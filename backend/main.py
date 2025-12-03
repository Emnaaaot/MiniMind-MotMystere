# backend/main.py
from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
from transformers import AutoTokenizer, AutoModelForCausalLM
import torch
import torch.nn.functional as F
import numpy as np
from spellchecker import SpellChecker
import re

app = FastAPI(title="MiniMind - Mot Mystère (FR stabilisé)")

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Endpoint test GET /
@app.get("/")
def root():
    return {"status": "backend OK", "message": "MiniMind backend is running!"}

# Chargement modèle FR
MODEL_NAME = "dbddv01/gpt2-french-small"
DEVICE = "cpu"

try:
    tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME)
    model = AutoModelForCausalLM.from_pretrained(MODEL_NAME)
    print("Modèle FR chargé:", MODEL_NAME)
except:
    MODEL_NAME = "distilgpt2"
    tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME)
    model = AutoModelForCausalLM.from_pretrained(MODEL_NAME)
    print("Fallback:", MODEL_NAME)

model.to(DEVICE)
model.eval()

# Correction orthographique FR
spell = SpellChecker(language="fr")

class Prompt(BaseModel):
    text: str
    top_k: int = 4
    max_new_tokens: int = 3

WORD_RE = re.compile(r"^[A-Za-zÀ-ÖØ-öø-ÿ']+$")

def sanitize_input_list(text):
    return [t for t in text.strip().split() if t.strip() != ""]

def correct_spelling_fr(text):
    words = text.split()
    corrected = []

    for w in words:
        core = w.strip(".,;:!?")
        if core == "":
            corrected.append(w)
            continue

        lower = core.lower()
        if lower in spell:
            corrected.append(core)
        else:
            cand = spell.correction(lower)
            if cand:
                if core[0].isupper():
                    cand = cand.capitalize()
                corrected.append(cand)
            else:
                corrected.append(core)

    return " ".join(corrected)

def clean_word_candidate(s):
    parts = re.split(r"[^\wÀ-ÖØ-öø-ÿ']+", s.strip())
    for p in parts:
        p = p.strip()
        if WORD_RE.match(p) and len(p) > 1:
            return p
    return None


@app.post("/predict")
def predict(prompt: Prompt):
    raw_text = prompt.text.strip()
    if raw_text == "":
        return {"error": "texte vide", "input_tokens": [], "candidates": []}

    # Correction orthographique
    text = raw_text
    if not text.endswith(" "):
        text += " "

    corrected_text = correct_spelling_fr(text)
    input_for_model = corrected_text

    # Encode
    input_enc = tokenizer(input_for_model, return_tensors="pt").to(DEVICE)
    top_k = max(1, min(8, prompt.top_k))
    max_new = max(1, min(6, prompt.max_new_tokens))

    # Génération
    with torch.no_grad():
        gen_out = model.generate(
            **input_enc,
            max_new_tokens=max_new,
            num_beams=top_k,
            num_return_sequences=top_k,
            early_stopping=True,
            return_dict_in_generate=True,
            output_scores=True
        )

    sequences = gen_out.sequences.cpu().numpy()

    candidates = []
    for seq in sequences:
        gen_ids = seq[input_enc["input_ids"].shape[1]:]
        if gen_ids.size == 0:
            continue

        gen_text = tokenizer.decode(gen_ids, skip_special_tokens=True).strip()
        first = clean_word_candidate(gen_text)
        if not first:
            continue

        # Probabilités correctes
        with torch.no_grad():
            outputs = model(input_ids=torch.tensor([seq]).to(DEVICE))
        logits = outputs.logits.detach()

        probs = []
        for j, tok_id in enumerate(gen_ids):
            pos = input_enc["input_ids"].shape[1] - 1 + j
            if pos >= logits.shape[1]:
                continue
            tok_logits = logits[0, pos].detach().cpu()
            prob = F.softmax(tok_logits, dim=-1)[int(tok_id)].item()
            probs.append(max(prob, 1e-12))

        seq_prob = float(np.prod(probs)) if probs else 0.0

        candidates.append({
            "word": first,
            "gen_text": gen_text,
            "seq_prob": seq_prob,
            "raw_ids": [int(i) for i in gen_ids.tolist()]
        })

    # Agrégation
    aggregated = {}
    for c in candidates:
        aggregated.setdefault(c["word"], 0.0)
        aggregated[c["word"]] += c["seq_prob"]

    # Fallback top-k logits
    if len(aggregated) < top_k:
        with torch.no_grad():
            logits = model(**input_enc).logits[:, -1, :].detach().cpu()[0].numpy()
        top_idx = np.argpartition(-logits, top_k)[:top_k]
        top_idx = top_idx[np.argsort(-logits[top_idx])]

        for idx in top_idx:
            decoded = tokenizer.decode([int(idx)], skip_special_tokens=True).strip()
            cleaned = clean_word_candidate(decoded)
            if cleaned:
                aggregated.setdefault(cleaned, 0.0)
                p = float(F.softmax(torch.tensor(logits), dim=0)[idx].item())
                aggregated[cleaned] += p

    # Normalize
    total = sum(aggregated.values()) or 1.0
    final = []
    emb = model.get_input_embeddings()

    last_id = input_enc["input_ids"][0, -1].item()
    last_vec = emb.weight[last_id].detach().cpu().numpy()

    for w, score in aggregated.items():
        raw_id = None
        for c in candidates:
            if c["word"] == w and len(c["raw_ids"]) > 0:
                raw_id = c["raw_ids"][0]
                break

        att = 0.0
        if raw_id is not None:
            cand_vec = emb.weight[raw_id].detach().cpu().numpy()
            att = float(np.dot(last_vec, cand_vec) /
                        (np.linalg.norm(last_vec) * np.linalg.norm(cand_vec) + 1e-9))

        final.append({
            "word": w,
            "prob": score / total,
            "attention": att
        })

    final = sorted(final, key=lambda x: -x["prob"])[:top_k]

    return {
        "input_text_raw": raw_text,
        "input_text_corrected": corrected_text,
        "input_tokens": sanitize_input_list(corrected_text),
        "candidates": final
    }
