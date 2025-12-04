const API_URL = "http://127.0.0.1:8000/predict";
const btn = document.getElementById("btnPredict");
const input = document.getElementById("userInput");
const statusSpan = document.getElementById("status");
const tokenRow = document.getElementById("tokenRow");
const cardsDiv = document.getElementById("cards");
const arrowSvg = document.getElementById("arrowSvg");

function countWords(s){
  return s.trim().split(/\s+/).filter(x => x.length>0).length;
}

function setStatus(t, warn=false){
  statusSpan.textContent = t;
  statusSpan.style.color = warn ? "crimson" : "#333";
}

async function predict(){
  const text = input.value.trim();
  const n = countWords(text);
  if(n < 2 || n > 9){
    setStatus("Écris une phrase de 5 à 9 mots (ex: 'Hier j'ai vu un grand chien').", true);
    return;
  }
  setStatus("Chargement…");
  try{
    const res = await fetch(API_URL, {
      method: "POST",
      headers: {"Content-Type":"application/json"},
      body: JSON.stringify({ text, top_k:4, max_new_tokens:3 })
    });
    if(!res.ok) throw new Error("Erreur serveur " + res.status);
    const data = await res.json();
    renderTokens(data.input_tokens || text.split(" "));
    renderCandidates(data.candidates || []);
    setStatus("Prêt — clique sur un mot pour voir l'attention");
  } catch(err){
    console.error(err);
    setStatus("Erreur : vérifie le backend", true);
  }
}

function renderTokens(tokens){
  tokenRow.innerHTML = "";
  tokens.forEach((t, i) => {
    const d = document.createElement("div");
    d.className = "token";
    d.textContent = t;
    d.dataset.idx = i;
    tokenRow.appendChild(d);
  });
}

function clearSelection(){
  document.querySelectorAll(".card.selected").forEach(c => c.classList.remove("selected"));
  clearArrows();
  resetTokenPulse();
}

function renderCandidates(candidates){
  cardsDiv.innerHTML = "";
  candidates.forEach((c, i) => {
    const card = document.createElement("div");
    card.className = "card";
    card.dataset.idx = i;
    card.innerHTML = `
      <div class="word">${c.word}</div>
      <div class="pct">${(c.prob*100).toFixed(1)}%</div>
      <div class="bar-bg"><div class="bar-fill" style="width:0%"></div></div>
    `;
    card.addEventListener("click", ()=> {
      clearSelection();
      card.classList.add("selected");
      onCardClick(i, c);
    });
    cardsDiv.appendChild(card);
    // animation de la barre
    setTimeout(()=> {
      const fill = card.querySelector(".bar-fill");
      fill.style.width = `${(c.prob*100).toFixed(1)}%`;
    }, 60 + i*80);
  });
}

function drawArrow(x1,y1,x2,y2, att){
  const ns = "http://www.w3.org/2000/svg";
  const line = document.createElementNS(ns, "line");
  line.setAttribute("x1", x1);
  line.setAttribute("y1", y1);
  line.setAttribute("x2", x1);
  line.setAttribute("y2", y1);
  line.setAttribute("stroke", `rgb(43, ${140 + att*100}, 255)`);
  line.setAttribute("stroke-width", 2);
  line.setAttribute("stroke-linecap","round");
  line.style.opacity = "0.0";
  line.style.transition = "all 450ms ease";
  arrowSvg.appendChild(line);
  requestAnimationFrame(()=>{
    line.style.opacity = "1.0";
    line.setAttribute("x2", x2);
    line.setAttribute("y2", y2);
  });
  return line;
}

function clearArrows(){
  while(arrowSvg.lastChild) arrowSvg.removeChild(arrowSvg.lastChild);
}

function centerOf(el){
  const r = el.getBoundingClientRect();
  return {x: r.left + r.width/2 + window.scrollX, y: r.top + r.height/2 + window.scrollY};
}

function resetTokenPulse(){
  document.querySelectorAll(".token").forEach(t => t.style.transform="scale(1)");
}

function onCardClick(cardIdx, candidate){
  clearArrows();
  resetTokenPulse();
  const tokenEls = Array.from(document.querySelectorAll(".token"));
  const cardEl = document.querySelector(`.card[data-idx="${cardIdx}"]`);
  if(!cardEl) return;
  const cardCenter = centerOf(cardEl);

  tokenEls.forEach((tEl, i) => {
    const tokCenter = centerOf(tEl);
    const svgRect = arrowSvg.getBoundingClientRect();
    const x1 = tokCenter.x - svgRect.left;
    const y1 = tokCenter.y - svgRect.top;
    const x2 = cardCenter.x - svgRect.left;
    const y2 = cardCenter.y - svgRect.top;

    const att = candidate.attention || 0;
    const line = drawArrow(x1,y1,x2,y2, att);
    const width = 1.2 + att*12*(1 - (i/tokenEls.length));
    line.setAttribute("stroke-width", width.toFixed(2));

    // pulse token selon attention
    tEl.style.transition = "transform 0.4s";
    tEl.style.transform = `scale(${1 + att*0.6})`;

    // fade out ligne
    setTimeout(()=> {
      line.style.opacity = "0";
      try { line.remove(); } catch(e){}
      tEl.style.transform = "scale(1)";
    }, 2200);
  });

  // popup explicatif
  const info = document.createElement("div");
  info.className = "explain-popup";
  info.textContent = `Mot choisi : ${candidate.word} — Probabilité ${(candidate.prob*100).toFixed(1)}%`;
  document.body.appendChild(info);
  const rect = cardEl.getBoundingClientRect();
  info.style.left = `${rect.right + 8 + window.scrollX}px`;
  info.style.top = `${rect.top + window.scrollY}px`;
  setTimeout(()=> info.remove(), 2400);
}

btn.addEventListener("click", predict);
input.addEventListener("keydown", (e) => { if(e.key === "Enter") predict(); });
window.addEventListener("resize", clearArrows);
