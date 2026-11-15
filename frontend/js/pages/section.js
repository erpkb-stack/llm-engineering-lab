import { esc, state } from "../api.js";
import { SECTIONS, setCrumbs } from "../app.js";
import { fidelityBadge } from "../components/render.js";

export async function renderSection(view, key) {
  const s = SECTIONS[key];
  if (!s) { view.innerHTML = `<div class="alert alert-warning">Unknown section.</div>`; return; }
  setCrumbs(esc(s.title));
  const mods = s.modules.map((id) => state.modules.find((m) => m.id === id)).filter(Boolean);
  view.innerHTML = `
    <div class="page-title">${s.icon} ${esc(s.title)}</div>
    <div class="page-sub">${esc(s.blurb)}</div>
    <div class="row g-3">${mods.map((m) => `
      <div class="col-md-6 col-xxl-4"><a class="lab-card module-card d-block text-decoration-none text-reset" href="#/module/${m.id}?from=${key}">
        <div class="d-flex align-items-start gap-2 mb-1"><div class="fw-semibold">${m.original_index ? `<span class="text-muted">${m.original_index}.</span> ` : ""}${esc(m.title)}</div>
        <div class="ms-auto">${fidelityBadge(m.fidelity)}</div></div>
        <div class="small text-muted mb-2">${esc(m.summary)}</div>
        <div class="d-flex gap-1"><span class="badge text-bg-secondary">${esc(m.category_label)}</span>${m.experimental ? '<span class="badge badge-exp">Experimental</span>' : ""}</div>
      </a></div>`).join("")}</div>`;
}
