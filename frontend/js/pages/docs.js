import { esc, get, state } from "../api.js";
import { SECTIONS, setCrumbs } from "../app.js";
import { docsHTML } from "../components/docs.js";
import { FID_LABEL, fidelityBadge } from "../components/render.js";

const DISTINCTIONS = [
  ["Tokenizer bias", "Tokenization effects (unequal token cost/fragmentation) are different from model-level bias and from decoding-time logit bias."],
  ["Logic bias", "A model behaviour / evaluation category for reasoning errors — not a standard tuning parameter."],
  ["Confirmation prediction", "Ambiguous: separate next-token prediction, confidence estimation, prediction verification and confirmation bias."],
  ["Cosine similarity trap", "A retrieval/evaluation limitation, not a model architecture."],
  ["Prefix caching", "Different from the per-sequence KV cache, though implementations reuse cached KV blocks."],
  ["Medusa heads", "Related to speculative decoding, but uses extra heads on the same model instead of a draft model."],
  ["Differential privacy", "A formal mechanism with accounting — not simply 'adding random noise'."],
];

export async function renderDocs(view, id) {
  if (id) {
    const m = await get(`/api/modules/${encodeURIComponent(id)}`);
    setCrumbs(`<a href="#/docs">Documentation</a> / ${esc(m.title)}`);
    view.innerHTML = `<div class="d-flex align-items-center mb-2"><div class="page-title">${esc(m.title)}</div><a class="btn btn-sm btn-primary ms-auto" href="#/module/${m.id}">Open playground →</a></div>
      <div class="page-sub">${esc(m.summary)}</div>${docsHTML(m)}`;
    return;
  }
  setCrumbs("Documentation");
  const apps = await get("/api/modules/applications");
  view.innerHTML = `
    <div class="page-title">Documentation</div>
    <div class="page-sub">Every module documents the same 12 sections: what, why, how, when to use, when to avoid, advantages, disadvantages, enterprise use case,
      Python implementation, architecture diagram, performance impact and production checklist.</div>
    <div class="row g-3">
      <div class="col-xl-7">
        <div class="lab-card"><h6>Module documentation</h6>
          ${Object.entries(SECTIONS).map(([k, s]) => `<div class="mb-2"><div class="small text-muted text-uppercase">${esc(s.title)}</div>
            ${s.modules.map((id) => { const m = state.modules.find((x) => x.id === id); return m ? `<a class="badge text-bg-secondary text-decoration-none me-1 mb-1" href="#/docs/${id}">${esc(m.title)}</a>` : ""; }).join("")}</div>`).join("")}
        </div>
        <div class="lab-card"><h6>Where Would I Use This in a Real Application?</h6>
          <div class="accordion" id="apps">${Object.entries(apps).map(([domain, items], i) => `
            <div class="accordion-item"><h2 class="accordion-header"><button class="accordion-button ${i ? "collapsed" : ""}" type="button" data-bs-toggle="collapse" data-bs-target="#app-${i}">
              ${esc(domain)} <span class="badge text-bg-secondary ms-2">${items.length}</span></button></h2>
              <div id="app-${i}" class="accordion-collapse collapse ${i ? "" : "show"}" data-bs-parent="#apps"><div class="accordion-body">
              ${items.length ? items.map((it) => `<div class="mb-2"><a href="#/module/${it.module}" class="fw-semibold">${esc(it.title)}</a><div class="small">${esc(it.text)}</div></div>`).join("") : '<span class="small text-muted">—</span>'}
              </div></div></div>`).join("")}</div>
        </div>
      </div>
      <div class="col-xl-5">
        <div class="lab-card"><h6>Fidelity levels (how to read every number)</h6>
          ${Object.entries(FID_LABEL).map(([k, v]) => `<div class="mb-2">${fidelityBadge(k)} <span class="small">${esc({ analytical: "Closed-form formulas (roofline, parameter counts, accounting) with stated hardware assumptions. Not a measurement.",
            computed: "The real algorithm executed exactly on real or toy inputs (BM25, BPE, quantization error, RDP accountant, DP-SGD on a toy model).",
            simulated: "Deterministic simulation with synthetic data or synthetic model behaviour; illustrates mechanisms, never benchmarks a real model.",
            real: "Output from a real model/library in this environment (Ollama, OpenAI-compatible API, Hugging Face) or measured wall-clock time." }[k])}</span></div>`).join("")}
        </div>
        <div class="lab-card"><h6>Important technical distinctions</h6><table class="table table-sm table-lab"><tbody>${DISTINCTIONS.map(([a, b]) => `<tr><th style="width:160px">${esc(a)}</th><td class="small">${esc(b)}</td></tr>`).join("")}</tbody></table></div>
        <div class="lab-card"><h6>Implementation levels</h6><ol class="small mb-0"><li><b>Level 1</b> — interactive mathematical simulations for every concept (this lab, default).</li>
          <li><b>Level 2</b> — real Python integrations: tokenizers, sentence-transformers embeddings & cross-encoders, PEFT LoRA training, Ollama/OpenAI-compatible generation (opt-in).</li>
          <li><b>Level 3</b> — GPU-backed training & inference (vLLM, bitsandbytes, Opacus) — reference code in each module's Implementation tab.</li></ol></div>
        <div class="lab-card"><h6>API</h6><p class="small mb-1">Interactive OpenAPI docs: <a href="/docs" target="_blank" rel="noopener">/docs</a> (Swagger) · <a href="/redoc" target="_blank" rel="noopener">/redoc</a></p>
          <p class="small text-muted mb-0">Authenticate with <span class="kbd">POST /api/auth/login</span> and use the returned <span class="kbd">access_token</span> as a Bearer token.</p></div>
      </div>
    </div>`;
}
