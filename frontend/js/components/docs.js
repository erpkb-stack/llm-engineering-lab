// Renders the 12-section documentation for a module.
import { esc } from "../api.js";
import { flowHTML } from "./render.js";

const p = (t) => `<p>${esc(t)}</p>`;
const ul = (xs) => `<ul>${(xs || []).map((x) => `<li>${esc(x)}</li>`).join("")}</ul>`;

export function docsHTML(m) {
  const d = m.docs;
  const sec = (n, title, body) => `<div class="doc-section"><h6>${n}. ${esc(title)}</h6>${body}</div>`;
  return `
  <div class="row g-3">
    <div class="col-xl-8">
      <div class="lab-card">
        ${m.experimental ? `<div class="alert alert-warning py-2 small">Experimental concept — this is an evaluation/analysis category, not a standard inference optimization or model parameter.</div>` : ""}
        ${sec(1, "What is it?", p(d.what))}
        ${sec(2, "Why does it exist?", p(d.why))}
        ${sec(3, "How does it work?", p(d.how) + (d.formula ? `<div class="mono p-2 rounded" style="background:var(--lab-surface-2)">${esc(d.formula)}</div>` : ""))}
        ${sec(4, "When should I use it?", p(d.when_use))}
        ${sec(5, "When should I avoid it?", p(d.when_avoid))}
        <div class="row"><div class="col-md-6">${sec(6, "Advantages", ul(d.pros))}</div><div class="col-md-6">${sec(7, "Disadvantages", ul(d.cons))}</div></div>
        ${sec(8, "Real-world enterprise use case", p(d.use_case))}
        ${sec(9, "Python implementation", `<p class="small text-muted">See the <b>Implementation</b> tab for an editor view.</p><pre class="code">${esc(d.code)}</pre>`)}
        ${sec(10, "Architecture diagram", flowHTML(d.diagram || []))}
        ${sec(11, "Performance impact", p(d.performance))}
        ${sec(12, "Production checklist", `<ul class="list-unstyled">${(d.checklist || []).map((c) => `<li>☐ ${esc(c)}</li>`).join("")}</ul>`)}
      </div>
    </div>
    <div class="col-xl-4">
      <div class="lab-card"><h6>Where would I use this in a real application?</h6>
        ${(d.applications || []).map((a) => `<div class="mb-2"><span class="badge text-bg-secondary">${esc(a.domain)}</span><div class="small mt-1">${esc(a.text)}</div></div>`).join("")}
        <a href="#/docs" class="small">See all application domains →</a>
      </div>
      <div class="lab-card"><h6>Simulation fidelity & limitations</h6><p class="small">${esc(m.fidelity_note)}</p>${d.limitations ? `<p class="small text-muted mb-0">${esc(d.limitations)}</p>` : ""}</div>
      ${d.references?.length ? `<div class="lab-card"><h6>References</h6>${ul(d.references)}</div>` : ""}
    </div>
  </div>`;
}
