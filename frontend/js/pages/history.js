import { api, can, del, download, esc, fmt, get, state, toast } from "../api.js";
import { setCrumbs } from "../app.js";
import { fidelityBadge, metricCards } from "../components/render.js";

export async function renderHistory(view) {
  setCrumbs("Experiment History");
  view.innerHTML = `
    <div class="page-title">Experiment History</div>
    <div class="page-sub">Your saved runs, A/B benchmarks and fine-tuning experiments, plus saved configurations. Select two runs of the same module to compare them.</div>
    <div class="lab-card mb-3 d-flex flex-wrap gap-2 align-items-center">
      <select class="form-select form-select-sm w-auto" id="h-module"><option value="">All modules</option>${state.modules.map((m) => `<option value="${m.id}">${esc(m.title)}</option>`).join("")}</select>
      <select class="form-select form-select-sm w-auto" id="h-kind"><option value="">All kinds</option><option value="run">Runs</option><option value="benchmark">Benchmarks</option><option value="fine_tune">Fine-tuning</option></select>
      <button class="btn btn-outline-primary btn-sm" id="h-cmp" disabled>Compare selected (2)</button>
      <div class="ms-auto btn-group"><button class="btn btn-outline-secondary btn-sm" id="h-json">⤓ Export JSON</button><button class="btn btn-outline-secondary btn-sm" id="h-csv">⤓ Export CSV</button></div>
    </div>
    <div class="lab-card"><div id="h-table"></div></div>
    <div id="h-detail" class="mt-3"></div>
    <div class="lab-card mt-3"><h6>Saved configurations</h6><div id="h-configs"></div></div>`;
  const sel = new Set();
  let rows = [];
  async function load() {
    const q = new URLSearchParams();
    const m = view.querySelector("#h-module").value, k = view.querySelector("#h-kind").value;
    if (m) q.set("module_id", m);
    if (k) q.set("kind", k);
    rows = await get(`/api/experiments?${q}`);
    sel.clear();
    view.querySelector("#h-cmp").disabled = true;
    view.querySelector("#h-table").innerHTML = rows.length ? `<div class="table-wrap" style="max-height:520px"><table class="table table-sm table-lab align-middle"><thead><tr>
      <th></th><th>#</th><th>Name</th><th>Module</th><th>Kind</th><th>Fidelity</th><th>Key metrics</th><th>Created</th><th></th></tr></thead><tbody>
      ${rows.map((e) => `<tr><td><input type="checkbox" class="form-check-input" data-sel="${e.id}"></td><td>${e.id}</td><td>${esc(e.name)}</td>
        <td><a href="#/module/${esc(e.module_id)}">${esc(state.modules.find((m) => m.id === e.module_id)?.title || e.module_id)}</a></td><td>${esc(e.kind)}</td><td>${fidelityBadge(e.mode)}</td>
        <td class="small">${Object.entries(e.metrics).slice(0, 3).map(([k, v]) => `${esc(k)}: <b>${esc(fmt(v))}</b>`).join(" · ")}</td>
        <td class="text-muted small">${new Date(e.created_at).toLocaleString()}</td>
        <td class="text-nowrap"><button class="btn btn-sm btn-outline-secondary" data-view="${e.id}">View</button>
          ${can("save") ? `<button class="btn btn-sm btn-outline-danger" data-del="${e.id}">✕</button>` : ""}</td></tr>`).join("")}</tbody></table></div>`
      : `<p class="text-muted small mb-0">No experiments match. Run a module and press <b>Run Experiment</b>, or save a Benchmark Studio comparison.</p>`;
    view.querySelectorAll("[data-sel]").forEach((cb) => cb.addEventListener("change", () => {
      cb.checked ? sel.add(Number(cb.dataset.sel)) : sel.delete(Number(cb.dataset.sel));
      view.querySelector("#h-cmp").disabled = sel.size !== 2;
    }));
    view.querySelectorAll("[data-view]").forEach((b) => b.addEventListener("click", () => show(Number(b.dataset.view))));
    view.querySelectorAll("[data-del]").forEach((b) => b.addEventListener("click", async () => {
      if (!confirm(`Delete experiment #${b.dataset.del}? This cannot be undone.`)) return;
      await del(`/api/experiments/${b.dataset.del}`);
      toast("Experiment deleted");
      load();
    }));
  }
  async function show(id) {
    const e = await get(`/api/experiments/${id}`);
    const params = btoa(JSON.stringify(e.params));
    view.querySelector("#h-detail").innerHTML = `<div class="lab-card"><div class="d-flex align-items-center mb-2"><h6 class="mb-0">#${e.id} · ${esc(e.name)}</h6>
      <span class="ms-2">${fidelityBadge(e.mode)}</span>${e.kind !== "fine_tune" ? `<a class="btn btn-sm btn-outline-primary ms-auto" href="#/module/${esc(e.module_id)}?params=${encodeURIComponent(params)}">Re-open with these parameters</a>` : ""}</div>
      ${metricCards(e.result.metrics || [])}
      <div class="row mt-3"><div class="col-md-6"><div class="small text-muted mb-1">Parameters</div><pre class="code">${esc(JSON.stringify(e.params, null, 2))}</pre></div>
      <div class="col-md-6">${e.baseline_params ? `<div class="small text-muted mb-1">Baseline parameters</div><pre class="code">${esc(JSON.stringify(e.baseline_params, null, 2))}</pre>` : ""}
      ${e.result.deltas ? `<div class="small text-muted mb-1">Deltas</div><pre class="code">${esc(e.result.deltas.map((d) => `${d.label}: ${fmt(d.baseline)} → ${fmt(d.variant)}`).join("\n"))}</pre>` : ""}</div></div></div>`;
    view.querySelector("#h-detail").scrollIntoView({ behavior: "smooth" });
  }
  view.querySelector("#h-cmp").addEventListener("click", async () => {
    const [a, b] = await Promise.all([...sel].map((id) => get(`/api/experiments/${id}`)));
    const am = Object.fromEntries((a.result.metrics || []).map((m) => [m.label, m]));
    const keys = (b.result.metrics || []).filter((m) => am[m.label]);
    view.querySelector("#h-detail").innerHTML = `<div class="lab-card"><h6>#${a.id} vs #${b.id}</h6>${a.module_id !== b.module_id ? '<div class="alert alert-warning py-1 small">Different modules — only shared metric names are compared.</div>' : ""}
      <table class="table table-sm table-lab"><thead><tr><th>Metric</th><th>#${a.id}</th><th>#${b.id}</th><th>Δ</th></tr></thead><tbody>
      ${keys.map((m) => { const x = am[m.label].value, y = m.value; return `<tr><td>${esc(m.label)}</td><td>${esc(fmt(x))}</td><td>${esc(fmt(y))}</td><td>${typeof x === "number" && typeof y === "number" ? esc(fmt(y - x)) : ""}</td></tr>`; }).join("")}</tbody></table>
      <div class="small text-muted">Changed parameters: ${Object.keys(b.params).filter((k) => JSON.stringify(a.params[k]) !== JSON.stringify(b.params[k])).map((k) => `<span class="badge text-bg-secondary">${esc(k)}</span>`).join(" ") || "none"}</div></div>`;
  });
  const exportAs = async (fmt_) => {
    const r = await api(`/api/experiments/export?format=${fmt_}`, { raw: true });
    download(`experiments.${fmt_}`, await r.text(), fmt_ === "csv" ? "text/csv" : "application/json");
  };
  view.querySelector("#h-json").addEventListener("click", () => exportAs("json"));
  view.querySelector("#h-csv").addEventListener("click", () => exportAs("csv"));
  view.querySelector("#h-module").addEventListener("change", load);
  view.querySelector("#h-kind").addEventListener("change", load);
  await load();
  const cfgs = await get("/api/configs");
  view.querySelector("#h-configs").innerHTML = cfgs.length ? `<table class="table table-sm table-lab"><thead><tr><th>Name</th><th>Module</th><th>Created</th><th></th></tr></thead><tbody>
    ${cfgs.map((c) => `<tr><td>${esc(c.name)}</td><td>${esc(state.modules.find((m) => m.id === c.module_id)?.title || c.module_id)}</td><td class="text-muted small">${new Date(c.created_at).toLocaleString()}</td>
    <td><a class="btn btn-sm btn-outline-primary" href="#/module/${esc(c.module_id)}?config=${c.id}">Load</a></td></tr>`).join("")}</tbody></table>`
    : `<p class="small text-muted mb-0">No saved configurations yet — use <b>Save Configuration</b> on any module.</p>`;
}
