// Interactive concept-module page: controls → live (debounced, WebSocket) simulation → visualizations.
import { can, debounce, del, download, esc, get, post, resultToCSV, toast } from "../api.js";
import { setCrumbs, SECTIONS } from "../app.js";
import { renderControls } from "../components/controls.js";
import { docsHTML } from "../components/docs.js";
import { editor } from "../components/monaco.js";
import { fidelityBadge, renderResult } from "../components/render.js";

class LiveSocket {
  constructor() { this.pending = new Map(); this.seq = 0; this.connect(); }
  connect() {
    const proto = location.protocol === "https:" ? "wss" : "ws";
    this.ws = new WebSocket(`${proto}://${location.host}/ws/modules`);
    this.ready = new Promise((res) => { this.ws.onopen = () => res(true); this.ws.onerror = () => res(false); });
    this.ws.onmessage = (ev) => {
      const msg = JSON.parse(ev.data);
      const p = this.pending.get(msg.request_id);
      if (!p) return;
      this.pending.delete(msg.request_id);
      msg.type === "result" ? p.resolve(msg.result) : p.reject(new Error(msg.message));
    };
    this.ws.onclose = () => { this.closed = true; this.pending.forEach((p) => p.reject(new Error("socket closed"))); this.pending.clear(); };
  }
  async run(module_id, params) {
    if (this.closed || !(await this.ready)) throw new Error("ws unavailable");
    const request_id = ++this.seq;
    return new Promise((resolve, reject) => {
      this.pending.set(request_id, { resolve, reject });
      this.ws.send(JSON.stringify({ module_id, params, request_id }));
      setTimeout(() => { if (this.pending.has(request_id)) { this.pending.delete(request_id); reject(new Error("timeout")); } }, 60000);
    });
  }
  close() { try { this.ws.close(); } catch { /* ignore */ } }
}

export async function renderModule(view, id, params) {
  const m = await get(`/api/modules/${encodeURIComponent(id)}`);
  const from = params.get("from");
  setCrumbs(`${from && SECTIONS[from] ? `<a href="#/section/${from}">${esc(SECTIONS[from].title)}</a> / ` : ""}${esc(m.title)}`);
  const ws = new LiveSocket();
  let current = null, baseline = null, lastReq = 0;
  view.innerHTML = `
    <div class="d-flex flex-wrap align-items-start gap-2 mb-2">
      <div>
        <div class="page-title">${m.original_index ? `<span class="text-muted">${m.original_index}.</span> ` : ""}${esc(m.title)}</div>
        <div class="page-sub mb-2">${esc(m.summary)}</div>
        <div class="d-flex gap-1 flex-wrap"><span class="badge text-bg-secondary">${esc(m.category_label)}</span>${fidelityBadge(m.fidelity)}
          ${m.experimental ? '<span class="badge badge-exp">Experimental concept</span>' : ""}${m.tags.includes("essential") ? '<span class="badge text-bg-info">Essential module</span>' : ""}</div>
      </div>
      <div class="ms-auto d-flex gap-2 flex-wrap">
        <button class="btn btn-primary btn-sm" data-act="run" title="Run now and save to Experiment History">▶ Run Experiment</button>
        <button class="btn btn-outline-secondary btn-sm" data-act="reset">↺ Reset</button>
        <button class="btn btn-outline-secondary btn-sm" data-act="save" ${can("save") ? "" : 'disabled title="Your role cannot save configurations"'}>💾 Save Configuration</button>
        <button class="btn btn-outline-secondary btn-sm" data-act="compare" title="Pin the current result as a baseline; later runs show deltas">⇄ Compare Results</button>
        <div class="btn-group">
          <button class="btn btn-outline-secondary btn-sm dropdown-toggle" data-bs-toggle="dropdown">⤓ Export Results</button>
          <ul class="dropdown-menu dropdown-menu-end"><li><button class="dropdown-item" data-act="export-json">JSON</button></li><li><button class="dropdown-item" data-act="export-csv">CSV</button></li></ul>
        </div>
        <a class="btn btn-outline-secondary btn-sm" href="#/benchmark?module=${esc(m.id)}">⚖ Benchmark</a>
      </div>
    </div>
    <ul class="nav nav-tabs mb-3" role="tablist">
      <li class="nav-item"><button class="nav-link active" data-bs-toggle="tab" data-bs-target="#tab-play" type="button">Playground</button></li>
      <li class="nav-item"><button class="nav-link" data-bs-toggle="tab" data-bs-target="#tab-learn" type="button">Learn (12 sections)</button></li>
      <li class="nav-item"><button class="nav-link" data-bs-toggle="tab" data-bs-target="#tab-code" type="button" data-code-tab>Implementation</button></li>
    </ul>
    <div class="tab-content">
      <div class="tab-pane fade show active" id="tab-play">
        <div class="row g-3">
          <div class="col-lg-4 col-xl-3">
            <div class="lab-card controls-panel">
              <div class="d-flex align-items-center mb-1"><h6 class="mb-0">Configuration</h6><span class="computing ms-auto" data-status></span></div>
              ${m.presets?.length ? `<select class="form-select form-select-sm my-2" data-presets><option value="">Load a preset scenario…</option>${m.presets.map((p, i) => `<option value="${i}">${esc(p.name)}</option>`).join("")}</select>` : ""}
              <div class="input-group input-group-sm my-2"><select class="form-select" data-configs><option value="">Saved configurations…</option></select>
                <button class="btn btn-outline-danger" data-act="del-config" title="Delete selected configuration">✕</button></div>
              <div data-controls></div>
              <div class="form-check form-switch mt-2 small"><input class="form-check-input" type="checkbox" id="live-toggle" checked><label class="form-check-label" for="live-toggle">Live update (debounced)</label></div>
            </div>
          </div>
          <div class="col-lg-8 col-xl-9">
            <div data-baseline class="mb-2"></div>
            <div data-results><div class="computing"><span class="spinner-dot"></span>Running…</div></div>
          </div>
        </div>
      </div>
      <div class="tab-pane fade" id="tab-learn">${docsHTML(m)}</div>
      <div class="tab-pane fade" id="tab-code"><div class="lab-card"><h6>Example implementation (Python)</h6><div class="monaco-host" data-editor></div>
        <p class="small text-muted mt-2 mb-0">Reference code using real libraries (PEFT, vLLM, sentence-transformers, Opacus, …). The lab's own simulation code lives in <span class="kbd">app/</span>.</p></div></div>
    </div>`;

  const status = view.querySelector("[data-status]");
  const resultsEl = view.querySelector("[data-results]");
  const baselineEl = view.querySelector("[data-baseline]");

  async function run(vals, { save = false } = {}) {
    const req = ++lastReq;
    status.innerHTML = `<span class="spinner-dot"></span>computing`;
    try {
      let res;
      try { res = await ws.run(m.id, vals); } catch { res = await post(`/api/modules/${m.id}/run`, { params: vals }); }
      if (req !== lastReq) return;  // stale response
      current = res;
      renderResult(resultsEl, res, { baseline });
      status.textContent = `updated ${new Date().toLocaleTimeString()}`;
      if (save) {
        if (!can("save")) { toast("Run complete. Your role (viewer) cannot save experiments.", "warn"); return; }
        const e = await post("/api/experiments", { module_id: m.id, name: `${m.title} run`, kind: "run", params: res.params, result: res,
          baseline_params: baseline?.params || null });
        toast(`Saved to Experiment History (#${e.id})`, "ok");
      }
    } catch (e) {
      if (req !== lastReq) return;
      status.textContent = "";
      resultsEl.innerHTML = `<div class="alert alert-danger">${esc(e.message)}</div>`;
    }
  }
  const debounced = debounce((v) => run(v), 350);
  const controls = renderControls(view.querySelector("[data-controls]"), m.controls, null, (vals) => {
    if (view.querySelector("#live-toggle").checked) debounced(vals);
  });

  // saved configurations
  const cfgSel = view.querySelector("[data-configs]");
  async function loadConfigs(selectId) {
    const cfgs = await get(`/api/configs?module_id=${m.id}`);
    cfgSel.innerHTML = `<option value="">Saved configurations (${cfgs.length})…</option>` + cfgs.map((c) => `<option value="${c.id}">${esc(c.name)}</option>`).join("");
    cfgSel._cfgs = cfgs;
    if (selectId) cfgSel.value = String(selectId);
  }
  cfgSel.addEventListener("change", () => {
    const c = cfgSel._cfgs?.find((x) => String(x.id) === cfgSel.value);
    if (c) { controls.set(c.params); run(controls.get()); toast(`Loaded “${c.name}”`); }
  });
  view.querySelector('[data-act="del-config"]').addEventListener("click", async () => {
    if (!cfgSel.value) return;
    if (!confirm("Delete this saved configuration?")) return;
    await del(`/api/configs/${cfgSel.value}`);
    toast("Configuration deleted");
    loadConfigs();
  });
  view.querySelector("[data-presets]")?.addEventListener("change", (e) => {
    const p = m.presets[Number(e.target.value)];
    if (p) { controls.set(p.params); run(controls.get()); }
  });

  const actions = {
    run: () => run(controls.get(), { save: true }),
    reset: () => { controls.reset(); run(controls.get()); },
    save: async () => {
      const name = prompt("Name this configuration:", `${m.title} — ${new Date().toLocaleDateString()}`);
      if (!name) return;
      const c = await post("/api/configs", { module_id: m.id, name, params: controls.get() });
      toast("Configuration saved", "ok");
      loadConfigs(c.id);
    },
    compare: () => {
      if (baseline) { baseline = null; baselineEl.innerHTML = ""; if (current) renderResult(resultsEl, current); toast("Baseline cleared"); return; }
      if (!current) return;
      baseline = current;
      baselineEl.innerHTML = `<div class="alert alert-info py-2 small mb-0 d-flex align-items-center">Baseline pinned — change parameters to see deltas on every metric.
        <button class="btn btn-sm btn-outline-info ms-auto" data-clear>Clear baseline</button></div>`;
      baselineEl.querySelector("[data-clear]").addEventListener("click", actions.compare);
      renderResult(resultsEl, current, { baseline });
    },
    "export-json": () => current && download(`${m.id}-result.json`, JSON.stringify({ ...current, baseline: baseline ? { params: baseline.params, metrics: baseline.metrics } : null }, null, 2)),
    "export-csv": () => current && download(`${m.id}-result.csv`, resultToCSV(current), "text/csv"),
  };
  view.querySelectorAll("[data-act]").forEach((b) => { const fn = actions[b.dataset.act]; if (fn) b.addEventListener("click", fn); });

  let edLoaded = false;
  view.querySelector("[data-code-tab]").addEventListener("shown.bs.tab", async () => {
    if (edLoaded) return;
    edLoaded = true;
    await editor(view.querySelector("[data-editor]"), m.docs.code, { language: "python", readOnly: true });
  });

  await loadConfigs();
  const cfgId = params.get("config");
  if (cfgId && cfgSel._cfgs?.some((c) => String(c.id) === cfgId)) {
    cfgSel.value = cfgId;
    controls.set(cfgSel._cfgs.find((c) => String(c.id) === cfgId).params);
  }
  if (params.get("params")) { try { controls.set(JSON.parse(atob(params.get("params")))); } catch { /* ignore */ } }
  run(controls.get());
  return () => ws.close();
}
