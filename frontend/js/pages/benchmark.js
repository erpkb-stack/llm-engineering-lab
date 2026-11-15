import { can, download, esc, fmt, get, post, state, toast } from "../api.js";
import { setCrumbs } from "../app.js";
import { renderChart } from "../components/charts.js";
import { renderControls } from "../components/controls.js";
import { fidelityBadge, metricCards } from "../components/render.js";

export async function renderBenchmark(view, params) {
  setCrumbs("Benchmark Studio");
  const initial = params.get("module") || "kv-cache";
  view.innerHTML = `
    <div class="page-title">Benchmark Studio</div>
    <div class="page-sub">A/B experiments: pick a baseline configuration, change one or more parameters, run both and compare. Every result carries its fidelity label —
      simulated and analytical numbers are never presented as measured benchmarks. Use <b>Live measurement</b> for real wall-clock numbers from a connected model.</div>
    <div class="lab-card mb-3 d-flex flex-wrap gap-2 align-items-center">
      <label class="small text-muted">Module</label>
      <select class="form-select form-select-sm w-auto" id="bm-module">${state.modules.map((m) => `<option value="${m.id}">${esc(m.title)}</option>`).join("")}</select>
      <button class="btn btn-outline-secondary btn-sm" id="bm-copy" title="Copy baseline parameters into the variant">Copy baseline → variant</button>
      <div class="form-check ms-2"><input class="form-check-input" type="checkbox" id="bm-save" ${can("benchmark") ? "checked" : "disabled"}><label class="form-check-label small" for="bm-save">Save to history</label></div>
      <button class="btn btn-primary btn-sm ms-auto" id="bm-run" ${can("benchmark") ? "" : 'disabled title="Your role cannot run benchmarks"'}>▶ Run A/B comparison</button>
      <div class="btn-group"><button class="btn btn-outline-secondary btn-sm dropdown-toggle" data-bs-toggle="dropdown" id="bm-export" disabled>⤓ Export</button>
        <ul class="dropdown-menu dropdown-menu-end"><li><button class="dropdown-item" data-exp="json">JSON</button></li><li><button class="dropdown-item" data-exp="csv">CSV</button></li></ul></div>
    </div>
    <div class="row g-3">
      <div class="col-lg-3"><div class="lab-card controls-panel"><h6>A · Baseline configuration</h6><div id="bm-a"></div></div></div>
      <div class="col-lg-3"><div class="lab-card controls-panel"><h6>B · Variant configuration <span class="small text-muted" id="bm-changed"></span></h6><div id="bm-b"></div></div></div>
      <div class="col-lg-6"><div id="bm-results"><div class="lab-card text-muted small">Configure A and B, then run the comparison.</div></div></div>
    </div>
    <div class="row g-3 mt-1">
      <div class="col-lg-6"><div class="lab-card"><h6>Parameter sweep</h6>
        <div class="d-flex flex-wrap gap-2 align-items-end mb-2">
          <div><label class="small text-muted">Parameter</label><select class="form-select form-select-sm" id="sw-key"></select></div>
          <div><label class="small text-muted">From</label><input class="form-control form-control-sm" id="sw-from" type="number" style="width:100px"></div>
          <div><label class="small text-muted">To</label><input class="form-control form-control-sm" id="sw-to" type="number" style="width:100px"></div>
          <div><label class="small text-muted">Points</label><input class="form-control form-control-sm" id="sw-n" type="number" value="8" min="2" max="25" style="width:70px"></div>
          <div><label class="small text-muted">Metric</label><select class="form-select form-select-sm" id="sw-metric"></select></div>
          <button class="btn btn-outline-primary btn-sm" id="sw-run" ${can("benchmark") ? "" : "disabled"}>Run sweep</button>
        </div><div class="chart-box"><canvas id="sw-chart"></canvas></div><div class="small text-muted mt-1" id="sw-note">Sweeps vary one parameter of configuration A.</div></div></div>
      <div class="col-lg-6"><div class="lab-card"><h6>Live measurement (real provider)</h6>
        <div class="input-group input-group-sm mb-2"><input class="form-control" id="lv-prompt" value="Explain the KV cache in two sentences.">
          <input class="form-control" id="lv-runs" type="number" value="3" min="1" max="10" style="max-width:80px" title="runs">
          <button class="btn btn-outline-primary" id="lv-run" ${can("benchmark") ? "" : "disabled"}>Measure</button></div>
        <div id="lv-out" class="small text-muted">Measures actual wall-clock latency against the provider configured in Settings. In simulation mode nothing is measured.</div></div></div>
    </div>`;
  const sel = view.querySelector("#bm-module");
  sel.value = state.modules.some((m) => m.id === initial) ? initial : "kv-cache";
  let mod, A, B, last = null;

  const markChanged = () => {
    const a = A.get(), b = B.get();
    const changed = Object.keys(b).filter((k) => JSON.stringify(a[k]) !== JSON.stringify(b[k]));
    view.querySelector("#bm-changed").textContent = changed.length ? `(${changed.length} changed: ${changed.join(", ")})` : "(identical to A)";
  };

  async function loadModule() {
    mod = await get(`/api/modules/${sel.value}`);
    A = renderControls(view.querySelector("#bm-a"), mod.controls, null, markChanged, { compact: true, idPrefix: "a" });
    B = renderControls(view.querySelector("#bm-b"), mod.controls, null, markChanged, { compact: true, idPrefix: "b" });
    // suggest a meaningful default variant: flip the first toggle or move the first slider
    const first = mod.controls.find((c) => c.type === "toggle") || mod.controls.find((c) => c.type === "slider");
    if (first?.type === "toggle") B.set({ [first.key]: !first.default });
    else if (first) B.set({ [first.key]: Math.min(first.max, Number((first.default * 2 || first.max).toFixed(6))) });
    markChanged();
    const nums = mod.controls.filter((c) => c.type === "slider");
    const swk = view.querySelector("#sw-key");
    swk.innerHTML = nums.map((c) => `<option value="${c.key}">${esc(c.label)}</option>`).join("");
    const setRange = () => { const c = nums.find((x) => x.key === swk.value); if (c) { view.querySelector("#sw-from").value = c.min; view.querySelector("#sw-to").value = c.max; } };
    swk.onchange = setRange;
    setRange();
    view.querySelector("#bm-results").innerHTML = `<div class="lab-card text-muted small">Configure A and B, then run the comparison.</div>`;
  }
  sel.addEventListener("change", loadModule);
  view.querySelector("#bm-copy").addEventListener("click", () => { B.set(A.get()); markChanged(); });

  view.querySelector("#bm-run").addEventListener("click", async () => {
    const out = view.querySelector("#bm-results");
    out.innerHTML = `<div class="computing"><span class="spinner-dot"></span>Running baseline and variant…</div>`;
    try {
      last = await post("/api/benchmarks/compare", { module_id: mod.id, baseline_params: A.get(), variant_params: B.get(), save: view.querySelector("#bm-save").checked,
        name: `${mod.title} A/B` });
      const numeric = last.deltas.filter((d) => typeof d.pct === "number" && isFinite(d.pct));
      out.innerHTML = `
        <div class="fidelity-banner mb-2">${fidelityBadge(last.fidelity)}<div>${esc(last.fidelity_note)}${last.experiment_id ? ` · saved as experiment #${last.experiment_id}` : ""}</div></div>
        <div class="lab-card"><h6>Changed parameters</h6>${Object.keys(last.changed).length ? Object.entries(last.changed).map(([k, v]) => `<span class="badge text-bg-secondary me-1">${esc(k)}: ${esc(JSON.stringify(v.from))} → ${esc(JSON.stringify(v.to))}</span>`).join("") : '<span class="small text-muted">none — A and B are identical</span>'}</div>
        <div class="lab-card"><h6>Metric deltas (B vs A)</h6><div class="table-wrap"><table class="table table-sm table-lab"><thead><tr><th>Metric</th><th>A · baseline</th><th>B · variant</th><th>Δ</th><th>Δ %</th></tr></thead><tbody>
          ${last.deltas.map((d) => `<tr><td>${esc(d.label)}</td><td>${esc(fmt(d.baseline))} ${esc(d.unit)}</td><td>${esc(fmt(d.variant))} ${esc(d.unit)}</td><td>${d.delta !== undefined ? esc(fmt(d.delta)) : ""}</td>
          <td class="${d.pct > 0 ? "text-success" : d.pct < 0 ? "text-danger" : ""}">${typeof d.pct === "number" ? d.pct.toFixed(1) + "%" : ""}</td></tr>`).join("")}</tbody></table></div></div>
        ${numeric.length ? `<div class="lab-card"><h6>Relative change per metric</h6><div class="chart-box"><canvas id="bm-chart"></canvas></div></div>` : ""}
        <div class="lab-card"><h6>Variant metrics</h6>${metricCards(last.variant.metrics, last.baseline)}</div>`;
      if (numeric.length) renderChart(out.querySelector("#bm-chart"), { kind: "bar", labels: numeric.map((d) => d.label.slice(0, 30)), y_label: "% change", datasets: [{ label: "Δ %", data: numeric.map((d) => d.pct) }] });
      view.querySelector("#bm-export").disabled = false;
      if (last.experiment_id) toast(`Benchmark saved (#${last.experiment_id})`, "ok");
    } catch (e) { out.innerHTML = `<div class="alert alert-danger">${esc(e.message)}</div>`; }
  });
  view.querySelectorAll("[data-exp]").forEach((b) => b.addEventListener("click", () => {
    if (!last) return;
    if (b.dataset.exp === "json") download(`${mod.id}-ab.json`, JSON.stringify({ module: mod.id, fidelity: last.fidelity, changed: last.changed, deltas: last.deltas, baseline_params: last.baseline.params, variant_params: last.variant.params }, null, 2));
    else download(`${mod.id}-ab.csv`, ["metric,unit,baseline,variant,delta,pct", ...last.deltas.map((d) => [d.label, d.unit, d.baseline, d.variant, d.delta ?? "", d.pct ?? ""].map((x) => `"${String(x).replace(/"/g, '""')}"`).join(","))].join("\n"), "text/csv");
  }));

  view.querySelector("#sw-run").addEventListener("click", async () => {
    const key = view.querySelector("#sw-key").value;
    const c = mod.controls.find((x) => x.key === key);
    const lo = Number(view.querySelector("#sw-from").value), hi = Number(view.querySelector("#sw-to").value);
    const n = Math.max(2, Math.min(25, Number(view.querySelector("#sw-n").value) || 8));
    let values = [...Array(n)].map((_, i) => (c.log && lo > 0 ? Math.exp(Math.log(lo) + (i / (n - 1)) * (Math.log(hi) - Math.log(lo))) : lo + (i / (n - 1)) * (hi - lo)));
    values = [...new Set(values.map((v) => Math.min(c.max, Math.max(c.min, Number.isInteger(c.default) && c.step >= 1 ? Math.round(v / c.step) * c.step : Number(v.toFixed(6))))))];
    view.querySelector("#sw-note").innerHTML = `<span class="spinner-dot"></span>Running ${values.length} configurations…`;
    try {
      const res = await post("/api/benchmarks/sweep", { module_id: mod.id, params: A.get(), sweep_key: key, values });
      const msel = view.querySelector("#sw-metric");
      const numericKeys = Object.keys(res.rows[0].metrics).filter((k) => res.rows.every((r) => typeof r.metrics[k] === "number"));
      const prev = msel.value;
      msel.innerHTML = numericKeys.map((k) => `<option value="${k}">${esc(res.rows[0].labels[k])}</option>`).join("");
      if (numericKeys.includes(prev)) msel.value = prev;
      const draw = () => renderChart(view.querySelector("#sw-chart"), { kind: "line", labels: res.rows.map((r) => fmt(r.value)), x_label: c.label, y_label: res.rows[0].labels[msel.value],
        datasets: [{ label: res.rows[0].labels[msel.value], data: res.rows.map((r) => r.metrics[msel.value]), pointRadius: 3 }] });
      msel.onchange = draw;
      draw();
      view.querySelector("#sw-note").innerHTML = `${fidelityBadge(res.fidelity)} sweep of <b>${esc(c.label)}</b> over ${values.length} values (other parameters from configuration A).`;
    } catch (e) { view.querySelector("#sw-note").textContent = e.message; }
  });

  view.querySelector("#lv-run").addEventListener("click", async () => {
    const out = view.querySelector("#lv-out");
    out.innerHTML = `<span class="spinner-dot"></span>Measuring…`;
    try {
      const r = await post("/api/benchmarks/live", { prompt: view.querySelector("#lv-prompt").value, runs: Number(view.querySelector("#lv-runs").value) || 3 });
      if (!r.runs) { out.innerHTML = `<div class="alert alert-warning py-2 mb-0">${esc(r.message)}</div>`; return; }
      out.innerHTML = `${fidelityBadge(r.measured ? "real" : "simulated")} mean ${fmt(r.mean_ms)} ms ± ${fmt(r.stdev_ms)} · ${esc(r.note)}
        <table class="table table-sm table-lab mt-2"><thead><tr><th>Run</th><th>Wall ms</th><th>Out tokens</th><th>tok/s</th><th>Provider</th></tr></thead><tbody>
        ${r.runs.map((x) => `<tr><td>${x.run}</td><td>${fmt(x.wall_ms)}</td><td>${x.output_tokens ?? "?"}</td><td>${x.tokens_per_s ? fmt(x.tokens_per_s) : "?"}</td><td>${esc(x.provider)}</td></tr>`).join("")}</tbody></table>`;
    } catch (e) { out.textContent = e.message; }
  });
  await loadModule();
}
