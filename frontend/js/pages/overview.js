import { esc, fmt, get, state } from "../api.js";
import { SECTIONS, setCrumbs } from "../app.js";
import { renderChart } from "../components/charts.js";
import { fidelityBadge, flowHTML, metricCards } from "../components/render.js";

export async function renderOverview(view) {
  setCrumbs("Overview");
  const d = await get("/api/dashboard/summary");
  const e = d.estimate;
  const metrics = [
    { key: "model", label: "Active model", value: d.active_model.label, hint: `${d.hardware.label} · ${d.dtype.toUpperCase()}` },
    { key: "params", label: "Parameters configured", value: d.active_model.params, hint: d.active_model.active_params !== d.active_model.params ? `${d.active_model.active_params} active / token` : `${d.active_model.layers} layers · d=${d.active_model.hidden}` },
    { key: "ctx", label: "Context window", value: d.active_model.context_window, unit: "tok" },
    { key: "in", label: "Input tokens", value: d.input_tokens, unit: "tok" },
    { key: "out", label: "Output tokens", value: d.output_tokens, unit: "tok" },
    { key: "lat", label: "Estimated latency", value: e.total_ms / 1000, unit: "s", hint: `TTFT ${fmt(e.ttft_ms)} ms · TPOT ${fmt(e.tpot_ms)} ms` },
    { key: "mem", label: "Estimated memory", value: e.total_gb, unit: "GB", hint: e.fits ? "fits on device" : "does NOT fit on device", status: e.fits ? "good" : "bad" },
    { key: "tps", label: "Throughput", value: e.tokens_per_s, unit: "tok/s", hint: `${e.decode_bound}-bound decode` },
    { key: "cost", label: "Estimated inference cost", value: e.cost_per_1k_tokens, unit: "$/1K tok", hint: `at $${d.hardware.usd_hr}/h (editable)` },
    ...Object.entries(d.quality).map(([k, v]) => ({ key: k, label: `Quality: ${k}`, value: v, hint: "Model Evaluation (synthetic set)" })),
  ];
  view.innerHTML = `
    <div class="overview-hero"><div><div class="page-title">AI Engineering Dashboard</div>
      <div class="page-sub">Welcome, ${esc(state.user.full_name || state.user.email)}. ${d.module_total} interactive modules · provider: <b>${esc(d.provider.name)}</b>${d.provider.real ? "" : " (deterministic simulation — no paid API needed)"}.</div></div>
      <div class="ms-auto">${fidelityBadge(d.fidelity)}</div></div>
    <div class="fidelity-banner mb-3">${fidelityBadge("analytical")}<div>${esc(d.fidelity_note)} Change the active model, hardware and efficiency assumptions in <a href="#/settings">Settings</a>.</div></div>
    ${metricCards(metrics)}
    <div class="row g-3 mt-1">
      <div class="col-xl-8"><div class="lab-card"><h6>Throughput vs. per-token latency as batch grows — ${esc(d.active_model.label)} on ${esc(d.hardware.label)}</h6><div class="chart-box"><canvas id="ov-batch"></canvas></div></div></div>
      <div class="col-xl-4"><div class="lab-card"><h6>Modules by category</h6><div class="chart-box"><canvas id="ov-cats"></canvas></div></div></div>
    </div>
    <div class="lab-card mt-3"><h6>Reference architecture of a production LLM application (and where each lab module fits)</h6>
      ${flowHTML([{ label: "Client", sub: "chat / API / agent" }, { label: "Gateway", sub: "AuthN/Z · PII masking · injection screen", cls: "accent" },
        { label: "Orchestrator", sub: "prompt engineering · tools" }, { label: "Retrieval", sub: "hybrid search · rerank · RAG", cls: "accent" },
        { label: "Inference server", sub: "continuous batching · paged KV · prefix cache", cls: "accent" }, { label: "Model", sub: "GQA · RoPE · MoE · quantized" },
        { label: "Evaluation & audit", sub: "faithfulness · calibration · logs" }])}
    </div>
    <div class="row g-3 mt-1" id="ov-sections"></div>
    <div class="row g-3 mt-1">
      <div class="col-lg-6"><div class="lab-card"><h6>Recent experiments</h6><div id="ov-recent"></div></div></div>
      <div class="col-lg-6"><div class="lab-card"><h6>Your workspace</h6>
        <div class="metric-grid">${[["Experiments", d.counts.experiments, "#/history"], ["Saved configurations", d.counts.configs, "#/history"], ["Fine-tuning jobs", d.counts.fine_tune_jobs, "#/fine-tuning"]]
          .map(([l, v, h]) => `<a href="${h}" class="metric-card text-decoration-none text-reset"><div class="label">${l}</div><div class="value">${v}</div></a>`).join("")}</div>
        <p class="small text-muted mt-3 mb-0">Simulation fidelity legend: ${["analytical", "computed", "simulated", "real"].map(fidelityBadge).join(" ")}</p></div></div>
    </div>`;
  renderChart(view.querySelector("#ov-batch"), {
    kind: "line", labels: d.batch_curve.map((b) => b.batch), x_label: "batch size", y_label: "tokens / s", dual_axis: true,
    datasets: [{ label: "throughput (tok/s)", data: d.batch_curve.map((b) => b.tokens_per_s) }, { label: "TPOT (ms)", data: d.batch_curve.map((b) => b.tpot_ms), yAxisID: "y1" }],
  });
  renderChart(view.querySelector("#ov-cats"), { kind: "doughnut", labels: Object.keys(d.modules_by_category), datasets: [{ label: "modules", data: Object.values(d.modules_by_category) }] });
  view.querySelector("#ov-sections").innerHTML = Object.entries(SECTIONS).map(([k, s]) => `
    <div class="col-md-6 col-xl-3"><a class="lab-card module-card d-block text-decoration-none text-reset" href="${k === "training" ? "#/fine-tuning" : "#/section/" + k}">
      <div class="fw-semibold mb-1">${s.icon} ${esc(s.title)}</div><div class="small text-muted mb-2">${esc(s.blurb)}</div>
      <div class="small">${s.modules.map((id) => esc(state.modules.find((m) => m.id === id)?.title || id)).join(" · ")}</div></a></div>`).join("");
  view.querySelector("#ov-recent").innerHTML = d.recent.length ? `<table class="table table-sm table-lab"><tbody>${d.recent.map((r) => `<tr><td><a href="#/module/${esc(r.module_id)}">${esc(r.name)}</a></td><td>${esc(r.kind)}</td><td>${fidelityBadge(r.mode)}</td><td class="text-muted">${new Date(r.created_at).toLocaleString()}</td></tr>`).join("")}</tbody></table>`
    : `<p class="text-muted small mb-0">No experiments yet. Open any module and press <b>Run Experiment</b>.</p>`;
}
