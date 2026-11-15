import { download, esc, fmt, post, state } from "../api.js";
import { setCrumbs } from "../app.js";
import { renderChart } from "../components/charts.js";
import { fidelityBadge } from "../components/render.js";

const MODELS = [["qwen2.5-0.5b", "Qwen2.5-0.5B"], ["tinyllama-1.1b", "TinyLlama-1.1B"], ["phi-3-mini", "Phi-3-mini"], ["llama-2-7b", "Llama-2-7B"], ["mistral-7b", "Mistral-7B"],
  ["llama-3-8b", "Llama-3-8B"], ["llama-2-13b", "Llama-2-13B"], ["mixtral-8x7b", "Mixtral-8x7B"], ["llama-3-70b", "Llama-3-70B"]];
const DEFAULT = ["tinyllama-1.1b", "mistral-7b", "llama-3-8b", "mixtral-8x7b", "llama-3-70b"];

export async function renderCompare(view) {
  setCrumbs("Model Comparison");
  const hw = state.hardwareProfiles || {};
  view.innerHTML = `
    <div class="page-title">Model Comparison</div>
    <div class="page-sub">Compare model architectures on memory, latency, throughput and cost for a given GPU, precision and workload.</div>
    <div class="lab-card mb-3"><div class="row g-2 align-items-end">
      <div class="col-lg-5"><label class="small text-muted">Models</label><div class="multiselect" id="mc-models">${MODELS.map(([k, l]) => `<label><input type="checkbox" value="${k}" ${DEFAULT.includes(k) ? "checked" : ""}><span>${l}</span></label>`).join("")}</div></div>
      <div class="col-md-2"><label class="small text-muted">GPU</label><select class="form-select form-select-sm" id="mc-hw">${Object.entries(hw).map(([k, v]) => `<option value="${k}" ${k === "a100_80" ? "selected" : ""}>${esc(v.label)}</option>`).join("")}</select></div>
      <div class="col-md-1"><label class="small text-muted">Precision</label><select class="form-select form-select-sm" id="mc-dtype">${["fp16", "bf16", "int8", "int4"].map((d) => `<option ${d === "bf16" ? "selected" : ""}>${d}</option>`).join("")}</select></div>
      <div class="col-md-1"><label class="small text-muted">Prompt</label><input class="form-control form-control-sm" id="mc-in" type="number" value="1024" min="1" max="131072"></div>
      <div class="col-md-1"><label class="small text-muted">Output</label><input class="form-control form-control-sm" id="mc-out" type="number" value="256" min="1" max="16384"></div>
      <div class="col-md-1"><label class="small text-muted">Batch</label><input class="form-control form-control-sm" id="mc-b" type="number" value="1" min="1" max="512"></div>
      <div class="col-md-1 d-flex gap-1"><button class="btn btn-primary btn-sm" id="mc-run">Compare</button><button class="btn btn-outline-secondary btn-sm" id="mc-csv" title="Export CSV">⤓</button></div>
    </div></div>
    <div id="mc-out-area"></div>`;
  let last = null;
  const run = async () => {
    const models = [...view.querySelectorAll("#mc-models input:checked")].map((i) => i.value);
    const body = { models, hardware: view.querySelector("#mc-hw").value, dtype: view.querySelector("#mc-dtype").value,
      prompt_tokens: Number(view.querySelector("#mc-in").value), output_tokens: Number(view.querySelector("#mc-out").value), batch: Number(view.querySelector("#mc-b").value) };
    const r = await post("/api/benchmarks/models", body);
    last = r;
    const out = view.querySelector("#mc-out-area");
    out.innerHTML = `<div class="fidelity-banner mb-3">${fidelityBadge(r.fidelity)}<div>${esc(r.fidelity_note)}</div></div>
      <div class="lab-card"><h6>${esc(r.hardware.label)} · ${esc(r.dtype.toUpperCase())}</h6><div class="table-wrap"><table class="table table-sm table-lab"><thead><tr>
      <th>Model</th><th>Params</th><th>Active</th><th>KV heads</th><th>Context</th><th>Weights GB</th><th>KV GB</th><th>Total GB</th><th>Fits?</th><th>TTFT ms</th><th>TPOT ms</th><th>tok/s</th><th>$/1K tok</th></tr></thead><tbody>
      ${r.rows.map((x) => `<tr><td>${esc(x.label)}</td><td>${x.params}</td><td>${x.active_params}</td><td>${x.kv_heads}</td><td>${x.context.toLocaleString()}</td><td>${fmt(x.weights_gb)}</td><td>${fmt(x.kv_gb)}</td><td>${fmt(x.total_gb)}</td>
      <td class="${x.fits ? "text-success" : "text-danger"}">${x.fits ? "yes" : "no"}</td><td>${fmt(x.ttft_ms)}</td><td>${fmt(x.tpot_ms)}</td><td>${fmt(x.tokens_per_s)}</td><td>${fmt(x.cost_per_1k_tokens)}</td></tr>`).join("")}</tbody></table></div></div>
      <div class="result-grid mt-3">
        <div class="lab-card"><h6>Memory (weights + KV)</h6><div class="chart-box"><canvas id="mc-c1"></canvas></div></div>
        <div class="lab-card"><h6>Per-token latency (TPOT)</h6><div class="chart-box"><canvas id="mc-c2"></canvas></div></div>
        <div class="lab-card"><h6>Throughput</h6><div class="chart-box"><canvas id="mc-c3"></canvas></div></div>
        <div class="lab-card"><h6>Cost per 1K tokens</h6><div class="chart-box"><canvas id="mc-c4"></canvas></div></div></div>`;
    const labels = r.rows.map((x) => x.label);
    renderChart(out.querySelector("#mc-c1"), { kind: "bar", labels, stacked: true, y_label: "GB", threshold: r.hardware.mem_gb,
      datasets: [{ label: "weights", data: r.rows.map((x) => x.weights_gb) }, { label: "KV cache", data: r.rows.map((x) => x.kv_gb) }] });
    renderChart(out.querySelector("#mc-c2"), { kind: "bar", labels, y_label: "ms", datasets: [{ label: "TPOT", data: r.rows.map((x) => x.tpot_ms) }] });
    renderChart(out.querySelector("#mc-c3"), { kind: "bar", labels, y_label: "tok/s", datasets: [{ label: "throughput", data: r.rows.map((x) => x.tokens_per_s) }] });
    renderChart(out.querySelector("#mc-c4"), { kind: "bar", labels, y_label: "$", datasets: [{ label: "$/1K tokens", data: r.rows.map((x) => x.cost_per_1k_tokens) }] });
  };
  view.querySelector("#mc-run").addEventListener("click", run);
  view.querySelector("#mc-csv").addEventListener("click", () => {
    if (!last) return;
    const cols = ["label", "params", "active_params", "weights_gb", "kv_gb", "total_gb", "fits", "ttft_ms", "tpot_ms", "tokens_per_s", "cost_per_1k_tokens"];
    download("model-comparison.csv", [cols.join(","), ...last.rows.map((x) => cols.map((c) => x[c]).join(","))].join("\n"), "text/csv");
  });
  await run();
}
