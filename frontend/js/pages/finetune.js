import { api, can, debounce, download, esc, fmt, get, post, state, toast } from "../api.js";
import { setCrumbs } from "../app.js";
import { renderChart } from "../components/charts.js";
import { renderControls } from "../components/controls.js";
import { fidelityBadge, metricCards } from "../components/render.js";

const CFG = [
  { key: "learning_rate", label: "Learning rate", type: "slider", default: 0.0002, min: 0.000001, max: 0.01, step: 0.000001, log: true, help: "Optimizer step size. LoRA ≈ 1e-4–3e-4, full FT ≈ 1e-5–5e-5." },
  { key: "epochs", label: "Epochs", type: "slider", default: 3, min: 1, max: 20, step: 1, help: "Passes over the training set." },
  { key: "batch_size", label: "Batch size", type: "slider", default: 8, min: 1, max: 128, step: 1, help: "Sequences per device step." },
  { key: "grad_accum", label: "Gradient accumulation", type: "slider", default: 1, min: 1, max: 64, step: 1, help: "Effective batch = batch × accumulation." },
  { key: "rank", label: "LoRA rank (r)", type: "slider", default: 16, min: 1, max: 256, step: 1, help: "Adapter rank (LoRA/QLoRA/PEFT)." },
  { key: "alpha", label: "LoRA alpha (α)", type: "slider", default: 32, min: 1, max: 512, step: 1, help: "Scaling α/r." },
  { key: "dropout", label: "LoRA dropout", type: "slider", default: 0.05, min: 0, max: 0.5, step: 0.01, help: "Adapter dropout." },
  { key: "weight_decay", label: "Weight decay", type: "slider", default: 0.0, min: 0, max: 0.3, step: 0.01, help: "L2-style regularization (AdamW)." },
  { key: "max_seq_len", label: "Maximum sequence length", type: "slider", default: 1024, min: 64, max: 8192, step: 64, help: "Truncation length in tokens." },
  { key: "quantization", label: "Base-model quantization", type: "select", default: "none", options: [{ value: "none", label: "None (BF16)" }, { value: "int8", label: "INT8" }, { value: "nf4", label: "NF4 4-bit (QLoRA)" }], help: "Quantize frozen base weights." },
  { key: "target_all_linear", label: "Target all linear layers", type: "toggle", default: true, help: "Otherwise only q_proj and v_proj." },
  { key: "seed", label: "Random seed", type: "slider", default: 0, min: 0, max: 9999, step: 1, help: "Reproducibility." },
];
const METHODS = [["lora", "LoRA", "Low-rank adapters on frozen BF16 weights."], ["qlora", "QLoRA", "LoRA on a frozen 4-bit NF4 base."],
  ["full", "Full fine-tuning", "Update every weight (16 B/param of training memory)."], ["peft", "PEFT (other adapters)", "Generic adapter/prefix methods via PEFT."]];

export async function renderFineTuning(view) {
  setCrumbs("Fine-Tuning Laboratory");
  const env = await get("/api/fine-tuning/environment");
  const realOK = env.environment.real_training_possible;
  let dataset = null, method = "lora", ws = null, jobId = null, lastMetrics = null;
  view.innerHTML = `
    <div class="page-title">Fine-Tuning Laboratory</div>
    <div class="page-sub">End-to-end workflow: base model → dataset → validation → method → configuration → training → metrics → adapter → base vs fine-tuned comparison.
      Simulation mode needs no GPU; real mode trains a LoRA adapter on <span class="kbd">${esc(env.real_model)}</span> with Hugging Face PEFT when the ML extras are installed.</div>
    ${can("fine_tune") ? "" : '<div class="alert alert-warning">Your role cannot start fine-tuning jobs (requires AI Engineer or Administrator).</div>'}
    <div class="row g-3">
      <div class="col-xl-5">
        <div class="lab-card stepper">
          <div class="step active" data-step="1"><div class="step-num">1</div><div class="flex-grow-1"><div class="fw-semibold">Select base model</div>
            <select class="form-select form-select-sm mt-1" id="ft-model">${env.models.map((m) => `<option value="${m.value}" ${m.value === "llama-3-8b" ? "selected" : ""}>${esc(m.label)}</option>`).join("")}</select></div></div>
          <hr>
          <div class="step" data-step="2"><div class="step-num">2</div><div class="flex-grow-1"><div class="fw-semibold">Upload training dataset</div>
            <div class="small text-muted">JSONL, CSV or JSON · schemas: ${Object.values(env.schemas).map(esc).join(" · ")}</div>
            <div class="input-group input-group-sm mt-1"><input type="file" class="form-control" id="ft-file" accept=".jsonl,.json,.csv">
              <button class="btn btn-outline-primary" id="ft-upload" ${can("fine_tune") ? "" : "disabled"}>Validate</button></div>
            <button class="btn btn-link btn-sm px-0" id="ft-sample" ${can("fine_tune") ? "" : "disabled"}>Use sample support-ticket dataset (20 examples)</button></div></div>
          <hr>
          <div class="step" data-step="3"><div class="step-num">3</div><div class="flex-grow-1"><div class="fw-semibold">Validate dataset</div><div id="ft-validation" class="small text-muted">Upload a file to see the validation report.</div></div></div>
          <hr>
          <div class="step" data-step="4"><div class="step-num">4</div><div class="flex-grow-1"><div class="fw-semibold">Fine-tuning approach</div>
            <div class="d-grid gap-1 mt-1" id="ft-methods">${METHODS.map(([k, l, d]) => `<label class="lab-card p-2 mb-0" style="cursor:pointer"><input type="radio" name="ft-method" value="${k}" ${k === "lora" ? "checked" : ""}>
              <b class="ms-1">${l}</b> <span class="small text-muted">— ${d}</span></label>`).join("")}</div></div></div>
          <hr>
          <div class="step" data-step="5"><div class="step-num">5</div><div class="flex-grow-1"><div class="fw-semibold">Configure</div><div id="ft-config"></div></div></div>
        </div>
      </div>
      <div class="col-xl-7">
        <div class="lab-card"><h6>Training plan (estimate) <span class="ms-1">${fidelityBadge("analytical")}</span></h6><div id="ft-plan" class="small text-muted">Validate a dataset to compute the plan.</div>
          <div id="ft-realnote" class="alert alert-info py-1 small mt-2 mb-0 d-none">Real mode trains <b>${esc(env.real_model)}</b> (LLMLAB_HF_MODEL) on this machine with LoRA; the plan above is an estimate for the selected base model on the hardware in Settings. Real metrics appear in steps 7–9.</div></div>
        <div class="lab-card stepper"><div class="step" data-step="6"><div class="step-num">6</div><div class="flex-grow-1"><div class="fw-semibold">Execute training</div>
          <div class="d-flex flex-wrap gap-2 align-items-center mt-1">
            <select class="form-select form-select-sm w-auto" id="ft-mode"><option value="simulation">Simulation mode (no GPU)</option><option value="real" ${realOK ? "" : "disabled"}>Real training (${esc((env.environment.training_device || "cpu").toUpperCase())}, ${esc(env.real_model)})${realOK ? "" : " — unavailable"}</option></select>
            <button class="btn btn-primary btn-sm" id="ft-start" disabled>▶ Start training</button><button class="btn btn-outline-danger btn-sm" id="ft-cancel" disabled>■ Cancel</button>
            <span class="small text-muted" id="ft-status"></span></div>
          ${realOK ? "" : `<div class="small text-muted mt-1">Real training needs <span class="kbd">pip install -r requirements-ml.txt</span> and <span class="kbd">LLMLAB_ENABLE_HF_MODELS=true</span>.</div>`}
          <div class="progress mt-2" style="height:8px"><div class="progress-bar" id="ft-progress" style="width:0%"></div></div></div></div></div>
        <div class="lab-card"><div class="d-flex"><h6>7 · Training metrics</h6><span class="ms-auto" id="ft-fid"></span></div>
          <div class="metric-grid mb-2" id="ft-live"></div><div class="chart-box tall"><canvas id="ft-loss"></canvas></div></div>
        <div class="lab-card"><h6>8 · Save adapter & experiment</h6><div id="ft-save" class="small text-muted">Available when training completes.</div></div>
        <div class="lab-card"><h6>9 · Base model vs fine-tuned model</h6><div id="ft-compare" class="small text-muted">Available when training completes.</div></div>
        <div class="lab-card"><h6>Previous jobs</h6><div id="ft-jobs"></div></div>
      </div>
    </div>`;
  const $ = (s) => view.querySelector(s);
  const step = (n, st) => view.querySelectorAll(".step").forEach((el) => { const k = Number(el.dataset.step); if (k === n) el.className = `step ${st}`; });
  const ctl = renderControls($("#ft-config"), CFG, null, () => planDebounced(), { compact: true, idPrefix: "ft" });
  view.querySelectorAll('input[name="ft-method"]').forEach((r) => r.addEventListener("change", () => {
    method = r.value;
    if (method === "qlora") ctl.set({ quantization: "nf4" });
    step(4, "done");
    planDebounced();
  }));
  $("#ft-model").addEventListener("change", () => { step(1, "done"); planDebounced(); });
  $("#ft-mode").addEventListener("change", () => $("#ft-realnote").classList.toggle("d-none", $("#ft-mode").value !== "real"));

  const body = () => ({ base_model: $("#ft-model").value, dataset_id: dataset?.dataset_id, method, mode: $("#ft-mode").value, ...ctl.get() });
  async function plan() {
    if (!dataset) return;
    try {
      const { plan: p } = await post("/api/fine-tuning/plan", body());
      $("#ft-plan").innerHTML = metricCards([
        { key: "tp", label: "Trainable parameters", value: p.trainable_params, hint: `of ${fmt(p.total_params)} (${(p.trainable_params / p.total_params * 100).toFixed(3)}%)` },
        { key: "mem", label: "Training memory", value: p.memory_gb, unit: "GB", status: p.fits ? "good" : "bad", hint: `${p.fits ? "fits" : "does NOT fit"} on ${p.hardware}` },
        { key: "steps", label: "Optimizer steps", value: p.steps },
        { key: "time", label: "Est. training time", value: p.est_train_seconds / 60, unit: "min", hint: "roofline at your MFU setting" },
        { key: "tok", label: "Training tokens", value: p.tokens }]);
      step(5, "done");
    } catch (e) { $("#ft-plan").textContent = e.message; }
  }
  const planDebounced = debounce(plan, 300);

  function showValidation(v) {
    dataset = v;
    step(2, "done");
    step(3, v.valid ? "done" : "active");
    $("#ft-validation").innerHTML = `
      <div class="mb-1">${v.valid ? '<span class="badge text-bg-success">valid</span>' : '<span class="badge text-bg-warning">valid with warnings</span>'}
        <b>${esc(v.filename)}</b> · schema <span class="kbd">${esc(v.schema)}</span> (${esc(v.schema_description)})</div>
      <table class="table table-sm table-lab mb-1"><tbody>
        <tr><td>Records</td><td>${v.total_records} total · ${v.valid_records} valid · ${v.invalid_records} invalid · ${v.duplicates_removed} duplicates removed</td></tr>
        <tr><td>Tokens (approx.)</td><td>mean ${fmt(v.tokens.mean)} · p50 ${v.tokens.p50} · p95 ${v.tokens.p95} · max ${v.tokens.max} · ${v.exceeding_max_seq_len} exceed ${v.max_seq_len}</td></tr>
        <tr><td>Split</td><td>${v.split.train} train / ${v.split.validation} validation</td></tr></tbody></table>
      ${v.errors.length ? `<details><summary>${v.errors.length} row errors</summary><ul>${v.errors.map((e) => `<li>row ${e.row}: ${esc(e.error)}</li>`).join("")}</ul></details>` : ""}
      <details><summary>Preview (normalized)</summary><pre class="code">${esc(JSON.stringify(v.preview, null, 2))}</pre></details>`;
    $("#ft-start").disabled = !can("fine_tune");
    plan();
  }
  $("#ft-upload").addEventListener("click", async () => {
    const f = $("#ft-file").files[0];
    if (!f) return toast("Choose a file first", "warn");
    const fd = new FormData();
    fd.append("file", f);
    fd.append("max_seq_len", String(ctl.get().max_seq_len));
    try { showValidation(await api("/api/fine-tuning/datasets/validate", { method: "POST", form: fd })); }
    catch (e) { $("#ft-validation").innerHTML = `<div class="text-danger">${esc(e.message)}</div>${e.body?.error?.details?.errors ? `<ul>${e.body.error.details.errors.slice(0, 10).map((x) => `<li>row ${x.row}: ${esc(x.error)}</li>`).join("")}</ul>` : ""}`; step(3, "active"); }
  });
  $("#ft-sample").addEventListener("click", async () => showValidation(await post("/api/fine-tuning/datasets/sample", {})));

  let chart = null, lastTrain = null, lastVal = null;
  const series = { train: [], val: [], steps: [] };
  function drawLoss(simulated) {
    chart = renderChart($("#ft-loss"), { kind: "line", labels: series.steps, x_label: "step", y_label: "loss",
      datasets: [{ label: "training loss", data: series.train }, { label: "validation loss", data: series.val }] });
    $("#ft-fid").innerHTML = fidelityBadge(simulated ? "simulated" : "real");
  }
  function onEvent(ev) {
    if (ev.type === "progress") {
      series.steps.push(ev.step); series.train.push(ev.train_loss); series.val.push(ev.val_loss ?? null);
      if (chart && series.steps.length % 3 && ev.step !== ev.total_steps) { chart.update("none"); } else drawLoss(ev.simulated);
      $("#ft-progress").style.width = `${Math.min(100, (ev.step / ev.total_steps) * 100)}%`;
      if (ev.train_loss !== null && ev.train_loss !== undefined) lastTrain = ev.train_loss;
      if (ev.val_loss !== null && ev.val_loss !== undefined) lastVal = ev.val_loss;
      $("#ft-live").innerHTML = metricCards([
        { key: "s", label: "Step", value: `${ev.step} / ${ev.total_steps}` }, { key: "tl", label: "Training loss", value: lastTrain },
        { key: "vl", label: "Validation loss", value: lastVal }, ...(ev.gpu_memory_gb ? [{ key: "g", label: "GPU memory (est.)", value: ev.gpu_memory_gb, unit: "GB" }] : []),
        ...(ev.elapsed_s_est ? [{ key: "t", label: "Training time (est.)", value: ev.elapsed_s_est / 60, unit: "min" }] : [])]);
    } else if (ev.type === "status") {
      $("#ft-status").textContent = `status: ${ev.status}${ev.mode ? ` (${ev.mode})` : ""}`;
      if (["completed", "failed", "cancelled"].includes(ev.status)) finish(ev);
    }
  }
  function finish(ev) {
    $("#ft-cancel").disabled = true;
    $("#ft-start").disabled = false;
    step(6, ev.status === "completed" ? "done" : "active");
    if (ev.status === "failed") { toast(ev.error || "Training failed", "error"); return; }
    lastMetrics = ev.metrics || {};
    if (!series.steps.length && lastMetrics.train_loss) {
      lastMetrics.train_loss.forEach((l, i) => { series.steps.push(i + 1); series.train.push(l); series.val.push(lastMetrics.val_loss?.[i] ?? null); });
      drawLoss(lastMetrics.simulated);
    }
    const sim = lastMetrics.simulated !== false;
    $("#ft-save").innerHTML = `<div>${fidelityBadge(sim ? "simulated" : "real")} Adapter written to <span class="kbd">${esc(lastMetrics.adapter_path || "data/adapters/job-" + jobId + (sim ? "-simulated" : ""))}</span>
      ${sim ? "(simulated run: adapter_config.json only, no weights)" : ""}</div>
      <div class="d-flex gap-2 mt-2"><button class="btn btn-sm btn-primary" id="ft-saveexp" ${can("save") ? "" : "disabled"}>Save experiment configuration</button>
      <button class="btn btn-sm btn-outline-secondary" id="ft-export">⤓ Export metrics JSON</button></div>`;
    $("#ft-saveexp")?.addEventListener("click", async () => {
      const e = await post("/api/experiments", { module_id: "lora", kind: "fine_tune", name: `Fine-tune job #${jobId} (${method}, ${sim ? "simulated" : "real"})`, params: body(),
        result: { fidelity: sim ? "simulated" : "real", metrics: [{ key: "final_val_loss", label: "Final validation loss", value: lastMetrics.final_val_loss },
          { key: "trainable", label: "Trainable parameters", value: lastMetrics.plan?.trainable_params ?? lastMetrics.trainable_params }], job_id: jobId } });
      toast(`Saved as experiment #${e.id}`, "ok");
    });
    $("#ft-export").addEventListener("click", () => download(`finetune-job-${jobId}.json`, JSON.stringify({ config: body(), metrics: lastMetrics }, null, 2)));
    const cmpTable = (lastMetrics.comparison || []).length ? `<table class="table table-sm table-lab"><thead><tr><th>Metric</th><th>Base model</th><th>Fine-tuned</th></tr></thead><tbody>
        ${lastMetrics.comparison.map(([k, a, b]) => `<tr><td>${esc(k)}</td><td>${fmt(a)}</td><td>${fmt(b)}</td></tr>`).join("")}</tbody></table>` : "";
    if (sim) {
      $("#ft-compare").innerHTML = `<div class="small mb-2">${fidelityBadge("simulated")} Values come from the simulator's loss model — they illustrate the comparison workflow, not a trained model.</div>${cmpTable}`;
    } else {
      $("#ft-compare").innerHTML = `<div class="small mb-2">${fidelityBadge("real")} ${esc(lastMetrics.model)} on ${esc(lastMetrics.device || "cpu")} · ${fmt(lastMetrics.trainable_params)} trainable / ${fmt(lastMetrics.total_params)} params · trained in ${fmt(lastMetrics.train_seconds)} s</div>
        ${cmpTable}<div class="small text-muted mb-1">Greedy generations on held-out validation prompts:</div>
        <table class="table table-sm table-lab"><thead><tr><th>Prompt</th><th>Base output</th><th>Fine-tuned output</th></tr></thead><tbody>
        ${(lastMetrics.samples || []).map((s) => `<tr><td>${esc(s.prompt.slice(0, 120))}</td><td>${esc(s.base)}</td><td>${esc(s.tuned)}</td></tr>`).join("")}</tbody></table>`;
    }
    step(7, "done"); step(8, "done"); step(9, "done");
    loadJobs();
  }
  $("#ft-start").addEventListener("click", async () => {
    if (!dataset) return;
    series.train = []; series.val = []; series.steps = []; lastTrain = null; lastVal = null;
    try {
      const j = await post("/api/fine-tuning/jobs", body());
      jobId = j.id;
      step(6, "active");
      $("#ft-start").disabled = true;
      $("#ft-cancel").disabled = false;
      $("#ft-status").textContent = `job #${j.id} queued`;
      ws?.close();
      ws = new WebSocket(`${location.protocol === "https:" ? "wss" : "ws"}://${location.host}/ws/fine-tuning/${j.id}`);
      ws.onmessage = (m) => onEvent(JSON.parse(m.data));
      ws.onerror = () => toast("Live progress connection failed — refresh the job list for results", "warn");
    } catch (e) { toast(e.message, "error"); }
  });
  $("#ft-cancel").addEventListener("click", () => jobId && post(`/api/fine-tuning/jobs/${jobId}/cancel`, {}));
  async function loadJobs() {
    const jobs = await get("/api/fine-tuning/jobs");
    $("#ft-jobs").innerHTML = jobs.length ? `<table class="table table-sm table-lab"><thead><tr><th>#</th><th>Model</th><th>Method</th><th>Mode</th><th>Status</th><th>Final val loss</th><th>Created</th></tr></thead><tbody>
      ${jobs.map((j) => `<tr><td>${j.id}</td><td>${esc(j.mode === "real" ? (j.metrics?.model || "(real model)") : j.config.base_model)}</td><td>${esc(j.config.method)}</td><td>${fidelityBadge(j.mode === "real" ? "real" : "simulated")}</td><td>${esc(j.status)}</td>
      <td>${fmt(j.metrics?.final_val_loss)}</td><td class="small text-muted">${new Date(j.created_at).toLocaleString()}</td></tr>`).join("")}</tbody></table>` : '<p class="small text-muted mb-0">No jobs yet.</p>';
  }
  loadJobs();
  void state;
  return () => ws?.close();
}
