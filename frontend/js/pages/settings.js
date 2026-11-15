import { can, esc, fmt, get, patch, post, put, state, toast } from "../api.js";
import { refreshBadges, setCrumbs } from "../app.js";
import { editor } from "../components/monaco.js";

const MODEL_LABELS = { "qwen2.5-0.5b": "Qwen2.5-0.5B", "tinyllama-1.1b": "TinyLlama-1.1B", "phi-3-mini": "Phi-3-mini (3.8B)", "llama-2-7b": "Llama-2-7B", "mistral-7b": "Mistral-7B",
  "llama-3-8b": "Llama-3-8B", "llama-2-13b": "Llama-2-13B", "mixtral-8x7b": "Mixtral-8x7B (MoE)", "llama-3-70b": "Llama-3-70B" };

export async function renderSettings(view) {
  setCrumbs("Settings");
  const { settings: s, hardware_profiles: hw } = await get("/api/settings");
  const ro = can("configure") ? "" : "disabled";
  view.innerHTML = `
    <div class="page-title">Settings</div>
    <div class="page-sub">Per-user configuration used by the dashboard and every analytical estimate. ${can("configure") ? "" : "<b>Your role is read-only.</b>"}</div>
    <div class="row g-3">
      <div class="col-xl-6">
        <div class="lab-card"><h6>Active model & hardware</h6>
          <div class="row g-2">
            <div class="col-md-6"><label class="form-label small">Active model</label><select class="form-select form-select-sm" id="s-model" ${ro}>${Object.entries(MODEL_LABELS).map(([k, v]) => `<option value="${k}">${v}</option>`).join("")}</select></div>
            <div class="col-md-6"><label class="form-label small">Hardware profile</label><select class="form-select form-select-sm" id="s-hw" ${ro}>${Object.entries(hw).map(([k, v]) => `<option value="${k}">${esc(v.label)}</option>`).join("")}</select></div>
            <div class="col-md-4"><label class="form-label small">Weight dtype</label><select class="form-select form-select-sm" id="s-dtype" ${ro}>${["fp32", "fp16", "bf16", "int8", "int4"].map((d) => `<option>${d}</option>`).join("")}</select></div>
            <div class="col-md-4"><label class="form-label small">Default input tokens</label><input class="form-control form-control-sm" type="number" id="s-in" min="1" max="131072" ${ro}></div>
            <div class="col-md-4"><label class="form-label small">Default output tokens</label><input class="form-control form-control-sm" type="number" id="s-out" min="1" max="131072" ${ro}></div>
            <div class="col-md-4"><label class="form-label small">Achieved bandwidth <span class="text-muted" id="s-me-v"></span></label><input type="range" class="form-range" id="s-me" min="0.05" max="1" step="0.05" ${ro}></div>
            <div class="col-md-4"><label class="form-label small">Achieved MFU <span class="text-muted" id="s-ce-v"></span></label><input type="range" class="form-range" id="s-ce" min="0.05" max="1" step="0.05" ${ro}></div>
            <div class="col-md-4"><label class="form-label small">$/GPU-hour override</label><input class="form-control form-control-sm" type="number" step="0.01" min="0" id="s-usd" placeholder="profile default" ${ro}></div>
          </div>
          <div class="small text-muted mt-2" id="s-hw-info"></div>
        </div>
        <div class="lab-card"><h6>Model provider</h6>
          <div class="row g-2">
            <div class="col-md-4"><label class="form-label small">Provider</label><select class="form-select form-select-sm" id="s-prov" ${ro}>
              <option value="simulation">Simulation (no model)</option><option value="ollama">Ollama (local)</option><option value="openai">OpenAI-compatible API</option><option value="huggingface">Hugging Face (local)</option></select></div>
            <div class="col-md-4"><label class="form-label small">Ollama model</label><input class="form-control form-control-sm" id="s-ollama" ${ro}></div>
            <div class="col-md-4"><label class="form-label small">OpenAI-compatible model</label><input class="form-control form-control-sm" id="s-oai-model" placeholder="server default" ${ro}></div>
            <div class="col-12"><label class="form-label small">OpenAI-compatible base URL</label><input class="form-control form-control-sm" id="s-oai-url" placeholder="server default (LLMLAB_OPENAI_BASE_URL)" ${ro}></div>
          </div>
          <p class="small text-muted mt-2 mb-2">API keys are never stored per user — set <span class="kbd">LLMLAB_OPENAI_API_KEY</span> on the server. Hugging Face models require <span class="kbd">LLMLAB_ENABLE_HF_MODELS=true</span>.</p>
          <button class="btn btn-sm btn-outline-secondary" id="s-check">Check provider status</button><div class="small mt-2" id="s-status"></div>
        </div>
        <div class="d-flex gap-2"><button class="btn btn-primary" id="s-save" ${ro}>Save settings</button><button class="btn btn-outline-secondary" id="s-defaults" ${ro}>Restore defaults</button></div>
      </div>
      <div class="col-xl-6">
        <div class="lab-card"><h6>Advanced: settings as JSON</h6><div class="monaco-host small" id="s-json"></div>
          <button class="btn btn-sm btn-outline-primary mt-2" id="s-json-apply" ${ro}>Apply JSON</button></div>
        <div class="lab-card"><h6>ML environment</h6><div id="s-env" class="small text-muted">Loading…</div></div>
        <div class="lab-card ${can("manage_users") ? "" : "d-none"}" id="s-admin"><h6>User management (Administrator)</h6>
          <div id="s-users"></div>
          <form class="row g-2 mt-2" id="s-newuser">
            <div class="col-md-4"><input class="form-control form-control-sm" name="email" placeholder="email" required></div>
            <div class="col-md-3"><input class="form-control form-control-sm" name="password" type="password" placeholder="password" required></div>
            <div class="col-md-3"><select class="form-select form-select-sm" name="role"><option value="viewer">Viewer</option><option value="researcher">Researcher</option><option value="engineer">AI Engineer</option><option value="admin">Administrator</option></select></div>
            <div class="col-md-2"><button class="btn btn-sm btn-primary w-100">Add</button></div>
          </form>
          <div class="small text-muted mt-1">Passwords need ≥10 chars with upper, lower, digit and symbol. Hashed with Argon2id.</div>
          <h6 class="mt-3">Audit log</h6><div id="s-audit" class="table-wrap"></div>
        </div>
      </div>
    </div>`;
  const $ = (id) => view.querySelector(id);
  let jsonEd;
  const fill = (x) => {
    $("#s-model").value = x.active_model; $("#s-hw").value = x.hardware; $("#s-dtype").value = x.dtype; $("#s-in").value = x.default_prompt_tokens; $("#s-out").value = x.default_output_tokens;
    $("#s-me").value = x.memory_efficiency; $("#s-ce").value = x.compute_efficiency; $("#s-usd").value = x.usd_per_hour_override ?? "";
    $("#s-prov").value = x.provider; $("#s-ollama").value = x.ollama_model || ""; $("#s-oai-model").value = x.openai_model || ""; $("#s-oai-url").value = x.openai_base_url || "";
    upd();
    jsonEd?.setValue(JSON.stringify(x, null, 2));
  };
  const upd = () => {
    $("#s-me-v").textContent = `${Math.round($("#s-me").value * 100)}%`; $("#s-ce-v").textContent = `${Math.round($("#s-ce").value * 100)}%`;
    const h = hw[$("#s-hw").value];
    $("#s-hw-info").textContent = `${h.label}: ${h.mem_gb} GB · ${h.bw_gbs} GB/s · ${h.tflops} dense FP16 TFLOPS · $${h.usd_hr}/h (assumption)`;
  };
  ["#s-me", "#s-ce", "#s-hw"].forEach((i) => $(i).addEventListener("input", upd));
  const collect = () => ({ active_model: $("#s-model").value, hardware: $("#s-hw").value, dtype: $("#s-dtype").value, default_prompt_tokens: Number($("#s-in").value),
    default_output_tokens: Number($("#s-out").value), memory_efficiency: Number($("#s-me").value), compute_efficiency: Number($("#s-ce").value),
    usd_per_hour_override: $("#s-usd").value === "" ? null : Number($("#s-usd").value), provider: $("#s-prov").value, ollama_model: $("#s-ollama").value,
    openai_model: $("#s-oai-model").value, openai_base_url: $("#s-oai-url").value });
  const save = async (body) => { const r = await put("/api/settings", body); fill(r.settings); state.settings = r.settings; await refreshBadges(); toast("Settings saved", "ok"); };
  $("#s-save").addEventListener("click", () => save(collect()));
  $("#s-defaults").addEventListener("click", async () => { const d = await get("/api/settings"); save(d.defaults); });
  fill(s);
  editor($("#s-json"), JSON.stringify(s, null, 2), { language: "json", readOnly: !can("configure") }).then((e) => { jsonEd = e; });
  $("#s-json-apply").addEventListener("click", () => { if (!jsonEd) return; try { save(JSON.parse(jsonEd.getValue())); } catch (e) { toast("Invalid JSON: " + e.message, "error"); } });
  $("#s-check").addEventListener("click", async () => {
    $("#s-status").textContent = "Checking…";
    const r = await get("/api/llm/status");
    const a = r.active;
    $("#s-status").innerHTML = `<b>${esc(a.provider)}</b>: ${a.available ? '<span class="text-success">available</span>' : '<span class="text-danger">unavailable</span>'}
      ${a.models ? ` · models: ${esc(a.models.join(", ") || "none pulled")}` : ""}${a.error ? ` · ${esc(a.error)}` : ""}${a.note ? ` · ${esc(a.note)}` : ""}`;
  });
  get("/api/llm/status").then((r) => {
    const e = r.environment;
    $("#s-env").innerHTML = `<table class="table table-sm table-lab mb-0"><tbody>${Object.entries(e).map(([k, v]) => `<tr><td>${esc(k)}</td><td>${v === null ? '<span class="text-muted">not installed</span>' : esc(String(v))}</td></tr>`).join("")}</tbody></table>
      <div class="mt-1">Install optional real-model support with <span class="kbd">pip install -r requirements-ml.txt</span>.</div>`;
  });
  if (can("manage_users")) {
    const loadUsers = async () => {
      const us = await get("/api/users");
      $("#s-users").innerHTML = `<table class="table table-sm table-lab"><thead><tr><th>Email</th><th>Role</th><th>Active</th></tr></thead><tbody>${us.map((u) => `<tr><td>${esc(u.email)}</td>
        <td><select class="form-select form-select-sm" data-role="${u.id}">${["viewer", "researcher", "engineer", "admin"].map((r) => `<option value="${r}" ${u.role === r ? "selected" : ""}>${r}</option>`).join("")}</select></td>
        <td><input type="checkbox" class="form-check-input" data-active="${u.id}" ${u.is_active ? "checked" : ""} ${u.id === state.user.id ? "disabled" : ""}></td></tr>`).join("")}</tbody></table>`;
      view.querySelectorAll("[data-role]").forEach((s_) => s_.addEventListener("change", async () => { await patch(`/api/users/${s_.dataset.role}`, { role: s_.value }); toast("Role updated", "ok"); }));
      view.querySelectorAll("[data-active]").forEach((c) => c.addEventListener("change", async () => { await patch(`/api/users/${c.dataset.active}`, { is_active: c.checked }); toast("Updated", "ok"); }));
      const au = await get("/api/audit?limit=40");
      $("#s-audit").innerHTML = `<table class="table table-sm table-lab"><thead><tr><th>Time</th><th>User</th><th>Action</th><th>Detail</th></tr></thead><tbody>${au.map((a) => `<tr><td class="small text-muted">${new Date(a.created_at).toLocaleString()}</td><td>${a.user_id ?? "—"}</td><td>${esc(a.action)}</td><td class="small mono">${esc(JSON.stringify(a.detail))}</td></tr>`).join("")}</tbody></table>`;
    };
    $("#s-newuser").addEventListener("submit", async (e) => {
      e.preventDefault();
      const f = new FormData(e.target);
      try { await post("/api/users", { email: f.get("email"), password: f.get("password"), role: f.get("role") }); toast("User created", "ok"); e.target.reset(); loadUsers(); }
      catch (ex) { toast(ex.message, "error"); }
    });
    loadUsers();
  }
  void fmt;
}
