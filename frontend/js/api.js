// Thin API client: cookie session + CSRF double-submit header, uniform error handling.
function cookie(name) {
  return document.cookie.split("; ").find((c) => c.startsWith(name + "="))?.split("=")[1];
}

export class ApiError extends Error {
  constructor(status, body) {
    super(body?.error?.message || `HTTP ${status}`);
    this.status = status;
    this.body = body;
  }
}

export async function api(path, { method = "GET", body, form, raw = false } = {}) {
  const headers = {};
  if (method !== "GET") headers["X-CSRF-Token"] = decodeURIComponent(cookie("llmlab_csrf") || "");
  let payload;
  if (form) payload = form;
  else if (body !== undefined) { headers["Content-Type"] = "application/json"; payload = JSON.stringify(body); }
  const r = await fetch(path, { method, headers, body: payload, credentials: "same-origin" });
  if (r.status === 401) { window.location.href = "/login"; throw new ApiError(401, {}); }
  if (raw) { if (!r.ok) throw new ApiError(r.status, await r.json().catch(() => ({}))); return r; }
  const data = await r.json().catch(() => ({}));
  if (!r.ok) throw new ApiError(r.status, data);
  return data;
}

export const get = (p) => api(p);
export const post = (p, body) => api(p, { method: "POST", body });
export const put = (p, body) => api(p, { method: "PUT", body });
export const patch = (p, body) => api(p, { method: "PATCH", body });
export const del = (p) => api(p, { method: "DELETE" });

// ---------------------------------------------------------------- UI helpers
export function toast(msg, kind = "info") {
  let c = document.querySelector(".toast-container");
  if (!c) { c = document.createElement("div"); c.className = "toast-container position-fixed bottom-0 end-0 p-3"; document.body.append(c); }
  const el = document.createElement("div");
  const bg = { info: "text-bg-primary", ok: "text-bg-success", error: "text-bg-danger", warn: "text-bg-warning" }[kind] || "text-bg-primary";
  el.className = `toast align-items-center border-0 ${bg}`;
  el.setAttribute("role", "status");
  el.innerHTML = `<div class="d-flex"><div class="toast-body"></div><button type="button" class="btn-close btn-close-white me-2 m-auto" data-bs-dismiss="toast"></button></div>`;
  el.querySelector(".toast-body").textContent = msg;
  c.append(el);
  const t = new bootstrap.Toast(el, { delay: 3500 });
  t.show();
  el.addEventListener("hidden.bs.toast", () => el.remove());
}

export function esc(s) {
  return String(s ?? "").replace(/[&<>"']/g, (c) => ({ "&": "&amp;", "<": "&lt;", ">": "&gt;", '"': "&quot;", "'": "&#39;" }[c]));
}

export function fmt(v, unit = "") {
  if (v === null || v === undefined) return "—";
  if (typeof v === "number") {
    const a = Math.abs(v);
    let s;
    if (Number.isInteger(v) && a < 1e6) s = v.toLocaleString();
    else if (a >= 1e15) s = v.toExponential(2);
    else if (a >= 1e6) { const [d, u] = a >= 1e12 ? [1e12, "T"] : a >= 1e9 ? [1e9, "B"] : [1e6, "M"]; s = (v / d).toFixed(2) + u; }
    else if (a >= 100) s = v.toFixed(1);
    else if (a >= 1) s = v.toFixed(2);
    else if (a === 0) s = "0";
    else if (a >= 0.001) s = v.toFixed(4);
    else s = v.toExponential(2);
    return s;
  }
  return String(v);
}

export function debounce(fn, ms) {
  let t;
  return (...a) => { clearTimeout(t); t = setTimeout(() => fn(...a), ms); };
}

export function download(filename, content, type = "application/json") {
  const blob = new Blob([content], { type });
  const a = document.createElement("a");
  a.href = URL.createObjectURL(blob);
  a.download = filename;
  a.click();
  setTimeout(() => URL.revokeObjectURL(a.href), 1000);
}

export function resultToCSV(result) {
  const rows = [["section", "label", "value", "unit"]];
  (result.metrics || []).forEach((m) => rows.push(["metric", m.label, m.value, m.unit || ""]));
  (result.comparison?.rows || []).forEach((r) => rows.push(["comparison:" + result.comparison.before_label, r.metric, r.before, r.unit || ""],
    ["comparison:" + result.comparison.after_label, r.metric, r.after, r.unit || ""]));
  Object.entries(result.params || {}).forEach(([k, v]) => rows.push(["param", k, Array.isArray(v) ? v.join("|") : v, ""]));
  return rows.map((r) => r.map((c) => `"${String(c ?? "").replace(/"/g, '""')}"`).join(",")).join("\n");
}

export const state = { user: null, modules: [], categories: {}, settings: null };
export const can = (perm) => !!state.user?.permissions?.includes(perm);
