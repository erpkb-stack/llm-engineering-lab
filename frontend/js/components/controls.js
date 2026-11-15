// Reusable parameter controls generated from the backend control specification.
import { esc, fmt } from "../api.js";

const LOG_STEPS = 1000;

function snap(v, c) {
  const step = c.step || 1;
  let out = Math.round((v - (c.min ?? 0)) / step) * step + (c.min ?? 0);
  out = Math.min(c.max, Math.max(c.min, out));
  const dec = (String(step).split(".")[1] || "").length;
  return Number(out.toFixed(Math.min(10, dec + 2)));
}
const toPos = (v, c) => (Math.log(v) - Math.log(c.min)) / (Math.log(c.max) - Math.log(c.min)) * LOG_STEPS;
const fromPos = (p, c) => snap(Math.exp(Math.log(c.min) + (p / LOG_STEPS) * (Math.log(c.max) - Math.log(c.min))), c);

function displayValue(c, v) {
  if (c.type === "toggle") return v ? "on" : "off";
  if (c.type === "slider" || c.type === "number") {
    if (c.key === "delta_exp") return `1e${v}`;
    if (c.log && v < 0.01) return Number(v).toExponential(1);
    return fmt(Number(v)) + (c.unit ? " " + c.unit : "");
  }
  return "";
}

/**
 * Render controls into `host`. Returns {get(), set(values), reset()}.
 * onChange(values, changedKey) is called on every user change.
 */
export function renderControls(host, controls, values, onChange, { compact = false, idPrefix = "c" } = {}) {
  host.innerHTML = "";
  const vals = { ...Object.fromEntries(controls.map((c) => [c.key, c.default])), ...(values || {}) };
  const setters = {};
  let lastGroup = null;
  controls.forEach((c) => {
    if (c.group && c.group !== lastGroup && !compact) {
      const g = document.createElement("div");
      g.className = "ctl-group-title";
      g.textContent = c.group;
      host.append(g);
      lastGroup = c.group;
    }
    const wrap = document.createElement("div");
    wrap.className = "ctl";
    const id = `${idPrefix}-${c.key}`;
    const info = c.help ? `<span class="info" data-bs-toggle="tooltip" data-bs-title="${esc(c.help)}" tabindex="0" aria-label="${esc(c.help)}">ⓘ</span>` : "";
    const reset = `<button class="reset" type="button" title="Reset to default (${esc(Array.isArray(c.default) ? c.default.join(", ") : c.default)})" aria-label="Reset ${esc(c.label)}">↺</button>`;
    const head = `<div class="ctl-head"><label for="${id}">${esc(c.label)}</label>${info}<span class="val" data-val></span>${reset}</div>`;
    let body = "";
    if (c.type === "slider" || c.type === "number") {
      const lg = c.log && c.min > 0;
      body = `<input type="range" class="form-range" id="${id}" ${lg ? `min="0" max="${LOG_STEPS}" step="1"` : `min="${c.min}" max="${c.max}" step="${c.step}"`}>
              <div class="ctl-range"><span>${fmt(c.min)}</span><span class="meta">default ${fmt(c.default)} · step ${c.step}${lg ? " · log scale" : ""}</span><span>${fmt(c.max)}</span></div>`;
    } else if (c.type === "select") {
      body = `<select class="form-select form-select-sm" id="${id}">${c.options.map((o) => `<option value="${esc(o.value)}">${esc(o.label)}</option>`).join("")}</select>`;
    } else if (c.type === "multiselect") {
      body = `<div class="multiselect" id="${id}">${c.options.map((o) => `<label><input type="checkbox" value="${esc(o.value)}"><span>${esc(o.label)}</span></label>`).join("")}</div>`;
    } else if (c.type === "toggle") {
      body = `<div class="form-check form-switch"><input class="form-check-input" type="checkbox" role="switch" id="${id}"></div>`;
    } else if (c.type === "textarea") {
      body = `<textarea class="form-control form-control-sm mono" id="${id}" rows="${compact ? 3 : 4}" maxlength="${c.max_length || 20000}"></textarea>`;
    } else {
      body = `<input class="form-control form-control-sm" id="${id}" type="text" maxlength="${c.max_length || 20000}">`;
    }
    wrap.innerHTML = head + body;
    if (c.type === "toggle") {  // toggle: put switch in header for compactness
      const sw = wrap.querySelector(".form-check");
      wrap.querySelector(".ctl-head").insertBefore(sw, wrap.querySelector("[data-val]"));
      sw.classList.add("mb-0", "ms-1");
    }
    host.append(wrap);
    const input = wrap.querySelector(`#${CSS.escape(id)}`);
    const valEl = wrap.querySelector("[data-val]");
    const lg = c.log && c.min > 0;

    const show = (v) => { valEl.textContent = displayValue(c, v); valEl.style.display = valEl.textContent ? "" : "none"; };
    const set = (v) => {
      vals[c.key] = v;
      if (c.type === "slider" || c.type === "number") input.value = lg ? toPos(Number(v), c) : v;
      else if (c.type === "toggle") input.checked = !!v;
      else if (c.type === "multiselect") input.querySelectorAll("input").forEach((i) => { i.checked = (v || []).includes(i.value); });
      else input.value = v ?? "";
      show(v);
    };
    setters[c.key] = set;
    set(vals[c.key]);

    const read = () => {
      if (c.type === "slider" || c.type === "number") return lg ? fromPos(Number(input.value), c) : snap(Number(input.value), c);
      if (c.type === "toggle") return input.checked;
      if (c.type === "multiselect") return [...input.querySelectorAll("input:checked")].map((i) => i.value);
      if (c.type === "select") { const o = c.options.find((o) => String(o.value) === input.value); return o ? o.value : input.value; }
      return input.value;
    };
    const evt = c.type === "slider" ? "input" : (c.type === "text" || c.type === "textarea") ? "input" : "change";
    input.addEventListener(evt, () => { const v = read(); vals[c.key] = v; show(v); onChange?.({ ...vals }, c.key); });
    wrap.querySelector(".reset").addEventListener("click", () => { set(c.default); onChange?.({ ...vals }, c.key); });
  });
  host.querySelectorAll('[data-bs-toggle="tooltip"]').forEach((el) => new bootstrap.Tooltip(el, { placement: "right", trigger: "hover focus" }));
  return {
    get: () => ({ ...vals }),
    set: (v) => Object.entries(v || {}).forEach(([k, x]) => setters[k]?.(x)),
    reset: () => controls.forEach((c) => setters[c.key](c.default)),
  };
}
