// Generic renderer for the uniform module result payload.
import { esc, fmt } from "../api.js";
import { renderChart, renderHeatmap } from "./charts.js";

export const FID_LABEL = { analytical: "Analytical estimate", computed: "Computed (exact)", simulated: "Simulated", real: "Real model / measured" };

export function fidelityBadge(f) {
  return `<span class="badge badge-fid fid-${esc(f)}" title="${esc(FID_LABEL[f] || f)}">${esc(FID_LABEL[f] || f)}</span>`;
}

export function metricCards(metrics, baseline) {
  const bm = Object.fromEntries((baseline?.metrics || []).map((m) => [m.key, m]));
  return `<div class="metric-grid">${(metrics || []).map((m) => {
    let delta = "";
    const b = bm[m.key];
    if (b && typeof b.value === "number" && typeof m.value === "number") {
      const d = m.value - b.value;
      const pct = b.value ? (d / Math.abs(b.value)) * 100 : null;
      const cls = Math.abs(d) < 1e-12 ? "flat" : d > 0 ? "up" : "down";
      delta = `<div class="delta ${cls}">vs baseline: ${d >= 0 ? "+" : ""}${fmt(d)}${pct !== null && isFinite(pct) ? ` (${pct >= 0 ? "+" : ""}${pct.toFixed(1)}%)` : ""}</div>`;
    } else if (b && b.value !== m.value) {
      delta = `<div class="delta flat">baseline: ${esc(fmt(b.value))}</div>`;
    }
    return `<div class="metric-card ${esc(m.status || "")}" title="${esc(m.hint || "")}">
      <div class="label">${esc(m.label)}</div>
      <div class="value">${esc(fmt(m.value))}${m.unit ? `<span class="unit">${esc(m.unit)}</span>` : ""}</div>
      ${m.hint ? `<div class="hint">${esc(m.hint)}</div>` : ""}${delta}</div>`;
  }).join("")}</div>`;
}

function tableHTML(t) {
  return `<div class="table-wrap"><table class="table table-sm table-lab"><thead><tr>${t.columns.map((c) => `<th>${esc(c)}</th>`).join("")}</tr></thead>
    <tbody>${t.rows.map((r) => `<tr>${r.map((c) => `<td>${esc(typeof c === "number" ? fmt(c) : c)}</td>`).join("")}</tr>`).join("")}</tbody></table></div>`;
}

function comparisonHTML(c) {
  if (!c) return "";
  const rows = c.rows.map((r) => {
    let bc = "", ac = "";
    if (typeof r.before === "number" && typeof r.after === "number" && r.better) {
      const improved = r.better === "lower" ? r.after < r.before : r.after > r.before;
      if (r.after !== r.before) { ac = improved ? "better" : "worse"; }
    }
    const ratio = typeof r.before === "number" && typeof r.after === "number" && r.before !== 0 ? r.after / r.before : null;
    return `<tr><td>${esc(r.metric)}</td><td class="${bc}">${esc(fmt(r.before))} ${esc(r.unit || "")}</td><td class="${ac}">${esc(fmt(r.after))} ${esc(r.unit || "")}</td>
      <td class="text-muted">${ratio !== null && isFinite(ratio) ? ratio.toFixed(2) + "×" : ""}</td></tr>`;
  }).join("");
  return `<div class="lab-card"><h6>${esc(c.title)}</h6><table class="table table-sm table-lab cmp-table"><thead><tr><th>Metric</th><th>${esc(c.before_label)}</th><th>${esc(c.after_label)}</th><th>ratio</th></tr></thead><tbody>${rows}</tbody></table></div>`;
}

export function flowHTML(nodes) {
  return `<div class="flow">${nodes.map((n, i) => `${i ? '<span class="flow-arrow">→</span>' : ""}<div class="flow-node ${esc(n.cls || "")}"><div>${esc(n.label)}</div>${n.sub ? `<div class="sub">${esc(n.sub)}</div>` : ""}</div>`).join("")}</div>`;
}

function tokensHTML(t) {
  if (t.inline) {
    return `<div class="tok-wrap inline">${t.items.map((i) => i.cls === "pii" ? `<span class="pii">${esc(i.text)}<sup>${esc(i.id)}</sup></span>` : esc(i.text)).join("")}</div>`;
  }
  return `<div class="tok-wrap">${t.items.map((i) => `<span class="tok ${esc(i.cls || "")}" title="${i.id !== undefined ? "id " + esc(i.id) : ""}"><span>${esc(i.text === " " ? "␣" : i.text.replace(/ /g, "·"))}</span>${t.show_ids !== false && i.id !== undefined && i.id !== null ? `<span class="id">${esc(i.id)}</span>` : ""}</span>`).join("")}</div>`;
}

function ringSVG(n, label) {
  const R = 90, cx = 130, cy = 115;
  const pts = [...Array(n)].map((_, i) => [cx + R * Math.cos((2 * Math.PI * i) / n - Math.PI / 2), cy + R * Math.sin((2 * Math.PI * i) / n - Math.PI / 2)]);
  const arrows = pts.map((p, i) => { const q = pts[(i + 1) % n]; return `<line x1="${p[0]}" y1="${p[1]}" x2="${q[0]}" y2="${q[1]}" stroke="#A77D9E" stroke-width="2" marker-end="url(#ah)" opacity=".9"/>`; }).join("");
  const nodes = pts.map((p, i) => `<circle cx="${p[0]}" cy="${p[1]}" r="17" fill="#714B67" stroke="#00CEB3" stroke-width="2"/><text x="${p[0]}" y="${p[1] + 4}" text-anchor="middle" style="fill:#fff">GPU${i}</text>`).join("");
  return `<div class="ring d-flex gap-3 align-items-center flex-wrap"><svg width="260" height="230" viewBox="0 0 260 230" role="img" aria-label="ring of ${n} GPUs">
    <defs><marker id="ah" markerWidth="8" markerHeight="8" refX="18" refY="3" orient="auto"><path d="M0,0 L0,6 L7,3 z" fill="#A77D9E"/></marker></defs>${arrows}${nodes}</svg>
    <div class="small text-muted" style="max-width:320px">${esc(label)}</div></div>`;
}

function panelHTML(p) {
  switch (p.kind) {
    case "code": return `<pre class="code">${esc(p.content)}</pre>`;
    case "formula": return `<div class="p-2 mono" style="font-size:1rem">W′ = W + (α / r) · B · A</div><div class="small text-muted">${esc(p.content)}</div>`;
    case "messages": return p.messages.map((m) => `<div class="msg ${esc(m.role)}"><div class="role">${esc(m.role)}</div>${esc(m.content)}</div>`).join("");
    case "tree": return p.levels.map((lv, i) => `<div class="tree-level"><span class="small text-muted" style="width:62px">depth ${i + 1}</span>${lv.map((n) => `<span class="tree-node ${n.accepted ? "ok" : ""}">${esc(n.label)}</span>`).join("")}</div>`).join("");
    case "ring": return ringSVG(p.nodes, p.label);
    case "block_table": return `<table class="table table-sm table-lab"><thead><tr><th>Sequence</th><th>Logical → physical block ids (block = ${p.block_size} tokens)</th></tr></thead><tbody>${p.sequences.map((s) => `<tr><td>${esc(s.seq)}</td><td class="mono">${s.physical.map((b, i) => `<span class="tok" title="logical ${i}"><span>#${b}</span><span class="id">L${i}</span></span>`).join(" ")}</td></tr>`).join("")}</tbody></table>`;
    case "moe_route": return p.tokens.map((t) => `<div class="mb-2"><div class="small mb-1">${esc(t.token)} → experts <b>${t.chosen.map((e) => "E" + e).join(", ") || "none (dropped)"}</b></div><div class="d-flex gap-1">${t.probs.map((pr, e) => `<div title="E${e}: ${pr}" style="flex:1;height:22px;border-radius:4px;background:${t.chosen.includes(e) ? "#00CEB3" : "#714B67"};opacity:${0.15 + 0.85 * Math.min(1, pr * 3)}"></div>`).join("")}</div></div>`).join("");
    case "concepts": return `<table class="table table-sm table-lab"><tbody>${p.items.map(([a, b]) => `<tr><th style="width:190px">${esc(a)}</th><td>${esc(b)}</td></tr>`).join("")}</tbody></table>`;
    case "schedule": return p.rows.map(([a, b]) => `<div class="mb-1"><span class="badge bg-secondary me-2" style="width:90px">${esc(a)}</span><span class="mono small">${esc(b)}</span></div>`).join("");
    default: return `<pre class="code">${esc(JSON.stringify(p, null, 2))}</pre>`;
  }
}

/** Render a full result into host. */
export function renderResult(host, res, { baseline = null } = {}) {
  const charts = res.charts || [];
  const parts = [];
  parts.push(`<div class="fidelity-banner mb-3">${fidelityBadge(res.fidelity)}<div>${esc(res.fidelity_note)}<span class="text-muted ms-2">· computed in ${res.compute_ms ?? "?"} ms</span></div></div>`);
  if (res.notes?.some((n) => n.level === "warn")) {
    parts.push(`<div class="notes mb-3">${res.notes.filter((n) => n.level === "warn").map((n) => `<div class="note warn">⚠ ${esc(n.text)}</div>`).join("")}</div>`);
  }
  parts.push(metricCards(res.metrics, baseline));
  parts.push(`<div class="result-grid mt-3" data-charts></div>`);
  host.innerHTML = parts.join("");
  const grid = host.querySelector("[data-charts]");
  charts.forEach((c) => {
    const card = document.createElement("div");
    card.className = "lab-card";
    card.innerHTML = `<h6>${esc(c.title)}</h6><div class="chart-box"><canvas aria-label="${esc(c.title)}"></canvas></div>`;
    grid.append(card);
    requestAnimationFrame(() => renderChart(card.querySelector("canvas"), c));
  });
  (res.heatmaps || []).forEach((h) => {
    const card = document.createElement("div");
    card.className = "lab-card heat";
    card.innerHTML = `<h6>${esc(h.title)}</h6>`;
    grid.append(card);
    renderHeatmap(card, h);
  });
  (res.flows || []).forEach((f) => {
    const card = document.createElement("div");
    card.className = "lab-card";
    card.innerHTML = `<h6>${esc(f.title)}</h6>${flowHTML(f.nodes)}`;
    grid.append(card);
  });
  (res.panels || []).forEach((p) => {
    const card = document.createElement("div");
    card.className = "lab-card";
    card.innerHTML = `<h6>${esc(p.title)}</h6>${panelHTML(p)}`;
    grid.append(card);
  });
  const below = document.createElement("div");
  below.className = "mt-3";
  below.innerHTML =
    (res.tokens || []).map((t) => `<div class="lab-card"><h6>${esc(t.title)}</h6>${tokensHTML(t)}</div>`).join("") +
    comparisonHTML(res.comparison) +
    (res.tables || []).map((t) => `<div class="lab-card"><h6>${esc(t.title)}</h6>${tableHTML(t)}</div>`).join("") +
    (res.notes?.some((n) => n.level !== "warn") ? `<div class="lab-card notes"><h6>Notes</h6>${res.notes.filter((n) => n.level !== "warn").map((n) => `<div class="note">${esc(n.text)}</div>`).join("")}</div>` : "") +
    (res.recommendations?.length ? `<div class="lab-card"><h6>Practical recommendations</h6><ul class="recs mb-0">${res.recommendations.map((r) => `<li>${esc(r)}</li>`).join("")}</ul></div>` : "");
  host.append(below);
}
