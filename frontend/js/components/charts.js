// Chart.js factory for the uniform chart spec returned by the backend.
// Brand palette: plum, teal, amber, coral, sky, then neutrals/tints
const PALETTE = ["#714B67", "#00CEB3", "#FBB130", "#FC787D", "#1BB6F9", "#6C757D", "#A77D9E", "#2F855A", "#5A3A52"];
const isDark = () => document.documentElement.dataset.bsTheme === "dark";
const seriesColors = () => (isDark() ? ["#C99BBD", ...PALETTE.slice(1)] : PALETTE);
const registry = new WeakMap();

const refLines = {
  id: "refLines",
  afterDatasetsDraw(chart, _args, opts) {
    const { ctx, chartArea: a, scales } = chart;
    ctx.save();
    ctx.setLineDash([6, 4]);
    ctx.lineWidth = 1.5;
    if (opts.y !== undefined && opts.y !== null && scales.y) {
      const y = scales.y.getPixelForValue(opts.y);
      if (y >= a.top && y <= a.bottom) {
        ctx.strokeStyle = "#D64550";
        ctx.beginPath(); ctx.moveTo(a.left, y); ctx.lineTo(a.right, y); ctx.stroke();
        ctx.fillStyle = "#D64550"; ctx.font = "11px sans-serif"; ctx.fillText(opts.yLabel || `threshold ${opts.y}`, a.left + 6, y - 4);
      }
    }
    if (opts.x !== undefined && opts.x !== null && scales.x) {
      const x = scales.x.getPixelForValue(opts.x);
      if (x >= a.left && x <= a.right) {
        ctx.strokeStyle = "#C98A12";
        ctx.beginPath(); ctx.moveTo(x, a.top); ctx.lineTo(x, a.bottom); ctx.stroke();
        ctx.fillStyle = "#C98A12"; ctx.font = "11px sans-serif"; ctx.fillText(opts.xLabel || String(opts.x), x + 4, a.top + 12);
      }
    }
    ctx.restore();
  },
};

function themeColors() {
  const cs = getComputedStyle(document.documentElement);
  return { text: cs.getPropertyValue("--lab-muted").trim() || "#93a0bd", grid: cs.getPropertyValue("--lab-border").trim() || "#24304f" };
}

function nearestIndex(labels, value) {
  if (!labels) return null;
  let best = 0, bd = Infinity;
  labels.forEach((l, i) => { const d = Math.abs(Number(l) - value); if (d < bd) { bd = d; best = i; } });
  return best;
}

export function renderChart(canvas, spec) {
  registry.get(canvas)?.destroy();
  const { text, grid } = themeColors();
  const kind = spec.kind === "scatter" ? "scatter" : spec.kind;
  const datasets = (spec.datasets || []).map((d, i) => {
    const color = seriesColors()[i % PALETTE.length];
    const ds = { ...d, borderColor: d.borderColor || color, backgroundColor: d.backgroundColor || (kind === "line" ? color : color + "cc"),
                 borderWidth: kind === "bar" ? 0 : 2, pointRadius: d.pointRadius ?? (kind === "line" ? 0 : 3), tension: 0.15, spanGaps: true };
    if (kind === "bar" && spec.highlight && spec.datasets.length === 1) {
      ds.backgroundColor = spec.highlight.map((h) => (h ? "#00A391" : "#FC787D") + "cc");
    }
    if (spec.dual_axis && d.yAxisID === "y1") ds.yAxisID = "y1";
    if (kind === "scatter" && d.showLine) { ds.showLine = true; ds.borderWidth = 2; }
    return ds;
  });
  const scales = {
    x: { type: spec.log_x ? "logarithmic" : (kind === "scatter" ? "linear" : "category"), title: { display: !!spec.x_label, text: spec.x_label, color: text },
         ticks: { color: text, maxRotation: 50, autoSkip: true, maxTicksLimit: 14 }, grid: { color: grid + "66" }, stacked: !!spec.stacked },
    y: { type: spec.log_y ? "logarithmic" : "linear", title: { display: !!spec.y_label, text: spec.y_label, color: text }, ticks: { color: text }, grid: { color: grid + "66" },
         stacked: !!spec.stacked, beginAtZero: !spec.log_y && kind === "bar" },
  };
  if (spec.log_x && kind !== "scatter") scales.x.type = "category";
  if (spec.dual_axis) scales.y1 = { position: "right", ticks: { color: text }, grid: { drawOnChartArea: false } };
  const radial = kind === "doughnut" || kind === "pie";
  if (radial) {
    datasets.forEach((d) => { d.backgroundColor = (spec.labels || []).map((_, i) => PALETTE[i % PALETTE.length]); d.borderWidth = 0; });
    return (() => {
      const c = new Chart(canvas, { type: kind, data: { labels: spec.labels, datasets },
        options: { responsive: true, maintainAspectRatio: false, plugins: { legend: { position: "right", labels: { color: text, font: { size: 11 } } } } } });
      registry.set(canvas, c);
      return c;
    })();
  }
  let vx = null;
  if (spec.vline !== undefined && spec.vline !== null) vx = kind === "scatter" ? spec.vline : nearestIndex(spec.labels, spec.vline);
  if (spec.vline_value !== undefined && spec.labels) vx = nearestIndex(spec.labels, spec.vline_value);
  const chart = new Chart(canvas, {
    type: kind,
    data: { labels: spec.labels || undefined, datasets },
    options: {
      responsive: true, maintainAspectRatio: false, animation: { duration: 250 }, interaction: { mode: kind === "scatter" ? "nearest" : "index", intersect: false },
      plugins: {
        legend: { display: datasets.length > 1 || kind === "scatter", labels: { color: text, boxWidth: 12, font: { size: 11 } } },
        tooltip: { callbacks: kind === "scatter" ? { label: (c) => `${c.raw.t ?? c.dataset.label}: (${Number(c.raw.x).toFixed(3)}, ${Number(c.raw.y).toFixed(3)})` } : {} },
        refLines: { y: spec.threshold, x: vx, xLabel: spec.vline !== undefined ? String(spec.vline) : (spec.vline_value !== undefined ? `C=${spec.vline_value}` : undefined) },
      },
      scales,
    },
    plugins: [refLines],
  });
  registry.set(canvas, chart);
  return chart;
}

// Heatmap on canvas (Chart.js has no native heatmap)
// Sequential ramp: light surface → plum (light theme) / dark surface → light plum (dark theme)
function ramp(t) {
  const [a, b] = isDark() ? [[46, 49, 64], [233, 190, 222]] : [[248, 244, 247], [90, 58, 82]];
  const c = a.map((x, i) => Math.round(x + (b[i] - x) * Math.max(0, Math.min(1, t))));
  return `rgb(${c[0]}, ${c[1]}, ${c[2]})`;
}

export function renderHeatmap(host, spec) {
  const M = spec.matrix || [];
  const rows = M.length, cols = rows ? M[0].length : 0;
  const cell = Math.max(6, Math.min(36, Math.floor(560 / Math.max(rows, cols, 1))));
  const canvas = document.createElement("canvas");
  canvas.width = cols * cell; canvas.height = rows * cell;
  canvas.style.maxWidth = canvas.width + "px";
  const ctx = canvas.getContext("2d");
  const flat = M.flat().filter((v) => v !== null && v !== undefined);
  let lo = Math.min(...flat), hi = Math.max(...flat);
  const tf = (v) => (spec.log_scale ? Math.log10(Math.max(v, 1e-6)) : v);
  if (spec.log_scale) { lo = tf(Math.max(lo, 1e-6)); hi = tf(hi); }
  const cat = [isDark() ? "#2E3140" : "#EEE7EC", "#714B67", "#00CEB3", "#FBB130", "#FC787D", "#1BB6F9", "#A77D9E", "#6C757D"];
  for (let i = 0; i < rows; i++) for (let j = 0; j < cols; j++) {
    const v = M[i][j];
    let color;
    if (spec.categorical) color = cat[(Number(v) || 0) % cat.length];
    else {
      const t = hi > lo ? (tf(v) - lo) / (hi - lo) : 0;
      color = ramp(t);
    }
    ctx.fillStyle = color;
    ctx.fillRect(j * cell, i * cell, cell - (cell > 10 ? 1 : 0), cell - (cell > 10 ? 1 : 0));
    if (spec.annotate && cell >= 30) {
      ctx.fillStyle = isDark() ? "#111827" : "#fff"; ctx.font = "12px sans-serif"; ctx.textAlign = "center"; ctx.textBaseline = "middle";
      ctx.fillText(String(v), j * cell + cell / 2, i * cell + cell / 2);
    }
  }
  if (spec.annotate && Array.isArray(spec.x_labels)) {
    const big = document.createElement("canvas");
    const pad = 110;
    big.width = cols * 120 + pad; big.height = rows * 60 + 30;
    const b = big.getContext("2d");
    for (let i = 0; i < rows; i++) for (let j = 0; j < cols; j++) {
      const v = M[i][j], t = hi > lo ? (v - lo) / (hi - lo) : 0;
      b.fillStyle = ramp(0.15 + 0.85 * t);
      b.fillRect(pad + j * 120, 30 + i * 60, 118, 58);
      b.fillStyle = "#fff"; b.font = "16px sans-serif"; b.textAlign = "center"; b.textBaseline = "middle";
      b.fillText(String(v), pad + j * 120 + 59, 30 + i * 60 + 29);
    }
    b.fillStyle = getComputedStyle(document.documentElement).getPropertyValue("--lab-muted"); b.font = "12px sans-serif"; b.textAlign = "center";
    spec.x_labels.forEach((l, j) => b.fillText(l, pad + j * 120 + 59, 15));
    b.textAlign = "right";
    (spec.y_labels || []).forEach((l, i) => b.fillText(l, pad - 6, 30 + i * 60 + 29));
    host.append(big);
    return;
  }
  host.append(canvas);
  const legend = document.createElement("div");
  legend.className = "d-flex justify-content-between small text-muted mt-1";
  legend.innerHTML = `<span>${typeof spec.y_labels === "string" ? "↓ " + spec.y_labels : ""}</span><span>${typeof spec.x_labels === "string" ? spec.x_labels + " →" : ""}</span>`;
  host.append(legend);
}
