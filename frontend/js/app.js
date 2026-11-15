import { esc, get, post, state, toast } from "./api.js";
import { renderBenchmark } from "./pages/benchmark.js";
import { renderCompare } from "./pages/compare.js";
import { renderDocs } from "./pages/docs.js";
import { renderFineTuning } from "./pages/finetune.js";
import { renderHistory } from "./pages/history.js";
import { renderModule } from "./pages/module.js";
import { renderOverview } from "./pages/overview.js";
import { renderSection } from "./pages/section.js";
import { renderSettings } from "./pages/settings.js";

// Dashboard sections (spec §3) → module ids. Modules may appear in several sections.
export const SECTIONS = {
  architecture: { title: "LLM Architecture", icon: "◇", blurb: "How the model represents text and routes computation.",
    modules: ["tokenizer-bias", "mixture-of-experts", "rope-scaling", "attention-mechanisms"] },
  inference: { title: "Inference Optimization", icon: "⚡", blurb: "Decoding, caching, quantization and serving techniques that cut latency and cost.",
    modules: ["kv-cache", "prefix-caching", "temperature-sampling", "speculative-decoding", "medusa-heads", "quantization", "paged-attention", "continuous-batching", "model-serving"] },
  retrieval: { title: "Retrieval Optimization", icon: "⌕", blurb: "Embeddings, lexical + semantic search, reranking and end-to-end RAG tuning.",
    modules: ["cosine-trap", "hybrid-search", "cross-encoder-reranking", "rag-optimization"] },
  attention: { title: "Attention Mechanisms", icon: "◎", blurb: "Attention variants, positional encoding, streaming and distributed attention.",
    modules: ["attention-mechanisms", "kv-cache", "rope-scaling", "attention-sink", "ring-attention"] },
  privacy: { title: "Privacy & Security", icon: "🛡", blurb: "Formal privacy, guardrails, governance and safe prompting.",
    modules: ["differential-privacy", "ai-security-governance", "prompt-engineering"] },
  quality: { title: "Quality & Evaluation", icon: "✓", blurb: "Measuring reasoning, calibration, retrieval and generation quality.",
    modules: ["model-evaluation", "logic-bias", "confirmation-prediction", "prompt-engineering"] },
  training: { title: "Fine-Tuning Concepts", icon: "⚙", blurb: "Parameter-efficient fine-tuning and low-precision training.",
    modules: ["lora", "qlora", "quantization", "differential-privacy"] },
};

const NAV = [
  { group: "Workspace" },
  { href: "#/overview", label: "Overview", icon: "▦" },
  { href: "#/section/architecture", label: "LLM Architecture", icon: "◇", section: "architecture" },
  { href: "#/section/inference", label: "Inference Optimization", icon: "⚡", section: "inference" },
  { href: "#/section/retrieval", label: "Retrieval Optimization", icon: "⌕", section: "retrieval" },
  { href: "#/fine-tuning", label: "Fine-Tuning Laboratory", icon: "⚙", section: "training" },
  { href: "#/section/attention", label: "Attention Mechanisms", icon: "◎", section: "attention" },
  { href: "#/section/privacy", label: "Privacy & Security", icon: "🛡", section: "privacy" },
  { href: "#/section/quality", label: "Quality & Evaluation", icon: "✓", section: "quality" },
  { group: "Experiments" },
  { href: "#/benchmark", label: "Benchmark Studio", icon: "⚖" },
  { href: "#/compare", label: "Model Comparison", icon: "≋" },
  { href: "#/history", label: "Experiment History", icon: "⟲" },
  { group: "Reference" },
  { href: "#/docs", label: "Documentation", icon: "📘" },
  { href: "#/settings", label: "Settings", icon: "⚙" },
];

function buildNav() {
  const nav = document.getElementById("nav");
  const route = location.hash || "#/overview";
  const modId = route.startsWith("#/module/") ? route.split("/")[2].split("?")[0] : null;
  nav.innerHTML = NAV.map((n) => {
    if (n.group) return `<div class="nav-section">${esc(n.group)}</div>`;
    const active = route.startsWith(n.href) ? "active" : "";
    let sub = "";
    if (n.section && (active || (modId && SECTIONS[n.section].modules.includes(modId) && route.includes("from=" + n.section)))) {
      sub = `<div class="sub">${SECTIONS[n.section].modules.map((id) => {
        const m = state.modules.find((x) => x.id === id);
        return m ? `<a href="#/module/${id}?from=${n.section}" class="${modId === id ? "active" : ""}">${esc(m.title)}</a>` : "";
      }).join("")}</div>`;
    }
    const count = n.section ? `<span class="num">${SECTIONS[n.section].modules.length}</span>` : "";
    return `<a href="${n.href}" class="${active}"><span class="nav-icon">${n.icon}</span>${esc(n.label)}${count}</a>${sub}`;
  }).join("") + `<div class="nav-section">All ${state.modules.length} modules</div><div class="sub">${state.modules.map((m) =>
    `<a href="#/module/${m.id}" class="${modId === m.id && !route.includes("from=") ? "active" : ""}">${m.original_index ? `<span class="text-muted me-1">${m.original_index}.</span>` : ""}${esc(m.title)}</a>`).join("")}</div>`;
}

export function setCrumbs(html) { document.getElementById("crumbs").innerHTML = html; }

export async function refreshBadges() {
  try {
    const s = await get("/api/settings");
    state.settings = s.settings;
    state.hardwareProfiles = s.hardware_profiles;
    const pb = document.getElementById("provider-badge");
    pb.textContent = s.settings.provider === "simulation" ? "Simulation mode" : `Provider: ${s.settings.provider}`;
    pb.className = "badge " + (s.settings.provider === "simulation" ? "text-bg-warning" : "text-bg-success");
    document.getElementById("model-badge").textContent = `${s.settings.active_model} · ${s.hardware_profiles[s.settings.hardware]?.label || s.settings.hardware}`;
  } catch { /* ignore */ }
}

let currentCleanup = null;
async function route() {
  const view = document.getElementById("view");
  currentCleanup?.();
  currentCleanup = null;
  const hash = location.hash || "#/overview";
  const [path, query] = hash.slice(2).split("?");
  const params = new URLSearchParams(query || "");
  const [page, arg] = path.split("/");
  buildNav();
  document.getElementById("sidebar").classList.remove("open");
  view.innerHTML = `<div class="computing"><span class="spinner-dot"></span>Loading…</div>`;
  try {
    switch (page) {
      case "module": currentCleanup = await renderModule(view, arg, params); break;
      case "section": await renderSection(view, arg); break;
      case "benchmark": await renderBenchmark(view, params); break;
      case "compare": await renderCompare(view); break;
      case "history": await renderHistory(view); break;
      case "fine-tuning": currentCleanup = await renderFineTuning(view); break;
      case "docs": await renderDocs(view, arg); break;
      case "settings": await renderSettings(view); break;
      default: await renderOverview(view);
    }
  } catch (e) {
    console.error(e);
    view.innerHTML = `<div class="alert alert-danger">Failed to load page: ${esc(e.message)}</div>`;
  }
  view.focus({ preventScroll: true });
  window.scrollTo(0, 0);
}

async function init() {
  try { const t = localStorage.getItem("llmlab-theme"); if (t) document.documentElement.dataset.bsTheme = t; } catch { /* storage unavailable */ }
  try {
    state.user = await get("/api/auth/me");
  } catch { return; }
  const mods = await get("/api/modules");
  state.modules = mods.modules;
  state.categories = mods.categories;
  document.getElementById("user-btn").textContent = state.user.email;
  document.getElementById("user-role").innerHTML = `Role: <b>${esc(state.user.role_label)}</b>`;
  document.getElementById("logout-btn").addEventListener("click", async () => {
    try { await post("/api/auth/logout", {}); } catch { /* ignore */ }
    location.href = "/login";
  });
  document.getElementById("theme-btn").addEventListener("click", () => {
    const next = document.documentElement.dataset.bsTheme === "dark" ? "light" : "dark";
    document.documentElement.dataset.bsTheme = next;
    try { localStorage.setItem("llmlab-theme", next); } catch { /* ignore */ }
    route();
  });
  document.getElementById("menu-btn").addEventListener("click", () => document.getElementById("sidebar").classList.toggle("open"));
  window.addEventListener("hashchange", route);
  await refreshBadges();
  route();
}

window.addEventListener("unhandledrejection", (e) => { if (e.reason?.status !== 401) toast(e.reason?.message || "Unexpected error", "error"); });
init();
