"""Module 10 (RoPE Scaling), 11 (Attention Sink), 12 (Mixture of Experts), 14 (Ring Attention)."""
from __future__ import annotations

import math

import numpy as np

from app.engine.common import active_arch, rng, softmax
from app.engine.hardware import HARDWARE, HARDWARE_OPTIONS, fmt_params
from app.engine.registry import Context, ModuleDef, Result, docs, register, select, slider, toggle


# ============================================================================ RoPE
def rope_freqs(d: int, base: float, method: str, s: float, orig: int, pos: int | None = None) -> tuple[np.ndarray, float]:
    """Return per-pair angular frequencies θ_i and the attention temperature multiplier."""
    i = np.arange(0, d, 2) / d
    theta = base ** (-i)
    if method == "none" or s <= 1:
        return theta, 1.0
    if method == "linear":
        return theta / s, 1.0
    if method == "ntk":
        return (base * s ** (d / (d - 2))) ** (-i), 1.0
    if method == "dynamic":
        cur = max(1.0, (pos or orig) / orig)
        return (base * cur ** (d / (d - 2))) ** (-i), 1.0
    # YaRN ("NTK-by-parts" + attention temperature), alpha=1, beta=32 as in the paper
    lam = 2 * math.pi / theta
    ratio = orig / lam
    gamma = np.clip((ratio - 1) / (32 - 1), 0, 1)
    return (1 - gamma) * theta / s + gamma * theta, 0.1 * math.log(s) + 1


def rope_sim(p: dict, ctx: Context) -> Result:
    d, base, orig, ext, s = p["head_dim"], float(p["base"]), p["original_context"], p["extended_context"], p["scaling_factor"]
    th0, _ = rope_freqs(d, base, "none", 1, orig)
    th, mscale = rope_freqs(d, base, p["method"], s, orig, ext)
    r = Result("computed")
    seen_max = orig * th0  # max angle each pair saw during training
    period_seen = seen_max >= 2 * math.pi
    new_max = ext * th
    ood = int(np.sum((new_max > seen_max * 1.0001) & ~period_seen))
    resolution = float(th[0] / th0[0])
    need = ext / orig
    r.metric("ood", "Out-of-distribution dim pairs", ood, f"/ {d // 2}", "low-frequency pairs whose angle at max position was never seen in training",
             status="bad" if ood else "good")
    r.metric("res", "High-freq resolution vs original", resolution, "×", "1.0 = adjacent tokens as distinguishable as in training")
    r.metric("needed", "Required extension factor", need, "×")
    r.metric("eff_base", "Effective base (lowest freq)", float(1 / th[-1] ** (1 / ((d - 2) / d))) if th[-1] > 0 else 0, "")
    r.metric("mscale", "Attention temperature factor", mscale, "", "YaRN scales logits by (0.1·ln s + 1)²" if p["method"] == "yarn" else "")
    if p["method"] in ("linear", "ntk", "yarn") and s < need - 1e-9:
        r.note(f"Scaling factor {s} is below the required {need:.1f}× — positions beyond {int(orig * s):,} remain extrapolated.", "warn")
    pairs = list(range(d // 2))
    r.chart("freq", "Rotation frequency per dimension pair (log scale)", "line", pairs, [
        {"label": "original θ_i", "data": th0.tolist()}, {"label": f"{p['method']} θ'_i", "data": th.tolist()}], "pair index i", "rad / token", log_y=True)
    m = p["position_index"]
    r.chart("rot", f"cos(m·θ_i) at position m = {m:,}", "line", pairs, [
        {"label": "original", "data": np.cos(m * th0).tolist()}, {"label": p["method"], "data": np.cos(m * th).tolist()}], "pair index i", "cos")
    rel = np.unique(np.linspace(0, ext, 300).astype(int))
    def score(theta):
        return [float(np.mean(np.cos(dl * theta))) for dl in rel]
    r.chart("decay", "Relative-position attention profile  (mean cos(Δ·θ_i) for q = k)", "line", rel.tolist(), [
        {"label": "original (extrapolated past training)", "data": score(th0)}, {"label": p["method"], "data": score(th)}],
        "relative distance Δ (tokens)", "score", vline=orig)
    r.chart("ood", "Max angle reached vs. seen in training", "line", pairs, [
        {"label": f"seen in training (≤{orig:,} tokens)", "data": np.minimum(seen_max, 1e6).tolist()},
        {"label": f"needed at {ext:,} tokens", "data": np.minimum(new_max, 1e6).tolist()}], "pair index i", "radians", log_y=True)
    r.compare("No scaling (extrapolate)", p["method"], [
        ("OOD dimension pairs", int(np.sum((ext * th0 > seen_max * 1.0001) & ~period_seen)), ood, "", "lower"),
        ("High-freq resolution", 1.0, resolution, "×", "higher")])
    r.recommend("Use a model already trained/fine-tuned for the target length when possible (e.g., Llama-3.1 128K RoPE scaling).",
                "Linear interpolation needs fine-tuning; NTK-aware/dynamic NTK work better zero-shot; YaRN fine-tunes efficiently.",
                "Always evaluate long-context retrieval (needle-in-a-haystack, RULER) at the actual target lengths.",
                "Implementation details (rope_scaling type, factor, original_max_position_embeddings) are model-specific — read the config.")
    return r


register(ModuleDef(
    id="rope-scaling", title="RoPE Scaling", category="architecture", original_index=10, fidelity="computed",
    summary="Rotary position embeddings and how linear, NTK-aware, dynamic and YaRN scaling extend context.",
    controls=[
        slider("head_dim", "Head dimension", 128, 32, 256, 32, "RoPE rotates head_dim/2 pairs."),
        slider("base", "RoPE base (θ)", 10000, 1000, 1000000, 1000, "Llama-2: 10,000; Llama-3: 500,000.", log=True),
        slider("original_context", "Original (trained) context", 4096, 512, 32768, 512, "Positions seen during pretraining.", unit="tok"),
        slider("extended_context", "Extended context", 32768, 1024, 262144, 1024, "Target context length.", unit="tok"),
        select("method", "RoPE scaling method", "yarn", [{"value": "none", "label": "None (extrapolate)"}, {"value": "linear", "label": "Linear (Position Interpolation)"},
                                                       {"value": "ntk", "label": "NTK-aware (base scaling)"}, {"value": "dynamic", "label": "Dynamic NTK"},
                                                       {"value": "yarn", "label": "YaRN (NTK-by-parts + temperature)"}], "RoPE configuration."),
        slider("scaling_factor", "Scaling factor s", 8.0, 1.0, 64.0, 0.5, "Usually extended / original."),
        slider("position_index", "Position index m", 8192, 0, 262144, 64, "Position to visualize rotations at."),
    ],
    simulate=rope_sim,
    docs=docs(
        what="RoPE encodes position by rotating each 2-D pair of query/key dimensions by angle m·θ_i, with θ_i = base^(−2i/d). The dot product q·k then depends only on relative distance.",
        why="Models trained to length L only see angles up to L·θ_i. Beyond L, low-frequency pairs reach unseen angles and attention degrades. Scaling methods remap positions/frequencies so long sequences look in-distribution.",
        how="Linear PI divides positions by s (all frequencies compressed — loses local resolution). NTK-aware increases the base so high frequencies stay intact and low frequencies interpolate. Dynamic NTK adjusts s with the current length. YaRN interpolates only low-frequency pairs (by wavelength ratio) and adds an attention temperature.",
        when_use="Long-document QA, codebase-wide reasoning, long agent transcripts — when the model's native window is too short.",
        when_avoid="If a native long-context model exists; when RAG can supply the relevant few thousand tokens more cheaply; without long-context evaluation.",
        pros=["Extends context 4–32× with little or no training", "No architecture change", "YaRN/NTK keep short-context quality largely intact"],
        cons=["Quality degrades with extreme factors", "Usually requires some long-context fine-tuning", "Quadratic attention cost and KV memory grow with the window"],
        use_case="A legal-tech firm extends a 4K model to 32K with YaRN plus 400 steps of fine-tuning on long contracts, then validates with needle-in-a-haystack tests.",
        code='''
import torch
def rope_inv_freq(dim, base=10000.0, method="yarn", s=8.0, orig=4096):
    inv = base ** (-torch.arange(0, dim, 2).float() / dim)
    if method == "linear":  return inv / s
    if method == "ntk":     return (base * s ** (dim / (dim - 2))) ** (-torch.arange(0, dim, 2).float() / dim)
    if method == "yarn":
        ratio = orig / (2 * torch.pi / inv)
        g = ((ratio - 1) / (32 - 1)).clamp(0, 1)
        return (1 - g) * inv / s + g * inv
    return inv

# Hugging Face config:
# config.rope_scaling = {"rope_type": "yarn", "factor": 8.0, "original_max_position_embeddings": 4096}
''',
        diagram=[{"label": "Position m"}, {"label": "θ_i = base^(−2i/d)"}, {"label": "Scale (PI/NTK/YaRN)", "cls": "accent"},
                 {"label": "Rotate q, k pairs"}, {"label": "q·k depends on Δ"}],
        performance="RoPE itself is cheap; the cost is the longer context: attention O(n²) compute (or O(n) memory with FlashAttention) and linear KV-cache growth.",
        checklist=["Confirm model's rope_scaling config", "Fine-tune briefly on long sequences", "Run long-context retrieval evals", "Check short-context regressions", "Budget KV memory for the new length"],
        applications=[{"domain": "Document intelligence", "text": "Whole-contract or whole-report analysis without chunking."},
                      {"domain": "AI-assisted software engineering", "text": "Repository-scale context for code understanding."}],
        limitations="Shows the positional geometry exactly; actual model quality at long range must be measured with long-context benchmarks.",
        references=["Su et al., 'RoFormer' (2021)", "Chen et al., 'Position Interpolation' (2023)", "Peng et al., 'YaRN' (2023)"],
    ),
))


# ============================================================================ Attention sink
def attn_logits(n: int, sink_strength: float, seed: int) -> np.ndarray:
    g = rng(seed)
    t = np.arange(n)[:, None]
    j = np.arange(n)[None, :]
    L = -0.08 * (t - j) + g.normal(0, 0.6, (n, n))
    L[:, :1] += sink_strength  # first token acts as a sink (empirically observed pattern)
    L[:, 1:4] += sink_strength * 0.35
    return np.where(j <= t, L, -np.inf)


def masked_softmax(L: np.ndarray, keep: np.ndarray) -> np.ndarray:
    Z = np.where(keep, L, -np.inf)
    Z = Z - np.max(Z, axis=1, keepdims=True)
    E = np.exp(Z)
    return E / E.sum(axis=1, keepdims=True)


def sink_sim(p: dict, ctx: Context) -> Result:
    n = 64
    scale = n / max(n, p["seq_len"])
    win = max(2, int(round(p["window"] * scale))) if p["seq_len"] > n else p["window"]
    win = min(win, n)
    sinks = p["sink_tokens"]
    L = attn_logits(n, p["sink_strength"], p["seed"])
    t = np.arange(n)[:, None]
    j = np.arange(n)[None, :]
    causal = j <= t
    full = masked_softmax(L, causal)
    window_only = masked_softmax(L, causal & (t - j < win))
    with_sinks = masked_softmax(L, causal & ((t - j < win) | (j < sinks)))
    V = rng(p["seed"] + 1).normal(0, 1, (n, 32))
    def shift(P):
        a, b = full @ V, P @ V
        return 1 - np.sum(a * b, 1) / (np.linalg.norm(a, axis=1) * np.linalg.norm(b, axis=1))
    sh_w, sh_s = shift(window_only), shift(with_sinks)
    mode = {"full": full, "window": window_only, "sinks": with_sinks}[p["mode"]]
    r = Result("simulated")
    r.set_fidelity("simulated", "Synthetic attention logits reproducing the empirically observed sink pattern; softmax, eviction and output-shift math are exact.")
    r.metric("sink_mass", "Attention mass on first 4 tokens (full)", float(full[n // 2:, :4].sum(1).mean() * 100), "%", "averaged over later queries")
    r.metric("shift_window", "Output deviation — window only", float(sh_w[win:].mean()), "", "1 − cos(full output, approx output)", status="bad" if sh_w[win:].mean() > 0.05 else None)
    r.metric("shift_sinks", "Output deviation — sinks + window", float(sh_s[win:].mean()), "", status="good" if sh_s[win:].mean() < sh_w[win:].mean() else None)
    arch = active_arch(ctx)
    kvtok = 2 * arch.layers * arch.kv_heads * arch.head_dim * 2
    T = p["seq_len"] + p["stream_tokens"]
    kept = min(T, p["window"] + sinks)
    r.metric("kv_full", "KV cache — full attention", T * kvtok / 1e9, "GB", f"{arch.label}, {T:,} tokens")
    r.metric("kv_stream", "KV cache — StreamingLLM", kept * kvtok / 1e9, "GB", f"{sinks} sinks + {p['window']} window")
    r.heatmap(f"Attention matrix ({p['mode']}, 64-token illustration, window {win}, {sinks} sinks)", mode.tolist(), x_labels="key position", y_labels="query position", log_scale=True)
    q = n - 1
    r.chart("row", f"Attention distribution of the last query (position {q})", "bar", list(range(n)), [
        {"label": "full", "data": full[q].tolist()}, {"label": "window only", "data": window_only[q].tolist()},
        {"label": f"{sinks} sinks + window", "data": with_sinks[q].tolist()}], "key position", "weight")
    r.chart("shift", "Output deviation from full attention per query position", "line", list(range(n)), [
        {"label": "window only (sinks evicted)", "data": sh_w.tolist()}, {"label": "sinks + window", "data": sh_s.tolist()}], "query position", "1 − cos")
    steps = np.linspace(0, p["stream_tokens"], 50).astype(int)
    r.chart("mem", "KV memory while streaming", "line", (p["seq_len"] + steps).tolist(), [
        {"label": "full KV cache", "data": ((p["seq_len"] + steps) * kvtok / 1e9).tolist()},
        {"label": "sinks + sliding window", "data": [min(p["seq_len"] + s, p["window"] + sinks) * kvtok / 1e9 for s in steps]}], "tokens processed", "GB")
    r.compare("Sliding window only", "Attention sinks + window", [("Output deviation", float(sh_w[win:].mean()), float(sh_s[win:].mean()), "", "lower"),
                                                                  ("KV memory", p["window"] * kvtok / 1e9, kept * kvtok / 1e9, "GB", "lower")])
    r.note("Sinks keep generation *stable*; they do not let the model remember evicted content. Information outside the window is gone.")
    r.recommend("For endless chat/streaming, keep ~4 sink tokens + a sliding window (StreamingLLM).",
                "Pair with summarization or retrieval memory if old content must be recalled.",
                "Use models with learned/trained sink tokens when available for the best stability.")
    return r


register(ModuleDef(
    id="attention-sink", title="Attention Sinks (StreamingLLM)", category="architecture", original_index=11, fidelity="simulated",
    summary="Keeping a few initial 'sink' tokens stabilizes sliding-window attention for infinite streams.",
    controls=[
        slider("sink_tokens", "Number of sink tokens", 4, 0, 16, 1, "Initial tokens always kept in the KV cache."),
        slider("window", "Sliding window size", 16, 2, 4096, 2, "Most recent tokens kept (illustration rescales to 64 tokens).", unit="tok"),
        slider("seq_len", "Sequence length", 64, 64, 1000000, 64, "Tokens before streaming generation.", log=True),
        slider("stream_tokens", "Streaming generation length", 100000, 0, 4000000, 1000, "Tokens generated afterwards.", log=True),
        select("mode", "Heatmap view", "sinks", [{"value": "full", "label": "Full causal attention"}, {"value": "window", "label": "Sliding window only"}, {"value": "sinks", "label": "Sinks + window"}], "Which attention pattern to show."),
        slider("sink_strength", "Sink logit strength", 5.0, 0.0, 8.0, 0.1, "How strongly queries attend to the first token."),
        slider("seed", "Random seed", 5, 0, 9999, 1, "Synthetic logits seed."),
    ],
    simulate=sink_sim,
    docs=docs(
        what="An attention sink is a token (typically the first few) that receives a large share of attention regardless of content. Models learn to 'park' excess softmax mass there.",
        why="Softmax forces weights to sum to 1. When no token is relevant, the mass must go somewhere; models learn to dump it on always-visible initial tokens. Evicting them (sliding-window KV) shifts all attention and destabilizes generation.",
        how="StreamingLLM keeps the KV entries of the first k (≈4) tokens plus a rolling window of recent tokens, re-assigning positions within the cache. Memory stays constant while perplexity stays stable over millions of tokens.",
        when_use="Long-running streaming generation, always-on assistants, log/stream processing, chat sessions longer than the context window.",
        when_avoid="Tasks needing recall of content older than the window (use retrieval/summarization), or when the full context fits.",
        pros=["Constant memory for unbounded streams", "No retraining required", "Big stability gain over plain sliding windows"],
        cons=["Does not extend true memory", "Window still bounds reasoning context", "Needs engine support for cache eviction + position handling"],
        use_case="A trading-desk monitoring assistant streams market commentary for hours; sinks + 4K window keep it coherent with fixed GPU memory.",
        code='''
# Conceptual StreamingLLM KV policy (per layer)
def evict(k_cache, v_cache, n_sink=4, window=4092):
    if k_cache.size(-2) <= n_sink + window:
        return k_cache, v_cache
    keep = lambda t: torch.cat([t[..., :n_sink, :], t[..., -window:, :]], dim=-2)
    return keep(k_cache), keep(v_cache)
# Hugging Face: from transformers import SinkCache; model.generate(..., past_key_values=SinkCache(window_length=4096, num_sink_tokens=4))
''',
        diagram=[{"label": "Sink tokens (keep)", "cls": "accent"}, {"label": "Evicted middle"}, {"label": "Sliding window (keep)", "cls": "accent"}, {"label": "Query attends sinks + window"}],
        performance="KV memory = (sinks + window) × KV bytes/token — constant. Compute per token is also bounded by the window.",
        checklist=["Choose window by quality/latency budget", "Keep ≥4 sinks", "Add memory (RAG/summary) for long-term facts", "Test stability over very long streams"],
        applications=[{"domain": "Enterprise chatbots", "text": "Day-long sessions without context overflow crashes."},
                      {"domain": "Agentic AI", "text": "Long-running monitoring agents processing event streams."}],
        limitations="Attention logits are synthetic; the qualitative effect matches published observations but magnitudes are illustrative.",
        references=["Xiao et al., 'Efficient Streaming Language Models with Attention Sinks' (2024)"],
    ),
))


# ============================================================================ Mixture of Experts
def moe_sim(p: dict, ctx: Context) -> Result:
    E, K, T, d_model, d_ff, layers = p["experts"], min(p["top_k"], p["experts"]), p["batch_tokens"], p["d_model"], p["d_ff"], p["layers"]
    g = rng(p["seed"])
    dr = 32
    centers = g.normal(0, 1, (6, dr))
    clusters = g.choice(6, size=T, p=np.array([0.35, 0.25, 0.15, 0.1, 0.1, 0.05]))
    X = centers[clusters] + g.normal(0, 0.7, (T, dr))
    Wr = g.normal(0, 1, (dr, E)) / np.sqrt(dr)
    bias = g.normal(0, 1, E) * p["router_skew"] * 2.5
    if p["aux_loss"]:
        bias *= 0.2  # effect of load-balancing training approximated by shrinking learned router bias
    logits = X @ Wr + bias
    if p["routing"] == "noisy":
        logits = logits + g.normal(0, 1, logits.shape)
    probs = softmax(logits)
    cap = int(math.ceil(p["capacity_factor"] * T * K / E))
    load = np.zeros(E, dtype=int)
    dropped = 0
    assign = []
    if p["routing"] == "expert_choice":
        cap_ec = int(math.ceil(T * K / E * p["capacity_factor"]))
        chosen = np.zeros((T, E), dtype=bool)
        for e in range(E):
            chosen[np.argsort(-probs[:, e])[:cap_ec], e] = True
        load = chosen.sum(0)
        unrouted = int((chosen.sum(1) == 0).sum())
        dropped = unrouted
        assign = [list(np.where(chosen[i])[0]) for i in range(T)]
    else:
        if p["routing"] == "hash":
            top = np.stack([(np.arange(T) * 2654435761 + k * 97) % E for k in range(K)], 1)
        else:
            top = np.argsort(-probs, axis=1)[:, :K]
        for i in range(T):
            got = []
            for e in top[i]:
                if load[e] < cap:
                    load[e] += 1
                    got.append(int(e))
                else:
                    dropped += 1
            assign.append(got)
    f = np.bincount(np.argmax(probs, 1), minlength=E) / T
    P = probs.mean(0)
    aux = float(E * np.sum(f * P))  # Switch Transformer load-balancing loss (1.0 = perfectly balanced)
    expert_params = 3 * d_model * d_ff
    attn = 4 * d_model * d_model
    total = layers * (attn + E * expert_params + d_model * E)
    active = layers * (attn + K * expert_params + d_model * E)
    dense_same_active = layers * (attn + K * expert_params)
    r = Result("simulated")
    r.set_fidelity("simulated", "Routing, capacity, token dropping and the Switch aux-loss are computed exactly on synthetic clustered token embeddings; parameter counts are exact for the given dims.")
    r.metric("total", "Total parameters", fmt_params(total), "", "all experts must be resident in memory")
    r.metric("active", "Active parameters / token", fmt_params(active), "", f"top-{K} of {E} experts")
    r.metric("ratio", "Active fraction", active / total * 100, "%")
    r.metric("dropped", "Dropped token-expert slots" if p["routing"] != "expert_choice" else "Tokens routed to no expert", dropped / (T * K if p["routing"] != "expert_choice" else T) * 100, "%",
             f"capacity = {cap} per expert", status="bad" if dropped else "good")
    r.metric("imbalance", "Load imbalance (max / mean)", float(load.max() / max(1e-9, load.mean())), "", "1.0 = perfect")
    r.metric("aux", "Load-balancing loss (Switch)", aux, "", "N·Σ f_i·P_i; 1.0 when uniform")
    r.metric("mem", "Weights memory BF16", total * 2 / 1e9, "GB")
    r.chart("util", "Expert utilization (tokens assigned)", "bar", [f"E{e}" for e in range(E)], [{"label": "tokens", "data": load.tolist()}], "expert", "tokens", threshold=cap)
    r.chart("prob", "Mean router probability per expert", "bar", [f"E{e}" for e in range(E)], [{"label": "P_i", "data": P.tolist()}], "expert", "prob")
    sample = [{"token": f"tok{i} (cluster {clusters[i]})", "probs": [round(float(x), 3) for x in probs[i]], "chosen": [int(e) for e in assign[i]]} for i in range(4)]
    r.panel("moe_route", "Sample token routing", tokens=sample, experts=E)
    r.table("Dense vs sparse comparison", ["Model", "Total params", "Active params/token", "FLOPs/token", "Memory (BF16)"], [
        ["Dense (same active compute)", fmt_params(dense_same_active), fmt_params(dense_same_active), f"{2 * dense_same_active / 1e9:.1f} GFLOP", f"{dense_same_active * 2 / 1e9:.1f} GB"],
        ["Dense (same total params)", fmt_params(total), fmt_params(total), f"{2 * total / 1e9:.1f} GFLOP", f"{total * 2 / 1e9:.1f} GB"],
        [f"MoE {E}×, top-{K}", fmt_params(total), fmt_params(active), f"{2 * active / 1e9:.1f} GFLOP", f"{total * 2 / 1e9:.1f} GB"]])
    r.compare("Dense (same total params)", "Sparse MoE", [("FLOPs per token", 2 * total / 1e9, 2 * active / 1e9, "GFLOP", "lower"),
                                                          ("Weight memory", total * 2 / 1e9, total * 2 / 1e9, "GB", "lower")])
    r.flow("MoE layer", [{"label": "Token hidden state"}, {"label": "Router (linear + softmax)", "cls": "accent"}, {"label": f"Top-{K} experts", "sub": f"of {E}"},
                         {"label": "Expert FFNs (parallel)"}, {"label": "Weighted sum"}])
    r.recommend("MoE trades memory for compute: you pay for total parameters in VRAM, but FLOPs only for active ones.",
                "Use a load-balancing auxiliary loss (or bias-based balancing) and capacity factor 1.0–1.25 in training.",
                "Serving: expert parallelism across GPUs; watch all-to-all communication and batch-size-dependent expert utilization.")
    return r


register(ModuleDef(
    id="mixture-of-experts", title="Mixture of Experts (MoE)", category="architecture", original_index=12, fidelity="simulated",
    summary="Sparse expert activation: a router sends each token to top-k of many feed-forward experts.",
    controls=[
        slider("experts", "Number of experts", 8, 2, 64, 1, "Expert FFNs per MoE layer."),
        slider("top_k", "Top-K experts per token", 2, 1, 8, 1, "Experts activated per token."),
        slider("d_model", "Hidden size (d_model)", 4096, 512, 8192, 256, "Model width."),
        slider("d_ff", "Expert FFN size (d_ff)", 14336, 1024, 32768, 256, "Each expert's intermediate size (SwiGLU)."),
        slider("layers", "Layers", 32, 1, 96, 1, "MoE layers."),
        select("routing", "Routing strategy", "topk", [{"value": "topk", "label": "Token-choice top-k (softmax)"}, {"value": "noisy", "label": "Noisy top-k"},
                                                      {"value": "expert_choice", "label": "Expert-choice"}, {"value": "hash", "label": "Hash routing"}], "How tokens are assigned."),
        slider("capacity_factor", "Expert capacity factor", 1.25, 0.5, 3.0, 0.05, "Capacity = factor × tokens × k / experts."),
        slider("batch_tokens", "Batch size (tokens)", 2048, 64, 16384, 64, "Tokens routed together."),
        slider("router_skew", "Router skew (imbalance)", 0.4, 0.0, 1.0, 0.05, "Bias toward popular experts."),
        toggle("aux_loss", "Load-balancing auxiliary loss", True, "Simulates a router trained with balancing loss."),
        slider("seed", "Random seed", 2, 0, 9999, 1, "Synthetic tokens seed."),
    ],
    simulate=moe_sim,
    docs=docs(
        what="An MoE transformer replaces the dense feed-forward block with many expert FFNs and a learned router that activates only the top-k experts per token.",
        why="Model quality scales with parameters, but compute scales with active parameters. MoE decouples them: e.g. Mixtral-8x7B has ~47B parameters but ~13B active per token.",
        how="Router logits = W_r·h; softmax; pick top-k; combine expert outputs weighted by router probabilities. Each expert has a capacity; overflow tokens are dropped (pass through residual). An auxiliary loss pushes balanced utilization.",
        when_use="Large-scale pretraining under compute budgets; serving where memory is plentiful but FLOPs/latency matter.",
        when_avoid="Memory-constrained edge deployment, tiny batch serving where expert weights are loaded but underutilized, or teams without distributed-systems expertise.",
        pros=["More quality per FLOP", "Expert specialization", "Faster than a dense model of the same total size"],
        cons=["All experts occupy memory", "Load imbalance and token dropping", "All-to-all communication in expert parallelism", "Harder fine-tuning and serving"],
        use_case="A multi-tenant platform serves Mixtral-8x7B on 2×A100-80GB: memory of a 47B model, decode cost closer to a 13B model.",
        code='''
import torch, torch.nn as nn, torch.nn.functional as F
class MoE(nn.Module):
    def __init__(self, d, d_ff, n_experts=8, k=2):
        super().__init__()
        self.router = nn.Linear(d, n_experts, bias=False); self.k = k
        self.experts = nn.ModuleList(nn.Sequential(nn.Linear(d, d_ff), nn.SiLU(), nn.Linear(d_ff, d)) for _ in range(n_experts))
    def forward(self, x):                                  # x: (tokens, d)
        probs = F.softmax(self.router(x), -1)
        w, idx = probs.topk(self.k, -1); w = w / w.sum(-1, keepdim=True)
        out = torch.zeros_like(x)
        for e, expert in enumerate(self.experts):
            tok, slot = (idx == e).nonzero(as_tuple=True)
            if tok.numel(): out[tok] += w[tok, slot, None] * expert(x[tok])
        f = F.one_hot(probs.argmax(-1), probs.size(-1)).float().mean(0)
        aux = probs.size(-1) * (f * probs.mean(0)).sum()   # Switch load-balancing loss
        return out, aux
''',
        diagram=[{"label": "Token h"}, {"label": "Router", "cls": "accent"}, {"label": "Top-k experts"}, {"label": "Expert FFNs", "cls": "accent"}, {"label": "Weighted combine"}],
        performance="FLOPs/token ≈ 2 × active params; memory ≈ total params × bytes. Throughput depends on expert load balance and interconnect for all-to-all.",
        checklist=["Budget memory for total params", "Monitor expert load and token drops", "Tune capacity factor", "Use expert parallelism with fast interconnect", "Evaluate per-domain quality"],
        applications=[{"domain": "Multi-agent enterprise platforms", "text": "Large shared MoE model serving diverse tasks efficiently."},
                      {"domain": "Enterprise chatbots", "text": "High-quality responses at mid-size-model latency."}],
        limitations="Token embeddings and router are synthetic; real routers' specialization patterns differ, but capacity/dropping mechanics are the same.",
        references=["Shazeer et al., 'Outrageously Large Neural Networks' (2017)", "Fedus et al., 'Switch Transformers' (2022)", "Jiang et al., 'Mixtral of Experts' (2024)"],
    ),
))


# ============================================================================ Ring attention
def ring_sim(p: dict, ctx: Context) -> Result:
    S, N, H, hd, bw_link = p["seq_len"], p["gpus"], p["heads"], p["head_dim"], p["link_gbs"] * 1e9
    hw = HARDWARE[p["gpu"]]
    F = hw["tflops"] * 1e12 * 0.5
    blk = S / N
    flops_step = 4 * blk * blk * H * hd  # QK^T and PV for one block pair (per layer)
    comm_bytes = 2 * blk * H * hd * 2  # K and V block, bf16
    t_comp = flops_step / F
    t_comm = comm_bytes / bw_link + p["hop_latency_us"] * 1e-6
    t_step = max(t_comp, t_comm)
    t_layer = N * t_step
    overlap_ok = t_comp >= t_comm
    hidden = H * hd
    mem_dev = (4 * blk * hidden * 2 + 2 * blk * hidden * 2 + p["block_size"] ** 2 * H * 4 / max(1, N)) / 1e9  # QKVO + recv buffer + tile
    mem_single = (4 * S * hidden * 2) / 1e9
    min_blk = F / (bw_link / (2 * 2))  # block size needed so compute hides communication
    r = Result("analytical")
    r.metric("tok_dev", "Tokens per device", int(blk))
    r.metric("mem_dev", "Activation memory / device (1 layer)", mem_dev, "GB", "Q,K,V,O blocks + receive buffer")
    r.metric("mem_single", "Single-device equivalent", mem_single, "GB", status="bad" if mem_single > hw["mem_gb"] else None)
    r.metric("t_comp", "Compute per ring step", t_comp * 1e3, "ms")
    r.metric("t_comm", "Communication per ring step", t_comm * 1e3, "ms")
    r.metric("overlap", "Compute hides communication?", "yes" if overlap_ok else "no — comm-bound", status="good" if overlap_ok else "bad")
    r.metric("min_block", "Min tokens/device to overlap", int(min_blk), "", "≈ FLOPS / (link bandwidth / 4 bytes)")
    r.metric("t_layer", "Attention time per layer", t_layer * 1e3, "ms")
    ns = [1, 2, 4, 8, 16, 32, 64]
    comp, comm, memd = [], [], []
    for n in ns:
        b = S / n
        comp.append(n * 4 * b * b * H * hd / F * 1e3)
        comm.append(n * (2 * b * H * hd * 2 / bw_link + p["hop_latency_us"] * 1e-6) * 1e3 if n > 1 else 0)
        memd.append(4 * b * hidden * 2 / 1e9)
    r.chart("scale", "Per-layer attention time vs. devices", "line", ns, [{"label": "compute (ms)", "data": comp}, {"label": "communication (ms, overlappable)", "data": comm}], "GPUs", "ms", log_y=True, log_x=True)
    r.chart("mem", "Activation memory per device vs. devices", "bar", ns, [{"label": "GB / device", "data": memd}], "GPUs", "GB", threshold=hw["mem_gb"])
    n_show = min(N, 8)
    sched = [[(dev - step) % n_show for dev in range(n_show)] for step in range(n_show)]
    r.heatmap(f"KV block held by each device at each ring step ({n_show} devices shown)", sched, x_labels="device", y_labels="ring step", categorical=True)
    r.panel("ring", "Ring topology", nodes=n_show, label=f"Each step: compute local Q × current KV block, send KV to next device ({comm_bytes / 1e6:.1f} MB)")
    r.compare("Single device (if it fit)", f"Ring attention on {N} GPUs", [("Activation memory / device", mem_single, mem_dev, "GB", "lower"),
                                                                           ("Attention time / layer", 4 * S * S * H * hd / F * 1e3, t_layer * 1e3, "ms", "lower")])
    r.note("Total attention FLOPs are unchanged — ring attention distributes them and their memory; speedup comes from parallel devices.")
    r.recommend("Keep tokens per device above the overlap threshold so KV transfers hide behind compute.",
                "Use NVLink/InfiniBand; PCIe-only clusters often become communication-bound.",
                "Combine with FlashAttention-style blockwise kernels inside each device and causal load balancing (striped/zigzag ring).")
    return r


register(ModuleDef(
    id="ring-attention", title="Ring Attention", category="architecture", original_index=14, fidelity="analytical",
    summary="Distribute very long sequences across GPUs; pass KV blocks around a ring while computing blockwise attention.",
    controls=[
        slider("seq_len", "Sequence length", 1048576, 8192, 8388608, 8192, "Total tokens.", log=True, unit="tok"),
        slider("gpus", "Number of GPUs", 8, 1, 64, 1, "Devices in the ring."),
        slider("block_size", "Inner block (tile) size", 1024, 128, 8192, 128, "Blockwise attention tile per device."),
        slider("heads", "Attention heads", 32, 8, 128, 8, "Heads per layer."),
        slider("head_dim", "Head dimension", 128, 64, 256, 64, "Per-head size."),
        select("gpu", "GPU type", "h100", HARDWARE_OPTIONS, "Compute per device."),
        slider("link_gbs", "Interconnect bandwidth", 450, 16, 900, 2, "Per-direction GB/s (PCIe4 ≈ 32, NVLink4 ≈ 450).", unit="GB/s"),
        slider("hop_latency_us", "Per-hop latency (communication overhead)", 10, 1, 200, 1, "Fixed latency per transfer.", unit="µs"),
    ],
    simulate=ring_sim,
    docs=docs(
        what="Ring Attention shards a sequence across N devices. Each device keeps its query block and passes key/value blocks around a ring; after N steps every query block has attended to every KV block.",
        why="Exact attention over millions of tokens cannot fit activations on one GPU. Blockwise computation with online softmax makes attention decomposable, and the ring overlaps communication with computation.",
        how="Device i holds Q_i, K_i, V_i. For step s it computes blockwise attention of Q_i against the KV block it currently holds (accumulating with online-softmax statistics), while sending that block to device i+1 and receiving from i−1. Communication is hidden when compute per block ≥ transfer time.",
        when_use="Training or inference with sequences far beyond single-device memory (books, genomes, long video, repository-scale code).",
        when_avoid="Contexts that fit on one device (FlashAttention suffices), slow interconnects, or workloads where retrieval is cheaper than full attention.",
        pros=["Context length scales linearly with device count", "Exact attention (no approximation)", "Communication overlapped with compute"],
        cons=["Requires multi-GPU clusters with fast links", "Causal masking creates load imbalance without striping", "Complex implementation"],
        use_case="A research team trains a 1M-token context model across 32 H100s with ring attention to reason over full legal case histories.",
        code='''
# Pseudocode for one layer on device `rank` (online-softmax accumulation)
q = local_q; k, v = local_k, local_v
acc, m, l = zeros_like(q), full(-inf), zeros()
for step in range(world_size):
    send_handle = isend((k, v), dst=(rank + 1) % world_size)       # overlap comm…
    s = q @ k.transpose(-1, -2) / sqrt(d)                           # …with compute
    m_new = maximum(m, s.max(-1)); p = exp(s - m_new[..., None])
    acc = acc * exp(m - m_new)[..., None] + p @ v
    l = l * exp(m - m_new) + p.sum(-1); m = m_new
    k, v = recv(src=(rank - 1) % world_size); send_handle.wait()
out = acc / l[..., None]
''',
        diagram=[{"label": "Shard sequence N ways"}, {"label": "Local Q·KV block", "cls": "accent"}, {"label": "Send KV → next"}, {"label": "Online softmax accumulate"}, {"label": "After N steps: exact attention", "cls": "accent"}],
        performance="Per-step compute ∝ (S/N)²·H·d; per-step communication ∝ (S/N)·H·d. Overlap requires S/N ≳ FLOPS / link-bandwidth × bytes factor.",
        checklist=["Verify tokens/device above overlap threshold", "Use NVLink/IB", "Stripe causal sequences for balance", "Profile comm vs compute", "Combine with sequence-parallel MLP"],
        applications=[{"domain": "Document intelligence", "text": "Million-token analysis of entire document collections."},
                      {"domain": "AI-assisted software engineering", "text": "Training models on whole repositories as single sequences."}],
        limitations="Roofline estimate at 50% MFU; ignores kernel overheads, causal imbalance and MLP sequence parallelism.",
        references=["Liu et al., 'Ring Attention with Blockwise Transformers for Near-Infinite Context' (2023)"],
    ),
))
