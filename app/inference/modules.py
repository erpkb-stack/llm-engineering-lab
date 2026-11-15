"""Module 3 (Temperature Sampling), Module 9 (Speculative Decoding), Module 15 (Medusa Heads), Quantization."""
from __future__ import annotations

import math

import numpy as np

from app.engine.common import active_hw, downsample, effs, hw_note, rng, softmax
from app.engine.hardware import DTYPE_BYTES, MODEL_OPTIONS, MODELS, fmt_params
from app.engine.registry import Context, ModuleDef, Result, docs, register, select, slider, text, toggle

# ============================================================================ Temperature
# Synthetic next-token logits that mimic the *shape* of real model distributions for each scenario.
# ok=False marks tokens that are factually wrong / break the JSON grammar.
SCENARIOS = {
    "factual": {"prompt": "Q: What is the capital of France?\nA: The capital of France is", "history": ["capital", "France"],
                "tokens": [(" Paris", 9.1, True), (" Lyon", 4.2, False), (" Marseille", 3.6, False), (" located", 3.4, True),
                           (" a", 3.1, True), (" the", 2.8, True), (" France", 2.5, False), (" Nice", 2.1, False),
                           (" Brussels", 1.4, False), (" London", 1.0, False), (" not", 0.6, False), (" Berlin", 0.4, False)]},
    "creative": {"prompt": "Write the opening line of a fantasy story: Once upon a time, there was a", "history": ["time"],
                 "tokens": [(" young", 5.0, True), (" little", 4.8, True), (" king", 4.6, True), (" girl", 4.5, True),
                            (" dragon", 4.4, True), (" village", 4.1, True), (" wizard", 3.9, True), (" small", 3.7, True),
                            (" mysterious", 3.2, True), (" forgotten", 2.9, True), (" clockwork", 2.2, True), (" time", 2.0, True)]},
    "json": {"prompt": 'Return JSON only. {"order_id": 4812, "status": ', "history": ["status"],
             "tokens": [(' "', 8.4, True), (" null", 4.0, True), ("'", 3.3, False), (" shipped", 3.1, False), (' ""', 2.6, True),
                        (" {", 1.9, False), (" Status", 1.5, False), (" pending", 1.4, False), (" \\n", 1.1, False),
                        (" status", 0.9, False), (" ```", 0.7, False), (" The", 0.5, False)]},
}


def sample_dist(logits: np.ndarray, toks: list[str], history: list[str], T: float, top_k: int, top_p: float, rep: float):
    z = logits.astype(float).copy()
    if rep != 1.0:  # CTRL-style repetition penalty on previously generated tokens
        for i, t in enumerate(toks):
            if t.strip() in history:
                z[i] = z[i] / rep if z[i] > 0 else z[i] * rep
    if T <= 1e-6:  # greedy
        probs = np.zeros_like(z)
        probs[int(np.argmax(z))] = 1.0
        return probs, z
    probs = softmax(z / T)
    keep = np.ones_like(probs, dtype=bool)
    order = np.argsort(-probs)
    if 0 < top_k < len(probs):
        keep[order[top_k:]] = False
    if top_p < 1.0:
        cum = np.cumsum(probs[order])
        cutoff = int(np.searchsorted(cum, top_p) + 1)
        keep[order[cutoff:]] = False
    probs = np.where(keep, probs, 0)
    return probs / probs.sum(), z


async def temperature_sim(p: dict, ctx: Context) -> Result:
    sc = SCENARIOS[p["scenario"]]
    toks = [t for t, _, _ in sc["tokens"]]
    logits = np.array([l for _, l, _ in sc["tokens"]])
    ok = np.array([o for _, _, o in sc["tokens"]])
    probs, adj = sample_dist(logits, toks, sc["history"], p["temperature"], p["top_k"], p["top_p"], p["repetition_penalty"])
    base = softmax(logits)
    g = rng(p["seed"])
    draws = g.choice(len(toks), size=p["runs"], p=probs)
    counts = np.bincount(draws, minlength=len(toks))
    r = Result("simulated")
    r.set_fidelity("simulated", "Real sampling algorithms (temperature, top-k, top-p, repetition penalty) applied to synthetic next-token logits.")
    ent = float(-(probs[probs > 0] * np.log2(probs[probs > 0])).sum())
    r.metric("argmax", "Most likely token", toks[int(np.argmax(adj))].strip())
    r.metric("selected", "Selected (run #1)", toks[int(draws[0])].strip())
    r.metric("entropy", "Entropy of sampling distribution", ent, "bits", "0 = deterministic")
    r.metric("candidates", "Tokens with non-zero probability", int((probs > 0).sum()))
    r.metric("p_bad", "P(incorrect / invalid token)", float(probs[~ok].sum() * 100), "%",
             "probability mass on tokens that are factually wrong or break the JSON grammar",
             status="bad" if probs[~ok].sum() > 0.05 else None)
    r.metric("distinct", f"Distinct outputs in {p['runs']} runs", int((counts > 0).sum()))
    r.chart("dist", "Next-token probability distribution", "bar", [t.strip() or repr(t) for t in toks], [
        {"label": "Raw model (T=1, no filtering)", "data": base.tolist()},
        {"label": f"After T={p['temperature']}, top-k={p['top_k']}, top-p={p['top_p']}", "data": probs.tolist()},
        {"label": f"Empirical frequency ({p['runs']} runs)", "data": (counts / p["runs"]).tolist()},
    ], "token", "probability")
    temps = np.linspace(0.05, 2.0, 40)
    bad = [float(sample_dist(logits, toks, sc["history"], t, p["top_k"], p["top_p"], p["repetition_penalty"])[0][~ok].sum()) for t in temps]
    ents = [float(-(lambda q: (q[q > 0] * np.log2(q[q > 0])).sum())(sample_dist(logits, toks, sc["history"], t, p["top_k"], p["top_p"], p["repetition_penalty"])[0])) for t in temps]
    r.chart("sweep", "Temperature sweep (current top-k / top-p)", "line", [round(t, 2) for t in temps], [
        {"label": "P(incorrect/invalid)", "data": bad}, {"label": "Entropy (bits)", "data": ents, "yAxisID": "y1"}],
        "temperature", "probability", dual_axis=True)
    r.table("Runs", ["Run", "Seed", "Selected token", "Correct / valid?"],
            [[i + 1, p["seed"] + i, toks[int(d)].strip(), "yes" if ok[int(d)] else "NO"] for i, d in enumerate(draws[:20])])
    r.panel("code", "Prompt", content=sc["prompt"])
    greedy, _ = sample_dist(logits, toks, sc["history"], 0.0, 0, 1.0, 1.0)
    r.compare("Greedy (T=0)", "Current settings", [
        ("Entropy", 0.0, ent, "bits", None), ("P(incorrect/invalid)", float(greedy[~ok].sum() * 100), float(probs[~ok].sum() * 100), "%", "lower"),
        ("Distinct outputs", 1, int((counts > 0).sum()), "", None)])
    if p["use_real_model"]:
        prov = ctx.providers
        if prov is None or not prov.has_real():
            r.note("No real model provider configured (Settings → Model provider). Showing simulation only.", "warn")
        else:
            outs = []
            for i in range(min(5, p["runs"])):
                res = await prov.generate(sc["prompt"], temperature=p["temperature"], top_p=p["top_p"], top_k=p["top_k"],
                                          max_tokens=12, seed=p["seed"] + i, repetition_penalty=p["repetition_penalty"])
                outs.append([i + 1, res.get("text", "").strip()[:120], res.get("provider"), f"{res.get('latency_ms', 0):.0f} ms"])
            r.table("Real model outputs (measured)", ["Run", "Output", "Provider", "Latency"], outs)
            r.note("Rows in 'Real model outputs' are actual generations from your configured provider.")
    advice = {"factual": "Use T=0–0.3 (or greedy) for factual QA; sampling only adds ways to be wrong.",
              "creative": "T≈0.7–1.0 with top-p 0.9–0.95 gives diversity without incoherence.",
              "json": "Use T=0 plus constrained decoding / JSON-schema mode; validate and retry on parse errors."}
    r.recommend(advice[p["scenario"]], "Set a seed (where supported) for reproducible evaluations.",
                "Temperature > 1.2 flattens the tail and samples low-quality tokens; pair high T with top-p.",
                "Repetition penalty fights loops but can also suppress legitimately repeated entities (names, JSON keys).")
    return r


register(ModuleDef(
    id="temperature-sampling", title="Temperature Sampling", category="inference", original_index=3, fidelity="simulated",
    summary="How temperature, top-k, top-p and repetition penalty reshape the next-token distribution.",
    controls=[
        select("scenario", "Scenario", "factual", [{"value": "factual", "label": "Factual QA"}, {"value": "creative", "label": "Creative writing"},
                                                   {"value": "json", "label": "Structured JSON"}], "Prompt type with its synthetic next-token logits."),
        slider("temperature", "Temperature", 0.7, 0.0, 2.0, 0.05, "Divides logits before softmax. 0 = greedy, >1 flattens."),
        slider("top_k", "Top-K", 0, 0, 12, 1, "Keep only the K most likely tokens (0 = disabled)."),
        slider("top_p", "Top-P (nucleus)", 1.0, 0.05, 1.0, 0.05, "Keep the smallest set whose cumulative probability ≥ p."),
        slider("repetition_penalty", "Repetition penalty", 1.0, 1.0, 2.0, 0.05, "Penalizes tokens already in the context (CTRL-style)."),
        slider("seed", "Random seed", 42, 0, 9999, 1, "Fixed seed → reproducible draws."),
        slider("runs", "Number of generations", 50, 1, 500, 1, "Independent samples to compare."),
        toggle("use_real_model", "Also run on real model (if configured)", False, "Uses Ollama / OpenAI-compatible provider from Settings."),
    ],
    simulate=temperature_sim,
    docs=docs(
        what="Temperature T rescales logits before softmax: p_i = exp(z_i/T) / Σ exp(z_j/T). Top-k and top-p then truncate the tail before sampling.",
        why="A model outputs a distribution, not an answer. The decoding policy decides how much of that uncertainty is exposed: low T for reliability, higher T for diversity.",
        how="T→0 approaches argmax (greedy). T=1 samples the model's own distribution. T>1 flattens it, giving more mass to unlikely tokens. Top-k keeps K candidates; top-p keeps the minimal nucleus with cumulative probability ≥ p. Repetition penalty divides positive logits of seen tokens.",
        when_use="Tune per task: extraction/QA/classification/JSON → T≈0; summarization → 0.2–0.5; brainstorming/creative → 0.7–1.0 with top-p.",
        when_avoid="Do not use high temperature for anything that must be correct or parseable. Do not rely on temperature for safety; it is not a guardrail.",
        pros=["Simple single knob for diversity", "Top-p adapts the candidate set to model confidence", "Enables self-consistency (sample N, vote)"],
        cons=["Non-determinism complicates testing", "High T increases hallucination and format errors", "Effects are model-specific — the same T behaves differently across models"],
        use_case="An insurance claims extractor returns invalid JSON 3% of the time at T=0.8; switching to T=0 with schema-constrained decoding removes parse failures.",
        code='''
import torch
def sample(logits, temperature=0.7, top_k=0, top_p=1.0, generator=None):
    if temperature == 0:
        return logits.argmax(-1)
    probs = torch.softmax(logits / temperature, dim=-1)
    if top_k:
        kth = torch.topk(probs, top_k).values[..., -1, None]
        probs = torch.where(probs < kth, 0.0, probs)
    if top_p < 1.0:
        sorted_p, idx = probs.sort(descending=True)
        mask = sorted_p.cumsum(-1) - sorted_p > top_p
        sorted_p[mask] = 0.0
        probs = torch.zeros_like(probs).scatter(-1, idx, sorted_p)
    return torch.multinomial(probs / probs.sum(-1, keepdim=True), 1, generator=generator)

# HF: model.generate(**inputs, do_sample=True, temperature=0.7, top_p=0.9, top_k=50, repetition_penalty=1.1)
''',
        diagram=[{"label": "Logits z"}, {"label": "Repetition penalty"}, {"label": "÷ temperature", "cls": "accent"},
                 {"label": "softmax"}, {"label": "top-k / top-p filter"}, {"label": "Sample"}],
        performance="Negligible compute cost. Indirect cost: higher T → more retries/validation failures and longer outputs.",
        checklist=["Set decoding params per endpoint, not globally", "Use T=0 + constrained decoding for structured outputs", "Log seed + params with each output",
                   "Evaluate quality at the exact production decoding settings", "Cap max_tokens"],
        applications=[{"domain": "Document intelligence", "text": "Extraction pipelines run at T=0 with schema validation."},
                      {"domain": "Customer support automation", "text": "Low T for policy answers; modest T for empathetic phrasing variants."},
                      {"domain": "AI-assisted software engineering", "text": "Code generation uses low T; sampling N at higher T + tests picks the best (pass@k)."}],
        limitations="Logits are synthetic but chosen to resemble real distributions; the sampling math is exactly what inference engines implement.",
        references=["Holtzman et al., 'The Curious Case of Neural Text Degeneration' (2020)", "Keskar et al., 'CTRL' (2019)"],
    ),
))


# ============================================================================ Speculative decoding
def _step_time(params_b: float, tokens: int, batch: int, bw: float, F: float, bytes_per=2.0) -> float:
    wb = params_b * 1e9 * bytes_per
    return max(wb / bw, 2 * params_b * 1e9 * tokens * batch / F)


def spec_sim(p: dict, ctx: Context) -> Result:
    hw = active_hw(ctx)
    me, ce = effs(ctx)
    bw, F = hw["bw_gbs"] * 1e9 * me, hw["tflops"] * 1e12 * ce
    a, gam, B = p["acceptance"], p["draft_tokens"], p["batch"]
    Pd, Pt = p["draft_b"], p["target_b"]

    def expected_tokens(alpha, g):
        return (1 - alpha ** (g + 1)) / (1 - alpha) if alpha < 1 else g + 1

    t_target = _step_time(Pt, 1, B, bw, F)
    t_draft = _step_time(Pd, 1, B, bw, F)

    def speed(alpha, g):
        it = g * t_draft + _step_time(Pt, g + 1, B, bw, F)
        return expected_tokens(alpha, g) * t_target / it

    s = speed(a, gam)
    best_g = max(range(1, 17), key=lambda g: speed(a, g))
    r = Result("analytical")
    r.metric("tokens_per_iter", "Expected tokens / target pass", expected_tokens(a, gam), "", "(1 − α^(γ+1)) / (1 − α)")
    r.metric("speedup", "Estimated speedup", s, "×", status="good" if s > 1.2 else ("bad" if s < 1 else None))
    r.metric("best_gamma", "Optimal draft length γ*", best_g)
    r.metric("cost_ratio", "Draft/target cost ratio c", t_draft / t_target, "", "per-token step time ratio")
    r.metric("tpot_std", "Standard TPOT", t_target * 1e3, "ms")
    r.metric("tpot_spec", "Speculative effective TPOT", t_target / s * 1e3, "ms")
    r.metric("mem", "Extra memory (draft weights)", Pd * 2, "GB", "plus the draft model's own KV cache")
    gs = list(range(1, 17))
    r.chart("speedup", "Speedup vs draft length for different acceptance rates", "line", gs,
            [{"label": f"α={al}", "data": [speed(al, g) for g in gs]} for al in (0.5, 0.6, 0.7, 0.8, 0.9)] +
            [{"label": f"current α={a}", "data": [speed(a, g) for g in gs], "borderDash": [6, 4]}], "draft tokens γ", "speedup ×")
    bs = [1, 2, 4, 8, 16, 32, 64, 128]
    r.chart("batch", "Speedup shrinks as batch grows (target becomes compute-bound)", "line", bs,
            [{"label": "speedup", "data": [expected_tokens(a, gam) * _step_time(Pt, 1, b, bw, F) / (gam * _step_time(Pd, 1, b, bw, F) + _step_time(Pt, gam + 1, b, bw, F)) for b in bs]}],
            "batch size", "×", log_x=True)
    # Monte Carlo trace (seeded) of draft proposals and verification
    g = rng(p["seed"])
    rows, total_tokens = [], 0
    for it in range(10):
        accepted = 0
        marks = []
        for _ in range(gam):
            if g.random() < a:
                accepted += 1
                marks.append("✓")
            else:
                marks.append("✗")
                break
        marks += ["·"] * (gam - len(marks))
        produced = accepted + 1  # target always contributes one token (correction or bonus)
        total_tokens += produced
        rows.append([it + 1, " ".join(marks), accepted, produced])
    r.table("Sample verification trace (seeded Monte Carlo)", ["Iteration", "Draft tokens (✓ accepted, ✗ rejected, · discarded)", "Accepted", "Tokens emitted"], rows)
    r.flow("One speculative iteration", [
        {"label": f"Draft model ({Pd}B)", "sub": f"proposes γ={gam} tokens"},
        {"label": f"Target model ({Pt}B)", "sub": "scores γ+1 positions in ONE pass", "cls": "accent"},
        {"label": "Rejection sampling", "sub": "accept prefix; resample at first mismatch"},
        {"label": "Output", "sub": f"≈{expected_tokens(a, gam):.2f} tokens / pass"}])
    r.compare("Standard autoregressive", "Speculative decoding", [
        ("Time per output token", t_target * 1e3, t_target / s * 1e3, "ms", "lower"),
        ("Target forward passes per 100 tokens", 100, 100 / expected_tokens(a, gam), "", "lower"),
        ("GPU memory for weights", Pt * 2, (Pt + Pd) * 2, "GB", "lower")])
    if s < 1:
        r.note("Speculation is SLOWER here: acceptance too low or draft too expensive relative to target.", "warn")
    if Pd >= Pt:
        r.note("Draft model should be much smaller than the target.", "warn")
    r.note("Output distribution is mathematically identical to the target model's (with the rejection-sampling acceptance rule).")
    r.note(hw_note(ctx))
    r.recommend("Measure α on your own traffic — it depends on domain, draft quality and temperature.",
                "Draft and target must share the tokenizer/vocabulary.",
                "Biggest wins at low batch (latency-bound serving); at high batch prefer continuous batching throughput.",
                "Alternatives without a separate draft model: Medusa heads, EAGLE, n-gram/prompt-lookup decoding.")
    return r


register(ModuleDef(
    id="speculative-decoding", title="Speculative Decoding", category="inference", original_index=9, fidelity="analytical",
    summary="A small draft model proposes tokens; the large target model verifies several in one pass.",
    controls=[
        slider("draft_b", "Draft model size", 1.0, 0.1, 13.0, 0.1, "Billions of parameters.", unit="B"),
        slider("target_b", "Target model size", 70.0, 1.0, 180.0, 1.0, "Billions of parameters.", unit="B"),
        slider("draft_tokens", "Draft tokens per iteration (γ)", 4, 1, 16, 1, "Tokens the draft proposes before verification."),
        slider("acceptance", "Per-token acceptance rate (α)", 0.75, 0.1, 0.98, 0.01, "Probability the target accepts each draft token."),
        slider("batch", "Verification batch size", 1, 1, 128, 1, "Concurrent sequences; higher batch → less idle compute to exploit."),
        slider("seed", "Random seed", 3, 0, 9999, 1, "For the Monte Carlo trace."),
    ],
    simulate=spec_sim,
    docs=docs(
        what="An exact acceleration technique: a cheap draft model guesses the next γ tokens; the target model evaluates all γ+1 positions in one forward pass and accepts the longest prefix consistent with its own distribution.",
        why="Single-sequence decoding is memory-bandwidth-bound: the GPU streams all weights to produce one token and its compute sits idle. Verifying several tokens per pass uses that idle compute.",
        how="For each draft token x with draft prob q(x) and target prob p(x), accept with probability min(1, p/q); on rejection, resample from the normalized residual max(0, p − q). This preserves the target distribution exactly. Expected tokens per pass = (1 − α^{γ+1})/(1 − α).",
        when_use="Latency-sensitive, low-to-moderate batch serving of large models where a well-aligned smaller model from the same family exists (e.g., Llama-3.2-1B drafting for Llama-3.1-70B).",
        when_avoid="High-batch throughput serving (compute already saturated), low acceptance domains, or when there is no tokenizer-compatible draft.",
        pros=["Lossless — identical output distribution", "2–3× latency reduction is common when α is high", "No retraining of the target"],
        cons=["Extra memory for the draft model and its KV cache", "Gains collapse when α is low or batch is large", "Operational complexity: two models to version and serve"],
        use_case="A code assistant on a 70B model uses a 1B draft; code is highly predictable (α≈0.8), so per-token latency falls ~2×.",
        code='''
from transformers import AutoModelForCausalLM, AutoTokenizer
tok = AutoTokenizer.from_pretrained("meta-llama/Llama-3.1-8B-Instruct")
target = AutoModelForCausalLM.from_pretrained("meta-llama/Llama-3.1-8B-Instruct", device_map="auto")
draft  = AutoModelForCausalLM.from_pretrained("meta-llama/Llama-3.2-1B-Instruct", device_map="auto")
inputs = tok("Write a binary search in Python.", return_tensors="pt").to(target.device)
out = target.generate(**inputs, assistant_model=draft, max_new_tokens=200)   # assisted generation

# vLLM: LLM(model=target_id, speculative_config={"model": draft_id, "num_speculative_tokens": 5})
''',
        diagram=[{"label": "Draft γ tokens"}, {"label": "Target verifies in 1 pass", "cls": "accent"}, {"label": "Accept prefix"},
                 {"label": "Resample at mismatch"}, {"label": "Repeat"}],
        performance="Speedup ≈ E[tokens/pass] × T_target / (γ·T_draft + T_verify(γ+1)). Memory overhead = draft weights + draft KV.",
        checklist=["Same tokenizer for draft and target", "Measure acceptance rate per workload", "Tune γ (usually 3–7)", "Benchmark at production batch sizes", "Monitor latency p50/p99, not just mean"],
        applications=[{"domain": "AI-assisted software engineering", "text": "Predictable code tokens give high acceptance rates."},
                      {"domain": "Enterprise chatbots", "text": "Reduces per-token latency for large flagship models in interactive chat."}],
        limitations="Speedup uses the i.i.d. acceptance model from Leviathan et al.; real acceptance varies by position and content.",
        references=["Leviathan et al., 'Fast Inference from Transformers via Speculative Decoding' (2023)", "Chen et al., 'Accelerating LLM Decoding with Speculative Sampling' (2023)"],
    ),
))


# ============================================================================ Medusa
def medusa_sim(p: dict, ctx: Context) -> Result:
    hw = active_hw(ctx)
    me, ce = effs(ctx)
    bw, F = hw["bw_gbs"] * 1e9 * me, hw["tflops"] * 1e12 * ce
    arch = MODELS[p["model"]]
    K, w, depth, a1, decay = p["heads"], p["tree_width"], min(p["verify_depth"], p["heads"]), p["acceptance"], p["decay"]
    head_acc = [a1 * (decay ** k) for k in range(K)]
    # heuristic: considering w candidates raises the chance the right token is among them
    tree_acc = [1 - (1 - acc) ** (1 + math.log2(w)) for acc in head_acc]
    nodes, level = 0, 1
    for _ in range(depth):
        level *= w
        nodes += level
    nodes = min(nodes, p["candidates"])
    exp_len = 1.0
    prod = 1.0
    for k in range(depth):
        prod *= tree_acc[k]
        exp_len += prod
    N = arch.total_params()
    head_params = K * (arch.hidden * arch.hidden + arch.hidden * arch.vocab)
    wb = N * 2
    t_std = max(wb / bw, 2 * N / F)
    t_med = max((wb + head_params * 2) / bw, 2 * (N + head_params) * (nodes + 1) / F)
    speed = exp_len * t_std / t_med
    r = Result("analytical")
    r.metric("exp_len", "Expected tokens / step", exp_len, "", "1 base token + accepted speculative tokens")
    r.metric("speedup", "Estimated speedup", speed, "×", status="good" if speed > 1.3 else ("bad" if speed < 1 else None))
    r.metric("nodes", "Tree candidates verified", nodes, "", "tokens in tree attention per step")
    r.metric("head_params", "Medusa head parameters", fmt_params(head_params), "", "each head: residual block + LM head")
    r.metric("head_mem", "Head memory (BF16)", head_params * 2 / 1e9, "GB")
    r.chart("head_acc", "Per-head acceptance (top-1 vs. with tree width)", "bar", [f"head {k + 1} (t+{k + 2})" for k in range(K)], [
        {"label": "top-1 acceptance", "data": head_acc}, {"label": f"top-{w} (tree) acceptance", "data": tree_acc}], "", "probability")
    ws = list(range(1, 11))
    def spd(width):
        n, lv = 0, 1
        for _ in range(depth):
            lv *= width
            n += lv
        n = min(n, p["candidates"])
        acc = [1 - (1 - a) ** (1 + math.log2(width)) for a in head_acc]
        e, pr = 1.0, 1.0
        for k in range(depth):
            pr *= acc[k]
            e += pr
        return e * t_std / max((wb + head_params * 2) / bw, 2 * (N + head_params) * (n + 1) / F)
    r.chart("width", "Speedup vs. tree width", "line", ws, [{"label": "speedup", "data": [spd(x) for x in ws]}], "tree width", "×")
    # candidate tree drawing with seeded accepted path
    g = rng(p["seed"])
    tree, path_ok = [], True
    for k in range(depth):
        level_nodes = []
        hit = g.random() < tree_acc[k] and path_ok
        chosen = int(g.integers(0, w)) if hit else -1
        for j in range(min(w, 6)):
            level_nodes.append({"label": f"h{k + 1}·c{j + 1}", "accepted": bool(hit and j == chosen)})
        path_ok = hit
        tree.append(level_nodes)
    r.panel("tree", "Candidate token tree (one sampled step; green = accepted path)", levels=tree)
    r.flow("Medusa step", [{"label": "Base model forward", "sub": "hidden state h_t"},
                           {"label": f"{K} Medusa heads", "sub": "predict t+2 … t+K+1", "cls": "accent"},
                           {"label": "Build candidate tree", "sub": f"width {w}, depth {depth}"},
                           {"label": "Tree attention verify", "sub": f"{nodes} candidates, 1 pass", "cls": "accent"},
                           {"label": "Accept longest valid path"}])
    r.compare("Standard decoding", "Medusa-assisted", [
        ("Tokens per forward pass", 1, exp_len, "", "higher"), ("Time per token", t_std * 1e3, t_std / speed * 1e3, "ms", "lower"),
        ("Extra parameters", 0, head_params / 1e6, "M", "lower")])
    r.note("Acceptance-vs-width uses a stated heuristic (1 − (1 − a)^(1+log2 w)); measure real head accuracy on your data.")
    r.note(hw_note(ctx))
    r.recommend("Train heads on data resembling production traffic (self-distillation preserves the model's style).",
                "Medusa-1 (frozen backbone) is lossless with typical-acceptance off; Medusa-2 joint training gives more speed but changes the model.",
                "Keep tree size modest — verification cost grows with candidates once compute-bound.")
    return r


register(ModuleDef(
    id="medusa-heads", title="Medusa Heads", category="inference", original_index=15, fidelity="analytical",
    summary="Extra decoding heads predict several future tokens; a tree of candidates is verified in one pass.",
    controls=[
        select("model", "Base model", "llama-2-7b", MODEL_OPTIONS, "Determines head size (hidden × vocab)."),
        slider("heads", "Number of Medusa heads", 4, 1, 6, 1, "Head k predicts token t+k+1."),
        slider("tree_width", "Tree width (top candidates per head)", 3, 1, 10, 1, "Candidates kept per head."),
        slider("verify_depth", "Verification depth", 4, 1, 6, 1, "Tree depth verified (≤ heads)."),
        slider("candidates", "Max candidate tokens", 64, 4, 256, 4, "Cap on tree nodes per verification pass."),
        slider("acceptance", "Head-1 top-1 acceptance", 0.6, 0.2, 0.95, 0.01, "Accuracy of the first head."),
        slider("decay", "Acceptance decay per head", 0.8, 0.4, 1.0, 0.01, "Later heads are less accurate."),
        slider("seed", "Random seed", 11, 0, 9999, 1, "For the sampled tree path."),
    ],
    simulate=medusa_sim,
    docs=docs(
        what="Medusa adds K lightweight heads on top of the base model's last hidden state; head k predicts the token k+1 positions ahead. Candidates form a tree that is verified with a special tree-attention mask in a single forward pass.",
        why="Speculative decoding needs a separate draft model. Medusa gets similar multi-token-per-pass speedups from the same model with small trained heads.",
        how="Heads are trained (backbone frozen in Medusa-1) to predict future tokens. At inference, top-w candidates from each head are combined into a sparse tree; the base model scores all tree nodes at once and the longest prefix that matches its own prediction (or typical acceptance) is kept.",
        when_use="Latency-bound serving where you control the model and can train heads; when no good draft model exists.",
        when_avoid="Closed models you cannot modify, high-batch serving, or when you cannot afford head training and evaluation.",
        pros=["No separate draft model", "Small parameter overhead", "2×-ish speedups reported"],
        cons=["Requires training heads per base model", "Implementation-specific tree attention kernels", "Gains depend on head accuracy and batch size"],
        use_case="An internal coding assistant on a fine-tuned 7B model trains Medusa heads on its own outputs to cut latency without operating a second model.",
        code='''
import torch, torch.nn as nn
class MedusaHead(nn.Module):
    def __init__(self, hidden, vocab):
        super().__init__()
        self.block = nn.Sequential(nn.Linear(hidden, hidden), nn.SiLU())
        self.lm_head = nn.Linear(hidden, vocab, bias=False)
    def forward(self, h):                  # residual block then vocabulary projection
        return self.lm_head(h + self.block(h))

heads = nn.ModuleList(MedusaHead(4096, 32000) for _ in range(4))
# training: loss_k = CE(heads[k](h_t), token_{t+k+2}); base model frozen (Medusa-1)
''',
        diagram=[{"label": "Base forward"}, {"label": "K heads", "cls": "accent"}, {"label": "Candidate tree"},
                 {"label": "Tree attention verify", "cls": "accent"}, {"label": "Accept path"}],
        performance="Each step costs one base forward over (tree nodes + 1) positions plus head compute. Speedup ≈ accepted length / step-cost ratio.",
        checklist=["Train heads on representative data", "Validate output equivalence (or quality) vs. baseline", "Tune tree shape per hardware", "Benchmark at real batch sizes"],
        applications=[{"domain": "AI-assisted software engineering", "text": "Faster completion from a self-hosted fine-tuned model."},
                      {"domain": "Enterprise chatbots", "text": "Latency reduction without a second model to operate."}],
        limitations="Acceptance numbers are user inputs + heuristic; actual Medusa acceptance must be measured.",
        references=["Cai et al., 'Medusa: Simple LLM Inference Acceleration Framework with Multiple Decoding Heads' (2024)"],
    ),
))


# ============================================================================ Quantization
NF4 = np.array([-1.0, -0.6961928009986877, -0.5250730514526367, -0.39491748809814453, -0.28444138169288635,
                -0.18477343022823334, -0.09105003625154495, 0.0, 0.07958029955625534, 0.16093020141124725,
                0.24611230194568634, 0.33791524171829224, 0.44070982933044434, 0.5626170039176941,
                0.7229568362236023, 1.0])


def to_bf16(x: np.ndarray) -> np.ndarray:
    u = x.astype(np.float32).view(np.uint32)
    rounded = ((u + 0x7FFF + ((u >> 16) & 1)) & 0xFFFF0000).astype(np.uint32)
    return rounded.view(np.float32)


def quantize(W: np.ndarray, fmt: str, granularity: str, group: int) -> tuple[np.ndarray, float]:
    """Returns (dequantized W, scale-overhead bits per weight)."""
    if fmt == "fp32":
        return W.copy(), 0.0
    if fmt == "fp16":
        return W.astype(np.float16).astype(np.float32), 0.0
    if fmt == "bf16":
        return to_bf16(W), 0.0
    if granularity == "tensor":
        blocks = W.reshape(1, -1)
    elif granularity == "channel":
        blocks = W
    else:
        blocks = W.reshape(-1, group)
    if fmt == "nf4":
        absmax = np.abs(blocks).max(axis=1, keepdims=True) + 1e-12
        normed = blocks / absmax
        idx = np.abs(normed[..., None] - NF4).argmin(-1)
        deq = NF4[idx] * absmax
    else:
        qmax = 127 if fmt == "int8" else 7
        scale = np.abs(blocks).max(axis=1, keepdims=True) / qmax + 1e-12
        deq = np.clip(np.round(blocks / scale), -qmax - 1, qmax) * scale
    overhead = 16 * blocks.shape[0] / W.size  # one fp16 scale per block
    return deq.reshape(W.shape), overhead


def quant_sim(p: dict, ctx: Context) -> Result:
    g = rng(p["seed"])
    d = 256
    W = g.normal(0, 0.02, size=(d, d)).astype(np.float32)
    n_out = int(p["outlier_pct"] / 100 * W.size)
    if n_out:
        idx = g.choice(W.size, n_out, replace=False)
        W.flat[idx] *= p["outlier_scale"]
    X = g.normal(0, 1, size=(64, d)).astype(np.float32)
    Y = X @ W
    arch = MODELS[p["model"]]
    hw = active_hw(ctx)
    me, _ = effs(ctx)
    rows, labels, err_w, sqnr, mem = [], [], [], [], []
    fmts = ["fp32", "fp16", "bf16", "int8", "int4", "nf4"]
    for f in fmts:
        Wq, ov = quantize(W, f, p["granularity"], p["group_size"])
        e = float(np.linalg.norm(W - Wq) / np.linalg.norm(W))
        eo = float(np.linalg.norm(Y - X @ Wq) / np.linalg.norm(Y))
        s = float(10 * np.log10((Y ** 2).sum() / max(((Y - X @ Wq) ** 2).sum(), 1e-30)))
        bits = DTYPE_BYTES[f] * 8 + (ov if f in ("int8", "int4", "nf4") else 0)
        gbm = arch.total_params() * bits / 8 / 1e9
        tpot = gbm * 1e9 / (hw["bw_gbs"] * 1e9 * me) * 1e3
        rows.append([f.upper(), round(bits, 2), round(gbm, 2), f"{e * 100:.3f}%", f"{eo * 100:.3f}%", round(min(s, 200), 1), round(tpot, 2), "yes" if gbm < hw["mem_gb"] * 0.9 else "no"])
        labels.append(f.upper())
        err_w.append(eo * 100)
        sqnr.append(min(s, 200))
        mem.append(gbm)
    sel = fmts.index(p["format"])
    r = Result("computed")
    r.set_fidelity("computed", "Quantization error is computed exactly on a synthetic weight matrix with injected outliers; memory and speed are analytical.")
    r.metric("bits", "Effective bits / weight", rows[sel][1], "bits", "includes per-block scale overhead")
    r.metric("mem", f"{arch.label} weight memory", mem[sel], "GB", status="bad" if mem[sel] > hw["mem_gb"] * 0.9 else "good")
    r.metric("out_err", "Relative output error ‖Y−Ŷ‖/‖Y‖", err_w[sel], "%")
    r.metric("sqnr", "Output SQNR", sqnr[sel], "dB", "higher is better")
    r.metric("speed", "Decode speedup vs FP16 (bandwidth-bound)", mem[1] / mem[sel], "×", "upper bound; dequant kernels cost extra")
    r.table(f"Quantization comparison ({p['granularity']} scaling" + (f", group {p['group_size']}" if p["granularity"] == "group" else "") + ")",
            ["Format", "Bits/weight", "Weights GB", "Weight error", "Output error", "SQNR dB", "Est. TPOT ms", f"Fits {hw['label']}"], rows)
    r.chart("mem", "Weight memory by format", "bar", labels, [{"label": "GB", "data": mem}], "", "GB")
    r.chart("err", "Output error by format (computed)", "bar", labels, [{"label": "relative output error %", "data": err_w}], "", "%", log_y=True)
    hist_w, edges = np.histogram(W, bins=40)
    r.chart("hist", "Weight distribution (outliers stretch the scale)", "bar", [f"{x:.3f}" for x in edges[:-1]], [{"label": "count", "data": hist_w.tolist()}], "value", "count", log_y=True)
    gran_rows = []
    for gr in ("tensor", "channel", "group"):
        Wq, _ = quantize(W, "int4" if p["format"] not in ("int8",) else "int8", gr, p["group_size"])
        gran_rows.append([gr, f"{np.linalg.norm(Y - X @ Wq) / np.linalg.norm(Y) * 100:.3f}%"])
    r.table("Scaling granularity effect (same bit width)", ["Granularity", "Output error"], gran_rows)
    r.compare("FP16", p["format"].upper(), [("Weights", mem[1], mem[sel], "GB", "lower"), ("Output error", err_w[1], err_w[sel], "%", "lower"),
                                           ("Est. TPOT", rows[1][6], rows[sel][6], "ms", "lower")])
    algo = {"rtn": "Round-to-nearest (shown above): fastest, no calibration; quality drops at 4-bit with outliers.",
            "gptq": "GPTQ: layer-wise second-order (Hessian) error compensation using calibration data; good 4-bit quality.",
            "awq": "AWQ: protects salient weight channels found from activation statistics by per-channel scaling before quantization.",
            "bnb": "bitsandbytes: LLM.int8() (outlier decomposition) and 4-bit NF4/FP4 on-the-fly dequantization; used by QLoRA."}
    r.note(algo[p["algorithm"]] + " This demo computes RTN error; calibration-based methods reduce it.")
    r.recommend("INT8 weight-only is usually near-lossless; validate INT4 on task evals, not just perplexity.",
                "Use group-wise scales (g=64–128) for 4-bit; per-tensor scaling is ruined by outliers.",
                "Quantization-aware training (QAT) recovers quality when post-training methods fall short.",
                "Weight-only quantization speeds up memory-bound decode; prefill (compute-bound) needs INT8/FP8 activations too.")
    return r


register(ModuleDef(
    id="quantization", title="Quantization", category="training", fidelity="computed", tags=["essential"],
    summary="Store weights in fewer bits (FP16 → INT8 → INT4/NF4) to cut memory and speed up decode.",
    controls=[
        select("model", "Model (for memory estimate)", "llama-3-8b", MODEL_OPTIONS, "Architecture used for memory/speed estimates."),
        select("format", "Target format", "int4", [{"value": f, "label": f.upper()} for f in ["fp32", "fp16", "bf16", "int8", "int4", "nf4"]], "Numeric format."),
        select("granularity", "Scale granularity", "group", [{"value": "tensor", "label": "Per-tensor"}, {"value": "channel", "label": "Per-channel (row)"},
                                                             {"value": "group", "label": "Group-wise"}], "How many weights share one scale."),
        select("group_size", "Group size", 64, [{"value": v, "label": str(v)} for v in (32, 64, 128, 256)], "Weights per scale (group-wise)."),
        slider("outlier_pct", "Outlier weights", 0.5, 0.0, 5.0, 0.1, "Percent of weights multiplied by the outlier scale.", unit="%"),
        slider("outlier_scale", "Outlier magnitude", 10.0, 1.0, 50.0, 1.0, "Multiplier applied to outliers."),
        select("algorithm", "Method (explanation)", "rtn", [{"value": "rtn", "label": "RTN"}, {"value": "gptq", "label": "GPTQ"},
                                                            {"value": "awq", "label": "AWQ"}, {"value": "bnb", "label": "bitsandbytes"}], "Production quantization method."),
        slider("seed", "Random seed", 0, 0, 9999, 1, "Synthetic weight matrix seed."),
    ],
    simulate=quant_sim,
    docs=docs(
        what="Representing weights (and optionally activations/KV) with lower-precision numbers plus scale factors: q = round(w / s), ŵ = q·s.",
        why="LLM decode speed is bounded by bytes moved per token and deployment is bounded by GPU memory. Halving bits roughly halves both.",
        how="Choose a format (INT8, INT4, NF4, FP8), a granularity (per-tensor, per-channel, group-wise) and a method: round-to-nearest, GPTQ (Hessian-aware error compensation), AWQ (activation-aware scaling), or bitsandbytes dynamic dequantization. QAT simulates quantization during training.",
        when_use="Serving on memory-constrained GPUs, edge/laptop deployment, maximizing batch size, QLoRA fine-tuning.",
        when_avoid="Tasks highly sensitive to small quality drops without evaluation (e.g., precise numerical reasoning) or when compute-bound prefill dominates and kernels lack low-bit support.",
        pros=["2–4× memory reduction", "Faster memory-bound decode", "Larger batches / longer contexts on the same GPU"],
        cons=["Quality degradation, especially at ≤4 bits and with outliers", "Kernel/hardware support varies", "Calibration data and evaluation required"],
        use_case="A healthcare provider runs Llama-3-8B on a single 24 GB L4 by serving AWQ INT4 weights, freeing memory for KV cache and longer clinical notes.",
        code='''
# bitsandbytes 4-bit NF4 (load-time quantization)
from transformers import AutoModelForCausalLM, BitsAndBytesConfig
import torch
bnb = BitsAndBytesConfig(load_in_4bit=True, bnb_4bit_quant_type="nf4",
                         bnb_4bit_use_double_quant=True, bnb_4bit_compute_dtype=torch.bfloat16)
model = AutoModelForCausalLM.from_pretrained("meta-llama/Llama-3.1-8B", quantization_config=bnb)

# Pre-quantized AWQ/GPTQ checkpoints in vLLM:
# LLM(model="TheBloke/Llama-2-7B-AWQ", quantization="awq")
''',
        diagram=[{"label": "FP16 weights"}, {"label": "Calibrate (GPTQ/AWQ)"}, {"label": "Quantize + scales", "cls": "accent"},
                 {"label": "Low-bit kernels"}, {"label": "Dequant on the fly"}],
        performance="Weight memory = params × bits/8 (+ scale overhead). Decode speedup ≤ bytes ratio; dequantization overhead and kernel quality determine real gains.",
        checklist=["Evaluate on task-specific benchmarks", "Use group-wise scaling for 4-bit", "Check kernel support on target GPU", "Keep embeddings/LM head higher precision if needed", "Re-validate after every model update"],
        applications=[{"domain": "Healthcare information systems", "text": "On-prem deployment on modest GPUs for privacy."},
                      {"domain": "AI-assisted software engineering", "text": "Local coding models on developer laptops (GGUF/INT4)."},
                      {"domain": "Financial services", "text": "Higher batch capacity per GPU for document processing."}],
        limitations="Error is measured on a synthetic matrix with RTN; real model quality must be measured on evals after GPTQ/AWQ.",
        references=["Frantar et al., 'GPTQ' (2023)", "Lin et al., 'AWQ' (2024)", "Dettmers et al., 'LLM.int8()' (2022)"],
    ),
))
