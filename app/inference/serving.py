"""Serving modules: PagedAttention, Continuous Batching, Model Serving Optimization, Attention variants."""
from __future__ import annotations

import math

import numpy as np

from app.engine.common import active_arch, active_hw, effs, hw_note, rng
from app.engine.hardware import DTYPE_BYTES, MODEL_OPTIONS, MODELS, estimate, kv_cache_bytes
from app.engine.registry import Context, ModuleDef, Result, docs, register, select, slider, toggle

WORKLOADS = {"chat": (300, 200, 250, 180), "mixed": (900, 900, 350, 300), "longdoc": (6000, 3000, 500, 300)}  # prompt mean/std, output mean/std


def sample_lengths(g, workload: str, n: int, max_len: int):
    pm, ps, om, os_ = WORKLOADS[workload]
    prompt = np.clip(g.normal(pm, ps, n), 16, max_len // 2).astype(int)
    out = np.clip(g.lognormal(math.log(om), 0.6, n), 8, max_len - prompt).astype(int)
    return prompt, out


# ============================================================================ PagedAttention
def paged_sim(p: dict, ctx: Context) -> Result:
    arch, hw = active_arch(ctx), active_hw(ctx)
    g = rng(p["seed"])
    kv_tok = kv_cache_bytes(arch, 1, 1, "bf16")
    weights = arch.total_params() * 2
    budget = hw["mem_gb"] * 1e9 * p["gpu_memory_utilization"] - weights
    r = Result("simulated")
    r.set_fidelity("simulated", "Exact allocator accounting over a deterministic synthetic request mix; memory sizes use the active model's real KV dimensions.")
    if budget <= 0:
        r.note(f"{arch.label} weights ({weights / 1e9:.1f} GB) do not fit in {hw['label']} at this utilization. Pick a smaller model or bigger GPU in Settings.", "warn")
        return r
    prompt, out = sample_lengths(g, p["workload"], 2000, p["max_model_len"])
    used = prompt + out
    n_samples = p["parallel_samples"]
    B = p["block_size"]
    # contiguous: reserve max_model_len per sequence (each parallel sample gets its own full copy)
    contig_per_req = p["max_model_len"] * n_samples
    # paged: blocks for shared prompt once + per-sample output blocks; partial last block wasted
    paged_per_req = np.ceil(prompt / B) * B + n_samples * np.ceil(out / B) * B
    actual_per_req = prompt + n_samples * out
    budget_tok = budget / kv_tok
    conc_contig = int(budget_tok // contig_per_req)
    conc_paged = int(budget_tok // paged_per_req.mean())
    waste_contig = 1 - actual_per_req.mean() / contig_per_req
    waste_paged = 1 - actual_per_req.mean() / paged_per_req.mean()
    r.metric("kv_budget", "KV cache budget", budget / 1e9, "GB", f"{p['gpu_memory_utilization']:.0%} of {hw['mem_gb']} GB − weights")
    r.metric("conc_contig", "Concurrent requests — contiguous", conc_contig, "", "reserves max_model_len per sequence")
    r.metric("conc_paged", "Concurrent requests — paged", conc_paged, "", f"{B}-token blocks on demand", status="good")
    r.metric("gain", "Batch capacity gain", conc_paged / max(1, conc_contig), "×")
    r.metric("waste_contig", "Memory wasted — contiguous", waste_contig * 100, "%", "internal fragmentation (reserved but unused)", status="bad")
    r.metric("waste_paged", "Memory wasted — paged", waste_paged * 100, "%", "only the partially filled last block")
    # timeline: 40 requests decode in lock-step, memory reserved vs used
    k = 24
    pr, ou = prompt[:k], out[:k]
    T = int(ou.max())
    steps = np.linspace(0, T, 60).astype(int)
    used_t = [float(sum(min(o, s) * n_samples + pp for pp, o in zip(pr, ou) if s <= o) * kv_tok / 1e9) for s in steps]
    paged_t = [float(sum((math.ceil(pp / B) + n_samples * math.ceil(max(1, min(o, s)) / B)) * B for pp, o in zip(pr, ou) if s <= o) * kv_tok / 1e9) for s in steps]
    contig_t = [float(sum(contig_per_req for pp, o in zip(pr, ou) if s <= o) * kv_tok / 1e9) for s in steps]
    r.chart("timeline", f"KV memory over time for {k} requests", "line", steps.tolist(), [
        {"label": "contiguous reservation", "data": contig_t}, {"label": "paged allocation", "data": paged_t}, {"label": "actually used", "data": used_t, "borderDash": [4, 4]}],
        "decode step", "GB")
    bs = [8, 16, 32, 64, 128]
    r.chart("block", "Waste vs. block size", "bar", bs, [{"label": "paged waste %", "data": [float((1 - actual_per_req.mean() / (np.ceil(prompt / b) * b + n_samples * np.ceil(out / b) * b).mean()) * 100) for b in bs]}], "block size", "%")
    blocks = []
    for i in range(6):
        nb = int(math.ceil((pr[i] + ou[i]) / B))
        blocks.append({"seq": f"req {i + 1}", "logical": min(nb, 12), "physical": [int(x) for x in g.choice(512, size=min(nb, 12), replace=False)]})
    r.panel("block_table", "Block table: logical blocks → scattered physical blocks", sequences=blocks, block_size=B)
    r.compare("Contiguous KV allocation", "PagedAttention", [("Max concurrent requests", conc_contig, conc_paged, "", "higher"), ("Wasted KV memory", waste_contig * 100, waste_paged * 100, "%", "lower")])
    if n_samples > 1:
        r.note(f"With {n_samples} parallel samples, paged KV shares the prompt blocks copy-on-write instead of duplicating them.")
    r.note(hw_note(ctx))
    r.recommend("Use a paged-KV engine (vLLM, TGI, TensorRT-LLM, SGLang) for multi-user serving.",
                "Set max_model_len to the real need — it no longer reserves memory, but caps admission and preemption behaviour.",
                "Tune gpu_memory_utilization (0.85–0.95) leaving headroom for activations and CUDA graphs.")
    return r


register(ModuleDef(
    id="paged-attention", title="PagedAttention", category="serving", fidelity="simulated", tags=["essential"],
    summary="Manage KV cache in fixed-size non-contiguous blocks (like OS virtual memory) to eliminate fragmentation.",
    controls=[
        select("workload", "Request mix", "chat", [{"value": "chat", "label": "Chat (short prompts/outputs)"}, {"value": "mixed", "label": "Mixed"}, {"value": "longdoc", "label": "Long documents"}], "Synthetic length distribution."),
        slider("max_model_len", "Max model length", 8192, 1024, 131072, 1024, "Contiguous allocators reserve this per sequence.", unit="tok"),
        select("block_size", "Block size", 16, [{"value": v, "label": f"{v} tokens"} for v in (8, 16, 32, 64, 128)], "Tokens per KV block."),
        slider("gpu_memory_utilization", "GPU memory utilization", 0.9, 0.5, 0.98, 0.01, "Fraction of GPU memory the engine may use."),
        slider("parallel_samples", "Parallel samples per request (n)", 1, 1, 8, 1, "Beam/parallel sampling sharing a prompt."),
        slider("seed", "Random seed", 21, 0, 9999, 1, "Request mix seed."),
    ],
    simulate=paged_sim,
    docs=docs(
        what="PagedAttention (vLLM) stores each sequence's KV cache in fixed-size blocks that can live anywhere in GPU memory, mapped by a per-sequence block table — like virtual memory pages.",
        why="Contiguous KV allocation must reserve the maximum possible length per request. Since actual lengths are unknown and vary widely, 60–80% of KV memory was wasted in earlier systems, limiting batch size and throughput.",
        how="Blocks are allocated on demand as tokens are generated. The attention kernel gathers keys/values through the block table. Blocks can be shared (prefix caching, parallel sampling) with copy-on-write, and swapped/preempted under pressure.",
        when_use="Any multi-request LLM serving on GPUs — it's the default in vLLM and similar engines.",
        when_avoid="Single-request local inference has little to gain; very small block sizes add kernel overhead.",
        pros=["Near-zero fragmentation (<4% typical waste)", "2–4× more concurrent requests", "Enables prefix sharing and copy-on-write"],
        cons=["Custom attention kernels needed", "Block-table indirection overhead", "Requires engine-level scheduler support"],
        use_case="An enterprise chat platform on A100s raises concurrent sessions per GPU ~3× by moving from a naive HF server to vLLM with paged KV.",
        code='''
from vllm import LLM, SamplingParams
llm = LLM(model="meta-llama/Llama-3.1-8B-Instruct",
          gpu_memory_utilization=0.90,   # weights + paged KV pool
          max_model_len=8192,            # admission limit, not a reservation
          block_size=16,
          enable_prefix_caching=True)
outs = llm.generate(prompts, SamplingParams(max_tokens=256, n=4))   # n samples share prompt blocks
''',
        diagram=[{"label": "Sequence tokens"}, {"label": "Logical blocks"}, {"label": "Block table", "cls": "accent"}, {"label": "Physical KV blocks (scattered)", "cls": "accent"}, {"label": "Paged attention kernel"}],
        performance="Batch capacity ≈ KV budget / (actual tokens rounded up to blocks) instead of KV budget / max_len.",
        checklist=["Use vLLM/TGI/TRT-LLM", "Tune gpu_memory_utilization and block size", "Monitor KV usage + preemptions", "Enable prefix caching", "Load-test with real length distributions"],
        applications=[{"domain": "Enterprise chatbots", "text": "Maximizing concurrent sessions per GPU."},
                      {"domain": "Multi-agent enterprise platforms", "text": "Many concurrent agent threads with variable lengths."}],
        limitations="Request lengths are synthetic; waste figures are exact for that mix.",
        references=["Kwon et al., 'Efficient Memory Management for LLM Serving with PagedAttention' (SOSP 2023)"],
    ),
))


# ============================================================================ Continuous batching
def batching_run(policy: str, arrivals, outs, prompts, max_batch, step_fn, prefill_fn, timeout):
    n = len(arrivals)
    t, i, active, done = 0.0, 0, [], {}
    queue = []
    useful = slots = 0
    first_tok = {}
    while len(done) < n:
        while i < n and arrivals[i] <= t:
            queue.append(i)
            i += 1
        if policy == "continuous":
            join = []
            while queue and len(active) < max_batch:
                join.append(queue.pop(0))
            if join:
                t += prefill_fn(sum(prompts[j] for j in join))
                for j in join:
                    active.append([j, outs[j]])
                    first_tok[j] = t
        else:
            if not active:
                if len(queue) >= max_batch or (queue and t - arrivals[queue[0]] >= timeout) or (queue and i >= n):
                    batch = queue[:max_batch]
                    queue = queue[max_batch:]
                    t += prefill_fn(sum(prompts[j] for j in batch))
                    active = [[j, outs[j]] for j in batch]
                    for j in batch:
                        first_tok[j] = t
        if not active:
            t = arrivals[i] if i < n and (policy == "continuous" or not queue) else t + 0.005
            continue
        dt = step_fn(len(active) if policy == "continuous" else len(active))
        t += dt
        slots += max_batch
        for a in active:
            if a[1] > 0:
                a[1] -= 1
                useful += 1
        if policy == "continuous":
            for a in [a for a in active if a[1] <= 0]:
                done[a[0]] = t
            active = [a for a in active if a[1] > 0]
        else:
            if all(a[1] <= 0 for a in active):  # static: batch finishes together (padding)
                for a in active:
                    done[a[0]] = t
                active = []
            else:
                for a in active:
                    if a[1] <= 0 and a[0] not in done:
                        pass
    lat = np.array([done[j] - arrivals[j] for j in range(n)])
    ttft = np.array([first_tok[j] - arrivals[j] for j in range(n)])
    span = max(done.values()) - arrivals[0]
    return {"throughput": float(sum(outs) / span), "lat_mean": float(lat.mean()), "lat_p95": float(np.percentile(lat, 95)),
            "ttft_mean": float(ttft.mean()), "util": useful / max(1, slots), "lat": lat}


def cb_sim(p: dict, ctx: Context) -> Result:
    arch, hw = active_arch(ctx), active_hw(ctx)
    me, ce = effs(ctx)
    bw, F = hw["bw_gbs"] * 1e9 * me, hw["tflops"] * 1e12 * ce
    wbytes = arch.total_params() * 2
    kv_tok = kv_cache_bytes(arch, 1, 1, "bf16")
    avg_ctx = 600
    def step_fn(b):
        return max((wbytes + b * avg_ctx * kv_tok) / bw, 2 * arch.active_params() * b / F)
    def prefill_fn(tokens):
        return max(2 * arch.active_params() * tokens / F, wbytes / bw)
    g = rng(p["seed"])
    n = p["requests"]
    arrivals = np.cumsum(g.exponential(1 / p["arrival_rate"], n))
    prompts = np.clip(g.normal(400, 200, n), 32, 4000).astype(int)
    outs = np.clip(g.lognormal(math.log(p["mean_output"]), p["output_spread"], n), 4, 4096).astype(int)
    st = batching_run("static", arrivals, outs, prompts, p["max_batch"], step_fn, prefill_fn, p["batch_timeout_ms"] / 1000)
    cb = batching_run("continuous", arrivals, outs, prompts, p["max_batch"], step_fn, prefill_fn, 0)
    sel = cb if p["policy"] == "continuous" else st
    r = Result("simulated")
    r.set_fidelity("simulated", "Discrete-event scheduling simulation over a seeded Poisson arrival stream; per-step decode/prefill times come from the roofline model.")
    r.metric("throughput", "Throughput", sel["throughput"], "tok/s")
    r.metric("lat_mean", "Mean request latency", sel["lat_mean"], "s")
    r.metric("lat_p95", "p95 latency", sel["lat_p95"], "s")
    r.metric("ttft", "Mean time to first token", sel["ttft_mean"] * 1000, "ms")
    r.metric("util", "Batch slot utilization", sel["util"] * 100, "%", "useful token slots ÷ total slots")
    r.metric("step", "Decode step @ max batch", step_fn(p["max_batch"]) * 1000, "ms")
    r.chart("cmp", "Static vs. continuous batching", "bar", ["Throughput (tok/s)", "Mean latency (s)", "p95 latency (s)", "Utilization (%)"], [
        {"label": "Static batching", "data": [st["throughput"], st["lat_mean"], st["lat_p95"], st["util"] * 100]},
        {"label": "Continuous batching", "data": [cb["throughput"], cb["lat_mean"], cb["lat_p95"], cb["util"] * 100]}], "", "", log_y=True)
    hist_s, edges = np.histogram(st["lat"], bins=20, range=(0, max(st["lat"].max(), cb["lat"].max())))
    hist_c, _ = np.histogram(cb["lat"], bins=edges)
    r.chart("lat", "Request latency distribution", "bar", [f"{e:.1f}" for e in edges[:-1]], [{"label": "static", "data": hist_s.tolist()}, {"label": "continuous", "data": hist_c.tolist()}], "latency (s)", "requests")
    rates = [p["arrival_rate"] * f for f in (0.25, 0.5, 1, 2)]
    m = min(n, 80)
    tp_s, tp_c = [], []
    for rate in rates:
        a2 = np.cumsum(rng(p["seed"]).exponential(1 / rate, m))
        tp_s.append(batching_run("static", a2, outs[:m], prompts[:m], p["max_batch"], step_fn, prefill_fn, p["batch_timeout_ms"] / 1000)["throughput"])
        tp_c.append(batching_run("continuous", a2, outs[:m], prompts[:m], p["max_batch"], step_fn, prefill_fn, 0)["throughput"])
    r.chart("load", "Throughput vs. offered load", "line", [round(x, 2) for x in rates], [{"label": "static", "data": tp_s}, {"label": "continuous", "data": tp_c}], "arrival rate (req/s)", "tok/s")
    r.panel("schedule", "Iteration-level scheduling (illustration)", rows=[
        ["static", "[A A A A][A A A ·][A A · ·][A · · ·]  ← waits for longest sequence, slots idle"],
        ["continuous", "[A B C D][A B E D][F B E D][F G E H]  ← finished sequences replaced every iteration"]])
    r.compare("Static batching", "Continuous batching", [("Throughput", st["throughput"], cb["throughput"], "tok/s", "higher"), ("p95 latency", st["lat_p95"], cb["lat_p95"], "s", "lower"),
                                                         ("Slot utilization", st["util"] * 100, cb["util"] * 100, "%", "higher")])
    r.note(hw_note(ctx))
    r.recommend("Use iteration-level (continuous) batching — default in vLLM, TGI, TensorRT-LLM, SGLang.",
                "Enable chunked prefill so long prompts don't stall decoding of running sequences.",
                "Cap max_num_seqs by KV memory; set per-tenant fairness/priorities in the scheduler.")
    return r


register(ModuleDef(
    id="continuous-batching", title="Continuous Batching", category="serving", fidelity="simulated", tags=["essential"],
    summary="Add and remove requests from the running batch every decode iteration to keep the GPU busy.",
    controls=[
        select("policy", "Policy shown in metrics", "continuous", [{"value": "continuous", "label": "Continuous (iteration-level)"}, {"value": "static", "label": "Static (request-level)"}], "Batching strategy."),
        slider("arrival_rate", "Arrival rate", 1.0, 0.2, 40.0, 0.1, "Poisson request rate.", unit="req/s"),
        slider("requests", "Requests simulated", 150, 20, 600, 10, "Stream length."),
        slider("max_batch", "Max batch size", 32, 1, 128, 1, "Concurrent sequences."),
        slider("mean_output", "Mean output tokens", 200, 16, 1024, 8, "Log-normal output lengths."),
        slider("output_spread", "Output length spread (σ)", 0.8, 0.1, 1.5, 0.05, "Higher spread → more padding waste in static batching."),
        slider("batch_timeout_ms", "Static batch wait timeout", 50, 0, 1000, 10, "Dynamic batching: max wait to fill a batch.", unit="ms"),
        slider("seed", "Random seed", 13, 0, 9999, 1, "Arrival stream seed."),
    ],
    simulate=cb_sim,
    docs=docs(
        what="Continuous (iteration-level) batching schedules at the granularity of a single decode step: finished sequences leave the batch immediately and queued requests join, instead of waiting for the whole batch to finish.",
        why="Output lengths vary by 10–100×. With static batching, short requests wait for the longest one and GPU slots sit idle (padding), wasting throughput and inflating latency.",
        how="Each iteration the scheduler: removes completed sequences, admits new ones (prefilling them, possibly chunked), and runs one decode step for all active sequences. KV memory (paged) limits how many can be active.",
        when_use="Every online LLM serving deployment with concurrent users.",
        when_avoid="Offline batch jobs with uniform lengths gain less (though still benefit); single-user local inference.",
        pros=["Large throughput gains (often several ×)", "Lower queueing latency", "Higher GPU utilization"],
        cons=["Complex scheduler", "Prefill/decode interference (mitigated by chunked prefill)", "Requires paged/dynamic KV memory"],
        use_case="A support-automation API serving 500 concurrent agents moves from a dynamic-batching server to vLLM continuous batching and cuts p95 latency while doubling tokens/s.",
        code='''
# vLLM's OpenAI-compatible server does continuous batching automatically:
#   vllm serve meta-llama/Llama-3.1-8B-Instruct --max-num-seqs 128 --enable-chunked-prefill
# Conceptual scheduler loop:
while True:
    running = [s for s in running if not s.finished]           # evict finished
    while waiting and can_allocate_kv(waiting[0]) and len(running) < MAX_SEQS:
        running.append(prefill(waiting.pop(0)))               # admit new requests
    decode_one_step(running)                                   # one token for every running sequence
''',
        diagram=[{"label": "Request queue"}, {"label": "Scheduler (per iteration)", "cls": "accent"}, {"label": "Admit / evict"}, {"label": "Batched decode step", "cls": "accent"}, {"label": "Stream tokens"}],
        performance="Throughput rises with average active batch size until compute-bound; latency stays near single-request decode time at moderate load.",
        checklist=["Use an engine with continuous batching", "Set max_num_seqs by memory", "Enable chunked prefill", "Load-test with realistic arrival and length distributions", "Watch queue time and preemptions"],
        applications=[{"domain": "Customer support automation", "text": "High-concurrency classification and drafting APIs."},
                      {"domain": "Enterprise chatbots", "text": "Many simultaneous chat sessions on shared GPUs."}],
        limitations="Simplified scheduler (no preemption, chunked prefill or priorities); step times are analytical.",
        references=["Yu et al., 'Orca: A Distributed Serving System for Transformer-Based Generative Models' (OSDI 2022)"],
    ),
))


# ============================================================================ Serving optimization
def serving_sim(p: dict, ctx: Context) -> Result:
    arch = MODELS[p["model"]]
    hw = active_hw(ctx)
    me, ce = effs(ctx)
    batches = [1, 2, 4, 8, 16, 32, 64, 128, 256]
    pts = []
    for b in batches:
        e = estimate(arch, hw, dtype=p["dtype"], prompt=p["prompt_tokens"], output=p["output_tokens"], batch=b, mem_eff=me, comp_eff=ce)
        pts.append((b, e))
    sel = estimate(arch, hw, dtype=p["dtype"], prompt=p["prompt_tokens"], output=p["output_tokens"], batch=p["batch"], mem_eff=me, comp_eff=ce)
    fits = [b for b, e in pts if e.total_gb <= hw["mem_gb"] * p["gpu_memory_utilization"]]
    max_b = max(fits) if fits else 0
    slo_ok = [b for b, e in pts if e.total_ms / 1000 <= p["slo_seconds"] and b in fits]
    best_b = max(slo_ok) if slo_ok else None
    r = Result("analytical")
    r.metric("tps", "Throughput @ batch", sel.tokens_per_s, "tok/s", f"batch {p['batch']}")
    r.metric("ttft", "TTFT", sel.ttft_ms, "ms", "streaming shows the first token after this")
    r.metric("tpot", "TPOT", sel.tpot_ms, "ms", sel.decode_bound + "-bound")
    r.metric("e2e", "End-to-end latency", sel.total_ms / 1000, "s", status="bad" if sel.total_ms / 1000 > p["slo_seconds"] else "good")
    r.metric("mem", "GPU memory (weights + KV)", sel.total_gb, "GB", status="bad" if not sel.fits else None)
    r.metric("maxb", "Max batch that fits", max_b)
    if best_b:
        e_best = dict(pts)[best_b]
        req_per_s = best_b / (e_best.total_ms / 1000)
        replicas = math.ceil(p["traffic_rps"] / req_per_s)
        r.metric("replicas", "Replicas needed for traffic", replicas, "", f"{req_per_s:.2f} req/s per replica at batch {best_b} within SLO")
        r.metric("cost_hr", "Fleet cost", replicas * hw["usd_hr"], "$/h", "uses the $/h assumption from Settings")
    else:
        r.metric("replicas", "Replicas needed", "SLO unreachable", "", "even batch 1 exceeds the latency SLO", status="bad")
    r.chart("tradeoff", "Throughput vs. latency as batch grows", "scatter", None, [
        {"label": "batch sizes", "data": [{"x": e.total_ms / 1000, "y": e.tokens_per_s, "t": f"batch {b}"} for b, e in pts if b in fits], "showLine": True}],
        "end-to-end latency (s)", "throughput (tok/s)")
    r.chart("perbatch", "Per-token latency and throughput by batch", "line", batches, [
        {"label": "TPOT (ms)", "data": [e.tpot_ms for _, e in pts]}, {"label": "throughput (tok/s)", "data": [e.tokens_per_s for _, e in pts], "yAxisID": "y1"}],
        "batch size", "ms", dual_axis=True, log_x=True)
    r.chart("mem", "Memory by batch", "bar", batches, [{"label": "weights", "data": [e.weights_gb for _, e in pts]}, {"label": "KV cache", "data": [e.kv_gb for _, e in pts]}],
            "batch size", "GB", stacked=True, threshold=hw["mem_gb"] * p["gpu_memory_utilization"])
    perceived = sel.ttft_ms / 1000 if p["streaming"] else sel.total_ms / 1000
    r.compare("Batch 1, no streaming", f"Batch {p['batch']}{', streaming' if p['streaming'] else ''}", [
        ("Throughput", pts[0][1].tokens_per_s, sel.tokens_per_s, "tok/s", "higher"), ("Perceived wait", pts[0][1].total_ms / 1000, perceived, "s", "lower"),
        ("Cost per 1K tokens", pts[0][1].cost_per_1k_tokens, sel.cost_per_1k_tokens, "$", "lower")])
    r.flow("Serving stack", [{"label": "Load balancer"}, {"label": "Autoscaler", "sub": "replicas from req/s & SLO"}, {"label": "Scheduler", "sub": "continuous batching", "cls": "accent"},
                             {"label": "Paged KV", "cls": "accent"}, {"label": "GPU kernels"}, {"label": "Token stream (SSE)"}])
    r.note(hw_note(ctx))
    r.recommend("Batch until you hit the latency SLO or memory limit — decode is memory-bound, so batching is nearly free throughput.",
                "Stream tokens (SSE/WebSocket) so perceived latency ≈ TTFT.",
                "Autoscale on queue depth / KV utilization, not CPU; keep warm replicas to avoid cold-start model loads.",
                "Quantize weights (INT8/INT4) or KV (FP8) to raise the batch ceiling.")
    return r


register(ModuleDef(
    id="model-serving", title="Model Serving Optimization", category="serving", fidelity="analytical", tags=["essential"],
    summary="Dynamic/continuous batching, memory management, streaming and autoscaling: the throughput–latency trade-off.",
    controls=[
        select("model", "Model", "llama-3-8b", MODEL_OPTIONS, "Served model."),
        select("dtype", "Weight dtype", "bf16", [{"value": d, "label": d.upper()} for d in ("fp16", "bf16", "int8", "int4")], "Weight precision."),
        slider("batch", "Batch size", 16, 1, 256, 1, "Concurrent sequences."),
        slider("prompt_tokens", "Prompt tokens", 1024, 16, 32768, 16, "Average prompt length."),
        slider("output_tokens", "Output tokens", 256, 8, 4096, 8, "Average output length."),
        slider("slo_seconds", "Latency SLO (end-to-end)", 10.0, 0.5, 60.0, 0.5, "Target max latency.", unit="s"),
        slider("traffic_rps", "Traffic", 5.0, 0.1, 500.0, 0.1, "Requests per second to serve.", log=True, unit="req/s"),
        slider("gpu_memory_utilization", "GPU memory utilization", 0.9, 0.5, 0.98, 0.01, "Fraction of memory usable."),
        toggle("streaming", "Stream responses", True, "Perceived latency becomes TTFT."),
    ],
    simulate=serving_sim,
    docs=docs(
        what="The set of system techniques that make LLM inference fast and economical: batching (dynamic/continuous), KV memory management (paged attention), quantization, streaming, autoscaling and capacity planning.",
        why="Decode is memory-bandwidth-bound: a GPU streams all weights to produce one token per sequence. Serving many sequences per weight read (batching) multiplies throughput, constrained by KV memory and latency SLOs.",
        how="Roofline: TPOT ≈ max((weights + KV)/bandwidth, 2·N·batch/FLOPS). Increase batch until compute-bound, memory-bound, or the SLO is hit; stream tokens; scale replicas by req/s capacity at the SLO.",
        when_use="Planning GPU capacity, choosing hardware/precision, setting batch limits and autoscaling policies.",
        when_avoid="Don't rely on analytical estimates for final sizing — load-test with real traffic and engines.",
        pros=["Order-of-magnitude cost reduction", "Predictable capacity planning", "Better UX with streaming"],
        cons=["Latency rises with batch", "Memory caps concurrency", "Many interacting knobs"],
        use_case="A platform team sizes a Llama-3-8B fleet: at batch 32 on A10Gs each replica serves ~2 req/s within a 10 s SLO, so 50 req/s needs ~25 replicas — then quantization halves it.",
        code='''
# Launch an OpenAI-compatible server with production knobs
# vllm serve meta-llama/Llama-3.1-8B-Instruct --max-num-seqs 64 --gpu-memory-utilization 0.9 \\
#      --enable-chunked-prefill --enable-prefix-caching --quantization fp8
import httpx, json
with httpx.stream("POST", "http://localhost:8000/v1/chat/completions",
                  json={"model": "meta-llama/Llama-3.1-8B-Instruct", "stream": True,
                        "messages": [{"role": "user", "content": "Hello"}]}) as r:
    for line in r.iter_lines():                 # Server-Sent Events: perceived latency = TTFT
        if line.startswith("data: ") and line != "data: [DONE]":
            print(json.loads(line[6:])["choices"][0]["delta"].get("content", ""), end="")
''',
        diagram=[{"label": "Traffic"}, {"label": "Autoscaler"}, {"label": "Continuous batching", "cls": "accent"}, {"label": "Paged KV + quantization", "cls": "accent"}, {"label": "Streaming"}],
        performance="Throughput ≈ batch / TPOT; TTFT ≈ prefill FLOPs / effective FLOPS; memory ≈ weights + batch × context × KV/token.",
        checklist=["Define SLOs (TTFT, TPOT, p95)", "Load test with real distributions", "Choose precision per quality evals", "Stream responses", "Autoscale on queue/KV metrics", "Track cost per 1K tokens"],
        applications=[{"domain": "Enterprise chatbots", "text": "Fleet sizing for thousands of daily users."},
                      {"domain": "AI-assisted software engineering", "text": "Low-TTFT code completion with streaming."},
                      {"domain": "Multi-agent enterprise platforms", "text": "Shared inference pools with per-tenant quotas."}],
        limitations="Roofline estimates with configurable efficiency factors; real engines add scheduling, kernel and network overheads.",
        references=["Pope et al., 'Efficiently Scaling Transformer Inference' (2022)", "vLLM documentation"],
    ),
))


# ============================================================================ Attention variants
def attn_variants_sim(p: dict, ctx: Context) -> Result:
    arch = MODELS[p["model"]]
    S, B = p["seq_len"], p["batch"]
    hd, H, L = arch.head_dim, arch.heads, arch.layers
    variants = {
        "MHA": {"kv": H, "win": S}, "GQA": {"kv": p["gqa_groups"], "win": S}, "MQA": {"kv": 1, "win": S},
        f"Sliding window ({p['window']})": {"kv": arch.kv_heads, "win": min(S, p["window"])},
    }
    rows, labels, kvs, scores = [], [], [], []
    for name, v in variants.items():
        kv = 2 * L * v["kv"] * hd * min(S, v["win"]) * B * 2
        flops = 4 * L * H * hd * S * min(S, v["win"]) * B
        score_naive = L * H * S * min(S, v["win"]) * B * 2  # materialized attention matrix (fp16), all layers sequential → per layer peak
        rows.append([name, v["kv"], f"{kv / 1e9:.2f}", f"{flops / 1e12:.2f}", f"{score_naive / L / 1e9:.2f}"])
        labels.append(name)
        kvs.append(kv / 1e9)
        scores.append(score_naive / L / 1e9)
    r = Result("analytical")
    r.metric("flash", "Attention-matrix memory saved by FlashAttention", scores[0], "GB", "naive per-layer S×S scores vs O(S) tiles")
    r.metric("gqa_saving", "GQA KV saving vs MHA", (1 - kvs[1] / kvs[0]) * 100, "%")
    r.metric("mqa_saving", "MQA KV saving vs MHA", (1 - kvs[2] / kvs[0]) * 100, "%")
    r.table(f"Attention variants for {arch.label} at {S:,} tokens × batch {B}", ["Variant", "KV heads", "KV cache GB", "Attention TFLOPs (prefill)", "Naive score matrix GB / layer"], rows)
    r.chart("kv", "KV cache by variant", "bar", labels, [{"label": "GB", "data": kvs}], "", "GB")
    lens = [1024, 4096, 16384, 65536, 131072]
    r.chart("scale", "Naive attention-score memory vs FlashAttention (per layer)", "line", lens, [
        {"label": "naive (materialize S×S)", "data": [H * s * s * B * 2 / 1e9 for s in lens]},
        {"label": "FlashAttention (tiles, O(S))", "data": [H * s * hd * B * 2 * 2 / 1e9 for s in lens]}], "sequence length", "GB", log_y=True, log_x=True)
    g = rng(1)
    n = 24
    mats = {}
    for name, (win, causal) in {"causal": (n, True), "sliding": (6, True)}.items():
        M = np.zeros((n, n))
        for i in range(n):
            for j in range(n):
                if (not causal or j <= i) and i - j < win:
                    M[i, j] = 1
        mats[name] = M
    r.heatmap("Mask: full causal attention", mats["causal"].tolist(), x_labels="key", y_labels="query", categorical=True)
    r.heatmap("Mask: sliding-window attention", mats["sliding"].tolist(), x_labels="key", y_labels="query", categorical=True)
    r.compare("MHA, naive kernel", "GQA + FlashAttention", [("KV cache", kvs[0], kvs[1], "GB", "lower"), ("Score-matrix memory / layer", scores[0], H * S * hd * B * 4 / 1e9, "GB", "lower")])
    r.recommend("GQA is the modern default: near-MHA quality with a fraction of the KV memory.",
                "FlashAttention changes memory, not math — always use it (SDPA in PyTorch, built into vLLM).",
                "Sliding-window/local attention bounds cost for very long inputs but limits direct long-range interactions.")
    return r


register(ModuleDef(
    id="attention-mechanisms", title="Attention Mechanisms (MHA / GQA / MQA / Sliding / Flash)", category="architecture", fidelity="analytical", tags=["essential"],
    summary="Compare multi-head, grouped-query, multi-query and sliding-window attention, and what FlashAttention changes.",
    controls=[
        select("model", "Model", "llama-3-8b", MODEL_OPTIONS, "Architecture dimensions."),
        slider("seq_len", "Sequence length", 8192, 512, 131072, 512, "Tokens.", unit="tok"),
        slider("batch", "Batch size", 4, 1, 64, 1, "Sequences."),
        slider("gqa_groups", "GQA KV heads", 8, 1, 64, 1, "KV heads for the GQA variant."),
        slider("window", "Sliding window", 4096, 128, 32768, 128, "Window for local attention.", unit="tok"),
    ],
    simulate=attn_variants_sim,
    docs=docs(
        what="Variants of scaled dot-product attention that trade expressiveness for memory and compute: MHA (one KV per head), GQA (heads share KV groups), MQA (one shared KV), sliding-window/local attention, and FlashAttention (an exact IO-aware kernel).",
        why="KV cache and attention-score memory dominate long-context and high-concurrency inference. Sharing KV heads and tiling the computation make long contexts practical.",
        how="softmax(QKᵀ/√d)V. GQA/MQA reduce the number of K/V projections; sliding windows restrict which keys each query sees; FlashAttention computes the same result in SRAM-sized tiles with online softmax, never materializing the S×S matrix.",
        when_use="Model selection (prefer GQA models for serving), long-context deployments, custom architectures.",
        when_avoid="Sliding-window alone where long-range dependencies matter; MQA where quality loss is unacceptable.",
        pros=["GQA/MQA: 4–32× smaller KV", "FlashAttention: exact, faster, O(S) memory", "Sliding window: linear cost"],
        cons=["MQA can reduce quality", "Sliding window loses direct long-range attention", "Kernel/hardware support varies"],
        use_case="A team chooses Mistral-7B (GQA + sliding window) over an MHA model of similar quality to serve 4× more concurrent long sessions.",
        code='''
import torch, torch.nn.functional as F
q = torch.randn(1, 32, 4096, 128, device="cuda", dtype=torch.bfloat16)   # 32 query heads
k = torch.randn(1, 8, 4096, 128, device="cuda", dtype=torch.bfloat16)    # 8 KV heads (GQA)
v = torch.randn_like(k)
k, v = k.repeat_interleave(4, dim=1), v.repeat_interleave(4, dim=1)      # expand groups
out = F.scaled_dot_product_attention(q, k, v, is_causal=True)            # dispatches to FlashAttention
''',
        diagram=[{"label": "Q (H heads)"}, {"label": "K,V (H_kv heads)", "cls": "accent"}, {"label": "Mask (causal / window)"}, {"label": "Tiled softmax (Flash)", "cls": "accent"}, {"label": "Output"}],
        performance="KV ∝ H_kv; attention FLOPs ∝ S × window; naive score memory ∝ H·S² vs FlashAttention ∝ S.",
        checklist=["Prefer GQA models for serving", "Ensure FlashAttention/SDPA is active", "Match window size to task", "Benchmark long-context quality"],
        applications=[{"domain": "Enterprise chatbots", "text": "GQA models maximize concurrent sessions."},
                      {"domain": "Document intelligence", "text": "Long-document processing with Flash/sliding attention."}],
        limitations="Analytical counts; ignores kernel-specific overheads.",
        references=["Ainslie et al., 'GQA' (2023)", "Shazeer, 'Fast Transformer Decoding: One Write-Head is All You Need' (2019)", "Dao et al., 'FlashAttention' (2022)"],
    ),
))
