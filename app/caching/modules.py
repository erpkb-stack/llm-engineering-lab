"""Module 1 (KV Cache) and Module 5 (Prefix Caching)."""
from __future__ import annotations

from collections import OrderedDict

import numpy as np

from app.engine.common import active_arch, active_hw, custom_arch, downsample, effs, hw_note, rng
from app.engine.hardware import DTYPE_BYTES, fmt_params
from app.engine.registry import Context, ModuleDef, Result, docs, register, select, slider, toggle

DTYPES = [{"value": "fp32", "label": "FP32 (4 B)"}, {"value": "fp16", "label": "FP16 (2 B)"},
          {"value": "bf16", "label": "BF16 (2 B)"}, {"value": "int8", "label": "INT8 (1 B)"}]


# =========================================================================== KV cache
def kv_cache_sim(p: dict, ctx: Context) -> Result:
    heads, kvh = p["heads"], p["kv_heads"]
    r = Result("analytical")
    if kvh > heads or heads % kvh:
        kvh_fixed = max(d for d in range(1, heads + 1) if heads % d == 0 and d <= kvh)
        r.note(f"kv_heads={kvh} does not evenly divide heads={heads}; using {kvh_fixed} (GQA requires groups of equal size).", "warn")
        kvh = kvh_fixed
    arch = custom_arch(p["layers"], heads, p["head_dim"], kvh)
    hw = active_hw(ctx)
    me, ce = effs(ctx)
    bw, F = hw["bw_gbs"] * 1e9 * me, hw["tflops"] * 1e12 * ce
    B, prompt, gen, bpe = p["batch"], p["seq_len"], p["gen_tokens"], DTYPE_BYTES[p["dtype"]]
    N = arch.total_params()
    wbytes = N * 2.0  # weights kept in 16-bit; the dtype control applies to the KV cache
    per_tok = 2 * arch.layers * kvh * arch.head_dim * bpe  # bytes per token per sequence
    per_tok_mha = 2 * arch.layers * heads * arch.head_dim * bpe
    steps = np.arange(1, gen + 1)
    ctxlen = prompt + steps
    kv_mem = per_tok * ctxlen * B
    # with cache: one new token per step; stream weights + whole cache once
    t_cache = np.maximum((wbytes + kv_mem) / bw, (2 * N * B + 4 * arch.layers * arch.hidden * ctxlen * B) / F)
    # without cache: recompute the full prefix every step
    t_nocache = np.maximum(wbytes / bw, (2 * N * ctxlen * B + 4 * arch.layers * arch.hidden * ctxlen ** 2 * B) / F)
    prefill = max((2 * N * prompt * B + 4 * arch.layers * arch.hidden * prompt ** 2 * B) / F, wbytes / bw)
    total_cache, total_nocache = prefill + t_cache.sum(), prefill + t_nocache.sum()
    kv_final = per_tok * (prompt + gen) * B
    free = hw["mem_gb"] * 1e9 * 0.9 - wbytes
    max_seqs = int(max(0, free) // (per_tok * (prompt + gen))) if per_tok else 0
    computed_kv_cache = gen  # new K/V projections per layer/head over the generation
    computed_kv_nocache = int(ctxlen.sum())
    enabled = p["enabled"]
    t_active = t_cache if enabled else t_nocache
    r.metric("kv_per_token", "KV bytes / token", per_tok / 1024, "KiB", "2 × layers × kv_heads × head_dim × bytes")
    r.metric("kv_total", "KV cache at end", kv_final / 1e9 if enabled else 0, "GB", f"batch {B} × {prompt + gen} tokens")
    r.metric("weights", "Weights (16-bit)", wbytes / 1e9, "GB", f"{fmt_params(N)} parameters (derived from dims)")
    r.metric("ttft", "Time to first token", prefill * 1e3, "ms", "prefill is identical with or without a cache")
    r.metric("tpot", "Avg time / output token", float(t_active.mean() * 1e3), "ms")
    r.metric("total", "Total generation time", (total_cache if enabled else total_nocache), "s")
    r.metric("reuse", "K/V reuse ratio", (1 - computed_kv_cache / computed_kv_nocache) * 100 if enabled else 0, "%",
             "share of K/V projections served from the cache instead of recomputed")
    r.metric("max_seqs", "Concurrent sequences that fit", max_seqs, "", f"(90% of {hw['mem_gb']} GB − weights) ÷ KV per sequence",
             status="bad" if max_seqs == 0 else None)
    idx = downsample(list(steps))
    lab = [int(steps[i]) for i in idx]
    r.chart("growth", "KV cache growth per generated token", "line", lab, [
        {"label": f"GQA ({kvh} KV heads)", "data": [kv_mem[i] / 1e9 for i in idx]},
        {"label": f"MHA ({heads} KV heads)", "data": [per_tok_mha * ctxlen[i] * B / 1e9 for i in idx]},
        {"label": "MQA (1 KV head)", "data": [per_tok / kvh * ctxlen[i] * B / 1e9 for i in idx]},
    ], "generated token", "GB")
    r.chart("latency", "Per-token decode latency", "line", lab, [
        {"label": "With KV cache", "data": [t_cache[i] * 1e3 for i in idx]},
        {"label": "Without cache (recompute)", "data": [t_nocache[i] * 1e3 for i in idx]},
    ], "generated token", "ms", log_y=True)
    cum_c, cum_n = np.cumsum(np.ones(gen)), np.cumsum(ctxlen)
    r.chart("reuse", "Cumulative K/V projections computed (per layer, per head)", "line", lab, [
        {"label": "With cache (1 per step)", "data": [cum_c[i] for i in idx]},
        {"label": "Without cache (whole prefix per step)", "data": [cum_n[i] for i in idx]},
    ], "generated token", "projections", log_y=True)
    r.flow("Decode step with a KV cache", [
        {"label": "New token x_t", "sub": "1 position"},
        {"label": "Project Q, K, V", "sub": f"{arch.layers} layers"},
        {"label": "Append K,V → cache", "sub": f"+{per_tok / 1024:.0f} KiB/token", "cls": "accent"},
        {"label": "Attend Q_t over cached K,V", "sub": "O(t) reads, no recompute"},
        {"label": "Next-token logits"},
    ])
    r.compare("No KV cache", "KV cache", [
        ("Total generation time", total_nocache, total_cache, "s", "lower"),
        ("Avg time / token", t_nocache.mean() * 1e3, t_cache.mean() * 1e3, "ms", "lower"),
        ("K/V projections computed", computed_kv_nocache, computed_kv_cache, "", "lower"),
        ("Extra memory (KV)", 0, kv_final / 1e9, "GB", "lower"),
    ])
    if kv_final + wbytes > hw["mem_gb"] * 1e9:
        r.note(f"Weights + KV cache ({(kv_final + wbytes) / 1e9:.1f} GB) exceed {hw['label']} memory — reduce batch/sequence, use GQA, an INT8/FP8 KV cache, or paging/offload.", "warn")
    r.note(hw_note(ctx))
    r.recommend("Always enable the KV cache for autoregressive serving; the question is how to *bound* it.",
                "Prefer GQA/MQA models for high-concurrency serving: KV memory scales with kv_heads, not query heads.",
                "Use a paged allocator (vLLM PagedAttention) to avoid fragmentation and enable prefix sharing.",
                "Quantize the KV cache (FP8/INT8) when memory-bound; validate quality on long-context evals.")
    return r


register(ModuleDef(
    id="kv-cache", title="KV Cache", category="inference", original_index=1, fidelity="analytical",
    summary="Reuse attention keys/values from earlier tokens instead of recomputing them each decode step.",
    controls=[
        toggle("enabled", "Enable KV cache", True, "When off, every decode step recomputes K/V for the full prefix."),
        slider("seq_len", "Prompt / context length", 4096, 128, 131072, 128, "Tokens already in context before generation.", unit="tok"),
        slider("gen_tokens", "Tokens to generate", 512, 16, 8192, 16, "Number of autoregressive decode steps.", unit="tok"),
        slider("layers", "Transformer layers", 32, 1, 128, 1, "Each layer keeps its own K and V cache."),
        slider("heads", "Query attention heads", 32, 1, 128, 1, "hidden size = heads × head_dim."),
        slider("kv_heads", "KV heads (GQA groups)", 8, 1, 128, 1, "= heads → MHA; 1 → MQA; in between → grouped-query attention."),
        slider("head_dim", "Head dimension", 128, 32, 256, 32, "Per-head vector size."),
        select("dtype", "KV cache data type", "bf16", DTYPES, "Bytes per stored element."),
        slider("batch", "Batch size (concurrent sequences)", 1, 1, 256, 1, "KV memory scales linearly with batch."),
    ],
    simulate=kv_cache_sim,
    docs=docs(
        what="A per-layer store of the attention Key and Value vectors of every token already processed, so each new decode step only computes K/V for the newest token.",
        why="In autoregressive generation token t attends to tokens 1..t. Without caching, generating T tokens recomputes O(T²) projections; with caching it is O(T) and decode becomes memory-bandwidth-bound instead of compute-bound.",
        how="During prefill the model computes K,V for all prompt tokens and stores them. Each decode step projects only the new token to Q,K,V, appends K,V to the cache, and attends Q over all cached keys. Memory per token = 2 × layers × kv_heads × head_dim × bytes.",
        when_use="Always for autoregressive inference (chat, completion, agents). It is on by default in Hugging Face `generate`, vLLM, TGI, TensorRT-LLM and llama.cpp.",
        when_avoid="Single forward passes (classification, embeddings, reranking) gain nothing. In extreme memory pressure you trade it for recomputation or offload — but rarely disable it.",
        pros=["Turns O(T²) decode compute into O(T)", "Order-of-magnitude lower per-token latency for long outputs", "Enables prefix sharing and paged memory management"],
        cons=["Memory grows linearly with sequence × batch — often larger than the weights at high concurrency", "Limits concurrency and maximum context", "Fragmentation without a paged allocator"],
        use_case="An enterprise chatbot serving 200 concurrent sessions of 8K tokens on Llama-3-8B: KV memory (≈1 GB/seq in BF16 with GQA) — not the 16 GB of weights — determines how many GPUs you need.",
        code='''
from transformers import AutoModelForCausalLM, AutoTokenizer
import torch

tok = AutoTokenizer.from_pretrained("TinyLlama/TinyLlama-1.1B-Chat-v1.0")
model = AutoModelForCausalLM.from_pretrained("TinyLlama/TinyLlama-1.1B-Chat-v1.0", torch_dtype=torch.bfloat16)
ids = tok("Explain KV caching in one sentence.", return_tensors="pt").input_ids

# Manual decode loop that reuses past_key_values (what generate() does internally)
out = model(ids, use_cache=True)
past, next_id = out.past_key_values, out.logits[:, -1].argmax(-1, keepdim=True)
for _ in range(32):
    out = model(next_id, past_key_values=past, use_cache=True)   # only ONE new position
    past, next_id = out.past_key_values, out.logits[:, -1].argmax(-1, keepdim=True)

# KV bytes per token = 2 * layers * kv_heads * head_dim * dtype_bytes
cfg = model.config
kv_per_tok = 2 * cfg.num_hidden_layers * cfg.num_key_value_heads * (cfg.hidden_size // cfg.num_attention_heads) * 2
print(f"{kv_per_tok/1024:.1f} KiB per token")
''',
        diagram=[{"label": "Prompt"}, {"label": "Prefill: K,V for all tokens", "cls": "accent"}, {"label": "KV cache (per layer)"},
                 {"label": "Decode: Q_t · K_cache", "cls": "accent"}, {"label": "Token t+1"}],
        performance="Decode per-token time ≈ (weights + KV bytes) / memory bandwidth. KV memory = 2·L·H_kv·d·s·b·bytes. Llama-3-8B (GQA, 8 KV heads) uses 128 KiB/token in BF16; Llama-2-7B (MHA, 32 KV heads) uses 512 KiB/token.",
        checklist=["Size GPU memory for weights + peak KV (max_seq × max_batch)", "Use GQA/MQA models for high-concurrency workloads",
                   "Enable paged KV management (vLLM) and set gpu_memory_utilization", "Consider FP8/INT8 KV cache after quality evaluation",
                   "Set max_model_len to what you actually need", "Monitor KV utilization and preemption/swap counts"],
        applications=[{"domain": "Enterprise chatbots", "text": "Every chat turn decodes with a KV cache; its size determines sessions per GPU."},
                      {"domain": "Agentic AI", "text": "Long tool-use traces grow the cache; budget memory for 32K+ contexts."},
                      {"domain": "AI-assisted software engineering", "text": "Code completion latency is dominated by per-token decode, which the cache keeps flat."}],
        formula="KV_bytes = 2 × n_layers × n_kv_heads × d_head × seq_len × batch × bytes_per_element",
        limitations="Latency is a roofline estimate with your configured efficiency factors; real kernels (FlashAttention, fused ops) and scheduling change absolute numbers.",
        references=["Pope et al., 'Efficiently Scaling Transformer Inference' (2022)", "Ainslie et al., 'GQA' (2023)"],
    ),
))


# =========================================================================== Prefix caching
def prefix_cache_sim(p: dict, ctx: Context) -> Result:
    arch, hw = active_arch(ctx), active_hw(ctx)
    me, ce = effs(ctx)
    F = hw["tflops"] * 1e12 * ce
    N = arch.active_params()
    block = p["block_size"]
    n_req, n_prefix, plen, slen = p["requests"], p["distinct_prefixes"], p["prefix_len"], p["suffix_len"]
    cap_blocks = p["cache_capacity"] // block
    g = rng(p["seed"])
    if p["pattern"] == "zipf":
        w = 1 / np.arange(1, n_prefix + 1) ** 1.1
        assign = g.choice(n_prefix, size=n_req, p=w / w.sum())
    elif p["pattern"] == "round_robin":
        assign = np.arange(n_req) % n_prefix
    else:
        assign = g.integers(0, n_prefix, size=n_req)
    blocks_per_prefix = plen // block  # only full blocks are cacheable
    cache: OrderedDict = OrderedDict()
    computed, hits_list, hit_tokens_total, evictions = [], [], 0, 0
    for pid in assign:
        hit_blocks = 0
        if p["enabled"]:
            for b in range(blocks_per_prefix):
                key = (int(pid), b)  # chain key: prefix id + block index ≙ hash of all tokens up to this block
                if key in cache:
                    cache.move_to_end(key)
                    hit_blocks += 1
                else:
                    break
            for b in range(hit_blocks, blocks_per_prefix):
                cache[(int(pid), b)] = True
                cache.move_to_end((int(pid), b))
                while len(cache) > cap_blocks:
                    cache.popitem(last=False)
                    evictions += 1
        hit_tok = hit_blocks * block
        hit_tokens_total += hit_tok
        computed.append(plen + slen - hit_tok)
        hits_list.append(hit_tok)
    computed = np.array(computed)
    baseline = np.full(n_req, plen + slen)
    saved = baseline.sum() - computed.sum()
    def ttft(tokens):
        return (2 * N * tokens + 4 * arch.layers * arch.hidden * tokens ** 2) / F * 1e3
    t_base, t_new = ttft(baseline.astype(float)), ttft(computed.astype(float))
    kv_tok = 2 * arch.layers * arch.kv_heads * arch.head_dim * 2
    r = Result("simulated")
    r.set_fidelity("simulated", "Exact LRU block-cache simulation over a deterministic synthetic request stream; TTFT/FLOPs are analytical estimates.")
    r.metric("hit_rate", "Prefix token hit rate", hit_tokens_total / max(1, plen * n_req) * 100, "%")
    r.metric("saved", "Prefill tokens saved", saved / baseline.sum() * 100, "%", f"{int(saved):,} of {int(baseline.sum()):,} tokens")
    r.metric("flops_saved", "Prefill FLOPs saved", 2 * N * saved / 1e15, "PFLOP", "≈ 2 × N_params × tokens")
    r.metric("ttft_before", "Mean TTFT (no cache)", float(t_base.mean()), "ms")
    r.metric("ttft_after", "Mean TTFT (prefix cache)", float(t_new.mean()), "ms")
    r.metric("cache_mem", "Cache memory (capacity)", cap_blocks * block * kv_tok / 1e9, "GB", f"{arch.label} BF16 KV")
    r.metric("evictions", "Block evictions", evictions, "", "LRU evictions due to capacity")
    show = min(n_req, 120)
    r.chart("per_request", f"Prefill tokens computed per request (first {show})", "bar", list(range(1, show + 1)), [
        {"label": "Reused from cache", "data": hits_list[:show]},
        {"label": "Computed", "data": computed[:show].tolist()},
    ], "request #", "tokens", stacked=True)
    r.chart("cumulative", "Cumulative prefill compute", "line", list(range(1, n_req + 1))[:: max(1, n_req // 200)], [
        {"label": "Without prefix cache", "data": np.cumsum(baseline)[:: max(1, n_req // 200)].tolist()},
        {"label": "With prefix cache", "data": np.cumsum(computed)[:: max(1, n_req // 200)].tolist()},
    ], "request #", "tokens")
    r.tokens("Request anatomy (block size %d)" % block, (
        [{"text": f"system/prefix ×{blocks_per_prefix} blocks", "cls": "shared"}] +
        ([{"text": f"partial block {plen % block} tok (not cacheable)", "cls": "partial"}] if plen % block else []) +
        [{"text": f"unique suffix {slen} tok", "cls": "unique"}]))
    r.compare("No prefix cache", "Prefix cache", [
        ("Prefill tokens computed", int(baseline.sum()), int(computed.sum()), "", "lower"),
        ("Mean TTFT", float(t_base.mean()), float(t_new.mean()), "ms", "lower"),
        ("Extra memory", 0, cap_blocks * block * kv_tok / 1e9, "GB", "lower"),
    ])
    if p["enabled"] and cap_blocks < blocks_per_prefix:
        r.note("Cache capacity is smaller than one prefix: blocks get evicted before reuse (thrashing).", "warn")
    r.note("Cache keys must include everything that changes the KV state: model/adapter version, exact token ids, and tenant id when prefixes contain private data.")
    r.recommend("Put stable content (system prompt, tool schemas, few-shot examples) first and volatile content last.",
                "Align long shared prefixes to block boundaries; partial blocks are not reused.",
                "Isolate caches per tenant if prefixes embed tenant data (timing side channels).",
                "In vLLM enable `enable_prefix_caching=True`; SGLang's RadixAttention does this automatically.")
    return r


register(ModuleDef(
    id="prefix-caching", title="Prefix Caching", category="inference", original_index=5, fidelity="simulated",
    summary="Reuse the prefill computation (KV states) of a shared prompt prefix across requests.",
    controls=[
        toggle("enabled", "Enable prefix cache", True, "Reuse cached KV blocks for matching prompt prefixes."),
        slider("prefix_len", "Shared prefix length", 2000, 0, 16384, 16, "System prompt / tool schema / few-shot tokens.", unit="tok"),
        slider("suffix_len", "Unique suffix length", 200, 1, 8192, 1, "Per-request user message tokens.", unit="tok"),
        slider("requests", "Number of requests", 200, 1, 2000, 1, "Requests in the simulated stream."),
        slider("distinct_prefixes", "Distinct prefixes", 3, 1, 64, 1, "How many different system prompts/tenants share the server."),
        select("pattern", "Arrival pattern", "zipf", [{"value": "zipf", "label": "Zipf (popular prefixes dominate)"},
                                                   {"value": "round_robin", "label": "Round-robin"},
                                                   {"value": "random", "label": "Uniform random"}], "Which prefix each request uses."),
        slider("cache_capacity", "Cache capacity", 8192, 0, 262144, 512, "Tokens of KV the cache may hold (LRU eviction).", unit="tok"),
        select("block_size", "Block size", 16, [{"value": 16, "label": "16 tokens"}, {"value": 32, "label": "32 tokens"},
                                               {"value": 64, "label": "64 tokens"}], "Only full blocks can be shared."),
        slider("seed", "Random seed", 7, 0, 9999, 1, "Deterministic request stream."),
    ],
    simulate=prefix_cache_sim,
    docs=docs(
        what="Caching the KV states computed during prefill for a prompt prefix, keyed by its exact tokens, so later requests starting with the same tokens skip that prefill work.",
        why="Production prompts share long fixed prefixes (system prompt, policies, tool definitions, retrieved boilerplate). Recomputing them per request wastes GPU time and inflates time-to-first-token.",
        how="KV memory is split into fixed-size blocks. Each full block is identified by a hash of all tokens up to and including it. A new request walks its prompt block by block; matching blocks are mapped in (copy-free) and only the remaining tokens are prefilled. Unused blocks are evicted (LRU).",
        when_use="Multi-user chatbots with a common system prompt, agent loops that resend growing transcripts, few-shot prompts, multi-turn conversations, batch jobs over the same document.",
        when_avoid="Workloads with unique prompts (no sharing), very short prefixes (< 1 block), or strict tenant isolation requirements without per-tenant cache partitioning.",
        pros=["Large TTFT reductions for long shared prefixes", "Free capacity: saved prefill FLOPs serve other requests", "Transparent to clients"],
        cons=["Consumes KV memory that could hold active sequences", "Only exact token-prefix matches hit — a single changed early token invalidates everything after it", "Potential cross-tenant timing side channel"],
        use_case="A customer-support assistant with a 3,000-token policy + tool schema prefix: every request after the first reuses the prefix and TTFT falls to the cost of the user's message.",
        code='''
from vllm import LLM, SamplingParams

llm = LLM(model="meta-llama/Llama-3.1-8B-Instruct", enable_prefix_caching=True)
SYSTEM = open("support_policy.txt").read()          # long, stable prefix FIRST
prompts = [f"{SYSTEM}\\n\\nUser: {q}\\nAssistant:" for q in user_questions]
outs = llm.generate(prompts, SamplingParams(max_tokens=256))
# Hosted APIs expose the same idea as "prompt caching" (cache_control / cached_tokens).
''',
        diagram=[{"label": "Request"}, {"label": "Hash prefix blocks", "cls": "accent"}, {"label": "Block table lookup"},
                 {"label": "Hit → map cached KV"}, {"label": "Prefill only the suffix", "cls": "accent"}, {"label": "Decode"}],
        performance="Saved prefill FLOPs ≈ 2·N·(hit tokens). TTFT falls roughly in proportion to the hit fraction of the prompt. Cost: cache memory = cached tokens × KV bytes/token.",
        checklist=["Order prompts: static → semi-static → dynamic", "Avoid timestamps/IDs at the top of prompts", "Include model + adapter + tenant in cache key",
                   "Monitor hit rate and eviction rate", "Reserve memory for active sequences before cache"],
        applications=[{"domain": "Enterprise chatbots", "text": "Shared corporate system prompt across thousands of users."},
                      {"domain": "Agentic AI", "text": "Each agent step resends the transcript; the prefix grows and is reused."},
                      {"domain": "Customer support automation", "text": "Fixed policies and tool schemas prepend every ticket."},
                      {"domain": "Multi-agent enterprise platforms", "text": "Agents sharing a common role/tool prefix reuse it across sessions."}],
        limitations="Request stream is synthetic. Real hit rates depend on your prompt templates and traffic mix.",
        references=["Kwon et al., 'vLLM / PagedAttention' (2023)", "Zheng et al., 'SGLang: RadixAttention' (2024)"],
    ),
))
