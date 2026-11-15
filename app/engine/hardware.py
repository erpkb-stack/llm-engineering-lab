"""Hardware profiles, model architecture presets and the analytical performance model.

Every number produced here is an *analytical estimate* (roofline-style), not a measurement.
Peak figures are vendor dense FP16/BF16 tensor throughput and memory bandwidth; achieved
efficiency factors are explicit, user-editable assumptions.
"""
from __future__ import annotations

import math
from copy import deepcopy
from dataclasses import asdict, dataclass

# name: memory GB, bandwidth GB/s, dense fp16 TFLOPS, default $/hour (editable assumption)
HARDWARE: dict[str, dict] = {
    "rtx4090": {"label": "NVIDIA RTX 4090 (24 GB)", "mem_gb": 24, "bw_gbs": 1008, "tflops": 165, "usd_hr": 0.70},
    "l4": {"label": "NVIDIA L4 (24 GB)", "mem_gb": 24, "bw_gbs": 300, "tflops": 121, "usd_hr": 0.80},
    "a10g": {"label": "NVIDIA A10G (24 GB)", "mem_gb": 24, "bw_gbs": 600, "tflops": 125, "usd_hr": 1.01},
    "a100_40": {"label": "NVIDIA A100 40 GB", "mem_gb": 40, "bw_gbs": 1555, "tflops": 312, "usd_hr": 3.00},
    "a100_80": {"label": "NVIDIA A100 80 GB SXM", "mem_gb": 80, "bw_gbs": 2039, "tflops": 312, "usd_hr": 3.70},
    "h100": {"label": "NVIDIA H100 80 GB SXM", "mem_gb": 80, "bw_gbs": 3350, "tflops": 989, "usd_hr": 6.00},
    "m2max": {"label": "Apple M2 Max (96 GB unified)", "mem_gb": 96, "bw_gbs": 400, "tflops": 13.6, "usd_hr": 0.0},
    "cpu": {"label": "Generic server CPU (64 GB)", "mem_gb": 64, "bw_gbs": 100, "tflops": 2.0, "usd_hr": 0.40},
}

DTYPE_BYTES = {"fp32": 4.0, "fp16": 2.0, "bf16": 2.0, "int8": 1.0, "int4": 0.5, "nf4": 0.5}


@dataclass(frozen=True)
class ModelArch:
    key: str
    label: str
    layers: int
    hidden: int
    heads: int
    kv_heads: int
    intermediate: int
    vocab: int
    context: int
    experts: int = 1
    top_k_experts: int = 1
    tied_embeddings: bool = False

    @property
    def head_dim(self) -> int:
        return self.hidden // self.heads

    def attn_params(self) -> int:
        h, kvd = self.hidden, self.kv_heads * self.head_dim
        return 2 * h * h + 2 * h * kvd  # q,o + k,v

    def mlp_params(self) -> int:
        return 3 * self.hidden * self.intermediate  # SwiGLU gate/up/down

    def total_params(self) -> int:
        per_layer = self.attn_params() + self.mlp_params() * self.experts + 2 * self.hidden
        if self.experts > 1:
            per_layer += self.hidden * self.experts  # router
        emb = self.vocab * self.hidden * (1 if self.tied_embeddings else 2)
        return self.layers * per_layer + emb + self.hidden

    def active_params(self) -> int:
        per_layer = self.attn_params() + self.mlp_params() * self.top_k_experts + 2 * self.hidden
        emb = self.vocab * self.hidden * (1 if self.tied_embeddings else 2)
        return self.layers * per_layer + emb

    def as_dict(self) -> dict:
        d = asdict(self)
        d.update(head_dim=self.head_dim, total_params=self.total_params(), active_params=self.active_params())
        return d


MODELS: dict[str, ModelArch] = {m.key: m for m in [
    ModelArch("qwen2.5-0.5b", "Qwen2.5-0.5B", 24, 896, 14, 2, 4864, 151936, 32768, tied_embeddings=True),
    ModelArch("tinyllama-1.1b", "TinyLlama-1.1B", 22, 2048, 32, 4, 5632, 32000, 2048),
    ModelArch("phi-3-mini", "Phi-3-mini (3.8B)", 32, 3072, 32, 32, 8192, 32064, 4096),
    ModelArch("llama-2-7b", "Llama-2-7B", 32, 4096, 32, 32, 11008, 32000, 4096),
    ModelArch("mistral-7b", "Mistral-7B", 32, 4096, 32, 8, 14336, 32000, 32768),
    ModelArch("llama-3-8b", "Llama-3-8B", 32, 4096, 32, 8, 14336, 128256, 8192),
    ModelArch("llama-2-13b", "Llama-2-13B", 40, 5120, 40, 40, 13824, 32000, 4096),
    ModelArch("mixtral-8x7b", "Mixtral-8x7B (MoE)", 32, 4096, 32, 8, 14336, 32000, 32768, experts=8, top_k_experts=2),
    ModelArch("llama-3-70b", "Llama-3-70B", 80, 8192, 64, 8, 28672, 128256, 8192),
]}

MODEL_OPTIONS = [{"value": k, "label": m.label} for k, m in MODELS.items()]
HARDWARE_OPTIONS = [{"value": k, "label": h["label"]} for k, h in HARDWARE.items()]

DEFAULT_SETTINGS: dict = {
    "active_model": "llama-3-8b",
    "hardware": "a10g",
    "dtype": "bf16",
    "memory_efficiency": 0.70,   # achieved fraction of peak memory bandwidth
    "compute_efficiency": 0.45,  # achieved fraction of peak tensor FLOPS (MFU)
    "usd_per_hour_override": None,
    "default_prompt_tokens": 1024,
    "default_output_tokens": 256,
    "provider": "simulation",
    "ollama_model": "llama3.2:1b",
    "openai_base_url": "",
    "openai_model": "",
}


def merge_settings(data: dict | None) -> dict:
    out = deepcopy(DEFAULT_SETTINGS)
    for k, v in (data or {}).items():
        if k not in out:
            continue
        if k == "active_model" and v not in MODELS:
            continue
        if k == "hardware" and v not in HARDWARE:
            continue
        if k == "dtype" and v not in DTYPE_BYTES:
            continue
        if k in ("memory_efficiency", "compute_efficiency"):
            try:
                v = min(1.0, max(0.05, float(v)))
            except (TypeError, ValueError):
                continue
        if k in ("default_prompt_tokens", "default_output_tokens"):
            try:
                v = int(min(131072, max(1, int(v))))
            except (TypeError, ValueError):
                continue
        if k == "usd_per_hour_override" and v is not None:
            try:
                v = max(0.0, float(v))
            except (TypeError, ValueError):
                continue
        if k == "provider" and v not in ("simulation", "ollama", "openai", "huggingface"):
            continue
        out[k] = v
    return out


# ----------------------------------------------------------------- perf model
def kv_cache_bytes(arch: ModelArch, seq: int, batch: int, dtype: str, kv_heads: int | None = None) -> float:
    kvh = arch.kv_heads if kv_heads is None else kv_heads
    return 2 * arch.layers * kvh * arch.head_dim * seq * batch * DTYPE_BYTES[dtype]


@dataclass
class PerfEstimate:
    weights_gb: float
    kv_gb: float
    activations_gb: float
    total_gb: float
    fits: bool
    ttft_ms: float
    tpot_ms: float
    total_ms: float
    tokens_per_s: float
    decode_bound: str
    cost_per_1k_tokens: float

    def as_dict(self) -> dict:
        return asdict(self)


def estimate(arch: ModelArch, hw: dict, *, dtype: str = "bf16", prompt: int = 1024, output: int = 256, batch: int = 1,
             mem_eff: float = 0.7, comp_eff: float = 0.45, kv_dtype: str | None = None, usd_hr: float | None = None,
             use_kv_cache: bool = True) -> PerfEstimate:
    """Roofline estimate for single-replica autoregressive inference.

    prefill:  compute-bound  ~ 2 * N_active * prompt * batch FLOPs  (+ attention 4*L*d*s^2)
    decode:   max(memory time to stream weights + KV, compute time for 2*N_active*batch FLOPs)
    """
    wbytes = arch.total_params() * DTYPE_BYTES[dtype]
    seq = prompt + output
    kvb = kv_cache_bytes(arch, seq, batch, kv_dtype or dtype) if use_kv_cache else 0.0
    act = batch * seq * arch.hidden * 2 * 4  # rough working-set for activations (bytes)
    total = wbytes + kvb + act
    bw = hw["bw_gbs"] * 1e9 * mem_eff
    flops = hw["tflops"] * 1e12 * comp_eff
    n_act = arch.active_params()
    attn_flops_prefill = 4 * arch.layers * arch.hidden * prompt * prompt * batch
    prefill_s = max((2 * n_act * prompt * batch + attn_flops_prefill) / flops, wbytes / bw)
    if use_kv_cache:
        avg_kv = kv_cache_bytes(arch, prompt + output // 2, batch, kv_dtype or dtype)
        mem_s = (wbytes + avg_kv) / bw
        comp_s = 2 * n_act * batch / flops
        tpot = max(mem_s, comp_s)
        bound = "memory-bandwidth" if mem_s >= comp_s else "compute"
    else:
        # Without a cache every step re-runs the full forward pass over the whole prefix.
        avg_len = prompt + output / 2
        comp_s = (2 * n_act * avg_len * batch + 4 * arch.layers * arch.hidden * avg_len * avg_len * batch) / flops
        tpot = max(comp_s, wbytes / bw)
        bound = "compute (full recompute)"
    total_s = prefill_s + tpot * output
    tps = batch / tpot if tpot > 0 else 0.0  # aggregate decode throughput
    rate = hw["usd_hr"] if usd_hr is None else usd_hr
    cost = (rate / 3600.0) * total_s / max(1, batch * (prompt + output)) * 1000
    return PerfEstimate(
        weights_gb=wbytes / 1e9, kv_gb=kvb / 1e9, activations_gb=act / 1e9, total_gb=total / 1e9,
        fits=total / 1e9 <= hw["mem_gb"], ttft_ms=prefill_s * 1e3, tpot_ms=tpot * 1e3, total_ms=total_s * 1e3,
        tokens_per_s=tps, decode_bound=bound, cost_per_1k_tokens=cost,
    )


def fmt_params(n: float) -> str:
    for unit, div in (("T", 1e12), ("B", 1e9), ("M", 1e6), ("K", 1e3)):
        if abs(n) >= div:
            return f"{n / div:.2f}{unit}"
    return f"{n:.0f}"


def safe_log(x: float) -> float:
    return math.log(max(x, 1e-300))
