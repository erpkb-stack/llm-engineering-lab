"""Shared helpers for module simulations."""
from __future__ import annotations

import hashlib

import numpy as np

from app.engine.hardware import HARDWARE, MODELS, ModelArch
from app.engine.registry import Context

APP_DOMAINS = ["Enterprise chatbots", "Financial services", "Healthcare information systems", "Customer support automation",
               "Agentic AI", "Document intelligence", "Semantic search", "AI-assisted software engineering",
               "Multi-agent enterprise platforms"]


def rng(seed: int | str = 0) -> np.random.Generator:
    if isinstance(seed, str):
        seed = int(hashlib.sha256(seed.encode()).hexdigest()[:8], 16)
    return np.random.default_rng(int(seed))


def active_arch(ctx: Context) -> ModelArch:
    return MODELS.get(ctx.settings.get("active_model"), MODELS["llama-3-8b"])


def active_hw(ctx: Context) -> dict:
    hw = dict(HARDWARE.get(ctx.settings.get("hardware"), HARDWARE["a10g"]))
    if ctx.settings.get("usd_per_hour_override") is not None:
        hw["usd_hr"] = float(ctx.settings["usd_per_hour_override"])
    return hw


def effs(ctx: Context) -> tuple[float, float]:
    return float(ctx.settings.get("memory_efficiency", 0.7)), float(ctx.settings.get("compute_efficiency", 0.45))


def custom_arch(layers: int, heads: int, head_dim: int, kv_heads: int, vocab: int = 32000) -> ModelArch:
    hidden = heads * head_dim
    inter = int(round(hidden * 8 / 3 / 256) * 256) or 256
    return ModelArch("custom", "Custom", layers, hidden, heads, kv_heads, inter, vocab, 131072)


def gb(x: float) -> float:
    return x / 1e9


def hw_note(ctx: Context) -> str:
    hw = active_hw(ctx)
    me, ce = effs(ctx)
    return (f"Hardware assumption: {hw['label']} — {hw['bw_gbs']} GB/s × {me:.0%} achieved bandwidth, "
            f"{hw['tflops']} TFLOPS × {ce:.0%} MFU. Change in Settings.")


def softmax(x: np.ndarray, axis: int = -1) -> np.ndarray:
    z = x - np.max(x, axis=axis, keepdims=True)
    e = np.exp(z)
    return e / e.sum(axis=axis, keepdims=True)


def downsample(xs: list, max_points: int = 200) -> list[int]:
    """Indices for evenly spaced samples of a long series (keeps charts responsive)."""
    n = len(xs)
    if n <= max_points:
        return list(range(n))
    step = n / max_points
    idx = sorted({int(i * step) for i in range(max_points)} | {n - 1})
    return idx
