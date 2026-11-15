"""Benchmark Studio: A/B comparison, parameter sweeps and model comparison.

In simulation mode every number is an analytical estimate or deterministic simulation and is
labelled with its fidelity. ``/api/benchmarks/live`` performs REAL wall-clock measurements
against the configured provider (or reports that none is configured).
"""
from __future__ import annotations

import statistics
import time

from fastapi import APIRouter, Depends, Request
from pydantic import BaseModel, Field
from sqlalchemy.ext.asyncio import AsyncSession

from app.auth.deps import Principal, audit, require
from app.db.models import Experiment
from app.db.session import get_db
from app.engine.hardware import DTYPE_BYTES, HARDWARE, MODELS, estimate, fmt_params
from app.engine.registry import get_module
from app.llm.providers import ProviderHub
from app.modules_api.router import execute
from app.users.router import load_settings

router = APIRouter(prefix="/api/benchmarks", tags=["benchmarks"])


class CompareRequest(BaseModel):
    module_id: str
    baseline_params: dict = Field(default_factory=dict)
    variant_params: dict = Field(default_factory=dict)
    save: bool = False
    name: str = ""


def metric_deltas(a: dict, b: dict) -> list[dict]:
    am = {m["key"]: m for m in a.get("metrics", [])}
    out = []
    for m in b.get("metrics", []):
        base = am.get(m["key"])
        if base is None:
            continue
        row = {"key": m["key"], "label": m["label"], "unit": m.get("unit", ""), "baseline": base["value"], "variant": m["value"]}
        if isinstance(base["value"], (int, float)) and isinstance(m["value"], (int, float)) and not isinstance(base["value"], bool):
            row["delta"] = m["value"] - base["value"]
            row["pct"] = (m["value"] - base["value"]) / abs(base["value"]) * 100 if base["value"] else None
        out.append(row)
    return out


@router.post("/compare")
async def compare(body: CompareRequest, request: Request, p: Principal = Depends(require("benchmark")), db: AsyncSession = Depends(get_db)):
    mod = get_module(body.module_id)
    settings = await load_settings(db, p.id)
    a = await execute(mod, body.baseline_params, settings)
    b = await execute(mod, {**body.baseline_params, **body.variant_params}, settings)
    result = {"module": mod.id, "baseline": a, "variant": b, "deltas": metric_deltas(a, b), "fidelity": b["fidelity"],
              "fidelity_note": b["fidelity_note"], "metrics": b["metrics"],
              "changed": {k: {"from": a["params"].get(k), "to": v} for k, v in b["params"].items() if a["params"].get(k) != v}}
    if body.save:
        e = Experiment(user_id=p.id, module_id=mod.id, kind="benchmark", name=body.name or f"{mod.title} A/B", params=b["params"],
                       baseline_params=a["params"], mode=b["fidelity"],
                       result={"metrics": b["metrics"], "deltas": result["deltas"], "changed": result["changed"], "fidelity": b["fidelity"],
                               "baseline_metrics": a["metrics"]})
        db.add(e)
        await db.commit()
        result["experiment_id"] = e.id
        await audit(db, request, p.id, "benchmark_saved", experiment_id=e.id, module=mod.id)
    return result


class SweepRequest(BaseModel):
    module_id: str
    params: dict = Field(default_factory=dict)
    sweep_key: str
    values: list = Field(min_length=1, max_length=25)


@router.post("/sweep")
async def sweep(body: SweepRequest, p: Principal = Depends(require("benchmark")), db: AsyncSession = Depends(get_db)):
    mod = get_module(body.module_id)
    settings = await load_settings(db, p.id)
    rows = []
    for v in body.values:
        res = await execute(mod, {**body.params, body.sweep_key: v}, settings)
        rows.append({"value": v, "metrics": {m["key"]: m["value"] for m in res["metrics"]}, "labels": {m["key"]: m["label"] for m in res["metrics"]}})
    return {"module": mod.id, "sweep_key": body.sweep_key, "rows": rows, "fidelity": mod.fidelity}


class ModelCompareRequest(BaseModel):
    models: list[str] = Field(default_factory=lambda: ["tinyllama-1.1b", "llama-3-8b", "mixtral-8x7b", "llama-3-70b"])
    hardware: str = "a100_80"
    dtype: str = "bf16"
    prompt_tokens: int = Field(1024, ge=1, le=131072)
    output_tokens: int = Field(256, ge=1, le=16384)
    batch: int = Field(1, ge=1, le=512)


@router.post("/models")
async def model_comparison(body: ModelCompareRequest, p: Principal = Depends(require("run")), db: AsyncSession = Depends(get_db)):
    s = await load_settings(db, p.id)
    hw = HARDWARE.get(body.hardware, HARDWARE["a100_80"])
    if body.dtype not in DTYPE_BYTES:
        body.dtype = "bf16"
    rows = []
    for key in body.models:
        if key not in MODELS:
            continue
        a = MODELS[key]
        e = estimate(a, hw, dtype=body.dtype, prompt=body.prompt_tokens, output=body.output_tokens, batch=body.batch,
                     mem_eff=s["memory_efficiency"], comp_eff=s["compute_efficiency"])
        rows.append({"model": key, "label": a.label, "params": fmt_params(a.total_params()), "active_params": fmt_params(a.active_params()),
                     "context": a.context, "kv_heads": a.kv_heads, **e.as_dict()})
    return {"hardware": hw, "dtype": body.dtype, "rows": rows, "fidelity": "analytical",
            "fidelity_note": "Roofline estimates using your Settings efficiency factors — not measurements."}


class LiveBenchRequest(BaseModel):
    prompt: str = Field("Explain the KV cache in two sentences.", max_length=4000)
    runs: int = Field(3, ge=1, le=10)
    max_tokens: int = Field(64, ge=1, le=512)


@router.post("/live")
async def live_benchmark(body: LiveBenchRequest, p: Principal = Depends(require("benchmark")), db: AsyncSession = Depends(get_db)):
    """Real wall-clock measurement against the configured provider."""
    hub = ProviderHub(await load_settings(db, p.id))
    if not hub.has_real():
        return {"measured": False, "message": "No real provider configured (Settings → Model provider). Nothing was measured."}
    runs = []
    for i in range(body.runs):
        t0 = time.perf_counter()
        res = await hub.generate(body.prompt, temperature=0.0, max_tokens=body.max_tokens, seed=i)
        wall = (time.perf_counter() - t0) * 1e3
        toks = res.get("eval_count") or res.get("new_tokens") or (res.get("usage") or {}).get("completion_tokens")
        runs.append({"run": i + 1, "wall_ms": wall, "output_tokens": toks, "tokens_per_s": (toks / (wall / 1e3)) if toks else None,
                     "provider": res.get("provider"), "measured": res.get("measured", False), "text": res.get("text", "")[:200]})
    walls = [r["wall_ms"] for r in runs]
    return {"measured": all(r["measured"] for r in runs), "runs": runs, "mean_ms": statistics.mean(walls),
            "stdev_ms": statistics.pstdev(walls), "note": "Measured end-to-end wall-clock latency including network and queueing."}
