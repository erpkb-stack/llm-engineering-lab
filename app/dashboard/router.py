"""Dashboard summary metrics for the active configuration."""
from __future__ import annotations

from fastapi import APIRouter, Depends
from sqlalchemy import func, select
from sqlalchemy.ext.asyncio import AsyncSession

from app.auth.deps import Principal, current_principal
from app.db.models import Experiment, FineTuneJob, SavedConfiguration
from app.db.session import get_db
from app.engine.common import active_arch, active_hw
from app.engine.hardware import estimate, fmt_params
from app.engine.registry import CATEGORIES, Context, all_modules
from app.evaluation.modules import eval_sim
from app.llm.providers import ProviderHub
from app.users.router import load_settings

router = APIRouter(prefix="/api/dashboard", tags=["dashboard"])


@router.get("/summary")
async def summary(p: Principal = Depends(current_principal), db: AsyncSession = Depends(get_db)):
    s = await load_settings(db, p.id)
    ctx = Context(settings=s)
    arch, hw = active_arch(ctx), active_hw(ctx)
    pin, pout = s["default_prompt_tokens"], s["default_output_tokens"]
    e = estimate(arch, hw, dtype=s["dtype"], prompt=pin, output=pout, mem_eff=s["memory_efficiency"], comp_eff=s["compute_efficiency"])
    q = eval_sim({"threshold": 0.5, "support_threshold": 0.6, "k": 3, "seed": 12}, ctx).data
    qm = {m["key"]: m["value"] for m in q["metrics"]}
    counts = {
        "experiments": (await db.execute(select(func.count()).select_from(Experiment).where(Experiment.user_id == p.id))).scalar_one(),
        "configs": (await db.execute(select(func.count()).select_from(SavedConfiguration).where(SavedConfiguration.user_id == p.id))).scalar_one(),
        "fine_tune_jobs": (await db.execute(select(func.count()).select_from(FineTuneJob).where(FineTuneJob.user_id == p.id))).scalar_one(),
    }
    recent = [{"id": x.id, "module_id": x.module_id, "kind": x.kind, "name": x.name, "mode": x.mode, "created_at": x.created_at.isoformat()}
              for x in (await db.execute(select(Experiment).where(Experiment.user_id == p.id).order_by(Experiment.id.desc()).limit(6))).scalars()]
    mods = all_modules()
    by_cat = {c: sum(1 for m in mods if m.category == c) for c in CATEGORIES}
    batch_curve = []
    for b in (1, 2, 4, 8, 16, 32, 64):
        eb = estimate(arch, hw, dtype=s["dtype"], prompt=pin, output=pout, batch=b, mem_eff=s["memory_efficiency"], comp_eff=s["compute_efficiency"])
        batch_curve.append({"batch": b, "tokens_per_s": eb.tokens_per_s, "tpot_ms": eb.tpot_ms, "memory_gb": eb.total_gb, "fits": eb.fits})
    hub = ProviderHub(s)
    return {
        "fidelity": "analytical",
        "fidelity_note": "Latency, memory, throughput and cost are roofline estimates for the active configuration. Quality metrics come from the "
                         "Model Evaluation module's synthetic labelled set.",
        "active_model": {"key": arch.key, "label": arch.label, "params": fmt_params(arch.total_params()), "active_params": fmt_params(arch.active_params()),
                         "context_window": arch.context, "layers": arch.layers, "hidden": arch.hidden, "heads": arch.heads, "kv_heads": arch.kv_heads},
        "hardware": hw, "dtype": s["dtype"], "input_tokens": pin, "output_tokens": pout,
        "estimate": e.as_dict(),
        "quality": {"F1": qm.get("f1"), "Faithfulness %": qm.get("faithfulness"), "nDCG@3": qm.get("ndcg"), "MRR": qm.get("mrr")},
        "counts": counts, "recent": recent, "modules_by_category": {CATEGORIES[c]: n for c, n in by_cat.items()}, "module_total": len(mods),
        "batch_curve": batch_curve, "provider": {"name": hub.active.name, "real": hub.has_real()},
    }
