"""Experiment history and saved configurations (per user), with JSON/CSV export."""
from __future__ import annotations

import csv
import io
import json

from fastapi import APIRouter, Depends, Query, Request
from fastapi.responses import Response
from pydantic import BaseModel, Field
from sqlalchemy import delete, select
from sqlalchemy.ext.asyncio import AsyncSession

from app.auth.deps import Principal, audit, current_principal, require
from app.core.errors import NotFoundError
from app.db.models import Experiment, SavedConfiguration
from app.db.session import get_db
from app.engine.registry import get_module

router = APIRouter(prefix="/api", tags=["experiments"])


class ExperimentIn(BaseModel):
    module_id: str
    name: str = Field("", max_length=200)
    kind: str = Field("run", pattern="^(run|benchmark|fine_tune)$")
    params: dict = Field(default_factory=dict)
    baseline_params: dict | None = None
    result: dict = Field(default_factory=dict)
    notes: str = Field("", max_length=5000)


class ConfigIn(BaseModel):
    module_id: str
    name: str = Field(min_length=1, max_length=200)
    params: dict = Field(default_factory=dict)


def exp_out(e: Experiment, full: bool = False) -> dict:
    d = {"id": e.id, "module_id": e.module_id, "kind": e.kind, "name": e.name, "mode": e.mode, "params": e.params,
         "baseline_params": e.baseline_params, "notes": e.notes, "created_at": e.created_at.isoformat() if e.created_at else None,
         "metrics": {m["label"]: m["value"] for m in (e.result.get("metrics") or [])} if isinstance(e.result, dict) else {}}
    if full:
        d["result"] = e.result
    return d


def _slim(result: dict) -> dict:
    """Store the full result but cap very large arrays to keep the DB lean."""
    s = json.dumps(result, default=str)
    if len(s) < 400_000:
        return result
    keep = {k: result.get(k) for k in ("metrics", "comparison", "notes", "fidelity", "fidelity_note", "module", "params", "mode", "baseline", "variant", "deltas")}
    keep["truncated"] = True
    return keep


@router.post("/experiments", status_code=201)
async def save_experiment(body: ExperimentIn, request: Request, p: Principal = Depends(require("save")), db: AsyncSession = Depends(get_db)):
    get_module(body.module_id) if body.kind != "fine_tune" else None
    e = Experiment(user_id=p.id, module_id=body.module_id, kind=body.kind, name=body.name or body.module_id, params=body.params,
                   baseline_params=body.baseline_params, result=_slim(body.result), notes=body.notes,
                   mode=str(body.result.get("fidelity") or body.result.get("mode") or "simulation"))
    db.add(e)
    await db.commit()
    await audit(db, request, p.id, "experiment_saved", experiment_id=e.id, module=body.module_id)
    return exp_out(e)


@router.get("/experiments")
async def list_experiments(module_id: str | None = None, kind: str | None = None, limit: int = Query(100, le=500),
                           p: Principal = Depends(current_principal), db: AsyncSession = Depends(get_db)):
    q = select(Experiment).where(Experiment.user_id == p.id).order_by(Experiment.id.desc()).limit(limit)
    if module_id:
        q = q.where(Experiment.module_id == module_id)
    if kind:
        q = q.where(Experiment.kind == kind)
    return [exp_out(e) for e in (await db.execute(q)).scalars()]


@router.get("/experiments/export")
async def export_experiments(format: str = Query("json", pattern="^(json|csv)$"), ids: str | None = None,
                             p: Principal = Depends(current_principal), db: AsyncSession = Depends(get_db)):
    q = select(Experiment).where(Experiment.user_id == p.id).order_by(Experiment.id)
    if ids:
        q = q.where(Experiment.id.in_([int(x) for x in ids.split(",") if x.strip().isdigit()]))
    rows = list((await db.execute(q)).scalars())
    if format == "json":
        return Response(json.dumps([exp_out(e, full=True) for e in rows], default=str, indent=2), media_type="application/json",
                        headers={"Content-Disposition": "attachment; filename=experiments.json"})
    buf = io.StringIO()
    w = csv.writer(buf)
    w.writerow(["id", "created_at", "module_id", "kind", "name", "mode", "metric", "value", "params"])
    for e in rows:
        metrics = exp_out(e)["metrics"] or {"": ""}
        for k, v in metrics.items():
            w.writerow([e.id, e.created_at.isoformat() if e.created_at else "", e.module_id, e.kind, e.name, e.mode, k, v, json.dumps(e.params)])
    return Response(buf.getvalue(), media_type="text/csv", headers={"Content-Disposition": "attachment; filename=experiments.csv"})


@router.get("/experiments/{exp_id}")
async def get_experiment(exp_id: int, p: Principal = Depends(current_principal), db: AsyncSession = Depends(get_db)):
    e = await db.get(Experiment, exp_id)
    if e is None or e.user_id != p.id:
        raise NotFoundError("Experiment not found")
    return exp_out(e, full=True)


@router.delete("/experiments/{exp_id}")
async def delete_experiment(exp_id: int, request: Request, p: Principal = Depends(require("save")), db: AsyncSession = Depends(get_db)):
    e = await db.get(Experiment, exp_id)
    if e is None or e.user_id != p.id:
        raise NotFoundError("Experiment not found")
    await db.delete(e)
    await db.commit()
    await audit(db, request, p.id, "experiment_deleted", experiment_id=exp_id)
    return {"ok": True}


# ------------------------------------------------------------ saved configurations
@router.get("/configs")
async def list_configs(module_id: str | None = None, p: Principal = Depends(current_principal), db: AsyncSession = Depends(get_db)):
    q = select(SavedConfiguration).where(SavedConfiguration.user_id == p.id).order_by(SavedConfiguration.id.desc())
    if module_id:
        q = q.where(SavedConfiguration.module_id == module_id)
    return [{"id": c.id, "module_id": c.module_id, "name": c.name, "params": c.params, "created_at": c.created_at.isoformat()} for c in (await db.execute(q)).scalars()]


@router.post("/configs", status_code=201)
async def save_config(body: ConfigIn, p: Principal = Depends(require("save")), db: AsyncSession = Depends(get_db)):
    clean = get_module(body.module_id).validate(body.params)
    c = SavedConfiguration(user_id=p.id, module_id=body.module_id, name=body.name, params=clean)
    db.add(c)
    await db.commit()
    return {"id": c.id, "module_id": c.module_id, "name": c.name, "params": c.params}


@router.delete("/configs/{config_id}")
async def delete_config(config_id: int, p: Principal = Depends(require("save")), db: AsyncSession = Depends(get_db)):
    res = await db.execute(delete(SavedConfiguration).where(SavedConfiguration.id == config_id, SavedConfiguration.user_id == p.id))
    await db.commit()
    if not res.rowcount:
        raise NotFoundError("Configuration not found")
    return {"ok": True}
