"""Concept-module catalog and execution endpoints."""
from __future__ import annotations

import inspect

from fastapi import APIRouter, Depends
from pydantic import BaseModel, Field
from sqlalchemy.ext.asyncio import AsyncSession
from starlette.concurrency import run_in_threadpool

from app.auth.deps import Principal, current_principal, require
from app.db.session import get_db
from app.engine.common import APP_DOMAINS
from app.engine.registry import CATEGORIES, Context, ModuleDef, all_modules, get_module
from app.llm.providers import ProviderHub
from app.users.router import load_settings

router = APIRouter(prefix="/api/modules", tags=["modules"])


class RunRequest(BaseModel):
    params: dict = Field(default_factory=dict)


async def execute(module: ModuleDef, params: dict, settings: dict) -> dict:
    """Validate, run (sync modules in a worker thread), and return the uniform result."""
    clean = module.validate(params)
    ctx = Context(settings=settings, providers=ProviderHub(settings))
    import time

    t0 = time.perf_counter()
    if inspect.iscoroutinefunction(module.simulate):
        result = await module.simulate(clean, ctx)
    else:
        result = await run_in_threadpool(module.simulate, clean, ctx)
    out = result.data
    out.update(module=module.id, params=clean, compute_ms=round((time.perf_counter() - t0) * 1e3, 2), mode=out["fidelity"])
    return out


@router.get("")
async def list_modules(_: Principal = Depends(current_principal)):
    mods = all_modules()
    return {"categories": CATEGORIES, "modules": [m.summary_dict() for m in mods]}


@router.get("/applications")
async def applications(_: Principal = Depends(current_principal)):
    """'Where would I use this in a real application?' — aggregated by domain."""
    out = {d: [] for d in APP_DOMAINS}
    for m in all_modules():
        for a in m.docs.get("applications", []):
            out.setdefault(a["domain"], []).append({"module": m.id, "title": m.title, "text": a["text"]})
    return out


@router.get("/{module_id}")
async def module_detail(module_id: str, _: Principal = Depends(current_principal)):
    return get_module(module_id).detail_dict()


@router.post("/{module_id}/run")
async def run_module(module_id: str, body: RunRequest, p: Principal = Depends(require("run")), db: AsyncSession = Depends(get_db)):
    return await execute(get_module(module_id), body.params, await load_settings(db, p.id))
