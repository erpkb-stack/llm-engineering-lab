"""Fine-Tuning Laboratory endpoints."""
from __future__ import annotations

import asyncio

from fastapi import APIRouter, Depends, File, Form, Request, UploadFile
from pydantic import BaseModel, Field
from sqlalchemy import select
from sqlalchemy.ext.asyncio import AsyncSession

from app.auth.deps import Principal, audit, current_principal, require
from app.core.config import get_settings
from app.core.errors import CapabilityUnavailable, NotFoundError, ParameterError
from app.db.models import FineTuneJob
from app.db.session import get_db
from app.engine.hardware import MODEL_OPTIONS, MODELS
from app.fine_tuning import jobs
from app.fine_tuning.datasets import SCHEMAS, load_dataset, validate
from app.llm.providers import ml_environment_async
from app.users.router import load_settings

router = APIRouter(prefix="/api/fine-tuning", tags=["fine-tuning"])
_tasks: set[asyncio.Task] = set()


class JobConfig(BaseModel):
    base_model: str = "llama-3-8b"
    dataset_id: str
    method: str = Field("lora", pattern="^(full|lora|qlora|peft)$")
    mode: str = Field("simulation", pattern="^(simulation|real)$")
    learning_rate: float = Field(2e-4, gt=0, le=1e-1)
    epochs: int = Field(3, ge=1, le=20)
    batch_size: int = Field(8, ge=1, le=256)
    grad_accum: int = Field(1, ge=1, le=64)
    rank: int = Field(16, ge=1, le=256)
    alpha: int = Field(32, ge=1, le=1024)
    dropout: float = Field(0.05, ge=0, le=0.5)
    weight_decay: float = Field(0.0, ge=0, le=0.5)
    max_seq_len: int = Field(1024, ge=32, le=32768)
    quantization: str = Field("none", pattern="^(none|int8|nf4)$")
    target_all_linear: bool = True
    seed: int = Field(0, ge=0, le=99999)
    name: str = Field("", max_length=200)


def job_out(j: FineTuneJob) -> dict:
    return {"id": j.id, "status": j.status, "mode": j.mode, "config": j.config, "dataset_summary": j.dataset_summary, "metrics": j.metrics,
            "adapter_path": j.adapter_path, "error": j.error, "created_at": j.created_at.isoformat() if j.created_at else None,
            "finished_at": j.finished_at.isoformat() if j.finished_at else None}


@router.get("/environment")
async def environment(_: Principal = Depends(current_principal)):
    return {"environment": await ml_environment_async(), "models": MODEL_OPTIONS, "schemas": SCHEMAS, "real_model": get_settings().hf_model}


@router.post("/datasets/validate")
async def validate_dataset(file: UploadFile = File(...), max_seq_len: int = Form(2048), p: Principal = Depends(require("fine_tune"))):
    s = get_settings()
    raw = await file.read(s.upload_max_bytes + 1)
    if len(raw) > s.upload_max_bytes:
        raise ParameterError(f"File exceeds {s.upload_max_bytes // (1024 * 1024)} MB limit")
    return validate(raw, file.filename or "upload.jsonl", p.id, max_seq_len)


@router.post("/datasets/sample")
async def use_sample(p: Principal = Depends(require("fine_tune"))):
    from app.core.config import BASE_DIR

    path = BASE_DIR / "data" / "samples" / "finetune_support.jsonl"
    return validate(path.read_bytes(), path.name, p.id, 2048)


@router.post("/plan")
async def plan_job(body: JobConfig, p: Principal = Depends(require("fine_tune")), db: AsyncSession = Depends(get_db)):
    s = await load_settings(db, p.id)
    recs = load_dataset(p.id, body.dataset_id)
    mean = sum(len((r["prompt"] + " " + r["completion"]).split()) * 1.33 for r in recs) / len(recs)
    cfg = body.model_dump()
    if body.base_model not in MODELS:
        raise ParameterError("Unknown base model")
    return {"plan": jobs.plan(cfg, int(len(recs) * 0.9) or 1, mean, s["hardware"], s["compute_efficiency"]), "fidelity": "analytical"}


@router.post("/jobs", status_code=201)
async def create_job(body: JobConfig, request: Request, p: Principal = Depends(require("fine_tune")), db: AsyncSession = Depends(get_db)):
    if body.base_model not in MODELS:
        raise ParameterError("Unknown base model")
    recs = load_dataset(p.id, body.dataset_id)
    env = await ml_environment_async()
    if body.mode == "real" and not env["real_training_possible"]:
        raise CapabilityUnavailable("Real training needs torch + transformers + peft installed (requirements-ml.txt) and LLMLAB_ENABLE_HF_MODELS=true. "
                                    "Use simulation mode instead.", details={"environment": env})
    lengths = sorted(len((r["prompt"] + " " + r["completion"]).split()) * 1.33 for r in recs)
    n_val = max(1, len(recs) // 10) if len(recs) >= 10 else 0
    summary = {"records": len(recs), "split": {"train": len(recs) - n_val, "validation": n_val}, "tokens": {"mean": sum(lengths) / len(lengths)}}
    job = FineTuneJob(user_id=p.id, status="queued", mode=body.mode, config=body.model_dump(), dataset_summary=summary)
    db.add(job)
    await db.commit()
    settings = await load_settings(db, p.id)
    coro = jobs.run_real(job.id, body.model_dump(), recs) if body.mode == "real" else jobs.run_simulated(job.id, body.model_dump(), summary, settings)
    task = asyncio.create_task(coro)
    _tasks.add(task)
    task.add_done_callback(_tasks.discard)
    await audit(db, request, p.id, "fine_tune_started", job_id=job.id, mode=body.mode, method=body.method)
    return job_out(job)


@router.get("/jobs")
async def list_jobs(p: Principal = Depends(current_principal), db: AsyncSession = Depends(get_db)):
    rows = (await db.execute(select(FineTuneJob).where(FineTuneJob.user_id == p.id).order_by(FineTuneJob.id.desc()).limit(50))).scalars()
    return [job_out(j) for j in rows]


@router.get("/jobs/{job_id}")
async def get_job(job_id: int, p: Principal = Depends(current_principal), db: AsyncSession = Depends(get_db)):
    j = await db.get(FineTuneJob, job_id)
    if j is None or j.user_id != p.id:
        raise NotFoundError("Job not found")
    return job_out(j)


@router.post("/jobs/{job_id}/cancel")
async def cancel_job(job_id: int, p: Principal = Depends(require("fine_tune")), db: AsyncSession = Depends(get_db)):
    j = await db.get(FineTuneJob, job_id)
    if j is None or j.user_id != p.id:
        raise NotFoundError("Job not found")
    jobs.bus.cancelled.add(job_id)
    return {"ok": True}
