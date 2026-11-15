"""Provider status and a direct generation endpoint (real-model integration interface)."""
from __future__ import annotations

from fastapi import APIRouter, Depends
from pydantic import BaseModel, Field
from sqlalchemy.ext.asyncio import AsyncSession

from app.auth.deps import Principal, current_principal, require
from app.db.session import get_db
from app.llm.providers import ProviderHub, ml_environment_async
from app.users.router import load_settings

router = APIRouter(prefix="/api/llm", tags=["llm"])


class GenerateRequest(BaseModel):
    prompt: str = Field(min_length=1, max_length=20000)
    temperature: float = Field(0.7, ge=0, le=2)
    top_p: float = Field(1.0, gt=0, le=1)
    max_tokens: int = Field(128, ge=1, le=2048)
    seed: int | None = None


@router.get("/status")
async def status(p: Principal = Depends(current_principal), db: AsyncSession = Depends(get_db)):
    hub = ProviderHub(await load_settings(db, p.id))
    return {"active": await hub.status(), "real": hub.has_real(), "environment": await ml_environment_async()}


@router.post("/generate")
async def generate(body: GenerateRequest, p: Principal = Depends(require("run")), db: AsyncSession = Depends(get_db)):
    hub = ProviderHub(await load_settings(db, p.id))
    return await hub.generate(body.prompt, temperature=body.temperature, top_p=body.top_p, max_tokens=body.max_tokens, seed=body.seed)
