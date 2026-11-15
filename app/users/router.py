"""User management (admin), per-user settings and audit log."""
from __future__ import annotations

from fastapi import APIRouter, Depends, Request
from pydantic import BaseModel, Field
from sqlalchemy import func, select
from sqlalchemy.ext.asyncio import AsyncSession

from app.auth.deps import Principal, audit, current_principal, require
from app.auth.router import UserOut, user_out
from app.core.errors import NotFoundError, ParameterError
from app.core.security import ROLES, hash_password, password_problems
from app.db.models import AuditLog, User, UserSettings
from app.db.session import get_db
from app.engine.hardware import DEFAULT_SETTINGS, HARDWARE, merge_settings

router = APIRouter(prefix="/api", tags=["users"])


class CreateUser(BaseModel):
    email: str = Field(min_length=3, max_length=255, pattern=r"^[^@\s]+@[^@\s]+$")
    full_name: str = ""
    password: str
    role: str = "viewer"


class UpdateUser(BaseModel):
    role: str | None = None
    is_active: bool | None = None


@router.get("/users", response_model=list[UserOut])
async def list_users(_: Principal = Depends(require("manage_users")), db: AsyncSession = Depends(get_db)):
    return [user_out(u) for u in (await db.execute(select(User).order_by(User.id))).scalars()]


@router.post("/users", response_model=UserOut, status_code=201)
async def create_user(body: CreateUser, request: Request, p: Principal = Depends(require("manage_users")), db: AsyncSession = Depends(get_db)):
    if body.role not in ROLES:
        raise ParameterError(f"role must be one of {ROLES}")
    problems = password_problems(body.password)
    if problems:
        raise ParameterError("Password must contain " + ", ".join(problems), details={"problems": problems})
    email = body.email.strip().lower()
    if (await db.execute(select(User).where(func.lower(User.email) == email))).scalar_one_or_none():
        raise ParameterError("A user with that email already exists")
    u = User(email=email, full_name=body.full_name, password_hash=hash_password(body.password), role=body.role)
    db.add(u)
    await db.commit()
    await audit(db, request, p.id, "user_created", email=email, role=body.role)
    return user_out(u)


@router.patch("/users/{user_id}", response_model=UserOut)
async def update_user(user_id: int, body: UpdateUser, request: Request, p: Principal = Depends(require("manage_users")), db: AsyncSession = Depends(get_db)):
    u = await db.get(User, user_id)
    if u is None:
        raise NotFoundError("User not found")
    if body.role is not None:
        if body.role not in ROLES:
            raise ParameterError(f"role must be one of {ROLES}")
        u.role = body.role
    if body.is_active is not None:
        if u.id == p.id and not body.is_active:
            raise ParameterError("You cannot deactivate yourself")
        u.is_active = body.is_active
    await db.commit()
    await audit(db, request, p.id, "user_updated", target=user_id, **body.model_dump(exclude_none=True))
    return user_out(u)


@router.get("/audit")
async def audit_log(limit: int = 100, _: Principal = Depends(require("view_audit")), db: AsyncSession = Depends(get_db)):
    rows = (await db.execute(select(AuditLog).order_by(AuditLog.id.desc()).limit(min(limit, 500)))).scalars()
    return [{"id": r.id, "user_id": r.user_id, "action": r.action, "detail": r.detail, "ip": r.ip, "created_at": r.created_at} for r in rows]


# ---------------------------------------------------------------- settings
async def load_settings(db: AsyncSession, user_id: int) -> dict:
    row = await db.get(UserSettings, user_id)
    return merge_settings(row.data if row else {})


@router.get("/settings")
async def get_user_settings(p: Principal = Depends(current_principal), db: AsyncSession = Depends(get_db)):
    return {"settings": await load_settings(db, p.id), "defaults": DEFAULT_SETTINGS,
            "hardware_profiles": HARDWARE}


@router.put("/settings")
async def put_user_settings(body: dict, request: Request, p: Principal = Depends(require("configure")), db: AsyncSession = Depends(get_db)):
    merged = merge_settings(body)  # validates & drops unknown keys
    row = await db.get(UserSettings, p.id)
    if row is None:
        db.add(UserSettings(user_id=p.id, data=merged))
    else:
        row.data = merged
    await db.commit()
    await audit(db, request, p.id, "settings_updated")
    return {"settings": merged}
