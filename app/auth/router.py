"""Login / logout / session endpoints (JWT in HttpOnly cookie + CSRF double-submit)."""
from __future__ import annotations

from datetime import datetime, timezone

from fastapi import APIRouter, Depends, Request, Response
from pydantic import BaseModel, Field
from sqlalchemy import func, select
from sqlalchemy.ext.asyncio import AsyncSession

from app.auth.deps import Principal, audit, current_principal
from app.core.config import get_settings
from app.core.errors import AuthError
from app.core.security import ROLE_LABELS, PERMISSIONS, create_access_token, login_throttle, new_csrf_token, verify_password
from app.db.models import User
from app.db.session import get_db

router = APIRouter(prefix="/api/auth", tags=["auth"])


class LoginRequest(BaseModel):
    email: str = Field(min_length=3, max_length=255)
    password: str = Field(min_length=1, max_length=256)
    remember_me: bool = False


class UserOut(BaseModel):
    id: int
    email: str
    full_name: str
    role: str
    role_label: str
    is_active: bool = True
    permissions: list[str]


def user_out(u: User) -> UserOut:
    return UserOut(id=u.id, email=u.email, full_name=u.full_name, role=u.role,
                   role_label=ROLE_LABELS.get(u.role, u.role), is_active=u.is_active, permissions=sorted(PERMISSIONS.get(u.role, set())))


class LoginResponse(BaseModel):
    user: UserOut
    access_token: str
    token_type: str = "bearer"
    expires_in: int
    csrf_token: str


@router.post("/login", response_model=LoginResponse)
async def login(body: LoginRequest, request: Request, response: Response, db: AsyncSession = Depends(get_db)):
    s = get_settings()
    email = body.email.strip().lower()
    key = f"{email}|{request.client.host if request.client else ''}"
    if login_throttle.is_locked(key):
        raise AuthError("Too many failed attempts. Try again later.")
    user = (await db.execute(select(User).where(func.lower(User.email) == email))).scalar_one_or_none()
    if user is None or not user.is_active or not verify_password(body.password, user.password_hash):
        login_throttle.fail(key)
        await audit(db, request, user.id if user else None, "login_failed", email=email)
        raise AuthError("Invalid email or password")
    login_throttle.reset(key)
    token, max_age = create_access_token(user.id, user.role, remember=body.remember_me)
    csrf = new_csrf_token()
    cookie_age = max_age if body.remember_me else None  # session cookie unless "remember me"
    response.set_cookie(s.session_cookie_name, token, max_age=cookie_age, httponly=True, secure=s.cookie_secure,
                        samesite=s.cookie_samesite, path="/")
    response.set_cookie(s.csrf_cookie_name, csrf, max_age=cookie_age, httponly=False, secure=s.cookie_secure,
                        samesite=s.cookie_samesite, path="/")
    user.last_login_at = datetime.now(timezone.utc)
    await db.commit()
    await audit(db, request, user.id, "login", remember=body.remember_me)
    return LoginResponse(user=user_out(user), access_token=token, expires_in=max_age, csrf_token=csrf)


@router.post("/logout")
async def logout(request: Request, response: Response, p: Principal = Depends(current_principal), db: AsyncSession = Depends(get_db)):
    s = get_settings()
    response.delete_cookie(s.session_cookie_name, path="/")
    response.delete_cookie(s.csrf_cookie_name, path="/")
    await audit(db, request, p.id, "logout")
    return {"ok": True}


@router.get("/me", response_model=UserOut)
async def me(p: Principal = Depends(current_principal)):
    return user_out(p.user)
