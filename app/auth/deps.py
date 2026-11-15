"""Authentication / authorization dependencies (dependency injection)."""
from __future__ import annotations

from dataclasses import dataclass

import jwt
from fastapi import Depends, Request
from sqlalchemy.ext.asyncio import AsyncSession

from app.core.config import get_settings
from app.core.errors import AuthError, PermissionDenied
from app.core.security import PERMISSIONS, csrf_matches, decode_access_token
from app.db.models import AuditLog, User
from app.db.session import get_db

UNSAFE_METHODS = {"POST", "PUT", "PATCH", "DELETE"}


@dataclass
class Principal:
    user: User
    via_cookie: bool

    @property
    def id(self) -> int:
        return self.user.id

    @property
    def role(self) -> str:
        return self.user.role

    def can(self, perm: str) -> bool:
        return perm in PERMISSIONS.get(self.user.role, set())


def _extract_token(request: Request) -> tuple[str | None, bool]:
    auth = request.headers.get("authorization", "")
    if auth.lower().startswith("bearer "):
        return auth[7:].strip(), False
    return request.cookies.get(get_settings().session_cookie_name), True


async def current_principal(request: Request, db: AsyncSession = Depends(get_db)) -> Principal:
    token, via_cookie = _extract_token(request)
    if not token:
        raise AuthError("Authentication required")
    try:
        claims = decode_access_token(token)
    except jwt.ExpiredSignatureError as exc:
        raise AuthError("Session expired") from exc
    except jwt.PyJWTError as exc:
        raise AuthError("Invalid session") from exc
    user = await db.get(User, int(claims["sub"]))
    if user is None or not user.is_active:
        raise AuthError("Account inactive or not found")
    # CSRF: double-submit cookie check for cookie-authenticated state-changing requests.
    if via_cookie and request.method in UNSAFE_METHODS:
        s = get_settings()
        if not csrf_matches(request.cookies.get(s.csrf_cookie_name), request.headers.get("x-csrf-token")):
            raise PermissionDenied("CSRF token missing or invalid")
    return Principal(user=user, via_cookie=via_cookie)


def require(perm: str):
    async def _dep(p: Principal = Depends(current_principal)) -> Principal:
        if not p.can(perm):
            raise PermissionDenied(f"Your role '{p.role}' lacks the '{perm}' permission")
        return p

    return _dep


async def audit(db: AsyncSession, request: Request | None, user_id: int | None, action: str, **detail) -> None:
    ip = request.client.host if request and request.client else ""
    db.add(AuditLog(user_id=user_id, action=action, detail=detail, ip=ip))
    await db.commit()
