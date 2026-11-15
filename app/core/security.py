"""Password hashing (Argon2id), password policy, JWT and CSRF helpers."""
from __future__ import annotations

import hmac
import re
import secrets
import time
import uuid
from collections import defaultdict, deque
from datetime import datetime, timedelta, timezone

import jwt
from argon2 import PasswordHasher
from argon2.exceptions import InvalidHashError, VerifyMismatchError

from app.core.config import get_settings

_hasher = PasswordHasher()  # Argon2id with library-recommended parameters

ROLES = ("admin", "engineer", "researcher", "viewer")
ROLE_LABELS = {"admin": "Administrator", "engineer": "AI Engineer", "researcher": "Researcher", "viewer": "Viewer"}

# Permission matrix (role-based access control).
PERMISSIONS: dict[str, set[str]] = {
    "admin": {"run", "save", "benchmark", "fine_tune", "manage_users", "view_audit", "configure"},
    "engineer": {"run", "save", "benchmark", "fine_tune", "configure"},
    "researcher": {"run", "save", "benchmark", "configure"},
    "viewer": {"run"},
}


def hash_password(password: str) -> str:
    return _hasher.hash(password)


def verify_password(password: str, password_hash: str) -> bool:
    try:
        return _hasher.verify(password_hash, password)
    except (VerifyMismatchError, InvalidHashError):
        return False


PASSWORD_RULES = [
    (r".{10,}", "at least 10 characters"),
    (r"[A-Z]", "an uppercase letter"),
    (r"[a-z]", "a lowercase letter"),
    (r"\d", "a digit"),
    (r"[^A-Za-z0-9]", "a symbol"),
]


def password_problems(password: str) -> list[str]:
    return [msg for pattern, msg in PASSWORD_RULES if not re.search(pattern, password or "")]


def create_access_token(user_id: int, role: str, *, remember: bool = False) -> tuple[str, int]:
    s = get_settings()
    lifetime = timedelta(days=s.remember_me_days) if remember else timedelta(minutes=s.access_token_minutes)
    now = datetime.now(timezone.utc)
    payload = {"sub": str(user_id), "role": role, "iat": now, "exp": now + lifetime, "jti": uuid.uuid4().hex, "typ": "access"}
    return jwt.encode(payload, s.secret_key, algorithm=s.jwt_algorithm), int(lifetime.total_seconds())


def decode_access_token(token: str) -> dict:
    s = get_settings()
    return jwt.decode(token, s.secret_key, algorithms=[s.jwt_algorithm], options={"require": ["exp", "sub", "jti"]})


def new_csrf_token() -> str:
    return secrets.token_urlsafe(32)


def csrf_matches(cookie_value: str | None, header_value: str | None) -> bool:
    return bool(cookie_value and header_value and hmac.compare_digest(cookie_value, header_value))


class LoginThrottle:
    """In-memory sliding-window throttle for failed logins (per email+IP).

    Suitable for a single-process deployment; use Redis for multi-replica setups.
    """

    def __init__(self) -> None:
        self._failures: dict[str, deque[float]] = defaultdict(deque)

    def _prune(self, key: str, window: int) -> deque[float]:
        q = self._failures[key]
        now = time.monotonic()
        while q and now - q[0] > window:
            q.popleft()
        return q

    def is_locked(self, key: str) -> bool:
        s = get_settings()
        return len(self._prune(key, s.login_lockout_seconds)) >= s.login_max_attempts

    def fail(self, key: str) -> None:
        self._failures[key].append(time.monotonic())

    def reset(self, key: str) -> None:
        self._failures.pop(key, None)


login_throttle = LoginThrottle()
