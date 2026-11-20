"""Public read-only demo account: configuration guards, the login-page endpoint and seeding."""
import pytest
from sqlalchemy import select

from app.core.config import Settings, get_settings
from app.db.models import User
from app.db.seed import seed_users
from app.db.session import get_sessionmaker

PROD = dict(environment="production", secret_key="x" * 40, cookie_secure=True,
            demo_user_password="Admin-Secret-123", seed_role_users=False)


def test_public_demo_disabled_by_default():
    assert Settings(_env_file=None).public_demo_enabled is False


def test_production_rejects_reusing_admin_password():
    with pytest.raises(ValueError, match="must differ"):
        Settings(_env_file=None, **PROD, public_demo_email="guest@lab", public_demo_password="Admin-Secret-123")


def test_production_rejects_role_users_with_public_demo():
    with pytest.raises(ValueError, match="SEED_ROLE_USERS"):
        Settings(_env_file=None, **{**PROD, "seed_role_users": True}, public_demo_email="guest@lab", public_demo_password="Guest-1")


def test_public_email_must_differ_from_admin():
    with pytest.raises(ValueError, match="PUBLIC_DEMO_EMAIL"):
        Settings(_env_file=None, demo_user_email="demo@llmlab.local", public_demo_email="DEMO@llmlab.local", public_demo_password="Guest-1")


def test_production_accepts_valid_public_demo():
    s = Settings(_env_file=None, **PROD, public_demo_email="guest@lab", public_demo_password="Guest-1")
    assert s.public_demo_enabled


def test_demo_endpoint_outside_production_shows_seeded_demo(anon):
    d = anon.get("/api/auth/demo").json()
    assert d["enabled"] and d["email"] == "demo@llmlab.local" and d["read_only"] is False


def test_demo_endpoint_hidden_in_production_without_public_demo(anon, monkeypatch):
    monkeypatch.setattr(get_settings(), "environment", "production")
    assert anon.get("/api/auth/demo").json() == {"enabled": False}


def _user(anon, email):
    async def q():
        async with get_sessionmaker()() as db:
            return (await db.execute(select(User).where(User.email == email))).scalar_one_or_none()
    return anon.portal.call(q)


def test_public_demo_is_seeded_as_viewer_and_kept_in_sync(anon, monkeypatch):
    s = get_settings()
    monkeypatch.setattr(s, "public_demo_email", "guest@llmlab.local")
    monkeypatch.setattr(s, "public_demo_password", "Guest-Pass-1")

    d = anon.get("/api/auth/demo").json()
    assert d == {"enabled": True, "email": "guest@llmlab.local", "password": "Guest-Pass-1",
                 "role": "viewer", "role_label": "Viewer", "read_only": True}

    anon.portal.call(seed_users)
    r = anon.post("/api/auth/login", json={"email": "guest@llmlab.local", "password": "Guest-Pass-1"})
    assert r.status_code == 200 and r.json()["user"]["role"] == "viewer"
    assert r.json()["user"]["permissions"] == ["run"]

    # Someone promotes the account and the env password is rotated: a restart repairs both.
    async def promote():
        async with get_sessionmaker()() as db:
            u = (await db.execute(select(User).where(User.email == "guest@llmlab.local"))).scalar_one()
            u.role = "admin"
            await db.commit()
    anon.portal.call(promote)
    monkeypatch.setattr(s, "public_demo_password", "Guest-Pass-2")
    anon.portal.call(seed_users)

    assert _user(anon, "guest@llmlab.local").role == "viewer"
    assert anon.post("/api/auth/login", json={"email": "guest@llmlab.local", "password": "Guest-Pass-1"}).status_code == 401
    assert anon.post("/api/auth/login", json={"email": "guest@llmlab.local", "password": "Guest-Pass-2"}).status_code == 200
