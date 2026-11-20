"""Database initialisation and demo-user seeding."""
from __future__ import annotations

import logging

from sqlalchemy import func, select

from app.core.config import get_settings
from app.core.security import hash_password, verify_password
from app.db.models import User
from app.db.session import Base, get_engine, get_sessionmaker

logger = logging.getLogger("llmlab.seed")


async def create_tables() -> None:
    async with get_engine().begin() as conn:
        await conn.run_sync(Base.metadata.create_all)


async def seed_users() -> None:
    s = get_settings()
    users = [(s.demo_user_email, s.demo_user_password, s.demo_user_role, "Demo User")]
    if s.seed_role_users:
        users += [
            ("engineer@llmlab.local", s.demo_user_password, "engineer", "Demo AI Engineer"),
            ("researcher@llmlab.local", s.demo_user_password, "researcher", "Demo Researcher"),
            ("viewer@llmlab.local", s.demo_user_password, "viewer", "Demo Viewer"),
        ]
    async with get_sessionmaker()() as db:
        for email, pw, role, name in users:
            existing = (await db.execute(select(User).where(User.email == email))).scalar_one_or_none()
            if existing is None:
                db.add(User(email=email, password_hash=hash_password(pw), role=role, full_name=name))
                logger.info("seeded_user", extra={"extra_fields": {"email": email, "role": role}})
        if s.public_demo_enabled:
            await _sync_public_demo(db, s.public_demo_email.strip(), s.public_demo_password)
        await db.commit()


async def _sync_public_demo(db, email: str, password: str) -> None:
    """Keep the published demo account in line with configuration on every start.

    Its credentials are public, so it is always forced to the least-privileged *viewer* role and
    its password follows LLMLAB_PUBLIC_DEMO_PASSWORD (rotating the env value rotates the account).
    """
    user = (await db.execute(select(User).where(func.lower(User.email) == email.lower()))).scalar_one_or_none()
    if user is None:
        db.add(User(email=email, password_hash=hash_password(password), role="viewer", full_name="Public Demo"))
        logger.info("seeded_user", extra={"extra_fields": {"email": email, "role": "viewer", "public_demo": True}})
        return
    changed = []
    if user.role != "viewer":
        user.role = "viewer"
        changed.append("role")
    if not verify_password(password, user.password_hash):
        user.password_hash = hash_password(password)
        changed.append("password")
    if not user.is_active:
        user.is_active = True
        changed.append("is_active")
    if changed:
        logger.info("public_demo_synced", extra={"extra_fields": {"email": email, "changed": changed}})
