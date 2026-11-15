"""Database initialisation and demo-user seeding."""
from __future__ import annotations

import logging

from sqlalchemy import select

from app.core.config import get_settings
from app.core.security import hash_password
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
        await db.commit()
