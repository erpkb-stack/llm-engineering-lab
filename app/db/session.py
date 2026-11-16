"""Async SQLAlchemy engine/session management (SQLite default, PostgreSQL optional)."""
from __future__ import annotations

from collections.abc import AsyncIterator
from pathlib import Path

from sqlalchemy.ext.asyncio import AsyncEngine, AsyncSession, async_sessionmaker, create_async_engine
from sqlalchemy.orm import DeclarativeBase

from sqlalchemy.pool import NullPool

from app.core.config import ON_SERVERLESS, get_settings


class Base(DeclarativeBase):
    pass


_engine: AsyncEngine | None = None
_sessionmaker: async_sessionmaker[AsyncSession] | None = None


def normalize_db_url(url: str) -> str:
    """Accept the plain ``postgres://…?sslmode=require`` URLs that Neon/Supabase/Vercel Postgres hand out."""
    for prefix in ("postgres://", "postgresql://"):
        if url.startswith(prefix):
            url = "postgresql+asyncpg://" + url[len(prefix):]
    if url.startswith("postgresql+asyncpg://"):
        url = url.replace("sslmode=", "ssl=").replace("&channel_binding=require", "").replace("channel_binding=require&", "")
    return url


def get_engine() -> AsyncEngine:
    global _engine, _sessionmaker
    if _engine is None:
        url = normalize_db_url(get_settings().database_url)
        if url.startswith("sqlite"):
            path = url.split("///", 1)[-1]
            if path and path != ":memory:":
                Path(path).parent.mkdir(parents=True, exist_ok=True)
        kw = {"poolclass": NullPool} if ON_SERVERLESS and not url.startswith("sqlite") else {}  # no pools across frozen instances
        _engine = create_async_engine(url, future=True, pool_pre_ping=not url.startswith("sqlite"), **kw)
        _sessionmaker = async_sessionmaker(_engine, expire_on_commit=False)
    return _engine


def get_sessionmaker() -> async_sessionmaker[AsyncSession]:
    get_engine()
    assert _sessionmaker is not None
    return _sessionmaker


async def get_db() -> AsyncIterator[AsyncSession]:
    async with get_sessionmaker()() as session:
        yield session


async def reset_engine() -> None:
    """Dispose the engine (used by tests to switch databases)."""
    global _engine, _sessionmaker
    if _engine is not None:
        await _engine.dispose()
    _engine = None
    _sessionmaker = None
