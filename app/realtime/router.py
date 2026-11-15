"""WebSocket endpoints: live module runs and fine-tuning progress.

Authentication uses the HttpOnly session cookie sent with the handshake; the Origin header
must match the Host to prevent cross-site WebSocket hijacking.
"""
from __future__ import annotations

import asyncio
import json
import logging
from urllib.parse import urlparse

import jwt
from fastapi import APIRouter, WebSocket, WebSocketDisconnect

from app.core.config import get_settings
from app.core.security import PERMISSIONS, decode_access_token
from app.db.models import FineTuneJob, User
from app.db.session import get_sessionmaker
from app.engine.registry import get_module
from app.fine_tuning.jobs import bus
from app.modules_api.router import execute
from app.users.router import load_settings

router = APIRouter(tags=["realtime"])
logger = logging.getLogger("llmlab.ws")


async def ws_user(ws: WebSocket) -> User | None:
    origin = ws.headers.get("origin")
    host = ws.headers.get("host", "")
    if origin and urlparse(origin).netloc != host:
        return None
    token = ws.cookies.get(get_settings().session_cookie_name)
    if not token:
        return None
    try:
        claims = decode_access_token(token)
    except jwt.PyJWTError:
        return None
    async with get_sessionmaker()() as db:
        u = await db.get(User, int(claims["sub"]))
        return u if u and u.is_active else None


@router.websocket("/ws/modules")
async def ws_modules(ws: WebSocket):
    """Client sends {"module_id", "params", "request_id"}; server replies with results (used for live parameter updates)."""
    user = await ws_user(ws)
    if user is None or "run" not in PERMISSIONS.get(user.role, set()):
        await ws.close(code=4401)
        return
    await ws.accept()
    async with get_sessionmaker()() as db:
        settings = await load_settings(db, user.id)
    try:
        while True:
            msg = json.loads(await ws.receive_text())
            if msg.get("type") == "settings_changed":
                async with get_sessionmaker()() as db:
                    settings = await load_settings(db, user.id)
                continue
            try:
                res = await execute(get_module(msg["module_id"]), msg.get("params") or {}, settings)
                await ws.send_text(json.dumps({"type": "result", "request_id": msg.get("request_id"), "result": res}, default=str))
            except Exception as exc:  # report validation errors to the client
                await ws.send_text(json.dumps({"type": "error", "request_id": msg.get("request_id"), "message": getattr(exc, "message", str(exc))}))
    except WebSocketDisconnect:
        return


@router.websocket("/ws/fine-tuning/{job_id}")
async def ws_job(ws: WebSocket, job_id: int):
    user = await ws_user(ws)
    if user is None:
        await ws.close(code=4401)
        return
    async with get_sessionmaker()() as db:
        job = await db.get(FineTuneJob, job_id)
    if job is None or job.user_id != user.id:
        await ws.close(code=4404)
        return
    await ws.accept()
    q = bus.subscribe(job_id)
    if job.status in ("completed", "failed", "cancelled") and not bus.history.get(job_id):
        await ws.send_text(json.dumps({"type": "status", "status": job.status, "metrics": job.metrics, "error": job.error}, default=str))
    try:
        while True:
            try:
                event = await asyncio.wait_for(q.get(), timeout=20)
            except asyncio.TimeoutError:
                await ws.send_text(json.dumps({"type": "ping"}))
                continue
            await ws.send_text(json.dumps(event, default=str))
            if event.get("type") == "status" and event.get("status") in ("completed", "failed", "cancelled"):
                break
    except WebSocketDisconnect:
        pass
    finally:
        bus.unsubscribe(job_id, q)
