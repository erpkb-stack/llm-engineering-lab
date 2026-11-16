"""LLM Engineering Lab — FastAPI application factory."""
from __future__ import annotations

import logging
import time
import uuid
from contextlib import asynccontextmanager
from pathlib import Path

from fastapi import FastAPI, Request
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import FileResponse, RedirectResponse
from fastapi.staticfiles import StaticFiles

from app.auth.router import router as auth_router
from app.benchmarking.router import router as bench_router
from app.core.config import BASE_DIR, ON_SERVERLESS, get_settings, session_key_configured
from app.core.errors import install_handlers
from app.core.logging import configure_logging, request_id_var
from app.dashboard.router import router as dashboard_router
from app.db.seed import create_tables, seed_users
from app.engine.registry import all_modules
from app.experiments.router import router as experiments_router
from app.fine_tuning.router import router as ft_router
from app.llm.router import router as llm_router
from app.modules_api.router import router as modules_router
from app.realtime.router import router as ws_router
from app.users.router import router as users_router

FRONTEND = Path(BASE_DIR) / "frontend"
logger = logging.getLogger("llmlab")

CSP = ("default-src 'self'; script-src 'self' https://cdn.jsdelivr.net; style-src 'self' 'unsafe-inline' https://cdn.jsdelivr.net; "
       "font-src 'self' data: https://cdn.jsdelivr.net; img-src 'self' data:; connect-src 'self' ws: wss: https://cdn.jsdelivr.net; "
       "worker-src 'self' blob:; frame-ancestors 'none'; base-uri 'self'; form-action 'self'")


@asynccontextmanager
async def lifespan(app: FastAPI):
    s = get_settings()
    configure_logging(s.log_level, s.log_json)
    if s.auto_create_tables:
        await create_tables()
    await seed_users()
    n = len(all_modules())
    logger.info("startup", extra={"extra_fields": {"environment": s.environment, "modules": n, "provider": s.llm_provider}})
    yield


def create_app() -> FastAPI:
    s = get_settings()
    app = FastAPI(
        title=s.app_name, version="1.0.0", lifespan=lifespan,
        description="Interactive LLM architecture, optimization, fine-tuning and inference playground. "
                    "Authenticate via POST /api/auth/login (use the returned `access_token` as a Bearer token in Swagger's Authorize dialog).",
    )
    install_handlers(app)
    if s.cors_origins:
        app.add_middleware(CORSMiddleware, allow_origins=s.cors_origins, allow_credentials=True, allow_methods=["*"], allow_headers=["*"])

    @app.middleware("http")
    async def request_context(request: Request, call_next):
        rid = request.headers.get("x-request-id") or uuid.uuid4().hex[:12]
        token = request_id_var.set(rid)
        t0 = time.perf_counter()
        try:
            response = await call_next(request)
        finally:
            request_id_var.reset(token)
        response.headers["X-Request-ID"] = rid
        response.headers["X-Content-Type-Options"] = "nosniff"
        response.headers["X-Frame-Options"] = "DENY"
        response.headers["Referrer-Policy"] = "strict-origin-when-cross-origin"
        if not request.url.path.startswith(("/docs", "/redoc", "/openapi")):
            response.headers["Content-Security-Policy"] = CSP
        if request.url.path.startswith("/api/"):
            response.headers["Cache-Control"] = "no-store"
            logging.getLogger("llmlab.http").info("request", extra={"extra_fields": {
                "method": request.method, "path": request.url.path, "status": response.status_code,
                "ms": round((time.perf_counter() - t0) * 1e3, 1), "request_id": rid}})
        return response

    for r in (auth_router, users_router, dashboard_router, modules_router, experiments_router, bench_router, ft_router, llm_router, ws_router):
        app.include_router(r)

    @app.get("/api/health", tags=["meta"])
    async def health():
        return {"status": "ok", "modules": len(all_modules()), "environment": s.environment,
                "serverless": ON_SERVERLESS, "session_key": "configured" if session_key_configured() else "missing"}

    app.mount("/static", StaticFiles(directory=FRONTEND), name="static")

    @app.get("/", include_in_schema=False)
    async def root(request: Request):
        return RedirectResponse("/app" if request.cookies.get(s.session_cookie_name) else "/login")

    @app.get("/login", include_in_schema=False)
    async def login_page():
        return FileResponse(FRONTEND / "login.html")

    @app.get("/app", include_in_schema=False)
    async def app_page():
        return FileResponse(FRONTEND / "app.html")

    return app


app = create_app()
