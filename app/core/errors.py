"""Centralized exception types and handlers."""
from __future__ import annotations

import logging

from fastapi import FastAPI, Request
from fastapi.exceptions import RequestValidationError
from fastapi.responses import JSONResponse
from starlette.exceptions import HTTPException as StarletteHTTPException

from app.core.logging import request_id_var

logger = logging.getLogger("llmlab.errors")


class AppError(Exception):
    status_code = 400
    code = "app_error"

    def __init__(self, message: str, *, details: dict | None = None):
        super().__init__(message)
        self.message = message
        self.details = details or {}


class NotFoundError(AppError):
    status_code = 404
    code = "not_found"


class PermissionDenied(AppError):
    status_code = 403
    code = "forbidden"


class AuthError(AppError):
    status_code = 401
    code = "unauthorized"


class ParameterError(AppError):
    status_code = 422
    code = "invalid_parameters"


class CapabilityUnavailable(AppError):
    """Raised when a real-model capability is requested but its dependency/backend is absent."""

    status_code = 409
    code = "capability_unavailable"


class ServerMisconfigured(AppError):
    """A deployment setting the app cannot work without is missing."""

    status_code = 503
    code = "server_misconfigured"


def _body(code: str, message: str, details=None) -> dict:
    return {"error": {"code": code, "message": message, "details": details or {}, "request_id": request_id_var.get()}}


def install_handlers(app: FastAPI) -> None:
    @app.exception_handler(AppError)
    async def _app_error(_: Request, exc: AppError):
        return JSONResponse(status_code=exc.status_code, content=_body(exc.code, exc.message, exc.details))

    @app.exception_handler(StarletteHTTPException)
    async def _http_error(_: Request, exc: StarletteHTTPException):
        return JSONResponse(status_code=exc.status_code, content=_body("http_error", str(exc.detail)), headers=getattr(exc, "headers", None))

    @app.exception_handler(RequestValidationError)
    async def _validation(_: Request, exc: RequestValidationError):
        errors = [{"loc": list(e.get("loc", [])), "msg": e.get("msg")} for e in exc.errors()]
        return JSONResponse(status_code=422, content=_body("validation_error", "Request validation failed", {"errors": errors}))

    @app.exception_handler(Exception)
    async def _unhandled(_: Request, exc: Exception):
        logger.exception("unhandled_error")
        return JSONResponse(status_code=500, content=_body("internal_error", "An unexpected error occurred"))
