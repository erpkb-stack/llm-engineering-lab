"""Environment-based application configuration.

All settings can be overridden with environment variables prefixed ``LLMLAB_``
(or via a ``.env`` file in the working directory). See ``.env.example``.
"""
from __future__ import annotations

import logging
import os
import secrets
from functools import lru_cache
from pathlib import Path
from typing import Literal

from pydantic import Field, model_validator
from pydantic_settings import BaseSettings, SettingsConfigDict

BASE_DIR = Path(__file__).resolve().parents[2]

# Serverless platforms (Vercel, AWS Lambda) mount the code read-only; only /tmp is writable.
ON_SERVERLESS = bool(os.environ.get("VERCEL") or os.environ.get("AWS_LAMBDA_FUNCTION_NAME"))
# Where the app writes at runtime (SQLite file, uploads, adapters). Bundled read-only data
# such as data/samples/ always stays under BASE_DIR.


def session_key_configured() -> bool:
    """False on serverless when no fixed LLMLAB_SECRET_KEY is set: each instance would sign sessions with its own key."""
    return not ON_SERVERLESS or bool(os.environ.get("LLMLAB_SECRET_KEY", "").strip())


RUNTIME_DIR = Path(os.environ.get("LLMLAB_RUNTIME_DIR") or ("/tmp/llmlab" if ON_SERVERLESS else BASE_DIR / "data"))


class Settings(BaseSettings):
    model_config = SettingsConfigDict(env_file=".env", env_prefix="LLMLAB_", extra="ignore")

    app_name: str = "LLM Engineering Lab"
    environment: Literal["development", "test", "production"] = "development"
    log_level: str = "INFO"
    log_json: bool = True

    # --- Security -----------------------------------------------------------
    # In development a random key is generated per process (sessions reset on restart).
    # In production the key MUST be supplied explicitly (validated below).
    secret_key: str = Field(default_factory=lambda: secrets.token_urlsafe(48))
    jwt_algorithm: str = "HS256"
    access_token_minutes: int = 60
    remember_me_days: int = 14
    cookie_secure: bool = ON_SERVERLESS  # Vercel always serves HTTPS
    cookie_samesite: Literal["lax", "strict", "none"] = "lax"
    session_cookie_name: str = "llmlab_session"
    csrf_cookie_name: str = "llmlab_csrf"
    login_max_attempts: int = 5
    login_lockout_seconds: int = 300
    cors_origins: list[str] = []

    # --- Database -----------------------------------------------------------
    database_url: str = f"sqlite+aiosqlite:///{RUNTIME_DIR / 'llmlab.db'}"
    auto_create_tables: bool = True

    # --- Demo account -------------------------------------------------------
    demo_user_email: str = "demo@llmlab.local"
    demo_user_password: str = "Demo@12345"
    demo_user_role: Literal["admin", "engineer", "researcher", "viewer"] = "admin"
    seed_role_users: bool = True  # also seeds engineer@/researcher@/viewer@llmlab.local

    # --- Model providers ----------------------------------------------------
    llm_provider: Literal["simulation", "ollama", "openai", "huggingface"] = "simulation"
    ollama_base_url: str = "http://localhost:11434"
    ollama_model: str = "llama3.2:1b"
    openai_base_url: str = "http://localhost:8000/v1"
    openai_api_key: str = ""
    openai_model: str = "gpt-4o-mini"
    hf_model: str = "HuggingFaceTB/SmolLM2-135M-Instruct"  # small, genuinely pretrained; trains on CPU
    hf_embedding_model: str = "sentence-transformers/all-MiniLM-L6-v2"
    hf_reranker_model: str = "cross-encoder/ms-marco-MiniLM-L-6-v2"
    enable_hf_models: bool = False  # opt-in: may download weights on first use
    provider_timeout_seconds: float = 120.0  # local CPU models can take ~30 s to cold-load

    # --- Fine-tuning --------------------------------------------------------
    upload_max_bytes: int = 10 * 1024 * 1024
    adapters_dir: str = str(RUNTIME_DIR / "adapters")

    @model_validator(mode="after")
    def _production_guards(self) -> "Settings":
        if not session_key_configured():
            # Each serverless instance would generate its own key, so sessions signed on one
            # instance are rejected by the next (random logouts). Warn loudly instead of crashing.
            logging.getLogger("llmlab").warning("LLMLAB_SECRET_KEY is not set: logins will not survive across serverless instances")
        if not self.secret_key:
            if self.environment == "production":
                raise ValueError("LLMLAB_SECRET_KEY must be set in production")
            self.secret_key = secrets.token_urlsafe(48)  # dev fallback for an empty value in .env
        if self.environment == "production":
            if len(self.secret_key) < 32:
                raise ValueError("LLMLAB_SECRET_KEY must be set (>=32 chars) in production")
            if not self.cookie_secure:
                raise ValueError("LLMLAB_COOKIE_SECURE must be true in production")
            if self.demo_user_password == "Demo@12345":
                raise ValueError("Change LLMLAB_DEMO_USER_PASSWORD in production")
        return self


@lru_cache
def get_settings() -> Settings:
    return Settings()
