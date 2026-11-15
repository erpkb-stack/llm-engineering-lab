"""Model provider abstraction: simulation (default), Ollama, OpenAI-compatible APIs, local Hugging Face.

Every response carries ``provider`` and ``measured`` so the UI can label real vs simulated output.
No provider is contacted unless the user selects it in Settings.
"""
from __future__ import annotations

import asyncio
import logging
import time
from functools import lru_cache
from typing import Any

import httpx

from app.core.config import get_settings
from app.core.errors import CapabilityUnavailable

logger = logging.getLogger("llmlab.providers")


class Provider:
    name = "base"
    real = False

    async def chat(self, messages: list[dict], **kw) -> dict:
        raise NotImplementedError

    async def generate(self, prompt: str, **kw) -> dict:
        return await self.chat([{"role": "user", "content": prompt}], **kw)

    async def status(self) -> dict:
        return {"provider": self.name, "available": True}


class SimulationProvider(Provider):
    name = "simulation"

    async def chat(self, messages, **kw) -> dict:
        last = messages[-1]["content"] if messages else ""
        return {"text": f"[simulation mode — no model configured] Received {len(last.split())} words.", "provider": self.name,
                "measured": False, "latency_ms": 0.0}


class OllamaProvider(Provider):
    name = "ollama"
    real = True

    def __init__(self, base_url: str, model: str, timeout: float):
        self.base_url, self.model, self.timeout = base_url.rstrip("/"), model, timeout

    async def chat(self, messages, temperature=0.7, top_p=1.0, top_k=0, max_tokens=256, seed=None, repetition_penalty=1.0, **_) -> dict:
        opts: dict[str, Any] = {"temperature": temperature, "top_p": top_p, "num_predict": max_tokens, "repeat_penalty": repetition_penalty}
        if top_k:
            opts["top_k"] = top_k
        if seed is not None:
            opts["seed"] = seed
        t0 = time.perf_counter()
        async with httpx.AsyncClient(timeout=self.timeout) as c:
            r = await c.post(f"{self.base_url}/api/chat", json={"model": self.model, "messages": messages, "stream": False, "options": opts})
            r.raise_for_status()
            data = r.json()
        return {"text": data.get("message", {}).get("content", ""), "provider": f"ollama:{self.model}", "measured": True,
                "latency_ms": (time.perf_counter() - t0) * 1e3, "eval_count": data.get("eval_count"),
                "eval_duration_ms": (data.get("eval_duration") or 0) / 1e6, "prompt_eval_count": data.get("prompt_eval_count")}

    async def status(self) -> dict:
        try:
            async with httpx.AsyncClient(timeout=3) as c:
                r = await c.get(f"{self.base_url}/api/tags")
                models = [m["name"] for m in r.json().get("models", [])]
            return {"provider": self.name, "available": True, "models": models, "model": self.model, "model_present": self.model in models}
        except Exception as exc:  # pragma: no cover - network dependent
            return {"provider": self.name, "available": False, "error": str(exc)[:200]}


class OpenAICompatibleProvider(Provider):
    name = "openai"
    real = True

    def __init__(self, base_url: str, api_key: str, model: str, timeout: float):
        self.base_url, self.api_key, self.model, self.timeout = base_url.rstrip("/"), api_key, model, timeout

    async def chat(self, messages, temperature=0.7, top_p=1.0, max_tokens=256, seed=None, **_) -> dict:
        body: dict[str, Any] = {"model": self.model, "messages": messages, "temperature": temperature, "top_p": top_p, "max_tokens": max_tokens}
        if seed is not None:
            body["seed"] = seed
        headers = {"Authorization": f"Bearer {self.api_key}"} if self.api_key else {}
        t0 = time.perf_counter()
        async with httpx.AsyncClient(timeout=self.timeout) as c:
            r = await c.post(f"{self.base_url}/chat/completions", json=body, headers=headers)
            r.raise_for_status()
            data = r.json()
        return {"text": data["choices"][0]["message"].get("content") or "", "provider": f"openai-compatible:{self.model}", "measured": True,
                "latency_ms": (time.perf_counter() - t0) * 1e3, "usage": data.get("usage")}

    async def status(self) -> dict:
        try:
            headers = {"Authorization": f"Bearer {self.api_key}"} if self.api_key else {}
            async with httpx.AsyncClient(timeout=3) as c:
                r = await c.get(f"{self.base_url}/models", headers=headers)
            return {"provider": self.name, "available": r.status_code < 400, "status_code": r.status_code, "model": self.model}
        except Exception as exc:  # pragma: no cover
            return {"provider": self.name, "available": False, "error": str(exc)[:200]}


@lru_cache(maxsize=2)
def _hf_pipeline(model_name: str):
    from transformers import AutoModelForCausalLM, AutoTokenizer  # type: ignore

    tok = AutoTokenizer.from_pretrained(model_name)
    model = AutoModelForCausalLM.from_pretrained(model_name)
    return tok, model


class HuggingFaceProvider(Provider):
    name = "huggingface"
    real = True

    def __init__(self, model: str):
        self.model = model

    async def chat(self, messages, temperature=0.7, top_p=1.0, top_k=0, max_tokens=64, seed=None, repetition_penalty=1.0, **_) -> dict:
        if not get_settings().enable_hf_models:
            raise CapabilityUnavailable("Set LLMLAB_ENABLE_HF_MODELS=true to allow local Hugging Face models (may download weights).")
        try:
            import torch  # type: ignore  # noqa: F401
        except ImportError as exc:
            raise CapabilityUnavailable("Install requirements-ml.txt (torch, transformers) for local Hugging Face inference.") from exc

        def run():
            import torch

            tok, model = _hf_pipeline(self.model)
            if getattr(tok, "chat_template", None):
                prompt = tok.apply_chat_template(messages, add_generation_prompt=True, tokenize=False)
            else:
                prompt = "\n".join(m["content"] for m in messages)
            ids = tok(prompt, return_tensors="pt", add_special_tokens=False)
            if seed is not None:
                torch.manual_seed(seed)
            t0 = time.perf_counter()
            out = model.generate(**ids, max_new_tokens=max_tokens, do_sample=temperature > 0, temperature=max(temperature, 1e-5),
                                 top_p=top_p, top_k=top_k or 0, repetition_penalty=repetition_penalty, pad_token_id=tok.eos_token_id)
            dt = (time.perf_counter() - t0) * 1e3
            text_ = tok.decode(out[0][ids["input_ids"].shape[1]:], skip_special_tokens=True)
            return {"text": text_, "provider": f"hf:{self.model}", "measured": True, "latency_ms": dt,
                    "new_tokens": int(out.shape[1] - ids["input_ids"].shape[1])}

        return await asyncio.to_thread(run)

    async def status(self) -> dict:
        try:
            import transformers  # type: ignore  # noqa: F401

            return {"provider": self.name, "available": get_settings().enable_hf_models, "model": self.model,
                    "note": "" if get_settings().enable_hf_models else "LLMLAB_ENABLE_HF_MODELS is false"}
        except ImportError:
            return {"provider": self.name, "available": False, "error": "transformers not installed"}


class ProviderHub:
    """Resolves the active provider from server config + the user's settings."""

    def __init__(self, user_settings: dict | None = None):
        s = get_settings()
        us = user_settings or {}
        choice = us.get("provider") or s.llm_provider
        if choice == "ollama":
            self.active: Provider = OllamaProvider(s.ollama_base_url, us.get("ollama_model") or s.ollama_model, s.provider_timeout_seconds)
        elif choice == "openai":
            self.active = OpenAICompatibleProvider(us.get("openai_base_url") or s.openai_base_url, s.openai_api_key,
                                                   us.get("openai_model") or s.openai_model, s.provider_timeout_seconds)
        elif choice == "huggingface":
            self.active = HuggingFaceProvider(s.hf_model)
        else:
            self.active = SimulationProvider()

    def has_real(self) -> bool:
        return self.active.real

    async def generate(self, prompt: str, **kw) -> dict:
        try:
            return await self.active.generate(prompt, **kw)
        except CapabilityUnavailable:
            raise
        except Exception as exc:
            logger.warning("provider_error", extra={"extra_fields": {"provider": self.active.name, "error": str(exc)[:200]}})
            return {"text": f"[provider error: {type(exc).__name__}: {str(exc)[:120]}]", "provider": self.active.name, "measured": False, "latency_ms": 0}

    async def chat(self, messages: list[dict], **kw) -> dict:
        try:
            return await self.active.chat(messages, **kw)
        except CapabilityUnavailable:
            raise
        except Exception as exc:
            return {"text": f"[provider error: {type(exc).__name__}: {str(exc)[:120]}]", "provider": self.active.name, "measured": False, "latency_ms": 0}

    async def status(self) -> dict:
        return await self.active.status()


_ENV_CACHE: dict | None = None


async def ml_environment_async() -> dict:
    """Non-blocking wrapper: importing torch/transformers can take many seconds, so probe once in a worker thread."""
    global _ENV_CACHE
    if _ENV_CACHE is None:
        _ENV_CACHE = await asyncio.to_thread(_probe_environment)
    return {**_ENV_CACHE, "hf_models_enabled": get_settings().enable_hf_models,
            "real_training_possible": bool(_ENV_CACHE.get("torch") and _ENV_CACHE.get("transformers") and _ENV_CACHE.get("peft") and get_settings().enable_hf_models)}


def ml_environment() -> dict:
    """Synchronous variant (cached after the first probe)."""
    global _ENV_CACHE
    if _ENV_CACHE is None:
        _ENV_CACHE = _probe_environment()
    return {**_ENV_CACHE, "hf_models_enabled": get_settings().enable_hf_models,
            "real_training_possible": bool(_ENV_CACHE.get("torch") and _ENV_CACHE.get("transformers") and _ENV_CACHE.get("peft") and get_settings().enable_hf_models)}


def _probe_environment() -> dict:
    """Report which optional ML libraries / accelerators are available (no downloads)."""
    info: dict[str, Any] = {}
    for mod in ("torch", "transformers", "tokenizers", "sentence_transformers", "peft", "bitsandbytes", "faiss", "chromadb", "opacus", "tiktoken", "trl", "datasets"):
        try:
            m = __import__(mod)
            info[mod] = getattr(m, "__version__", "installed")
        except Exception:
            info[mod] = None
    accel = "cpu"
    if info.get("torch"):
        try:
            import torch  # type: ignore

            accel = "cuda" if torch.cuda.is_available() else ("mps" if getattr(torch.backends, "mps", None) and torch.backends.mps.is_available() else "cpu")
        except Exception:
            pass
    info["accelerator"] = accel
    info["training_device"] = "cuda" if accel == "cuda" else "cpu"  # the lab's Trainer uses CUDA when present, otherwise CPU (MPS not used)
    return info
