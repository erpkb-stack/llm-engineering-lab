"""Opt-in integration tests against real models.

    LLMLAB_TEST_OLLAMA_MODEL=llama3.2:3b pytest tests/test_real_models.py      # needs a running Ollama
    LLMLAB_TEST_HF=1 pytest tests/test_real_models.py                          # needs requirements-ml.txt (downloads ~450 MB once)

Skipped by default so the suite runs anywhere without models.
"""
import asyncio
import os

import pytest

from app.engine.hardware import DEFAULT_SETTINGS
from app.engine.registry import get_module
from app.modules_api.router import execute

OLLAMA_MODEL = os.getenv("LLMLAB_TEST_OLLAMA_MODEL")
HF = os.getenv("LLMLAB_TEST_HF") == "1"


def run(module_id, params, settings):
    return asyncio.run(execute(get_module(module_id), params, settings))


@pytest.mark.skipif(not OLLAMA_MODEL, reason="set LLMLAB_TEST_OLLAMA_MODEL to run against Ollama")
def test_ollama_paths():
    s = {**DEFAULT_SETTINGS, "provider": "ollama", "ollama_model": OLLAMA_MODEL}
    r = run("temperature-sampling", {"use_real_model": True, "runs": 2}, s)
    real = next(t for t in r["tables"] if "Real" in t["title"])
    assert all(row[2].startswith("ollama:") for row in real["rows"])
    r = run("logic-bias", {"use_real_model": True, "task": "numerical", "test_cases": 3, "instruction_strength": 0.0}, s)
    assert r["fidelity"] == "real" and len(r["tables"][0]["rows"]) == 3
    r = run("prompt-engineering", {"use_real_model": True}, s)
    row = next(t for t in r["tables"] if "Real" in t["title"])["rows"][0]
    assert "JSON" in row[1] and ("followed" in row[2] or "ignored" in row[2])


@pytest.mark.skipif(not HF, reason="set LLMLAB_TEST_HF=1 (requires requirements-ml.txt)")
def test_hf_embeddings_and_reranker(monkeypatch):
    from app.core.config import get_settings

    monkeypatch.setattr(get_settings(), "enable_hf_models", True)
    r = run("cosine-trap", {"embedding_model": "minilm"}, DEFAULT_SETTINGS)
    assert r["fidelity"] == "real"
    r = run("cross-encoder-reranking", {}, DEFAULT_SETTINGS)
    assert r["fidelity"] == "real"
    before, after = r["comparison"]["rows"][0]["before"], r["comparison"]["rows"][0]["after"]
    assert after > before  # a real ms-marco cross-encoder improves nDCG on the labelled KB
