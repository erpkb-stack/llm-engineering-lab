import os
import tempfile

import pytest

_tmp = tempfile.mkdtemp(prefix="llmlab-test-")
os.environ.update({
    "LLMLAB_ENVIRONMENT": "test",
    "LLMLAB_DATABASE_URL": f"sqlite+aiosqlite:///{_tmp}/test.db",
    "LLMLAB_SECRET_KEY": "test-secret-key-0123456789-abcdefghijklmnopqrstuvwxyz",
    "LLMLAB_LOG_JSON": "false",
    "LLMLAB_LOG_LEVEL": "WARNING",
    "LLMLAB_ADAPTERS_DIR": f"{_tmp}/adapters",
    "LLMLAB_LLM_PROVIDER": "simulation",
    "LLMLAB_ENABLE_HF_MODELS": "false",  # hermetic: never pick up a developer .env that enables real models
})

from fastapi.testclient import TestClient  # noqa: E402

from app.main import create_app  # noqa: E402

PASSWORD = "Demo@12345"


@pytest.fixture(scope="session")
def app():
    return create_app()


@pytest.fixture(scope="session")
def anon(app):
    with TestClient(app) as c:
        yield c


def _login(app, email):
    c = TestClient(app)
    c.__enter__()
    r = c.post("/api/auth/login", json={"email": email, "password": PASSWORD})
    assert r.status_code == 200, r.text
    c.headers["X-CSRF-Token"] = c.cookies.get("llmlab_csrf")
    return c


@pytest.fixture(scope="session")
def admin(app, anon):
    c = _login(app, "demo@llmlab.local")
    yield c
    c.__exit__(None, None, None)


@pytest.fixture(scope="session")
def viewer(app, anon):
    c = _login(app, "viewer@llmlab.local")
    yield c
    c.__exit__(None, None, None)


@pytest.fixture(scope="session")
def engineer(app, anon):
    c = _login(app, "engineer@llmlab.local")
    yield c
    c.__exit__(None, None, None)
