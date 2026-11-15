from fastapi.testclient import TestClient

from app.core.security import hash_password, password_problems, verify_password


def test_health(anon):
    r = anon.get("/api/health")
    assert r.status_code == 200 and r.json()["modules"] >= 27


def test_password_hashing_is_argon2():
    h = hash_password("Correct@Horse1")
    assert h.startswith("$argon2id$")
    assert verify_password("Correct@Horse1", h) and not verify_password("wrong", h)


def test_password_policy():
    assert password_problems("Demo@12345") == []
    assert set(password_problems("short")) >= {"at least 10 characters", "an uppercase letter", "a digit", "a symbol"}


def test_protected_endpoints_require_auth(app):
    with TestClient(app) as c:
        for path in ("/api/auth/me", "/api/modules", "/api/dashboard/summary", "/api/experiments", "/api/settings"):
            assert c.get(path).status_code == 401, path


def test_login_sets_secure_cookies(app):
    with TestClient(app) as c:
        r = c.post("/api/auth/login", json={"email": "demo@llmlab.local", "password": "Demo@12345", "remember_me": True})
        assert r.status_code == 200
        body = r.json()
        assert body["user"]["role"] == "admin" and body["access_token"]
        set_cookie = ",".join(r.headers.get_list("set-cookie"))
        assert "llmlab_session=" in set_cookie and "HttpOnly" in set_cookie and "SameSite=lax" in set_cookie
        assert c.get("/api/auth/me").json()["email"] == "demo@llmlab.local"


def test_wrong_password_and_lockout(app):
    with TestClient(app) as c:
        for _ in range(5):
            assert c.post("/api/auth/login", json={"email": "researcher@llmlab.local", "password": "nope"}).status_code == 401
        r = c.post("/api/auth/login", json={"email": "researcher@llmlab.local", "password": "Demo@12345"})
        assert r.status_code == 401 and "Too many" in r.json()["error"]["message"]


def test_csrf_required_for_cookie_auth(app):
    with TestClient(app) as c:
        c.post("/api/auth/login", json={"email": "engineer@llmlab.local", "password": "Demo@12345"})
        r = c.post("/api/modules/kv-cache/run", json={"params": {}})
        assert r.status_code == 403 and "CSRF" in r.json()["error"]["message"]
        r = c.post("/api/modules/kv-cache/run", json={"params": {}}, headers={"X-CSRF-Token": c.cookies.get("llmlab_csrf")})
        assert r.status_code == 200


def test_bearer_token_without_csrf(app):
    with TestClient(app) as c:
        tok = c.post("/api/auth/login", json={"email": "engineer@llmlab.local", "password": "Demo@12345"}).json()["access_token"]
    with TestClient(app) as c2:
        r = c2.post("/api/modules/kv-cache/run", json={"params": {}}, headers={"Authorization": f"Bearer {tok}"})
        assert r.status_code == 200
        assert c2.get("/api/auth/me", headers={"Authorization": "Bearer garbage"}).status_code == 401


def test_logout_clears_session(app):
    with TestClient(app) as c:
        c.post("/api/auth/login", json={"email": "engineer@llmlab.local", "password": "Demo@12345"})
        r = c.post("/api/auth/logout", headers={"X-CSRF-Token": c.cookies.get("llmlab_csrf")})
        assert r.status_code == 200
        assert c.get("/api/auth/me").status_code == 401


def test_rbac_viewer(viewer):
    assert viewer.post("/api/modules/kv-cache/run", json={"params": {}}).status_code == 200  # viewers can run
    assert viewer.post("/api/experiments", json={"module_id": "kv-cache", "result": {}}).status_code == 403
    assert viewer.post("/api/benchmarks/compare", json={"module_id": "kv-cache"}).status_code == 403
    assert viewer.post("/api/fine-tuning/datasets/sample").status_code == 403
    assert viewer.put("/api/settings", json={"hardware": "h100"}).status_code == 403
    assert viewer.get("/api/users").status_code == 403


def test_rbac_admin_user_management(admin, engineer):
    assert engineer.get("/api/users").status_code == 403
    users = admin.get("/api/users").json()
    assert {u["role"] for u in users} >= {"admin", "engineer", "researcher", "viewer"}
    r = admin.post("/api/users", json={"email": "weak@llmlab.local", "password": "weak", "role": "viewer"})
    assert r.status_code == 422
    r = admin.post("/api/users", json={"email": "new.user@llmlab.local", "password": "Str0ng@Passw0rd", "role": "researcher"})
    assert r.status_code == 201 and r.json()["role"] == "researcher"
    uid = r.json()["id"]
    assert admin.patch(f"/api/users/{uid}", json={"role": "engineer"}).json()["role"] == "engineer"
    assert any(a["action"] == "user_created" for a in admin.get("/api/audit").json())


def test_security_headers(admin):
    r = admin.get("/app")
    assert "default-src 'self'" in r.headers["content-security-policy"]
    assert r.headers["x-frame-options"] == "DENY" and r.headers["x-content-type-options"] == "nosniff"


def test_settings_roundtrip(engineer):
    r = engineer.put("/api/settings", json={"hardware": "h100", "active_model": "mistral-7b", "memory_efficiency": 5, "unknown": 1})
    s = r.json()["settings"]
    assert s["hardware"] == "h100" and s["active_model"] == "mistral-7b" and s["memory_efficiency"] == 1.0 and "unknown" not in s
    assert engineer.get("/api/dashboard/summary").json()["active_model"]["key"] == "mistral-7b"
