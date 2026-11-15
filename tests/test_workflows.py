"""Integration tests: experiments, configurations, benchmarks, fine-tuning lab, WebSockets."""
import json
import time

from fastapi.testclient import TestClient


def test_experiment_lifecycle_and_export(engineer):
    res = engineer.post("/api/modules/kv-cache/run", json={"params": {"batch": 2}}).json()
    e = engineer.post("/api/experiments", json={"module_id": "kv-cache", "name": "kv test", "params": res["params"], "result": res}).json()
    assert e["mode"] == "analytical"
    listed = engineer.get("/api/experiments?module_id=kv-cache").json()
    assert any(x["id"] == e["id"] for x in listed)
    full = engineer.get(f"/api/experiments/{e['id']}").json()
    assert full["result"]["metrics"]
    csv_ = engineer.get("/api/experiments/export?format=csv")
    assert csv_.status_code == 200 and "metric" in csv_.text.splitlines()[0] and "kv test" in csv_.text
    js = engineer.get("/api/experiments/export?format=json").json()
    assert any(x["id"] == e["id"] for x in js)
    assert engineer.delete(f"/api/experiments/{e['id']}").json()["ok"]
    assert engineer.get(f"/api/experiments/{e['id']}").status_code == 404


def test_experiments_are_private_per_user(engineer, admin):
    e = engineer.post("/api/experiments", json={"module_id": "lora", "result": {"metrics": []}}).json()
    assert admin.get(f"/api/experiments/{e['id']}").status_code == 404


def test_saved_configurations(engineer):
    c = engineer.post("/api/configs", json={"module_id": "lora", "name": "r=8", "params": {"rank": 8}}).json()
    assert c["params"]["rank"] == 8 and c["params"]["alpha"] == 32  # validated + defaults filled
    assert any(x["id"] == c["id"] for x in engineer.get("/api/configs?module_id=lora").json())
    assert engineer.post("/api/configs", json={"module_id": "lora", "name": "bad", "params": {"rank": 9999}}).status_code == 422
    assert engineer.delete(f"/api/configs/{c['id']}").json()["ok"]


def test_benchmark_compare_and_sweep(engineer):
    r = engineer.post("/api/benchmarks/compare", json={"module_id": "kv-cache", "baseline_params": {"enabled": False},
                                                        "variant_params": {"enabled": True}, "save": True}).json()
    assert r["changed"] == {"enabled": {"from": False, "to": True}}
    total = next(d for d in r["deltas"] if d["key"] == "total")
    assert total["variant"] < total["baseline"]
    assert r["experiment_id"]
    s = engineer.post("/api/benchmarks/sweep", json={"module_id": "lora", "params": {}, "sweep_key": "rank", "values": [4, 8, 16]}).json()
    assert [row["value"] for row in s["rows"]] == [4, 8, 16]
    m = engineer.post("/api/benchmarks/models", json={"models": ["llama-3-8b", "llama-3-70b"], "hardware": "a10g"}).json()
    fits = {row["model"]: row["fits"] for row in m["rows"]}
    assert fits == {"llama-3-8b": True, "llama-3-70b": False}
    live = engineer.post("/api/benchmarks/live", json={}).json()
    assert live["measured"] is False  # simulation mode never fabricates measurements


def test_dataset_validation(engineer):
    good = "\n".join(json.dumps({"prompt": f"q{i}", "completion": f"a{i}"}) for i in range(12)) + "\n{bad json\n" + json.dumps({"prompt": "", "completion": "x"})
    r = engineer.post("/api/fine-tuning/datasets/validate", files={"file": ("d.jsonl", good.encode(), "application/json")}).json()
    assert r["schema"] == "prompt_completion" and r["valid_records"] == 12 and r["invalid_records"] == 2 and not r["valid"]
    csv_ = "instruction,input,output\nSay hi,,hello\nSay bye,,bye\n"
    r = engineer.post("/api/fine-tuning/datasets/validate", files={"file": ("d.csv", csv_.encode(), "text/csv")}).json()
    assert r["schema"] == "alpaca" and r["valid_records"] == 2
    chat = json.dumps([{"messages": [{"role": "user", "content": "hi"}, {"role": "assistant", "content": "hello"}]}])
    assert engineer.post("/api/fine-tuning/datasets/validate", files={"file": ("d.json", chat.encode(), "application/json")}).json()["schema"] == "chat"
    bad = engineer.post("/api/fine-tuning/datasets/validate", files={"file": ("d.jsonl", b'{"foo": 1}', "application/json")})
    assert bad.status_code == 422
    assert engineer.post("/api/fine-tuning/datasets/validate", files={"file": ("d.txt", b"x", "text/plain")}).status_code == 422


def test_simulated_fine_tuning_job(engineer):
    ds = engineer.post("/api/fine-tuning/datasets/sample").json()
    cfg = {"dataset_id": ds["dataset_id"], "method": "lora", "mode": "simulation", "epochs": 2, "batch_size": 4}
    plan = engineer.post("/api/fine-tuning/plan", json=cfg).json()["plan"]
    assert 0 < plan["trainable_params"] < plan["total_params"]
    job = engineer.post("/api/fine-tuning/jobs", json=cfg).json()
    for _ in range(100):
        j = engineer.get(f"/api/fine-tuning/jobs/{job['id']}").json()
        if j["status"] in ("completed", "failed"):
            break
        time.sleep(0.1)
    assert j["status"] == "completed", j
    assert j["metrics"]["simulated"] is True and j["metrics"]["train_loss"]
    real = engineer.post("/api/fine-tuning/jobs", json={**cfg, "mode": "real"})
    assert real.status_code == 409  # real models disabled in the test env → explicit capability error, not a fake run


def test_websocket_module_run(app):
    with TestClient(app) as c:
        c.post("/api/auth/login", json={"email": "engineer@llmlab.local", "password": "Demo@12345"})
        with c.websocket_connect("/ws/modules") as ws:
            ws.send_text(json.dumps({"module_id": "rope-scaling", "params": {"method": "linear"}, "request_id": 7}))
            msg = json.loads(ws.receive_text())
            assert msg["type"] == "result" and msg["request_id"] == 7 and msg["result"]["params"]["method"] == "linear"
            ws.send_text(json.dumps({"module_id": "rope-scaling", "params": {"bogus": 1}, "request_id": 8}))
            assert json.loads(ws.receive_text())["type"] == "error"


def test_websocket_rejects_anonymous(app):
    import pytest
    from starlette.websockets import WebSocketDisconnect

    with TestClient(app) as c:
        with pytest.raises(WebSocketDisconnect):
            with c.websocket_connect("/ws/modules") as ws:
                ws.receive_text()
