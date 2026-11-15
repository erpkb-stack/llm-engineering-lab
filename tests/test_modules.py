import json

import pytest

from app.engine.registry import all_modules

ORIGINAL = {1: "kv-cache", 2: "tokenizer-bias", 3: "temperature-sampling", 4: "cosine-trap", 5: "prefix-caching", 6: "cross-encoder-reranking",
            7: "lora", 8: "logic-bias", 9: "speculative-decoding", 10: "rope-scaling", 11: "attention-sink", 12: "mixture-of-experts",
            13: "confirmation-prediction", 14: "ring-attention", 15: "medusa-heads", 16: "differential-privacy"}
ESSENTIAL = {"quantization", "qlora", "rag-optimization", "hybrid-search", "prompt-engineering", "model-evaluation", "model-serving",
             "paged-attention", "continuous-batching", "ai-security-governance", "attention-mechanisms"}
DOC_KEYS = {"what", "why", "how", "when_use", "when_avoid", "pros", "cons", "use_case", "code", "diagram", "performance", "checklist", "applications"}
MODULE_IDS = [m.id for m in all_modules()]


def test_catalog_contains_all_required_modules(admin):
    data = admin.get("/api/modules").json()
    mods = {m["id"]: m for m in data["modules"]}
    for idx, mid in ORIGINAL.items():
        assert mods[mid]["original_index"] == idx
    assert ESSENTIAL <= set(mods)
    assert mods["logic-bias"]["experimental"] and mods["confirmation-prediction"]["experimental"]


@pytest.mark.parametrize("module_id", MODULE_IDS)
def test_module_contract_and_defaults(admin, module_id):
    detail = admin.get(f"/api/modules/{module_id}").json()
    assert DOC_KEYS <= set(detail["docs"]), "every module documents the 12 sections"
    for c in detail["controls"]:
        assert c["help"], f"{c['key']} needs a tooltip"
        if c["type"] == "slider":
            assert c["min"] <= c["default"] <= c["max"] and c["step"] > 0
    r = admin.post(f"/api/modules/{module_id}/run", json={"params": {}})
    assert r.status_code == 200, r.text
    res = r.json()
    assert res["fidelity"] in {"analytical", "computed", "simulated", "real"} and res["fidelity_note"]
    assert res["metrics"], "every module reports metrics"
    assert res["comparison"] and res["comparison"]["rows"], "every module provides a before/after comparison"
    assert res["recommendations"]
    json.dumps(res)


@pytest.mark.parametrize("module_id", ["kv-cache", "cosine-trap", "differential-privacy", "mixture-of-experts", "prefix-caching"])
def test_modules_are_deterministic(admin, module_id):
    a = admin.post(f"/api/modules/{module_id}/run", json={"params": {}}).json()["metrics"]
    b = admin.post(f"/api/modules/{module_id}/run", json={"params": {}}).json()["metrics"]
    assert a == b


def test_controls_affect_output(admin):
    """Every slider must change the demonstration (no dead controls) — spot-check on KV cache."""
    base = {m["key"]: m["value"] for m in admin.post("/api/modules/kv-cache/run", json={"params": {}}).json()["metrics"]}
    for key, val in {"seq_len": 8192, "layers": 16, "kv_heads": 32, "head_dim": 64, "batch": 4, "dtype": "fp32", "gen_tokens": 64}.items():
        m = {x["key"]: x["value"] for x in admin.post("/api/modules/kv-cache/run", json={"params": {key: val}}).json()["metrics"]}
        assert m != base, key
    off = {x["key"]: x["value"] for x in admin.post("/api/modules/kv-cache/run", json={"params": {"enabled": False}}).json()["metrics"]}
    assert off["kv_total"] == 0 and off["total"] > base["total"]


def test_parameter_validation(admin):
    assert admin.post("/api/modules/kv-cache/run", json={"params": {"seq_len": 10 ** 9}}).status_code == 422
    assert admin.post("/api/modules/kv-cache/run", json={"params": {"bogus": 1}}).status_code == 422
    assert admin.post("/api/modules/kv-cache/run", json={"params": {"dtype": "fp7"}}).status_code == 422
    assert admin.post("/api/modules/nope/run", json={"params": {}}).status_code == 404


def test_presets_apply(admin):
    detail = admin.get("/api/modules/cosine-trap").json()
    assert len(detail["presets"]) >= 3
    for p in detail["presets"]:
        assert admin.post("/api/modules/cosine-trap/run", json={"params": p["params"]}).status_code == 200


def test_applications_cover_all_domains(admin):
    apps = admin.get("/api/modules/applications").json()
    assert len(apps) == 9 and all(len(v) >= 1 for v in apps.values())
