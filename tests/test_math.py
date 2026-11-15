"""Unit tests that pin the technical correctness of the simulation engine."""
import math

import numpy as np

from app.attention.modules import rope_freqs
from app.engine.hardware import MODELS, kv_cache_bytes
from app.evaluation.logic_cases import CASES
from app.fine_tuning.modules import lora_params, module_shapes
from app.inference.modules import quantize
from app.privacy.accountant import epsilon
from app.privacy.modules import find_pii, luhn_ok
from app.retrieval.core import BM25, ndcg_at_k, precision_at_k, recall_at_k, rrf
from app.tokenization.bpe import ByteLevelBPE


def test_parameter_counts_match_published_sizes():
    assert abs(MODELS["llama-2-7b"].total_params() / 1e9 - 6.74) < 0.02
    assert abs(MODELS["llama-3-8b"].total_params() / 1e9 - 8.03) < 0.02
    assert abs(MODELS["mixtral-8x7b"].total_params() / 1e9 - 46.7) < 0.2
    assert abs(MODELS["mixtral-8x7b"].active_params() / 1e9 - 12.9) < 0.1
    assert abs(MODELS["llama-3-70b"].total_params() / 1e9 - 70.6) < 0.2


def test_kv_cache_formula():
    a = MODELS["llama-3-8b"]  # 32 layers, 8 KV heads, head_dim 128 → 128 KiB/token in BF16
    assert kv_cache_bytes(a, 1, 1, "bf16") == 2 * 32 * 8 * 128 * 2 == 131072
    assert kv_cache_bytes(MODELS["llama-2-7b"], 1, 1, "bf16") == 524288  # MHA: 512 KiB/token


def test_dp_accountant_matches_tf_privacy_reference():
    # TF-Privacy MNIST reference: N=60000, batch 256, σ=1.1, 60 epochs, δ=1e-5 → ε ≈ 3.01
    eps, _ = epsilon(256 / 60000, 1.1, int(60 * 60000 / 256), 1e-5)
    assert abs(eps - 3.01) < 0.05
    assert epsilon(0.01, 2.0, 1000, 1e-5)[0] < epsilon(0.01, 1.0, 1000, 1e-5)[0]  # more noise → less privacy loss


def test_ir_metrics():
    rel = {"a": 2, "b": 1}
    assert ndcg_at_k(["a", "b", "c"], rel, 3) == 1.0
    assert ndcg_at_k(["c", "b", "a"], rel, 3) < 1.0
    assert precision_at_k(["a", "c"], rel, 2) == 0.5 and recall_at_k(["a", "c"], rel, 2) == 0.5
    fused = rrf([[0, 1, 2], [2, 1, 0]], k=60)
    assert fused[1] == 2 / 62


def test_bm25_prefers_matching_document():
    bm = BM25(["the cat sat on the mat", "dogs chase cars", "a cat and a dog"])
    s = bm.scores("cat mat")
    assert int(np.argmax(s)) == 0 and s[1] == 0


def test_bpe_is_lossless_and_language_weighting_matters():
    tok_en = ByteLevelBPE(800, 0.0)
    tok_ml = ByteLevelBPE(800, 1.0)
    text = "कृपया मेरा पासवर्ड रीसेट करें। Please reset my password."
    assert "".join(t for t, _ in tok_en.encode("Please reset my password.")) == "Please reset my password."
    hindi = "कृपया मेरा पासवर्ड रीसेट करें"
    assert len(tok_ml.encode(hindi)) < len(tok_en.encode(hindi))
    assert len(tok_en.encode(text)) > 0


def test_nf4_beats_uniform_int4_on_gaussian_weights():
    W = np.random.default_rng(0).normal(0, 1, (128, 256)).astype(np.float32)
    e_nf4 = np.sqrt(((W - quantize(W, "nf4", "group", 64)[0]) ** 2).mean())
    e_int4 = np.sqrt(((W - quantize(W, "int4", "group", 64)[0]) ** 2).mean())
    e_int8 = np.sqrt(((W - quantize(W, "int8", "group", 64)[0]) ** 2).mean())
    assert e_int8 < e_nf4 < e_int4


def test_lora_param_count():
    a = MODELS["llama-2-7b"]
    # q,v on 4096×4096 with r=8: 32 layers × 2 × 8 × (4096+4096) = 4,194,304 (matches PEFT's print_trainable_parameters)
    assert lora_params(a, ["q_proj", "v_proj"], 8) == 4_194_304
    assert module_shapes(a)["down_proj"] == (11008, 4096)


def test_rope_scaling_methods():
    th, _ = rope_freqs(128, 10000, "none", 1, 4096)
    lin, _ = rope_freqs(128, 10000, "linear", 4, 4096)
    ntk, _ = rope_freqs(128, 10000, "ntk", 4, 4096)
    yarn, m = rope_freqs(128, 10000, "yarn", 4, 4096)
    assert np.allclose(lin, th / 4)
    assert math.isclose(ntk[0], th[0]) and ntk[-1] < th[-1]  # NTK keeps high freq, stretches low freq
    assert math.isclose(yarn[0], th[0]) and math.isclose(yarn[-1], th[-1] / 4) and m > 1


def test_speculative_expected_tokens():
    from app.inference.modules import spec_sim
    from app.engine.hardware import DEFAULT_SETTINGS
    from app.engine.registry import Context, get_module

    mod = get_module("speculative-decoding")
    res = spec_sim(mod.validate({"acceptance": 0.8, "draft_tokens": 4}), Context(settings=DEFAULT_SETTINGS)).data
    m = {x["key"]: x["value"] for x in res["metrics"]}
    assert math.isclose(m["tokens_per_iter"], (1 - 0.8 ** 5) / (1 - 0.8), rel_tol=1e-6)


def test_pii_detection_and_luhn():
    assert luhn_ok("4111 1111 1111 1111") and not luhn_ok("4111 1111 1111 1112")
    found = {f["type"] for f in find_pii("mail a.b@x.io, card 4111-1111-1111-1111, ssn 123-45-6789, ip 10.0.0.1", ["email", "credit_card", "ssn", "ip_address", "phone"])}
    assert found == {"email", "credit_card", "ssn", "ip_address"}


def test_logic_verifiers_compute_expected_answers():
    answers = {c["id"]: c["verify"]() for c in CASES}
    assert answers["cond-1"] == "no" and answers["cond-2"] == "no" and answers["contra-1"] == "no"
    assert answers["num-1"] == "5" and answers["num-2"] == "no" and answers["num-3"] == "408"
    assert answers["multi-2"] == "180" and answers["multi-3"] == "17:15" and answers["neg-1"] == "9"
