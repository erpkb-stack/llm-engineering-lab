"""Cross-encoder relevance scoring.

Uses a real ``sentence_transformers.CrossEncoder`` when installed and enabled; otherwise a
deterministic *simulated* cross-encoder: a rule-based joint (query, document) scorer that —
like a real cross-encoder and unlike a bi-encoder — looks at both texts together (term
coverage with synonym matching, negation and number consistency, answer-type cues).
"""
from __future__ import annotations

import math
import re
from functools import lru_cache

from app.core.config import get_settings
from app.retrieval.core import _CONCEPT_OF, content_words, stem

NEG = {"not", "no", "never", "cannot", "can't", "without", "don't", "doesn't", "isn't"}
NUM = re.compile(r"\d+(?:\.\d+)?")
QUANT_CUES = ("how many", "how much", "limit", "when", "how long", "what is the")


@lru_cache(maxsize=1)
def _real_model():
    from sentence_transformers import CrossEncoder  # type: ignore

    return CrossEncoder(get_settings().hf_reranker_model)


def real_reranker_available() -> bool:
    if not get_settings().enable_hf_models:
        return False
    try:
        import sentence_transformers  # noqa: F401  # type: ignore

        return True
    except Exception:
        return False


def _match(qw: str, dwords: set[str], dstems: set[str], dconcepts: set[int]) -> float:
    if qw in dwords or stem(qw) in dstems:
        return 1.0
    c = _CONCEPT_OF.get(qw, _CONCEPT_OF.get(stem(qw)))
    return 0.75 if c is not None and c in dconcepts else 0.0


def simulated_cross_score(query: str, doc: str) -> float:
    qws = content_words(query)
    dws = content_words(doc)
    if not qws or not dws:
        return 0.0
    dset, dstems = set(dws), {stem(w) for w in dws}
    dconcepts = {_CONCEPT_OF[w] for w in dws if w in _CONCEPT_OF} | {_CONCEPT_OF[stem(w)] for w in dws if stem(w) in _CONCEPT_OF}
    weights = [1.0 + min(len(w), 10) / 10 for w in qws]  # longer words ≈ more informative
    cov = sum(wt * _match(w, dset, dstems, dconcepts) for w, wt in zip(qws, weights)) / sum(weights)
    score = 1 / (1 + math.exp(-8 * (cov - 0.45)))
    qneg = any(w in NEG for w in query.lower().split())
    dneg = any(w in NEG for w in doc.lower().split())
    if qneg != dneg and cov > 0.5:
        score *= 0.85
    qn, dn = set(NUM.findall(query)), set(NUM.findall(doc))
    if qn and dn and not (qn & dn):
        score *= 0.6
    if any(c in query.lower() for c in QUANT_CUES) and dn:
        score = min(1.0, score + 0.08)
    return round(float(score), 4)


def cross_scores(query: str, docs: list[str]) -> tuple[list[float], str]:
    if real_reranker_available():
        m = _real_model()
        raw = m.predict([(query, d) for d in docs])
        return [float(1 / (1 + math.exp(-x))) for x in raw], "real"
    return [simulated_cross_score(query, d) for d in docs], "simulated"


RERANKERS = {
    "minilm-l6": {"label": "ms-marco-MiniLM-L-6 (22M)", "params": 22e6},
    "minilm-l12": {"label": "ms-marco-MiniLM-L-12 (33M)", "params": 33e6},
    "bge-base": {"label": "bge-reranker-base (278M)", "params": 278e6},
    "bge-large": {"label": "bge-reranker-large (560M)", "params": 560e6},
}


def rerank_latency_ms(model: str, k: int, tokens_per_pair: int, device: str) -> float:
    flops = 2 * RERANKERS[model]["params"] * tokens_per_pair * k
    eff = 0.5e12 if device == "cpu" else 20e12  # assumed effective FLOP/s: modern multi-core CPU vs datacenter GPU (fp16)
    return flops / eff * 1e3 + (2.0 if device == "gpu" else 0.5)
