"""Retrieval primitives: BM25, embedding models, vector index, fusion, IR metrics.

Embedding backends:
* ``hash``     — signed feature hashing of word uni/bi-grams + char trigrams (lexical; exact, fast).
* ``concept``  — bag-of-concepts using a small synonym lexicon (captures synonyms, blind to
                 word order and negation — like many real sentence embeddings in failure cases).
* ``minilm``   — real sentence-transformers model when installed and enabled (LLMLAB_ENABLE_HF_MODELS).
FAISS is used for the vector index when installed; otherwise an exact NumPy inner-product index.
"""
from __future__ import annotations

import hashlib
import json
import math
import re
from collections import Counter
from functools import lru_cache
from pathlib import Path

import numpy as np

from app.core.config import BASE_DIR, get_settings

WORD = re.compile(r"[a-z0-9]+(?:'[a-z]+)?")
STOP = set("a an the of to in for on and or is are was were be been it this that with as at by from i my me we you your do does can how what when which who whom will would should could".split())


def words(text: str) -> list[str]:
    return WORD.findall(text.lower())


def content_words(text: str) -> list[str]:
    return [w for w in words(text) if w not in STOP]


def stem(w: str) -> str:
    for suf in ("ing", "ies", "ed", "es", "s"):
        if len(w) > len(suf) + 2 and w.endswith(suf):
            return w[: -len(suf)] + ("y" if suf == "ies" else "")
    return w


# ------------------------------------------------------------------- BM25
class BM25:
    def __init__(self, docs: list[str], k1: float = 1.5, b: float = 0.75):
        self.k1, self.b = k1, b
        self.toks = [[stem(w) for w in content_words(d)] for d in docs]
        self.N = len(docs)
        self.avgdl = sum(map(len, self.toks)) / max(1, self.N)
        df: Counter = Counter()
        for t in self.toks:
            df.update(set(t))
        self.idf = {w: math.log(1 + (self.N - n + 0.5) / (n + 0.5)) for w, n in df.items()}
        self.tf = [Counter(t) for t in self.toks]

    def scores(self, query: str) -> np.ndarray:
        q = [stem(w) for w in content_words(query)]
        out = np.zeros(self.N)
        for i, tf in enumerate(self.tf):
            dl = len(self.toks[i])
            for w in q:
                if w in tf:
                    f = tf[w]
                    out[i] += self.idf[w] * f * (self.k1 + 1) / (f + self.k1 * (1 - self.b + self.b * dl / self.avgdl))
        return out


# ------------------------------------------------------------------- Embeddings
CONCEPTS = [
    {"vacation", "holiday", "holidays", "leave", "annual", "pto", "time-off", "days"},
    {"password", "credential", "credentials", "login", "log", "sign", "signin", "locked", "lockout", "account"},
    {"reset", "forgot", "forgotten", "recover", "unlock"},
    {"laptop", "computer", "device", "notebook", "hardware", "pc"},
    {"broken", "damaged", "faulty", "cracked", "replacement", "replace", "repair"},
    {"expense", "expenses", "reimbursement", "reimbursed", "reimburse", "receipt", "receipts", "claim"},
    {"travel", "travelling", "traveling", "trip", "flight", "flights", "hotel", "business"},
    {"money", "dollars", "cost", "costs", "budget", "price", "paid", "payment", "pay", "salary", "payroll"},
    {"chatgpt", "ai", "chatbot", "chatbots", "assistant", "llm", "external"},
    {"customer", "client", "confidential", "restricted", "data", "records", "financial"},
    {"remote", "home", "remotely", "wfh", "vpn"},
    {"baby", "parental", "maternity", "paternity", "caregiver", "caregivers", "fathers", "mothers", "parent"},
    {"phishing", "suspicious", "scam", "link", "email", "emails", "malicious"},
    {"incident", "breach", "leak", "lost", "security"},
    {"supplier", "suppliers", "vendor", "vendors", "invoice", "invoices"},
    {"bank", "banks", "banking", "loan", "loans", "deposit"},
    {"river", "water", "flood", "flooded", "shore"},
    {"office", "building", "open", "closed", "hours"},
    {"software", "install", "installing", "installation", "application", "app"},
    {"sick", "ill", "illness", "doctor", "medical"},
    {"safe", "safety", "secure", "risk", "dangerous", "harmful"},
    {"approve", "approved", "approval", "allow", "allowed", "permitted", "permission"},
    {"drug", "medication", "dose", "dosage", "patient", "aspirin", "ibuprofen"},
    {"dog", "dogs", "puppy", "cat", "cats", "pet", "pets"},
    {"bite", "bit", "attacked", "chased"},
    {"increase", "increased", "rise", "rose", "grew", "growth", "up"},
    {"decrease", "decreased", "fall", "fell", "drop", "dropped", "down", "decline"},
    {"revenue", "sales", "income", "earnings", "profit"},
    {"car", "automobile", "vehicle", "vehicles", "parking", "garage"},
]
_CONCEPT_OF = {w: i for i, grp in enumerate(CONCEPTS) for w in grp}


def _h(s: str, dim: int) -> tuple[int, float]:
    d = hashlib.blake2b(s.encode(), digest_size=8).digest()
    v = int.from_bytes(d, "little")
    return v % dim, (1.0 if (v >> 63) & 1 else -1.0)


def hash_embed(text: str, dim: int = 384) -> np.ndarray:
    v = np.zeros(dim)
    ws = content_words(text)
    feats = [f"w:{stem(w)}" for w in ws] + [f"b:{a}_{b}" for a, b in zip(ws, ws[1:])]
    t = f"  {text.lower()}  "
    feats += [f"c:{t[i:i + 3]}" for i in range(len(t) - 2)]
    for f in feats:
        i, s = _h(f, dim)
        v[i] += s * (2.0 if f.startswith("w:") else 0.5 if f.startswith("c:") else 1.0)
    n = np.linalg.norm(v)
    return v / n if n else v


def concept_embed(text: str, dim: int = 384) -> np.ndarray:
    v = np.zeros(dim)
    for w in content_words(text):
        key = _CONCEPT_OF.get(w, _CONCEPT_OF.get(stem(w)))
        token = f"concept:{key}" if key is not None else f"w:{stem(w)}"
        i, s = _h(token, dim)
        v[i] += s
    n = np.linalg.norm(v)
    return v / n if n else v


@lru_cache(maxsize=2)
def _st_model(name: str):
    from sentence_transformers import SentenceTransformer  # type: ignore

    return SentenceTransformer(name)


def real_embeddings_available() -> bool:
    if not get_settings().enable_hf_models:
        return False
    try:
        import sentence_transformers  # noqa: F401  # type: ignore

        return True
    except Exception:
        return False


def embed(texts: list[str], model: str = "hash", dim: int = 384) -> tuple[np.ndarray, str]:
    """Returns (L2-normalized matrix, backend-used)."""
    if model == "minilm":
        if real_embeddings_available():
            m = _st_model(get_settings().hf_embedding_model)
            return np.asarray(m.encode(texts, normalize_embeddings=True)), "real"
        model = "concept"
    fn = hash_embed if model == "hash" else concept_embed
    return np.vstack([fn(t, dim) for t in texts]) if texts else np.zeros((0, dim)), model


class VectorIndex:
    """Inner-product index over normalized vectors (FAISS if available, else exact NumPy)."""

    def __init__(self, vectors: np.ndarray):
        self.vectors = vectors.astype("float32")
        self.backend = "numpy"
        try:
            import faiss  # type: ignore

            self.index = faiss.IndexFlatIP(self.vectors.shape[1])
            self.index.add(self.vectors)
            self.backend = "faiss"
        except Exception:
            self.index = None

    def scores(self, q: np.ndarray) -> np.ndarray:
        return self.vectors @ q.astype("float32")


# ------------------------------------------------------------------- Fusion & metrics
def rrf(rankings: list[list[int]], k: int = 60) -> dict[int, float]:
    out: dict[int, float] = {}
    for ranking in rankings:
        for r, idx in enumerate(ranking):
            out[idx] = out.get(idx, 0.0) + 1.0 / (k + r + 1)
    return out


def minmax(x: np.ndarray) -> np.ndarray:
    lo, hi = float(np.min(x)), float(np.max(x))
    return (x - lo) / (hi - lo) if hi > lo else np.zeros_like(x)


def precision_at_k(ranked: list[str], rel: dict[str, int], k: int) -> float:
    return sum(1 for d in ranked[:k] if rel.get(d, 0) > 0) / k if k else 0.0


def recall_at_k(ranked: list[str], rel: dict[str, int], k: int) -> float:
    pos = [d for d, g in rel.items() if g > 0]
    return sum(1 for d in ranked[:k] if d in pos) / len(pos) if pos else 0.0


def mrr(ranked: list[str], rel: dict[str, int]) -> float:
    for i, d in enumerate(ranked):
        if rel.get(d, 0) > 0:
            return 1.0 / (i + 1)
    return 0.0


def ndcg_at_k(ranked: list[str], rel: dict[str, int], k: int) -> float:
    dcg = sum((2 ** rel.get(d, 0) - 1) / math.log2(i + 2) for i, d in enumerate(ranked[:k]))
    ideal = sorted(rel.values(), reverse=True)[:k]
    idcg = sum((2 ** g - 1) / math.log2(i + 2) for i, g in enumerate(ideal))
    return dcg / idcg if idcg else 0.0


@lru_cache(maxsize=1)
def load_kb() -> dict:
    path = Path(BASE_DIR) / "data" / "samples" / "kb.json"
    return json.loads(path.read_text())


def doc_text(d: dict) -> str:
    return f"{d['title']}. {d['text']}"
