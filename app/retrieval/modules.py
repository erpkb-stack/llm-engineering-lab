"""Module 4 (Cosine Similarity Trap), Hybrid Search, RAG Optimization."""
from __future__ import annotations

import re
from functools import lru_cache

import numpy as np

from app.engine.registry import Context, ModuleDef, Result, docs, register, select, slider, text, toggle
from app.reranking.scorer import simulated_cross_score
from app.retrieval.core import (BM25, VectorIndex, content_words, doc_text, embed, hash_embed, load_kb, minmax, mrr,
                                ndcg_at_k, precision_at_k, recall_at_k, rrf, stem)

EMBEDDERS = [{"value": "hash", "label": "Hashed n-gram (lexical)"}, {"value": "concept", "label": "Bag-of-concepts (synonym-aware)"},
             {"value": "minilm", "label": "all-MiniLM-L6-v2 (real, if installed)"}]
NEG = {"not", "no", "never", "cannot", "without", "isn't", "don't"}
ANTONYMS = [("increase", "decrease"), ("increased", "decreased"), ("safe", "unsafe"), ("approved", "rejected"), ("rose", "fell"),
            ("before", "after"), ("allowed", "forbidden"), ("above", "below"), ("more", "less")]

COSINE_PRESETS = [
    {"name": "Medication safety (negation)", "params": {"query": "Is it safe to take ibuprofen with aspirin?", "candidates":
        "+ Taking ibuprofen together with aspirin can reduce aspirin's heart-protective effect and raise bleeding risk; ask a doctor first.\n"
        "It is safe to take ibuprofen with aspirin.\nIt is not safe to take ibuprofen with aspirin.\n"
        "It is safe to take ibuprofen with food.\nAspirin is not safe for dogs.\nIbuprofen and aspirin price comparison at local pharmacies."}},
    {"name": "Role reversal", "params": {"query": "The dog bit the man.", "candidates":
        "The man bit the dog.\nThe dog did not bite the man.\n+ A man was attacked and bitten by a dog.\nDogs and men are often friends.\n+ The man was bitten by the dog."}},
    {"name": "Polysemy: bank", "params": {"query": "How do I open a savings account at the bank?", "candidates":
        "+ To open a savings account, bring photo ID to any bank branch or apply online.\nThe river bank flooded after heavy rain.\n"
        "Bank holidays calendar for 2026.\nA bank of servers powers the data center.\n+ Savings accounts can be opened online in ten minutes."}},
    {"name": "Numbers & dates", "params": {"query": "Revenue increased 12% in Q3 2024", "candidates":
        "Revenue decreased 12% in Q3 2024\nRevenue increased 12% in Q3 2023\nRevenue increased 21% in Q3 2024\n"
        "+ Q3 2024 sales grew by 12 percent year over year\nCosts increased 12% in Q3 2024"}},
]


def parse_candidates(raw: str) -> list[tuple[str, bool]]:
    out = []
    for line in raw.splitlines():
        line = line.strip()
        if not line:
            continue
        rel = line.startswith("+")
        out.append((line.lstrip("+").strip(), rel))
    return out


def trap_reasons(q: str, d: str) -> list[str]:
    qw, dw = q.lower().split(), d.lower().split()
    reasons = []
    if any(w.strip(".,?") in NEG for w in qw) != any(w.strip(".,?") in NEG for w in dw):
        reasons.append("negation flips meaning")
    qs, ds = set(content_words(q)), set(content_words(d))
    if qs == ds and [w for w in content_words(q)] != [w for w in content_words(d)]:
        reasons.append("same words, different order/roles")
    qn, dn = set(re.findall(r"\d+", q)), set(re.findall(r"\d+", d))
    if qn and dn and qn != dn:
        reasons.append("numbers/dates differ")
    for a, b in ANTONYMS:
        if (a in qw and b in [x.strip(".,") for x in dw]) or (b in qw and a in [x.strip(".,") for x in dw]):
            reasons.append(f"antonym ({a}/{b})")
    if len(qs & ds) / max(1, len(qs)) > 0.5 and not reasons:
        reasons.append("high lexical overlap")
    return reasons


def cosine_sim(p: dict, ctx: Context) -> Result:
    cands = parse_candidates(p["candidates"])[: p["num_candidates"]]
    r = Result("computed")
    if not cands:
        r.note("Add at least one candidate document (one per line; prefix with + to mark it truly relevant).", "warn")
        return r
    texts = [c for c, _ in cands]
    rel = np.array([x for _, x in cands])
    all_sims = {}
    backend_used = None
    for m in ("hash", "concept", "minilm"):
        E, used = embed([p["query"]] + texts, m, p["dimension"])
        if m == "minilm" and used != "real":
            continue
        all_sims[m] = E[1:] @ E[0]
        if m == p["embedding_model"] or (p["embedding_model"] == "minilm" and m == "minilm"):
            backend_used = used
    model = p["embedding_model"] if p["embedding_model"] in all_sims else "concept"
    if p["embedding_model"] == "minilm" and "minilm" not in all_sims:
        r.note("Real MiniLM unavailable (install sentence-transformers and set LLMLAB_ENABLE_HF_MODELS=true). Using bag-of-concepts.", "warn")
    else:
        r.set_fidelity("real" if backend_used == "real" else "computed")
    sims = all_sims[model]
    th = p["threshold"]
    above = sims >= th
    fp = int((above & ~rel).sum())
    fn = int((~above & rel).sum())
    tp = int((above & rel).sum())
    r.metric("top", "Top-ranked candidate", "relevant ✓" if rel[int(np.argmax(sims))] else "NOT relevant ✗",
             status="good" if rel[int(np.argmax(sims))] else "bad")
    r.metric("fp", "False positives ≥ threshold", fp, "", "irrelevant docs that pass the threshold", status="bad" if fp else "good")
    r.metric("fn", "False negatives < threshold", fn, "", "relevant docs filtered out", status="bad" if fn else "good")
    r.metric("precision", "Precision at threshold", tp / max(1, tp + fp) * 100, "%")
    r.metric("gap", "Best relevant − best irrelevant", float(sims[rel].max() - sims[~rel].max()) if rel.any() and (~rel).any() else 0, "",
             "negative = an irrelevant doc outranks every relevant one")
    labels = [t[:38] + ("…" if len(t) > 38 else "") for t in texts]
    r.chart("sims", "Cosine similarity to the query", "bar", labels,
            [{"label": {"hash": "lexical hash", "concept": "bag-of-concepts", "minilm": "MiniLM (real)"}[m], "data": s.tolist()} for m, s in all_sims.items()],
            "", "cosine", threshold=th, highlight=[bool(x) for x in rel])
    rows = []
    for i in np.argsort(-sims):
        reasons = trap_reasons(p["query"], texts[i])
        verdict = ("✓ true positive" if rel[i] else "✗ FALSE POSITIVE") if above[i] else ("✗ missed (false negative)" if rel[i] else "correctly rejected")
        rows.append([round(float(sims[i]), 3), texts[i], "relevant" if rel[i] else "not relevant", verdict, ", ".join(reasons) or "—"])
    r.table("Ranking with semantic verdicts", ["Cosine", "Candidate", "Ground truth", "At threshold", "Why similarity misleads"], rows)
    # 2D projection (PCA) of embeddings
    E, _ = embed([p["query"]] + texts, model, p["dimension"])
    Ec = E - E.mean(0)
    _, _, vt = np.linalg.svd(Ec, full_matrices=False)
    xy = Ec @ vt[:2].T
    r.chart("pca", "Embedding space (PCA 2-D projection)", "scatter", None, [
        {"label": "query", "data": [{"x": float(xy[0, 0]), "y": float(xy[0, 1]), "t": "query"}], "pointRadius": 9},
        {"label": "relevant", "data": [{"x": float(xy[i + 1, 0]), "y": float(xy[i + 1, 1]), "t": texts[i][:40]} for i in range(len(texts)) if rel[i]], "pointRadius": 7},
        {"label": "not relevant", "data": [{"x": float(xy[i + 1, 0]), "y": float(xy[i + 1, 1]), "t": texts[i][:40]} for i in range(len(texts)) if not rel[i]], "pointRadius": 7},
    ], "PC1", "PC2")
    # dimension effect: unrelated-pair similarity under hashing
    kb = [doc_text(d) for d in load_kb()["documents"]]
    dims = [8, 16, 32, 64, 128, 256, 512, 1024]
    means = []
    for dm in dims:
        V = np.vstack([hash_embed(t, dm) for t in kb])
        S = V @ V.T
        means.append(float(np.abs(S[np.triu_indices(len(kb), 1)]).mean()))
    r.chart("dims", "Avg |cosine| between unrelated KB docs vs. vector dimension (hash embeddings)", "line", dims,
            [{"label": "mean |cos| of unrelated pairs", "data": means}], "dimension", "|cos|", log_x=True)
    # Verification layer: demote candidates whose relation to the query is flagged by consistency rules
    demote = {"negation flips meaning", "same words, different order/roles", "numbers/dates differ"}
    verified = np.array([sims[i] * (0.5 if (demote & set(trap_reasons(p["query"], t)) or any(x.startswith("antonym") for x in trap_reasons(p["query"], t))) else 1.0)
                         for i, t in enumerate(texts)])
    vb = int(np.argmax(verified))
    r.compare("Cosine threshold only", "Cosine + consistency rules (negation/roles/numbers/antonyms)", [
        ("Top result relevant?", "yes" if rel[int(np.argmax(sims))] else "no", "yes" if rel[vb] else "no", "", None),
        ("False positives at threshold", fp, int(((verified >= th) & ~rel).sum()), "", "lower")])
    if not rel[vb]:
        r.note("Even with consistency rules the top result is still wrong. Similarity plus surface rules cannot verify meaning — e.g. a bare "
               "restatement of the query ('It is safe to…') or a swapped subject ('Costs' vs 'Revenue'). That needs grounding in trusted sources "
               "and an entailment (NLI) or cross-encoder check trained for the task.", "warn")
    r.note("Cosine similarity measures *direction overlap of vectors*, not truth, entailment or answer correctness.")
    r.recommend("Never use a single global similarity threshold as a correctness gate; calibrate per query type and corpus.",
                "Add a cross-encoder reranker or NLI/entailment check for negation, numbers and roles.",
                "Combine with lexical (BM25) signals for exact identifiers, and metadata filters for scope.",
                "Evaluate retrieval with labeled queries (Recall@k, nDCG), not by eyeballing similarity scores.")
    return r


register(ModuleDef(
    id="cosine-trap", title="Cosine Similarity Trap", category="retrieval", original_index=4, fidelity="computed",
    summary="High cosine similarity ≠ semantic correctness: negation, role reversal, polysemy and numbers.",
    presets=COSINE_PRESETS,
    controls=[
        text("query", "Query", COSINE_PRESETS[0]["params"]["query"], "The search query.", group="Input"),
        text("candidates", "Candidate documents (one per line, prefix + for truly relevant)", COSINE_PRESETS[0]["params"]["candidates"],
             "Ground-truth labels let the lab count false positives.", multiline=True, group="Input"),
        slider("threshold", "Similarity threshold", 0.5, 0.0, 1.0, 0.01, "Docs at/above this cosine are 'retrieved'."),
        select("embedding_model", "Embedding model", "concept", EMBEDDERS, "Vector representation used for the main verdicts."),
        slider("num_candidates", "Number of candidates", 10, 1, 30, 1, "Use only the first N lines."),
        slider("dimension", "Vector dimension", 384, 8, 1024, 8, "Hash/concept embedding size — low dims cause collisions."),
    ],
    simulate=cosine_sim,
    docs=docs(
        what="Cosine similarity is the normalized dot product of two embedding vectors. It is a geometric closeness score that retrieval systems use as a proxy for relevance.",
        why="Embeddings compress text into a fixed vector; texts that share topic, vocabulary or style land close together. That is useful for recall, but it means 'X is safe' and 'X is not safe' can be near-identical vectors.",
        how="Embedding models are trained to place paraphrases close; they are weakly sensitive to negation, argument order, numbers and entity swaps, and polysemous words pull different senses together. Thresholds learned on one corpus do not transfer to another.",
        when_use="Use cosine similarity for candidate generation (recall) in semantic search and RAG.",
        when_avoid="Do not use it as the final arbiter of correctness, answer validity, deduplication of facts, or a safety gate.",
        pros=["Fast (ANN indexes)", "Captures paraphrase and synonymy", "Language-agnostic with multilingual models"],
        cons=["Blind spots: negation, roles, numbers, antonyms", "Score scale differs by model — thresholds don't transfer", "Topic similarity masquerades as answer relevance"],
        use_case="A pharmacy chatbot retrieves 'It is safe to take ibuprofen with food' for a question about ibuprofen + aspirin; a cross-encoder + clinical rules layer prevents the unsafe answer.",
        code='''
from sentence_transformers import SentenceTransformer, CrossEncoder, util
emb = SentenceTransformer("sentence-transformers/all-MiniLM-L6-v2")
q = "Is it safe to take ibuprofen with aspirin?"
docs = ["It is safe to take ibuprofen with aspirin.", "It is not safe to take ibuprofen with aspirin."]
print(util.cos_sim(emb.encode(q), emb.encode(docs)))        # both very high!

ce = CrossEncoder("cross-encoder/ms-marco-MiniLM-L-6-v2")   # joint scoring is more discriminative
print(ce.predict([(q, d) for d in docs]))
# For factual consistency use an NLI model (entailment / contradiction), not cosine.
''',
        diagram=[{"label": "Query"}, {"label": "Embed"}, {"label": "Cosine top-k", "cls": "accent"}, {"label": "Rerank / NLI check"}, {"label": "Answer"}],
        performance="Cosine on normalized vectors is a dot product: O(d) per pair; ANN search is sub-linear. The cost of the trap is quality, not compute.",
        checklist=["Label a query set and measure precision/recall", "Calibrate thresholds per corpus/model", "Add reranking for top-k", "Handle negation and numbers with rules/NLI",
                   "Monitor false-positive retrievals in production feedback"],
        applications=[{"domain": "Healthcare information systems", "text": "Negation ('no history of diabetes') must not match 'history of diabetes'."},
                      {"domain": "Financial services", "text": "'Revenue increased' vs 'decreased' are near-identical vectors."},
                      {"domain": "Semantic search", "text": "Polysemy (bank, apple, python) pollutes results without context."}],
        limitations="Built-in embeddings are transparent lexical/concept models; real neural embeddings fail on the same categories, typically with higher absolute similarity values.",
        references=["Ettinger, 'What BERT is not' (2020)", "Reimers & Gurevych, 'Sentence-BERT' (2019)"],
    ),
))


# ============================================================================ Hybrid search
@lru_cache(maxsize=8)
def _kb_index(model: str):
    kb = load_kb()
    texts = [doc_text(d) for d in kb["documents"]]
    E, used = embed(texts, model)
    return kb, texts, E, used


@lru_cache(maxsize=32)
def _bm25(texts: tuple, k1: float, b: float) -> BM25:
    return BM25(list(texts), k1=k1, b=b)


@lru_cache(maxsize=512)
def _qvec(query: str, model: str) -> np.ndarray:
    return embed([query], model)[0][0]


def rank_all(query: str, texts, E, model: str, k1: float, b: float, rrf_k: int, alpha: float, allowed: np.ndarray):
    bm = _bm25(tuple(texts), k1, b).scores(query)
    vs = VectorIndex(E).scores(_qvec(query, model))
    bm_m, vs_m = np.where(allowed, bm, -1e9), np.where(allowed, vs, -1e9)
    order_bm = [int(i) for i in np.argsort(-bm_m) if allowed[i]]
    order_vs = [int(i) for i in np.argsort(-vs_m) if allowed[i]]
    fused = rrf([order_bm, order_vs], rrf_k)
    order_rrf = sorted(fused, key=lambda i: -fused[i])
    w = alpha * minmax(vs) + (1 - alpha) * minmax(bm)
    order_w = [int(i) for i in np.argsort(-np.where(allowed, w, -1e9)) if allowed[i]]
    return {"bm25": order_bm, "vector": order_vs, "rrf": order_rrf, "weighted": order_w}, bm, vs, fused, w


def hybrid_sim(p: dict, ctx: Context) -> Result:
    kb, texts, E, used = _kb_index(p["embedding_model"])
    ids = [d["id"] for d in kb["documents"]]
    r = Result("real" if used == "real" else "computed")
    allowed_all = np.ones(len(ids), dtype=bool)
    # aggregate metrics over all labeled queries (no metadata filter so qrels stay valid)
    methods = {"bm25": "BM25 only", "vector": "Vector only", "rrf": "Hybrid (RRF)", "weighted": f"Hybrid (weighted α={p['alpha']})"}
    agg = {m: {"recall": [], "mrr": [], "ndcg": []} for m in methods}
    k = p["top_k"]
    for q in kb["queries"]:
        orders = rank_all(q["text"], texts, E, p["embedding_model"], p["k1"], p["b"], p["rrf_k"], p["alpha"], allowed_all)[0]
        for m, o in orders.items():
            ranked = [ids[i] for i in o]
            agg[m]["recall"].append(recall_at_k(ranked, q["qrels"], k))
            agg[m]["mrr"].append(mrr(ranked, q["qrels"]))
            agg[m]["ndcg"].append(ndcg_at_k(ranked, q["qrels"], k))
    rows = [[methods[m], f"{np.mean(v['recall']):.3f}", f"{np.mean(v['mrr']):.3f}", f"{np.mean(v['ndcg']):.3f}"] for m, v in agg.items()]
    r.table(f"Aggregate over {len(kb['queries'])} labeled queries (k={k})", ["Method", f"Recall@{k}", "MRR", f"nDCG@{k}"], rows)
    r.chart("agg", "Retrieval quality by method (labeled queries)", "bar", [f"Recall@{k}", "MRR", f"nDCG@{k}"],
            [{"label": methods[m], "data": [float(np.mean(v["recall"])), float(np.mean(v["mrr"])), float(np.mean(v["ndcg"]))]} for m, v in agg.items()], "", "score")
    sel = p["fusion"]
    for key, name in (("recall", "Recall"), ("ndcg", "nDCG")):
        r.metric(f"{key}_sel", f"{name}@{k} — {methods[sel]}", float(np.mean(agg[sel][key])))
    r.metric("mrr_sel", f"MRR — {methods[sel]}", float(np.mean(agg[sel]["mrr"])))
    r.metric("gain", f"nDCG gain vs BM25", (np.mean(agg[sel]["ndcg"]) - np.mean(agg["bm25"]["ndcg"])) * 100, "pts")
    # alpha sweep
    alphas = [round(x, 1) for x in np.linspace(0, 1, 11)]
    sweep = []
    for a in alphas:
        vals = []
        for q in kb["queries"]:
            o = rank_all(q["text"], texts, E, p["embedding_model"], p["k1"], p["b"], p["rrf_k"], a, allowed_all)[0]["weighted"]
            vals.append(ndcg_at_k([ids[i] for i in o], q["qrels"], k))
        sweep.append(float(np.mean(vals)))
    r.chart("alpha", "Weighted fusion: nDCG vs α (0 = pure BM25, 1 = pure vector)", "line", alphas, [{"label": f"nDCG@{k}", "data": sweep}], "α", "nDCG")
    # current query detail (with metadata filter)
    qtext = p["custom_query"].strip() or next(q["text"] for q in kb["queries"] if q["id"] == p["query_id"])
    qrels = next((q["qrels"] for q in kb["queries"] if q["text"] == qtext), {})
    allowed = np.array([p["department"] in ("all", d["dept"]) for d in kb["documents"]])
    orders, bm, vs, fused, w = rank_all(qtext, texts, E, p["embedding_model"], p["k1"], p["b"], p["rrf_k"], p["alpha"], allowed)
    pos = {m: {i: n + 1 for n, i in enumerate(o)} for m, o in orders.items()}
    detail = []
    for i in orders[sel][:k]:
        detail.append([ids[i], kb["documents"][i]["dept"], kb["documents"][i]["title"], pos["bm25"].get(i, "—"), pos["vector"].get(i, "—"),
                       round(float(bm[i]), 2), round(float(vs[i]), 3), round(fused.get(i, 0), 4), {2: "high", 1: "partial"}.get(qrels.get(ids[i], 0), "—")])
    r.table(f"Results for “{qtext}” ({methods[sel]}, dept={p['department']})",
            ["Doc", "Dept", "Title", "BM25 rank", "Vector rank", "BM25", "Cosine", "RRF", "Label"], detail)
    if qrels:
        ranked = [ids[i] for i in orders[sel]]
        r.compare("BM25 only", methods[sel], [
            (f"nDCG@{k} (this query)", ndcg_at_k([ids[i] for i in orders["bm25"]], qrels, k), ndcg_at_k(ranked, qrels, k), "", "higher"),
            ("MRR (this query)", mrr([ids[i] for i in orders["bm25"]], qrels), mrr(ranked, qrels), "", "higher")])
    else:
        r.note("Custom query has no relevance labels — per-query metrics unavailable; aggregate metrics still shown.")
    r.flow("Hybrid retrieval pipeline", [{"label": "Query"}, {"label": "Metadata pre-filter", "sub": p["department"]},
                                         {"label": "BM25 (lexical)", "cls": "accent"}, {"label": "Vector (semantic)", "cls": "accent"},
                                         {"label": "Fusion", "sub": sel.upper()}, {"label": f"Top-{k}"}])
    r.recommend("RRF is robust because it ignores incomparable score scales; tune α only with labeled data.",
                "BM25 wins on exact identifiers (error codes, SKUs, names); vectors win on paraphrases ('vacation' ↔ 'annual leave').",
                "Apply metadata filters (tenant, ACL, department, date) before ranking — never after generation.")
    return r


HYBRID_CONTROLS = [
    select("query_id", "Labeled query", "q2", [{"value": q["id"], "label": q["text"]} for q in load_kb()["queries"]], "Query with relevance judgments."),
    text("custom_query", "Custom query (overrides)", "", "Leave blank to use the labeled query."),
    select("fusion", "Retrieval strategy", "rrf", [{"value": "bm25", "label": "BM25 only"}, {"value": "vector", "label": "Vector only"},
                                                   {"value": "rrf", "label": "Hybrid: Reciprocal Rank Fusion"}, {"value": "weighted", "label": "Hybrid: weighted scores"}], "How results are combined."),
    slider("alpha", "Vector weight α (weighted fusion)", 0.5, 0.0, 1.0, 0.05, "score = α·vector + (1−α)·BM25 after min-max normalization."),
    slider("rrf_k", "RRF constant k", 60, 1, 100, 1, "Larger k flattens rank contributions."),
    slider("top_k", "Top-K", 3, 1, 10, 1, "Results evaluated."),
    select("embedding_model", "Embedding model", "concept", EMBEDDERS, "Vector side of the hybrid."),
    select("department", "Metadata filter: department", "all", ["all", "hr", "it", "finance", "security", "general"], "Pre-filter applied to the current query view."),
    slider("k1", "BM25 k1", 1.5, 0.1, 3.0, 0.1, "Term-frequency saturation."),
    slider("b", "BM25 b", 0.75, 0.0, 1.0, 0.05, "Document-length normalization."),
]

register(ModuleDef(
    id="hybrid-search", title="Hybrid Search (BM25 + Vector)", category="retrieval", fidelity="computed", tags=["essential"],
    summary="Combine lexical BM25 and semantic vector retrieval with rank fusion and metadata filtering.",
    controls=HYBRID_CONTROLS, simulate=hybrid_sim,
    docs=docs(
        what="Hybrid search runs a lexical retriever (BM25) and a dense vector retriever in parallel and fuses their rankings.",
        why="Lexical search excels at exact terms (IDs, codes, names) but misses paraphrases; dense retrieval captures meaning but misses rare exact tokens. Enterprise queries contain both.",
        how="BM25 scores term matches with IDF weighting and length normalization. Vectors score cosine similarity. Reciprocal Rank Fusion adds 1/(k+rank) from each list; weighted fusion mixes normalized scores with α.",
        when_use="Almost every enterprise RAG / search deployment, especially with product codes, policy names, acronyms or multilingual users.",
        when_avoid="Tiny corpora where either retriever alone is perfect, or extremely latency-constrained paths where only one index is feasible.",
        pros=["Higher recall and nDCG across query types", "RRF needs no score calibration", "Robust to embedding model weaknesses"],
        cons=["Two indexes to build and maintain", "More tuning parameters", "Slightly higher latency"],
        use_case="An IT helpdesk search where 'error 0x80070005' needs BM25 while 'my laptop is broken' needs semantic matching to 'hardware ticket / damaged laptop'.",
        code='''
from rank_bm25 import BM25Okapi
from sentence_transformers import SentenceTransformer
import numpy as np
bm25 = BM25Okapi([d.lower().split() for d in docs])
emb = SentenceTransformer("BAAI/bge-small-en-v1.5"); D = emb.encode(docs, normalize_embeddings=True)

def hybrid(query, k=5, rrf_k=60):
    lex = np.argsort(-bm25.get_scores(query.lower().split()))
    sem = np.argsort(-(D @ emb.encode(query, normalize_embeddings=True)))
    score = {}
    for ranking in (lex, sem):
        for r, i in enumerate(ranking):
            score[i] = score.get(i, 0) + 1 / (rrf_k + r + 1)
    return sorted(score, key=score.get, reverse=True)[:k]
# Managed options: Elasticsearch/OpenSearch hybrid queries, Weaviate/Qdrant/pgvector + tsvector.
''',
        diagram=[{"label": "Query"}, {"label": "Filter (ACL/tenant)"}, {"label": "BM25", "cls": "accent"}, {"label": "Vector", "cls": "accent"},
                 {"label": "RRF fusion"}, {"label": "Rerank (optional)"}],
        performance="Two retrievals in parallel (~ms each with indexes); fusion is negligible. Index storage roughly doubles.",
        checklist=["Build labeled query set with graded relevance", "Compare BM25 / vector / hybrid on nDCG@k", "Pre-filter by ACL/tenant", "Tune analyzers (stemming, synonyms)", "Monitor zero-result and low-score queries"],
        applications=[{"domain": "Semantic search", "text": "Enterprise intranet search across policies and tickets."},
                      {"domain": "Customer support automation", "text": "Matching product codes plus natural-language symptoms."},
                      {"domain": "Document intelligence", "text": "Contract search by clause meaning and exact party names."}],
        limitations="Corpus is a 28-document synthetic KB with 12 labeled queries — enough to show the effect, not to rank production systems.",
        references=["Robertson & Zaragoza, 'The Probabilistic Relevance Framework: BM25' (2009)", "Cormack et al., 'Reciprocal Rank Fusion' (2009)"],
    ),
))


# ============================================================================ RAG optimization
EVIDENCE = {"q1": "accrue 25 days of paid annual leave", "q2": "open the self-service portal and choose forgot password",
            "q3": "must never be pasted into external ai tools", "q4": "hotel costs are capped at 250 dollars per night",
            "q5": "submit expenses within 30 days with itemized receipts", "q6": "work remotely up to three days per week",
            "q7": "open a hardware ticket and it will provide a loaner device", "q8": "offices close on national bank holidays",
            "q9": "report suspicious emails with the report phishing button", "q10": "supplier invoices are paid within 45 days of receipt",
            "q11": "secondary caregivers receive 6 weeks", "q12": "requests for new software require a security review"}


def build_chunks(strategy: str, size: int, overlap: int) -> list[str]:
    kb = load_kb()["documents"]
    if strategy == "paragraph":
        out, cur = [], []
        for d in kb:
            para = doc_text(d).split()
            if cur and len(cur) + len(para) > size:
                out.append(" ".join(cur))
                cur = []
            cur += para
        return out + ([" ".join(cur)] if cur else [])
    corpus = " ".join(doc_text(d) for d in kb)
    if strategy == "fixed":
        w = corpus.split()
        step = max(1, size - overlap)
        return [" ".join(w[i:i + size]) for i in range(0, len(w), step) if w[i:i + size]]
    sents = re.split(r"(?<=[.!?])\s+", corpus)
    out, cur = [], []
    for s in sents:
        sw = s.split()
        if cur and len(cur) + len(sw) > size:
            out.append(" ".join(cur))
            keep = cur[-overlap:] if overlap else []
            cur = keep[:]
        cur += sw
    return out + ([" ".join(cur)] if cur else [])


def compress(query: str, chunk: str) -> str:
    qs = {stem(w) for w in content_words(query)}
    sents = re.split(r"(?<=[.!?])\s+", chunk)
    keep = [s for s in sents if qs & {stem(w) for w in content_words(s)}]
    return " ".join(keep) if keep else chunk


@lru_cache(maxsize=64)
def _rag_index(strategy: str, size: int, overlap: int, model: str):
    chunks = build_chunks(strategy, size, overlap)
    E, used = embed(chunks, model)
    return chunks, E, BM25(chunks), used


@lru_cache(maxsize=4096)
def _xscore(query: str, chunk: str) -> float:
    return simulated_cross_score(query, chunk)


def run_rag(p: dict, size: int, overlap: int) -> dict:
    chunks, E, bm, used = _rag_index(p["strategy"], size, max(0, min(overlap, size - 1)), p["embedding_model"])
    kb = load_kb()
    hits, ctx_tokens, prec = [], [], []
    for q in kb["queries"]:
        b = bm.scores(q["text"])
        v = E @ _qvec(q["text"], p["embedding_model"])
        if p["retriever"] == "bm25":
            order = list(np.argsort(-b))
        elif p["retriever"] == "vector":
            order = list(np.argsort(-v))
        else:
            f = rrf([list(np.argsort(-b)), list(np.argsort(-v))])
            order = sorted(f, key=lambda i: -f[i])
        cand = order[: p["top_k"] * 3] if p["rerank"] else order[: p["top_k"]]
        if p["rerank"]:
            cand = sorted(cand, key=lambda i: -_xscore(q["text"], chunks[i]))[: p["top_k"]]
        ctx = [compress(q["text"], chunks[i]) if p["compression"] else chunks[i] for i in cand]
        joined = " ".join(ctx).lower()
        ev = EVIDENCE[q["id"]]
        hit = ev in re.sub(r"\s+", " ", joined)
        hits.append(hit)
        ntok = int(len(joined.split()) * 1.33)
        ctx_tokens.append(ntok)
        prec.append(len(ev.split()) / max(1, len(joined.split())) if hit else 0)
    return {"chunks": chunks, "recall": float(np.mean(hits)), "tokens": float(np.mean(ctx_tokens)), "precision": float(np.mean(prec)),
            "hits": hits, "backend": used}


def rag_sim(p: dict, ctx: Context) -> Result:
    res = run_rag(p, p["chunk_size"], p["chunk_overlap"])
    r = Result("real" if res["backend"] == "real" else "computed")
    r.metric("recall", "Answer-evidence recall@k", res["recall"] * 100, "%", "questions whose evidence sentence is fully inside the retrieved context",
             status="good" if res["recall"] >= 0.9 else ("bad" if res["recall"] < 0.6 else None))
    r.metric("tokens", "Avg context tokens / query", res["tokens"], "tok", "≈ words × 1.33")
    r.metric("precision", "Context precision", res["precision"] * 100, "%", "evidence words ÷ context words")
    r.metric("chunks", "Chunks in index", len(res["chunks"]))
    r.metric("cost", "Prompt tokens per 1K queries", res["tokens"] * 1000 / 1e6, "M tok")
    sizes = [20, 30, 40, 60, 80, 120, 160, 240, 320]
    sweep = [run_rag(p, s, min(p["chunk_overlap"], s - 1)) for s in sizes]
    r.chart("sweep", "Chunk size trade-off", "line", sizes, [
        {"label": "Evidence recall %", "data": [x["recall"] * 100 for x in sweep]},
        {"label": "Context tokens / query", "data": [x["tokens"] for x in sweep], "yAxisID": "y1"}], "chunk size (words)", "recall %", dual_axis=True)
    ovs = [0, 5, 10, 20, 30, 40]
    ov = [run_rag(p, p["chunk_size"], o)["recall"] * 100 for o in ovs]
    r.chart("overlap", f"Overlap effect at chunk size {p['chunk_size']}", "bar", ovs, [{"label": "Evidence recall %", "data": ov}], "overlap (words)", "%")
    kb = load_kb()
    r.table("Per-question outcome", ["Question", "Evidence retrieved?"], [[q["text"], "✓" if h else "✗ missed"] for q, h in zip(kb["queries"], res["hits"])])
    r.tokens("First chunks of the index", [{"text": c[:90] + "…", "cls": "chunk"} for c in res["chunks"][:6]])
    base = run_rag({**p, "rerank": False, "compression": False, "retriever": "vector"}, p["chunk_size"], p["chunk_overlap"])
    r.compare("Vector-only, no rerank/compression", "Current pipeline", [
        ("Evidence recall", base["recall"] * 100, res["recall"] * 100, "%", "higher"), ("Context tokens", base["tokens"], res["tokens"], "tok", "lower"),
        ("Context precision", base["precision"] * 100, res["precision"] * 100, "%", "higher")])
    r.flow("RAG pipeline", [{"label": "Documents"}, {"label": f"Chunk ({p['strategy']})", "sub": f"{p['chunk_size']}w / {p['chunk_overlap']}w overlap"},
                            {"label": "Embed + index"}, {"label": f"Retrieve ({p['retriever']})", "cls": "accent"},
                            {"label": "Rerank" if p["rerank"] else "(no rerank)"}, {"label": "Compress" if p["compression"] else "(no compression)"},
                            {"label": "LLM prompt", "cls": "accent"}])
    r.note("Evidence recall measures whether the answer is *available* to the LLM; generation faithfulness must be evaluated separately (see Model Evaluation).")
    r.recommend("Chunk on semantic boundaries (sentences/sections) with modest overlap (10–20%).",
                "Retrieve wide (top-20 hybrid), rerank narrow (top-3–5), then compress to cut prompt tokens.",
                "Track evidence recall and context tokens together — both drive answer quality and cost.")
    return r


register(ModuleDef(
    id="rag-optimization", title="RAG Optimization", category="retrieval", fidelity="computed", tags=["essential"],
    summary="Tune chunking, overlap, retrieval strategy, reranking and context compression end-to-end.",
    controls=[
        select("strategy", "Chunking strategy", "sentence", [{"value": "fixed", "label": "Fixed word windows"}, {"value": "sentence", "label": "Sentence-aware packing"},
                                                            {"value": "paragraph", "label": "Document/section packing"}], "How the corpus is split."),
        slider("chunk_size", "Chunk size", 60, 10, 400, 10, "Words per chunk (≈ ×1.33 tokens).", unit="words"),
        slider("chunk_overlap", "Chunk overlap", 10, 0, 100, 5, "Words repeated between consecutive chunks.", unit="words"),
        select("retriever", "Retriever", "hybrid", [{"value": "bm25", "label": "BM25"}, {"value": "vector", "label": "Vector"}, {"value": "hybrid", "label": "Hybrid (RRF)"}], "First-stage retrieval."),
        select("embedding_model", "Embedding model", "concept", EMBEDDERS, "Dense representation."),
        slider("top_k", "Top-K chunks in context", 3, 1, 10, 1, "Chunks passed to the LLM."),
        toggle("rerank", "Cross-encoder rerank", True, "Retrieve 3×K then rerank to K."),
        toggle("compression", "Context compression", False, "Keep only sentences sharing terms with the query."),
    ],
    simulate=rag_sim,
    docs=docs(
        what="RAG optimization is the systematic tuning of the retrieval half of retrieval-augmented generation: how documents are chunked, embedded, retrieved, filtered, reranked and compressed before reaching the LLM.",
        why="Most RAG failures are retrieval failures: the answer was never in the context, or was buried in irrelevant tokens. Every token in context also costs latency and money.",
        how="Chunk size trades recall granularity against context dilution; overlap prevents evidence being split across boundaries; hybrid retrieval maximizes recall; reranking maximizes precision; compression removes irrelevant sentences.",
        when_use="Any production RAG: knowledge assistants, support bots, document Q&A, agent memory.",
        when_avoid="When the full corpus fits in context cheaply and latency allows (long-context models), or when answers require aggregation over the entire corpus (use structured queries/SQL instead).",
        pros=["Large quality gains for little compute", "Lower token cost with compression", "Measurable with labeled questions"],
        cons=["Many interacting knobs", "Optimal settings are corpus-specific", "Compression can drop needed context"],
        use_case="A financial-services policy assistant raises evidence recall from 70% to 95% by switching from 512-token fixed chunks to sentence-aware 150-token chunks with hybrid retrieval and reranking.",
        code='''
from langchain_text_splitters import RecursiveCharacterTextSplitter
splitter = RecursiveCharacterTextSplitter(chunk_size=600, chunk_overlap=80, separators=["\\n\\n", "\\n", ". ", " "])
chunks = splitter.split_text(handbook)

candidates = hybrid_retrieve(question, k=20)                       # recall
reranked = cross_encoder_rerank(question, candidates)[:4]          # precision
context = "\\n\\n".join(compress(question, c) for c in reranked)    # cost
prompt = f"Answer ONLY from the context. Cite sources.\\n\\n{context}\\n\\nQ: {question}"
''',
        diagram=[{"label": "Chunk"}, {"label": "Embed/Index"}, {"label": "Hybrid retrieve", "cls": "accent"}, {"label": "Rerank"}, {"label": "Compress"}, {"label": "Generate", "cls": "accent"}],
        performance="Context tokens drive prefill latency and cost linearly; reranking adds tens of ms; retrieval is ms-scale.",
        checklist=["Create a labeled evaluation set (question → evidence)", "Sweep chunk size/overlap per corpus", "Use hybrid + rerank", "Measure context tokens and evidence recall", "Re-evaluate on corpus updates"],
        applications=[{"domain": "Enterprise chatbots", "text": "HR/IT knowledge assistants grounded in policy documents."},
                      {"domain": "Document intelligence", "text": "Q&A over contracts, reports and filings."},
                      {"domain": "Agentic AI", "text": "Agent memory retrieval with tight context budgets."}],
        limitations="Evaluates evidence availability with exact-substring matching on a small synthetic corpus.",
        references=["Lewis et al., 'Retrieval-Augmented Generation' (2020)", "Gao et al., 'RAG for LLMs: A Survey' (2024)"],
    ),
))
