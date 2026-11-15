"""Module 6 — Cross-Encoder Reranking."""
from __future__ import annotations

import numpy as np

from app.engine.registry import Context, ModuleDef, Result, docs, register, select, slider, text, toggle
from app.reranking.scorer import RERANKERS, cross_scores, rerank_latency_ms
from app.retrieval.core import doc_text, embed, load_kb, ndcg_at_k, precision_at_k
from app.retrieval.modules import EMBEDDERS


def _pipeline(query, texts, ids, E, model_emb, top_k, final_k, threshold):
    qv, _ = embed([query], model_emb)
    sims = E @ qv[0]
    first = [int(i) for i in np.argsort(-sims)[:top_k]]
    scores, backend = cross_scores(query, [texts[i] for i in first])
    order = [first[j] for j in np.argsort(-np.array(scores))]
    sc_map = {first[j]: scores[j] for j in range(len(first))}
    final = [i for i in order if sc_map[i] >= threshold][:final_k]
    return sims, first, order, sc_map, final, backend


def rerank_sim(p: dict, ctx: Context) -> Result:
    kb = load_kb()
    ids = [d["id"] for d in kb["documents"]]
    texts = [doc_text(d) for d in kb["documents"]]
    E, used = embed(texts, p["embedding_model"])
    qtext = p["custom_query"].strip() or next(q["text"] for q in kb["queries"] if q["id"] == p["query_id"])
    qrels = next((q["qrels"] for q in kb["queries"] if q["text"] == qtext), {})
    sims, first, order, sc, final, backend = _pipeline(qtext, texts, ids, E, p["embedding_model"], p["top_k"], p["final_k"], p["threshold"])
    r = Result("real" if backend == "real" else "simulated")
    if backend != "real":
        r.set_fidelity("simulated", "Bi-encoder retrieval and IR metrics are computed exactly; the cross-encoder is a deterministic rule-based joint scorer "
                                    "(enable LLMLAB_ENABLE_HF_MODELS with sentence-transformers for a real ms-marco cross-encoder).")
    K = p["final_k"]
    vec_ranked = [ids[i] for i in first]
    rr_ranked = [ids[i] for i in order]
    shown_final = [ids[i] for i in final]
    if qrels:
        r.metric("p_before", f"Precision@{K} — vector only", precision_at_k(vec_ranked, qrels, K))
        r.metric("p_after", f"Precision@{K} — reranked", precision_at_k(rr_ranked, qrels, K))
        r.metric("n_before", f"nDCG@{K} — vector only", ndcg_at_k(vec_ranked, qrels, K))
        r.metric("n_after", f"nDCG@{K} — reranked", ndcg_at_k(rr_ranked, qrels, K))
    lat_vec = 3.0
    lat_rr = rerank_latency_ms(p["reranker"], p["top_k"], p["tokens_per_pair"], p["device"])
    r.metric("latency", "Added rerank latency (est.)", lat_rr, "ms", f"{p['top_k']} pairs × {p['tokens_per_pair']} tokens on {p['device'].upper()}")
    r.metric("kept", "Docs passing relevance threshold", len(final), "", f"threshold {p['threshold']}")
    rows = []
    for rank, i in enumerate(first):
        rows.append([rank + 1, order.index(i) + 1, ids[i], kb["documents"][i]["title"], round(float(sims[i]), 3), round(sc[i], 3),
                     "kept" if i in final else "dropped", {2: "high", 1: "partial"}.get(qrels.get(ids[i], 0), "—")])
    r.table(f"“{qtext}” — first-stage top-{p['top_k']} and reranked positions", ["Vector rank", "Reranked", "Doc", "Title", "Cosine", "Cross score", "Final", "Label"], rows)
    r.chart("ranks", "Bi-encoder score vs. cross-encoder score", "bar", [ids[i] for i in first], [
        {"label": "cosine (bi-encoder)", "data": [float(sims[i]) for i in first]}, {"label": "cross-encoder", "data": [sc[i] for i in first]}], "", "score",
        highlight=[qrels.get(ids[i], 0) > 0 for i in first])
    # aggregate across labeled queries
    agg_b, agg_a, agg_pb, agg_pa = [], [], [], []
    for q in kb["queries"]:
        _, f1, o1, _, _, _ = _pipeline(q["text"], texts, ids, E, p["embedding_model"], p["top_k"], p["final_k"], 0.0)
        agg_b.append(ndcg_at_k([ids[i] for i in f1], q["qrels"], K))
        agg_a.append(ndcg_at_k([ids[i] for i in o1], q["qrels"], K))
        agg_pb.append(precision_at_k([ids[i] for i in f1], q["qrels"], K))
        agg_pa.append(precision_at_k([ids[i] for i in o1], q["qrels"], K))
    r.chart("agg", f"All {len(kb['queries'])} labeled queries", "bar", [f"Precision@{K}", f"nDCG@{K}"], [
        {"label": "Vector search only", "data": [float(np.mean(agg_pb)), float(np.mean(agg_b))]},
        {"label": "Vector + rerank", "data": [float(np.mean(agg_pa)), float(np.mean(agg_a))]}], "", "score")
    ks = [5, 10, 20, 28]
    r.chart("lat", "Rerank latency vs. candidates (estimate)", "line", ks, [
        {"label": f"{RERANKERS[m]['label']} on {p['device'].upper()}", "data": [rerank_latency_ms(m, k, p["tokens_per_pair"], p["device"]) for k in ks]} for m in RERANKERS],
        "candidates reranked", "ms", log_y=True)
    r.compare("Vector search only", "Vector + cross-encoder", [
        (f"Mean nDCG@{K} (all queries)", float(np.mean(agg_b)), float(np.mean(agg_a)), "", "higher"),
        (f"Mean Precision@{K}", float(np.mean(agg_pb)), float(np.mean(agg_pa)), "", "higher"),
        ("Retrieval latency", lat_vec, lat_vec + lat_rr, "ms", "lower")])
    r.flow("Two-stage retrieval", [{"label": "Query"}, {"label": "Bi-encoder ANN", "sub": f"top-{p['top_k']} (fast, independent encodings)"},
                                   {"label": "Cross-encoder", "sub": "[CLS] query [SEP] doc → score", "cls": "accent"},
                                   {"label": "Threshold + top-K", "sub": f"keep ≥{p['threshold']}, K={K}"}, {"label": "LLM context"}])
    r.recommend("Rerank 20–50 candidates down to 3–5; beyond ~100 the latency rarely pays off.",
                "Use a GPU or a small reranker (MiniLM) for interactive latency; batch pairs.",
                "Reranker scores are better calibrated than cosine for 'no good answer' thresholds — still validate per domain.")
    return r


register(ModuleDef(
    id="cross-encoder-reranking", title="Cross-Encoder Reranking", category="retrieval", original_index=6, fidelity="simulated",
    summary="Bi-encoders retrieve fast; cross-encoders read query and document together to rank precisely.",
    controls=[
        select("query_id", "Labeled query", "q4", [{"value": q["id"], "label": q["text"]} for q in load_kb()["queries"]], "Query with relevance labels."),
        text("custom_query", "Custom query (overrides)", "", "Leave blank to use a labeled query."),
        select("embedding_model", "First-stage embedding model", "hash", EMBEDDERS, "Bi-encoder used for initial retrieval."),
        slider("top_k", "Top-K retrieved (first stage)", 10, 2, 28, 1, "Candidates passed to the reranker."),
        slider("final_k", "Final reranked count", 3, 1, 10, 1, "Results kept after reranking."),
        slider("threshold", "Relevance threshold", 0.3, 0.0, 1.0, 0.05, "Minimum cross-encoder score to keep a document."),
        select("reranker", "Reranker model (latency estimate)", "minilm-l6", [{"value": k, "label": v["label"]} for k, v in RERANKERS.items()], "Size drives latency."),
        select("device", "Reranker hardware", "cpu", [{"value": "cpu", "label": "CPU"}, {"value": "gpu", "label": "GPU"}], "Where reranking runs."),
        slider("tokens_per_pair", "Tokens per (query, doc) pair", 128, 32, 512, 32, "Sequence length per cross-encoder pass."),
    ],
    simulate=rerank_sim,
    docs=docs(
        what="A cross-encoder is a transformer that takes the query and a candidate document concatenated as one input and outputs a relevance score. A bi-encoder embeds each independently and compares vectors.",
        why="Bi-encoders must compress a document into one vector before seeing the query, losing fine-grained interactions. Cross-encoders attend across query and document tokens, catching negation, specificity and answer presence.",
        how="Stage 1: bi-encoder/BM25 retrieves top-K cheaply from millions of docs. Stage 2: cross-encoder scores each (query, doc) pair — K forward passes — and reorders. Optionally drop documents below a score threshold.",
        when_use="RAG and search where precision in the top few results matters (LLM context slots are scarce), and latency budgets allow +20–200 ms.",
        when_avoid="Very high QPS with tight latency and no GPU; when first-stage precision is already sufficient; huge candidate sets (cost is linear in K).",
        pros=["Large precision/nDCG gains", "Better-calibrated relevance scores for thresholds", "Drop-in second stage"],
        cons=["K forward passes per query", "Needs GPU for low latency at scale", "Cannot pre-compute document representations"],
        use_case="A legal-document assistant retrieves 50 clauses with hybrid search and reranks to the 5 most relevant, reducing hallucinated citations.",
        code='''
from sentence_transformers import CrossEncoder
reranker = CrossEncoder("cross-encoder/ms-marco-MiniLM-L-6-v2", max_length=512)
cands = vector_store.similarity_search(query, k=30)
scores = reranker.predict([(query, c.page_content) for c in cands], batch_size=32)
top = [c for _, c in sorted(zip(scores, cands), key=lambda t: -t[0])][:5]
''',
        diagram=[{"label": "Query"}, {"label": "ANN / BM25 top-K"}, {"label": "Cross-encoder ×K", "cls": "accent"}, {"label": "Reorder + threshold"}, {"label": "Top-N to LLM"}],
        performance="Latency ≈ K × 2·params·tokens / effective FLOPs. MiniLM-L6 on GPU reranks 50 pairs in ~10–20 ms; large rerankers on CPU can take seconds.",
        checklist=["Measure nDCG@k with and without reranking", "Size K by latency budget", "Batch pairs and use GPU", "Truncate documents sensibly", "Cache scores for repeated queries"],
        applications=[{"domain": "Semantic search", "text": "Precision ranking for enterprise search results."},
                      {"domain": "Document intelligence", "text": "Selecting the exact clauses/passages for answer generation."},
                      {"domain": "Customer support automation", "text": "Choosing the single best KB article to show agents."}],
        limitations="Without the optional model, the cross-encoder is simulated with a transparent joint scoring rule; latency is an estimate.",
        references=["Nogueira & Cho, 'Passage Re-ranking with BERT' (2019)"],
    ),
))
