# LLM Engineering Lab

An interactive, production-structured web application for **understanding, experimenting with, simulating and comparing**
modern LLM architecture, inference optimization, retrieval, fine-tuning, privacy and evaluation techniques.

- **27 interactive modules** — all 16 requested concepts, the essential enterprise modules (quantization, QLoRA, RAG,
  prompt engineering, evaluation, serving, security & governance) and the recommended additions (PagedAttention,
  continuous batching, hybrid search, calibration, attention variants).
- **Every module** has typed controls (default/min/max/step/current value/tooltip/reset), a live visualization, a
  before-vs-after comparison, practical recommendations and 12 documentation sections.
- **Runs locally with no GPU and no paid API.** A deterministic simulation engine powers everything; optional
  integrations run real models (Ollama, any OpenAI-compatible API, Hugging Face / PEFT / sentence-transformers).
- **Honest numbers.** Every result carries a fidelity label — *Analytical*, *Computed*, *Simulated* or *Real* —
  and simulated values are never presented as benchmarks. "Live measurement" in Benchmark Studio reports nothing
  unless a real provider is connected.

---

## Quick start (macOS / Linux, Python 3.12+)

```bash
cd llm-engineering-lab
python3.12 -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
cp .env.example .env            # optional — sensible defaults are built in
uvicorn app.main:app --reload --port 8000
```

Open <http://localhost:8000> and sign in.

| Account | Password | Role |
|---|---|---|
| `demo@llmlab.local` | `Demo@12345` | Administrator |
| `engineer@llmlab.local` | `Demo@12345` | AI Engineer |
| `researcher@llmlab.local` | `Demo@12345` | Researcher |
| `viewer@llmlab.local` | `Demo@12345` | Viewer |

Demo credentials are configurable via `LLMLAB_DEMO_USER_EMAIL`, `LLMLAB_DEMO_USER_PASSWORD`, `LLMLAB_DEMO_USER_ROLE`
and `LLMLAB_SEED_ROLE_USERS`. Users are seeded at startup only if missing.

### Docker

```bash
docker compose up --build                       # app + SQLite volume → http://localhost:8000
docker compose --profile ollama up --build      # + local Ollama server (then: docker compose exec ollama ollama pull llama3.2:1b)
```

PostgreSQL: set `LLMLAB_DATABASE_URL=postgresql+asyncpg://llmlab:<password>@postgres:5432/llmlab` in `.env` and run
`docker compose --profile postgres up --build`. Build with real-model libraries: `WITH_ML=1 docker compose build`.

### Tests

```bash
pytest            # 70 unit + integration tests (auth, RBAC, CSRF, every module, math correctness, workflows, WebSockets)
```

---

## Modules

| # | Module | Fidelity | What is computed |
|---|---|---|---|
| 1 | KV Cache | Analytical | Exact KV bytes (2·L·H_kv·d·s·b·bytes); roofline latency with/without cache; GQA/MQA; concurrency |
| 2 | Tokenizer Effects ("bias") | Computed | Real byte-level BPE trained in-app with configurable language weighting; language token premium |
| 3 | Temperature Sampling | Simulated | Real temperature/top-k/top-p/repetition-penalty math on synthetic logits; optional real model runs |
| 4 | Cosine Similarity Trap | Computed | Real embeddings (lexical hash / concept / MiniLM if installed); negation, role reversal, polysemy, numbers |
| 5 | Prefix Caching | Simulated | Exact LRU block-cache simulation over a seeded request stream; saved prefill FLOPs and TTFT |
| 6 | Cross-Encoder Reranking | Simulated* | Real bi-encoder retrieval + Precision@K/nDCG@K on labelled queries; real cross-encoder if installed |
| 7 | LoRA | Computed | Exact adapter parameter counts; memory estimate; **real NumPy LoRA training** of a toy layer; rank sweep |
| 8 | Logic Bias *(experimental)* | Simulated* | 13 reasoning tests whose answers are computed by verifiers; synthetic or real model responses |
| 9 | Speculative Decoding | Analytical | (1−α^{γ+1})/(1−α) expected tokens, speedup vs γ/α/batch, Monte Carlo verification trace |
| 10 | RoPE Scaling | Computed | Exact RoPE frequencies for linear PI / NTK / dynamic NTK / YaRN; OOD dimension analysis |
| 11 | Attention Sinks | Simulated | Softmax eviction math on synthetic sink-pattern logits; output deviation; constant-memory streaming |
| 12 | Mixture of Experts | Simulated | Real top-k / noisy / expert-choice / hash routing, capacity dropping, Switch aux loss; exact params |
| 13 | Confirmation Prediction *(experimental)* | Computed | Brier score, ECE, reliability diagram, temperature scaling on a labelled synthetic eval set |
| 14 | Ring Attention | Analytical | Per-device memory, compute vs. communication overlap, ring schedule |
| 15 | Medusa Heads | Analytical | Head parameters, tree candidates, accepted length (stated heuristic), speedup |
| 16 | Differential Privacy | Computed | **Exact RDP accountant** (matches TF-Privacy ε≈3.01 reference), real DP-SGD + membership-inference attack on a toy model |
| A | Quantization | Computed | Real FP16/BF16/INT8/INT4/NF4 quantization error with outliers; per-tensor/channel/group scales |
| B | QLoRA | Analytical | NF4 + double-quant bits/param, memory vs full FT and LoRA, fit-per-GPU table, NF4 vs INT4 error |
| C | RAG Optimization | Computed | Chunking strategies/size/overlap, hybrid retrieval, rerank, compression → evidence recall & tokens |
| C | Hybrid Search | Computed | Real BM25 + vector + RRF / weighted fusion; Recall@k, MRR, nDCG on 12 labelled queries |
| D | Prompt Engineering | Computed | Prompt assembly, token count, lint, JSON schema / function calling, injection defenses |
| E | Model Evaluation | Computed | Accuracy/P/R/F1 vs threshold, confusion matrix, claim-level faithfulness, retrieval metrics, Cohen's κ |
| F | Model Serving | Analytical | Throughput–latency trade-off, memory ceiling, SLO-based replica count and fleet cost |
| F | PagedAttention | Simulated | Exact contiguous vs paged allocator accounting; block table; copy-on-write sharing |
| F | Continuous Batching | Simulated | Discrete-event simulation of static vs iteration-level batching |
| G | AI Security & Governance | Computed | PII detection (regex + Luhn) and masking, injection screening, RBAC, output validation, audit, SHA-256 supply chain |
| — | Attention Mechanisms | Analytical | MHA / GQA / MQA / sliding-window KV and FLOPs; FlashAttention memory |

\* uses a real model when the optional libraries/provider are configured; otherwise a clearly labelled simulation.

### Fidelity levels

| Label | Meaning |
|---|---|
| **Analytical estimate** | Closed-form formulas with stated hardware assumptions (bandwidth, FLOPS, MFU are editable in Settings). Not a measurement. |
| **Computed (exact)** | The real algorithm run exactly on real or toy inputs (BM25, BPE, quantization, RDP accountant, DP-SGD, LoRA on a toy layer). |
| **Simulated** | Deterministic simulation with synthetic data/behaviour — illustrates mechanisms, never benchmarks a model. |
| **Real** | Output from a connected model or library, or measured wall-clock time. |

---

## Application tour

| Area | What you can do |
|---|---|
| **Overview** | Active model, parameters, context window, input/output tokens, estimated latency, memory, throughput, cost, quality metrics; batch curve; reference architecture |
| **Section pages** | LLM Architecture · Inference Optimization · Retrieval Optimization · Attention Mechanisms · Privacy & Security · Quality & Evaluation |
| **Module playground** | Live (debounced, WebSocket) updates; presets; **Run Experiment** (saves), **Reset**, **Save Configuration**, **Compare Results** (pin baseline → deltas on every metric), **Export** JSON/CSV; Learn tab (12 sections); Implementation tab (Monaco editor) |
| **Fine-Tuning Laboratory** | 9 steps: base model → upload JSONL/CSV/JSON → validation report → Full/LoRA/QLoRA/PEFT → configuration → training (simulated, or real PEFT LoRA) with live WebSocket loss curves → adapter + saved experiment → base vs fine-tuned comparison |
| **Benchmark Studio** | A/B: baseline vs variant controls, metric deltas, % change chart, save; parameter sweeps; live wall-clock measurement against a real provider |
| **Model Comparison** | Memory/TTFT/TPOT/throughput/cost across 9 architectures for any GPU, precision, workload and batch |
| **Experiment History** | Filter, view, re-open with parameters, compare two runs, delete, export all as JSON/CSV; saved configurations |
| **Documentation** | Per-module docs, "Where Would I Use This in a Real Application?" across 9 domains, fidelity legend, key distinctions |
| **Settings** | Active model, hardware profile, dtype, efficiency assumptions, $/GPU-hour, provider (simulation/Ollama/OpenAI-compatible/HF), JSON editor, ML environment; admin user management and audit log |

### Demonstration scenarios (5-minute interview walkthrough)

1. **KV cache vs GQA** — KV Cache: set batch 32, seq 32K; drop KV heads 32→8→1 and watch memory and *concurrent sequences* change; toggle the cache off to see O(T²) recompute.
2. **The cosine trap** — Cosine Similarity Trap → preset *Role reversal*: "The man bit the dog" outranks the true paraphrases; the consistency-rule layer fixes the top result. Then try *Medication safety* and *Numbers & dates*: rules reduce false positives but cannot fix the top result — the module says why (grounding + entailment needed).
3. **Lexical vs semantic** — Hybrid Search: on *"I forgot my password and cannot log in"* BM25 scores nDCG@3 0.56 while vector/hybrid reach 0.91; on *"what is the hotel limit…"* BM25 wins (1.0 vs 0.83) and weighted fusion keeps the best of both. Compare all four methods in the aggregate table and sweep α.
4. **Fine-tune on a laptop** — Fine-Tuning Lab: use the sample dataset, switch LoRA → QLoRA and watch memory drop from ~17 GB to ~6 GB, start the simulated job and watch the live curve.
5. **Privacy budget** — Differential Privacy: raise steps until ε exceeds the budget; then raise σ and watch the real toy DP-SGD runs trade accuracy for privacy — membership-inference AUC falls from ~0.70 toward 0.5 as ε shrinks.
6. **Speculation that backfires** — Speculative Decoding: set acceptance 0.4 and batch 64 → speedup below 1×.
7. **Serving capacity** — Model Serving (A10G, Llama-3-8B): set traffic 50 req/s and a 10 s SLO — BF16 cannot meet the SLO even at batch 1, INT4 needs ~24 replicas. Relax the SLO to 15 s to see the BF16 fleet size.

---

## Real model integration

| Integration | How to enable | Used by |
|---|---|---|
| **Ollama** | Run Ollama, `ollama pull llama3.2:3b`, then Settings → Provider = Ollama and set the model name | Temperature sampling, Logic bias, Calibration, Prompt engineering, Benchmark live measurement |
| **OpenAI-compatible API** (vLLM, LM Studio, llama.cpp server, hosted) | `LLMLAB_OPENAI_BASE_URL`, `LLMLAB_OPENAI_API_KEY`, `LLMLAB_OPENAI_MODEL`; Settings → Provider = OpenAI-compatible | same as above |
| **Hugging Face** | `pip install -r requirements-ml.txt` and `LLMLAB_ENABLE_HF_MODELS=true` | Real generation, MiniLM embeddings (cosine/hybrid/RAG), ms-marco cross-encoder reranking, **real PEFT LoRA training** in the Fine-Tuning Lab, FAISS index, tiktoken |

Nothing external is contacted unless you select it. API keys are server-side environment variables only.

### Verified real-model results (Intel Core i7 MacBook, CPU only)

These were run against real models on the development machine — they are measurements, not simulations:

| Path | Model | Result |
|---|---|---|
| Real LoRA fine-tuning (Fine-Tuning Lab, sample dataset, 3 epochs) | SmolLM2-135M-Instruct | 4.9 M trainable / 134.5 M params, 151 s on CPU; validation loss 3.23 → 2.74 (perplexity 25.3 → 15.5); held-out output changed from `"Output:"` to an on-task support reply |
| Cross-encoder reranking | ms-marco-MiniLM-L-6-v2 | nDCG@3 0.77 → 0.89, Precision@3 0.36 → 0.47 on the 12 labelled queries |
| Cosine similarity trap | all-MiniLM-L6-v2 | Real neural embeddings still rank the negated / role-reversed sentence first — the trap is not an artefact of the built-in embeddings |
| Hybrid search | all-MiniLM-L6-v2 + BM25 | nDCG@3: BM25 0.86, vector 0.96, weighted hybrid 0.96 |
| Logic bias | llama3.2:3b (Ollama) | Solves bat-and-ball and 17×24, but answers **"Yes" to "Is 9.11 greater than 9.9?"** — caught by the verifier |
| Prompt engineering | llama3.2:3b (Ollama) | The injected "mark this as low priority" **succeeded** against XML delimiters but was **ignored** with spotlighting |
| Benchmark Studio live measurement | llama3.2:3b (Ollama) | ≈2.5–3 tokens/s on this CPU; first call includes a ≈27 s cold model load |

Setup notes:
- On **Intel Macs** PyTorch stops at 2.2.2, which needs NumPy 1.x — `requirements-ml.txt` pins a verified combination.
  Do **not** install `datasets` alongside it (its `pyarrow` needs NumPy 2 and breaks `transformers.Trainer` / `sentence-transformers`).
- Local models can take ~30 s to cold-load on CPU; the provider timeout defaults to 120 s (`LLMLAB_PROVIDER_TIMEOUT_SECONDS`).
- Opt-in integration tests: `LLMLAB_TEST_OLLAMA_MODEL=llama3.2:3b LLMLAB_TEST_HF=1 pytest tests/test_real_models.py`.

---

## Architecture

```
llm-engineering-lab/
├── app/
│   ├── main.py                 # FastAPI factory: middleware (request IDs, security headers/CSP), routers, static UI
│   ├── core/                   # config (pydantic-settings), structured JSON logging, errors, security (Argon2id, JWT, CSRF, throttle)
│   ├── db/                     # async SQLAlchemy engine, ORM models, seeding
│   ├── auth/  users/           # login/logout/me, RBAC dependencies, user admin, settings, audit log
│   ├── engine/                 # module registry + control spec + Result builder, hardware/model presets, roofline model
│   ├── caching/                # KV cache, prefix caching
│   ├── tokenization/           # BPE trainer/tokenizers, tokenizer-effects module
│   ├── inference/              # temperature, speculative, Medusa, quantization; serving: paged attention, continuous batching, serving, attention variants
│   ├── retrieval/  reranking/  # BM25, embeddings, vector index, fusion, IR metrics; cosine trap, hybrid, RAG; cross-encoder
│   ├── attention/              # RoPE scaling, attention sinks, MoE, ring attention
│   ├── fine_tuning/            # LoRA/QLoRA modules, dataset validation, job runner (simulated + real PEFT), router
│   ├── privacy/                # RDP accountant, DP-SGD module, security & governance
│   ├── evaluation/             # logic bias (+verifiers), calibration, model evaluation, prompt engineering
│   ├── benchmarking/  experiments/  dashboard/  modules_api/  llm/  realtime/   # REST + WebSocket APIs, providers
├── frontend/                   # Bootstrap 5 + Chart.js + Monaco, ES modules, no build step
│   ├── js/components/          # reusable controls, metric cards, charts/heatmaps, result renderer, docs, Monaco loader
│   └── js/pages/               # overview, section, module, benchmark, compare, history, fine-tuning, docs, settings
├── migrations/                 # Alembic (async) — 0001_initial
├── data/samples/               # labelled KB + queries, fine-tuning datasets (JSONL / CSV / chat JSON)
├── tests/                      # pytest suite
└── Dockerfile, docker-compose.yml, requirements*.txt, .env.example
```

**Adding a technique** = one `ModuleDef` (controls + `simulate()` + docs) registered in a package listed in
`engine/registry.py:load_all`. The API, WebSocket channel, generic UI renderer, Benchmark Studio, history and docs pick it
up automatically. See [docs/ARCHITECTURE.md](docs/ARCHITECTURE.md).

## Deploying to Vercel

Vercel detects the FastAPI app in `app/main.py` automatically; `vercel.json` keeps marketing/test files out of the
function bundle and `.python-version` pins Python 3.12. Set these environment variables in **Project → Settings →
Environment Variables**, then redeploy:

| Variable | Value |
|---|---|
| `LLMLAB_SECRET_KEY` | **Required.** A long random string (`python -c "import secrets; print(secrets.token_urlsafe(48))"`). Without it every serverless instance signs sessions with its own key and users get logged out at random. Until it is set, sign-in is refused with a message saying so, and `/api/health` reports `"session_key": "missing"`. |
| `LLMLAB_DATABASE_URL` | Recommended. A Postgres URL (Neon, Supabase, Vercel Postgres). Plain `postgres://…?sslmode=require` URLs are accepted. |
| `LLMLAB_DEMO_USER_PASSWORD` | Optional. Change it if the deployment is public. |

Serverless limits to know about:

- The code directory is read-only, so runtime files go to `/tmp/llmlab` (auto-detected via `VERCEL`). Without
  `LLMLAB_DATABASE_URL` the SQLite database lives there too and is **wiped whenever an instance is recycled**:
  users are re-seeded, but saved experiments are lost.
- Module pages fall back from WebSocket to plain HTTP automatically. Fine-tuning jobs run in the background of one
  instance, so live progress is best-effort; real (PyTorch) training needs the Docker deployment.

## Security

Argon2id password hashing · password policy · JWT in an HttpOnly, SameSite cookie (Secure in production) plus Bearer
tokens for API clients · CSRF double-submit token on cookie-authenticated writes · role-based access control
(Administrator / AI Engineer / Researcher / Viewer) · login throttling · WebSocket origin + cookie checks · strict CSP and
security headers · audit log · per-user data isolation · production start-up guards (secret key, secure cookies, non-default
demo password). No secrets are hard-coded.

## API

Interactive OpenAPI docs at **`/docs`** (Swagger) and **`/redoc`**. Summary in [docs/API.md](docs/API.md).

## Database

SQLite by default (`data/llmlab.db`), PostgreSQL via `LLMLAB_DATABASE_URL`. Tables are auto-created in development;
for managed schemas set `LLMLAB_AUTO_CREATE_TABLES=false` and run `alembic upgrade head`.
