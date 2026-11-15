# Architecture

## Request flow

```
Browser (ES modules, Bootstrap, Chart.js, Monaco)
   │  REST (cookie + CSRF)            WebSocket /ws/modules (live parameter updates)
   ▼
FastAPI ── middleware: request-id, security headers/CSP, structured access log
   │
   ├─ auth.deps.current_principal / require(perm)      ← dependency-injected RBAC
   ├─ modules_api.execute(module, params, settings)    ← validation → simulate() (threadpool or async) → Result
   │       └─ engine.registry.ModuleDef
   │             controls: [Control]  (typed spec: default/min/max/step/options/help)
   │             simulate(params, Context) -> Result   (pure, deterministic, seeded)
   │             docs: 12 sections + applications
   ├─ benchmarking (A/B deltas, sweeps, model comparison, live provider timing)
   ├─ fine_tuning (dataset validation, jobs → JobBus → /ws/fine-tuning/{id})
   ├─ llm.providers.ProviderHub (simulation | Ollama | OpenAI-compatible | Hugging Face)
   └─ db (async SQLAlchemy: users, settings, configs, experiments, jobs, audit)
```

## The module contract

```python
register(ModuleDef(
    id="my-technique", title="My Technique", category="inference", fidelity="analytical",
    summary="One line.",
    controls=[slider("x", "X", 4, 1, 16, 1, "What X does."), toggle("on", "Enable", True, "…")],
    simulate=my_sim,            # (params: dict, ctx: Context) -> Result   (may be async)
    docs=docs(what=…, why=…, how=…, when_use=…, when_avoid=…, pros=[…], cons=[…], use_case=…,
              code="python example", diagram=[{"label": …}], performance=…, checklist=[…],
              applications=[{"domain": "Enterprise chatbots", "text": …}]),
))
```

`Result` is a builder for a uniform payload: `metric()`, `chart()`, `table()`, `tokens()`, `flow()`, `heatmap()`,
`panel()`, `compare()` (before/after), `note()`, `recommend()`, `set_fidelity()`. The frontend renders any payload
generically, so new modules need no UI code.

## Honesty rules enforced in code

1. Every `Result` has a fidelity (`analytical` / `computed` / `simulated` / `real`) and a human-readable note.
2. Real model output is only labelled `real` when it came from a provider call; provider errors are reported as text, never replaced with made-up output.
3. `/api/benchmarks/live` returns `measured: false` and no numbers when no real provider is configured.
4. Real fine-tuning returns HTTP 409 (`capability_unavailable`) when torch/peft are missing — it never silently simulates.
5. Hardware peak figures and efficiency factors are visible and editable (Settings).

## Analytical performance model (engine/hardware.py)

- Weights bytes = params × bytes(dtype); KV bytes = 2·L·H_kv·d_head·seq·batch·bytes.
- Prefill (TTFT) ≈ max((2·N_active·prompt·batch + 4·L·d·prompt²·batch) / (FLOPS·MFU), weights / (BW·η)).
- Decode (TPOT) ≈ max((weights + KV) / (BW·η), 2·N_active·batch / (FLOPS·MFU)).
- Cost = $/hour × time / tokens.

## Scaling notes

- The login throttle and fine-tuning JobBus are in-process; for multi-replica deployments move them to Redis.
- Real training runs in a worker thread; production deployments should use a job queue (Celery/RQ/Arq) and GPU workers.
- PostgreSQL is supported through `asyncpg`; schema is managed by Alembic.
