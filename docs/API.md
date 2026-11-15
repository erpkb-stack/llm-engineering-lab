# API reference (summary)

Full interactive documentation: `GET /docs` (Swagger UI) and `GET /redoc`.

Authentication: `POST /api/auth/login` sets an HttpOnly session cookie and returns an `access_token`.
Browser clients send the `llmlab_csrf` cookie value as `X-CSRF-Token` on POST/PUT/PATCH/DELETE.
API clients may instead send `Authorization: Bearer <access_token>` (no CSRF header needed).

Errors use one shape: `{"error": {"code", "message", "details", "request_id"}}`.

| Method | Path | Permission | Description |
|---|---|---|---|
| POST | `/api/auth/login` | — | `{email, password, remember_me}` → user, token, csrf |
| POST | `/api/auth/logout` | any | Clear session |
| GET | `/api/auth/me` | any | Current user, role, permissions |
| GET | `/api/modules` | any | Module catalog + categories |
| GET | `/api/modules/applications` | any | "Where would I use this?" grouped by 9 domains |
| GET | `/api/modules/{id}` | any | Controls spec, docs (12 sections), presets, fidelity |
| POST | `/api/modules/{id}/run` | run | `{params}` → result (metrics, charts, tables, heatmaps, flows, comparison, notes, recommendations, fidelity) |
| WS | `/ws/modules` | run | Send `{module_id, params, request_id}` → `{type: result|error, request_id, result}` |
| GET | `/api/dashboard/summary` | any | Dashboard metrics for the active configuration |
| GET/PUT | `/api/settings` | any / configure | Per-user settings (model, hardware, efficiencies, provider) |
| GET/POST/DELETE | `/api/experiments[/{id}]` | any / save | Experiment history (per user) |
| GET | `/api/experiments/export?format=json|csv` | any | Export |
| GET/POST/DELETE | `/api/configs[/{id}]` | any / save | Saved configurations (validated against the module spec) |
| POST | `/api/benchmarks/compare` | benchmark | `{module_id, baseline_params, variant_params, save}` → deltas |
| POST | `/api/benchmarks/sweep` | benchmark | `{module_id, params, sweep_key, values[]}` |
| POST | `/api/benchmarks/models` | run | Analytical model comparison across architectures |
| POST | `/api/benchmarks/live` | benchmark | Real wall-clock measurement against the configured provider |
| GET | `/api/fine-tuning/environment` | any | Installed ML libraries, accelerator, real-training availability |
| POST | `/api/fine-tuning/datasets/validate` | fine_tune | multipart `file` (.jsonl/.json/.csv) → validation report + `dataset_id` |
| POST | `/api/fine-tuning/datasets/sample` | fine_tune | Use the bundled sample dataset |
| POST | `/api/fine-tuning/plan` | fine_tune | Memory / steps / time estimate |
| POST/GET | `/api/fine-tuning/jobs[/{id}]` | fine_tune / any | Start (simulation or real) and inspect jobs |
| POST | `/api/fine-tuning/jobs/{id}/cancel` | fine_tune | Cancel |
| WS | `/ws/fine-tuning/{id}` | owner | Live progress events |
| GET | `/api/llm/status` | any | Provider availability + ML environment |
| POST | `/api/llm/generate` | run | Direct generation through the active provider |
| GET/POST/PATCH | `/api/users[/{id}]` | manage_users | User administration |
| GET | `/api/audit` | view_audit | Audit log |
| GET | `/api/health` | — | Liveness |

Permissions by role: **Administrator** – everything; **AI Engineer** – run, save, benchmark, fine_tune, configure;
**Researcher** – run, save, benchmark, configure; **Viewer** – run.
