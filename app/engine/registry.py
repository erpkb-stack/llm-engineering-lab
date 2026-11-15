"""Module registry, control specification, parameter validation and result builder.

A *module* is a self-contained concept demo: metadata + documentation + a list of
typed controls + a pure ``simulate(params, ctx) -> Result`` function. The frontend
renders any module generically from this contract, so adding a technique means adding
one ``ModuleDef`` — no frontend or router changes.
"""
from __future__ import annotations

import time
from collections.abc import Callable
from dataclasses import dataclass, field
from typing import Any

from app.core.errors import NotFoundError, ParameterError

CATEGORIES = {
    "inference": "Inference & Decoding",
    "architecture": "Architecture & Attention",
    "training": "Fine-Tuning & Training",
    "retrieval": "RAG & Retrieval",
    "quality": "Quality, Safety & Governance",
    "serving": "Model Serving",
}

# Fidelity levels shown on every result so simulated numbers are never mistaken for benchmarks.
FIDELITY = {
    "analytical": "Analytical estimate from closed-form formulas and stated hardware assumptions — not a measurement.",
    "computed": "Exact computation of the algorithm on the given inputs (real math, toy scale).",
    "simulated": "Deterministic simulation with synthetic data — illustrates behavior, not a real model's output.",
    "real": "Executed against a real model/library in this environment.",
}


@dataclass
class Control:
    key: str
    label: str
    type: str  # slider | select | toggle | text | textarea | number
    default: Any
    min: float | None = None
    max: float | None = None
    step: float | None = None
    options: list[dict] | None = None
    help: str = ""
    group: str = "Parameters"
    unit: str = ""
    log: bool = False  # slider in log scale (frontend hint)
    max_length: int = 20000

    def to_dict(self) -> dict:
        return {k: v for k, v in self.__dict__.items()}

    def coerce(self, value: Any) -> Any:
        if self.type in ("slider", "number"):
            try:
                num = float(value)
            except (TypeError, ValueError) as exc:
                raise ParameterError(f"'{self.key}' must be numeric") from exc
            if self.min is not None and num < self.min - 1e-12:
                raise ParameterError(f"'{self.key}'={num} is below minimum {self.min}")
            if self.max is not None and num > self.max + 1e-12:
                raise ParameterError(f"'{self.key}'={num} exceeds maximum {self.max}")
            is_int = isinstance(self.default, int) and not isinstance(self.default, bool) and (self.step or 1) >= 1
            return int(round(num)) if is_int else num
        if self.type == "toggle":
            if isinstance(value, str):
                return value.lower() in ("1", "true", "yes", "on")
            return bool(value)
        if self.type == "select":
            allowed = [o["value"] for o in (self.options or [])]
            for a in allowed:
                if value == a or str(value) == str(a):
                    return a
            raise ParameterError(f"'{self.key}' must be one of {allowed}")
        if self.type == "multiselect":
            allowed = {o["value"] for o in (self.options or [])}
            vals = list(value) if isinstance(value, (list, tuple)) else [value]
            bad = [v for v in vals if v not in allowed]
            if bad:
                raise ParameterError(f"'{self.key}' has invalid values {bad}")
            return vals
        text = "" if value is None else str(value)
        if len(text) > self.max_length:
            raise ParameterError(f"'{self.key}' exceeds {self.max_length} characters")
        return text


def slider(key, label, default, lo, hi, step=1, help="", **kw) -> Control:
    return Control(key, label, "slider", default, lo, hi, step, help=help, **kw)


def select(key, label, default, options, help="", **kw) -> Control:
    opts = [o if isinstance(o, dict) else {"value": o, "label": str(o)} for o in options]
    return Control(key, label, "select", default, options=opts, help=help, **kw)


def toggle(key, label, default, help="", **kw) -> Control:
    return Control(key, label, "toggle", default, help=help, **kw)


def text(key, label, default, help="", multiline=False, **kw) -> Control:
    return Control(key, label, "textarea" if multiline else "text", default, help=help, **kw)


def multiselect(key, label, default, options, help="", **kw) -> Control:
    opts = [o if isinstance(o, dict) else {"value": o, "label": str(o)} for o in options]
    return Control(key, label, "multiselect", default, options=opts, help=help, **kw)


@dataclass
class Context:
    """Per-request execution context injected into simulate functions."""

    settings: dict
    providers: Any = None  # app.llm.providers.ProviderHub


class Result:
    """Fluent builder for the uniform result payload consumed by the frontend."""

    def __init__(self, fidelity: str = "analytical"):
        self.data: dict = {"fidelity": fidelity, "fidelity_note": FIDELITY[fidelity], "metrics": [], "charts": [],
                           "tables": [], "tokens": [], "flows": [], "heatmaps": [], "notes": [], "comparison": None,
                           "recommendations": [], "panels": []}

    def set_fidelity(self, fidelity: str, note: str | None = None) -> "Result":
        self.data["fidelity"] = fidelity
        self.data["fidelity_note"] = note or FIDELITY[fidelity]
        return self

    def metric(self, key, label, value, unit="", hint="", fmt=None, status=None) -> "Result":
        self.data["metrics"].append({"key": key, "label": label, "value": _clean(value), "unit": unit, "hint": hint,
                                     "fmt": fmt, "status": status})
        return self

    def chart(self, id, title, kind, labels, datasets, x_label="", y_label="", **opts) -> "Result":
        self.data["charts"].append({"id": id, "title": title, "kind": kind, "labels": _clean(labels),
                                    "datasets": _clean(datasets), "x_label": x_label, "y_label": y_label, **opts})
        return self

    def table(self, title, columns, rows, **opts) -> "Result":
        self.data["tables"].append({"title": title, "columns": columns, "rows": _clean(rows), **opts})
        return self

    def tokens(self, title, items, **opts) -> "Result":
        self.data["tokens"].append({"title": title, "items": items, **opts})
        return self

    def flow(self, title, nodes, **opts) -> "Result":
        """nodes: list of {label, sub?, cls?}; rendered as a left-to-right pipeline diagram."""
        self.data["flows"].append({"title": title, "nodes": nodes, **opts})
        return self

    def heatmap(self, title, matrix, x_labels=None, y_labels=None, **opts) -> "Result":
        self.data["heatmaps"].append({"title": title, "matrix": _clean(matrix), "x_labels": x_labels,
                                      "y_labels": y_labels, **opts})
        return self

    def panel(self, kind, title, **payload) -> "Result":
        self.data["panels"].append({"kind": kind, "title": title, **_clean(payload)})
        return self

    def note(self, text_, level="info") -> "Result":
        self.data["notes"].append({"level": level, "text": text_})
        return self

    def compare(self, before_label, after_label, rows, title="Before vs. after") -> "Result":
        """rows: (metric, before, after, unit, better) where better in {'lower','higher',None}."""
        self.data["comparison"] = {
            "title": title, "before_label": before_label, "after_label": after_label,
            "rows": [{"metric": r[0], "before": _clean(r[1]), "after": _clean(r[2]), "unit": r[3] if len(r) > 3 else "",
                      "better": r[4] if len(r) > 4 else None} for r in rows],
        }
        return self

    def recommend(self, *items: str) -> "Result":
        self.data["recommendations"].extend(items)
        return self


def _clean(v):
    """Make values JSON-safe (numpy scalars, NaN/inf, tuples)."""
    import math

    try:
        import numpy as np
    except ImportError:  # pragma: no cover
        np = None
    if isinstance(v, dict):
        return {k: _clean(x) for k, x in v.items()}
    if isinstance(v, (list, tuple)):
        return [_clean(x) for x in v]
    if np is not None:
        if isinstance(v, np.ndarray):
            return _clean(v.tolist())
        if isinstance(v, np.generic):
            v = v.item()
    if isinstance(v, float):
        if math.isnan(v) or math.isinf(v):
            return None
        return round(v, 6) if abs(v) < 1e15 else v
    return v


@dataclass
class ModuleDef:
    id: str
    title: str
    category: str
    summary: str
    controls: list[Control]
    simulate: Callable[[dict, Context], Result]
    docs: dict
    fidelity: str = "analytical"
    experimental: bool = False
    original_index: int | None = None  # position in the 16 original modules
    tags: list[str] = field(default_factory=list)
    presets: list[dict] = field(default_factory=list)  # [{"name": ..., "params": {...}}]

    def defaults(self) -> dict:
        return {c.key: c.default for c in self.controls}

    def validate(self, params: dict | None) -> dict:
        params = params or {}
        known = {c.key for c in self.controls}
        unknown = set(params) - known
        if unknown:
            raise ParameterError(f"Unknown parameters for {self.id}: {sorted(unknown)}")
        return {c.key: c.coerce(params.get(c.key, c.default)) for c in self.controls}

    def run(self, params: dict | None, ctx: Context) -> dict:
        clean = self.validate(params)
        t0 = time.perf_counter()
        result = self.simulate(clean, ctx)
        out = result.data
        out.update(module=self.id, params=clean, compute_ms=round((time.perf_counter() - t0) * 1e3, 2))
        return out

    def summary_dict(self) -> dict:
        return {"id": self.id, "title": self.title, "category": self.category, "category_label": CATEGORIES[self.category],
                "summary": self.summary, "fidelity": self.fidelity, "experimental": self.experimental,
                "original_index": self.original_index, "tags": self.tags}

    def detail_dict(self) -> dict:
        d = self.summary_dict()
        d.update(controls=[c.to_dict() for c in self.controls], docs=self.docs, fidelity_note=FIDELITY[self.fidelity],
                 presets=self.presets)
        return d


_REGISTRY: dict[str, ModuleDef] = {}


def register(module: ModuleDef) -> ModuleDef:
    if module.id in _REGISTRY:
        raise ValueError(f"duplicate module id {module.id}")
    if module.category not in CATEGORIES:
        raise ValueError(f"unknown category {module.category}")
    _REGISTRY[module.id] = module
    return module


def get_module(module_id: str) -> ModuleDef:
    load_all()
    try:
        return _REGISTRY[module_id]
    except KeyError as exc:
        raise NotFoundError(f"Unknown module '{module_id}'") from exc


def all_modules() -> list[ModuleDef]:
    load_all()
    return sorted(_REGISTRY.values(), key=lambda m: (m.original_index or 100, m.title))


_loaded = False


def load_all() -> None:
    """Import every module package once so their ``register`` calls run."""
    global _loaded
    if _loaded:
        return
    _loaded = True
    import importlib

    for name in ("app.caching.modules", "app.tokenization.modules", "app.inference.modules", "app.retrieval.modules",
                 "app.reranking.modules", "app.fine_tuning.modules", "app.attention.modules", "app.privacy.modules",
                 "app.evaluation.modules", "app.inference.serving"):
        importlib.import_module(name)


def docs(*, what, why, how, when_use, when_avoid, pros, cons, use_case, code, diagram, performance, checklist,
         applications, formula: str | None = None, limitations: str | None = None, references: list[str] | None = None) -> dict:
    """Uniform documentation schema — the 12 required sections plus extras."""
    return {"what": what, "why": why, "how": how, "when_use": when_use, "when_avoid": when_avoid, "pros": pros,
            "cons": cons, "use_case": use_case, "code": code.strip("\n"), "diagram": diagram, "performance": performance,
            "checklist": checklist, "applications": applications, "formula": formula, "limitations": limitations,
            "references": references or []}
