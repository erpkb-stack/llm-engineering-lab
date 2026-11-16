"""Dataset upload parsing, schema detection and validation for the Fine-Tuning Laboratory."""
from __future__ import annotations

import csv
import hashlib
import io
import json
import uuid
from pathlib import Path

from app.core.config import RUNTIME_DIR
from app.core.errors import ParameterError

UPLOAD_DIR = Path(RUNTIME_DIR) / "uploads"
SCHEMAS = {
    "chat": "messages: [{role, content}, …]",
    "alpaca": "instruction, (input), output",
    "prompt_completion": "prompt, completion",
    "qa": "question, answer",
    "text": "text (raw causal-LM)",
}


def parse(raw: bytes, filename: str) -> tuple[list, list[dict]]:
    """Returns (records, parse_errors)."""
    name = filename.lower()
    try:
        text = raw.decode("utf-8-sig")
    except UnicodeDecodeError as exc:
        raise ParameterError("File must be UTF-8 encoded") from exc
    errors: list[dict] = []
    if name.endswith(".jsonl"):
        recs = []
        for i, line in enumerate(text.splitlines(), 1):
            if not line.strip():
                continue
            try:
                recs.append(json.loads(line))
            except json.JSONDecodeError as exc:
                errors.append({"row": i, "error": f"invalid JSON: {exc.msg}"})
        return recs, errors
    if name.endswith(".json"):
        try:
            data = json.loads(text)
        except json.JSONDecodeError as exc:
            raise ParameterError(f"Invalid JSON: {exc.msg} at line {exc.lineno}") from exc
        if isinstance(data, dict):
            data = data.get("data") or data.get("train") or data.get("examples") or []
        if not isinstance(data, list):
            raise ParameterError("JSON must be an array of records (or {\"data\": [...]})")
        return data, errors
    if name.endswith(".csv"):
        return list(csv.DictReader(io.StringIO(text))), errors
    raise ParameterError("Supported formats: .jsonl, .json, .csv")


def detect_schema(rec: dict) -> str | None:
    if not isinstance(rec, dict):
        return None
    keys = {k.lower() for k in rec}
    if "messages" in keys:
        return "chat"
    if {"instruction", "output"} <= keys:
        return "alpaca"
    if {"prompt", "completion"} <= keys:
        return "prompt_completion"
    if {"question", "answer"} <= keys:
        return "qa"
    if "text" in keys:
        return "text"
    return None


def normalize(rec: dict, schema: str) -> tuple[dict | None, str | None]:
    r = {k.lower(): v for k, v in rec.items()}
    if schema == "chat":
        msgs = r.get("messages")
        if not isinstance(msgs, list) or not msgs:
            return None, "messages must be a non-empty list"
        for m in msgs:
            if not isinstance(m, dict) or m.get("role") not in ("system", "user", "assistant") or not str(m.get("content", "")).strip():
                return None, "each message needs role ∈ {system,user,assistant} and non-empty content"
        if msgs[-1]["role"] != "assistant":
            return None, "last message must be from the assistant (the training target)"
        prompt = "\n".join(f"{m['role']}: {m['content']}" for m in msgs[:-1])
        return {"prompt": prompt, "completion": str(msgs[-1]["content"])}, None
    if schema == "alpaca":
        ins, out = str(r.get("instruction") or "").strip(), str(r.get("output") or "").strip()
        if not ins or not out:
            return None, "instruction and output must be non-empty"
        inp = str(r.get("input") or "").strip()
        return {"prompt": ins + (f"\n\nInput: {inp}" if inp else ""), "completion": out}, None
    if schema in ("prompt_completion", "qa"):
        a, b = ("prompt", "completion") if schema == "prompt_completion" else ("question", "answer")
        pa, pb = str(r.get(a) or "").strip(), str(r.get(b) or "").strip()
        if not pa or not pb:
            return None, f"{a} and {b} must be non-empty"
        return {"prompt": pa, "completion": pb}, None
    t = str(r.get("text") or "").strip()
    return ({"prompt": "", "completion": t}, None) if t else (None, "text must be non-empty")


def approx_tokens(s: str) -> int:
    return int(len(s.split()) * 1.33) + 1


def validate(raw: bytes, filename: str, user_id: int, max_seq_len: int = 2048) -> dict:
    recs, errors = parse(raw, filename)
    if not recs:
        raise ParameterError("No records found in file", details={"parse_errors": errors[:20]})
    schemas = [detect_schema(r) for r in recs[:50]]
    schema = max(set(s for s in schemas if s), key=schemas.count, default=None)
    if schema is None:
        raise ParameterError("Could not detect a supported schema", details={"supported": SCHEMAS, "example_keys": list(recs[0].keys()) if isinstance(recs[0], dict) else []})
    clean, seen, dups, too_long, lengths = [], set(), 0, 0, []
    for i, rec in enumerate(recs, 1):
        if detect_schema(rec) != schema:
            errors.append({"row": i, "error": f"record does not match schema '{schema}'"})
            continue
        norm, err = normalize(rec, schema)
        if err:
            errors.append({"row": i, "error": err})
            continue
        h = hashlib.sha1((norm["prompt"] + "\x00" + norm["completion"]).encode()).hexdigest()
        if h in seen:
            dups += 1
            continue
        seen.add(h)
        n = approx_tokens(norm["prompt"]) + approx_tokens(norm["completion"])
        if n > max_seq_len:
            too_long += 1
        lengths.append(n)
        clean.append(norm)
    if not clean:
        raise ParameterError("No valid records after validation", details={"errors": errors[:50]})
    ds_id = uuid.uuid4().hex[:12]
    out_dir = UPLOAD_DIR / str(user_id)
    out_dir.mkdir(parents=True, exist_ok=True)
    (out_dir / f"{ds_id}.jsonl").write_text("\n".join(json.dumps(c) for c in clean))
    lengths.sort()
    n_val = max(1, len(clean) // 10) if len(clean) >= 10 else 0
    return {
        "dataset_id": ds_id, "filename": filename, "schema": schema, "schema_description": SCHEMAS[schema],
        "total_records": len(recs), "valid_records": len(clean), "invalid_records": len(errors), "duplicates_removed": dups,
        "exceeding_max_seq_len": too_long, "max_seq_len": max_seq_len,
        "tokens": {"mean": sum(lengths) / len(lengths), "p50": lengths[len(lengths) // 2], "p95": lengths[int(len(lengths) * 0.95) - 1 if len(lengths) > 1 else 0],
                   "max": lengths[-1], "total": sum(lengths)},
        "split": {"train": len(clean) - n_val, "validation": n_val},
        "errors": errors[:50], "preview": clean[:3],
        "valid": len(errors) == 0 and too_long == 0,
    }


def load_dataset(user_id: int, ds_id: str) -> list[dict]:
    if not ds_id.isalnum():
        raise ParameterError("Invalid dataset id")
    path = UPLOAD_DIR / str(user_id) / f"{ds_id}.jsonl"
    if not path.exists():
        raise ParameterError("Dataset not found — upload and validate it first")
    return [json.loads(line) for line in path.read_text().splitlines() if line.strip()]
