"""Fine-tuning job execution with live progress (WebSocket fan-out).

* Simulation mode: transparent analytical loss model + memory/time estimates, streamed live.
* Real mode: Hugging Face Trainer + PEFT LoRA on a small local model (opt-in, CPU/GPU/MPS).
"""
from __future__ import annotations

import asyncio
import logging
import math
from collections import defaultdict
from datetime import datetime, timezone
from pathlib import Path

import numpy as np

from app.core.config import get_settings
from app.db.models import FineTuneJob
from app.db.session import get_sessionmaker
from app.engine.common import rng
from app.engine.hardware import HARDWARE, MODELS
from app.fine_tuning.modules import TARGETS, activation_bytes, lora_params

logger = logging.getLogger("llmlab.finetune")


class JobBus:
    """In-process pub/sub for job progress events (history replayed to late subscribers)."""

    def __init__(self) -> None:
        self.history: dict[int, list[dict]] = defaultdict(list)
        self.subs: dict[int, set[asyncio.Queue]] = defaultdict(set)
        self.cancelled: set[int] = set()

    def publish(self, job_id: int, event: dict) -> None:
        self.history[job_id].append(event)
        if len(self.history[job_id]) > 2000:
            self.history[job_id] = self.history[job_id][-2000:]
        for q in list(self.subs[job_id]):
            q.put_nowait(event)

    def subscribe(self, job_id: int) -> asyncio.Queue:
        q: asyncio.Queue = asyncio.Queue()
        for e in self.history.get(job_id, []):
            q.put_nowait(e)
        self.subs[job_id].add(q)
        return q

    def unsubscribe(self, job_id: int, q: asyncio.Queue) -> None:
        self.subs[job_id].discard(q)


bus = JobBus()


def plan(config: dict, n_train: int, mean_tokens: float, hardware: str, comp_eff: float) -> dict:
    a = MODELS[config["base_model"]]
    hw = HARDWARE[hardware]
    N = a.total_params()
    method = config["method"]
    targets = [t["value"] for t in TARGETS] if config.get("target_all_linear", True) else ["q_proj", "v_proj"]
    nl = N if method == "full" else lora_params(a, targets, config["rank"])
    seq = min(config["max_seq_len"], max(16, int(mean_tokens)))
    act = activation_bytes(a, config["batch_size"], seq, True)
    if method == "full":
        mem = N * 16 + act
    elif method == "qlora":
        mem = N * (4.13 / 8) + nl * 14 + act + 1e9
    else:
        mem = N * 2 + nl * 14 + act
    tokens = n_train * mean_tokens * config["epochs"]
    flops = (6 if method == "full" else 4) * a.active_params() * tokens * (1.25 if method == "qlora" else 1.0)
    secs = flops / (hw["tflops"] * 1e12 * comp_eff)
    steps = max(1, math.ceil(n_train / (config["batch_size"] * config["grad_accum"]))) * config["epochs"]
    return {"trainable_params": int(nl), "total_params": int(N), "memory_gb": mem / 1e9, "fits": mem / 1e9 < hw["mem_gb"] * 0.92,
            "est_train_seconds": secs, "steps": steps, "tokens": tokens, "hardware": hw["label"]}


def simulated_curve(config: dict, steps: int, n_train: int, seed: int) -> tuple[list[float], list[float]]:
    """Analytical loss model (documented assumptions, NOT a measurement)."""
    g = rng(seed)
    method, lr = config["method"], config["learning_rate"]
    l0 = 2.4
    floor = {"full": 0.95, "lora": 1.05 + 0.25 / math.sqrt(config["rank"]), "qlora": 1.08 + 0.25 / math.sqrt(config["rank"]), "peft": 1.15}[method]
    good_lr = 2e-5 if method == "full" else 2e-4
    ratio = lr / good_lr
    speed = 0.12 * min(ratio, 3) if ratio < 3 else 0.12
    unstable = ratio > 5
    train, val = [], []
    for t in range(1, steps + 1):
        frac = t / steps
        lt = floor + (l0 - floor) * math.exp(-speed * t * 25 / steps / max(0.15, min(1.0, ratio)))
        if unstable:
            lt += 0.4 * math.sin(t / 3) ** 2 + frac * 0.6 * (ratio / 10)
        lt += g.normal(0, 0.03)
        overfit = max(0.0, frac * config["epochs"] - 1.0) * 0.12 * min(1.0, math.sqrt(500 / max(1, n_train))) * (1 - config["dropout"])
        lv = lt + 0.05 + overfit - config["weight_decay"] * 0.5 * frac
        train.append(round(float(lt), 4))
        val.append(round(float(lv), 4))
    return train, val


async def _finish(job_id: int, status: str, metrics: dict, error: str | None = None, adapter_path: str | None = None) -> None:
    async with get_sessionmaker()() as db:
        job = await db.get(FineTuneJob, job_id)
        if job:
            job.status, job.metrics, job.error, job.adapter_path = status, metrics, error, adapter_path
            job.finished_at = datetime.now(timezone.utc)
            await db.commit()
    bus.publish(job_id, {"type": "status", "status": status, "metrics": metrics, "error": error})


async def run_simulated(job_id: int, config: dict, summary: dict, settings: dict) -> None:
    p = plan(config, summary["split"]["train"], summary["tokens"]["mean"], settings["hardware"], settings["compute_efficiency"])
    steps_total = p["steps"]
    n_events = min(steps_total, 120)
    train, val = simulated_curve(config, n_events, summary["split"]["train"], config.get("seed", 0))
    bus.publish(job_id, {"type": "status", "status": "running", "plan": p, "mode": "simulation"})
    async with get_sessionmaker()() as db:
        job = await db.get(FineTuneJob, job_id)
        job.status = "running"
        await db.commit()
    for i in range(n_events):
        if job_id in bus.cancelled:
            await _finish(job_id, "cancelled", {"plan": p, "train_loss": train[:i], "val_loss": val[:i]})
            return
        step = int((i + 1) / n_events * steps_total)
        bus.publish(job_id, {"type": "progress", "step": step, "total_steps": steps_total, "train_loss": train[i], "val_loss": val[i],
                             "gpu_memory_gb": round(p["memory_gb"] * (0.97 + 0.03 * math.sin(i)), 2),
                             "elapsed_s_est": round(p["est_train_seconds"] * (i + 1) / n_events, 1), "simulated": True})
        await asyncio.sleep(0.04)
    base_ppl = math.exp(2.4)
    metrics = {"plan": p, "train_loss": train, "val_loss": val, "final_train_loss": train[-1], "final_val_loss": val[-1],
               "base_val_ppl": base_ppl, "tuned_val_ppl": math.exp(val[-1]), "simulated": True,
               "comparison": [["Validation loss", 2.45, val[-1]], ["Validation perplexity", base_ppl, math.exp(val[-1])],
                              ["Trainable parameters", p["total_params"], p["trainable_params"]]]}
    path = Path(get_settings().adapters_dir) / f"job-{job_id}-simulated"
    path.mkdir(parents=True, exist_ok=True)
    import json

    (path / "adapter_config.json").write_text(json.dumps({"simulated": True, "peft_type": "LORA" if config["method"] != "full" else None,
                                                           "base_model": MODELS[config["base_model"]].label, **config}, indent=2))
    await _finish(job_id, "completed", metrics, adapter_path=str(path))


async def run_real(job_id: int, config: dict, records: list[dict]) -> None:
    """Real PEFT LoRA training on a small local HF model (runs in a worker thread)."""
    loop = asyncio.get_running_loop()
    s = get_settings()
    out_dir = Path(s.adapters_dir) / f"job-{job_id}"
    async with get_sessionmaker()() as db:
        job = await db.get(FineTuneJob, job_id)
        job.status = "running"
        await db.commit()
    bus.publish(job_id, {"type": "status", "status": "running", "mode": "real", "model": s.hf_model})

    def train() -> dict:
        import time

        import torch
        from peft import LoraConfig, get_peft_model
        from transformers import AutoModelForCausalLM, AutoTokenizer, PrinterCallback, Trainer, TrainerCallback, TrainingArguments

        tok = AutoTokenizer.from_pretrained(s.hf_model)
        tok.pad_token = tok.pad_token or tok.eos_token
        base = AutoModelForCausalLM.from_pretrained(s.hf_model)
        n_val = max(1, len(records) // 10)
        train_r, val_r = records[n_val:], records[:n_val]
        max_len = min(config["max_seq_len"], 512)

        def prompt_text(r: dict) -> str:
            return (r["prompt"] + "\n") if r["prompt"] else ""

        def enc(batch):
            """Tokenize prompt+completion, pad to the longest example, and train on completion tokens only."""
            rows = []
            for r in batch:
                p_ids = tok(prompt_text(r), add_special_tokens=False)["input_ids"]
                c_ids = tok(r["completion"] + tok.eos_token, add_special_tokens=False)["input_ids"]
                ids = (p_ids + c_ids)[:max_len]
                labels = ([-100] * len(p_ids) + c_ids)[:max_len]
                rows.append((ids, labels))
            width = max(len(i) for i, _ in rows)
            out = []
            for ids, labels in rows:
                pad = width - len(ids)
                out.append({"input_ids": torch.tensor(ids + [tok.pad_token_id] * pad), "attention_mask": torch.tensor([1] * len(ids) + [0] * pad),
                            "labels": torch.tensor(labels + [-100] * pad)})
            return out

        train_ds, val_ds = enc(train_r), enc(val_r)
        sample_prompts = [prompt_text(r) for r in val_r[:3]]

        def gen(model):
            model.eval()
            outs = []
            for pr in sample_prompts:
                ids = tok(pr, return_tensors="pt")
                with torch.no_grad():
                    o = model.generate(**ids, max_new_tokens=40, do_sample=False, pad_token_id=tok.pad_token_id)
                outs.append(tok.decode(o[0][ids["input_ids"].shape[1]:], skip_special_tokens=True).strip())
            return outs

        before = gen(base)
        targets = "all-linear" if config.get("target_all_linear", True) else None  # None → PEFT's per-architecture default (e.g. q_proj, v_proj)
        model = get_peft_model(base, LoraConfig(r=config["rank"], lora_alpha=config["alpha"], lora_dropout=config["dropout"],
                                                target_modules=targets, task_type="CAUSAL_LM"))
        trainable = sum(p.numel() for p in model.parameters() if p.requires_grad)
        total = sum(p.numel() for p in model.parameters()) - trainable

        class Progress(TrainerCallback):
            def on_log(self, args, state, control, logs=None, **kw):
                if logs and ("loss" in logs or "eval_loss" in logs):
                    loop.call_soon_threadsafe(bus.publish, job_id, {"type": "progress", "step": state.global_step, "total_steps": state.max_steps,
                                                                    "train_loss": logs.get("loss"), "val_loss": logs.get("eval_loss"), "simulated": False})
                if job_id in bus.cancelled:
                    control.should_training_stop = True

        t0 = time.time()
        args = TrainingArguments(output_dir=str(out_dir / "ckpt"), per_device_train_batch_size=min(config["batch_size"], 8),
                                 per_device_eval_batch_size=8, gradient_accumulation_steps=config["grad_accum"], num_train_epochs=config["epochs"],
                                 learning_rate=config["learning_rate"], weight_decay=config["weight_decay"], logging_steps=1, eval_strategy="epoch",
                                 save_strategy="no", report_to=[], disable_tqdm=True, label_names=["labels"], seed=config.get("seed", 0),
                                 use_cpu=not torch.cuda.is_available())
        trainer = Trainer(model=model, args=args, train_dataset=train_ds, eval_dataset=val_ds, callbacks=[Progress()])
        trainer.remove_callback(PrinterCallback)
        base_eval = trainer.evaluate()["eval_loss"]  # LoRA B=0 at init → identical to the base model
        trainer.train()
        ev = trainer.evaluate()
        model.save_pretrained(str(out_dir))
        after = gen(model)
        hist = trainer.state.log_history
        return {"trainable_params": trainable, "total_params": total, "train_seconds": time.time() - t0, "final_val_loss": ev.get("eval_loss"),
                "base_val_loss": base_eval, "base_val_ppl": math.exp(base_eval), "tuned_val_ppl": math.exp(ev["eval_loss"]),
                "train_loss": [h["loss"] for h in hist if "loss" in h], "val_loss": [h["eval_loss"] for h in hist if "eval_loss" in h],
                "samples": [{"prompt": p_, "base": b, "tuned": a_} for p_, b, a_ in zip(sample_prompts, before, after)],
                "comparison": [["Validation loss (completion tokens)", base_eval, ev["eval_loss"]],
                               ["Validation perplexity", math.exp(base_eval), math.exp(ev["eval_loss"])],
                               ["Trainable parameters", total, trainable]],
                "gpu_memory_gb": (torch.cuda.max_memory_allocated() / 1e9) if torch.cuda.is_available() else None, "simulated": False,
                "model": s.hf_model, "device": "cuda" if torch.cuda.is_available() else "cpu"}

    try:
        metrics = await asyncio.to_thread(train)
        await _finish(job_id, "cancelled" if job_id in bus.cancelled else "completed", metrics, adapter_path=str(out_dir))
    except Exception as exc:
        logger.exception("real_training_failed")
        await _finish(job_id, "failed", {}, error=f"{type(exc).__name__}: {exc}"[:500])
