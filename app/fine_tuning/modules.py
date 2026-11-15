"""Module 7 (LoRA) and QLoRA."""
from __future__ import annotations

import numpy as np

from app.engine.common import rng
from app.engine.hardware import HARDWARE, HARDWARE_OPTIONS, MODEL_OPTIONS, MODELS, ModelArch, fmt_params
from app.engine.registry import Context, ModuleDef, Result, docs, multiselect, register, select, slider, toggle
from app.inference.modules import NF4, quantize

TARGETS = [{"value": v, "label": v} for v in ("q_proj", "k_proj", "v_proj", "o_proj", "gate_proj", "up_proj", "down_proj")]


def module_shapes(a: ModelArch) -> dict[str, tuple[int, int]]:
    kvd = a.kv_heads * a.head_dim
    return {"q_proj": (a.hidden, a.hidden), "k_proj": (a.hidden, kvd), "v_proj": (a.hidden, kvd), "o_proj": (a.hidden, a.hidden),
            "gate_proj": (a.hidden, a.intermediate), "up_proj": (a.hidden, a.intermediate), "down_proj": (a.intermediate, a.hidden)}


def lora_params(a: ModelArch, targets: list[str], r: int) -> int:
    sh = module_shapes(a)
    per_layer = sum(r * (i + o) for t in targets for i, o in [sh[t]])
    return per_layer * a.layers * (a.experts if any(t in ("gate_proj", "up_proj", "down_proj") for t in targets) and a.experts > 1 else 1)


def activation_bytes(a: ModelArch, batch: int, seq: int, checkpointing: bool) -> float:
    # Korthikanti et al. (2022): ~ s·b·h·(34 + 5·a·s/h) bytes per layer in 16-bit without recomputation
    if checkpointing:
        return 2.0 * seq * batch * a.hidden * a.layers + seq * batch * a.hidden * (34 + 5 * a.heads * seq / a.hidden)
    return seq * batch * a.hidden * (34 + 5 * a.heads * seq / a.hidden) * a.layers


def toy_lora_train(rank: int, alpha: float, dropout: float, lr: float, steps: int, batch: int, seed: int, d: int = 64, true_rank: int = 8):
    """Real LoRA training (NumPy, Adam) of W0 + (α/r)·B·A towards W0 + ΔW where rank(ΔW)=true_rank."""
    g = rng(seed)
    W0 = g.normal(0, 1 / np.sqrt(d), (d, d))
    dW = (g.normal(0, 1, (d, true_rank)) @ g.normal(0, 1, (true_rank, d))) / (np.sqrt(d) * true_rank ** 0.5) * 0.5
    Xv = g.normal(0, 1, (256, d))
    Yv = Xv @ (W0 + dW).T
    A = g.normal(0, 1 / np.sqrt(d), (rank, d))
    B = np.zeros((d, rank))
    s = alpha / rank
    mA, vA, mB, vB = (np.zeros_like(A), np.zeros_like(A), np.zeros_like(B), np.zeros_like(B))
    b1, b2, eps = 0.9, 0.999, 1e-8
    train_l, val_l = [], []
    for t in range(1, steps + 1):
        X = g.normal(0, 1, (batch, d))
        Y = X @ (W0 + dW).T + g.normal(0, 0.01, (batch, d))
        Xd = X * (g.random(X.shape) >= dropout) / (1 - dropout) if dropout > 0 else X
        H = Xd @ A.T                      # (b, r)
        pred = X @ W0.T + s * H @ B.T
        err = pred - Y
        loss = float((err ** 2).mean())
        gpred = 2 * err / err.size
        gB = s * gpred.T @ H
        gA = s * (gpred @ B).T @ Xd
        for P, G, m, v in ((A, gA, mA, vA), (B, gB, mB, vB)):
            m[:] = b1 * m + (1 - b1) * G
            v[:] = b2 * v + (1 - b2) * G * G
            P -= lr * (m / (1 - b1 ** t)) / (np.sqrt(v / (1 - b2 ** t)) + eps)
        if t % max(1, steps // 60) == 0 or t == steps:
            train_l.append((t, loss))
            val_l.append((t, float(((Xv @ (W0 + s * B @ A).T - Yv) ** 2).mean())))
    base_val = float(((Xv @ W0.T - Yv) ** 2).mean())
    sv = np.linalg.svd(dW, compute_uv=False)
    return train_l, val_l, base_val, sv


def lora_sim(p: dict, ctx: Context) -> Result:
    a = MODELS[p["model"]]
    targets = p["target_modules"] or ["q_proj", "v_proj"]
    N = a.total_params()
    nl = lora_params(a, targets, p["rank"])
    r = Result("computed")
    r.set_fidelity("computed", "Parameter counts are exact for the chosen architecture; memory is an analytical estimate; the training curve is a REAL "
                               "LoRA optimisation of a 64×64 toy layer in NumPy (not an LLM).")
    act = activation_bytes(a, p["batch_size"], p["seq_len"], p["grad_checkpointing"])
    full_mem = N * 16 + act       # bf16 weights+grads, fp32 master + Adam m,v
    lora_mem = N * 2 + nl * 16 + act
    r.metric("base", "Base model parameters", fmt_params(N), "", "frozen")
    r.metric("trainable", "Trainable LoRA parameters", fmt_params(nl), "", f"{len(targets)} target modules × {a.layers} layers")
    r.metric("pct", "Trainable fraction", nl / N * 100, "%")
    r.metric("scale", "Scaling α / r", p["alpha"] / p["rank"])
    r.metric("adapter_mb", "Adapter file size (BF16)", nl * 2 / 1e6, "MB")
    r.metric("mem_lora", "Est. training memory (LoRA)", lora_mem / 1e9, "GB")
    r.metric("mem_full", "Est. training memory (full FT)", full_mem / 1e9, "GB")
    steps_total = int(np.ceil(p["dataset_size"] / p["batch_size"])) * p["epochs"]
    r.metric("steps", "Optimizer steps", steps_total)
    # real toy training
    toy_steps = int(min(300, max(60, steps_total)))
    toy_lr = p["learning_rate"] * 25
    tr, va, base_val, sv = toy_lora_train(p["rank"], p["alpha"], p["dropout"], toy_lr, toy_steps, min(64, p["batch_size"] * 4), p["seed"])
    r.chart("loss", f"LoRA training on toy layer (rank {p['rank']}, true update rank 8)", "line", [t for t, _ in tr], [
        {"label": "train MSE", "data": [l for _, l in tr]}, {"label": "validation MSE", "data": [l for _, l in va]},
        {"label": "no adaptation (base W0)", "data": [base_val] * len(tr), "borderDash": [5, 5]}], "step", "MSE", log_y=True)
    ranks = [1, 2, 4, 8, 16, 32]
    finals = [toy_lora_train(rk, p["alpha"], p["dropout"], toy_lr, 150, 32, p["seed"])[1][-1][1] for rk in ranks]
    r.chart("rank", "Final validation error vs. LoRA rank (toy)", "line", ranks, [{"label": "val MSE after 150 steps", "data": finals}], "rank r", "MSE", log_y=True)
    r.chart("sv", "Singular values of the true weight update ΔW", "bar", list(range(1, 17)), [{"label": "σ_i", "data": sv[:16].tolist()}], "index", "σ")
    sh = module_shapes(a)
    r.table("Per-module parameters (one layer)", ["Module", "W shape (in×out)", "Full params", f"LoRA params (r={p['rank']})", "Targeted"],
            [[m, f"{i}×{o}", f"{i * o:,}", f"{p['rank'] * (i + o):,}", "✓" if m in targets else ""] for m, (i, o) in sh.items()])
    r.chart("mem", "Training memory breakdown (estimate)", "bar", ["Full fine-tuning", "LoRA"], [
        {"label": "weights", "data": [N * 2 / 1e9, N * 2 / 1e9]}, {"label": "gradients", "data": [N * 2 / 1e9, nl * 2 / 1e9]},
        {"label": "optimizer + master", "data": [N * 12 / 1e9, nl * 12 / 1e9]}, {"label": "activations", "data": [act / 1e9, act / 1e9]}], "", "GB", stacked=True)
    r.panel("formula", "Low-rank update", latex="W' = W + \\frac{\\alpha}{r} B A,\\quad B\\in\\mathbb{R}^{d\\times r}, A\\in\\mathbb{R}^{r\\times k}",
            content=f"W' = W + (α/r)·B·A   with d={a.hidden}, r={p['rank']}: a {a.hidden}×{a.hidden} update costs {2 * a.hidden * p['rank']:,} params instead of {a.hidden ** 2:,}.")
    r.compare("Full fine-tuning", "LoRA", [("Trainable params", N / 1e6, nl / 1e6, "M", "lower"), ("Training memory", full_mem / 1e9, lora_mem / 1e9, "GB", "lower"),
                                           ("Checkpoint size", N * 2 / 1e9, nl * 2 / 1e9, "GB", "lower")])
    fits = [h["label"] for h in HARDWARE.values() if lora_mem / 1e9 < h["mem_gb"] * 0.92]
    r.note("LoRA fits on: " + (", ".join(fits) if fits else "no single listed device — use QLoRA, smaller batch, or gradient checkpointing."),
           "info" if fits else "warn")
    if p["learning_rate"] > 1e-3:
        r.note("Learning rate above 1e-3 is unusually high for LoRA on LLMs (typical 1e-4 – 3e-4).", "warn")
    r.note(f"Toy optimisation uses Adam with lr × 25 (= {toy_lr:.4g}) so the 64-dim problem converges in a few hundred steps.")
    r.recommend("Start with r=8–16, α=2r, all linear layers targeted, lr 1e-4–2e-4.",
                "Rank below the intrinsic rank of the needed update underfits (see the rank chart); very high rank approaches full FT cost.",
                "Keep one base model in memory and hot-swap task adapters (multi-LoRA serving in vLLM/LoRAX).")
    return r


register(ModuleDef(
    id="lora", title="LoRA (Low-Rank Adaptation)", category="training", original_index=7, fidelity="computed",
    summary="Fine-tune by learning a low-rank update BA on frozen weights — a fraction of the parameters and memory.",
    controls=[
        select("model", "Base model", "llama-3-8b", MODEL_OPTIONS, "Architecture for parameter and memory counts."),
        slider("rank", "Rank (r)", 16, 1, 256, 1, "Inner dimension of B (d×r) and A (r×k)."),
        slider("alpha", "Alpha (α)", 32, 1, 512, 1, "Update is scaled by α/r."),
        slider("dropout", "LoRA dropout", 0.05, 0.0, 0.5, 0.01, "Dropout on the adapter input."),
        multiselect("target_modules", "Target modules", ["q_proj", "k_proj", "v_proj", "o_proj"], TARGETS, "Linear layers that receive adapters."),
        slider("learning_rate", "Learning rate", 0.0002, 0.00001, 0.01, 0.00001, "Adam learning rate.", log=True),
        slider("epochs", "Epochs", 3, 1, 10, 1, "Passes over the dataset."),
        slider("batch_size", "Batch size", 8, 1, 64, 1, "Sequences per optimizer step."),
        slider("dataset_size", "Training examples", 5000, 100, 100000, 100, "Dataset size."),
        slider("seq_len", "Sequence length", 1024, 128, 8192, 128, "Tokens per example."),
        toggle("grad_checkpointing", "Gradient checkpointing", True, "Recompute activations to save memory."),
        slider("seed", "Random seed", 1, 0, 9999, 1, "Toy training seed."),
    ],
    simulate=lora_sim,
    docs=docs(
        what="LoRA freezes the pretrained weight W and learns a low-rank correction ΔW = (α/r)·B·A for selected linear layers, where B∈ℝ^{d×r}, A∈ℝ^{r×k} and r ≪ min(d,k).",
        why="Full fine-tuning updates and stores all parameters plus optimizer state (~16 bytes/param). Empirically, task adaptation updates have low intrinsic rank, so a small factorized update captures most of the benefit.",
        how="A is initialised randomly, B to zero (so training starts at the base model). Only A and B receive gradients. At inference the update can be merged into W (no latency cost) or kept separate to hot-swap adapters.",
        when_use="Domain/style/task adaptation with limited GPUs, many task-specific variants of one base model, rapid experimentation.",
        when_avoid="Teaching large amounts of new knowledge or a new language (full FT / continued pretraining is stronger), or when the base model is far from the target behavior.",
        pros=["~0.1–1% trainable parameters", "Much lower memory: no optimizer state for frozen weights", "Tiny adapters (MBs) — many per base model", "Mergeable → zero inference overhead"],
        cons=["Can underperform full FT on complex tasks", "Rank/target/α hyperparameters matter", "Adapter sprawl requires management and versioning"],
        use_case="A customer-support platform keeps one Llama-3-8B base and 40 per-client LoRA adapters (≈40 MB each), served with multi-LoRA batching.",
        code='''
from transformers import AutoModelForCausalLM, AutoTokenizer, TrainingArguments
from peft import LoraConfig, get_peft_model
from trl import SFTTrainer
from datasets import load_dataset

base = AutoModelForCausalLM.from_pretrained("meta-llama/Llama-3.1-8B", torch_dtype="bfloat16", device_map="auto")
cfg = LoraConfig(r=16, lora_alpha=32, lora_dropout=0.05, task_type="CAUSAL_LM",
                 target_modules=["q_proj", "k_proj", "v_proj", "o_proj", "gate_proj", "up_proj", "down_proj"])
model = get_peft_model(base, cfg)
model.print_trainable_parameters()        # trainable params: ~42M || all params: ~8.07B || 0.52%

ds = load_dataset("json", data_files="data/samples/finetune_support.jsonl")["train"]
trainer = SFTTrainer(model=model, train_dataset=ds,
                     args=TrainingArguments("out/lora", per_device_train_batch_size=8, num_train_epochs=3,
                                            learning_rate=2e-4, bf16=True, gradient_checkpointing=True, logging_steps=10))
trainer.train()
model.save_pretrained("adapters/support-v1")   # adapter only (MBs)
# merged = model.merge_and_unload()            # fold BA into W for zero-overhead serving
''',
        diagram=[{"label": "x"}, {"label": "Frozen W (d×k)"}, {"label": "A (r×k) ↓", "cls": "accent"}, {"label": "B (d×r) ↑", "cls": "accent"},
                 {"label": "Wx + (α/r)·BAx"}],
        performance="Trainable params per adapted matrix = r·(d_in + d_out). Training memory ≈ 2 B/param (frozen BF16) + 16 B/adapter param + activations.",
        checklist=["Choose targets (all linear layers usually best)", "Tune r and α together (α≈2r)", "Use a held-out validation set", "Version adapters with base-model hash",
                   "Evaluate vs base on task + safety evals", "Decide merge vs hot-swap for serving"],
        applications=[{"domain": "Customer support automation", "text": "Per-client tone and product adapters on a shared base."},
                      {"domain": "Financial services", "text": "Adapting to internal report formats without moving data off-prem."},
                      {"domain": "Multi-agent enterprise platforms", "text": "Specialist adapters per agent role, hot-swapped at inference."}],
        formula="W' = W + (α/r)·B·A",
        limitations="The LLM memory figures are estimates (activation formula from Korthikanti et al.); real usage depends on framework, kernels and fragmentation.",
        references=["Hu et al., 'LoRA: Low-Rank Adaptation of Large Language Models' (2021)", "Korthikanti et al., 'Reducing Activation Recomputation' (2022)"],
    ),
))


# ============================================================================ QLoRA
def quant_bits(qtype: str, double_quant: bool, block: int = 64) -> float:
    base = {"nf4": 4, "fp4": 4, "int8": 8}[qtype]
    if double_quant:
        return base + 8 / block + 32 / (block * 256)
    return base + 32 / block


def qlora_sim(p: dict, ctx: Context) -> Result:
    a = MODELS[p["model"]]
    hw = HARDWARE[p["gpu"]]
    N = a.total_params()
    targets = [t["value"] for t in TARGETS]
    nl = lora_params(a, targets, p["rank"])
    act = activation_bytes(a, p["batch_size"], p["seq_len"], p["grad_checkpointing"])
    bits = quant_bits(p["quant_type"], p["double_quant"])
    overhead = 1.0e9
    q_weights = N * bits / 8
    rows = {
        "Full FT (BF16 + Adam)": [N * 2, N * 2, N * 12, act, overhead],
        "LoRA (BF16 base)": [N * 2, nl * 2, nl * 12, act, overhead],
        f"QLoRA ({p['quant_type'].upper()}{' + DQ' if p['double_quant'] else ''})": [q_weights, nl * 2, nl * 12, act, overhead],
    }
    totals = {k: sum(v) / 1e9 for k, v in rows.items()}
    r = Result("analytical")
    q_name = list(rows)[2]
    r.metric("qbits", "Bits per base weight", bits, "bits", "4-bit codes + quantization constants")
    r.metric("qweights", "Quantized base weights", q_weights / 1e9, "GB")
    r.metric("dq_saving", "Double-quant saving", N * ((4 + 32 / 64) - quant_bits(p["quant_type"], True)) / 8 / 1e9 if p["quant_type"] != "int8" else 0, "GB")
    r.metric("total", "Est. QLoRA training memory", totals[q_name], "GB", status="good" if totals[q_name] < hw["mem_gb"] * 0.92 else "bad")
    r.metric("fits", f"Fits on {hw['label']}?", "yes" if totals[q_name] < hw["mem_gb"] * 0.92 else "no", status="good" if totals[q_name] < hw["mem_gb"] * 0.92 else "bad")
    r.metric("trainable", "Trainable LoRA params", fmt_params(nl), "", "all linear layers")
    r.chart("mem", "Memory: full FT vs LoRA vs QLoRA (estimate)", "bar", list(rows), [
        {"label": lab, "data": [v[i] / 1e9 for v in rows.values()]} for i, lab in enumerate(["base weights", "gradients", "optimizer + master", "activations", "CUDA/runtime overhead"])],
        "", "GB", stacked=True, threshold=hw["mem_gb"])
    g = rng(p["seed"])
    W = g.normal(0, 1, (128, 256)).astype(np.float32)
    Y = W
    errs = []
    for f in ("nf4", "int4"):
        Wq, _ = quantize(W, f, "group", 64)
        errs.append(float(np.sqrt(((Y - Wq) ** 2).mean())))
    r.chart("nf4", "NF4 vs uniform INT4 on normally-distributed weights (computed RMSE, block 64)", "bar", ["NF4", "INT4 (uniform)"], [{"label": "RMSE", "data": errs}], "", "RMSE")
    r.chart("levels", "NF4 code values (quantiles of N(0,1), normalized)", "scatter", None,
            [{"label": "NF4 levels", "data": [{"x": i, "y": float(v)} for i, v in enumerate(NF4)]}], "code", "value")
    models = ["tinyllama-1.1b", "phi-3-mini", "llama-3-8b", "llama-2-13b", "mixtral-8x7b", "llama-3-70b"]
    r.table("Which models fit for fine-tuning on this GPU?", ["Model", "Full FT GB", "LoRA GB", "QLoRA GB", f"QLoRA fits {hw['mem_gb']} GB?"], [
        [MODELS[m].label, round((MODELS[m].total_params() * 16 + activation_bytes(MODELS[m], p["batch_size"], p["seq_len"], p["grad_checkpointing"])) / 1e9, 1),
         round((MODELS[m].total_params() * 2 + lora_params(MODELS[m], targets, p["rank"]) * 16 + activation_bytes(MODELS[m], p["batch_size"], p["seq_len"], p["grad_checkpointing"])) / 1e9, 1),
         round(q := (MODELS[m].total_params() * bits / 8 + lora_params(MODELS[m], targets, p["rank"]) * 14 + activation_bytes(MODELS[m], p["batch_size"], p["seq_len"], p["grad_checkpointing"]) + overhead) / 1e9, 1),
         "✓" if q < hw["mem_gb"] * 0.92 else "✗"] for m in models])
    r.flow("QLoRA forward/backward", [{"label": "NF4 base weights", "sub": "frozen, 4-bit"}, {"label": "Dequantize → BF16", "sub": "per block, on the fly", "cls": "accent"},
                                      {"label": "W·x + (α/r)·B·A·x"}, {"label": "Backprop into A,B only", "cls": "accent"},
                                      {"label": "Paged AdamW" if p["paged_optimizer"] else "AdamW", "sub": "spills to CPU on spikes" if p["paged_optimizer"] else ""}])
    r.compare("LoRA (BF16 base)", q_name, [("Training memory", totals["LoRA (BF16 base)"], totals[q_name], "GB", "lower"),
                                            ("Base weight memory", N * 2 / 1e9, q_weights / 1e9, "GB", "lower"),
                                            ("Step time (relative)", 1.0, 1.25, "×", "lower")])
    r.note("QLoRA step time is typically slower than BF16 LoRA because of dequantization; the 1.25× shown is an illustrative assumption, not a measurement.")
    if p["paged_optimizer"]:
        r.note("Paged optimizers avoid OOM on memory spikes by paging optimizer state to CPU via unified memory; they do not reduce steady-state memory.")
    r.recommend("Use NF4 + double quantization + BF16 compute dtype (bnb_4bit_compute_dtype=bfloat16).",
                "Target all linear layers — QLoRA paper found this necessary to match 16-bit full fine-tuning quality.",
                "Merge adapters into a de-quantized BF16 copy for serving, or serve with a 4-bit base + adapter.")
    return r


register(ModuleDef(
    id="qlora", title="QLoRA", category="training", fidelity="analytical", tags=["essential"],
    summary="Fine-tune LoRA adapters on top of a frozen 4-bit NF4 base model to fit large models on one GPU.",
    controls=[
        select("model", "Base model", "llama-3-8b", MODEL_OPTIONS, "Model to fine-tune."),
        select("gpu", "GPU", "rtx4090", HARDWARE_OPTIONS, "Target training device."),
        select("quant_type", "Quantization type", "nf4", [{"value": "nf4", "label": "NF4 (4-bit NormalFloat)"}, {"value": "fp4", "label": "FP4"}, {"value": "int8", "label": "INT8 (LLM.int8)"}], "Base-weight storage format."),
        toggle("double_quant", "Double quantization", True, "Quantize the per-block scales themselves (saves ~0.37 bits/param)."),
        slider("rank", "LoRA rank", 16, 1, 128, 1, "Adapter rank."),
        slider("batch_size", "Micro-batch size", 4, 1, 32, 1, "Sequences per step."),
        slider("seq_len", "Sequence length", 2048, 256, 8192, 256, "Tokens per sequence."),
        toggle("grad_checkpointing", "Gradient checkpointing", True, "Trade compute for activation memory."),
        toggle("paged_optimizer", "Paged optimizer", True, "Page optimizer state to CPU under pressure."),
        slider("seed", "Random seed", 0, 0, 9999, 1, "For the NF4 error computation."),
    ],
    simulate=qlora_sim,
    docs=docs(
        what="QLoRA backpropagates through a frozen base model stored in 4-bit NormalFloat (NF4) into LoRA adapters kept in 16-bit, with double quantization of scales and paged optimizers.",
        why="Even with LoRA, BF16 base weights for a 70B model need 140 GB. Storing them in ~4.1 bits cuts that to ~36 GB while preserving fine-tuning quality.",
        how="NF4 places 16 code values at quantiles of a standard normal — optimal for normally distributed weights. Each 64-weight block stores an absmax scale; double quantization stores those scales in 8-bit. Weights are dequantized to BF16 on the fly per matmul; gradients flow only into LoRA parameters.",
        when_use="Fine-tuning 7B–70B models on a single 24–80 GB GPU, cost-constrained experimentation, privacy-preserving on-prem fine-tuning.",
        when_avoid="When you have ample GPU memory (BF16 LoRA is faster), or need maximum throughput training at scale.",
        pros=["~4× less base-weight memory", "Matches 16-bit LoRA quality in the original paper's evaluations", "Enables 65–70B fine-tuning on one 48–80 GB GPU"],
        cons=["Slower steps due to dequantization", "bitsandbytes/CUDA dependency", "Merging into 4-bit weights loses precision — merge into a BF16 copy"],
        use_case="A hospital fine-tunes Llama-3-8B on de-identified discharge summaries on a single on-prem RTX 4090 using QLoRA.",
        code='''
import torch
from transformers import AutoModelForCausalLM, BitsAndBytesConfig
from peft import LoraConfig, get_peft_model, prepare_model_for_kbit_training

bnb = BitsAndBytesConfig(load_in_4bit=True, bnb_4bit_quant_type="nf4",
                         bnb_4bit_use_double_quant=True, bnb_4bit_compute_dtype=torch.bfloat16)
model = AutoModelForCausalLM.from_pretrained("meta-llama/Llama-3.1-8B", quantization_config=bnb, device_map="auto")
model = prepare_model_for_kbit_training(model, use_gradient_checkpointing=True)
model = get_peft_model(model, LoraConfig(r=16, lora_alpha=32, lora_dropout=0.05,
                                         target_modules="all-linear", task_type="CAUSAL_LM"))
# TrainingArguments(..., optim="paged_adamw_8bit", bf16=True)
''',
        diagram=[{"label": "BF16 checkpoint"}, {"label": "NF4 + double quant", "cls": "accent"}, {"label": "Frozen 4-bit base"}, {"label": "BF16 LoRA adapters", "cls": "accent"}, {"label": "Paged AdamW"}],
        performance="Base memory ≈ params × ~4.13 bits / 8. Adapter + optimizer memory is small; activations dominate at long sequence lengths.",
        checklist=["Use NF4 + double quant + BF16 compute", "Target all linear layers", "Enable gradient checkpointing", "Use paged 8-bit AdamW", "Evaluate merged model in BF16"],
        applications=[{"domain": "Healthcare information systems", "text": "On-prem fine-tuning on sensitive data with a single GPU."},
                      {"domain": "Financial services", "text": "Low-cost domain adaptation of larger models for analysis tasks."}],
        limitations="Memory is estimated; real usage varies with framework version, CUDA allocator behaviour and sequence packing.",
        references=["Dettmers et al., 'QLoRA: Efficient Finetuning of Quantized LLMs' (2023)"],
    ),
))
