"""Module 2 — Tokenizer Bias (tokenization effects)."""
from __future__ import annotations

from app.engine.registry import Context, ModuleDef, Result, docs, register, select, slider, text, toggle
from app.tokenization.bpe import SAMPLES, ByteLevel, ByteLevelBPE, CharLevel, WordLevel, optional_tokenizers

TOKENIZERS = [{"value": "bpe", "label": "Byte-level BPE (trained here)"}, {"value": "word", "label": "Word-level (closed vocab)"},
              {"value": "char", "label": "Character-level"}, {"value": "byte", "label": "Raw UTF-8 bytes"}]


def build(name: str, vocab: int, weight: float):
    if name == "bpe":
        return ByteLevelBPE(vocab, weight)
    if name == "word":
        return WordLevel(vocab, weight)
    if name == "char":
        return CharLevel()
    return ByteLevel()


def tok_sim(p: dict, ctx: Context) -> Result:
    r = Result("computed")
    txt = p["text"] or " "
    tk = build(p["tokenizer"], p["vocab_size"], p["multilingual_weight"])
    pieces = tk.encode(txt)
    truncated = len(pieces) > p["max_tokens"]
    shown = pieces[: p["max_tokens"]]
    unk = sum(1 for _, i in pieces if p["tokenizer"] == "word" and i == 0)
    r.tokens(f"Tokenized output — {len(pieces)} tokens" + (" (truncated)" if truncated else ""),
             [{"text": t, "id": i, "cls": "unk" if (p["tokenizer"] == "word" and i == 0) else None} for t, i in shown],
             show_ids=p["show_ids"])
    r.metric("tokens", "Token count", len(pieces), "", status="warn" if truncated else None)
    r.metric("chars_per_tok", "Characters / token", len(txt) / max(1, len(pieces)), "", "higher = more efficient")
    r.metric("bytes_per_tok", "UTF-8 bytes / token", len(txt.encode()) / max(1, len(pieces)))
    r.metric("vocab", "Vocabulary size", getattr(tk, "vocab_size", 0))
    if p["tokenizer"] == "word":
        r.metric("unk", "Unknown ([UNK]) tokens", unk, "", "information lost — the model never sees these words", status="bad" if unk else None)
    if truncated:
        r.note(f"Input exceeds max length {p['max_tokens']}: {len(pieces) - p['max_tokens']} tokens would be silently dropped.", "warn")

    # Cross-language fertility with the selected tokenizer config
    eng = len(tk.encode(SAMPLES["English"]))
    rows, labels, counts = [], [], []
    for lang, s in SAMPLES.items():
        n = len(tk.encode(s))
        labels.append(lang)
        counts.append(n)
        rows.append([lang, len(s), n, round(len(s) / n, 2), round(n / eng, 2)])
    r.chart("fertility", "Tokens for the *same* sentence across languages", "bar", labels,
            [{"label": "tokens", "data": counts}], "", "tokens")
    r.table("Language premium (same meaning, selected tokenizer)", ["Sample", "Chars", "Tokens", "Chars/token", "× English"], rows)

    # Tokenizer comparison on the user's text
    if p["compare"]:
        comp = {"BPE": ByteLevelBPE(p["vocab_size"], p["multilingual_weight"]), "Word": WordLevel(p["vocab_size"], p["multilingual_weight"]),
                "Char": CharLevel(), "Byte": ByteLevel()}
        extra = optional_tokenizers()
        crow = [[k, len(v.encode(txt)), "built-in"] for k, v in comp.items()]
        crow += [[k, len(fn(txt)), "real pretrained"] for k, fn in extra.items()]
        r.table("Tokenizer comparison on your text", ["Tokenizer", "Tokens", "Source"], crow)
        r.chart("compare", "Token count by tokenizer", "bar", [c[0] for c in crow], [{"label": "tokens", "data": [c[1] for c in crow]}])
        if not extra:
            r.note("Install `tiktoken` or cache Hugging Face tokenizers locally to add real pretrained tokenizers to this comparison.")
    r.compare("English-weighted BPE (weight 0)", f"Current (weight {p['multilingual_weight']})", [
        ("Hindi sample tokens", len(ByteLevelBPE(p["vocab_size"], 0.0).encode(SAMPLES["Hindi"])), len(tk.encode(SAMPLES["Hindi"])) if p["tokenizer"] == "bpe" else len(ByteLevelBPE(p["vocab_size"], p["multilingual_weight"]).encode(SAMPLES["Hindi"])), "", "lower"),
        ("Chinese sample tokens", len(ByteLevelBPE(p["vocab_size"], 0.0).encode(SAMPLES["Chinese"])), len(ByteLevelBPE(p["vocab_size"], p["multilingual_weight"]).encode(SAMPLES["Chinese"])), "", "lower"),
        ("English sample tokens", len(ByteLevelBPE(p["vocab_size"], 0.0).encode(SAMPLES["English"])), len(ByteLevelBPE(p["vocab_size"], p["multilingual_weight"]).encode(SAMPLES["English"])), "", "lower"),
    ], title="Effect of training-corpus language weighting on BPE")
    r.note("This is a *tokenization effect*: unequal cost, context consumption and fragmentation. It is not the same as model-level bias "
           "(learned associations in weights) nor decoding-time logit bias (an additive offset on chosen token ids).")
    r.recommend("Measure tokens-per-character on your real traffic per language before sizing context windows and cost.",
                "Choose multilingual tokenizers (large vocab, balanced corpus) for non-English products.",
                "Never assume max_tokens = words; truncate on token count, from the side that matters.",
                "Normalize numbers/IDs when exact digits matter — tokenizers split them inconsistently.")
    return r


register(ModuleDef(
    id="tokenizer-bias", title="Tokenizer Effects (\"Tokenizer Bias\")", category="architecture", original_index=2, fidelity="computed",
    summary="How vocabulary design and training-corpus frequency make some text cost more tokens and fragment differently.",
    controls=[
        text("text", "Input text", "Please reset my password. Por favor restablezca mi contraseña. कृपया मेरा पासवर्ड रीसेट करें।", "Any text, any language.", multiline=True, group="Input"),
        select("tokenizer", "Tokenizer", "bpe", TOKENIZERS, "Algorithm used for the main tokenization view."),
        slider("vocab_size", "Vocabulary size", 800, 260, 2200, 20, "BPE: 256 base bytes + learned merges. Word-level: words kept; others → [UNK]."),
        slider("multilingual_weight", "Non-English corpus weight", 0.0, 0.0, 1.0, 0.25, "Token-frequency weighting of Spanish/Hindi/Chinese/Arabic text in the training corpus (English = 1.0)."),
        slider("max_tokens", "Maximum token length", 128, 8, 2048, 8, "Truncation limit applied to the input."),
        toggle("show_ids", "Show token IDs", True, "Display the integer id under every token."),
        toggle("compare", "Compare tokenizers", True, "Tokenize the same input with every available tokenizer."),
    ],
    simulate=tok_sim,
    docs=docs(
        what="Tokenization maps text to integer ids from a fixed vocabulary. The vocabulary is learned from a training corpus, so text resembling that corpus is encoded compactly and everything else is split into many small pieces.",
        why="Models have finite vocabularies and context windows. Subword algorithms (BPE, WordPiece, Unigram) balance vocabulary size against sequence length and eliminate out-of-vocabulary failures.",
        how="BPE starts from bytes and repeatedly merges the most frequent adjacent pair in the training corpus. Frequent strings become single tokens; rare scripts stay near byte level. The same sentence in Hindi can cost several times the tokens of English when the corpus is English-dominated.",
        when_use="Analyse tokenization whenever you estimate cost/latency, set context limits, support multiple languages, handle code/numbers, or choose between model families.",
        when_avoid="Do not attribute every behavior difference to the tokenizer; model-level bias comes from training data and objectives, and logit bias is an explicit decoding-time adjustment.",
        pros=["Subword tokenizers have no OOV problem", "Compact encoding for in-distribution text", "Byte fallback handles any input"],
        cons=["Unequal token cost across languages (a 'language tax')", "Numbers and identifiers split inconsistently, hurting arithmetic and exact matching", "Word-level vocabularies lose rare words entirely ([UNK])"],
        use_case="A bank launching a Hindi + English assistant finds Hindi conversations consume ~3× the context and cost; they switch to a model with a multilingual tokenizer and budget context per language.",
        code='''
from transformers import AutoTokenizer
for name in ["gpt2", "meta-llama/Llama-3.1-8B", "google/gemma-2-2b"]:
    tok = AutoTokenizer.from_pretrained(name)
    for text in ["Please reset my password.", "कृपया मेरा पासवर्ड रीसेट करें।"]:
        ids = tok.encode(text, add_special_tokens=False)
        print(name, len(text), len(ids), tok.convert_ids_to_tokens(ids)[:12])

# Logit bias is DIFFERENT: an explicit decoding-time offset, e.g. OpenAI-compatible APIs
# client.chat.completions.create(..., logit_bias={token_id: -100})  # ban a token
''',
        diagram=[{"label": "Raw text"}, {"label": "Normalize + pre-tokenize"}, {"label": "Apply BPE merges", "cls": "accent"},
                 {"label": "Token ids"}, {"label": "Embedding lookup"}],
        performance="Cost, latency and context consumption scale with token count, so a 3× token premium is a 3× cost and a 3× smaller effective context window for that language.",
        checklist=["Measure chars/token per language on real data", "Budget context by tokens, not words", "Test numeric/ID handling",
                   "Pin tokenizer version with the model", "Check special-token handling and chat templates"],
        applications=[{"domain": "Financial services", "text": "Account numbers and amounts fragment into many tokens; validate exact-match extraction."},
                      {"domain": "Healthcare information systems", "text": "Drug names and codes (ICD-10) are rare strings; tokenization affects recall."},
                      {"domain": "Customer support automation", "text": "Multilingual support costs differ by language — price per locale."}],
        limitations="The built-in BPE is trained on a tiny embedded corpus to make the frequency effect visible; production tokenizers are trained on trillions of tokens, but the mechanism is identical.",
        references=["Sennrich et al., 'Neural Machine Translation of Rare Words with Subword Units' (2016)", "Petrov et al., 'Language Model Tokenizers Introduce Unfairness Between Languages' (2023)"],
    ),
))
