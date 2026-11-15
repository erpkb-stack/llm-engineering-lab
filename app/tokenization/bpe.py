"""Small, real tokenizers implemented from scratch (no downloads needed).

* ByteLevelBPE — trains real BPE merges over UTF-8 bytes on an embedded corpus
  whose language mix is configurable (to demonstrate vocabulary coverage effects).
* WordLevel / Char / Byte tokenizers for comparison.
* Optional Hugging Face / tiktoken tokenizers if those libraries are installed.
"""
from __future__ import annotations

import re
from collections import Counter
from functools import lru_cache

CORPUS = {
    "en": """The quick brown fox jumps over the lazy dog. Large language models read text as tokens, not words.
A tokenizer splits text into pieces that appear often in its training data. Common English words become a single token,
while rare words are broken into several smaller pieces. The model was trained on documents about finance, health,
software, science and customer support. Please reset my password because I cannot log in to my account.
The invoice total is due at the end of the month and the payment was approved by the bank.
Retrieval augmented generation combines search with generation to ground answers in documents.
Engineers measure latency, throughput and memory when they deploy models in production systems.
The patient reported a headache and the doctor recommended rest and water. Our support team answers questions every day.
Machine learning models learn patterns from data, and the quality of the data matters more than the size of the model.
This product helps teams write better software, review code faster and ship features with confidence.""",
    "es": """El rápido zorro marrón salta sobre el perro perezoso. Los modelos de lenguaje leen el texto como tokens.
Por favor restablezca mi contraseña porque no puedo iniciar sesión en mi cuenta. La factura vence a fin de mes.
El paciente informó dolor de cabeza y el médico recomendó descanso. Nuestro equipo de soporte responde preguntas.""",
    "hi": """तेज़ भूरी लोमड़ी आलसी कुत्ते के ऊपर कूदती है। भाषा मॉडल पाठ को टोकन के रूप में पढ़ते हैं।
कृपया मेरा पासवर्ड रीसेट करें क्योंकि मैं अपने खाते में लॉग इन नहीं कर पा रहा हूँ। चालान महीने के अंत में देय है।
मरीज़ ने सिरदर्द की शिकायत की और डॉक्टर ने आराम की सलाह दी। हमारी सहायता टीम हर दिन प्रश्नों का उत्तर देती है।""",
    "zh": """敏捷的棕色狐狸跳过了懒狗。大型语言模型把文本当作词元来读取。请重置我的密码，因为我无法登录我的账户。
发票总额在月底到期，付款已获银行批准。病人说头痛，医生建议休息和喝水。我们的支持团队每天回答问题。""",
    "ar": """الثعلب البني السريع يقفز فوق الكلب الكسول. نماذج اللغة تقرأ النص كرموز وليس ككلمات.
يرجى إعادة تعيين كلمة المرور لأنني لا أستطيع تسجيل الدخول إلى حسابي. الفاتورة مستحقة في نهاية الشهر.
أبلغ المريض عن صداع وأوصى الطبيب بالراحة. فريق الدعم لدينا يجيب على الأسئلة كل يوم.""",
    "code": """def compute_total(items):\n    return sum(item.price * item.quantity for item in items)\n
for i in range(10):\n    print(f"value={i}")\nclass User:\n    def __init__(self, name, email):\n        self.name = name\n""",
}

SAMPLES = {
    "English": "Please reset my password because I cannot log in to my account.",
    "Spanish": "Por favor restablezca mi contraseña porque no puedo iniciar sesión en mi cuenta.",
    "Hindi": "कृपया मेरा पासवर्ड रीसेट करें क्योंकि मैं अपने खाते में लॉग इन नहीं कर पा रहा हूँ।",
    "Chinese": "请重置我的密码，因为我无法登录我的账户。",
    "Arabic": "يرجى إعادة تعيين كلمة المرور لأنني لا أستطيع تسجيل الدخول إلى حسابي.",
    "Python code": "def compute_total(items): return sum(i.price * i.quantity for i in items)",
    "Numbers": "Invoice 4839201 total $12,345.67 due 2026-11-30, ref ID 0x7f3a9c.",
}

PRETOKENIZE = re.compile(r"""'s|'t|'re|'ve|'m|'ll|'d| ?\w+| ?\d+| ?[^\s\w]+|\s+(?!\S)|\s+""", re.UNICODE)


def _training_words(multilingual_weight: float) -> Counter:
    """Word frequency table. English always weight 1; other languages repeated by weight×4."""
    words: Counter = Counter()
    for lang, text in CORPUS.items():
        reps = 4 if lang == "en" else int(round(multilingual_weight * 4))
        if lang == "code":
            reps = 2
        for _ in range(reps):
            for w in PRETOKENIZE.findall(text):
                words[w] += 1
    return words


@lru_cache(maxsize=16)
def train_bpe(multilingual_weight: float, max_merges: int = 2000) -> list[tuple[bytes, bytes]]:
    """Classic BPE (Sennrich et al.) over UTF-8 bytes. Returns merges in rank order."""
    vocab = {tuple(bytes([b]) for b in w.encode("utf-8")): c for w, c in _training_words(multilingual_weight).items()}
    merges: list[tuple[bytes, bytes]] = []
    for _ in range(max_merges):
        pairs: Counter = Counter()
        for sym, c in vocab.items():
            for a, b in zip(sym, sym[1:]):
                pairs[(a, b)] += c
        if not pairs:
            break
        best, cnt = max(pairs.items(), key=lambda kv: (kv[1], kv[0]))
        if cnt < 2:
            break
        merges.append(best)
        merged = best[0] + best[1]
        new_vocab = {}
        for sym, c in vocab.items():
            out, i = [], 0
            while i < len(sym):
                if i < len(sym) - 1 and sym[i] == best[0] and sym[i + 1] == best[1]:
                    out.append(merged)
                    i += 2
                else:
                    out.append(sym[i])
                    i += 1
            t = tuple(out)
            new_vocab[t] = new_vocab.get(t, 0) + c
        vocab = new_vocab
    return merges


class ByteLevelBPE:
    name = "bpe"

    def __init__(self, vocab_size: int, multilingual_weight: float):
        all_merges = train_bpe(round(multilingual_weight, 2))
        self.merges = all_merges[: max(0, vocab_size - 256)]
        self.ranks = {m: i for i, m in enumerate(self.merges)}
        self.vocab: dict[bytes, int] = {bytes([i]): i for i in range(256)}
        for i, (a, b) in enumerate(self.merges):
            self.vocab[a + b] = 256 + i
        self.vocab_size = len(self.vocab)
        self._cache: dict[str, list[bytes]] = {}

    def _bpe(self, word: str) -> list[bytes]:
        if word in self._cache:
            return self._cache[word]
        sym = [bytes([b]) for b in word.encode("utf-8")]
        while len(sym) > 1:
            ranked = [(self.ranks.get((a, b), 1 << 30), i) for i, (a, b) in enumerate(zip(sym, sym[1:]))]
            rank, i = min(ranked)
            if rank == 1 << 30:
                break
            sym = sym[:i] + [sym[i] + sym[i + 1]] + sym[i + 2:]
        self._cache[word] = sym
        return sym

    def encode(self, text: str) -> list[tuple[str, int]]:
        out = []
        for w in PRETOKENIZE.findall(text):
            for piece in self._bpe(w):
                out.append((piece.decode("utf-8", errors="replace"), self.vocab[piece]))
        return out


class WordLevel:
    name = "word"

    def __init__(self, vocab_size: int, multilingual_weight: float):
        counts = _training_words(multilingual_weight)
        common = [w.strip() for w, _ in counts.most_common() if w.strip()]
        uniq = list(dict.fromkeys(common))[: max(1, vocab_size - 2)]
        self.vocab = {"[UNK]": 0, "[PAD]": 1, **{w: i + 2 for i, w in enumerate(uniq)}}
        self.vocab_size = len(self.vocab)

    def encode(self, text: str):
        return [(w, self.vocab.get(w, 0)) for w in re.findall(r"\w+|[^\s\w]", text)]


class CharLevel:
    name = "char"
    vocab_size = 0x110000

    def encode(self, text: str):
        return [(c, ord(c)) for c in text]


class ByteLevel:
    name = "byte"
    vocab_size = 256

    def encode(self, text: str):
        return [(bytes([b]).decode("latin-1") if 32 <= b < 127 else f"<0x{b:02X}>", b) for b in text.encode("utf-8")]


def optional_tokenizers() -> dict:
    """Real pretrained tokenizers if available locally (no network access is attempted)."""
    found = {}
    try:
        import tiktoken  # type: ignore

        enc = tiktoken.get_encoding("cl100k_base")
        found["tiktoken-cl100k"] = lambda t: [(enc.decode([i]), i) for i in enc.encode(t)]
    except Exception:
        pass
    try:
        from transformers import AutoTokenizer  # type: ignore

        for name in ("gpt2", "bert-base-uncased"):
            try:
                tok = AutoTokenizer.from_pretrained(name, local_files_only=True)
                found[f"hf-{name}"] = (lambda tk: (lambda t: [(tk.convert_ids_to_tokens(i), i) for i in tk.encode(t, add_special_tokens=False)]))(tok)
            except Exception:
                continue
    except Exception:
        pass
    return found
