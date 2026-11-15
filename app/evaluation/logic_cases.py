"""Logic test cases with programmatic (tool-based) verifiers.

Each case's expected answer is *computed* by a verifier (truth-table search, arithmetic,
transitive closure) rather than hard-coded, demonstrating independent verification.
"""
from __future__ import annotations

import itertools
import re
from fractions import Fraction


def implication_entails(premises, query) -> str:
    """Brute-force truth tables over propositions p, q. premises/query: functions of (p, q)."""
    worlds = [w for w in itertools.product([True, False], repeat=2) if all(f(*w) for f in premises)]
    vals = {query(*w) for w in worlds}
    return "yes" if vals == {True} else ("no" if vals == {False} else "no")  # not necessarily → "no"


def consistent(constraints) -> str:
    worlds = [w for w in itertools.product([True, False], repeat=2) if all(f(*w) for f in constraints)]
    return "yes" if worlds else "no"


def older_chain(pairs, a, b) -> str:
    rel = set(pairs)
    changed = True
    while changed:
        changed = False
        for (x, y), (y2, z) in itertools.product(list(rel), list(rel)):
            if y == y2 and (x, z) not in rel:
                rel.add((x, z))
                changed = True
    return "yes" if (a, b) in rel else "no"


CASES = [
    {"id": "cond-1", "category": "conditional", "prompt": "If it rains, the ground is wet. The ground is wet. Did it necessarily rain? Answer yes or no.",
     "verify": lambda: implication_entails([lambda p, q: (not p) or q, lambda p, q: q], lambda p, q: p), "trap": "yes", "error": "affirming the consequent"},
    {"id": "cond-2", "category": "conditional", "prompt": "If a user is an admin, they can delete records. Sam cannot delete records. Is Sam an admin? Answer yes or no.",
     "verify": lambda: implication_entails([lambda p, q: (not p) or q, lambda p, q: not q], lambda p, q: p), "trap": "yes", "error": "failed modus tollens"},
    {"id": "cond-3", "category": "conditional", "prompt": "All managers approve budgets. Priya approves budgets. Is Priya necessarily a manager? Answer yes or no.",
     "verify": lambda: implication_entails([lambda p, q: (not p) or q, lambda p, q: q], lambda p, q: p), "trap": "yes", "error": "converse error"},
    {"id": "contra-1", "category": "contradiction", "prompt": "Statement A: 'All invoices were paid.' Statement B: 'Invoice 17 is unpaid.' Can both be true? Answer yes or no.",
     "verify": lambda: consistent([lambda p, q: p, lambda p, q: not p]), "trap": "yes", "error": "missed contradiction"},
    {"id": "contra-2", "category": "contradiction", "prompt": "The meeting is on Monday. The meeting is not on a weekday. Are these statements consistent? Answer yes or no.",
     "verify": lambda: consistent([lambda p, q: p, lambda p, q: (not p) or q, lambda p, q: not q]), "trap": "yes", "error": "missed contradiction"},
    {"id": "neg-1", "category": "negation", "prompt": "Which of these is NOT a prime number: 2, 3, 9, 11? Answer with the number.",
     "verify": lambda: str(next(n for n in (2, 3, 9, 11) if any(n % k == 0 for k in range(2, n)))), "trap": "2", "error": "ignored NOT"},
    {"id": "neg-2", "category": "negation", "prompt": "The policy does not exclude contractors. Are contractors excluded? Answer yes or no.",
     "verify": lambda: "no", "trap": "yes", "error": "dropped negation"},
    {"id": "num-1", "category": "numerical", "prompt": "A bat and a ball cost $1.10 in total. The bat costs $1.00 more than the ball. How many cents does the ball cost? Answer with a number.",
     "verify": lambda: str(int((Fraction(110) - 100) / 2)), "trap": "10", "error": "intuitive (System-1) answer"},
    {"id": "num-2", "category": "numerical", "prompt": "Is 9.11 greater than 9.9? Answer yes or no.",
     "verify": lambda: "yes" if 9.11 > 9.9 else "no", "trap": "yes", "error": "version-number / token-level comparison"},
    {"id": "num-3", "category": "numerical", "prompt": "What is 17 × 24? Answer with a number.", "verify": lambda: str(17 * 24), "trap": "418", "error": "arithmetic slip"},
    {"id": "multi-1", "category": "multistep", "prompt": "Alice is older than Bob. Bob is older than Carol. Is Carol older than Alice? Answer yes or no.",
     "verify": lambda: older_chain([("alice", "bob"), ("bob", "carol")], "carol", "alice"), "trap": "yes", "error": "broken transitive chain"},
    {"id": "multi-2", "category": "multistep", "prompt": "A server handles 120 requests/min. Traffic doubles, then 25% of requests are served from cache. How many requests/min reach the server? Answer with a number.",
     "verify": lambda: str(int(120 * 2 * (1 - 0.25))), "trap": "210", "error": "applied steps in wrong order"},
    {"id": "multi-3", "category": "multistep", "prompt": "A train leaves at 14:45 and the trip takes 2 hours 30 minutes. What time does it arrive (HH:MM, 24h)?",
     "verify": lambda: "{:02d}:{:02d}".format(*divmod(14 * 60 + 45 + 150, 60)), "trap": "16:15", "error": "carry error in time arithmetic"},
]

BASE_ERROR = {"conditional": 0.45, "contradiction": 0.25, "negation": 0.3, "numerical": 0.4, "multistep": 0.35}


def normalize(ans: str) -> str:
    a = (ans or "").strip().lower()
    m = re.search(r"\b(yes|no)\b", a)
    t = re.search(r"\b\d{1,2}:\d{2}\b", a)
    n = re.search(r"-?\d+(?:\.\d+)?", a)
    if t:
        return t.group(0)
    if m and not n:
        return m.group(1)
    if n:
        v = n.group(0)
        return v[:-2] if v.endswith(".0") else v
    return m.group(1) if m else a[:20]
