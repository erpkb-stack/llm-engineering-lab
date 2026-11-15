"""Rényi-DP accountant for the Poisson-subsampled Gaussian mechanism (DP-SGD).

Implements the exact integer-order RDP bound of Mironov, Talwar & Zhang (2019),
the same computation as TensorFlow Privacy's ``_compute_log_a_int``:

    A_α = Σ_{k=0}^{α} C(α,k) (1−q)^{α−k} q^k exp((k² − k) / (2σ²))
    RDP(α) = log(A_α) / (α − 1)

and the standard conversion ε(δ) = min_α [T·RDP(α) + log(1/δ)/(α − 1)].
The conversion is valid but not the tightest available (Balle et al. 2020 give tighter);
results therefore upper-bound ε. Assumes Poisson sampling with rate q = batch / N.
"""
from __future__ import annotations

import math
from functools import lru_cache

ORDERS = tuple(list(range(2, 64)) + [80, 96, 128, 192, 256])


def _logsumexp(xs: list[float]) -> float:
    m = max(xs)
    return m + math.log(sum(math.exp(x - m) for x in xs))


@lru_cache(maxsize=4096)
def rdp_subsampled_gaussian(q: float, sigma: float, alpha: int) -> float:
    if q <= 0:
        return 0.0
    if sigma <= 0:
        return math.inf
    if q >= 1.0:
        return alpha / (2 * sigma ** 2)
    terms = []
    for k in range(alpha + 1):
        log_coef = math.lgamma(alpha + 1) - math.lgamma(k + 1) - math.lgamma(alpha - k + 1)
        terms.append(log_coef + (alpha - k) * math.log1p(-q) + k * math.log(q) + (k * k - k) / (2 * sigma ** 2))
    return _logsumexp(terms) / (alpha - 1)


def epsilon(q: float, sigma: float, steps: int, delta: float) -> tuple[float, int]:
    best, best_order = math.inf, ORDERS[0]
    for a in ORDERS:
        eps = steps * rdp_subsampled_gaussian(q, sigma, a) + math.log(1 / delta) / (a - 1)
        if eps < best:
            best, best_order = eps, a
    return best, best_order


def sigma_for_epsilon(q: float, steps: int, delta: float, target_eps: float) -> float | None:
    lo, hi = 0.3, 50.0
    if epsilon(q, hi, steps, delta)[0] > target_eps:
        return None
    for _ in range(40):
        mid = (lo + hi) / 2
        if epsilon(q, mid, steps, delta)[0] > target_eps:
            lo = mid
        else:
            hi = mid
    return hi
