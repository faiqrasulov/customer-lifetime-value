"""BG/NBD model (Fader, Hardie & Lee 2005), implemented from the paper's equations.

Generative story for one customer, starting with a first purchase at time 0:

- while alive, purchases follow a Poisson process with rate ``lam ~ Gamma(r, rate=alpha)``;
- after each *repeat* purchase the customer drops out for good with probability
  ``p ~ Beta(a, b)``.

Data per customer: ``x`` repeat purchases, last one at ``t_x``, observed until ``T``.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
from scipy.special import betaln, expit, gammaln, hyp2f1

from clv.mle import MLEResult, fit_log_params

PARAM_NAMES = ("r", "alpha", "a", "b")


@dataclass(frozen=True)
class BGNBDParams:
    """Population parameters: Gamma(r, alpha) purchase rates, Beta(a, b) dropout probabilities."""

    r: float
    alpha: float
    a: float
    b: float

    def as_array(self) -> np.ndarray:
        return np.array([self.r, self.alpha, self.a, self.b], dtype=np.float64)


def _arrays(*values) -> list[np.ndarray]:
    return [np.asarray(v, dtype=np.float64) for v in values]


def loglik(params: BGNBDParams | np.ndarray, x, t_x, T) -> np.ndarray:
    """Per-customer log-likelihood, eq. (6) of Fader, Hardie & Lee (2005), in log space."""
    r, alpha, a, b = np.asarray(
        params.as_array() if isinstance(params, BGNBDParams) else params, dtype=np.float64
    )
    x, t_x, T = _arrays(x, t_x, T)
    a1 = gammaln(r + x) - gammaln(r) + r * np.log(alpha)
    a2 = betaln(a, b + x) - betaln(a, b) - (r + x) * np.log(alpha + T)
    repeat = x > 0
    x_safe = np.where(repeat, x, 1.0)
    a3 = betaln(a + 1.0, b + x_safe - 1.0) - betaln(a, b) - (r + x) * np.log(alpha + t_x)
    return a1 + np.logaddexp(a2, np.where(repeat, a3, -np.inf))


def fit(x, t_x, T, weights=None, init: tuple[float, ...] = (1.0, 1.0, 1.0, 1.0)) -> MLEResult:
    """Maximum likelihood estimates of (r, alpha, a, b) with Hessian standard errors."""
    x, t_x, T = _arrays(x, t_x, T)
    if np.any(x < 0) or np.any(t_x < 0) or np.any(t_x > T) or np.any((x == 0) & (t_x != 0)):
        raise ValueError("need x >= 0, 0 <= t_x <= T and t_x = 0 when x = 0")
    # Start alpha near the data's time scale so the optimiser begins in a sensible region.
    init_arr = np.asarray(init, dtype=np.float64)
    if init == (1.0, 1.0, 1.0, 1.0):
        init_arr[1] = max(float(np.mean(T)), 1e-3)
    return fit_log_params(lambda th: loglik(th, x, t_x, T), PARAM_NAMES, init_arr, weights)


def loglik_no_dropout(params, x, T) -> np.ndarray:
    """Log-likelihood of the nested model without dropout (p = 0, i.e. the NBD timing model).

    BG/NBD approaches it as a -> 0 or b -> infinity; comparing the two log-likelihoods shows
    how much the data support a dropout process at all.
    """
    r, alpha = np.asarray(params, dtype=np.float64)
    x, T = _arrays(x, T)
    return gammaln(r + x) - gammaln(r) + r * np.log(alpha) - (r + x) * np.log(alpha + T)


def fit_no_dropout(x, T) -> MLEResult:
    """MLE of (r, alpha) for the no-dropout (NBD) model."""
    x, T = _arrays(x, T)
    init = np.array([1.0, max(float(np.mean(T)), 1e-3)])
    return fit_log_params(lambda th: loglik_no_dropout(th, x, T), ("r", "alpha"), init)


def params_from(result: MLEResult) -> BGNBDParams:
    return BGNBDParams(*map(float, result.estimate))


def p_alive(params: BGNBDParams, x, t_x, T) -> np.ndarray:
    """P(customer still active at T | x, t_x, T), eq. (13). Always 1 when x = 0."""
    x, t_x, T = _arrays(x, t_x, T)
    log_odds_dead = (
        np.log(params.a)
        - np.log(np.where(x > 0, params.b + x - 1.0, 1.0))
        + (params.r + x) * (np.log(params.alpha + T) - np.log(params.alpha + t_x))
    )
    return np.where(x > 0, expit(-log_odds_dead), 1.0)


def _expected_if_alive(params: BGNBDParams, t, x, T) -> np.ndarray:
    """E[purchases in (T, T+t] | alive at T, x, T].

    Posterior given alive: lam ~ Gamma(r+x, alpha+T), p ~ Beta(a, b+x). Eq. (10)'s factor
    ``((alpha+T)/(alpha+T+t))^(r+x) * 2F1(r+x, b+x; a+b+x-1; z)`` is rewritten with Euler's
    transformation as ``(1-z)^(a-1) * 2F1(a+b-1-r, a-1; a+b+x-1; z)``: the parameters of the
    hypergeometric function no longer grow with x, which keeps it accurate for heavy buyers.
    """
    r, alpha, a, b = params.r, params.alpha, params.a, params.b
    t, x, T = _arrays(t, x, T)
    z = t / (alpha + T + t)
    c = a + b + x - 1.0
    tail = np.exp((a - 1.0) * np.log1p(-z)) * hyp2f1(a + b - 1.0 - r, a - 1.0, c, z)
    return c / (a - 1.0) * (1.0 - tail)


def expected_transactions(params: BGNBDParams, t) -> np.ndarray:
    """E[X(t)]: expected repeat purchases in (0, t] for a customer born at 0, eq. (9)."""
    t = np.asarray(t, dtype=np.float64)
    return _expected_if_alive(params, t, np.zeros_like(t), np.zeros_like(t))


def conditional_expected_transactions(params: BGNBDParams, t, x, t_x, T) -> np.ndarray:
    """E[Y(t) | x, t_x, T]: expected purchases in (T, T+t], eq. (10)."""
    return p_alive(params, x, t_x, T) * _expected_if_alive(params, t, x, T)


def simulate(
    params: BGNBDParams, T, holdout: float = 0.0, rng: np.random.Generator | None = None
) -> dict[str, np.ndarray]:
    """Simulate customers from the generative process over (0, T] and (T, T + holdout].

    Returns ``x``, ``t_x``, ``T``, ``holdout`` (purchases in the holdout window), ``alive``
    (still active at T), and the latent ``lam`` and ``p``.
    """
    rng = rng or np.random.default_rng(0)
    T = np.asarray(T, dtype=np.float64)
    n = T.size
    lam = rng.gamma(params.r, 1.0 / params.alpha, size=n)
    p = rng.beta(params.a, params.b, size=n)
    now = np.zeros(n)
    x = np.zeros(n, dtype=np.int64)
    t_x = np.zeros(n)
    hold = np.zeros(n, dtype=np.int64)
    alive_at_T = np.ones(n, dtype=bool)
    active = np.ones(n, dtype=bool)
    end = T + holdout
    while active.any():
        idx = np.flatnonzero(active)
        now[idx] += rng.exponential(1.0, size=idx.size) / lam[idx]
        done = now[idx] > end[idx]
        active[idx[done]] = False
        idx = idx[~done]
        in_cal = now[idx] <= T[idx]
        cal_idx, hold_idx = idx[in_cal], idx[~in_cal]
        x[cal_idx] += 1
        t_x[cal_idx] = now[cal_idx]
        hold[hold_idx] += 1
        dies = rng.random(idx.size) < p[idx]
        active[idx[dies]] = False
        alive_at_T[cal_idx[dies[in_cal]]] = False
    return {"x": x, "t_x": t_x, "T": T, "holdout": hold, "alive": alive_at_T, "lam": lam, "p": p}
