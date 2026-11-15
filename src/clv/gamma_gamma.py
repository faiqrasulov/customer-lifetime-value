"""Gamma-Gamma spend model (Fader, Hardie & Lee 2005; Fader & Hardie 2013 note).

Each transaction value of a customer is ``z ~ Gamma(p, rate=nu)``, and ``nu ~ Gamma(q, rate=gamma)``
across customers. Given x transactions with mean ``m``, ``m ~ Gamma(p*x, rate=x*nu)``; integrating
out ``nu`` gives the marginal likelihood used for estimation. The model assumes a customer's
average spend is independent of how often they buy, which :func:`frequency_spend_correlation`
lets you check.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
from scipy.special import gammaln

from clv.mle import MLEResult, fit_log_params

PARAM_NAMES = ("p", "q", "gamma")


@dataclass(frozen=True)
class GammaGammaParams:
    """Shape ``p`` of a customer's spend; Gamma(q, gamma) prior on the spend rate ``nu``."""

    p: float
    q: float
    gamma: float

    def as_array(self) -> np.ndarray:
        return np.array([self.p, self.q, self.gamma], dtype=np.float64)

    @property
    def population_mean_spend(self) -> float:
        """E[M] = p * gamma / (q - 1) (infinite when q <= 1)."""
        return self.p * self.gamma / (self.q - 1.0) if self.q > 1.0 else float("inf")


def loglik(params: GammaGammaParams | np.ndarray, x, m) -> np.ndarray:
    """Per-customer log-likelihood of the mean spend ``m`` over ``x >= 1`` transactions."""
    p, q, g = np.asarray(
        params.as_array() if isinstance(params, GammaGammaParams) else params, dtype=np.float64
    )
    x = np.asarray(x, dtype=np.float64)
    m = np.asarray(m, dtype=np.float64)
    px = p * x
    return (
        gammaln(px + q)
        - gammaln(px)
        - gammaln(q)
        + q * np.log(g)
        + (px - 1.0) * np.log(m)
        + px * np.log(x)
        - (px + q) * np.log(g + x * m)
    )


def fit(x, m, weights=None) -> MLEResult:
    """MLE of (p, q, gamma) from repeat customers (x >= 1, m > 0)."""
    x = np.asarray(x, dtype=np.float64)
    m = np.asarray(m, dtype=np.float64)
    if np.any(x < 1) or np.any(m <= 0):
        raise ValueError("Gamma-Gamma needs customers with x >= 1 and m > 0")
    # Method-of-moments-flavoured start: p = 1, q = 2, gamma = mean spend.
    init = np.array([1.0, 2.0, float(np.mean(m))])
    return fit_log_params(lambda th: loglik(th, x, m), PARAM_NAMES, init, weights)


def params_from(result: MLEResult) -> GammaGammaParams:
    return GammaGammaParams(*map(float, result.estimate))


def conditional_expected_spend(params: GammaGammaParams, x, m) -> np.ndarray:
    """E[M | x, m] = p (gamma + x m) / (p x + q - 1): a credibility-weighted mean of the
    population mean and the customer's own mean. For x = 0 it is the population mean."""
    x = np.asarray(x, dtype=np.float64)
    m = np.where(x > 0, np.asarray(m, dtype=np.float64), 0.0)
    return params.p * (params.gamma + x * m) / (params.p * x + params.q - 1.0)


def frequency_spend_correlation(x, m) -> dict[str, float]:
    """Pearson and Spearman correlation between frequency and mean spend (repeat buyers)."""
    from scipy import stats

    x = np.asarray(x, dtype=np.float64)
    m = np.asarray(m, dtype=np.float64)
    keep = x > 0
    return {
        "pearson": float(stats.pearsonr(x[keep], m[keep])[0]),
        "spearman": float(stats.spearmanr(x[keep], m[keep])[0]),
        "n": int(keep.sum()),
    }


def simulate(params: GammaGammaParams, x, rng: np.random.Generator | None = None) -> np.ndarray:
    """Mean spend over ``x`` transactions for each customer (x >= 1) under the model."""
    rng = rng or np.random.default_rng(0)
    x = np.asarray(x, dtype=np.int64)
    nu = rng.gamma(params.q, 1.0 / params.gamma, size=x.size)
    totals = rng.gamma(params.p * x, 1.0 / nu)
    return totals / x
