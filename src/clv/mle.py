"""Maximum likelihood on log-parameters with Hessian-based standard errors."""

from __future__ import annotations

from collections.abc import Callable
from dataclasses import dataclass

import numpy as np
from scipy.optimize import minimize


@dataclass(frozen=True)
class MLEResult:
    """Point estimates, delta-method standard errors and fit diagnostics."""

    names: tuple[str, ...]
    estimate: np.ndarray
    se: np.ndarray  # NaN when the Hessian is not positive definite
    loglik: float
    n_obs: int
    converged: bool
    message: str

    def as_dict(self) -> dict[str, dict[str, float]]:
        """``{name: {"estimate", "se", "ci95_low", "ci95_high"}}`` (log-scale Wald CI)."""
        out = {}
        for k, name in enumerate(self.names):
            est, se = float(self.estimate[k]), float(self.se[k])
            log_se = se / est if np.isfinite(se) and se / est < 50 else np.nan
            out[name] = {
                "estimate": est,
                "se": se,
                "ci95_low": est * float(np.exp(-1.959964 * log_se)),
                "ci95_high": est * float(np.exp(1.959964 * log_se)),
            }
        return out


def numerical_hessian(f: Callable[[np.ndarray], float], x0: np.ndarray, step: float = 1e-4
                      ) -> np.ndarray:
    """Central-difference Hessian of a scalar function."""
    x0 = np.asarray(x0, dtype=np.float64)
    k = x0.size
    h = step * np.maximum(1.0, np.abs(x0))
    hess = np.empty((k, k))
    f0 = f(x0)
    for i in range(k):
        ei = np.zeros(k)
        ei[i] = h[i]
        hess[i, i] = (f(x0 + ei) - 2.0 * f0 + f(x0 - ei)) / h[i] ** 2
        for j in range(i):
            ej = np.zeros(k)
            ej[j] = h[j]
            val = (
                f(x0 + ei + ej) - f(x0 + ei - ej) - f(x0 - ei + ej) + f(x0 - ei - ej)
            ) / (4.0 * h[i] * h[j])
            hess[i, j] = hess[j, i] = val
    return hess


def _projected_gradient(f: Callable[[np.ndarray], float], x: np.ndarray,
                        bounds: tuple[float, float], step: float = 1e-5) -> float:
    """Max |central-difference gradient|, ignoring components that push against a bound."""
    grad = np.empty_like(x)
    for i in range(x.size):
        e = np.zeros_like(x)
        e[i] = step
        grad[i] = (f(x + e) - f(x - e)) / (2 * step)
    at_lower = (x <= bounds[0] + 1e-8) & (grad > 0)
    at_upper = (x >= bounds[1] - 1e-8) & (grad < 0)
    return float(np.max(np.abs(np.where(at_lower | at_upper, 0.0, grad))))


def fit_log_params(
    loglik: Callable[[np.ndarray], np.ndarray],
    names: tuple[str, ...],
    init: np.ndarray,
    weights: np.ndarray | None = None,
    bounds: tuple[float, float] = (-15.0, 15.0),
) -> MLEResult:
    """Maximise ``sum(weights * loglik(theta))`` over ``log(theta)`` with L-BFGS-B.

    ``loglik`` maps a positive parameter vector to per-observation log-likelihoods. Standard
    errors come from the inverse numerical Hessian of the total negative log-likelihood in log
    space, mapped back with the delta method: ``se(theta) = theta * se(log theta)``.
    """
    w = None if weights is None else np.asarray(weights, dtype=np.float64)

    def total(log_theta: np.ndarray) -> float:
        ll = loglik(np.exp(log_theta))
        val = float(np.sum(ll if w is None else w * ll))
        return -val if np.isfinite(val) else 1e300

    n_obs = int(len(loglik(np.asarray(init, dtype=np.float64))) if w is None else w.sum())
    scale = 1.0 / max(n_obs, 1)
    start = np.log(np.asarray(init, dtype=np.float64))

    def objective(log_theta: np.ndarray) -> float:
        return total(log_theta) * scale

    opts = {"maxiter": 5000, "ftol": 1e-15, "gtol": 1e-10}
    first = minimize(objective, start, method="L-BFGS-B", jac="3-point",
                     bounds=[bounds] * len(names), options=opts)
    # One restart from the optimum: finite-difference gradients can stop the first run early.
    second = minimize(objective, first.x, method="L-BFGS-B", jac="3-point",
                      bounds=[bounds] * len(names), options=opts)
    res = second if second.fun <= first.fun else first
    log_hat = np.asarray(res.x, dtype=np.float64)
    # L-BFGS-B's line search can report failure on a flat ridge (e.g. BG/NBD when the data show
    # no dropout) although the optimum has been reached; judge convergence by the gradient too.
    converged = bool(first.success or second.success
                     or _projected_gradient(objective, log_hat, bounds) < 1e-5)
    theta = np.exp(log_hat)
    se = np.full(len(names), np.nan)
    hess = numerical_hessian(total, log_hat)
    if np.all(np.isfinite(hess)):
        try:
            cov = np.linalg.inv(hess)
            diag = np.diag(cov)
            if np.all(diag > 0) and np.all(np.linalg.eigvalsh(hess) > 0):
                se = theta * np.sqrt(diag)
        except np.linalg.LinAlgError:
            pass
    return MLEResult(
        names=names,
        estimate=theta,
        se=se,
        loglik=-total(log_hat),
        n_obs=n_obs,
        converged=converged,
        message=str(res.message),
    )
