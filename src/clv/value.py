"""Customer lifetime value = expected purchases x expected spend x margin, discounted monthly."""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from clv import bgnbd, gamma_gamma
from clv.bgnbd import BGNBDParams
from clv.gamma_gamma import GammaGammaParams

WEEKS_PER_MONTH = 365.25 / 12.0 / 7.0


@dataclass(frozen=True)
class ValueAssumptions:
    """Business assumptions behind the money numbers (not estimated from the data).

    ``margin``: gross margin on revenue. The dataset has no cost data, so 25% is an assumed,
    mid-range figure for a gift wholesaler; it scales every CLV linearly and does not change
    any ranking. ``annual_discount_rate``: cost of capital, converted to a monthly rate.
    """

    months: int = 12
    margin: float = 0.25
    annual_discount_rate: float = 0.10

    @property
    def monthly_discount_rate(self) -> float:
        return (1.0 + self.annual_discount_rate) ** (1.0 / 12.0) - 1.0


def monthly_expected_transactions(
    params: BGNBDParams, x, t_x, T, months: int
) -> np.ndarray:
    """Expected purchases in each of the next ``months`` months, shape (n_customers, months)."""
    x, t_x, T = (np.asarray(v, dtype=np.float64) for v in (x, t_x, T))
    edges = np.arange(months + 1) * WEEKS_PER_MONTH
    cum = np.stack(
        [bgnbd.conditional_expected_transactions(params, e, x, t_x, T) for e in edges[1:]], axis=1
    )
    return np.diff(np.concatenate([np.zeros((cum.shape[0], 1)), cum], axis=1), axis=1)


def customer_lifetime_value(
    bg: BGNBDParams,
    gg: GammaGammaParams,
    x,
    t_x,
    T,
    m,
    assumptions: ValueAssumptions | None = None,
) -> dict[str, np.ndarray]:
    """Discounted CLV over the horizon plus its parts, per customer.

    Purchases in month k are discounted by ``(1 + d)^k``. Returns ``clv``,
    ``expected_transactions`` (undiscounted, whole horizon), ``expected_spend`` and ``p_alive``.
    """
    a = assumptions or ValueAssumptions()
    per_month = monthly_expected_transactions(bg, x, t_x, T, a.months)
    spend = gamma_gamma.conditional_expected_spend(gg, x, m)
    discount = (1.0 + a.monthly_discount_rate) ** -np.arange(1, a.months + 1)
    return {
        "clv": a.margin * spend * (per_month @ discount),
        "expected_transactions": per_month.sum(axis=1),
        "expected_spend": spend,
        "p_alive": bgnbd.p_alive(bg, x, t_x, T),
    }
