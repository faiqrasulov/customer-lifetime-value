import numpy as np
import pytest

from clv import bgnbd, gamma_gamma
from clv.bgnbd import BGNBDParams
from clv.gamma_gamma import GammaGammaParams
from clv.value import WEEKS_PER_MONTH, ValueAssumptions, customer_lifetime_value

BG = BGNBDParams(0.8, 10.0, 0.4, 3.0)
GG = GammaGammaParams(3.0, 4.0, 400.0)
X, TX, T, M = np.array([0, 2, 10]), np.array([0.0, 20.0, 45.0]), np.array([30.0, 40.0, 50.0]), \
    np.array([0.0, 250.0, 600.0])


def test_undiscounted_clv_is_expected_purchases_times_spend_times_margin():
    a = ValueAssumptions(months=12, margin=1.0, annual_discount_rate=0.0)
    out = customer_lifetime_value(BG, GG, X, TX, T, M, a)
    horizon = 12 * WEEKS_PER_MONTH
    n = bgnbd.conditional_expected_transactions(BG, horizon, X, TX, T)
    np.testing.assert_allclose(out["expected_transactions"], n, rtol=1e-10)
    np.testing.assert_allclose(out["clv"], n * gamma_gamma.conditional_expected_spend(GG, X, M),
                               rtol=1e-10)


def test_discounting_and_margin_scale_value_down():
    base = customer_lifetime_value(BG, GG, X, TX, T, M, ValueAssumptions(margin=1.0,
                                                                          annual_discount_rate=0.0))
    disc = customer_lifetime_value(BG, GG, X, TX, T, M, ValueAssumptions(margin=0.25))
    assert np.all(disc["clv"] < 0.25 * base["clv"])
    assert np.all(disc["clv"] > 0.25 * base["clv"] / 1.10)  # at most one year of discounting
    assert ValueAssumptions().monthly_discount_rate == pytest.approx(1.1 ** (1 / 12) - 1)
