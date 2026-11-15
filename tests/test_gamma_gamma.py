import math

import numpy as np
import pytest
from scipy import integrate

from clv import gamma_gamma
from clv.gamma_gamma import GammaGammaParams

# Fader & Hardie (2013) CDNOW estimates.
CDNOW = GammaGammaParams(p=6.25, q=3.74, gamma=15.44)


@pytest.mark.parametrize("x, m", [(1, 20.0), (3, 35.5), (12, 8.0)])
def test_loglik_matches_integration_over_nu(x, m):
    """Mean of x Gamma(p, nu) spends is Gamma(p x, x nu); integrate nu ~ Gamma(q, gamma)."""
    p, q, g = CDNOW.p, CDNOW.q, CDNOW.gamma

    def integrand(nu):
        log_m = (p * x * math.log(x * nu) - math.lgamma(p * x) + (p * x - 1) * math.log(m)
                 - x * nu * m)
        log_nu = q * math.log(g) - math.lgamma(q) + (q - 1) * math.log(nu) - g * nu
        return math.exp(log_m + log_nu)

    direct, _ = integrate.quad(integrand, 0, 5, epsabs=0.0, epsrel=1e-11, limit=200)
    assert math.log(direct) == pytest.approx(float(gamma_gamma.loglik(CDNOW, x, m)), abs=1e-7)


@pytest.fixture(scope="module")
def simulated():
    rng = np.random.default_rng(11)
    x = rng.integers(1, 15, size=6000)
    nu = rng.gamma(CDNOW.q, 1.0 / CDNOW.gamma, size=x.size)
    m = rng.gamma(CDNOW.p * x, 1.0 / nu) / x
    return x, m, nu


def test_mle_recovers_simulated_parameters(simulated):
    x, m, _ = simulated
    res = gamma_gamma.fit(x, m)
    assert res.converged and np.all(np.isfinite(res.se))
    z = (res.estimate - CDNOW.as_array()) / res.se
    assert np.all(np.abs(z) < 3.5), z
    np.testing.assert_allclose(res.estimate, CDNOW.as_array(), rtol=0.25)


def test_simulate_matches_generative_process():
    rng = np.random.default_rng(5)
    m = gamma_gamma.simulate(CDNOW, np.full(200_000, 1), rng=rng)
    # E[m | x = 1] = p * E[1/nu] = p * gamma / (q - 1)
    assert m.mean() == pytest.approx(CDNOW.population_mean_spend, rel=0.02)


def test_conditional_expectation_is_posterior_mean_of_true_spend(simulated):
    x, m, nu = simulated
    expected = gamma_gamma.conditional_expected_spend(CDNOW, x, m)
    true_mean = CDNOW.p / nu
    for lo, hi in ((1, 2), (2, 5), (5, 15)):
        sel = (x >= lo) & (x < hi)
        diff = true_mean[sel] - expected[sel]
        assert abs(diff.mean()) < 4 * diff.std() / np.sqrt(sel.sum())
    # Shrinkage: always between the population mean and the customer's own mean.
    lo_b = np.minimum(m, CDNOW.population_mean_spend)
    hi_b = np.maximum(m, CDNOW.population_mean_spend)
    assert np.all((expected >= lo_b - 1e-9) & (expected <= hi_b + 1e-9))


def test_no_history_gives_population_mean():
    out = gamma_gamma.conditional_expected_spend(CDNOW, [0], [0.0])
    assert out[0] == pytest.approx(CDNOW.population_mean_spend)


def test_frequency_spend_correlation_and_input_checks():
    corr = gamma_gamma.frequency_spend_correlation([1, 2, 3, 4, 0], [10.0, 20.0, 30.0, 40.0, 0.0])
    assert corr["n"] == 4 and corr["pearson"] == pytest.approx(1.0)
    with pytest.raises(ValueError):
        gamma_gamma.fit([0, 1], [5.0, 5.0])
