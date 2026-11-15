import math

import numpy as np
import pytest
from scipy import integrate
from scipy.special import betaln, hyp2f1

from clv import bgnbd
from clv.bgnbd import BGNBDParams

# Fader, Hardie & Lee (2005) CDNOW estimates (weeks): a realistic, identifiable setting.
CDNOW = BGNBDParams(r=0.243, alpha=4.414, a=0.793, b=2.426)


def _individual_likelihood(lam, p, x, t_x, T):
    """Eq. (3) of the paper for one customer with known (lambda, p)."""
    alive = (1 - p) ** x * lam**x * math.exp(-lam * T)
    dead = p * (1 - p) ** (x - 1) * lam**x * math.exp(-lam * t_x) if x > 0 else 0.0
    return alive + dead


@pytest.mark.parametrize("x, t_x, T", [(0, 0.0, 10.0), (1, 3.0, 12.0), (4, 20.5, 30.0)])
def test_loglik_matches_direct_integration(x, t_x, T):
    """Closed form vs numerically integrating eq. (3) over the Gamma and Beta mixing laws."""
    r, alpha, a, b = 1.3, 6.0, 1.5, 4.0
    log_gamma_norm = r * math.log(alpha) - math.lgamma(r)
    log_beta_norm = -(math.lgamma(a) + math.lgamma(b) - math.lgamma(a + b))

    def integrand(p, lam):
        if lam == 0.0 or p in (0.0, 1.0):
            return 0.0
        dens = math.exp(log_gamma_norm + (r - 1) * math.log(lam) - alpha * lam + log_beta_norm
                        + (a - 1) * math.log(p) + (b - 1) * math.log1p(-p))
        return _individual_likelihood(lam, p, x, t_x, T) * dens

    direct, _ = integrate.dblquad(integrand, 0, 4, 0, 1, epsabs=0.0, epsrel=1e-9)
    ours = float(bgnbd.loglik(BGNBDParams(r, alpha, a, b), x, t_x, T))
    assert math.log(direct) == pytest.approx(ours, abs=1e-6)


def test_loglik_is_vectorised_and_finite():
    rng = np.random.default_rng(1)
    x = rng.integers(0, 30, size=200)
    T = rng.uniform(1, 52, size=200)
    t_x = np.where(x > 0, rng.uniform(0, 1, size=200) * T, 0.0)
    ll = bgnbd.loglik(CDNOW, x, t_x, T)
    assert ll.shape == (200,) and np.all(np.isfinite(ll))
    one_by_one = [float(bgnbd.loglik(CDNOW, a, b, c)) for a, b, c in zip(x, t_x, T, strict=True)]
    np.testing.assert_allclose(ll, one_by_one, rtol=1e-12)


@pytest.fixture(scope="module")
def simulated():
    rng = np.random.default_rng(2024)
    T = rng.uniform(26.0, 52.0, size=8000)  # staggered acquisition, as in real cohorts
    return bgnbd.simulate(CDNOW, T, holdout=39.0, rng=rng)


def test_mle_recovers_simulated_parameters(simulated):
    res = bgnbd.fit(simulated["x"], simulated["t_x"], simulated["T"])
    assert res.converged
    truth = CDNOW.as_array()
    assert np.all(np.isfinite(res.se))
    z = (res.estimate - truth) / res.se
    assert np.all(np.abs(z) < 3.5), z
    np.testing.assert_allclose(res.estimate, truth, rtol=0.3)


def test_unconditional_expectation_matches_monte_carlo():
    rng = np.random.default_rng(7)
    t = 30.0
    sim = bgnbd.simulate(CDNOW, np.full(200_000, t), rng=rng)
    mc_mean = sim["x"].mean()
    mc_se = sim["x"].std() / np.sqrt(sim["x"].size)
    assert abs(float(bgnbd.expected_transactions(CDNOW, t)) - mc_mean) < 4 * mc_se


def test_conditional_expectation_matches_monte_carlo(simulated):
    """E[E[Y | data] 1{x in bucket}] = E[Y 1{x in bucket}] for every frequency bucket."""
    s = simulated
    pred = bgnbd.conditional_expected_transactions(CDNOW, 39.0, s["x"], s["t_x"], s["T"])
    bucket = np.minimum(s["x"], 4)
    for b in range(5):
        m = bucket == b
        diff = s["holdout"][m] - pred[m]
        assert abs(diff.mean()) < 4 * diff.std() / np.sqrt(m.sum()) + 1e-9, b


def test_p_alive_matches_simulated_survival(simulated):
    s = simulated
    pa = bgnbd.p_alive(CDNOW, s["x"], s["t_x"], s["T"])
    assert np.all(pa[s["x"] == 0] == 1.0)
    for b in (1, 2, 3):
        m = s["x"] == b
        diff = s["alive"][m].astype(float) - pa[m]
        assert abs(diff.mean()) < 4 * diff.std() / np.sqrt(m.sum()), b


def _quadrature_expected_if_alive(params, t, x, T):
    """E[(1 - exp(-lam p t)) / p] with lam ~ Gamma(r+x, alpha+T), p ~ Beta(a, b+x).

    The lambda integral is closed form; the p integral uses u = p**a, which removes the
    p**(a-1) singularity (essential when a is tiny).
    """
    r, al, a, b = params.r, params.alpha, params.a, params.b
    scale = t / (al + T)

    def f(u):
        p = u ** (1.0 / a)
        if p == 0.0:
            g = (r + x) * scale
        else:
            g = -np.expm1(-(r + x) * np.log1p(p * scale)) / p
        return g * (1.0 - p) ** (b + x - 1.0)

    val, _ = integrate.quad(f, 0.0, 1.0, limit=500, epsabs=0.0, epsrel=1e-11)
    return val / a / np.exp(betaln(a, b + x))


@pytest.mark.parametrize("params", [CDNOW, BGNBDParams(0.83, 10.0, 0.0057, 8.5),
                                    BGNBDParams(2.0, 5.0, 3.0, 10.0)])
def test_expectation_given_alive_matches_quadrature_and_paper_form(params):
    """Euler-transformed 2F1 vs (i) eq. (10) as printed and (ii) 1-D quadrature over p."""
    for x, T, t in [(0, 0.0, 13.0), (3, 30.0, 13.0), (40, 52.0, 52.0)]:
        ours = float(bgnbd._expected_if_alive(params, t, x, T))
        r, al, a, b = params.r, params.alpha, params.a, params.b
        z = t / (al + T + t)
        paper = (a + b + x - 1) / (a - 1) * (
            1 - ((al + T) / (al + T + t)) ** (r + x) * hyp2f1(r + x, b + x, a + b + x - 1, z))

        quad = _quadrature_expected_if_alive(params, t, x, T)
        assert ours == pytest.approx(paper, rel=1e-9)
        assert ours == pytest.approx(quad, rel=1e-7)


def test_euler_form_stays_finite_where_paper_form_overflows():
    mpmath = pytest.importorskip("mpmath")
    params = BGNBDParams(5.0, 1.0, 0.3, 0.7)
    x, T, t = 200, 5.0, 200.0
    mpmath.mp.dps = 50
    ref = (mpmath.mpf(params.a) + params.b + x - 1) / (mpmath.mpf(params.a) - 1) * (
        1 - ((params.alpha + T) / mpmath.mpf(params.alpha + T + t)) ** (params.r + x)
        * mpmath.hyp2f1(params.r + x, params.b + x, mpmath.mpf(params.a) + params.b + x - 1,
                        mpmath.mpf(t) / (params.alpha + T + t)))
    assert float(bgnbd._expected_if_alive(params, t, x, T)) == pytest.approx(float(ref), rel=1e-9)


def test_no_dropout_limit_and_nested_model():
    """With a -> 0 BG/NBD reduces to the NBD: E[Y(t)|x, T] = (r + x) / (alpha + T) * t."""
    params = BGNBDParams(0.8, 9.0, 1e-9, 5.0)
    x, t_x, T, t = np.array([0, 3, 12]), np.array([0.0, 10.0, 40.0]), np.array([20.0] * 3), 13.0
    np.testing.assert_allclose(
        bgnbd.conditional_expected_transactions(params, t, x, t_x, T),
        (params.r + x) / (params.alpha + T) * t, rtol=1e-6)
    np.testing.assert_allclose(bgnbd.loglik(params, x, t_x, T),
                               bgnbd.loglik_no_dropout([0.8, 9.0], x, T), atol=1e-6)


def test_fit_rejects_inconsistent_input():
    with pytest.raises(ValueError):
        bgnbd.fit([1, 0], [5.0, 1.0], [10.0, 10.0])  # x = 0 but t_x > 0
    with pytest.raises(ValueError):
        bgnbd.fit([2], [12.0], [10.0])  # t_x > T
