"""Tests for covariance_shrinkage.inference: variance_difference_test."""

import math

import numpy as np
import pytest

from covariance_shrinkage.inference import variance_difference_test


def two_sided_p(t: float) -> float:
    """P(|Z| > |t|) for a standard normal Z."""
    return math.erfc(abs(t) / math.sqrt(2))


def clustered_and_iid(t: int, rng: np.random.Generator) -> tuple[np.ndarray, np.ndarray]:
    """Two independent series with variance 1: a has volatility clustering, b is iid N(0, 1).

    a_t = exp(s_t / 2) z_t with s_t a stationary AR(1) (coefficient 0.8, innovation sd 0.6),
    centred so that E exp(s_t) = 1. The variances are equal, but the squares of a are
    autocorrelated, so the daily differences of squared returns are too.
    """
    phi, sd = 0.8, 0.6
    v = sd**2 / (1 - phi**2)
    s = np.empty(t)
    s[0] = rng.normal(0.0, math.sqrt(v))
    shocks = rng.normal(0.0, sd, t)
    for i in range(1, t):
        s[i] = phi * s[i - 1] + shocks[i]
    a = np.exp((s - v / 2) / 2) * rng.standard_normal(t)
    return a, rng.standard_normal(t)


# --- known answers ----------------------------------------------------------------------

# Both series have mean 0. Squares: a -> 1, 1, 1, 1 (variance 1); b -> 4, 0, 0, 4 (variance 2).
# Daily differences of squares: -3, 1, 1, -3, mean -1; deviations from the mean -2, 2, 2, -2.
# Autocovariances (divided by T = 4): lag 0: 16/4 = 4; lag 1: (-4 + 4 - 4)/4 = -1;
# lag 2: (-4 - 4)/4 = -2; lag 3: (-2)(-2)/4 = 1.
A4 = np.array([1.0, -1.0, 1.0, -1.0])
B4 = np.array([2.0, 0.0, 0.0, -2.0])


def test_known_answer_iid_standard_error():
    # lags = 0: long-run variance 4, se = sqrt(4 / 4) = 1, t = -1.
    res = variance_difference_test(A4, B4, lags=0)
    assert res.diff == pytest.approx(-1.0)
    assert res.se == pytest.approx(1.0)
    assert res.t == pytest.approx(-1.0)
    assert res.p_value == pytest.approx(two_sided_p(1.0))
    assert res.lags == 0


def test_known_answer_one_lag():
    # lags = 1: weight 1/2, long-run variance 4 + 2 * (1/2) * (-1) = 3, se = sqrt(3/4).
    res = variance_difference_test(A4, B4, lags=1)
    assert res.diff == pytest.approx(-1.0)
    assert res.se == pytest.approx(math.sqrt(3 / 4))
    assert res.t == pytest.approx(-1.0 / math.sqrt(3 / 4))
    assert res.p_value == pytest.approx(two_sided_p(1.0 / math.sqrt(3 / 4)))


def test_known_answer_two_lags():
    # lags = 2: weights 2/3, 1/3: 4 + 2 * (2/3) * (-1) + 2 * (1/3) * (-2) = 4/3, se = sqrt(1/3).
    res = variance_difference_test(A4, B4, lags=2)
    assert res.se == pytest.approx(math.sqrt(1 / 3))
    assert res.lags == 2


def test_known_answer_largest_lag():
    # lags = T - 1 = 3 is allowed: weights 3/4, 1/2, 1/4:
    # 4 + 2 * (3/4) * (-1) + 2 * (1/2) * (-2) + 2 * (1/4) * 1 = 1, se = sqrt(1/4) = 0.5.
    res = variance_difference_test(A4, B4, lags=3)
    assert res.se == pytest.approx(0.5)
    assert res.lags == 3


def test_numpy_integer_lags_accepted():
    # e.g. a lag taken from np.arange in a robustness loop.
    res = variance_difference_test(A4, B4, lags=np.int64(1))
    assert res.se == pytest.approx(math.sqrt(3 / 4))


def test_bartlett_weights_keep_the_variance_positive():
    # The daily differences of squares alternate 4, -4, 4, ... (T = 8, mean 0). Lag 0: 16;
    # lag 1: 7 * (-16) / 8 = -14. Truncation at lag 1 would give 16 - 28 < 0; the Bartlett
    # weight 1/2 gives 16 - 14 = 2, so se = sqrt(2 / 8) = 0.5, and diff = 0.
    a = np.array([2.0, 0.0, -2.0, 0.0, 2.0, 0.0, -2.0, 0.0])
    b = np.array([0.0, 2.0, 0.0, -2.0, 0.0, 2.0, 0.0, -2.0])
    res = variance_difference_test(a, b, lags=1)
    assert res.diff == pytest.approx(0.0, abs=1e-12)
    assert res.se == pytest.approx(0.5)
    assert res.p_value == pytest.approx(1.0)


def test_b_twice_a(rng):
    # Var(2a) = 4 Var(a), so diff = -3 Var(a) exactly, and with T = 1000 the difference is
    # far beyond noise (t is about -sqrt(T / 2), around -22, for normal data).
    a = rng.standard_normal(1000)
    res = variance_difference_test(a, 2 * a)
    assert res.diff == pytest.approx(-3 * a.var())
    assert res.t < -10
    assert res.p_value < 1e-10


@pytest.mark.parametrize("t, expected", [(1000, 7), (27000, 22), (4, 1)])
def test_default_lags(t, expected, rng):
    # floor(0.75 * T^(1/3)): 0.75 * 10 = 7.5, 0.75 * 30 = 22.5, 0.75 * 1.587 = 1.19.
    a, b = rng.standard_normal(t), rng.standard_normal(t)
    assert variance_difference_test(a, b).lags == expected


# --- invariances ------------------------------------------------------------------------


def test_swapping_the_series_flips_the_sign(rng):
    a, b = rng.standard_normal(500), 1.2 * rng.standard_normal(500)
    ab, ba = variance_difference_test(a, b), variance_difference_test(b, a)
    assert ba.diff == pytest.approx(-ab.diff)
    assert ba.t == pytest.approx(-ab.t)
    assert ba.se == pytest.approx(ab.se)
    assert ba.p_value == pytest.approx(ab.p_value)


def test_scaling_both_series(rng):
    # Variances scale by c^2, so diff and se do too; t and the p-value don't change.
    a, b, c = rng.standard_normal(500), 1.2 * rng.standard_normal(500), 0.01
    res, scaled = variance_difference_test(a, b), variance_difference_test(c * a, c * b)
    assert scaled.diff == pytest.approx(c**2 * res.diff)
    assert scaled.se == pytest.approx(c**2 * res.se)
    assert scaled.t == pytest.approx(res.t)
    assert scaled.p_value == pytest.approx(res.p_value)


def test_each_series_is_centred_at_its_own_mean(rng):
    # Adding different constants to a and b changes neither variance.
    a, b = rng.standard_normal(500), 1.2 * rng.standard_normal(500)
    res, shifted = variance_difference_test(a, b), variance_difference_test(a + 0.5, b - 2.0)
    assert shifted.diff == pytest.approx(res.diff)
    assert shifted.se == pytest.approx(res.se)
    assert shifted.t == pytest.approx(res.t)


# --- simulation with a known truth ------------------------------------------------------


def test_size_under_equal_variances_iid(rng):
    # H0 true: correlated pairs (rho = 0.5), iid normal, T = 500, default lags (5).
    # 2,000 repetitions: the binomial sd of a 5 % rate is 0.5 percentage points, and a
    # reference implementation rejects 5.3 % with this seed; the band is about +-3 sd.
    chol = np.linalg.cholesky(np.array([[1.0, 0.5], [0.5, 1.0]]))
    rejections = 0
    for _ in range(2000):
        x = rng.standard_normal((500, 2)) @ chol.T
        rejections += variance_difference_test(x[:, 0], x[:, 1]).p_value < 0.05
    assert 0.035 <= rejections / 2000 <= 0.07


def test_newey_west_corrects_for_volatility_clustering(rng):
    # H0 true, but the squares of a are autocorrelated. The iid standard error (lags = 0) is
    # too small, so the test over-rejects badly; 30 lags bring it much closer to 5 %.
    # Reference implementation, 1,000 repetitions, T = 2,000: about 18 % at lags = 0 and
    # 7-8 % at lags = 30 (still above 5 %: the volatility process also fattens the tails).
    # Binomial sd: 1.2 points at 18 %, 0.9 points at 8 %.
    rejections = {0: 0, 30: 0}
    for _ in range(1000):
        a, b = clustered_and_iid(2000, rng)
        for lags in rejections:
            rejections[lags] += variance_difference_test(a, b, lags=lags).p_value < 0.05
    assert rejections[0] / 1000 > 0.13
    assert rejections[30] / 1000 < 0.11


# --- bad input --------------------------------------------------------------------------


@pytest.mark.parametrize(
    "a, b",
    [
        (np.ones((3, 2)), np.ones((3, 2))),  # not 1-D
        (np.array([0.1, 0.2, 0.3]), np.array([0.1, 0.2])),  # lengths differ
        (np.array([0.1]), np.array([0.2])),  # T < 2
        (np.array([0.1, np.nan, 0.3]), np.array([0.1, 0.2, 0.3])),  # NaN
        (np.array([0.1, 0.2, 0.3]), np.array([0.1, np.inf, 0.3])),  # inf
    ],
)
def test_rejects_bad_input(a, b):
    with pytest.raises(ValueError):
        variance_difference_test(a, b)


@pytest.mark.parametrize("lags", [-1, 4, 10])
def test_rejects_lags_outside_range(lags):
    with pytest.raises(ValueError):
        variance_difference_test(A4, B4, lags=lags)


@pytest.mark.parametrize("lags", [1.5, 1.0, "1"])
def test_rejects_non_integer_lags(lags):
    with pytest.raises(TypeError):
        variance_difference_test(A4, B4, lags=lags)


def test_identical_series_raise(rng):
    # The daily difference is 0 every day: diff = se = 0 and t would be 0 / 0.
    a = rng.standard_normal(100)
    with pytest.raises(ValueError):
        variance_difference_test(a, a.copy())


@pytest.mark.parametrize("c", [0.1, 1.0, 1000.0])
def test_series_equal_up_to_a_constant_raise(c, rng):
    # b = a + c has the same demeaned returns, so the daily difference is 0 up to rounding:
    # the long-run variance comes out around 1e-31 (1e-26 for c = 1000), not exactly 0, and
    # the t-statistic would be rounding noise divided by rounding noise.
    a = rng.standard_normal(1000)
    with pytest.raises(ValueError):
        variance_difference_test(a, a + c)


def test_highly_correlated_daily_returns_do_not_raise(rng):
    # Realistic daily scale (sd 0.5 %) and two strategies with correlation 0.9999: the
    # long-run variance is small in absolute terms (about 4 (1 - rho^2) sd^4, i.e. ~5e-13),
    # but it is signal, not rounding noise. A threshold on the variance itself (e.g. 1e-12)
    # would wrongly raise here; one relative to the data's scale doesn't.
    sd, rho = 0.005, 0.9999
    a = sd * rng.standard_normal(1000)
    b = rho * a + math.sqrt(1 - rho**2) * sd * rng.standard_normal(1000)
    res = variance_difference_test(a, b)
    assert res.se > 0
    assert math.isfinite(res.t)
