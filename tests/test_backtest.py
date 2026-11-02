"""Tests for covariance_shrinkage.backtest: run_backtest and annualised_volatility."""

import numpy as np
import pytest

from covariance_shrinkage.backtest import annualised_volatility, run_backtest
from covariance_shrinkage.estimators import linear_shrinkage, sample_cov
from covariance_shrinkage.portfolio import gmv_weights
from conftest import spread_covariance, simulate


def equal(W: np.ndarray) -> np.ndarray:
    """1/N, defined here so these tests don't depend on portfolio.py."""
    return np.full(W.shape[1], 1.0 / W.shape[1])


def inverse_variance(W: np.ndarray) -> np.ndarray:
    """Weights proportional to 1/variance over the window: a rule that uses the data."""
    inv = 1.0 / W.var(axis=0)
    return inv / inv.sum()


# --- annualised_volatility --------------------------------------------------------------


def test_annualised_volatility_known_answer():
    # Mean 0, squared deviations sum to 4e-4, n - 1 = 3: daily variance 4e-4 / 3,
    # annualised variance 252 * 4e-4 / 3 = 0.0336.
    returns = np.array([0.01, -0.01, 0.01, -0.01])
    np.testing.assert_allclose(annualised_volatility(returns), np.sqrt(0.0336))


def test_annualised_volatility_of_a_constant_series_is_zero():
    assert annualised_volatility(np.full(10, 0.001)) == pytest.approx(0.0, abs=1e-15)


def test_annualised_volatility_uses_periods_per_year():
    returns = np.array([0.01, -0.01, 0.01, -0.01])
    ratio = annualised_volatility(returns, periods_per_year=12) / annualised_volatility(returns)
    np.testing.assert_allclose(ratio, np.sqrt(12 / 252))


@pytest.mark.parametrize("returns", [np.array([0.01]), np.ones((3, 2)), np.array([0.01, np.nan, 0.0])])
def test_annualised_volatility_rejects_bad_input(returns):
    with pytest.raises(ValueError):
        annualised_volatility(returns)


# --- run_backtest: known answers --------------------------------------------------------


def test_buy_and_hold_hand_computed():
    # Two assets, window 2, holding 3: rebalances at rows 2 and 5. Rows 0-1 are only
    # estimation data. Equal weights, then no trading, so the weights drift.
    R = np.array(
        [
            [0.00, 0.00],
            [0.00, 0.00],
            [0.10, 0.00],  # values 0.55 + 0.50 = 1.05      -> +5%
            [0.10, 0.00],  # values 0.605 + 0.50 = 1.105    -> 0.055 / 1.05
            [0.00, 0.10],  # values 0.605 + 0.55 = 1.155    -> 0.05 / 1.105
            [0.00, -0.10],  # rebalance to (0.5, 0.5): 0.5 + 0.45 = 0.95 -> -5%
            [0.00, 0.00],  # unchanged                      -> 0
            [0.20, 0.00],  # values 0.6 + 0.45 = 1.05       -> 0.10 / 0.95
        ]
    )
    result = run_backtest(R, equal, window=2, holding=3)
    expected = [0.05, 0.055 / 1.05, 0.05 / 1.105, -0.05, 0.0, 0.10 / 0.95]
    np.testing.assert_allclose(result.returns, expected)
    # Fixed weights (daily rebalancing) would give 0.05 on each of the first three days.
    np.testing.assert_array_equal(result.rebalance_days, [2, 5])
    np.testing.assert_allclose(result.weights, [[0.5, 0.5], [0.5, 0.5]])


def test_buy_and_hold_with_a_short_position_hand_computed():
    # Weights (1.5, -0.5), window 2, holding 3: one rebalance, at row 2. The shorted asset
    # rises, so the short's value grows in magnitude and the NAV shrinks: the effective
    # weights drift from (1.5, -0.5) towards more leverage.
    R = np.array(
        [
            [0.00, 0.00],
            [0.00, 0.00],
            [0.00, 0.10],  # values 1.5 - 0.55  = 0.95  -> -5%
            [0.00, 0.10],  # values 1.5 - 0.605 = 0.895 -> -0.055 / 0.95
            [0.10, 0.00],  # values 1.65 - 0.605 = 1.045 -> 0.15 / 0.895
        ]
    )
    result = run_backtest(R, lambda W: np.array([1.5, -0.5]), window=2, holding=3)
    expected = [-0.05, -0.055 / 0.95, 0.15 / 0.895]
    np.testing.assert_allclose(result.returns, expected)
    # After row 3 the effective weights are (1.5, -0.605) / 0.895 = (1.676, -0.676): gross
    # exposure 2.35, up from 2. Fixed weights would give -0.05 on row 3, not -0.058.
    np.testing.assert_allclose(result.weights, [[1.5, -0.5]])


def test_rejects_a_bankrupt_portfolio():
    # Weights (3, -2) and the shorted asset up 60% in a day: values 3 - 3.2 = -0.2 < 0.
    R = np.array([[0.00, 0.00], [0.00, 0.00], [0.00, 0.60], [0.00, 0.00]])
    with pytest.raises(ValueError, match="<= 0"):
        run_backtest(R, lambda W: np.array([3.0, -2.0]), window=2, holding=2)


def test_output_shapes_and_rebalance_days(rng):
    R = 0.01 * rng.standard_normal((30, 4))
    result = run_backtest(R, equal, window=10, holding=7)
    np.testing.assert_array_equal(result.rebalance_days, [10, 17, 24])
    assert result.returns.shape == (20,)
    assert result.weights.shape == (3, 4)
    np.testing.assert_allclose(result.weights.sum(axis=1), 1.0)


def test_last_holding_period_ends_with_the_data(rng):
    R = 0.01 * rng.standard_normal((25, 3))
    result = run_backtest(R, equal, window=10, holding=7)
    np.testing.assert_array_equal(result.rebalance_days, [10, 17, 24])
    assert result.returns.shape == (15,)  # holding periods of 7, 7 and 1 days


def test_weight_rule_sees_exactly_the_estimation_window(rng):
    R = 0.01 * rng.standard_normal((40, 3))
    seen = []

    def spy(W):
        seen.append(W.copy())
        return equal(W)

    result = run_backtest(R, spy, window=10, holding=5)
    assert len(seen) == len(result.rebalance_days)
    for W, t in zip(seen, result.rebalance_days):
        np.testing.assert_array_equal(W, R[t - 10 : t])


# --- run_backtest: no lookahead ---------------------------------------------------------


def test_changing_later_returns_changes_nothing_earlier(rng):
    R = 0.01 * rng.standard_normal((300, 10))
    k = 150
    changed = R.copy()
    changed[k:] = 0.01 * rng.standard_normal((300 - k, 10))
    before = run_backtest(R, inverse_variance, window=60, holding=21)
    after = run_backtest(changed, inverse_variance, window=60, holding=21)

    early = before.rebalance_days <= k
    np.testing.assert_array_equal(after.weights[early], before.weights[early])
    np.testing.assert_array_equal(after.returns[: k - 60], before.returns[: k - 60])
    # ... and the change does matter later on, so the test has teeth.
    assert not np.allclose(after.weights[~early], before.weights[~early])


# --- run_backtest: missing data ---------------------------------------------------------


def test_asset_with_missing_data_is_left_out_of_that_rebalance_only(rng):
    # Rebalances at 10, 20, 30, 40, 50 (window 10, holding 10). A NaN in asset 2 on row 25
    # lies in the holding period of t = 20 and in the estimation window of t = 30.
    R = 0.01 * rng.standard_normal((60, 3))
    R[25, 2] = np.nan
    result = run_backtest(R, equal, window=10, holding=10)
    np.testing.assert_allclose(result.weights[:, 2], [1 / 3, 0, 0, 1 / 3, 1 / 3])
    np.testing.assert_allclose(result.weights[1], [0.5, 0.5, 0.0])
    assert np.isfinite(result.returns).all()


def test_rejects_an_empty_universe(rng):
    R = 0.01 * rng.standard_normal((40, 3))
    R[12, :] = np.nan
    with pytest.raises(ValueError):
        run_backtest(R, equal, window=10, holding=10)


# --- run_backtest: bad input ------------------------------------------------------------


@pytest.mark.parametrize(
    "window, holding",
    [(1, 5), (40, 5), (50, 5), (10, 0)],  # window < 2, window >= T_total (twice), holding < 1
)
def test_rejects_bad_window_or_holding(rng, window, holding):
    with pytest.raises(ValueError):
        run_backtest(0.01 * rng.standard_normal((40, 3)), equal, window=window, holding=holding)


@pytest.mark.parametrize(
    "rule",
    [
        lambda W: np.ones(W.shape[1]),  # doesn't sum to 1
        lambda W: np.full(W.shape[1] + 1, 1.0 / (W.shape[1] + 1)),  # wrong shape
        lambda W: np.full(W.shape[1], np.nan),  # NaN: every comparison with it is False
        lambda W: np.r_[np.nan, np.full(W.shape[1] - 1, 1.0 / (W.shape[1] - 1))],  # one NaN
        lambda W: np.r_[np.inf, -np.inf, np.ones(W.shape[1] - 2)],  # infinite
    ],
)
def test_rejects_a_bad_weight_rule(rng, rule):
    with pytest.raises(ValueError):
        run_backtest(0.01 * rng.standard_normal((40, 3)), rule, window=10, holding=5)


def test_rejects_one_dimensional_returns():
    with pytest.raises(ValueError):
        run_backtest(np.zeros(40), equal, window=10, holding=5)


# --- end to end: simulation with a known truth ------------------------------------------


def test_shrinkage_beats_the_sample_covariance_out_of_sample(rng):
    # N = 40, window 50 (N/T = 0.8), 20 monthly rebalances, data from a known Sigma.
    # At this ratio the sample covariance is very noisy, so GMV on linear shrinkage should
    # have clearly lower realised volatility (in this set-up about half: ratio 0.46 on
    # average over 30 seeds, at most 0.53).
    sigma = 1e-4 * spread_covariance(40, rng)
    R = simulate(sigma, 50 + 20 * 21, rng)
    sample = run_backtest(R, lambda W: gmv_weights(sample_cov(W)), window=50)
    shrunk = run_backtest(R, lambda W: gmv_weights(linear_shrinkage(W)[0]), window=50)
    assert annualised_volatility(shrunk.returns) < 0.9 * annualised_volatility(sample.returns)
