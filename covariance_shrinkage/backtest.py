"""Rolling out-of-sample backtest of a portfolio weight rule.

Conventions:
- t denotes a rebalance day: the first day the new portfolio is held. The estimation window
  is rows t - window, ..., t - 1 of R, so day t is never used to form the weights at t.
- The holding period is rows t, ..., t + holding - 1, buy-and-hold: no trading between
  rebalances, so the weights drift with prices.
- Universe filter, as Ledoit-Wolf (2017): an asset with any NaN in the estimation window or
  the holding period is left out at that rebalance (weight 0). This looks ahead, deliberately;
  it is the same for every weight rule.
"""

from collections.abc import Callable
from dataclasses import dataclass

import numpy as np

WeightRule = Callable[[np.ndarray], np.ndarray]
"""Maps a window of returns, shape (window, n), to weights, shape (n,), summing to 1."""


@dataclass(frozen=True)
class BacktestResult:
    returns: np.ndarray
    """Daily out-of-sample portfolio returns, shape (T_total - window,): row i is day window + i."""
    rebalance_days: np.ndarray
    """Rebalance days t (row indices of R), shape (K,), increasing."""
    weights: np.ndarray
    """Weights set at each rebalance, shape (K, N); 0 for assets left out at that rebalance."""


def run_backtest(
    R: np.ndarray, weight_rule: WeightRule, window: int, holding: int = 21
) -> BacktestResult:
    """Run the rolling out-of-sample backtest.

    Rebalance days are t = window, window + holding, window + 2 * holding, ... as long as
    t < T_total. The last holding period ends with the data, so it can be shorter.

    At each rebalance day t:
    1. The universe is every asset with no NaN in rows t - window, ..., end - 1, where
       end = min(t + holding, T_total).
    2. The weights are weight_rule(R[t - window : t, universe]); assets outside the
       universe get weight 0.
    3. Buy-and-hold over rows t, ..., end - 1: start with total value 1 split according to
       the weights; each position's value is multiplied by (1 + its return) every day, with
       no trading. The portfolio's return on a day is its total value at the end of the
       day divided by its total value at the end of the previous day, minus 1.

    Args:
        R: array of shape (T_total, N), simple returns in decimals, days as rows; may
            contain NaN (handled by the universe filter).
        weight_rule: see ``WeightRule``.
        window: estimation window length T, >= 2.
        holding: rebalance every ``holding`` days, >= 1 (21 = monthly).

    Returns:
        A BacktestResult.

    Raises:
        ValueError: if R is not 2-D, window < 2, holding < 1, window >= T_total, the
            universe is empty at some rebalance, or weight_rule returns weights of the
            wrong shape, not finite, or not summing to 1 (tolerance 1e-8), or the portfolio
            value falls to 0 or below within a holding period (bankrupt: later returns would
            be meaningless).
    """
    if R.ndim != 2:
        raise ValueError("R must be 2-D")
    if window < 2:
        raise ValueError("window must be >= 2")
    if holding < 1:
        raise ValueError("holding must be >= 1")
    if window >= R.shape[0]:
        raise ValueError("window must be < number of rows in R")

    T, N = R.shape
    rebalance_days = []
    weights_list = []
    returns_list = []

    for t in range(window, T, holding):
        end = min(T, t + holding)
        universe = ~np.isnan(R[t - window : end]).any(axis=0)

        if not universe.any():
            raise ValueError(f"empty universe at rebalance day {t}")

        # Rebalance: compute weights for the universe, then put them into a
        # full-length array with 0s for the rest.
        w = np.asarray(weight_rule(R[t - window : t, universe]))

        if w.shape != (universe.sum(),):
            raise ValueError(
                f"weight_rule returned shape {w.shape}, expected ({universe.sum()},)"
            )
        if not np.isfinite(w).all():
            raise ValueError("weight_rule returned non-finite weights")
        if abs(w.sum() - 1.0) > 1e-8:
            raise ValueError(f"weights sum to {w.sum()}, not 1")

        weights = np.zeros(N)
        weights[universe] = w
        rebalance_days.append(t)
        weights_list.append(weights)

        # Buy-and-hold over the holding period, starting with total
        # value 1 split according to the weights.
        values = np.cumprod(1 + R[t:end, universe], axis=0) * w
        curr_vals = values.sum(axis=1)
        if (curr_vals <= 0).any():
            first = np.argmax(curr_vals <= 0)
            raise ValueError(
                f"portfolio value {curr_vals[first]:.4g} <= 0 on day {t + first} "
                f"(holding period from rebalance day {t})"
            )
        prev_vals = np.concatenate([[1.0], curr_vals[:-1]])
        curr_returns = curr_vals / prev_vals - 1.0
        returns_list.append(curr_returns)

    return BacktestResult(
        returns=np.concatenate(returns_list),
        rebalance_days=np.array(rebalance_days),
        weights=np.array(weights_list),
    )


def annualised_volatility(returns: np.ndarray, periods_per_year: int = 252) -> float:
    """Annualised volatility of a series of daily returns.

    The sample standard deviation (normalised by n - 1) times sqrt(periods_per_year).

    Raises:
        ValueError: if returns is not 1-D, has fewer than 2 entries, or contains NaN.
    """
    if returns.ndim != 1:
        raise ValueError("returns must be 1-D")
    if len(returns) < 2:
        raise ValueError("returns must have at least 2 entries")
    if np.isnan(returns).any():
        raise ValueError("returns must not contain NaN")
    return returns.std(ddof=1) * np.sqrt(periods_per_year)
