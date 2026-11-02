"""How much does the look-ahead in the universe filter change the out-of-sample SDs?

Run from the repo root:  poetry run python scripts/universe_filter_check.py

run_backtest drops a portfolio with a missing return in the estimation window or the holding
period (as Ledoit-Wolf, 2017, footnote 9). The holding-period half looks ahead. This script
counts the rebalances where that half matters, and reruns gmv_table.py's backtests with the
filter on the window only: such a portfolio is kept, and its value is held constant once its
data stops.

Writes results/universe_filter.md (the table as printed).
"""

from pathlib import Path

import numpy as np

from covariance_shrinkage.backtest import annualised_volatility
from covariance_shrinkage.data import load_french_100_daily, load_french_factors_daily
from covariance_shrinkage.estimators import linear_shrinkage, sample_cov
from covariance_shrinkage.portfolio import equal_weights, gmv_weights

START = "1973-01-01"
WINDOWS = (125, 250, 500)
HOLDING = 21
RESULTS = Path("results")

RULES = {
    "1/N": equal_weights,
    "sample": lambda W: gmv_weights(sample_cov(W)),
    "linear_shrinkage": lambda W: gmv_weights(linear_shrinkage(W)[0]),
}


def backtest(R: np.ndarray, rule, window: int, look_ahead: bool) -> tuple[np.ndarray, list[int], int]:
    """Daily portfolio returns, the rebalance days where the holding-period half of the filter
    removes a portfolio, and the largest number removed at one rebalance."""
    returns = []
    affected = []
    most_removed = 0
    for t in range(window, R.shape[0], HOLDING):
        end = min(R.shape[0], t + HOLDING)
        in_window = ~np.isnan(R[t - window : t]).any(axis=0)
        in_both = ~np.isnan(R[t - window : end]).any(axis=0)
        removed = int((in_window & ~in_both).sum())
        if removed:
            affected.append(t)
            most_removed = max(most_removed, removed)

        universe = in_both if look_ahead else in_window
        w = np.asarray(rule(R[t - window : t, universe]))
        # A return of 0 holds the value constant once the data stops.
        held = np.nan_to_num(R[t:end, universe], nan=0.0)
        values = (np.cumprod(1 + held, axis=0) * w).sum(axis=1)
        returns.append(values / np.concatenate([[1.0], values[:-1]]) - 1.0)
    return np.concatenate(returns), affected, most_removed


def main() -> None:
    df = load_french_100_daily()
    df = df[df.index >= START]
    rf = load_french_factors_daily()["RF"].reindex(df.index).to_numpy()
    R = df.to_numpy()
    oos_start = max(WINDOWS)

    lines = [
        "| T   | Estimator        | Affected | Most removed | SD, filter | SD, window only | Change |",
        "| --- | ---------------- | -------- | ------------ | ---------- | --------------- | ------ |",
    ]
    notes = []
    for T in WINDOWS:
        sub = R[oos_start - T :]
        for name, rule in RULES.items():
            filtered, affected, most_removed = backtest(sub, rule, T, look_ahead=True)
            kept, _, _ = backtest(sub, rule, T, look_ahead=False)
            sd = 100 * annualised_volatility(filtered - rf[oos_start:])
            sd_kept = 100 * annualised_volatility(kept - rf[oos_start:])
            lines.append(
                f"| {T:<3} | {name:<16} | {len(affected):>8} | {most_removed:>12} "
                f"| {sd:>10.3f} | {sd_kept:>15.3f} | {sd_kept - sd:>+6.3f} |"
            )
        days = df.index[oos_start - T :][affected]
        notes.append(f"T = {T}, affected rebalances: " + ", ".join(f"{d:%Y-%m-%d}" for d in days))

    n_rebalances = len(range(max(WINDOWS), R.shape[0], HOLDING))
    text = "\n".join(lines) + f"\n\n{n_rebalances} rebalances. " + "\n".join(notes) + "\n"
    print(text)
    RESULTS.mkdir(exist_ok=True)
    (RESULTS / "universe_filter.md").write_text(text)


if __name__ == "__main__":
    main()
