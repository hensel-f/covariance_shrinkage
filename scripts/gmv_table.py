"""Out-of-sample GMV performance: estimator x window length, on the French 100 portfolios.

Run from the repo root:  poetry run python scripts/gmv_table.py [--lags L]

Every window length is evaluated on the same out-of-sample days: the first holding period
starts on the first trading day of START plus the longest window, and each shorter window
takes its estimation data from just before that day. The rebalance days are therefore the
same for every T, and the return series are paired across estimators and window lengths.

Measures, as Ledoit-Wolf (2017, p. 4368), on daily returns in excess of the risk-free rate
(French's daily RF), annualised with 252 days and in percent:
- AV: average excess return times 252.
- SD: standard deviation of excess returns times sqrt(252). The primary measure: GMV
  minimises variance, so it is the one that is tested.
- SR: AV / SD. AV and SR are reported, not tested.

The t-statistics compare each estimator's SD with BASELINE's (variance_difference_test,
Newey-West): t > 0 means a higher out-of-sample variance than the baseline. --lags sets the
Newey-West lag L; without it, the test's default floor(0.75 n^(1/3)) on the n out-of-sample days.
The t-statistic measures evidence, not size: the effect size is the variance ratio (the
estimator's out-of-sample variance over the baseline's), reported next to it.

Writes results/gmv_table.csv (full precision, one row per window length and estimator) and
results/gmv_table.md (the table as printed); with --lags L, gmv_table_L<L>.csv and .md.
"""

import argparse
import sys
import time
from pathlib import Path

import numpy as np
import pandas as pd

from covariance_shrinkage.backtest import annualised_volatility, run_backtest
from covariance_shrinkage.data import load_french_100_daily, load_french_factors_daily
from covariance_shrinkage.estimators import linear_shrinkage, sample_cov
from covariance_shrinkage.inference import variance_difference_test
from covariance_shrinkage.portfolio import equal_weights, gmv_weights

START = "1973-01-01"
WINDOWS = (125, 250, 500)
HOLDING = 21
PERIODS_PER_YEAR = 252
RESULTS = Path("results")

RULES = {
    "1/N": equal_weights,
    "sample": lambda W: gmv_weights(sample_cov(W)),
    "linear_shrinkage": lambda W: gmv_weights(linear_shrinkage(W)[0]),
}
# The sample covariance while the question is whether shrinkage helps at all. Switch to
# "linear_shrinkage" once nonlinear shrinkage is added: the question then is whether it beats
# linear shrinkage.
BASELINE = "sample"


def markdown_table(rows: list[list[str]]) -> str:
    """A markdown table with every column padded to its widest cell, so it also reads in a
    terminal. rows[0] is the header."""
    widths = [max(len(row[j]) for row in rows) for j in range(len(rows[0]))]
    lines = ["| " + " | ".join(c.ljust(w) for c, w in zip(row, widths)) + " |" for row in rows]
    lines.insert(1, "| " + " | ".join("-" * w for w in widths) + " |")
    return "\n".join(lines)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    parser.add_argument(
        "--lags", type=int, default=None, metavar="L",
        help="Newey-West lag of the variance test (default: floor(0.75 n^(1/3)))",
    )
    lags = parser.parse_args().lags
    if lags is not None and lags < 0:
        parser.error("--lags must be >= 0")

    df = load_french_100_daily()
    df = df[df.index >= START]
    rf = load_french_factors_daily()["RF"].reindex(df.index)
    if rf.isna().any():
        raise ValueError(f"no risk-free rate on {rf.index[rf.isna()][:5].tolist()} ...")
    R = df.to_numpy()
    oos_start = max(WINDOWS)
    oos_days = df.index[oos_start:]
    rf_oos = rf.to_numpy()[oos_start:]
    period = (
        f"Data {df.index[0]:%Y-%m-%d} to {df.index[-1]:%Y-%m-%d}, N = {R.shape[1]}; "
        f"out of sample {oos_days[0]:%Y-%m-%d} to {oos_days[-1]:%Y-%m-%d} "
        f"({len(oos_days)} days), rebalanced every {HOLDING} days."
    )
    print(period, file=sys.stderr)

    # Daily out-of-sample returns in excess of the risk-free rate, by (estimator, T).
    excess: dict[tuple[str, int], np.ndarray] = {}
    for T in WINDOWS:
        for name, rule in RULES.items():
            tic = time.perf_counter()
            try:
                res = run_backtest(R[oos_start - T :], rule, window=T, holding=HOLDING)
            except ValueError as err:
                print(f"{name}, T = {T}: backtest failed: {err}", file=sys.stderr)
                continue
            excess[(name, T)] = res.returns - rf_oos
            print(
                f"  {name:>16}, T = {T}: {len(res.rebalance_days)} rebalances, "
                f"{time.perf_counter() - tic:.1f} s",
                file=sys.stderr,
            )

    lengths = {len(r) for r in excess.values()}
    assert lengths == {len(oos_days)}, f"return series of lengths {lengths}, not {len(oos_days)}"

    records = []
    for T in WINDOWS:
        base = excess.get((BASELINE, T))
        for name in RULES:
            r = excess.get((name, T))
            rec = {"T": T, "N_over_T": R.shape[1] / T, "estimator": name}
            if r is not None:
                av = 100 * PERIODS_PER_YEAR * r.mean()
                sd = 100 * annualised_volatility(r, PERIODS_PER_YEAR)
                rec |= {"AV": av, "SD": sd, "SR": av / sd}
                if name != BASELINE and base is not None:
                    test = variance_difference_test(r, base, lags=lags)
                    rec |= {
                        "var_ratio": r.var() / base.var(),
                        "t_SD": test.t,
                        "p_SD": test.p_value,
                        "lags": test.lags,
                    }
            records.append(rec)
    table = pd.DataFrame(records)
    if "lags" in table:
        table["lags"] = table["lags"].astype("Int64")  # an integer column, empty for the baseline
    table["baseline"] = BASELINE

    def fmt(x, spec: str) -> str:
        return "" if pd.isna(x) else format(x, spec)

    used = table["lags"].dropna().unique() if "lags" in table else []
    lag_label = f"L = {int(used[0])}" if len(used) == 1 else "L"
    rows = [[
        "T", "N/T", "Estimator", "AV (%)", "SD (%)", "SR",
        f"Var / var({BASELINE})", f"t (SD vs {BASELINE}, {lag_label})",
    ]]
    for rec in table.to_dict("records"):
        failed = pd.isna(rec.get("SD"))
        rows.append([
            str(rec["T"]),
            f"{rec['N_over_T']:.1f}",
            rec["estimator"],
            "failed" if failed else fmt(rec["AV"], ".2f"),
            fmt(rec.get("SD"), ".2f"),
            fmt(rec.get("SR"), ".2f"),
            fmt(rec.get("var_ratio"), ".3f"),
            fmt(rec.get("t_SD"), "+.2f"),
        ])
    note = (
        f"{period}\n"
        f"Daily returns in excess of the risk-free rate, annualised with {PERIODS_PER_YEAR} days. "
        f"AV: mean x {PERIODS_PER_YEAR}; SD: sd x sqrt({PERIODS_PER_YEAR}); SR: AV / SD. "
        f"Var / var({BASELINE}): out-of-sample variance relative to {BASELINE} (effect size). "
        f"t: paired variance-difference test against {BASELINE}, Newey-West with lag {lag_label}; "
        f"t < 0 means lower out-of-sample variance than {BASELINE}."
    )
    markdown = markdown_table(rows) + "\n\n" + note + "\n"

    stem = "gmv_table" if lags is None else f"gmv_table_L{lags}"
    RESULTS.mkdir(exist_ok=True)
    table.to_csv(RESULTS / f"{stem}.csv", index=False)
    (RESULTS / f"{stem}.md").write_text(markdown)
    print("\n" + markdown)
    print(f"Written to {RESULTS / stem}.csv and .md.", file=sys.stderr)


if __name__ == "__main__":
    main()
