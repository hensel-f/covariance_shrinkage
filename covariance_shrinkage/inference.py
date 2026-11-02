"""Statistical inference on out-of-sample backtest results.

Conventions:
- Inputs are paired return series: entry t of both series is the same day. The tests use the
  pairing; they are not two-sample tests.
- Variances are in the units of the inputs (daily, for daily returns); annualising is the
  caller's job.
- Standard errors are Newey-West (Bartlett weights), so autocorrelation in the squared
  returns (volatility clustering) is allowed for.
"""

from dataclasses import dataclass

import numpy as np
from scipy.stats import norm


@dataclass(frozen=True)
class VarianceTestResult:
    diff: float
    """Variance of a minus variance of b, each normalised by T (not T - 1), in input units."""
    se: float
    """Newey-West standard error of diff."""
    t: float
    """diff / se."""
    p_value: float
    """Two-sided p-value for H0: equal variances, from the standard normal distribution."""
    lags: int
    """The Newey-West lag L actually used."""


def variance_difference_test(
    a: np.ndarray, b: np.ndarray, lags: int | None = None
) -> VarianceTestResult:
    """Test whether two paired return series have equal variances.

    The statistic is the difference of the two sample variances, each normalised by T and
    centred at its own sample mean. Its standard error is Newey-West with Bartlett weights
    1 - k / (lags + 1), k = 1, ..., lags, applied to sample autocovariances normalised by T.
    lags = 0 gives the i.i.d. standard error.

    Args:
        a, b: returns of the two strategies on the same T days, shape (T,) each.
        lags: the Newey-West lag L, an integer (int or NumPy integer) with 0 <= L < T. None
            means floor(0.75 * T ** (1/3)).

    Returns:
        A VarianceTestResult. diff < 0 means a has the lower variance.

    Raises:
        TypeError: if lags is not None and not an integer (a float such as 1.0, or a bool,
            included).
        ValueError: if a or b is not 1-D, their lengths differ, T < 2, either contains NaN or
            inf, lags is outside 0 <= lags < T, or the estimated long-run variance is not
            positive relative to the data's scale: sqrt(long-run variance) <= 1e-10 *
            (var(a) + var(b)), i.e. zero up to rounding (e.g. a and b equal, or equal up to a
            constant, so the difference is 0 on every day and t is 0 / 0).
    """
    # bool is a subclass of int, so True would otherwise pass as lags = 1.
    if lags is not None and (not isinstance(lags, (int, np.integer)) or isinstance(lags, bool)):
        raise TypeError(f"lags must be an integer or None, got {type(lags)}")
    if a.ndim != 1 or b.ndim != 1:
        raise ValueError(f"a and b must be 1-D, got {a.ndim} and {b.ndim}")
    if a.shape[0] != b.shape[0]:
        raise ValueError(f"a and b must have the same length, got {a.shape[0]} and {b.shape[0]}")
    if np.isnan(a).any() or np.isnan(b).any():
        raise ValueError("a and b must not contain NaN")
    if np.isinf(a).any() or np.isinf(b).any():
        raise ValueError("a and b must not contain inf")
    if a.shape[0] < 2 or b.shape[0] < 2:
        raise ValueError(f"length of a and b must be at least 2, got {a.shape[0]} and {b.shape[0]}")

    T = a.shape[0]
    if lags is None:
        lags = int(np.floor(0.75 * T ** (1 / 3)))
    if lags < 0 or lags >= T:
        raise ValueError(f"lags must be in 0 <= lags < T, got {lags} with T = {T}")

    mu_a = np.mean(a)
    mu_b = np.mean(b)
    # Difference of demeaned sample variances, normalised by T (not T - 1):
    d = (a - mu_a) ** 2 - (b - mu_b) ** 2
    mu_d = np.mean(d)
    variance_d = np.var(d - mu_d, ddof=0)  # demeaned population variance of d, normalised by T
    # Autocovariances of d, normalised by T, for lags 1, ..., L.
    gamma = 1/T * np.array([(d[k:]- mu_d) @ (np.roll(d, k)[k:] - mu_d) for k in range(1, lags + 1)])
    # Newey-West long-run variance estimate, Bartlett weights 1 - k / (lags + 1):
    w_NW = np.array([1 - k / (lags + 1) for k in range(1, lags + 1)])
    # The long-run variance is the variance plus twice the sum of the weighted autocovariances.
    omega_NW = variance_d + 2 * w_NW @ gamma
    scale = np.var(a) + np.var(b)

    # The long-run variance is an estimate of the variance of the difference of sample variances,
    # which is always non-negative. If it is not positive, the difference of sample variances is
    # zero up to rounding noise, e.g. if a and b are equal or equal up to a constant. In that case,
    # the t-statistic is 0 / 0, which is undefined, so raise an error.
    if omega_NW <= (1e-10 * scale) ** 2:
        raise ValueError(
            f"Estimated long-run variance is not positive (relative to the data's scale), got "
            f"{omega_NW}. This can happen if a and b have identical demeaned returns, e.g. "
            "b = a or b = a + constant, so the difference is 0 on every day."
        )
    # Standard error of the difference of variances, normalised by T (not T - 1):
    se = float(np.sqrt(omega_NW / T))
    # The t-statistic is the difference divided by its standard error,
    # and the p-value is two-sided from the standard normal distribution.
    t_stat = float(mu_d / se)
    # Two-sided p-value from the standard normal distribution,
    # using the survival function norm.sf = (1 - cdf) for numerical stability in the tails:
    p = float(2 * norm.sf(abs(t_stat)))

    return VarianceTestResult(diff=float(mu_d), se=se, t=t_stat, p_value=p, lags=int(lags))
