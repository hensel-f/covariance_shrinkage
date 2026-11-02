"""Covariance estimators.

Each estimator takes a window of returns ``R`` of shape (T, N): T days as rows, N assets as
columns, simple returns in decimals, all finite. It returns an N x N covariance matrix.

Conventions: demean each column, normalise by 1/T.
"""

import numpy as np


def sample_cov(R: np.ndarray) -> np.ndarray:
    """The sample covariance matrix S of a window of returns.

    Demeaned and normalised by 1/T (not 1/(T - 1)).

    Args:
        R: array of shape (T, N), T >= 2, N >= 1, all entries finite.

    Returns:
        Array of shape (N, N), symmetric positive semi-definite. Its rank is at most
        T - 1, so it is singular whenever N >= T.

    Raises:
        ValueError: if R is not 2-D, has T < 2 or N < 1, or contains NaN or inf.
    """
    R = np.asarray(R, dtype=float)
    if R.ndim != 2:
        raise ValueError(f"R must be 2-D, got {R.ndim}-D")
    T, N = R.shape
    if T < 2 or N < 1 or not np.isfinite(R).all():
        raise ValueError(
            f"R must have shape (T, N) with T >= 2, N >= 1 and all entries finite, "
            f"got shape {R.shape}"
        )

    R = R - np.mean(R, axis=0)
    return (R.transpose() @ R) / T


def linear_shrinkage(R: np.ndarray) -> tuple[np.ndarray, float]:
    """Ledoit-Wolf (2004) linear shrinkage of the sample covariance towards a scaled identity.

    S* = delta * m * I + (1 - delta) * S, where S = sample_cov(R), m * I is the target
    (m the average sample variance) and delta in [0, 1] is the estimated optimal shrinkage
    intensity of Ledoit & Wolf (2004, J. Multivariate Analysis 88), eq. (14) with
    Lemmas 3.2-3.5 (p. 380). Using the paper's norm, ||A||^2 = tr(A A^T) / N, and S as
    defined in ``sample_cov``. The result must match scikit-learn's ``LedoitWolf``
    (``covariance_`` and ``shrinkage_``).

    Args:
        R: array of shape (T, N), T >= 2, N >= 1, all entries finite. N > T is allowed.

    Returns:
        (S_star, delta): S_star of shape (N, N), symmetric, positive definite whenever
        delta > 0 and the average variance m is positive; delta a float in [0, 1].
        If S already equals m * I (for instance N = 1), there is nothing to shrink:
        return (S, 0.0).

    Raises:
        ValueError: on the same inputs as ``sample_cov``.
    """
    R = np.asarray(R, dtype=float)
    S = sample_cov(R)
    T, N = R.shape
    I_N = np.identity(N)

    m = np.trace(S) / N
    # The paper's norm tr(A A^T)/N as a sum of squares:
    d = np.sum((S - m * I_N) * (S - m * I_N)) / N

    R_demeaned = R - np.mean(R, axis=0)
    b_bar = 0.0
    for k in range(T):
        row = R_demeaned[k, :]
        O = np.outer(row, row) - S
        b_bar += np.sum(O * O) / N
    b_bar = b_bar / T**2
    b = min(b_bar, d)

    if d != 0:
        delta = b / d
    else:
        delta = 0.0

    S_star = delta * m * I_N + (1 - delta) * S
    return S_star, delta
