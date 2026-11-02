"""Portfolio weight rules.

Weights are 1-D arrays of shape (N,) that sum to 1 (fully invested); short positions
(negative weights) are allowed. With returns R of shape (T, N), ``R @ w`` is the vector of
daily portfolio returns at fixed weights.
"""

import numpy as np
from scipy import linalg

# Relative tolerance for the symmetry check in gmv_weights. Rounding in X.T @ X / T leaves
# an asymmetry of order 1e-16 relative to the largest entry; a real mistake (a wrong index,
# an estimator returning a non-symmetric matrix) is many orders of magnitude larger.
SYMMETRY_TOL = 1e-10


def equal_weights(R: np.ndarray) -> np.ndarray:
    """The 1/N portfolio.

    Takes the window of returns, shape (T, N), only so that it has the same signature as
    the other weight rules used by the backtest; the values are ignored.

    Returns:
        Array of shape (N,), every entry 1/N.
    """
    return np.ones(R.shape[1]) / R.shape[1]


def gmv_weights(Sigma: np.ndarray) -> np.ndarray:
    """Global minimum-variance weights for a covariance matrix.

    The solution of: minimise w^T Sigma w subject to 1^T w = 1, no other constraints.

    Implementation note:
    Instead of computing the inverse of Sigma, the weights are computed by solving a linear
    system via the Cholesky factorisation (which doubles as an assertion of positive
    definiteness). Normalization is the last step to ensure the weights sum to 1.

    Args:
        Sigma: array of shape (N, N), symmetric positive definite.

    Returns:
        Array of shape (N,), summing to 1.

    Raises:
        ValueError: if Sigma is not square, not symmetric (up to floating-point tolerance),
            contains NaN or inf, or is not positive definite (e.g. a sample covariance with
            N >= T). No pseudo-inverse fallback: that would silently change the estimator.
    """

    Sigma = np.asarray(Sigma, dtype=float)

    if not Sigma.ndim == 2 or Sigma.shape[0] != Sigma.shape[1] or Sigma.shape[0] == 0:
        raise ValueError(
            f"Sigma must be a square matrix, it is of shape: {Sigma.shape}."
        )

    if not np.isfinite(Sigma).all():
        raise ValueError("Sigma contains NaN or inf.")

    # Checking that Sigma is symmetric, relative to its own scale:
    scale = np.max(np.abs(Sigma))
    asym = np.max(np.abs(Sigma - Sigma.T))
    if asym > SYMMETRY_TOL * scale:
        raise ValueError("Sigma is not symmetric.")

    try:
        L = linalg.cholesky(Sigma, lower=True)
    except linalg.LinAlgError as err:
        raise ValueError("Sigma is not positive definite (e.g. N >= T).") from err

    w = linalg.cho_solve((L, True), np.ones(Sigma.shape[0]))
    w = w / np.sum(w)

    return w
