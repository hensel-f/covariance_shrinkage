"""Shared pytest setup: a seeded random generator, so simulated data are reproducible."""

import numpy as np
import pytest


@pytest.fixture
def rng() -> np.random.Generator:
    return np.random.default_rng(0)


def spread_covariance(n: int, rng: np.random.Generator) -> np.ndarray:
    """A covariance with eigenvalues spread evenly over [0.5, 1.5] and random eigenvectors."""
    q, _ = np.linalg.qr(rng.standard_normal((n, n)))
    return q @ np.diag(np.linspace(0.5, 1.5, n)) @ q.T


def simulate(sigma: np.ndarray, t: int, rng: np.random.Generator) -> np.ndarray:
    """t iid Gaussian draws with mean zero and covariance sigma, shape (t, n)."""
    return rng.multivariate_normal(np.zeros(len(sigma)), sigma, size=t)
