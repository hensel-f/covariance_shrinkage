"""Tests for covariance_shrinkage.portfolio: equal_weights and gmv_weights."""

import numpy as np
import pytest

from covariance_shrinkage.portfolio import equal_weights, gmv_weights
from conftest import spread_covariance


def test_equal_weights(rng):
    w = equal_weights(rng.standard_normal((30, 4)))
    assert w.shape == (4,)
    np.testing.assert_allclose(w, 0.25)


# --- gmv_weights: known answers ---------------------------------------------------------


def test_gmv_shape_and_full_investment(rng):
    w = gmv_weights(spread_covariance(6, rng))
    assert w.shape == (6,)
    np.testing.assert_allclose(w.sum(), 1.0)


@pytest.mark.parametrize("scale", [1.0, 0.0004, 25.0])
def test_gmv_of_a_multiple_of_the_identity_is_equal_weights(scale):
    np.testing.assert_allclose(gmv_weights(scale * np.eye(5)), 0.2)


def test_gmv_of_uncorrelated_assets_is_inverse_variance():
    # Weights proportional to 1/variance: (1, 1/2, 1/4) / 1.75.
    w = gmv_weights(np.diag([1.0, 2.0, 4.0]))
    np.testing.assert_allclose(w, np.array([1.0, 0.5, 0.25]) / 1.75)


def test_gmv_two_assets_closed_form():
    # w_1 = (s2^2 - s12) / (s1^2 + s2^2 - 2 s12) = (0.09 - 0.006) / (0.04 + 0.09 - 0.012).
    sigma = np.array([[0.04, 0.006], [0.006, 0.09]])
    w1 = 0.084 / 0.118
    np.testing.assert_allclose(gmv_weights(sigma), [w1, 1 - w1])


def test_gmv_allows_short_positions():
    # Volatilities 1 and 2, correlation 0.9: hedge the riskier asset by shorting it.
    # w_1 = (4 - 1.8) / (1 + 4 - 3.6) = 2.2 / 1.4 > 1, so w_2 < 0.
    w = gmv_weights(np.array([[1.0, 1.8], [1.8, 4.0]]))
    np.testing.assert_allclose(w, [2.2 / 1.4, 1 - 2.2 / 1.4])
    assert w[1] < 0


# --- gmv_weights: the defining property and invariances ---------------------------------


def test_gmv_has_lower_variance_than_any_other_fully_invested_portfolio(rng):
    sigma = spread_covariance(8, rng)
    w = gmv_weights(sigma)
    for _ in range(200):
        v = rng.standard_normal(8)
        other = w + (v - v.mean())  # still sums to 1
        assert w @ sigma @ w <= other @ sigma @ other + 1e-12


def test_gmv_ignores_the_scale_of_sigma(rng):
    sigma = spread_covariance(6, rng)
    np.testing.assert_allclose(gmv_weights(1e-4 * sigma), gmv_weights(sigma))


def test_gmv_permuting_assets_permutes_the_weights(rng):
    sigma = spread_covariance(6, rng)
    p = rng.permutation(6)
    np.testing.assert_allclose(gmv_weights(sigma[np.ix_(p, p)]), gmv_weights(sigma)[p])


# --- gmv_weights: bad input -------------------------------------------------------------


def test_gmv_rejects_a_singular_sample_covariance(rng):
    # N = 20 > T = 10: rank-deficient, so no minimum-variance solution exists.
    R = rng.standard_normal((10, 20))
    S = np.cov(R, rowvar=False, bias=True)
    with pytest.raises(ValueError):
        gmv_weights(S)


@pytest.mark.parametrize(
    "sigma",
    [
        np.ones((2, 3)),  # not square
        np.ones(3),  # not a matrix
        np.zeros((0, 0)),  # no assets
        np.array([[1.0, 0.5], [0.0, 1.0]]),  # not symmetric
        np.array([[1.0, np.nan], [np.nan, 1.0]]),
        np.array([[1.0, 2.0], [2.0, 1.0]]),  # symmetric but indefinite (eigenvalues 3, -1)
    ],
)
def test_gmv_rejects_bad_input(sigma):
    with pytest.raises(ValueError):
        gmv_weights(sigma)


@pytest.mark.parametrize("scale", [1.0, 1e-4, 1e-9])
def test_gmv_rejects_a_small_asymmetry_at_any_scale(scale):
    # One off-diagonal entry off by 1e-6 relative: far above rounding error, so a real
    # mistake. Cholesky reads one triangle only, so without the check it would be ignored.
    sigma = scale * np.array([[1.0, 0.5], [0.5 + 1e-6, 1.0]])
    with pytest.raises(ValueError, match="symmetric"):
        gmv_weights(sigma)


@pytest.mark.parametrize("scale", [1.0, 1e-4, 1e-9])
def test_gmv_accepts_rounding_asymmetry(rng, scale):
    # Asymmetry at the level of floating-point rounding (1e-14 relative) is not an error.
    sigma = scale * spread_covariance(6, rng)
    noise = 1e-14 * np.abs(sigma).max() * np.triu(rng.standard_normal((6, 6)), k=1)
    w = gmv_weights(sigma + noise)
    np.testing.assert_allclose(w, gmv_weights(sigma), atol=1e-10)
