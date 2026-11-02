"""Tests for covariance_shrinkage.estimators: sample_cov and linear_shrinkage."""

import numpy as np
import pytest
from sklearn.covariance import LedoitWolf

from covariance_shrinkage.estimators import linear_shrinkage, sample_cov
from conftest import simulate, spread_covariance


# --- sample_cov: known answers ----------------------------------------------------------


def test_sample_cov_hand_computed():
    # Column means (3, 3); deviations [[-1, -2], [1, 2]]; divided by T = 2.
    # 1/(T-1) would give [[2, 4], [4, 8]]; no demeaning would give [[10, 11], [11, 13]].
    R = np.array([[2.0, 1.0], [4.0, 5.0]])
    np.testing.assert_allclose(sample_cov(R), [[1.0, 2.0], [2.0, 4.0]])


def test_sample_cov_single_asset_is_its_variance():
    R = np.array([[1.0], [2.0], [3.0], [6.0]])  # mean 3, squared deviations 4, 1, 0, 9
    np.testing.assert_allclose(sample_cov(R), [[14.0 / 4]])


def test_sample_cov_shape_and_symmetry(rng):
    S = sample_cov(rng.standard_normal((50, 7)))
    assert S.shape == (7, 7)
    np.testing.assert_allclose(S, S.T)


# --- sample_cov: invariances ------------------------------------------------------------


def test_sample_cov_ignores_a_constant_added_to_each_asset(rng):
    R = rng.standard_normal((40, 5))
    shift = rng.standard_normal(5)
    np.testing.assert_allclose(sample_cov(R + shift), sample_cov(R), atol=1e-12)


def test_sample_cov_scales_quadratically(rng):
    R = rng.standard_normal((40, 5))
    np.testing.assert_allclose(sample_cov(3.0 * R), 9.0 * sample_cov(R))


def test_sample_cov_permuting_assets_permutes_the_matrix(rng):
    R = rng.standard_normal((40, 5))
    p = rng.permutation(5)
    np.testing.assert_allclose(sample_cov(R[:, p]), sample_cov(R)[np.ix_(p, p)])


# --- sample_cov: simulation with a known truth ------------------------------------------


def test_sample_cov_is_singular_when_n_exceeds_t(rng):
    # Demeaning costs one degree of freedom: rank <= T - 1 = 9 < N = 20.
    S = sample_cov(rng.standard_normal((10, 20)))
    assert np.linalg.matrix_rank(S) <= 9


def test_sample_cov_error_shrinks_with_sample_size(rng):
    sigma = spread_covariance(10, rng)
    small = np.linalg.norm(sample_cov(simulate(sigma, 100, rng)) - sigma)
    large = np.linalg.norm(sample_cov(simulate(sigma, 20_000, rng)) - sigma)
    # Frobenius error ~ 1/sqrt(T): 200 times more data, about 14 times smaller.
    assert large < small / 4
    assert large < 0.1


# --- sample_cov: bad input --------------------------------------------------------------


@pytest.mark.parametrize(
    "R",
    [
        np.ones(5),  # 1-D
        np.ones((1, 3)),  # T = 1
        np.ones((5, 0)),  # N = 0
        np.array([[1.0, np.nan], [2.0, 3.0], [0.0, 1.0]]),
        np.array([[1.0, np.inf], [2.0, 3.0], [0.0, 1.0]]),
    ],
)
def test_sample_cov_rejects_bad_input(R):
    with pytest.raises(ValueError):
        sample_cov(R)


# --- linear_shrinkage: the reference implementation -------------------------------------


@pytest.mark.parametrize("t, n", [(250, 100), (125, 100), (60, 200), (30, 5)])
def test_linear_shrinkage_matches_scikit_learn(rng, t, n):
    R = 0.01 * rng.standard_t(df=5, size=(t, n)) + 0.0005
    S_star, delta = linear_shrinkage(R)
    reference = LedoitWolf().fit(R)
    np.testing.assert_allclose(delta, reference.shrinkage_, rtol=1e-10)
    np.testing.assert_allclose(S_star, reference.covariance_, rtol=1e-10, atol=1e-16)


# --- linear_shrinkage: properties -------------------------------------------------------


def test_linear_shrinkage_intensity_is_between_zero_and_one(rng):
    for t, n in [(20, 50), (100, 10), (1000, 5)]:
        _, delta = linear_shrinkage(rng.standard_normal((t, n)))
        assert 0.0 <= delta <= 1.0


def test_linear_shrinkage_is_positive_definite_when_n_exceeds_t(rng):
    S_star, _ = linear_shrinkage(rng.standard_normal((30, 80)))
    assert np.linalg.eigvalsh(S_star).min() > 0


def test_linear_shrinkage_keeps_the_trace(rng):
    # The target m * I has the same trace as S, so any convex combination does too.
    R = rng.standard_normal((40, 15))
    S_star, _ = linear_shrinkage(R)
    np.testing.assert_allclose(np.trace(S_star), np.trace(sample_cov(R)))


def test_linear_shrinkage_keeps_eigenvectors_and_pulls_eigenvalues_in(rng):
    R = rng.standard_normal((40, 15))
    S = sample_cov(R)
    S_star, _ = linear_shrinkage(R)
    np.testing.assert_allclose(S_star @ S, S @ S_star, atol=1e-12)  # same eigenvectors
    sample_eigs, shrunk_eigs = np.linalg.eigvalsh(S), np.linalg.eigvalsh(S_star)
    assert shrunk_eigs.min() >= sample_eigs.min()
    assert shrunk_eigs.max() <= sample_eigs.max()


def test_linear_shrinkage_is_rotation_equivariant(rng):
    # Rotating the assets rotates the estimate and leaves the intensity alone.
    R = rng.standard_normal((40, 8))
    q, _ = np.linalg.qr(rng.standard_normal((8, 8)))
    S_star, delta = linear_shrinkage(R)
    S_rot, delta_rot = linear_shrinkage(R @ q)
    np.testing.assert_allclose(delta_rot, delta)
    np.testing.assert_allclose(S_rot, q.T @ S_star @ q, atol=1e-12)


def test_linear_shrinkage_scales_quadratically_and_keeps_the_intensity(rng):
    R = rng.standard_normal((40, 8))
    S_star, delta = linear_shrinkage(R)
    S_scaled, delta_scaled = linear_shrinkage(0.01 * R)
    np.testing.assert_allclose(delta_scaled, delta)
    np.testing.assert_allclose(S_scaled, 1e-4 * S_star)


def test_linear_shrinkage_ignores_a_constant_added_to_each_asset(rng):
    R = rng.standard_normal((40, 8))
    shift = rng.standard_normal(8)
    S_star, delta = linear_shrinkage(R)
    S_shifted, delta_shifted = linear_shrinkage(R + shift)
    np.testing.assert_allclose(delta_shifted, delta, atol=1e-12)
    np.testing.assert_allclose(S_shifted, S_star, atol=1e-12)


def test_linear_shrinkage_single_asset_is_not_shrunk():
    R = np.array([[1.0], [2.0], [3.0], [6.0]])
    S_star, delta = linear_shrinkage(R)
    assert delta == 0.0
    np.testing.assert_allclose(S_star, sample_cov(R))


# --- linear_shrinkage: simulation with a known truth ------------------------------------


def test_linear_shrinkage_shrinks_almost_fully_when_the_truth_is_the_target(rng):
    # Sigma = 2 I, so alpha^2 = 0 and the optimal intensity beta^2 / (alpha^2 + beta^2) is 1.
    _, delta = linear_shrinkage(simulate(2.0 * np.eye(50), 100, rng))
    assert delta > 0.9


def test_linear_shrinkage_intensity_falls_with_more_data(rng):
    # Eigenvalues uniform on [0.5, 1.5]: alpha^2 ~ 1/12, beta^2 ~ (N/T) mu^2 with mu = 1, so
    # the optimal intensity is about 0.86 at N/T = 0.5 and about 0.23 at N/T = 0.025.
    sigma = spread_covariance(50, rng)
    _, delta_short = linear_shrinkage(simulate(sigma, 100, rng))
    _, delta_long = linear_shrinkage(simulate(sigma, 2000, rng))
    assert delta_short > 0.7
    assert delta_long < 0.4


def test_linear_shrinkage_beats_the_sample_covariance_on_average(rng):
    # Ledoit-Wolf 2004, Theorem 3.4: lower expected Frobenius loss than S (a linear
    # combination of I and S with intensity 0). N/T = 0.5, 20 draws; in this set-up the
    # mean loss ratio is about 0.15, so 0.5 leaves a wide margin.
    sigma = spread_covariance(50, rng)
    ratios = []
    for _ in range(20):
        R = simulate(sigma, 100, rng)
        ratios.append(
            np.linalg.norm(linear_shrinkage(R)[0] - sigma) ** 2
            / np.linalg.norm(sample_cov(R) - sigma) ** 2
        )
    assert np.mean(ratios) < 0.5


# --- linear_shrinkage: bad input --------------------------------------------------------


@pytest.mark.parametrize(
    "R",
    [np.ones(5), np.ones((1, 3)), np.array([[1.0, np.nan], [2.0, 3.0], [0.0, 1.0]])],
)
def test_linear_shrinkage_rejects_bad_input(R):
    with pytest.raises(ValueError):
        linear_shrinkage(R)
