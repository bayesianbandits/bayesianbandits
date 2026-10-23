"""Tests for the argmax probabilities behind Thompson sampling's propensity."""

from typing import Any, Optional

import numpy as np
import pytest
from numpy.typing import NDArray
from scipy.stats import norm
from scipy.stats import t as student_t

from bayesianbandits._argmax_probability import (
    argmax_probabilities,
    independent_argmax_probabilities,
)


def _f64(a: Any) -> NDArray[np.float64]:
    return np.asarray(a, dtype=np.float64)


def _monte_carlo(
    mean: NDArray[np.float64],
    cov: NDArray[np.float64],
    n_draws: int,
    rng: np.random.Generator,
    df: Optional[NDArray[np.float64]] = None,
) -> NDArray[np.float64]:
    n, n_arms = mean.shape
    L = np.linalg.cholesky(cov + 1e-10 * np.eye(n_arms))
    r = np.einsum("nij,npj->npi", L, rng.standard_normal((n, n_draws, n_arms)))
    if df is not None:
        r *= np.sqrt(df[:, None] / rng.chisquare(df[:, None], (n, n_draws)))[:, :, None]
    picks = (r + mean[:, None, :]).argmax(axis=2)
    return np.stack([(picks == a).mean(axis=1) for a in range(n_arms)], axis=1)


@pytest.mark.parametrize("df", [None, 3.0])
def test_matches_monte_carlo(df: Optional[float]) -> None:
    """Rank-3 covariances: the shared-learner case."""
    rng = np.random.default_rng(0)
    n, n_arms = 4, 5
    mean = _f64(2 * rng.normal(size=(n, n_arms)))
    A = rng.normal(size=(n, n_arms, 3))
    cov = _f64(A @ A.transpose(0, 2, 1))
    dfs = None if df is None else np.full(n, df)

    probs = argmax_probabilities(mean, cov, dfs)

    assert np.allclose(probs.sum(axis=1), 1.0, atol=3e-3)
    assert np.abs(probs - _monte_carlo(mean, cov, 400_000, rng, dfs)).max() < 5e-3


def test_tail_of_rank_deficient_scores() -> None:
    """scipy's Genz and a direct 2-D quadrature both give 1.2698143e-10,
    which Monte Carlo cannot see."""
    A = np.array([[[1.0, 0.0], [0.8, 0.6], [0.5, 0.9], [-0.2, 1.1]]])
    mean = np.array([[-4.0, 0.0, 0.3, 0.6]])

    probs = argmax_probabilities(mean, _f64(A @ A.transpose(0, 2, 1)))

    assert probs[0, 0] == pytest.approx(1.2698143e-10, rel=1e-5)


def test_two_arms_closed_form() -> None:
    mean = np.array([[0.0, 8.0]])
    exact = norm.cdf(-8 / np.sqrt(2))

    assert argmax_probabilities(mean, _f64([np.eye(2)]))[0, 0] == pytest.approx(
        exact, rel=1e-6
    )
    assert independent_argmax_probabilities(mean, np.ones((1, 2)))[
        0, 0
    ] == pytest.approx(exact, rel=1e-6)


def test_independent_matches_general() -> None:
    rng = np.random.default_rng(1)
    n, n_arms = 5, 4
    mean = _f64(rng.normal(size=(n, n_arms)))
    var = _f64(rng.uniform(0.2, 2.0, (n, n_arms)))

    got = independent_argmax_probabilities(mean, var)
    expected = argmax_probabilities(
        mean, _f64(var[:, :, None] * np.eye(n_arms)), n_points=4096
    )

    assert np.allclose(got.sum(axis=1), 1.0, atol=1e-6)
    assert np.abs(got - expected).max() < 2e-3


def test_independent_student_t() -> None:
    """Each arm its own t, as independent NIG learners give; an infinite
    ``df`` is the Gaussian arm."""
    rng = np.random.default_rng(2)
    n_arms = 4
    mean = _f64(rng.normal(size=(1, n_arms)))
    sd = _f64(rng.uniform(0.5, 1.5, (1, n_arms)))
    df = np.array([[3.0, 5.0, 4.0, 8.0]])

    probs = independent_argmax_probabilities(mean, _f64(sd**2), df)
    draws = mean + sd * student_t.rvs(df, size=(400_000, n_arms), random_state=rng)
    mc = np.bincount(draws.argmax(axis=1), minlength=n_arms) / len(draws)

    assert np.allclose(probs.sum(axis=1), 1.0, atol=1e-6)
    assert np.abs(probs[0] - mc).max() < 5e-3
    assert np.allclose(
        independent_argmax_probabilities(
            mean, _f64(sd**2), np.full((1, n_arms), np.inf)
        ),
        independent_argmax_probabilities(mean, _f64(sd**2)),
        atol=1e-6,
    )


def test_ties_and_single_arm() -> None:
    """Identical scores go to the lowest index, as ``argmax`` breaks ties."""
    tied = argmax_probabilities(np.zeros((1, 3)), np.ones((1, 3, 3)))
    assert tied.tolist() == [[1.0, 0.0, 0.0]]
    assert argmax_probabilities(np.zeros((1, 1)), np.ones((1, 1, 1))).tolist() == [
        [1.0]
    ]
