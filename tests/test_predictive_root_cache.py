"""Tests for the X-keyed predictive sampling caches.

The decision loop of a bandit agent samples the same candidate matrix
against a static posterior thousands of times between updates. The
estimators serve that loop from a predictive root (joint ``sample``)
and a per-row sd vector (``sample_marginal``) cached on the estimator,
keyed by the content of ``X`` and invalidated whenever the posterior
changes.
"""

import numpy as np
import pytest
from scipy.sparse import csc_array
from scipy.sparse import random as sparse_random

import bayesianbandits._sparse_bayesian_linear_regression as sblr
from bayesianbandits import (
    BayesianGLM,
    NormalInverseGammaRegressor,
    NormalRegressor,
)


def _fitted_normal(sparse: bool = False, p: int = 30, seed: int = 0):
    rng = np.random.default_rng(seed)
    if sparse:
        X = csc_array(sparse_random(100, p, density=0.3, random_state=seed))
    else:
        X = rng.standard_normal((100, p))
    y = np.asarray(X @ rng.standard_normal(p)).ravel() + rng.standard_normal(100)
    est = NormalRegressor(alpha=1.0, beta=1.0, sparse=sparse, random_state=seed)
    est.fit(X, y)
    return est, rng


def _analytic_predictive_cov(est, X):
    dense = X.toarray() if hasattr(X, "toarray") else np.asarray(X)
    cov_inv = est.cov_inv_
    if hasattr(cov_inv, "toarray"):
        prec = cov_inv.toarray()
    else:
        upper = np.triu(cov_inv)
        prec = upper + upper.T - np.diag(np.diag(upper))
    return dense @ np.linalg.solve(prec, dense.T)


@pytest.mark.parametrize("sparse", [False, True])
def test_repeated_x_builds_root_with_exact_law(sparse: bool) -> None:
    est, rng = _fitted_normal(sparse=sparse)
    if sparse:
        X = csc_array(sparse_random(4, 30, density=0.5, random_state=7))
    else:
        X = rng.standard_normal((4, 30))

    # dense: first sighting rents (weight space), second builds the
    # root. Sparse: the reach-limited operator is cheap enough to
    # build and cache on first sight.
    est.sample(X, size=1)
    if not sparse:
        assert est._predictive_root_cache[1] is None
    est.sample(X, size=1)
    root = est._predictive_root_cache[1]
    assert root is not None

    # the cached root is an exact square root of X Λ⁻¹ Xᵀ
    np.testing.assert_allclose(
        root.T @ root, _analytic_predictive_cov(est, X), rtol=1e-8, atol=1e-10
    )

    # draws served from the cache follow the posterior predictive law
    draws = np.vstack([est.sample(X, size=1) for _ in range(4000)])
    mean = np.asarray(X @ est.coef_).ravel()
    np.testing.assert_allclose(draws.mean(axis=0), mean, atol=0.15)
    np.testing.assert_allclose(
        np.cov(draws.T),
        _analytic_predictive_cov(est, X),
        atol=0.15,
    )


def test_partial_fit_and_decay_invalidate_root_cache() -> None:
    est, rng = _fitted_normal()
    X = rng.standard_normal((4, 30))
    est.sample(X, size=1)
    est.sample(X, size=1)
    assert est._predictive_root_cache[1] is not None

    est.partial_fit(X, np.zeros(4))
    assert "_predictive_root_cache" not in est.__dict__

    est.sample(X, size=1)
    est.sample(X, size=1)
    assert est._predictive_root_cache[1] is not None
    est.decay(X, decay_rate=0.9)
    assert "_predictive_root_cache" not in est.__dict__

    # samples after the update center on the updated posterior mean
    draws = np.vstack([est.sample(X, size=1) for _ in range(3000)])
    np.testing.assert_allclose(
        draws.mean(axis=0), np.asarray(X @ est.coef_).ravel(), atol=0.2
    )


def test_changed_or_mutated_x_is_a_miss() -> None:
    est, rng = _fitted_normal()
    X = rng.standard_normal((4, 30))
    est.sample(X, size=1)
    est.sample(X, size=1)
    assert est._predictive_root_cache[1] is not None

    # in-place mutation of the caller's matrix must not hit the cache
    X[0, 0] += 1.0
    est.sample(X, size=1)
    assert est._predictive_root_cache[1] is None  # fresh sighting recorded

    est.sample(X, size=1)
    root = est._predictive_root_cache[1]
    assert root is not None
    np.testing.assert_allclose(
        root.T @ root, _analytic_predictive_cov(est, X), rtol=1e-8, atol=1e-10
    )


def test_more_rows_than_features_never_caches() -> None:
    est, rng = _fitted_normal(p=5)
    X = rng.standard_normal((8, 5))
    est.sample(X, size=1)
    est.sample(X, size=1)
    assert "_predictive_root_cache" not in est.__dict__


def test_gram_fallback_matches_analytic_cov(monkeypatch) -> None:
    """Force the blocked Gram path and check the root is still exact."""
    est, rng = _fitted_normal(sparse=True)
    X = csc_array(sparse_random(4, 30, density=0.5, random_state=7))
    monkeypatch.setattr(sblr, "_PREDICTIVE_DRAWS_ELEMS", 16)
    root = sblr.predictive_root(est._precision_factor, X)
    assert root is not None
    np.testing.assert_allclose(
        root.T @ root, _analytic_predictive_cov(est, X), rtol=1e-8, atol=1e-10
    )


def test_nig_cached_draws_follow_t_law() -> None:
    rng = np.random.default_rng(3)
    X_train = rng.standard_normal((200, 20))
    y = X_train @ rng.standard_normal(20) + rng.standard_normal(200)
    est = NormalInverseGammaRegressor(random_state=3)
    est.fit(X_train, y)
    X = rng.standard_normal((3, 20))

    est.sample(X, size=1)
    est.sample(X, size=1)
    assert est._predictive_root_cache[1] is not None

    draws = np.vstack([est.sample(X, size=1) for _ in range(4000)])
    mean = X @ est.coef_
    np.testing.assert_allclose(draws.mean(axis=0), mean, atol=0.2)
    # marginal variance of the t: (b/a) · xᵀΛ⁻¹x · df/(df-2), df = 2a
    df = 2 * est.a_
    base = np.diag(_analytic_predictive_cov(est, X)) * (est.b_ / est.a_)
    expected_var = base * df / (df - 2)
    np.testing.assert_allclose(draws.var(axis=0), expected_var, rtol=0.2)


def test_glm_cached_draws_match_weight_space_law() -> None:
    rng = np.random.default_rng(4)
    X_train = rng.standard_normal((200, 20))
    y = (rng.random(200) < 0.5).astype(np.float64)
    est = BayesianGLM(alpha=1.0, link="logit", random_state=4)
    est.fit(X_train, y)
    X = rng.standard_normal((3, 20))

    est.sample(X, size=1)
    est.sample(X, size=1)
    assert est._predictive_root_cache[1] is not None
    draws = np.vstack([est.sample(X, size=1) for _ in range(2000)])
    assert draws.shape == (2000, 3)
    assert np.all((draws > 0) & (draws < 1))


def test_marginal_sd_cache_hits_and_invalidates() -> None:
    est, rng = _fitted_normal()
    X = rng.standard_normal((4, 30))

    est.sample_marginal(X, size=2)
    cached_sd = est._marginal_sd_cache[1]
    est.sample_marginal(X, size=2)
    assert est._marginal_sd_cache[1] is cached_sd  # served from cache

    est.partial_fit(X, np.zeros(4))
    assert "_marginal_sd_cache" not in est.__dict__
    est.sample_marginal(X, size=2)
    assert est._marginal_sd_cache[1] is not cached_sd


def test_reach_budget_bail_falls_back_to_weight_space(monkeypatch) -> None:
    """A factor too dense for the reach solver must fall back to the
    exact weight-space path, not fail or serve bad draws."""
    est, rng = _fitted_normal(sparse=True)
    X = csc_array(sparse_random(4, 30, density=0.5, random_state=7))
    solver = est._precision_factor._reach_solver
    monkeypatch.setattr(solver, "_call_budget", 1)
    draws = np.vstack([est.sample(X, size=1) for _ in range(2000)])
    np.testing.assert_allclose(
        draws.mean(axis=0), np.asarray(X @ est.coef_).ravel(), atol=0.2
    )
    # entry budget bail behaves the same
    monkeypatch.setattr(solver, "_call_budget", 10_000)
    monkeypatch.setattr(solver, "_entry_budget", 0)
    s = est.sample(X, size=1)
    assert s.shape == (1, 4) and np.isfinite(s).all()


@pytest.mark.cython
def test_reach_solve_matches_dense_triangular_solve() -> None:
    """Direct check of the Cython kernel against scipy's dense solve."""
    from bayesianbandits._reach import reach_solve
    from scipy.linalg import solve_triangular

    rng = np.random.default_rng(0)
    p = 60
    L = np.tril(rng.normal(0, 0.3, (p, p)), -1) + np.diag(rng.uniform(1, 2, p))
    Lc = csc_array(L)
    b = np.zeros(p)
    nodes = np.array([3, 17, 40], dtype=np.int64)
    vals = rng.standard_normal(3)
    b[nodes] = vals
    work = np.zeros(p)
    stamp = np.zeros(p, dtype=np.int64)
    result = reach_solve(
        np.ascontiguousarray(Lc.data),
        np.ascontiguousarray(Lc.indices, dtype=np.int32),
        np.ascontiguousarray(Lc.indptr, dtype=np.int32),
        nodes,
        vals,
        work,
        stamp,
        1,
        p,
        p * p,
    )
    assert result is not None
    reach, out, entries = result
    ref = solve_triangular(L, b, lower=True)
    full = np.zeros(p)
    full[reach] = out
    np.testing.assert_allclose(full, ref, rtol=1e-10, atol=1e-12)
    assert np.all(work == 0.0)  # scratch restored
    # budget bail returns None and leaves scratch clean
    assert (
        reach_solve(
            np.ascontiguousarray(Lc.data),
            np.ascontiguousarray(Lc.indices, dtype=np.int32),
            np.ascontiguousarray(Lc.indptr, dtype=np.int32),
            nodes,
            vals,
            work,
            stamp,
            2,
            1,
            p * p,
        )
        is None
    )
    assert np.all(work == 0.0)
