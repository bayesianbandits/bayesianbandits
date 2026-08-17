"""Benchmarks for the support-covariance route: one solve per distinct
column touched instead of one per draw or per row.

Each pair benchmarks the same call with the route on and off; the
``declines_`` pair guards the region where the pre-existing path is
better.
"""

from contextlib import contextmanager
from unittest import mock

import numpy as np
import pytest
import scipy.sparse as sp

from bayesianbandits import NormalRegressor
from bayesianbandits import _support_covariance as sc

N_FEATURES = 100_000


@contextmanager
def _gate_off():
    with mock.patch.object(sc, "build", lambda *a, **k: None):
        yield


def _fit(n_features, rng):
    est = NormalRegressor(alpha=1.0, beta=1.0, sparse=True, random_state=0)
    X = sp.csc_array(sp.random(200, n_features, density=1e-4, random_state=42))
    est.fit(X, rng.standard_normal(200))
    return est


def _design(n_features, n_rows, n_u, rng, nnz_per_row=8):
    support = np.sort(rng.choice(n_features, n_u, replace=False))
    rows, cols, vals = [], [], []
    for i in range(n_rows):
        pick = rng.choice(n_u, nnz_per_row, replace=False)
        rows += [i] * nnz_per_row
        cols += list(support[pick])
        vals += list(rng.standard_normal(nnz_per_row))
    return sp.csc_array(sp.csr_array((vals, (rows, cols)), shape=(n_rows, n_features)))


@pytest.fixture(scope="module")
def arms_96():
    """96 arms over a 49-column support: the production agent's shape."""
    rng = np.random.default_rng(0)
    return _fit(N_FEATURES, rng), _design(N_FEATURES, 96, 49, rng)


@pytest.fixture(scope="module")
def rows_2000():
    """Many prediction rows over a small support."""
    rng = np.random.default_rng(1)
    return _fit(N_FEATURES, rng), _design(N_FEATURES, 2_000, 49, rng)


@pytest.fixture(scope="module")
def wide_support_10_rows():
    """Few rows, wide support: the route must decline here."""
    rng = np.random.default_rng(2)
    return _fit(N_FEATURES, rng), _design(N_FEATURES, 10, 800, rng)


# -- joint draws, size=500 (the UCB/EXP3A regime) ------------------------------


@pytest.mark.benchmark(group="support_cov_joint_arms96")
def test_sample_500_arms96_route(benchmark, arms_96):
    est, X = arms_96
    benchmark(est.sample, X, size=500)


@pytest.mark.benchmark(group="support_cov_joint_arms96")
def test_sample_500_arms96_weight_space(benchmark, arms_96):
    est, X = arms_96
    with _gate_off():
        benchmark(est.sample, X, size=500)


# -- per-row standard deviations ----------------------------------------------


@pytest.mark.benchmark(group="support_cov_marginal_rows2000")
def test_sample_marginal_500_rows2000_route(benchmark, rows_2000):
    est, X = rows_2000
    benchmark(est.sample_marginal, X, size=500)


@pytest.mark.benchmark(group="support_cov_marginal_rows2000")
def test_sample_marginal_500_rows2000_half_solves(benchmark, rows_2000):
    est, X = rows_2000
    with _gate_off():
        benchmark(est.sample_marginal, X, size=500)


# -- the region the gate must keep declining ----------------------------------


@pytest.mark.benchmark(group="support_cov_declines_wide")
def test_sample_marginal_500_wide_support_route_declines(
    benchmark, wide_support_10_rows
):
    est, X = wide_support_10_rows
    benchmark(est.sample_marginal, X, size=500)


@pytest.mark.benchmark(group="support_cov_declines_wide")
def test_sample_marginal_500_wide_support_half_solves(benchmark, wide_support_10_rows):
    est, X = wide_support_10_rows
    with _gate_off():
        benchmark(est.sample_marginal, X, size=500)


# -- Thompson hot path: must be untouched -------------------------------------


@pytest.mark.benchmark(group="support_cov_thompson")
def test_sample_1_arms96_route_declines(benchmark, arms_96):
    est, X = arms_96
    benchmark(est.sample, X, size=1)
