"""Benchmarks for forgetting rules: exponential, stabilized (KZ), and SIFt."""

import numpy as np
import pytest
from scipy.sparse import csc_array
from scipy.sparse import random as sparse_random

from bayesianbandits._forgetting import (
    ExponentialForgetting,
    SiftForgetting,
    StabilizedForgetting,
    filter_batch,
)

# ---------------------------------------------------------------------------
# Data helpers
# ---------------------------------------------------------------------------


def _make_dense_precision(n, rng):
    """alpha * I + X.T @ X with dense X, giving a realistic dense PD matrix."""
    X_hist = rng.standard_normal((max(n, 200), n))
    return np.eye(n) + X_hist.T @ X_hist / X_hist.shape[0]


def _make_sparse_precision(n, rng, density=None):
    """alpha * I + X.T @ X with sparse X, giving a sparse PD matrix."""
    if density is None:
        nnz_per_row = 10
        density = min(nnz_per_row / n, 0.1)
    X_hist = csc_array(sparse_random(200, n, density=density, random_state=rng))
    eye = csc_array((np.ones(n), (np.arange(n), np.arange(n))), shape=(n, n))
    return eye + X_hist.T @ X_hist / 200


def _make_batch(n, p, rng, sparse=False, density=None):
    """Generate a (p, n) batch and (p,) targets."""
    if sparse:
        if density is None:
            nnz_per_row = 10
            density = min(nnz_per_row / n, 0.1)
        X = csc_array(sparse_random(p, n, density=density, random_state=rng))
    else:
        X = rng.standard_normal((p, n))
    y = rng.standard_normal(p)
    return X, y


# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------

BATCH_SIZE = 5
LAM = 0.95


@pytest.fixture
def forgetting_dense_100():
    rng = np.random.default_rng(42)
    R = _make_dense_precision(100, rng)
    X, y = _make_batch(100, BATCH_SIZE, rng)
    return R, X, y


@pytest.fixture
def forgetting_dense_1k():
    rng = np.random.default_rng(42)
    R = _make_dense_precision(1_000, rng)
    X, y = _make_batch(1_000, BATCH_SIZE, rng)
    return R, X, y


@pytest.fixture
def forgetting_sparse_1k():
    rng = np.random.default_rng(42)
    R = _make_sparse_precision(1_000, rng)
    X, y = _make_batch(1_000, BATCH_SIZE, rng, sparse=True)
    return R, X, y


@pytest.fixture
def forgetting_sparse_100k():
    rng = np.random.default_rng(42)
    R = _make_sparse_precision(100_000, rng, density=0.0001)
    X, y = _make_batch(100_000, BATCH_SIZE, rng, sparse=True, density=0.0001)
    return R, X, y


@pytest.fixture
def forgetting_sparse_1m():
    rng = np.random.default_rng(42)
    R = _make_sparse_precision(1_000_000, rng, density=0.00001)
    X, y = _make_batch(1_000_000, BATCH_SIZE, rng, sparse=True, density=0.00001)
    return R, X, y


# ---------------------------------------------------------------------------
# filter_batch benchmarks
# ---------------------------------------------------------------------------


def test_filter_batch_dense_100(benchmark, forgetting_dense_100):
    _, X, y = forgetting_dense_100
    benchmark(filter_batch, X, y, eps=1e-12)


def test_filter_batch_dense_1k(benchmark, forgetting_dense_1k):
    _, X, y = forgetting_dense_1k
    benchmark(filter_batch, X, y, eps=1e-12)


def test_filter_batch_sparse_1k(benchmark, forgetting_sparse_1k):
    _, X, y = forgetting_sparse_1k
    benchmark(filter_batch, X, y, eps=1e-12)


def test_filter_batch_sparse_100k(benchmark, forgetting_sparse_100k):
    _, X, y = forgetting_sparse_100k
    benchmark(filter_batch, X, y, eps=1e-12)


@pytest.mark.slow
def test_filter_batch_sparse_1m(benchmark, forgetting_sparse_1m):
    _, X, y = forgetting_sparse_1m
    benchmark(filter_batch, X, y, eps=1e-12)


# ---------------------------------------------------------------------------
# Forgetting rule benchmarks (parametrized over rules)
# ---------------------------------------------------------------------------

RULES = [
    pytest.param(ExponentialForgetting(), id="exponential"),
    pytest.param(StabilizedForgetting(alpha=1.0), id="stabilized"),
    pytest.param(SiftForgetting(eps=1e-12), id="sift"),
]


@pytest.mark.parametrize("rule", RULES)
def test_rule_dense_100(benchmark, rule, forgetting_dense_100):
    R, X, y = forgetting_dense_100
    benchmark(rule, R, X, y, LAM)


@pytest.mark.parametrize("rule", RULES)
def test_rule_dense_1k(benchmark, rule, forgetting_dense_1k):
    R, X, y = forgetting_dense_1k
    benchmark(rule, R, X, y, LAM)


@pytest.mark.parametrize("rule", RULES)
def test_rule_sparse_1k(benchmark, rule, forgetting_sparse_1k):
    R, X, y = forgetting_sparse_1k
    benchmark(rule, R, X, y, LAM)


@pytest.mark.parametrize("rule", RULES)
def test_rule_sparse_100k(benchmark, rule, forgetting_sparse_100k):
    R, X, y = forgetting_sparse_100k
    benchmark(rule, R, X, y, LAM)


@pytest.mark.slow
@pytest.mark.parametrize("rule", RULES)
def test_rule_sparse_1m(benchmark, rule, forgetting_sparse_1m):
    R, X, y = forgetting_sparse_1m
    benchmark(rule, R, X, y, LAM)
