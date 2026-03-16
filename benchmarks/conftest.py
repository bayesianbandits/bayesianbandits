"""Benchmark fixtures for estimator performance testing."""

import copy

import numpy as np
import pytest
from scipy.sparse import csc_array
from scipy.sparse import random as sparse_random

from bayesianbandits._estimators import (
    BayesianGLM,
    NormalInverseGammaRegressor,
    NormalRegressor,
)


def _make_estimator(estimator_type, sparse):
    """Create an unfitted estimator of the given type."""
    if estimator_type == "normal":
        return NormalRegressor(alpha=1.0, beta=1.0, sparse=sparse)
    elif estimator_type == "nig":
        return NormalInverseGammaRegressor(sparse=sparse)
    elif estimator_type == "glm_logit":
        return BayesianGLM(alpha=1.0, link="logit", sparse=sparse)
    elif estimator_type == "glm_log":
        return BayesianGLM(alpha=1.0, link="log", sparse=sparse)
    else:
        raise ValueError(f"Unknown estimator type: {estimator_type}")


def _make_data(kind, n_features, n_obs=200, rng=None):
    """Generate training and test data at the given scale."""
    if rng is None:
        rng = np.random.default_rng(42)

    if kind == "dense":
        X_train = rng.standard_normal((n_obs, n_features))
        X_test = rng.standard_normal((10, n_features))
    else:
        nnz_per_row = 10
        density = min(nnz_per_row / n_features, 0.1)
        X_train = csc_array(
            sparse_random(n_obs, n_features, density=density, random_state=42)
        )
        X_test = csc_array(
            sparse_random(10, n_features, density=density, random_state=43)
        )
    return X_train, X_test


def _fit_estimator(estimator_type, kind, n_features):
    """Build and fit one estimator, returning (est, X_test, name)."""
    is_sparse = kind == "sparse"
    est = _make_estimator(estimator_type, sparse=is_sparse)

    rng = np.random.default_rng(42)
    X_train, X_test = _make_data(kind, n_features, n_obs=200, rng=rng)

    if estimator_type == "glm_logit":
        y = rng.integers(0, 2, size=X_train.shape[0]).astype(np.float64)
    elif estimator_type == "glm_log":
        y = rng.poisson(3, size=X_train.shape[0]).astype(np.float64)
    else:
        y = rng.standard_normal(X_train.shape[0])

    est.fit(X_train, y)
    return est, X_test, f"{estimator_type}-{kind}-{n_features}"


@pytest.fixture
def normal_dense_100():
    return _fit_estimator("normal", "dense", 100)


@pytest.fixture
def normal_dense_100_fresh(normal_dense_100):
    """Deep-copy factory so partial_fit benchmarks don't accumulate state."""
    est, X_test, name = normal_dense_100
    return lambda: (copy.deepcopy(est), X_test, name)


@pytest.fixture
def normal_dense_1k():
    return _fit_estimator("normal", "dense", 1_000)


@pytest.fixture
def normal_dense_1k_fresh(normal_dense_1k):
    est, X_test, name = normal_dense_1k
    return lambda: (copy.deepcopy(est), X_test, name)


@pytest.fixture
def normal_sparse_1k():
    return _fit_estimator("normal", "sparse", 1_000)


@pytest.fixture
def normal_sparse_1k_fresh(normal_sparse_1k):
    est, X_test, name = normal_sparse_1k
    return lambda: (copy.deepcopy(est), X_test, name)


@pytest.fixture
def normal_sparse_1m():
    return _fit_estimator("normal", "sparse", 1_000_000)


@pytest.fixture
def normal_sparse_1m_fresh(normal_sparse_1m):
    est, X_test, name = normal_sparse_1m
    return lambda: (copy.deepcopy(est), X_test, name)


@pytest.fixture
def glm_logit_dense_100():
    return _fit_estimator("glm_logit", "dense", 100)


@pytest.fixture
def glm_logit_dense_100_fresh(glm_logit_dense_100):
    est, X_test, name = glm_logit_dense_100
    return lambda: (copy.deepcopy(est), X_test, name)


@pytest.fixture
def glm_logit_dense_1k():
    return _fit_estimator("glm_logit", "dense", 1_000)


@pytest.fixture
def glm_logit_dense_1k_fresh(glm_logit_dense_1k):
    est, X_test, name = glm_logit_dense_1k
    return lambda: (copy.deepcopy(est), X_test, name)


@pytest.fixture
def glm_logit_sparse_1k():
    return _fit_estimator("glm_logit", "sparse", 1_000)


@pytest.fixture
def glm_logit_sparse_1k_fresh(glm_logit_sparse_1k):
    est, X_test, name = glm_logit_sparse_1k
    return lambda: (copy.deepcopy(est), X_test, name)


@pytest.fixture
def glm_logit_sparse_100k():
    return _fit_estimator("glm_logit", "sparse", 100_000)


@pytest.fixture
def glm_logit_sparse_100k_fresh(glm_logit_sparse_100k):
    est, X_test, name = glm_logit_sparse_100k
    return lambda: (copy.deepcopy(est), X_test, name)


@pytest.fixture
def glm_logit_sparse_1m():
    return _fit_estimator("glm_logit", "sparse", 1_000_000)


@pytest.fixture
def glm_logit_sparse_1m_fresh(glm_logit_sparse_1m):
    est, X_test, name = glm_logit_sparse_1m
    return lambda: (copy.deepcopy(est), X_test, name)
