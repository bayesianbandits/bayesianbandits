"""Tests for EmpiricalBayesNormalRegressor."""

import pickle
from unittest import mock

import numpy as np
import pytest
from sklearn.base import clone

from bayesianbandits import EmpiricalBayesNormalRegressor
from bayesianbandits._sparse_bayesian_linear_regression import SparseSolver


suitespare_envvar_params = [
    SparseSolver.SUPERLU,
    SparseSolver.CHOLMOD,
]


@pytest.fixture(
    params=suitespare_envvar_params,
    autouse=True,
)
def suitesparse_envvar(request):
    """Allows running test suite with and without CHOLMOD."""
    with mock.patch(
        "bayesianbandits._sparse_bayesian_linear_regression.solver", request.param
    ):
        yield


@pytest.fixture
def regression_data():
    rng = np.random.default_rng(42)
    X = rng.standard_normal((100, 5))
    w_true = np.array([1.0, -2.0, 0.5, 0.0, 3.0])
    y = X @ w_true + rng.standard_normal(100) * 0.1
    return X, y


@pytest.mark.parametrize("sparse", [True, False])
class TestEBNormalRegressor:
    def test_fit_smoke(self, regression_data, sparse):
        X, y = regression_data
        model = EmpiricalBayesNormalRegressor(alpha=1.0, beta=1.0, sparse=sparse)
        model.fit(X, y)

        assert hasattr(model, "log_evidence_")
        assert hasattr(model, "n_eb_iterations_")
        assert hasattr(model, "eb_converged_")
        assert np.isfinite(model.log_evidence_)

        preds = model.predict(X)
        assert preds.shape == (100,)

        samples = model.sample(X, size=3)
        assert samples.shape == (3, 100)

    def test_hyperparams_change(self, regression_data, sparse):
        X, y = regression_data
        model = EmpiricalBayesNormalRegressor(alpha=1.0, beta=1.0, sparse=sparse)
        model.fit(X, y)

        assert model.alpha != 1.0 or model.beta != 1.0

    def test_evidence_monotonicity(self, regression_data, sparse):
        X, y = regression_data
        evidences = []

        model = EmpiricalBayesNormalRegressor(
            alpha=1.0, beta=1.0, n_eb_iter=1, eb_tol=0.0, sparse=sparse
        )
        for _ in range(10):
            model.fit(X, y)
            evidences.append(model.log_evidence_)
            # Use the updated hyperparams for next iteration
            model = EmpiricalBayesNormalRegressor(
                alpha=model.alpha, beta=model.beta, n_eb_iter=1, eb_tol=0.0,
                sparse=sparse,
            )

        for i in range(1, len(evidences)):
            assert evidences[i] >= evidences[i - 1] - 1e-6, (
                f"Evidence decreased at step {i}: {evidences[i]} < {evidences[i-1]}"
            )

    def test_convergence_flag(self, regression_data, sparse):
        X, y = regression_data
        model = EmpiricalBayesNormalRegressor(
            alpha=1.0, beta=1.0, n_eb_iter=100, eb_tol=1e-2, sparse=sparse,
        )
        model.fit(X, y)
        assert model.eb_converged_ is True
        assert model.n_eb_iterations_ < 100

    def test_partial_fit_updates_hyperparams(self, regression_data, sparse):
        X, y = regression_data
        model = EmpiricalBayesNormalRegressor(alpha=1.0, beta=1.0, sparse=sparse)
        model.fit(X[:50], y[:50])

        alpha_before = model.alpha
        beta_before = model.beta

        model.partial_fit(X[50:], y[50:])

        assert model.alpha != alpha_before or model.beta != beta_before

    def test_n_eb_iter_zero(self, regression_data, sparse):
        X, y = regression_data

        model_eb = EmpiricalBayesNormalRegressor(
            alpha=1.0, beta=1.0, n_eb_iter=0, random_state=42, sparse=sparse,
        )
        model_eb.fit(X, y)

        from bayesianbandits import NormalRegressor

        model_plain = NormalRegressor(
            alpha=1.0, beta=1.0, random_state=42, sparse=sparse,
        )
        model_plain.fit(X, y)

        np.testing.assert_allclose(model_eb.coef_, model_plain.coef_)
        np.testing.assert_allclose(model_eb.predict(X), model_plain.predict(X))

    def test_get_set_params(self, sparse):
        model = EmpiricalBayesNormalRegressor(alpha=2.0, beta=3.0, n_eb_iter=5, sparse=sparse)
        params = model.get_params()
        assert params["alpha"] == 2.0
        assert params["beta"] == 3.0
        assert params["n_eb_iter"] == 5

        model.set_params(alpha=10.0)
        assert model.alpha == 10.0

    def test_clone(self, sparse):
        model = EmpiricalBayesNormalRegressor(alpha=2.0, beta=3.0, n_eb_iter=7, sparse=sparse)
        cloned = clone(model)
        assert cloned.get_params() == model.get_params()
        assert cloned is not model

    def test_pickle_roundtrip(self, regression_data, sparse):
        X, y = regression_data
        model = EmpiricalBayesNormalRegressor(alpha=1.0, beta=1.0, sparse=sparse)
        model.fit(X, y)

        data = pickle.dumps(model)
        loaded = pickle.loads(data)

        np.testing.assert_allclose(loaded.predict(X), model.predict(X))
        assert loaded.log_evidence_ == model.log_evidence_
        assert loaded.alpha == model.alpha
        assert loaded.beta == model.beta
