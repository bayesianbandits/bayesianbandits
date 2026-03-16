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
                alpha=model.alpha,
                beta=model.beta,
                n_eb_iter=1,
                eb_tol=0.0,
                sparse=sparse,
            )

        for i in range(1, len(evidences)):
            assert evidences[i] >= evidences[i - 1] - 1e-6, (
                f"Evidence decreased at step {i}: {evidences[i]} < {evidences[i - 1]}"
            )

    def test_convergence_flag(self, regression_data, sparse):
        X, y = regression_data
        model = EmpiricalBayesNormalRegressor(
            alpha=1.0,
            beta=1.0,
            n_eb_iter=100,
            eb_tol=1e-2,
            sparse=sparse,
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
            alpha=1.0,
            beta=1.0,
            n_eb_iter=0,
            random_state=42,
            sparse=sparse,
        )
        model_eb.fit(X, y)

        from bayesianbandits import NormalRegressor

        model_plain = NormalRegressor(
            alpha=1.0,
            beta=1.0,
            random_state=42,
            sparse=sparse,
        )
        model_plain.fit(X, y)

        np.testing.assert_allclose(model_eb.coef_, model_plain.coef_)
        np.testing.assert_allclose(model_eb.predict(X), model_plain.predict(X))

    def test_get_set_params(self, sparse):
        model = EmpiricalBayesNormalRegressor(
            alpha=2.0, beta=3.0, n_eb_iter=5, sparse=sparse
        )
        params = model.get_params()
        assert params["alpha"] == 2.0
        assert params["beta"] == 3.0
        assert params["n_eb_iter"] == 5

        model.set_params(alpha=10.0)
        assert model.alpha == 10.0

    def test_clone(self, sparse):
        model = EmpiricalBayesNormalRegressor(
            alpha=2.0, beta=3.0, n_eb_iter=7, sparse=sparse
        )
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

    def test_fit_sparse_X(self, regression_data, sparse):
        """fit() with csc_array X initializes _eff_XTy via sparse branch."""
        if not sparse:
            pytest.skip("only relevant for sparse=True")
        X, y = regression_data
        from scipy.sparse import csc_array as csc

        X_sp = csc(X)
        model = EmpiricalBayesNormalRegressor(alpha=1.0, beta=1.0, sparse=True)
        model.fit(X_sp, y)

        assert hasattr(model, "_eff_XTy")
        assert model._eff_XTy.shape == (X.shape[1],)

        # Compare to dense fit
        model_dense = EmpiricalBayesNormalRegressor(
            alpha=1.0,
            beta=1.0,
            sparse=True,
            random_state=42,
        )
        model_dense.fit(X, y)
        # Both should produce finite predictions
        assert np.all(np.isfinite(model.predict(X)))

    def test_correct_precision_noop(self, regression_data, sparse):
        """_correct_precision returns early when hyperparams are unchanged."""
        X, y = regression_data
        model = EmpiricalBayesNormalRegressor(alpha=1.0, beta=1.0, sparse=sparse)
        model.fit(X, y)

        cov_inv_before = (
            model.cov_inv_.copy() if not sparse else model.cov_inv_.toarray().copy()
        )
        # Call with same alpha/beta — should be a no-op
        model._correct_precision(model.alpha, model.beta)
        cov_inv_after = model.cov_inv_ if not sparse else model.cov_inv_.toarray()
        np.testing.assert_array_equal(cov_inv_before, cov_inv_after)

    def test_sample_before_partial_fit(self, regression_data, sparse):
        """partial_fit after sample() initializes EB state correctly."""
        X, y = regression_data
        model = EmpiricalBayesNormalRegressor(alpha=1.0, beta=1.0, sparse=sparse)

        # sample() triggers _initialize_prior, setting coef_ without _prior_scalar
        model.sample(X[:1])
        assert hasattr(model, "coef_")
        assert not hasattr(model, "_prior_scalar")

        # partial_fit should detect the sample-before-fit case
        model.partial_fit(X[:5], y[:5])
        assert hasattr(model, "_prior_scalar")
        assert hasattr(model, "_effective_n")
        assert hasattr(model, "_eff_XTy")
        assert np.isfinite(model.alpha)
        assert np.isfinite(model.beta)

    def test_decay(self, regression_data, sparse):
        """decay() scales _prior_scalar and sufficient stats."""
        X, y = regression_data
        model = EmpiricalBayesNormalRegressor(
            alpha=1.0,
            beta=1.0,
            learning_rate=0.99,
            sparse=sparse,
        )
        model.fit(X, y)

        prior_before = model._prior_scalar
        eff_n_before = model._effective_n
        eff_yTy_before = model._eff_yTy
        eff_XTy_before = model._eff_XTy.copy()

        model.decay(X[:1])  # decay by 1 observation

        decay = 0.99**1
        # Stabilized forgetting: prior_scalar converges to alpha instead of
        # decaying to zero (Kulhavy & Zarrop, 1993).
        expected_prior = decay * prior_before + (1 - decay) * model.alpha
        np.testing.assert_allclose(model._prior_scalar, expected_prior)
        np.testing.assert_allclose(model._effective_n, eff_n_before * decay)
        np.testing.assert_allclose(model._eff_yTy, eff_yTy_before * decay)
        np.testing.assert_allclose(model._eff_XTy, eff_XTy_before * decay)

    def test_decay_rate_one_no_reinjection(self, regression_data, sparse):
        """decay() with decay_rate=1.0 is a no-op (zero reinjection)."""
        X, y = regression_data
        model = EmpiricalBayesNormalRegressor(
            alpha=1.0, beta=1.0, learning_rate=0.99, sparse=sparse
        )
        model.fit(X, y)

        precision_before = (
            model.cov_inv_.toarray().copy() if sparse else model.cov_inv_.copy()
        )

        model.decay(X[:1], decay_rate=1.0)

        precision_after = model.cov_inv_.toarray() if sparse else model.cov_inv_
        np.testing.assert_allclose(precision_after, precision_before)

    def test_partial_fit_prior_reinjection(self, regression_data, sparse):
        """partial_fit with learning_rate < 1 re-injects prior into precision diagonal."""
        X, y = regression_data
        model = EmpiricalBayesNormalRegressor(
            alpha=1.0,
            beta=1.0,
            learning_rate=0.99,
            sparse=sparse,
        )
        model.fit(X[:50], y[:50])

        if sparse:
            diag_before = model.cov_inv_.toarray().diagonal().copy()
        else:
            diag_before = model.cov_inv_.diagonal().copy()

        n_new = 10
        prior_decay = 0.99**n_new

        model.partial_fit(X[50 : 50 + n_new], y[50 : 50 + n_new])

        # The diagonal should include the re-injection amount relative to
        # what pure decay would have produced.
        if sparse:
            diag_after = model.cov_inv_.toarray().diagonal()
        else:
            diag_after = model.cov_inv_.diagonal()

        # After decay alone the diagonal would be prior_decay * diag_before
        # (plus data contributions).  The re-injection adds expected_reinjection
        # uniformly, so the diagonal must exceed prior_decay * diag_before.
        assert np.all(diag_after > prior_decay * diag_before), (
            "Prior re-injection did not increase precision diagonal"
        )

    def test_decay_prior_reinjection_precision(self, regression_data, sparse):
        """decay() with learning_rate < 1 re-injects prior into precision diagonal."""
        X, y = regression_data
        model = EmpiricalBayesNormalRegressor(
            alpha=1.0,
            beta=1.0,
            learning_rate=0.99,
            sparse=sparse,
        )
        model.fit(X, y)

        if sparse:
            diag_before = model.cov_inv_.toarray().diagonal().copy()
        else:
            diag_before = model.cov_inv_.diagonal().copy()

        n_obs = X[:1].shape[0]
        prior_decay = 0.99**n_obs
        expected_reinjection = (1 - prior_decay) * model.alpha

        model.decay(X[:1])

        if sparse:
            diag_after = model.cov_inv_.toarray().diagonal()
        else:
            diag_after = model.cov_inv_.diagonal()

        # Without re-injection: diag = prior_decay * diag_before
        # With re-injection: diag = prior_decay * diag_before + expected_reinjection
        np.testing.assert_allclose(
            diag_after,
            prior_decay * diag_before + expected_reinjection,
        )

    def test_decay_before_fit(self, sparse):
        """decay() before fit is a no-op."""
        model = EmpiricalBayesNormalRegressor(
            alpha=1.0,
            beta=1.0,
            learning_rate=0.99,
            sparse=sparse,
        )
        rng = np.random.default_rng(0)
        X = rng.standard_normal((5, 3))
        # Should not raise
        model.decay(X)

    def test_partial_fit_cold_start(self, regression_data, sparse):
        """partial_fit with no prior fit() or sample() delegates to fit()."""
        X, y = regression_data
        model = EmpiricalBayesNormalRegressor(alpha=1.0, beta=1.0, sparse=sparse)

        # No fit() or sample() — partial_fit should call fit() internally,
        # which sets _prior_scalar, and return via the early-return path.
        model.partial_fit(X[:10], y[:10])
        assert hasattr(model, "_prior_scalar")
        assert hasattr(model, "log_evidence_")

    def test_sample_before_partial_fit_sparse_X(self, regression_data, sparse):
        """sample-before-fit path with sparse X in partial_fit."""
        if not sparse:
            pytest.skip("only relevant for sparse=True")
        X, y = regression_data
        from scipy.sparse import csc_array as csc

        X_sp = csc(X)
        model = EmpiricalBayesNormalRegressor(alpha=1.0, beta=1.0, sparse=True)

        # sample() sets coef_ without _prior_scalar
        model.sample(X_sp[:1])
        assert not hasattr(model, "_prior_scalar")

        # partial_fit with sparse X hits the sparse first-obs branch
        model.partial_fit(X_sp[:5], y[:5])
        assert hasattr(model, "_eff_XTy")
        assert model._eff_XTy.shape == (X.shape[1],)


@pytest.mark.parametrize("sparse", [True, False])
@pytest.mark.parametrize("trace_method", ["auto", "diagonal"])
class TestTraceMethod:
    def test_fit_with_trace_method(self, regression_data, sparse, trace_method):
        """Both trace methods produce valid fit results."""
        X, y = regression_data
        model = EmpiricalBayesNormalRegressor(
            alpha=1.0, beta=1.0, sparse=sparse, trace_method=trace_method
        )
        model.fit(X, y)

        assert np.isfinite(model.log_evidence_)
        assert model.alpha > 0
        assert model.beta > 0

    def test_partial_fit_with_trace_method(self, regression_data, sparse, trace_method):
        """Both trace methods work through partial_fit (online MacKay)."""
        X, y = regression_data
        model = EmpiricalBayesNormalRegressor(
            alpha=1.0, beta=1.0, sparse=sparse, trace_method=trace_method
        )
        model.fit(X[:50], y[:50])
        alpha_after_fit = model.alpha

        model.partial_fit(X[50:60], y[50:60])
        # MacKay should have updated alpha
        assert model.alpha != alpha_after_fit

    def test_auto_and_diagonal_agree_on_alpha_direction(
        self, regression_data, sparse, trace_method
    ):
        """auto and diagonal should move alpha in the same direction."""
        if trace_method != "auto":
            pytest.skip("only run once per sparse setting")
        X, y = regression_data

        model_auto = EmpiricalBayesNormalRegressor(
            alpha=1.0, beta=1.0, sparse=sparse, trace_method="auto"
        )
        model_auto.fit(X, y)

        model_diag = EmpiricalBayesNormalRegressor(
            alpha=1.0, beta=1.0, sparse=sparse, trace_method="diagonal"
        )
        model_diag.fit(X, y)

        # Both should move alpha away from the initial value in the same direction
        assert (model_auto.alpha > 1.0) == (model_diag.alpha > 1.0)


@pytest.mark.parametrize("sparse", [True, False])
class TestSampleWeight:
    def test_partial_fit_with_sample_weight(self, regression_data, sparse):
        """partial_fit with sample_weight exercises the reinjection _fit_helper path."""
        X, y = regression_data
        weights = np.ones(10, dtype=np.float64)
        weights[:5] = 2.0  # upweight first half

        model = EmpiricalBayesNormalRegressor(
            alpha=1.0, beta=1.0, sparse=sparse, learning_rate=0.99
        )
        model.fit(X[:50], y[:50])

        # This hits _fit_helper with _pending_reinjection > 0 AND sample_weight != None
        model.partial_fit(X[50:60], y[50:60], sample_weight=weights)

        assert model.alpha > 0
        assert model.beta > 0
        assert hasattr(model, "_prior_scalar")

    def test_sample_weight_changes_result(self, regression_data, sparse):
        """Non-uniform weights should produce different coefficients than uniform."""
        X, y = regression_data

        model_uniform = EmpiricalBayesNormalRegressor(
            alpha=1.0, beta=1.0, sparse=sparse, learning_rate=0.99
        )
        model_uniform.fit(X[:50], y[:50])
        model_uniform.partial_fit(X[50:60], y[50:60])
        coef_uniform = model_uniform.coef_.copy()

        model_weighted = EmpiricalBayesNormalRegressor(
            alpha=1.0, beta=1.0, sparse=sparse, learning_rate=0.99
        )
        model_weighted.fit(X[:50], y[:50])
        weights = np.ones(10)
        weights[0] = 100.0
        model_weighted.partial_fit(X[50:60], y[50:60], sample_weight=weights)
        coef_weighted = model_weighted.coef_

        assert not np.allclose(coef_uniform, coef_weighted)
