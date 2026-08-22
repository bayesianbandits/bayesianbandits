"""Tests for Empirical Bayes estimators."""

import pickle
from unittest import mock

import numpy as np
import pytest
import scipy.sparse as sp
from numpy.testing import assert_allclose
from sklearn.base import clone

from bayesianbandits import (
    EmpiricalBayesDirichletClassifier,
    EmpiricalBayesGammaRegressor,
    EmpiricalBayesNormalRegressor,
)
from bayesianbandits._estimators import NormalRegressor
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

    @staticmethod
    def _dense_precision(model):
        """Full symmetric precision (dense cov_inv_ is upper-triangle-only)."""
        if model.sparse:
            return model.cov_inv_.toarray()
        upper = np.triu(model.cov_inv_)
        return upper + upper.T - np.diag(np.diag(upper))

    def test_partial_fit_keeps_coef_consistent_with_information(
        self, regression_data, sparse
    ):
        """After a MacKay step, Λ·coef_ must equal β·(Xᵀy): the posterior
        mean has to be re-solved against the rescaled precision."""
        X, y = regression_data
        model = EmpiricalBayesNormalRegressor(alpha=1.0, beta=1.0, sparse=sparse)
        model.fit(X[:50], y[:50])
        # fit() has already moved both, so the ratios that decide the
        # branch are relative to here, not to the constructor arguments.
        alpha_old, beta_old = model.alpha, model.beta
        model.partial_fit(X[50:], y[50:])

        ratio_alpha = model.alpha / alpha_old
        ratio_beta = model.beta / beta_old
        assert not np.isclose(ratio_alpha, ratio_beta)  # a real diagonal shift

        precision = self._dense_precision(model)
        np.testing.assert_allclose(
            precision @ model.coef_,
            model.beta * model._eff_XTy,
            rtol=1e-8,
            atol=1e-8,
        )

    def test_correct_precision_resolves_coef(self, regression_data, sparse):
        """Direct check of _correct_precision with a forced α/β change."""
        X, y = regression_data
        model = EmpiricalBayesNormalRegressor(
            alpha=1.0, beta=1.0, n_eb_iter=0, sparse=sparse
        )
        model.fit(X, y)

        alpha_old, beta_old = model.alpha, model.beta
        prec_old = self._dense_precision(model)
        coef_old = model.coef_.copy()
        prior_old = model._prior_scalar
        data_old = prec_old - prior_old * np.eye(prec_old.shape[0])

        model.alpha = alpha_old * 3.0
        model.beta = beta_old * 0.5
        model._correct_precision(alpha_old, beta_old)

        prec_new = self._dense_precision(model)
        data_new = prec_new - model._prior_scalar * np.eye(prec_new.shape[0])
        np.testing.assert_allclose(data_new, 0.5 * data_old, rtol=1e-10, atol=1e-12)
        np.testing.assert_allclose(model._prior_scalar, 3.0 * prior_old)

        eta_new = 0.5 * (prec_old @ coef_old)
        expected = np.linalg.solve(prec_new, eta_new)
        np.testing.assert_allclose(model.coef_, expected, rtol=1e-8, atol=1e-10)
        # The cached factor (rebuilt on demand) agrees with the raw matrix.
        np.testing.assert_allclose(
            model._precision_factor.solve(eta_new), expected, rtol=1e-8, atol=1e-10
        )

    def test_partial_fit_tracks_full_fit(self, sparse):
        """Chunked partial_fit from far-off α, β must land on the full fit.

        With ``learning_rate=1`` the online learner sees the same
        sufficient statistics as ``fit``, so its MacKay fixed point is
        the same one. Before ``_correct_precision`` re-solved ``coef_``
        it drifted (α off by 1.8x, β off by 0.5x, coef off by 0.3).
        """
        rng = np.random.default_rng(7)
        n, p = 2000, 10
        X = rng.standard_normal((n, p))
        w_true = rng.standard_normal(p)
        y = X @ w_true + 0.5 * rng.standard_normal(n)

        full = EmpiricalBayesNormalRegressor(alpha=1e4, beta=1e-3, sparse=sparse)
        full.fit(X, y)

        # n_eb_iter=1 keeps the first chunk (which routes to fit) from
        # converging on its own, so every chunk takes one MacKay step.
        online = EmpiricalBayesNormalRegressor(
            alpha=1e4, beta=1e-3, n_eb_iter=1, sparse=sparse
        )
        for start in range(0, n, 50):
            online.partial_fit(X[start : start + 50], y[start : start + 50])

        np.testing.assert_allclose(online.alpha, full.alpha, rtol=1e-2)
        np.testing.assert_allclose(online.beta, full.beta, rtol=1e-2)
        np.testing.assert_allclose(online.coef_, full.coef_, atol=1e-3)

    def test_correct_precision_pure_rescale(self, regression_data, sparse):
        """α and β moving by the same ratio is a pure rescale.

        ``diag_correction`` is then exactly zero: the whole precision
        shares one factor, so the cached factorization absorbs it and
        the mean is unchanged (``Λ`` and ``η`` scale together).
        """
        X, y = regression_data
        model = EmpiricalBayesNormalRegressor(
            alpha=1.0, beta=1.0, n_eb_iter=0, sparse=sparse
        )
        model.fit(X, y)
        factor_before = model._precision_factor  # populate the cache

        alpha_old, beta_old = model.alpha, model.beta
        prec_old = self._dense_precision(model)
        coef_old = model.coef_.copy()
        prior_old = model._prior_scalar

        # A power of two keeps both ratios exactly 4.0, so
        # diag_correction lands on 0.0 rather than near it.
        model.alpha = alpha_old * 4.0
        model.beta = beta_old * 4.0
        model._correct_precision(alpha_old, beta_old)

        np.testing.assert_allclose(
            self._dense_precision(model), 4.0 * prec_old, rtol=1e-12, atol=1e-12
        )
        np.testing.assert_allclose(model._prior_scalar, 4.0 * prior_old)
        # The mean is untouched -- no solve, and no drift either.
        np.testing.assert_array_equal(model.coef_, coef_old)
        # The factor was scaled in place, not dropped and rebuilt.
        assert "_precision_factor" in model.__dict__
        assert model._precision_factor is not factor_before
        np.testing.assert_allclose(
            model._precision_factor.solve(4.0 * (prec_old @ coef_old)),
            coef_old,
            rtol=1e-8,
            atol=1e-10,
        )

    def test_partial_fit_never_forces_a_factorization(self, regression_data, sparse):
        """``partial_fit`` must not build a factorization it will not use.

        The MacKay correction re-solves ``coef_`` against a diagonally
        shifted precision, but the next ``partial_fit`` factors its own
        updated precision and needs only the information vector -- so
        the solve is deferred rather than paid here.  Every
        factorization a ``partial_fit`` does need, ``_fit_helper``
        builds as a byproduct of its own solve and caches eagerly; a
        lazy rebuild means one was thrown away.
        """
        X, y = regression_data
        model = EmpiricalBayesNormalRegressor(alpha=1.0, beta=1.0, sparse=sparse)
        model.fit(X[:20], y[:20])

        cached = NormalRegressor.__dict__["_precision_factor"]
        original = cached.func
        rebuilds = []

        def counting(self):
            rebuilds.append(1)
            return original(self)

        cached.func = counting
        try:
            for start in range(20, 100, 20):
                model.partial_fit(X[start : start + 20], y[start : start + 20])
            assert rebuilds == []
            # Reading the mean is what pays for the deferred solve.
            model.coef_
            assert len(rebuilds) == 1
        finally:
            cached.func = original

    def test_reading_coef_between_updates_changes_nothing(
        self, regression_data, sparse
    ):
        """The deferred solve is invisible: predicting, sampling or
        decaying between updates must not move the trajectory."""
        X, y = regression_data

        def run(peek):
            model = EmpiricalBayesNormalRegressor(
                alpha=1.0, beta=1.0, sparse=sparse, random_state=0
            )
            model.fit(X[:20], y[:20])
            for start in range(20, 100, 20):
                model.partial_fit(X[start : start + 20], y[start : start + 20])
                if peek:
                    model.predict(X[:1])
            return model

        quiet, peeked = run(peek=False), run(peek=True)
        np.testing.assert_allclose(peeked.coef_, quiet.coef_, rtol=1e-12, atol=1e-14)
        np.testing.assert_allclose(peeked.alpha, quiet.alpha, rtol=1e-12)
        np.testing.assert_allclose(peeked.beta, quiet.beta, rtol=1e-12)
        np.testing.assert_allclose(
            self._dense_precision(peeked),
            self._dense_precision(quiet),
            rtol=1e-12,
            atol=1e-12,
        )

    def test_pending_coef_solve_survives_pickling(self, regression_data, sparse):
        """A pickle taken with the solve still pending restores a model
        that resolves to the same mean -- the factor is not pickled, so
        the information vector has to travel with it."""
        X, y = regression_data
        model = EmpiricalBayesNormalRegressor(alpha=1.0, beta=1.0, sparse=sparse)
        model.fit(X[:50], y[:50])
        model.partial_fit(X[50:], y[50:])
        assert "_pending_eta" in model.__dict__  # nothing has read coef_ yet

        restored = pickle.loads(pickle.dumps(model))
        np.testing.assert_allclose(restored.coef_, model.coef_, rtol=1e-8, atol=1e-10)

    def test_failed_partial_fit_keeps_the_pending_solve(self, regression_data, sparse):
        """A rejected update must not swallow the pending correction:
        ``coef_`` is still solvable, and still the corrected mean."""
        X, y = regression_data
        model = EmpiricalBayesNormalRegressor(alpha=1.0, beta=1.0, sparse=sparse)
        model.fit(X[:50], y[:50])
        model.partial_fit(X[50:], y[50:])
        assert "_pending_eta" in model.__dict__
        expected = model.__dict__["_pending_eta"].copy()

        with pytest.raises(ValueError):
            model.partial_fit(X[:10], y[:5])  # mismatched lengths

        assert "_pending_eta" in model.__dict__
        np.testing.assert_array_equal(model.__dict__["_pending_eta"], expected)
        precision = self._dense_precision(model)
        np.testing.assert_allclose(
            precision @ model.coef_, expected, rtol=1e-8, atol=1e-8
        )

    def test_coef_raises_before_fitting(self, sparse):
        """An unfitted model has no mean to report, deferred or not."""
        model = EmpiricalBayesNormalRegressor(alpha=1.0, beta=1.0, sparse=sparse)
        assert not hasattr(model, "coef_")
        with pytest.raises(AttributeError, match="coef_"):
            model.coef_


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

    def test_fit_sample_weight_n_eb_iter_zero(self, regression_data, sparse):
        """fit() with sample_weight and n_eb_iter=0 must respect weights."""
        X, y = regression_data

        model_uniform = EmpiricalBayesNormalRegressor(
            alpha=1.0, beta=1.0, n_eb_iter=0, sparse=sparse, random_state=42
        )
        model_uniform.fit(X, y)

        weights = np.ones(len(y))
        weights[0] = 100.0
        model_weighted = EmpiricalBayesNormalRegressor(
            alpha=1.0, beta=1.0, n_eb_iter=0, sparse=sparse, random_state=42
        )
        model_weighted.fit(X, y, sample_weight=weights)

        assert not np.allclose(model_uniform.coef_, model_weighted.coef_)


@pytest.mark.parametrize("sparse", [True, False])
class TestPriorScalar:
    """``_prior_scalar`` is the prior's real contribution to Λ's diagonal.

    Under forgetting ``_fit_helper`` scales the prior by
    ``learning_rate ** n_samples``, so ``alpha`` itself overstates it.
    MacKay's ``gamma = p - prior_scalar * tr(Λ⁻¹)`` is sensitive to
    that: every unobserved feature contributes exactly
    ``1 / prior_scalar`` to the trace, so overstating the scalar pushes
    the sum past ``p`` and drives gamma negative.
    """

    def test_prior_scalar_matches_precision_diagonal(self, sparse):
        """The tracked scalar equals the prior floor of Λ's diagonal."""
        rng = np.random.default_rng(0)
        X = rng.standard_normal((200, 8))
        y = X @ rng.standard_normal(8) + 0.1 * rng.standard_normal(200)

        model = EmpiricalBayesNormalRegressor(
            alpha=1.0, beta=1.0, learning_rate=0.99, sparse=sparse
        )
        model.fit(X, y)

        diagonal = model.cov_inv_.diagonal() if sparse else np.diag(model.cov_inv_)
        assert model._prior_scalar <= np.min(diagonal) * (1 + 1e-9)
        assert model._prior_scalar == pytest.approx(
            model.learning_rate ** X.shape[0] * model.alpha
        )

    def test_gamma_stays_in_range_when_features_are_unobserved(self, sparse):
        """Wide data with all-zero columns keeps gamma in (0, min(n, p)].

        With ``alpha`` in place of ``prior_scalar`` the trace overshoots
        ``p``, gamma goes negative, and the clip plus the beta/alpha
        guardrail silently reject every update -- EB becomes a no-op
        exactly where it matters most.
        """
        rng = np.random.default_rng(0)
        n_samples, n_features = 30, 400
        X = np.zeros((n_samples, n_features))
        # Only the first 20 features are ever observed.
        X[:, :20] = rng.standard_normal((n_samples, 20))
        y = X[:, :20] @ rng.standard_normal(20) + 0.1 * rng.standard_normal(n_samples)

        model = EmpiricalBayesNormalRegressor(
            alpha=1.0, beta=1.0, learning_rate=0.99999, sparse=sparse
        )
        model.fit(X, y)

        gamma = n_features - model._prior_scalar * model._precision_factor.trace_inv()
        assert 0 < gamma <= min(n_samples, n_features)

    def test_eb_is_not_a_noop_on_wide_data(self, sparse):
        """Hyperparameters actually move when most features are unobserved."""
        rng = np.random.default_rng(0)
        X = np.zeros((30, 400))
        X[:, :20] = rng.standard_normal((30, 20))
        y = X[:, :20] @ rng.standard_normal(20) + 0.1 * rng.standard_normal(30)

        model = EmpiricalBayesNormalRegressor(
            alpha=1.0, beta=1.0, learning_rate=0.99999, sparse=sparse
        )
        model.fit(X, y)

        assert model.alpha != 1.0 or model.beta != 1.0
        assert np.isfinite(model.log_evidence_)

    def test_partial_fit_updates_hyperparameters_on_wide_data(self, sparse):
        """The online MacKay step is not a no-op either."""
        rng = np.random.default_rng(0)
        X = np.zeros((30, 400))
        X[:, :20] = rng.standard_normal((30, 20))
        y = X[:, :20] @ rng.standard_normal(20) + 0.1 * rng.standard_normal(30)

        model = EmpiricalBayesNormalRegressor(
            alpha=1.0, beta=1.0, learning_rate=0.99999, sparse=sparse
        )
        model.fit(X[:20], y[:20])
        before = (model.alpha, model.beta)
        model.partial_fit(X[20:], y[20:])
        assert (model.alpha, model.beta) != before


class TestShiftDiagonal:
    """``_shift_diagonal`` adds ``shift * I`` to a sparse precision in place."""

    @staticmethod
    def _model():
        return EmpiricalBayesNormalRegressor(alpha=1.0, beta=1.0, sparse=True)

    def test_zero_shift_is_a_no_op(self):
        model = self._model()
        cov_inv = sp.csc_array(sp.eye_array(4, format="csc"))
        before = cov_inv.data.copy()
        model._shift_diagonal(cov_inv, 0.0)
        assert np.array_equal(cov_inv.data, before)
        assert "_diag_pos" not in model.__dict__

    def test_missing_diagonal_entry_falls_back_to_setdiag(self):
        model = self._model()
        cov_inv = sp.csc_array(sp.diags_array([1.0, 0.0, 1.0], format="csc"))
        cov_inv.eliminate_zeros()
        assert cov_inv.nnz == 2
        model._shift_diagonal(cov_inv, 0.5)
        assert_allclose(cov_inv.diagonal(), [1.5, 0.5, 1.5])
        assert "_diag_pos" not in model.__dict__


@pytest.mark.parametrize("sparse", [True, False])
class TestEBReporting:
    """EB should expose the objective it maximizes, and say when it stops.

    ``_eb_mackay_step_online`` used to discard the log evidence it had
    already computed, so ``log_evidence_`` went stale after ``fit`` --
    unlike the Dirichlet and Gamma estimators, which refresh theirs on
    every ``partial_fit`` -- and a guardrail rejection left no trace at
    all.
    """

    @staticmethod
    def _data(seed=0):
        rng = np.random.default_rng(seed)
        X = rng.standard_normal((60, 5))
        y = X @ rng.standard_normal(5) + 0.1 * rng.standard_normal(60)
        return X, y

    def test_log_evidence_is_refreshed_by_partial_fit(self, sparse):
        X, y = self._data()
        model = EmpiricalBayesNormalRegressor(alpha=1.0, beta=1.0, sparse=sparse)
        model.fit(X[:30], y[:30])
        after_fit = model.log_evidence_

        model.partial_fit(X[30:], y[30:])
        assert np.isfinite(model.log_evidence_)
        assert model.log_evidence_ != after_fit

    def test_log_evidence_matches_the_online_update(self, sparse):
        """The recorded value is the one the MacKay step computed."""
        from bayesianbandits._empirical_bayes import mackay_update_normal_online

        X, y = self._data()
        model = EmpiricalBayesNormalRegressor(alpha=1.0, beta=1.0, sparse=sparse)
        model.fit(X[:30], y[:30])

        captured = {}
        real = mackay_update_normal_online

        def spy(*args, **kwargs):
            result = real(*args, **kwargs)
            captured["log_evidence"] = result.log_evidence
            return result

        with mock.patch(
            "bayesianbandits._eb_estimators.mackay_update_normal_online", spy
        ):
            model.partial_fit(X[30:], y[30:])

        assert model.log_evidence_ == captured["log_evidence"]

    def test_healthy_fit_rejects_nothing(self, sparse):
        X, y = self._data()
        model = EmpiricalBayesNormalRegressor(alpha=1.0, beta=1.0, sparse=sparse)
        model.fit(X, y)
        assert model.eb_updates_rejected_ == 0

        model.partial_fit(X[:5], y[:5])
        assert model.eb_updates_rejected_ == 0

    def test_rejected_updates_are_counted_and_leave_hyperparameters_alone(self, sparse):
        """A guardrail rejection is visible instead of silent."""
        from bayesianbandits._empirical_bayes import MacKayUpdate

        X, y = self._data()
        model = EmpiricalBayesNormalRegressor(alpha=1.0, beta=1.0, sparse=sparse)
        model.fit(X[:30], y[:30])
        alpha, beta = model.alpha, model.beta
        before = model.eb_updates_rejected_

        rejection = MacKayUpdate(alpha, beta, -123.5, True)
        with mock.patch(
            "bayesianbandits._eb_estimators.mackay_update_normal_online",
            return_value=rejection,
        ):
            model.partial_fit(X[30:40], y[30:40])
            model.partial_fit(X[40:50], y[40:50])

        assert model.eb_updates_rejected_ == before + 2
        assert model.alpha == alpha
        assert model.beta == beta
        assert model.log_evidence_ == -123.5

    def test_fit_resets_the_rejection_count(self, sparse):
        X, y = self._data()
        model = EmpiricalBayesNormalRegressor(alpha=1.0, beta=1.0, sparse=sparse)
        model.fit(X[:30], y[:30])
        model.eb_updates_rejected_ = 7
        model.fit(X, y)
        assert model.eb_updates_rejected_ == 0

    def test_reporting_state_exists_without_an_explicit_fit(self, sparse):
        """sample() then partial_fit skips fit(), so the state starts there."""
        X, y = self._data()
        model = EmpiricalBayesNormalRegressor(alpha=1.0, beta=1.0, sparse=sparse)
        model.sample(X[:1])  # initializes the prior without fitting
        model.partial_fit(X, y)

        assert model.eb_updates_rejected_ >= 0
        assert np.isfinite(model.log_evidence_)


# ---------------------------------------------------------------------------
# EmpiricalBayesDirichletClassifier
# ---------------------------------------------------------------------------


@pytest.fixture
def dirichlet_data():
    """Synthetic Dirichlet data: 5 groups, class 0 is more common."""
    rng = np.random.default_rng(42)
    X_parts = []
    y_parts = []
    for group_id in range(1, 6):
        n = 30
        # Class 0 is ~70%, class 1 is ~30%
        p = rng.dirichlet([7.0, 3.0])
        ys = rng.choice([0, 1], size=n, p=p)
        X_parts.append(np.full((n, 1), group_id))
        y_parts.append(ys)
    X = np.vstack(X_parts)
    y = np.concatenate(y_parts)
    return X, y


class TestEBDirichletClassifier:
    def test_fit_smoke(self, dirichlet_data):
        """Basic fit produces EB attributes and valid predictions."""
        X, y = dirichlet_data
        model = EmpiricalBayesDirichletClassifier({0: 1.0, 1: 1.0}, random_state=42)
        model.fit(X, y)

        assert hasattr(model, "log_evidence_")
        assert hasattr(model, "n_eb_iterations_")
        assert hasattr(model, "eb_converged_")
        assert np.isfinite(model.log_evidence_)

        proba = model.predict_proba(X[:1])
        assert proba.shape == (1, 2)
        np.testing.assert_allclose(proba.sum(), 1.0)

    def test_convergence_flag(self, dirichlet_data):
        """Model converges within tolerance."""
        X, y = dirichlet_data
        model = EmpiricalBayesDirichletClassifier(
            {0: 1.0, 1: 1.0}, n_eb_iter=1000, eb_tol=1e-4, random_state=42
        )
        model.fit(X, y)
        assert model.eb_converged_

    def test_partial_fit_updates_prior(self, dirichlet_data):
        """partial_fit performs one Minka step, changing prior_."""
        X, y = dirichlet_data
        model = EmpiricalBayesDirichletClassifier({0: 1.0, 1: 1.0}, random_state=42)
        model.fit(X[:30], y[:30])
        prior_after_fit = model.prior_.copy()

        model.partial_fit(X[30:60], y[30:60])
        assert not np.allclose(model.prior_, prior_after_fit), (
            "partial_fit did not update the prior"
        )

    def test_partial_fit_corrects_posteriors(self, dirichlet_data):
        """Posteriors are adjusted by new_prior - old_prior."""
        X, y = dirichlet_data
        model = EmpiricalBayesDirichletClassifier({0: 1.0, 1: 1.0}, random_state=42)
        model.fit(X, y)

        # Record current state
        old_prior = model.prior_.copy()
        old_posteriors = {k: v.copy() for k, v in model.known_alphas_.items()}

        model.partial_fit(X[:30], y[:30])

        # The posterior correction: new_post = old_post + (new_prior - old_prior) + new_counts
        # At minimum, the prior shift should be reflected
        prior_delta = model.prior_ - old_prior
        if np.any(np.abs(prior_delta) > 1e-10):
            # For groups that didn't receive new data, the shift should match
            # For groups that did, additional counts are added
            for key in old_posteriors:
                diff = model.known_alphas_[key] - old_posteriors[key]
                # diff should be at least prior_delta (plus any new counts)
                # Just check it's not zero — correction happened
                assert np.any(np.abs(diff) > 1e-10)

    def test_n_eb_iter_zero(self, dirichlet_data):
        """n_eb_iter=0 disables EB; prior stays at initial values."""
        X, y = dirichlet_data
        model = EmpiricalBayesDirichletClassifier(
            {0: 1.0, 1: 1.0}, n_eb_iter=0, random_state=42
        )
        model.fit(X, y)

        assert model.log_evidence_ == -np.inf
        assert model.n_eb_iterations_ == 0
        assert not model.eb_converged_
        # Prior should be unchanged
        np.testing.assert_array_equal(model.prior_, np.array([1.0, 1.0]))

    def test_decay_stabilized_forgetting(self):
        """After many decays, alphas converge to prior, not zero."""
        # Use balanced data so the EB prior doesn't go to extremes
        rng = np.random.default_rng(99)
        X_parts, y_parts = [], []
        for gid in range(1, 4):
            ys = rng.choice([0, 1], size=30, p=[0.5, 0.5])
            X_parts.append(np.full((30, 1), gid))
            y_parts.append(ys)
        X = np.vstack(X_parts)
        y = np.concatenate(y_parts)

        model = EmpiricalBayesDirichletClassifier(
            {0: 1.0, 1: 1.0}, learning_rate=0.9, random_state=42
        )
        model.fit(X, y)
        prior_after_fit = model.prior_.copy()

        # Decay ALL groups (pass unique group IDs)
        group_ids = np.array([[1], [2], [3]])
        for _ in range(200):
            model.decay(group_ids)

        # Alphas should converge to prior, not zero
        for key in list(model.known_alphas_.keys()):
            np.testing.assert_allclose(
                model.known_alphas_[key],
                prior_after_fit,
                atol=1e-2,
                rtol=0.05,
            )
            # Prior components should be reasonable for balanced data
            assert np.all(model.known_alphas_[key] > 0.01), (
                "Alphas decayed to near-zero despite stabilized forgetting"
            )

    def test_batch_partial_fit_matches_sequential(self):
        """Batch and sequential partial_fit converge to same base measure.

        Phase 1 diverges the models: one gets 5 groups in a single
        partial_fit (1 Minka step), the other gets them one group at a
        time (5 Minka steps). Phase 2 feeds both models the same
        sequential data, and the base measures converge.
        """
        rng = np.random.default_rng(42)
        true_alpha = np.array([3.0, 1.0])
        lr = 0.9

        batch_model = EmpiricalBayesDirichletClassifier(
            {0: 1.0, 1: 1.0}, learning_rate=lr, random_state=42
        )
        seq_model = EmpiricalBayesDirichletClassifier(
            {0: 1.0, 1: 1.0}, learning_rate=lr, random_state=42
        )

        # Phase 1: diverge
        all_X, all_y = [], []
        for g in range(5):
            theta = rng.dirichlet(true_alpha)
            obs = rng.choice([0, 1], size=20, p=theta)
            X_g = np.full((20, 1), g)
            all_X.append(X_g)
            all_y.append(obs)
            seq_model.partial_fit(X_g, obs)
        batch_model.partial_fit(np.vstack(all_X), np.concatenate(all_y))

        # Phase 2: converge (same sequential data for both)
        for g in range(5, 50):
            theta = rng.dirichlet(true_alpha)
            obs = rng.choice([0, 1], size=20, p=theta)
            X_g = np.full((20, 1), g)
            batch_model.partial_fit(X_g, obs)
            seq_model.partial_fit(X_g, obs)

        batch_m = batch_model.prior_ / batch_model.prior_.sum()
        seq_m = seq_model.prior_ / seq_model.prior_.sum()
        true_m = true_alpha / true_alpha.sum()

        # Both should recover the true base measure
        np.testing.assert_allclose(
            batch_m,
            true_m,
            atol=0.05,
            err_msg="Batch model did not recover the true base measure",
        )
        np.testing.assert_allclose(
            seq_m,
            true_m,
            atol=0.05,
            err_msg="Sequential model did not recover the true base measure",
        )

        # And agree with each other
        np.testing.assert_allclose(
            batch_m,
            seq_m,
            atol=0.01,
            err_msg="Base measures did not converge after shared updates",
        )

    def test_partial_fit_no_negative_counts_under_decay(self):
        """Stabilized forgetting prevents negative counts with lr < 1."""
        rng = np.random.default_rng(42)
        model = EmpiricalBayesDirichletClassifier(
            {0: 1.0, 1: 1.0, 2: 1.0}, learning_rate=0.8, random_state=0
        )

        for i in range(200):
            g = rng.integers(0, 10)
            cls = rng.choice([0, 1, 2], p=[0.7, 0.2, 0.1])
            model.partial_fit(np.array([[g]]), np.array([cls]))

            if i % 3 == 0:
                model.decay(np.array([[rng.integers(0, 10)]]))

            for key, val in model.known_alphas_.items():
                if hasattr(val, "__len__"):
                    counts = val - model.prior_
                    assert np.all(counts >= -1e-10), (
                        f"Negative counts at step {i}, group {key}: {counts}"
                    )

    def test_multiple_groups(self, dirichlet_data):
        """EB pools information across groups."""
        X, y = dirichlet_data
        model = EmpiricalBayesDirichletClassifier({0: 1.0, 1: 1.0}, random_state=42)
        model.fit(X, y)

        # Since class 0 is more common across all groups, the
        # EB-tuned prior should favor class 0
        assert model.alphas[0] > model.alphas[1], (
            f"Expected alpha[0] > alpha[1] for skewed data, got {model.alphas}"
        )

    def test_get_set_params(self):
        """sklearn get_params/set_params contract."""
        model = EmpiricalBayesDirichletClassifier(
            {0: 1.0, 1: 1.0}, n_eb_iter=5, eb_tol=1e-3
        )
        params = model.get_params()
        assert params["n_eb_iter"] == 5
        assert params["eb_tol"] == 1e-3

        model.set_params(n_eb_iter=20)
        assert model.n_eb_iter == 20

    def test_clone(self, dirichlet_data):
        """sklearn clone produces equivalent estimator."""
        X, y = dirichlet_data
        model = EmpiricalBayesDirichletClassifier({0: 1.0, 1: 1.0}, random_state=42)
        model.fit(X, y)

        cloned = clone(model)
        assert cloned.n_eb_iter == model.n_eb_iter
        assert not hasattr(cloned, "log_evidence_")

    def test_pickle_roundtrip(self, dirichlet_data):
        """Fitted model survives pickle/unpickle."""
        X, y = dirichlet_data
        model = EmpiricalBayesDirichletClassifier({0: 1.0, 1: 1.0}, random_state=42)
        model.fit(X, y)

        data = pickle.dumps(model)
        loaded = pickle.loads(data)

        np.testing.assert_array_equal(
            model.predict_proba(X[:5]),
            loaded.predict_proba(X[:5]),
        )

    def test_sample_weight_doubles_equivalent_to_duplicate(self, dirichlet_data):
        """Weight=2 on each sample produces same posteriors as duplicating data."""
        X, y = dirichlet_data

        model_dup = EmpiricalBayesDirichletClassifier({0: 1.0, 1: 1.0}, random_state=42)
        model_dup.fit(np.vstack([X, X]), np.concatenate([y, y]))

        model_wt = EmpiricalBayesDirichletClassifier({0: 1.0, 1: 1.0}, random_state=42)
        weights = 2.0 * np.ones(X.shape[0])
        model_wt.fit(X, y, sample_weight=weights)

        for key in model_dup.known_alphas_:
            np.testing.assert_allclose(
                model_wt.known_alphas_[key],
                model_dup.known_alphas_[key],
                rtol=1e-10,
            )

    def test_sample_weight_shape_mismatch_raises(self, dirichlet_data):
        """Mismatched sample_weight shape raises ValueError."""
        X, y = dirichlet_data
        weights = np.ones(X.shape[0] + 5, dtype=np.float64)
        model = EmpiricalBayesDirichletClassifier({0: 1.0, 1: 1.0}, random_state=42)
        with pytest.raises(ValueError, match="sample_weight"):
            model.fit(X, y, sample_weight=weights)

    def test_decay_before_fit_preserves_prior(self):
        """Decay on unfitted model: stabilized forgetting gives lr*α + (1-lr)*α = α."""
        lr = 0.9
        model = EmpiricalBayesDirichletClassifier(
            {0: 2.0, 1: 3.0}, learning_rate=lr, random_state=42
        )
        model.decay(np.array([[1], [2]]))

        expected = np.array([2.0, 3.0])
        for key in [1, 2]:
            np.testing.assert_allclose(
                model.known_alphas_[key],
                expected,
                rtol=1e-12,
                err_msg="Stabilized decay on fresh prior should be identity",
            )


class TestEBGammaRegressor:
    """Tests for EmpiricalBayesGammaRegressor."""

    @pytest.fixture
    def gamma_poisson_data(self):
        """Synthetic Gamma-Poisson data: 5 groups with rates drawn from Gamma(3, 1)."""
        rng = np.random.default_rng(42)
        true_alpha, true_beta = 3.0, 1.0
        X_parts, y_parts = [], []
        for group_id in range(1, 6):
            rate = rng.gamma(true_alpha, 1.0 / true_beta)
            counts = rng.poisson(rate, size=30)
            X_parts.append(np.full((30, 1), group_id))
            y_parts.append(counts)
        X = np.vstack(X_parts)
        y = np.concatenate(y_parts)
        return X, y

    def test_fit_smoke(self, gamma_poisson_data):
        """Basic fit produces valid attributes and predictions."""
        X, y = gamma_poisson_data
        model = EmpiricalBayesGammaRegressor(alpha=1.0, beta=1.0, random_state=42)
        model.fit(X, y)

        assert hasattr(model, "log_evidence_")
        assert np.isfinite(model.log_evidence_)
        assert hasattr(model, "n_eb_iterations_")
        assert hasattr(model, "eb_converged_")

        preds = model.predict(X[:5])
        assert preds.shape == (5,)
        assert np.all(preds > 0)

        samples = model.sample(X[:5], size=3)
        assert samples.shape == (3, 5)
        assert np.all(samples > 0)

    def test_hyperparams_change(self, gamma_poisson_data):
        """EB tuning changes alpha and/or beta from initial values."""
        X, y = gamma_poisson_data
        model = EmpiricalBayesGammaRegressor(alpha=1.0, beta=1.0, random_state=42)
        model.fit(X, y)

        assert model.alpha != 1.0 or model.beta != 1.0, (
            "EB should change at least one hyperparameter"
        )

    def test_convergence_flag(self, gamma_poisson_data):
        """With enough iterations, EB converges."""
        X, y = gamma_poisson_data
        model = EmpiricalBayesGammaRegressor(
            alpha=1.0, beta=1.0, n_eb_iter=1000, eb_tol=1e-4, random_state=42
        )
        model.fit(X, y)
        assert model.eb_converged_

    def test_evidence_monotonicity(self, gamma_poisson_data):
        """Repeatedly fitting with n_eb_iter=1, evidence is non-decreasing."""
        X, y = gamma_poisson_data
        evidences = []
        alpha, beta = 1.0, 1.0

        for _ in range(10):
            model = EmpiricalBayesGammaRegressor(
                alpha=alpha, beta=beta, n_eb_iter=1, random_state=42
            )
            model.fit(X, y)
            evidences.append(model.log_evidence_)
            alpha = model.alpha
            beta = model.beta

        for i in range(1, len(evidences)):
            assert evidences[i] >= evidences[i - 1] - 1e-6, (
                f"Evidence decreased: {evidences[i]} < {evidences[i - 1]}"
            )

    def test_partial_fit_updates_prior(self, gamma_poisson_data):
        """partial_fit changes the prior via online EB."""
        X, y = gamma_poisson_data
        model = EmpiricalBayesGammaRegressor(alpha=1.0, beta=1.0, random_state=42)
        model.fit(X[:60], y[:60])
        prior_before = model.prior_.copy()

        model.partial_fit(X[60:], y[60:])
        assert not np.allclose(model.prior_, prior_before), (
            "Prior should change after partial_fit"
        )

    def test_partial_fit_corrects_posteriors(self, gamma_poisson_data):
        """partial_fit shifts unaffected group posteriors by exactly prior_delta."""
        X, y = gamma_poisson_data
        # Fit on groups 1,2 (first 60 rows)
        model = EmpiricalBayesGammaRegressor(alpha=1.0, beta=1.0, random_state=42)
        model.fit(X[:60], y[:60])

        old_prior = model.prior_.copy()
        old_posteriors = {k: v.copy() for k, v in model.coef_.items()}

        # partial_fit on group 3 only (rows 60:90)
        model.partial_fit(X[60:90], y[60:90])

        prior_delta = model.prior_ - old_prior
        # Groups 1 and 2 were not in the batch, so their posteriors
        # should shift by exactly prior_delta (the EB correction).
        for key in [1, 2]:
            diff = model.coef_[key] - old_posteriors[key]
            np.testing.assert_allclose(
                diff,
                prior_delta,
                atol=1e-12,
                err_msg=f"Group {key} (not in batch) should shift by exactly prior_delta",
            )

    def test_n_eb_iter_zero(self, gamma_poisson_data):
        """n_eb_iter=0 disables EB; prior stays at initial values."""
        X, y = gamma_poisson_data
        model = EmpiricalBayesGammaRegressor(
            alpha=1.0, beta=1.0, n_eb_iter=0, random_state=42
        )
        model.fit(X, y)

        assert model.log_evidence_ == -np.inf
        assert model.n_eb_iterations_ == 0
        assert not model.eb_converged_
        np.testing.assert_array_equal(model.prior_, np.array([1.0, 1.0]))

    def test_decay_stabilized_forgetting(self):
        """After many decays, coef_ converges to prior, not zero."""
        rng = np.random.default_rng(99)
        X_parts, y_parts = [], []
        for gid in range(1, 4):
            counts = rng.poisson(3.0, size=30)
            X_parts.append(np.full((30, 1), gid))
            y_parts.append(counts)
        X = np.vstack(X_parts)
        y = np.concatenate(y_parts)

        model = EmpiricalBayesGammaRegressor(
            alpha=1.0, beta=1.0, learning_rate=0.9, random_state=42
        )
        model.fit(X, y)
        prior_after_fit = model.prior_.copy()

        group_ids = np.array([[1], [2], [3]])
        for _ in range(200):
            model.decay(group_ids)

        for key in list(model.coef_.keys()):
            np.testing.assert_allclose(
                model.coef_[key],
                prior_after_fit,
                atol=1e-2,
                rtol=0.05,
            )
            assert np.all(model.coef_[key] > 0.01), (
                "Params decayed to near-zero despite stabilized forgetting"
            )

    def test_decay_before_fit_preserves_prior(self):
        """Decay on unfitted model: stabilized forgetting gives lr*p + (1-lr)*p = p."""
        lr = 0.9
        model = EmpiricalBayesGammaRegressor(
            alpha=2.0, beta=3.0, learning_rate=lr, random_state=42
        )
        model.decay(np.array([[1], [2]]))

        expected = np.array([2.0, 3.0])
        for key in [1, 2]:
            np.testing.assert_allclose(
                model.coef_[key],
                expected,
                rtol=1e-12,
                err_msg="Stabilized decay on fresh prior should be identity",
            )

    def test_prior_recovery_large_sample(self):
        """With many groups, EB recovers the true Gamma prior parameters."""
        rng = np.random.default_rng(42)
        true_alpha, true_beta = 3.0, 1.0

        X_parts, y_parts = [], []
        for group_id in range(1, 51):
            rate = rng.gamma(true_alpha, 1.0 / true_beta)
            counts = rng.poisson(rate, size=100)
            X_parts.append(np.full((100, 1), group_id))
            y_parts.append(counts)
        X = np.vstack(X_parts)
        y = np.concatenate(y_parts)

        model = EmpiricalBayesGammaRegressor(alpha=1.0, beta=1.0, random_state=42)
        model.fit(X, y)

        recovered_mean = model.alpha / model.beta
        true_mean = true_alpha / true_beta
        assert abs(recovered_mean - true_mean) < 0.6, (
            f"Prior mean {recovered_mean:.3f} too far from true {true_mean:.1f}"
        )

        recovered_var = model.alpha / model.beta**2
        true_var = true_alpha / true_beta**2
        assert abs(recovered_var - true_var) < 1.5, (
            f"Prior variance {recovered_var:.3f} too far from true {true_var:.1f}"
        )

    def test_prior_recovery_misspecified_init(self):
        """EB recovers even from a badly misspecified initial prior."""
        rng = np.random.default_rng(42)
        true_alpha, true_beta = 3.0, 1.0

        X_parts, y_parts = [], []
        for group_id in range(1, 51):
            rate = rng.gamma(true_alpha, 1.0 / true_beta)
            counts = rng.poisson(rate, size=100)
            X_parts.append(np.full((100, 1), group_id))
            y_parts.append(counts)
        X = np.vstack(X_parts)
        y = np.concatenate(y_parts)

        model_good = EmpiricalBayesGammaRegressor(alpha=1.0, beta=1.0, random_state=42)
        model_good.fit(X, y)

        model_bad = EmpiricalBayesGammaRegressor(
            alpha=100.0, beta=0.01, random_state=42
        )
        model_bad.fit(X, y)

        np.testing.assert_allclose(
            model_bad.alpha / model_bad.beta,
            model_good.alpha / model_good.beta,
            rtol=0.01,
            err_msg="Misspecified init should converge to same prior mean",
        )

    def test_partial_fit_no_negative_counts_under_decay(self):
        """Stabilized forgetting prevents negative counts with lr < 1."""
        rng = np.random.default_rng(42)
        model = EmpiricalBayesGammaRegressor(
            alpha=1.0, beta=1.0, learning_rate=0.8, random_state=0
        )

        for i in range(200):
            g = rng.integers(0, 10)
            count = rng.poisson(3.0)
            model.partial_fit(np.array([[g]]), np.array([count]))

            if i % 3 == 0:
                model.decay(np.array([[rng.integers(0, 10)]]))

            for key, val in model.coef_.items():
                counts = val - model.prior_
                assert np.all(counts >= -1e-10), (
                    f"Negative counts at step {i}, group {key}: {counts}"
                )

    def test_sample_weight_doubles_equivalent_to_duplicate(self, gamma_poisson_data):
        """Weight=2 on each sample produces same posteriors as duplicating data."""
        X, y = gamma_poisson_data

        model_dup = EmpiricalBayesGammaRegressor(alpha=1.0, beta=1.0, random_state=42)
        model_dup.fit(np.vstack([X, X]), np.concatenate([y, y]))

        model_wt = EmpiricalBayesGammaRegressor(alpha=1.0, beta=1.0, random_state=42)
        weights = 2.0 * np.ones(X.shape[0])
        model_wt.fit(X, y, sample_weight=weights)

        for key in model_dup.coef_:
            np.testing.assert_allclose(
                model_wt.coef_[key],
                model_dup.coef_[key],
                rtol=1e-10,
            )

    def test_sample_weight_shape_mismatch_raises(self, gamma_poisson_data):
        """Mismatched sample_weight shape raises ValueError."""
        X, y = gamma_poisson_data
        weights = np.ones(X.shape[0] + 5, dtype=np.float64)
        model = EmpiricalBayesGammaRegressor(alpha=1.0, beta=1.0, random_state=42)
        with pytest.raises(ValueError, match="sample_weight"):
            model.fit(X, y, sample_weight=weights)

    def test_get_set_params(self):
        """sklearn get_params/set_params contract."""
        model = EmpiricalBayesGammaRegressor(
            alpha=1.0, beta=1.0, n_eb_iter=5, eb_tol=1e-3
        )
        params = model.get_params()
        assert params["n_eb_iter"] == 5
        assert params["eb_tol"] == 1e-3

        model.set_params(n_eb_iter=20)
        assert model.n_eb_iter == 20

    def test_clone(self, gamma_poisson_data):
        """sklearn clone produces equivalent estimator."""
        X, y = gamma_poisson_data
        model = EmpiricalBayesGammaRegressor(alpha=1.0, beta=1.0, random_state=42)
        model.fit(X, y)

        cloned = clone(model)
        assert cloned.n_eb_iter == model.n_eb_iter
        assert not hasattr(cloned, "log_evidence_")

    def test_pickle_roundtrip(self, gamma_poisson_data):
        """Fitted model survives pickle/unpickle."""
        X, y = gamma_poisson_data
        model = EmpiricalBayesGammaRegressor(alpha=1.0, beta=1.0, random_state=42)
        model.fit(X, y)

        data = pickle.dumps(model)
        loaded = pickle.loads(data)

        np.testing.assert_array_equal(
            model.predict(X[:5]),
            loaded.predict(X[:5]),
        )

    def test_single_em_step_matches_hand_computation(self):
        """One EM iteration on a 2-group example matches hand-derived values.

        Prior: Gamma(alpha=2, beta=1) (rate parameterization).
        Group 1: count=3, exposure=2 -> posterior [5, 3].
        Group 2: count=1, exposure=1 -> posterior [3, 2].

        E-step:
            E[lambda_1] = (3+2)/(2+1) = 5/3
            E[lambda_2] = (1+2)/(1+1) = 3/2
            E[log lambda_1] = psi(5) - log(3)
            E[log lambda_2] = psi(3) - log(2)

        M-step (beta first, then alpha via Newton):
            mean_E_lambda = (5/3 + 3/2) / 2 = 19/12
            beta_new = alpha_old / mean_E_lambda = 2 / (19/12) = 24/19
            alpha_new = 3.705... (5 Newton iterations converged)
        """
        from scipy.special import digamma

        from bayesianbandits._empirical_bayes import negbin_update_gamma_poisson

        prior = np.array([2.0, 1.0])
        posterior_params = {
            "g1": np.array([5.0, 3.0]),
            "g2": np.array([3.0, 2.0]),
        }

        result, _, _ = negbin_update_gamma_poisson(posterior_params, prior, n_iter=1)

        # Beta: closed-form given old alpha
        # mean_E_lambda = ((3+2)/(2+1) + (1+2)/(1+1)) / 2 = 19/12
        expected_beta = 2.0 / (19.0 / 12.0)  # = 24/19
        np.testing.assert_allclose(result[1], expected_beta, atol=1e-14)

        # Alpha: verify E-step feeds correct target into Newton
        E_log_lambda = np.array(
            [
                float(digamma(5)) - np.log(3),
                float(digamma(3)) - np.log(2),
            ]
        )
        mean_E_log_lambda = E_log_lambda.mean()
        target = np.log(19.0 / 12.0) - mean_E_log_lambda
        # At convergence: log(alpha) - psi(alpha) = target
        residual = abs(np.log(result[0]) - float(digamma(result[0])) - target)
        assert residual < 1e-10, (
            f"Newton did not converge: log(a)-psi(a)={np.log(result[0]) - float(digamma(result[0])):.10f}, "
            f"target={target:.10f}, residual={residual:.2e}"
        )


class TestFailedUpdateLeavesEstimatorIntact:
    """A ``partial_fit`` that raises must not have moved the posterior."""

    @pytest.mark.parametrize("cls", [NormalRegressor, EmpiricalBayesNormalRegressor])
    @pytest.mark.parametrize("sparse", [False, True])
    def test_partial_fit_that_raises_changes_nothing(self, cls, sparse) -> None:
        rng = np.random.default_rng(0)
        X = rng.standard_normal((30, 4))
        y = rng.standard_normal(30)
        X_fit = sp.csc_array(X) if sparse else X

        est = cls(alpha=1.0, beta=1.0, learning_rate=0.9, sparse=sparse)
        est.partial_fit(X_fit[:15], y[:15])

        def _dense(matrix):
            return np.asarray(
                matrix.todense() if sparse else matrix, dtype=np.float64
            ).copy()

        prec_before = _dense(est.cov_inv_)
        coef_before = est.coef_.copy()

        if sparse:
            # Inherited, so patching it on the base covers both estimators.
            failure = mock.patch.object(
                NormalRegressor,
                "_sparse_factor",
                side_effect=np.linalg.LinAlgError("boom"),
            )
        else:
            module = (
                "bayesianbandits._eb_estimators"
                if cls is EmpiricalBayesNormalRegressor
                else "bayesianbandits._estimators"
            )
            failure = mock.patch(
                f"{module}.cho_factor", side_effect=np.linalg.LinAlgError("boom")
            )

        with failure, pytest.raises(np.linalg.LinAlgError):
            est.partial_fit(X_fit[15:], y[15:])

        # Only the upper triangle of the dense precision is meaningful.
        assert np.array_equal(np.triu(_dense(est.cov_inv_)), np.triu(prec_before))
        assert np.array_equal(est.coef_, coef_before)
