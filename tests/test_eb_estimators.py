"""Tests for Empirical Bayes estimators."""

import pickle
from unittest import mock

import numpy as np
import pytest
from sklearn.base import clone

from bayesianbandits import (
    EmpiricalBayesDirichletClassifier,
    EmpiricalBayesNormalRegressor,
)
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
