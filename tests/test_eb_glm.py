"""Tests for EmpiricalBayesGLM."""

import pickle
from unittest import mock

import numpy as np
import pytest
import scipy.sparse as sp
from scipy.special import expit
from sklearn.base import clone

from bayesianbandits import (
    BayesianGLM,
    EmpiricalBayesGLM,
    LaplaceApproximator,
    RVGAApproximator,
)
from bayesianbandits._empirical_bayes import glm_log_likelihood
from bayesianbandits._sparse_bayesian_linear_regression import SparseSolver


@pytest.fixture(
    params=[SparseSolver.SUPERLU, SparseSolver.CHOLMOD],
    autouse=True,
)
def suitesparse_envvar(request):
    with mock.patch(
        "bayesianbandits._sparse_bayesian_linear_regression.solver", request.param
    ):
        yield


def _simulate(link, n=200, p=5, seed=0, alpha_true=2.0):
    rng = np.random.default_rng(seed)
    X = rng.standard_normal((n, p)) / np.sqrt(p)
    w = rng.normal(scale=alpha_true**-0.5, size=p)
    eta = X @ w
    if link == "logit":
        y = rng.binomial(1, expit(eta)).astype(np.float64)
    else:
        y = rng.poisson(np.exp(eta)).astype(np.float64)
    return X, y


def _X(X, sparse):
    return sp.csc_array(X) if sparse else X


def _diag(model):
    return np.asarray(model.cov_inv_.diagonal()).ravel().copy()


def _dense_prec(model):
    P = model.cov_inv_.toarray() if model.sparse else np.asarray(model.cov_inv_)
    return np.triu(P) + np.triu(P, 1).T


@pytest.mark.parametrize("link", ["logit", "log"])
@pytest.mark.parametrize("sparse", [True, False])
class TestEBGLM:
    def test_fit_smoke(self, link, sparse):
        X, y = _simulate(link)
        model = EmpiricalBayesGLM(link=link, sparse=sparse).fit(_X(X, sparse), y)
        assert model.alpha > 0
        assert np.isfinite(model.log_evidence_)
        assert model.n_eb_iterations_ >= 1
        assert model.eb_updates_rejected_ == 0
        assert model.predict(_X(X[:3], sparse)).shape == (3,)
        assert model.sample(_X(X[:3], sparse), size=2).shape == (2, 3)

    def test_alpha_moves_from_init(self, link, sparse):
        X, y = _simulate(link)
        model = EmpiricalBayesGLM(alpha=100.0, link=link, sparse=sparse)
        model.fit(_X(X, sparse), y)
        assert model.alpha != 100.0

    def test_evidence_monotonicity(self, link, sparse):
        X, y = _simulate(link)
        evidences = []
        alpha = 50.0
        for _ in range(10):
            model = EmpiricalBayesGLM(
                alpha=alpha, link=link, n_eb_iter=1, eb_tol=0.0, sparse=sparse
            )
            model.fit(_X(X, sparse), y)
            evidences.append(model.log_evidence_)
            alpha = model.alpha
        # MacKay's fixed point holds H_data constant in alpha, while the
        # Laplace evidence also moves through H_data(theta_MAP(alpha)), so
        # the iteration converges geometrically to within ~1e-4 of the
        # evidence argmax rather than onto it: monotone up to that slack.
        for i in range(1, len(evidences)):
            assert evidences[i] >= evidences[i - 1] - 1e-3, evidences
        assert evidences[-1] > evidences[0] + 1.0

    def test_convergence_flag(self, link, sparse):
        X, y = _simulate(link)
        model = EmpiricalBayesGLM(link=link, n_eb_iter=100, eb_tol=1e-6, sparse=sparse)
        model.fit(_X(X, sparse), y)
        assert model.eb_converged_
        assert model.n_eb_iterations_ < 100

    def test_fit_posterior_is_plain_glm_at_tuned_alpha(self, link, sparse):
        """After fit, the posterior equals BayesianGLM fitted with the
        converged alpha and the same approximator."""
        X, y = _simulate(link)
        eb = EmpiricalBayesGLM(link=link, sparse=sparse).fit(_X(X, sparse), y)
        plain = BayesianGLM(
            alpha=eb.alpha,
            link=link,
            sparse=sparse,
            approximator=LaplaceApproximator(n_iter=25, tol=1e-6),
        ).fit(_X(X, sparse), y)
        np.testing.assert_allclose(eb.coef_, plain.coef_, rtol=1e-8)
        np.testing.assert_allclose(_dense_prec(eb), _dense_prec(plain), rtol=1e-8)

    def test_log_evidence_matches_hand_formula(self, link, sparse):
        """fit's log_evidence_ is the Laplace evidence at the alpha used for
        the last EB iteration, which with n_eb_iter=1 is the initial one."""
        X, y = _simulate(link)
        alpha0 = 3.0
        model = EmpiricalBayesGLM(
            alpha=alpha0, link=link, n_eb_iter=1, sparse=sparse
        ).fit(_X(X, sparse), y)
        plain = BayesianGLM(
            alpha=alpha0,
            link=link,
            sparse=sparse,
            approximator=LaplaceApproximator(n_iter=25, tol=1e-6),
        ).fit(_X(X, sparse), y)
        theta = plain.coef_
        P = _dense_prec(plain)
        p = X.shape[1]
        expected = (
            glm_log_likelihood(X, y, theta, link)
            + 0.5 * p * np.log(alpha0)
            - 0.5 * alpha0 * theta @ theta
            - 0.5 * np.linalg.slogdet(P)[1]
        )
        np.testing.assert_allclose(model.log_evidence_, expected, rtol=1e-8)

    def test_recovers_true_alpha(self, link, sparse):
        X, y = _simulate(link, n=4000, p=40, seed=3, alpha_true=4.0)
        model = EmpiricalBayesGLM(alpha=0.1, link=link, sparse=sparse, n_eb_iter=50)
        model.fit(_X(X, sparse), y)
        assert 0.4 < model.alpha / 4.0 < 2.5

    def test_n_eb_iter_zero(self, link, sparse):
        X, y = _simulate(link)
        model = EmpiricalBayesGLM(alpha=2.0, link=link, n_eb_iter=0, sparse=sparse)
        model.fit(_X(X, sparse), y)
        assert model.alpha == 2.0
        assert model.log_evidence_ == -np.inf
        assert model.n_eb_iterations_ == 0
        assert not model.eb_converged_
        # partial_fit still tunes
        model.partial_fit(_X(X[:20], sparse), y[:20])
        assert model.alpha != 2.0
        assert np.isfinite(model.log_evidence_)

    def test_partial_fit_updates_alpha_and_precision(self, link, sparse):
        X, y = _simulate(link)
        model = EmpiricalBayesGLM(link=link, sparse=sparse).fit(
            _X(X[:100], sparse), y[:100]
        )
        alpha_before = model.alpha
        ev_before = model.log_evidence_
        diag_before = _diag(model)
        prior_before = model._prior_scalar
        model.partial_fit(_X(X[100:], sparse), y[100:])
        assert model.alpha != alpha_before
        assert model.log_evidence_ != ev_before
        assert np.all(np.isfinite(_diag(model)))
        assert not np.allclose(diag_before, _diag(model))
        # prior scalar was rescaled to the new alpha (no decay here)
        np.testing.assert_allclose(
            model._prior_scalar, prior_before * model.alpha / alpha_before
        )

    def test_correct_precision_leaves_data_component_alone(self, link, sparse):
        X, y = _simulate(link)
        model = EmpiricalBayesGLM(link=link, sparse=sparse).fit(
            _X(X[:100], sparse), y[:100]
        )
        # Replay the step partial_fit performs, watching the pieces.
        model._pending_floor = 0.0
        BayesianGLM.partial_fit(model, _X(X[100:], sparse), y[100:])
        P_before = _dense_prec(model)
        theta_before = model.coef_.copy()
        s_before = model._prior_scalar
        data_before = P_before - s_before * np.eye(X.shape[1])
        alpha_old = model.alpha
        model.alpha = 2.5 * alpha_old
        model._correct_precision(alpha_old)
        P_after = _dense_prec(model)
        data_after = P_after - model._prior_scalar * np.eye(X.shape[1])
        np.testing.assert_allclose(data_after, data_before, atol=1e-10)
        np.testing.assert_allclose(model._prior_scalar, 2.5 * s_before)
        # The mode moves to the new prior's MAP under the quadratic model.
        np.testing.assert_allclose(
            model.coef_, np.linalg.solve(P_after, P_before @ theta_before), atol=1e-10
        )
        # The cached factor is the fresh one: its logdet matches P_after.
        np.testing.assert_allclose(
            model._precision_factor.logdet(), np.linalg.slogdet(P_after)[1]
        )
        assert model.sample(_X(X[:2], sparse)).shape == (1, 2)

    def test_correct_precision_noop(self, link, sparse):
        X, y = _simulate(link)
        model = EmpiricalBayesGLM(link=link, sparse=sparse).fit(_X(X, sparse), y)
        _ = model._precision_factor
        model._correct_precision(model.alpha)
        assert "_precision_factor" in model.__dict__

    def test_chunked_partial_fit_tracks_batch_fit(self, link, sparse):
        X, y = _simulate(link, n=2000, p=10, seed=5)
        batch = EmpiricalBayesGLM(link=link, sparse=sparse).fit(_X(X, sparse), y)
        online = EmpiricalBayesGLM(link=link, sparse=sparse)
        for start in range(0, 2000, 100):
            online.partial_fit(
                _X(X[start : start + 100], sparse), y[start : start + 100]
            )
        assert 0.5 < online.alpha / batch.alpha < 2.0
        np.testing.assert_allclose(online.coef_, batch.coef_, atol=0.3)

    def test_sample_before_partial_fit(self, link, sparse):
        X, y = _simulate(link)
        model = EmpiricalBayesGLM(link=link, sparse=sparse)
        model.sample(_X(X[:1], sparse))
        assert hasattr(model, "coef_")
        assert not hasattr(model, "_prior_scalar")
        model.partial_fit(_X(X[:50], sparse), y[:50])
        assert hasattr(model, "_prior_scalar")
        assert model._effective_n == 50.0
        assert model.n_eb_iterations_ == 0
        assert model.eb_updates_rejected_ == 0
        assert np.isfinite(model.alpha)
        assert np.isfinite(model.log_evidence_)
        # precision diagonal is consistent with the tracked prior scalar
        P = _dense_prec(model)
        data = P - model._prior_scalar * np.eye(X.shape[1])
        assert np.all(np.linalg.eigvalsh(data) > -1e-8)

    def test_partial_fit_cold_start_runs_fit(self, link, sparse):
        X, y = _simulate(link)
        model = EmpiricalBayesGLM(link=link, sparse=sparse)
        model.partial_fit(_X(X, sparse), y)
        assert model.n_eb_iterations_ >= 1
        assert hasattr(model, "_prior_scalar")

    def test_decay_reinjects_prior(self, link, sparse):
        X, y = _simulate(link)
        model = EmpiricalBayesGLM(link=link, learning_rate=0.9, sparse=sparse)
        model.fit(_X(X, sparse), y)
        diag_before = _diag(model)
        s_before = model._prior_scalar
        n_eff_before = model._effective_n
        model.decay(_X(X[:3], sparse))
        g = 0.9**3
        np.testing.assert_allclose(
            _diag(model), g * diag_before + (1 - g) * model.alpha
        )
        np.testing.assert_allclose(
            model._prior_scalar, g * s_before + (1 - g) * model.alpha
        )
        np.testing.assert_allclose(model._effective_n, g * n_eff_before)

    def test_repeated_decay_converges_to_alpha(self, link, sparse):
        X, y = _simulate(link)
        model = EmpiricalBayesGLM(link=link, learning_rate=0.5, sparse=sparse)
        model.fit(_X(X, sparse), y)
        for _ in range(60):
            model.decay(_X(X[:1], sparse))
        np.testing.assert_allclose(_diag(model), model.alpha, rtol=1e-6)
        np.testing.assert_allclose(model._prior_scalar, model.alpha, rtol=1e-6)

    def test_decay_rate_one_no_reinjection(self, link, sparse):
        X, y = _simulate(link)
        model = EmpiricalBayesGLM(link=link, learning_rate=0.9, sparse=sparse)
        model.fit(_X(X, sparse), y)
        before = _dense_prec(model)
        model.decay(_X(X[:3], sparse), decay_rate=1.0)
        np.testing.assert_allclose(_dense_prec(model), before)

    def test_decay_before_fit(self, link, sparse):
        model = EmpiricalBayesGLM(link=link, learning_rate=0.9, sparse=sparse)
        model.decay(np.zeros((3, 5)))
        assert not hasattr(model, "coef_")

    def test_partial_fit_prior_reinjection(self, link, sparse):
        """With learning_rate < 1 the prior component after partial_fit is
        γⁿ·s_old + (1 - γⁿ)·alpha_old, then rescaled by the MacKay step."""
        X, y = _simulate(link)
        model = EmpiricalBayesGLM(link=link, learning_rate=0.95, sparse=sparse)
        model.fit(_X(X[:100], sparse), y[:100])
        s_old, alpha_old = model._prior_scalar, model.alpha
        g = 0.95**20
        model.partial_fit(_X(X[100:120], sparse), y[100:120])
        expected = (g * s_old + (1 - g) * alpha_old) * model.alpha / alpha_old
        np.testing.assert_allclose(model._prior_scalar, expected)
        # Precision minus the prior part is a PSD data Hessian.
        data = _dense_prec(model) - model._prior_scalar * np.eye(X.shape[1])
        assert np.all(np.linalg.eigvalsh(data) > -1e-8)

    def test_reinjection_is_zero_centered(self, link, sparse):
        """Stabilized re-injection shrinks toward zero: the posterior mean
        after partial_fit equals the base GLM run against the explicit
        product prior N(m, (γⁿP)⁻¹)·N(0, (sI)⁻¹)."""
        X, y = _simulate(link)
        model = EmpiricalBayesGLM(link=link, learning_rate=0.95, sparse=sparse)
        model.fit(_X(X[:100], sparse), y[:100])
        m, P = model.coef_.copy(), _dense_prec(model)
        n = 20
        g = 0.95**n
        s = (1 - g) * model.alpha
        P_eq = g * P + s * np.eye(X.shape[1])
        m_eq = np.linalg.solve(P_eq, g * P @ m)

        model._pending_floor = model.alpha
        model._fit_helper(_X(X[100 : 100 + n], sparse), y[100 : 100 + n])

        from bayesianbandits._estimators import compute_effective_weights

        ref = BayesianGLM(link=link, sparse=sparse, approximator=model.approximator_)
        ref._initialize_prior(_X(X, sparse))
        ref.coef_ = m_eq
        ref.cov_inv_ = sp.csc_array(P_eq) if sparse else P_eq
        ref._fit_helper(
            _X(X[100 : 100 + n], sparse),
            y[100 : 100 + n],
            compute_effective_weights(n, None, 0.95),
        )
        np.testing.assert_allclose(model.coef_, ref.coef_, atol=1e-8)

    def test_sample_weight_matches_duplication(self, link, sparse):
        X, y = _simulate(link, n=60)
        sw = np.array([1.0, 2.0, 3.0] * 20)
        a = EmpiricalBayesGLM(link=link, sparse=sparse).fit(_X(X, sparse), y, sw)
        X_rep = np.repeat(X, sw.astype(int), axis=0)
        y_rep = np.repeat(y, sw.astype(int))
        b = EmpiricalBayesGLM(link=link, sparse=sparse).fit(_X(X_rep, sparse), y_rep)
        np.testing.assert_allclose(a.alpha, b.alpha, rtol=1e-6)
        np.testing.assert_allclose(a.log_evidence_, b.log_evidence_, rtol=1e-6)

    def test_rvga_approximator(self, link, sparse):
        X, y = _simulate(link)
        model = EmpiricalBayesGLM(
            link=link, sparse=sparse, approximator=RVGAApproximator()
        )
        model.fit(_X(X[:100], sparse), y[:100])
        model.partial_fit(_X(X[100:], sparse), y[100:])
        assert np.isfinite(model.alpha) and model.alpha > 0
        assert np.isfinite(model.log_evidence_)

    def test_trace_method_diagonal(self, link, sparse):
        X, y = _simulate(link)
        model = EmpiricalBayesGLM(link=link, sparse=sparse, trace_method="diagonal")
        model.fit(_X(X, sparse), y)
        model.partial_fit(_X(X[:10], sparse), y[:10])
        assert np.isfinite(model.alpha) and model.alpha > 0

    def test_get_set_params_and_clone(self, link, sparse):
        model = EmpiricalBayesGLM(
            alpha=2.0, link=link, n_eb_iter=3, eb_tol=1e-2, sparse=sparse
        )
        params = model.get_params()
        assert params["n_eb_iter"] == 3 and params["eb_tol"] == 1e-2
        assert params["link"] == link
        model.set_params(n_eb_iter=7)
        assert clone(model).n_eb_iter == 7

    def test_pickle_roundtrip(self, link, sparse):
        X, y = _simulate(link)
        model = EmpiricalBayesGLM(link=link, sparse=sparse).fit(_X(X, sparse), y)
        _ = model._precision_factor
        restored = pickle.loads(pickle.dumps(model))
        assert "_precision_factor" not in restored.__dict__
        assert restored.alpha == model.alpha
        assert restored._prior_scalar == model._prior_scalar
        np.testing.assert_allclose(
            restored.predict(_X(X[:5], sparse)), model.predict(_X(X[:5], sparse))
        )
        restored.partial_fit(_X(X[:10], sparse), y[:10])


class TestEBGLMGuardrail:
    def test_rejected_updates_are_counted_and_alpha_kept(self):
        X, y = _simulate("logit", n=50, p=5)
        model = EmpiricalBayesGLM(link="logit").fit(X, y)
        alpha = model.alpha
        with mock.patch(
            "bayesianbandits._eb_estimators.mackay_update_glm",
            return_value=mock.Mock(alpha=alpha, log_evidence=-1.0, rejected=True),
        ):
            model.partial_fit(X[:10], y[:10])
        assert model.eb_updates_rejected_ == 1
        assert model.alpha == alpha
        assert model.log_evidence_ == -1.0

    def test_fit_resets_rejection_count(self):
        X, y = _simulate("logit", n=50, p=5)
        model = EmpiricalBayesGLM(link="logit").fit(X, y)
        model.eb_updates_rejected_ = 4
        model.fit(X, y)
        assert model.eb_updates_rejected_ == 0
