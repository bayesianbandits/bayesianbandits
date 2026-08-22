"""Tests for EmpiricalBayesGLM."""

import pickle
from copy import deepcopy
from typing import cast
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

    def test_secant_acceleration_is_load_bearing(self, link, sparse):
        """Weak signal with p comparable to n puts the evidence maximum at
        an alpha MacKay's fixed-point iteration crawls toward. The secant
        reaches the same fixed point in a fraction of the iterations."""
        import bayesianbandits._eb_estimators as ebm

        X, y = _simulate(link, n=200, p=40, seed=7, alpha_true=2.0)
        accelerated = EmpiricalBayesGLM(link=link, sparse=sparse, n_eb_iter=30)
        accelerated.fit(_X(X, sparse), y)
        with mock.patch.object(ebm.SecantRootFinder, "next", lambda self, u, h: u + h):
            plain = EmpiricalBayesGLM(link=link, sparse=sparse, n_eb_iter=300)
            plain.fit(_X(X, sparse), y)
        assert accelerated.eb_converged_ and plain.eb_converged_
        assert accelerated.n_eb_iterations_ <= plain.n_eb_iterations_
        if link == "logit":
            # The crawl: plain MacKay needs ~100 iterations to settle at
            # eb_tol=1e-9, and at the default tolerance stops early (17)
            # because successive evidence changes shrink with the step
            # while alpha is still short.
            assert accelerated.n_eb_iterations_ * 3 <= plain.n_eb_iterations_
        assert 0.9 < accelerated.alpha / plain.alpha < 1.1
        assert abs(accelerated.log_evidence_ - plain.log_evidence_) < 1e-2

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
        # Both IRLS runs stop within tol=1e-6 of the mode; the EB one is
        # warm-started from the previous iteration, the plain one from 0.
        np.testing.assert_allclose(eb.coef_, plain.coef_, atol=1e-5)
        np.testing.assert_allclose(_dense_prec(eb), _dense_prec(plain), atol=1e-5)

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
        # eb_alpha_tol=inf: exactly one MacKay step, so the bookkeeping
        # below can be checked against that step alone.
        model = EmpiricalBayesGLM(link=link, sparse=sparse, eb_alpha_tol=np.inf)
        model.fit(_X(X[:100], sparse), y[:100])
        alpha_before = model.alpha
        ev_before = model.log_evidence_
        diag_before = _diag(model)
        prior_before = model._prior_scalar
        model.partial_fit(_X(X[100:], sparse), y[100:])
        assert model.alpha != alpha_before
        assert model.log_evidence_ != ev_before
        assert np.all(np.isfinite(_diag(model)))
        assert not np.allclose(diag_before, _diag(model))
        # prior scalar was rescaled to the new alpha (no decay here), and
        # the matching diagonal shift is booked for the next update
        np.testing.assert_allclose(
            model._prior_scalar, prior_before * model.alpha / alpha_before
        )
        np.testing.assert_allclose(
            model._pending_shift, prior_before * (model.alpha / alpha_before - 1)
        )

    def test_correct_precision_defers_the_shift(self, link, sparse):
        """The MacKay rescale is booked, not applied: the stored posterior
        and its factor are untouched until the next update."""
        X, y = _simulate(link)
        model = EmpiricalBayesGLM(link=link, sparse=sparse).fit(
            _X(X[:100], sparse), y[:100]
        )
        assert model._pending_shift == 0.0
        P_before = _dense_prec(model)
        theta_before = model.coef_.copy()
        s_before = model._prior_scalar
        factor_before = model._precision_factor
        alpha_old = model.alpha
        model.alpha = 2.5 * alpha_old
        model._correct_precision(alpha_old)
        np.testing.assert_array_equal(_dense_prec(model), P_before)
        np.testing.assert_array_equal(model.coef_, theta_before)
        assert model._precision_factor is factor_before
        np.testing.assert_allclose(model._prior_scalar, 2.5 * s_before)
        np.testing.assert_allclose(model._pending_shift, 1.5 * s_before)

    def test_deferred_shift_matches_eager_resolve(self, link, sparse):
        """Applying the booked shift at the next update gives the same
        posterior as rescaling Λ and re-solving θ = Λ_new⁻¹·Λ_old·θ_old
        right away, then updating from that."""
        X, y = _simulate(link)
        model = EmpiricalBayesGLM(link=link, sparse=sparse).fit(
            _X(X[:100], sparse), y[:100]
        )
        P_old = _dense_prec(model)
        theta_old = model.coef_.copy()
        alpha_old = model.alpha
        model.alpha = 2.5 * alpha_old
        model._correct_precision(alpha_old)
        shift = model._pending_shift
        model._pending_floor = 0.0
        model._fit_helper(_X(X[100:120], sparse), y[100:120])
        assert model._pending_shift == 0.0

        P_new = P_old + shift * np.eye(X.shape[1])
        eager = BayesianGLM(link=link, sparse=sparse, approximator=model.approximator_)
        eager._initialize_prior(_X(X, sparse))
        eager.coef_ = np.linalg.solve(P_new, P_old @ theta_old)
        eager.cov_inv_ = sp.csc_array(P_new) if sparse else P_new
        eager._fit_helper(_X(X[100:120], sparse), y[100:120])
        np.testing.assert_allclose(model.coef_, eager.coef_, atol=1e-8)
        np.testing.assert_allclose(_dense_prec(model), _dense_prec(eager), atol=1e-8)
        # The data component is untouched by the shift.
        data_new = _dense_prec(model) - model._prior_scalar * np.eye(X.shape[1])
        assert np.all(np.linalg.eigvalsh(data_new) > -1e-8)

    def test_decay_applies_pending_shift(self, link, sparse):
        X, y = _simulate(link)
        model = EmpiricalBayesGLM(link=link, learning_rate=0.9, sparse=sparse).fit(
            _X(X, sparse), y
        )
        alpha_old = model.alpha
        model.alpha = 0.5 * alpha_old
        model._correct_precision(alpha_old)
        shift = model._pending_shift
        assert shift < 0
        diag_before = _diag(model)
        s_logical = model._prior_scalar
        model.decay(_X(X[:2], sparse))
        g = 0.9**2
        assert model._pending_shift == 0.0
        np.testing.assert_allclose(
            _diag(model), g * (diag_before + shift) + (1 - g) * model.alpha
        )
        np.testing.assert_allclose(
            model._prior_scalar, g * s_logical + (1 - g) * model.alpha
        )

    def test_decay_moves_the_mean_with_the_pending_shift(self, link, sparse):
        """A booked shift is only exact if the mode moves with it. The
        quadratic model puts it at ``Λ_new⁻¹·Λ_old·θ_old``; landing the
        shift on the diagonal alone would leave a mean that still belongs
        to the old alpha, and keep leaving it as the shifts compound."""
        X, y = _simulate(link)
        model = EmpiricalBayesGLM(link=link, learning_rate=1.0, sparse=sparse).fit(
            _X(X, sparse), y
        )
        alpha_old = model.alpha
        model.alpha = 0.5 * alpha_old
        model._correct_precision(alpha_old)
        shift = model._pending_shift
        prec_old = _dense_prec(model)
        coef_old = model.coef_.copy()
        expected = np.linalg.solve(
            prec_old + shift * np.eye(X.shape[1]), prec_old @ coef_old
        )

        model.decay(_X(X[:2], sparse), decay_rate=1.0)

        assert not np.allclose(expected, coef_old)  # the shift really moves it
        np.testing.assert_allclose(model.coef_, expected, atol=1e-10)

    def test_pending_shift_lands_the_same_through_decay_or_partial_fit(
        self, link, sparse
    ):
        """The deferral has two consumers and they have to agree: updating
        after a decay that landed the shift must match letting the next
        ``partial_fit`` fold it into its own solve."""
        X, y = _simulate(link, n=300)
        model = EmpiricalBayesGLM(
            link=link, learning_rate=1.0, n_eb_iter=1, sparse=sparse
        )
        model.partial_fit(_X(X[:100], sparse), y[:100])
        model.partial_fit(_X(X[100:200], sparse), y[100:200])
        assert model._pending_shift != 0.0

        through_decay, through_fit = deepcopy(model), deepcopy(model)
        through_decay.decay(_X(X[:1], sparse), decay_rate=1.0)
        through_decay.partial_fit(_X(X[200:], sparse), y[200:])
        through_fit.partial_fit(_X(X[200:], sparse), y[200:])

        np.testing.assert_allclose(through_decay.coef_, through_fit.coef_, atol=1e-10)
        # Loose only against the secant finder's own stopping point; landing
        # the shift without the re-solve moves alpha by percents, not ulps.
        np.testing.assert_allclose(through_decay.alpha, through_fit.alpha, rtol=1e-6)

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

    def test_partial_fit_iterates_to_the_fixed_point(self, link, sparse):
        """After a cold start far from the fixed point, partial_fit keeps
        stepping on the Laplace approximation it holds until alpha
        settles, so it lands near where fit on the same data does."""
        X, y = _simulate(link, n=400, p=40, seed=7, alpha_true=2.0)
        online = EmpiricalBayesGLM(link=link, sparse=sparse)
        one_step = EmpiricalBayesGLM(link=link, sparse=sparse, eb_alpha_tol=np.inf)
        for start in range(0, 400, 20):
            online.partial_fit(_X(X[start : start + 20], sparse), y[start : start + 20])
            one_step.partial_fit(
                _X(X[start : start + 20], sparse), y[start : start + 20]
            )
        batch = EmpiricalBayesGLM(link=link, sparse=sparse, n_eb_iter=50)
        batch.fit(_X(X, sparse), y)
        assert 0.8 < online.alpha / batch.alpha < 1.25
        np.testing.assert_allclose(online.coef_, batch.coef_, atol=0.1)
        # and a single step per batch is not enough on the logit data
        if link == "logit":
            assert abs(np.log(online.alpha / batch.alpha)) < abs(
                np.log(one_step.alpha / batch.alpha)
            )

    def test_partial_fit_in_steady_state_takes_one_step(self, link, sparse):
        X, y = _simulate(link, n=2000, p=10, seed=5)
        model = EmpiricalBayesGLM(link=link, sparse=sparse).fit(
            _X(X[:1500], sparse), y[:1500]
        )
        import bayesianbandits._eb_estimators as ebm

        with mock.patch.object(
            ebm.EmpiricalBayesGLM,
            "_apply_alpha_now",
            autospec=True,
            side_effect=ebm.EmpiricalBayesGLM._apply_alpha_now,
        ) as eager:
            model.partial_fit(_X(X[1500:1510], sparse), y[1500:1510])
        eager.assert_not_called()

    def test_alpha_recovers_from_a_no_signal_start(self, link, sparse):
        """Rows that carry no information push alpha up; the ceiling keeps
        it finite and theta representable, and once informative rows
        arrive alpha comes back to what fit finds."""
        X, y = _simulate(link, n=2000, p=10, seed=7, alpha_true=2.0)
        model = EmpiricalBayesGLM(link=link, sparse=sparse)
        model.partial_fit(_X(X[:40], sparse), y[:40])
        H_max = _diag(model).max() - (model._prior_scalar - model._pending_shift)
        assert model.alpha <= 1e10 * H_max * (1 + 1e-9)
        assert np.all(np.isfinite(model.coef_))
        for start in range(40, 2000, 20):
            model.partial_fit(_X(X[start : start + 20], sparse), y[start : start + 20])
        batch = EmpiricalBayesGLM(link=link, sparse=sparse, n_eb_iter=50)
        batch.fit(_X(X, sparse), y)
        assert 0.8 < model.alpha / batch.alpha < 1.25

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
        # precision diagonal is consistent with the stored prior scalar
        P = _dense_prec(model)
        stored = model._prior_scalar - model._pending_shift
        data = P - stored * np.eye(X.shape[1])
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
        model = EmpiricalBayesGLM(
            link=link, learning_rate=0.95, sparse=sparse, eb_alpha_tol=np.inf
        )
        model.fit(_X(X[:100], sparse), y[:100])
        s_old, alpha_old = model._prior_scalar, model.alpha
        g = 0.95**20
        model.partial_fit(_X(X[100:120], sparse), y[100:120])
        expected = (g * s_old + (1 - g) * alpha_old) * model.alpha / alpha_old
        np.testing.assert_allclose(model._prior_scalar, expected)
        # The stored precision still carries the pre-step prior part;
        # minus that, it is a PSD data Hessian.
        stored = model._prior_scalar - model._pending_shift
        np.testing.assert_allclose(stored, g * s_old + (1 - g) * alpha_old)
        data = _dense_prec(model) - stored * np.eye(X.shape[1])
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
        cloned = cast(EmpiricalBayesGLM, clone(model))
        assert cloned.n_eb_iter == 7

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


class TestFailedPartialFitLeavesEBStateIntact:
    """A ``partial_fit`` that raises must not have moved the EB bookkeeping."""

    @staticmethod
    def _fitted(sparse):
        X, y = _simulate("log", n=40, p=4)
        model = EmpiricalBayesGLM(link="log", learning_rate=0.9, sparse=sparse)
        return model.fit(_X(X, sparse), y), X, y

    @pytest.mark.parametrize("sparse", [False, True])
    def test_prior_scalar_and_floor_survive_a_raising_update(self, sparse):
        model, X, y = self._fitted(sparse)
        before = (model._prior_scalar, model._pending_floor, model.alpha)

        with mock.patch.object(
            model.approximator_,
            "update_posterior",
            side_effect=np.linalg.LinAlgError("injected"),
        ):
            with pytest.raises(np.linalg.LinAlgError):
                model.partial_fit(_X(X[:5], sparse), y[:5])

        assert (model._prior_scalar, model._pending_floor, model.alpha) == before

    @pytest.mark.parametrize("sparse", [False, True])
    def test_a_failed_update_does_not_change_the_next_one(self, sparse):
        clean, X, y = self._fitted(sparse)
        poisoned, _, _ = self._fitted(sparse)

        with mock.patch.object(
            poisoned.approximator_,
            "update_posterior",
            side_effect=np.linalg.LinAlgError("injected"),
        ):
            with pytest.raises(np.linalg.LinAlgError):
                poisoned.partial_fit(_X(X[:5], sparse), y[:5])

        clean.partial_fit(_X(X[:5], sparse), y[:5])
        poisoned.partial_fit(_X(X[:5], sparse), y[:5])

        assert poisoned.alpha == clean.alpha
        assert poisoned._prior_scalar == clean._prior_scalar
        np.testing.assert_array_equal(poisoned.coef_, clean.coef_)
        np.testing.assert_allclose(_dense_prec(poisoned), _dense_prec(clean))

    def test_a_first_ever_partial_fit_that_raises_leaves_no_prior_scalar(self):
        X, y = _simulate("log", n=40, p=4)
        model = EmpiricalBayesGLM(link="log", learning_rate=0.9)

        with mock.patch.object(
            EmpiricalBayesGLM,
            "fit",
            side_effect=np.linalg.LinAlgError("injected"),
        ):
            with pytest.raises(np.linalg.LinAlgError):
                model.partial_fit(X[:5], y[:5])

        assert not hasattr(model, "_prior_scalar")
