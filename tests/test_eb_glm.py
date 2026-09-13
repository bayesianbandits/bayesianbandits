"""Tests for EmpiricalBayesGLM."""

import pickle
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
    StabilizedForgetting,
)
from bayesianbandits._empirical_bayes import glm_log_likelihood


@pytest.fixture(autouse=True)
def suitesparse_envvar(sparse_solver):
    """Run every test in this module against both sparse backends."""
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
        assert model.eb_converged_ and 1 <= model.n_eb_iterations_ < 10
        assert model.eb_updates_rejected_ == 0
        assert model.predict(_X(X[:3], sparse)).shape == (3,)
        assert model.sample(_X(X[:3], sparse), size=2).shape == (2, 3)

    def test_evidence_monotonicity(self, link, sparse):
        X, y = _simulate(link)
        evidences = []
        alpha = 50.0
        for _ in range(10):
            # re-anchored each round, so only plain MacKay is one objective
            model = EmpiricalBayesGLM(
                alpha=alpha,
                link=link,
                n_eb_iter=1,
                eb_tol=0.0,
                sparse=sparse,
                alpha_prior_strength=0.0,
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
        # Each precision is built at its run's last pre-convergence
        # iterate, so they agree only to O(tol · scale).
        np.testing.assert_allclose(eb.coef_, plain.coef_, atol=1e-5)
        np.testing.assert_allclose(_dense_prec(eb), _dense_prec(plain), atol=1e-4)

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
        k = model.alpha_prior_strength
        expected = (
            glm_log_likelihood(X, y, theta, link)
            + 0.5 * p * np.log(alpha0)
            - 0.5 * alpha0 * theta @ theta
            - 0.5 * np.linalg.slogdet(P)[1]
            + 0.5 * k * np.log(alpha0)
            - 0.5 * k  # alpha / alpha0 at the anchor
        )
        np.testing.assert_allclose(model.log_evidence_, expected, rtol=1e-8)

    @pytest.mark.parametrize("k", [0.0, 0.5])
    def test_converged_alpha_is_the_mackay_fixed_point(self, link, sparse, k):
        """At convergence alpha = (gamma + k) / (||theta||^2 + k/alpha0) with
        gamma and theta taken from an independent plain GLM fit at that
        alpha."""
        X, y = _simulate(link)
        eb = EmpiricalBayesGLM(
            link=link,
            sparse=sparse,
            n_eb_iter=100,
            eb_tol=1e-10,
            alpha_prior_strength=k,
        )
        eb.fit(_X(X, sparse), y)
        plain = BayesianGLM(
            alpha=eb.alpha,
            link=link,
            sparse=sparse,
            approximator=LaplaceApproximator(n_iter=50, tol=1e-10),
        ).fit(_X(X, sparse), y)
        theta = plain.coef_
        gamma = X.shape[1] - eb.alpha * np.trace(np.linalg.inv(_dense_prec(plain)))
        np.testing.assert_allclose(
            eb.alpha, (gamma + k) / (theta @ theta + k / eb._alpha0), rtol=1e-5
        )

    def test_partial_fit_tracks_full_fit(self, link, sparse):
        """Chunked partial_fit from a far-off alpha lands near the full
        fit's fixed point. The slack is sequential Laplace: even at fixed
        alpha the chunked posterior is 2-4% off the batch one."""
        X, y = _simulate(link, n=4000, p=10, seed=7)
        # the two starts would anchor differently; compare plain MacKay
        full = EmpiricalBayesGLM(
            alpha=1e3, link=link, sparse=sparse, n_eb_iter=50, alpha_prior_strength=0.0
        )
        full.fit(_X(X, sparse), y)
        for alpha0 in (1e3, 1e-2):
            online = EmpiricalBayesGLM(
                alpha=alpha0,
                link=link,
                sparse=sparse,
                n_eb_iter=1,
                alpha_prior_strength=0.0,
            )
            for start in range(0, 4000, 50):
                online.partial_fit(
                    _X(X[start : start + 50], sparse), y[start : start + 50]
                )
            np.testing.assert_allclose(online.alpha, full.alpha, rtol=6e-2)
            np.testing.assert_allclose(online.coef_, full.coef_, atol=5e-2)

    def test_recovers_true_alpha(self, link, sparse):
        X, y = _simulate(link, n=4000, p=40, seed=3, alpha_true=4.0)
        # alpha0 is 40x too small, so the hyperprior's bound would bind
        model = EmpiricalBayesGLM(
            alpha=0.1, link=link, sparse=sparse, n_eb_iter=50, alpha_prior_strength=0.0
        )
        model.fit(_X(X, sparse), y)
        assert 0.4 < model.alpha / 4.0 < 2.5

    def test_hyperprior_keeps_alpha_off_the_guardrail(self, link, sparse):
        """Near-zero coefficients: plain alpha hits the ceiling, regularized stays bounded."""
        X, y = _simulate(link, n=30, p=5, seed=5, alpha_true=1e4)
        p, alpha0 = 5, 1.0
        k = EmpiricalBayesGLM().alpha_prior_strength
        bound = (p + k) * alpha0 / k * (1 + 1e-9)

        plain = EmpiricalBayesGLM(
            link=link, sparse=sparse, n_eb_iter=50, alpha_prior_strength=0.0
        ).fit(_X(X, sparse), y)
        assert plain.alpha > 100

        model = EmpiricalBayesGLM(link=link, sparse=sparse, n_eb_iter=50)
        model.fit(_X(X, sparse), y)
        assert model.alpha <= bound
        assert model.eb_converged_ and model.eb_updates_rejected_ == 0

        online = EmpiricalBayesGLM(link=link, sparse=sparse)
        for start in range(0, 30, 5):
            online.partial_fit(_X(X[start : start + 5], sparse), y[start : start + 5])
            assert online.alpha <= bound
        assert online.eb_updates_rejected_ == 0

    def test_negative_prior_strength_raises(self, link, sparse):
        X, y = _simulate(link)
        model = EmpiricalBayesGLM(link=link, sparse=sparse, alpha_prior_strength=-1.0)
        with pytest.raises(ValueError, match="alpha_prior_strength"):
            model.fit(_X(X, sparse), y)

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

    def test_correct_precision_moves_the_mean_with_the_prior(self, link, sparse):
        """Rescaling the prior part of Λ is a diagonal shift, and the mode
        under the shifted precision is Λ_new⁻¹·Λ_old·θ_old."""
        X, y = _simulate(link)
        model = EmpiricalBayesGLM(link=link, sparse=sparse).fit(
            _X(X[:100], sparse), y[:100]
        )
        P_old = _dense_prec(model)
        theta_old = model.coef_.copy()
        s_old, alpha_old = model._prior_scalar, model.alpha
        model.alpha = 2.5 * alpha_old
        model._correct_precision(alpha_old)
        P_new = P_old + 1.5 * s_old * np.eye(X.shape[1])
        np.testing.assert_allclose(model._prior_scalar, 2.5 * s_old)
        np.testing.assert_allclose(_dense_prec(model), P_new, atol=1e-10)
        np.testing.assert_allclose(
            model.coef_, np.linalg.solve(P_new, P_old @ theta_old), atol=1e-8
        )

    def test_correct_precision_noop(self, link, sparse):
        X, y = _simulate(link)
        model = EmpiricalBayesGLM(link=link, sparse=sparse).fit(_X(X, sparse), y)
        _ = model._precision_factor
        model._correct_precision(model.alpha)
        assert "_precision_factor" in model.__dict__

    def test_alpha_recovers_from_a_no_signal_start(self, link, sparse):
        """Rows that carry no information push alpha up; the ceiling keeps
        it finite and theta representable, and once informative rows
        arrive alpha comes back to what fit finds."""
        X, y = _simulate(link, n=2000, p=10, seed=7, alpha_true=2.0)
        model = EmpiricalBayesGLM(link=link, sparse=sparse)
        model.partial_fit(_X(X[:40], sparse), y[:40])
        H_max = _diag(model).max() - model._prior_scalar
        assert model.alpha <= 1e10 * H_max * (1 + 1e-9)
        assert np.all(np.isfinite(model.coef_))
        for start in range(40, 2000, 20):
            model.partial_fit(_X(X[start : start + 20], sparse), y[start : start + 20])
        batch = EmpiricalBayesGLM(link=link, sparse=sparse, n_eb_iter=50)
        batch.fit(_X(X, sparse), y)
        assert 0.8 < model.alpha / batch.alpha < 1.25

    def test_alpha0_keeps_the_constructor_alpha(self, link, sparse):
        X, y = _simulate(link)
        model = EmpiricalBayesGLM(alpha=3.0, link=link, sparse=sparse)
        model.fit(_X(X, sparse), y)
        assert model.alpha != 3.0 and model._alpha0 == 3.0
        model.fit(_X(X, sparse), y)
        assert model._alpha0 == 3.0

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
        data = P - model._prior_scalar * np.eye(X.shape[1])
        assert np.all(np.linalg.eigvalsh(data) > -1e-8)

    def test_decay_reinjects_prior(self, link, sparse):
        X, y = _simulate(link)
        model = EmpiricalBayesGLM(
            link=link, forgetting=StabilizedForgetting(0.9), sparse=sparse
        )
        model.fit(_X(X, sparse), y)
        diag_before = _diag(model)
        s_before = model._prior_scalar
        n_eff_before = model._effective_n
        model.decay(decay_rate=0.9, steps=3)
        g = 0.9**3
        np.testing.assert_allclose(
            _diag(model), g * diag_before + (1 - g) * model.alpha
        )
        np.testing.assert_allclose(
            model._prior_scalar, g * s_before + (1 - g) * model.alpha
        )
        np.testing.assert_allclose(model._effective_n, g * n_eff_before)

    def test_partial_fit_prior_reinjection(self, link, sparse):
        """With learning_rate < 1 the prior component after partial_fit is
        γⁿ·s_old + (1 - γⁿ)·alpha_old, then rescaled by the MacKay step."""
        X, y = _simulate(link)
        model = EmpiricalBayesGLM(
            link=link, forgetting=StabilizedForgetting(0.95), sparse=sparse
        )
        model.fit(_X(X[:100], sparse), y[:100])
        s_old, alpha_old = model._prior_scalar, model.alpha
        g = 0.95**20
        model.partial_fit(_X(X[100:120], sparse), y[100:120])
        expected = (g * s_old + (1 - g) * alpha_old) * model.alpha / alpha_old
        np.testing.assert_allclose(model._prior_scalar, expected)
        # Minus the prior part, the precision is a PSD data Hessian.
        data = _dense_prec(model) - expected * np.eye(X.shape[1])
        assert np.all(np.linalg.eigvalsh(data) > -1e-8)

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

    def test_get_set_params_and_clone(self, link, sparse):
        model = EmpiricalBayesGLM(
            alpha=2.0,
            link=link,
            n_eb_iter=3,
            eb_tol=1e-2,
            sparse=sparse,
            alpha_prior_strength=0.25,
        )
        params = model.get_params()
        assert params["n_eb_iter"] == 3 and params["eb_tol"] == 1e-2
        assert params["link"] == link
        assert params["alpha_prior_strength"] == 0.25
        model.set_params(n_eb_iter=7)
        cloned = cast(EmpiricalBayesGLM, clone(model))
        assert cloned.n_eb_iter == 7
        assert cloned.alpha_prior_strength == 0.25

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

    def test_pickle_after_online_update(self, link, sparse):
        X, y = _simulate(link)
        model = EmpiricalBayesGLM(
            link=link, sparse=sparse, forgetting=StabilizedForgetting(0.99)
        )
        model.fit(_X(X, sparse), y)
        model.partial_fit(_X(X[:50], sparse), y[:50])
        model.decay(decay_rate=0.99, steps=5)
        restored = pickle.loads(pickle.dumps(model))
        assert "_factor_hint" not in restored.__dict__
        np.testing.assert_allclose(
            restored.predict(_X(X[:5], sparse)), model.predict(_X(X[:5], sparse))
        )

    def test_failed_update_leaves_no_stale_factor(self, link, sparse):
        X, y = _simulate(link)
        model = EmpiricalBayesGLM(link=link, sparse=sparse).fit(_X(X, sparse), y)
        before = model.predict(_X(X[:5], sparse))
        with mock.patch.object(
            model.approximator_, "update_posterior", side_effect=RuntimeError
        ):
            with pytest.raises(RuntimeError):
                model.partial_fit(_X(X[:10], sparse), y[:10])
        if sparse:
            assert "_precision_factor" not in model.__dict__
        np.testing.assert_allclose(model.predict(_X(X[:5], sparse)), before)
        model.sample(_X(X[:5], sparse))

    def test_failed_fit_does_not_leave_partial_stats(self, link, sparse):
        X, y = _simulate(link)
        model = EmpiricalBayesGLM(link=link, sparse=sparse)
        with mock.patch.object(model, "_fit_helper", side_effect=RuntimeError):
            with pytest.raises(RuntimeError):
                model.fit(_X(X, sparse), y)
        assert not hasattr(model, "_effective_n")
        model.decay(decay_rate=0.9, steps=5)


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

    def test_a_rejected_step_during_fit_is_counted(self):
        X, y = _simulate("logit", n=50, p=5)
        model = EmpiricalBayesGLM(link="logit", n_eb_iter=3)
        with mock.patch(
            "bayesianbandits._eb_estimators.mackay_update_glm",
            return_value=mock.Mock(
                alpha=1.0,
                alpha_min=1e-6,
                alpha_max=1e6,
                log_evidence=-1.0,
                rejected=True,
            ),
        ):
            model.fit(X, y)
        # Constant evidence converges on the second step.
        assert model.eb_updates_rejected_ == 2
        assert model.alpha == 1.0
        model.fit(X, y)
        assert model.eb_updates_rejected_ == 0


class TestFailedPartialFitLeavesEBStateIntact:
    """A ``partial_fit`` that raises must not have moved the EB bookkeeping."""

    @staticmethod
    def _fitted(sparse):
        X, y = _simulate("log", n=40, p=4)
        model = EmpiricalBayesGLM(
            link="log", forgetting=StabilizedForgetting(0.9), sparse=sparse
        )
        return model.fit(_X(X, sparse), y), X, y

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


class TestEffectiveN:
    def test_counts_the_effective_row_weights(self):
        X, y = _simulate("log", n=4, p=3)
        w = np.array([2.0, 3.0, 1.0, 4.0])
        model = EmpiricalBayesGLM(link="log", forgetting=StabilizedForgetting(0.9)).fit(
            X, y, w
        )
        assert model._effective_n == pytest.approx(
            np.sum(w * 0.9 ** np.arange(3, -1, -1))
        )


@pytest.mark.parametrize("sparse", [True, False])
def test_alpha_untouched_when_laplace_does_not_converge(sparse):
    from sklearn.exceptions import ConvergenceWarning

    X, y = _simulate("log")
    # tol=0 can never be met, so every Laplace update reports non-convergence.
    model = EmpiricalBayesGLM(
        link="log",
        alpha=1.0,
        sparse=sparse,
        approximator=LaplaceApproximator(n_iter=2, tol=0.0),
    )
    with pytest.warns(ConvergenceWarning):
        model.fit(_X(X, sparse), y)
    assert model.alpha == 1.0
    assert not model.eb_converged_
    assert model.n_eb_iterations_ == 0

    with pytest.warns(ConvergenceWarning):
        model.partial_fit(_X(X[:20], sparse), y[:20])
    assert model.alpha == 1.0
