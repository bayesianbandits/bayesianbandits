"""Tests for the GLM MacKay helpers: ``glm_log_likelihood`` and
``mackay_update_glm``.  Everything is checked against brute-force dense
computations on tiny problems."""

from __future__ import annotations

import math

import numpy as np
import pytest
from numpy.typing import NDArray
from scipy.linalg import cho_factor
from scipy.optimize import minimize
from scipy.sparse import csc_array
from scipy.special import expit
from scipy.stats import bernoulli, poisson

from bayesianbandits._empirical_bayes import (
    _MAX_SECANT_LOG_STEP,
    MacKayGLMUpdate,
    SecantRootFinder,
    glm_log_likelihood,
    mackay_update_glm,
    secant_log_alpha,
)
from bayesianbandits._sparse_bayesian_linear_regression import (
    DenseFactor,
    PrecisionFactor,
    create_sparse_factor,
)

# ---------------------------------------------------------------------------
# Brute-force reference implementations
# ---------------------------------------------------------------------------


def _make_dense_factor(precision: NDArray[np.float64]) -> DenseFactor:
    cho = cho_factor(precision, lower=False, check_finite=False)
    return DenseFactor(_U=cho[0], _n_features=cho[0].shape[0])


def _as_input(
    precision: NDArray[np.float64], sparse: bool
) -> tuple[NDArray[np.float64] | csc_array, PrecisionFactor]:
    if sparse:
        prec_in = csc_array(precision)
        return prec_in, create_sparse_factor(prec_in)
    return precision, _make_dense_factor(precision)


def _neg_log_post(
    theta: NDArray[np.float64],
    X: NDArray[np.float64],
    y: NDArray[np.float64],
    alpha: float,
    link: str,
) -> float:
    return -glm_log_likelihood(X, y, theta, link) + 0.5 * alpha * float(theta @ theta)


def _data_hessian(
    X: NDArray[np.float64], theta: NDArray[np.float64], link: str
) -> NDArray[np.float64]:
    eta = X @ theta
    if link == "logit":
        mu = expit(eta)
        w = mu * (1.0 - mu)
    else:
        w = np.exp(eta)
    return np.asarray((X * w[:, None]).T @ X, dtype=np.float64)


def _laplace_posterior(
    X: NDArray[np.float64], y: NDArray[np.float64], alpha: float, link: str
) -> tuple[NDArray[np.float64], NDArray[np.float64]]:
    """MAP by a generic optimizer, Hessian analytically."""
    p = X.shape[1]
    res = minimize(
        _neg_log_post,
        np.zeros(p),
        args=(X, y, alpha, link),
        method="BFGS",
        options={"gtol": 1e-10},
    )
    theta = np.asarray(res.x, dtype=np.float64)
    precision = np.asarray(
        alpha * np.eye(p) + _data_hessian(X, theta, link), dtype=np.float64
    )
    return theta, precision


def _reference_log_evidence(
    X: NDArray[np.float64],
    y: NDArray[np.float64],
    theta: NDArray[np.float64],
    precision: NDArray[np.float64],
    alpha: float,
    link: str,
) -> float:
    """l(theta) + log N(theta | 0, alpha^-1 I) + p/2 log 2pi - 1/2 log|Lambda|."""
    p = X.shape[1]
    log_lik = glm_log_likelihood(X, y, theta, link)
    log_prior = 0.5 * p * np.log(alpha / (2 * np.pi)) - 0.5 * alpha * float(
        theta @ theta
    )
    laplace = 0.5 * p * np.log(2 * np.pi) - 0.5 * np.linalg.slogdet(precision)[1]
    return float(log_lik + log_prior + laplace)


def _simulate(
    link: str, n: int, p: int, seed: int, alpha_true: float = 2.0
) -> tuple[NDArray[np.float64], NDArray[np.float64]]:
    rng = np.random.default_rng(seed)
    X = rng.normal(size=(n, p)) / np.sqrt(p)
    w = rng.normal(scale=alpha_true**-0.5, size=p)
    eta = X @ w
    if link == "logit":
        y = rng.binomial(1, expit(eta)).astype(np.float64)
    else:
        y = rng.poisson(np.exp(eta)).astype(np.float64)
    return X, y


# ---------------------------------------------------------------------------
# glm_log_likelihood
# ---------------------------------------------------------------------------


class TestGLMLogLikelihood:
    def test_logit_matches_bernoulli_logpmf(self) -> None:
        X, y = _simulate("logit", 40, 3, seed=0)
        theta = np.array([0.5, -1.0, 2.0])
        expected = bernoulli.logpmf(y, expit(X @ theta)).sum()
        np.testing.assert_allclose(
            glm_log_likelihood(X, y, theta, "logit"), expected, rtol=1e-12
        )

    def test_log_matches_poisson_logpmf(self) -> None:
        X, y = _simulate("log", 40, 3, seed=1)
        theta = np.array([0.5, -1.0, 0.3])
        expected = poisson.logpmf(y, np.exp(X @ theta)).sum()
        np.testing.assert_allclose(
            glm_log_likelihood(X, y, theta, "log"), expected, rtol=1e-12
        )

    @pytest.mark.parametrize("link", ["logit", "log"])
    def test_sparse_matches_dense(self, link: str) -> None:
        X, y = _simulate(link, 40, 3, seed=2)
        theta = np.array([0.5, -1.0, 0.3])
        np.testing.assert_allclose(
            glm_log_likelihood(csc_array(X), y, theta, link),
            glm_log_likelihood(X, y, theta, link),
            rtol=1e-12,
        )

    def test_sample_weight_scales_rows(self) -> None:
        X, y = _simulate("logit", 10, 2, seed=3)
        theta = np.array([0.2, -0.4])
        sw = np.arange(1, 11, dtype=np.float64)
        X_rep = np.repeat(X, sw.astype(int), axis=0)
        y_rep = np.repeat(y, sw.astype(int))
        np.testing.assert_allclose(
            glm_log_likelihood(X, y, theta, "logit", sample_weight=sw),
            glm_log_likelihood(X_rep, y_rep, theta, "logit"),
            rtol=1e-12,
        )

    def test_unknown_link_raises(self) -> None:
        X, y = _simulate("logit", 5, 2, seed=4)
        with pytest.raises(ValueError, match="Unknown link"):
            glm_log_likelihood(X, y, np.zeros(2), "probit")


# ---------------------------------------------------------------------------
# mackay_update_glm
# ---------------------------------------------------------------------------


@pytest.mark.parametrize("link", ["logit", "log"])
@pytest.mark.parametrize("sparse", [False, True])
class TestMacKayUpdateGLM:
    alpha = 1.5

    def _run(
        self, link: str, sparse: bool, seed: int = 10
    ) -> tuple[
        NDArray[np.float64],
        NDArray[np.float64],
        NDArray[np.float64],
        NDArray[np.float64],
        MacKayGLMUpdate,
    ]:
        X, y = _simulate(link, 60, 4, seed=seed)
        theta, precision = _laplace_posterior(X, y, self.alpha, link)
        log_lik = glm_log_likelihood(X, y, theta, link)
        prec_in, factor = _as_input(precision, sparse)
        update = mackay_update_glm(
            theta,
            prec_in,
            self.alpha,
            prior_scalar=self.alpha,
            effective_n=float(X.shape[0]),
            log_lik=log_lik,
            factor=factor,
        )
        return X, y, theta, precision, update

    def test_alpha_matches_effective_dof(self, link: str, sparse: bool) -> None:
        """alpha_new = tr(Lambda^-1 H_data) / ||theta||^2."""
        X, _, theta, precision, update = self._run(link, sparse)
        H = precision - self.alpha * np.eye(X.shape[1])
        gamma = np.trace(np.linalg.solve(precision, H))
        np.testing.assert_allclose(update.alpha, gamma / (theta @ theta), rtol=1e-8)
        assert not update.rejected

    def test_log_evidence_matches_reference(self, link: str, sparse: bool) -> None:
        X, y, theta, precision, update = self._run(link, sparse)
        expected = _reference_log_evidence(X, y, theta, precision, self.alpha, link)
        np.testing.assert_allclose(update.log_evidence, expected, rtol=1e-10)

    def test_fixed_point_maximizes_laplace_evidence(
        self, link: str, sparse: bool
    ) -> None:
        """Iterating the update converges to the argmax of the Laplace
        evidence as a function of alpha (found here by grid search)."""
        X, y = _simulate(link, 200, 3, seed=11)

        def evidence(a: float) -> float:
            th, prec = _laplace_posterior(X, y, a, link)
            return _reference_log_evidence(X, y, th, prec, a, link)

        alpha = self.alpha
        for _ in range(50):
            theta, precision = _laplace_posterior(X, y, alpha, link)
            prec_in, factor = _as_input(precision, sparse)
            update = mackay_update_glm(
                theta,
                prec_in,
                alpha,
                prior_scalar=alpha,
                effective_n=float(X.shape[0]),
                log_lik=glm_log_likelihood(X, y, theta, link),
                factor=factor,
            )
            if abs(update.alpha - alpha) < 1e-9 * alpha:
                break
            alpha = update.alpha

        grid = np.exp(np.linspace(np.log(alpha) - 1.0, np.log(alpha) + 1.0, 41))
        values = [evidence(a) for a in grid]
        alpha_grid = grid[int(np.argmax(values))]
        # Grid spacing is 5% in log-alpha; the fixed point must be the
        # best grid cell or its neighbor.
        assert abs(np.log(alpha) - np.log(alpha_grid)) < 0.06

    def test_prior_scalar_not_alpha_in_gamma(self, link: str, sparse: bool) -> None:
        """Under forgetting the prior's diagonal contribution is the
        decayed prior_scalar; the update must use it, not alpha."""
        X, y = _simulate(link, 60, 4, seed=12)
        decay = 0.5
        prior_scalar = decay * self.alpha
        theta, _ = _laplace_posterior(X, y, prior_scalar, link)
        H = _data_hessian(X, theta, link)
        precision = np.asarray(prior_scalar * np.eye(X.shape[1]) + H, dtype=np.float64)
        prec_in, factor = _as_input(precision, sparse)
        update = mackay_update_glm(
            theta,
            prec_in,
            self.alpha,
            prior_scalar=prior_scalar,
            effective_n=float(X.shape[0]),
            log_lik=0.0,
            factor=factor,
        )
        gamma = np.trace(np.linalg.solve(precision, H))
        np.testing.assert_allclose(update.alpha, gamma / (theta @ theta), rtol=1e-8)

    def test_diagonal_trace_method_runs(self, link: str, sparse: bool) -> None:
        X, y, theta, precision, exact = self._run(link, sparse)
        prec_in, factor = _as_input(precision, sparse)
        approx = mackay_update_glm(
            theta,
            prec_in,
            self.alpha,
            prior_scalar=self.alpha,
            effective_n=float(X.shape[0]),
            log_lik=0.0,
            factor=factor,
            trace_method="diagonal",
        )
        assert approx.alpha > 0
        assert np.isfinite(approx.log_evidence)
        # Same logdet, different trace: alpha differs but not wildly.
        assert 0.2 < approx.alpha / exact.alpha < 5.0


class TestMacKayUpdateGLMGuardrail:
    def test_separable_underdetermined_is_not_driven_to_zero(self) -> None:
        """p >> n separable logistic data starting from a tiny alpha: the
        gamma <= effective_n cap keeps alpha_new = gamma / ||theta||^2 a
        sane value rather than chasing alpha -> 0, and the
        ill-conditioning guardrail has nothing to reject."""
        rng = np.random.default_rng(0)
        n, p = 5, 50
        X = rng.normal(size=(n, p))
        y = (X[:, 0] > 0).astype(np.float64)  # perfectly separable
        alpha = 1e-9
        theta, precision = _laplace_posterior(X, y, alpha, "logit")
        factor = _make_dense_factor(precision)
        update = mackay_update_glm(
            theta,
            precision,
            alpha,
            prior_scalar=alpha,
            effective_n=float(n),
            log_lik=glm_log_likelihood(X, y, theta, "logit"),
            factor=factor,
        )
        assert not update.rejected
        assert update.alpha > 1e3 * alpha
        assert update.alpha <= n / float(theta @ theta) * (1 + 1e-12)

    def test_ill_conditioned_update_is_rejected(self) -> None:
        """max(diag H_data) / alpha_new above the ceiling keeps alpha."""
        p = 4
        alpha = 1e-3
        theta = np.full(p, 1e6)  # ||theta||^2 = 4e12 -> alpha_new ~ 1e-12
        precision = np.asarray(
            (alpha + 1.0) * np.eye(p), dtype=np.float64
        )  # H_data diagonal = 1
        update = mackay_update_glm(
            theta,
            precision,
            alpha,
            prior_scalar=alpha,
            effective_n=100.0,
            log_lik=0.0,
            factor=_make_dense_factor(precision),
        )
        assert update.rejected
        assert update.alpha == alpha
        assert np.isfinite(update.log_evidence)

    def test_overgrown_alpha_is_rejected(self) -> None:
        """alpha_new above 1e10 x max(diag H_data) keeps alpha."""
        p = 4
        alpha = 1e3
        theta = np.full(p, 1e-8)  # ||theta||^2 = 4e-16 -> alpha_new ~ 1e15
        precision = np.asarray((alpha + 1.0) * np.eye(p), dtype=np.float64)
        update = mackay_update_glm(
            theta,
            precision,
            alpha,
            prior_scalar=alpha,
            effective_n=100.0,
            log_lik=0.0,
            factor=_make_dense_factor(precision),
        )
        assert update.rejected
        assert update.alpha == alpha
        np.testing.assert_allclose(update.alpha_min, 1e-10)
        np.testing.assert_allclose(update.alpha_max, 1e10)

    def test_tiny_gamma_is_not_floored_away(self) -> None:
        """When the prior dominates, gamma ~ tr(H)/alpha is tiny but real,
        and the step that shrinks alpha must go through."""
        p = 4
        alpha = 1e9
        h = 1.0  # H = I
        precision = np.asarray((alpha + h) * np.eye(p), dtype=np.float64)
        # theta = Lambda^-1 eta with ||eta||^2 = 16 > tr(H) = 4: the data
        # argue for a smaller alpha, by the factor tr(H)/||eta||^2 = 1/4.
        eta = np.full(p, 2.0)
        theta = np.asarray(eta / (alpha + h), dtype=np.float64)
        update = mackay_update_glm(
            theta,
            precision,
            alpha,
            prior_scalar=alpha,
            effective_n=100.0,
            log_lik=0.0,
            factor=_make_dense_factor(precision),
        )
        assert not update.rejected
        np.testing.assert_allclose(update.alpha, alpha / 4, rtol=1e-6)

    def test_zero_theta_keeps_alpha(self) -> None:
        p = 3
        precision = np.asarray(2.0 * np.eye(p), dtype=np.float64)
        update = mackay_update_glm(
            np.zeros(p),
            precision,
            2.0,
            prior_scalar=2.0,
            effective_n=10.0,
            log_lik=-1.0,
            factor=_make_dense_factor(precision),
        )
        assert update.alpha == 2.0
        assert not update.rejected

    def test_gamma_capped_by_effective_n(self) -> None:
        """With effective_n < p, gamma cannot exceed effective_n."""
        rng = np.random.default_rng(1)
        p = 6
        theta = rng.normal(size=p)
        precision = np.asarray(
            1e6 * np.eye(p), dtype=np.float64
        )  # all dofs well-determined
        update = mackay_update_glm(
            theta,
            precision,
            1.0,
            prior_scalar=1.0,
            effective_n=2.0,
            log_lik=0.0,
            factor=_make_dense_factor(precision),
        )
        np.testing.assert_allclose(update.alpha, 2.0 / (theta @ theta))


class TestSecantLogAlpha:
    """secant_log_alpha accelerates the fixed-point iteration u <- F(u)
    through the residual h(u) = F(u) - u."""

    def test_linear_fixed_point_is_hit_in_one_step(self) -> None:
        # F(u) = r·u + c has the fixed point c / (1 - r); the residual is
        # linear in u, so the secant lands on it exactly.
        r, c = 0.5, 1.0
        F = lambda u: r * u + c  # noqa: E731
        u0, u1 = 0.0, F(0.0)
        u2, used = secant_log_alpha(u1, F(u1) - u1, u0, F(u0) - u0)
        assert used
        np.testing.assert_allclose(u2, c / (1 - r))

    def test_plain_step_without_contraction(self) -> None:
        # Residual growing with u: slope >= 0, no contraction to exploit.
        u, h = 1.0, 0.5
        assert secant_log_alpha(u, h, 0.0, 0.2) == (u + h, False)

    def test_plain_step_when_history_is_degenerate(self) -> None:
        u, h = 1.0, 0.5
        assert secant_log_alpha(u, h, 1.0, 0.3) == (u + h, False)
        assert secant_log_alpha(u, h, 0.0, math.nan) == (u + h, False)

    def test_never_steps_against_mackay(self) -> None:
        # Negative slope but the root lies behind: h and step disagree in
        # sign only if h changed sign, which the secant then refuses.
        u, h = 1.0, -0.1
        u_next, used = secant_log_alpha(u, h, 0.0, 0.3)
        # slope = -0.4, step = -h/slope = -0.25: same sign as h, accepted
        assert used and u_next == pytest.approx(u - 0.25)
        u_next, used = secant_log_alpha(u, 0.1, 0.0, 0.3)
        # slope = -0.2, step = 0.5 > 0 = sign of h: accepted
        assert used and u_next == pytest.approx(1.5)

    def test_step_is_capped(self) -> None:
        # Nearly equal residuals extrapolate far; the cap bounds it.
        u_next, used = secant_log_alpha(1.0, 0.5, 0.0, 0.5001)
        assert used
        assert u_next == pytest.approx(1.0 + _MAX_SECANT_LOG_STEP)


class TestSecantRootFinder:
    @staticmethod
    def _solve(h, u0, tol=1e-10, max_iter=50):
        finder = SecantRootFinder()
        u = u0
        for k in range(max_iter):
            hk = h(u)
            if abs(hk) <= tol:
                return u, k, finder
            u = finder.next(u, hk)
        raise AssertionError("did not converge")

    def test_first_step_is_plain_mackay(self):
        finder = SecantRootFinder()
        assert finder.next(0.0, 0.3) == 0.3
        assert not finder.bracketed
        assert finder.bracket_width == math.inf

    def test_zero_or_nonfinite_residual_stays_put(self):
        finder = SecantRootFinder()
        assert finder.next(1.0, 0.0) == 1.0
        assert finder.next(1.0, math.nan) == 1.0

    def test_linear_residual_converges_in_two_accelerated_steps(self):
        # F'(u) = 0.9: the plain iteration moves 10% of the way per
        # step; the secant through two points is exact on a line.
        root = 3.0
        u, k, _ = self._solve(lambda u: 0.1 * (root - u), u0=0.0)
        assert abs(u - root) < 1e-9
        assert k <= 3

    def test_bracket_forms_on_sign_change_and_only_shrinks(self):
        root = 1.0
        h = lambda u: math.tanh(root - u)  # noqa: E731
        finder = SecantRootFinder()
        u = -4.0
        widths = []
        for _ in range(30):
            hk = h(u)
            if abs(hk) < 1e-12:
                break
            u = finder.next(u, hk)
            if finder.bracketed:
                widths.append(finder.bracket_width)
        assert finder.bracketed
        assert abs(u - root) < 1e-9
        assert all(b <= a for a, b in zip(widths, widths[1:]))
        assert widths[-1] < 1e-4

    def test_illinois_beats_plain_regula_falsi_on_a_bent_residual(self):
        # h(u) = 1 - e^u with a root at 0 is convex enough that plain
        # regula falsi pins the left endpoint and converges linearly.
        h = lambda u: (1.0 - math.exp(u)) * 0.5  # noqa: E731
        u, k, finder = self._solve(h, u0=-5.0)
        assert abs(u) < 1e-9
        assert finder.bracketed
        assert k < 20

    def test_reset_forgets_the_bracket(self):
        finder = SecantRootFinder()
        finder.next(0.0, 1.0)
        finder.next(2.0, -1.0)
        assert finder.bracketed
        finder.reset()
        assert not finder.bracketed
        assert finder.next(0.0, 0.3) == 0.3

    def test_bracketed_step_lands_inside_the_bracket(self):
        finder = SecantRootFinder()
        finder.next(0.0, 1.0)
        u = finder.next(5.0, -0.01)
        assert 0.0 < u < 5.0
        # A huge residual on one side cannot push the step outside.
        u = finder.next(u, 1e6)
        assert 0.0 < u < 5.0
