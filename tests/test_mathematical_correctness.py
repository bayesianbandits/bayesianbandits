"""Mathematical correctness tests for all estimators.

Verifies two properties for each estimator:
1. Sufficient statistics are correct after update (fit/partial_fit)
2. Posterior predictive distribution is correct after pull (sample)

All examples are small enough to verify by hand.
"""

from unittest import mock

import numpy as np
import pytest
import scipy.sparse as sp
from numpy.testing import assert_allclose
from scipy.stats import dirichlet as scipy_dirichlet
from scipy.stats import gamma as scipy_gamma

from bayesianbandits import (
    BayesianGLM,
    DirichletClassifier,
    GammaRegressor,
    NormalInverseGammaRegressor,
    NormalRegressor,
)
from bayesianbandits._eb_estimators import EmpiricalBayesNormalRegressor
from bayesianbandits._gaussian import LaplaceApproximator
from bayesianbandits._sparse_bayesian_linear_regression import SparseSolver

# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------

suitesparse_envvar_params = [
    SparseSolver.SUPERLU,
    SparseSolver.CHOLMOD,
]


@pytest.fixture(params=suitesparse_envvar_params, autouse=True)
def suitesparse_envvar(request):
    """Run each test with both CHOLMOD and SuperLU sparse solvers."""
    with mock.patch(
        "bayesianbandits._sparse_bayesian_linear_regression.solver", request.param
    ):
        yield


def _cov_inv_dense(est):
    """Return cov_inv_ as a full symmetric dense array.

    Dense mode stores only the upper triangle for speed; this helper
    mirrors it so tests can compare against the full expected matrix.
    """
    if sp.issparse(est.cov_inv_):
        return est.cov_inv_.toarray()
    A = np.array(est.cov_inv_)
    # Mirror upper triangle to lower
    return A + np.triu(A, k=1).T


# ===========================================================================
# 1. DirichletClassifier
# ===========================================================================


class TestDirichletClassifierMath:
    def test_sufficient_stats_unweighted(self):
        """prior {0:1, 1:1}, observe y=[0,0,1] at x=1 => alpha=[3, 2]."""
        clf = DirichletClassifier({0: 1, 1: 1}, random_state=0)
        X = np.array([[1], [1], [1]])
        y = np.array([0, 0, 1])
        clf.fit(X, y)

        assert_allclose(clf.known_alphas_[1], [3.0, 2.0], atol=1e-10)

    def test_sufficient_stats_weighted(self):
        """Weighted update: weight goes to the observed class."""
        clf = DirichletClassifier({0: 1, 1: 1}, random_state=0)
        X = np.array([[1], [1]])
        y = np.array([0, 1])
        clf.fit(X, y, sample_weight=np.array([3.0, 1.5]))

        assert_allclose(clf.known_alphas_[1], [4.0, 2.5], atol=1e-10)

    def test_sufficient_stats_incremental_equals_batch(self):
        """partial_fit one-at-a-time == batch fit."""
        X = np.array([[1], [1], [1]])
        y = np.array([0, 0, 1])

        batch = DirichletClassifier({0: 1, 1: 1}, random_state=0)
        batch.fit(X, y)

        seq = DirichletClassifier({0: 1, 1: 1}, random_state=0)
        for i in range(len(y)):
            seq.partial_fit(X[i : i + 1], y[i : i + 1])

        assert_allclose(seq.known_alphas_[1], batch.known_alphas_[1], atol=1e-10)

    def test_posterior_predictive_moments(self):
        """Dirichlet(3,2) moments: mean=[0.6, 0.4], var=[0.04, 0.04]."""
        clf = DirichletClassifier({0: 1, 1: 1}, random_state=42)
        X = np.array([[1], [1], [1]])
        y = np.array([0, 0, 1])
        clf.fit(X, y)

        samples = clf.sample(np.array([[1]]), size=50_000)  # (50000, 1, 2)
        samples = samples[:, 0, :]  # (50000, 2)

        alpha = np.array([3.0, 2.0])
        S = alpha.sum()
        expected_mean = alpha / S
        expected_var = alpha * (S - alpha) / (S**2 * (S + 1))

        assert_allclose(samples.mean(axis=0), expected_mean, atol=0.01)
        assert_allclose(samples.var(axis=0), expected_var, atol=0.01)

    def test_posterior_predictive_scipy_reference(self):
        """Samples match scipy.stats.dirichlet with the same RNG state."""
        clf = DirichletClassifier({0: 1, 1: 1}, random_state=0)
        X = np.array([[1], [1], [1]])
        y = np.array([0, 0, 1])
        clf.fit(X, y)

        alpha = clf.known_alphas_[1].copy()

        # Reset RNG to a known state for both
        rng = np.random.default_rng(99)
        clf.random_state_ = rng
        estimator_samples = clf.sample(np.array([[1]]), size=100)[:, 0, :]

        rng2 = np.random.default_rng(99)
        ref_samples = scipy_dirichlet.rvs(alpha, size=100, random_state=rng2)

        assert_allclose(estimator_samples, ref_samples, atol=1e-10)


# ===========================================================================
# 2. GammaRegressor
# ===========================================================================


class TestGammaRegressorMath:
    def test_sufficient_stats_unweighted(self):
        """prior (2,3), observe y=[1,3,5] => alpha=2+1+3+5=11, beta=3+3=6."""
        model = GammaRegressor(alpha=2, beta=3, random_state=0)
        X = np.array([[1], [1], [1]])
        y = np.array([1, 3, 5])
        model.fit(X, y)

        assert_allclose(model.coef_[1], [11.0, 6.0], atol=1e-10)

    def test_sufficient_stats_weighted(self):
        """Weighted: alpha += w*y, beta += w."""
        model = GammaRegressor(alpha=2, beta=3, random_state=0)
        X = np.array([[1], [1]])
        y = np.array([2, 4])
        model.fit(X, y, sample_weight=np.array([0.5, 2.0]))

        # alpha = 2 + 0.5*2 + 2.0*4 = 2 + 1 + 8 = 11
        # beta  = 3 + 0.5 + 2.0 = 5.5
        assert_allclose(model.coef_[1], [11.0, 5.5], atol=1e-10)

    def test_sufficient_stats_incremental_equals_batch(self):
        """partial_fit one-at-a-time == batch fit."""
        X = np.array([[1], [1], [1]])
        y = np.array([1, 3, 5])

        batch = GammaRegressor(alpha=2, beta=3, random_state=0)
        batch.fit(X, y)

        seq = GammaRegressor(alpha=2, beta=3, random_state=0)
        for i in range(len(y)):
            seq.partial_fit(X[i : i + 1], y[i : i + 1])

        assert_allclose(seq.coef_[1], batch.coef_[1], atol=1e-10)

    def test_posterior_predictive_moments(self):
        """Gamma(11, 6): mean=11/6, var=11/36."""
        model = GammaRegressor(alpha=2, beta=3, random_state=42)
        X = np.array([[1], [1], [1]])
        y = np.array([1, 3, 5])
        model.fit(X, y)

        samples = model.sample(np.array([[1]]), size=50_000)  # (50000, 1)
        samples = samples[:, 0]

        assert_allclose(samples.mean(), 11.0 / 6, atol=0.05)
        assert_allclose(samples.var(), 11.0 / 36, atol=0.05)

    def test_posterior_predictive_scipy_reference(self):
        """Samples match scipy.stats.gamma with the same RNG state."""
        model = GammaRegressor(alpha=2, beta=3, random_state=0)
        X = np.array([[1], [1], [1]])
        y = np.array([1, 3, 5])
        model.fit(X, y)

        alpha, beta = model.coef_[1]

        rng = np.random.default_rng(99)
        model.random_state_ = rng
        estimator_samples = model.sample(np.array([[1]]), size=100)[:, 0]

        rng2 = np.random.default_rng(99)
        ref_samples = scipy_gamma.rvs(
            alpha, scale=1 / beta, size=100, random_state=rng2
        )

        assert_allclose(estimator_samples, ref_samples, atol=1e-10)


# ===========================================================================
# 3. NormalRegressor
# ===========================================================================


@pytest.mark.parametrize("sparse", [True, False])
class TestNormalRegressorMath:
    def test_sufficient_stats_identity_design(self, sparse):
        """X=I, y=[3,5], alpha=1, beta=2.

        Lambda_n = alpha*I + beta*X^T X = I + 2*I = 3*I
        mu_n = Lambda_n^{-1} (alpha*0 + beta*X^T y) = (1/3)*2*[3,5] = [2, 10/3]
        """
        X = np.array([[1.0, 0.0], [0.0, 1.0]])
        y = np.array([3.0, 5.0])
        if sparse:
            X = sp.csc_array(X)

        reg = NormalRegressor(alpha=1, beta=2, sparse=sparse, random_state=0)
        reg.fit(X, y)

        assert_allclose(reg.coef_, [2.0, 10.0 / 3], atol=1e-10)
        assert_allclose(_cov_inv_dense(reg), [[3.0, 0.0], [0.0, 3.0]], atol=1e-10)

    def test_sufficient_stats_correlated_design(self, sparse):
        """X=[[1,1]], y=[4], alpha=1, beta=2.

        X^T X = [[1,1],[1,1]], Lambda_n = I + 2*[[1,1],[1,1]] = [[3,2],[2,3]]
        mu_n = [[3,2],[2,3]]^{-1} @ [8,8] = (1/5)*[[3,-2],[-2,3]] @ [8,8] = [1.6, 1.6]
        """
        X = np.array([[1.0, 1.0]])
        y = np.array([4.0])
        if sparse:
            X = sp.csc_array(X)

        reg = NormalRegressor(alpha=1, beta=2, sparse=sparse, random_state=0)
        reg.fit(X, y)

        assert_allclose(reg.coef_, [1.6, 1.6], atol=1e-10)
        assert_allclose(_cov_inv_dense(reg), [[3.0, 2.0], [2.0, 3.0]], atol=1e-10)

    def test_posterior_predictive_moments(self, sparse):
        """From identity design: X_test=[1,0] => E[y]=2, Var[y]=1/3."""
        X = np.array([[1.0, 0.0], [0.0, 1.0]])
        y = np.array([3.0, 5.0])
        X_test = np.array([[1.0, 0.0]])
        if sparse:
            X_fit = sp.csc_array(X)
            X_test_s = sp.csc_array(X_test)
        else:
            X_fit = X
            X_test_s = X_test

        reg = NormalRegressor(alpha=1, beta=2, sparse=sparse, random_state=42)
        reg.fit(X_fit, y)

        samples = reg.sample(X_test_s, size=100_000)[:, 0]

        assert_allclose(samples.mean(), 2.0, atol=0.01)
        assert_allclose(samples.var(), 1.0 / 3, atol=0.01)

    def test_posterior_predictive_empirical_covariance(self, sparse):
        """Use X_test=I to recover weight samples; check full covariance.

        With X_test = I, sample(X_test) returns w samples directly.
        Empirical cov should match inv(cov_inv_).
        """
        X = np.array([[1.0, 0.0], [0.0, 1.0]])
        y = np.array([3.0, 5.0])
        if sparse:
            X_fit = sp.csc_array(X)
            X_test = sp.csc_array(np.eye(2))
        else:
            X_fit = X
            X_test = np.eye(2)

        reg = NormalRegressor(alpha=1, beta=2, sparse=sparse, random_state=42)
        reg.fit(X_fit, y)

        # (100000, 2) — each row is a weight sample
        weight_samples = reg.sample(X_test, size=100_000)

        theoretical_mean = reg.coef_
        theoretical_cov = np.linalg.inv(_cov_inv_dense(reg))

        assert_allclose(weight_samples.mean(axis=0), theoretical_mean, atol=0.01)
        assert_allclose(np.cov(weight_samples.T), theoretical_cov, atol=0.01)


# ===========================================================================
# 4. NormalInverseGammaRegressor
# ===========================================================================


@pytest.mark.parametrize("sparse", [True, False])
class TestNormalInverseGammaRegressorMath:
    def test_sufficient_stats_unweighted(self, sparse):
        """X=[[1],[2],[3]], y=[2,4,6], mu=0, lam=1, a=2, b=1.

        X^T X = 14, Lambda_n = 1 + 14 = 15
        mu_n = 15^{-1} * (0 + 28) = 28/15
        a_n = 2 + 3/2 = 3.5
        b_n = 1 + 0.5*(56 + 0 - (28/15)*15*(28/15))
             = 1 + 0.5*(56 - 784/15)
             = 1 + 0.5*(56/15)
             = 1 + 28/15 = 43/15
        """
        X = np.array([[1.0], [2.0], [3.0]])
        y = np.array([2.0, 4.0, 6.0])
        if sparse:
            X = sp.csc_array(X)

        reg = NormalInverseGammaRegressor(
            mu=0.0, lam=1.0, a=2.0, b=1.0, sparse=sparse, random_state=0
        )
        reg.fit(X, y)

        assert_allclose(reg.coef_[0], 28.0 / 15, atol=1e-10)
        assert_allclose(_cov_inv_dense(reg), [[15.0]], atol=1e-10)
        assert_allclose(reg.a_, 3.5, atol=1e-10)
        assert_allclose(reg.b_, 43.0 / 15, atol=1e-10)

    def test_posterior_predictive_moments(self, sparse):
        """Marginal over w is t_{2a}(mu_n, (b/a)/lam_n).

        df=7, loc=28/15, scale^2 = (43/15)/(3.5*15) = 43/787.5
        Mean = 28/15, Var = scale^2 * df/(df-2) = (43/787.5)*(7/5)
        """
        X = np.array([[1.0], [2.0], [3.0]])
        y = np.array([2.0, 4.0, 6.0])
        X_test = np.array([[1.0]])
        if sparse:
            X_fit = sp.csc_array(X)
            X_test_s = sp.csc_array(X_test)
        else:
            X_fit = X
            X_test_s = X_test

        reg = NormalInverseGammaRegressor(
            mu=0.0, lam=1.0, a=2.0, b=1.0, sparse=sparse, random_state=42
        )
        reg.fit(X_fit, y)

        samples = reg.sample(X_test_s, size=50_000)[:, 0]

        expected_mean = 28.0 / 15
        scale_sq = (43.0 / 15) / (3.5 * 15)  # b/a / lam_n
        df = 7.0
        expected_var = scale_sq * df / (df - 2)

        assert_allclose(samples.mean(), expected_mean, atol=0.05)
        assert_allclose(samples.var(), expected_var, atol=0.05)

    def test_posterior_predictive_kurtosis(self, sparse):
        """t-distribution has heavier tails than Gaussian; verify via kurtosis.

        For t(df), excess kurtosis = 6/(df-4) when df>4. Here df=7, so
        excess kurtosis = 6/3 = 2.0. A Gaussian would have 0.
        """
        X = np.array([[1.0], [2.0], [3.0]])
        y = np.array([2.0, 4.0, 6.0])
        if sparse:
            X_fit = sp.csc_array(X)
        else:
            X_fit = X

        reg = NormalInverseGammaRegressor(
            mu=0.0, lam=1.0, a=2.0, b=1.0, sparse=sparse, random_state=42
        )
        reg.fit(X_fit, y)

        df = 2 * reg.a_  # 7
        loc = reg.coef_[0]
        scale_sq = reg.b_ / (reg.a_ * _cov_inv_dense(reg)[0, 0])
        expected_var = scale_sq * df / (df - 2)
        expected_kurtosis = 6.0 / (df - 4)  # excess kurtosis for t(df)

        X_test = np.array([[1.0]])
        if sparse:
            X_test = sp.csc_array(X_test)
        samples = reg.sample(X_test, size=200_000)[:, 0]

        from scipy.stats import kurtosis

        assert_allclose(samples.mean(), loc, atol=0.02)
        assert_allclose(samples.var(), expected_var, atol=0.02)
        assert_allclose(kurtosis(samples), expected_kurtosis, atol=0.2)


# ===========================================================================
# 5. BayesianGLM
# ===========================================================================


@pytest.mark.parametrize("sparse", [True, False])
class TestBayesianGLMMath:
    def test_sufficient_stats_logit_single_irls_step(self, sparse):
        """One IRLS step from w=0 with logit link, alpha=1.

        X = [[-2],[-1],[1],[2]], y = [0,0,1,1]
        mu = sigmoid(0) = 0.5 for all, W = diag(0.25)
        z = eta + (y - mu) / (mu*(1-mu)) = 0 + (y-0.5)/0.25 = [-2,-2,2,2]
        X^T W X = 0.25*(4+1+1+4) = 2.5
        Lambda = alpha*I + X^T W X = 1 + 2.5 = 3.5
        X^T (W*z) = [[-2,-1,1,2]] @ (0.25*[-2,-2,2,2]) = [[-2,-1,1,2]] @ [-0.5,-0.5,0.5,0.5]
                   = 1 + 0.5 + 0.5 + 1 = 3
        coef_ = 3 / 3.5 = 6/7
        """
        X = np.array([[-2.0], [-1.0], [1.0], [2.0]])
        y = np.array([0.0, 0.0, 1.0, 1.0])
        if sparse:
            X = sp.csc_array(X)

        glm = BayesianGLM(
            alpha=1.0,
            link="logit",
            approximator=LaplaceApproximator(n_iter=1),
            sparse=sparse,
            random_state=0,
        )
        glm.fit(X, y)

        assert_allclose(glm.coef_[0], 6.0 / 7, atol=1e-10)
        assert_allclose(_cov_inv_dense(glm)[0, 0], 3.5, atol=1e-10)

    def test_sufficient_stats_log_single_irls_step(self, sparse):
        """One IRLS step from w=0 with log link, alpha=1.

        X = [[1],[2]], y = [3,7]
        mu = exp(0) = 1, dmu/deta = mu = 1, W = diag(1,1)
        z = 0 + (y-1)/1 = [2, 6]
        X^T W X = 1 + 4 = 5
        Lambda = 1 + 5 = 6
        X^T (W*z) = 1*2 + 2*6 = 14
        coef_ = 14/6 = 7/3
        """
        X = np.array([[1.0], [2.0]])
        y = np.array([3.0, 7.0])
        if sparse:
            X = sp.csc_array(X)

        glm = BayesianGLM(
            alpha=1.0,
            link="log",
            approximator=LaplaceApproximator(n_iter=1),
            sparse=sparse,
            random_state=0,
        )
        glm.fit(X, y)

        assert_allclose(glm.coef_[0], 7.0 / 3, atol=1e-10)
        assert_allclose(_cov_inv_dense(glm)[0, 0], 6.0, atol=1e-10)

    def test_posterior_predictive_logit_bounded(self, sparse):
        """All logit predictions must be in [0, 1]."""
        X = np.array([[-2.0], [-1.0], [1.0], [2.0]])
        y = np.array([0.0, 0.0, 1.0, 1.0])
        if sparse:
            X_fit = sp.csc_array(X)
        else:
            X_fit = X

        glm = BayesianGLM(alpha=1.0, link="logit", sparse=sparse, random_state=0)
        glm.fit(X_fit, y)

        X_test = np.array([[0.0], [1.0], [-1.0]])
        if sparse:
            X_test = sp.csc_array(X_test)
        samples = glm.sample(X_test, size=10_000)

        assert np.all(samples >= 0.0)
        assert np.all(samples <= 1.0)

    def test_posterior_predictive_log_positive(self, sparse):
        """All log-link predictions must be positive."""
        X = np.array([[1.0], [2.0], [3.0]])
        y = np.array([3.0, 7.0, 12.0])
        if sparse:
            X_fit = sp.csc_array(X)
        else:
            X_fit = X

        glm = BayesianGLM(alpha=1.0, link="log", sparse=sparse, random_state=0)
        glm.fit(X_fit, y)

        X_test = np.array([[1.5], [2.5]])
        if sparse:
            X_test = sp.csc_array(X_test)
        samples = glm.sample(X_test, size=10_000)

        assert np.all(samples > 0.0)

    def test_posterior_predictive_logit_moments(self, sparse):
        """At X_test=[[1]], positive coef => mean prediction > 0.5.

        The data has positive x -> y=1, negative x -> y=0, so the MAP
        coefficient is positive. sigmoid(positive) > 0.5.
        Also at X_test=[[-1]], mean prediction should be < 0.5 by symmetry.
        """
        X = np.array([[-2.0], [-1.0], [1.0], [2.0]])
        y = np.array([0.0, 0.0, 1.0, 1.0])
        if sparse:
            X_fit = sp.csc_array(X)
        else:
            X_fit = X

        glm = BayesianGLM(alpha=1.0, link="logit", sparse=sparse, random_state=42)
        glm.fit(X_fit, y)

        # At x=1: sigmoid(coef * 1) > 0.5 since coef > 0
        X_pos = np.array([[1.0]])
        X_neg = np.array([[-1.0]])
        if sparse:
            X_pos = sp.csc_array(X_pos)
            X_neg = sp.csc_array(X_neg)

        pos_samples = glm.sample(X_pos, size=50_000)[:, 0]
        neg_samples = glm.sample(X_neg, size=50_000)[:, 0]

        assert pos_samples.mean() > 0.5
        assert neg_samples.mean() < 0.5
        # Symmetric data + symmetric prior => predictions should mirror
        assert_allclose(pos_samples.mean(), 1 - neg_samples.mean(), atol=0.02)


# ===========================================================================
# 6. EmpiricalBayesNormalRegressor
# ===========================================================================


@pytest.mark.parametrize("sparse", [True, False])
class TestEmpiricalBayesNormalRegressorMath:
    def test_sufficient_stats_no_eb(self, sparse):
        """With n_eb_iter=0, should match NormalRegressor exactly.

        Also verify EB-specific sufficient stats.
        """
        X = np.array([[1.0], [2.0], [3.0]])
        y = np.array([1.0, 2.0, 3.0])
        if sparse:
            X_s = sp.csc_array(X)
        else:
            X_s = X

        eb = EmpiricalBayesNormalRegressor(
            alpha=1.0, beta=1.0, n_eb_iter=0, sparse=sparse, random_state=0
        )
        eb.fit(X_s, y)

        nr = NormalRegressor(alpha=1.0, beta=1.0, sparse=sparse, random_state=0)
        nr.fit(X_s, y)

        assert_allclose(eb.coef_, nr.coef_, atol=1e-10)
        assert_allclose(_cov_inv_dense(eb), _cov_inv_dense(nr), atol=1e-10)

        # EB-specific sufficient stats
        assert_allclose(eb._prior_scalar, 1.0, atol=1e-10)
        assert_allclose(eb._effective_n, 3.0, atol=1e-10)
        assert_allclose(eb._eff_yTy, 14.0, atol=1e-10)  # 1+4+9
        assert_allclose(eb._eff_XTy, [14.0], atol=1e-10)  # 1*1+2*2+3*3

    def test_sufficient_stats_eb_consistency(self, sparse):
        """After EB convergence, cov_inv_ ~ alpha*I + beta*X^T X."""
        X = np.array([[1.0, 0.0], [0.0, 1.0], [1.0, 1.0]])
        y = np.array([2.0, 3.0, 5.0])
        if sparse:
            X_s = sp.csc_array(X)
        else:
            X_s = X

        eb = EmpiricalBayesNormalRegressor(
            alpha=1.0, beta=1.0, n_eb_iter=50, sparse=sparse, random_state=0
        )
        eb.fit(X_s, y)

        assert eb.eb_converged_

        XtX = X.T @ X  # always dense for reference calculation
        expected_precision = eb.alpha * np.eye(X.shape[1]) + eb.beta * XtX
        assert_allclose(_cov_inv_dense(eb), expected_precision, atol=1e-6)

    def test_posterior_predictive_matches_normal_regressor(self, sparse):
        """With n_eb_iter=0 and same seed, samples should be identical."""
        X = np.array([[1.0], [2.0], [3.0]])
        y = np.array([1.0, 2.0, 3.0])
        X_test = np.array([[1.5]])
        if sparse:
            X_s = sp.csc_array(X)
            X_test_s = sp.csc_array(X_test)
        else:
            X_s = X
            X_test_s = X_test

        eb = EmpiricalBayesNormalRegressor(
            alpha=1.0, beta=1.0, n_eb_iter=0, sparse=sparse, random_state=0
        )
        eb.fit(X_s, y)

        nr = NormalRegressor(alpha=1.0, beta=1.0, sparse=sparse, random_state=0)
        nr.fit(X_s, y)

        # Reset both to the same RNG state
        eb.random_state_ = np.random.default_rng(42)
        nr.random_state_ = np.random.default_rng(42)

        eb_samples = eb.sample(X_test_s, size=100)
        nr_samples = nr.sample(X_test_s, size=100)

        assert_allclose(eb_samples, nr_samples, atol=1e-10)


# ===========================================================================
# 7. Decay correctness
# ===========================================================================


class TestDirichletClassifierDecay:
    def test_decay_scales_concentration(self):
        """decay(gamma) multiplies each alpha_k by gamma."""
        clf = DirichletClassifier({0: 1, 1: 1}, random_state=0)
        X = np.array([[1], [1], [1]])
        y = np.array([0, 0, 1])
        clf.fit(X, y)
        # known_alphas_[1] = [3, 2]

        gamma = 0.8
        clf.decay(np.array([[1]]), decay_rate=gamma)

        assert_allclose(clf.known_alphas_[1], [3.0 * gamma, 2.0 * gamma], atol=1e-10)

    def test_decay_preserves_mean(self):
        """Dirichlet mean alpha_k / sum(alpha) is invariant under uniform scaling."""
        clf = DirichletClassifier({0: 1, 1: 1}, random_state=0)
        X = np.array([[1], [1], [1]])
        y = np.array([0, 0, 1])
        clf.fit(X, y)

        mean_before = clf.known_alphas_[1] / clf.known_alphas_[1].sum()
        clf.decay(np.array([[1]]), decay_rate=0.7)
        mean_after = clf.known_alphas_[1] / clf.known_alphas_[1].sum()

        assert_allclose(mean_after, mean_before, atol=1e-10)


class TestGammaRegressorDecay:
    def test_decay_scales_both_params(self):
        """decay(gamma) multiplies both alpha and beta by gamma."""
        model = GammaRegressor(alpha=2, beta=3, random_state=0)
        X = np.array([[1], [1], [1]])
        y = np.array([1, 3, 5])
        model.fit(X, y)
        # coef_[1] = [11, 6]

        gamma = 0.9
        model.decay(np.array([[1]]), decay_rate=gamma)

        assert_allclose(model.coef_[1], [11.0 * gamma, 6.0 * gamma], atol=1e-10)

    def test_decay_preserves_mean(self):
        """Gamma mean alpha/beta is invariant under uniform scaling."""
        model = GammaRegressor(alpha=2, beta=3, random_state=0)
        X = np.array([[1], [1], [1]])
        y = np.array([1, 3, 5])
        model.fit(X, y)

        mean_before = model.coef_[1][0] / model.coef_[1][1]
        model.decay(np.array([[1]]), decay_rate=0.7)
        mean_after = model.coef_[1][0] / model.coef_[1][1]

        assert_allclose(mean_after, mean_before, atol=1e-10)


@pytest.mark.parametrize("sparse", [True, False])
class TestNormalRegressorDecay:
    def test_decay_scales_precision(self, sparse):
        """decay with n rows scales cov_inv_ by gamma^n, mean unchanged."""
        X = np.array([[1.0, 0.0], [0.0, 1.0]])
        y = np.array([3.0, 5.0])
        if sparse:
            X = sp.csc_array(X)

        reg = NormalRegressor(alpha=1, beta=2, sparse=sparse, random_state=0)
        reg.fit(X, y)

        coef_before = reg.coef_.copy()
        prec_before = _cov_inv_dense(reg).copy()

        gamma = 0.8
        # Decay with 2 rows => factor = gamma^2
        X_decay = np.array([[0.0, 0.0], [0.0, 0.0]])
        if sparse:
            X_decay = sp.csc_array(X_decay)
        reg.decay(X_decay, decay_rate=gamma)

        factor = gamma**2
        assert_allclose(reg.coef_, coef_before, atol=1e-10)  # mean unchanged
        assert_allclose(_cov_inv_dense(reg), factor * prec_before, atol=1e-10)

    def test_decay_single_row(self, sparse):
        """decay with 1 row scales cov_inv_ by gamma."""
        X = np.array([[1.0, 0.0], [0.0, 1.0]])
        y = np.array([3.0, 5.0])
        if sparse:
            X = sp.csc_array(X)

        reg = NormalRegressor(alpha=1, beta=2, sparse=sparse, random_state=0)
        reg.fit(X, y)

        prec_before = _cov_inv_dense(reg).copy()

        gamma = 0.5
        X_decay = np.array([[0.0, 0.0]])
        if sparse:
            X_decay = sp.csc_array(X_decay)
        reg.decay(X_decay, decay_rate=gamma)

        assert_allclose(_cov_inv_dense(reg), gamma * prec_before, atol=1e-10)


@pytest.mark.parametrize("sparse", [True, False])
class TestNIGDecay:
    def test_decay_scales_all_params(self, sparse):
        """decay scales cov_inv_, a_, b_ by gamma^n. Mean unchanged."""
        X = np.array([[1.0], [2.0], [3.0]])
        y = np.array([2.0, 4.0, 6.0])
        if sparse:
            X = sp.csc_array(X)

        reg = NormalInverseGammaRegressor(
            mu=0.0, lam=1.0, a=2.0, b=1.0, sparse=sparse, random_state=0
        )
        reg.fit(X, y)

        coef_before = reg.coef_.copy()
        prec_before = _cov_inv_dense(reg).copy()
        a_before = reg.a_
        b_before = reg.b_

        gamma = 0.9
        X_decay = np.array([[0.0]])
        if sparse:
            X_decay = sp.csc_array(X_decay)
        reg.decay(X_decay, decay_rate=gamma)

        assert_allclose(reg.coef_, coef_before, atol=1e-10)
        assert_allclose(_cov_inv_dense(reg), gamma * prec_before, atol=1e-10)
        assert_allclose(reg.a_, gamma * a_before, atol=1e-10)
        assert_allclose(reg.b_, gamma * b_before, atol=1e-10)


@pytest.mark.parametrize("sparse", [True, False])
class TestBayesianGLMDecay:
    def test_decay_scales_precision(self, sparse):
        """decay scales cov_inv_ by gamma^n. Mean unchanged."""
        X = np.array([[-2.0], [-1.0], [1.0], [2.0]])
        y = np.array([0.0, 0.0, 1.0, 1.0])
        if sparse:
            X = sp.csc_array(X)

        glm = BayesianGLM(alpha=1.0, link="logit", sparse=sparse, random_state=0)
        glm.fit(X, y)

        coef_before = glm.coef_.copy()
        prec_before = _cov_inv_dense(glm).copy()

        gamma = 0.75
        X_decay = np.array([[0.0]])
        if sparse:
            X_decay = sp.csc_array(X_decay)
        glm.decay(X_decay, decay_rate=gamma)

        assert_allclose(glm.coef_, coef_before, atol=1e-10)
        assert_allclose(_cov_inv_dense(glm), gamma * prec_before, atol=1e-10)


@pytest.mark.parametrize("sparse", [True, False])
class TestEBDecay:
    def test_decay_stabilized_forgetting(self, sparse):
        """EB decay: _prior_scalar follows the stabilized forgetting formula.

        _prior_scalar_new = gamma * _prior_scalar_old + (1 - gamma) * alpha
        _effective_n, _eff_yTy, _eff_XTy all scaled by gamma.

        To make this non-trivial, we manually set _prior_scalar != alpha
        before decaying (simulating what happens after EB tunes alpha).
        """
        X = np.array([[1.0], [2.0], [3.0]])
        y = np.array([1.0, 2.0, 3.0])
        if sparse:
            X_s = sp.csc_array(X)
        else:
            X_s = X

        eb = EmpiricalBayesNormalRegressor(
            alpha=1.0, beta=1.0, n_eb_iter=0, sparse=sparse, random_state=0
        )
        eb.fit(X_s, y)

        # Simulate a state where _prior_scalar has drifted from alpha
        # (as happens after EB updates alpha)
        eb._prior_scalar = 5.0

        prior_scalar_before = eb._prior_scalar
        eff_n_before = eb._effective_n
        eff_yTy_before = eb._eff_yTy
        eff_XTy_before = eb._eff_XTy.copy()

        gamma = 0.8
        X_decay = np.array([[0.0]])
        if sparse:
            X_decay = sp.csc_array(X_decay)
        eb.decay(X_decay, decay_rate=gamma)

        # _prior_scalar: 0.8 * 5.0 + 0.2 * 1.0 = 4.2
        expected_prior_scalar = gamma * prior_scalar_before + (1 - gamma) * eb.alpha
        assert_allclose(eb._prior_scalar, expected_prior_scalar, atol=1e-10)
        assert expected_prior_scalar != prior_scalar_before  # non-trivial check
        assert_allclose(eb._effective_n, gamma * eff_n_before, atol=1e-10)
        assert_allclose(eb._eff_yTy, gamma * eff_yTy_before, atol=1e-10)
        assert_allclose(eb._eff_XTy, gamma * eff_XTy_before, atol=1e-10)

    def test_repeated_decay_prior_scalar_converges(self, sparse):
        """From a non-fixed-point start, _prior_scalar converges to alpha.

        Recurrence: s_new = gamma * s_old + (1 - gamma) * alpha
        Fixed point: s = alpha. Starting from s=10.0, should converge.
        """
        X = np.array([[1.0], [2.0]])
        y = np.array([1.0, 2.0])
        if sparse:
            X_s = sp.csc_array(X)
        else:
            X_s = X

        eb = EmpiricalBayesNormalRegressor(
            alpha=2.0, beta=1.0, n_eb_iter=0, sparse=sparse, random_state=0
        )
        eb.fit(X_s, y)

        # Start far from fixed point
        eb._prior_scalar = 10.0

        X_decay = np.array([[0.0]])
        if sparse:
            X_decay = sp.csc_array(X_decay)

        for _ in range(200):
            eb.decay(X_decay, decay_rate=0.9)

        # Fixed point of s = 0.9*s + 0.1*2.0 is s = 2.0 = alpha
        assert_allclose(eb._prior_scalar, eb.alpha, atol=1e-4)


# ===========================================================================
# 8. Batch equals sequential (cross-cutting)
# ===========================================================================


class TestBatchEqualsSequentialDirichlet:
    def test_batch_vs_sequential(self):
        X = np.array([[1], [1], [1], [2], [2]])
        y = np.array([0, 0, 1, 1, 1])

        batch = DirichletClassifier({0: 1, 1: 1}, random_state=0)
        batch.fit(X, y)

        seq = DirichletClassifier({0: 1, 1: 1}, random_state=0)
        for i in range(len(y)):
            seq.partial_fit(X[i : i + 1], y[i : i + 1])

        for key in batch.known_alphas_:
            assert_allclose(
                seq.known_alphas_[key], batch.known_alphas_[key], atol=1e-10
            )


class TestBatchEqualsSequentialGamma:
    def test_batch_vs_sequential(self):
        X = np.array([[1], [1], [2], [2]])
        y = np.array([3, 5, 2, 4])

        batch = GammaRegressor(alpha=2, beta=3, random_state=0)
        batch.fit(X, y)

        seq = GammaRegressor(alpha=2, beta=3, random_state=0)
        for i in range(len(y)):
            seq.partial_fit(X[i : i + 1], y[i : i + 1])

        for key in batch.coef_:
            assert_allclose(seq.coef_[key], batch.coef_[key], atol=1e-10)


@pytest.mark.parametrize("sparse", [True, False])
class TestBatchEqualsSequentialNormal:
    def test_batch_vs_sequential(self, sparse):
        X = np.array([[1.0, 0.0], [0.0, 1.0], [1.0, 1.0]])
        y = np.array([2.0, 3.0, 5.0])
        if sparse:
            X = sp.csc_array(X)

        batch = NormalRegressor(alpha=1, beta=1, sparse=sparse, random_state=0)
        batch.fit(X, y)

        seq = NormalRegressor(alpha=1, beta=1, sparse=sparse, random_state=0)
        for i in range(len(y)):
            if sparse:
                Xi = sp.csc_array(X[i : i + 1].toarray())
            else:
                Xi = X[i : i + 1]
            seq.partial_fit(Xi, y[i : i + 1])

        assert_allclose(seq.coef_, batch.coef_, atol=1e-10)
        assert_allclose(_cov_inv_dense(seq), _cov_inv_dense(batch), atol=1e-10)


@pytest.mark.parametrize("sparse", [True, False])
class TestBatchEqualsSequentialNIG:
    def test_batch_vs_sequential(self, sparse):
        X = np.array([[1.0], [2.0], [3.0]])
        y = np.array([2.0, 4.0, 6.0])
        if sparse:
            X = sp.csc_array(X)

        batch = NormalInverseGammaRegressor(
            mu=0.0, lam=1.0, a=2.0, b=1.0, sparse=sparse, random_state=0
        )
        batch.fit(X, y)

        seq = NormalInverseGammaRegressor(
            mu=0.0, lam=1.0, a=2.0, b=1.0, sparse=sparse, random_state=0
        )
        for i in range(len(y)):
            if sparse:
                Xi = sp.csc_array(X[i : i + 1].toarray())
            else:
                Xi = X[i : i + 1]
            seq.partial_fit(Xi, y[i : i + 1])

        assert_allclose(seq.coef_, batch.coef_, atol=1e-10)
        assert_allclose(_cov_inv_dense(seq), _cov_inv_dense(batch), atol=1e-10)
        assert_allclose(seq.a_, batch.a_, atol=1e-10)
        assert_allclose(seq.b_, batch.b_, atol=1e-10)
