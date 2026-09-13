"""``forgetting=``: the per-update forgetting API on the estimators."""

import pickle
from typing import Any

import numpy as np
import pytest
from numpy.testing import assert_allclose
from scipy import sparse as sp

from bayesianbandits import (
    BayesianGLM,
    DirichletClassifier,
    EmpiricalBayesGammaRegressor,
    EmpiricalBayesGLM,
    EmpiricalBayesNormalRegressor,
    ExponentialForgetting,
    FeatureWiseForgetting,
    GammaRegressor,
    NormalInverseGammaRegressor,
    NormalRegressor,
    SiftForgetting,
    StabilizedForgetting,
)
from bayesianbandits._gaussian import RVGAApproximator


def _dense(P):
    return P.toarray() if sp.issparse(P) else np.asarray(P)


def _sym(P):
    """The dense fit paths keep only the upper triangle current."""
    P = _dense(P)
    return np.triu(P) + np.triu(P, 1).T


def _data(n=30, p=4, seed=0):
    rng = np.random.default_rng(seed)
    X = rng.standard_normal((n, p))
    y = X @ np.linspace(1, -1, p) + rng.normal(0, 0.1, n)
    return X, y


class TestUniformRulesOnUpdate:
    @pytest.mark.parametrize("sparse", [False, True])
    def test_exponential_is_one_step_per_row(self, sparse):
        """A batch of n rows scales the prior by rate**n and weighs its
        older rows less: the same posterior as n single-row updates."""
        X, y = _data()
        one = NormalRegressor(
            alpha=1.0, beta=2.0, sparse=sparse, forgetting=ExponentialForgetting(0.9)
        )
        many = NormalRegressor(
            alpha=1.0, beta=2.0, sparse=sparse, forgetting=ExponentialForgetting(0.9)
        )
        wrap = (lambda A: sp.csc_array(A)) if sparse else (lambda A: A)
        one.fit(wrap(X[:10]), y[:10])
        many.fit(wrap(X[:10]), y[:10])
        one.partial_fit(wrap(X[10:]), y[10:])
        for i in range(10, 30):
            many.partial_fit(wrap(X[i : i + 1]), y[i : i + 1])
        assert_allclose(one.coef_, many.coef_, rtol=1e-10)
        assert_allclose(_sym(one.cov_inv_), _sym(many.cov_inv_), rtol=1e-10)

    @pytest.mark.parametrize("sparse", [False, True])
    def test_stabilized_floors_after_the_first_fit_only(self, sparse):
        X, y = _data()
        wrap = (lambda A: sp.csc_array(A)) if sparse else (lambda A: A)
        est = NormalRegressor(
            alpha=2.0, beta=1.0, sparse=sparse, forgetting=StabilizedForgetting(0.8)
        )
        plain = NormalRegressor(
            alpha=2.0, beta=1.0, sparse=sparse, forgetting=ExponentialForgetting(0.8)
        )
        est.fit(wrap(X[:10]), y[:10])
        plain.fit(wrap(X[:10]), y[:10])
        # fit: nothing to forget yet, so no floor is added.
        assert_allclose(_sym(est.cov_inv_), _sym(plain.cov_inv_))
        est.partial_fit(wrap(X[10:15]), y[10:15])
        plain.partial_fit(wrap(X[10:15]), y[10:15])
        g = 0.8**5
        assert_allclose(
            _sym(est.cov_inv_), _sym(plain.cov_inv_) + (1 - g) * 2.0 * np.eye(4)
        )

    def test_stabilized_after_sample_before_fit_is_not_floored(self):
        X, y = _data()
        est = NormalRegressor(alpha=2.0, beta=1.0, forgetting=StabilizedForgetting(0.8))
        plain = NormalRegressor(
            alpha=2.0, beta=1.0, forgetting=ExponentialForgetting(0.8)
        )
        est.sample(X[:2])  # initializes the prior without fitting
        plain.sample(X[:2])
        est.partial_fit(X[:10], y[:10])
        plain.partial_fit(X[:10], y[:10])
        assert_allclose(_sym(est.cov_inv_), _sym(plain.cov_inv_))

    def test_no_rule_is_no_forgetting(self):
        X, y = _data()
        est = NormalRegressor(alpha=1.0, beta=1.0).fit(X[:10], y[:10])
        before = _sym(est.cov_inv_)
        est.partial_fit(X[10:], y[10:])
        gain = _sym(est.cov_inv_) - before
        assert np.all(np.linalg.eigvalsh(gain) >= -1e-10)

    def test_nig_dense_stabilized_floors_at_lam(self):
        X, y = _data()
        est = NormalInverseGammaRegressor(lam=2.0, forgetting=StabilizedForgetting(0.8))
        plain = NormalInverseGammaRegressor(
            lam=2.0, forgetting=ExponentialForgetting(0.8)
        )
        est.fit(X[:10], y[:10])
        plain.fit(X[:10], y[:10])
        est.partial_fit(X[10:15], y[10:15])
        plain.partial_fit(X[10:15], y[10:15])
        g = 0.8**5
        assert_allclose(
            _sym(est.cov_inv_), _sym(plain.cov_inv_) + (1 - g) * 2.0 * np.eye(4)
        )

    def test_nig_forgets_the_inverse_gamma_under_a_uniform_rule_only(self):
        X, y = _data()
        uni = NormalInverseGammaRegressor(forgetting=ExponentialForgetting(0.9))
        uni.fit(X[:10], y[:10])
        a, b = uni.a_, uni.b_
        uni.partial_fit(X[10:15], y[10:15])
        # Older rows of the batch weigh less: the effective count is a
        # geometric sum, not 5.
        assert uni.a_ == pytest.approx(0.9**5 * a + 0.5 * np.sum(0.9 ** np.arange(5)))
        assert uni.b_ < 0.9**5 * b + 0.5 * np.sum(y[10:15] ** 2) + 1e-9
        dire = NormalInverseGammaRegressor(forgetting=FeatureWiseForgetting(0.9))
        dire.fit(X[:10], y[:10])
        a = dire.a_
        dire.partial_fit(X[10:15], y[10:15])
        assert dire.a_ == pytest.approx(a + 2.5)


class TestDirectionalRulesOnUpdate:
    def test_feature_wise_scales_only_the_observed_features(self):
        X, y = _data(p=4)
        est = NormalRegressor(
            alpha=1.0, beta=1.0, forgetting=FeatureWiseForgetting(0.5)
        )
        est.fit(X[:10], y[:10])
        R = _sym(est.cov_inv_)
        # One row touching features 0 and 1 only.
        x = np.array([[1.0, 2.0, 0.0, 0.0]])
        est.partial_fit(x, np.array([0.3]))
        d = np.array([np.sqrt(0.5), np.sqrt(0.5), 1.0, 1.0])
        expected = (d[:, None] * R) * d[None, :] + x.T @ x
        assert_allclose(_sym(est.cov_inv_), expected, rtol=1e-10)

    def test_feature_wise_keeps_the_sparsity_pattern(self):
        rng = np.random.default_rng(0)
        p = 40
        rows = []
        for _ in range(60):
            r = np.zeros(p)
            r[0] = 1.0  # shared column
            r[1 + rng.integers(p - 1)] = 1.0
            rows.append(r)
        X = sp.csc_array(np.array(rows))
        y = rng.standard_normal(60)
        est = NormalRegressor(
            alpha=1.0, beta=1.0, sparse=True, forgetting=FeatureWiseForgetting(0.9)
        )
        est.fit(X[:30], y[:30])
        pattern = set(zip(*est.cov_inv_.nonzero()))
        est.partial_fit(X[30:], y[30:])
        data_pattern = set(zip(*((X.T @ X).nonzero())))
        assert set(zip(*est.cov_inv_.nonzero())) <= pattern | data_pattern

    def test_sift_downdates_along_the_batch(self):
        X, y = _data(p=3)
        est = NormalRegressor(alpha=1.0, beta=1.0, forgetting=SiftForgetting(0.5))
        est.fit(X[:10], y[:10])
        R = _sym(est.cov_inv_)
        x = np.array([[1.0, 0.0, 0.0]])
        est.partial_fit(x, np.array([0.0]))
        w = R @ x[0]
        expected = R - 0.5 * np.outer(w, w) / (x[0] @ w) + x.T @ x
        assert_allclose(_sym(est.cov_inv_), expected, rtol=1e-10)

    def test_directional_rules_leave_the_mean_where_the_data_puts_it(self):
        """Forgetting alone moves no coefficient: with a batch that adds
        nothing (zero rows) the mean is unchanged."""
        X, y = _data(p=3)
        for rule in (FeatureWiseForgetting(0.5), SiftForgetting(0.5)):
            est = NormalRegressor(alpha=1.0, beta=1.0, forgetting=rule)
            est.fit(X[:10], y[:10])
            coef = est.coef_.copy()
            est.partial_fit(np.zeros((2, 3)), np.zeros(2))
            assert_allclose(est.coef_, coef, atol=1e-12)

    def test_glm_accepts_a_directional_rule(self):
        X, y = _data(p=3)
        y = (y > 0).astype(float)
        est = BayesianGLM(alpha=1.0, forgetting=FeatureWiseForgetting(0.5))
        est.fit(X[:10], y[:10])
        R = _sym(est.cov_inv_)
        x = np.array([[1.0, 0.0, 0.0]])
        est.partial_fit(x, np.array([1.0]))
        d = np.array([np.sqrt(0.5), 1.0, 1.0])
        forgotten = (d[:, None] * R) * d[None, :]
        # The Laplace update adds a rank-one Fisher term to the forgotten prior.
        gain = _sym(est.cov_inv_) - forgotten
        assert np.all(np.linalg.eigvalsh(gain) >= -1e-10)
        assert np.linalg.matrix_rank(gain, tol=1e-8) == 1

    def test_sift_is_refused_on_a_sparse_estimator(self):
        X, y = _data()
        est = NormalRegressor(
            alpha=1.0, beta=1.0, sparse=True, forgetting=SiftForgetting(0.9)
        )
        with pytest.raises(TypeError, match="FeatureWiseForgetting"):
            est.fit(sp.csc_array(X), y)

    def test_a_non_rule_is_refused(self):
        X, y = _data()
        est = NormalRegressor(alpha=1.0, beta=1.0, forgetting=0.9)  # type: ignore[arg-type]
        with pytest.raises(TypeError, match="ExponentialForgetting"):
            est.fit(X, y)


class TestUniformOnlyEstimators:
    @pytest.mark.parametrize(
        "make",
        [
            lambda rule: EmpiricalBayesNormalRegressor(forgetting=rule),
            lambda rule: EmpiricalBayesGLM(forgetting=rule),
            lambda rule: DirichletClassifier({0: 1.0, 1: 1.0}, forgetting=rule),
            lambda rule: GammaRegressor(alpha=1.0, beta=1.0, forgetting=rule),
        ],
    )
    def test_directional_rules_are_refused(self, make):
        est = make(FeatureWiseForgetting(0.9))
        if isinstance(est, (DirichletClassifier, GammaRegressor)):
            X, y = np.array([[1], [1]]), np.array([0, 1])
        else:
            X, y = np.eye(3), np.array([1.0, 0.0, 1.0])
        with pytest.raises(TypeError, match="forgets uniformly"):
            est.fit(X, y)

    def test_eb_glm_sparse_rvga_forgets_through_the_cached_factor(self):
        X, y = _data(n=40, p=3, seed=1)
        y = (y > 0).astype(float)
        est = EmpiricalBayesGLM(
            sparse=True,
            approximator=RVGAApproximator(),
            forgetting=StabilizedForgetting(0.9),
        )
        est.fit(sp.csc_array(X[:30]), y[:30])
        est.sample(sp.csc_array(X[:2]))  # caches the factor
        s, alpha_old = est._prior_scalar, est.alpha
        est.partial_fit(sp.csc_array(X[30:]), y[30:])
        g = 0.9**10
        assert est._prior_scalar == pytest.approx(
            (g * s + (1 - g) * alpha_old) * est.alpha / alpha_old
        )
        assert np.all(np.isfinite(est.coef_))

    def test_eb_stabilized_matches_the_old_learning_rate_bookkeeping(self):
        X, y = _data()
        est = EmpiricalBayesNormalRegressor(
            alpha=1.0, beta=1.0, forgetting=StabilizedForgetting(0.99)
        )
        est.fit(X[:20], y[:20])
        s, alpha_old = est._prior_scalar, est.alpha
        est.partial_fit(X[20:25], y[20:25])
        g = 0.99**5
        # The online MacKay step then rescales the prior part to the new alpha.
        assert est._prior_scalar == pytest.approx(
            (g * s + (1 - g) * alpha_old) * est.alpha / alpha_old
        )

    def test_eb_exponential_lets_the_prior_scalar_decay(self):
        X, y = _data()
        est = EmpiricalBayesNormalRegressor(
            alpha=1.0, beta=1.0, forgetting=ExponentialForgetting(0.9)
        )
        est.fit(X[:20], y[:20])
        s, alpha_old = est._prior_scalar, est.alpha
        est.partial_fit(X[20:25], y[20:25])
        assert est._prior_scalar == pytest.approx(0.9**5 * s * est.alpha / alpha_old)

    def test_grouped_stabilized_mixes_the_prior_back_per_row(self):
        clf = DirichletClassifier(
            {0: 1.0, 1: 3.0}, forgetting=StabilizedForgetting(0.5)
        )
        clf.fit(np.array([[1]]), np.array([0]))
        prior = np.array([1.0, 3.0])
        # fit: prior scaled by 0.5, one row added, prior mixed back at 0.5.
        after_fit = 0.5 * prior + np.array([1.0, 0.0]) + 0.5 * prior
        assert_allclose(clf.known_alphas_[1], after_fit)
        clf.partial_fit(np.array([[1], [1]]), np.array([1, 1]))
        # Two rows: 0.25 on the old value, the first row at 0.5, the second
        # at 1, and 0.75 of the prior back.
        expected = 0.25 * after_fit + 1.5 * np.array([0.0, 1.0]) + 0.75 * prior
        assert_allclose(clf.known_alphas_[1], expected)

    def test_eb_gamma_stabilized_returns_to_the_prior(self):
        model = EmpiricalBayesGammaRegressor(
            alpha=2.0, beta=3.0, forgetting=StabilizedForgetting(0.5), random_state=0
        )
        model.fit(np.array([[1], [1], [1]]), np.array([2, 3, 4]))
        for _ in range(60):
            model.partial_fit(np.array([[2]]), np.array([1]))  # another group
        # Group 1 is only ticked by decay; group 2's rows do not touch it.
        assert model.coef_[1][0] > model.prior_[0]


class TestLegacyPickles:
    def _relabel(self, est, learning_rate):
        state = est.__getstate__()
        state.pop("forgetting")
        state["learning_rate"] = learning_rate
        fresh: Any = object.__new__(type(est))
        fresh.__setstate__(state)
        return fresh

    def test_plain_rate_becomes_exponential(self):
        X, y = _data()
        est = NormalRegressor(alpha=1.0, beta=1.0).fit(X, y)
        loaded = self._relabel(est, 0.95)
        assert loaded.forgetting == ExponentialForgetting(0.95)
        assert not hasattr(loaded, "learning_rate")
        assert_allclose(loaded.coef_, est.coef_)

    def test_rate_one_becomes_no_rule(self):
        X, y = _data()
        est = NormalRegressor(alpha=1.0, beta=1.0).fit(X, y)
        assert self._relabel(est, 1.0).forgetting is None

    def test_eb_rate_becomes_stabilized(self):
        X, y = _data()
        est = EmpiricalBayesNormalRegressor().fit(X, y)
        assert self._relabel(est, 0.9).forgetting == StabilizedForgetting(0.9)

    def test_grouped_rate_converts_too(self):
        clf = DirichletClassifier({0: 1.0, 1: 1.0}).fit(np.array([[1]]), np.array([0]))
        assert self._relabel(clf, 0.8).forgetting == ExponentialForgetting(0.8)
        model = GammaRegressor(alpha=1.0, beta=1.0).fit(np.array([[1]]), np.array([2]))
        assert self._relabel(model, 0.8).forgetting == ExponentialForgetting(0.8)

    def test_current_pickles_round_trip(self):
        X, y = _data()
        est = NormalRegressor(
            alpha=1.0, beta=1.0, forgetting=FeatureWiseForgetting(0.9)
        ).fit(X, y)
        loaded: Any = pickle.loads(pickle.dumps(est))
        assert loaded.forgetting == FeatureWiseForgetting(0.9)
        loaded.partial_fit(X[:3], y[:3])
        est.partial_fit(X[:3], y[:3])
        assert_allclose(loaded.coef_, est.coef_)


class TestCoverageUnderFeatureWise:
    def test_intercept_plus_arm_predictions_stay_calibrated(self):
        """Intercept plus one-hot arms, 90/5/5 pulls, stationary rewards:
        the cell-mean intervals under feature-wise forgetting must cover
        the truth about as often as they claim."""
        rng = np.random.default_rng(3)
        means = np.array([1.0, 0.5, 0.0])
        est = NormalRegressor(
            alpha=1.0, beta=4.0, forgetting=FeatureWiseForgetting(0.98)
        )

        def row(a):
            x = np.zeros(4)
            x[0] = 1.0
            x[1 + a] = 1.0
            return x

        hits = 0
        total = 0
        for t in range(1500):
            a = rng.choice(3, p=[0.9, 0.05, 0.05])
            x = row(a)[None, :]
            est.partial_fit(x, np.array([means[a] + rng.normal(0, 0.5)]))
            if t >= 500:
                S = np.linalg.inv(_sym(est.cov_inv_))
                for arm in range(3):
                    r = row(arm)
                    sd = np.sqrt(r @ S @ r)
                    hits += abs(r @ est.coef_ - means[arm]) < 1.96 * sd
                    total += 1
        assert hits / total > 0.9
        assert np.linalg.eigvalsh(_sym(est.cov_inv_))[0] >= 1.0 - 1e-9
