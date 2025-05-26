from typing import Literal
from unittest import mock

import numpy as np
import pytest
import scipy.sparse as sp
from numpy.testing import assert_almost_equal
from numpy.typing import NDArray
from sklearn.datasets import make_regression

from bayesianbandits import (
    NormalInverseGammaRegressor,
    NormalRegressor,
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
    with mock.patch("bayesianbandits._estimators.solver", request.param):
        yield


@pytest.fixture
def X() -> NDArray[np.int_]:
    """Return X."""
    return np.array([1, 1, 1, 2, 2, 2, 3, 3, 3]).reshape(-1, 1)


@pytest.fixture
def y() -> NDArray[np.int_]:
    """Return y."""
    return np.array([1, 1, 2, 2, 2, 3, 3, 3, 3])


@pytest.mark.parametrize("sparse", [True, False])
def test_normal_regressor_init(sparse: bool) -> None:
    """Test NormalRegressor init."""

    clf = NormalRegressor(alpha=1, beta=1, sparse=sparse, random_state=0)
    assert clf is not None


@pytest.mark.parametrize("sparse", [True, False])
def test_normal_regressor_fit(
    X: NDArray[np.int_],
    y: NDArray[np.int_],
    sparse: bool,
) -> None:
    """Test NormalRegressor fit."""

    clf = NormalRegressor(alpha=1, beta=1, sparse=sparse, random_state=0)
    if sparse:
        X_fit = sp.csc_array(X)
    else:
        X_fit = X
    clf.fit(X_fit, y)

    # worked these out by hand and by ref implementations
    if sparse:
        cov_inv_ = clf.cov_inv_.toarray()  # type: ignore
    else:
        cov_inv_ = clf.cov_inv_

    assert_almost_equal(clf.coef_, np.array([1.04651163]))
    assert_almost_equal(cov_inv_, np.array([[43.0]]))


@pytest.mark.parametrize("sparse", [True, False])
def test_normal_regressor_partial_fit_never_fit(
    X: NDArray[np.int_],
    y: NDArray[np.int_],
    sparse: bool,
) -> None:
    """Test NormalRegressor partial_fit."""

    clf = NormalRegressor(alpha=1, beta=1, sparse=sparse, random_state=0)
    if sparse:
        X_fit = sp.csc_array(X)
    else:
        X_fit = X
    clf.partial_fit(X_fit, y)

    if sparse:
        cov_inv_ = clf.cov_inv_.toarray()  # type: ignore
    else:
        cov_inv_ = clf.cov_inv_
    # worked these out by hand and by ref implementations
    assert_almost_equal(clf.coef_, np.array([1.04651163]))
    assert_almost_equal(cov_inv_, np.array([[43.0]]))


@pytest.mark.parametrize("sparse", [True, False])
def test_normal_regressor_partial_fit_already_fit(
    X: NDArray[np.int_],
    y: NDArray[np.int_],
    sparse: bool,
) -> None:
    """Test NormalRegressor partial_fit."""

    clf = NormalRegressor(alpha=1, beta=1, sparse=sparse, random_state=0)
    if sparse:
        X_fit = sp.csc_array(X)
    else:
        X_fit = X
    clf.fit(X_fit, y)
    clf.partial_fit(X_fit, y)

    if sparse:
        cov_inv_ = clf.cov_inv_.toarray()  # type: ignore
    else:
        cov_inv_ = clf.cov_inv_
    # worked these out by hand and by ref implementations
    assert_almost_equal(clf.coef_, np.array([1.0588235]))
    assert_almost_equal(cov_inv_, np.array([[85.0]]))


@pytest.mark.parametrize("sparse", [True, False])
def test_normal_regressor_predict(
    X: NDArray[np.int_],
    y: NDArray[np.int_],
    sparse: bool,
) -> None:
    """Test NormalRegressor predict."""
    if sparse:
        X_fit = sp.csc_array(X)
    else:
        X_fit = X

    clf = NormalRegressor(alpha=1, beta=1, sparse=sparse, random_state=0)
    clf.fit(X_fit, y)
    assert_almost_equal(
        clf.predict(X_fit),
        np.array(
            [
                1.0465116,
                1.0465116,
                1.0465116,
                2.0930233,
                2.0930233,
                2.0930233,
                3.1395349,
                3.1395349,
                3.1395349,
            ]
        ),
    )


@pytest.mark.parametrize("sparse", [True, False])
def test_normal_regressor_predict_no_fit(
    X: NDArray[np.int_],
    y: NDArray[np.int_],
    sparse: bool,
) -> None:
    """Test NormalRegressor predict."""
    if sparse:
        X_fit = sp.csc_array(X)
    else:
        X_fit = X

    clf = NormalRegressor(alpha=1, beta=1, sparse=sparse, random_state=0)
    assert_almost_equal(
        clf.predict(X_fit),
        np.array(
            [
                0.0,
                0.0,
                0.0,
                0.0,
                0.0,
                0.0,
                0.0,
                0.0,
                0.0,
            ]
        ),
    )


@pytest.mark.parametrize("sparse", [True, False])
def test_normal_regressor_sample(
    X: NDArray[np.int_],
    y: NDArray[np.int_],
    sparse: bool,
) -> None:
    """Test NormalRegressor sample."""
    if sparse:
        X_fit = sp.csc_array(X)
    else:
        X_fit = X

    clf = NormalRegressor(alpha=1, beta=1, sparse=sparse, random_state=0)
    clf.fit(X_fit, y)

    assert_almost_equal(
        clf.sample(X_fit),
        np.array(
            [
                [
                    1.0656853,
                    1.0656853,
                    1.0656853,
                    2.1313706,
                    2.1313706,
                    2.1313706,
                    3.1970559,
                    3.1970559,
                    3.1970559,
                ]
            ]
        ),
    )


@pytest.mark.parametrize("sparse", [True, False])
def test_normal_regressor_sample_no_fit(
    X: NDArray[np.int_],
    y: NDArray[np.int_],
    sparse: bool,
) -> None:
    """Test NormalRegressor sample."""
    if sparse:
        X_fit = sp.csc_array(X)
    else:
        X_fit = X

    clf = NormalRegressor(alpha=1, beta=1, sparse=sparse, random_state=0)

    assert_almost_equal(
        clf.sample(X_fit),
        np.array(
            [
                [
                    0.1257302,
                    0.1257302,
                    0.1257302,
                    0.2514604,
                    0.2514604,
                    0.2514604,
                    0.3771907,
                    0.3771907,
                    0.3771907,
                ]
            ]
        ),
    )


@pytest.mark.parametrize("sparse", [True, False])
def test_normal_regressor_decay(
    X: NDArray[np.int_],
    y: NDArray[np.int_],
    sparse: bool,
) -> None:
    """Test NormalRegressor decay only increases variance."""
    if sparse:
        X_fit = sp.csc_array(X)
    else:
        X_fit = X

    clf = NormalRegressor(
        alpha=1, beta=1, learning_rate=0.9, sparse=sparse, random_state=0
    )
    clf.fit(X_fit, y)

    pre_decay = clf.predict(X_fit)

    clf.decay(X_fit)

    assert_almost_equal(clf.predict(X_fit), pre_decay)


@pytest.mark.parametrize("sparse", [True, False])
def test_normal_regressor_manual_decay(
    X: NDArray[np.int_],
    y: NDArray[np.int_],
    sparse: bool,
) -> None:
    """Test NormalRegressor decay only increases variance."""
    if sparse:
        X_fit = sp.csc_array(X)
    else:
        X_fit = X

    clf = NormalRegressor(
        alpha=1, beta=1, learning_rate=1.0, sparse=sparse, random_state=0
    )
    clf.fit(X_fit, y)

    pre_decay = clf.predict(X_fit)

    clf.decay(X_fit, decay_rate=0.9)

    assert_almost_equal(clf.predict(X_fit), pre_decay)


@pytest.mark.parametrize("sparse", [True, False])
def test_normal_regressor_serialization(
    X: NDArray[np.int_],
    y: NDArray[np.int_],
    sparse: bool,
) -> None:
    """Test that NormalRegressor can be serialized."""
    from io import BytesIO

    import joblib

    if sparse:
        X_fit = sp.csc_array(X)
    else:
        X_fit = X

    clf = NormalRegressor(
        alpha=1, beta=1, learning_rate=1.0, sparse=sparse, random_state=0
    )
    clf.fit(X_fit, y)

    pre_dump_cov_inv = clf.cov_inv_

    dumped = BytesIO()
    joblib.dump(clf, dumped)

    agent = joblib.load(dumped)
    assert agent is not None

    post_dump_cov_inv = agent.cov_inv_

    assert_almost_equal(pre_dump_cov_inv, post_dump_cov_inv)

    clf.sample(X_fit)

    dumped_after_sample = BytesIO()
    joblib.dump(clf, dumped_after_sample)

    agent_after_sample = joblib.load(dumped_after_sample)
    assert agent_after_sample is not None

    post_dump_cov_inv_after_sample = agent_after_sample.cov_inv_

    assert_almost_equal(post_dump_cov_inv, post_dump_cov_inv_after_sample)


@pytest.mark.parametrize("obs", [1, 2, 3, 4])
@pytest.mark.parametrize("covariates", [1, 2, 3, 4])
@pytest.mark.parametrize("sparse", [True, False])
def test_normal_regressor_predict_covariates(
    obs: Literal[1, 2, 3, 4], covariates: Literal[1, 2, 3, 4], sparse: bool
):
    """Test NormalRegressor predict with covariates."""

    X = np.random.rand(obs, covariates)
    y = np.random.rand(obs)

    if sparse:
        X_fit = sp.csc_array(X)
    else:
        X_fit = X

    clf = NormalRegressor(alpha=1, beta=1, sparse=sparse, random_state=0)
    clf.fit(X_fit, y)

    pred = clf.predict(X_fit)

    assert pred.shape == (obs,)

    single_pred = clf.predict(X_fit[[0]])  # type: ignore
    assert single_pred.shape == (1,)


@pytest.mark.parametrize("obs", [1, 2, 3, 4])
@pytest.mark.parametrize("covariates", [1, 2, 3, 4])
@pytest.mark.parametrize("size", [1, 2, 3, 4])
@pytest.mark.parametrize("sparse", [True, False])
def test_normal_regressor_sample_covariates(
    obs: Literal[1, 2, 3, 4],
    covariates: Literal[1, 2, 3, 4],
    size: Literal[1, 2, 3, 4],
    sparse: bool,
):
    """Test NormalRegressor predict with covariates."""

    X = np.random.rand(obs, covariates)
    y = np.random.rand(obs)

    if sparse:
        X_fit = sp.csc_array(X)
    else:
        X_fit = X

    clf = NormalRegressor(alpha=1, beta=1, sparse=sparse, random_state=0)
    clf.fit(X_fit, y)

    pred = clf.sample(X_fit, size=size)

    assert pred.shape == (size, obs)

    single_pred = clf.sample(X_fit[[0]], size=size)  # type: ignore
    assert single_pred.shape == (size, 1)


@pytest.mark.parametrize("sparse", [True, False])
def test_normal_regressor_fit_with_weights(sparse: bool) -> None:
    """Test that sample weights affect the posterior correctly."""
    X, y = make_regression(n_samples=10, n_features=2, noise=0.1, random_state=42)  # type: ignore

    # Fit without weights
    reg1 = NormalRegressor(alpha=1.0, beta=1.0, sparse=sparse, random_state=0)
    if sparse:
        X_fit = sp.csc_array(X)
    else:
        X_fit = X
    reg1.fit(X_fit, y)

    # Fit with non-uniform weights
    weights = np.array([0.5, 2.0, 1.0, 0.5, 2.0, 1.0, 0.5, 2.0, 1.0, 0.5])
    reg2 = NormalRegressor(alpha=1.0, beta=1.0, sparse=sparse, random_state=0)
    reg2.fit(X_fit, y, sample_weight=weights)

    # Coefficients should be different
    assert not np.allclose(reg1.coef_, reg2.coef_)

    # Covariance should also be different
    if sparse:
        assert not np.allclose(reg1.cov_inv_.toarray(), reg2.cov_inv_.toarray())  # type: ignore
    else:
        assert not np.allclose(reg1.cov_inv_, reg2.cov_inv_)


@pytest.mark.parametrize("sparse", [True, False])
def test_normal_regressor_weight_equals_duplication(sparse: bool) -> None:
    """Test that weight=2 is equivalent to seeing a sample twice."""
    X = np.array([[1, 2], [3, 4], [5, 6]])
    y = np.array([1, 2, 3])

    # Fit with duplicated data
    reg1 = NormalRegressor(alpha=0.1, beta=1.0, sparse=sparse, random_state=0)
    X_dup = np.vstack([X, X])
    y_dup = np.hstack([y, y])
    if sparse:
        X_dup_fit = sp.csc_array(X_dup)
    else:
        X_dup_fit = X_dup
    reg1.fit(X_dup_fit, y_dup)

    # Fit with weight=2
    reg2 = NormalRegressor(alpha=0.1, beta=1.0, sparse=sparse, random_state=0)
    weights = np.array([2.0, 2.0, 2.0])
    if sparse:
        X_fit = sp.csc_array(X)
    else:
        X_fit = X
    reg2.fit(X_fit, y, sample_weight=weights)

    assert_almost_equal(reg1.coef_, reg2.coef_)
    if sparse:
        assert_almost_equal(reg1.cov_inv_.toarray(), reg2.cov_inv_.toarray())  # type: ignore
    else:
        assert_almost_equal(reg1.cov_inv_, reg2.cov_inv_)


@pytest.mark.parametrize("sparse", [True, False])
def test_normal_regressor_zero_weights(sparse: bool) -> None:
    """Test that samples with weight=0 are effectively ignored."""
    X = np.array([[1, 2], [3, 4], [5, 6], [7, 8]])
    y = np.array([1, 2, 3, 4])

    # Fit on subset
    reg1 = NormalRegressor(alpha=0.1, beta=1.0, sparse=sparse, random_state=0)
    X_subset = X[[0, 2]]
    y_subset = y[[0, 2]]
    if sparse:
        X_subset_fit = sp.csc_array(X_subset)
    else:
        X_subset_fit = X_subset
    reg1.fit(X_subset_fit, y_subset)

    # Fit on full data with zero weights
    reg2 = NormalRegressor(alpha=0.1, beta=1.0, sparse=sparse, random_state=0)
    weights = np.array([1.0, 0.0, 1.0, 0.0])
    if sparse:
        X_fit = sp.csc_array(X)
    else:
        X_fit = X
    reg2.fit(X_fit, y, sample_weight=weights)

    assert_almost_equal(reg1.coef_, reg2.coef_)
    if sparse:
        assert_almost_equal(reg1.cov_inv_.toarray(), reg2.cov_inv_.toarray())  # type: ignore
    else:
        assert_almost_equal(reg1.cov_inv_, reg2.cov_inv_)


@pytest.mark.parametrize("sparse", [True, False])
def test_normal_regressor_all_zero_weights(sparse: bool) -> None:
    """Test edge case where all weights are zero."""
    X = np.array([[1, 2], [3, 4]])
    y = np.array([1, 2])

    reg = NormalRegressor(alpha=1.0, beta=1.0, sparse=sparse, random_state=0)
    weights = np.zeros(len(y))
    if sparse:
        X_fit = sp.csc_array(X)
    else:
        X_fit = X
    reg.fit(X_fit, y, sample_weight=weights)

    # Should only have the prior
    assert_almost_equal(reg.coef_, np.zeros(X.shape[1]))
    if sparse:
        assert_almost_equal(reg.cov_inv_.toarray(), np.eye(X.shape[1]))  # type: ignore
    else:
        assert_almost_equal(reg.cov_inv_, np.eye(X.shape[1]))


@pytest.mark.parametrize("sparse", [True, False])
def test_normal_regressor_partial_fit_with_weights(sparse: bool) -> None:
    """Test partial_fit with sample weights."""
    X, y = make_regression(n_samples=10, n_features=2, noise=0.1, random_state=42)  # type: ignore
    weights = np.random.rand(10) * 2  # Random weights between 0 and 2

    # Single fit with learning_rate=1 (no decay)
    reg1 = NormalRegressor(
        alpha=0.1, beta=1.0, learning_rate=1.0, sparse=sparse, random_state=0
    )
    if sparse:
        X_fit = sp.csc_array(X)
    else:
        X_fit = X
    reg1.fit(X_fit, y, sample_weight=weights)

    # Sequential partial_fit with learning_rate=1 (no decay)
    reg2 = NormalRegressor(
        alpha=0.1, beta=1.0, learning_rate=1.0, sparse=sparse, random_state=0
    )
    if sparse:
        X1_fit = sp.csc_array(X[:5])
        X2_fit = sp.csc_array(X[5:])
    else:
        X1_fit = X[:5]
        X2_fit = X[5:]
    reg2.partial_fit(X1_fit, y[:5], sample_weight=weights[:5])
    reg2.partial_fit(X2_fit, y[5:], sample_weight=weights[5:])

    assert_almost_equal(reg1.coef_, reg2.coef_)
    if sparse:
        assert_almost_equal(reg1.cov_inv_.toarray(), reg2.cov_inv_.toarray())  # type: ignore
    else:
        assert_almost_equal(reg1.cov_inv_, reg2.cov_inv_)


@pytest.mark.parametrize("sparse", [True, False])
def test_normal_regressor_mathematical_correctness(sparse: bool) -> None:
    """Test mathematical correctness of weighted Bayesian linear regression."""
    # Simple 1D case for easy manual calculation
    X = np.array([[1], [2], [3]])
    y = np.array([1, 2, 3])
    weights = np.array([2.0, 1.0, 0.5])

    alpha = 1.0  # Prior precision
    beta = 1.0  # Noise precision

    reg = NormalRegressor(alpha=alpha, beta=beta, sparse=sparse, random_state=0)
    if sparse:
        X_fit = sp.csc_array(X)
    else:
        X_fit = X
    reg.fit(X_fit, y, sample_weight=weights)

    # Manual calculation
    # X^T W X = [1, 2, 3] @ diag([2, 1, 0.5]) @ [[1], [2], [3]]
    #         = [1, 2, 3] @ [[2], [2], [1.5]] = 1*2 + 2*2 + 3*1.5 = 10.5
    # X^T W y = [1, 2, 3] @ diag([2, 1, 0.5]) @ [1, 2, 3]
    #         = [1, 2, 3] @ [2, 2, 1.5] = 1*2 + 2*2 + 3*1.5 = 10.5

    # Posterior precision: Λ = α*I + β*X^T W X = 1*1 + 1*10.5 = 11.5
    # Posterior mean: μ = Λ^(-1) * β * X^T W y = (1/11.5) * 1 * 10.5 = 10.5/11.5

    expected_coef = 10.5 / 11.5
    expected_cov_inv = 11.5

    assert_almost_equal(reg.coef_[0], expected_coef, decimal=6)
    if sparse:
        assert_almost_equal(reg.cov_inv_[0, 0], expected_cov_inv, decimal=6)
    else:
        assert_almost_equal(reg.cov_inv_[0, 0], expected_cov_inv, decimal=6)


@pytest.mark.parametrize("sparse", [True, False])
def test_normal_regressor_weights_learning_rate_interaction(sparse: bool) -> None:
    """Test that sample weights and learning_rate interact correctly."""
    X = np.array([[1, 0], [0, 1], [1, 1]])
    y = np.array([1, 2, 3])
    weights = np.array([2.0, 1.0, 0.5])

    # With learning rate < 1
    reg = NormalRegressor(
        alpha=8.0, beta=1.0, learning_rate=0.5, sparse=sparse, random_state=0
    )
    if sparse:
        X_fit = sp.csc_array(X)
    else:
        X_fit = X
    reg.fit(X_fit, y, sample_weight=weights)

    # The effective weights combine decay_factors and sample_weight
    # decay_factors = [0.5^2, 0.5^1, 0.5^0] = [0.25, 0.5, 1.0]
    # effective_weights = weights * decay_factors = [2.0*0.25, 1.0*0.5, 0.5*1.0] = [0.5, 0.5, 0.5]

    # Fit same data with uniform weights AND SAME learning rate
    reg_uniform = NormalRegressor(
        alpha=1.0, beta=1.0, learning_rate=1.0, sparse=sparse, random_state=0
    )
    reg_uniform.fit(X_fit, y, sample_weight=np.array([0.5, 0.5, 0.5]))

    # Should be very close (exact up to numerical precision)
    assert_almost_equal(reg.coef_, reg_uniform.coef_)
    if sparse:
        assert_almost_equal(reg.cov_inv_.toarray(), reg_uniform.cov_inv_.toarray())  # type: ignore
    else:
        assert_almost_equal(reg.cov_inv_, reg_uniform.cov_inv_)

    # Also test that weights scale linearly
    # If we double all weights, the precision matrix should scale
    reg_double = NormalRegressor(
        alpha=1.0, beta=1.0, learning_rate=0.5, sparse=sparse, random_state=0
    )
    reg_double.fit(X_fit, y, sample_weight=weights * 2)

    # The precision matrix should scale with the weights
    # But coefficients might change due to the prior contribution
    if sparse:
        # Check that doubling weights increases the precision
        assert np.all(
            np.diag(reg_double.cov_inv_.toarray()) > np.diag(reg.cov_inv_.toarray())  # type: ignore
        )
    else:
        assert np.all(np.diag(reg_double.cov_inv_) > np.diag(reg.cov_inv_))


def test_normal_regressor_weight_validation() -> None:
    """Test input validation for weights."""
    X = np.array([[1, 2], [3, 4]])
    y = np.array([1, 2])

    reg = NormalRegressor(alpha=1.0, beta=1.0)

    # Wrong shape
    with pytest.raises(ValueError, match="sample_weight.shape"):
        reg.fit(X, y, sample_weight=np.array([1.0]))  # Too few

    with pytest.raises(ValueError, match="sample_weight.shape"):
        reg.fit(X, y, sample_weight=np.array([1.0, 2.0, 3.0]))  # Too many


@pytest.mark.parametrize("sparse", [True, False])
def test_normal_regressor_fractional_weights(sparse: bool) -> None:
    """Test that fractional weights work correctly."""
    X = np.array([[1, 0], [0, 1], [1, 1]])
    y = np.array([1.5, 2.3, 3.7])

    # Use fractional weights
    weights = np.array([0.7, 1.3, 0.6])

    reg = NormalRegressor(alpha=0.5, beta=2.0, sparse=sparse, random_state=0)
    if sparse:
        X_fit = sp.csc_array(X)
    else:
        X_fit = X
    reg.fit(X_fit, y, sample_weight=weights)

    # Test that we get reasonable predictions
    preds = reg.predict(X_fit)

    # Weighted model should still fit the data reasonably well
    weighted_error = np.sum(weights * (y - preds) ** 2)
    assert weighted_error < 5.0  # Reasonable threshold for this toy problem


@pytest.mark.parametrize("sparse", [True, False])
def test_normal_regressor_importance_sampling(sparse: bool) -> None:
    """Test importance sampling scenario for regression."""
    np.random.seed(42)

    # True linear model: y = 2*x1 + 3*x2 + noise
    n_samples = 100
    X = np.random.randn(n_samples, 2)
    true_coef = np.array([2.0, 3.0])
    y = X @ true_coef + 0.1 * np.random.randn(n_samples)

    # Create biased sampling: undersample high y values
    # This simulates a scenario where we're less likely to see extreme outcomes
    y_percentile = np.percentile(y, 75)
    sampling_probs = np.where(y > y_percentile, 0.2, 0.8)
    sampling_probs = sampling_probs / sampling_probs.sum()

    # Sample with bias
    indices = np.random.choice(n_samples, size=80, p=sampling_probs)
    X_biased = X[indices]
    y_biased = y[indices]

    # Calculate importance weights (inverse of sampling probability)
    importance_weights = 1.0 / (sampling_probs[indices] * n_samples)
    importance_weights = importance_weights / importance_weights.mean()  # Normalize

    # Fit with and without weights
    reg_weighted = NormalRegressor(alpha=0.01, beta=10.0, sparse=sparse, random_state=0)
    reg_biased = NormalRegressor(alpha=0.01, beta=10.0, sparse=sparse, random_state=0)

    if sparse:
        X_biased_fit = sp.csc_array(X_biased)
    else:
        X_biased_fit = X_biased

    reg_weighted.fit(X_biased_fit, y_biased, sample_weight=importance_weights)
    reg_biased.fit(X_biased_fit, y_biased)

    # Weighted estimate should be closer to true coefficients
    weighted_error = np.linalg.norm(reg_weighted.coef_ - true_coef)
    biased_error = np.linalg.norm(reg_biased.coef_ - true_coef)

    assert weighted_error < biased_error


@pytest.mark.parametrize("mu_shape", ["scalar", "vector"])
@pytest.mark.parametrize("lam_shape", ["scalar", "vector", "matrix"])
@pytest.mark.parametrize("sparse", [True, False])
def test_normal_inverse_gamma_regressor_fit(
    X: NDArray[np.int_],
    y: NDArray[np.int_],
    mu_shape: Literal["scalar", "vector"],
    lam_shape: Literal["scalar", "vector", "matrix"],
    sparse: bool,
) -> None:
    """Test NormalRegressor fit."""

    prior_mu = 0.0
    if mu_shape == "vector":
        prior_mu = np.full_like(X[0], prior_mu)

    prior_lam = 1.0
    if lam_shape == "vector":
        prior_lam = np.full_like(X[0], prior_lam)
    elif lam_shape == "matrix":
        prior_lam = np.eye(X.shape[1])
        if sparse:
            prior_lam = sp.csc_array(prior_lam)

    clf = NormalInverseGammaRegressor(
        mu=prior_mu, lam=prior_lam, random_state=0, sparse=sparse
    )
    if sparse:
        X_fit = sp.csc_array(X)
    else:
        X_fit = X
    clf.fit(X_fit, y)

    if sparse:
        assert isinstance(clf.cov_inv_, sp.csc_array)
        cov_inv_ = clf.cov_inv_.toarray()
    else:
        cov_inv_ = clf.cov_inv_
    assert isinstance(cov_inv_, np.ndarray)
    # worked these out by hand and by ref implementations
    assert_almost_equal(clf.coef_, np.array([1.04651163]))
    assert_almost_equal(cov_inv_, np.array([[43.0]]))
    assert_almost_equal(clf.a_, 4.6)
    assert_almost_equal(clf.b_, 1.5534883720930197)


@pytest.mark.parametrize("sparse", [True, False])
def test_normal_inverse_gamma_regressor_partial_fit_never_fit(
    X: NDArray[np.int_],
    y: NDArray[np.int_],
    sparse: bool,
) -> None:
    """Test NormalRegressor partial_fit."""

    clf = NormalInverseGammaRegressor(random_state=0, sparse=sparse)
    if sparse:
        X_fit = sp.csc_array(X)
    else:
        X_fit = X
    clf.partial_fit(X_fit, y)

    if sparse:
        assert isinstance(clf.cov_inv_, sp.csc_array)
        cov_inv_ = clf.cov_inv_.toarray()
    else:
        cov_inv_ = clf.cov_inv_

    assert isinstance(cov_inv_, np.ndarray)
    # worked these out by hand and by ref implementations
    assert_almost_equal(clf.coef_, np.array([1.04651163]))
    assert_almost_equal(cov_inv_, np.array([[43.0]]))
    assert_almost_equal(clf.a_, 4.6)
    assert_almost_equal(clf.b_, 1.5534883720930197)


@pytest.mark.parametrize("sparse", [True, False])
def test_normal_inverse_gamma_regressor_partial_fit_already_fit(
    X: NDArray[np.int_],
    y: NDArray[np.int_],
    sparse: bool,
) -> None:
    """Test NormalRegressor partial_fit."""

    clf = NormalInverseGammaRegressor(random_state=0, sparse=sparse)
    if sparse:
        X_fit = sp.csc_array(X)
    else:
        X_fit = X
    clf.fit(X_fit, y)
    clf.partial_fit(X_fit, y)

    if sparse:
        assert isinstance(clf.cov_inv_, sp.csc_array)
        cov_inv_ = clf.cov_inv_.toarray()
    else:
        cov_inv_ = clf.cov_inv_

    assert isinstance(cov_inv_, np.ndarray)
    # worked these out by hand and by ref implementations
    # worked these out by hand and by ref implementations
    assert_almost_equal(clf.coef_, np.array([1.05882353]))
    assert_almost_equal(cov_inv_, np.array([[85.0]]))
    assert_almost_equal(clf.a_, 9.1)
    assert_almost_equal(clf.b_, 2.452941176470587)


@pytest.mark.parametrize("sparse", [True, False])
def test_normal_inverse_gamma_regressor_predict(
    X: NDArray[np.int_],
    y: NDArray[np.int_],
    sparse: bool,
) -> None:
    """Test NormalRegressor predict."""

    clf = NormalInverseGammaRegressor(random_state=0, sparse=sparse)
    if sparse:
        X_fit = sp.csc_array(X)
    else:
        X_fit = X
    clf.fit(X_fit, y)
    assert_almost_equal(
        clf.predict(X_fit),
        np.array(
            [
                1.0465116,
                1.0465116,
                1.0465116,
                2.0930233,
                2.0930233,
                2.0930233,
                3.1395349,
                3.1395349,
                3.1395349,
            ]
        ),
    )


@pytest.mark.parametrize("sparse", [True, False])
def test_normal_inverse_gamma_regressor_predict_no_fit(
    X: NDArray[np.int_],
    y: NDArray[np.int_],
    sparse: bool,
) -> None:
    """Test NormalRegressor predict."""

    clf = NormalInverseGammaRegressor(random_state=0, sparse=sparse)
    if sparse:
        X_fit = sp.csc_array(X)
    else:
        X_fit = X
    assert_almost_equal(
        clf.predict(X_fit),
        np.array(
            [
                0.0,
                0.0,
                0.0,
                0.0,
                0.0,
                0.0,
                0.0,
                0.0,
                0.0,
            ]
        ),
    )


@pytest.mark.parametrize("sparse", [True, False])
def test_normal_inverse_gamma_regressor_sample(
    X: NDArray[np.int_],
    y: NDArray[np.int_],
    sparse: bool,
) -> None:
    """Test NormalRegressor sample."""

    clf = NormalInverseGammaRegressor(random_state=0, sparse=sparse)
    if sparse:
        X_fit = sp.csc_array(X)
    else:
        X_fit = X
    clf.fit(X_fit, y)

    assert_almost_equal(
        clf.sample(X_fit),
        np.array(
            [
                [
                    1.1036933,
                    1.1036933,
                    1.1036933,
                    2.2073866,
                    2.2073866,
                    2.2073866,
                    3.3110799,
                    3.3110799,
                    3.3110799,
                ]
            ]
        ),
    )


@pytest.mark.parametrize("sparse", [True, False])
def test_normal_inverse_gamma_regressor_sample_no_fit(
    X: NDArray[np.int_],
    y: NDArray[np.int_],
    sparse: bool,
) -> None:
    """Test NormalRegressor sample."""

    clf = NormalInverseGammaRegressor(random_state=0, sparse=sparse)

    if sparse:
        X_fit = sp.csc_array(X)
    else:
        X_fit = X

    assert_almost_equal(
        clf.sample(X_fit),
        np.array(
            [
                [
                    1.9315241,
                    1.9315241,
                    1.9315241,
                    3.8630482,
                    3.8630482,
                    3.8630482,
                    5.7945723,
                    5.7945723,
                    5.7945723,
                ]
            ]
        ),
    )


@pytest.mark.parametrize("sparse", [True, False])
def test_normal_inverse_gamma_regressor_decay(
    X: NDArray[np.int_],
    y: NDArray[np.int_],
    sparse: bool,
) -> None:
    """Test NormalRegressor decay only increases variance."""

    clf = NormalInverseGammaRegressor(random_state=0, learning_rate=0.9, sparse=sparse)
    if sparse:
        X_fit = sp.csc_array(X)
    else:
        X_fit = X

    clf.fit(X_fit, y)

    pre_decay = clf.predict(X_fit)

    clf.decay(X_fit)

    assert_almost_equal(clf.predict(X_fit), pre_decay)


@pytest.mark.parametrize("sparse", [True, False])
def test_normal_inverse_gamma_prior_exceptions(sparse) -> None:
    """Test exceptions are raised when priors are not the right shape."""

    # mu must be scalar or vector
    with pytest.raises(ValueError):
        est = NormalInverseGammaRegressor(mu=np.array([[1, 2], [2, 3]]), sparse=sparse)
        est.sample(np.array([[1, 2]]))

    # lam must be scalar, vector, or matrix
    with pytest.raises(ValueError):
        est = NormalInverseGammaRegressor(
            lam=np.array([[[1, 2], [2, 3]]]), sparse=sparse
        )
        est.sample(np.array([[1, 2]]))


@pytest.mark.parametrize("sparse", [True, False])
def test_normal_inverse_gamma_regressor_manual_decay(
    X: NDArray[np.int_],
    y: NDArray[np.int_],
    sparse: bool,
) -> None:
    """Test NormalRegressor decay only increases variance."""

    clf = NormalInverseGammaRegressor(random_state=0, learning_rate=1.0, sparse=sparse)
    if sparse:
        X_fit = sp.csc_array(X)
    else:
        X_fit = X
    clf.fit(X_fit, y)

    pre_decay = clf.predict(X_fit)

    clf.decay(X_fit, decay_rate=0.9)

    assert_almost_equal(clf.predict(X_fit), pre_decay)


@pytest.mark.parametrize("sparse", [True, False])
def test_normal_inverse_gamma_regressor_serialization(
    X: NDArray[np.int_],
    y: NDArray[np.int_],
    sparse: bool,
) -> None:
    """Test that NormalRegressor can be serialized."""
    from io import BytesIO

    import joblib

    if sparse:
        X_fit = sp.csc_array(X)
    else:
        X_fit = X

    clf = NormalInverseGammaRegressor(random_state=0, learning_rate=1.0, sparse=sparse)
    clf.fit(X_fit, y)

    pre_dump_cov_inv = clf.cov_inv_

    dumped = BytesIO()
    joblib.dump(clf, dumped)

    agent = joblib.load(dumped)
    assert agent is not None

    post_dump_cov_inv = agent.cov_inv_

    assert_almost_equal(pre_dump_cov_inv, post_dump_cov_inv)  # type: ignore

    clf.sample(X_fit)

    dumped_after_sample = BytesIO()
    joblib.dump(clf, dumped_after_sample)

    agent_after_sample = joblib.load(dumped_after_sample)
    assert agent_after_sample is not None

    post_dump_cov_inv_after_sample = agent_after_sample.cov_inv_

    assert_almost_equal(post_dump_cov_inv, post_dump_cov_inv_after_sample)


@pytest.mark.parametrize("obs", [1, 2, 3, 4])
@pytest.mark.parametrize("covariates", [1, 2, 3, 4])
@pytest.mark.parametrize("sparse", [True, False])
def test_normal_inverse_gamma_regressor_predict_covariates(
    obs: Literal[1, 2, 3, 4], covariates: Literal[1, 2, 3, 4], sparse: bool
):
    """Test NormalRegressor predict with covariates."""

    X = np.random.rand(obs, covariates)
    y = np.random.rand(obs)

    if sparse:
        X_fit = sp.csc_array(X)
    else:
        X_fit = X

    clf = NormalInverseGammaRegressor(sparse=sparse)
    clf.fit(X_fit, y)

    if sparse:
        assert isinstance(clf.cov_inv_, sp.csc_array)

    pred = clf.predict(X_fit)

    assert pred.shape == (obs,)

    single_pred = clf.predict(X_fit[[0]])  # type: ignore
    assert single_pred.shape == (1,)


@pytest.mark.parametrize("obs", [1, 2, 3, 4])
@pytest.mark.parametrize("covariates", [1, 2, 3, 4])
@pytest.mark.parametrize("size", [1, 2, 3, 4])
@pytest.mark.parametrize("sparse", [True, False])
def test_normal_inverse_gamma_regressor_sample_covariates(
    obs: Literal[1, 2, 3, 4],
    covariates: Literal[1, 2, 3, 4],
    size: Literal[1, 2, 3, 4],
    sparse: bool,
):
    """Test NormalRegressor predict with covariates."""

    X = np.random.rand(obs, covariates)
    y = np.random.rand(obs)

    if sparse:
        X_fit = sp.csc_array(X)
    else:
        X_fit = X

    clf = NormalInverseGammaRegressor(sparse=sparse)
    clf.fit(X_fit, y)

    if sparse:
        assert isinstance(clf.cov_inv_, sp.csc_array)

    pred = clf.sample(X_fit, size=size)

    assert pred.shape == (size, obs)

    single_pred = clf.sample(X_fit[[0]], size=size)  # type: ignore
    assert single_pred.shape == (size, 1)


class TestDenseVsSparseLearnedPrecisionMatrix:
    @pytest.fixture(scope="class", params=[None] * 5)
    def X_y(self, request):
        X, y, _ = make_regression(
            n_samples=100, n_features=500, random_state=None, coef=True
        )
        # Clip X values near zero to make sparse
        X[X < 0.1] = 0
        # Scale X to be bigger
        X *= 10
        return X, y

    def test_learned_precision_matrix_is_identical_normal(
        self, X_y, suitesparse_envvar
    ):
        X, y = X_y
        sparse_clf = NormalRegressor(alpha=1, beta=1, sparse=True, random_state=0)
        dense_clf = NormalRegressor(alpha=1, beta=1, sparse=False, random_state=0)
        sparse_clf.fit(X, y)
        dense_clf.fit(X, y)
        assert isinstance(sparse_clf.cov_inv_, sp.csc_array)
        assert isinstance(dense_clf.cov_inv_, np.ndarray)

        assert_almost_equal(sparse_clf.cov_inv_.toarray(), dense_clf.cov_inv_)  # type: ignore
        assert_almost_equal(sparse_clf.coef_, dense_clf.coef_)

    def test_learned_precision_matrix_is_identical_normal_inverse_gamma(
        self, X_y, suitesparse_envvar
    ):
        X, y = X_y
        sparse_clf = NormalInverseGammaRegressor(sparse=True, random_state=0)
        dense_clf = NormalInverseGammaRegressor(sparse=False, random_state=0)
        sparse_clf.fit(X, y)
        dense_clf.fit(X, y)
        assert isinstance(sparse_clf.cov_inv_, sp.csc_array)
        assert isinstance(dense_clf.cov_inv_, np.ndarray)

        assert_almost_equal(sparse_clf.cov_inv_.toarray(), dense_clf.cov_inv_)  # type: ignore
        assert_almost_equal(sparse_clf.coef_, dense_clf.coef_)


@pytest.mark.parametrize("sparse", [True, False])
def test_normal_inverse_gamma_regressor_fit_with_weights(sparse: bool) -> None:
    """Test that sample weights affect the posterior correctly."""
    X, y = make_regression(n_samples=10, n_features=2, noise=0.1, random_state=42)  # type: ignore

    # Fit without weights
    reg1 = NormalInverseGammaRegressor(sparse=sparse, random_state=0)
    if sparse:
        X_fit = sp.csc_array(X)
    else:
        X_fit = X
    reg1.fit(X_fit, y)

    # Fit with non-uniform weights
    weights = np.array([0.5, 2.0, 1.0, 0.5, 2.0, 1.0, 0.5, 2.0, 1.0, 0.5])
    reg2 = NormalInverseGammaRegressor(sparse=sparse, random_state=0)
    reg2.fit(X_fit, y, sample_weight=weights)

    # Coefficients should be different
    assert not np.allclose(reg1.coef_, reg2.coef_)

    # Covariance should also be different
    if sparse:
        assert not np.allclose(reg1.cov_inv_.toarray(), reg2.cov_inv_.toarray())  # type: ignore
    else:
        assert not np.allclose(reg1.cov_inv_, reg2.cov_inv_)  # type: ignore

    # Variance parameters should also be different
    assert reg1.a_ != reg2.a_
    assert reg1.b_ != reg2.b_


@pytest.mark.parametrize("sparse", [True, False])
def test_normal_inverse_gamma_regressor_weight_equals_duplication(sparse: bool) -> None:
    """Test that weight=2 is equivalent to seeing a sample twice."""
    X = np.array([[1, 2], [3, 4], [5, 6]])
    y = np.array([1, 2, 3])

    # Fit with duplicated data
    reg1 = NormalInverseGammaRegressor(sparse=sparse, random_state=0)
    X_dup = np.vstack([X, X])
    y_dup = np.hstack([y, y])
    if sparse:
        X_dup_fit = sp.csc_array(X_dup)
    else:
        X_dup_fit = X_dup
    reg1.fit(X_dup_fit, y_dup)

    # Fit with weight=2
    reg2 = NormalInverseGammaRegressor(sparse=sparse, random_state=0)
    weights = np.array([2.0, 2.0, 2.0])
    if sparse:
        X_fit = sp.csc_array(X)
    else:
        X_fit = X
    reg2.fit(X_fit, y, sample_weight=weights)

    assert_almost_equal(reg1.coef_, reg2.coef_)
    if sparse:
        assert_almost_equal(reg1.cov_inv_.toarray(), reg2.cov_inv_.toarray())  # type: ignore
    else:
        assert_almost_equal(reg1.cov_inv_, reg2.cov_inv_)  # type: ignore
    assert_almost_equal(reg1.a_, reg2.a_)
    assert_almost_equal(reg1.b_, reg2.b_)


@pytest.mark.parametrize("sparse", [True, False])
def test_normal_inverse_gamma_regressor_zero_weights(sparse: bool) -> None:
    """Test that samples with weight=0 are effectively ignored."""
    X = np.array([[1, 2], [3, 4], [5, 6], [7, 8]])
    y = np.array([1, 2, 3, 4])

    # Fit on subset
    reg1 = NormalInverseGammaRegressor(sparse=sparse, random_state=0)
    X_subset = X[[0, 2]]
    y_subset = y[[0, 2]]
    if sparse:
        X_subset_fit = sp.csc_array(X_subset)
    else:
        X_subset_fit = X_subset
    reg1.fit(X_subset_fit, y_subset)

    # Fit on full data with zero weights
    reg2 = NormalInverseGammaRegressor(sparse=sparse, random_state=0)
    weights = np.array([1.0, 0.0, 1.0, 0.0])
    if sparse:
        X_fit = sp.csc_array(X)
    else:
        X_fit = X
    reg2.fit(X_fit, y, sample_weight=weights)

    assert_almost_equal(reg1.coef_, reg2.coef_)
    if sparse:
        assert_almost_equal(reg1.cov_inv_.toarray(), reg2.cov_inv_.toarray())  # type: ignore
    else:
        assert_almost_equal(reg1.cov_inv_, reg2.cov_inv_)  # type: ignore
    assert_almost_equal(reg1.a_, reg2.a_)
    assert_almost_equal(reg1.b_, reg2.b_)


@pytest.mark.parametrize("sparse", [True, False])
def test_normal_inverse_gamma_regressor_all_zero_weights(sparse: bool) -> None:
    """Test edge case where all weights are zero."""
    X = np.array([[1, 2], [3, 4]])
    y = np.array([1, 2])

    reg = NormalInverseGammaRegressor(
        mu=0.0, lam=1.0, a=2.0, b=2.0, sparse=sparse, random_state=0
    )
    weights = np.zeros(len(y))
    if sparse:
        X_fit = sp.csc_array(X)
    else:
        X_fit = X
    reg.fit(X_fit, y, sample_weight=weights)

    # Should only have the prior
    assert_almost_equal(reg.coef_, np.zeros(X.shape[1]))
    if sparse:
        assert_almost_equal(reg.cov_inv_.toarray(), np.eye(X.shape[1]))  # type: ignore
    else:
        assert_almost_equal(reg.cov_inv_, np.eye(X.shape[1]))  # type: ignore
    # Variance parameters should remain at prior
    assert_almost_equal(reg.a_, 2.0)
    assert_almost_equal(reg.b_, 2.0)


@pytest.mark.parametrize("sparse", [True, False])
def test_normal_inverse_gamma_regressor_partial_fit_with_weights(sparse: bool) -> None:
    """Test partial_fit with sample weights."""
    X, y = make_regression(n_samples=10, n_features=2, noise=0.1, random_state=42)  # type: ignore
    weights = np.random.rand(10) * 2  # Random weights between 0 and 2

    # Single fit with learning_rate=1 (no decay)
    reg1 = NormalInverseGammaRegressor(learning_rate=1.0, sparse=sparse, random_state=0)
    if sparse:
        X_fit = sp.csc_array(X)
    else:
        X_fit = X
    reg1.fit(X_fit, y, sample_weight=weights)

    # Sequential partial_fit with learning_rate=1 (no decay)
    reg2 = NormalInverseGammaRegressor(learning_rate=1.0, sparse=sparse, random_state=0)
    if sparse:
        X1_fit = sp.csc_array(X[:5])
        X2_fit = sp.csc_array(X[5:])
    else:
        X1_fit = X[:5]
        X2_fit = X[5:]
    reg2.partial_fit(X1_fit, y[:5], sample_weight=weights[:5])
    reg2.partial_fit(X2_fit, y[5:], sample_weight=weights[5:])

    assert_almost_equal(reg1.coef_, reg2.coef_)
    if sparse:
        assert_almost_equal(reg1.cov_inv_.toarray(), reg2.cov_inv_.toarray())  # type: ignore
    else:
        assert_almost_equal(reg1.cov_inv_, reg2.cov_inv_)  # type: ignore
    assert_almost_equal(reg1.a_, reg2.a_)
    assert_almost_equal(reg1.b_, reg2.b_)


@pytest.mark.parametrize("sparse", [True, False])
def test_normal_inverse_gamma_regressor_mathematical_correctness(sparse: bool) -> None:
    """Test mathematical correctness of weighted Bayesian linear regression with unknown variance."""
    # Simple 1D case for easy manual calculation
    X = np.array([[1], [2], [3]])
    y = np.array([1, 2, 3])
    weights = np.array([2.0, 1.0, 0.5])

    # Use simple priors
    mu = 0.0
    lam = 1.0
    a = 1.0
    b = 1.0

    reg = NormalInverseGammaRegressor(
        mu=mu, lam=lam, a=a, b=b, sparse=sparse, random_state=0
    )
    if sparse:
        X_fit = sp.csc_array(X)
    else:
        X_fit = X
    reg.fit(X_fit, y, sample_weight=weights)

    # Manual calculation
    # X^T W X = 10.5 (same as NormalRegressor test)
    # X^T W y = 10.5
    # V_n = lam + X^T W X = 1 + 10.5 = 11.5
    # m_n = V_n^(-1) * (lam * mu + X^T W y) = (1/11.5) * (0 + 10.5) = 10.5/11.5

    # For variance parameters:
    # a_n = a + 0.5 * sum(weights) = 1 + 0.5 * 3.5 = 2.75
    # b_n = b + 0.5 * (y^T W y + mu * lam * mu - m_n * V_n * m_n)
    #     = 1 + 0.5 * (weighted_y_squared - m_n^2 * V_n)

    expected_coef = 10.5 / 11.5
    expected_cov_inv = 11.5
    expected_a = 2.75
    expected_b = 1.0 + 0.5 * (
        np.sum(weights * y**2) - (expected_coef**2 * expected_cov_inv)
    )

    assert_almost_equal(reg.coef_[0], expected_coef, decimal=6)
    if sparse:
        assert_almost_equal(reg.cov_inv_[0, 0], expected_cov_inv, decimal=6)
    else:
        assert_almost_equal(reg.cov_inv_[0, 0], expected_cov_inv, decimal=6)
    assert_almost_equal(reg.a_, expected_a, decimal=6)
    assert_almost_equal(reg.b_, expected_b, decimal=6)


@pytest.mark.parametrize("sparse", [True, False])
def test_normal_inverse_gamma_regressor_weights_learning_rate_interaction(
    sparse: bool,
) -> None:
    """Test that sample weights and learning_rate interact correctly."""
    X = np.array([[1, 0], [0, 1], [1, 1]])
    y = np.array([1, 2, 3])
    weights = np.array([2.0, 1.0, 0.5])

    # With learning rate < 1
    # Prior parameters need to compensate for decay
    reg = NormalInverseGammaRegressor(
        mu=0.0, lam=8.0, a=8.0, b=8.0, learning_rate=0.5, sparse=sparse, random_state=0
    )
    if sparse:
        X_fit = sp.csc_array(X)
    else:
        X_fit = X
    reg.fit(X_fit, y, sample_weight=weights)

    # Fit same effective model with no decay
    reg_uniform = NormalInverseGammaRegressor(
        mu=0.0, lam=1.0, a=1.0, b=1.0, learning_rate=1.0, sparse=sparse, random_state=0
    )
    reg_uniform.fit(X_fit, y, sample_weight=np.array([0.5, 0.5, 0.5]))

    # Should be very close (exact up to numerical precision)
    assert_almost_equal(reg.coef_, reg_uniform.coef_)
    if sparse:
        assert_almost_equal(reg.cov_inv_.toarray(), reg_uniform.cov_inv_.toarray())  # type: ignore
    else:
        assert_almost_equal(reg.cov_inv_, reg_uniform.cov_inv_)  # type: ignore
    assert_almost_equal(reg.a_, reg_uniform.a_)
    assert_almost_equal(reg.b_, reg_uniform.b_)


def test_normal_inverse_gamma_regressor_weight_validation() -> None:
    """Test input validation for weights."""
    X = np.array([[1, 2], [3, 4]])
    y = np.array([1, 2])

    reg = NormalInverseGammaRegressor()

    # Wrong shape
    with pytest.raises(ValueError, match="sample_weight.shape"):
        reg.fit(X, y, sample_weight=np.array([1.0]))  # Too few

    with pytest.raises(ValueError, match="sample_weight.shape"):
        reg.fit(X, y, sample_weight=np.array([1.0, 2.0, 3.0]))  # Too many


@pytest.mark.parametrize("sparse", [True, False])
def test_normal_inverse_gamma_regressor_sample_with_weights(sparse: bool) -> None:
    """Test that sampling works correctly with weighted training."""
    X = np.array([[1, 0], [0, 1], [1, 1]])
    y = np.array([1, 2, 3])
    weights = np.array([2.0, 1.0, 0.5])

    reg = NormalInverseGammaRegressor(sparse=sparse, random_state=0)
    if sparse:
        X_fit = sp.csc_array(X)
    else:
        X_fit = X
    reg.fit(X_fit, y, sample_weight=weights)

    # Should be able to sample without errors (samples from multivariate t)
    samples = reg.sample(X_fit, size=10)
    assert samples.shape == (10, 3)
    assert np.all(np.isfinite(samples))


@pytest.mark.parametrize("sparse", [True, False])
def test_normal_inverse_gamma_regressor_variance_update_with_weights(
    sparse: bool,
) -> None:
    """Test that variance parameters are updated correctly with weights."""
    X = np.array([[1], [1], [1]])
    y = np.array([1, 2, 3])

    # All same weight
    reg1 = NormalInverseGammaRegressor(a=1.0, b=1.0, sparse=sparse, random_state=0)
    if sparse:
        X_fit = sp.csc_array(X)
    else:
        X_fit = X
    reg1.fit(X_fit, y, sample_weight=np.array([1.0, 1.0, 1.0]))

    # Different weights that sum to 3
    reg2 = NormalInverseGammaRegressor(a=1.0, b=1.0, sparse=sparse, random_state=0)
    reg2.fit(X_fit, y, sample_weight=np.array([0.5, 2.0, 0.5]))

    # a_n should be same (depends on sum of weights)
    assert_almost_equal(reg1.a_, reg2.a_)

    # b_n should be different (depends on weighted sum of squares)
    assert reg1.b_ != reg2.b_
