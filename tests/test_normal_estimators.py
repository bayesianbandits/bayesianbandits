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
