from pathlib import Path
from typing import cast
from unittest.mock import patch

import joblib
import numpy as np
import pytest
import scipy.sparse as sp
from numpy.testing import assert_allclose, assert_array_almost_equal
from scipy.linalg import cholesky
from scipy.stats import Covariance, multivariate_normal, multivariate_t
from sklearn.datasets import make_regression

from bayesianbandits._estimators import multivariate_t_sample_from_covariance
from bayesianbandits._sparse_bayesian_linear_regression import (
    CovViaSparsePrecision,
    multivariate_normal_sample_from_sparse_covariance,
    multivariate_t_sample_from_sparse_covariance,
    sparse_cholesky,
)


@pytest.fixture(params=["identity", "diag"])
def sparse_array(request):
    if request.param == "identity":
        return cast(sp.csc_array, sp.eye(100, format="csc"))
    elif request.param == "diag":
        return cast(sp.csc_array, sp.diags([5] * 100, format="csc"))


@pytest.fixture(params=[True, False], ids=["suitesparse", "no_suitesparse"])
def suitesparse_envvar(request, monkeypatch):
    """Allows running test suite with and without CHOLMOD."""
    with patch(
        "bayesianbandits._sparse_bayesian_linear_regression.use_suitesparse",
        request.param,
    ):
        yield


@pytest.mark.usefixtures("suitesparse_envvar")
class TestSparseCholesky:
    def test_sparse_cholesky(self, sparse_array):
        chol = sparse_cholesky(sparse_array)

        assert_array_almost_equal(chol.toarray(), cholesky(sparse_array.toarray()))

    def test_sparse_cholesky_ill_conditioned_matrices(self):
        this_file_path = Path(__file__)
        test_data_dir = this_file_path.parent / "ill_conditioned_matrices"
        for file_path in test_data_dir.glob("*"):
            sparse_array = joblib.load(file_path)
            sparse_cholesky(sparse_array)


@pytest.mark.parametrize("size", [1, 10])
def test_multivariate_normal_sample_from_sparse_covariance_ill_conditioned_matrices(
    size,
):
    this_file_path = Path(__file__)
    test_data_dir = this_file_path.parent / "ill_conditioned_matrices"
    for file_path in test_data_dir.glob("*"):
        sparse_array = joblib.load(file_path)
        cov = CovViaSparsePrecision(sparse_array)
        samples = multivariate_normal_sample_from_sparse_covariance(
            mean=None, cov=cov, size=size
        )
        assert samples.shape == (size, sparse_array.shape[0])


class TestCovViaSparsePrecision:
    @pytest.fixture(scope="class")
    def X_y(self):
        X, y, _ = make_regression(
            n_samples=100, n_features=500, random_state=None, coef=True
        )
        # Clip X values near zero to make sparse
        X[X < 0.1] = 0
        # Scale X to be bigger
        X *= 10
        return X, y

    @pytest.fixture(scope="class")
    def precision_matrix(self, X_y):
        X, y = X_y

        return X.T @ X + np.eye(X.shape[1])

    def test_inversion(self, precision_matrix):
        sparse_cov = CovViaSparsePrecision(
            sp.csc_array(precision_matrix), use_suitesparse=False
        )
        scipy_cov = Covariance.from_precision(precision_matrix)

        assert_array_almost_equal(scipy_cov.covariance, sparse_cov.covariance.toarray())

    @pytest.mark.parametrize(
        "matrix",
        [
            joblib.load(file)
            for file in Path(__file__).parent.glob("ill_conditioned_matrices/*")
        ],
    )
    def test_colorize_ill_conditioned_matrices(self, matrix):
        """
        These aren't actually going to be the same, but they should be close. We'll
        test by taking a large number of samples and checking that the variances are
        close.
        """
        scipy_sparse_cov = CovViaSparsePrecision(
            sp.csc_array(matrix), use_suitesparse=False
        )
        suitesparse_cov = CovViaSparsePrecision(
            sp.csc_array(matrix), use_suitesparse=True
        )

        random_samples = np.random.default_rng(0).normal(size=(1000, matrix.shape[0]))
        scipy_samples = scipy_sparse_cov.colorize(random_samples)
        suitesparse_samples = suitesparse_cov.colorize(random_samples)

        assert_allclose(
            scipy_samples.var(axis=0), suitesparse_samples.var(axis=0), rtol=0.5
        )

    def test_mvn_sample_no_suitesparse(self, precision_matrix):
        sparse_cov = CovViaSparsePrecision(
            sp.csc_array(precision_matrix), use_suitesparse=False
        )
        suitesparse_cov = CovViaSparsePrecision(
            sp.csc_array(precision_matrix), use_suitesparse=True
        )
        scipy_cov = Covariance.from_precision(precision_matrix)

        sparse_samples = multivariate_normal_sample_from_sparse_covariance(
            mean=None, cov=sparse_cov, size=80000, random_state=0
        )
        suitesparse_samples = multivariate_normal_sample_from_sparse_covariance(
            mean=None, cov=suitesparse_cov, size=80000, random_state=0
        )
        scipy_samples = multivariate_normal.rvs(
            mean=None, cov=scipy_cov, size=80000, random_state=0  # type: ignore
        )

        sparse_emp_cov = np.cov(sparse_samples.T)
        suitesparse_emp_cov = np.cov(suitesparse_samples.T)
        scipy_emp_cov = np.cov(scipy_samples.T)

        assert_allclose(sparse_emp_cov, suitesparse_emp_cov, atol=0.03)
        assert_allclose(sparse_emp_cov, scipy_emp_cov, atol=0.03)
        assert_allclose(suitesparse_emp_cov, scipy_emp_cov, atol=0.03)

    def test_mvn_sample_suitesparse(self, precision_matrix):
        sparse_cov = CovViaSparsePrecision(
            sp.csc_array(precision_matrix), use_suitesparse=True
        )
        scipy_cov = Covariance.from_precision(precision_matrix)

        rs_1 = np.random.default_rng(0)
        rs_2 = np.random.default_rng(0)

        sparse_samples = multivariate_normal_sample_from_sparse_covariance(
            mean=None, cov=sparse_cov, size=80000, random_state=rs_1
        )
        scipy_samples = multivariate_normal.rvs(
            mean=None, cov=scipy_cov, size=80000, random_state=rs_2  # type: ignore
        )

        assert_allclose(
            np.cov(sparse_samples.T),
            np.cov(scipy_samples.T),
            atol=0.02,
        )

    def test_mvt_sample_scipy_vs_bb(self, precision_matrix):
        scipy_cov = Covariance.from_precision(precision_matrix)

        rs_1 = np.random.default_rng(0)
        rs_2 = np.random.default_rng(0)

        # The way that multivariate_t uses the random state is different from
        # how we use it, so we're going to mock out the multivariate normal sampling
        # in our implementation (as our above tests show that it's correct) and
        # just test the t distribution sampling.
        with patch("bayesianbandits._estimators.multivariate_normal.rvs") as mock_mvn:
            mock_mvn.side_effect = lambda mean, shape, size, random_state: random_state.multivariate_normal(
                np.zeros(scipy_cov.shape[0]), shape, size=size
            )
            sparse_samples = multivariate_t_sample_from_covariance(
                loc=None, shape=scipy_cov.covariance, size=100, random_state=rs_1, df=3
            )

        scipy_samples = multivariate_t.rvs(
            loc=None, shape=scipy_cov.covariance, size=100, random_state=rs_2, df=3
        )

        assert_array_almost_equal(sparse_samples, scipy_samples)

    def test_mvt_sampling_against_scipy(self, precision_matrix):
        sparse_cov = CovViaSparsePrecision(
            sp.csc_array(precision_matrix), use_suitesparse=False
        )
        suitesparse_cov = CovViaSparsePrecision(
            sp.csc_array(precision_matrix), use_suitesparse=True
        )
        scipy_cov = Covariance.from_precision(precision_matrix)

        sparse_samples = multivariate_t_sample_from_covariance(
            loc=None, shape=sparse_cov, size=80000, random_state=0, df=300
        )
        suitesparse_samples = multivariate_t_sample_from_covariance(
            loc=None, shape=suitesparse_cov, size=80000, random_state=0, df=300
        )
        scipy_samples = multivariate_t_sample_from_sparse_covariance(
            loc=None, shape=scipy_cov, size=80000, random_state=0, df=300  # type: ignore
        )

        sparse_emp_cov = np.cov(sparse_samples.T)
        suitesparse_emp_cov = np.cov(suitesparse_samples.T)
        scipy_emp_cov = np.cov(scipy_samples.T)

        assert_allclose(sparse_emp_cov, suitesparse_emp_cov, atol=0.03)
        assert_allclose(sparse_emp_cov, scipy_emp_cov, atol=0.03)
        assert_allclose(suitesparse_emp_cov, scipy_emp_cov, atol=0.03)
