from pathlib import Path
from typing import cast

import joblib
import numpy as np
import pytest
import scipy.sparse as sp
from numpy.testing import assert_allclose, assert_array_almost_equal
from scipy.linalg import cholesky
from scipy.stats import Covariance
from sklearn.datasets import make_regression

from bayesianbandits._sparse_bayesian_linear_regression import (
    CovViaSparsePrecision,
    multivariate_normal_sample_from_sparse_covariance,
    sparse_cholesky,
)


@pytest.fixture(params=["identity", "diag"])
def sparse_array(request):
    if request.param == "identity":
        return cast(sp.csc_array, sp.eye(100, format="csc"))
    elif request.param == "diag":
        return cast(sp.csc_array, sp.diags([5] * 100, format="csc"))


@pytest.fixture(params=[0, 1], ids=["suitesparse", "no_suitesparse"])
def suitesparse_envvar(request, monkeypatch):
    """Allows running test suite with and without CHOLMOD."""
    monkeypatch.setenv("BB_NO_SUITESPARSE", str(request.param))
    yield request.param
    monkeypatch.delenv("BB_NO_SUITESPARSE")


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
    @pytest.fixture(scope="class", params=[None] * 10)
    def X_y(self, request):
        X, y, _ = make_regression(
            n_samples=100, n_features=500, random_state=request.param, coef=True
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

    def test_colorize_no_suitesparse(self, precision_matrix):
        sparse_cov = CovViaSparsePrecision(
            sp.csc_array(precision_matrix), use_suitesparse=False
        )
        scipy_cov = Covariance.from_precision(precision_matrix)

        random_samples = np.random.default_rng(0).normal(
            size=(100, precision_matrix.shape[0])
        )

        assert_array_almost_equal(
            scipy_cov.colorize(random_samples), sparse_cov.colorize(random_samples)
        )

    def test_colorize_suitesparse(self, precision_matrix):
        sparse_cov = CovViaSparsePrecision(
            sp.csc_array(precision_matrix), use_suitesparse=True
        )
        scipy_cov = Covariance.from_precision(precision_matrix)

        random_samples = np.random.default_rng(0).normal(
            size=(100, precision_matrix.shape[0])
        )

        assert_array_almost_equal(
            scipy_cov.colorize(random_samples), sparse_cov.colorize(random_samples)
        )

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
