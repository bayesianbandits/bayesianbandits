import pytest
from typing import cast
import scipy.sparse as sp
from bayesianbandits._sparse_bayesian_linear_regression import (
    sparse_cholesky,
    CovViaSparsePrecision,
    multivariate_normal_sample_from_sparse_covariance,
)
from scipy.linalg import cholesky
from numpy.testing import assert_array_almost_equal
import joblib
from pathlib import Path


@pytest.fixture(params=[0, 1], autouse=True, ids=["cholmod", "no_cholmod"])
def cholmod_envvar(request, monkeypatch):
    """Allows running test suite with and without CHOLMOD."""
    monkeypatch.setenv("BB_NO_SUITESPARSE", str(request.param))
    yield request.param
    monkeypatch.delenv("BB_NO_SUITESPARSE")


@pytest.fixture(params=["identity", "diag"])
def sparse_array(request):
    if request.param == "identity":
        return cast(sp.csc_array, sp.eye(100, format="csc"))
    elif request.param == "diag":
        return cast(sp.csc_array, sp.diags([5] * 100, format="csc"))


def test_sparse_cholesky(sparse_array):
    chol = sparse_cholesky(sparse_array)

    assert_array_almost_equal(chol.toarray(), cholesky(sparse_array.toarray()))


def test_sparse_cholesky_ill_conditioned_matrices():
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
