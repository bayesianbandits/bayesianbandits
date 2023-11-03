import pytest
from typing import cast
import scipy.sparse as sp
from bayesianbandits._sparse_bayesian_linear_regression import sparse_cholesky
from scipy.linalg import cholesky
from numpy.testing import assert_array_almost_equal
import joblib
from pathlib import Path


@pytest.fixture(params=["identity", "diag"])
def sparse_matrix(request):
    if request.param == "identity":
        return cast(sp.csc_matrix, sp.eye(100, format="csc"))
    elif request.param == "diag":
        return cast(sp.csc_matrix, sp.diags([5] * 100, format="csc"))


def test_sparse_cholesky(sparse_matrix):
    chol = sparse_cholesky(sparse_matrix)

    assert_array_almost_equal(chol.toarray(), cholesky(sparse_matrix.toarray()))


def test_sparse_cholesky_ill_conditioned_matrices():
    this_file_path = Path(__file__)
    test_data_dir = this_file_path.parent / "ill_conditioned_matrices"
    for file_path in test_data_dir.glob("*"):
        sparse_matrix = joblib.load(file_path)
        sparse_cholesky(sparse_matrix)
