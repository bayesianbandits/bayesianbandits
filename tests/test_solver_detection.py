import sys

import pytest


@pytest.fixture
def no_cholmod(monkeypatch):
    monkeypatch.setitem(sys.modules, "sksparse.cholmod", None)


@pytest.fixture
def no_umfpack(monkeypatch):
    monkeypatch.setitem(sys.modules, "scikits.umfpack", None)


@pytest.fixture
def no_suitesparse_env_vars(monkeypatch):
    monkeypatch.setenv("BB_NO_SUITESPARSE", "1")


def test_no_suitesparse(no_cholmod, no_umfpack):
    if "bayesianbandits._sparse_bayesian_linear_regression" in sys.modules:
        del sys.modules["bayesianbandits._sparse_bayesian_linear_regression"]
    from bayesianbandits._sparse_bayesian_linear_regression import SparseSolver, solver

    assert solver == SparseSolver.SUPERLU


def test_no_cholmod_yes_umfpack(no_cholmod):
    if "bayesianbandits._sparse_bayesian_linear_regression" in sys.modules:
        del sys.modules["bayesianbandits._sparse_bayesian_linear_regression"]
    from bayesianbandits._sparse_bayesian_linear_regression import SparseSolver, solver

    assert solver == SparseSolver.UMFPACK


@pytest.mark.skipif(
    sys.version_info
    > (
        3,
        11,
    ),
    reason="sksparse is not yet compatible with Python 3.12",
)
def test_yes_cholmod_no_umfpack(no_umfpack):
    if "bayesianbandits._sparse_bayesian_linear_regression" in sys.modules:
        del sys.modules["bayesianbandits._sparse_bayesian_linear_regression"]
    from bayesianbandits._sparse_bayesian_linear_regression import SparseSolver, solver

    assert solver == SparseSolver.CHOLMOD


@pytest.mark.skipif(
    sys.version_info
    > (
        3,
        11,
    ),
    reason="sksparse is not yet compatible with Python 3.12",
)
def test_yes_cholmod_yes_umfpack():
    if "bayesianbandits._sparse_bayesian_linear_regression" in sys.modules:
        del sys.modules["bayesianbandits._sparse_bayesian_linear_regression"]
    from bayesianbandits._sparse_bayesian_linear_regression import SparseSolver, solver

    assert solver == SparseSolver.CHOLMOD


def test_no_suitesparse_env_vars(no_suitesparse_env_vars):
    if "bayesianbandits._sparse_bayesian_linear_regression" in sys.modules:
        del sys.modules["bayesianbandits._sparse_bayesian_linear_regression"]
    from bayesianbandits._sparse_bayesian_linear_regression import SparseSolver, solver

    assert solver == SparseSolver.SUPERLU
