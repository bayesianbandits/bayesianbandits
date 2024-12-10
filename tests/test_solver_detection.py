import sys

import pytest


@pytest.fixture
def no_cholmod(monkeypatch: pytest.MonkeyPatch):
    monkeypatch.setitem(sys.modules, "sksparse.cholmod", None)


@pytest.fixture
def no_suitesparse_env_vars(monkeypatch: pytest.MonkeyPatch):
    monkeypatch.setenv("BB_NO_SUITESPARSE", "1")


def test_no_suitesparse(no_cholmod: None):
    if "bayesianbandits._sparse_bayesian_linear_regression" in sys.modules:
        del sys.modules["bayesianbandits._sparse_bayesian_linear_regression"]
    from bayesianbandits._sparse_bayesian_linear_regression import SparseSolver, solver

    assert solver == SparseSolver.SUPERLU


def test_yes_cholmod_no_umfpack() -> None:
    if "bayesianbandits._sparse_bayesian_linear_regression" in sys.modules:
        del sys.modules["bayesianbandits._sparse_bayesian_linear_regression"]
    from bayesianbandits._sparse_bayesian_linear_regression import SparseSolver, solver

    assert solver == SparseSolver.CHOLMOD


def test_yes_cholmod_yes_umfpack() -> None:
    if "bayesianbandits._sparse_bayesian_linear_regression" in sys.modules:
        del sys.modules["bayesianbandits._sparse_bayesian_linear_regression"]
    from bayesianbandits._sparse_bayesian_linear_regression import SparseSolver, solver

    assert solver == SparseSolver.CHOLMOD


def test_no_suitesparse_env_vars(no_suitesparse_env_vars: None):
    if "bayesianbandits._sparse_bayesian_linear_regression" in sys.modules:
        del sys.modules["bayesianbandits._sparse_bayesian_linear_regression"]
    from bayesianbandits._sparse_bayesian_linear_regression import SparseSolver, solver

    assert solver == SparseSolver.SUPERLU
