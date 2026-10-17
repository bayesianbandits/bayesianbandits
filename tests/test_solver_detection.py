import sys

import pytest

import bayesianbandits

MODULE = "bayesianbandits._sparse_bayesian_linear_regression"


@pytest.fixture(autouse=True)
def reimported_module(monkeypatch: pytest.MonkeyPatch):
    """Each test imports the module afresh to rerun solver detection. The
    original goes back afterwards, both in ``sys.modules`` and on the
    package, which the re-import rebinds: the estimators hold the
    original's functions, while lazy imports and ``mock.patch`` resolve
    through these two, so a leaked copy splits them between modules."""
    original = sys.modules[MODULE]
    monkeypatch.delitem(sys.modules, MODULE)
    monkeypatch.setattr(bayesianbandits, "_sparse_bayesian_linear_regression", original)


@pytest.fixture
def no_cholmod(monkeypatch: pytest.MonkeyPatch):
    monkeypatch.setitem(sys.modules, "sksparse.cholmod", None)


@pytest.fixture
def no_suitesparse_env_vars(monkeypatch: pytest.MonkeyPatch):
    monkeypatch.setenv("BB_NO_SUITESPARSE", "1")


def test_no_suitesparse(no_cholmod: None):
    from bayesianbandits._sparse_bayesian_linear_regression import SparseSolver, solver

    assert solver == SparseSolver.SUPERLU


def test_yes_cholmod_no_umfpack() -> None:
    from bayesianbandits._sparse_bayesian_linear_regression import SparseSolver, solver

    assert solver == SparseSolver.CHOLMOD


def test_yes_cholmod_yes_umfpack() -> None:
    from bayesianbandits._sparse_bayesian_linear_regression import SparseSolver, solver

    assert solver == SparseSolver.CHOLMOD


def test_no_suitesparse_env_vars(no_suitesparse_env_vars: None):
    from bayesianbandits._sparse_bayesian_linear_regression import SparseSolver, solver

    assert solver == SparseSolver.SUPERLU
