"""Shared fixtures for the test suite.

Plain (non-fixture) helpers live in ``tests/_helpers.py`` -- importing
them from here would be ambiguous, since ``benchmarks/conftest.py``
shadows the ``conftest`` module name when pytest runs from the repo
root.
"""

from unittest import mock

import pytest

from bayesianbandits._sparse_bayesian_linear_regression import SparseSolver


@pytest.fixture(params=[SparseSolver.SUPERLU, SparseSolver.CHOLMOD])
def sparse_solver(request):
    """Run a test against both SuperLU and CHOLMOD factors.

    Modules that want every test solver-parameterized wrap this in a
    module-level autouse fixture::

        @pytest.fixture(autouse=True)
        def suitesparse_envvar(sparse_solver):
            yield
    """
    with mock.patch(
        "bayesianbandits._sparse_bayesian_linear_regression.solver", request.param
    ):
        yield request.param
