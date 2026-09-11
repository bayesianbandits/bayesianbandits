"""Issue #296: the dense ``partial_fit`` paths hand ``X`` to ``dsyrk`` and
``dgemv`` through a Fortran-contiguous view, so f2py never copies a
C-ordered ``X`` on the way in. The wrappers below fail the test the moment
a BLAS call receives a matrix operand that would be copied.
"""

from typing import Any, Callable, Literal

import numpy as np
import pytest

from bayesianbandits import (
    EmpiricalBayesNormalRegressor,
    NormalInverseGammaRegressor,
    NormalRegressor,
    _blas_helpers,
    _eb_estimators,
    _estimators,
)

ESTIMATORS = [
    lambda: NormalRegressor(alpha=1.0, beta=1.0),
    lambda: NormalInverseGammaRegressor(mu=0.0, lam=1.0, a=1.0, b=1.0),
    lambda: EmpiricalBayesNormalRegressor(alpha=1.0, beta=1.0),
]


def _guarded(fn: Callable[..., Any]) -> Callable[..., Any]:
    def wrapper(alpha: float, a: np.ndarray, *args: Any, **kwargs: Any) -> Any:
        assert a.flags.f_contiguous, "matrix operand would be copied by f2py"
        return fn(alpha, a, *args, **kwargs)

    return wrapper


@pytest.fixture
def guard_blas(monkeypatch: pytest.MonkeyPatch) -> None:
    for module in (_blas_helpers, _estimators, _eb_estimators):
        for name in ("dsyrk", "dgemv"):
            if hasattr(module, name):
                monkeypatch.setattr(module, name, _guarded(getattr(module, name)))


@pytest.mark.parametrize("make", ESTIMATORS, ids=["normal", "nig", "eb-normal"])
@pytest.mark.parametrize("order", ["C", "F"])
@pytest.mark.usefixtures("guard_blas")
def test_dense_partial_fit_passes_fortran_views(
    make: Callable[[], Any], order: Literal["C", "F"]
) -> None:
    rng = np.random.default_rng(0)
    X = np.asarray(rng.standard_normal((40, 6)), order=order)
    y = rng.standard_normal(40)
    model = make()
    model.partial_fit(X, y)
    model.partial_fit(X, y)  # incremental path, after the prior exists


@pytest.mark.parametrize("make", ESTIMATORS, ids=["normal", "nig", "eb-normal"])
def test_dense_partial_fit_is_layout_invariant(make: Callable[[], Any]) -> None:
    rng = np.random.default_rng(1)
    X = rng.standard_normal((40, 6))
    y = rng.standard_normal(40)
    c, f = make(), make()
    for _ in range(2):
        c.partial_fit(X, y)
        f.partial_fit(np.asfortranarray(X), y)
    np.testing.assert_allclose(c.coef_, f.coef_, rtol=1e-12)
    np.testing.assert_allclose(np.triu(c.cov_inv_), np.triu(f.cov_inv_), rtol=1e-12)
