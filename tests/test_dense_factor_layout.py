"""Issue #298: scipy 1.18 ``cho_factor`` returns C-ordered factors, and
every f2py solve against such a factor transposes it first. The dense
paths factor through ``dpotrf`` instead, so the cached factor stays
Fortran-contiguous on every scipy version.
"""

from typing import Any, Callable

import numpy as np
import pytest
from scipy.linalg import LinAlgError

from bayesianbandits import (
    BayesianGLM,
    EmpiricalBayesNormalRegressor,
    NormalInverseGammaRegressor,
    NormalRegressor,
)
from bayesianbandits._blas_helpers import cho_factor_f
from bayesianbandits._sparse_bayesian_linear_regression import DenseFactor

ESTIMATORS = {
    "normal": lambda: NormalRegressor(alpha=1.0, beta=1.0),
    "nig": lambda: NormalInverseGammaRegressor(mu=0.0, lam=1.0, a=1.0, b=1.0),
    "eb-normal": lambda: EmpiricalBayesNormalRegressor(alpha=1.0, beta=1.0),
    "glm-laplace": lambda: BayesianGLM(alpha=1.0, link="logit"),
}


def _factor(model: Any) -> DenseFactor:
    factor = model._precision_factor
    assert isinstance(factor, DenseFactor)
    return factor


@pytest.mark.parametrize("make", ESTIMATORS.values(), ids=list(ESTIMATORS))
def test_cached_factor_is_fortran_after_fit(make: Callable[[], Any]) -> None:
    rng = np.random.default_rng(0)
    X = rng.standard_normal((40, 6))
    y = rng.integers(0, 2, 40).astype(float)  # binary, valid for every link
    model = make()
    model.partial_fit(X, y)
    assert _factor(model)._U.flags.f_contiguous
    model.partial_fit(X, y)
    assert _factor(model)._U.flags.f_contiguous


def test_lazy_factor_is_fortran() -> None:
    model = NormalRegressor(alpha=1.0, beta=1.0)
    model.partial_fit(np.eye(3), np.ones(3))
    del model._precision_factor  # force the lazy cached_property path
    assert _factor(model)._U.flags.f_contiguous


def test_cho_factor_f_matches_scipy_and_rejects_indefinite() -> None:
    rng = np.random.default_rng(1)
    A = rng.standard_normal((5, 5))
    P = np.asfortranarray(A @ A.T + 5 * np.eye(5))
    U, lower = cho_factor_f(P)
    assert not lower and U.flags.f_contiguous
    np.testing.assert_allclose(np.triu(U).T @ np.triu(U), P, rtol=1e-12)
    with pytest.raises(LinAlgError, match="leading minor"):
        cho_factor_f(np.asfortranarray(-np.eye(3)))
