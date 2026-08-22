"""``prior_floor`` (stabilized forgetting re-injection) in the Laplace
and R-VGA posterior updates."""

from __future__ import annotations

from typing import Any

import numpy as np
import pytest
from numpy.typing import NDArray
from scipy.sparse import csc_array
from scipy.special import expit

from bayesianbandits._gaussian import (
    update_gaussian_posterior_laplace,
    update_gaussian_posterior_rvga,
)


def _data(seed: int = 0, n: int = 30, p: int = 4):
    rng = np.random.default_rng(seed)
    X = rng.normal(size=(n, p))
    y = rng.binomial(1, expit(X @ rng.normal(size=p))).astype(np.float64)
    prior_mean = rng.normal(size=p)
    A = rng.normal(size=(p, p))
    prior_precision = A @ A.T + p * np.eye(p)
    return X, y, prior_mean, prior_precision


def _dense(a: Any) -> NDArray[np.float64]:
    return np.asarray(a.toarray() if hasattr(a, "toarray") else a, dtype=np.float64)


def _call(method: str, sparse: bool, X, y, prior_mean, prior_precision, **kw):
    fn = (
        update_gaussian_posterior_laplace
        if method == "laplace"
        else update_gaussian_posterior_rvga
    )
    if sparse:
        X = csc_array(X)
        prior_precision = csc_array(prior_precision)
    post = fn(
        X,
        y,
        prior_mean,
        prior_precision,
        link="logit",
        sparse=sparse,
        n_iter=10,
        tol=1e-10,
        **kw,
    )
    prec = _dense(post.precision)
    return np.asarray(post.mean), np.triu(prec) + np.triu(prec, 1).T


@pytest.mark.parametrize("method", ["laplace", "rvga"])
@pytest.mark.parametrize("sparse", [False, True])
class TestPriorFloor:
    def test_zero_floor_is_a_no_op(self, method: str, sparse: bool) -> None:
        X, y, m, P = _data()
        a = _call(method, sparse, X, y, m, P, learning_rate=0.9)
        b = _call(method, sparse, X, y, m, P, learning_rate=0.9, prior_floor=0.0)
        np.testing.assert_allclose(a[0], b[0])
        np.testing.assert_allclose(a[1], b[1])

    def test_no_decay_ignores_floor(self, method: str, sparse: bool) -> None:
        X, y, m, P = _data()
        a = _call(method, sparse, X, y, m, P)
        b = _call(method, sparse, X, y, m, P, prior_floor=5.0)
        np.testing.assert_allclose(a[0], b[0])
        np.testing.assert_allclose(a[1], b[1])

    def test_equals_zero_centered_prior_shift(self, method: str, sparse: bool) -> None:
        """Decay + floor == a single update whose prior is the product of
        N(m, (γⁿP)⁻¹) and N(0, (sI)⁻¹) with s = (1 - γⁿ)·floor."""
        X, y, m, P = _data()
        lr, floor = 0.95, 2.0
        n = X.shape[0]
        g = lr**n
        s = (1.0 - g) * floor

        got = _call(method, sparse, X, y, m, P, learning_rate=lr, prior_floor=floor)

        # Equivalent prior: precision γⁿP + sI, mean solving (γⁿP + sI) m' = γⁿP m.
        P_eq = g * P + s * np.eye(P.shape[0])
        m_eq = np.linalg.solve(P_eq, g * P @ m)
        # Run with learning_rate=1 but the same per-row weights: the
        # effective weights depend on learning_rate, so replicate them
        # via sample_weight.
        from bayesianbandits._estimators import compute_effective_weights

        w = compute_effective_weights(n, None, lr)
        want = _call(method, sparse, X, y, m_eq, P_eq, sample_weight=w)

        np.testing.assert_allclose(got[0], want[0], atol=1e-8)
        np.testing.assert_allclose(got[1], want[1], atol=1e-8)

    def test_caller_prior_not_mutated(self, method: str, sparse: bool) -> None:
        X, y, m, P = _data()
        P_F = np.asfortranarray(P)
        P_copy = P_F.copy()
        _call(method, False, X, y, m, P_F, learning_rate=0.9, prior_floor=3.0)
        np.testing.assert_array_equal(P_F, P_copy)


@pytest.mark.parametrize("method", ["laplace", "rvga"])
@pytest.mark.parametrize("sparse", [False, True])
@pytest.mark.parametrize("lr", [1.0, 0.95])
def test_prior_shift_is_a_zero_centered_precision_shift(
    method: str, sparse: bool, lr: float
) -> None:
    """prior_shift adds γⁿ·shift to the decayed prior precision only:
    equals an update from the product prior N(m, (γⁿP)⁻¹)·N(0, (γⁿs I)⁻¹).
    With lr=1 the dense scaled prior aliases the caller's array, which
    must come back untouched."""
    X, y, m, P = _data()
    n = X.shape[0]
    g = lr**n
    shift = 0.7
    P_F = np.asfortranarray(P)
    P_copy = P_F.copy()
    got = _call(method, sparse, X, y, m, P_F, learning_rate=lr, prior_shift=shift)
    np.testing.assert_array_equal(P_F, P_copy)

    from bayesianbandits._estimators import compute_effective_weights

    P_eq = g * P + g * shift * np.eye(P.shape[0])
    m_eq = np.linalg.solve(P_eq, g * P @ m)
    w = compute_effective_weights(n, None, lr)
    want = _call(method, sparse, X, y, m_eq, P_eq, sample_weight=w)
    np.testing.assert_allclose(got[0], want[0], atol=1e-8)
    np.testing.assert_allclose(got[1], want[1], atol=1e-8)


def test_rvga_sparse_chunked_shift_applies_once() -> None:
    """Chunked R-VGA applies prior_shift on the first chunk only; later
    chunks decay it with the rest of the prior."""
    _, y, m, P = _data(n=40)
    n = y.shape[0]
    X = np.zeros((n, P.shape[0]))
    lr, shift = 0.97, 0.7
    one, chunked = (
        update_gaussian_posterior_rvga(
            csc_array(X),
            y,
            m,
            csc_array(P),
            link="logit",
            sparse=True,
            learning_rate=lr,
            prior_shift=shift,
            n_iter=3,
            batch_size=bs,
        )
        for bs in (None, 7)
    )
    g = lr**n
    want_prec = g * P + g * shift * np.eye(P.shape[0])
    want_mean = np.linalg.solve(want_prec, g * P @ m)
    for post in (one, chunked):
        np.testing.assert_allclose(_dense(post.precision), want_prec, atol=1e-10)
        np.testing.assert_allclose(_dense(post.mean), want_mean, atol=1e-10)


def test_rvga_sparse_chunked_floor_composes() -> None:
    """Minibatched R-VGA re-injects per chunk.  With no data signal
    (all-zero X) the data Hessian vanishes and chunked must equal
    one-shot exactly: precision γⁿP + (1 - γⁿ)·floor·I, and mean
    solving that against γⁿ·P·m, because per-chunk re-injection
    telescopes."""
    _, y, m, P = _data(n=40)
    n = y.shape[0]
    X = np.zeros((n, P.shape[0]))
    lr, floor = 0.97, 2.0
    one, chunked = (
        update_gaussian_posterior_rvga(
            csc_array(X),
            y,
            m,
            csc_array(P),
            link="logit",
            sparse=True,
            learning_rate=lr,
            prior_floor=floor,
            n_iter=3,
            batch_size=bs,
        )
        for bs in (None, 7)
    )
    g = lr**n
    want_prec = g * P + (1.0 - g) * floor * np.eye(P.shape[0])
    want_mean = np.linalg.solve(want_prec, g * P @ m)
    for post in (one, chunked):
        np.testing.assert_allclose(_dense(post.precision), want_prec, atol=1e-10)
        np.testing.assert_allclose(_dense(post.mean), want_mean, atol=1e-10)
