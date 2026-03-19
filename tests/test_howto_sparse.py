"""Validate code snippets from docs/howto/sparse.rst."""

import numpy as np
from scipy.sparse import csc_array, issparse
from scipy.sparse import random as sparse_random

from bayesianbandits import (
    Arm,
    ContextualAgent,
    NormalRegressor,
    ThompsonSampling,
)


def _make_sparse_agent():
    arms = [
        Arm(
            f"variant_{i}",
            learner=NormalRegressor(alpha=1.0, beta=1.0, sparse=True),
        )
        for i in range(3)
    ]
    return ContextualAgent(arms, ThompsonSampling(), random_seed=42)


def test_sparse_pull_and_update():
    """Enable sparse mode, pull, and update with csc_array context."""
    agent = _make_sparse_agent()

    # -- mirrors the RST snippet --
    X = sparse_random(1, 10000, density=0.01, format="csc", random_state=42)
    (action,) = agent.pull(X)
    agent.update(X, np.array([1.0]))

    # The updated arm's precision matrix should be sparse
    updated_arm = agent.arm(action)
    assert issparse(updated_arm.learner.cov_inv_)


def test_sparse_precision_is_csc():
    """Precision matrix is stored as csc_array when sparse=True."""
    agent = _make_sparse_agent()

    X = sparse_random(1, 100, density=0.1, format="csc", random_state=42)
    (action,) = agent.pull(X)
    agent.update(X, np.array([1.0]))

    precision = agent.arm(action).learner.cov_inv_
    assert isinstance(precision, csc_array)


def test_sparse_multiple_updates():
    """Multiple sparse updates accumulate correctly."""
    agent = _make_sparse_agent()
    token = "variant_0"

    X = sparse_random(1, 100, density=0.1, format="csc", random_state=42)
    agent.select_for_update(token).update(X, np.array([1.0]))

    precision_after_one = agent.arm(token).learner.cov_inv_.copy()

    agent.select_for_update(token).update(X, np.array([2.0]))

    precision_after_two = agent.arm(token).learner.cov_inv_

    # Second update should increase precision (more data)
    diff = precision_after_two - precision_after_one
    assert diff.nnz > 0


def test_sparse_batch_update():
    """Batch update works with sparse multi-row context."""
    agent = _make_sparse_agent()
    token = "variant_0"

    X_batch = sparse_random(5, 100, density=0.1, format="csc", random_state=42)
    y_batch = np.array([1.0, 2.0, 3.0, 4.0, 5.0])

    agent.select_for_update(token).update(X_batch, y_batch)

    assert issparse(agent.arm(token).learner.cov_inv_)
    assert hasattr(agent.arm(token).learner, "coef_")
