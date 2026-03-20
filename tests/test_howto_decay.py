"""Validate code snippets from docs/howto/decay.rst."""

import numpy as np

from bayesianbandits import (
    Arm,
    ContextualAgent,
    EmpiricalBayesNormalRegressor,
    NormalRegressor,
    ThompsonSampling,
)


def test_decoupled_decay():
    """Decouple decay from updates via explicit decay() calls."""
    arms = [
        Arm(f"variant_{i}", learner=NormalRegressor(alpha=1.0, beta=1.0))
        for i in range(3)
    ]
    agent = ContextualAgent(arms, ThompsonSampling(), random_seed=42)

    X = np.array([[1.0, 2.0]])
    (action,) = agent.pull(X)
    agent.update(X, y=np.array([1.0]))

    # Capture precision before decay
    updated_arm = agent.arm(action)
    precision_before = updated_arm.learner.cov_inv_.copy()

    # Decay all arms
    agent.decay(np.array([[0.0, 0.0]]), decay_rate=0.95)

    # Precision should have shrunk
    precision_after = updated_arm.learner.cov_inv_
    assert np.all(np.diag(precision_after) < np.diag(precision_before))


def test_effective_window_size():
    """Verify the effective window size approximation."""
    # After n decay steps, weight is gamma^n
    # Weight drops below 1/e when n ~ 1/(1-gamma)
    gamma = 0.99
    n_eff = 1 / (1 - gamma)  # ~100
    assert abs(n_eff - 100) < 1

    weight_at_n_eff = gamma**n_eff
    # Should be close to 1/e ~ 0.368
    assert abs(weight_at_n_eff - 1 / np.e) < 0.01


def test_no_decay_by_default():
    """learning_rate=1.0 means no decay on partial_fit."""
    learner = NormalRegressor(alpha=1.0, beta=1.0)
    X = np.array([[1.0]])
    y = np.array([1.0])
    learner.fit(X, y)

    precision_after_fit = learner.cov_inv_.copy()

    # partial_fit with learning_rate=1.0 should only add information
    learner.partial_fit(X, y)
    precision_after_update = learner.cov_inv_

    # Precision should have increased (more data), not decreased
    assert np.all(np.diag(precision_after_update) >= np.diag(precision_after_fit))


def test_coupled_decay_depends_on_batch_size():
    """Show that coupled decay (learning_rate<1) varies with batch size."""
    # Two learners with same learning_rate but different batch sizes
    lr = 0.99

    learner_one = NormalRegressor(alpha=1.0, beta=1.0, learning_rate=lr)
    learner_ten = NormalRegressor(alpha=1.0, beta=1.0, learning_rate=lr)

    # Fit both on same initial data
    X_init = np.array([[1.0]])
    y_init = np.array([1.0])
    learner_one.fit(X_init, y_init)
    learner_ten.fit(X_init, y_init)

    # Update one with 1 observation (decay = 0.99^1)
    learner_one.partial_fit(np.array([[2.0]]), np.array([2.0]))

    # Update other with 10 observations (decay = 0.99^10)
    X_ten = np.full((10, 1), 2.0)
    y_ten = np.full(10, 2.0)
    learner_ten.partial_fit(X_ten, y_ten)

    # The prior was decayed more in the 10-observation case
    # (0.99^10 ~ 0.904 vs 0.99^1 = 0.99)
    # So the precision matrices differ despite same learning_rate
    assert not np.allclose(learner_one.cov_inv_, learner_ten.cov_inv_)


def test_eb_stabilized_forgetting():
    """EmpiricalBayesNormalRegressor re-injects prior after decay."""
    learner = EmpiricalBayesNormalRegressor(
        alpha=1.0,
        beta=1.0,
        learning_rate=1.0,
    )

    X = np.array([[1.0, 2.0]])
    y = np.array([1.0])
    learner.fit(X, y)

    # Decay aggressively many times
    for _ in range(100):
        learner.decay(X, decay_rate=0.5)

    # With stabilized forgetting, the prior contribution converges to alpha
    # instead of zero. The diagonal should stay bounded above zero.
    diag = np.diag(learner.cov_inv_)
    assert np.all(diag > 0.5), f"Prior collapsed: diag = {diag}"
