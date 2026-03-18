"""Validate code snippets from docs/howto/delayed-rewards.rst."""

import numpy as np

from bayesianbandits import (
    Agent,
    Arm,
    ContextualAgent,
    NormalRegressor,
    ThompsonSampling,
)


def _make_contextual_agent():
    arms = [
        Arm(f"variant_{i}", learner=NormalRegressor(alpha=1.0, beta=1.0))
        for i in range(3)
    ]
    return ContextualAgent(arms, ThompsonSampling(), random_seed=42)


def test_pull_now_update_later():
    """Section 1: the basic pattern from the RST."""
    agent = _make_contextual_agent()

    # -- mirrors the RST snippet --
    X_batch = np.array([[1.0, 2.0]])
    actions = agent.pull(X_batch)

    for token, X_row, reward in zip(
        actions,
        X_batch,
        [1.0],
    ):
        agent.select_for_update(token).update(
            np.atleast_2d(X_row), np.atleast_1d(reward)
        )

    # The arm that was selected should now be fitted
    updated_arm = agent.arm(actions[0])
    assert hasattr(updated_arm.learner, "coef_")


def test_batch_update_same_as_sequential():
    """Section 2: batch update produces same posterior as sequential."""
    X_all = np.array([[1.0, 2.0], [3.0, 4.0], [5.0, 6.0]])
    y_all = np.array([1.0, 2.0, 3.0])

    # Sequential updates
    agent_seq = _make_contextual_agent()
    token = "variant_0"
    for i in range(len(y_all)):
        agent_seq.select_for_update(token).update(X_all[i : i + 1], y_all[i : i + 1])

    # Batch update -- mirrors the RST snippet
    agent_batch = _make_contextual_agent()
    rewards_grouped_by_arm = {
        token: {"contexts": [X_all[i : i + 1] for i in range(len(y_all))], "rewards": y_all},
    }
    for tk, group in rewards_grouped_by_arm.items():
        X_batch = np.vstack(group["contexts"])
        y_batch = np.array(group["rewards"])
        agent_batch.select_for_update(tk).update(X_batch, y_batch)

    seq_learner = agent_seq.arm(token).learner
    batch_learner = agent_batch.arm(token).learner

    np.testing.assert_allclose(seq_learner.coef_, batch_learner.coef_)
    np.testing.assert_allclose(seq_learner.cov_inv_, batch_learner.cov_inv_)


def test_multiple_pulls_before_update():
    """Section 3: pull twice, update once via select_for_update."""
    agent = _make_contextual_agent()

    X = np.array([[1.0, 2.0]])
    (token_a,) = agent.pull(X)

    # Capture precision of token_a after pull (prior only)
    precision_before = agent.arm(token_a).learner.cov_inv_.copy()

    (_,) = agent.pull(X)  # second pull overwrites arm_to_update

    # Update only token_a via select_for_update
    agent.select_for_update(token_a).update(X, np.array([1.0]))

    # token_a precision should have increased (data was incorporated)
    precision_after = agent.arm(token_a).learner.cov_inv_
    assert np.all(np.diag(precision_after) >= np.diag(precision_before))


def test_select_for_update_returns_self():
    """API: chaining works via select_for_update returning self."""
    agent = _make_contextual_agent()

    X = np.array([[1.0, 2.0]])
    (token,) = agent.pull(X)

    result = agent.select_for_update(token)
    assert result is agent


def test_decay_then_update():
    """Section 4: decay widens posterior, then update tightens it."""
    agent = _make_contextual_agent()
    token = "variant_0"

    # Initial fit
    X = np.array([[1.0, 2.0]])
    agent.select_for_update(token).update(X, np.array([1.0]))
    precision_after_fit = agent.arm(token).learner.cov_inv_.copy()

    # Decay widens posterior (lower precision) -- RST uses 0.99
    agent.decay(np.array([[0.0, 0.0]]), decay_rate=0.99)
    precision_after_decay = agent.arm(token).learner.cov_inv_.copy()
    assert np.all(np.diag(precision_after_decay) < np.diag(precision_after_fit))

    # Update with yesterday's rewards tightens it back
    agent.select_for_update(token).update(X, np.array([2.0]))
    precision_after_update = agent.arm(token).learner.cov_inv_
    assert np.all(np.diag(precision_after_update) > np.diag(precision_after_decay))


def test_noncontextual_delayed():
    """Section 5: non-contextual Agent with delayed update pattern."""
    # -- mirrors the RST snippet --
    arms = [
        Arm(f"slot_{i}", learner=NormalRegressor(alpha=1.0, beta=1.0)) for i in range(3)
    ]
    agent = Agent(arms, ThompsonSampling(), random_seed=42)

    (token,) = agent.pull()
    # later...
    agent.select_for_update(token).update(np.array([1.0]))

    updated_arm = agent.arm(token)
    assert hasattr(updated_arm.learner, "coef_")


def test_wrong_arm_without_select():
    """Troubleshooting: pull overwrites arm_to_update."""
    agent = _make_contextual_agent()

    X = np.array([[1.0, 2.0]])
    (token_a,) = agent.pull(X)
    (token_b,) = agent.pull(X)

    # Without select_for_update, update goes to token_b (last pulled)
    agent.update(X, np.array([1.0]))
    assert hasattr(agent.arm(token_b).learner, "coef_")

    # With select_for_update, we can target token_a explicitly
    agent2 = _make_contextual_agent()
    agent2.pull(X)
    agent2.pull(X)
    agent2.select_for_update(token_a).update(X, np.array([1.0]))
    assert hasattr(agent2.arm(token_a).learner, "coef_")
