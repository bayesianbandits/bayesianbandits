"""Validate code snippets from docs/howto/production.rst."""

import tempfile
from pathlib import Path

import joblib
import numpy as np

from bayesianbandits import (
    Agent,
    Arm,
    GammaRegressor,
    ThompsonSampling,
)


def _make_agent():
    arms = [
        Arm("ad_a", learner=GammaRegressor(alpha=1, beta=1)),
        Arm("ad_b", learner=GammaRegressor(alpha=1, beta=1)),
    ]
    return Agent(arms, ThompsonSampling(), random_seed=42)


def test_joblib_roundtrip():
    """Serialize with joblib, reload, verify learned state preserved."""
    agent = _make_agent()

    (choice,) = agent.pull()
    agent.update(np.array([1.0]))

    with tempfile.TemporaryDirectory() as tmpdir:
        path = Path(tmpdir) / "agent.pkl"
        joblib.dump(agent, path, compress=True)
        loaded = joblib.load(path)

    # Learned state is preserved
    assert loaded.arms[0].learner.coef_[1][0] == agent.arms[0].learner.coef_[1][0]
    assert loaded.arms[1].learner.coef_[1][0] == agent.arms[1].learner.coef_[1][0]


def test_reseed_after_load():
    """Reseed RNG after loading to avoid deterministic replay."""
    agent = _make_agent()
    agent.pull()
    agent.update(np.array([1.0]))

    with tempfile.TemporaryDirectory() as tmpdir:
        path = Path(tmpdir) / "agent.pkl"
        joblib.dump(agent, path)

        # Two loads from same file share frozen RNG state
        loaded_a = joblib.load(path)
        loaded_b = joblib.load(path)
        pulls_a = [loaded_a.pull()[0] for _ in range(20)]
        pulls_b = [loaded_b.pull()[0] for _ in range(20)]
        assert pulls_a == pulls_b

        # After reseeding, the Generator is fresh
        loaded_c = joblib.load(path)
        old_rng = loaded_c.rng
        loaded_c.rng = 999
        assert loaded_c.rng is not old_rng


def test_reseed_propagates_to_learners():
    """Setting rng propagates to all arm learners."""
    agent = _make_agent()
    agent.rng = 123

    # All arm learners should share the agent's Generator
    for arm in agent.arms:
        assert arm.learner.random_state is agent.rng


def test_add_arm():
    """Add a new arm at runtime."""
    agent = _make_agent()

    # Train existing arms
    (choice,) = agent.pull()
    agent.update(np.array([1.0]))

    # Add new arm
    agent.add_arm(Arm("ad_c", learner=GammaRegressor(alpha=1, beta=1)))

    assert len(agent.arms) == 3
    tokens = [arm.action_token for arm in agent.arms]
    assert "ad_c" in tokens

    # New arm should work in pull
    (choice,) = agent.pull()
    assert choice in tokens


def test_remove_arm():
    """Remove an arm at runtime."""
    agent = _make_agent()
    agent.remove_arm("ad_a")

    assert len(agent.arms) == 1
    assert agent.arms[0].action_token == "ad_b"


def test_add_arm_duplicate_raises():
    """Adding an arm with a duplicate token raises ValueError."""
    agent = _make_agent()
    try:
        agent.add_arm(Arm("ad_a", learner=GammaRegressor(alpha=1, beta=1)))
        assert False, "Should have raised ValueError"
    except ValueError:
        pass


def test_remove_arm_missing_raises():
    """Removing a nonexistent arm raises KeyError."""
    agent = _make_agent()
    try:
        agent.remove_arm("nonexistent")
        assert False, "Should have raised KeyError"
    except KeyError:
        pass


def test_independent_copies_via_load():
    """Each joblib.load produces an independent copy (thread safety pattern)."""
    agent = _make_agent()
    agent.pull()
    agent.update(np.array([1.0]))

    with tempfile.TemporaryDirectory() as tmpdir:
        path = Path(tmpdir) / "agent.pkl"
        joblib.dump(agent, path)

        worker_a = joblib.load(path)
        worker_b = joblib.load(path)

        worker_a.rng = 111
        worker_b.rng = 222

        # Mutating one copy doesn't affect the other
        worker_a.pull()
        worker_a.update(np.array([5.0]))

        # worker_b still has original learned state
        assert (
            worker_b.arms[0].learner.coef_[1][0]
            == agent.arms[0].learner.coef_[1][0]
        )
