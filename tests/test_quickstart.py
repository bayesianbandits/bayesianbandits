"""Validate that the code snippets in docs/quickstart.rst keep working."""

from collections import Counter

import numpy as np

from bayesianbandits import (
    Agent,
    Arm,
    NormalInverseGammaRegressor,
    ThompsonSampling,
)


def _make_agent():
    return Agent(
        arms=[
            Arm("ad_a", learner=NormalInverseGammaRegressor()),
            Arm("ad_b", learner=NormalInverseGammaRegressor()),
        ],
        policy=ThompsonSampling(),
    )


def test_quickstart_setup_and_pull_update():
    """Setup + pull-update loop from the quick-start guide."""
    agent = _make_agent()

    (choice,) = agent.pull()
    assert choice in ("ad_a", "ad_b")
    agent.update(np.array([0.15]))


def test_quickstart_simulation():
    """Simulation: ad_b has higher profit despite lower revenue."""
    agent = _make_agent()

    rng = np.random.default_rng(42)
    true_revenue = {"ad_a": 0.50, "ad_b": 0.40}
    cost = {"ad_a": 0.35, "ad_b": 0.10}

    choices = []
    for _ in range(500):
        (choice,) = agent.pull()
        profit = rng.normal(true_revenue[choice], scale=0.1) - cost[choice]
        agent.update(np.array([profit]))
        choices.append(choice)

    counts = Counter(choices)
    # ad_b has higher profit ($0.30 vs $0.15), so it should win
    assert counts["ad_b"] > counts["ad_a"], f"Expected ad_b to win, got {counts}"
