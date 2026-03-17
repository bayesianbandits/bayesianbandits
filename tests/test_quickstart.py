"""Validate that the code snippets in docs/quickstart.rst keep working."""

from collections import Counter

import numpy as np

from bayesianbandits import (
    Agent,
    Arm,
    NormalInverseGammaRegressor,
    ThompsonSampling,
)


def test_quickstart_setup_and_pull_update():
    """Setup + pull-update loop from the quick-start guide."""
    agent = Agent(
        arms=[
            Arm("ad_a", learner=NormalInverseGammaRegressor()),
            Arm("ad_b", learner=NormalInverseGammaRegressor()),
        ],
        policy=ThompsonSampling(),
    )

    # pull-update loop (one round)
    (choice,) = agent.pull()
    assert choice in ("ad_a", "ad_b")
    agent.update(np.array([1.0]))


def test_quickstart_simulation():
    """Simulation from the quick-start guide: ad_b should win."""
    agent = Agent(
        arms=[
            Arm("ad_a", learner=NormalInverseGammaRegressor()),
            Arm("ad_b", learner=NormalInverseGammaRegressor()),
        ],
        policy=ThompsonSampling(),
    )

    rng = np.random.default_rng(42)
    true_means = {"ad_a": 1.0, "ad_b": 1.3}

    choices = []
    for _ in range(500):
        (choice,) = agent.pull()
        revenue = rng.normal(true_means[choice], scale=0.5)
        agent.update(np.array([revenue]))
        choices.append(choice)

    counts = Counter(choices)
    # ad_b should be pulled substantially more often
    assert counts["ad_b"] > counts["ad_a"], f"Expected ad_b to win, got {counts}"
