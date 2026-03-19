===========
Quick Start
===========

Install with pip (Python 3.10+)::

    pip install -U bayesianbandits

We have two ad creatives for the same product. They earn different revenue per
click, but they also have different costs per impression, so the one with the
highest gross revenue isn't necessarily the most profitable. We want a bandit
that learns to maximize *profit* (revenue minus cost) per impression.

Setup
=====

.. code-block:: python

    import numpy as np
    from bayesianbandits import Agent, Arm, NormalInverseGammaRegressor, ThompsonSampling

    agent = Agent(
        arms=[
            Arm("ad_a", learner=NormalInverseGammaRegressor()),
            Arm("ad_b", learner=NormalInverseGammaRegressor()),
        ],
        policy=ThompsonSampling(),
    )

Each :class:`~bayesianbandits.Arm` wraps a Bayesian model (here, a normal-inverse-gamma regression
that learns the mean and variance of each ad's profit). :class:`~bayesianbandits.ThompsonSampling`
draws from these posteriors and picks the arm with the highest sample. Early on,
the posteriors are wide and the agent explores. As data comes in, they tighten
and the agent exploits.

The pull-update loop
====================

The whole API is two methods: ``pull()`` to pick an arm, and ``update()`` to
teach the agent what happened.

.. code-block:: python

    (choice,) = agent.pull()

    # compute profit for whichever ad we served
    profit = get_revenue(choice) - get_cost(choice)

    agent.update(np.array([profit]))

That's one round. In production you'd run this continuously: on each
impression, pull an arm, serve the ad, observe the profit (potentially much later!), update the model.

A simulation
============

To see this working end to end, we can simulate the problem. Ad A earns more
per click on average ($0.50 vs $0.40), but it costs much more per impression
($0.35 vs $0.10). Ad B is more profitable:

.. code-block:: python

    rng = np.random.default_rng(42)
    true_revenue = {"ad_a": 0.50, "ad_b": 0.40}
    cost = {"ad_a": 0.35, "ad_b": 0.10}

    choices = []
    for _ in range(500):
        (choice,) = agent.pull()
        profit = rng.normal(true_revenue[choice], scale=0.1) - cost[choice]
        agent.update(np.array([profit]))
        choices.append(choice)

.. code-block:: python

    print(f"Ad B (the profitable one) was chosen {choices.count('ad_b') / len(choices):.0%} of the time")
    # Ad B (the profitable one) was chosen 86% of the time

Ad A has higher revenue, but ad B has higher profit ($0.30 vs $0.15). The
agent figures this out and shifts traffic accordingly. It still pulls ad A
occasionally. Thompson sampling never stops exploring entirely, which is what
you want if the underlying rates might change.

Under the hood, the :class:`~bayesianbandits.NormalInverseGammaRegressor` maintains a conjugate
posterior over both the mean and the noise variance of each arm's profit. Each
``update()`` is a rank-1 precision update, not a refit, so the computational
cost per observation is constant regardless of how much data you've seen.

Where to go from here
=====================

**Custom reward functions** (profit = revenue - cost)
    :doc:`howto/reward-functions` shows how to write reward functions that
    turn probability or revenue samples into profit.

**Using context features** (user demographics, item attributes)
    :doc:`Linear Bandits <notebooks/linear-bandits>` covers per-arm contextual
    models with :class:`~bayesianbandits.ContextualAgent`.

**Sharing knowledge across arms** (hundreds of products, cold start)
    :doc:`Hybrid Bandits <notebooks/hybrid-bandits>` demonstrates the
    shared-learner approach with :class:`~bayesianbandits.LipschitzContextualAgent`.

**Pipelines and preprocessing** (DataFrames, JSON, sparse features)
    :doc:`howto/pipelines` covers using sklearn transformers with bandits.
