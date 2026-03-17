===========
Quick Start
===========

We have two ad creatives for the same product. They earn different revenue per
click, but they also have different costs per impression — so the one with the
highest gross revenue isn't necessarily the most profitable. We want a bandit
that learns to maximize *profit* (revenue minus cost) rather than raw revenue.

Setup
=====

Each arm's learner models raw revenue. The ``reward_function`` transforms those
samples into profit before the policy compares them — the bandit explores in
revenue-space but decides in profit-space.

.. code-block:: python

    import numpy as np
    from bayesianbandits import Agent, Arm, NormalInverseGammaRegressor, ThompsonSampling

    costs = {"ad_a": 0.35, "ad_b": 0.10}

    agent = Agent(
        arms=[
            Arm(
                "ad_a",
                learner=NormalInverseGammaRegressor(),
                reward_function=lambda revenue: revenue - costs["ad_a"],
            ),
            Arm(
                "ad_b",
                learner=NormalInverseGammaRegressor(),
                reward_function=lambda revenue: revenue - costs["ad_b"],
            ),
        ],
        policy=ThompsonSampling(),
    )

Each ``Arm`` wraps a Bayesian model — here, a normal-inverse-gamma regression
that learns the mean and variance of each ad's revenue. ``ThompsonSampling``
draws from these posteriors, applies each arm's reward function, and picks the
arm with the highest profit sample. Early on, the posteriors are wide and the
agent explores. As data comes in, they tighten and the agent exploits.

The pull-update loop
====================

The whole API is two methods: ``pull()`` to pick an arm, and ``update()`` to
teach the agent what happened. You always update with the raw observation
(revenue) — the reward function is only used internally for decision-making.

.. code-block:: python

    (choice,) = agent.pull()

    # observe revenue for whichever ad we served
    revenue = get_revenue(choice)

    agent.update(np.array([revenue]))

That's one round. In production you'd run this continuously — on each
impression, pull an arm, serve the ad, observe the revenue, update the model.

A simulation
============

To see this working end to end, we can simulate the problem. Ad A earns more
per click on average ($0.50 vs $0.40), but it costs much more per impression
($0.35 vs $0.10). Ad B is more profitable:

.. code-block:: python

    rng = np.random.default_rng(42)
    true_revenue = {"ad_a": 0.50, "ad_b": 0.40}

    choices = []
    for _ in range(500):
        (choice,) = agent.pull()
        revenue = rng.normal(true_revenue[choice], scale=0.1)
        agent.update(np.array([revenue]))
        choices.append(choice)

.. code-block:: python

    from collections import Counter
    Counter(choices)
    # Counter({'ad_b': 428, 'ad_a': 72})

Ad A has higher revenue, but ad B has higher profit ($0.30 vs $0.15). The
agent figures this out and shifts traffic accordingly. It still pulls ad A
occasionally — Thompson sampling never stops exploring entirely, which is what
you want if the underlying rates might change.

Under the hood, the ``NormalInverseGammaRegressor`` maintains a conjugate
posterior over both the mean and the noise variance of each arm's revenue. Each
``update()`` is a rank-1 precision update, not a refit — so the computational
cost per observation is constant regardless of how much data you've seen.

Where to go from here
=====================

**Binary outcomes** (click / no click)
    The :doc:`Thompson Sampling tutorial <notebooks/demo>` uses
    ``DirichletClassifier`` with a custom reward function.

**Count data** (transactions per week)
    :doc:`Contextual Bandits for Count Data <notebooks/counts>` uses
    ``GammaRegressor`` with Upper Confidence Bound.

**Using context features** (user demographics, item attributes)
    :doc:`Linear Bandits <notebooks/linear-bandits>` covers per-arm contextual
    models with ``ContextualAgent``.

**Sharing knowledge across arms** (hundreds of products, cold start)
    :doc:`Hybrid Bandits <notebooks/hybrid-bandits>` demonstrates the
    shared-learner approach with ``LipschitzContextualAgent``.
