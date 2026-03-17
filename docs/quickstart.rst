===========
Quick Start
===========

We have two ad creatives for the same product and we want to figure out which
one earns more revenue per impression. We could run a fixed A/B test, but a
bandit will shift traffic toward the winner as it learns — we don't have to
wait for the test to "finish" before we start benefiting from what we've
learned.

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

Each ``Arm`` wraps a Bayesian model — here, a normal-inverse-gamma regression
that learns the mean and variance of each ad's revenue. ``ThompsonSampling``
draws from these posteriors and picks the arm with the highest sample. Early on,
the posteriors are wide and the agent explores. As data comes in, they tighten
and the agent exploits.

The pull-update loop
====================

The whole API is two methods: ``pull()`` to pick an arm, and ``update()`` to
teach the agent what happened.

.. code-block:: python

    (choice,) = agent.pull()

    # observe revenue for whichever ad we served
    revenue = get_revenue(choice)

    agent.update(np.array([revenue]))

That's one round. In production you'd run this continuously — on each
impression, pull an arm, serve the ad, observe the revenue, update the model.

A simulation
============

To see this working end to end, we can simulate the problem. Say ``ad_b``
actually earns 30% more on average:

.. code-block:: python

    rng = np.random.default_rng(42)
    true_means = {"ad_a": 1.0, "ad_b": 1.3}

    choices = []
    for _ in range(500):
        (choice,) = agent.pull()
        revenue = rng.normal(true_means[choice], scale=0.5)
        agent.update(np.array([revenue]))
        choices.append(choice)

.. code-block:: python

    from collections import Counter
    Counter(choices)
    # Counter({'ad_b': 422, 'ad_a': 78})

The agent figures out pretty quickly that ``ad_b`` earns more and shifts traffic
accordingly. It still pulls ``ad_a`` occasionally — Thompson sampling never
stops exploring entirely, which is what you want if the underlying rates might
change.

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
