=====================================
Separating Inference from Decisions
=====================================

:doc:`worldview` argues that the forecast and the policy are different kinds of
objects. This page explains the full separation and why the library enforces it
architecturally.


Three questions, not one
========================

A bandit answers three questions at every step:

1. **What do I believe about the world?** (the posterior)
2. **What do I value?** (the utility function)
3. **How do I decide?** (the decision rule)

These are independent. Changing what you value (costs change, success metrics
change) doesn't change what you believe. Changing how you decide (Thompson
sampling to UCB) doesn't change either of the other two. Savage's axioms [1]_
derive this decomposition: rational decision-making separates into beliefs
(probability) and preferences (utility), determined independently.

Bayesian decision theory operationalizes this: compute the posterior (inference),
apply the loss function (utility), select an action (decision rule) [2]_
[3]_. The Bayes action minimizes posterior expected loss, but in practice you
use a tractable decision rule like Thompson sampling or UCB instead. bayesianbandits makes the same decomposition literal in its API: the
learner holds the posterior, the reward function encodes utility, and the policy
is the decision rule.


The learner models what happens
===============================

The learner maintains a posterior over the generative process: click-through
rate, revenue per impression, conversion probability. Whatever the world
actually produces when you take an action.

``update()`` trains on raw outcomes. The learner never sees the reward function.
This means the posterior stays valid if your utility changes. Ad costs go up,
you redefine profit, you add a penalty for inventory risk: none of these
require retraining. The posterior is about *what happens*, not about *what you
want to happen*.

This is the standard Bayesian decision-theoretic setup [2]_: the posterior is
computed first, and it remains valid under any loss function you might later
apply to it.


The reward function is a utility function
==========================================

The library calls it "reward function" following bandit convention, but it's
the utility function from decision theory. It maps posterior samples to *value*:
what an outcome is worth to you.

``profit = revenue - cost`` is a reward function. So is
``profit = revenue - cost - 0.1 * inventory_risk``. The reward function is
applied at decision time (during ``pull()``), not at training time (during
``update()``). The learner's posterior samples are raw predictions about
observables; the reward function transforms them into decision-relevant
quantities.

Raiffa and Schlaifer [3]_ work through this separation specifically for
conjugate models: maintain a conjugate posterior, apply a loss function to the
predictive distribution, choose the Bayes action. That's exactly the flow in
bayesianbandits. See :doc:`/howto/reward-functions` for how to write one.


The policy is a decision rule
==============================

Given posterior samples transformed by the reward function, the policy picks an
arm. Thompson sampling draws one sample per arm and picks the highest. UCB
picks the arm with the highest upper quantile. Epsilon-greedy picks the highest
mean and occasionally randomizes.


Orthogonality in practice
==========================

The three axes are independent in the code. Each can be swapped without
touching the others:

**Change what you value.** Costs change, so you update the reward function.
The learner's posterior over raw outcomes is unaffected. No retraining.

**Change how you decide.** Swap Thompson sampling for UCB to get deterministic
behavior during a holdout test. The learner and reward function are unchanged.

**Change what you believe.** Switch from ``NormalRegressor`` to
``EmpiricalBayesNormalRegressor`` for automatic hyperparameter tuning. The
reward function and policy are unchanged.

Each is a one-line change. If your system tangles inference with utility (e.g., training
on transformed rewards), changing the utility function means retraining from
scratch. If it tangles inference with the decision rule, you can't swap
policies without rebuilding the model. Keeping them separate means each
component has its own lifecycle. See :doc:`/howto/pipelines` for how the
learner integrates with sklearn transformers, and :doc:`/howto/decay` for
tuning the learner's memory independently of the other two axes.


.. rubric:: References

.. [1] Savage, L. J. (1954). *The Foundations of Statistics.* Wiley.

.. [2] Berger, J. O. (1985). *Statistical Decision Theory and Bayesian
   Analysis* (2nd ed.). Springer-Verlag.

.. [3] Raiffa, H. and Schlaifer, R. (1961). *Applied Statistical Decision
   Theory.* Wiley.
