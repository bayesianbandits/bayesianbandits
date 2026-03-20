Writing Custom Reward Functions
===============================

A reward function separates what the learner models (outcomes) from
what those outcomes are worth (utility). If your learner already
models the quantity you want to maximize (revenue via
``NormalRegressor``, profit via ``NormalInverseGammaRegressor``), the
default identity function is fine and you don't need a custom reward
function. See :doc:`/notebooks/linear-bandits` for how far you can
get without one.

.. note::

   ``update()`` always trains on raw outcomes, not transformed
   rewards. The learner models what actually happens; the reward
   function captures what it's worth. You can change your cost
   structure or utility function without retraining.


Map binary outcomes to profit
-----------------------------

A ``BayesianGLM`` with ``link="logit"`` models click-through or
conversion probability. But different arms may have different revenue per
conversion and different costs per impression. The learner trains on
raw binary outcomes (0/1); the reward function turns probability
samples into expected profit.

.. code-block:: python

   import numpy as np
   from bayesianbandits import Arm, ContextualAgent, BayesianGLM, ThompsonSampling

   def make_profit_reward(revenue, cost):
       """Expected profit = P(convert) * revenue - cost."""
       def reward(samples):
           return samples * revenue - cost
       return reward

   arms = [
       Arm(
           "campaign_A",
           reward_function=make_profit_reward(revenue=50.0, cost=10.0),
           learner=BayesianGLM(alpha=1.0, link="logit"),
       ),
       Arm(
           "campaign_B",
           reward_function=make_profit_reward(revenue=20.0, cost=2.0),
           learner=BayesianGLM(alpha=1.0, link="logit"),
       ),
       Arm(
           "campaign_C",
           reward_function=make_profit_reward(revenue=100.0, cost=30.0),
           learner=BayesianGLM(alpha=1.0, link="logit"),
       ),
   ]
   agent = ContextualAgent(arms, ThompsonSampling(), random_seed=42)

   X = np.array([[1.0, 0.0, 25.0]])  # user features
   (action,) = agent.pull(X)

   # Update gets the raw binary outcome, NOT the profit
   agent.update(X, np.array([1]))  # conversion happened

``BayesianGLM`` with the logit link already returns probability
samples in [0, 1], so ``samples * revenue - cost`` is all you need.


Use context in the reward function
-----------------------------------

When utility depends on context (user-specific shipping cost,
regional tax), the reward function can accept a second parameter
named ``X`` to receive the context matrix. The parameter **must** be
named ``X``: the library inspects the function signature to detect
context-awareness. See :ref:`troubleshooting <reward-troubleshooting>`
if this isn't working.

.. code-block:: python

   from bayesianbandits import (
       Arm, ContextualAgent, NormalRegressor, ThompsonSampling,
   )

   def net_profit_reward(samples, X):
       """Subtract per-row cost (last column of X) from gross revenue."""
       cost = X[:, -1]
       return samples - cost

   arms = [
       Arm(
           f"product_{i}",
           reward_function=net_profit_reward,
           learner=NormalRegressor(alpha=1.0, beta=1.0),
       )
       for i in range(3)
   ]
   agent = ContextualAgent(arms, ThompsonSampling(), random_seed=42)

   X = np.array([[5.0, 1.2]])  # features + cost
   (action,) = agent.pull(X)
   agent.update(X, y=np.array([6.5]))  # raw gross revenue


Express non-linear utility
--------------------------

Reward functions don't have to be linear. Here's an asymmetric loss
that penalizes underperformance more than overperformance:

.. code-block:: python

   def asymmetric_loss(samples, target=10.0, penalty=3.0):
       """Penalize undershoot 3x relative to overshoot."""
       diff = samples - target
       return np.where(diff < 0, penalty * diff, diff)

Other examples: threshold utility (``np.maximum(samples - threshold, 0)``),
diminishing returns (``np.log1p(samples)``).


Batch reward functions for shared-learner bandits
-------------------------------------------------

When the learner models one thing (revenue) but you need to transform
it per-arm (multiply by a margin that differs across products), use a
batch reward function.

With :class:`~bayesianbandits.LipschitzContextualAgent` and many
arms, processing rewards one arm at a time is inefficient. A batch
reward function handles all arms in a single vectorized call.

Two signatures:

- ``f(samples, action_tokens) -> rewards``
- ``f(samples, action_tokens, X) -> rewards`` (context-aware; third
  param must be named ``X``)

Input ``samples`` shape is ``(n_arms, n_contexts, size)``. Output
must match.

.. code-block:: python

   import pandas as pd
   from sklearn.compose import ColumnTransformer
   from sklearn.preprocessing import OneHotEncoder, StandardScaler
   from bayesianbandits import (
       Arm, LipschitzContextualAgent, NormalRegressor,
       ThompsonSampling, ArmColumnFeaturizer, LearnerPipeline,
   )

   MARGINS = {
       "product_0": 0.3,
       "product_1": 0.5,
       "product_2": 0.8,
   }

   def margin_reward(samples, action_tokens):
       """Multiply revenue samples by per-product margin."""
       multipliers = np.array([MARGINS[t] for t in action_tokens])
       return samples * multipliers[:, np.newaxis, np.newaxis]

   arm_tokens = list(MARGINS.keys())

   sample_enriched = pd.DataFrame({
       "score": [0.8, 0.6, 0.9] * 3,
       "arm": arm_tokens * 3,
   })
   ct = ColumnTransformer([
       ("num", StandardScaler(), ["score"]),
       ("arm", OneHotEncoder(sparse_output=False), ["arm"]),
   ])
   ct.fit(sample_enriched)

   shared_learner = LearnerPipeline(
       steps=[("preprocess", ct)],
       learner=NormalRegressor(alpha=1.0, beta=1.0),
   )
   arms = [Arm(token) for token in arm_tokens]
   agent = LipschitzContextualAgent(
       arms=arms,
       policy=ThompsonSampling(),
       arm_featurizer=ArmColumnFeaturizer("arm"),
       learner=shared_learner,
       batch_reward_function=margin_reward,
       random_seed=42,
   )

   context = pd.DataFrame({"score": [0.7]})
   (action,) = agent.pull(context)


.. _reward-troubleshooting:

If something goes wrong
-----------------------

**Shape mismatch error from the policy.**
Your reward function must return shape ``(size, n_contexts)``. For
most estimators (``NormalRegressor``, ``BayesianGLM``,
``NormalInverseGammaRegressor``, ``GammaRegressor``) the input is
already that shape, so element-wise operations just work.
``DirichletClassifier`` is the exception: its samples are
``(size, n_contexts, n_classes)`` and the reward function must
collapse the class axis (e.g. ``samples[..., 1] * revenue - cost``).
For batch reward functions, the contract is ``(n_arms, n_contexts,
size)`` in and out.

**TypeError: reward() missing 1 required positional argument.**
You wrote a context-aware reward function but named the parameter
something other than ``X``. The library checks
``inspect.signature()`` for a parameter literally named ``X``. If it
doesn't find one, it calls your function with ``samples`` only, and
your function raises because it expected two arguments. Rename the
parameter to ``X``.

**PicklingError when serializing the agent.**
Lambdas, closures, and factory-produced functions (like
``make_profit_reward`` above) are not picklable with standard
``pickle``. If you need to serialize agents, use a callable class
with ``__call__`` instead, or use a serialization library that
handles closures (e.g. ``cloudpickle``).

**Reward function has side effects or mutable state.**
Thompson sampling calls the reward function on every ``pull()``.
A reward function that mutates external state (counters, accumulators,
caches with size limits) will behave unpredictably. Keep reward
functions stateless.


.. seealso::

   :class:`~bayesianbandits.Arm` for full reward function type
   signatures.
