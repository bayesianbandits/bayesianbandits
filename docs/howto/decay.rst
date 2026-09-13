Choosing and Tuning a Decay Rate
================================

Decay lets a bandit adapt to non-stationary environments. There are
two ways to apply it:

.. note::

   Decay scales the precision matrix, which increases posterior
   variance without moving the posterior mean. Your point estimate
   stays the same; you just become less confident in it. Wider
   posteriors drive re-exploration via more diverse Thompson samples
   and higher UCB values.

``forgetting=`` on the estimator
    Forgetting is coupled to ``partial_fit``: every update forgets
    before incorporating the new data, one step per row under a
    uniform rule such as ``ExponentialForgetting(0.99)``. This is for
    change that happens *because you observed*.

Explicit ``agent.decay()`` calls
    Forgetting is decoupled from updates. You call ``decay()`` on your
    own schedule, independently of when observations arrive. This is
    for change that happens *with time*.


Start with no decay
--------------------

If you are unsure whether your environment is non-stationary, start
with no ``forgetting`` rule (the default) and no ``decay()`` calls.
Adding decay when you don't need it throws away information and
widens your posterior for no benefit.


Decouple decay from updates
-----------------------------

Consider a product recommendation system. You ``pull()`` thousands of
times per day as users visit the site, and ``update()`` as purchases
arrive. But user preferences don't shift on a per-request basis --
they shift over weeks or months. If you set ``forgetting=`` on the
estimator, the amount of forgetting depends on how many observations
land in each update batch, not how fast tastes actually change. Leave
it unset and call ``decay()`` on a schedule that matches the
timescale of change in your environment:

.. code-block:: python

   from bayesianbandits import (
       Arm, ContextualAgent, NormalRegressor, ThompsonSampling,
   )
   import numpy as np

   arms = [
       Arm(f"product_{i}", learner=NormalRegressor(alpha=1.0, beta=1.0))
       for i in range(3)
   ]
   agent = ContextualAgent(arms, ThompsonSampling(), random_seed=42)

   # Throughout the day: pull and update as users visit
   X = np.array([[1.0, 2.0]])  # user features
   (action,) = agent.pull(X)
   agent.update(X, y=np.array([1.0]))  # purchase signal

   # Once per day (e.g. nightly cron): decay all arms
   agent.decay(decay_rate=0.95)

One call is one tick. If the job missed a few days, pass
``steps=3`` and the rate is raised to that power. Per-observation
forgetting through ``forgetting=`` is usually too aggressive for the
same reason: most real systems make many decisions per natural time
period (thousands of recommendations per day), and forgetting once
per observation in that setting tracks traffic, not time.

``decay_rate=0.95`` is shorthand for passing the learner's default
rule, :class:`~bayesianbandits.ExponentialForgetting`, at that rate.
The rule can be passed explicitly, and the rule carries its rate:

.. code-block:: python

   from bayesianbandits import StabilizedForgetting

   agent.decay(StabilizedForgetting(0.95), steps=1)


Choose a decay rate
--------------------

The decay rate ``gamma`` controls how many effective observations the
model remembers. After ``n`` decay steps, an observation's weight is
``gamma^n``.  A rough rule of thumb: the effective window size is
approximately ``1 / (1 - gamma)`` observations before the weight
drops below ``1/e``:

====== ================
gamma  Effective window
====== ================
0.999  ~1000
0.99   ~100
0.95   ~20
0.9    ~10
====== ================

Start conservative (closer to 1.0). You can always decay more
aggressively later.


Avoid over-decay
-----------------

Aggressive decay can cause problems:

- **Near-singular precision matrix**: with ``NormalRegressor``, the
  precision matrix ``Lambda`` is scaled by ``gamma^n`` on each decay
  step. If ``gamma`` is too small or decay is called too frequently,
  ``Lambda`` approaches zero and the Cholesky factorization fails.

- **Prior washed out**: the prior contribution ``alpha * I``  decays
  along with the data. After enough decay steps, the model is
  effectively unregularized.

:class:`~bayesianbandits.StabilizedForgetting` fixes both. After
scaling by ``gamma``, it adds ``(1 - gamma) * alpha`` back onto the
precision diagonal, so the prior contribution converges to ``alpha``
instead of zero and no direction can wind up past the prior. Any
estimator accepts it in ``decay``:

.. code-block:: python

   from bayesianbandits import StabilizedForgetting

   agent.decay(StabilizedForgetting(0.95))

The empirical Bayes estimators
(:class:`~bayesianbandits.EmpiricalBayesNormalRegressor`,
:class:`~bayesianbandits.EmpiricalBayesGLM`) use it by default, with
the floor at their tuned ``alpha``, so for them ``decay_rate=0.95`` is
already stabilized. For the other estimators the default is plain
exponential forgetting, for compatibility; pass the rule to opt in.

See the :doc:`delayed reward example </notebooks/delayed-reward>` for
a full simulation that tunes the decay rate with optuna and shows how
too much decay hurts.
