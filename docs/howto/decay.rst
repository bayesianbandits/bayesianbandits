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

``learning_rate < 1`` on the estimator
    Decay is coupled to ``partial_fit``: every update automatically
    down-weights the prior by ``learning_rate ** n_samples`` before
    incorporating new data.

Explicit ``agent.decay()`` calls
    Decay is decoupled from updates. You call ``decay()`` on your own
    schedule, independently of when observations arrive.


Start with no decay
--------------------

If you are unsure whether your environment is non-stationary, start
with ``learning_rate=1.0`` (the default) and no ``decay()`` calls.
Adding decay when you don't need it throws away information and
widens your posterior for no benefit.


Decouple decay from updates
-----------------------------

Consider a product recommendation system. You ``pull()`` thousands of
times per day as users visit the site, and ``update()`` as purchases
arrive. But user preferences don't shift on a per-request basis --
they shift over weeks or months. If you set ``learning_rate < 1``,
the amount of forgetting depends on how many observations land in
each update batch, not how fast tastes actually change. Keep
``learning_rate=1.0`` and call ``decay()`` on a schedule that matches
the timescale of change in your environment:

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
   agent.decay(np.array([[0.0, 0.0]]), decay_rate=0.95)

Pass a 1-row array -- ``decay()`` uses ``X.shape[0]`` as the
exponent, so a 100-row array would apply ``0.95^100`` instead of
``0.95`` [1]_.

.. [1] ``decay()`` raises ``gamma`` to the power of ``X.shape[0]``,
   so a 1-row array gives one decay step (``gamma^1``). This
   exponent exists so that ``partial_fit`` on a batch of 10
   observations gives the same posterior as fitting them one at a
   time with decay between each. In practice, per-observation decay
   via ``learning_rate < 1`` is often too aggressive: most real
   systems make many decisions per natural time period (thousands of
   recommendations per day), and decaying once per observation in
   that setting forgets too fast.


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

:class:`~bayesianbandits.EmpiricalBayesNormalRegressor` mitigates the
second problem with stabilized forgetting: after each decay step, it
re-injects ``(1 - gamma^n) * alpha`` onto the precision diagonal so
the prior contribution converges to ``alpha`` instead of zero. If you
need decay and want a safety net against prior collapse, use EB:

.. code-block:: python

   from bayesianbandits import EmpiricalBayesNormalRegressor

   learner = EmpiricalBayesNormalRegressor(
       alpha=1.0,
       beta=1.0,
       learning_rate=1.0,  # still decouple from partial_fit
   )

Then call ``decay()`` on a schedule as above.

See the :doc:`delayed reward example </notebooks/delayed-reward>` for
a full simulation that tunes the decay rate with optuna and shows how
too much decay hurts.
