Deploying to Production
=======================


Serialize with joblib
----------------------

``joblib`` is a dependency of scikit-learn, so it's already installed.

.. note::

   Lambdas, closures, and factory-produced functions (like the
   ``make_profit_reward`` pattern in :doc:`reward-functions`) are not
   picklable with standard ``pickle``. If your arms use reward
   functions like these, use a callable class with ``__call__``
   instead, or serialize with ``cloudpickle``.

.. code-block:: python

   import joblib
   import numpy as np
   from bayesianbandits import Agent, Arm, GammaRegressor, ThompsonSampling

   arms = [
       Arm("ad_a", learner=GammaRegressor(alpha=1, beta=1)),
       Arm("ad_b", learner=GammaRegressor(alpha=1, beta=1)),
   ]
   agent = Agent(arms, ThompsonSampling(), random_seed=42)

   (choice,) = agent.pull()
   agent.update(np.array([1.0]))

   joblib.dump(agent, "agent.pkl", compress=True)
   loaded = joblib.load("agent.pkl")

   # Learned state is preserved
   assert loaded.arms[0].learner.coef_[1][0] == agent.arms[0].learner.coef_[1][0]

Uncompressed pickles of precision matrices can be large. The learned
state compresses efficiently: a sparse model with 1M features and 4M
nonzeros is a couple hundred KB at rest.


Reseed the RNG after loading
-----------------------------

After deserialization, the RNG state is frozen from save time. Every
copy loaded from the same file replays the exact same exploration
sequence. Reseed immediately after loading:

.. code-block:: python

   loaded = joblib.load("agent.pkl")
   loaded.rng = None  # seeds from OS entropy

This creates a fresh ``numpy.random.Generator`` and propagates it to
all arm learners. Pass an ``int`` instead if you need reproducibility.


Add and remove arms at runtime
-------------------------------

New arms start with a fresh prior. Existing arms keep their learned
state:

.. code-block:: python

   from bayesianbandits import Arm, GammaRegressor

   # Add a new arm
   loaded.add_arm(Arm("ad_c", learner=GammaRegressor(alpha=1, beta=1)))

   # Remove an underperforming arm
   loaded.remove_arm("ad_a")

   joblib.dump(loaded, "agent.pkl")

Removing an arm is destructive: its learned state is gone on
re-serialization. Action tokens must be unique across arms.


.. important::

   **Isolate from your application server.** BLAS and LAPACK, which
   back every ``pull()`` and ``update()`` call, will eagerly use all
   available cores. A single Cholesky solve can saturate a machine for
   the duration of the call. If the bandit lives on the same server as
   your application, a burst of pulls can starve your request-handling
   threads. Run the bandit in a separate process or on a dedicated
   host.

   Agents are mutable and not thread-safe. Keep one agent per process.
   ``joblib.load`` produces an independent copy, so you can run
   multiple reader processes that pull concurrently and funnel updates
   through a single writer.
