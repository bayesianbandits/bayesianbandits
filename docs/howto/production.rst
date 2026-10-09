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


Save state without pickle
-------------------------

A pickle restores classes by name, so a checkpoint can break when the
library is refactored, and loading one runs whatever code it names.
``state_dict`` returns only what the agent learned -- each arm's
posterior and tuned hyperparameters, the arm queued for update, and the
generator state -- as dicts, lists, numpy arrays and Python scalars.
Build the agent in code as before and load the state into it:

.. code-block:: python

   import numpy as np
   from bayesianbandits import Agent, Arm, GammaRegressor, ThompsonSampling

   def make_agent():
       arms = [
           Arm("ad_a", learner=GammaRegressor(alpha=1, beta=1)),
           Arm("ad_b", learner=GammaRegressor(alpha=1, beta=1)),
       ]
       return Agent(arms, ThompsonSampling())

   agent = make_agent()
   agent.pull()
   agent.update(np.array([1.0]))

   state = agent.state_dict()  # store with any codec for plain data
   loaded = make_agent()
   loaded.load_state_dict(state)

   # The loaded agent continues the original's random stream
   assert loaded.pull() == agent.pull()

A learner's state describes its posterior, not the class: a ``family``
(``gaussian``, ``dirichlet`` or ``gamma``) and blocks such as ``prior``
and ``posterior``, each with its own version, which loading checks. So
any estimator of a family loads it -- a dense model's state into a
sparse one, or a ``NormalRegressor``'s into an
``EmpiricalBayesNormalRegressor``, which tunes on from there. An agent
stores ``[token, state]`` pairs, so tokens that are not strings survive
JSON, and loading raises if the tokens differ from its arms. A
``LipschitzContextualAgent`` stores its shared learner once.

The generator continues where it left off, so reseed with
``loaded.rng = None`` for copies that should explore differently.
Factorizations are rebuilt rather than stored, so after a ``decay``, or
a sparse update without ``scikit-sparse``, draws can differ from the
original's in the last bit, as after unpickling.


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
