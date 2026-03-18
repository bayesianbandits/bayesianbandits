Handling Delayed Rewards
========================

With contextual bandits, you make decisions for many contexts at once
and collect rewards later. You serve ads to a batch of users now and
learn which ones converted hours later. You recommend products to
thousands of visitors and get purchase signals overnight. The API reflects this:
``pull()`` and ``update()`` are independent calls. There is no
requirement that they alternate.


The basic pattern
-----------------

Pull for a batch of contexts, store the action tokens alongside
request IDs, update later when rewards arrive.

.. code-block:: python

   from bayesianbandits import (
       Arm, ContextualAgent, NormalRegressor, ThompsonSampling,
   )
   import numpy as np

   arms = [
       Arm(f"variant_{i}", learner=NormalRegressor(alpha=1.0, beta=1.0))
       for i in range(3)
   ]
   agent = ContextualAgent(arms, ThompsonSampling(), random_seed=42)

   # Decision time
   X_batch = np.array([[1.0, 2.0]])
   actions = agent.pull(X_batch)
   # Store (request_id, action, context) in your database

   # Later, when rewards arrive
   for token, X_row, reward in zip(
       actions,
       X_batch,
       [1.0],  # rewards from your database
   ):
       agent.select_for_update(token).update(
           np.atleast_2d(X_row), np.atleast_1d(reward)
       )

Nothing about the posterior update depends on how much time has passed
since the pull.


Batch updates for the same arm
-------------------------------

When multiple rewards for the same arm arrive at once (common with a
nightly batch job), group them into a single ``update()`` call rather
than looping row by row. The learner does one precision-matrix update
instead of N sequential ones:

.. code-block:: python

   # Group rewards by arm token
   for token, group in rewards_grouped_by_arm.items():
       X_batch = np.vstack(group["contexts"])
       y_batch = np.array(group["rewards"])
       agent.select_for_update(token).update(X_batch, y_batch)

The posterior is the same whether you update in one batch or one row
at a time: the sufficient statistics are identical either way. (With
``learning_rate < 1``, each row triggers decay, so order matters. See
:doc:`decay` for why decoupling decay from updates avoids this.)


Multiple pulls before any update
---------------------------------

You can pull many times before updating. Each pull samples from the
current posterior. The only state ``pull()`` sets is
``arm_to_update``, which you override with ``select_for_update()``
anyway.


Interaction with decay
-----------------------

If you run ``agent.decay()`` on a schedule, a delayed observation is
still incorporated normally when it arrives. The model has already
decayed (widened its uncertainty), and the observation tightens it
back. Decay reflects time passing; the update reflects new
information.

If decay is aggressive and rewards arrive late, the posterior
variance has already grown by the time the observation lands. The
observation has *more* influence on the decayed posterior than it
would have on a tighter one, so a single stale reward can pull the
model further than you might expect.

.. code-block:: python

   # Daily loop
   X_today = np.array([[1.0, 2.0]])
   actions = agent.pull(X_today)
   # ... store decisions ...

   # Nightly: decay + update with yesterday's rewards
   agent.decay(np.array([[0.0, 0.0]]), decay_rate=0.99)
   for token, X_row, reward in yesterdays_rewards:
       agent.select_for_update(token).update(
           np.atleast_2d(X_row), np.atleast_1d(reward)
       )

See :doc:`decay` for choosing the rate.


Non-contextual agents
----------------------

:class:`~bayesianbandits.Agent` works the same way, just without context:

.. code-block:: python

   from bayesianbandits import Agent

   arms = [
       Arm(f"slot_{i}", learner=NormalRegressor(alpha=1.0, beta=1.0))
       for i in range(3)
   ]
   agent = Agent(arms, ThompsonSampling(), random_seed=42)

   (token,) = agent.pull()
   # later...
   agent.select_for_update(token).update(np.array([1.0]))


If something goes wrong
------------------------

**"I updated the wrong arm."**
``pull()`` sets ``arm_to_update`` to the last selected arm. If you pull
again before updating, the previous selection is overwritten. Always
use ``select_for_update(token)`` explicitly rather than relying on the
implicit state from ``pull()``.
