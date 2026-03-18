bayesianbandits
===============

``bayesianbandits`` is a Python library for multi-armed bandits with Bayesian
learning. It provides conjugate estimators (binary, continuous, count rewards),
three agent types (classic, contextual, shared-learner), and exploration
policies (Thompson sampling, UCB, epsilon-greedy, EXP3) -- all with O(1)
online updates and a two-method API: ``pull()`` to decide, ``update()`` to
learn.

.. toctree::
   :maxdepth: 2
   :hidden:

   getting-started
   howto/index
   math/index
   examples
   api
