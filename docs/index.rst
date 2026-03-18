.. bayesianbandits documentation master file, created by
   sphinx-quickstart on Fri Feb 24 11:39:20 2023.
   You can adapt this file completely to your liking, but it should at least
   contain the root `toctree` directive.

Welcome to bayesianbandits's documentation!
===========================================

`bayesianbandits` is a Python library for multi-armed bandits with Bayesian
learning. It provides conjugate estimators (binary, continuous, count rewards),
three agent types (classic, contextual, shared-learner), and exploration
policies (Thompson sampling, UCB, epsilon-greedy, EXP3) — all with O(1)
online updates and a two-method API: ``pull()`` to decide, ``update()`` to
learn.

If you have any questions or suggestions, please feel free to open an issue
on the project page on GitHub or contact me at <rishi@kulkarni.science>.

.. toctree::
   :maxdepth: 3
   :caption: Contents:

   installation
   usage
   api


Indices and tables
==================

* :ref:`genindex`
* :ref:`modindex`
* :ref:`search`
