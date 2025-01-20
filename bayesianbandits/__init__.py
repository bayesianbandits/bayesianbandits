"""
========================================
bayesianbandits (:mod:`bayesianbandits`)
========================================

.. currentmodule:: bayesianbandits

A Python library for Bayesian Multi-Armed Bandits.

This library implements a variety of multi-armed bandit algorithms, including
epsilon-greedy, Thompson sampling, and upper confidence bound. It also
handles a number of common problems in multi-armed bandit problems, including
contextual bandits, delayed reward, and restless bandits.

This library is designed to be easy to use and extend. It is built on top of
scikit-learn, and uses scikit-learn-style estimators to model the arms. This
allows you to use any scikit-learn estimator that supports the `partial_fit`
and `sample` methods as an arm in a bandit. Restless bandits also require the
`decay` method.

The Agent API found in `bayesianbandits.api` is stable and is battle-tested in
production environments.

Agent API
=========

The Agent API is the most ergonomic way to use this library in production. It is
designed to maximize your IDE's ability to autocomplete and type-check your
code. Additionally, it is designed to make it easy to modify the arms and the
policies of your bandit as your needs change.

The Agent API requires a slightly different interface for choice policies than
the old `Bandit` API, but these policies and the policy decorators use the same
underlying code. Both are available for backwards compatibility.

.. autosummary::
    :toctree: _autosummary

    Agent
    ContextualAgent
    EpsilonGreedy
    ThompsonSampling
    UpperConfidenceBound
    Arm


Estimators
==========

These estimators are the underlying models for the arms in a bandit. They
should be passed to the `learner` argument of the `bandit` decorator. Each
of these Bayesian estimators can be converted to a recursive estimator by
passing a `learning_rate` argument to the constructor that is less than 1.
Each of them implement a `decay` method that uses the `learning_rate` to
increase the variance of the prior. This is a type of state-space model that
is useful for restless bandits.

.. autosummary::
    :toctree: _autosummary

    DirichletClassifier
    GammaRegressor
    NormalRegressor
    NormalInverseGammaRegressor

Exceptions
==========

These are custom exceptions raised by the bandit classes.

.. autosummary::
    :toctree: _autosummary

    DelayedRewardException
    DelayedRewardWarning

"""

from ._arm import Arm

from ._estimators import (
    DirichletClassifier,
    GammaRegressor,
    NormalInverseGammaRegressor,
    NormalRegressor,
)

from .api import (
    Agent,
    ContextualAgent,
    EpsilonGreedy,
    ThompsonSampling,
    UpperConfidenceBound,
)
