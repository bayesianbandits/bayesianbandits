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

This library is still under development, and the API is subject to change.

Bandit and Arm Classes
======================

The `Arm` class is the base class for all arms in a bandit. Its constructor
takes two arguments, `action_function` and `reward_function`, which represent
the action taken by the `pull` method of the arm and the mechanism for computing
the reward from the outcome of the action.

.. autosummary::
    :toctree: _autosummary

    Bandit
    Arm


Bandit Decorators
=================

These class decorators can be used to specialize `Bandit` subclasses for
particular problems.

.. autosummary::
    :toctree: _autosummary

    contextual
    restless

Policies
========

These functions can be used to create policy functions for bandits. They should
be passed to the `policy` argument of the `bandit` decorator.

.. autosummary::
    :toctree: _autosummary

    epsilon_greedy
    thompson_sampling
    upper_confidence_bound

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
from ._basebandit import (
    Bandit,
    DelayedRewardException,
    DelayedRewardWarning,
    contextual,
    restless,
)
from ._estimators import (
    DirichletClassifier,
    GammaRegressor,
    NormalInverseGammaRegressor,
    NormalRegressor,
)
from ._policy_decorators import (
    epsilon_greedy,
    thompson_sampling,
    upper_confidence_bound,
)
