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

.. autosummary::

    Agent
    ContextualAgent
    LipschitzContextualAgent
    EpsilonGreedy
    ThompsonSampling
    UpperConfidenceBound
    InformationDirectedSampling
    EXP3A
    Arm

Pipelines
=========
Pipelines enable the use of sklearn transformers with Bayesian bandits,
providing preprocessing capabilities at different levels.

.. autosummary::

    AgentPipeline
    LearnerPipeline

Arm Featurizers
===============
Arm featurizers enable shared model bandits by transforming context features
based on action tokens. They support vectorized operations for efficient
multi-arm processing.

.. autosummary::

    ArmFeaturizer
    ArmColumnFeaturizer
    FunctionArmFeaturizer


Estimators
==========

These estimators are the underlying models for the arms in a bandit. They
should be passed to the `learner` argument of an `Arm`. Each of them
implements a `decay` method that forgets: the clock ticked, and the
posterior widens by a forgetting rule so the model can follow rewards
that change with time (a restless bandit).

.. autosummary::

    BayesianGLM
    DirichletClassifier
    GammaRegressor
    NormalRegressor
    NormalInverseGammaRegressor

Forgetting Rules
================

A forgetting rule says how a posterior widens and carries its own rate.
Pass one as an estimator's `forgetting=` to forget on every `partial_fit`,
for change that happens because you observed, or to `decay` on a
schedule, for change that happens with time. The uniform rules serve
both; the directional rules forget only along what a batch excited and
belong on `forgetting=`.

.. autosummary::

    ExponentialForgetting
    StabilizedForgetting
    FeatureWiseForgetting
    SiftForgetting

Empirical Bayes Estimators
==========================

These estimators automatically tune their hyperparameters via evidence
maximization (MacKay's update rules). They are drop-in replacements for their
base estimators and are especially useful when hyperparameters are unknown or
when the environment may be non-stationary (their ``decay`` defaults to
stabilized forgetting, which never forgets the prior).

.. autosummary::

    EmpiricalBayesDirichletClassifier
    EmpiricalBayesGammaRegressor
    EmpiricalBayesGLM
    EmpiricalBayesNormalRegressor

Posterior Approximators
=======================
These control how :class:`BayesianGLM` approximates the non-conjugate
posterior. Pass to the ``approximator`` argument.

.. autosummary::

    LaplaceApproximator
    RVGAApproximator

Memory Accounting
=================
Agents, arms, learners, and precision factors each carry a
``memory_usage`` property reporting the bytes they retain, broken down
by part. The same walk is available as a function for anything else.
Buffers reachable more than once -- a precision matrix shared with its
factorization, a learner shared across arms -- are charged once, so a
total is what dropping the object would free.

.. autosummary::

    memory_usage
    MemoryUsage

"""

from ._arm import Arm
from ._arm_featurizer import ArmFeaturizer
from ._draw_kind import DrawKind
from ._eb_estimators import (
    EmpiricalBayesDirichletClassifier,
    EmpiricalBayesGammaRegressor,
    EmpiricalBayesGLM,
    EmpiricalBayesNormalRegressor,
)
from ._estimators import (
    BayesianGLM,
    DirichletClassifier,
    GammaRegressor,
    NormalInverseGammaRegressor,
    NormalRegressor,
)
from ._forgetting import (
    ExponentialForgetting,
    FeatureWiseForgetting,
    SiftForgetting,
    StabilizedForgetting,
)
from ._gaussian import LaplaceApproximator, RVGAApproximator
from ._memory import MemoryUsage, memory_usage
from .api import (
    Agent,
    ContextualAgent,
    EpsilonGreedy,
    LipschitzContextualAgent,
    ThompsonSampling,
    UpperConfidenceBound,
)
from .featurizers import ArmColumnFeaturizer, FunctionArmFeaturizer
from .pipelines import AgentPipeline, LearnerPipeline
from .policies import EXP3A, InformationDirectedSampling

__all__ = [
    "Arm",
    "ArmFeaturizer",
    "DrawKind",
    "ArmColumnFeaturizer",
    "FunctionArmFeaturizer",
    "BayesianGLM",
    "DirichletClassifier",
    "EmpiricalBayesDirichletClassifier",
    "EmpiricalBayesGammaRegressor",
    "EmpiricalBayesGLM",
    "EmpiricalBayesNormalRegressor",
    "GammaRegressor",
    "NormalInverseGammaRegressor",
    "NormalRegressor",
    "LaplaceApproximator",
    "RVGAApproximator",
    "MemoryUsage",
    "memory_usage",
    "Agent",
    "ContextualAgent",
    "LipschitzContextualAgent",
    "EpsilonGreedy",
    "ThompsonSampling",
    "UpperConfidenceBound",
    "InformationDirectedSampling",
    "EXP3A",
    "AgentPipeline",
    "LearnerPipeline",
    "ExponentialForgetting",
    "StabilizedForgetting",
    "FeatureWiseForgetting",
    "SiftForgetting",
]
