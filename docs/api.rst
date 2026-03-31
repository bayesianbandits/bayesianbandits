API Reference
=============

Agent API
---------

The Agent API is the most ergonomic way to use this library in production. It is
designed to maximize your IDE's ability to autocomplete and type-check your
code. Additionally, it is designed to make it easy to modify the arms and the
policies of your bandit as your needs change.

.. autosummary::
   :toctree: generated/

   bayesianbandits.Agent
   bayesianbandits.ContextualAgent
   bayesianbandits.LipschitzContextualAgent
   bayesianbandits.Arm

Policy Functions
----------------

.. autosummary::
   :toctree: generated/

   bayesianbandits.EpsilonGreedy
   bayesianbandits.ThompsonSampling
   bayesianbandits.UpperConfidenceBound
   bayesianbandits.EXP3A

Pipelines
---------

Pipelines enable the use of sklearn transformers with Bayesian bandits,
providing preprocessing capabilities at different levels.

.. autosummary::
   :toctree: generated/

   bayesianbandits.AgentPipeline
   bayesianbandits.LearnerPipeline

Arm Featurizers
---------------

Arm featurizers enable shared model bandits by transforming context features
based on action tokens. They support vectorized operations for efficient
multi-arm processing.

.. autosummary::
   :toctree: generated/

   bayesianbandits.ArmFeaturizer
   bayesianbandits.ArmColumnFeaturizer
   bayesianbandits.FunctionArmFeaturizer

Estimators
----------

These estimators are the underlying models for the arms in a bandit. They
should be passed to the ``learner`` argument of the ``bandit`` decorator. Each
of these Bayesian estimators can be converted to a recursive estimator by
passing a ``learning_rate`` argument to the constructor that is less than 1.
Each of them implement a ``decay`` method that uses the ``learning_rate`` to
increase the variance of the prior. This is a type of state-space model that
is useful for restless bandits.

.. autosummary::
   :toctree: generated/

   bayesianbandits.BayesianGLM
   bayesianbandits.DirichletClassifier
   bayesianbandits.GammaRegressor
   bayesianbandits.NormalRegressor
   bayesianbandits.NormalInverseGammaRegressor

Empirical Bayes Estimators
--------------------------

These estimators automatically tune their hyperparameters via evidence
maximization (MacKay's update rules). They are drop-in replacements for their
base estimators and are especially useful when hyperparameters are unknown or
when the environment may be non-stationary (pair with ``learning_rate < 1``
for decay as a defensive default).

.. autosummary::
   :toctree: generated/

   bayesianbandits.EmpiricalBayesDirichletClassifier
   bayesianbandits.EmpiricalBayesGammaRegressor
   bayesianbandits.EmpiricalBayesNormalRegressor

Utilities
---------

.. autosummary::
   :toctree: generated/

   bayesianbandits.LaplaceApproximator
