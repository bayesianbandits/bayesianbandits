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
   bayesianbandits.InformationDirectedSampling
   bayesianbandits.EXP3A

Every policy declares the weakest posterior draws it can correctly
consume. An agent supplies anything at least that strong and picks
whichever is cheapest, so a policy never names a sampling method.

.. autosummary::
   :toctree: generated/

   bayesianbandits.DrawKind

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
should be passed to the ``learner`` argument of an ``Arm``. Each of them
implements a ``decay`` method that forgets: the clock ticked, and the
posterior widens by a forgetting rule so the model can follow rewards that
change with time (a restless bandit).

.. autosummary::
   :toctree: generated/

   bayesianbandits.BayesianGLM
   bayesianbandits.DirichletClassifier
   bayesianbandits.GammaRegressor
   bayesianbandits.NormalRegressor
   bayesianbandits.NormalInverseGammaRegressor

Forgetting Rules
----------------

A forgetting rule says how a posterior widens and carries its own rate.
The uniform rules are what ``decay`` applies on a schedule; the
directional rules forget only along what a batch excited. See
:doc:`math/forgetting` for the equations.

.. autosummary::
   :toctree: generated/

   bayesianbandits.ExponentialForgetting
   bayesianbandits.StabilizedForgetting
   bayesianbandits.FeatureWiseForgetting
   bayesianbandits.SiftForgetting

Empirical Bayes Estimators
--------------------------

These estimators automatically tune their hyperparameters via evidence
maximization (MacKay's update rules). They are drop-in replacements for their
base estimators and are especially useful when hyperparameters are unknown or
when the environment may be non-stationary (their ``decay`` defaults to
stabilized forgetting, which never forgets the prior).

.. autosummary::
   :toctree: generated/

   bayesianbandits.EmpiricalBayesDirichletClassifier
   bayesianbandits.EmpiricalBayesGammaRegressor
   bayesianbandits.EmpiricalBayesGLM
   bayesianbandits.EmpiricalBayesNormalRegressor

Posterior Approximators
-----------------------

These control how :class:`~bayesianbandits.BayesianGLM` approximates
the non-conjugate posterior. Pass to the ``approximator`` argument.

.. autosummary::
   :toctree: generated/

   bayesianbandits.LaplaceApproximator
   bayesianbandits.RVGAApproximator
