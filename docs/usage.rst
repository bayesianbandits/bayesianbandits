==================
Usage and Examples
==================

This guide provides a progressive learning path from basic concepts to advanced production scenarios.

Getting Started
===============

Start here if you're new to multi-armed bandits or this library.

.. toctree::
    :maxdepth: 1

    notebooks/demo

**What you'll learn**: Your first bandit with Thompson sampling. Covers the basic pull-update cycle and core concepts like exploration vs exploitation.

Core Concepts
=============

Essential techniques for effective bandit implementation.

.. toctree::
    :maxdepth: 1

    notebooks/counts
    notebooks/linear-bandits

**Bayesian Updating** (`counts`): Understanding how bandits learn from rewards using conjugate priors and posterior distributions.

**Contextual Bandits** (`linear-bandits`): Using user and item features to make personalized decisions with linear models.

Advanced Techniques
===================

Powerful methods for improved efficiency and performance.

.. toctree::
    :maxdepth: 1

    notebooks/hybrid-bandits
    notebooks/persistence

**Cross-Arm Learning** (`hybrid-bandits`): Share knowledge across similar arms for faster learning and better sample efficiency. Essential for large action spaces.

**Production Deployment** (`persistence`): Patterns for saving, loading, and updating bandits in live systems with proper serialization.

Specialized Scenarios
=====================

Handle challenging real-world conditions.

.. toctree::
    :maxdepth: 1

    notebooks/adversarial
    notebooks/delayed-reward
    notebooks/offline-learning

**Adversarial Environments** (`adversarial`): Robust bandits for non-stationary environments using EXP3A algorithm.

**Delayed Feedback** (`delayed-reward`): Handling scenarios where rewards arrive hours or days after actions.

**Historical Data** (`offline-learning`): Bootstrap bandits using existing data before deploying online.

Agent Types Guide
=================

Choose the right agent for your use case:

**Agent**
    Non-contextual bandits where all users/items are identical. Use for simple A/B testing.

**ContextualAgent** 
    Contextual bandits with separate models per arm. Use when arms are completely different (e.g., different product categories).

**LipschitzContextualAgent**
    Contextual bandits with shared model across arms. Use for large action spaces where arms are similar (e.g., thousands of articles, products).

Pipeline Integration
====================

Integrate with sklearn preprocessing pipelines:

.. code-block:: python

    from bayesianbandits.pipelines import AgentPipeline
    from sklearn.preprocessing import StandardScaler
    
    # Preprocess contexts before bandit sees them
    pipeline = AgentPipeline([
        ('scaler', StandardScaler()),
        ('agent', ContextualAgent(arms, policy))
    ])
    
    # Or preprocess features within learners
    from bayesianbandits.pipelines import LearnerPipeline
    
    learner = LearnerPipeline([
        ('preprocessor', StandardScaler()),
        ('regressor', NormalRegressor())
    ])
