==================
Usage and Examples
==================

This guide provides a progressive learning path from basic concepts to advanced production scenarios.

Getting Started
===============

Start here if you're new to multi-armed bandits or this library.

.. toctree::
    :maxdepth: 1

    introduction
    quickstart

**Introduction** (``introduction``): Why multi-armed bandits, why Bayesian,
and which estimator / agent / policy combination to use.

**Quick Start** (``quickstart``): Get a working bandit in 5 minutes. The
fastest path from zero to a bandit that learns.

Core Concepts
=============

Essential techniques for effective bandit implementation.

.. toctree::
    :maxdepth: 1

    notebooks/demo
    notebooks/counts
    notebooks/linear-bandits

**Binary Outcomes** (`demo`): Thompson sampling with binary rewards (click / no click), custom reward functions, batch updates, and model persistence.

**Bayesian Updating** (`counts`): Understanding how bandits learn from rewards using conjugate priors and posterior distributions.

**Contextual Bandits** (`linear-bandits`): Using user and item features to make personalized decisions with linear models.

Advanced Techniques
===================

Powerful methods for improved efficiency and performance.

.. toctree::
    :maxdepth: 1

    notebooks/empirical-bayes
    notebooks/hybrid-bandits
    notebooks/persistence

**Automatic Hyperparameter Tuning** (`empirical-bayes`): Learn prior precision and noise precision from data via evidence maximization, eliminating sensitivity to hyperparameter choices.

**Cross-Arm Learning** (`hybrid-bandits`): Use ``LipschitzContextualAgent`` with different design matrix structures to express disjoint, hybrid, or hierarchical bandits. Demonstrates how the design matrix encodes assumptions about arm relationships.

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

