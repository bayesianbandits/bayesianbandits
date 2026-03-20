============
Introduction
============

When to use a bandit
====================

Some decisions are worth making once. If you're choosing a button color or a
checkout flow, run an A/B test, pick the winner, ship it, and move on. The
answer won't change, and there's no reason to keep a model running.

Other decisions you need to know you're somewhere close to optimal at all
times. Dynamic pricing, marketing mix, ad creative rotation, personalized
recommendations. The best action shifts continuously, and you don't want to
rerun an A/B test every time it does. A bandit learns from every observation
and adjusts immediately, so you stay near-optimal without ever stopping to
redesign an experiment. You pay by carrying a model in production, but you
gain by never leaving value on the table.

Why Bayesian
============

The bandit problem has a natural Bayesian structure. You maintain *beliefs*
about how good each option is (a posterior distribution over parameters). You
have a *policy* for acting on those beliefs (Thompson sampling, UCB, etc.).
You observe an outcome, and you update your beliefs via Bayes' rule. The
posterior is your state; the policy is your decision rule.

Real bandit problems are anytime (no known horizon), contextual (decisions
depend on who and what), and nonstationary (the world drifts). Posteriors
quantify uncertainty for exploration without a fixed sample-size calculation.
Conditioning on covariates is just regression. Discounting old observations
through the prior lets you track a changing environment.

Why conjugate models
====================

Bandits learn online. Every observation updates the model immediately, and the
next decision depends on the updated state. This rules out any method that
requires batch retraining or iterating to convergence between decisions. MCMC
produces samples from the posterior, not a parametric representation you can
incrementally update; when new data arrives, you'd have to rerun the chain
over the full dataset. What remains are conjugate models (closed-form
posteriors that accept rank-1 updates) and variational approximations.

This library uses conjugate models. Each observation is a rank-1 update to a
precision matrix, O(d²) whether it's your thousandth or your ten millionth.
Memory is O(d²) in feature dimension, independent of observation count.
Drawing from the posterior is a Cholesky solve; you can make promises about latency
and throughput in production.

Choosing your setup
===================

Building a bandit requires three independent choices: an estimator, an agent,
and a policy. Pick one from each.

Estimator: what does your reward look like?
-------------------------------------------

The estimator is the Bayesian model inside each arm.

**Intercept-only models** (no covariates, one parameter per arm)

:class:`~bayesianbandits.DirichletClassifier`
    Binary or categorical outcomes (click / no-click, convert / bounce).
    Dirichlet-Multinomial conjugate posterior over class probabilities.

:class:`~bayesianbandits.GammaRegressor`
    Count or rate data (transactions per period, events per session).
    Gamma-Poisson conjugate posterior over the rate parameter.

These are intercept-only models. Each unique value of the first feature gets
its own posterior. They show up frequently in examples and tutorials because
they're easy to reason about, but they can't condition on covariates, which
limits their usefulness in production.

**Linear models with covariates** (conditionally normal outcomes)

:class:`~bayesianbandits.NormalRegressor`
    Bayesian linear regression with known noise variance. Gaussian prior on
    weights, exact conjugate updates. Use this when you can set the noise
    precision yourself or estimate it offline.

:class:`~bayesianbandits.NormalInverseGammaRegressor`
    Bayesian linear regression with unknown noise variance. Normal-inverse-gamma
    prior over weights and variance jointly. The marginal posterior over weights
    is a multivariate t, giving heavier-tailed uncertainty when you have little
    data.

:class:`~bayesianbandits.EmpiricalBayesNormalRegressor`
    Extends :class:`~bayesianbandits.NormalRegressor` with automatic
    hyperparameter tuning via MacKay's evidence maximization. Learns both the
    prior precision and noise precision from data, so you don't need to get the
    initial values right. With decay enabled, uses stabilized forgetting
    (Kulhavy & Zarrop) to keep regularization active rather than letting the
    prior wash out.

**Generalized linear models** (non-normal outcomes with covariates)

:class:`~bayesianbandits.BayesianGLM`
    Bayesian GLM with Laplace approximation via iteratively reweighted least
    squares. Supports logit link (binary outcomes) and log link (count data).
    Use this when you need covariates for binary or count outcomes.

Agent: do you have context? How many arms?
------------------------------------------

The general case is :class:`~bayesianbandits.LipschitzContextualAgent`, which
uses a single shared learner where the design matrix encodes arm identity,
context features, and any relationships between arms. How you construct the
design matrix determines what the bandit can learn: disjoint blocks give you
independent arms, shared columns let arms borrow strength, and interaction
terms let context affect arms differently. See the
:doc:`hybrid bandits tutorial <notebooks/hybrid-bandits>` for examples.

The other two agents are convenience wrappers:

:class:`~bayesianbandits.Agent`
    No context, independent arms. Each arm gets its own learner with an
    intercept-only model. The classic K-armed bandit.

:class:`~bayesianbandits.ContextualAgent`
    Context features, but each arm still gets its own independent learner.
    No cross-arm learning.

Policy: how do you want to explore?
------------------------------------

**Thompson sampling** (the default choice)
    :class:`~bayesianbandits.ThompsonSampling` draws a sample from each arm's
    posterior and picks the highest. Explores more when uncertain, exploits
    when confident. Never stops exploring entirely, so it adapts if underlying
    rates change.

**Upper confidence bound** (explicit optimism)
    :class:`~bayesianbandits.UpperConfidenceBound` picks the arm with the
    highest upper quantile of its posterior. More aggressive exploration of
    uncertain arms than Thompson sampling, deterministic given the same state.

**Epsilon-greedy** (a simple knob)
    :class:`~bayesianbandits.EpsilonGreedy` exploits the best arm with
    probability 1 - epsilon and explores uniformly at random otherwise. Easy to
    explain, easy to tune, but doesn't use uncertainty information.

**EXP3** (adversarial environments)
    :class:`~bayesianbandits.EXP3A` makes no stochastic assumptions about
    rewards. Use it when the environment is adversarial or non-stationary in
    ways that violate the assumptions of the other policies.

Where to start
==============

The :doc:`quickstart` walks through a complete pull-update loop in under
5 minutes. The :doc:`API reference <api>` has full details on every class
mentioned above.
