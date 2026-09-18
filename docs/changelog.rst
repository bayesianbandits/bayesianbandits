Changelog
=========

Unreleased
----------

**Breaking changes**

- ``learning_rate`` is removed from every estimator. Forgetting on each
  ``partial_fit`` is now ``forgetting=``, which takes a rule:
  ``ExponentialForgetting(0.99)`` on the plain estimators,
  ``StabilizedForgetting(0.99)`` on the empirical Bayes ones. The numbers
  are unchanged, and pickles carrying ``learning_rate`` are converted on
  load. See :doc:`/howto/decay` (#303)
- ``decay`` is ``decay(forgetting=None, *, decay_rate=None, steps=1)`` on
  every estimator, arm, agent and pipeline. It no longer takes a context
  array (pass ``steps=``), ``decay_rate`` is keyword-only, and a call with
  neither raises. A custom learner used inside an ``Arm`` must accept the
  new signature (#302, #303)
- :class:`~bayesianbandits.ContextualAgent` and :class:`~bayesianbandits.Agent`
  require one learner per arm and raise from ``add_arm`` otherwise. Arms
  sharing a learner were sampled independently, discarding their
  correlation; use :class:`~bayesianbandits.LipschitzContextualAgent` to
  share a model (#267)
- :class:`~bayesianbandits.AgentPipeline` raises ``TypeError`` on a
  non-contextual ``Agent``, whose steps had nothing to transform and never
  ran. Use the agent directly (#291)
- :class:`~bayesianbandits.EmpiricalBayesNormalRegressor` regularizes the
  MacKay ``alpha`` update with a Gamma hyperprior, weighted by the new
  ``alpha_prior_strength`` (default ``0.2``; ``0.0`` restores the previous
  update). Learned alphas change under the default (#288)
- ``fit``'s first parameter is ``X`` on every estimator; it was ``X_fit`` on
  ``NormalRegressor`` and ``EmpiricalBayesNormalRegressor`` (#292)
- Seeded trajectories change under ``UpperConfidenceBound``, ``EXP3A``,
  ``EpsilonGreedy`` and joint ``sample``, since the new sampling routes
  consume randomness differently. Every distribution is unchanged (#258,
  #269)

**New features**

- The forgetting rules are public, each carrying its own ``rate``:
  :class:`~bayesianbandits.ExponentialForgetting`,
  :class:`~bayesianbandits.StabilizedForgetting`,
  :class:`~bayesianbandits.SiftForgetting`, and the new sparsity-preserving
  :class:`~bayesianbandits.FeatureWiseForgetting`. Pass a uniform rule to
  ``decay(rule, steps=)`` or any rule as an estimator's ``forgetting=``.
  See :doc:`/math/forgetting` (#300, #302, #303)
- :class:`~bayesianbandits.EmpiricalBayesGLM`: ``BayesianGLM`` with MacKay
  tuning of the prior precision. See :doc:`/math/glm-eb` (#283, #288)
- :class:`~bayesianbandits.InformationDirectedSampling`: variance-based
  IDS after Russo & Van Roy (2018), with ``top_k`` (#270)
- :class:`~bayesianbandits.DrawKind`: a policy declares the weakest draws
  it can consume through a ``consumes`` attribute, and the agent picks the
  cheapest exact sampling route. Custom policies default to ``JOINT`` (#270)
- ``sample_marginal`` and ``sample_reward_space`` on ``NormalRegressor``,
  ``NormalInverseGammaRegressor`` and ``BayesianGLM``: exact per-row and
  reward-space joint draws at a cost independent of the feature count
  (#258, #269)
- :func:`~bayesianbandits.memory_usage` on agents, arms, learners,
  pipelines and precision factors, reporting retained bytes by part (#271)
- ``BayesianGLM`` damps each IRLS step and raises ``ConvergenceWarning``
  when the iteration budget runs out short of the mode (#285)

**Bug fixes**

- ``EmpiricalBayesNormalRegressor``: the ``beta`` statistics ignored row
  weights, silently skipping the update under forgetting or
  ``sample_weight`` (#279); MacKay's ``gamma`` used the undecayed prior
  scalar, making EB a no-op on wide sparse designs (#275); ``coef_`` was
  not re-solved after a MacKay rescale (#277)
- ``LaplaceApproximator`` and ``RVGAApproximator`` reject ``n_iter <= 0``
  instead of returning the prior as a fitted posterior (#293)
- A ``partial_fit`` that raises mid-update no longer leaves the posterior
  half-advanced (#278, #281)

**Performance**

- Posterior sampling on ``NormalRegressor``, ``NormalInverseGammaRegressor``
  and ``BayesianGLM`` is faster, most of all for joint draws with many
  samples and on sparse models with many never-observed features (#265,
  #269, #273)
- ``UpperConfidenceBound``, ``EXP3A`` and ``EpsilonGreedy`` pull faster;
  ``ThompsonSampling`` is unchanged (#258, #260)
- ``InformationDirectedSampling.select`` is 50-100x faster with identical
  decisions (#270, #273)
- Dense ``partial_fit`` is faster, and no longer slows down under scipy
  1.18 (#297, #299)
- Sparse empirical Bayes ``partial_fit`` is faster (#276, #277, #283)
- Sparse models hold less memory in long-lived serving processes (#278,
  #290)

**Documentation**

- Notebook: :doc:`How do you pick a forgetting strategy?
  </notebooks/forgetting>` (#304)
- Math reference for :doc:`EmpiricalBayesGLM </math/glm-eb>` (#283) and
  :doc:`feature-wise forgetting </math/forgetting>` (#300)

**Infrastructure**

- Dependency bumps to clear Dependabot alerts: pillow, mistune, soupsieve
  (#257)

1.4.0 (2026-07-31)
------------------

**New features**

- Empirical Bayes Gamma regressor: Gamma-Poisson regressor with automatic
  prior tuning via the Negative Binomial marginal likelihood, using Minka's
  fixed-point EM with the generalized Newton update for the shape parameter.
  Stabilized forgetting re-injects the tuned prior after each decay step
  (#244)
- SIFt directional forgetting: forgets only in the excited directions of each
  batch, retaining full precision in unexcited directions. Guarantees an
  eigenvalue floor without artificial prior injection (Lai & Bernstein 2024)
  (#245)
- ``RVGAApproximator`` for ``BayesianGLM``: R-VGA posterior approximation that
  replaces Laplace's point-estimate curvature with expected curvature under
  the approximate posterior, correcting systematic bias for non-Gaussian
  likelihoods. Supports the exact log-link closed form, an analytical probit
  approximation, and Gauss-Hermite quadrature for the logit link, with
  minibatched updates for large sparse models (#254)

**Performance**

- Cache the precision Cholesky factor and thread it through posterior
  approximators to avoid redundant factorizations (#253)
- Reuse the ``dsymv`` result to optimize dense linear regression updates (#247)
- Drop a redundant O(p\ :sup:`2`) matvec in ``NormalInverseGammaRegressor._fit_helper``
  (#246)

**Documentation**

- Mathematical reference for forgetting strategies (exponential, stabilized
  Kulhavy-Zarrop, and directional SIFt), with expanded ``_forgetting.py``
  docstrings and cross-references from the normal and empirical-Bayes pages
  (#248)
- Gamma empirical-Bayes math reference page (#244)
- R-VGA GLM notebook and example demonstrating the expected-curvature
  approximation (#254)

**Infrastructure**

- Test against scikit-learn 1.9.0, drop 1.5.2 (#252)
- Dependency bumps for security advisories: urllib3 2.7.0 (#249), dev
  dependencies (#250), pytest 9.0.3 (#251)

1.3.0 (2026-03-28)
------------------

**New features**

- Empirical Bayes Dirichlet classifier with automatic prior tuning via
  Minka's fixed-point iteration for the Dirichlet-Multinomial marginal
  likelihood, with stabilized forgetting
- Empirical Bayes normal regressor with automatic hyperparameter tuning via
  MacKay's evidence maximization (#200)
- Kulhavy-Zarrop stabilized forgetting to prevent prior collapse under decay
  (#202)
- Takahashi recursion for efficient trace computation in sparse precision
  matrices (#204), with Cython implementation (#206)
- Sparse factor caching to avoid redundant factorizations (#198)
- ``rng`` property with setter for reseeding agents and pipelines after
  deserialization (#224)

**Performance**

- BLAS-level optimizations for NormalRegressor (#219), BayesianGLM IRLS (#218),
  and EmpiricalBayesNormalRegressor (#220)
- Refactored sparse factor classes for better performance and reuse (#213, #214)
- Benchmark suite with pytest-benchmark (#217, #219, #220, #221)
- Modernized Cython code with typed memoryviews (#212)

**Documentation**

- Complete documentation overhaul following Diataxis framework
- How-to guides: pipelines, decay, reward functions, delayed rewards,
  production deployment, sparse features
- Mathematical reference: NormalRegressor, NIG, empirical Bayes, Dirichlet EB,
  intercept-only models, GLM, exploration policies
- Explanation pages: "Knowledge Is Prediction" (worldview), "Separating
  Inference from Decisions" (decision theory)
- Comprehensive docstrings for all estimators, policies, agents, and arms
- Quick-start guide (#223)

**Infrastructure**

- Cross-platform wheel builds via cibuildwheel (Linux x86_64/aarch64, macOS
  arm64, Windows x86_64)
- Migrated from black + flake8 to ruff (#215)
- NumPy 2.0 dependency, scikit-sparse 0.5.0 (#188, #205)
- Pickling support fix for BayesianGLM (#196)
