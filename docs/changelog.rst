Changelog
=========

Unreleased
----------

**Breaking changes**

- ``learning_rate`` is removed from every estimator. Forgetting on each
  ``partial_fit`` is now ``forgetting=``, which takes a rule that carries
  its own rate: ``NormalRegressor(..., learning_rate=0.99)`` becomes
  ``forgetting=ExponentialForgetting(0.99)``, and the empirical Bayes
  estimators, which always floored at the prior, take
  ``forgetting=StabilizedForgetting(0.99)``; ``learning_rate=1.0`` is the
  default ``forgetting=None``. A uniform rule steps once per row, so the
  numbers are unchanged, and models pickled with ``learning_rate`` are
  converted on load (#303)
- ``decay`` has one signature everywhere,
  ``decay(forgetting=None, *, decay_rate=None, steps=1)``. It no longer
  takes a context array (pass ``steps=`` for the number of ticks),
  ``decay_rate`` is keyword-only on the agents and pipelines, and a call
  with neither a rule nor ``decay_rate`` raises instead of falling back to
  the constructor rate. A custom learner used inside an ``Arm`` must accept
  the new signature (#302, #303)
- ``PosteriorApproximator.update_posterior`` takes ``prior_decay`` (the
  factor already raised to the batch size) in place of ``learning_rate``,
  with ``sample_weight`` arriving as the final per-row weights, and gains
  optional ``prior_floor`` and ``coef_init`` keywords. ``BayesianGLM`` only
  passes them when set; ``EmpiricalBayesGLM`` requires both (#283, #303)
- ``ContextualAgent`` and ``Agent`` require one learner per arm and raise
  from ``add_arm`` (so also from the constructor) when two arms share one.
  Arms sharing a learner were sampled independently, so joint draws lost
  all cross-arm correlation. Sharing one model across arms is
  ``LipschitzContextualAgent``'s job, and the error says so (#267)
- ``NonContextualAgentPipeline`` is removed and ``AgentPipeline`` is a class
  rather than a factory. Its steps transform context, which a plain
  ``Agent`` has none of, so they never ran; wrapping an ``Agent`` now raises
  ``TypeError`` pointing at ``LearnerPipeline``. ``ContextualAgentPipeline``
  remains as an alias (#291)
- ``EmpiricalBayesNormalRegressor`` regularizes the MacKay ``alpha`` update
  with a Gamma hyperprior at the constructor's ``alpha``, weighted by the
  new ``alpha_prior_strength`` (default ``0.2``). Plain MacKay ran ``alpha``
  to the guardrail on arms with near-zero coefficients and locked them
  there. Learned alphas change under the default; ``0.0`` restores the
  previous update (#288)
- ``fit``'s first parameter is ``X`` on every estimator; it was ``X_fit`` on
  ``NormalRegressor`` and ``EmpiricalBayesNormalRegressor`` (#292)
- Seeded trajectories change: ``UpperConfidenceBound``, ``EXP3A`` and
  ``EpsilonGreedy`` draw through the marginal path, and joint ``sample``
  draws consume different randomness depending on the route taken. Every
  distribution is unchanged (verified by KS tests); Thompson sampling is
  unchanged to the last bits on dense models and bit-for-bit on sparse
  models whose features have all been observed (#258, #269)

**New features**

- The forgetting rules are public: ``ExponentialForgetting``,
  ``StabilizedForgetting``, ``SiftForgetting``, and the new
  ``FeatureWiseForgetting`` (Saelid & Foss 1983), each carrying its own
  ``rate``. Pass a uniform rule to ``decay(rule, steps=)`` on any
  estimator, arm, agent or pipeline, or any rule as an estimator's
  ``forgetting=`` to apply on each ``partial_fit``. The directional rules
  forget only along what the batch excited; feature-wise preserves the
  sparsity pattern, so it is the one for sparse estimators, where
  ``SiftForgetting`` is refused. The grouped conjugate models and the
  empirical Bayes estimators take the uniform rules only (#300, #302, #303)
- ``EmpiricalBayesGLM``: ``BayesianGLM`` with MacKay tuning of the prior
  precision on the Laplace approximation, with the same Gamma hyperprior
  and stabilized forgetting as ``EmpiricalBayesNormalRegressor`` (#283,
  #288)
- ``InformationDirectedSampling``: variance-based information-directed
  sampling after Russo & Van Roy (2018), sampling from the two-arm mixture
  that minimizes the information ratio. ``top_k`` draws sequentially
  without replacement (#270)
- ``DrawKind``: a policy declares the weakest draws it can consume
  (``MARGINAL_ONLY < CONTEXT_JOINT < JOINT``) through a ``consumes``
  attribute, and the agent picks the cheapest exact sampling route. Custom
  policies default to ``JOINT``, which is always correct and never the
  cheapest (#270)
- ``sample_marginal`` and ``sample_reward_space`` on ``NormalRegressor``,
  ``NormalInverseGammaRegressor`` and ``BayesianGLM``: exact per-row
  marginal draws, and joint draws factored in reward space, both with
  per-draw cost independent of the feature count. ``Arm`` and
  ``LearnerPipeline`` forward ``sample_marginal`` (#258, #269)
- ``memory_usage`` on agents, arms, learners, pipelines and precision
  factors, plus a ``memory_usage()`` function, reporting retained bytes by
  part with shared buffers charged once (#271)
- ``BayesianGLM``'s Laplace IRLS damps each Newton step by halving and
  reports convergence: ``GaussianPosterior.converged`` is new,
  ``BayesianGLM`` raises sklearn's ``ConvergenceWarning`` when the budget
  runs out, and ``EmpiricalBayesGLM`` skips the MacKay step on a
  non-converged fit (#285)
- ``PolicyDefaultUpdate`` implements ``__call__``, so a custom policy needs
  only ``samples_needed`` and ``select`` (#291)

**Bug fixes**

- ``EmpiricalBayesNormalRegressor`` under per-update forgetting or
  ``sample_weight``: the ``beta`` sufficient statistics were unweighted, so
  the residual sum came out negative and the update was silently skipped
  in batches (#279); MacKay's ``gamma`` used ``alpha`` where it meant the
  decayed prior scalar, so EB became a no-op on wide, sparsely observed
  designs (#275); ``coef_`` was not re-solved after a MacKay rescale, so
  each step re-applied the old shrinkage (#277). Chunked ``partial_fit``
  now lands on the full fit's fixed point, and ``log_evidence_`` reports
  the objective actually maximized
- ``LaplaceApproximator`` and ``RVGAApproximator`` reject ``n_iter <= 0``
  instead of returning the prior as a fitted posterior or raising
  ``UnboundLocalError`` (#293)
- A ``partial_fit`` that raises mid-update no longer leaves the posterior
  or the empirical Bayes prior scalar half-advanced (#278, #281)

**Performance**

- Joint ``sample`` picks the cheapest of three exact routes (weight space,
  row side, or the columns a sparse ``X`` touches) from ``size``,
  ``n_rows`` and ``|U|``; sparse factors carry never-observed features as
  a diagonal, so factorization and sampling are sized by the observed
  block. Sparse ``sample(size=500)`` at 2\ :sup:`20` features from 37 s to
  under 0.1 s; dense ``size=1`` pull-plus-update at p=1000 about 10x (#269)
- ``UpperConfidenceBound``, ``EXP3A`` and ``EpsilonGreedy`` draw through
  the marginal path, which is exact for the per-arm statistics they
  consume; ``ThompsonSampling`` is unchanged. Joint draws are kept when a
  ``LipschitzContextualAgent`` carries a user-supplied
  ``batch_reward_function``, which may combine arms within a draw (#258,
  #260)
- ``InformationDirectedSampling.select`` is 50-100x faster with identical
  decisions: a dense 96-arm pull at ``samples=1000`` and 64 contexts from
  4.4 s to 75 ms (#270, #273)
- Draws are written into their output buffers rather than copied there:
  up to 7.8x on sparse joint draws and 3.6x on ``sample_marginal``'s
  arithmetic (#273)
- Dense factors go through ``dpotrf`` so the cached Cholesky stays
  Fortran-ordered under scipy 1.18, whose ``cho_factor`` returns C order
  and made every solve against the cache 4-17x slower (#299)
- Dense ``partial_fit`` hands Fortran views of ``X`` to BLAS instead of
  letting f2py copy them, about 3x on the update at n=5000, p=256 (#297)
- Sparse empirical Bayes ``partial_fit``: symbolic analyses are reused
  across MacKay shifts and stabilized decays, the Takahashi recursion is
  supernodal with a bounded workspace, and each step pays one Cholesky
  rather than two (#276, #277, #278, #283)
- Sparse factors no longer retain the gather pattern only ``refactorize``
  read, 72 MiB per worker at 2\ :sup:`20` hashed features (#290)
- ``sample`` and ``sample_marginal`` no longer copy ``X`` on validation
  (#265)

**Documentation**

- Notebook: :doc:`How do you pick a forgetting strategy?
  </notebooks/forgetting>`, on covariance windup, the directional rules,
  and why a stabilized tick is still needed for change with time (#304)
- Math reference for ``EmpiricalBayesGLM`` (#283) and for feature-wise
  forgetting, including what it gives up (#300)
- Class pages render inherited members, so ``sample_marginal``,
  ``sample_reward_space``, ``decay`` and ``memory_usage`` appear on every
  estimator (#292, #294)
- README: current test matrix and features, fixed the hybrid-bandits
  example (#274)

**Infrastructure**

- Dependency bumps to clear Dependabot alerts: pillow 12.3.0, mistune
  3.3.4, soupsieve 2.9.2 (#257)

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
