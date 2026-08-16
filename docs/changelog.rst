Changelog
=========

Unreleased
----------

**Breaking changes**

- Policies declare what they *consume* rather than which sampling methods
  they tolerate. The ``marginal_ok`` and ``reward_space_ok`` flags on
  ``PolicyProtocol`` are replaced by a single ``consumes: DrawKind``,
  naming the weakest draws a policy can correctly consume over a totally
  ordered lattice::

      MARGINAL_ONLY  <  CONTEXT_JOINT  <  JOINT

  Agents satisfy it with anything at least that strong and pick whichever
  is cheapest, so no policy names a sampling method any more. Supplying
  more structure than asked is always sound, so an agent may widen the
  requirement silently and can never narrow it: ``max`` is the only
  combinator, and ``JOINT`` is the default, meaning a policy that does not
  consider this gets correctness rather than speed.

  Migration is mechanical: ``marginal_ok = True`` becomes
  ``consumes = DrawKind.MARGINAL_ONLY``, ``reward_space_ok = True``
  becomes ``consumes = DrawKind.CONTEXT_JOINT``, and both ``False``
  becomes ``consumes = DrawKind.JOINT``.

  The flags could not express ``CONTEXT_JOINT`` at all. Per-context
  reward-space blocks are joint across the arms of one context and
  independent across contexts, which is strictly between marginal and
  fully joint draws; it is what ``InformationDirectedSampling`` needs, and
  it was previously implied by a method name rather than stated. The two
  booleans were also independent over levels that are not, so they could
  be set to combinations with no coherent meaning (#270)

- The ``elementwise`` opt-out on batch reward functions is removed, along
  with ``is_elementwise_batch_reward``. A supplied
  ``batch_reward_function`` now always widens the agent's requirement to
  at least ``CONTEXT_JOINT``. The attribute was an unverifiable promise
  that a function mapped each draw cell independently, and a wrong one
  silently manufactured spread that quantile-based policies read as
  uncertainty. Joint draws are no longer expensive enough to be worth that
  risk, now that they reduce through the cheapest of three exact routes.
  Agents supplying no reward function, or per-arm reward functions only,
  are unaffected: those normalize to ``batch_identity``, which combines
  nothing and does not widen (#270)

- ``ContextualAgent`` and ``Agent`` now require each arm to have its own
  learner, and raise from ``add_arm`` (so also from the constructor) when
  two arms share one. Neither agent gives arms a way to differ in
  features, so arms sharing a learner were statistically
  indistinguishable; worse, the policies on that path sample one arm at a
  time, which draws a separate weight vector per arm and silently
  discards the dependence a shared posterior implies. A policy declaring
  ``marginal_ok = False`` could therefore ask for joint draws and receive
  independent ones: on arms sharing a learner, measured cross-arm
  correlation was 0.00 where the true value was 1.00. Sharing one model
  across arms is ``LipschitzContextualAgent``'s job -- it distinguishes
  arms with an ``arm_featurizer`` and samples the shared learner once for
  all of them -- and the error says so. Two distinct ``LearnerPipeline``
  objects wrapping the same estimator also count as sharing (#267)

- The batched arm-sampling path is removed: ``batch_sample_arms``,
  ``can_batch_arms``, ``stack_features``, and the ``LearnerWithTransform``
  protocol (all private). It could only engage for arms sharing a
  ``final_estimator`` while differing in ``transform``, which is the
  configuration the invariant above now rejects -- and no shipped learner
  ever implemented ``final_estimator``, so it never engaged in practice.
  With per-arm learners, drawing one arm at a time is the correct joint
  law, so the remaining path needs no batching (#267)

**New features**

- ``InformationDirectedSampling``: a variance-based information-directed
  sampling (IDS) policy after Russo & Van Roy (2018). Each round it
  estimates every arm's expected regret and variance-based information
  gain from joint Monte Carlo posterior draws and samples from the
  two-arm distribution minimizing the information ratio
  :math:`\Delta(\pi)^2 / v(\pi)`, exploiting cross-arm correlation under
  shared learners. ``top_k`` returns sequential IDS draws without
  replacement, each slot re-solving the subgame of remaining arms (#270)

- ``sample_reward_space`` on ``NormalRegressor``,
  ``NormalInverseGammaRegressor``, and ``BayesianGLM``: joint draws from
  the exact posterior predictive, factored in reward space so per-draw
  cost is independent of the feature count. Distributionally identical
  to ``sample``; with ``block_size=k``, consecutive groups of ``k`` rows
  are drawn jointly within and independently across groups (#269)

- ``sample_marginal`` on ``NormalRegressor``, ``NormalInverseGammaRegressor``,
  and ``BayesianGLM``: iid draws from each prediction row's exact marginal
  posterior predictive, computed with one triangular half-solve per row
  against the cached precision factor (neither :math:`\Lambda^{-1}` nor any
  :math:`n \times n` matrix is ever formed, and per-draw cost is independent
  of the feature count). ``Arm.sample_marginal`` and
  ``LearnerPipeline.sample_marginal`` forward to it, falling back to joint
  ``sample`` for learners without it (or whose class overrides ``sample``
  without it). Unlike ``sample`` -- whose rows within one draw share a
  weight vector -- draws are independent across rows, so it serves per-row
  statistics only (#258)

**Performance**

- A number of performance updates to posterior sampling on
  ``NormalRegressor``, ``NormalInverseGammaRegressor`` and ``BayesianGLM``,
  none of which changes a distribution. Joint ``sample`` draws go through
  the cheapest exact route for the call: weight space, the row-side
  factor behind ``sample_reward_space``, or, for a sparse ``X``, a
  covariance over the columns it touches. Sparse precision factors
  factor only the features some observation has touched and carry the
  rest as a diagonal, so factorization, updates and every sampling
  path are sized by that block rather than by ``n_features``, and
  memory no longer grows with the never-observed features a query
  touches. The dense factor no longer materializes :math:`U^{-1}` on
  every update, scaling is a field on every factor rather than a
  wrapper, and the sampling paths keep to one BLAS thread pool and stop
  copying ``X`` and their BLAS operands. Expected gains: on a sparse
  model hashing into :math:`2^{20}` features of which ~26k had been
  observed, factorization and a 32-row ``partial_fit`` about 5x faster,
  Thompson ``sample`` over 96 arms about 4x (26 ms to 7 ms),
  ``sample(size=8)`` about 30x, ``sample(size=500)`` from 37 s to under
  0.1 s, and ``sample_marginal(size=500)`` about 20x (2.3 s to 0.1 s
  over 96 arms, 22 s to 1 s over 960 rows); on dense models, ``size = 1``
  pull-plus-update at :math:`d` = 1,000 about 10x, and large-``size``
  joint draws up to ~45x on the shapes swept (#269)

- ``UpperConfidenceBound``, ``EXP3A``, and ``EpsilonGreedy`` now draw
  through the marginal path (they consume only per-arm, per-context
  statistics, for which marginal draws are exact), giving large speedups
  for their Monte Carlo estimates, dense and sparse alike.
  ``ThompsonSampling`` is unchanged, byte-for-byte. Custom policies opt in
  by declaring ``consumes = DrawKind.MARGINAL_ONLY``, and subclasses of the
  built-in policies opt out with ``consumes = DrawKind.JOINT``, which their
  ``__call__`` and the agents both honor. The marginal path is never used
  for a learner whose class overrides ``sample`` without also overriding
  ``sample_marginal``, so customized joint sampling is not silently
  bypassed (#258)

- The marginal path is likewise never used when a
  ``LipschitzContextualAgent`` carries a user-supplied
  ``batch_reward_function``. That function sees a whole draw at once and
  may combine arms within it (share of total, cannibalization, softmax
  over a slate), which requires the arms of a draw to be jointly
  distributed; iid marginal draws leave such a reward's mean intact but
  manufacture spread that a quantile-based policy reads as uncertainty. A
  per-arm ``Arm.reward_function`` is applied one arm at a time and never
  affects sampling, so the common cases -- no reward function, or per-arm
  functions only -- keep the speedup (#258)

- ``sample`` and ``sample_marginal`` no longer copy ``X`` while validating
  it. Neither mutates the validated array nor retains a reference to it,
  so the copy was pure overhead, costing O(nnz(X)) per draw on sparse
  models. ``fit`` and ``partial_fit`` are unchanged (#265)

**Behavioral changes**

- Seeded agent trajectories under ``UpperConfidenceBound``, ``EXP3A``, and
  ``EpsilonGreedy`` change: the marginal path consumes different amounts of
  randomness than joint sampling. Per-row marginals are identical (verified
  by KS tests) and decisions converge to the same choices as ``samples``
  grows, but for arms sharing one model the per-arm Monte Carlo estimates
  no longer share weight draws, so finite-sample selection noise among
  near-tied arms increases; raise ``samples`` to compensate (marginal draws
  are much cheaper per draw), or opt the policy out with
  ``consumes = DrawKind.JOINT`` (#258)

- Seeded ``sample`` trajectories change: a reduced draw consumes
  ``n_rows`` or :math:`|U|` normals rather than one per feature, a
  partitioned sparse factor draws one normal per observed feature plus
  one per distinct never-observed feature the query touches, and dense
  draws solve against ``U`` instead of multiplying by :math:`U^{-1}`,
  which differs in the last bits. Every distribution is unchanged
  (verified by per-row KS tests); Thompson sampling (``size = 1``) on a
  sparse model whose every feature has been observed is bit-for-bit
  unchanged (#269)

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
