Changelog
=========

Unreleased
----------

**Breaking changes**

- ``ContextualAgent`` and ``Agent`` now require each arm to have its own
  learner, and raise from ``add_arm`` (so also from the constructor) when
  two arms share one. Neither agent gives arms a way to differ in
  features, so arms sharing a learner were statistically
  indistinguishable; worse, the policies on that path sample one arm at a
  time, which draws a separate weight vector per arm and silently discards
  the dependence a shared posterior implies. A policy asking for joint
  draws could therefore receive independent ones: on arms sharing a
  learner, measured cross-arm correlation was 0.00 where the true value
  was 1.00. Sharing one model across arms is
  ``LipschitzContextualAgent``'s job -- it distinguishes arms with an
  ``arm_featurizer`` and samples the shared learner once for all of them
  -- and the error says so. Two distinct ``LearnerPipeline`` objects
  wrapping the same estimator also count as sharing (#267)

- The batched arm-sampling path is removed: ``batch_sample_arms``,
  ``can_batch_arms``, ``stack_features``, and the ``LearnerWithTransform``
  protocol (all private). It could only engage for arms sharing a
  ``final_estimator`` while differing in ``transform``, which is the
  configuration the invariant above now rejects -- and no shipped learner
  ever implemented ``final_estimator``, so it never engaged in practice.
  With per-arm learners, drawing one arm at a time is the correct joint
  law, so the remaining path needs no batching (#267)

**New features**

- ``sample_reward_space`` on ``NormalRegressor``,
  ``NormalInverseGammaRegressor``, and ``BayesianGLM``: joint draws from
  the exact posterior predictive, computed by QR-factoring ``half_solve``
  output into a triangular square root of :math:`X \Lambda^{-1} X^T` and
  drawing directly in reward space -- distributionally identical to
  ``sample``, but per-draw cost is independent of the feature count
  (neither :math:`\Lambda^{-1}` nor any covariance is ever formed;
  rank-deficient ``X`` is exact). With ``block_size=k``, consecutive row
  groups are drawn jointly within and independently across groups, which no
  other route can produce, so it stays an explicit request rather than a
  cost decision. Full-mode draws need no request: ``sample`` reduces through
  this route on its own whenever it is cheapest (see below) (#269)

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

- Joint ``sample`` draws now reduce through whichever exact route is
  cheapest. :math:`\operatorname{Cov}(Xw) = X \Lambda^{-1} X^T` has rank at
  most :math:`\min(n_{\text{rows}}, |U|, p)`, and there is a reduction to a
  small square root on each side, so the three routes differ only in how
  many triangular solves against the cached factor they cost:

  .. list-table::
     :header-rows: 1

     * - route
       - solves
       - square root
     * - weight space
       - ``size``
       - :math:`p \times p`, cached
     * - row side
       - ``n_rows``
       - :math:`n_{\text{rows}} \times n_{\text{rows}}`
     * - column side
       - :math:`|U|`
       - :math:`|U| \times |U|`

  The choice is therefore :math:`\min(\text{size}, n_{\text{rows}}, |U|)`,
  three exactly known integers, and the gate holds no calibration
  constants. Weight space is the one whose square root is *cached* -- it
  factors :math:`\Lambda`, which does not depend on ``X`` -- so it builds
  nothing and pays per draw, while the other two refactor on every call.
  That is why neither can win at small ``size``: there is nothing to
  amortize. ``size = 1`` therefore always stays on weight space, and
  Thompson sampling is bit-for-bit unchanged.

  Previously the column side was chosen inside ``sample`` and the row side
  out at the agent behind a policy flag, so neither could account for what
  the other would have picked. Measured end-to-end over 36 dense shapes
  (:math:`p` in {100, 1000}, ``n_rows`` in {1, 10, 32, 96, 320}, ``size``
  in {1, 100, 500, 1000}): no regressions, and speedups to 45.9x. Sparse
  gains reach 860x (:math:`p` = 100,000, one row, ``size`` = 1000: 6.0 s to
  7.0 ms) (#269)

- Every step of the reward-space path after the half-solve is taken from
  ``scipy.linalg`` rather than ``numpy`` (``dgeqrf`` for the QR, ``dgemm``
  for the draws, ``dgemv`` for the predictive mean). ``numpy`` and
  ``scipy`` bind separate copies of OpenBLAS, each with its own thread
  pool, and alternating between them within one call parks and unparks
  both. On a 28-core box that cost more than every flop on the path: a
  100x100 QR took 81 ms through ``numpy.linalg.qr`` and 0.9 ms through
  ``dgeqrf``. Reward-space sampling is up to 102x faster than before this
  routing (#269)

- ``UpperConfidenceBound``, ``EXP3A``, and ``EpsilonGreedy`` now draw
  through the marginal path (they consume only per-arm, per-context
  statistics, for which marginal draws are exact), giving large speedups
  for their Monte Carlo estimates, dense and sparse alike.
  ``ThompsonSampling`` and joint ``sample`` are unchanged, byte-for-byte.
  Custom policies can opt in by setting ``marginal_ok = True`` (declared on
  ``PolicyProtocol``), and subclasses of the built-in policies can opt out
  with ``marginal_ok = False``, which their ``__call__`` and the agents both
  honor. The marginal path is never used
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

- The column-side reduction above. Writing :math:`U` for the columns a
  sparse design matrix touches, :math:`X = X_U E_U^T` exactly, so
  :math:`\operatorname{Cov}(Xw) = X_U (\Lambda^{-1})_{U,U} X_U^T`: the whole
  predictive law follows from one :math:`|U| \times |U|` block, obtained
  with :math:`|U|` solves against the cached factor. Every draw after that
  is dense linear algebra on that small block, never touching the factor
  again. The reformulation is exact, and the cost carries no dependence on
  the factor's elimination tree. Dense models never take it, since there
  :math:`|U|` is the full feature count and there is nothing to reduce. On a
  production agent (:math:`2^{20}` features, 96 arms over a 49-column
  support) a ``size=500`` joint draw goes from 31.5 s to 0.99 s, and
  additionally drops the ``size``-proportional weight-space allocation,
  which at that width exhausted memory beyond roughly 2,600 draws. It also
  serves ``sample_marginal``, where it is taken when :math:`|U|` beats the
  row count (#269)

- ``sample`` and ``sample_marginal`` no longer copy ``X`` while validating
  it. Neither mutates the validated array nor retains a reference to it, so
  the copy was pure overhead, costing O(nnz(X)) per draw on sparse models.
  ``fit``, ``partial_fit``, and ``predict`` are unchanged (#265)

- ``SuperLUSparseFactor.solve`` no longer refactorizes the precision on
  every call for a dense right-hand side, going through the cached
  triangular factor instead. Sparse right-hand sides are unchanged. This
  also settles a discrepancy between the backends: SuperLU returned a
  1-D result for a single-column 2-D input, where CHOLMOD preserved the
  column; both now preserve it (#269)

**Behavioral changes**

- Seeded agent trajectories under ``UpperConfidenceBound``, ``EXP3A``, and
  ``EpsilonGreedy`` change: the marginal path consumes different amounts of
  randomness than joint sampling. Per-row marginals are identical (verified
  by KS tests) and decisions converge to the same choices as ``samples``
  grows, but for arms sharing one model the per-arm Monte Carlo estimates
  no longer share weight draws, so finite-sample selection noise among
  near-tied arms increases; raise ``samples`` to compensate (marginal draws
  are much cheaper per draw), or opt the policy out with
  ``marginal_ok = False`` (#258)

- Seeded ``sample`` trajectories change wherever a reduction applies, since
  the row and column routes consume ``n_rows`` and :math:`|U|` normals per
  draw rather than one per feature. The draws are exact and jointly
  distributed either way, and agreement with the weight-space path is
  verified by KS tests per row. Thompson sampling is unaffected: at
  ``size = 1`` no reduction can win, so none is taken and the output is
  bit-for-bit identical (enforced by a test) (#263, #266)

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
