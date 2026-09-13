Changelog
=========

Unreleased
----------

**New features**

- The forgetting rules are public, ``ExponentialForgetting``,
  ``StabilizedForgetting``, ``FeatureWiseForgetting`` and
  ``SiftForgetting``, and each carries its own ``rate``. The uniform two
  have a ``tick`` for the clock, and every rule has an ``update`` for a
  batch. ``decay(rule, steps=)`` accepts a uniform rule on every
  estimator, arm, agent and pipeline, so stabilized forgetting (Kulhavy &
  Zarrop) is available to the plain estimators, with the floor at their
  ``alpha``; the empirical Bayes estimators default to it, as before.
  ``decay_rate=`` is shorthand for the estimator's default rule at that
  rate. Passing a directional rule to ``decay`` is a ``TypeError``
  pointing at where it belongs: the learner's update.

- ``forgetting=`` on every estimator: the rule applied on each
  ``partial_fit`` before the batch is absorbed, for change that happens
  because you observed. The linear estimators take any rule; the
  directional ones, ``FeatureWiseForgetting`` and ``SiftForgetting``,
  forget only along what the batch excites and leave everything else at
  full precision. ``SiftForgetting`` is refused on a sparse estimator,
  since its correction fills in the precision; the grouped conjugate
  models and the empirical Bayes estimators take the uniform rules only.
  Combined with ``decay()`` on a schedule this is the TrueSkill 2 model
  of change: some from each observation, some from the passage of time.

**Breaking changes**

- ``learning_rate`` is removed from every estimator. What it did,
  forgetting on each ``partial_fit`` before the batch is absorbed, is now
  ``forgetting=`` taking a rule that carries its rate. To migrate:

  - ``NormalRegressor(..., learning_rate=0.99)`` and the other plain
    estimators: ``forgetting=ExponentialForgetting(0.99)``.
  - ``EmpiricalBayesNormalRegressor(..., learning_rate=0.99)`` and the
    other empirical Bayes estimators, which always forgot with the prior
    floored: ``forgetting=StabilizedForgetting(0.99)``.
  - ``learning_rate=1.0``: drop it; ``forgetting=None`` is the default.

  The numbers are unchanged: a uniform rule takes one step per row, so
  a batch of ``n`` rows scales the prior by ``rate ** n`` and weighs its
  older rows less, as before. Models pickled with ``learning_rate`` load
  unchanged and are converted on load, to the rule the class used to
  apply. ``decay()`` no longer falls back to a constructor rate: with
  neither a rule nor ``decay_rate`` it raises. The
  ``PosteriorApproximator`` protocol takes ``prior_decay`` (the factor
  already raised to the batch size) instead of ``learning_rate``, with
  ``sample_weight`` arriving as the final per-row weights.

- ``decay`` has one signature everywhere,
  ``decay(forgetting=None, *, decay_rate=None, steps=1)``, and the
  ``Learner`` protocol requires it. It no longer takes a context array:
  the array was only ever read for its row count (and, on the grouped
  conjugate models, for which groups to tick), so pass ``steps=`` for the
  number of ticks and let a tick reach every group. ``decay_rate`` is keyword-only on the
  agents and pipelines, where it used to be positional, and
  ``Agent.decay(0.9)`` is a ``TypeError`` that says to pass
  ``decay_rate=``. A learner of your own with the old
  ``decay(X, *, decay_rate=None)`` signature no longer works inside an
  ``Arm``: arms call ``decay(forgetting, decay_rate=..., steps=...)``.
  Internally, the estimators' private ``_apply_decay`` hook is replaced
  by ``_apply_tick(rule, steps)``, the EB ``_reinject_prior`` helper is
  gone, and the rule objects in ``_forgetting.py`` are built with a
  ``rate`` and applied through ``update``/``tick`` rather than called.

- ``NonContextualAgentPipeline`` is removed, and ``AgentPipeline`` is now
  a class rather than a factory dispatching between it and
  ``ContextualAgentPipeline``. A pipeline's steps transform the context on
  its way to the agent; a non-contextual ``Agent`` has no context, so the
  steps never ran. The class validated its ``steps``, stored them, and
  exposed them through ``named_steps``, ``__len__``, and ``__getitem__``
  without ever applying them. ``AgentPipeline`` now raises ``TypeError``
  on an ``Agent``, pointing at ``LearnerPipeline`` for preprocessing the
  arms' features. Code wrapping a plain ``Agent`` should drop the wrapper
  and use the agent, which has the same ``pull``/``update``/``decay``
  interface. ``ContextualAgentPipeline`` remains as an alias for
  ``AgentPipeline``, so the two are now one class under two names, but
  ``repr`` reports ``AgentPipeline`` (#291)

- ``EmpiricalBayesNormalRegressor`` and ``EmpiricalBayesGLM`` regularize
  the MacKay update of ``alpha`` with a Gamma hyperprior at the
  constructor's ``alpha``, weighted by the new ``alpha_prior_strength``
  (default ``0.2`` pseudo-observations): ``(γ + k) / (‖θ‖² + k/α₀)``. Plain
  MacKay ran ``alpha`` to the guardrail on arms with near-zero coefficients
  and locked them there. Learned alphas change under the default and are
  bounded by ``(γ + k)·α₀/k``; ``0.0`` restores the previous update, and
  ``log_evidence_`` includes the log hyperprior when the strength is
  positive (#286)

- ``PolicyProtocol`` gains a ``consumes: DrawKind`` attribute, naming the
  weakest draws a policy can correctly consume over a totally ordered
  lattice::

      MARGINAL_ONLY  <  CONTEXT_JOINT  <  JOINT

  Agents satisfy it with anything at least that strong and pick whichever
  is cheapest (see ``DrawKind`` for the semantics). Custom, structurally
  typed policies should declare a ``consumes`` attribute;
  ``PolicyDefaultUpdate`` and the agents default it to ``JOINT``, which
  is always correct and never the cheapest.

  ``CONTEXT_JOINT`` is new: per-context reward-space blocks are joint
  across the arms of one context and independent across contexts, which
  is strictly between marginal and fully joint draws. It is what
  ``InformationDirectedSampling`` needs (#270)

- ``ContextualAgent`` and ``Agent`` now require each arm to have its own
  learner, and raise from ``add_arm`` (so also from the constructor) when
  two arms share one. Neither agent gives arms a way to differ in
  features, so arms sharing a learner were statistically
  indistinguishable; worse, the policies on that path sample one arm at a
  time, which draws a separate weight vector per arm and silently
  discards the dependence a shared posterior implies. A policy requiring
  joint draws could therefore ask for them and receive independent ones:
  on arms sharing a learner, measured cross-arm correlation was 0.00
  where the true value was 1.00. Sharing one model
  across arms is ``LipschitzContextualAgent``'s job -- it distinguishes
  arms with an ``arm_featurizer`` and samples the shared learner once for
  all of them -- and the error says so. Two distinct ``LearnerPipeline``
  objects wrapping the same estimator also count as sharing (#267)

- ``mackay_update_nig`` and ``mackay_update_glm`` are removed (private,
  never exported or documented). They were evidence-maximization
  updates written ahead of the empirical-Bayes NIG and GLM estimators
  that would call them; neither shipped, so nothing in the package has
  ever used them. The Gamma sibling that did ship,
  ``negbin_update_gamma_poisson``, stays. Should those estimators be
  built, both functions and their tests are recoverable from history
  (#273)

- ``multivariate_t_sample_from_covariance`` is removed (private, never
  exported or documented). It reimplemented
  ``scipy.stats.multivariate_t.rvs`` against a ``Covariance`` object,
  and nothing has called it since joint draws moved to the reduction
  routes; the tests that used it as an oracle now compare against
  ``scipy.stats.multivariate_t`` directly. Its removal also drops this
  package's only import of a private scipy internal
  (``scipy.stats._multivariate._squeeze_output``) (#273)

- The batched arm-sampling path is removed: ``batch_sample_arms``,
  ``can_batch_arms``, ``stack_features``, and the ``LearnerWithTransform``
  protocol (all private). It could only engage for arms sharing a
  ``final_estimator`` while differing in ``transform``, which is the
  configuration the invariant above now rejects -- and no shipped learner
  ever implemented ``final_estimator``, so it never engaged in practice.
  With per-arm learners, drawing one arm at a time is the correct joint
  law, so the remaining path needs no batching (#267)

**New features**

- ``FeatureWiseForgetting`` joins the forgetting rules in
  ``_forgetting.py``: vector-type forgetting (Saelid & Foss 1983) with the
  per-feature factors set from the batch support, ``Λ̄ = D Λ D`` with
  ``D = diag(γ^{m_i/2})``. It forgets only the observed features, leaves
  every unobserved feature's marginal and every correlation unchanged, and
  preserves the sparsity pattern at exponential-forgetting cost, where the
  SIFt correction is dense on the neighbourhood of the active features.
  Not yet wired into the estimators. The math page documents what it gives
  up: it is a coordinate stretch rather than a Bayesian update and can
  over-tighten combinations with strongly correlated unobserved features
  (#300)

**Internal**

- ``NormalRegressor`` and ``BayesianGLM`` now share a single private base,
  ``_BayesianLinearModel``, replacing the ``_SparseFactorMixin`` and
  ``_RewardSpacePredictiveMixin`` pair. The two estimators had duplicated
  every method that stack existed to share: ``_precision_factor``, ``cov_``
  and ``decay`` were identical, ``fit`` and ``partial_fit`` differed only in
  a local name, and ``predict``, ``sample``, ``sample_marginal`` and
  ``sample_reward_space`` only in a ``_inverse_link`` on the return. The base
  owns all of them and applies the inverse link, which is the identity unless
  a subclass overrides it. ``NormalInverseGammaRegressor.decay`` was likewise
  the shared body plus two lines, and is now an ``_apply_decay`` override.

  Neither mixin was really a mixin: with no base to inherit declarations
  from, they had to restate the attributes they read, which forced
  ``cast(Any, super())`` at the two empirical-Bayes call sites.
  ``_StabilizedPriorMixin`` inherits the new base for the same reason and
  keeps only the EB state it manages.

  No public name, signature or behavior changes; the removed mixins were
  private and unexported. One narrow exception: ``fit``'s first parameter is
  now ``X`` on every estimator. It was ``X_fit`` on ``NormalRegressor`` and
  ``EmpiricalBayesNormalRegressor`` alone, against ``X`` on the other six,
  so a caller passing it by keyword to those two must rename it (#292)

- Class documentation pages now render inherited members. The autosummary
  template asked for ``:members:`` only, so anything defined on a base was
  silently absent: every estimator page would have lost the methods above.
  ``memory_usage`` is documented for the first time on the classes that
  have it, and each policy page gains the inherited ``update`` (#292)

**New features**

- ``PolicyDefaultUpdate`` implements ``__call__``, so a policy subclassing
  it now needs only ``samples_needed`` and ``select``: draw the samples
  the policy asked for, then let it choose. All five shipped policies had
  carried a byte-identical copy of that two-line body under its own pair
  of ``@overload`` stubs. The base also declares ``samples_needed`` and
  ``select``, naming the contract it calls into (#291)

- ``EmpiricalBayesGLM``: ``BayesianGLM`` with MacKay evidence-framework
  tuning of the prior precision ``alpha``, applied to the Laplace
  approximation. ``fit`` alternates IRLS with MacKay steps until the
  Laplace evidence settles; ``partial_fit`` takes one MacKay step on
  the posterior in hand and moves it to the new ``alpha``. Stabilized
  forgetting keeps the tuned prior load-bearing under
  ``learning_rate < 1``, as for ``EmpiricalBayesNormalRegressor``.

  ``PosteriorApproximator.update_posterior`` gains two optional keywords
  for it: ``prior_floor`` (the stabilized-forgetting re-injection) and
  ``coef_init`` (a warm start for the EB refits). ``BayesianGLM`` only
  passes them when set, so an existing custom approximator keeps working
  there; one used with ``EmpiricalBayesGLM`` must accept both (#283)

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

- The support-covariance route writes each row block's draws straight
  into the output buffer. The result has to be C-ordered
  ``(n_rows, size)`` for its transpose to satisfy the layout contract,
  and ``dgemm`` builds Fortran-ordered results, so taking each block
  back as a return value made it a transposed copy of itself into the
  buffer: the whole output moved twice, the second time at the ~2 GB/s
  a strided copy runs rather than the ~20 a contiguous one does. Handing
  ``dgemm`` the block's own transpose as its output operand costs
  nothing and removes the copy. Sparse joint draws over many rows gain
  3.4x on the draw itself at 2000 rows by 500 draws (7.8x by 2000) and
  2.5x end to end on ``sample``. Draws move in the last bits, the
  accumulation order being the transposed one's (#273)

- The layout normalization behind every agent's draw tensor
  (``draw_contiguous``) now slabs its copy whenever a slab holds a full
  cache line of draws, where it previously required 64 of them and fell
  back to numpy's strided copy below that. What decides which copy wins
  is the contiguous run each write lays down, not the number of draws
  per slab, and eight float64 draws is already a whole line. The old
  threshold therefore declined exactly where the slab loop wins most:
  a tensor with wide leading axes gets a small slab *because* it is
  wide, so 96 arms by 64 contexts by 1000 draws took the fallback and
  paid 57.6 ms where slabbing costs 21.5. Over a sweep of eleven shapes
  the new threshold picks the faster branch in nine, worst regression
  1.11x on a 0.16 ms array, and the shapes the guard exists for (a slab
  of one, two or four draws) still take numpy's copy. Values are
  unchanged; only in-library callers handing over a non-conforming
  layout are affected, since the agents' own tensors already conform
  and pass through untouched (#273)

- ``sample_marginal`` builds its draws through in-place passes over the
  normals rather than an expression per operator. ``mean + sd * z``
  allocates a ``(size, n_rows)`` temporary per operator and reads each
  back for the next; folding them into ``z``, which nothing else holds,
  measured 3.6x on that arithmetic (5.6 ms to 1.5 at 2000 rows by 500
  draws) and 1.29x end to end, the remainder being the Gaussian RNG
  itself. The ``NormalInverseGammaRegressor`` also folds its
  multivariate-t scale into the chi-square draws in place, where it
  spent three more temporaries of the same size. Values, random stream
  and layout are all unchanged (#273)

- ``InformationDirectedSampling.select`` reads each arm's win count off
  the tally it already computes rather than reducing the
  ``(n_contexts, M, n_draws)`` weight block a second time, whenever no
  draw is exactly tied. The two agree by construction there: an untied
  weight row sums to exactly its arm's win count. Ties still take the
  reduction, since splitting a tied draw leaves fractional counts.
  Bit-identical decisions, about 7% off the whole of ``select`` at 96
  arms by 64 contexts (#273)

- The precision factors skip their scale division when they carry no
  scale, which is every factor but a decayed one and the
  Normal-Inverse-Gamma shape. It was a full pass over, and a second
  allocation of, an operand-sized array to divide by one (#273)

- ``InformationDirectedSampling.select`` is 50-100x faster, exactly.
  It normalizes its draw tensor to draw-contiguous layout first (the
  agents' transposed views left the draw axis with the largest stride,
  which alone made the conditional-mean GEMM 1000x slower), conditions
  only on the arms that win at least one draw (every other arm's
  :math:`p(A^* = b)` term is identically zero), recovers the means and
  :math:`\mathbb{E}[\max]` from that GEMM's own entries instead of
  separate passes, and scans only the Pareto frontier of the
  :math:`(v_a, \Delta_a)` points for the optimal pair (moving weight
  from a dominated arm to its dominator never increases the ratio), one
  interior root per unordered frontier pair. No step excludes a possible
  optimum, so the selected mixture is identical. A dense 96-arm pull at
  ``samples=1000`` goes from 612 ms to 13 ms at 8 contexts and 4.4 s to
  75 ms at 64; ``top_k=4`` from 1.50 s to 20 ms (#270)

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
