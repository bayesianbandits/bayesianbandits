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

- ``sample_reward_space`` on ``NormalRegressor``,
  ``NormalInverseGammaRegressor``, and ``BayesianGLM``: joint draws from
  the exact posterior predictive, factored in reward space (one triangular
  half-solve per prediction row against the cached precision factor, then
  a QR), so per-draw cost is independent of the feature count.
  Distributionally identical to ``sample``; with ``block_size=k``,
  consecutive groups of ``k`` rows are drawn jointly within and
  independently across groups (#269)

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

- Dense sampling no longer forms an explicit :math:`U^{-1}`. The precision
  factor is rebuilt after every ``partial_fit``, so materializing the
  inverse charged an :math:`O(d^3)` triangular inversion to every update
  -- 13x to 17x the Cholesky that preceded it, and the largest single term
  in a pull-and-update round -- to save nothing, since solving against
  ``U`` costs the same per draw as multiplying by the inverse.
  ``DenseFactor.colorize`` now calls ``dtrsm``, which also keeps the draw
  inside scipy's BLAS pool. A dense ``size = 1`` pull plus update at
  :math:`d` = 1,000 falls from 106 ms to 11 ms. ``trace_inv`` genuinely
  needs the inverse and still builds it lazily, now through ``dtrtri``
  rather than a solve against a dense identity (#269)

- Joint ``sample`` draws reduce through whichever exact route is cheapest.
  :math:`\operatorname{Cov}(Xw) = X \Lambda^{-1} X^T` has a square root
  on each side: weight space costs ``size`` solves against the cached
  factor, the row side ``n_rows`` solves and an ``n_rows x n_rows`` QR,
  and, for sparse ``X`` touching ``|U|`` columns, the column side ``|U|``
  solves and the ``|U| x |U|`` block of :math:`\Lambda^{-1}` (an identity,
  since :math:`X = X_U E_U^T`). The choice is
  ``min(size, n_rows, |U|)``, three known integers with no calibration
  constants; at ``size = 1`` nothing beats weight space, so Thompson
  sampling still draws there (the entry below changes how cheaply that
  route runs, not which route is chosen). Over 36 dense shapes (:math:`p` in
  {100, 1000}, ``n_rows`` up to 320, ``size`` up to 1000): no regressions,
  speedups to 45.9x. Sparse: :math:`p` = 100,000, one row, ``size`` = 1000
  goes from 6.0 s to 7.0 ms; a :math:`2^{20}`-feature agent with 96 arms
  over a 49-column support draws ``size=500`` in 0.99 s instead of 31.5 s,
  without the ``size``-proportional weight-space allocation that ran out
  of memory beyond ~2,600 draws. The column side also serves
  ``sample_marginal`` when :math:`|U|` beats the row count (#269)

  Those dense figures were measured before the BLAS thread-pool fixes
  below, which cut the weight-space side by up to 8.9x and so narrowed
  the row side's dense margin; the swept shapes stay wins, but past them
  the ``2 n_rows <= size`` guard is now slightly loose. At
  :math:`p` = 1,000 with 448 rows and ``size`` = 900 the row side is
  taken and runs 0.76x to 0.81x, the crossover there having moved from
  ``2 n_rows`` to roughly ``4 n_rows``. The guard is left alone
  deliberately: tightening it would need a fitted constant, and dense
  cost models are dependent on a BLAS threading configuration the
  library cannot observe.

- Sparse factors factor only the features some observation has touched.
  ``Λ = αI + Σ x_t x_tᵀ`` puts an off-diagonal entry between two
  features only when one observation touched both, so a feature no
  observation has touched holds nothing but its diagonal -- exactly --
  and ``Λ`` is block diagonal between those features and the rest.
  Their posterior is an independent scalar with variance ``1/Λ_jj``.
  ``create_sparse_factor`` now detects them from the stored pattern
  (both sides: the column's own length and how many entries anywhere
  reference its row, so triangular storage cannot fool it) and hands
  CHOLMOD or SuperLU only the observed block; every factor operation
  scatters its result back across the split. Nothing above the factor
  changes, and when nothing is trivial the block is the whole matrix,
  as before. On a production model hashing into :math:`2^{20}` features
  of which 26,119 had been observed (97.5% trivial), factorization went
  from 2.9 s to 0.39 s and a 32-row ``partial_fit`` from 3.1 s to
  0.43 s; ``sample(size=1)`` over 96 arms from 27.7 ms to 13.8 ms,
  ``sample(size=8)`` 253 ms to 97 ms, ``sample_reward_space(size=1)``
  8.0 s to 2.3 s. The posterior is unchanged to rounding: the support
  covariance over 49 production features agrees to 9e-16 with the
  unpartitioned factor (#269)

- The precision factors draw their own weight-space normals.
  ``sample_at(features, size, rng)`` replaces ``colorize`` and
  ``colorize_at`` on every factor, dense and sparse: zero-mean draws of
  ``w ~ N(0, Λ⁻¹)`` at the features asked for, and the factor decides how
  many normals that takes. A partitioned sparse factor draws one per
  block feature plus one per *distinct* never-observed feature in the
  query, never one per never-observed feature it was not asked about;
  a dense factor, or a sparse one with nothing trivial, draws them all,
  bitwise as before. A repeated feature repeats its draw, and callers
  read the result through the design matrix, so rows touching one
  feature share it: the joint law of ``Xw``, enforced by the interface
  rather than by care. Every ``sample`` now falls back to one
  weight-space path for dense and sparse ``X`` alike, and the internal
  ``multivariate_normal_sample_from_precision``,
  ``multivariate_t_sample_from_precision`` and the
  ``..._from_sparse_precision`` alias are removed with the branches
  they served. On the production model above, ``sample(size=1)`` over
  96 arms goes from 13.8 ms to 3.2 ms and ``sample(size=8)`` from
  97 ms to 8.3 ms; end to end from the branch tip, ``sample(size=1)``
  is 28.1 ms to 3.2 ms (#269)

- The support-covariance route hands the factor a sparse right-hand
  side, and the sparse factors solve a sparse right-hand side at their
  block rows only. ``E_U`` is ``|U|`` ones; building it as a dense
  ``(n_features, k)`` block, and scattering a dense ``(n_features, k)``
  result back to read ``|U|`` rows of it, cost the route
  ``O(n_features · |U|)`` however small the observed block -- so once
  weight-space solves ran on the block, the route was mispriced by the
  block-to-matrix ratio wherever its gate fired. Neither backend is
  handed the sparse operand itself: its entries are classified in
  ``O(nnz log n)``, the observed rows densified (``m x k``) for one
  BLAS-3 solve, and the result assembled straight into CSC. That also
  closes a correctness hole -- CHOLMOD's own sparse solve reads the
  operand's index arrays with the factor's integer type, and an
  int64-indexed operand (what scipy returns from row-indexing a CSC)
  against an int32 factor gives the wrong answer silently -- and gives
  SuperLU a real sparse-RHS solve, where it re-factorized the whole
  matrix through ``spsolve`` on every call. On the production model:
  ``support_covariance`` over 49 features 398 ms to 71 ms, over 489
  features 4.0 s to 0.7 s; ``sample(size=500)`` over one context of 96
  arms 854 ms to 87 ms and over ten stacked contexts 8.5 s to 0.8 s
  (#269)

- ``half_solve`` takes a sparse right-hand side and returns it compact,
  and the marginal, reward-space and row-side joint paths hand it one.
  Every caller reads a half-solve only through the Gram ``Bᵀ B``, and
  for a sparse operand the rows at never-observed features it does not
  touch are exactly zero -- so a partitioned factor now returns the
  block rows plus one row per distinct never-observed feature touched,
  ``(n_factored + t, k)``, with the same Gram, and never forms the
  ``(n_features, k)`` operand or result those paths used to densify per
  block: ``Θ(n_rows · n_features)`` on the marginal path became
  ``Θ(n_rows · n_factored)``. ``PrecisionFactor.n_factored`` -- the
  features the factor actually factors, all of them for a dense factor
  -- is the unit every scratch bound and route gate on those paths now
  uses in place of ``n_features``, so at :math:`2^{20}` features with 26k
  observed the marginal path takes 80 rows per block instead of 2, and
  the gates price the per-row path at what it costs. On the production
  model, ``sample_marginal(size=1)`` over 96 arms goes from 1.7 s to
  90 ms and over ten stacked contexts from 17.3 s to 0.97 s;
  ``sample_reward_space(size=1)`` over 96 arms from 8.3 s to 0.13 s
  (#269)

- Scaling is a field on every precision factor, not a wrapper.
  ``ScaledSparseFactor`` is removed; ``CholmodSparseFactor``,
  ``SuperLUSparseFactor`` and ``DenseFactor`` carry ``_scale``, and
  ``scale_factor`` returns a shallow copy with the scalar composed in,
  for any of the three. ``decay`` scales the cached factor on dense
  models too, where it used to discard it and refactorize at the next
  pull -- :math:`O(d^3)` per decay -- and the Normal-Inverse-Gamma
  ``shape_`` composes ``a/b`` into the cached factor instead of building
  a scaled copy of the dense triangle by hand; a dense ``decay`` plus
  pull at :math:`d` = 1,000 goes from 14.7 ms to 1.9 ms. Dense NIG draws
  now differ from before at rounding (the scale rides in ``dtrsm``'s
  ``alpha`` rather than in a rescaled triangle); everything else on the
  dense side is bitwise. ``refactorize`` yields a factor of the new
  matrix itself, scale reset, and invalidates any scaled views of the
  same factorization, as the wrapper's did (#269)

- A sparse weight-space ``sample`` never materializes the weight vector.
  The draw :math:`w = \hat{w} + Mz` is read only through ``X``, so the
  passes that followed the triangular solve (undoing the factor's
  fill-reducing permutation, a scaled factor's rescale, adding
  :math:`\hat{w}`, and the CSC product's own sweep over all :math:`p`
  columns) spent :math:`O(\text{size} \cdot p)` on entries no prediction
  row touches. They now run on the :math:`|U|` columns ``X`` does touch,
  leaving one :math:`O(p)` scan for the support and
  :math:`O(\mathrm{nnz}(X))` arithmetic; only the normals and the solve
  stay proportional to :math:`p`, and neither is avoidable in weight
  space. At :math:`p = 2^{20}` with 96 arms over a 49-column support,
  ``NormalInverseGammaRegressor.sample`` at ``size = 1`` falls from
  27.9 ms to 22.7 ms, of which 20.4 ms is now the normals plus the one
  solve. The same reduction serves ``NormalRegressor`` and
  ``BayesianGLM``, and the whole weight-space regime rather than
  ``size = 1`` alone (1.33x at ``size = 8``). Results agree with the
  previous path to 3e-16 rather than bitwise, since the per-draw scalars
  are applied once to the reduced draw instead of elementwise to
  :math:`p` entries. A dense ``X`` against a sparse model reads every
  column and is left on the old path, unchanged (#269)

- The reward-space, support-covariance, weight-space, and dense sampling
  paths are ``scipy.linalg`` throughout (``dgeqrf``, ``dgemm``,
  ``dgemv``, ``dtrsm``, ``dpotrf``). ``numpy`` and ``scipy`` bind
  separate copies of OpenBLAS with separate thread pools, and
  alternating between them within one call parks and unparks both: on a
  28-core box a 100x100 QR took 81 ms through ``numpy.linalg.qr`` and
  0.9 ms through ``dgeqrf``. The last two crossings to go were the
  weight-space tail ``draws @ X.T``, which followed the triangular solve
  that produced ``draws``, and the predictive mean ``X @ coef`` ahead of
  the reduced draw. Routing both through ``dgemm``/``dgemv`` took a
  dense ``sample`` at :math:`d` = 1,000 with 320 rows and ``size`` = 100
  from 40.0 ms to 4.5 ms. Operands are passed in whichever orientation
  they already hold, since a C-contiguous array's transpose is
  Fortran-contiguous and the naive spelling would copy both (#269)

- ``SuperLUSparseFactor.solve`` goes through the cached triangular factor
  for a dense right-hand side instead of refactorizing on every call, and
  now preserves the column of a single-column 2-D input, as CHOLMOD does
  (#269)

- Every dense right-hand side handed to a sparse factor is now
  Fortran-ordered. CHOLMOD's dense format is column-major, so a C-ordered
  block is copied before the solve begins -- and on the
  support-covariance route that block is ``(n_features, |U|)``, large
  enough that the copy outweighed the solve. At :math:`p` = 100,000 with
  96 rows over a 49-column support, ``sample`` at ``size`` = 100 goes
  from 19.5 ms to 9.8 ms and ``sample_marginal`` at ``size`` = 10 from
  19.8 ms to 9.6 ms (#269)

- The sparse factors no longer precompute state only ``colorize`` needs,
  which every ``fit``/``partial_fit`` was paying for and neither uses.
  Sparse ``partial_fit`` at :math:`p` = 100,000 falls from 13.0 ms to
  10.0 ms under CHOLMOD (#269)

- ``CholmodSparseFactor`` inverts the fill-reducing permutation by
  scattering rather than sorting, :math:`O(p)` instead of
  :math:`O(p \log p)`. It is rebuilt with the factor, so a Thompson pull
  pays it every round: a ``sample(size=1)`` and ``partial_fit`` round at
  :math:`p` = 100,000 falls from 11.9 ms to 11.1 ms (#269)

- ``SuperLUSparseFactor`` retains its decomposition, so ``solve`` runs
  both triangular sweeps in one ``gstrs`` call instead of two trips
  through ``spsolve_triangular``; the symmetric ``L`` that sampling needs
  is derived from it on demand. At :math:`p` = 100,000, ``sample`` and
  ``sample_marginal`` over a 49-column support run 4.3x faster and
  ``partial_fit`` 1.8x. The factor holds ``L`` and ``U`` rather than one
  folded ``L``, costing roughly one extra ``nnz(L)`` when sampling (#269)

- ``UpperConfidenceBound``, ``EXP3A``, and ``EpsilonGreedy`` now draw
  through the marginal path (they consume only per-arm, per-context
  statistics, for which marginal draws are exact), giving large speedups
  for their Monte Carlo estimates, dense and sparse alike.
  ``ThompsonSampling`` is unchanged, byte-for-byte.
  Custom policies can opt in by setting ``marginal_ok = True`` (declared
  on ``PolicyProtocol``), and subclasses of the built-in policies can
  opt out with ``marginal_ok = False``, which their ``__call__`` and the
  agents both honor. The marginal path is never used for a learner whose
  class overrides ``sample`` without also overriding ``sample_marginal``,
  so customized joint sampling is not silently bypassed (#258)

- The marginal path is likewise never used when a
  ``LipschitzContextualAgent`` carries a user-supplied
  ``batch_reward_function``. That function sees a whole draw at once and
  may combine arms within it (share of total, cannibalization, softmax
  over a slate), which requires the arms of a draw to be jointly
  distributed; iid marginal draws leave such a reward's mean intact but
  manufacture spread that a quantile-based policy reads as uncertainty.
  A batch function that maps each ``(arm, context, draw)`` cell
  independently can opt back into the faster path by carrying a truthy
  ``elementwise`` attribute. Per-arm ``Arm.reward_function``s are applied
  one arm at a time and never affect sampling, so the common cases --
  no reward function, or per-arm functions only -- keep the speedup (#258)

- ``sample`` and ``sample_marginal`` no longer copy ``X`` while validating
  it. Neither mutates the validated array nor retains a reference to it,
  so the copy was pure overhead, costing O(nnz(X)) per draw on sparse
  models. ``fit`` and ``partial_fit`` are unchanged (#265)

- ``sample_reward_space``, ``predict``, and ``predict_proba`` likewise no
  longer copy ``X``: they only read it (#269)

**Behavioral changes**

- Seeded agent trajectories under ``UpperConfidenceBound``, ``EXP3A``, and
  ``EpsilonGreedy`` change: the marginal path consumes different amounts
  of randomness than joint sampling. Per-row marginals are identical
  (verified by KS tests) and decisions converge to the same choices as
  ``samples`` grows, but for arms sharing one model the per-arm Monte
  Carlo estimates no longer share weight draws, so finite-sample
  selection noise among near-tied arms increases; raise ``samples`` to
  compensate (marginal draws are much cheaper per draw), or opt the
  policy out with ``marginal_ok = False`` (#258)

- Seeded ``sample`` trajectories change wherever a reduction applies: the
  row and column routes consume ``n_rows`` and :math:`|U|` normals per
  draw rather than one per feature. The draws are exact and jointly
  distributed either way (verified by per-row KS tests). Thompson sampling
  (``size = 1``) never reduces, and on sparse models is bit-for-bit
  unchanged (#269)

- Dense seeded draws change in the last bits, Thompson sampling included:
  ``colorize`` solves against ``U`` instead of multiplying by an explicit
  ``U^{-1}``, the same operation in exact arithmetic but a different
  rounding order (the two agree to ~2e-16). No distribution changes (#269)

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
