Changelog
=========

Unreleased
----------

**New features**

- ``InformationDirectedSampling``: a variance-based information-directed
  sampling (IDS) policy after Russo & Van Roy (2018). Each round it
  estimates every arm's expected regret and variance-based information
  gain from joint Monte Carlo posterior draws and samples from the
  two-arm distribution minimizing the information ratio
  :math:`\Delta(\pi)^2 / v(\pi)`, exploiting cross-arm correlation under
  shared learners. ``top_k`` returns sequential IDS draws without
  replacement, each slot re-solving the subgame of remaining arms (#240)

- ``sample_reward_space`` on ``NormalRegressor``,
  ``NormalInverseGammaRegressor``, and ``BayesianGLM``: joint draws from
  the exact posterior predictive, computed by QR-factoring
  ``half_solve`` output into a triangular square root of
  :math:`X \Lambda^{-1} X^T` and drawing directly in reward space --
  distributionally identical to ``sample``, but per-draw cost is
  independent of the feature count (neither :math:`\Lambda^{-1}` nor any
  covariance is ever formed; rank-deficient ``X`` is exact). With
  ``block_size=k``, consecutive row groups are drawn jointly within and
  independently across groups via one batched QR -- the shape
  per-context policies need. ``InformationDirectedSampling`` opts in via
  ``reward_space_ok`` (declared on ``PolicyProtocol``), routed through
  ``batch_sample_arms(..., reward_space=True)``,
  ``LearnerPipeline.sample_reward_space``, and
  ``LipschitzContextualAgent``, and gated on the one cost both paths pay
  in the same currency: triangular solves against the cached factor,
  ``n`` for reward space against ``size`` for weight space. Both are
  exactly known integers, so the gate holds no calibration constants.
  ``sample`` itself is unchanged, byte-for-byte, and Thompson sampling's
  ``size=1`` hot path never routes here (#240)

  Every step after the half-solve is taken from ``scipy.linalg`` rather
  than ``numpy`` (``dgeqrf`` for the QR, ``dgemm`` for the draws,
  ``dgemv`` for the predictive mean). ``numpy`` and ``scipy`` bind
  separate copies of OpenBLAS, each with its own thread pool, and
  alternating between them within a call parks and unparks both. On a
  28-core box that cost more than every flop on the path: a 100x100 QR
  took 81 ms through ``numpy.linalg.qr`` and 0.9 ms through ``dgeqrf``.
  Reward-space sampling is up to 102x faster than it was before this
  routing, and Thompson sampling's ``size=1`` path is unaffected either
  way (#263)

- ``sample_marginal`` on ``NormalRegressor``, ``NormalInverseGammaRegressor``,
  and ``BayesianGLM``: iid draws from each prediction row's exact marginal
  posterior predictive, computed with one triangular half-solve per row
  against the cached precision factor (neither :math:`\Lambda^{-1}` nor any
  :math:`n \times n` matrix is ever formed, and per-draw cost is independent
  of the feature count). ``Arm.sample_marginal``, ``LearnerPipeline.sample_marginal``,
  and ``batch_sample_arms(..., marginal=True)`` forward to it, falling back
  to joint ``sample`` for learners without it (or whose class overrides
  ``sample`` without it). Unlike ``sample`` -- whose
  rows within one draw share a weight vector -- draws are independent
  across rows, so it serves per-row statistics only (#258)

**Performance**

- ``UpperConfidenceBound``, ``EXP3A``, and ``EpsilonGreedy`` now draw
  through the marginal path (they consume only per-arm, per-context
  statistics, for which marginal draws are exact), giving large speedups
  for their Monte Carlo estimates, dense and sparse alike.
  ``ThompsonSampling`` and joint ``sample`` are unchanged, byte-for-byte.
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

**Behavioral changes**

- Seeded agent trajectories under ``UpperConfidenceBound``, ``EXP3A``, and
  ``EpsilonGreedy`` change: the marginal path consumes different amounts
  of randomness than joint sampling. Per-row marginals are identical
  (verified by KS tests) and decisions converge to the same choices as
  ``samples`` grows, but for arms sharing one model the per-arm Monte
  Carlo estimates no longer share weight draws, so finite-sample
  selection noise among near-tied arms increases; raise ``samples`` to
  compensate (marginal draws are much cheaper per draw), or opt the
  policy out with ``marginal_ok = False``.
  ``sample`` itself is bit-for-bit identical to previous versions (#258)

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
