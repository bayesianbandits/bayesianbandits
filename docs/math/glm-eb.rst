EmpiricalBayesGLM
=================

Automatic tuning of the prior precision :math:`\alpha` via MacKay's
evidence maximization [1]_, applied to the Laplace approximation,
with stabilized forgetting [2]_ to prevent prior collapse under
exponential decay.

Builds on the posterior update from :doc:`glm`, which is inherited
unchanged, and follows :doc:`empirical-bayes` closely. Read both
first; this page covers what changes when the likelihood is not
Gaussian.


Symbols
-------

In addition to the symbols defined in :doc:`glm`:

.. list-table::
   :header-rows: 1
   :widths: 15 85

   * - Symbol
     - Meaning
   * - :math:`\hat{\boldsymbol{\theta}}`
     - MAP estimate (the IRLS mode), stored as ``coef_``
   * - :math:`\mathbf{H}`
     - Data Hessian at the mode,
       :math:`\mathbf{X}^\top\!\mathbf{W}\mathbf{X}`, with
       :math:`\mathbf{W}` the IRLS weights times the effective row
       weights
   * - :math:`s_t`
     - Cumulative (decayed) prior contribution to the precision
       diagonal (stored as ``_prior_scalar``), so
       :math:`\boldsymbol{\Lambda} = s_t \mathbf{I} + \mathbf{H}`
   * - :math:`\gamma_{\text{eff}}`
     - Effective number of well-determined parameters. Distinct from
       the decay factor :math:`\gamma`.
   * - :math:`N_{\text{eff}}`
     - Decayed, weighted effective sample size (``_effective_n``)
   * - :math:`\ell_{\text{eff}}`
     - Decayed, weighted running log-likelihood at the mode
       (``_eff_loglik``)


Differences from the Normal case
--------------------------------

Only :math:`\alpha` is tuned. It plays the same role as in the
Normal model: the prior precision of
:math:`\mathbf{w} \sim \mathcal{N}(\mathbf{0}, \alpha^{-1}\mathbf{I})`,
which sets how strongly the coefficients are shrunk toward zero.
The Normal's second hyperparameter :math:`\beta` is the precision of
the *likelihood*, the scatter of :math:`y` around
:math:`\mathbf{X}\mathbf{w}`, and has no counterpart here: a
Bernoulli or Poisson likelihood has no free scale, its variance is a
fixed function of its mean, so there is no dispersion to estimate
from residuals.

The evidence is not available in closed form. The Laplace
approximation replaces it with the Gaussian evidence of the
second-order expansion around the mode:

.. math::

   \log p(\mathbf{y} \mid \mathbf{X}, \alpha)
   \approx \ell(\hat{\boldsymbol{\theta}})
   + \tfrac{p}{2}\log\alpha
   - \tfrac{\alpha}{2}\,\|\hat{\boldsymbol{\theta}}\|^2
   - \tfrac{1}{2}\log|\boldsymbol{\Lambda}|

where :math:`\ell` is the log-likelihood (Bernoulli, or Poisson
including the :math:`\log y!` normalizer) evaluated at the mode.
Stored as ``log_evidence_``.

Because the mode and the Hessian both depend on :math:`\alpha`, this
is only the evidence *at the current approximation*. The IRLS has to
converge for it to mean anything, which is why the default
approximator is ``LaplaceApproximator(n_iter=25, tol=1e-6)`` rather
than the base class's five fixed iterations.


MacKay's update
---------------

With :math:`\boldsymbol{\Lambda} = s_t \mathbf{I} + \mathbf{H}` and
the mode :math:`\hat{\boldsymbol{\theta}}`, the update is the
:math:`\alpha` half of the Normal case:

.. math::

   \gamma_{\text{eff}}
   = p - s_t\,\operatorname{tr}(\boldsymbol{\Lambda}^{-1}),
   \qquad
   \alpha_{\text{new}}
   = \frac{\gamma_{\text{eff}}}{\|\hat{\boldsymbol{\theta}}\|^2}

:math:`\gamma_{\text{eff}}` is clipped to
:math:`[\,p \cdot 10^{-13},\; \min(N_{\text{eff}}, p)\,]`. The lower
floor sits just above the cancellation noise of the subtraction when
the prior dominates :math:`\mathbf{H}`; the Normal's larger floor
would inflate :math:`\alpha_{\text{new}}` there and reject the very
steps that bring an overgrown :math:`\alpha` back down.

**Fixed point vs. evidence maximum.** MacKay's rule treats
:math:`\mathbf{H}` as constant in :math:`\alpha`. For a GLM it is
not: a different :math:`\alpha` gives a different mode, and the
Hessian is evaluated there. The iteration therefore converges to a
fixed point that sits near, not at, the maximum of the Laplace
evidence, within a few percent of :math:`\alpha` in practice.


Guard rail
~~~~~~~~~~

The Normal's :math:`\beta / \alpha` ratio has no analogue, but the
same degeneracies exist. With separable logistic data
:math:`\|\hat{\boldsymbol{\theta}}\|` grows without bound and MacKay
drives :math:`\alpha \to 0`. With data that carry no information the
evidence increases in :math:`\alpha` without bound and the mode
shrinks until it underflows, which pins :math:`\alpha` forever.

The largest diagonal entry of :math:`\mathbf{H}` is a cheap lower
bound on its largest eigenvalue, so :math:`\alpha_{\text{new}}` is
accepted only within :math:`10^{10}` of
:math:`\max_i \mathbf{H}_{ii}` on either side: the band where
:math:`\boldsymbol{\Lambda}` is neither numerically singular nor
numerically the prior. Outside it :math:`\alpha` stays where it was,
``eb_updates_rejected_`` is incremented, and tuning resumes once the
data argue for it.


fit vs. partial_fit
--------------------

``fit`` runs IRLS once from :math:`\boldsymbol{\theta} = 0`, then
alternates a MacKay step with a refit at the new :math:`\alpha` until
:math:`|\Delta \log p| < \texttt{eb\_tol}` or ``n_eb_iter`` is
reached. Each refit is from scratch in the Bayesian sense (the prior
is reset to :math:`\alpha_{\text{new}} \mathbf{I}`) but IRLS is
warm-started at the previous mode and refactorizes from the previous
factor, so it typically converges in a few iterations. A MacKay step
that is rejected, or that leaves :math:`\alpha` unchanged, skips the
refit.

``partial_fit`` runs one MacKay step per call:

1. Apply stabilized forgetting to the prior scalar (see below); the
   re-injection is folded into the IRLS prior rather than added
   afterwards, so the one factorization covers both.
2. Perform the Laplace update (inherited from
   :class:`~bayesianbandits.BayesianGLM`), starting from the current
   posterior.
3. Accumulate :math:`N_{\text{eff}}` and :math:`\ell_{\text{eff}}`:
   the batch's log-likelihood at the mode is added to the decayed
   running sum.
4. Run one MacKay step.
5. Correct the precision (see below).

Under ``partial_fit`` the log-likelihood term of ``log_evidence_``
is therefore a decayed sum of per-batch values, each evaluated at
the mode right after that batch, not the log-likelihood of all the
data at the current mode. It is exact only after ``fit``.


Precision correction
--------------------

After MacKay changes :math:`\alpha`, only the prior part of the
precision moves:

.. math::

   \boldsymbol{\Lambda}_{\text{corrected}}
   = \boldsymbol{\Lambda}
   + \left(\frac{\alpha_{\text{new}}}{\alpha_{\text{old}}} - 1\right)
     s_t\,\mathbf{I},
   \qquad
   s_t \leftarrow s_t\,\frac{\alpha_{\text{new}}}{\alpha_{\text{old}}}

a diagonal shift that leaves the sparsity pattern alone, so a sparse
factor is refactorized numerically without a new symbolic analysis.

The mode moves with it. The information vector
:math:`\boldsymbol{\Lambda}\hat{\boldsymbol{\theta}}` is held fixed
and re-solved against the corrected precision:

.. math::

   \hat{\boldsymbol{\theta}}
   \leftarrow \boldsymbol{\Lambda}_{\text{corrected}}^{-1}\,
   \boldsymbol{\Lambda}\,\hat{\boldsymbol{\theta}}

For the Normal this is exact. For a GLM it is one Newton step's worth
of approximation: the true new mode would need IRLS to re-converge.
The next ``partial_fit`` starts its IRLS from this point, so the
error is corrected as data arrive.


Stabilized forgetting
---------------------

Identical to :doc:`empirical-bayes`. After each decay of
:math:`n` rows, :math:`(1 - \gamma^n)\,\alpha` is re-injected onto
the precision diagonal and the prior scalar follows

.. math::

   s_{t+n} = \gamma^n\, s_t + (1 - \gamma^n)\,\alpha

converging to :math:`\alpha`, so the prior's contribution never
vanishes and the tuned :math:`\alpha` stays load-bearing. During
``decay``, :math:`N_{\text{eff}}` and :math:`\ell_{\text{eff}}` are
scaled by :math:`\gamma^n`.

``decay`` is only defined on a fitted model; decaying before the
first update is not supported.


Hyperparameter semantics
-------------------------

.. list-table::
   :header-rows: 1
   :widths: 20 50 30

   * - Parameter
     - Controls
     - Practical guidance
   * - ``alpha``
     - Initial prior precision. EB tunes this automatically.
     - Start with 1.0. It matters for the first few observations
       before EB has data to work with.
   * - ``n_eb_iter``
     - Maximum EB iterations during ``fit``. Each iteration runs one
       MacKay step and a warm-started refit.
     - 10 (default). Set to 0 to disable EB during ``fit``.
   * - ``eb_tol``
     - Convergence tolerance on log evidence change between
       iterations.
     - 1e-4 (default).
   * - ``approximator``
     - Posterior approximation; see :doc:`glm`.
     - Default ``LaplaceApproximator(n_iter=25, tol=1e-6)``. With
       ``RVGAApproximator`` the precision is an expected rather than
       observed curvature and ``log_evidence_`` is a heuristic.
   * - ``forgetting``
     - Forgetting rule applied on ``partial_fit``, carrying its rate
       :math:`\gamma`. See :doc:`/howto/decay`.
     - ``None`` (default) for stationary environments.
   * - ``trace_method``
     - Method for computing
       :math:`\operatorname{tr}(\boldsymbol{\Lambda}^{-1})`.
     - ``'auto'`` (default) is exact. Use ``'diagonal'`` for very
       large :math:`p` when the precision is diagonally dominant.


References
----------

.. [1] MacKay, D. J. C. (1992). "Bayesian Interpolation."
   *Neural Computation*, 4(3), 415--447.

.. [2] Kulhavy, R. & Zarrop, M. B. (1993). "On a general concept of
   forgetting." *International Journal of Control*, 58(4), 905--924.
