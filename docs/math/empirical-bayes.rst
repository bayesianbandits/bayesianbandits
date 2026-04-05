EmpiricalBayesNormalRegressor
=============================

Automatic tuning of :math:`\alpha` and :math:`\beta` via MacKay's
evidence maximization [1]_, with stabilized forgetting [2]_ to prevent
prior collapse under exponential decay.

Builds on the posterior update from :doc:`normal`, which is inherited
unchanged. Read that page first.


Symbols
-------

In addition to the symbols defined in :doc:`normal`:

.. list-table::
   :header-rows: 1
   :widths: 15 85

   * - Symbol
     - Meaning
   * - :math:`\gamma_{\text{eff}}`
     - Effective number of well-determined parameters (MacKay's
       :math:`\gamma`). Distinct from the decay factor :math:`\gamma`.
   * - :math:`s_t`
     - Cumulative (decayed) prior contribution to the precision
       diagonal (stored as ``_prior_scalar``)
   * - :math:`N_{\text{eff}}`
     - Decayed effective sample size (stored as ``_effective_n``)
   * - :math:`\mathbf{y}^\top\!\mathbf{y}_{\text{eff}}`
     - Decayed sum of squared targets (stored as ``_eff_yTy``)
   * - :math:`\mathbf{X}^\top\!\mathbf{y}_{\text{eff}}`
     - Decayed cross-product vector (stored as ``_eff_XTy``)


MacKay's evidence maximization
-------------------------------

Given the posterior precision
:math:`\boldsymbol{\Lambda} = s_t \mathbf{I} + \beta\,\mathbf{X}^\top\!\mathbf{X}_{\text{eff}}`
and the posterior mean :math:`\boldsymbol{\mu}_n`, MacKay's update
rules [1]_ adjust :math:`\alpha` and :math:`\beta` to maximize the
log marginal likelihood (evidence).


Effective number of parameters
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

.. math::

   \gamma_{\text{eff}}
   = p - \alpha\,\operatorname{tr}(\boldsymbol{\Lambda}^{-1})

Clipped to :math:`[\varepsilon,\; \min(N_{\text{eff}},\, p)]`. This
quantity measures how many weight components are determined by data
rather than by the prior. When :math:`\gamma_{\text{eff}} \approx p`,
the data determines all parameters; when
:math:`\gamma_{\text{eff}} \approx 0`, the prior dominates.

Computing :math:`\operatorname{tr}(\boldsymbol{\Lambda}^{-1})`:

``trace_method='auto'`` (default)
   Exact via Cholesky (dense) or Takahashi recursion [3]_ (sparse).

``trace_method='diagonal'``
   :math:`O(p)` approximation
   :math:`\operatorname{tr}(\boldsymbol{\Lambda}^{-1}) \approx
   \sum_i 1/\Lambda_{ii}`, valid when the precision is strongly
   diagonally dominant.


Alpha update
~~~~~~~~~~~~

.. math::

   \alpha_{\text{new}}
   = \frac{\gamma_{\text{eff}}}
          {\boldsymbol{\mu}_n^\top \boldsymbol{\mu}_n}

Increasing :math:`\alpha` strengthens regularization, shrinking the
weights toward zero. When all parameters are well-determined
(:math:`\gamma_{\text{eff}} \approx p`) and the weights are large,
:math:`\alpha` decreases, relaxing regularization.


Beta update
~~~~~~~~~~~

Textbook MacKay computes :math:`\beta` from the residual sum of
squares, which requires :math:`N`, :math:`\mathbf{y}^\top\mathbf{y}`,
:math:`\mathbf{X}^\top\mathbf{y}`, and
:math:`\mathbf{X}^\top\mathbf{X}`. In a batch setting you have all
of these directly. Online, you need to maintain them as running totals
under decay.

Three of the four are scalar or vector quantities that accumulate
naturally:

.. math::

   N_{\text{eff}} &\leftarrow \gamma^n\, N_{\text{eff}} + n_{\text{new}} \\
   \mathbf{y}^\top\!\mathbf{y}_{\text{eff}}
   &\leftarrow \gamma^n\, \mathbf{y}^\top\!\mathbf{y}_{\text{eff}}
   + \mathbf{y}_{\text{new}}^\top \mathbf{y}_{\text{new}} \\
   \mathbf{X}^\top\!\mathbf{y}_{\text{eff}}
   &\leftarrow \gamma^n\, \mathbf{X}^\top\!\mathbf{y}_{\text{eff}}
   + \mathbf{X}_{\text{new}}^\top \mathbf{y}_{\text{new}}

The fourth, :math:`\mathbf{X}^\top\!\mathbf{X}_{\text{eff}}`, is a
:math:`p \times p` matrix that would be expensive to store separately.
Instead, it is recovered from the precision matrix:

.. math::

   \mathbf{X}^\top\!\mathbf{X}_{\text{eff}}
   = \frac{\boldsymbol{\Lambda} - s_t\,\mathbf{I}}{\beta}

This works because
:math:`\boldsymbol{\Lambda} = s_t\,\mathbf{I} + \beta\,\mathbf{X}^\top\!\mathbf{X}_{\text{eff}}`
by construction, and the tracked prior scalar :math:`s_t` separates
the prior and data contributions exactly, even after multiple rounds
of decay.

With all four statistics in hand, the RSS and :math:`\beta` update
are:

.. math::

   \text{RSS}
   = \mathbf{y}^\top\!\mathbf{y}_{\text{eff}}
   - 2\,\boldsymbol{\mu}_n^\top \mathbf{X}^\top\!\mathbf{y}_{\text{eff}}
   + \boldsymbol{\mu}_n^\top \mathbf{X}^\top\!\mathbf{X}_{\text{eff}}\,
     \boldsymbol{\mu}_n

.. math::

   \beta_{\text{new}}
   = \frac{N_{\text{eff}} - \gamma_{\text{eff}}}{\text{RSS}}

These are the standard MacKay equations; the only difference from batch
is that the sufficient statistics are maintained incrementally under
decay. Multiple incremental updates produce the same sufficient
statistics as a single batch update over the same data, with one
caveat: the guard rails are checked after each step.


Guard rails
~~~~~~~~~~~~

When :math:`\beta_{\text{new}} / \alpha_{\text{new}} > 10^{10}`, the
update is rejected entirely (both :math:`\alpha` and :math:`\beta`
revert). In the underdetermined regime (:math:`p \gg N`), MacKay can
drive :math:`\alpha \to 0` and :math:`\beta \to \infty`, making the
precision matrix ill-conditioned. The hard cutoff freezes EB tuning
until enough data arrives to make the problem well-determined.


Log marginal likelihood
~~~~~~~~~~~~~~~~~~~~~~~

.. math::

   \log p(\mathbf{y} \mid \mathbf{X}, \alpha, \beta)
   = \tfrac{p}{2}\log\alpha
   + \tfrac{N_{\text{eff}}}{2}\log\beta
   - \tfrac{1}{2}\log|\boldsymbol{\Lambda}|
   - \tfrac{1}{2}(\beta\,\text{RSS} + \alpha\,\|\boldsymbol{\mu}_n\|^2)
   - \tfrac{N_{\text{eff}}}{2}\log 2\pi

Stored as ``log_evidence_`` after ``fit``.


fit vs. partial_fit
--------------------

The MacKay step is the same in both cases. ``fit`` iterates it to
convergence on fixed data (up to ``n_eb_iter`` times, checking
:math:`|\Delta \log p| < \texttt{eb\_tol}`), then refits the posterior
with the converged hyperparameters. ``partial_fit`` runs one step per
call, since new data is imminent. In both cases the sufficient
statistics and the precision-matrix decomposition provide what MacKay
needs.

During ``partial_fit``:

1. Apply stabilized forgetting to the prior scalar (see below).
2. Perform the recursive Bayesian update (inherited from
   :class:`~bayesianbandits.NormalRegressor`).
3. Accumulate the three running sufficient statistics (above).
4. Run one MacKay step.
5. Correct the precision matrix (see below).


Precision correction
--------------------

After MacKay changes :math:`\alpha` and :math:`\beta`, the precision
matrix must be corrected to reflect the new hyperparameters without
recomputing from scratch.

The precision decomposes as:

.. math::

   \boldsymbol{\Lambda}
   = s_t\,\mathbf{I} + \beta\,\mathbf{X}^\top\!\mathbf{X}_{\text{eff}}

The correction rescales each component independently:

.. math::

   \boldsymbol{\Lambda}_{\text{corrected}}
   = \frac{\beta_{\text{new}}}{\beta_{\text{old}}}\,\boldsymbol{\Lambda}
   + \left(
       s_t \frac{\alpha_{\text{new}}}{\alpha_{\text{old}}}
       - \frac{\beta_{\text{new}}}{\beta_{\text{old}}}\, s_t
     \right) \mathbf{I}

The first term scales everything (prior + data) by the beta ratio.
The diagonal correction then adjusts the prior part to use the new
alpha ratio instead. After correction,
:math:`s_t \leftarrow s_t \cdot \alpha_{\text{new}} / \alpha_{\text{old}}`.


Stabilized forgetting
---------------------

**The problem.** Under exponential decay with factor :math:`\gamma < 1`,
the prior contribution to the precision diagonal decays as
:math:`\gamma^t \alpha \to 0`. After enough decay steps the model is
effectively unregularized, and the precision matrix can become
ill-conditioned.

**The solution.** After each decay step, re-inject
:math:`(1 - \gamma^n)\,\alpha` onto the diagonal. The prior scalar
:math:`s_t` evolves as:

.. math::

   s_{t+n}
   = \gamma^n\, s_t + (1 - \gamma^n)\,\alpha

This is a geometric recursion that converges to :math:`\alpha`
regardless of the starting value. The prior contribution never vanishes,
maintaining a regularization floor.

**Interaction with EB.** The stabilization target is :math:`\alpha`,
which is itself updated by MacKay. EB changes :math:`\alpha` slowly
(one step per ``partial_fit``), so stabilization prevents prior
collapse between updates.

See :doc:`forgetting` for the broader context of forgetting
strategies, including directional forgetting which addresses the
isotropy limitation of stabilized forgetting.

During ``decay``, the sufficient statistics are also decayed:

.. math::

   N_{\text{eff}} &\leftarrow \gamma^n\, N_{\text{eff}} \\
   \mathbf{y}^\top\!\mathbf{y}_{\text{eff}}
   &\leftarrow \gamma^n\, \mathbf{y}^\top\!\mathbf{y}_{\text{eff}} \\
   \mathbf{X}^\top\!\mathbf{y}_{\text{eff}}
   &\leftarrow \gamma^n\, \mathbf{X}^\top\!\mathbf{y}_{\text{eff}}


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
     - Start with 1.0. The initial value matters for the first few
       observations before EB has data to work with.
   * - ``beta``
     - Initial noise precision. EB tunes this automatically.
     - Start with 1.0, or set to :math:`1/\sigma^2` if the noise
       scale is roughly known.
   * - ``n_eb_iter``
     - Maximum EB iterations during ``fit``. Each iteration refits
       the posterior and runs one MacKay step.
     - 10 (default) is usually sufficient. Set to 0 to disable EB
       during ``fit``.
   * - ``eb_tol``
     - Convergence tolerance on log evidence change between
       iterations.
     - 1e-4 (default). Lower values give tighter convergence at the
       cost of more iterations.
   * - ``learning_rate``
     - Decay factor :math:`\gamma`. See :doc:`/howto/decay`.
     - 1.0 (default) for stationary environments.
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

.. [3] Takahashi, K., Fagan, J., & Chin, M.-S. (1973). "Formation of
   a sparse bus impedance matrix and its application to short circuit
   study." 8th PICA Conference Proceedings.
