Forgetting Strategies
=====================

Four strategies for shrinking the posterior precision matrix before
a recursive Bayesian update.  Each of the first three addresses a
limitation of the previous one; the fourth trades exactness for the
sparsity pattern.  The update and sampling steps are unchanged;
see :doc:`normal` for those details and :doc:`/howto/decay` for
practical tuning guidance.


Symbols
-------

.. list-table::
   :header-rows: 1
   :widths: 15 85

   * - Symbol
     - Meaning
   * - :math:`\boldsymbol{\Lambda}`
     - Posterior precision matrix before forgetting
   * - :math:`\bar{\boldsymbol{\Lambda}}`
     - Precision matrix after forgetting (input to the update step)
   * - :math:`\gamma`
     - Forgetting factor, in :math:`(0, 1]`
   * - :math:`\alpha`
     - Prior precision scalar
   * - :math:`\beta`
     - Noise precision
   * - :math:`\mathbf{X}`
     - Design matrix (batch of observations)
   * - :math:`\bar{\mathbf{X}}`
     - Filtered design matrix (rank-:math:`q` approximation)
   * - :math:`q`
     - Rank of the filtered batch (:math:`q \le \min(n_{\text{batch}}, p)`)
   * - :math:`p`
     - Number of features
   * - :math:`\varepsilon`
     - Eigenvalue threshold for batch filtering
   * - :math:`m_i`
     - Number of rows of the batch in which feature :math:`i` is nonzero
   * - :math:`\mathbf{D}`
     - Diagonal matrix of per-feature factors :math:`\gamma^{m_i/2}`


The update loop
---------------

The forgetting rule maps
:math:`\boldsymbol{\Lambda} \to \bar{\boldsymbol{\Lambda}}`.
The subsequent update is standard:

.. math::

   \boldsymbol{\Lambda}_n
   &= \bar{\boldsymbol{\Lambda}}
   + \beta\,\mathbf{X}^\top \mathbf{W} \mathbf{X} \\
   \boldsymbol{\eta}_n
   &= \bar{\boldsymbol{\Lambda}}\,\boldsymbol{\mu}_{\text{old}}
   + \beta\,\mathbf{X}^\top \mathbf{W}\mathbf{y} \\
   \boldsymbol{\mu}_n
   &= \boldsymbol{\Lambda}_n^{-1}\,\boldsymbol{\eta}_n

The four strategies below differ only in the forgetting rule.


Exponential forgetting
----------------------

.. math::

   \bar{\boldsymbol{\Lambda}} = \gamma\,\boldsymbol{\Lambda}

Scalar multiply on the precision.  Preserves sparsity, no parameters
beyond :math:`\gamma`, reduces to no forgetting when
:math:`\gamma = 1`.

Equivalent to the predict step of a Kalman filter for a random-walk
model with process noise
:math:`\mathbf{Q} = (1 - \gamma)\,\boldsymbol{\Lambda}^{-1}`.

**Covariance windup.**
Each step scales every eigenvalue by :math:`\gamma`.  The update
replenishes precision only in directions excited by the current batch.
Unexcited directions lose :math:`1 - \gamma` per step and never
recover; their eigenvalues approach zero and the Cholesky fails.  The
prior :math:`\alpha\mathbf{I}` decays along with everything else.


Stabilized forgetting (Kulhavy--Zarrop)
----------------------------------------

.. math::

   \bar{\boldsymbol{\Lambda}}
   = \gamma\,\boldsymbol{\Lambda}
   + (1 - \gamma)\,\alpha\,\mathbf{I}

After scaling by :math:`\gamma`, a fraction of the prior is added
back [1]_.  No eigenvalue of :math:`\bar{\boldsymbol{\Lambda}}` can
fall below :math:`(1 - \gamma)\,\alpha`.

**Convergence.**
The prior scalar :math:`s_t` on the diagonal evolves as
:math:`s_{t+1} = \gamma\,s_t + (1 - \gamma)\,\alpha`, which has
fixed point :math:`s^* = \alpha`.  See :doc:`empirical-bayes` for
the interaction with EB hyperparameter tuning.

**Isotropy.**
Every direction loses the same fraction of precision and gets the
same floor, regardless of excitation.


Directional forgetting (SIFt)
------------------------------

.. math::

   \bar{\boldsymbol{\Lambda}}
   = \boldsymbol{\Lambda}
   - (1 - \gamma)\,
     \boldsymbol{\Lambda}\bar{\mathbf{X}}^\top
     \bigl(\bar{\mathbf{X}}\,\boldsymbol{\Lambda}\,
           \bar{\mathbf{X}}^\top\bigr)^{-1}
     \bar{\mathbf{X}}\,\boldsymbol{\Lambda}

The correction term projects the precision onto the subspace excited
by the batch and removes :math:`1 - \gamma` of it.  Directions
orthogonal to the batch are unchanged; fully excited directions
receive the same :math:`\gamma` scaling as exponential forgetting
[2]_.

Before applying the formula, the batch is filtered: eigenvalues of
the Gram matrix below :math:`\varepsilon` are discarded, leaving a
rank-:math:`q` approximation :math:`\bar{\mathbf{X}}` that preserves
:math:`\mathbf{X}^\top\!\mathbf{X}` and
:math:`\mathbf{X}^\top\!\mathbf{y}` in the surviving subspace [3]_.


Precision retention
~~~~~~~~~~~~~~~~~~~

:math:`\bar{\boldsymbol{\Lambda}} \succeq \gamma\,\boldsymbol{\Lambda}`.

Directional forgetting retains at least as much precision as
exponential forgetting.  Strictly more when :math:`q < p`
(Proposition 5 in [3]_).


Eigenvalue floor
~~~~~~~~~~~~~~~~

After arbitrarily many forget–update cycles:

.. math::

   \lambda_{\min}(\boldsymbol{\Lambda}_k)
   \ge \min\!\Bigl(
     \frac{\varepsilon}{1 - \gamma},\;
     \lambda_{\min}(\boldsymbol{\Lambda}_0)
   \Bigr)

No prior injection is needed (Theorem 3 in [3]_).  Compare with the
stabilized floor :math:`(1 - \gamma)\,\alpha`.


Computational cost
~~~~~~~~~~~~~~~~~~

**Dense.**
The inner Gram
:math:`\bar{\mathbf{X}}\,\boldsymbol{\Lambda}\,\bar{\mathbf{X}}^\top`
is :math:`q \times q`.  Cholesky is :math:`O(q^3)`.  The dominant
cost is the rank-:math:`q` symmetric update via ``dsyrk`` at
:math:`O(p^2 q)`, the same asymptotic cost as the standard RLS
update.

**Sparse.**
If the batch activates :math:`k_0` columns of a sparse design matrix,
the linear algebra operates on a :math:`k_0 \times k_0` dense
submatrix extracted from the sparse precision.  The sparse path uses
pivoted Cholesky (``dpstrf``) rather than eigendecomposition, which
handles rank deficiency and is faster for the small Gram matrices
that arise in practice.


Feature-wise forgetting (vector-type)
-------------------------------------

.. math::

   \bar{\boldsymbol{\Lambda}} = \mathbf{D}\,\boldsymbol{\Lambda}\,\mathbf{D},
   \qquad
   \mathbf{D} = \operatorname{diag}\bigl(\gamma^{m_i/2}\bigr)

Each feature is forgotten by :math:`\gamma` once per row of the batch
in which it appears, and features absent from the batch are not
forgotten at all.  Entry :math:`(i, j)` of the precision is scaled by
:math:`\gamma^{(m_i + m_j)/2}`, so the diagonal of an observed feature
scales by :math:`\gamma^{m_i}` and its cross terms with unobserved
features by :math:`\gamma^{m_i/2}`.  When every feature appears in
every row this is exponential forgetting; when the batch is one-hot it
touches one row and one column per active feature.

Per-parameter forgetting factors applied as a congruence of the
covariance,
:math:`\bar{\mathbf{P}} = \boldsymbol{\Lambda}_f^{-1/2}\,\mathbf{P}\,\boldsymbol{\Lambda}_f^{-1/2}`,
are the *vector variable forgetting factor* of [4]_ [5]_ and the
*selective forgetting* of [6]_; the form above is equation (12) of
[7]_ and of [8]_.  Those works fix the factors per parameter from prior
knowledge of how fast each one drifts.  Setting them from the batch
support, :math:`\gamma` on the active features and :math:`1`
elsewhere, is what makes the rule directional: it is the oracle
"multiple forgetting" baseline that [3]_ reports as comparable to
SIFt, with the oracle read off the regressor.


What it preserves
~~~~~~~~~~~~~~~~~

In covariance form the rule is
:math:`\bar{\boldsymbol{\Sigma}} = \mathbf{D}^{-1}\boldsymbol{\Sigma}\mathbf{D}^{-1}`:
the standard deviation of each observed coefficient is inflated by
:math:`\gamma^{-m_i/2}` and nothing else moves.  Consequently

- every correlation between coefficients is unchanged;
- the marginal posterior of every unobserved feature is unchanged
  (the Schur complement of the untouched block is invariant, which is
  what fixes the exponent at one half);
- the sparsity pattern of :math:`\boldsymbol{\Lambda}` is unchanged,
  so a sparse factorization refactorizes numerically on the same
  symbolic analysis;
- positive definiteness is preserved, by congruence.

The cost is one pass over the nonzeros of :math:`\boldsymbol{\Lambda}`,
the same as exponential forgetting.


What it gives up
~~~~~~~~~~~~~~~~

The rule is a coordinate stretch of the posterior about its mean, not
a Bayesian update: it is neither conditioning nor a transition model.
:math:`\boldsymbol{\Lambda} - \bar{\boldsymbol{\Lambda}}` is not
positive semidefinite in general, so the rule is not *proper* in the
sense of [9]_, which notes the same of the multiple-forgetting scheme
of [8]_.  Concretely, stretching a tilted ellipse along one axis
rotates its principal axes, so the combination of an observed feature
with a correlated unobserved one that the posterior claims to know
best moves to a new combination, and that combination can come out
tighter than any data supported.  The rotation is in the same
direction as the exact update; the overclaim is in its width, and it
grows with the correlation.  Relative to a Kalman step inflating the
observed coefficient by the same amount, the best-known combination is
about 2% too tight at correlation :math:`0.8` and
:math:`\gamma = 0.98`, 8% at :math:`0.95`, and 29% at :math:`0.99`.
The next observation of that feature re-pins the combination from
data.

SIFt is the exact counterpart: it adds noise along the excited
direction rather than stretching, which can never tighten any
direction, and it pays for that with a correction that is dense on the
neighbourhood of the active features.  In the special case where the
observed feature is conditionally independent of every other
(its row of :math:`\boldsymbol{\Lambda}` is diagonal) the two rules
coincide.


Choosing :math:`\gamma`
~~~~~~~~~~~~~~~~~~~~~~~~

The same :math:`\gamma` means different amounts of memory under
different rules.  Per row, exponential forgetting removes
:math:`p \log\gamma` of log-determinant from the precision,
feature-wise removes :math:`k \log\gamma` where :math:`k` is the number
of active features, and SIFt removes exactly :math:`\log\gamma`.  A
SIFt factor of :math:`\gamma^k` is therefore a reasonable starting
point for the memory of a feature-wise factor of :math:`\gamma`.


Choosing a strategy
-------------------

.. list-table::
   :header-rows: 1
   :widths: 20 25 25 30

   * - Strategy
     - Strengths
     - Limitations
     - Best for
   * - Exponential
     - Simplest; no extra parameters; preserves sparsity
     - Covariance windup under non-uniform excitation
     - Uniformly excited features; prototyping
   * - Stabilized
     - Prior floor prevents collapse; integrates with Empirical
       Bayes
     - Isotropic; decays rare features at the same rate as
       common ones
     - Environments where EB tunes :math:`\alpha`; safety net
       against collapse
   * - Directional (SIFt)
     - Preserves precision in unexcited directions; natural
       eigenvalue floor; exact
     - Correction is dense on the neighbourhood of the excited
       features, so a sparse precision fills in
     - Dense or correlated continuous features
   * - Feature-wise
     - Forgets only observed features; preserves the sparsity
       pattern; exponential cost
     - A coordinate stretch, not a Bayesian update; can
       over-tighten combinations with strongly correlated
       unobserved features
     - One-hot and hierarchical sparse designs

Stabilized and directional forgetting address orthogonal problems
(prior collapse vs. isotropic decay) and can be combined.  On one-hot
hierarchies, feature-wise and SIFt reach the same predictive accuracy
and calibration; feature-wise does so at the cost and pattern of
exponential forgetting, while SIFt's precision becomes dense over
every feature ever observed.  On dense low-rank regressors feature-wise
reduces to exponential forgetting and only SIFt protects the unexcited
directions.


Adaptation from SIFt-RLS
-------------------------

SIFt-RLS [3]_ is formulated for classical recursive least squares.
Adapting it to the precision-parameterized Bayesian estimators in
this library required several changes.  This section documents each
and why correctness is preserved.


Precision-only updates
~~~~~~~~~~~~~~~~~~~~~~

Algorithm 1 of [3]_ maintains both the information matrix
:math:`R_k` and the covariance
:math:`P_k \triangleq R_k^{-1}`, updating the covariance via the
matrix inversion lemma (equation 31 in [3]_).  Only the precision
is kept here: the SIFt step (equation 27) and the information
update (equation 29) are both defined on :math:`R_k`, and the
posterior mean is recovered by Cholesky solve.  The covariance is
never formed.
:math:`\boldsymbol{\Lambda}` is positive definite at every step
(Corollary 1 in [3]_), so the solve is well-conditioned.


Gram eigendecomposition instead of SVD
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

[3]_ filters the regressor via compact SVD, thresholding singular
values below :math:`\sqrt{\varepsilon}` (Section 4.1).  The Gram
matrix :math:`\mathbf{G} = \mathbf{X}\mathbf{X}^\top` is
eigendecomposed instead.  Its eigenvalues are the squared singular
values of :math:`\mathbf{X}`, so thresholding at
:math:`\varepsilon` is equivalent to thresholding singular values at
:math:`\sqrt{\varepsilon}`.  The eigenvectors of :math:`\mathbf{G}`
are the left singular vectors :math:`\mathbf{U}_k`, the only factor
needed to form
:math:`\bar{\mathbf{X}} = \mathbf{U}_q^\top \mathbf{X}`
(:math:`\mathbf{V}_k` need not be computed; footnote 3 in [3]_).


Pivoted Cholesky for sparse batch filtering
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

When :math:`\mathbf{X}` is sparse with :math:`k_0` active (nonzero)
columns, the dense :math:`k_0 \times k_0` active-column Gram
:math:`\mathbf{G}_a = \mathbf{X}_a^\top \mathbf{X}_a` is factored
via LAPACK pivoted Cholesky (``dpstrf``) with tolerance
:math:`\varepsilon`:

.. math::

   \mathbf{P}^\top \mathbf{G}_a \mathbf{P}
   = \mathbf{L}_{:r}\,\mathbf{L}_{:r}^\top

where :math:`r` is the numerical rank and :math:`\mathbf{P}` is a
permutation.  Setting
:math:`\bar{\mathbf{X}}_a = \mathbf{L}_{:r}[\mathbf{P}^{-1}]^\top`
gives
:math:`\bar{\mathbf{X}}_a^\top \bar{\mathbf{X}}_a = \mathbf{G}_a`,
preserving sufficient statistics in the surviving subspace.  The
nonzero eigenvalues of :math:`\mathbf{G}_a` are identical to those of
the full :math:`p \times p` Gram, so the rank determination is
equivalent.  Pivoted Cholesky is :math:`O(k_0^2 r)` versus
:math:`O(k_0^3)` for eigendecomposition.


Symmetry preservation
~~~~~~~~~~~~~~~~~~~~~

Computing the SIFt correction via ``solve(H, w.T)`` produces an
asymmetric result due to rounding, and the error accumulates over
many steps.  Algorithm 1 of [3]_ corrects this with
:math:`R_k \leftarrow \tfrac{1}{2}(R_k + R_k^\top)` (line 12).

Instead,
:math:`\mathbf{H} = \bar{\mathbf{X}}\,\boldsymbol{\Lambda}\,\bar{\mathbf{X}}^\top`
is Cholesky-factored as :math:`\mathbf{L}\mathbf{L}^\top` and the
correction is computed as
:math:`\mathbf{V}^\top\!\mathbf{V}` where
:math:`\mathbf{V} = \mathbf{L}^{-1}\mathbf{w}^\top` and
:math:`\mathbf{w} = \boldsymbol{\Lambda}\bar{\mathbf{X}}^\top`.
:math:`\mathbf{V}^\top\!\mathbf{V}` is a Gram matrix, hence
symmetric by construction:

.. math::

   \mathbf{V}^\top\!\mathbf{V}
   = \mathbf{w}\,\mathbf{L}^{-\top}\mathbf{L}^{-1}\mathbf{w}^\top
   = \mathbf{w}\,\mathbf{H}^{-1}\mathbf{w}^\top
   = \boldsymbol{\Lambda}\bar{\mathbf{X}}^\top
     (\bar{\mathbf{X}}\,\boldsymbol{\Lambda}\,
      \bar{\mathbf{X}}^\top)^{-1}
     \bar{\mathbf{X}}\,\boldsymbol{\Lambda}

The rank-:math:`q` update is applied via ``dsyrk``, which writes only
one triangle, so no post-hoc symmetrization is needed.


Sparse precision downdate
~~~~~~~~~~~~~~~~~~~~~~~~~

[3]_ does not discuss sparse precision matrices.  When
:math:`\bar{\mathbf{X}}` has nonzero entries only in :math:`k_0`
columns,
:math:`\boldsymbol{\Lambda}\bar{\mathbf{X}}^\top` is nonzero only
in the rows of :math:`\boldsymbol{\Lambda}` that interact with those
columns.  Call this row set :math:`\mathcal{N}`.  The correction
:math:`\mathbf{V}^\top\!\mathbf{V}` is a
:math:`|\mathcal{N}| \times |\mathcal{N}|` dense block embedded in
the :math:`p \times p` sparse matrix; all linear algebra is done on
that block.  Entries outside :math:`\mathcal{N}` are structurally
zero, so no approximation is introduced.


References
----------

.. [1] Kulhavy, R. & Zarrop, M. B. (1993). "On a general concept of
   forgetting." *International Journal of Control*, 58(4), 905--924.

.. [2] Cao, L. & Schwartz, H. M. (2000). "A directional forgetting
   algorithm based on the decomposition of the information matrix."
   *Automatica*, 36(11), 1725--1731.

.. [3] Lai, B. & Bernstein, D. S. (2024). "SIFt-RLS: Subspace of
   Information Forgetting Recursive Least Squares."
   *arXiv:2404.10844*.

.. [4] Saelid, S. & Foss, B. (1983). "Adaptive controllers with a vector
   variable forgetting factor." *Proceedings of the 22nd IEEE Conference
   on Decision and Control*, 1488--1494.

.. [5] Saelid, S., Egeland, O. & Foss, B. (1985). "A solution to the
   blow-up problem in adaptive controllers." *Modeling, Identification
   and Control*, 6(1), 39--56.

.. [6] Parkum, J. E., Poulsen, N. K. & Holst, J. (1992). "Recursive
   forgetting algorithms." *International Journal of Control*, 55(1),
   109--128.

.. [7] Fraccaroli, F., Peruffo, A. & Zorzi, M. (2015). "A new recursive
   least-squares method with multiple forgetting schemes."
   *arXiv:1503.07338*.

.. [8] Vahidi, A., Stefanopoulou, A. & Peng, H. (2005). "Recursive least
   squares with forgetting for online estimation of vehicle mass and
   road grade: theory and experiments." *Vehicle System Dynamics*,
   43(1), 31--55.

.. [9] Lai, B. & Bernstein, D. S. (2024). "Generalized forgetting
   recursive least squares: stability and robustness guarantees."
   *IEEE Transactions on Automatic Control*. *arXiv:2308.04259*.
