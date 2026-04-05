Forgetting Strategies
=====================

Three strategies for shrinking the posterior precision matrix before
a recursive Bayesian update.  Each addresses a limitation of the
previous one.  The update and sampling steps are unchanged;
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

The three strategies below differ only in the forgetting rule.


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
       eigenvalue floor
     - Slightly more compute per step
     - Sparse / high-dimensional features with heterogeneous
       excitation

Stabilized and directional forgetting address orthogonal problems
(prior collapse vs. isotropic decay) and can be combined.


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
