Forgetting Strategies
=====================

In non-stationary environments, old data collected under conditions
that no longer hold actively misleads the model.  Forgetting is a
concrete operation on the posterior precision matrix: shrink it to
become less confident, then let new data rebuild confidence in the
directions it excites.

The three strategies below differ only in *how* the precision is
shrunk.  Everything else -- the Bayesian update, the Cholesky solve,
the sampling -- is unchanged.

See :doc:`/howto/decay` for practical guidance on *when* and *how
much* to decay.


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

Every recursive Bayesian update with forgetting has three steps.
First, apply a forgetting rule to get
:math:`\bar{\boldsymbol{\Lambda}}` from :math:`\boldsymbol{\Lambda}`.
Then update:

.. math::

   \boldsymbol{\Lambda}_n
   &= \bar{\boldsymbol{\Lambda}}
   + \beta\,\mathbf{X}^\top \mathbf{W} \mathbf{X} \\
   \boldsymbol{\eta}_n
   &= \bar{\boldsymbol{\Lambda}}\,\boldsymbol{\mu}_{\text{old}}
   + \beta\,\mathbf{X}^\top \mathbf{W}\mathbf{y} \\
   \boldsymbol{\mu}_n
   &= \boldsymbol{\Lambda}_n^{-1}\,\boldsymbol{\eta}_n

The forgetting rule
:math:`\boldsymbol{\Lambda} \to \bar{\boldsymbol{\Lambda}}` is the
entire design space.  Everything else is fixed by Bayes' rule.


Exponential forgetting
----------------------

.. math::

   \bar{\boldsymbol{\Lambda}} = \gamma\,\boldsymbol{\Lambda}

A scalar multiply on the precision matrix.  It preserves sparsity
structure, introduces no parameters beyond :math:`\gamma`, and reduces
to the standard (no-forgetting) update when :math:`\gamma = 1`.

**Kalman interpretation.**
Exponential forgetting is the predict step of a Kalman filter for a
random-walk state-space model
:math:`\mathbf{w}_t = \mathbf{w}_{t-1} + \boldsymbol{\epsilon}` with
process noise
:math:`\mathbf{Q} = (1 - \gamma)\,\boldsymbol{\Sigma}`, where
:math:`\boldsymbol{\Sigma} = \boldsymbol{\Lambda}^{-1}`.

**Covariance windup.**
Each decay step scales every eigenvalue of the precision by
:math:`\gamma`.  The subsequent update adds
:math:`\beta\,\mathbf{X}^\top\mathbf{W}\mathbf{X}`, but this only
replenishes precision in directions excited by the current batch.
Directions that receive no new information lose a fraction
:math:`1 - \gamma` of their precision on every step and never recover.
Over time, those eigenvalues approach zero and the Cholesky
factorization fails.  The prior contribution :math:`\alpha\mathbf{I}`
decays along with everything else, so the model eventually loses its
regularization too.

This is the Bayesian analog of integrator windup in control theory:
the covariance accumulates uncertainty without bound in unexcited
directions.


Stabilized forgetting (Kulhavy--Zarrop)
----------------------------------------

.. math::

   \bar{\boldsymbol{\Lambda}}
   = \gamma\,\boldsymbol{\Lambda}
   + (1 - \gamma)\,\alpha\,\mathbf{I}

After scaling the precision by :math:`\gamma`, a fraction of the
prior is added back [1]_.  The term
:math:`(1 - \gamma)\,\alpha\,\mathbf{I}` is a floor: no eigenvalue
of :math:`\bar{\boldsymbol{\Lambda}}` can fall below
:math:`(1 - \gamma)\,\alpha`, regardless of how many decay steps have
occurred.

**Steady-state convergence.**
Let :math:`s_t` track the prior contribution on the precision
diagonal.  Under stabilized forgetting:

.. math::

   s_{t+1} = \gamma\,s_t + (1 - \gamma)\,\alpha

This linear recurrence has fixed point :math:`s^* = \alpha`.  Starting
from any :math:`s_0`, the prior contribution converges to
:math:`\alpha`.  The model can never become less confident than its
prior.  See :doc:`empirical-bayes` for how this interacts with
Empirical Bayes hyperparameter tuning.

**The isotropy limitation.**
Stabilized forgetting is still isotropic: every direction in parameter
space loses the same fraction of precision and gets the same floor.
If some features are excited by every observation (dense user
attributes) while others are rare (event indicators triggered a few
times a year), the rare features sit at the floor between events,
unable to accumulate precision above it.


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

The second term is a projection.  It identifies how much of the
current precision lives in the subspace excited by the batch, and
removes a fraction :math:`1 - \gamma` of it.  Directions orthogonal
to the batch are untouched; fully excited directions receive the same
:math:`\gamma` scaling as exponential forgetting [2]_.

Before applying the formula, the batch is filtered via
eigendecomposition of the Gram matrix: eigenvalues below
:math:`\varepsilon` are discarded, leaving a rank-:math:`q`
approximation :math:`\bar{\mathbf{X}}` that preserves the sufficient
statistics :math:`\mathbf{X}^\top\!\mathbf{X}` and
:math:`\mathbf{X}^\top\!\mathbf{y}` in the surviving subspace [3]_.


Precision retention
~~~~~~~~~~~~~~~~~~~

:math:`\bar{\boldsymbol{\Lambda}} \succeq \gamma\,\boldsymbol{\Lambda}`.

Directional forgetting always retains at least as much precision as
exponential forgetting.  When the batch rank :math:`q < p`, it
retains strictly more.  The bound is tight only when :math:`q = p`,
which rarely occurs in practice (Proposition 5 in [3]_).


Eigenvalue floor
~~~~~~~~~~~~~~~~

After arbitrarily many forget-update cycles:

.. math::

   \lambda_{\min}(\boldsymbol{\Lambda}_k)
   \ge \min\!\Bigl(
     \frac{\varepsilon}{1 - \gamma},\;
     \lambda_{\min}(\boldsymbol{\Lambda}_0)
   \Bigr)

No artificial prior injection is needed.  The floor emerges from the
geometry of the directional forgetting itself: only directions that
receive new information are forgotten, so precision cannot collapse
(Theorem 3 in [3]_).  Compare with stabilized forgetting, where the
floor :math:`(1 - \gamma)\,\alpha` is explicitly injected.


Computational cost
~~~~~~~~~~~~~~~~~~

**Dense.**
The inner Gram matrix
:math:`\bar{\mathbf{X}}\,\boldsymbol{\Lambda}\,\bar{\mathbf{X}}^\top`
is :math:`q \times q`, where :math:`q` is the batch rank (typically
single digits).  Its Cholesky costs :math:`O(q^3)`.  The dominant
cost is the rank-:math:`q` symmetric update to the
:math:`p \times p` precision, implemented as a BLAS-3 ``dsyrk`` call
at :math:`O(p^2 q)` -- the same asymptotic cost as the standard RLS
update.

**Sparse.**
The correction touches only features that interact with the batch in
the precision matrix's sparsity graph.  If the batch activates
:math:`k_0` columns of a sparse design matrix, the linear algebra
operates on a :math:`k_0 \times k_0` dense submatrix extracted from
the sparse precision.  The inner Cholesky is :math:`k_0 \times k_0`.
The sparse path uses pivoted Cholesky (``dpstrf``) rather than
eigendecomposition, which handles rank deficiency and is faster for
the small Gram matrices that arise in practice.


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
     - Isotropic -- decays rare features at the same rate as
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
