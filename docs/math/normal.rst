NormalRegressor
===============

Bayesian linear regression with known noise variance (known
:math:`\beta`). Gaussian prior on weights, exact conjugate updates in
the precision parameterization.

See also: :doc:`normal-inverse-gamma` (unknown variance),
:doc:`empirical-bayes` (automatic hyperparameter tuning).


Symbols
-------

.. list-table::
   :header-rows: 1
   :widths: 15 85

   * - Symbol
     - Meaning
   * - :math:`p`
     - Number of features (dimensionality of the weight vector)
   * - :math:`n`
     - Number of observations in a batch
   * - :math:`\mathbf{w}`
     - Weight vector (posterior mean stored as ``coef_``)
   * - :math:`\alpha`
     - Prior precision scalar
   * - :math:`\beta`
     - Noise precision (inverse noise variance, :math:`1/\sigma^2`)
   * - :math:`\boldsymbol{\Lambda}`
     - Posterior precision matrix (stored as ``cov_inv_``)
   * - :math:`\boldsymbol{\mu}`
     - Posterior mean of the weight vector (stored as ``coef_``)
   * - :math:`\gamma`
     - Learning rate / decay factor, in :math:`(0, 1]`
   * - :math:`\mathbf{W}`
     - Diagonal matrix of effective sample weights
   * - :math:`w_i`
     - User-supplied sample weight for observation :math:`i`


Generative model
----------------

**Prior:**

.. math::

   \mathbf{w} \sim \mathcal{N}(\mathbf{0},\; \alpha^{-1} \mathbf{I})

The prior precision matrix is :math:`\boldsymbol{\Lambda}_0 = \alpha \mathbf{I}`
and the prior mean is :math:`\boldsymbol{\mu}_0 = \mathbf{0}`.

**Likelihood:**

.. math::

   y_i \mid \mathbf{x}_i, \mathbf{w}
   \sim \mathcal{N}(\mathbf{x}_i^\top \mathbf{w},\; \beta^{-1})

Here :math:`\beta` is the noise *precision* (inverse variance), not the
variance itself.

**References:** Bishop (2006) Section 3.3 [1]_, Murphy (2012) Chapter 7 [2]_.


Update equations
----------------

The posterior is Gaussian:
:math:`\mathbf{w} \mid \mathcal{D} \sim \mathcal{N}(\boldsymbol{\mu}_n, \boldsymbol{\Lambda}_n^{-1})`.

**Precision update:**

.. math::

   \boldsymbol{\Lambda}_n
   = \gamma^n \,\boldsymbol{\Lambda}_{\text{old}}
   + \beta\, \mathbf{X}^\top \mathbf{W} \mathbf{X}

**Mean update (via the information vector):**

.. math::

   \boldsymbol{\eta}_n
   &= \gamma^n \,\boldsymbol{\Lambda}_{\text{old}}\,\boldsymbol{\mu}_{\text{old}}
   + \beta\, \mathbf{X}^\top \mathbf{W} \mathbf{y} \\
   \boldsymbol{\mu}_n
   &= \boldsymbol{\Lambda}_n^{-1}\, \boldsymbol{\eta}_n

The implementation stores :math:`\boldsymbol{\Lambda}` and computes
:math:`\boldsymbol{\eta}` as an intermediate; the mean is recovered
by a single Cholesky solve.

When :math:`\gamma = 1` (the default), these reduce to the standard
conjugate update.


Effective weights within a batch
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

When :math:`\gamma < 1` and a batch of :math:`n` observations is
processed in a single ``partial_fit`` call, each observation receives an
effective weight that depends on its position in the batch:

.. math::

   w_{\text{eff}, i}
   = w_i \cdot \gamma^{\,n - 1 - i},
   \qquad i = 0, 1, \ldots, n{-}1

Earlier observations in the batch are decayed more than later ones.
This ensures that processing a batch of :math:`n` observations in one
call gives the same posterior as processing them one at a time with
decay between each. When :math:`\gamma = 1`, all effective weights
equal the user-supplied weights (or 1 if none are provided).


Sampling
--------

The ``sample`` method draws weight vectors from the posterior and
projects them through the design matrix:

.. math::

   \mathbf{w}_s \sim \mathcal{N}(\boldsymbol{\mu}_n,\;
   \boldsymbol{\Lambda}_n^{-1}),
   \qquad
   \hat{\mathbf{y}} = \mathbf{X}\,\mathbf{w}_s

Sampling from the precision parameterization uses
:math:`\mathbf{w}_s = \boldsymbol{\mu}_n + \mathbf{L}^{-\top}\mathbf{z}`
where :math:`\mathbf{L}\mathbf{L}^\top = \boldsymbol{\Lambda}_n` and
:math:`\mathbf{z} \sim \mathcal{N}(\mathbf{0}, \mathbf{I})`.

.. note::

   ``sample`` produces draws of the *expected reward*
   :math:`\mathbf{X}\mathbf{w}_s`, not noisy reward realizations.
   Observation noise :math:`\varepsilon \sim \mathcal{N}(0, \beta^{-1})`
   is not added. Thompson sampling needs samples of the expected value
   under parameter uncertainty, not noisy outcomes.


Standalone decay
----------------

The ``decay`` method scales the precision matrix without observing new
data:

.. math::

   \boldsymbol{\Lambda} \leftarrow \gamma^n \,\boldsymbol{\Lambda}

where :math:`n` is the number of rows in the ``X`` argument. The
posterior mean is unchanged because
:math:`(\gamma^n \boldsymbol{\Lambda})^{-1}(\gamma^n \boldsymbol{\Lambda}\,\boldsymbol{\mu})
= \boldsymbol{\mu}`.

This is equivalent to a random-walk state-space model where the
transition shrinks the effective precision by :math:`\gamma` per time
step. See :doc:`/howto/decay` for practical guidance.


Hyperparameter semantics
-------------------------

.. list-table::
   :header-rows: 1
   :widths: 20 50 30

   * - Parameter
     - Controls
     - Practical guidance
   * - ``alpha``
     - Prior precision. Sets :math:`\boldsymbol{\Lambda}_0 = \alpha \mathbf{I}`.
       Higher values give stronger regularization toward zero.
     - Start with 1.0. Range: 0.01 to 100.
   * - ``beta``
     - Noise precision :math:`1/\sigma^2`. Scales the data
       contribution to the precision matrix. Higher values mean
       observations are trusted more (lower noise).
     - Set to :math:`1/\sigma^2` if the noise scale is known.
       Otherwise consider
       :class:`~bayesianbandits.NormalInverseGammaRegressor`.
   * - ``learning_rate``
     - Decay factor :math:`\gamma`. Applied per observation during
       ``partial_fit`` and per row during ``decay``.
     - 1.0 (default) for stationary environments. See
       :doc:`/howto/decay` for tuning advice.


References
----------

.. [1] Bishop, C. M. (2006). *Pattern Recognition and Machine
   Learning*, Section 3.3. Springer.

.. [2] Murphy, K. P. (2012). *Machine Learning: A Probabilistic
   Perspective*, Chapter 7. MIT Press.
