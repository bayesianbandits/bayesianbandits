NormalInverseGammaRegressor
===========================

Bayesian linear regression with unknown noise variance. Places a
joint Normal-Inverse-Gamma prior on the weights and noise variance.
The marginal posterior over weights is a multivariate t, giving
heavier-tailed uncertainty when data is scarce.

See also: :doc:`normal` (known variance),
:doc:`empirical-bayes` (automatic hyperparameter tuning).


Symbols
-------

.. list-table::
   :header-rows: 1
   :widths: 15 85

   * - Symbol
     - Meaning
   * - :math:`p`
     - Number of features
   * - :math:`n`
     - Number of observations in a batch
   * - :math:`\mathbf{w}`
     - Weight vector (posterior mean stored as ``coef_``)
   * - :math:`\sigma^2`
     - Noise variance (random, integrated out)
   * - :math:`\boldsymbol{\mu}_0`
     - Prior mean of the weights
   * - :math:`\boldsymbol{\Lambda}_0`
     - Prior precision matrix of the weights (conditioned on
       :math:`\sigma^2`)
   * - :math:`a_0, b_0`
     - Prior shape and rate of the Inverse-Gamma on :math:`\sigma^2`
   * - :math:`\boldsymbol{\Lambda}_n`
     - Posterior precision matrix (stored as ``cov_inv_``)
   * - :math:`\boldsymbol{\mu}_n`
     - Posterior mean (stored as ``coef_``)
   * - :math:`a_n, b_n`
     - Posterior shape and rate (stored as ``a_``, ``b_``)
   * - :math:`\gamma`
     - Decay factor, in :math:`(0, 1]`
   * - :math:`w_i`
     - Effective sample weight for observation :math:`i`


Generative model
----------------

**Joint prior:**

.. math::

   \mathbf{w} \mid \sigma^2
   &\sim \mathcal{N}(\boldsymbol{\mu}_0,\;
     \sigma^2 \boldsymbol{\Lambda}_0^{-1}) \\
   \sigma^2
   &\sim \mathrm{IG}(a_0,\, b_0)

Note the covariance of :math:`\mathbf{w}` scales with
:math:`\sigma^2`.

**Likelihood:**

.. math::

   y_i \mid \mathbf{x}_i, \mathbf{w}, \sigma^2
   \sim \mathcal{N}(\mathbf{x}_i^\top \mathbf{w},\; \sigma^2)

**Reference:** Murphy (2012) Chapter 7 [1]_.


Update equations
----------------

Posterior update:

.. math::

   \boldsymbol{\Lambda}_n
   = \gamma^n\,\boldsymbol{\Lambda}_{\text{old}}
     + \mathbf{X}^\top \mathbf{W} \mathbf{X}

.. math::

   \boldsymbol{\mu}_n
   = \boldsymbol{\Lambda}_n^{-1}\!\left(
     \gamma^n\,\boldsymbol{\Lambda}_{\text{old}}\,
       \boldsymbol{\mu}_{\text{old}}
     + \mathbf{X}^\top \mathbf{W} \mathbf{y}
   \right)

.. math::

   a_n = \gamma^n\, a_{\text{old}}
     + \tfrac{1}{2} \sum_i w_i

.. math::

   b_n = \gamma^n\, b_{\text{old}}
     + \tfrac{1}{2}\!\left(
       \mathbf{y}^\top (\mathbf{W} \odot \mathbf{y})
       + \gamma^n\,\boldsymbol{\mu}_{\text{old}}^\top
         \boldsymbol{\Lambda}_{\text{old}}\,
         \boldsymbol{\mu}_{\text{old}}
       - \boldsymbol{\mu}_n^\top
         \boldsymbol{\Lambda}_n\,\boldsymbol{\mu}_n
     \right)

where :math:`\mathbf{W} = \mathrm{diag}(w_1, \ldots, w_n)` contains
effective sample weights (incorporating within-batch decay, as in
:doc:`normal`). When :math:`\gamma = 1` and all weights are 1, these
reduce to the standard NIG conjugate update.

Note that :math:`\beta` does not appear in the precision update
(contrast with :doc:`normal`). The noise precision
:math:`1/\sigma^2` is unknown and integrated out; the precision matrix
here is the *conditional* precision given :math:`\sigma^2`, not the
marginal precision.


Marginal posterior
------------------

Integrating out :math:`\sigma^2`, the marginal posterior of the
weights is a multivariate t:

.. math::

   \mathbf{w} \mid \mathcal{D}
   \sim t_{2a_n}\!\left(
     \boldsymbol{\mu}_n,\;
     \frac{b_n}{a_n}\,\boldsymbol{\Lambda}_n^{-1}
   \right)

with :math:`2a_n` degrees of freedom, location
:math:`\boldsymbol{\mu}_n`, and shape covariance
:math:`(b_n / a_n)\,\boldsymbol{\Lambda}_n^{-1}`.

With little data, :math:`a_n` is small and the tails are heavy.


Sampling
--------

The ``sample`` method draws from the multivariate t:

1. Draw :math:`u \sim \chi^2(2a_n) / (2a_n)`
2. Draw :math:`\mathbf{z} \sim \mathcal{N}(\mathbf{0},\,
   (b_n/a_n)\,\boldsymbol{\Lambda}_n^{-1})`
3. Return :math:`\boldsymbol{\mu}_n + \mathbf{z} / \sqrt{u}`

Predictions are :math:`\hat{\mathbf{y}} = \mathbf{X}\,\mathbf{w}_s`
as in :doc:`normal`. No observation noise is added.


Decay
-----

Standalone ``decay`` scales all three posterior statistics:

.. math::

   \boldsymbol{\Lambda} &\leftarrow \gamma^n\,\boldsymbol{\Lambda} \\
   a &\leftarrow \gamma^n\, a \\
   b &\leftarrow \gamma^n\, b

The mean is unchanged. The t distribution widens: fewer effective
degrees of freedom and higher scale.

See :doc:`/howto/decay` for practical guidance.


Hyperparameter semantics
-------------------------

.. list-table::
   :header-rows: 1
   :widths: 20 50 30

   * - Parameter
     - Controls
     - Practical guidance
   * - ``mu``
     - Prior mean of the weights.
     - 0.0 (default). A scalar is broadcast to all features.
   * - ``lam``
     - Prior precision of the weights (conditioned on
       :math:`\sigma^2`). Scalar gives :math:`\lambda \mathbf{I}`,
       vector gives :math:`\mathrm{diag}(\boldsymbol{\lambda})`.
     - 1.0 (default). Higher values regularize more strongly.
   * - ``a``
     - Prior shape of the IG on :math:`\sigma^2`. Controls how
       informative the prior variance estimate is.
     - 0.1 (default). Small values give a diffuse prior on the noise
       level.
   * - ``b``
     - Prior rate of the IG. The prior mean of :math:`\sigma^2` is
       :math:`b/(a-1)` for :math:`a > 1`.
     - 0.1 (default).
   * - ``learning_rate``
     - Decay factor :math:`\gamma`. See :doc:`/howto/decay`.
     - 1.0 (default) for stationary environments.


References
----------

.. [1] Murphy, K. P. (2012). *Machine Learning: A Probabilistic
   Perspective*, Chapter 7. MIT Press.
