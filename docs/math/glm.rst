Bayesian GLM
============

Bayesian generalized linear model for non-Gaussian likelihoods (binary
and count data). The posterior is not conjugate; it is approximated as
a Gaussian at the MAP via Laplace approximation / IRLS.

See also: :doc:`normal` (exact conjugate, Gaussian likelihood).


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
     - Weight vector (MAP estimate stored as ``coef_``)
   * - :math:`\alpha`
     - Prior precision scalar
   * - :math:`\boldsymbol{\Lambda}`
     - Posterior precision matrix (stored as ``cov_inv_``)
   * - :math:`\boldsymbol{\eta}`
     - Linear predictor :math:`\mathbf{X}\mathbf{w}`
   * - :math:`g^{-1}`
     - Inverse link function (sigmoid or exp)
   * - :math:`\boldsymbol{\mu}`
     - Mean response :math:`g^{-1}(\boldsymbol{\eta})`
   * - :math:`\mathbf{W}`
     - Diagonal IRLS weight matrix
   * - :math:`\mathbf{z}`
     - Working response (pseudo-targets)
   * - :math:`\gamma`
     - Decay factor, in :math:`(0, 1]`


Prior
-----

.. math::

   \mathbf{w} \sim \mathcal{N}(\mathbf{0},\; \alpha^{-1}\mathbf{I})


Likelihood
----------

**Logit link** (Bernoulli):

.. math::

   p(y_i = 1 \mid \mathbf{x}_i, \mathbf{w})
   = \sigma(\mathbf{x}_i^\top \mathbf{w})
   = \frac{1}{1 + \exp(-\mathbf{x}_i^\top \mathbf{w})}

**Log link** (Poisson):

.. math::

   \mathbb{E}[y_i \mid \mathbf{x}_i, \mathbf{w}]
   = \exp(\mathbf{x}_i^\top \mathbf{w})

Neither likelihood is conjugate to the Gaussian prior.

**Reference:** Murphy (2012) Chapter 8 [1]_.


Laplace approximation via IRLS
-------------------------------

The posterior is approximated as:

.. math::

   p(\mathbf{w} \mid \mathcal{D})
   \approx \mathcal{N}(\hat{\mathbf{w}},\;
     \boldsymbol{\Lambda}^{-1})

where :math:`\hat{\mathbf{w}}` is the MAP estimate and
:math:`\boldsymbol{\Lambda}` is the Hessian of the negative
log-posterior evaluated at the MAP.

The MAP is found by iteratively reweighted least squares (IRLS). Each
iteration:

1. Compute the linear predictor and link derivatives:

   .. math::

      \boldsymbol{\eta} &= \mathbf{X}\mathbf{w} \\
      \boldsymbol{\mu} &= g^{-1}(\boldsymbol{\eta}) \\
      \frac{d\boldsymbol{\mu}}{d\boldsymbol{\eta}}
      &= \begin{cases}
        \boldsymbol{\mu}(1 - \boldsymbol{\mu}) & \text{logit} \\
        \boldsymbol{\mu} & \text{log}
      \end{cases}

2. Form IRLS weights and working response:

   .. math::

      \mathbf{W} &= \mathrm{diag}\!\left(
        \frac{d\boldsymbol{\mu}}{d\boldsymbol{\eta}}
      \right) \\
      \mathbf{z} &= \boldsymbol{\eta}
        + \frac{\mathbf{y} - \boldsymbol{\mu}}
               {d\boldsymbol{\mu}/d\boldsymbol{\eta}}

3. Solve the weighted least squares problem:

   .. math::

      \boldsymbol{\Lambda}
      &= \gamma^n\,\boldsymbol{\Lambda}_{\text{old}}
        + \mathbf{X}^\top \mathbf{W} \mathbf{X} \\
      \mathbf{w}
      &\leftarrow \boldsymbol{\Lambda}^{-1}\!\left(
        \gamma^n\,\boldsymbol{\Lambda}_{\text{old}}\,
          \mathbf{w}_{\text{old}}
        + \mathbf{X}^\top (\mathbf{W} \odot \mathbf{z})
      \right)

4. Check convergence:
   :math:`\|\mathbf{w}_{\text{new}} - \mathbf{w}_{\text{old}}\|_\infty < \texttt{tol}`.

The default runs 5 iterations per ``partial_fit`` call
(``LaplaceApproximator(n_iter=5)``). For fast online updates where the
previous posterior is a good initialization, ``n_iter=1`` (a single
Newton step) is often sufficient.


Sampling
--------

Weight vectors are drawn from the Gaussian approximation:

.. math::

   \mathbf{w}_s \sim \mathcal{N}(\hat{\mathbf{w}},\;
     \boldsymbol{\Lambda}^{-1})

Predictions are transformed through the inverse link:

.. math::

   \hat{\mathbf{y}} = g^{-1}(\mathbf{X}\,\mathbf{w}_s)

For the logit link, samples are probabilities in :math:`(0, 1)`. For
the log link, samples are positive rates.


Decay
-----

.. math::

   \boldsymbol{\Lambda} \leftarrow \gamma^n\,\boldsymbol{\Lambda}

Same as :doc:`normal`: scales the precision, widens the posterior,
mean unchanged. See :doc:`/howto/decay`.


Hyperparameter semantics
-------------------------

.. list-table::
   :header-rows: 1
   :widths: 20 50 30

   * - Parameter
     - Controls
     - Practical guidance
   * - ``alpha``
     - Prior precision. Sets
       :math:`\boldsymbol{\Lambda}_0 = \alpha\mathbf{I}`.
     - 1.0 (default). Higher values regularize more.
   * - ``link``
     - Likelihood family. ``'logit'`` for binary outcomes,
       ``'log'`` for counts.
     - Match to your outcome type.
   * - ``approximator``
     - Posterior approximation strategy. Default is
       ``LaplaceApproximator(n_iter=5, tol=1e-4)``.
     - See the IRLS section above for ``n_iter`` guidance.
   * - ``learning_rate``
     - Decay factor :math:`\gamma`. See :doc:`/howto/decay`.
     - 1.0 (default) for stationary environments.


Trade-offs vs. conjugate models
--------------------------------

The Laplace approximation buys flexible likelihoods (logistic, Poisson)
at a cost:

- The posterior is Gaussian at the MAP, not exact. Tail behavior and
  multimodality are lost.
- IRLS iterations are more expensive than a single conjugate update
  (:math:`O(n\_iter \cdot p^2 n)` vs. :math:`O(p^2 n)`).
- With ``n_iter=1``, the cost matches a conjugate update, but the
  approximation is only good when the previous posterior is close to
  the new MAP.

For Gaussian outcomes, :class:`~bayesianbandits.NormalRegressor` is
exact and cheaper. For binary or count outcomes,
:class:`~bayesianbandits.NormalRegressor` is also viable: by
Bernstein-von Mises, the linear model's posterior concentrates
correctly even under likelihood misspecification, and in practice
it performs well (see the :doc:`/notebooks/linear-bandits` example).
The GLM buys you a proper likelihood at the cost of IRLS iterations.


References
----------

.. [1] Murphy, K. P. (2012). *Machine Learning: A Probabilistic
   Perspective*, Chapter 8. MIT Press.
