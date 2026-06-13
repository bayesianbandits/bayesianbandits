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


R-VGA: expected-curvature alternative
--------------------------------------

The Laplace approximation evaluates curvature at a point estimate
(the MAP). R-VGA (Lambert, Bonnabel, Bach 2022 [2]_) replaces this
with the *expected curvature* under the current approximate posterior,
correcting a systematic bias for non-Gaussian likelihoods.

Each IRLS iteration replaces point-estimate weights with expected
weights:

.. math::

   \tilde{\mathbf{W}} = \mathrm{diag}\!\left(
     \mathbb{E}_{q(\mathbf{w})}\!\left[
       \frac{d\boldsymbol{\mu}}{d\boldsymbol{\eta}}
     \right]
   \right)

where the expectation is over the current Gaussian approximation
:math:`q(\mathbf{w}) = \mathcal{N}(\hat{\mathbf{w}},
\boldsymbol{\Lambda}^{-1})`.
For each observation, this reduces to a univariate integral
:math:`\mathbb{E}[\,h'(\eta_i)\,]` with
:math:`\eta_i \sim \mathcal{N}(m_i, v_i)`, where
:math:`m_i = \mathbf{x}_i^\top\hat{\mathbf{w}}` and
:math:`v_i = \mathbf{x}_i^\top\boldsymbol{\Lambda}^{-1}\mathbf{x}_i`.

**Computing expected weights.** The path depends on the link:

- **Log link** (exact): the lognormal mean gives
  :math:`\mathbb{E}[\exp(\eta_i)] = \exp(m_i + v_i/2)` in closed
  form, and for the log link the expected response and the expected
  Fisher weight coincide. No quadrature is needed.
- **Probit approximation** (default for logit link): the MacKay
  probit approximation to the logistic sigmoid.
  :math:`k_i = \beta / \sqrt{v_i + \beta^2}` with
  :math:`\beta = \sqrt{8/\pi}`, giving
  :math:`\mathbb{E}[\sigma(\eta_i)] \approx \sigma(k_i m_i)`.
  Within 5% of Gauss-Hermite and avoids the quadrature cost.
- **Gauss-Hermite quadrature** (logit link with
  ``use_probit=False``): evaluates the integrand at ``n_gh_nodes``
  points. 20 nodes suffice in practice.

**Effect on the posterior.** For logit link, the expected weight
:math:`\mathbb{E}[\sigma'(\eta)]` is always :math:`\le \sigma'(\hat\eta)` when
the posterior mean is near zero (because :math:`\sigma'` peaks at
:math:`\eta = 0` and posterior variance spreads mass into the
tails). This yields wider posteriors than Laplace, correcting the
well-known underestimation of posterior uncertainty.

**When to use R-VGA.** Laplace's systematically narrow posteriors
cause underexploration in Thompson sampling; R-VGA's better-calibrated
posteriors explore more early on and converge to better policies. The
per-coordinate effect is percent-level, but it compounds: each update's
posterior is the next update's prior, and Thompson sampling draws all
:math:`p` coordinates jointly. In the demonstration notebook
(:doc:`/notebooks/rvga-glm`), a 5-arm bandit with :math:`p = 20` and
single-observation updates accumulates about 6% less cumulative regret
over 300 rounds (paired difference over 50 simulations,
:math:`2.1 \pm 0.4` s.e.). The cost is
:math:`O(p^2 n)` per iteration for the predictive variance solve; at
:math:`p = 1000`, the :math:`O(p^3)` Cholesky dominates and the
overhead is negligible.

**Large sparse batches.** The sparse path computes predictive
variances through an :math:`n \times n` Gram matrix, so a single
joint update over a large batch would allocate :math:`8 n^2` bytes.
Batches larger than ``batch_size`` (default 2048, capping the Gram at
roughly 34 MB) are therefore processed as *sequential* minibatch
updates, threading each chunk's posterior into the next chunk's
prior. This is R-VGA's native recursive formulation: decay and sample
weights compose exactly across chunks, and in sparse high-dimensional
regimes the result is nearly indistinguishable from the joint update
(coefficient differences on the order of :math:`10^{-3}` in our
benchmarks). The one semantic change is that the posterior becomes
mildly dependent on row order, as in any recursive update. Set
``batch_size=None`` to force a single joint update.

Usage::

    from bayesianbandits import BayesianGLM, RVGAApproximator

    model = BayesianGLM(
        link="logit",
        approximator=RVGAApproximator(),  # probit path, 5 IRLS iters
    )

    # Or with Gauss-Hermite for maximum accuracy:
    model = BayesianGLM(
        link="logit",
        approximator=RVGAApproximator(use_probit=False, n_gh_nodes=20),
    )


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
     - ``RVGAApproximator()`` for better-calibrated uncertainty.
       See the R-VGA section above.
   * - ``learning_rate``
     - Decay factor :math:`\gamma`. See :doc:`/howto/decay`.
     - 1.0 (default) for stationary environments.


Trade-offs
----------

**Laplace vs. R-VGA.** Laplace is faster (no predictive variance
solve) but systematically underestimates posterior width for logit
link. R-VGA corrects this at a cost of roughly 1.5--5x slower ``fit``
(the overhead shrinks as :math:`p` grows because the
:math:`O(p^3)` Cholesky dominates). When calibrated uncertainty
matters, R-VGA is the better default.

**GLM vs. conjugate models.** Both approximations buy flexible
likelihoods (logistic, Poisson) at a cost:

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

.. [2] Lambert, M., Bonnabel, S. & Bach, F. (2022). The recursive
   variational Gaussian approximation (R-VGA). *Statistics and
   Computing*, 32, 10.
