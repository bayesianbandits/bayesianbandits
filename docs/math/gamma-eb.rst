EmpiricalBayesGammaRegressor
============================

Tunes the Gamma prior parameters via the Negative Binomial marginal
likelihood, using Minka's EM algorithm [1]_ with stabilized
forgetting to prevent prior collapse under exponential decay.

Builds on the posterior update from :doc:`intercept-only`, which is
inherited unchanged. Read that page first.


Symbols
-------

.. list-table::
   :header-rows: 1
   :widths: 15 85

   * - Symbol
     - Meaning
   * - :math:`\alpha`
     - Gamma shape parameter (prior)
   * - :math:`\beta`
     - Gamma rate parameter (prior)
   * - :math:`\lambda_g`
     - Latent rate for group :math:`g`
   * - :math:`c_g`
     - Effective count in group :math:`g`:
       :math:`c_g = \alpha_g^{\text{post}} - \alpha`
   * - :math:`n_g`
     - Effective exposure in group :math:`g`:
       :math:`n_g = \beta_g^{\text{post}} - \beta`
   * - :math:`G`
     - Number of groups (unique values of the first feature)
   * - :math:`\psi`
     - Digamma function
   * - :math:`\psi'`
     - Trigamma function


Negative Binomial marginal likelihood
--------------------------------------

The Gamma-Poisson marginal for :math:`G` groups with effective
counts :math:`c_g` and exposures :math:`n_g` under a shared
prior :math:`\text{Gamma}(\alpha, \beta)`:

.. math::

   \log p(\text{data} \mid \alpha, \beta)
   = \sum_{g=1}^{G} \left[
     \log\Gamma(c_g + \alpha)
     - \log\Gamma(\alpha)
     - \log\Gamma(c_g + 1)
     + \alpha \log\frac{\beta}{\beta + n_g}
     + c_g \log\frac{n_g}{\beta + n_g}
   \right]


EM derivation (Minka 2002, section 2.1)
----------------------------------------

The marginal likelihood is maximized via EM, treating the
per-group rate :math:`\lambda_g` as a latent variable [1]_.


E-step
~~~~~~

Compute posterior moments of :math:`\lambda_g` under
:math:`\text{Gamma}(\alpha + c_g,\, \beta + n_g)`:

.. math::

   \mathbb{E}[\lambda_g]
   &= \frac{\alpha + c_g}{\beta + n_g} \\
   \mathbb{E}[\log \lambda_g]
   &= \psi(\alpha + c_g) - \log(\beta + n_g)


M-step
~~~~~~

Solve the Gamma MLE on expected sufficient statistics.

**Rate update** (closed-form given :math:`\alpha`):

.. math::

   \beta = \frac{\alpha}{\bar{\mathbb{E}}[\lambda]}

where :math:`\bar{\mathbb{E}}[\lambda] =
\frac{1}{G} \sum_g \mathbb{E}[\lambda_g]`.

**Shape update** (generalized Newton [1]_, section 1). The
fixed-point equation is:

.. math::

   \log \alpha - \psi(\alpha)
   = \log \bar{\mathbb{E}}[\lambda]
     - \bar{\mathbb{E}}[\log \lambda]

solved by the iteration:

.. math::

   \frac{1}{\alpha^{\text{new}}} = \frac{1}{\alpha}
   + \frac{\log \bar{\mathbb{E}}[\lambda]
     - \bar{\mathbb{E}}[\log \lambda]
     - \log \alpha + \psi(\alpha)}
        {\alpha^2 \left(\frac{1}{\alpha}
        - \psi'(\alpha)\right)}


Counts and exposure extraction
-------------------------------

The effective counts and exposures are derived from the difference
between posterior and prior parameters:

.. math::

   c_g = \alpha_g^{\text{post}} - \alpha, \qquad
   n_g = \beta_g^{\text{post}} - \beta

With stabilized forgetting (see below), this gives the decayed
counts and exposures exactly.


Stabilized forgetting
---------------------

**The problem.** Uniform decay
:math:`(\alpha_g, \beta_g) \leftarrow \gamma\,(\alpha_g, \beta_g)`
drives all parameters to zero.

**The solution.** Every decay (both explicit via ``decay()`` and
implicit via ``learning_rate`` in ``partial_fit``) re-injects the
EB-tuned prior:

.. math::

   \mathbf{p}_g \leftarrow \gamma^n \, \mathbf{p}_g
   + (1 - \gamma^n) \, \mathbf{p}^{\text{prior}}

where :math:`\mathbf{p}_g = (\alpha_g, \beta_g)` and :math:`n`
is the number of observations. This maintains the invariant:

.. math::

   \mathbf{p}_g - \mathbf{p}^{\text{prior}}
   = \gamma^n \, (c_g, n_g)

so effective counts and exposures are correctly decayed and the
prior contribution never vanishes.


fit vs. partial_fit
-------------------

``fit`` iterates the EM step to convergence on fixed data (up to
``n_eb_iter`` times, checking
:math:`|\Delta \log p| < \texttt{eb\_tol}`), then refits the
posterior with the converged prior. ``partial_fit`` runs one EM
step per call, correcting all group posteriors by the prior change
:math:`\mathbf{p}^{\text{new}} - \mathbf{p}^{\text{old}}`.


Hyperparameter semantics
-------------------------

.. list-table::
   :header-rows: 1
   :widths: 20 50 30

   * - Parameter
     - Controls
     - Practical guidance
   * - ``alpha``
     - Initial Gamma shape. EB tunes this.
     - Start at 1.0.
   * - ``beta``
     - Initial Gamma rate. EB tunes this.
     - Start at 1.0.
   * - ``n_eb_iter``
     - Maximum EM iterations during ``fit``.
     - 10 (default). Set to 0 to disable EB during ``fit``.
   * - ``eb_tol``
     - Convergence tolerance on log evidence change.
     - 1e-4 (default).
   * - ``learning_rate``
     - Decay factor :math:`\gamma`. See :doc:`/howto/decay`.
     - 1.0 (default) for stationary environments.


Robustness to misspecified priors
----------------------------------

The EB prior mean :math:`\alpha/\beta` converges to the average
group rate regardless of initialization. The prior strength
(concentration) converges more slowly but is less critical for
predictions.

.. code-block:: python

   import numpy as np
   from bayesianbandits import EmpiricalBayesGammaRegressor

   rng = np.random.default_rng(42)
   true_alpha, true_beta = 3.0, 1.0  # true mean rate = 3.0

   # Wrong initial prior: mean rate = 0.1
   model = EmpiricalBayesGammaRegressor(
       alpha=1.0, beta=10.0, random_state=0
   )

   for g in range(200):
       rate = rng.gamma(true_alpha, 1.0 / true_beta)
       obs = rng.poisson(rate, size=rng.poisson(10) + 1)
       model.partial_fit(np.full((len(obs), 1), g), obs)

   # Recovered prior mean: alpha/beta ~ 3.0
   print(f"Prior mean rate: {model.alpha / model.beta:.2f}")


References
----------

.. [1] Minka, T. P. (2002). "Estimating a Gamma distribution."
   Microsoft Research Technical Report.
