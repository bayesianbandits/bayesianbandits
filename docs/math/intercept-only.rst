Intercept-Only Models
=====================

Two conjugate models for problems without covariates: one for
categorical outcomes, one for rates. Both are stratified by the first
feature value, maintaining an independent posterior per group.


DirichletClassifier
-------------------

Conjugate Dirichlet-Multinomial model for binary or categorical
outcomes (click/no-click, class selection).


Symbols
~~~~~~~

.. list-table::
   :header-rows: 1
   :widths: 15 85

   * - Symbol
     - Meaning
   * - :math:`K`
     - Number of classes
   * - :math:`\alpha_k`
     - Concentration parameter for class :math:`k`
   * - :math:`\theta_k`
     - Probability of class :math:`k`
   * - :math:`w_i`
     - Sample weight for observation :math:`i`
   * - :math:`\gamma`
     - Decay factor


Prior and likelihood
~~~~~~~~~~~~~~~~~~~~

.. math::

   \boldsymbol{\theta} \sim \mathrm{Dirichlet}(\alpha_1, \ldots, \alpha_K)

.. math::

   y_i \mid \boldsymbol{\theta} \sim \mathrm{Categorical}(\boldsymbol{\theta})


Update
~~~~~~

.. math::

   \alpha_k^{\text{post}}
   = \alpha_k^{\text{prior}}
   + \sum_{i=1}^{N} w_i \, \mathbb{1}[y_i = k]


Posterior mean
~~~~~~~~~~~~~~

.. math::

   \mathbb{E}[\theta_k]
   = \frac{\alpha_k}{\sum_j \alpha_j}


Sampling
~~~~~~~~

.. math::

   \boldsymbol{\theta} \sim \mathrm{Dirichlet}(\alpha_1^{\text{post}},
     \ldots, \alpha_K^{\text{post}})

via ``scipy.stats.dirichlet``.


Decay
~~~~~

.. math::

   \alpha_k \leftarrow \gamma\, \alpha_k \quad \forall k

Scaling all concentrations uniformly preserves the mean
:math:`\alpha_k / \sum_j \alpha_j` but increases posterior variance.

**Reference:** Murphy (2012) Chapter 3 [1]_.


GammaRegressor
--------------

Conjugate Gamma-Poisson model for count or rate data (transactions
per period, events per session).


Symbols
~~~~~~~

.. list-table::
   :header-rows: 1
   :widths: 15 85

   * - Symbol
     - Meaning
   * - :math:`\lambda`
     - Rate parameter (random)
   * - :math:`\alpha`
     - Gamma shape parameter
   * - :math:`\beta`
     - Gamma rate parameter (inverse scale)
   * - :math:`w_i`
     - Sample weight for observation :math:`i`
   * - :math:`\gamma`
     - Decay factor


Prior and likelihood
~~~~~~~~~~~~~~~~~~~~

.. math::

   \lambda \sim \mathrm{Gamma}(\alpha, \beta)

.. math::

   y_i \mid \lambda \sim \mathrm{Poisson}(\lambda)


Update
~~~~~~

.. math::

   \alpha^{\text{post}}
   &= \alpha^{\text{prior}} + \sum_{i=1}^{N} w_i\, y_i \\
   \beta^{\text{post}}
   &= \beta^{\text{prior}} + \sum_{i=1}^{N} w_i


Posterior moments
~~~~~~~~~~~~~~~~~

.. math::

   \mathbb{E}[\lambda] = \frac{\alpha}{\beta}, \qquad
   \mathrm{Var}[\lambda] = \frac{\alpha}{\beta^2}


Sampling
~~~~~~~~

.. math::

   \lambda \sim \mathrm{Gamma}(\alpha^{\text{post}},\,
     \beta^{\text{post}})

via ``scipy.stats.gamma`` with ``scale = 1/beta``.


Decay
~~~~~

.. math::

   \alpha \leftarrow \gamma\,\alpha, \qquad
   \beta \leftarrow \gamma\,\beta

Both parameters scale equally, so the mean :math:`\alpha/\beta` is
preserved and the variance :math:`\alpha/\beta^2` increases.

**Reference:** Murphy (2012) Chapter 3 [1]_.


When to use each
-----------------

Use :class:`~bayesianbandits.DirichletClassifier` when the outcome is
categorical (binary conversions, multi-class selections). Use
:class:`~bayesianbandits.GammaRegressor` when the outcome is a
positive count or rate.

Both are intercept-only: each unique value of the first feature gets
its own independent posterior. They cannot condition on covariates. For
problems with covariates, use :class:`~bayesianbandits.NormalRegressor`
or :class:`~bayesianbandits.BayesianGLM`.


References
----------

.. [1] Murphy, K. P. (2012). *Machine Learning: A Probabilistic
   Perspective*, Chapter 3. MIT Press.
