EmpiricalBayesDirichletClassifier
==================================

Tunes the Dirichlet prior concentration parameters via
Minka's fixed-point iteration [1]_ for the Dirichlet-Multinomial
marginal likelihood, with stabilized forgetting to prevent prior
collapse under exponential decay.

Builds on the posterior update from :doc:`intercept-only`, which is
inherited unchanged. Read that page first.


Symbols
-------

.. list-table::
   :header-rows: 1
   :widths: 15 85

   * - Symbol
     - Meaning
   * - :math:`\boldsymbol{\alpha}`
     - Dirichlet prior concentration parameters, length :math:`K`
   * - :math:`s`
     - Scalar concentration :math:`s = \sum_k \alpha_k`
   * - :math:`\mathbf{m}`
     - Base measure :math:`m_k = \alpha_k / s`,
       :math:`\sum_k m_k = 1`
   * - :math:`c_{gk}`
     - Effective count of class :math:`k` in group :math:`g`
   * - :math:`c_g`
     - Total effective counts in group :math:`g`:
       :math:`c_g = \sum_k c_{gk}`
   * - :math:`G`
     - Number of groups (unique values of the first feature)
   * - :math:`K`
     - Number of classes
   * - :math:`\psi`
     - Digamma function


Dirichlet-Multinomial marginal likelihood
-----------------------------------------

The log marginal likelihood (evidence) for :math:`G` groups with
effective counts :math:`c_{gk}` under prior
:math:`\boldsymbol{\alpha}`:

.. math::

   \log p(\text{data} \mid \boldsymbol{\alpha})
   = \sum_{g=1}^{G} \left[
     \log\Gamma(s)
     - \log\Gamma(c_g + s)
     + \sum_{k=1}^{K} \left(
       \log\Gamma(c_{gk} + \alpha_k)
       - \log\Gamma(\alpha_k)
     \right)
   \right]


(s, m) decomposition
--------------------

The prior is decomposed as :math:`\alpha_k = s \cdot m_k` where
:math:`s` controls overall prior strength and :math:`\mathbf{m}`
controls prior shape. Optimizing :math:`s` and :math:`\mathbf{m}` in
alternation converges faster than optimizing
:math:`\boldsymbol{\alpha}` directly [1]_.


m update (base measure)
~~~~~~~~~~~~~~~~~~~~~~~

Fixing :math:`s`, the per-component update is:

.. math::

   \alpha_k^{\text{new}} = \alpha_k \cdot
   \frac{\sum_g [\psi(c_{gk} + \alpha_k) - \psi(\alpha_k)]}
        {\sum_g [\psi(c_g + s) - \psi(s)]}

Then renormalize: :math:`m_k^{\text{new}} = \alpha_k^{\text{new}}
/ \sum_j \alpha_j^{\text{new}}`.


s update (concentration)
~~~~~~~~~~~~~~~~~~~~~~~~

Fixing :math:`\mathbf{m}`, the scalar update is:

.. math::

   s^{\text{new}} = s \cdot
   \frac{\sum_g \sum_k m_k [\psi(c_{gk} + s \cdot m_k)
         - \psi(s \cdot m_k)]}
        {\sum_g [\psi(c_g + s) - \psi(s)]}

Each (m, s) step is a lower-bound maximization, so the log marginal
likelihood is non-decreasing across iterations.


Counts extraction
-----------------

The effective counts are derived from the difference between posterior
and prior concentrations:

.. math::

   c_{gk} = \alpha_{gk}^{\text{post}} - \alpha_k^{\text{prior}}

With stabilized forgetting (see below), this gives the decayed counts
exactly.


Stabilized forgetting
---------------------

**The problem.** Uniform decay
:math:`\boldsymbol{\alpha}_g \leftarrow \gamma \,
\boldsymbol{\alpha}_g` drives all concentrations to zero.

**The solution.** Every decay (both explicit via ``decay()`` and
implicit via ``learning_rate`` in ``partial_fit``) re-injects the
EB-tuned prior:

.. math::

   \boldsymbol{\alpha}_g \leftarrow \gamma^n \, \boldsymbol{\alpha}_g
   + (1 - \gamma^n) \, \boldsymbol{\alpha}^{\text{prior}}

where :math:`n` is the number of observations. This maintains the
invariant:

.. math::

   \boldsymbol{\alpha}_g - \boldsymbol{\alpha}^{\text{prior}}
   = \gamma^n \, \mathbf{c}_g

so the effective counts are correctly decayed and the prior
contribution never vanishes.


fit vs. partial_fit
-------------------

``fit`` iterates the Minka step to convergence on fixed data (up to
``n_eb_iter`` times, checking
:math:`|\Delta \log p| < \texttt{eb\_tol}`), then refits the
posterior with the converged prior. ``partial_fit`` runs one step per
call, correcting all group posteriors by the prior change
:math:`\boldsymbol{\alpha}^{\text{new}} -
\boldsymbol{\alpha}^{\text{old}}`.


Hyperparameter semantics
-------------------------

.. list-table::
   :header-rows: 1
   :widths: 20 50 30

   * - Parameter
     - Controls
     - Practical guidance
   * - ``alphas``
     - Initial prior concentrations. EB tunes these.
     - Start uniform (e.g. ``{0: 1, 1: 1}``).
   * - ``n_eb_iter``
     - Maximum EB iterations during ``fit``.
     - 10 (default). Set to 0 to disable EB during ``fit``.
   * - ``eb_tol``
     - Convergence tolerance on log evidence change.
     - 1e-4 (default).
   * - ``learning_rate``
     - Decay factor :math:`\gamma`. See :doc:`/howto/decay`.
     - 1.0 (default) for stationary environments.


Robustness to misspecified priors
----------------------------------

The base measure :math:`\mathbf{m}` converges to the true shape
regardless of initialization. The scalar concentration :math:`s`
converges more slowly (a known property of Dirichlet MLE), but
:math:`\mathbf{m}` determines the relative class weights, which is
what matters for predictions.

.. code-block:: python

   import numpy as np
   from bayesianbandits import EmpiricalBayesDirichletClassifier

   rng = np.random.default_rng(42)
   true_alpha = np.array([3.0, 1.0])  # 75/25 split

   # Wrong initial prior: heavily favors class 2
   clf = EmpiricalBayesDirichletClassifier(
       {1: 1.0, 2: 5.0}, random_state=0
   )

   for g in range(200):
       theta = rng.dirichlet(true_alpha)
       obs = rng.choice([1, 2], size=rng.poisson(10) + 1, p=theta)
       clf.partial_fit(np.full((len(obs), 1), g), obs)

   # Recovered base measure: m ≈ [0.75, 0.25]
   s = sum(clf.alphas.values())
   m = {k: v / s for k, v in clf.alphas.items()}


References
----------

.. [1] Minka, T. P. (2000; revised 2003, 2012). "Estimating a
   Dirichlet distribution." Technical report, MIT.
