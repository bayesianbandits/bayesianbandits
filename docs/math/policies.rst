Exploration Policies
====================

A policy maps posterior samples to arm selections. All four share a
common interface:

1. Draw posterior predictive samples for each arm:
   :math:`\tilde{\theta}_a \sim p(\theta_a \mid \mathcal{D}_a)`,
   then :math:`\tilde{r}_a = g_a(\tilde{\theta}_a, \mathbf{x}_t)`
   where :math:`g_a` is the reward function.
2. Compute a decision statistic from those samples (policy-specific).
3. Select the arm that maximizes the statistic.

The update rule is shared by Thompson sampling, UCB, and
epsilon-greedy: call ``arm.update(X, y)`` directly. EXP3A overrides
this with importance-weighted updates.

.. note::

   The regret bounds listed below are proven for stationary,
   non-contextual bandits with exact conjugate posteriors. This library
   targets contextual, non-stationary settings with approximate
   posteriors and decay. The formal guarantees do not directly apply.

.. list-table:: Samples per arm
   :header-rows: 1
   :widths: 30 20 50

   * - Policy
     - Samples
     - Reason
   * - :class:`~bayesianbandits.ThompsonSampling`
     - 1
     - One posterior draw suffices for probability matching
   * - :class:`~bayesianbandits.UpperConfidenceBound`
     - 1000
     - Monte Carlo quantile estimation needs many draws
   * - :class:`~bayesianbandits.EpsilonGreedy`
     - 1000
     - Accurate posterior mean via Monte Carlo
   * - :class:`~bayesianbandits.EXP3A`
     - 100
     - Only needs mean estimates; 100 draws are typically sufficient


Thompson Sampling
-----------------

**Selection rule.** Draw a single sample from each arm's posterior
predictive and select the arm with the highest sample:

.. math::

   \tilde{\theta}_a \sim p(\theta_a \mid \mathcal{D}_a)
   \quad \forall a, \qquad
   a^* = \arg\max_a \; g_a(\tilde{\theta}_a, \mathbf{x}_t)

This is *probability matching*: each arm is selected with probability
equal to its posterior probability of being optimal.

**Hyperparameters.** None.

**Update rule.** Default: ``arm.update(X, y)``.

**Regret bounds.** For the :math:`K`-armed stochastic bandit with
exact conjugate posteriors [1]_ [2]_:

.. math::

   \mathbb{E}[\mathrm{Regret}(T)]
   = O\!\left(\sqrt{KT \ln K}\right)

with an asymptotically optimal problem-dependent bound of
:math:`O\!\left(\sum_{a:\Delta_a>0} \frac{\ln T}{\Delta_a}\right)`
matching the Lai-Robbins lower bound.

Upper Confidence Bound
----------------------

Strictly, this is an upper *credible* bound (a posterior quantile),
not the frequentist confidence bound of classical UCB. The name is
conventional.

**Selection rule.** Compute the :math:`\alpha`-quantile of each arm's
posterior predictive and select the arm with the highest quantile:

.. math::

   a^* = \arg\max_a \; Q_\alpha\!\left(
     g_a(\theta_a, \mathbf{x}_t) \mid \mathcal{D}_a
   \right)

The quantile is estimated via Monte Carlo: draw :math:`S` samples per
arm, compute ``np.quantile(samples, alpha)``, then argmax.

**Hyperparameters.**

.. list-table::
   :header-rows: 1
   :widths: 20 50 30

   * - Parameter
     - Controls
     - Practical guidance
   * - ``alpha``
     - Quantile level (default 0.68). Controls optimism.
       :math:`\alpha = 0.5` is the median (greedy on the posterior
       mean). :math:`\alpha = 0.68` is roughly one standard deviation
       for Gaussian posteriors. :math:`\alpha = 0.95` is aggressive
       exploration.
     - Start with 0.68. Increase for more exploration.
   * - ``samples``
     - Number of Monte Carlo draws per arm (default 1000).
     - 1000 is usually sufficient. Reduce for lower latency if
       quantile accuracy is not critical.

**Update rule.** Default: ``arm.update(X, y)``.

**Regret bounds.** With a quantile schedule
:math:`\alpha_t = 1 - 1/(t \log^c(T))`, Bayesian UCB achieves [3]_:

.. math::

   \mathbb{E}[\mathrm{Regret}(T)]
   = O\!\left(\sum_{a:\Delta_a>0}
   \frac{\ln T}{\Delta_a}\right)

With a fixed :math:`\alpha` (as implemented), the problem-dependent
bound is not guaranteed, but a minimax rate of
:math:`O(\sqrt{KT \ln T})` holds under mild conditions.

Epsilon-Greedy
--------------

**Selection rule.** Exploit the posterior mean with probability
:math:`1 - \varepsilon`, explore uniformly at random with probability
:math:`\varepsilon`:

.. math::

   a_t =
   \begin{cases}
   \displaystyle\arg\max_a \; \hat{\mu}_a(\mathbf{x}_t)
     & \text{with probability } 1 - \varepsilon \\[4pt]
   \text{Uniform}\{1, \ldots, K\}
     & \text{with probability } \varepsilon
   \end{cases}

where :math:`\hat{\mu}_a(\mathbf{x}_t)` is the Monte Carlo posterior
mean for arm :math:`a` in context :math:`\mathbf{x}_t`.

**Hyperparameters.**

.. list-table::
   :header-rows: 1
   :widths: 20 50 30

   * - Parameter
     - Controls
     - Practical guidance
   * - ``epsilon``
     - Exploration probability (default 0.1).
       :math:`\varepsilon = 0` is pure greedy (no exploration).
       :math:`\varepsilon = 1` is pure random.
     - 0.01 to 0.2 is typical. Start with 0.1.
   * - ``samples``
     - Number of Monte Carlo draws per arm (default 1000).
     - 1000 is usually sufficient for accurate mean estimation.

**Update rule.** Default: ``arm.update(X, y)``.

**Regret bounds.** For fixed :math:`\varepsilon` [4]_:

.. math::

   \mathbb{E}[\mathrm{Regret}(T)]
   \le \varepsilon\,T\,\Delta_{\max}
   + \sum_{a:\Delta_a>0} \frac{C}{\Delta_a}

With a decaying schedule :math:`\varepsilon_t = O(K/t)`, the minimax
rate :math:`O(\sqrt{KT})` is achievable. This implementation uses a
fixed :math:`\varepsilon` (no schedule).


EXP3A
------

Standard EXP3 [4]_ is noncontextual and model-free: it maintains its
own cumulative weight tables with no underlying belief system. This
implementation keeps the parts that give adversarial robustness
(uniform mixing, importance-weighted updates) but replaces the weight
tables with Bayesian posterior means. That swap is what makes it
contextual, compatible with decay, and anytime: posterior means don't
grow with :math:`T`, so :math:`\eta` needs no horizon-dependent
tuning. The default variant is EXP3-IX [5]_. The
:doc:`/notebooks/adversarial` notebook demonstrates both adversarial
robustness and exploitation of a weak opponent.

Unlike the other policies, EXP3A overrides the update rule with
importance-weighted observations.


Selection rule
~~~~~~~~~~~~~~

1. Compute posterior mean rewards via Monte Carlo:

   .. math::

      \hat{\mu}_a(\mathbf{x}_t)
      = \frac{1}{S} \sum_{s=1}^{S}
        g_a(\tilde{\theta}_a^{(s)}, \mathbf{x}_t)

2. Compute exponential weights (with log-sum-exp stability):

   .. math::

      w_a = \exp\!\bigl(\eta \cdot
        (\hat{\mu}_a - \max_j \hat{\mu}_j)\bigr)

3. Mix with uniform exploration:

   .. math::

      P_t(a) = (1 - \gamma_{\text{forced}})
        \frac{w_a}{\sum_j w_j}
      + \frac{\gamma_{\text{forced}}}{K}

4. Sample arm :math:`a_t` according to :math:`P_t`.

When :math:`\gamma_{\text{forced}} = 0` (default), this is a pure
softmax over posterior means.


Update rule (importance-weighted)
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

EXP3A overrides the default update. After observing reward :math:`y`
for arm :math:`a_t`:

1. Recompute the selection probabilities :math:`P_t(a)` using the
   current posterior (stateless design).

2. Compute the importance weight:

   .. math::

      w_t = \frac{1}{P_t(a_t) + \gamma_{\text{ix}}}

3. Update the selected arm with the weighted observation:

   .. math::

      \text{arm}_{a_t}.\text{update}(\mathbf{X},\, \mathbf{y},\,
        \text{sample\_weight} = w_t)

The importance weighting corrects for selection bias: arms pulled
frequently get down-weighted, arms pulled rarely get up-weighted.


Two variants
~~~~~~~~~~~~

**EXP3-IX** (default): ``gamma=0``, ``ix_gamma=eta/2``. No forced
exploration; the IX regularization term :math:`\gamma_{\text{ix}}`
in the denominator bounds importance weights at
:math:`1/\gamma_{\text{ix}}`, preventing catastrophic updates when an
arm has very low selection probability. Recommended by Neu (2015) [5]_
for better empirical performance and high-probability bounds.

**Standard EXP3**: ``gamma>0``, ``ix_gamma=0``. Forced exploration via
:math:`\gamma`-mixing ensures every arm has at least
:math:`\gamma/K` selection probability. Unbiased importance weights
(no IX regularization). Original formulation from Auer et al. (2002)
[4]_.


Hyperparameters
~~~~~~~~~~~~~~~

.. list-table::
   :header-rows: 1
   :widths: 20 50 30

   * - Parameter
     - Controls
     - Practical guidance
   * - ``gamma``
     - Forced exploration rate (default 0.0). Each arm is guaranteed
       at least :math:`\gamma/K` selection probability.
     - 0.0 for EXP3-IX (recommended). Set > 0 for standard EXP3.
   * - ``eta``
     - Temperature for exponential weights (default 1.0). Higher
       values create sharper distinctions between arms (more
       exploitation).
     - Must be calibrated to reward scale. If rewards are in
       :math:`[0, 1]`, :math:`\eta = 1.0` is reasonable. If rewards
       are in :math:`[0, 100]`, use :math:`\eta = 0.01`.
   * - ``ix_gamma``
     - IX regularization (default ``eta/2``). Bounds importance
       weights at :math:`1/\gamma_{\text{ix}}`.
     - Use the default (``eta/2``) for EXP3-IX. Set to 0 for
       standard EXP3 (unbiased weights).
   * - ``samples``
     - Number of Monte Carlo draws per arm (default 100).
     - 100 is usually sufficient for mean estimation.


Regret bounds
~~~~~~~~~~~~~

EXP3-IX achieves a high-probability bound of [5]_:

.. math::

   \mathrm{Regret}(T) = O\!\left(\sqrt{KT \ln K}\right)

Standard EXP3 achieves the same minimax rate in expectation [4]_.

References
----------

.. [1] Chapelle, O. & Li, L. (2011). "An empirical evaluation of
   Thompson sampling." *NeurIPS 24*, 2249--2257.

.. [2] Agrawal, S. & Goyal, N. (2012). "Analysis of Thompson sampling
   for the multi-armed bandit problem." *COLT*, JMLR W&CP 23,
   39.1--39.26.

.. [3] Kaufmann, E., Cappe, O., & Garivier, A. (2012). "On Bayesian
   upper confidence bounds for bandit problems." *AISTATS*, JMLR W&CP
   22, 592--600.

.. [4] Auer, P., Cesa-Bianchi, N., & Fischer, P. (2002).
   "Finite-time analysis of the multiarmed bandit problem."
   *Machine Learning*, 47(2--3), 235--256.

.. [5] Neu, G. (2015). "Explore no more: Improved high-probability
   regret bounds for non-stochastic bandits." *NeurIPS 28*,
   3168--3176.
