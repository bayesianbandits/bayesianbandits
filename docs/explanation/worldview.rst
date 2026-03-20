========================
Knowledge Is Prediction
========================

The question
============

Every bandit system answers the same question: *given what I know right now,
what will I observe next if I take action a?*

The question is about the next observable outcome: the click, the conversion,
the revenue. Which arm is best is a downstream decision. What θ is doesn't
matter; you'll never observe it.

This is the **forecast**:

.. math::

   p(r_{t+1} \mid a, \text{history})

The posterior predictive distribution over the next reward, conditioned on an
action and everything you've observed so far. It compresses history into a
sufficient statistic for the decision problem.

.. note::

   **For RL readers.** A bandit is a single-state MDP with discount γ = 0.
   The sample-average update for an arm's value is TD(0). Every arm's
   estimated value is a general value function (GVF) with zero discount.
   Bandits aren't simplified RL; they're a special case, and the RL
   perspective has something specific to say about what knowledge means in
   this setting.


What counts as knowledge
========================

Sutton argues that the content of knowledge is verifiable predictions about
future observations [1]_. You can't verify θ. You *can* verify "the next
reward under action *a* will be drawn from this distribution." A model's
knowledge is exactly the predictions it makes.

State is whatever summary of the past is sufficient to produce those
predictions. The posterior is a state representation in this sense: it
compresses all past observations into a sufficient statistic for the predictive
distribution, not for θ (that's a different, weaker claim).

Predictive state representations (PSRs) formalize this [2]_. A PSR defines
state entirely in terms of predictions about future observables, no hidden
variables required. If the predictions are calibrated, the model is doing its
job, regardless of whether the parameters inside it correspond to anything
real.


The forecast is fundamental
===========================

The **posterior predictive** :math:`p(r_{t+1} \mid a, \text{history})` is the
object that matters. The parameter posterior
:math:`p(\theta \mid \text{history})` exists to produce it. If you had the
predictive without parameters, you wouldn't need parameters.

bayesianbandits maintains conjugate posteriors over parameters because they're
the cheapest way to produce calibrated forecasts in real time. One matrix
operation per observation. O(d²) regardless of how many observations you've
seen. Cholesky sampling for Thompson draws. No MCMC, no variational inference,
no mini-batches. Parameters are load-bearing for *computation*, not for
*meaning*.

This is the same tradeoff at work in `approximate hierarchical Bayes for
online learning
<https://rukulkarni.com/blog/approximate-hierarchical-bayes-online-learning/>`_:
the ideal model has load-bearing properties, and the job of a practical system
is to preserve those properties while making the computation feasible.


Deciding is not forecasting
===========================

The forecast tells you what will happen. It doesn't tell you what to do.

Thompson sampling draws a plausible mean reward from the forecast, picks the
action that looks best. UCB picks the action whose optimistic forecast is
highest. Epsilon-greedy picks the highest expected reward and occasionally
randomizes. The decision rule consumes the forecast.

This is why bayesianbandits separates the learner (which maintains the
forecast) from the policy (which acts on it). The learner updates when you
observe an outcome. The policy reads the current forecast and decides.

This separation has consequences that go further than forecast vs. policy. The
full story, including how the reward function fits in as a third axis, is in
:doc:`decision-theory`.


The forecast is always well-defined
====================================

Most bandit theory frames the problem as: there exists a true best arm, and
your job is to find it while wasting as few pulls as possible on inferior
arms. That framing requires stationarity (the best arm stays the best), a
fixed parameter (θ exists and doesn't change), and usually a known horizon T.
Algorithms built on it need epoch resets, the doubling trick, or explicit
horizon parameters to function, and they discard learned state at each restart.

Production bandits don't live there. The world drifts, there may be no fixed
best arm, and you don't want to throw away what you've learned on a schedule.
You want a bandit that runs forever.

The forecast doesn't need any of that. Given history, what will I observe next
under action *a*? That question is well-defined at every time step, stationary
or not. When the world changes, the forecast adapts. In bayesianbandits, decay
re-inflates uncertainty and new observations sharpen it again.

.. note::

   **For RL readers.** The pull/update/decay loop is literally a Kalman filter
   with a decision layer. The conjugate update is the measurement step:
   incorporate what you just observed. Decay is the prediction step: time has
   passed, the world may have moved, so uncertainty grows on all arms,
   including unobserved ones. Thompson sampling acts on the current state.


Further reading
================

The predictive view of knowledge originates in Sutton's work on TD learning
and general value functions [1]_, formalized by predictive state
representations [2]_. Russo and Van Roy develop the same idea in a Bayesian RL
setting: the information ratio that governs explore/exploit is defined over
predictive distributions, not parameter posteriors [3]_ [4]_ [5]_. Fong,
Holmes, and Walker [6]_ show that coherent predictive distributions can be
defined without ever specifying a parametric model.


.. rubric:: References

.. [1] Sutton, R. S. and Barto, A. G. (2018). *Reinforcement Learning: An
   Introduction.* 2nd ed. MIT Press.

.. [2] Littman, M. L., Sutton, R. S., and Singh, S. (2001). "Predictive
   representations of state." *Advances in Neural Information Processing
   Systems* 14.

.. [3] Russo, D. and Van Roy, B. (2014). "Learning to optimize via posterior
   sampling." *Mathematics of Operations Research*, 39(4), 1221--1243.

.. [4] Russo, D. and Van Roy, B. (2018). "An information-theoretic analysis
   of Thompson sampling." *Journal of Machine Learning Research*, 19(96),
   1--30.

.. [5] Lu, X. and Van Roy, B. (2023). "Reinforcement learning, bit by bit."
   *Foundations and Trends in Machine Learning*, 16(6), 733--865.

.. [6] Fong, E., Holmes, C., and Walker, S. G. (2023). "Martingale posterior
   distributions." *Journal of the Royal Statistical Society: Series B*,
   85(5), 1357--1391.
