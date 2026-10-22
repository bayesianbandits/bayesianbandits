"""
Upper confidence bound policy for Bayesian bandits.
"""

from typing import (
    List,
    Optional,
    Tuple,
    Union,
    overload,
)

import numpy as np
from numpy.typing import NDArray

from .._arm import Arm, ContextType, TokenType
from .._draw_kind import DrawKind
from ._base import PolicyDefaultUpdate


def _quantile(samples: NDArray[np.float64], q: float) -> NDArray[np.float64]:
    """``np.quantile(samples, q, axis=-1)``, bit-identical, 3x faster.

    One partition at the lower order statistic; the upper one is the
    min of the tail it leaves. ``np.quantile`` partitions at both.
    """
    pos = q * (samples.shape[-1] - 1)
    lo = int(pos)
    frac = pos - lo
    part = np.partition(samples, lo, axis=-1)
    a = part[..., lo]
    if frac == 0.0:
        return a
    b = part[..., lo + 1 :].min(axis=-1)
    # numpy's lerp, monotone at both ends
    return b - (b - a) * (1 - frac) if frac >= 0.5 else a + (b - a) * frac


class UpperConfidenceBound(PolicyDefaultUpdate[ContextType, TokenType]):
    """
    Policy object for Bayesian upper confidence bound.

    At each round, posterior samples are used to estimate the
    :math:`\\alpha`-quantile of each arm's reward distribution, and the arm
    with the highest quantile is selected:

    .. math::

        a^* = \\arg\\max_a \\;
        Q_{\\alpha}\\!\\bigl(g_a(\\theta_a) \\mid \\mathcal{D}_a\\bigr)

    where :math:`Q_{\\alpha}` denotes the :math:`\\alpha`-quantile of the
    posterior predictive reward, estimated via Monte Carlo with ``samples``
    draws.

    Parameters
    ----------
    alpha : float, default=0.68
        Quantile level used as the upper confidence bound. Higher values
        produce more optimistic estimates and encourage exploration.
        The default of 0.68 corresponds roughly to a one-standard-deviation
        bound for a Gaussian posterior.
    samples : int, default=1000
        Number of posterior samples used to estimate the quantile.

    Examples
    --------
    Default one-sigma bound:

    >>> from bayesianbandits import Agent, Arm, NormalInverseGammaRegressor
    >>> from bayesianbandits import UpperConfidenceBound
    >>>
    >>> arms = [
    ...     Arm(f"arm_{i}", learner=NormalInverseGammaRegressor())
    ...     for i in range(3)
    ... ]
    >>> agent = Agent(arms, UpperConfidenceBound())

    More aggressive exploration with a higher quantile:

    >>> agent = Agent(arms, UpperConfidenceBound(alpha=0.95))

    See Also
    --------
    ThompsonSampling : Randomized exploration via posterior sampling.
        Generally the default choice; explores proportionally to uncertainty.
    EpsilonGreedy : Simple exploration via random arm selection.
        Does not use posterior uncertainty to guide exploration.
    EXP3A : Adversarial robustness via importance-weighted updates.
        Use when reward generation may be adversarial or non-stochastic.

    Notes
    -----
    **Regret bounds (standard setting).** Kaufmann et al. (2012) show that
    Bayesian UCB with a quantile schedule
    :math:`\\alpha_t = 1 - 1/(t \\log^c(T))` achieves

    .. math::

        \\mathbb{E}[\\mathrm{Regret}(T)]
        = O\\!\\left(\\sum_{a:\\Delta_a>0}
        \\frac{\\ln T}{\\Delta_a}\\right)

    matching the Lai-Robbins lower bound up to constants [2]_. With a fixed
    :math:`\\alpha` (as used here), the problem-dependent bound is not
    guaranteed, but a minimax rate of :math:`O(\\sqrt{KT \\ln T})` still
    holds under mild conditions.

    **Applicability to this library.** The bounds above assume stationary
    rewards, exact conjugate posteriors, and a non-contextual setting [1]_.
    This library uses a fixed quantile level, supports contextual features,
    approximate posteriors, and variance-increasing decay for
    non-stationarity. Under these modifications the formal guarantees do not
    directly apply, though the optimism-in-the-face-of-uncertainty principle
    that drives UCB's exploration is preserved.

    References
    ----------
    .. [1] Chapelle, O. and Li, L. (2011). "An empirical evaluation of
       Thompson sampling." Advances in Neural Information Processing Systems
       24, 2249-2257.

    .. [2] Kaufmann, E., Cappe, O., and Garivier, A. (2012). "On Bayesian
       upper confidence bounds for bandit problems." Proceedings of the 15th
       International Conference on Artificial Intelligence and Statistics
       (AISTATS), JMLR W&CP 22, 592-600.
    """

    def __repr__(self) -> str:
        return f"UpperConfidenceBound(alpha={self.alpha}, samples={self.samples})"

    def __init__(self, alpha: float = 0.68, samples: int = 1000):
        self.alpha = alpha
        self.samples = samples

    #: Consumes only per-(arm, context) statistics, so iid marginal
    #: draws (``sample_marginal``) are exact for this policy.
    consumes = DrawKind.MARGINAL_ONLY

    @property
    def samples_needed(self) -> int:
        """Number of samples per arm per context needed for decision making."""
        return self.samples

    @overload
    def select(
        self,
        samples: NDArray[np.float64],  # Shape: (n_arms, n_contexts, samples_needed)
        arms: List[Arm[ContextType, TokenType]],
        rng: np.random.Generator,
        top_k: None = None,
    ) -> List[Arm[ContextType, TokenType]]: ...

    @overload
    def select(
        self,
        samples: NDArray[np.float64],  # Shape: (n_arms, n_contexts, samples_needed)
        arms: List[Arm[ContextType, TokenType]],
        rng: np.random.Generator,
        top_k: int,
    ) -> List[List[Arm[ContextType, TokenType]]]: ...

    def select(
        self,
        samples: NDArray[np.float64],  # Shape: (n_arms, n_contexts, samples_needed)
        arms: List[Arm[ContextType, TokenType]],
        rng: np.random.Generator,
        top_k: Optional[int] = None,
    ) -> Union[
        List[Arm[ContextType, TokenType]], List[List[Arm[ContextType, TokenType]]]
    ]:
        """Select arms based on pre-generated samples using upper confidence bounds."""
        if top_k is None:
            return self.select_with_probabilities(samples, arms, rng)[0]

        # Compute upper confidence bounds using quantiles
        # samples shape: (n_arms, n_contexts, samples_needed)
        ucb_values = _quantile(samples, self.alpha)

        # Use argpartition for efficiency, but handle case where top_k >= n_arms
        return [
            [
                arms[idx]
                for idx in np.argpartition(
                    -ucb_values[:, i], min(top_k, len(arms) - 1)
                )[:top_k]
            ]
            for i in range(ucb_values.shape[1])
        ]

    def select_with_probabilities(
        self,
        samples: NDArray[np.float64],  # Shape: (n_arms, n_contexts, samples_needed)
        arms: List[Arm[ContextType, TokenType]],
        rng: np.random.Generator,
    ) -> Tuple[List[Arm[ContextType, TokenType]], NDArray[np.float64]]:
        """Select one arm per context and return the ``(n_contexts, n_arms)``
        distribution it was drawn from: one-hot, since UCB is deterministic
        given the samples."""
        choices = _quantile(samples, self.alpha).argmax(axis=0)
        probs = np.zeros((samples.shape[1], len(arms)))
        probs[np.arange(samples.shape[1]), choices] = 1.0
        return list(map(arms.__getitem__, choices.tolist())), probs
