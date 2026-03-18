"""
Upper confidence bound policy for Bayesian bandits.
"""

from typing import (
    List,
    Optional,
    Union,
    overload,
)

import numpy as np
from numpy.typing import NDArray

from .._arm import Arm, ContextType, TokenType, batch_sample_arms
from ._base import PolicyDefaultUpdate


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
        # Compute upper confidence bounds using quantiles
        # samples shape: (n_arms, n_contexts, samples_needed)
        ucb_values = np.quantile(samples, self.alpha, axis=-1)

        if top_k is None:
            best_indices = ucb_values.argmax(axis=0)
            return [arms[idx] for idx in best_indices]
        else:
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

    @overload
    def __call__(
        self,
        arms: List[Arm[ContextType, TokenType]],
        X: ContextType,
        rng: np.random.Generator,
        top_k: None = None,
    ) -> List[Arm[ContextType, TokenType]]: ...

    @overload
    def __call__(
        self,
        arms: List[Arm[ContextType, TokenType]],
        X: ContextType,
        rng: np.random.Generator,
        top_k: int,
    ) -> List[List[Arm[ContextType, TokenType]]]: ...

    def __call__(
        self,
        arms: List[Arm[ContextType, TokenType]],
        X: ContextType,
        rng: np.random.Generator,
        top_k: Optional[int] = None,
    ) -> Union[
        List[Arm[ContextType, TokenType]], List[List[Arm[ContextType, TokenType]]]
    ]:
        """Choose arm(s) using upper confidence bound."""
        samples = batch_sample_arms(arms, X, size=self.samples_needed)
        if samples is None:
            samples = np.array([arm.sample(X, self.samples_needed) for arm in arms])
            # Convert from (n_arms, size, n_contexts) to (n_arms, n_contexts, size)
            samples = samples.transpose(0, 2, 1)
        return self.select(samples, arms, rng, top_k)
