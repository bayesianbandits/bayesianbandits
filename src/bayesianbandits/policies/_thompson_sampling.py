"""
Thompson sampling policy for Bayesian bandits.
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


class ThompsonSampling(PolicyDefaultUpdate[ContextType, TokenType]):
    """
    Policy object for Thompson sampling.

    Thompson sampling chooses the best arm with probability equal to the
    probability that the arm is the best arm. At each round, a single sample
    is drawn from each arm's posterior and the arm with the highest sample is
    selected:

    .. math::

        \\tilde{\\theta}_a \\sim p(\\theta_a \\mid \\mathcal{D}_a)
        \\quad \\forall a, \\qquad
        a^* = \\arg\\max_a \\; g_a(\\tilde{\\theta}_a)

    where :math:`g_a` is the reward function (possibly context-dependent) and
    :math:`\\mathcal{D}_a` is the data observed for arm :math:`a`.

    Notes
    -----
    **Regret bounds (standard setting).** For the :math:`K`-armed stochastic
    bandit with Beta-Bernoulli or Gaussian conjugate models, Thompson sampling
    achieves a Bayesian expected regret of

    .. math::

        \\mathbb{E}[\\mathrm{Regret}(T)]
        = O\\!\\left(\\sqrt{KT \\ln K}\\right)

    and an asymptotically optimal problem-dependent bound of
    :math:`O\\!\\left(\\sum_{a:\\Delta_a>0}
    \\frac{\\ln T}{\\Delta_a}\\right)` matching the Lai-Robbins lower bound
    [2]_.

    **Applicability to this library.** The bounds above are proven for
    stationary, non-contextual bandits with exact conjugate posteriors. This
    library targets anytime, contextual, and potentially non-stationary
    problems, and the Bayesian learners may use approximate posteriors (e.g.
    Laplace approximations) or variance-increasing decay. Under these
    modifications the classical regret guarantees do not formally apply,
    though the core explore-exploit mechanism---probability matching via
    posterior sampling---is preserved.

    References
    ----------
    .. [1] Chapelle, O. and Li, L. (2011). "An empirical evaluation of
       Thompson sampling." Advances in Neural Information Processing Systems
       24, 2249-2257.

    .. [2] Agrawal, S. and Goyal, N. (2012). "Analysis of Thompson sampling
       for the multi-armed bandit problem." Proceedings of the 25th Annual
       Conference on Learning Theory (COLT), JMLR W&CP 23, 39.1-39.26.
    """

    def __repr__(self) -> str:
        return "ThompsonSampling()"

    @property
    def samples_needed(self) -> int:
        """Number of samples per arm per context needed for decision making."""
        return 1

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
        """Select arms based on pre-generated samples."""
        # samples shape: (n_arms, n_contexts, 1)
        values = samples[..., 0]  # Shape: (n_arms, n_contexts)

        if top_k is None:
            best_indices = values.argmax(axis=0)
            return [arms[idx] for idx in best_indices]
        else:
            # Use argpartition for efficiency, but handle case where top_k >= n_arms
            return [
                [
                    arms[idx]
                    for idx in np.argpartition(
                        -values[:, i], min(top_k, len(arms) - 1)
                    )[:top_k]
                ]
                for i in range(values.shape[1])
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
        """Choose arm(s) using Thompson sampling."""
        samples = batch_sample_arms(arms, X, size=self.samples_needed)
        if samples is None:
            samples = np.array([arm.sample(X, self.samples_needed) for arm in arms])
            # Convert from (n_arms, size, n_contexts) to (n_arms, n_contexts, size)
            samples = samples.transpose(0, 2, 1)
        return self.select(samples, arms, rng, top_k)
