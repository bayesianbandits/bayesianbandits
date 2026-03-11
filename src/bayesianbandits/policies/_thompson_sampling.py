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
    probability that the arm is the best arm. That is, it takes a sample from
    each arm's posterior distribution and chooses the arm with the highest
    sample.

    Notes
    -----
    The implementation here is based on the implementation in [1]_.

    References
    ----------
    .. [1] Chapelle, Olivier, and Lihong Li. "An empirical evaluation of
       thompson sampling." Advances in neural information processing systems
       24 (2011): 2249-2257.
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
