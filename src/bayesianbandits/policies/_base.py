"""
Base classes for Bayesian bandit policies.
"""

from typing import (
    Generic,
    List,
    Optional,
    Union,
    overload,
)

import numpy as np
from numpy.typing import NDArray

from .._arm import Arm, ContextType, TokenType
from .._draw_kind import DrawKind


class PolicyDefaultUpdate(Generic[ContextType, TokenType]):
    #: Safe default satisfying ``PolicyProtocol``: fully joint draws
    #: serve every policy. Subclasses declare a weaker requirement to
    #: get cheaper draws (see :class:`~bayesianbandits.DrawKind`).
    consumes: DrawKind = DrawKind.JOINT

    @property
    def samples_needed(self) -> int:
        """Number of samples per arm per context needed for decision making."""
        raise NotImplementedError

    def _draw_samples(
        self,
        arms: List[Arm[ContextType, TokenType]],
        X: ContextType,
        size: int,
    ) -> NDArray[np.float64]:
        """Draw ``(n_arms, n_contexts, size)`` samples for ``select``.

        Samples each arm from its own learner, marginally when the
        policy consumes ``MARGINAL_ONLY`` (iid per-row draws are exact
        for policies reading only per-(arm, context) statistics) and
        jointly otherwise. ``CONTEXT_JOINT`` and ``JOINT`` coincide
        here: every arm has its own learner, so there is no cross-arm
        dependence for a context block to preserve.

        Drawing one arm at a time is the correct joint law here because
        agents using this path give every arm an independent learner --
        a shared learner is rejected at ``add_arm``, since without
        arm-specific features the arms would be indistinguishable, and
        sampling them separately would break the dependence they have
        through the shared posterior. Sharing a learner across arms is
        :class:`~bayesianbandits.LipschitzContextualAgent`'s job, and it
        samples that learner once for all arms rather than coming
        through here.
        """
        # Stacking transposed builds (n_arms, n_contexts, size) in one
        # copy with the draw axis contiguous (the layout contract).
        return np.array(
            [
                (
                    arm.sample_marginal
                    if self.consumes == DrawKind.MARGINAL_ONLY
                    else arm.sample
                )(X, size).T
                for arm in arms
            ]
        )

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
        raise NotImplementedError

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
        """Draw the samples the policy asked for, then let it choose."""
        samples = self._draw_samples(arms, X, self.samples_needed)
        return self.select(samples, arms, rng, top_k)

    def update(
        self,
        arm: Arm[ContextType, TokenType],
        X: ContextType,
        y: NDArray[np.float64],
        all_arms: List[Arm[ContextType, TokenType]],
        rng: np.random.Generator,
        sample_weight: Optional[NDArray[np.float64]] = None,
    ) -> None:
        """Default update implementation that simply updates the arm."""
        arm.update(X, y, sample_weight=sample_weight)
