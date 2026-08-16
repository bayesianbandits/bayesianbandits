"""
Base classes for Bayesian bandit policies.
"""

from typing import (
    Generic,
    List,
    Optional,
)

import numpy as np
from numpy.typing import NDArray

from .._arm import Arm, ContextType, TokenType


class PolicyDefaultUpdate(Generic[ContextType, TokenType]):
    #: Safe default satisfying ``PolicyProtocol``: joint ``sample``
    #: draws serve every policy. Subclasses opt into cheaper sampling
    #: modes by overriding this (see
    #: :class:`~bayesianbandits.api.PolicyProtocol` for the semantics).
    marginal_ok: bool = False
    reward_space_ok: bool = False

    def _draw_samples(
        self,
        arms: List[Arm[ContextType, TokenType]],
        X: ContextType,
        size: int,
    ) -> NDArray[np.float64]:
        """Draw ``(n_arms, n_contexts, size)`` samples for ``select``.

        Samples each arm from its own learner, marginally when
        ``marginal_ok`` (iid per-row draws are exact for policies
        consuming only per-(arm, context) statistics) and jointly
        otherwise.

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
        samples = np.array(
            [
                (arm.sample_marginal if self.marginal_ok else arm.sample)(X, size)
                for arm in arms
            ]
        )
        # Convert from (n_arms, size, n_contexts) to (n_arms, n_contexts, size)
        return samples.transpose(0, 2, 1)

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
