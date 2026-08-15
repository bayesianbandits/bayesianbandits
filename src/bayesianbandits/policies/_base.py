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

from .._arm import Arm, ContextType, TokenType, batch_sample_arms


class PolicyDefaultUpdate(Generic[ContextType, TokenType]):
    #: Safe defaults satisfying ``PolicyProtocol``: joint ``sample``
    #: draws serve every policy. Subclasses opt into cheaper sampling
    #: modes by overriding these (see
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

        Tries one batched call across arms sharing a learner, forwarding
        both capability flags (``marginal_ok``: iid per-row draws are
        exact for policies consuming only per-(arm, context) statistics;
        ``reward_space_ok``: per-context joint blocks are exact for
        per-context decisions); falls back to per-arm sampling
        (marginal when ``marginal_ok``) otherwise. Always returns a 3-D
        array, re-expanding the size axis ``batch_sample_arms`` squeezes
        when ``size == 1``.
        """
        samples = batch_sample_arms(
            arms,
            X,
            size=size,
            marginal=self.marginal_ok,
            reward_space=self.reward_space_ok,
        )
        if samples is None:
            samples = np.array(
                [
                    (arm.sample_marginal if self.marginal_ok else arm.sample)(X, size)
                    for arm in arms
                ]
            )
            # Convert from (n_arms, size, n_contexts) to (n_arms, n_contexts, size)
            samples = samples.transpose(0, 2, 1)
        elif samples.ndim == 2:
            samples = samples[:, :, np.newaxis]
        return samples

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
