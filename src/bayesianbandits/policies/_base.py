"""
Base classes for Bayesian bandit policies.
"""

from typing import (
    Any,
    Generic,
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

# Arms up to which one stable sort beats partitioning: 1.7x at 10 arms by
# 10k contexts, while past ~128 the sort's log factor loses (2.5x at 512
# arms by 64 contexts)
_SORT_ARMS = 128


def _top_k_indices(values: NDArray[np.floating[Any]], top_k: int) -> NDArray[np.intp]:
    """``(n_contexts, min(top_k, n_arms))`` arm indices per context, best
    ``values`` first, ties to the lower index as ``argmax`` breaks them, so
    slot 0 is the ``top_k=None`` pick. ``values`` is ``(n_arms, n_contexts)``.

    Past ``_SORT_ARMS`` arms, linear in them instead of a full sort:
    partition to each context's k-th best value, keep everything above it
    plus the lowest-indexed arms tied at it, and sort only those k. NaN
    ranks last on both routes."""
    # np.partition would make NaN the k-th value, which nothing ties
    values = np.where(np.isnan(values), -np.inf, values)
    n_arms = values.shape[0]
    if top_k >= n_arms or n_arms <= _SORT_ARMS:
        return np.argsort(-values, axis=0, kind="stable")[:top_k].T
    kth = -np.partition(-values, top_k - 1, axis=0)[top_k - 1]
    above = values > kth
    tied = values == kth
    chosen = above | (tied & (np.cumsum(tied, axis=0) <= top_k - above.sum(axis=0)))
    # exactly top_k per context, ascending index (a stable sort of a bool)
    idx = np.argsort(~chosen, axis=0, kind="stable")[:top_k]
    order = np.argsort(-np.take_along_axis(values, idx, axis=0), axis=0, kind="stable")
    return np.take_along_axis(idx, order, axis=0).T


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

    def select_with_probabilities(
        self,
        samples: NDArray[np.float64],  # Shape: (n_arms, n_contexts, samples_needed)
        arms: List[Arm[ContextType, TokenType]],
        rng: np.random.Generator,
    ) -> Tuple[List[Arm[ContextType, TokenType]], NDArray[np.float64]]:
        """Select one arm per context and return the ``(n_contexts, n_arms)``
        distribution it was drawn from, given the samples."""
        raise NotImplementedError(
            f"{type(self).__name__} does not report selection probabilities"
        )

    def call_with_probabilities(
        self,
        arms: List[Arm[ContextType, TokenType]],
        X: ContextType,
        rng: np.random.Generator,
    ) -> Tuple[List[Arm[ContextType, TokenType]], NDArray[np.float64]]:
        """Draw the samples the policy asked for, then
        ``select_with_probabilities``."""
        samples = self._draw_samples(arms, X, self.samples_needed)
        return self.select_with_probabilities(samples, arms, rng)

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
