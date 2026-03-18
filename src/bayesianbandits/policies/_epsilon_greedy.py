"""
Epsilon-greedy policy for Bayesian bandits.
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


class EpsilonGreedy(PolicyDefaultUpdate[ContextType, TokenType]):
    r"""
    Policy object for :math:`\varepsilon`-greedy selection.

    With probability :math:`1 - \varepsilon` the arm with the highest
    estimated posterior mean reward is selected (exploit); with probability
    :math:`\varepsilon` an arm is chosen uniformly at random (explore):

    .. math::

        a_t =
        \begin{cases}
        \arg\max_a \; \hat{\mu}_a(x_t)
            & \text{with probability } 1 - \varepsilon, \\
        \text{Uniform}\{1, \ldots, K\}
            & \text{with probability } \varepsilon,
        \end{cases}

    where :math:`\hat{\mu}_a(x_t) = \mathbb{E}_{\theta_a \mid
    \mathcal{D}_a}[g_a(\theta_a, x_t)]` is estimated via Monte Carlo with
    ``samples`` posterior draws.

    Parameters
    ----------
    epsilon : float, default=0.1
        Probability of exploration.
    samples : int, default=1000
        Number of posterior samples used to estimate the arm means.

    Notes
    -----
    **Regret bounds (standard setting).** For a :math:`K`-armed stochastic
    bandit with fixed :math:`\varepsilon`, the expected regret is bounded by

    .. math::

        \mathbb{E}[\mathrm{Regret}(T)]
        \;\le\; \varepsilon\,T\,\Delta_{\max}
        \;+\; \sum_{a:\Delta_a>0}
        \frac{C}{\Delta_a}

    where :math:`\Delta_a = \mu^* - \mu_a` is the sub-optimality gap and
    :math:`C` depends on the concentration of the mean estimator [2]_.
    With a decaying schedule :math:`\varepsilon_t = O(K / t)` the minimax
    rate :math:`O(\sqrt{KT})` is achievable.

    **Applicability to this library.** The classical analysis assumes
    stationary rewards and empirical sample means with known concentration
    properties [1]_. This implementation replaces the empirical mean with a
    Bayesian posterior mean (which may use approximate inference) and
    supports contextual features and variance-increasing decay for
    non-stationarity. Under these modifications the formal bounds do not
    directly apply, but the basic explore-exploit trade-off controlled by
    :math:`\varepsilon` is preserved.

    References
    ----------
    .. [1] Chapelle, O. and Li, L. (2011). "An empirical evaluation of
       Thompson sampling." Advances in Neural Information Processing Systems
       24, 2249-2257.

    .. [2] Auer, P., Cesa-Bianchi, N., and Fischer, P. (2002).
       "Finite-time analysis of the multiarmed bandit problem." Machine
       Learning, 47(2-3), 235-256.
    """

    def __repr__(self) -> str:
        return f"EpsilonGreedy(epsilon={self.epsilon}, samples={self.samples})"

    def __init__(self, epsilon: float = 0.1, samples: int = 1000):
        self.epsilon = epsilon
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
        """Select arms based on pre-generated samples using epsilon-greedy."""
        # Compute means across samples dimension
        means = samples.mean(axis=-1)  # Shape: (n_arms, n_contexts)

        # Apply epsilon-greedy exploration per context
        # Decide which contexts explore
        choice_idx_to_explore = rng.random(size=means.shape[1]) < self.epsilon  # type: ignore[misc]

        # How many arms to mark as infinite
        k = 1 if top_k is None else top_k

        # For each exploring context, set k random arms to np.inf
        values = means.copy()
        for idx, explore in enumerate(choice_idx_to_explore):  # type: ignore[misc]
            if explore:
                random_arms = rng.choice(
                    means.shape[0],
                    size=min(k, means.shape[0]),
                    replace=False,
                )
                values[random_arms, idx] = np.inf

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
        """Choose arm(s) using epsilon-greedy."""
        samples = batch_sample_arms(arms, X, size=self.samples_needed)
        if samples is None:
            samples = np.array([arm.sample(X, self.samples_needed) for arm in arms])
            # Convert from (n_arms, size, n_contexts) to (n_arms, n_contexts, size)
            samples = samples.transpose(0, 2, 1)
        return self.select(samples, arms, rng, top_k)
