from __future__ import annotations
from functools import partial
from typing import Callable, Optional, cast

import numpy as np
from numpy.typing import ArrayLike, NDArray

from ._typing import ArmProtocol, BanditProtocol


def epsilon_greedy(
    epsilon: float = 0.1,
    *,
    samples: int = 1000,
) -> Callable[[BanditProtocol, Optional[ArrayLike]], ArmProtocol]:
    """Creates an epsilon-greedy choice algorithm. To be used with the
    `bandit` decorator.

    Parameters
    ----------
    epsilon : float, default=0.1
        Probability of choosing a random arm.
    samples : int, default=1000
        Number of samples to use to compute the mean of the posterior.

    Returns
    -------
    Callable[[BanditProtocol, Optional[ArrayLike]], ArmProtocol]
        Closure that chooses an arm using epsilon-greedy.
    """

    def _choose_arm(
        self: BanditProtocol,
        X: Optional[ArrayLike] = None,
    ) -> ArmProtocol:
        """Choose an arm using epsilon-greedy."""

        if self.rng.random() < epsilon:  # type: ignore
            return self.rng.choice(list(self.arms.values()))  # type: ignore
        else:
            key_func = partial(_compute_arm_mean, X=X, samples=samples)
            return max(self.arms.values(), key=key_func)  # type: ignore

    return _choose_arm


def upper_confidence_bound(
    alpha: float = 0.68,
    *,
    samples: int = 1000,
) -> Callable[[BanditProtocol, Optional[ArrayLike]], ArmProtocol]:
    """Creates a UCB choice algorithm. To be used with the
    `bandit` decorator.

    Actually uses the upper bound of the one-sided credible interval,
    which will deviate from the upper bound of the one-sided confidence
    interval depending on the strength of the prior.

    Parameters
    ----------
    alpha : float, default=0.68
        Upper bound of the one-sided prediction interval.
    samples : int, default=1000
        Number of samples to use to compute the upper bound of the
        credible interval. The larger `alpha` is, the larger `samples`
        should be.

    Returns
    -------
    Callable[[BanditProtocol, Optional[ArrayLike]], ArmProtocol]
        Closure that chooses an arm using UCB.
    """

    if not 0 < alpha < 1:
        raise ValueError("alpha must be in (0, 1).")

    def _choose_arm(
        self: BanditProtocol,
        X: Optional[ArrayLike] = None,
    ) -> ArmProtocol:
        """Choose an arm using UCB1."""

        key_func = partial(_compute_arm_upper_bound, X=X, alpha=alpha, samples=samples)
        return max(self.arms.values(), key=key_func)  # type: ignore

    return _choose_arm


def thompson_sampling() -> Callable[[BanditProtocol, Optional[ArrayLike]], ArmProtocol]:
    """Creates a Thompson sampling choice algorithm. To be used with the
    `bandit` decorator.

    Returns
    -------
    Callable[[BanditProtocol, Optional[ArrayLike]], ArmProtocol]
        Closure that chooses an arm using Thompson sampling.
    """

    def _choose_arm(
        self: BanditProtocol,
        X: Optional[ArrayLike] = None,
    ) -> ArmProtocol:
        """Choose an arm using Thompson sampling."""

        key_func = partial(_draw_one_sample, X=X)
        return max(self.arms.values(), key=key_func)  # type: ignore

    return _choose_arm


def _draw_one_sample(
    arm: ArmProtocol,
    X: Optional[ArrayLike] = None,
) -> np.float_:
    """Draw one sample from the posterior distribution for the arm."""
    return arm.sample(X, size=1).item()  # type: ignore


def _compute_arm_upper_bound(
    arm: ArmProtocol,
    X: Optional[ArrayLike] = None,
    *,
    alpha: float = 0.68,
    samples=1000,
) -> np.float_:
    """Compute the upper bound of a one-sided credible interval with size
    `alpha` from the posterior distribution for the arm."""
    posterior_samples = arm.sample(X, size=samples)
    posterior_samples = cast(NDArray[np.float64], posterior_samples)

    return np.quantile(posterior_samples, q=alpha)  # type: ignore


def _compute_arm_mean(
    arm: ArmProtocol,
    X: Optional[ArrayLike] = None,
    *,
    samples=1000,
) -> np.float_:
    """Compute the mean of the posterior distribution for the arm."""
    posterior_samples = arm.sample(X, size=samples)
    posterior_samples = cast(NDArray[np.float64], posterior_samples)

    return np.mean(posterior_samples)
