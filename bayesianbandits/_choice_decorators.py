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

    def _compute_arm_mean(
        arm: ArmProtocol,
        X: Optional[ArrayLike] = None,
    ) -> np.float_:
        """Compute the mean of the posterior distribution for the arm."""
        if arm.learner is None:
            raise ValueError("Learner is not set.")
        posterior_samples = arm.sample(X, size=samples)
        posterior_samples = cast(NDArray[np.float64], posterior_samples)

        return np.mean(posterior_samples)

    def _choose_arm(
        self: BanditProtocol,
        X: Optional[ArrayLike] = None,
    ) -> ArmProtocol:
        """Choose an arm using epsilon-greedy."""

        if self.rng.random() < epsilon:  # type: ignore
            return self.rng.choice(list(self.arms.values()))  # type: ignore
        else:
            key_func = partial(_compute_arm_mean, X=X)
            return max(self.arms.values(), key=key_func)  # type: ignore

    return _choose_arm
