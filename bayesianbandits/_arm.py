from __future__ import annotations
from typing import Callable, Optional, cast

import numpy as np
from numpy.typing import ArrayLike, NDArray

from ._typing import DecayingLearner, Learner


class Arm:
    """Arm of a bandit.

    Parameters
    ----------
    action_function : Callable
        Nullary function to call when the arm is pulled. Should have
        either directly produce the reward or have a side effect that
        eventually produces reward. For example, if the arm represents an action
        to take in an experiment, the action function should perform the
        database query to update the experiment table with the action to take.
        Later, the `update` method should be called with the computed reward.
    reward_function : Callable
        Function to call to compute the reward. Takes the output of the learner's
        `sample` function as input and should return a scalar reward.
    learner : Optional[Learner], default=None
        Learner to use for the arm. If None, the arm cannot be used.

    Examples
    --------
    >>> from bayesianbandits import Arm
    >>> import numpy as np
    >>> def action_function():
    ...     print("Action taken.")
    >>> def reward_function(sample):
    ...     return sample
    >>> class MyLearner:
    ...     def sample(self, X, size=1):
    ...         np.random.seed(0)
    ...         return np.random.normal(size=size)
    ...     def partial_fit(self, X, y):
    ...         pass
    >>> learner = MyLearner()
    >>> arm = Arm(action_function, reward_function, learner)
    >>> arm.pull()
    Action taken.
    >>> arm.update(1)

    """

    def __init__(
        self,
        action_function: Callable[[], None],
        reward_function: Callable[[ArrayLike], ArrayLike],
        learner: Optional[Learner] = None,
    ) -> None:
        self.action_function = action_function
        self.reward_function = reward_function
        self.learner = learner

    def pull(self) -> None:
        """Pull the arm."""
        if self.learner is None:
            raise ValueError("Learner is not set.")
        return self.action_function()

    def sample(
        self,
        X: Optional[ArrayLike] = None,
        size: int = 1,
    ) -> NDArray[np.float_]:
        """Sample from learner and compute the reward."""
        if self.learner is None:
            raise ValueError("Learner is not set.")
        if X is None:
            X_new = np.array([[1]])
        else:
            X_new = np.atleast_2d(X)

        return self.reward_function(self.learner.sample(X_new, size))  # type: ignore

    def update(self, X: ArrayLike, y: Optional[ArrayLike] = None) -> None:
        """Update the learner.

        If y is None, the data in X is used as the target and X is set to
        a `len(X)` rows of ones.
        """
        if self.learner is None:
            raise ValueError("Learner is not set.")
        if y is None:
            y_fit = np.atleast_1d(X)
            X_fit = np.ones_like(y_fit, dtype=np.float64)[:, np.newaxis]
        else:
            y_fit, X_fit = np.atleast_1d(y), np.atleast_2d(X)

        self.learner.partial_fit(X_fit, y_fit)

    def decay(self, X: ArrayLike, y: Optional[ArrayLike] = None) -> None:
        """Decay the learner.

        Takes a `y` argument for consistency with `update` but does not use it."""
        if self.learner is None:
            raise ValueError("Learner is not set.")
        if not hasattr(self.learner, "decay"):
            raise ValueError("Learner does not have a decay method.")
        if y is None:
            y_fit = np.atleast_1d(X)
            X_fit = np.ones_like(y_fit, dtype=np.float64)[:, np.newaxis]
        else:
            y_fit, X_fit = np.atleast_1d(y), np.atleast_2d(X)

        cast(DecayingLearner, self.learner).decay(X_fit)

    def __repr__(self) -> str:
        return (
            f"Arm(action_function={self.action_function},"
            f" reward_function={self.reward_function}"
        )
