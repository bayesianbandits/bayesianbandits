from __future__ import annotations
from typing import Callable, Dict, Optional, TypeVar, cast, Any

import numpy as np
from numpy.typing import ArrayLike, NDArray

from ._typing import DecayingLearner, Learner, ActionToken


R = TypeVar("R")


def requires_learner(func: Callable[..., R]) -> Callable[..., R]:
    """Decorator to check if the arm has a learner set."""

    def wrapper(self: Arm, *args: Any, **kwargs: Dict[str, Any]):
        if self.learner is None:
            raise ValueError("Learner is not set.")
        return func(self, *args, **kwargs)

    return wrapper


class Arm:
    """Arm of a bandit.

    Parameters
    ----------
    action_token : Any
        Token to return when the arm is pulled. This should be something processed
        by the user's code to execute the action associated with the arm.
    reward_function : Callable
        Function to call to compute the reward. Takes the output of the learner's
        `sample` function as input and should return a scalar reward.
    learner : Optional[Learner], default=None
        Learner to use for the arm. If None, the arm cannot be used.

    Examples
    --------
    >>> from bayesianbandits import Arm
    >>> import numpy as np
    >>> def reward_function(sample):
    ...     return sample
    >>> class MyLearner:
    ...     def sample(self, X, size=1):
    ...         np.random.seed(0)
    ...         return np.random.normal(size=size)
    ...     def partial_fit(self, X, y):
    ...         pass
    >>> learner = MyLearner()
    >>> arm = Arm("Action taken.", reward_function, learner)
    >>> arm.pull()
    'Action taken.'
    >>> arm.update(1)

    """

    def __init__(
        self,
        action_token: Any,
        reward_function: Callable[[ArrayLike], ArrayLike],
        learner: Optional[Learner] = None,
    ) -> None:
        self.action_token = ActionToken(action_token)
        self.reward_function = reward_function
        self.learner = learner

    @requires_learner
    def pull(self) -> ActionToken:
        """Pull the arm."""
        return self.action_token

    @requires_learner
    def sample(
        self,
        X: Optional[ArrayLike] = None,
        size: int = 1,
    ) -> NDArray[np.float_]:
        """Sample from learner and compute the reward."""
        if X is None:
            X_new = np.array([[1]])
        else:
            X_new = np.atleast_2d(X)

        return self.reward_function(self.learner.sample(X_new, size))  # type: ignore

    @requires_learner
    def update(self, X: ArrayLike, y: Optional[ArrayLike] = None) -> None:
        """Update the learner.

        If y is None, the data in X is used as the target and X is set to
        a `len(X)` rows of ones.
        """

        if y is None:
            y_fit = np.atleast_1d(X)
            X_fit = np.ones_like(y_fit, dtype=np.float64)[:, np.newaxis]
        else:
            y_fit, X_fit = np.atleast_1d(y), np.atleast_2d(X)

        assert self.learner is not None  # for type checker
        self.learner.partial_fit(X_fit, y_fit)

    @requires_learner
    def decay(
        self,
        X: NDArray[np.float_],
    ) -> None:
        """Decay the learner.

        Takes a `y` argument for consistency with `update` but does not use it."""
        if not hasattr(self.learner, "decay"):
            raise ValueError("Learner does not have a decay method.")

        cast(DecayingLearner, self.learner).decay(X)

    def __repr__(self) -> str:
        return (
            f"Arm(action_token={self.action_token},"
            f" reward_function={self.reward_function}"
        )
