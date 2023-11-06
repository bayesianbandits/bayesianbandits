from __future__ import annotations

from functools import wraps
from typing import Any, Callable, Optional, TypeVar, Union, cast

import numpy as np
from numpy.typing import NDArray
from scipy.sparse import csc_array
from typing_extensions import Concatenate, ParamSpec

from ._typing import ActionToken, DecayingLearner, Learner

P = ParamSpec("P")
R = TypeVar("R", covariant=True)
RewardFunction = Union[
    Callable[..., np.float_],
    Callable[..., NDArray[np.float_]],
    Callable[..., Union[np.float_, NDArray[np.float_]]],
]


def requires_learner(
    func: Callable[Concatenate[Arm, P], R]
) -> Callable[Concatenate[Arm, P], R]:
    """Decorator to check if the arm has a learner set."""

    @wraps(func)
    def wrapper(self: "Arm", *args: P.args, **kwargs: P.kwargs) -> R:
        if self.learner is None:
            raise ValueError("Learner is not set.")
        return func(self, *args, **kwargs)

    return wrapper


def identity(
    x: Union[np.float_, NDArray[np.float_]]
) -> Union[np.float_, NDArray[np.float_]]:
    return x


class Arm:
    """Arm of a bandit.

    Parameters
    ----------
    action_token : Any
        Token to return when the arm is pulled. This should be something processed
        by the user's code to execute the action associated with the arm.
    reward_function : Optional[RewardFunction], default=None
        Function to call to compute the reward. Takes the output of the learner's
        `sample` function as input and should return a scalar reward. Should take
        a single scalar or array-like argument and return a scalar or
        1D array. If None, the identity function is used.
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
    >>> arm.update(np.array([[1]]), np.array([1]))

    """

    def __init__(
        self,
        action_token: Any,
        reward_function: Optional[RewardFunction] = None,
        learner: Optional[Learner] = None,
    ) -> None:
        self.action_token = ActionToken(action_token)
        if reward_function is None:
            reward_function = identity
        self.reward_function = reward_function
        self.learner = learner

    def __set_name__(self, owner: type, name: str) -> None:
        self.name = name

    @requires_learner
    def pull(self) -> ActionToken:
        """Pull the arm."""
        return self.action_token

    @requires_learner
    def sample(
        self,
        X: Union[csc_array, NDArray[np.float_]],
        size: int = 1,
    ) -> NDArray[np.float_]:
        """Sample from learner and compute the reward."""
        return self.reward_function(self.learner.sample(X, size))  # type: ignore

    @requires_learner
    def update(
        self, X: Union[csc_array, NDArray[np.float_]], y: NDArray[np.float_]
    ) -> None:
        """Update the learner.

        If y is None, the data in X is used as the target and X is set to
        a `len(X)` rows of ones.
        """
        assert self.learner is not None  # for type checker
        # sparse learners are supported by some, but not all learners
        self.learner.partial_fit(X, y)  # type: ignore

    @requires_learner
    def decay(
        self,
        X: Union[csc_array, NDArray[np.float_]],
        *,
        decay_rate: Optional[float] = None,
    ) -> None:
        """Decay the learner.

        Takes a `y` argument for consistency with `update` but does not use it."""
        if not hasattr(self.learner, "decay"):
            raise ValueError("Learner does not have a decay method.")
        # sparse learners are supported by some, but not all learners
        cast(DecayingLearner, self.learner).decay(X, decay_rate=decay_rate)  # type: ignore

    def __repr__(self) -> str:
        return (
            f"Arm(action_token={self.action_token},"
            f" reward_function={self.reward_function}"
        )
