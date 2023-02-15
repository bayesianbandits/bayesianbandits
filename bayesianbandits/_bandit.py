from typing import Callable, Optional, Protocol, Union
import numpy as np


class Learner(Protocol):
    """Learner protocol for the model underlying each arm.

    Each Learner must implement the following methods:
    - `sample`
    - `partial_fit`

    """

    def sample(self, X, size: int = 1) -> Union[float, np.ndarray]:
        ...

    def partial_fit(self, X, y):
        ...


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
        Function to call to compute the reward. Takes
        the output of the learner's `sample` function as input.
    learner : Optional[Learner], default=None
        Learner to use for the arm. If None, the arm cannot be used.

    Examples
    --------
    >>> from bayesianbandits import Learner, Arm
    >>> import numpy as np
    >>> def action_function():
    ...     print("Action taken.")
    >>> def reward_function(sample):
    ...     return sample
    >>> class MyLearner(Learner):
    ...     def sample(self, X, size=1):
    ...         np.random.seed(0)
    ...         return np.random.normal(size=size)
    ...     def partial_fit(self, X, y):
    ...         pass
    >>> learner = MyLearner()
    >>> arm = Arm(action_function, reward_function, learner)
    >>> arm.pull()
    Action taken.
    >>> arm.sample()
    array([1.76405235])
    >>> arm.update(np.array([1.76405235]))

    """

    def __init__(
        self,
        action_function: Callable,
        reward_function: Callable,
        learner: Optional[Learner] = None,
    ):
        self.action_function = action_function
        self.reward_function = reward_function
        self.learner = learner

    def pull(self):
        """Pull the arm."""
        if self.learner is None:
            raise ValueError("Learner is not set.")
        self.action_function()

    def sample(self, X=None, size: int = 1):
        """Sample from learner and compute the reward."""
        if self.learner is None:
            raise ValueError("Learner is not set.")
        X = X or np.array([[1]])
        return self.reward_function(self.learner.sample(X, size))

    def update(self, X, y=None):
        """Update the learner.

        If y is None, the data in X is used as the target and X is set to
        a `len(X)` rows of ones.
        """
        if self.learner is None:
            raise ValueError("Learner is not set.")
        if y is None:
            y, X = np.atleast_1d(X), np.array([[1]]).repeat(len(X), axis=0)
        self.learner.partial_fit(X, y)

    def __repr__(self):
        return (
            f"Arm(action_function={self.action_function},"
            f" reward_function={self.reward_function})"
        )
