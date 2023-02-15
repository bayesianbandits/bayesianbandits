from typing import Callable, Iterable, Optional, Protocol, Union, runtime_checkable
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


@runtime_checkable
class ArmBanditProtocol(Protocol):
    """Protocol for Arms and Bandits. Bandits themselves can be used as arms
    in other bandits, so both must implement the same minimal interface.

    Each Arm or Bandit must implement the following methods:
    - `pull`
    - `sample`
    - `update`

    """

    def pull(self):
        ...

    def sample(self, X=None, size: int = 1):
        ...

    def update(self, X, y=None):
        ...


class ChoiceAlgorithm(Protocol):
    """Choice algorithm protocol for choosing which arm to pull.

    Each ChoiceAlgorithm must implement the following methods:
    - `choose`

    """

    def choose(self, arms: Iterable[ArmBanditProtocol]) -> ArmBanditProtocol:
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
        Function to call to compute the reward. Takes the output of the learner's
        `sample` function as input and should return a scalar reward.
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

    def __set_name__(self, owner, name):
        self.name = name

    def __get__(self, obj, objtype=None):
        if obj is None:
            return self
        return obj.__dict__[self.name]

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
            f" reward_function={self.reward_function}),"
            f" learner={self.learner}"
        )


def bandit(learner: Learner, choice_algorithm: ChoiceAlgorithm) -> ArmBanditProtocol:
    """Decorator to create a context-free multi-armed bandit from a class definition
    with arms defined as attributes.

    Parameters
    ----------
    learner : Learner
        Learner to use for each arm in the bandit.
    choice_algorithm : ChoiceAlgorithm
        Choice algorithm to use for choosing which arm to pull.

    Returns
    -------
    ArmBanditProtocol
        Bandit class instance with arms defined as attributes.
    """

    def wrapper(cls):
        for _, attr in cls.__dict__.items():
            if isinstance(attr, Arm):
                if attr.learner is None:
                    attr.learner = learner

        if not hasattr(cls, "arms"):
            cls._arms = {
                name: attr
                for name, attr in cls.__dict__.items()
                if isinstance(attr, ArmBanditProtocol)
            }

        if not hasattr(cls, "choice_algorithm"):
            cls._choice_algorithm = choice_algorithm

        if not hasattr(cls, "pull"):
            cls.pull = _bandit_choose_and_pull

        if not hasattr(cls, "last_arm_pulled"):
            cls.last_arm_pulled = None

        if not hasattr(cls, "update"):
            cls.update = _bandit_update

        if not hasattr(cls, "sample"):
            cls.sample = _bandit_sample

        return cls

    return wrapper


def _bandit_choose_and_pull(self):
    """Choose an arm and pull it. Set `last_arm_pulled` to the name of the arm
    that was pulled.

    This method is added to the bandit class by the `bandit` decorator.
    """
    arm = self.choice_algorithm.choose(self.arms.values())
    self.last_arm_pulled = arm.name
    arm.pull()


def _bandit_update(self, X, y=None):
    """Update the learner for the last arm pulled.

    This method is added to the bandit class by the `bandit` decorator.
    """
    if self.last_arm_pulled is None:
        raise ValueError("No arm has been pulled yet.")
    arm = self.arms[self.last_arm_pulled]
    arm.update(X, y)


def _bandit_sample(self, X=None, size: int = 1):
    """Sample from the learner using the choice algorithm.

    This method is added to the bandit class by the `bandit` decorator."""
    arm = self.choice_algorithm.choose(self.arms.values())
    return arm.sample(X, size)
