from copy import deepcopy
from dataclasses import dataclass, field
from functools import cached_property, partial
from typing import Any, Callable, Dict, Optional, Union, cast

import numpy as np
from numpy.typing import ArrayLike, NDArray
from sklearn import clone  # type: ignore

from ._typing import ArmProtocol, BanditProtocol, Learner


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
        self.action_function()

    def sample(
        self,
        X: Optional[NDArray[Any]] = None,
        size: int = 1,
    ) -> NDArray[np.float_]:
        """Sample from learner and compute the reward."""
        if self.learner is None:
            raise ValueError("Learner is not set.")
        X_new = X or np.array([[1]])

        return self.reward_function(self.learner.sample(X_new, size))  # type: ignore

    def update(self, X: ArrayLike, y: Optional[ArrayLike] = None) -> None:
        """Update the learner.

        If y is None, the data in X is used as the target and X is set to
        a `len(X)` rows of ones.
        """
        if self.learner is None:
            raise ValueError("Learner is not set.")
        if y is None:
            y_fit, X_fit = np.atleast_1d(X), np.array([[1]]).repeat(
                len(np.array(X)), axis=0
            )
        else:
            y_fit, X_fit = np.atleast_1d(y), np.atleast_2d(X)

        self.learner.partial_fit(X_fit, y_fit)

    def mean(self, X: Optional[NDArray[Any]], size: int = 1000) -> ArrayLike:
        """
        Mean of the posterior for X.
        """
        if self.learner is None:
            raise ValueError("Learner is not set.")
        posterior_samples = self.sample(X, size=size)
        posterior_samples = cast(NDArray[np.float64], posterior_samples)

        return np.mean(posterior_samples)

    def __repr__(self) -> str:
        return (
            f"Arm(action_function={self.action_function},"
            f" reward_function={self.reward_function}"
        )


def epsilon_greedy(
    epsilon: float = 0.1,
) -> Callable[[BanditProtocol, Optional[ArrayLike]], ArmProtocol]:
    """Creates an epsilon-greedy choice algorithm. To be used with the
    `bandit` decorator.

    Parameters
    ----------
    epsilon : float, default=0.1
        Probability of choosing a random arm.

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
            return max(self.arms.values(), key=lambda arm: arm.mean(X))

    return _choose_arm


def bandit(
    learner: Learner,
    choice: Callable[[BanditProtocol, Optional[ArrayLike]], ArmProtocol],
    **options: Any,
) -> Callable[[object], BanditProtocol]:
    """Decorator to create a context-free multi-armed bandit from a class definition
    with arms defined as attributes.

    Parameters
    ----------
    learner : Learner
        Learner to use for each arm in the bandit.
    choice : Callable[..., ArmProtocol]
        Choice algorithm to use for choosing which arm to pull.

    Returns
    -------
    ArmBanditProtocol
        Bandit class instance with arms defined as attributes.
    """

    contextual: bool = options.get("contextual", False)

    if not contextual:

        def _bandit_choose_and_pull(self: BanditProtocol) -> None:
            """Choose an arm and pull it. Set `last_arm_pulled` to the name of the
            arm that was pulled.

            This method is added to the bandit class by the `bandit` decorator.
            """
            arm = self.choice_algorithm(X=None)
            self.last_arm_pulled = arm
            arm.pull()

        def _bandit_update(self: BanditProtocol, y: ArrayLike) -> None:
            """Update the learner for the last arm pulled.

            This method is added to the bandit class by the `bandit` decorator.

            Parameters
            ----------
            y : ArrayLike
                Outcome for the last arm pulled.

            Raises
            ------
            ValueError
                If no arm has been pulled yet.
            """
            if self.last_arm_pulled is None:
                raise ValueError("No arm has been pulled yet.")
            self.last_arm_pulled.update(np.atleast_1d(y))

        def _bandit_sample(self: BanditProtocol, size: int = 1) -> ArrayLike:
            """Sample from the bandit by choosing an arm according to the choice
            algorithm and sampling from the arm's learner.

            This method is added to the bandit class by the `bandit` decorator.

            Parameters
            ----------
            size : int, default=1
                Number of samples to draw.
            """
            # choose an arm, draw a sample, and repeat `size` times
            # TODO: this is not the most efficient way to do this
            # could be vectorized or parallelized
            return np.array(
                [self.choice_algorithm(X=None).sample() for _ in range(size)]
            )

    def _bandit_post_init(self: BanditProtocol) -> None:
        """Moves all class attributes that are instances of `Arm` to instance
        attributes.

        This ensures that the bandit can be pickled."""

        # initialize the rng. this has to be done this way because the
        # bandit dataclass is frozen
        setattr(self, "rng", np.random.default_rng(self.rng))

        # initialize the arms with copies of the learner and
        # point the learner rng to the bandit rng
        for arm in self.arms.values():
            arm.learner = cast(Learner, clone(learner))
            arm.learner.set_params(random_state=self.rng)

    def wrapper(cls: object) -> BanditProtocol:
        """Adds methods to the bandit class."""

        # annotate the arm variables as Arms so that dataclasses
        # know they're not class variables.
        if not hasattr(cls, "__annotations__"):
            setattr(cls, "__annotations__", {})

        for name, attr in cls.__dict__.items():
            if isinstance(attr, Arm):
                cls.__annotations__[name] = Arm
                # set the arm to be a field with a defaultfactory of deepcopying
                # the arm

                setattr(cls, name, field(default_factory=partial(deepcopy, attr)))

        # annotate rng as a random generator, int, or None, and give it a default
        # value of None
        cls.__annotations__["rng"] = Union[np.random.Generator, int, None]
        setattr(cls, "rng", None)

        # annotate last_arm_pulled as an ArmProtocol or None, and make sure
        # it is not initialized
        cls.__annotations__["last_arm_pulled"] = Union[ArmProtocol, None]
        setattr(cls, "last_arm_pulled", field(default=None, init=False))

        # set arms as a cached_property so that it's only computed once
        # per instance
        def _arms(self: BanditProtocol) -> Dict[str, ArmProtocol]:
            return {
                name: attr
                for name, attr in self.__dict__.items()
                if isinstance(attr, ArmProtocol)
            }

        setattr(cls, "arms", cached_property(_arms))
        cls.arms.__set_name__(cls, "arms")  # type: ignore

        cls = cast(BanditProtocol, cls)
        setattr(cls, "__post_init__", _bandit_post_init)
        setattr(cls, "pull", _bandit_choose_and_pull)
        setattr(cls, "update", _bandit_update)
        setattr(cls, "sample", _bandit_sample)
        setattr(cls, "choice_algorithm", choice)

        return dataclass(cls)  # type: ignore

    return wrapper


if __name__ == "__main__":
    from ._estimators import DirichletClassifier

    clf = DirichletClassifier({1: 1, 2: 1, 3: 1})

    def reward_func(x: ArrayLike) -> ArrayLike:
        return np.take(x, 0, axis=-1)  # type: ignore

    def action_func(x: int) -> None:
        print(f"action{x}")

    arm1 = partial(action_func, 1)
    arm2 = partial(action_func, 2)
    arm3 = partial(action_func, 3)

    @bandit(learner=clf, choice=epsilon_greedy(epsilon=0.1))
    class Experiment:
        arm1 = Arm(arm1, reward_func)
        arm2 = Arm(arm2, reward_func)
        arm3 = Arm(arm3, reward_func)
