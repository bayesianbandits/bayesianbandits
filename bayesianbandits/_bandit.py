from copy import deepcopy
from dataclasses import dataclass, field
from functools import cached_property, partial
from typing import Any, Callable, Dict, Optional, Union, cast

import numpy as np
from numpy.typing import ArrayLike, NDArray
from sklearn import clone  # type: ignore

from ._typing import ArmProtocol, BanditProtocol, Learner, BanditConstructor


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

    def __repr__(self) -> str:
        return (
            f"Arm(action_function={self.action_function},"
            f" reward_function={self.reward_function}"
        )


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


def bandit(
    learner: Learner,
    choice: Callable[[BanditProtocol, Optional[ArrayLike]], ArmProtocol],
    **options: Any,
) -> Callable[[type], BanditConstructor]:
    """Decorator to create a multi-armed bandit from a class definition.

    The class definition should define the arms as attributes. The
    attributes should be instances of `Arm`. Instances of the decorated
    class will have the arms as attributes and will implement the following
    methods:
    - `pull`: Pull the arm according to the `choice` algorithm.
    - `sample`: Sample from the posterior distribution of the bandit.
    - `update`: Update one of the arms with new data.

    Parameters
    ----------
    learner : Learner
        Learner to use for each arm in the bandit.
    choice : Callable[[BanditProtocol, Optional[ArrayLike]], ArmProtocol]
        Constructor for making a choice algorithm to use for
        choosing which arm to pull.

    Returns
    -------
    Callable[[object], BanditConstructor]
        Class decorator that creates a bandit class from a class definition with
        arms defined as attributes.

    Raises
    ------
    ValueError
        If the class definition does not have any arms defined as attributes.


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

    def wrapper(cls: type) -> BanditConstructor:
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

        if Arm not in cls.__annotations__.values():
            raise ValueError(f"No arms defined in the {cls.__name__} definition.")

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
        def _arms(self: BanditProtocol) -> Dict[str, Arm]:
            return {
                name: attr
                for name, attr in self.__dict__.items()
                if isinstance(attr, Arm)
            }

        setattr(cls, "arms", cached_property(_arms))
        cls.arms.__set_name__(cls, "arms")  # type: ignore

        setattr(cls, "__post_init__", _bandit_post_init)
        setattr(cls, "pull", _bandit_choose_and_pull)
        setattr(cls, "update", _bandit_update)
        setattr(cls, "sample", _bandit_sample)
        setattr(cls, "choice_algorithm", choice)

        return dataclass(cls)

    return wrapper
