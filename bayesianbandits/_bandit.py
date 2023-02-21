from copy import deepcopy
from dataclasses import dataclass, field
from functools import cached_property, partial, wraps
from typing import (
    Any,
    Callable,
    Dict,
    MutableMapping,
    Optional,
    Type,
    Union,
    cast,
)

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
            y_fit = np.atleast_1d(X)
            X_fit = np.ones_like(y_fit, dtype=np.float64)[:, np.newaxis]
        else:
            y_fit, X_fit = np.atleast_1d(y), np.atleast_2d(X)

        self.learner.partial_fit(X_fit, y_fit)

    def __repr__(self) -> str:
        return (
            f"Arm(action_function={self.action_function},"
            f" reward_function={self.reward_function}"
        )


def bandit(
    learner: Learner,
    policy: Callable[[BanditProtocol, Optional[ArrayLike]], ArmProtocol],
) -> Callable[[type], Type[BanditProtocol]]:
    """Decorator to create a contextual multi-armed bandit from a class definition.

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

    def _bandit_pull(self: BanditProtocol, X: ArrayLike) -> None:
        """Choose an arm and pull it. Set `last_arm_pulled` to the name of the
        arm that was pulled.

        This method is added to the bandit class by the `bandit` decorator.

        Parameters
        ----------
        X : ArrayLike
            Context for the bandit.
        """
        arm = self.policy(X=np.atleast_2d(X))
        self.last_arm_pulled = arm
        arm.pull()

    def _bandit_update(self: BanditProtocol, X: ArrayLike, y: ArrayLike) -> None:
        """Update the learner for the last arm pulled.

        This method is added to the bandit class by the `bandit` decorator.

        Parameters
        ----------
        X: ArrayLike
            Context for the last arm pulled.s
        y : ArrayLike
            Outcome for the last arm pulled.

        Raises
        ------
        ValueError
            If no arm has been pulled yet.
        """
        self.arm_to_update.update(X, y)

    def _bandit_sample(
        self: BanditProtocol,
        X: ArrayLike,
        *,
        size: int = 1,
    ) -> ArrayLike:
        """Sample from the bandit by choosing an arm according to the
        context vector `X`. For each sample, the arm is chosen according
        to the `policy` algorithm and then a sample is drawn from the
        learner.

        Parameters
        ----------
        X : ArrayLike
            Context for the bandit.
        size : int, default=1
            Number of samples to draw.
        """
        # choose an arm, draw a sample, and repeat `size` times
        # TODO: this is not the most efficient way to do this
        # but I can't imagine a situation where this would be a bottleneck.
        return np.array(
            [
                self.policy(X=np.atleast_2d(X)).sample(X=np.atleast_2d(X))
                for _ in range(size)
            ]
        )

    def arm_to_update(self: BanditProtocol) -> ArmProtocol:
        """Returns the arm that was last pulled."""
        if self.last_arm_pulled is None:
            raise ValueError("No arm has been pulled yet.")
        return self.last_arm_pulled

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

    def wrapper(cls: type) -> Type[BanditProtocol]:
        """Adds methods to the bandit class."""

        # annotate the arm variables as Arms so that dataclasses
        # know they're not class variables.
        if not hasattr(cls, "__annotations__"):
            setattr(cls, "__annotations__", {})

        for name, attr in cls.__dict__.items():
            if isinstance(attr, ArmProtocol):
                cls.__annotations__[name] = ArmProtocol
                # set the arm to be a field with a defaultfactory of deepcopying
                # the arm
                setattr(cls, name, field(default_factory=partial(deepcopy, attr)))

        if ArmProtocol not in cls.__annotations__.values():
            raise ValueError(f"No arms defined in the {cls.__name__} definition.")

        # annotate rng as a random generator, int, or None, and give it a default
        # value of None
        cls.__annotations__["rng"] = Union[np.random.Generator, int, None]
        setattr(cls, "rng", field(default=None))

        # annotate last_arm_pulled as an ArmProtocol or None, and make sure
        # it is not initialized
        cls.__annotations__["last_arm_pulled"] = Union[ArmProtocol, None]
        setattr(cls, "last_arm_pulled", field(default=None, init=False))

        setattr(
            cls,
            "arm_to_update",
            property(arm_to_update, doc="Returns the arm that was last pulled."),
        )

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

        setattr(cls, "__post_init__", _bandit_post_init)
        setattr(cls, "pull", _bandit_pull)
        setattr(cls, "update", _bandit_update)
        setattr(cls, "sample", _bandit_sample)
        setattr(cls, "policy", policy)

        return dataclass(cls)

    return wrapper


def contextfree(
    cls: Type[BanditProtocol],
) -> Type[BanditProtocol]:
    """Decorator for making a bandit context-free.

    This decorator adds methods to the bandit class that allow it to be used
    in a context-free setting. The `pull` and `sample` methods will take no
    arguments, and the `update` method will take a single argument `y`.

    Parameters
    ----------
    cls : BanditConstructor
        Bandit class to make contextual.

    Returns
    -------
    BanditConstructor
        Contextual bandit class.

    Raises
    ------
    ValueError
        If the bandit is already contextual.
    """

    if (
        not hasattr(cls, "pull")
        or not hasattr(cls, "sample")
        or not hasattr(cls, "update")
    ):
        raise ValueError("Decorated class must be a bandit. Are you missing @bandit?")

    orig_pull = cls.pull
    orig_sample = cls.sample
    orig_update = cls.update

    @wraps(orig_pull)
    def _contextfree_pull(self: BanditProtocol, **kwargs: Any) -> None:
        """Choose an arm and pull it. Set `last_arm_pulled` to the name of the
        arm that was pulled.

        """
        return orig_pull(self, X=None, **kwargs)

    @wraps(orig_update)
    def _contextfree_update(self: BanditProtocol, y: ArrayLike, **kwargs: Any) -> None:
        """Update the learner for the last arm pulled.

        Parameters
        ----------
        y : ArrayLike
            Outcome for the last arm pulled.

        Raises
        ------
        ValueError
            If no arm has been pulled yet.
        """
        return orig_update(self, X=y, y=None, **kwargs)

    @wraps(orig_sample)
    def _contextfree_sample(
        self: BanditProtocol, *, size: int = 1, **kwargs: Any
    ) -> ArrayLike:
        """Sample from the bandit by choosing an arm according to the choice
        algorithm and sampling from the arm's learner.

        Parameters
        ----------
        size : int, default=1
            Number of samples to draw.
        """
        return orig_sample(self, X=None, size=size, **kwargs)

    setattr(cls, "pull", _contextfree_pull)
    setattr(cls, "sample", _contextfree_sample)
    setattr(cls, "update", _contextfree_update)

    return cls


def delayed_reward(
    cls: Optional[Type[BanditProtocol]] = None,
    *,
    cache: Optional[MutableMapping[Any, str]] = None,
) -> Type[BanditProtocol]:
    """Decorator for handling delayed rewards.

    This decorator adds a `unique_id` argument to the `pull` and `update`
    methods. Upon pulling, the `unique_id` and the name of the arm that was
    pulled are stored in the cache. Upon updating, the `unique_id` is used to
    look up the arm that was pulled and the reward is passed to the arm's
    learner.

    Parameters
    ----------
    cls : BanditConstructor, optional
        Bandit class to modify. If not provided, the decorator can be used
        as a function.
    cache : MutableMapping, optional
        Cache to use for storing the unique ids. If not provided, an
        in-memory dictionary is used. As long as the cache exposes a `dict`-like
        interface, it can be used.

    """

    if cache is None:
        cache = {}

    def _delayed_reward_impl(cls: Type[BanditProtocol]) -> Type[BanditProtocol]:
        if (
            not hasattr(cls, "pull")
            or not hasattr(cls, "sample")
            or not hasattr(cls, "update")
        ):
            raise ValueError(
                "Decorated class must be a bandit. Are you missing @bandit?"
            )

        # change the `arm_to_update` property to have a setter that looks up
        # the arm in the cache
        def arm_to_update_getter(self: BanditProtocol) -> ArmProtocol:
            if self.last_arm_pulled is None:
                raise ValueError("No arm has been pulled yet.")
            return self.last_arm_pulled

        def arm_to_update_setter(self: BanditProtocol, unique_id: Any) -> None:
            try:
                self.last_arm_pulled = self.arms[cache[unique_id]]
            except KeyError:
                raise ValueError(f"No event with unique_id {unique_id}.")

        setattr(
            cls,
            "arm_to_update",
            property(fget=arm_to_update_getter, fset=arm_to_update_setter),
        )

        orig_pull = cls.pull
        orig_update = cls.update
        orig_post_init = cls.__post_init__  # type: ignore

        @wraps(orig_pull)
        def _delayed_reward_pull(
            self: BanditProtocol,
            *args: ArrayLike,
            unique_id: Any,
            **kwargs: ArrayLike,
        ) -> None:
            """
            Choose an arm and pull it. Save the unique id and the name of the
            arm that was pulled in the cache.

            Parameters
            ----------
            unique_id : Any
                Unique id for the event.
            """
            orig_pull(self, *args, **kwargs)
            (cache[unique_id],) = [
                k for k, v in self.arms.items() if v is self.last_arm_pulled
            ]

        @wraps(orig_update)
        def _delayed_reward_update(
            self: BanditProtocol,
            *args: ArrayLike,
            unique_id: Any,
            **kwargs: ArrayLike,
        ) -> None:
            """
            Update the learner for the arm corresponding to the unique id.

            Parameters
            ----------
            unique_id : Any
                Unique id for the event.

            """
            self.arm_to_update = unique_id  # type: ignore
            orig_update(self, *args, **kwargs)
            del cache[unique_id]

        @wraps(orig_post_init)  # type: ignore
        def _delayed_reward_post_init(self: BanditProtocol) -> None:
            orig_post_init(self)
            self.__cache__ = cache  # type: ignore

        setattr(cls, "pull", _delayed_reward_pull)
        setattr(cls, "update", _delayed_reward_update)
        setattr(cls, "__post_init__", _delayed_reward_post_init)

        return cast(Type[BanditProtocol], cls)

    if cls is not None:
        return _delayed_reward_impl(cls)

    return cast(Type[BanditProtocol], _delayed_reward_impl)
