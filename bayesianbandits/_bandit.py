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

from ._typing import ArmProtocol, BanditProtocol, Learner, DecayingLearner


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
        self.action_function()

    def sample(
        self,
        X: Optional[ArrayLike] = None,
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

    def _bandit_pull(
        self: BanditProtocol,
        X: Optional[ArrayLike] = None,
        **kwargs: Any,
    ) -> None:
        """Choose an arm and pull it. Set `last_arm_pulled` to the name of the
        arm that was pulled.

        This method is added to the bandit class by the `bandit` decorator.

        Parameters
        ----------
        X : ArrayLike
            Context for the bandit.
        """
        if X is None and self._contextual:
            raise ValueError("X must be an array-like for a contextual bandit.")
        elif X is not None and not self._contextual:
            raise ValueError("X must be None for a non-contextual bandit.")

        arm = self.policy(X=X)
        self.last_arm_pulled = arm
        arm.pull()

    def _bandit_update(
        self: BanditProtocol,
        X: ArrayLike,
        y: Optional[ArrayLike] = None,
        **kwargs: Any,
    ) -> None:
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
        if y is None and self._contextual:
            raise ValueError(
                "X and y must both be array-likes for a contextual bandit."
            )
        elif y is not None and not self._contextual:
            raise ValueError(
                "The second argument must be None for a non-contextual bandit."
                " The first argument to `update` must be the outcome."
            )

        self.arm_to_update.update(X, y)

    def _bandit_sample(
        self: BanditProtocol,
        X: Optional[ArrayLike] = None,
        *,
        size: int = 1,
        **kwargs: Any,
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
        if X is None and self._contextual:
            raise ValueError("X must be an array-like for a contextual bandit.")
        elif X is not None and not self._contextual:
            raise ValueError("X must be None for a non-contextual bandit.")
        # choose an arm, draw a sample, and repeat `size` times
        # TODO: this is not the most efficient way to do this
        # but I can't imagine a situation where this would be a bottleneck.
        return np.array([self.policy(X=X).sample(X=X) for _ in range(size)])

    def arm_to_update(self: BanditProtocol) -> ArmProtocol:
        """Returns the arm that was last pulled."""
        if self.last_arm_pulled is None:
            raise ValueError("No arm has been pulled yet.")
        return self.last_arm_pulled

    def _bandit_post_init(self: BanditProtocol) -> None:
        """Moves all class attributes that are instances of `Arm` to instance
        attributes.

        This ensures that the bandit can be pickled."""

        self.rng = np.random.default_rng(self.rng)
        self._contextual = False

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


def contextual(
    cls: Type[BanditProtocol],
) -> Type[BanditProtocol]:
    """Decorator for making a bandit contextual.

    This decorator adds methods to the bandit class that allow it to be used
    in a contextual setting. The `pull` and `sample` methods will take `X`
    arguments, and the `update` method will take `X` and `y`.

    Parameters
    ----------
    cls : BanditConstructor
        Bandit class to make contextual.

    Returns
    -------
    BanditConstructor
        Contextual bandit class.
    """

    check_is_bandit(cls)

    orig_post_init = cls.__post_init__  # type: ignore

    def _contextual_post_init(self: BanditProtocol) -> None:
        """Moves all class attributes that are instances of `Arm` to instance
        attributes.

        This ensures that the bandit can be pickled."""

        orig_post_init(self)
        self._contextual = True

    setattr(cls, "__post_init__", _contextual_post_init)

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
        check_is_bandit(cls)

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
            X: Optional[ArrayLike] = None,
            **kwargs: Any,
        ) -> None:
            """
            Choose an arm and pull it. Save the unique id and the name of the
            arm that was pulled in the cache.

            Parameters
            ----------
            unique_id : Any
                Unique id for the event.
            """
            unique_id = kwargs.pop("unique_id", None)
            if unique_id is None:
                raise ValueError("unique_id must be provided.")
            orig_pull(self, X)
            (cache[unique_id],) = [
                k for k, v in self.arms.items() if v is self.last_arm_pulled
            ]

        @wraps(orig_update)
        def _delayed_reward_update(
            self: BanditProtocol,
            X: ArrayLike,
            y: Optional[ArrayLike] = None,
            **kwargs: Any,
        ) -> None:
            """
            Update the learner for the arm corresponding to the unique id.

            Parameters
            ----------
            unique_id : Any
                Unique id for the event.

            """
            unique_id = kwargs.pop("unique_id", None)
            if unique_id is None:
                raise ValueError("unique_id must be provided.")

            self.arm_to_update = unique_id  # type: ignore
            orig_update(self, X, y)
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


def restless(
    cls: Type[BanditProtocol],
) -> Type[BanditProtocol]:
    """Decorator for restless bandits.

    This decorator ensures that the `decay` method of each unselected
    arm is called at each update. The specific behavior of the `decay`
    method is left to the learner.

    """

    check_is_bandit(cls)

    orig_update = cls.update

    @wraps(orig_update)
    def _restless_update(
        self: BanditProtocol,
        X: ArrayLike,
        y: Optional[ArrayLike] = None,
        **kwargs: ArrayLike,
    ) -> None:
        """Update the learner for the last arm pulled.

        If `y` is not provided, the `X` argument is assumed to be
        the outcome and the context is a vector of ones.

        Parameters
        ----------
        X : ArrayLike
            Context for the last arm pulled.
        y : Optional[ArrayLike], optional
            Outcome for the last arm pulled.

        Raises
        ------
        ValueError
            If no arm has been pulled yet.

        """
        orig_update(self, X, y)
        for arm in self.arms.values():
            if arm is not self.last_arm_pulled:
                arm.decay(X, y)

    setattr(cls, "update", _restless_update)

    return cast(Type[BanditProtocol], cls)


def check_is_bandit(cls):
    if (
        not hasattr(cls, "pull")
        or not hasattr(cls, "sample")
        or not hasattr(cls, "update")
    ):
        raise ValueError("Decorated class must be a bandit. Are you missing @bandit?")
