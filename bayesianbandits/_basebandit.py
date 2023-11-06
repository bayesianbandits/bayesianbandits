from copy import deepcopy
from dataclasses import Field, dataclass, field
from functools import cached_property, partial
from typing import (
    Any,
    ClassVar,
    Collection,
    Dict,
    Hashable,
    List,
    MutableMapping,
    Optional,
    Tuple,
    Type,
    TypeVar,
    Union,
    cast,
    overload,
)
from warnings import warn

import numpy as np
from numpy.typing import ArrayLike, NDArray
from scipy.sparse import csc_array, issparse
from sklearn.base import clone
from typing_extensions import Literal, TypeGuard, dataclass_transform

from ._arm import Arm
from ._np_utils import groupby_array
from ._policy_decorators import ArmChoicePolicy
from ._typing import BanditProtocol, Learner

_B = TypeVar("_B", bound="Bandit")


@dataclass_transform(field_specifiers=(Field[Any], field))
@dataclass
class Bandit:
    """
    Base class for bandits. This class is not meant to be instantiated directly.
    Instead, it should be subclassed and the `learner` and `policy` arguments to
    the `__init_subclass__` method should be set. Optionally, the `delayed_reward`
    argument can be set to `True` to enable delayed rewards.

    Subclass parameters should be passed to `__init_subclass__` during
    subclassing as keyword arguments. These parameters will be added to
    the subclass as class attributes.

    Subclass Parameters
    -------------------

    learner : Learner
        Learner underlying each arm. Must implement `partial_fit` and `sample`.
    policy : Callable[..., Arm]
        Policy for choosing arms. Must take a `Bandit` instance as its first
        argument and return an `Arm` instance.
    delayed_reward : bool, optional
        Whether or not rewards are measured between pulls, by default False.


    Parameters
    ----------
    rng : Union[np.random.Generator, int, None], optional
        Random number generator to use for choosing arms, by default None
    cache : Optional[MutableMapping[Any, str]], optional
        Cache to use for storing arms when `delayed_reward` is set to `True`,
        by default None. If `delayed_reward` is set to `True` and `cache` is
        not set, a `dict` will be used.

    Attributes
    ----------
    learner : Learner
        Learner underlying each arm. Must implement `partial_fit` and `sample`.
    policy : Callable[..., Arm]
        Policy for choosing arms. Must take a `Bandit` instance as its first
        argument and return an `Arm` instance.
    rng: Union[np.random.Generator, None, int]
        Random number generator to use for choosing arms.
    arms: Dict[str, Arm]
        Dictionary of arms.
    last_arm_pulled: Optional[Arm]
        Last arm pulled.
    cache : Optional[MutableMapping[Any, str]]
        Cache to use for storing arms when `delayed_reward` is set to `True`.

    Examples
    --------

    Minimally, a subclass of `Bandit` must pass a `learner` and `policy` to
    `__init_subclass__`. Additionally, all subclasses must define some
    `Arm` instances as class attributes. These will be used to
    initialize the arms of the bandit.

    >>> from bayesianbandits import Arm, GammaRegressor, epsilon_greedy
    >>> clf = GammaRegressor(alpha=1, beta=1)
    >>> policy = epsilon_greedy(0.1)
    >>> def reward_func(x):
    ...     return x
    >>> class MyBandit(Bandit, learner=clf, policy=policy):
    ...     arm1 = Arm("Action 1!", reward_func)
    ...     arm2 = Arm("Action 2!", reward_func)
    >>> bandit = MyBandit(rng=0)

    Once subclassed and instantiated, the `pull` method can be used to pull
    arms. For non-contextual bandits, the `pull` method takes no arguments.
    After pulling an arm, the `update` method can be used to update the
    learner underlying the arm.

    >>> bandit.pull()
    'Action 1!'
    >>> bandit.update(1)

    For delayed reward bandits, the subclass must set the `delayed_reward`
    argument to `__init_subclass__` to `True`.

    >>> class MyDelayedBandit(Bandit, learner=clf, policy=policy, delayed_reward=True):
    ...     arm1 = Arm("Action 1!", reward_func)
    ...     arm2 = Arm("Action 2!", reward_func)
    >>> bandit = MyDelayedBandit(cache={}, rng=0)

    When `delayed_reward` is set to `True`, the `pull` method takes an additional
    argument, `unique_id`, which is used to identify the arm pulled. This is
    used to retrieve the arm from the `cache` when the `update` method is called.

    >>> bandit.pull(unique_id=1)
    'Action 1!'
    >>> bandit.update(1, unique_id=1)

    """

    rng: Union[np.random.Generator, None, int] = field(default=None, repr=False)
    last_arm_pulled: Optional[Arm] = field(default=None, init=False, repr=False)
    cache: Optional[MutableMapping[Any, str]] = field(default=None, repr=False)

    learner: ClassVar[Learner]
    policy: ClassVar[ArmChoicePolicy]

    _delayed_reward: ClassVar[bool] = False

    def __init_subclass__(
        cls,
        /,
        learner: Learner,
        policy: ArmChoicePolicy,
        delayed_reward: bool = False,
        **kwargs: Dict[str, Any],
    ):
        """Initialize a subclass of `Bandit`.

        Parameters
        ----------
        learner : Learner
            Learner to use for each arm.
        policy : Callable[..., Arm]
            Policy to use for choosing arms.
        """
        super().__init_subclass__(**kwargs)

        # set learner and policy as class variables
        cls.learner = learner
        cls.policy = staticmethod(policy)  # type: ignore

        # if delayed reward, set cache as an initializable instance variable,
        # otherwise leave it uninitializable
        if delayed_reward is True:
            cls._delayed_reward = True
            setattr(cls, "cache", field(default_factory=dict, repr=False))
        else:
            setattr(cls, "cache", field(default=None, init=False, repr=False))

        # make sure cache is annotated for dataclass magic
        annotations: dict[str, Any] = getattr(cls, "__annotations__", {})
        annotations["cache"] = Optional[MutableMapping[Any, Arm]]

        # collect and annotate all Arm instances
        arm_annotations: Dict[str, Any] = {}

        for name, attr in cls.__dict__.items():
            if isinstance(attr, Arm):
                arm_annotations[name] = Arm
                setattr(
                    cls,
                    name,
                    field(default_factory=partial(deepcopy, attr), init=False),
                )

        # make sure all Arm instances are first in the annotations
        # for dataclass magic - this is unnecessary in Python 3.10
        other_annotations = {
            k: v for k, v in annotations.items() if k not in arm_annotations
        }
        arm_annotations.update(other_annotations)

        setattr(cls, "__annotations__", arm_annotations)

        # modifies cls in-place
        dataclass(cls)

    @overload
    def pull(self, X: Union[ArrayLike, csc_array], /) -> Any:
        ...

    @overload
    def pull(self, X: Union[ArrayLike, csc_array], /, *, unique_id: Any) -> Any:
        ...

    @overload
    def pull(self, /) -> Any:
        ...

    @overload
    def pull(self, /, *, unique_id: Any) -> Any:
        ...

    def pull(
        self,
        X: Union[ArrayLike, csc_array, None] = None,
        /,
        **kwargs: Any,
    ) -> Union[Any, List[Any]]:
        """Choose an arm and pull it. Set `last_arm_pulled` to the name of the
        arm that was pulled.

        This method is added to the bandit class by the `bandit` decorator.

        Parameters
        ----------
        X : Optional[ArrayLike]
            Context for the bandit. Only provided when the @contextual
            decorator is used.

        Keyword Arguments
        -----------------
        unique_id : Any | Collection[Any]
            Unique identifier for the pull. Required when the `@delayed_reward`
            decorator is used. If a Collection is provided and the bandit is
            contextual, the `X` array must have the same number of rows as the
            length of the Collection. The pull will be performed as a batch and
            the tokens will be returned as a list.

        Returns
        -------
        Any | List[Any]
            Token of the arm pulled. If `unique_id` is a Collection, a list of
            tokens will be returned.

        Raises
        ------
        DelayedRewardException
            Raised when `unique_id` is not provided and `delayed_reward = True`,
            or when any `unique_id` has already been used.
        ValueError
            Raised when `unique_id` is not hashable, or when `X` does not have
            the same number of rows as `unique_id` when `unique_id` is a
            Collection. Also raised when `X` has more than one row when
            `delayed_reward = False`.
        """
        assert isinstance(self.rng, np.random.Generator)  # for the type checker

        X_pull, _ = _validate_arrays(X, None, self._contextual, check_y=False)
        unique_id = None

        if self._delayed_reward is True:
            unique_id: Union[Hashable, Collection[Hashable]] = kwargs.get(
                "unique_id", None
            )

            if unique_id is None:
                raise DelayedRewardException(
                    "The `unique_id` keyword argument is required when the "
                    "`delayed_reward = True`."
                )

            elif isinstance(unique_id, Collection) and not isinstance(unique_id, str):
                return self._pull_batch_delayed_reward(
                    X_pull, cast(Collection[Any], unique_id)
                )

            else:
                return self._pull_single_delayed_reward(X_pull, unique_id)

        return self._pull_single(X_pull)

    def _pull_single(self, X_pull: NDArray[np.float_]) -> Any:
        """Makes a single decision and pulls one arm.

        If given, validates that `X_pull` has only one row, as making several
        decisions for multiple context vectors is not possible when
        `delayed_reward = False`.

        Parameters
        ----------
        X_pull : NDArray[np.float_]
            Context vector - must have only one row.

        Returns
        -------
        Any
            Token of the arm pulled.

        Raises
        ------
        ValueError
            Raised when `X_pull` has more than one row.
        """
        assert isinstance(self.rng, np.random.Generator)  # for the type checker

        if X_pull.shape[0] > 1:
            raise ValueError(
                "The `X` array must have only one row when `delayed_reward = False`."
            )

        arm = self.policy(self.arms, X_pull, self.rng)

        assert isinstance(arm, Arm)  # for the type checker
        ret_val = arm.pull()
        self.last_arm_pulled = arm
        return ret_val

    def _pull_single_delayed_reward(
        self, X: NDArray[np.float_], unique_id: Hashable
    ) -> Any:
        """Makes a single decision and pulls one arm for a delayed reward bandit.

        By calling `_pull_single`, this method validates that `X` has only one
        row. It also validates that `unique_id` is hashable and has not been
        used before. This branch is only taken when `delayed_reward = True` and
        a single `unique_id` is provided.

        Parameters
        ----------
        X : NDArray[np.float_]
            Context vector - must have only one row.
        unique_id : Hashable
            Unique identifier for the pull.

        Returns
        -------
        Any
            Token of the arm pulled.

        Raises
        ------
        ValueError
            Raised when `X` has more than one row.
        ValueError
            Raised when `unique_id` is not hashable.
        DelayedRewardException
            Raised when `unique_id` has already been used.
        """

        assert self.cache is not None

        if not _validate_unique_id(unique_id):
            raise ValueError(
                "The unique_id must be hashable. "
                "Please use a hashable unique identifier."
            )

        if unique_id in self.cache:
            raise DelayedRewardException(
                f"The unique_id {unique_id} has already been used. "
                "Please use a unique identifier."
            )

        ret_val = self._pull_single(X)
        self.cache[unique_id] = cast(Arm, self.last_arm_pulled).name

        return ret_val

    def _pull_batch_delayed_reward(
        self, X: NDArray[np.float_], unique_ids: Collection[Any]
    ) -> List[Any]:
        """Makes a batch of decisions and pulls a batch of arms for a delayed
        reward bandit.

        This method validates that `X` has the same number of rows as
        `unique_ids`. It also validates that `unique_ids` is a collection of
        hashable objects and that none of the `unique_ids` have been used
        before. This branch is only taken when `delayed_reward = True` and a
        collection of `unique_ids` is provided.

        Parameters
        ----------
        X : NDArray[np.float_]
            Context array - must have the same number of rows as `unique_ids`.
        unique_ids : Collection[Any]
            Unique identifiers for the pulls.

        Returns
        -------
        List[Any]
            Tokens of the arms pulled, in the same order as `unique_ids`.

        Raises
        ------
        ValueError
            Raised when `X` does not have the same number of rows as
            `unique_ids`.
        DelayedRewardException
            Raised when any of the `unique_ids` have already been used.
        ValueError
            Raised when `unique_ids` is not a collection of hashable objects.
        """
        assert self.cache is not None  # for the type checker
        assert isinstance(self.rng, np.random.Generator)  # for the type checker

        if not _validate_unique_ids(unique_ids):
            raise ValueError(
                "All unique_ids must be hashable. "
                "Please use hashable unique identifiers."
            )

        _already_used_ids = [
            unique_id for unique_id in unique_ids if unique_id in self.cache
        ]
        if len(_already_used_ids) > 0:
            raise DelayedRewardException(
                f"The unique_ids {_already_used_ids} have already been used. "
                "Please use unique identifiers."
            )

        if self._contextual:
            if len(unique_ids) != X.shape[0]:
                raise ValueError(
                    "The number of unique_ids must match the number of rows in `X`."
                )
            arms = self.policy(self.arms, X, self.rng)
        else:
            arms = self.policy(self.arms, X.repeat(len(unique_ids), axis=0), self.rng)

        if isinstance(arms, Arm):
            arms = [arms]

        ret_vals = [arm.pull() for arm in arms]

        for unique_id, arm in zip(unique_ids, arms):
            self.cache[unique_id] = arm.name

        self.last_arm_pulled = arms[-1]

        return ret_vals

    @overload
    def update(self, y: ArrayLike, /) -> None:
        ...

    @overload
    def update(self, y: ArrayLike, /, *, unique_id: Any) -> None:
        ...

    @overload
    def update(self, X: Union[ArrayLike, csc_array], y: ArrayLike, /) -> None:
        ...

    @overload
    def update(
        self, X: Union[ArrayLike, csc_array], y: ArrayLike, /, *, unique_id: Any
    ) -> None:
        ...

    def update(
        self,
        X: Union[ArrayLike, csc_array],
        y: Optional[ArrayLike] = None,
        /,
        **kwargs: Any,
    ) -> None:
        """Update the learner for the last arm pulled.

        This method is added to the bandit class by the `bandit` decorator.

        Parameters
        ----------
        X: ArrayLike
            Context for the bandit. Only provided when the @contextual
            decorator is used.
        y : ArrayLike
            Outcome for the last arm pulled.

        Raises
        ------
        ValueError
            If no arm has been pulled yet.

        Keyword Arguments
        -----------------
        unique_id : Any | Collection[Any]
            Unique identifier for the pull. Required when the `@delayed_reward`
            decorator is used. If a Collection is provided, all arrays must have
            the same length as the Collection. The update will be performed as a
            batch.
        """
        X_fit, y_fit = _validate_arrays(X, y, self._contextual, check_y=True)

        if self.__class__._delayed_reward is True:
            unique_id: Union[Collection[Any], str, None] = kwargs.get("unique_id", None)
            if unique_id is None:
                raise DelayedRewardException(
                    "The `unique_id` keyword argument is required when the "
                    "`delayed_reward = True`."
                )
            # check if `unique_id` is a non-string iterable
            elif isinstance(unique_id, Collection) and not isinstance(unique_id, str):
                return self._update_batch(
                    X_fit, y_fit, cast(Collection[Any], unique_id)
                )

            try:
                assert self.cache is not None  # this is here for the type checker
                arm = self.cache.pop(unique_id)
            except KeyError:
                raise DelayedRewardException(
                    f"The unique_id {unique_id} is not in the cache. "
                    "Please use a valid unique identifier."
                )
            try:
                arm_to_update = self.arms[arm]
            except KeyError:
                warn(
                    DelayedRewardWarning(
                        f"The arm {arm} is not in the bandit. Skipping."
                    ),
                    stacklevel=2,
                )
                return

        else:
            arm_to_update = cast(Arm, self.last_arm_pulled)

        arm_to_update.update(X_fit, y_fit)

    def _update_batch(
        self, X: NDArray[np.float_], y: NDArray[np.float_], unique_ids: Collection[Any]
    ):
        # fetch the arms names from the cache
        assert self.cache is not None  # for the type checker

        if len(unique_ids) != X.shape[0]:
            raise ValueError(
                "`X`, `y`, and `unique_ids` must have the same number of rows."
            )

        # get indexes of unique_ids that are in the cache
        present_ids = np.array(
            [idx for idx, v in enumerate(unique_ids) if v in self.cache], dtype=int
        )

        # get the arm names from the cache, get None for missing ids
        arm_names = np.array(
            [self.cache.pop(unique_id, None) for unique_id in unique_ids],
            dtype=str,
        )

        # raise if all ids are missing
        if len(present_ids) == 0:
            raise DelayedRewardException(
                f"None of the unique_ids {unique_ids} are in the cache. "
                "Please use valid unique identifiers."
            )

        # warn if some ids are missing
        if (missing_ids := len(arm_names) - len(present_ids)) > 0:
            warn(
                f"{missing_ids} unique_ids not in the cache. Skipping those updates.",
                DelayedRewardWarning,
                stacklevel=2,
            )

        # update the arms, dropping the missing ids
        for X_part, y_part, arms in groupby_array(
            X[present_ids],
            y[present_ids],
            arm_names[present_ids],
            by=arm_names[present_ids],
        ):
            arm_name = arms[0]
            try:
                arm_to_update = self.arms[arm_name]
            except KeyError:
                warn(
                    DelayedRewardWarning(
                        f"The arm {arm_name} is not in the bandit. Skipping."
                    ),
                    stacklevel=2,
                )
                continue

            arm_to_update.update(X_part, y_part)

    @overload
    def sample(self, X: Union[ArrayLike, csc_array], /, *, size: int = 1) -> ArrayLike:
        ...

    @overload
    def sample(self, /, *, size: int = 1) -> ArrayLike:
        ...

    def sample(
        self,
        X: Union[ArrayLike, csc_array, None] = None,
        /,
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
        X : Optional[ArrayLike]
            Context for the bandit. Only provided when the @contextual
            decorator is used.
        size : int, default=1
            Number of samples to draw.

        """
        assert isinstance(self.rng, np.random.Generator)  # for the type checker
        X_sample, _ = _validate_arrays(X, None, self._contextual, check_y=False)
        # choose an arm, draw a sample, and repeat `size` times
        # TODO: this is not the most efficient way to do this
        # but I can't imagine a situation where this would be a bottleneck.
        return np.array(
            [
                cast(Arm, self.policy(self.arms, X_sample, self.rng)).sample(X_sample)
                for _ in range(size)
            ]
        )

    @overload
    def decay(
        self, /, *, decay_rate: Optional[float] = None, decay_last_arm: bool = True
    ) -> None:
        ...

    @overload
    def decay(
        self,
        X: Union[ArrayLike, csc_array],
        /,
        *,
        decay_rate: Optional[float] = None,
        decay_last_arm: bool = True,
    ) -> None:
        ...

    def decay(
        self,
        X: Union[ArrayLike, csc_array, None] = None,
        /,
        *,
        decay_rate: Optional[float] = None,
        decay_last_arm: bool = True,
    ) -> None:
        """Decay the all arms in the bandit.

        Optionally, decay the last arm pulled.

        Parameters
        ----------
        X : Optional[ArrayLike]
            Context for the bandit. Only provided when the @contextual
            decorator is used.
        decay_last_arm : bool, default=True
            Whether to decay the last arm pulled.
        """
        X_decay, _ = _validate_arrays(
            X, None, contextual=self._contextual, check_y=False
        )

        for arm in self.arms.values():
            if decay_last_arm or arm is not self.last_arm_pulled:
                arm.decay(X_decay, decay_rate=decay_rate)

    def __post_init__(self) -> None:
        """Moves all class attributes that are instances of `Arm` to instance
        attributes.

        This ensures that the bandit can be pickled."""

        self.rng = np.random.default_rng(self.rng)
        self._contextual = False

        # initialize the arms with copies of the learner and
        # point the learner rng to the bandit rng
        for arm in self.arms.values():
            self._set_learner(arm)

        if Arm not in self.__annotations__.values():
            raise ValueError(
                "A bandit must have at least one arm. "
                "Add an arm to the class definition."
            )

    def _set_learner(self, arm: Arm) -> None:
        """Set the learner for an arm, if it is not already set."""
        if arm.learner is None:
            arm.learner = clone(self.learner)  # type: ignore
            arm.learner.set_params(random_state=self.rng)

    def __setstate__(self, state: Dict[str, Any]) -> None:
        """Set the state of the bandit.

        This method is called by pickle when loading a bandit from disk.
        This implementation ensures that arms that are no longer defined
        are deleted and arms that are newly defined are initialized, while
        preserving the state of the arms that are still defined.
        """
        self.__dict__.update(state)

        # 0.4.6+ compatibility: add name to arms - this attribute was added in 0.4.6
        # and is not present in older versions, breaking old pickles
        for arm_name, arm in self.arms.items():
            if not hasattr(arm, "name"):
                arm.name = arm_name

        currently_defined_arms = {
            name: attr
            for name, attr in self.__class__.__annotations__.items()
            if attr is Arm
        }

        # delete arms that are no longer defined
        to_del = [
            arm_name for arm_name in self.arms if arm_name not in currently_defined_arms
        ]

        for arm_name in to_del:
            del self.arms[arm_name]
            del self.__dict__[arm_name]

        # initialize arms that are newly defined
        for arm_name, _ in currently_defined_arms.items():
            if arm_name not in self.arms:
                new_arm = self.__class__.__dataclass_fields__[
                    arm_name
                ].default_factory()
                self._set_learner(new_arm)
                setattr(self, arm_name, new_arm)  # type: ignore
                self.arms[arm_name] = self.__dict__[arm_name]

    @cached_property
    def arms(self: BanditProtocol) -> Dict[str, Arm]:
        return {
            name: attr for name, attr in self.__dict__.items() if isinstance(attr, Arm)
        }


@overload
def _validate_arrays(
    X: Union[ArrayLike, csc_array],
    y: Optional[ArrayLike],
    /,
    contextual: bool,
    check_y: Literal[True] = True,
) -> Tuple[NDArray[np.float_], NDArray[np.float_]]:
    ...


@overload
def _validate_arrays(
    X: Union[ArrayLike, csc_array, None],
    y: Literal[None],
    /,
    contextual: bool,
    check_y: Literal[False] = False,
) -> Tuple[NDArray[np.float_], None]:
    ...


def _validate_unique_id(
    unique_id: Any,
) -> TypeGuard[Hashable]:
    return isinstance(unique_id, Hashable)


def _validate_unique_ids(
    unique_ids: Collection[Any],
) -> TypeGuard[Collection[Hashable]]:
    for unique_id in unique_ids:
        if not _validate_unique_id(unique_id):
            return False
    return True


def _validate_arrays(
    X: Union[ArrayLike, csc_array, None],
    y: Optional[ArrayLike],
    /,
    contextual: bool,
    check_y: bool = True,
) -> Tuple[Union[NDArray[np.float_], csc_array], Optional[NDArray[np.float_]]]:
    """Validate the `X` and `y` arrays.

    Parameters
    ----------
    X : ArrayLike
        Context for the bandit. Only provided when the @contextual
        decorator is used. Otherwise, this position is used for `y`.
    y : Optional[ArrayLike]
        Reward for the bandit. Only provided when the @contextual
        decorator is used. Otherwise, this position should be None.
    contextual : bool
        Whether the bandit is contextual.
    check_y : bool, default=True
        Whether to check the `y` array.

    Returns
    -------
    Tuple[NDArray, NDArray]
        Validated `X` and `y` arrays.
    """
    if check_y:
        should_not_be_none_if_contextual = y
    else:
        should_not_be_none_if_contextual = X

    if contextual and should_not_be_none_if_contextual is None:
        raise ValueError("Context must be provided for a contextual bandit.")
    if not contextual and should_not_be_none_if_contextual is not None:
        raise ValueError("Context must be None for a non-contextual bandit.")

    if contextual:
        X = np.atleast_2d(cast(ArrayLike, X)) if not issparse(X) else X
        y = np.atleast_1d(cast(ArrayLike, y)) if check_y else None
    else:
        y = np.atleast_1d(cast(ArrayLike, X)) if check_y else None
        X = (  # type: ignore
            np.ones_like(y, dtype=float)[:, np.newaxis]
            if check_y
            else np.array([[1]], dtype=float)
        )

    if check_y:
        if X.shape[0] != y.shape[0]:  # type: ignore
            raise ValueError(
                "The number of rows in `X` must match the number of rows in `y`."
            )
    assert isinstance(X, (np.ndarray, csc_array))  # for the type checker
    return X, y


def contextual(
    cls: Type[_B],
) -> Type[_B]:
    """Decorator for making a bandit contextual.

    This decorator adds methods to a Bandit subclass that allow it to be used
    in a contextual setting. The `pull` and `sample` methods will take `X`
    arguments, and the `update` method will take `X` and `y`.

    Parameters
    ----------
    cls : Bandit
        Bandit class to make contextual.

    Returns
    -------
    Bandit
        Contextual bandit class.

    Examples
    --------

    A contextual bandit can be created by decorating a Bandit subclass with
    the `contextual` decorator.

    >>> from bayesianbandits import Arm, GammaRegressor, epsilon_greedy
    >>> clf = GammaRegressor(alpha=1, beta=1)
    >>> policy = epsilon_greedy(0.1)
    >>> def reward_func(x):
    ...     return x
    >>> @contextual
    ... class MyBandit(Bandit, learner=clf, policy=policy):
    ...     arm1 = Arm("Action 1!", reward_func)
    ...     arm2 = Arm("Action 2!", reward_func)
    >>> bandit = MyBandit(rng=0)

    The `pull`, `sample`, and `update` methods now take an `X` argument.

    >>> bandit.pull(1)
    'Action 1!'
    >>> bandit.update(1, 1)

    """

    check_is_bandit(cls)

    orig_post_init = cls.__post_init__

    def _contextual_post_init(self: _B) -> None:
        orig_post_init(self)
        self._contextual = True  # type: ignore

    setattr(cls, "__post_init__", _contextual_post_init)

    return cls


def restless(
    cls: Type[_B],
) -> Type[_B]:
    """Decorator for restless bandits.

    This decorator ensures that the `decay` method of each unselected
    arm is called at each update. The specific behavior of the `decay`
    method is left to the learner.

    """

    check_is_bandit(cls)

    orig_update = cls.update  # type: ignore

    def _restless_update(
        self: Bandit,
        X: ArrayLike,
        y: Optional[ArrayLike] = None,
        **kwargs: ArrayLike,
    ) -> None:
        orig_update(self, X, y, **kwargs)  # type: ignore

        if self._contextual:  # type: ignore
            self.decay(X, decay_last_arm=False)
        else:
            self.decay(decay_last_arm=False)

    setattr(cls, "update", _restless_update)

    return cls


def check_is_bandit(cls: type):
    if not issubclass(cls, Bandit):
        raise ValueError("This decorator can only be used on a Bandit subclass.")


class DelayedRewardException(Exception):
    """Exception raised when the user does not handle delayed reward bandits
    correctly.

    For example, if the user tries to reuse a `unique_id`."""

    pass


class DelayedRewardWarning(UserWarning):
    """Warning raised when the user does not handle delayed reward bandits
    correctly.

    For example, if the user tries to reuse a `unique_id`."""

    pass
