from typing import Any, Callable, Iterable, Optional, Union, cast

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

    def __set_name__(self, owner: Optional[BanditProtocol], name: str) -> None:
        self.name = name

    def __get__(
        self, obj: Optional[BanditProtocol], objtype: Optional[BanditProtocol] = None
    ) -> "Arm":
        if obj is None:
            return self
        return obj.__dict__[self.name]

    def pull(self) -> None:
        """Pull the arm."""
        if self.learner is None:
            raise ValueError("Learner is not set.")
        self.action_function()

    def sample(
        self,
        X: Optional[NDArray[Any]] = None,
        size: int = 1,
    ) -> ArrayLike:
        """Sample from learner and compute the reward."""
        if self.learner is None:
            raise ValueError("Learner is not set.")
        X_new = X or np.array([[1]])

        return self.reward_function(self.learner.sample(X_new, size))

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

    def mean(self, X: NDArray[Any], size: int = 1000) -> ArrayLike:
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
            f" reward_function={self.reward_function}),"
            f" learner={self.learner}"
        )


def epsilon_greedy(
    epsilon: float = 0.1,
) -> Callable[[Iterable[ArmProtocol], Optional[ArrayLike]], ArmProtocol]:
    """Returns a closure that chooses an arm using epsilon-greedy."""

    def _choose_arm(
        arms: Iterable[ArmProtocol],
        X: Optional[ArrayLike] = None,
        rng: Union[np.random.Generator, int, None] = None,
    ) -> ArmProtocol:
        """Choose an arm using epsilon-greedy."""
        if not isinstance(rng, np.random.Generator):
            rng = np.random.default_rng(rng)

        if np.random.rand() < epsilon:
            return rng.choice(list(arms))  # type: ignore
        else:
            return max(arms, key=lambda arm: arm.mean(X))

    return _choose_arm


def bandit(
    learner: Learner,
    choice: Callable[[Iterable[ArmProtocol], Optional[ArrayLike]], ArmProtocol],
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
    rng: Union[np.random.Generator, int, None] = options.get("rng", None)

    if not isinstance(rng, np.random.Generator):
        rng = np.random.default_rng(rng)

    if not contextual:

        @classmethod
        def _bandit_choose_and_pull(cls: BanditProtocol) -> None:
            """Choose an arm and pull it. Set `last_arm_pulled` to the name of the
            arm that was pulled.

            This method is added to the bandit class by the `bandit` decorator.
            """
            arm = cls.choice_algorithm(cls.arms.values(), X=None, rng=cls.rng)
            cls.last_arm_pulled = arm
            arm.pull()

        @classmethod
        def _bandit_update(cls: BanditProtocol, y: ArrayLike) -> None:
            """Update the learner for the last arm pulled.

            This method is added to the bandit class by the `bandit` decorator.
            """
            if cls.last_arm_pulled is None:
                raise ValueError("No arm has been pulled yet.")
            cls.last_arm_pulled.update(np.atleast_1d(y))

        @classmethod
        def _bandit_sample(cls: BanditProtocol, size: int = 1) -> ArrayLike:
            """Sample from the learner using the choice algorithm.

            This method is added to the bandit class by the `bandit` decorator."""
            arm = cls.choice_algorithm()
            return arm.sample(X=None, size=size)

    def wrapper(cls: object) -> BanditProtocol:
        for _, attr in cls.__dict__.items():
            if isinstance(attr, Arm):
                if attr.learner is None:
                    learner_clone = cast(Learner, clone(learner))
                    attr.learner = learner_clone
                    attr.learner.set_params(random_state=rng)

        if not hasattr(cls, "arms"):
            setattr(
                cls,
                "arms",
                {
                    name: attr
                    for name, attr in cls.__dict__.items()
                    if isinstance(attr, ArmProtocol)
                },
            )

        if not hasattr(cls, "rng"):
            setattr(cls, "rng", rng)

        if not hasattr(cls, "choice_algorithm"):
            setattr(cls, "choice_algorithm", staticmethod(choice))

        if not hasattr(cls, "pull"):
            setattr(cls, "pull", _bandit_choose_and_pull)

        if not hasattr(cls, "last_arm_pulled"):
            setattr(cls, "last_arm_pulled", None)

        if not hasattr(cls, "update"):
            setattr(cls, "update", _bandit_update)

        if not hasattr(cls, "sample"):
            setattr(cls, "sample", _bandit_sample)

        cls = cast(BanditProtocol, cls)

        return cls

    return wrapper


if __name__ == "__main__":
    from ._estimators import DirichletClassifier

    clf = DirichletClassifier({1: 1, 2: 1, 3: 1})

    def reward_func(x: ArrayLike) -> ArrayLike:
        return np.take(x, 0, axis=-1)  # type: ignore

    @bandit(learner=clf, choice=epsilon_greedy(epsilon=0.9), rng=0)
    class Experiment:
        arm1 = Arm(lambda: print("arm1"), reward_func)
        arm2 = Arm(lambda: print("arm2"), reward_func)
        arm3 = Arm(lambda: print("arm3"), reward_func)
