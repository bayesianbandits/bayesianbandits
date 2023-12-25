from typing import (
    Any,
    Callable,
    Generic,
    List,
    Optional,
    TypeVar,
    Union,
    cast,
)

import numpy as np
from numpy.typing import NDArray
from scipy.sparse import csc_array  # type: ignore
from typing_extensions import Self

from ._arm import Arm
from ._basebandit import _validate_arrays  # type: ignore
from ._policy_decorators import (
    _compute_arm_mean,  # type: ignore
    _compute_arm_upper_bound,  # type: ignore
    _draw_one_sample,  # type: ignore
)
from ._typing import ActionToken

AT = TypeVar("AT", bound=Arm[Any])

Policy = Callable[
    [list[AT], Union[NDArray[np.float_], csc_array], np.random.Generator], List[AT]
]


class ContextualMultiArmedBandit(Generic[AT]):
    """Agent for a contextual multi-armed bandit.k-

    Parameters
    ----------
    arms : List[Arm]
        List of arms to choose from. All arms must have a learner and a unique
        action token.
    policy : Callable[[List[Arm], NDArray[np.float_], Generator], Arm]
        Function to choose an arm from the list of arms. Takes the list of arms
        and the context as input and returns the chosen arm.
    random_seed : int, default=None
        Seed for the random number generator. If None, a random seed is used.
    """

    def __init__(
        self,
        arms: List[AT],
        policy: Policy[AT],
        random_seed: Union[int, None, np.random.Generator] = None,
    ):
        self._arms: List[AT] = arms
        if not len(arms) > 0:
            raise ValueError("At least one arm is required.")
        unique_tokens = set(arm.action_token for arm in arms)
        if not len(unique_tokens) == len(arms):
            raise ValueError("All arms must have unique action tokens.")
        if not all(arm.learner is not None for arm in arms):  # type: ignore
            raise ValueError("All arms must have a learner.")

        self.policy: Callable[
            [list[AT], Union[NDArray[np.float_], csc_array], np.random.Generator],
            List[AT],
        ] = policy

        self.arm_to_update: AT = arms[0]

        self.rng: np.random.Generator = np.random.default_rng(random_seed)
        for arm in self.arms:
            arm.learner.random_state = random_seed

    @property
    def arms(self) -> List[AT]:
        return self._arms

    def add_arm(self, arm: AT) -> None:
        """Add an arm to the bandit.

        Parameters
        ----------
        arm : Arm
            Arm to add to the bandit.

        Raises
        ------
        ValueError
            If the arm's action token is already in the bandit.
        """
        current_tokens = set(arm.action_token for arm in self.arms)
        if arm.action_token in current_tokens:
            raise ValueError("All arms must have unique action tokens.")
        self.arms.append(arm)

    def remove_arm(self, token: Any) -> None:
        """Remove an arm from the bandit.

        Parameters
        ----------
        token : Any
            Action token of the arm to remove.

        Raises
        ------
        KeyError
            If the arm's action token is not in the bandit.
        """
        for i, arm in enumerate(self._arms):
            if arm.action_token == token:
                self._arms.pop(i)
                break
        else:
            raise KeyError(f"Arm with token {token} not found.")

    def arm(self, token: Any) -> Self:
        """Set the `arm_to_update`.

        Parameters
        ----------
        token : Any
            Action token of the arm to update.

        Returns
        -------
        Self
            Self for chaining.

        Raises
        ------
        KeyError
            If the arm's action token is not in the bandit.
        """

        for arm in self.arms:
            if arm.action_token == token:
                self.arm_to_update = arm
                break
        else:
            raise KeyError(f"Arm with token {token} not found.")
        return self

    def pull(self, X: Union[NDArray[np.float_], csc_array]) -> List[ActionToken]:
        """Choose an arm and pull it based on the context(s).

        Parameters
        ----------
        X : Union[NDArray[np.float_], csc_array]
            Context matrix to use for choosing an arm.

        Returns
        -------
        List[ActionToken]
            List of action tokens for the pulled arms.
        """
        X_pull, _ = _validate_arrays(X, None, contextual=True, check_y=False)
        arms = self.policy(self.arms, X_pull, self.rng)
        self.arm_to_update = arms[-1]
        return [arm.pull() for arm in arms]

    def update(
        self, X: Union[NDArray[np.float_], csc_array], y: NDArray[np.float_]
    ) -> None:
        """Update the `arm_to_update` with the context(s) and the reward(s).

        Parameters
        ----------
        X : Union[NDArray[np.float_], csc_array]
            Context matrix to use for updating the arm.
        y : NDArray[np.float_]
            Reward(s) to use for updating the arm.
        """
        X_updated, y_update = _validate_arrays(X, y, contextual=True, check_y=True)
        self.arm_to_update.update(X_updated, y_update)

    def decay(
        self,
        X: Union[NDArray[np.float_], csc_array],
        decay_rate: Optional[float] = None,
    ) -> None:
        """Decay all arms of the bandit len(X) times.

        Parameters
        ----------
        X : Union[NDArray[np.float_], csc_array]
            Context matrix to use for decaying the arm.
        decay_rate : Optional[float], default=None
            Decay rate to use for decaying the arm. If None, the decay rate
            of the arm's learner is used.
        """
        X_decay, _ = _validate_arrays(X, None, contextual=True, check_y=False)
        for arm in self.arms:
            arm.decay(X_decay, decay_rate=decay_rate)


class MultiArmedBandit(Generic[AT]):
    def __init__(
        self,
        arms: List[AT],
        policy: Policy[AT],
        random_seed: Union[int, None, np.random.Generator] = None,
    ):
        self._inner = ContextualMultiArmedBandit(arms, policy, random_seed=random_seed)

    @property
    def rng(self) -> np.random.Generator:
        return self._inner.rng

    @property
    def arm_to_update(self) -> AT:
        return self._inner.arm_to_update

    @property
    def arms(self) -> List[AT]:
        return self._inner.arms

    def add_arm(self, arm: AT) -> None:
        """Add an arm to the bandit.

        Parameters
        ----------
        arm : Arm
            Arm to add to the bandit.

        Raises
        ------
        ValueError
            If the arm's action token is already in the bandit.
        """
        self._inner.add_arm(arm)

    def remove_arm(self, token: Any) -> None:
        """Remove an arm from the bandit.

        Parameters
        ----------
        token : Any
            Action token of the arm to remove.

        Raises
        ------
        KeyError
            If the arm's action token is not in the bandit.
        """
        self._inner.remove_arm(token)

    def arm(self, token: Any) -> Self:
        """Set the `arm_to_update`.

        Parameters
        ----------
        token : Any
            Action token of the arm to update.

        Returns
        -------
        Self
            Self for chaining.

        Raises
        ------
        KeyError
            If the arm's action token is not in the bandit.
        """
        self._inner.arm(token)
        return self

    def pull(self):
        """Choose an arm and pull it.

        Returns
        -------
        List[ActionToken]
            List containing the action token for the pulled arm.
        """
        X_pull, _ = _validate_arrays(None, None, contextual=False, check_y=False)
        return self._inner.pull(X_pull)

    def update(self, y: NDArray[np.float_]) -> None:
        """Update the `arm_to_update` with an observed reward.

        Parameters
        ----------
        y : NDArray[np.float_]
            Reward(s) to use for updating the arm.
        """
        X_update, y_update = _validate_arrays(y, None, contextual=False, check_y=True)
        self._inner.update(X_update, y_update)

    def decay(self, decay_rate: Optional[float] = None) -> None:
        """Decay all arms of the bandit.

        Parameters
        ----------
        decay_rate : Optional[float], default=None
            Decay rate to use for decaying the arm. If None, the decay rate
            of the arm's learner is used.
        """
        X_decay, _ = _validate_arrays(None, None, contextual=False, check_y=False)
        self._inner.decay(X_decay, decay_rate=decay_rate)


def epsilon_greedy(
    epsilon: float = 0.1,
    *,
    samples: int = 1000,
) -> Policy[AT]:
    """Creates an epsilon-greedy choice algorithm.

    Parameters
    ----------
    epsilon : float, default=0.1
        Probability of choosing a random arm.
    samples : int, default=1000
        Number of samples to use to compute the mean of the posterior.

    Returns
    -------
    Callable[[BanditProtocol, NDArray[np.float_]], Arm]
        Closure that chooses an arm using epsilon-greedy.
    """

    def _choose_arm(
        arms: list[AT],
        X: Union[NDArray[np.float_], csc_array],
        rng: np.random.Generator,
    ) -> List[AT]:
        """Choose an arm using epsilon-greedy."""
        means = np.stack(
            tuple(_compute_arm_mean(arm, X, samples=samples) for arm in arms)
        )

        best_arm_indexes = np.atleast_1d(
            cast(NDArray[np.int_], np.argmax(means, axis=0))
        )

        choice_idx_to_explore = rng.random(size=len(best_arm_indexes)) < epsilon

        return [
            arms[cast(int, choice)] if not explore else cast(AT, rng.choice(arms))  # type: ignore
            for explore, choice in zip(choice_idx_to_explore, best_arm_indexes)
        ]

    return _choose_arm


def upper_confidence_bound(
    alpha: float = 0.68,
    *,
    samples: int = 1000,
) -> Policy[AT]:
    """Creates an upper confidence bound choice algorithm.

    Parameters
    ----------
    c : float, default=1.0
        Constant to control the exploration-exploitation tradeoff.
    samples : int, default=1000
        Number of samples to use to compute the mean of the posterior.

    Returns
    -------
    Callable[[BanditProtocol, NDArray[np.float_]], Arm]
        Closure that chooses an arm using upper confidence bound.
    """

    def _choose_arm(
        arms: list[AT],
        X: Union[NDArray[np.float_], csc_array],
        rng: np.random.Generator,
    ) -> List[AT]:
        """Choose an arm using upper confidence bound."""

        upper_bounds = np.stack(
            tuple(
                _compute_arm_upper_bound(arm, X, alpha=alpha, samples=samples)
                for arm in arms
            )
        )

        best_arm_indexes = np.atleast_1d(
            cast(NDArray[np.int_], np.argmax(upper_bounds, axis=0))
        )

        return [arms[cast(int, choice)] for choice in best_arm_indexes]

    return _choose_arm


def thompson_sampling(*, batch_size: Optional[int] = None) -> Policy[AT]:
    """Creates a Thompson sampling choice algorithm.

    Returns
    -------
    Callable[[BanditProtocol, NDArray[np.float_]], Arm]
        Closure that chooses an arm using Thompson sampling.
    """

    def _choose_arm(
        arms: list[AT],
        X: Union[NDArray[np.float_], csc_array],
        rng: np.random.Generator,
    ) -> List[AT]:
        """Choose an arm using Thompson sampling."""

        samples = np.stack(
            tuple(_draw_one_sample(arm, X, batch_size=batch_size) for arm in arms)
        )

        best_arm_indexes = np.atleast_1d(
            cast(NDArray[np.int_], np.argmax(samples, axis=0))
        )

        return [arms[cast(int, choice)] for choice in best_arm_indexes]

    return _choose_arm
