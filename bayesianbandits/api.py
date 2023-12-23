from typing import Any, Callable, Generic, List, Optional, TypeVar, Union

import numpy as np
from numpy.typing import NDArray
from scipy.sparse import csc_array
from typing_extensions import Self

from ._arm import Arm
from ._basebandit import _validate_arrays
from ._policy_decorators import (
    _compute_arm_mean,
    _compute_arm_upper_bound,
    _draw_one_sample,
)
from ._typing import ActionToken

AT = TypeVar("AT", bound=Arm)

Policy = Callable[
    [list[AT], Union[NDArray[np.float_], csc_array], np.random.Generator], List[AT]
]


class ContextualMultiArmedBandit(Generic[AT]):
    def __init__(
        self,
        arms: List[AT],
        policy: Policy,
        random_seed: Union[int, None, np.random.Generator] = None,
    ):
        self._arms: List[AT] = arms
        if not len(arms) > 0:
            raise ValueError("At least one arm is required.")
        unique_tokens = set(arm.action_token for arm in arms)
        if not len(unique_tokens) == len(arms):
            raise ValueError("All arms must have unique action tokens.")
        if not all(arm.learner is not None for arm in arms):
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
        current_tokens = set(arm.action_token for arm in self.arms)
        if arm.action_token in current_tokens:
            raise ValueError("All arms must have unique action tokens.")
        self.arms.append(arm)

    def remove_arm(self, token: Any) -> None:
        for i, arm in enumerate(self._arms):
            if arm.action_token == token:
                self._arms.pop(i)
                break
        else:
            raise KeyError(f"Arm with token {token} not found.")

    def arm(self, token: Any) -> Self:
        for arm in self.arms:
            if arm.action_token == token:
                self.arm_to_update = arm
                break
        else:
            raise KeyError(f"Arm with token {token} not found.")
        return self

    def pull(self, X: Union[NDArray[np.float_], csc_array]) -> List[ActionToken]:
        X_pull, _ = _validate_arrays(X, None, contextual=True, check_y=False)
        arms = self.policy(self.arms, X_pull, self.rng)
        self.arm_to_update = arms[-1]
        return [arm.pull() for arm in arms]

    def update(
        self, X: Union[NDArray[np.float_], csc_array], y: NDArray[np.float_]
    ) -> None:
        X_updated, y_update = _validate_arrays(X, y, contextual=True, check_y=True)
        self.arm_to_update.update(X_updated, y_update)

    def decay(
        self,
        X: Union[NDArray[np.float_], csc_array],
        decay_rate: Optional[float] = None,
    ) -> None:
        X_decay, _ = _validate_arrays(X, None, contextual=True, check_y=False)
        for arm in self.arms:
            arm.decay(X_decay, decay_rate=decay_rate)


class MultiArmedBandit(ContextualMultiArmedBandit[AT]):
    def pull(self):
        X_pull, _ = _validate_arrays(None, None, contextual=False, check_y=False)
        return super().pull(X_pull)

    def update(self, y: NDArray[np.float_]) -> None:
        X_update, y_update = _validate_arrays(y, None, contextual=False, check_y=True)
        super().update(X_update, y_update)

    def decay(self, decay_rate: Optional[float] = None) -> None:
        X_decay, _ = _validate_arrays(None, None, contextual=False, check_y=False)
        super().decay(X_decay, decay_rate=decay_rate)


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

        best_arm_indexes = np.atleast_1d(np.argmax(means, axis=0))

        choice_idx_to_explore = rng.random(size=len(best_arm_indexes)) < epsilon
        final_choices: List[AT] = []
        for explore, choice in zip(choice_idx_to_explore, best_arm_indexes):
            if explore:
                final_choices.append(rng.choice(arms))  # type: ignore
            else:
                final_choices.append(arms[choice])

        return final_choices

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

        best_arm_indexes = np.atleast_1d(np.argmax(upper_bounds, axis=0))

        return [arms[choice] for choice in best_arm_indexes]

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

        best_arm_indexes = np.atleast_1d(np.argmax(samples, axis=0))

        return [arms[choice] for choice in best_arm_indexes]

    return _choose_arm
