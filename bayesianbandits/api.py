"""
=======================================================
Agent API (:mod:`bayesianbandits.api`)
=======================================================

.. currentmodule:: bayesianbandits.api

Fully typed API for Bayesian bandits.

This module contains the API for Bayesian bandits. It passes strict type
checking with `pyright` and is recommended for use in production code. On top
of the type checking, this API makes it much easier to add or remove arms, change
the policy function, and serialize/deserialize bandits in live services.

It uses the same estimators and `Arm` class as the original API, but
defines its own `ContextualAgent` and `Agent` classes, as well as a
`Policy` type alias for the policy functions.

This API splits the `Bandit` class into two classes, `ContextualAgent`
and `Agent`. This split enables safer typing, as the contextual bandit
always takes a context matrix as input, while the non-contextual bandit does not.

Additionally, this API deprecates the `delayed_reward` decorator, as it modifies
the function signatures of the `pull` and `update` methods. Instead, this API
enables batch pulls and updates, but leaves it up to the user to keep track of
matching updates with the correct pulls. Library users reported that this was
what they were doing anyway, so this API change should encourage better practices.

.. note::

    Migrating from the original API to this API should be straightforward. Just
    instantiate the `ContextualAgent` or `MultiArmedBandit` class
    with `list(arms.values())` and the policy function of the original `Bandit`
    subclass.

Bandit Classes
==============

.. autosummary::
    :toctree: _autosummary

    ContextualAgent
    Agent

Policy Functions
================

.. autosummary::
    :toctree: _autosummary

    EpsilonGreedy
    ThompsonSampling
    UpperConfidenceBound

"""

from typing import (
    Any,
    Generic,
    List,
    Optional,
    Protocol,
    TypeVar,
    Union,
    cast,
)

import numpy as np
from numpy.typing import NDArray
from scipy.sparse import csc_array  # type: ignore
from typing_extensions import Self

from ._arm import Arm
from ._typing import DecayingLearner

T = TypeVar("T")
L = TypeVar("L", bound=DecayingLearner)


class PolicyProtocol(Protocol[L, T]):
    def __call__(
        self,
        arms: List[Arm[L, T]],
        X: Union[NDArray[np.float64], csc_array],
        rng: np.random.Generator,
    ) -> List[Arm[L, T]]: ...


P = TypeVar("P", bound=PolicyProtocol[Any, Any])


class ContextualAgent(Generic[L, T, P]):
    """Agent for a contextual multi-armed bandit problem.

    Parameters
    ----------
    arms : List[Arm]
        List of arms to choose from. All arms must have a learner and a unique
        action token.
    policy : Callable[[List[Arm], NDArray[np.float64], Generator], Arm]
        Function to choose an arm from the list of arms. Takes the list of arms
        and the context as input and returns the chosen arm.
    random_seed : int, default=None
        Seed for the random number generator. If None, a random seed is used.

    Attributes
    ----------
    arms : List[Arm]
        List of arms to choose from. All arms must have a learner and a unique
        action token.
    policy : Callable[[List[Arm], NDArray[np.float64], Generator], Arm]
        Function to choose an arm from the list of arms. Takes the list of arms
        and the context as input and returns the chosen arm.
    arm_to_update : Arm
        Arm to update with the next reward.
    rng : Generator
        Random number generator used for choosing arms and decaying.


    Examples
    --------

    Minimally, an agent can be instantiated with a list of arms and a policy
    function. The arms should have a learner and a unique action token.

    >>> from bayesianbandits import Arm, NormalInverseGammaRegressor
    >>> from bayesianbandits import ContextualAgent, ThompsonSampling
    >>> arms = [
    ...     Arm(0, learner=NormalInverseGammaRegressor()),
    ...     Arm(1, learner=NormalInverseGammaRegressor()),
    ... ]
    >>> agent = ContextualAgent(arms, ThompsonSampling(), random_seed=0)

    The agent can then be used to choose an arm and pull it. The `pull` method
    takes a context matrix as input and returns the action token of the chosen
    arm.

    >>> import numpy as np
    >>> X = np.array([[1.0, 15.0]])
    >>> agent.pull(X)
    [1]

    The agent can then be updated with the observed reward. The `update` method
    takes a context matrix and a reward vector as input. By default, the last
    pulled arm is updated.

    >>> y = np.array([100.0])
    >>> agent.update(X, y)
    >>> agent.arm_to_update.learner.predict(X)
    array([99.55947137])

    The agent can also be 'fluently' updated by chaining the `arm` method with
    the `update` method. This is useful when the reward is not immediately
    available, and a batch of updates needs to be made later. The user is
    trusted with calling `arm` with the right action token.

    >>> agent.select_for_update(1).update(X, y)
    >>> agent.arm_to_update is arms[1]
    True
    >>> agent.arm_to_update.learner.predict(X)
    array([99.77924945])
    """

    def __repr__(self) -> str:
        learners = set(type(arm.learner) for arm in self.arms)
        action_space = set(arm.action_token for arm in self.arms)
        reward_function = set(arm.reward_function for arm in self.arms)

        return (
            f"ContextualAgent(policy={self.policy}, random_seed={self.rng},\n"
            f"arms={action_space}, reward_function={reward_function},\n"
            f"learners={learners})"
        )

    def __init__(
        self,
        arms: List[Arm[L, T]],
        policy: P,
        random_seed: Union[int, None, np.random.Generator] = None,
    ):
        self.policy: P = policy

        self.rng: np.random.Generator = np.random.default_rng(random_seed)
        self._arms: List[Arm[L, T]] = []
        for arm in arms:
            self.add_arm(arm)

        if len(self.arms) == 0:
            raise ValueError("At least one arm is required.")

        self.arm_to_update = arms[0]

    @property
    def arms(self) -> List[Arm[L, T]]:
        return self._arms

    def add_arm(self, arm: Arm[L, T]) -> None:
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

        arm.learner.random_state = self.rng
        self.arms.append(arm)

    def remove_arm(self, token: T) -> None:
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

    def arm(self, token: T) -> Arm[L, T]:
        """Get an arm by its action token.

        Parameters
        ----------
        token : Any
            Action token of the arm to get.

        Returns
        -------
        Arm
            Arm with the action token.

        Raises
        ------
        KeyError
            If the arm's action token is not in the bandit.
        """
        for arm in self.arms:
            if arm.action_token == token:
                return arm
        raise KeyError(f"Arm with token {token} not found.")

    def select_for_update(self, token: T) -> Self:
        """Set the `arm_to_update` and return self for chaining.

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

    def pull(self, X: Union[NDArray[np.float64], csc_array]) -> List[T]:
        """Choose an arm and pull it based on the context(s).

        Parameters
        ----------
        X : Union[NDArray[np.float64], csc_array]
            Context matrix to use for choosing an arm.

        Returns
        -------
        List[ActionToken]
            List of action tokens for the pulled arms.
        """
        arms = self.policy(self.arms, X, self.rng)
        self.arm_to_update = arms[-1]
        return [arm.pull() for arm in arms]

    def update(
        self, X: Union[NDArray[np.float64], csc_array], y: NDArray[np.float64]
    ) -> None:
        """Update the `arm_to_update` with the context(s) and the reward(s).

        Parameters
        ----------
        X : Union[NDArray[np.float64], csc_array]
            Context matrix to use for updating the arm.
        y : NDArray[np.float64]
            Reward(s) to use for updating the arm.
        """
        assert X.shape is not None, "X must be a 2D array."
        if X.shape[0] != y.shape[0]:
            raise ValueError(
                "The number of rows in `X` must match the number of rows in `y`."
            )
        self.arm_to_update.update(X, y)

    def decay(
        self,
        X: Union[NDArray[np.float64], csc_array],
        decay_rate: Optional[float] = None,
    ) -> None:
        """Decay all arms of the bandit len(X) times.

        Parameters
        ----------
        X : Union[NDArray[np.float64], csc_array]
            Context matrix to use for decaying the arm.
        decay_rate : Optional[float], default=None
            Decay rate to use for decaying the arm. If None, the decay rate
            of the arm's learner is used.
        """
        for arm in self.arms:
            arm.decay(X, decay_rate=decay_rate)


class Agent(Generic[L, T, P]):
    """
    Agent for a non-contextual multi-armed bandit problem.

    The non-contextual bandit is a special case of the contextual bandit where
    the context matrix is a single column of ones. This class is a wrapper
    around the `ContextualAgent` class that automatically synthesizes the
    context matrix for the `pull`, `update`, and `decay` methods.

    Parameters
    ----------
    arms : List[Arm]
        List of arms to choose from. All arms must have a learner and a unique
        action token.
    policy : Callable[[List[Arm], NDArray[np.float64], Generator], Arm]
        Function to choose an arm from the list of arms. Takes the list of arms
        and the context as input and returns the chosen arm.
    random_seed : int, default=None
        Seed for the random number generator. If None, a random seed is used.

    Attributes
    ----------
    arms : List[Arm]
        List of arms to choose from. All arms must have a learner and a unique
        action token.
    policy : Callable[[List[Arm], NDArray[np.float64], Generator], Arm]
        Function to choose an arm from the list of arms. Takes the list of arms
        and the context as input and returns the chosen arm.
    arm_to_update : Arm
        Arm to update with the next reward.

    Examples
    --------
    The key difference between the `Agent` and the `ContextualAgent` is that
    the `Agent` does not take a context matrix as input. Instead, the context
    matrix is synthesized automatically.

    >>> from bayesianbandits import Arm, NormalInverseGammaRegressor
    >>> from bayesianbandits import Agent, ThompsonSampling
    >>> arms = [
    ...     Arm(0, learner=NormalInverseGammaRegressor()),
    ...     Arm(1, learner=NormalInverseGammaRegressor()),
    ... ]
    >>> agent = Agent(arms, ThompsonSampling(), random_seed=0)
    >>> agent.pull()
    [1]

    This is equivalent to calling the `pull` method with a context matrix containing
    only a global intercept. The `update` and `decay` methods work the same way.

    >>> import numpy as np
    >>> y = np.array([100.0])
    >>> agent.update(y)
    >>> agent.select_for_update(0).update(y)
    """

    def __repr__(self) -> str:
        learners = set(type(arm.learner) for arm in self.arms)
        action_space = set(arm.action_token for arm in self.arms)
        reward_function = set(arm.reward_function for arm in self.arms)

        return (
            f"Agent(policy={self.policy}, random_seed={self.rng},\n"
            f"arms={action_space}, reward_function={reward_function},\n"
            f"learners={learners})"
        )

    def __init__(
        self,
        arms: List[Arm[L, T]],
        policy: P,
        random_seed: Union[int, None, np.random.Generator] = None,
    ):
        self._inner: ContextualAgent[L, T, P] = ContextualAgent(
            arms, policy, random_seed=random_seed
        )

    @property
    def policy(self) -> P:
        return self._inner.policy

    @policy.setter
    def policy(
        self,
        policy: P,
    ) -> None:
        self._inner.policy = policy

    @property
    def rng(self) -> np.random.Generator:
        return self._inner.rng

    @property
    def arm_to_update(self) -> Arm[L, T]:
        return self._inner.arm_to_update

    @property
    def arms(self) -> List[Arm[L, T]]:
        return self._inner.arms

    def add_arm(self, arm: Arm[L, T]) -> None:
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

    def select_for_update(self, token: T) -> Self:
        """Set the `arm_to_update` and return self for chaining.

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
        self._inner.select_for_update(token)
        return self

    def arm(self, token: T) -> Arm[L, T]:
        """Get an arm by its action token.

        Parameters
        ----------
        token : Any
            Action token of the arm to get.

        Returns
        -------
        Self
            Self for chaining.

        Raises
        ------
        KeyError
            If the arm's action token is not in the bandit.
        """
        return self._inner.arm(token)

    def pull(self):
        """Choose an arm and pull it.

        Returns
        -------
        List[ActionToken]
            List containing the action token for the pulled arm.
        """
        return self._inner.pull(np.array([[1]], dtype=np.float64))

    def update(self, y: NDArray[np.float64]) -> None:
        """Update the `arm_to_update` with an observed reward.

        Parameters
        ----------
        y : NDArray[np.float64]
            Reward(s) to use for updating the arm.
        """
        X_update: NDArray[np.float64] = np.ones_like(y, dtype=np.float64)[:, np.newaxis]
        self._inner.update(X_update, y)

    def decay(self, decay_rate: Optional[float] = None) -> None:
        """Decay all arms of the bandit.

        Parameters
        ----------
        decay_rate : Optional[float], default=None
            Decay rate to use for decaying the arm. If None, the decay rate
            of the arm's learner is used.
        """
        self._inner.decay(np.array([[1]], dtype=np.float64), decay_rate=decay_rate)


class EpsilonGreedy:
    """
    Policy object for epsilon-greedy.

    Epsilon-greedy chooses the best arm with probability 1 - epsilon and a random
    arm with probability epsilon.

    Parameters
    ----------
    epsilon : float, default=0.1
        Probability of exploration.
    samples : int, default=1000
        Number of samples to use for computing the arm means.

    Notes
    -----
    The implementation here is based on the implementation in [1]_.

    References
    ----------
    .. [1] Chapelle, Olivier, and Lihong Li. "An empirical evaluation of
       thompson sampling." Advances in neural information processing systems
       24 (2011): 2249-2257.
    """

    def __repr__(self) -> str:
        return f"EpsilonGreedy(epsilon={self.epsilon}, samples={self.samples})"

    def __init__(self, epsilon: float = 0.1, samples: int = 1000):
        self.epsilon = epsilon
        self.samples = samples

    def arm_summary(
        self,
        arms: List[Arm[L, T]],
        X: Union[NDArray[np.float64], csc_array],
        rng: np.random.Generator,
    ) -> NDArray[np.float64]:
        """Return a summary of the arms."""
        means = np.stack(
            tuple(_compute_arm_mean(arm, X, samples=self.samples) for arm in arms)
        )

        return means

    def postprocess(
        self, arm_summary: NDArray[np.float64], rng: np.random.Generator
    ) -> NDArray[np.float64]:
        # Pick random rows to explore
        choice_idx_to_explore = cast(
            NDArray[np.bool_], rng.random(size=arm_summary.shape[1]) < self.epsilon
        )
        # Within the rows to explore, pick a random column and set to np.inf
        for idx, explore in enumerate(choice_idx_to_explore):
            if explore:
                arm_summary[rng.integers(arm_summary.shape[0]), idx] = np.inf  # type: ignore

        return arm_summary

    def __call__(
        self,
        arms: List[Arm[L, T]],
        X: Union[NDArray[np.float64], csc_array],
        rng: np.random.Generator,
    ) -> List[Arm[L, T]]:
        """Choose an arm using epsilon-greedy."""
        means = self.arm_summary(arms, X, rng)
        means = self.postprocess(means, rng)

        best_arm_indexes = np.atleast_1d(cast(NDArray[np.int_], means.argmax(axis=0)))

        return [arms[cast(int, choice)] for choice in best_arm_indexes]


class ThompsonSampling:
    """
    Policy object for Thompson sampling.

    Thompson sampling chooses the best arm with probability equal to the
    probability that the arm is the best arm. That is, it takes a sample from
    each arm's posterior distribution and chooses the arm with the highest
    sample.

    Notes
    -----
    The implementation here is based on the implementation in [1]_.

    References
    ----------
    .. [1] Chapelle, Olivier, and Lihong Li. "An empirical evaluation of
       thompson sampling." Advances in neural information processing systems
       24 (2011): 2249-2257.
    """

    def __repr__(self) -> str:
        return "ThompsonSampling()"

    def arm_summary(
        self,
        arms: List[Arm[L, T]],
        X: Union[NDArray[np.float64], csc_array],
        rng: np.random.Generator,
    ) -> NDArray[np.float64]:
        """Return a summary of the arms."""
        samples = np.stack(tuple(_draw_one_sample(arm, X) for arm in arms))

        return samples

    def postprocess(
        self, arm_summary: NDArray[np.float64], rng: np.random.Generator
    ) -> NDArray[np.float64]:
        return arm_summary

    def __call__(
        self,
        arms: List[Arm[L, T]],
        X: Union[NDArray[np.float64], csc_array],
        rng: np.random.Generator,
    ) -> List[Arm[L, T]]:
        """Choose an arm using Thompson sampling."""

        samples = self.arm_summary(arms, X, rng)
        samples = self.postprocess(samples, rng)

        best_arm_indexes = np.atleast_1d(cast(NDArray[np.int_], samples.argmax(axis=0)))

        return [arms[cast(int, choice)] for choice in best_arm_indexes]


class UpperConfidenceBound:
    """
    Policy object for upper confidence bound.

    Upper confidence bound takes `samples` samples from each arm's posterior
    distribution and chooses the arm with the highest upper bound, as defined
    by the `alpha` parameter.

    Parameters
    ----------
    alpha : float, default=0.68
        Confidence level (one-sided)
    samples : int, default=1000
        Number of samples to use for computing the arm upper bounds.

    Notes
    -----
    The implementation here is based on the implementation in [1]_.

    References
    ----------
    .. [1] Chapelle, Olivier, and Lihong Li. "An empirical evaluation of
       thompson sampling." Advances in neural information processing systems
       24 (2011): 2249-2257.
    """

    def __repr__(self) -> str:
        return f"UpperConfidenceBound(alpha={self.alpha}, samples={self.samples})"

    def __init__(self, alpha: float = 0.68, samples: int = 1000):
        self.alpha = alpha
        self.samples = samples

    def arm_summary(
        self,
        arms: List[Arm[L, T]],
        X: Union[NDArray[np.float64], csc_array],
        rng: np.random.Generator,
    ) -> NDArray[np.float64]:
        """Return a summary of the arms."""
        upper_bounds = np.stack(
            tuple(
                _compute_arm_upper_bound(arm, X, alpha=self.alpha, samples=self.samples)
                for arm in arms
            )
        )

        return upper_bounds

    def postprocess(
        self, arm_summary: NDArray[np.float64], rng: np.random.Generator
    ) -> NDArray[np.float64]:
        return arm_summary

    def __call__(
        self,
        arms: List[Arm[L, T]],
        X: Union[NDArray[np.float64], csc_array],
        rng: np.random.Generator,
    ) -> List[Arm[L, T]]:
        """Choose an arm using upper confidence bound."""

        upper_bounds = self.arm_summary(arms, X, rng)
        upper_bounds = self.postprocess(upper_bounds, rng)

        best_arm_indexes = np.atleast_1d(
            cast(NDArray[np.int_], upper_bounds.argmax(axis=0))
        )

        return [arms[cast(int, choice)] for choice in best_arm_indexes]


def _draw_one_sample(
    arm: Arm,
    X: Union[NDArray[np.float64], csc_array],
) -> NDArray[np.float64]:
    """Draw one sample from the posterior distribution for the arm."""
    return arm.sample(X, size=1).squeeze(axis=0)


def _compute_arm_upper_bound(
    arm: Arm,
    X: Union[NDArray[np.float64], csc_array],
    *,
    alpha: float = 0.68,
    samples: int = 1000,
) -> np.float64:
    """Compute the upper bound of a one-sided credible interval with size
    `alpha` from the posterior distribution for the arm."""
    posterior_samples = arm.sample(X, size=samples)

    return np.quantile(posterior_samples, q=alpha, axis=0)  # type: ignore


def _compute_arm_mean(
    arm: Arm,
    X: Union[NDArray[np.float64], csc_array],
    *,
    samples: int = 1000,
) -> np.float64:
    """Compute the mean of the posterior distribution for the arm."""
    posterior_samples = arm.sample(X, size=samples)
    return np.mean(posterior_samples, axis=0)
