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
    LipschitzContextualAgent

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
    Sequence,
    Sized,
    Union,
    cast,
    overload,
)

import numpy as np
from numpy.typing import NDArray
from typing_extensions import Self

from ._arm import Arm, ContextType, Learner, TokenType
from ._arm_featurizer import ArmFeaturizer
from .policies import (  # noqa: F401
    EpsilonGreedy,
    ThompsonSampling,
    UpperConfidenceBound,
)


class PolicyProtocol(Protocol[ContextType, TokenType]):
    @overload
    def __call__(
        self,
        arms: List[Arm[ContextType, TokenType]],
        X: ContextType,
        rng: np.random.Generator,
        top_k: None = None,
    ) -> List[Arm[ContextType, TokenType]]: ...

    @overload
    def __call__(
        self,
        arms: List[Arm[ContextType, TokenType]],
        X: ContextType,
        rng: np.random.Generator,
        top_k: int,
    ) -> List[List[Arm[ContextType, TokenType]]]: ...

    def __call__(
        self,
        arms: List[Arm[ContextType, TokenType]],
        X: ContextType,
        rng: np.random.Generator,
        top_k: Optional[int] = None,
    ) -> Union[
        List[Arm[ContextType, TokenType]], List[List[Arm[ContextType, TokenType]]]
    ]: ...

    def update(
        self,
        arm: Arm[ContextType, TokenType],
        X: ContextType,
        y: NDArray[np.float64],
        all_arms: List[Arm[ContextType, TokenType]],
        rng: np.random.Generator,
        sample_weight: Optional[NDArray[np.float64]] = None,
    ) -> None: ...

    @property
    def samples_needed(self) -> int: ...

    @overload
    def select(
        self,
        samples: NDArray[np.float64],  # Shape: (n_arms, n_contexts, samples_needed)
        arms: List[Arm[ContextType, TokenType]],
        rng: np.random.Generator,
        top_k: None = None,
    ) -> List[Arm[ContextType, TokenType]]: ...

    @overload
    def select(
        self,
        samples: NDArray[np.float64],  # Shape: (n_arms, n_contexts, samples_needed)
        arms: List[Arm[ContextType, TokenType]],
        rng: np.random.Generator,
        top_k: int,
    ) -> List[List[Arm[ContextType, TokenType]]]: ...

    def select(
        self,
        samples: NDArray[np.float64],  # Shape: (n_arms, n_contexts, samples_needed)
        arms: List[Arm[ContextType, TokenType]],
        rng: np.random.Generator,
        top_k: Optional[int] = None,
    ) -> Union[
        List[Arm[ContextType, TokenType]], List[List[Arm[ContextType, TokenType]]]
    ]: ...


class ContextualAgent(Generic[ContextType, TokenType]):
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
        arms: Sequence[Arm[ContextType, TokenType]],
        policy: PolicyProtocol[ContextType, TokenType],
        random_seed: Union[int, None, np.random.Generator] = None,
    ):
        self.policy: PolicyProtocol[ContextType, TokenType] = policy

        self.rng: np.random.Generator = np.random.default_rng(random_seed)
        self._arms: List[Arm[ContextType, TokenType]] = []
        for arm in arms:
            self.add_arm(arm)

        if len(self.arms) == 0:
            raise ValueError("At least one arm is required.")

        self.arm_to_update = arms[0]

    @property
    def arms(self) -> List[Arm[ContextType, TokenType]]:
        return self._arms

    def add_arm(self, arm: Arm[Any, TokenType]) -> None:
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

        assert arm.learner is not None, "Arm must have a learner."
        arm.learner.random_state = self.rng
        self.arms.append(arm)

    def remove_arm(self, token: TokenType) -> None:
        """Remove an arm from the bandit.

        Parameters
        ----------
        token : TokenType
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

    def arm(self, token: TokenType) -> Arm[Any, TokenType]:
        """Get an arm by its action token.

        Parameters
        ----------
        token : TokenType
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

    def select_for_update(self, token: TokenType) -> Self:
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

    @overload
    def pull(self, X: ContextType) -> List[TokenType]: ...

    @overload
    def pull(self, X: ContextType, *, top_k: int) -> List[List[TokenType]]: ...

    def pull(
        self, X: ContextType, *, top_k: int | None = None
    ) -> List[TokenType] | List[List[TokenType]]:
        """Choose arm(s) and pull based on the context(s).

        Parameters
        ----------
        X : ContextType
            Context matrix to use for choosing arms.
        top_k : int, optional
            Number of arms to select per context. If None (default),
            selects single best arm per context. If specified, selects
            top k arms per context.

        Returns
        -------
        List[TokenType] or List[List[TokenType]]
            If top_k is None: List of action tokens (one per context)
            If top_k is int: List of lists of action tokens

        Notes
        -----
        When top_k is None, arm_to_update is set to the last selected arm.
        When top_k is specified, arm_to_update is NOT updated - you must
        explicitly call select_for_update() before update() to specify
        which arm's feedback you're providing.
        """

        if top_k is None:
            arms = self.policy(self.arms, X, self.rng)

            self.arm_to_update = arms[-1]
            return [arm.pull() for arm in arms]
        else:
            arms_lists = self.policy(self.arms, X, self.rng, top_k=top_k)

            # Don't update arm_to_update - ambiguous which to choose
            return [[arm.pull() for arm in arms_list] for arms_list in arms_lists]

    def update(
        self,
        X: ContextType,
        y: NDArray[np.float64],
        sample_weight: Optional[NDArray[np.float64]] = None,
    ) -> None:
        """Update the `arm_to_update` with the context(s) and the reward(s).

        Parameters
        ----------
        X : ContextType
            Context matrix to use for updating the arm.
        y : NDArray[np.float64]
            Reward(s) to use for updating the arm.
        sample_weight : Optional[NDArray[np.float64]], default=None
            Sample weights to use for updating the arm. If None, all samples
            are weighted equally.
        """
        self.policy.update(
            self.arm_to_update, X, y, self.arms, self.rng, sample_weight=sample_weight
        )

    def decay(
        self,
        X: ContextType,
        decay_rate: Optional[float] = None,
    ) -> None:
        """Decay all arms of the bandit len(X) times.

        Parameters
        ----------
        X : ContextType
            Context matrix to use for decaying the arm.
        decay_rate : Optional[float], default=None
            Decay rate to use for decaying the arm. If None, the decay rate
            of the arm's learner is used.
        """
        for arm in self.arms:
            arm.decay(X, decay_rate=decay_rate)


class Agent(Generic[TokenType]):
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
        arms: Sequence[Arm[Any, TokenType]],
        policy: PolicyProtocol[Any, TokenType],
        random_seed: Union[int, None, np.random.Generator] = None,
    ) -> None:
        # Type constraint: ContextType must be compatible with NDArray[np.float64]
        # We use Any here because we can't express the constraint that ContextType
        # must be a supertype of NDArray[np.float64] in Python's type system
        self._inner: ContextualAgent[NDArray[np.float64], TokenType] = ContextualAgent(
            cast(Sequence[Arm[NDArray[np.float64], TokenType]], arms),
            cast(PolicyProtocol[NDArray[np.float64], TokenType], policy),
            random_seed=random_seed,
        )

    @property
    def policy(self) -> PolicyProtocol[NDArray[np.float64], TokenType]:
        return self._inner.policy

    @policy.setter
    def policy(
        self,
        policy: PolicyProtocol[NDArray[np.float64], TokenType],
    ) -> None:
        self._inner.policy = policy

    @property
    def rng(self) -> np.random.Generator:
        return self._inner.rng

    @property
    def arm_to_update(self) -> Arm[NDArray[np.float64], TokenType]:
        return self._inner.arm_to_update

    @property
    def arms(self) -> List[Arm[NDArray[np.float64], TokenType]]:
        return self._inner.arms

    def add_arm(self, arm: Arm[NDArray[np.float64], TokenType]) -> None:
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

    def select_for_update(self, token: TokenType) -> Self:
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

    def arm(self, token: TokenType) -> Arm[NDArray[np.float64], TokenType]:
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

    @overload
    def pull(self) -> List[TokenType]: ...

    @overload
    def pull(self, *, top_k: int) -> List[List[TokenType]]: ...

    def pull(
        self, *, top_k: int | None = None
    ) -> List[TokenType] | List[List[TokenType]]:
        """Choose arm(s) and pull.

        Parameters
        ----------
        top_k : int, optional
            Number of arms to select. If None (default), selects single
            best arm. If specified, selects top k arms.

        Returns
        -------
        List[TokenType] or List[List[TokenType]]
            If top_k is None: List containing single action token [token]
            If top_k is int: List containing a list of k action tokens [[token1, token2, ...]]

        Notes
        -----
        When top_k is None, arm_to_update is set to the selected arm.
        When top_k is specified, arm_to_update is NOT updated - you must
        explicitly call select_for_update() before update() to specify
        which arm's feedback you're providing.
        """
        X_dummy = np.array([[1]], dtype=np.float64)
        if top_k is None:
            return self._inner.pull(X_dummy)
        return self._inner.pull(X_dummy, top_k=top_k)

    def update(
        self,
        y: NDArray[np.float64],
        sample_weight: Optional[NDArray[np.float64]] = None,
    ) -> None:
        """Update the `arm_to_update` with an observed reward.

        Parameters
        ----------
        y : NDArray[np.float64]
            Reward(s) to use for updating the arm.
        sample_weight : Optional[NDArray[np.float64]], default=None
            Sample weights to use for updating the arm. If None, all samples
            are weighted equally.
        """
        X_update: NDArray[np.float64] = np.ones_like(y, dtype=np.float64)[:, np.newaxis]
        self._inner.update(X_update, y, sample_weight=sample_weight)

    def decay(self, decay_rate: Optional[float] = None) -> None:
        """Decay all arms of the bandit.

        Parameters
        ----------
        decay_rate : Optional[float], default=None
            Decay rate to use for decaying the arm. If None, the decay rate
            of the arm's learner is used.
        """
        self._inner.decay(np.array([[1]], dtype=np.float64), decay_rate=decay_rate)


class LipschitzContextualAgent(Generic[TokenType]):
    """
    Agent for Lipschitz/continuous contextual bandits with shared learner.

    This agent is designed for scenarios where a single learner instance is shared
    across all arms, enabling efficient batched operations for large action spaces.
    The key insight is setting the same learner object on all arms, so existing
    policies work unchanged while the learner accumulates knowledge across all
    arm-context pairs.

    Parameters
    ----------
    arms : Sequence[Arm]
        List of arms to choose from. Arms can have learner=None initially - the
        shared learner will be set on all arms during initialization.
    policy : PolicyProtocol
        Policy function to choose arms. All existing policies are compatible.
    arm_featurizer : ArmFeaturizer
        Featurizer to transform contexts with arm features in vectorized batches.
    learner : Learner
        Shared learner instance that will be set on all arms.
    random_seed : Union[int, None, np.random.Generator], default=None
        Seed for the random number generator. If None, a random seed is used.

    Attributes
    ----------
    arms : List[Arm]
        List of arms, all sharing the same learner instance.
    policy : PolicyProtocol
        Policy function for arm selection.
    arm_featurizer : ArmFeaturizer
        Vectorized arm feature transformer.
    learner : Learner
        Shared learner instance across all arms.
    arm_to_update : Arm
        Arm to update with the next reward.
    rng : np.random.Generator
        Random number generator.

    Examples
    --------
    Create a Lipschitz contextual agent for product recommendation:

    >>> import numpy as np
    >>> from bayesianbandits import Arm, NormalRegressor, ThompsonSampling
    >>> from bayesianbandits import ArmColumnFeaturizer
    >>>
    >>> # Define action space - product IDs
    >>> product_ids = list(range(100))
    >>>
    >>> # Create arms without learners initially
    >>> arms = [Arm(token, learner=None) for token in product_ids]
    >>>
    >>> # Create agent with shared learner
    >>> agent = LipschitzContextualAgent(
    ...     arms=arms,
    ...     policy=ThompsonSampling(),
    ...     arm_featurizer=ArmColumnFeaturizer(column_name='product_id'),
    ...     learner=NormalRegressor(alpha=1.0, beta=1.0),
    ...     random_seed=0
    ... )
    >>>
    >>> # Use normally - single call handles all arms efficiently
    >>> X = np.array([[25, 50000], [35, 75000]])  # age, income
    >>> selected_products = agent.pull(X)  # Returns [product_id1, product_id2]
    >>>
    >>> # Update with observed rewards
    >>> for token, context, reward in zip(selected_products, X, [1.0, 0.5]):
    ...     agent.select_for_update(token).update(np.atleast_2d(context), np.array([reward]))


    Notes
    -----
    Performance Benefits:
    - **Vectorized Operations**: Single featurizer call for all arms vs N separate calls
    - **Shared Model**: Single learner forward pass vs N separate model calls
    - **Memory Efficiency**: Better cache locality with batched operations
    - **Large Action Spaces**: Significant speedup when N >> 100 arms

    The learner sees enriched feature vectors and is unaware of the bandit structure.
    During pull(), contexts are enriched for ALL arms in a single vectorized call.
    During update(), contexts are enriched only for the selected arm.
    """

    def __repr__(self) -> str:
        learner_type = type(self.learner)
        action_space = set(arm.action_token for arm in self.arms)
        reward_functions = set(arm.reward_function for arm in self.arms)

        return (
            f"LipschitzContextualAgent(policy={self.policy}, "
            f"arm_featurizer={self.arm_featurizer}, random_seed={self.rng},\n"
            f"arms={action_space}, reward_functions={reward_functions},\n"
            f"shared_learner={learner_type})"
        )

    def __init__(
        self,
        arms: Sequence[Arm[Any, TokenType]],
        policy: PolicyProtocol[Any, TokenType],
        arm_featurizer: ArmFeaturizer[TokenType],
        learner: Learner[Any],
        random_seed: Union[int, None, np.random.Generator] = None,
    ):
        self.arms: List[Arm[Any, TokenType]] = list(arms)
        self.policy: PolicyProtocol[Any, TokenType] = policy
        self.arm_featurizer: ArmFeaturizer[TokenType] = arm_featurizer
        self.learner: Learner[Any] = learner
        self.rng: np.random.Generator = np.random.default_rng(random_seed)

        # Set random state on learner
        self.learner.random_state = self.rng

        # Set the shared learner on all arms
        for arm in self.arms:
            arm.learner = self.learner

        if len(self.arms) == 0:
            raise ValueError("At least one arm is required.")

        self.arm_to_update: Arm[Any, TokenType] = self.arms[0]

    def _reshape_samples(
        self, samples: NDArray[np.float64], n_arms: int, n_contexts: int
    ) -> NDArray[np.float64]:
        """
        Unified reshape for both 2D and 3D learner outputs.

        Converts learner output to (n_arms, n_contexts, size, ...) for policy consumption.

        Parameters
        ----------
        samples : NDArray[np.float64]
            Learner output with shape (size, n_contexts*n_arms, ...)
        n_arms : int
            Number of arms
        n_contexts : int
            Number of contexts

        Returns
        -------
        NDArray[np.float64]
            Reshaped array with shape (n_arms, n_contexts, size, ...)
        """
        if samples.ndim == 2:
            # 2D case: (size, n_contexts*n_arms) -> (n_arms, n_contexts, size)
            return samples.T.reshape(n_arms, n_contexts, -1)
        else:
            # 3D+ case: (size, n_contexts*n_arms, ...) -> (n_arms, n_contexts, size, ...)
            samples_moved = np.moveaxis(samples, 0, 1)  # Move size to position 1
            new_shape = (n_arms, n_contexts) + samples_moved.shape[1:]
            return samples_moved.reshape(new_shape)

    def add_arm(self, arm: Arm[Any, TokenType]) -> None:
        """
        Add an arm to the agent and set the shared learner.

        Parameters
        ----------
        arm : Arm[Any, TokenType]
            Arm to add to the agent.

        Raises
        ------
        ValueError
            If the arm's action token is already in the agent.
        """
        current_tokens = set(arm.action_token for arm in self.arms)
        if arm.action_token in current_tokens:
            raise ValueError("All arms must have unique action tokens.")

        arm.learner = self.learner
        self.arms.append(arm)

    def remove_arm(self, token: TokenType) -> None:
        """
        Remove an arm from the agent.

        Parameters
        ----------
        token : TokenType
            Action token of the arm to remove.

        Raises
        ------
        KeyError
            If the arm's action token is not in the agent.
        """
        for i, arm in enumerate(self.arms):
            if arm.action_token == token:
                self.arms.pop(i)
                break
        else:
            raise KeyError(f"Arm with token {token} not found.")

    def arm(self, token: TokenType) -> Arm[Any, TokenType]:
        """
        Get an arm by its action token.

        Parameters
        ----------
        token : TokenType
            Action token of the arm to get.

        Returns
        -------
        Arm[Any, TokenType]
            Arm with the action token.

        Raises
        ------
        KeyError
            If the arm's action token is not in the agent.
        """
        for arm in self.arms:
            if arm.action_token == token:
                return arm
        raise KeyError(f"Arm with token {token} not found.")

    def select_for_update(self, token: TokenType) -> Self:
        """
        Set the `arm_to_update` and return self for chaining.

        Parameters
        ----------
        token : TokenType
            Action token of the arm to update.

        Returns
        -------
        Self
            Self for chaining.

        Raises
        ------
        KeyError
            If the arm's action token is not in the agent.
        """
        for arm in self.arms:
            if arm.action_token == token:
                self.arm_to_update = arm
                break
        else:
            raise KeyError(f"Arm with token {token} not found.")
        return self

    @overload
    def pull(self, X: Sized) -> List[TokenType]: ...

    @overload
    def pull(self, X: Sized, *, top_k: int) -> List[List[TokenType]]: ...

    def pull(
        self, X: Sized, *, top_k: int | None = None
    ) -> List[TokenType] | List[List[TokenType]]:
        """
        Choose arm(s) and pull based on the context(s).

        Parameters
        ----------
        X : Sized
            Context matrix to use for choosing arms.
        top_k : int, optional
            Number of arms to select per context. If None (default),
            selects single best arm per context. If specified, selects
            top k arms per context.

        Returns
        -------
        List[TokenType] or List[List[TokenType]]
            If top_k is None: List of action tokens (one per context)
            If top_k is int: List of lists of action tokens

        Notes
        -----
        When top_k is None, arm_to_update is set to the last selected arm.
        When top_k is specified, arm_to_update is NOT updated - you must
        explicitly call select_for_update() before update() to specify
        which arm's feedback you're providing.

        The method performs vectorized operations:
        1. Single featurizer call for all arms (major efficiency gain)
        2. Single learner sample call for all arm-context pairs
        3. Efficient reshape and reward function application
        4. Policy selection using standard interface
        """
        # 1. Get action tokens
        action_tokens = [arm.action_token for arm in self.arms]

        # 2. Enrich context with arm features (VECTORIZED - 1 call for N arms)
        X_enriched = self.arm_featurizer.transform(
            X, action_tokens=action_tokens
        )
        # Shape: (n_contexts * n_arms, n_features_enriched)

        # 3. Get samples from learner (SINGLE MODEL CALL)
        samples = self.learner.sample(X_enriched, size=self.policy.samples_needed)
        # Shape: (size, n_contexts * n_arms) or (size, n_contexts * n_arms, n_classes)

        # 4. Unified reshape
        samples = self._reshape_samples(samples, len(self.arms), len(X))
        # Shape: (n_arms, n_contexts, size, ...)

        # 5. Apply reward functions (handles multi-output -> single reward)
        processed_samples: List[NDArray[np.float64]] = []
        for i, arm in enumerate(self.arms):
            arm_samples = arm.reward_function(samples[i])
            processed_samples.append(arm_samples)
        samples = np.array(processed_samples)
        # Final shape: (n_arms, n_contexts, size)

        # 6. Let policy select arms
        selected_arms = self.policy.select(samples, self.arms, self.rng, top_k)

        # 7. Handle return format and arm_to_update
        if top_k is None:
            # Standard case: update arm_to_update and return tokens
            selected_arms_list = cast(List[Arm[Any, TokenType]], selected_arms)
            self.arm_to_update = selected_arms_list[-1]
            return [arm.pull() for arm in selected_arms_list]
        else:
            # Top-k case: don't update arm_to_update, return nested lists
            selected_arms_nested = cast(List[List[Arm[Any, TokenType]]], selected_arms)
            return [
                [arm.pull() for arm in arms_list] for arms_list in selected_arms_nested
            ]

    def update(
        self,
        X: Sized,
        y: NDArray[np.float64],
        sample_weight: Optional[NDArray[np.float64]] = None,
    ) -> None:
        """
        Update the `arm_to_update` with the context(s) and the reward(s).

        Parameters
        ----------
        X : Sized
            Context matrix to use for updating the arm.
        y : NDArray[np.float64]
            Reward(s) to use for updating the arm.
        sample_weight : Optional[NDArray[np.float64]], default=None
            Sample weights to use for updating the arm. If None, all samples
            are weighted equally.

        Notes
        -----
        This method enriches contexts with ONLY the selected arm's features,
        then delegates to the policy's update method which will call arm.update()
        using the shared learner.
        """
        # Enrich context with ONLY the selected arm's features
        X_enriched = self.arm_featurizer.transform(
            X, action_tokens=[self.arm_to_update.action_token]
        )

        # Let the policy handle the update
        # The policy will call arm.update(), which uses our shared learner
        self.policy.update(
            self.arm_to_update, X_enriched, y, self.arms, self.rng, sample_weight
        )

    def decay(
        self,
        X: Sized,
        decay_rate: Optional[float] = None,
    ) -> None:
        """
        Decay the shared learner with all arms' features.

        Parameters
        ----------
        X : Sized
            Context matrix to use for decaying.
        decay_rate : Optional[float], default=None
            Decay rate to use. If None, the learner's default decay rate is used.

        Notes
        -----
        This method enriches contexts with ALL arms' features and applies
        decay to the shared learner once. This is more efficient than
        decaying each arm separately.
        """
        # Get all action tokens
        action_tokens = [arm.action_token for arm in self.arms]

        # Enrich context with all arm features
        X_enriched = self.arm_featurizer.transform(
            X, action_tokens=action_tokens
        )

        # Decay the shared learner once
        self.learner.decay(X_enriched, decay_rate=decay_rate)
