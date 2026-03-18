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

    ContextualAgent
    Agent
    LipschitzContextualAgent

Policy Functions
================

.. autosummary::

    EpsilonGreedy
    ThompsonSampling
    UpperConfidenceBound

"""

from typing import (
    Any,
    Dict,
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

from ._arm import (
    Arm,
    BatchRewardFunction,
    ContextAwareBatchRewardFunction,
    ContextType,
    Learner,
    TokenType,
    _accepts_context_batch,
    batch_identity,
    is_identity_function,
)
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
    """
    Agent for a contextual multi-armed bandit problem.

    At each round the agent observes a context :math:`x_t`, selects an
    arm :math:`a_t` according to the configured policy, and later receives
    a reward :math:`r_t`. Each arm maintains an independent Bayesian
    learner, so the posterior for arm :math:`a` is updated only when that
    arm is selected:

    .. math::

        a_t = \\pi\\bigl(\\{p(\\theta_a \\mid \\mathcal{D}_a)\\}_{a=1}^{K},
        \\; x_t\\bigr), \\qquad
        \\mathcal{D}_{a_t} \\leftarrow \\mathcal{D}_{a_t}
        \\cup \\{(x_t, r_t)\\}

    where :math:`\\pi` is the policy (e.g. Thompson sampling, UCB, or
    :math:`\\varepsilon`-greedy) and :math:`K` is the number of arms.

    Parameters
    ----------
    arms : Sequence[Arm[ContextType, TokenType]]
        Arms to choose from. Each arm must carry a fitted or unfitted
        learner and a unique action token that identifies it.
    policy : PolicyProtocol[ContextType, TokenType]
        Policy object that implements arm selection given posteriors and
        context. Built-in options: :class:`ThompsonSampling`,
        :class:`UpperConfidenceBound`, :class:`EpsilonGreedy`.
    random_seed : int, np.random.Generator, or None, default=None
        Controls the random number generator shared by the policy and
        all learners. Pass an int for reproducible results across calls.

    See Also
    --------
    Agent : Non-contextual (intercept-only) agent.
    LipschitzContextualAgent : Shared-learner agent with configurable
        design matrix; generalizes this agent.

    Notes
    -----
    **Independent learners.** Each arm owns a separate learner instance
    that is updated only with observations from that arm. This is the
    standard approach when arms have independent reward distributions
    [1]_. For parameter sharing across arms, see
    :class:`LipschitzContextualAgent`.

    **Batch contexts.** Both :meth:`pull` and :meth:`update` accept
    matrices with multiple rows, producing one decision or update per
    row. This enables efficient batch serving but requires the user to
    match rewards to the correct arms when using delayed feedback [2]_.

    **Serialization.** The agent (including all arm learners) is
    pickle-compatible, making it straightforward to persist to a
    database or message queue for use in live services.

    References
    ----------
    .. [1] Chapelle, O. and Li, L. (2011). "An empirical evaluation of
       Thompson sampling." Advances in Neural Information Processing
       Systems 24, 2249-2257.

    .. [2] Li, L., Chu, W., Langford, J., and Schapire, R. E. (2010).
       "A contextual-bandit approach to personalized news article
       recommendation." Proceedings of the 19th International Conference
       on World Wide Web (WWW), 661-670.

    Examples
    --------
    Create an agent with two arms and pull for a single context:

    >>> from bayesianbandits import Arm, NormalInverseGammaRegressor
    >>> from bayesianbandits import ContextualAgent, ThompsonSampling
    >>> arms = [
    ...     Arm(0, learner=NormalInverseGammaRegressor()),
    ...     Arm(1, learner=NormalInverseGammaRegressor()),
    ... ]
    >>> agent = ContextualAgent(arms, ThompsonSampling(), random_seed=0)

    The ``pull`` method takes a context matrix and returns one action
    token per row:

    >>> import numpy as np
    >>> X = np.array([[1.0, 15.0]])
    >>> agent.pull(X)
    [1]

    By default the last pulled arm is queued for update. Call
    ``update`` with the same context and the observed reward:

    >>> y = np.array([100.0])
    >>> agent.update(X, y)
    >>> agent.arm_to_update.learner.predict(X)
    array([99.55947137])

    For delayed rewards, explicitly select which arm to update using
    the fluent :meth:`select_for_update` interface:

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

        self._arms: List[Arm[ContextType, TokenType]] = []
        self.rng = random_seed
        for arm in arms:
            self.add_arm(arm)

        if len(self.arms) == 0:
            raise ValueError("At least one arm is required.")

        self.arm_to_update = arms[0]

    @property
    def rng(self) -> np.random.Generator:
        return self._rng

    @rng.setter
    def rng(self, value: Union[int, None, np.random.Generator]) -> None:
        self._rng = np.random.default_rng(value)
        for arm in self._arms:
            assert arm.learner is not None
            arm.learner.random_state = self._rng

    def __setstate__(self, state: Dict[str, Any]) -> None:
        if "rng" in state and "_rng" not in state:
            state["_rng"] = state.pop("rng")
        self.__dict__.update(state)

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
    Agent for a non-contextual (classic) multi-armed bandit problem.

    Implements the :math:`K`-armed bandit where the agent selects an arm
    :math:`a_t` without observing any side information. Internally this
    is a thin wrapper around :class:`ContextualAgent` with the context
    fixed to a single intercept column :math:`x = [1]`:

    .. math::

        a_t = \\pi\\bigl(\\{p(\\theta_a \\mid \\mathcal{D}_a)\\}_{a=1}^{K}
        \\bigr), \\qquad
        \\mathcal{D}_{a_t} \\leftarrow \\mathcal{D}_{a_t}
        \\cup \\{r_t\\}

    All ``pull``, ``update``, and ``decay`` calls automatically
    synthesize the intercept context, so the caller never needs to
    provide a feature matrix.

    Parameters
    ----------
    arms : Sequence[Arm[Any, TokenType]]
        Arms to choose from. Each arm must carry a fitted or unfitted
        learner and a unique action token that identifies it.
    policy : PolicyProtocol[Any, TokenType]
        Policy object that implements arm selection given posteriors.
        Built-in options: :class:`ThompsonSampling`,
        :class:`UpperConfidenceBound`, :class:`EpsilonGreedy`.
    random_seed : int, np.random.Generator, or None, default=None
        Controls the random number generator shared by the policy and
        all learners. Pass an int for reproducible results across calls.

    See Also
    --------
    ContextualAgent : Agent that conditions decisions on a feature
        matrix.
    LipschitzContextualAgent : Shared-learner agent with configurable
        design matrix; generalizes both Agent and ContextualAgent.

    Notes
    -----
    Because the context is always a single intercept, every learner
    reduces to an intercept-only model. For example,
    :class:`NormalRegressor` becomes a simple Bayesian estimate of the
    mean reward, and :class:`DirichletClassifier` maintains a posterior
    over class probabilities. See [1]_ for an empirical comparison of
    policies in this setting.

    References
    ----------
    .. [1] Chapelle, O. and Li, L. (2011). "An empirical evaluation of
       Thompson sampling." Advances in Neural Information Processing
       Systems 24, 2249-2257.

    Examples
    --------
    Create a non-contextual agent and pull:

    >>> from bayesianbandits import Arm, NormalInverseGammaRegressor
    >>> from bayesianbandits import Agent, ThompsonSampling
    >>> arms = [
    ...     Arm(0, learner=NormalInverseGammaRegressor()),
    ...     Arm(1, learner=NormalInverseGammaRegressor()),
    ... ]
    >>> agent = Agent(arms, ThompsonSampling(), random_seed=0)
    >>> agent.pull()
    [1]

    No context matrix is needed. The ``update`` and ``decay`` methods
    similarly take only a reward vector:

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

    @rng.setter
    def rng(self, value: Union[int, None, np.random.Generator]) -> None:
        self._inner.rng = value

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
        token : TokenType
            Action token of the arm to get.

        Returns
        -------
        Arm[NDArray[np.float64], TokenType]
            Arm with the given action token.

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
    Contextual agent with a shared learner and a configurable design matrix.

    This is the most general agent in the library. The design matrix,
    constructed by the ``arm_featurizer``, encodes your assumptions about
    how arms relate to each other and to the context. By choosing the
    right design matrix structure, you can express a spectrum of models:

    - **One-hot arms only** (no context features): recovers
      :class:`Agent` (non-contextual bandits).
    - **One-hot arms interacted with context** (block-diagonal design
      matrix): recovers :class:`ContextualAgent` (disjoint bandits,
      independent parameters per arm).
    - **Shared features + arm-specific intercepts**: hybrid bandits
      with cross-arm learning and Bayesian shrinkage toward shared
      structure (see the ``hybrid-bandits`` tutorial).
    - **Continuous arm features**: Lipschitz-style bandits (the
      namesake), where nearby arms in feature space share information.

    Formally, the agent uses a single shared learner that conditions
    on both context and arm features:

    .. math::

        \\tilde{x}_{a} = \\phi(x, a), \\qquad
        r \\mid \\tilde{x}_{a} \\sim p(r \\mid \\theta, \\tilde{x}_{a})

    where :math:`\\phi` is the arm featurizer that constructs the design
    matrix from context :math:`x` and arm identity :math:`a`, and
    :math:`\\theta` is the shared parameter vector. At each round,
    posterior samples for **all** arms are drawn in a single vectorized
    call:

    .. math::

        a^* = \\pi\\bigl(
        \\{\\tilde{\\theta} \\sim p(\\theta \\mid \\mathcal{D})\\},
        \\; \\{\\phi(x, a)\\}_{a=1}^{K}\\bigr)

    Parameters
    ----------
    arms : Sequence[Arm[Any, TokenType]]
        Arms to choose from. Arms may have ``learner=None``; the
        shared learner is set on every arm during initialization.
    policy : PolicyProtocol[Any, TokenType]
        Policy object for arm selection. All built-in policies
        (:class:`ThompsonSampling`, :class:`UpperConfidenceBound`,
        :class:`EpsilonGreedy`) are compatible.
    arm_featurizer : ArmFeaturizer[TokenType]
        Featurizer that constructs the design matrix from
        ``(context, action_tokens)`` in a single vectorized call.
        The structure of this matrix encodes assumptions about how
        arms relate to each other and to the context -- see Notes.
    learner : Learner
        Shared learner instance that will be set on all arms. Because
        all arms share this object, updates to any arm improve
        predictions for every arm.
    batch_reward_function : BatchRewardFunction or ContextAwareBatchRewardFunction or None, default=None
        Optional function that processes rewards for all arms at once.

        *Traditional* signature::

            def batch_reward(samples, action_tokens):
                # samples: shape (n_arms, n_contexts, size, ...)
                # action_tokens: list of length n_arms
                return rewards  # shape (n_arms, n_contexts, size)

        *Context-aware* signature::

            def batch_reward(samples, action_tokens, X):
                # X: original context, shape (n_contexts, n_features)
                return rewards  # shape (n_arms, n_contexts, size)

        The ``action_tokens`` list is ordered to match the first
        dimension of ``samples``. If None and all arms use the identity
        reward function, an optimized batch identity is used
        automatically.
    random_seed : int, np.random.Generator, or None, default=None
        Controls the random number generator shared by the policy and
        the learner. Pass an int for reproducible results across calls.

    See Also
    --------
    ContextualAgent : Independent-learner agent; equivalent to this
        agent with a block-diagonal design matrix (no parameter sharing).
    Agent : Non-contextual (intercept-only) agent; equivalent to this
        agent with one-hot arm indicators and no context features.
    ArmColumnFeaturizer : Default featurizer that appends an arm
        identifier column to the context matrix.

    Notes
    -----
    **Vectorized pull.** During :meth:`pull`, contexts are enriched for
    *all* arms in a single featurizer call, followed by a single
    learner ``sample`` call for the entire ``(n_arms * n_contexts)``
    batch. This yields significant speedups when :math:`K \\gg 100`.

    **Selective update.** During :meth:`update`, contexts are enriched
    only for the selected arm, so the update cost is independent of
    :math:`K`.

    **Design matrix as assumption encoding.** The structure of
    :math:`\\phi(x, a)` is the mechanism by which you encode domain
    knowledge about the relationship between arms [3]_.  A block-diagonal
    design matrix (one-hot arms interacted with context) yields fully
    independent parameters per arm -- equivalent to
    :class:`ContextualAgent`.  Adding shared columns (e.g. user
    features that affect all arms) introduces cross-arm learning: the
    shared learner pools data across arms for those features while
    keeping arm-specific effects separate.  This creates a "poor man's
    hierarchical model" where Bayesian priors automatically shrink
    arm-specific effects toward the shared structure.  See the
    ``hybrid-bandits`` tutorial for a worked example.

    **Relationship to other agents.** :class:`Agent` and
    :class:`ContextualAgent` are special cases of this agent with
    particular design matrix structures.  This makes
    ``LipschitzContextualAgent`` the most general agent in the library,
    suitable for any problem where you can describe the arm structure
    through features.

    **Name origin.** The class name comes from the Lipschitz bandit
    literature [1]_ [2]_, where rewards vary smoothly with continuous
    arm features.  The agent is not limited to that setting -- it works
    equally well with discrete arms and arbitrary feature structures.

    References
    ----------
    .. [1] Slivkins, A. (2014). "Contextual bandits with similarity
       information." Journal of Machine Learning Research, 15(1),
       2533-2568.

    .. [2] Krishnamurthy, A., Langford, J., Slivkins, A., and Zhang, C.
       (2020). "Contextual bandits with continuous actions: smoothing,
       zooming, and adapting." Journal of Machine Learning Research,
       21(137), 1-45.

    .. [3] Li, L., Chu, W., Langford, J., and Schapire, R. E. (2010).
       "A contextual-bandit approach to personalized news article
       recommendation." Proceedings of the 19th International Conference
       on World Wide Web (WWW), 661-670.

    Examples
    --------
    Create an agent for product recommendation with 100 products:

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

    Using a batch reward function for revenue optimization:

    >>> # Pre-compute revenue array for all products (vectorized approach)
    >>> n_products = 100
    >>> product_revenues = np.random.uniform(0.5, 3.0, n_products)  # Revenue per product
    >>>
    >>> # Create vectorized batch reward function
    >>> def revenue_batch_reward(samples, action_tokens):
    ...     # Direct numpy indexing - fully vectorized
    ...     multipliers = product_revenues[action_tokens]
    ...     # Broadcast to match samples shape: (n_arms, n_contexts, size)
    ...     return samples * multipliers[:, np.newaxis, np.newaxis]
    >>>
    >>> # Create agent with batch reward function
    >>> agent = LipschitzContextualAgent(
    ...     arms=arms,
    ...     policy=ThompsonSampling(),
    ...     arm_featurizer=ArmColumnFeaturizer(column_name="product_id"),
    ...     learner=NormalRegressor(alpha=1, beta=1),
    ...     batch_reward_function=revenue_batch_reward
    ... )

    Using a context-aware batch reward function:

    >>> # Context-aware: calculate gross profit from prices, costs, and taxes
    >>> # Arms represent different price points
    >>> price_points = np.array([9.99, 14.99, 19.99, 24.99, 29.99])
    >>> arms = [Arm(i, learner=None) for i in range(len(price_points))]
    >>>
    >>> def gross_profit_reward(samples, action_tokens, X):
    ...     # X contains: [customer_value, cost_per_unit, tax_rate]
    ...     costs = X[:, 1]      # shape: (n_contexts,)
    ...     tax_rates = X[:, 2]  # shape: (n_contexts,)
    ...
    ...     # Get prices for selected arms
    ...     prices = price_points[action_tokens]  # shape: (n_arms,)
    ...
    ...     # Vectorized profit calculation for all (arm, context) pairs
    ...     # Revenue after tax: price * (1 - tax_rate)
    ...     # Gross profit: revenue_after_tax - cost
    ...     revenue_after_tax = prices[:, np.newaxis] * (1 - tax_rates[np.newaxis, :])
    ...     gross_profit = revenue_after_tax - costs[np.newaxis, :]
    ...
    ...     # Apply profit multiplier to samples, clamping negative profits to 0
    ...     profit_multiplier = np.maximum(gross_profit, 0)
    ...     return samples * profit_multiplier[:, :, np.newaxis]
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
        batch_reward_function: Optional[
            Union[BatchRewardFunction, ContextAwareBatchRewardFunction]
        ] = None,
        random_seed: Union[int, None, np.random.Generator] = None,
    ):
        self.policy: PolicyProtocol[Any, TokenType] = policy
        self.arm_featurizer: ArmFeaturizer[TokenType] = arm_featurizer
        self.learner: Learner[Any] = learner
        self.batch_reward_function = batch_reward_function

        # Initialize arms list before rng setter (which iterates _arms)
        self.arms: List[Arm[Any, TokenType]] = []
        self.rng = random_seed
        for arm in arms:
            self.add_arm(arm)

        if len(self.arms) == 0:
            raise ValueError("At least one arm is required.")

        # Check identity optimization after all arms are added
        self._check_identity_optimization()

        # Warn if individual functions will be ignored
        # Only warn if user provided a batch function (not auto-optimized)
        if batch_reward_function is not None:
            # Check if any arm has a non-identity reward function
            if any(not is_identity_function(arm.reward_function) for arm in self.arms):
                import warnings

                warnings.warn(
                    "batch_reward_function provided; individual arm reward_functions "
                    "will be ignored during pull()",
                    UserWarning,
                )

        self.arm_to_update: Arm[Any, TokenType] = self.arms[0]

    @property
    def rng(self) -> np.random.Generator:
        return self._rng

    @rng.setter
    def rng(self, value: Union[int, None, np.random.Generator]) -> None:
        self._rng = np.random.default_rng(value)
        self.learner.random_state = self._rng

    def __setstate__(self, state: Dict[str, Any]) -> None:
        if "rng" in state and "_rng" not in state:
            state["_rng"] = state.pop("rng")
        self.__dict__.update(state)

    def _check_identity_optimization(self) -> None:
        """Update batch_reward_function based on current arms."""
        # Only optimize if batch_reward_function is None (can set to batch_identity)
        # or is batch_identity (can set to None)
        if self.batch_reward_function is None:
            # Check if all arms use the identity function
            if self.arms and all(
                is_identity_function(arm.reward_function) for arm in self.arms
            ):
                self.batch_reward_function = batch_identity
        elif self.batch_reward_function is batch_identity:
            # Check if we should disable optimization
            if not (
                self.arms
                and all(is_identity_function(arm.reward_function) for arm in self.arms)
            ):
                self.batch_reward_function = None

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

        # Recheck identity optimization
        self._check_identity_optimization()

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

        # Recheck identity optimization
        self._check_identity_optimization()

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
        X_enriched = self.arm_featurizer.transform(X, action_tokens=action_tokens)
        # Shape: (n_contexts * n_arms, n_features_enriched)

        # 3. Get samples from learner (SINGLE MODEL CALL)
        samples = self.learner.sample(X_enriched, size=self.policy.samples_needed)
        # Shape: (size, n_contexts * n_arms) or (size, n_contexts * n_arms, n_classes)

        # 4. Unified reshape
        samples = self._reshape_samples(samples, len(self.arms), len(X))
        # Shape: (n_arms, n_contexts, size, ...)

        # 5. Apply reward functions
        if self.batch_reward_function is not None:
            # Use efficient batch function
            expected_shape = samples.shape[:3]  # (n_arms, n_contexts, size)

            if _accepts_context_batch(self.batch_reward_function):
                # Context-aware batch function
                context_aware_func = cast(
                    ContextAwareBatchRewardFunction, self.batch_reward_function
                )
                result = context_aware_func(samples, action_tokens, X)
            else:
                # Traditional batch function
                traditional_func = cast(BatchRewardFunction, self.batch_reward_function)
                result = traditional_func(samples, action_tokens)

            # Validate output shape
            if result.shape[:3] != expected_shape:
                raise ValueError(
                    f"batch_reward_function returned wrong shape. "
                    f"Expected shape[:3]={expected_shape}, got shape={result.shape}"
                )
            samples = result
        else:
            # Fall back to individual arm functions
            from ._arm import apply_reward_function

            processed_samples: List[NDArray[np.float64]] = []
            for i, arm in enumerate(self.arms):
                arm_samples = apply_reward_function(arm.reward_function, samples[i], X)
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
        This method enriches contexts with a single arm's features and applies
        decay to the shared learner once. This ensures we decay based on the
        number of contexts, not the number of arms.
        """
        # Use any single arm's token - we just need the enriched shape for one arm
        single_token = [self.arms[0].action_token]

        # Enrich context with single arm features
        X_enriched = self.arm_featurizer.transform(X, action_tokens=single_token)

        # Decay the shared learner once
        self.learner.decay(X_enriched, decay_rate=decay_rate)
