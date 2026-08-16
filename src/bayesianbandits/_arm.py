from __future__ import annotations

import importlib.util
import inspect
from functools import wraps
from typing import (
    Any,
    Callable,
    Generic,
    List,
    Optional,
    Protocol,
    Sized,
    TypeVar,
    Union,
    cast,
)

import numpy as np
from numpy.typing import NDArray
from typing_extensions import Concatenate, ParamSpec, Self, TypeGuard

from ._memory import MemoryUsageMixin

HAS_PANDAS = importlib.util.find_spec("pandas") is not None

P = ParamSpec("P")
R = TypeVar("R", covariant=True)
ContextType = TypeVar("ContextType", bound=Sized)

# Traditional reward function type
TraditionalRewardFunction = Callable[[NDArray[np.float64]], NDArray[np.float64]]

# Context-aware reward function type
ContextAwareRewardFunction = Callable[
    [NDArray[np.float64], ContextType], NDArray[np.float64]
]

# Union type for backward compatibility
RewardFunction = Union[
    TraditionalRewardFunction, ContextAwareRewardFunction[ContextType]
]

TokenType = TypeVar("TokenType")
X_contra = TypeVar("X_contra", contravariant=True)  # Contravariant for input types
A = TypeVar("A", bound="Arm[Any, Any]")

# Batch reward function types for LipschitzContextualAgent
BatchRewardFunction = Callable[
    [
        NDArray[np.float64],  # samples: shape (n_arms, n_contexts, size, ...)
        List[Any],  # action_tokens: length n_arms (ordered by arm index)
    ],
    NDArray[np.float64],  # returns: shape (n_arms, n_contexts, size)
]
"""
BatchRewardFunction processes rewards for multiple arms in a single call.

Parameters
----------
samples : NDArray[np.float64]
    Samples from the learner with shape (n_arms, n_contexts, size, ...).
    The first dimension corresponds to arms in the same order as action_tokens.
action_tokens : List[Any]
    List of action tokens, one per arm, in the order corresponding to the
    arms in the agent's arms list. This order matches the first dimension
    of the samples array.

Returns
-------
NDArray[np.float64]
    Reward values with shape (n_arms, n_contexts, size), maintaining the
    same arm ordering as the input.

Notes
-----
The action_tokens list is ordered to match the agent's arms list order,
NOT necessarily in numerical or alphabetical order. For example, if
arms were added with tokens [5, 2, 8], then action_tokens = [5, 2, 8]
and samples[0] corresponds to token 5, samples[1] to token 2, etc.
"""

# Context-aware batch reward function
ContextAwareBatchRewardFunction = Callable[
    [
        NDArray[np.float64],  # samples: shape (n_arms, n_contexts, size, ...)
        List[Any],  # action_tokens: length n_arms (ordered by arm index)
        Sized,  # X: shape (n_contexts, n_features)
    ],
    NDArray[np.float64],  # returns: shape (n_arms, n_contexts, size)
]
"""
Context-aware batch reward function that also receives context information.

Parameters
----------
samples : NDArray[np.float64]
    Samples from the learner with shape (n_arms, n_contexts, size, ...).
    The first dimension corresponds to arms in the same order as action_tokens.
action_tokens : List[Any]
    List of action tokens, one per arm, in the order corresponding to the
    arms in the agent's arms list.
X : Sized
    Original context data with shape (n_contexts, n_features), before
    arm featurization. This is the same context passed to pull().

Returns
-------
NDArray[np.float64]
    Reward values with shape (n_arms, n_contexts, size), maintaining the
    same arm ordering as the input.
"""

# ContextType must be both iterable and have a length


class Learner(Protocol[X_contra]):
    """Protocol defining the learner interface with contravariant X type parameter."""

    def sample(self, X: X_contra, size: int = 1) -> NDArray[np.float64]: ...

    def partial_fit(
        self,
        X: X_contra,
        y: NDArray[np.float64],
        sample_weight: Optional[NDArray[np.float64]] = None,
    ) -> Self: ...
    def decay(self, X: X_contra, *, decay_rate: Optional[float] = None) -> None: ...
    def predict(self, X: X_contra) -> NDArray[np.float64]: ...
    @property
    def random_state(self) -> Union[np.random.Generator, int, None]: ...
    @random_state.setter
    def random_state(self, value: Union[np.random.Generator, int, None]) -> None: ...


def requires_learner(
    func: Callable[Concatenate[A, P], R],
) -> Callable[Concatenate[A, P], R]:
    """Decorator to check if the arm has a learner set."""

    @wraps(func)
    def wrapper(self: A, *args: P.args, **kwargs: P.kwargs) -> R:
        if self.learner is None:
            raise ValueError("Learner is not set.")
        return func(self, *args, **kwargs)

    return wrapper


def _accepts_context(
    func: RewardFunction,
) -> TypeGuard[ContextAwareRewardFunction[ContextType]]:
    """Detect if reward function accepts 'X' context parameter."""
    try:
        sig = inspect.signature(func)
        return "X" in sig.parameters
    except (ValueError, TypeError):
        return False


def _accepts_context_batch(
    func: Union[BatchRewardFunction, ContextAwareBatchRewardFunction],
) -> TypeGuard[ContextAwareBatchRewardFunction]:
    """Detect if batch reward function accepts 'X' context parameter."""
    try:
        sig = inspect.signature(func)
        params = list(sig.parameters.keys())
        # Check if third parameter exists and is named 'X'
        return len(params) >= 3 and params[2] == "X"
    except (ValueError, TypeError):
        return False


def identity(x: NDArray[np.float64]) -> NDArray[np.float64]:
    return x


def is_identity_function(func: Any) -> bool:
    """Check if a function is the identity function."""
    return func is identity


def batch_identity(
    samples: NDArray[np.float64], action_tokens: List[Any]
) -> NDArray[np.float64]:
    """Batch identity function that ignores action_tokens."""
    return samples


def apply_reward_function(
    reward_function: RewardFunction,
    samples: NDArray[np.float64],
    context: Optional[ContextType] = None,
) -> NDArray[np.float64]:
    """
    Apply a reward function with automatic context detection.

    This wrapper reduces cyclomatic complexity by centralizing the logic
    for determining whether to pass context to a reward function.

    Parameters
    ----------
    reward_function : RewardFunction
        The reward function to apply
    samples : NDArray[np.float64]
        The samples to transform
    context : Optional[ContextType]
        The context to pass if the function accepts it

    Returns
    -------
    NDArray[np.float64]
        The transformed samples
    """
    if context is not None and _accepts_context(reward_function):
        # TypeGuard ensures this function accepts context
        context_func = cast(ContextAwareRewardFunction[ContextType], reward_function)
        return context_func(samples, context)
    else:
        # Traditional function
        traditional_func = cast(TraditionalRewardFunction, reward_function)
        return traditional_func(samples)


class Arm(MemoryUsageMixin, Generic[ContextType, TokenType]):
    """Single arm of a multi-armed bandit.

    An arm pairs a Bayesian learner with an action token and an
    optional reward function. The learner maintains a posterior
    distribution over the outcome model; the arm draws from this
    posterior and composes the result with the reward function to
    produce reward samples that drive policy decisions.

    Type Parameters
    ---------------
    ContextType : Input array type (e.g. ``NDArray``, ``DataFrame``).
    TokenType : Action token type returned by ``pull``.

    Parameters
    ----------
    action_token : TokenType
        Identifier returned when this arm is selected. Can be any
        hashable value (int, str, enum, etc.).
    reward_function : callable, default=None
        Transforms raw learner samples into reward values used by the
        policy. If None, the identity function is used (raw samples
        are treated as rewards directly). See *Notes* for the formal
        definition.
    learner : Learner, default=None
        Bayesian estimator that maintains the posterior over outcomes.
        Must implement the ``Learner`` protocol (``sample``,
        ``partial_fit``, ``decay``, ``predict``). Typically set to
        None when arms are passed to a ``LipschitzContextualAgent``,
        which assigns a shared learner during initialization.

    Attributes
    ----------
    action_token : TokenType
        The action identifier for this arm.
    reward_function : callable
        The reward transformation applied to posterior samples.
    learner : Learner or None
        The Bayesian estimator backing this arm.

    See Also
    --------
    Agent : Non-contextual multi-armed bandit agent.
    ContextualAgent : Contextual multi-armed bandit agent.
    LipschitzContextualAgent : Contextual agent with shared learner
        for large or continuous action spaces.

    Notes
    -----
    The arm's design reflects the separation between *inference*,
    *utility*, and *decision* in Bayesian decision theory [1]_:

    - The **learner** performs inference, maintaining a posterior
      :math:`p(\\theta_a \\mid \\mathcal{D}_a)` over an outcome
      model (e.g. click-through rates, conversion counts).
    - The **reward function** :math:`g_a` encodes the utility — it
      maps raw outcomes to decision-relevant values (e.g. expected
      revenue, profit margin).
    - The **policy** selects the action that maximizes expected
      utility under posterior uncertainty.

    Sampling from arm :math:`a` given context :math:`x` produces:

    .. math::

        \\tilde{r}_a(x)
        = g_a\\!\\bigl(\\tilde{\\theta}_a(x)\\bigr),
        \\qquad
        \\tilde{\\theta}_a(x) \\sim p(\\theta_a \\mid \\mathcal{D}_a)

    where :math:`\\tilde{\\theta}_a(x)` is a posterior predictive
    draw and :math:`g_a` is the reward function. This is a
    Monte Carlo approximation to the expected utility:

    .. math::

        U_a(x)
        = \\mathbb{E}_{\\theta_a \\mid \\mathcal{D}_a}
          \\bigl[g_a\\!\\bigl(\\theta_a(x)\\bigr)\\bigr]
        \\approx \\frac{1}{S} \\sum_{s=1}^{S}
          g_a\\!\\bigl(\\tilde{\\theta}_a^{(s)}(x)\\bigr)

    This separation is what allows the same learner (e.g. a
    ``GammaRegressor`` modeling click rates) to be used with
    different reward functions depending on the business objective,
    without retraining.

    The policy receives these reward samples
    :math:`\\tilde{r}_a(x)` for every arm and selects the arm to
    play according to its selection rule (e.g. Thompson sampling
    picks the arm with the highest single draw, UCB picks the arm
    with the highest upper quantile across many draws).

    **Disjoint vs. shared learners.** The ``Arm`` class supports two
    fundamentally different bandit architectures depending on how
    learners are assigned:

    *Disjoint* — each arm carries its own independent learner with
    parameters :math:`\\theta_a`. The posteriors are separate:

    .. math::

        p(\\theta_a \\mid \\mathcal{D}_a)
        \\perp
        p(\\theta_b \\mid \\mathcal{D}_b)
        \\quad \\text{for } a \\neq b

    An observation for arm :math:`a` updates only that arm's
    posterior. This is the standard multi-armed bandit setup used by
    ``Agent`` and ``ContextualAgent``, and is appropriate when the
    arms represent qualitatively different actions with no shared
    structure.

    *Shared* — all arms reference the same learner instance with a
    single parameter vector :math:`\\theta`. Each arm augments the
    context :math:`x` with arm-specific features via a featurizer
    :math:`\\phi_a`, so arm :math:`a` effectively models:

    .. math::

        \\tilde{\\theta}_a(x)
        = f\\!\\bigl(\\phi_a(x);\\, \\theta\\bigr),
        \\qquad
        \\theta \\sim p(\\theta \\mid \\mathcal{D})

    where :math:`\\mathcal{D} = \\bigcup_a \\mathcal{D}_a` pools
    observations across all arms. An observation for any arm updates
    the shared posterior, enabling generalization across the action
    space. This is the architecture used by
    ``LipschitzContextualAgent`` and is appropriate when the action
    space is large or continuous and rewards vary smoothly with the
    action (Lipschitz continuity).

    **Posterior updates.** After observing outcome :math:`y` given
    context :math:`x`, calling ``update(x, y)`` performs the
    conjugate (or approximate) Bayesian update:

    .. math::

        p(\\theta_a \\mid \\mathcal{D}_a)
        \\;\\longrightarrow\\;
        p(\\theta_a \\mid \\mathcal{D}_a \\cup \\{(x, y)\\})

    Crucially, the update is performed on the *raw outcome*
    :math:`y`, not the transformed reward :math:`g_a(y)`. The
    learner models what *actually happens* (the data-generating
    process), while the reward function captures what that outcome
    is *worth*. These are fundamentally different quantities.

    Consider a marketing example: a learner models whether a user
    converts (a binary outcome), but different arms correspond to
    campaigns with different costs. The observable :math:`y \\in
    \\{0, 1\\}` is the same regardless of cost — a conversion is a
    conversion. The cost structure lives entirely in the reward
    function :math:`g_a`, which might compute
    ``revenue - cost_a`` for each arm. Training the learner on
    :math:`g_a(y)` would conflate the conversion model with the
    cost model, making it impossible to update costs without
    retraining. By keeping them separate, the posterior over
    conversion rates remains valid even if campaign costs change,
    and only the reward function needs updating.

    **Decay.** For restless bandit problems where the reward
    distribution may change over time, calling ``decay`` shrinks the
    learner's posterior precision, increasing uncertainty and allowing
    the model to adapt to non-stationarity. See the individual
    estimator documentation for the specific decay mechanics.

    References
    ----------
    .. [1] Russo, Daniel J., et al. "A Tutorial on Thompson
       Sampling." *Foundations and Trends in Machine Learning*
       11.1 (2018): 1-96.

    Examples
    --------
    Basic arm with identity reward (raw posterior samples = rewards):

    >>> import numpy as np
    >>> from bayesianbandits import Arm, GammaRegressor
    >>> arm = Arm(action_token="ad_A", learner=GammaRegressor(1, 1))
    >>> arm.update(np.array([[1]]), np.array([5]))
    >>> arm.sample(np.array([[1]]), size=3).shape
    (3, 1)

    Arm with a reward function that converts click-through rate
    to expected revenue:

    >>> revenue_per_click = 0.50
    >>> arm = Arm(
    ...     action_token="ad_B",
    ...     reward_function=lambda ctr: ctr * revenue_per_click,
    ...     learner=GammaRegressor(1, 1),
    ... )
    >>> arm.update(np.array([[1]]), np.array([3]))
    >>> samples = arm.sample(np.array([[1]]), size=2)
    >>> samples.shape
    (2, 1)
    """

    def __init__(
        self,
        action_token: TokenType,
        reward_function: Optional[RewardFunction] = None,
        learner: Optional[Learner[ContextType]] = None,
    ) -> None:
        self.action_token: TokenType = action_token
        self.reward_function = identity if reward_function is None else reward_function
        self.learner = learner

    def __set_name__(self, owner: type, name: str) -> None:
        self.name = name

    @requires_learner
    def pull(self) -> TokenType:
        """Return this arm's action token.

        This is typically called by the agent after the policy has
        selected this arm. The token identifies the action to take
        in the environment.

        Returns
        -------
        TokenType
            The action token for this arm.
        """
        return self.action_token

    @requires_learner
    def sample(self, X: ContextType, size: int = 1) -> NDArray[np.float64]:
        """Draw posterior predictive samples and apply the reward function.

        Computes :math:`\\tilde{r}_a(x) = g_a(\\tilde{\\theta}_a(x))`
        where :math:`\\tilde{\\theta}_a(x)` is drawn from the
        learner's posterior predictive distribution and :math:`g_a`
        is the reward function.

        Parameters
        ----------
        X : ContextType
            Context matrix of shape ``(n_contexts, n_features)``.
        size : int, default=1
            Number of posterior samples to draw per context.

        Returns
        -------
        NDArray[np.float64]
            Reward samples of shape ``(size, n_contexts)``.
        """
        assert self.learner is not None
        samples = self.learner.sample(X, size)
        return apply_reward_function(self.reward_function, samples, X)

    @requires_learner
    def sample_marginal(self, X: ContextType, size: int = 1) -> NDArray[np.float64]:
        """Draw per-context marginal reward samples.

        Uses the learner's ``sample_marginal`` when available -- iid
        draws from each context's exact marginal posterior predictive,
        much cheaper than ``sample`` for large ``size``. Falls back to
        joint ``sample`` when the learner has no ``sample_marginal``,
        when the learner's class overrides ``sample`` without also
        overriding ``sample_marginal``, or when an ``Arm`` subclass
        overrides ``sample`` -- custom sampling behavior must not be
        bypassed; per-context marginals are identical either way.
        Draws are independent across contexts, so use this only for
        per-context statistics such as means and quantiles, never for
        comparisons across contexts within a draw.

        Parameters
        ----------
        X : ContextType
            Context matrix of shape ``(n_contexts, n_features)``.
        size : int, default=1
            Number of samples to draw per context.

        Returns
        -------
        NDArray[np.float64]
            Reward samples of shape ``(size, n_contexts)``.
        """
        assert self.learner is not None
        if type(self).sample is not Arm.sample:
            return self.sample(X, size)
        samples = resolve_marginal_sampler(self.learner)(X, size)
        return apply_reward_function(self.reward_function, samples, X)

    @requires_learner
    def update(
        self,
        X: ContextType,
        y: NDArray[np.float64],
        sample_weight: Optional[NDArray[np.float64]] = None,
    ) -> None:
        """Perform a Bayesian posterior update with observed outcomes.

        Updates the learner's posterior using the conjugate (or
        approximate) update rule:

        .. math::

            p(\\theta \\mid \\mathcal{D})
            \\;\\longrightarrow\\;
            p(\\theta \\mid \\mathcal{D} \\cup \\{(X, y)\\})

        Parameters
        ----------
        X : ContextType
            Context matrix of shape ``(n_samples, n_features)``.
        y : NDArray[np.float64]
            Observed outcomes of shape ``(n_samples,)``.
        sample_weight : NDArray[np.float64] or None, default=None
            Per-sample weights. Used for importance-weighted updates
            in adversarial bandit algorithms (e.g. EXP3).
        """
        assert self.learner is not None
        self.learner.partial_fit(X, y, sample_weight)

    @requires_learner
    def decay(self, X: ContextType, *, decay_rate: Optional[float] = None) -> None:
        """Increase posterior uncertainty for non-stationary environments.

        Shrinks the learner's posterior precision, allowing the model
        to forget old observations and adapt to changing reward
        distributions (restless bandit setting).

        Parameters
        ----------
        X : ContextType
            Context matrix (required by the learner interface; used
            by context-dependent estimators).
        decay_rate : float or None, default=None
            Override the learner's default ``learning_rate``. Values
            less than 1 geometrically shrink posterior precision.
        """
        assert self.learner is not None
        self.learner.decay(X, decay_rate=decay_rate)

    def __repr__(self) -> str:
        return (
            f"Arm(action_token={self.action_token},"
            f" reward_function={self.reward_function})"
        )


def _defines_before_sample(learner: Any, method: str) -> bool:
    """Walking the MRO from the most-derived class, is ``method`` defined
    at or above the first ``sample`` override?

    The first class defining either name decides, so a class that
    overrides ``sample`` without also overriding ``method`` (e.g. to
    clip or transform draws) never has its custom sampling bypassed.
    """
    for klass in type(learner).__mro__:
        if method in vars(klass):
            return True
        if "sample" in vars(klass):
            return False
    return False


def resolve_marginal_sampler(learner: Any) -> Callable[..., NDArray[np.float64]]:
    """Resolve a learner's marginal sampler, falling back to joint ``sample``.

    Returns the learner's ``sample_marginal`` only when it is safe to
    use (see :func:`_defines_before_sample`); per-row marginals are
    identical either way, the fallback is merely slower.
    """
    if _defines_before_sample(learner, "sample_marginal"):
        return cast(Callable[..., NDArray[np.float64]], learner.sample_marginal)
    return cast(Callable[..., NDArray[np.float64]], learner.sample)


def resolve_reward_space_sampler(
    learner: Any, n_rows: int, size: int, block_size: Optional[int] = None
) -> Optional[Callable[..., NDArray[np.float64]]]:
    """Resolve a learner's reward-space joint sampler, or None to use ``sample``.

    Returns a ``(X, size) -> samples`` callable bound to the learner's
    ``sample_reward_space`` with ``block_size`` applied, only when it is
    safe *and* profitable to use: ``sample_reward_space`` must be
    defined at or above any ``sample`` override (see
    :func:`_defines_before_sample` -- the two are distributionally
    identical for the built-in learners, but not necessarily for a
    subclass), and the learner's ``_use_reward_space`` flop model must
    prefer reward space for this ``(n_rows, size)`` shape. A learner
    that defines ``sample_reward_space`` without the ``_use_reward_space``
    gate is never routed to reward space. Returns None otherwise;
    callers fall back to ``sample``, which is always correct.
    """
    if not _defines_before_sample(learner, "sample_reward_space"):
        return None
    gate = getattr(learner, "_use_reward_space", None)
    if gate is None or not gate(n_rows, size, block_size):
        return None

    def sampler(X: Any, size: int = 1, _learner: Any = learner) -> NDArray[np.float64]:
        return cast(
            NDArray[np.float64],
            _learner.sample_reward_space(X, size, block_size=block_size),
        )

    return sampler


def _context_major_permutation(n_arms: int, n_contexts: int) -> NDArray[np.intp]:
    """Row order gathering each context's arm rows into one consecutive block.

    Stacked arm features are arm-major (row ``a * n_contexts + c``); the
    blocked reward-space sampler needs each context's rows adjacent so a
    block is jointly drawn per context.
    """
    return np.arange(n_arms * n_contexts).reshape(n_arms, n_contexts).T.ravel()


def _take_rows(X: Any, indices: NDArray[np.intp]) -> Any:
    """Positionally select rows from an array, sparse matrix, or DataFrame.

    Plain ``X[indices]`` selects *columns by label* on a DataFrame, so
    DataFrames must go through ``iloc``.
    """
    if hasattr(X, "iloc"):
        return X.iloc[indices]
    return X[indices]


def _sample_context_major_blocks(
    joint_sampler: Callable[..., NDArray[np.float64]],
    X_stacked: Any,
    n_arms: int,
    n_contexts: int,
    size: int,
) -> NDArray[np.float64]:
    """Draw per-context joint blocks from arm-major stacked feature rows.

    Permutes the arm-major stack (row ``a * n_contexts + c``)
    context-major so each context's arm rows form one consecutive
    jointly-drawn block, samples, and restores the (arm, context) axes.
    Returns shape ``(n_arms, n_contexts, size)``, a zero-copy view of
    the drawn array.
    """
    if n_arms == 1 or n_contexts == 1:
        # the permutation is the identity; skip the gather-copy
        X_ctx_major = X_stacked
    else:
        perm = _context_major_permutation(n_arms, n_contexts)
        X_ctx_major = _take_rows(X_stacked, perm)
    drawn = joint_sampler(X_ctx_major, size=size)
    return drawn.reshape(size, n_contexts, n_arms).transpose(2, 1, 0)


def posterior_identity(learner: Any) -> Any:
    """The object carrying ``learner``'s posterior state.

    A :class:`~bayesianbandits.LearnerPipeline` delegates to an inner
    learner, so two pipelines wrapping one estimator share a posterior
    even though the pipeline objects differ. Comparing this identity
    catches that, where comparing the learners themselves would not.
    """
    return getattr(learner, "learner", learner)
