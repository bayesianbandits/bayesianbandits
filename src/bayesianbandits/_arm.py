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
from scipy.sparse import csr_matrix, issparse  # type: ignore[import]
from scipy.sparse import vstack as sparse_vstack  # type: ignore[import]
from typing_extensions import Concatenate, ParamSpec, Self, TypeGuard

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


class LearnerWithTransform(Learner[X_contra], Protocol[X_contra]):
    """Protocol for learners that support transformation (e.g., Pipeline)."""

    def transform(self, X: X_contra) -> Any: ...
    @property
    def final_estimator(self) -> Learner[Any]: ...


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


class Arm(Generic[ContextType, TokenType]):
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


def can_batch_arms(arms: List[Arm[Any, Any]]) -> bool:
    """Check if arms can be batched (all share the same model)."""
    if not arms:
        return False

    # Fast path: check first arm has required interface
    first_arm = arms[0]
    first_learner = first_arm.learner
    if not (
        first_learner is not None
        and hasattr(first_learner, "transform")
        and hasattr(first_learner, "final_estimator")
    ):
        return False

    # Type guard via hasattr checks
    # Check all share the same model instance
    first_model = getattr(first_learner, "final_estimator")
    return all(
        arm.learner is not None
        and hasattr(arm.learner, "final_estimator")
        and getattr(arm.learner, "final_estimator") is first_model
        for arm in arms[1:]
    )


def stack_features(feature_list: List[Any]) -> Any:
    """Stack feature arrays, handling DataFrames, dense and sparse arrays."""
    if not feature_list:
        raise ValueError("Empty feature list")

    if len(feature_list) == 1:
        return feature_list[0]

    first = feature_list[0]

    # Case 1: Pandas DataFrames (if pandas available)
    if HAS_PANDAS:
        import pandas as pd  # type: ignore[import] # Re-import for type narrowing

        if isinstance(first, pd.DataFrame):
            if not all(isinstance(x, pd.DataFrame) for x in feature_list):
                raise ValueError(
                    "Cannot stack mixed DataFrame and non-DataFrame objects"
                )
            return pd.concat(feature_list, ignore_index=True, sort=False, copy=False)

    # Case 2: Sparse arrays
    if issparse(first):
        if not all(issparse(x) for x in feature_list):
            raise ValueError("Cannot stack mixed sparse and dense arrays")
        # Convert to CSR for efficient row operations
        return sparse_vstack([csr_matrix(x) for x in feature_list], format="csr")  # type: ignore[return-value]

    # Case 3: Dense arrays - numpy is always available
    arrays = [np.asarray(x) for x in feature_list]

    # Check shapes are compatible
    first_shape = arrays[0].shape[1:] if arrays[0].ndim > 1 else ()
    for arr in arrays[1:]:
        if (arr.shape[1:] if arr.ndim > 1 else ()) != first_shape:
            raise ValueError("Incompatible shapes for stacking")

    return np.vstack(arrays)


def batch_sample_arms(
    arms: List[Arm[ContextType, TokenType]], X: ContextType, size: int = 1
) -> Optional[NDArray[np.float64]]:
    """
    Batch sample from arms that share the same model.

    This is optimized for the common case where all arms share a single model
    (e.g., recommendation systems with one model and many items).

    Parameters
    ----------
    arms : List[Arm]
        Arms to sample from. Must all share the same final_estimator.
    X : ContextType
        Context data
    size : int, default=1
        Number of samples to draw from each arm

    Returns
    -------
    samples : NDArray or None
        If batching is possible, returns array of shape (n_arms, n_contexts).
        If batching is not possible, returns None.

    Examples
    --------
    >>> import numpy as np
    >>> from bayesianbandits import Arm, NormalRegressor
    >>> # Create arms with the same model instance
    >>> shared_model = NormalRegressor(alpha=1.0, beta=1.0)
    >>> arms = [Arm(i, learner=shared_model) for i in range(3)]
    >>> X = np.array([[1.0, 2.0], [3.0, 4.0]])
    >>> # Batched sampling returns None if models are different
    >>> samples = batch_sample_arms(arms, X, size=1)
    >>> samples is None  # Same model for all arms, but no Pipeline
    True
    """
    if not can_batch_arms(arms):
        return None

    # Transform features for each arm
    n_arms = len(arms)
    feature_list: List[Any] = []

    # Pre-check context length for memory allocation
    n_contexts = len(X)

    # Transform all features
    for arm in arms:
        # We know from can_batch_arms that all learners are LearnerWithTransform
        learner = cast(LearnerWithTransform[Any], arm.learner)
        X_transformed = learner.transform(X)
        feature_list.append(X_transformed)

    # Stack features
    try:
        X_stacked = stack_features(feature_list)
    except ValueError:
        return None

    # Single batched sample
    # We know from can_batch_arms that first learner is LearnerWithTransform
    first_learner = cast(LearnerWithTransform[Any], arms[0].learner)
    model = first_learner.final_estimator
    samples = model.sample(X_stacked, size=size)

    # Reshape based on size
    if size == 1:
        samples = samples.reshape(n_arms, n_contexts)
    else:
        samples = samples.reshape(size, n_arms, n_contexts).transpose(1, 2, 0)

    # Apply reward functions if non-identity
    # Check using function identity rather than name comparison
    identity_func = arms[0].reward_function if arms else None
    if identity_func and all(arm.reward_function is identity_func for arm in arms):
        return samples

    # Pre-allocate result array
    results = np.empty_like(samples)
    for i, arm in enumerate(arms):
        results[i] = apply_reward_function(arm.reward_function, samples[i], X)

    return results
