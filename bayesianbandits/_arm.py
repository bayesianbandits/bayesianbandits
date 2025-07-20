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
from typing_extensions import TypeGuard

import numpy as np
from numpy.typing import NDArray
from scipy.sparse import csr_matrix, issparse  # type: ignore[import]
from scipy.sparse import vstack as sparse_vstack  # type: ignore[import]
from typing_extensions import Concatenate, ParamSpec, Self

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
    """Check if a function is the identity function.

    This handles both direct reference checks and module reloading issues.
    """
    return func is identity or (
        hasattr(func, "__name__")
        and func.__name__ == "identity"
        and hasattr(func, "__module__")
        and func.__module__ == "bayesianbandits._arm"
    )


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
    """Arm of a bandit with type-safe X parameter.

    Type Parameters
    ---------------
    ContextType : Input array type
    TokenType : Action token type
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
        """Pull the arm."""
        return self.action_token

    @requires_learner
    def sample(self, X: ContextType, size: int = 1) -> NDArray[np.float64]:
        """Sample from learner and compute the reward."""
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
        """Update the learner."""
        assert self.learner is not None
        self.learner.partial_fit(X, y, sample_weight)

    @requires_learner
    def decay(self, X: ContextType, *, decay_rate: Optional[float] = None) -> None:
        """Decay the learner."""
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
