from __future__ import annotations

from functools import wraps
from typing import (
    Any,
    Callable,
    Generic,
    Iterable,
    List,
    Optional,
    Protocol,
    TypeVar,
    Union,
)

import numpy as np
from numpy.typing import NDArray
from scipy.sparse import csr_matrix, issparse
from scipy.sparse import vstack as sparse_vstack
from typing_extensions import Concatenate, ParamSpec, Self

try:
    import pandas as pd

    HAS_PANDAS = True
except ImportError:
    HAS_PANDAS = False

P = ParamSpec("P")
R = TypeVar("R", covariant=True)
RewardFunction = Union[
    Callable[..., np.float64],
    Callable[..., NDArray[np.float64]],
    Callable[..., Union[np.float64, NDArray[np.float64]]],
]
TokenType = TypeVar("TokenType")
X_contra = TypeVar("X_contra", contravariant=True)  # Contravariant for input types
A = TypeVar("A", bound="Arm[Any, Any]")
ContextType = TypeVar("ContextType", bound=Iterable[Any])


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


def identity(
    x: Union[np.float64, NDArray[np.float64]],
) -> Union[np.float64, NDArray[np.float64]]:
    return x


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
        return self.reward_function(self.learner.sample(X, size))  # type: ignore[return-value]

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
    if not (
        hasattr(first_arm.learner, "transform")
        and hasattr(first_arm.learner, "final_estimator")
    ):
        return False

    # Check all share the same model instance
    first_model = first_arm.learner.final_estimator  # type: ignore
    return all(
        arm.learner.final_estimator is first_model  # type: ignore
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
    if HAS_PANDAS and isinstance(first, pd.DataFrame):  # type: ignore[has-type]
        if not all(isinstance(x, pd.DataFrame) for x in feature_list):  # type: ignore[has-type]
            raise ValueError("Cannot stack mixed DataFrame and non-DataFrame objects")
        return pd.concat(feature_list, ignore_index=True, sort=False, copy=False)  # type: ignore[has-type]

    # Case 2: Sparse arrays
    if issparse(first):
        if not all(issparse(x) for x in feature_list):
            raise ValueError("Cannot stack mixed sparse and dense arrays")
        # Convert to CSR for efficient row operations
        return sparse_vstack([csr_matrix(x) for x in feature_list], format="csr")

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
    >>> from bayesianbandits import Arm, BayesianGLM, Pipeline
    >>> # All arms must share the same model
    >>> shared_model = BayesianGLM(alpha=1.0)
    >>> arms = []
    >>> for i in range(100):
    ...     pipeline = Pipeline([
    ...         ('model', shared_model)  # Same instance!
    ...     ])
    ...     arms.append(Arm(i, learner=pipeline))
    >>>
    >>> # Fast batched sampling
    >>> samples = batch_sample_arms(arms, X=np.random.rand(10, 5), size=5)
    """
    if not can_batch_arms(arms):
        return None

    # Transform features for each arm
    n_arms = len(arms)
    feature_list = []

    # Pre-check context length for memory allocation
    n_contexts = len(X) if hasattr(X, "__len__") else 1  # type: ignore[has-type]

    # Transform all features
    for arm in arms:
        X_transformed = arm.learner.transform(X)  # type: ignore[return-value]
        feature_list.append(X_transformed)

    # Stack features
    try:
        X_stacked = stack_features(feature_list)
    except ValueError:
        return None

    # Single batched sample
    model = arms[0].learner.final_estimator  # type: ignore[attr-defined]
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
        results[i] = arm.reward_function(samples[i])

    return results
