from __future__ import annotations

from functools import wraps
from typing import (
    Any,
    Callable,
    Generic,
    Iterable,
    Optional,
    Protocol,
    TypeVar,
    Union,
)

import numpy as np
from numpy.typing import NDArray
from typing_extensions import Concatenate, ParamSpec, Self

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
