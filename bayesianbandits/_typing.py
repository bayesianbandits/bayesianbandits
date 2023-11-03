from __future__ import annotations

from typing import (
    TYPE_CHECKING,
    Any,
    Callable,
    Dict,
    NewType,
    Optional,
    Protocol,
    Union,
    runtime_checkable,
)

import numpy as np
from numpy.typing import ArrayLike, NDArray

if TYPE_CHECKING:
    from ._arm import Arm

ActionToken = NewType("TokenType", Any)


class Learner(Protocol):
    """Learner protocol for the model underlying each arm.

    Each Learner must implement the following methods:
    - `sample`
    - `partial_fit`

    """

    random_state: Union[np.random.Generator, int, None]

    def sample(
        self,
        X: NDArray[Any],
        size: int = 1,
    ) -> NDArray[np.float_]:
        ...

    def partial_fit(self, X: NDArray[Any], y: NDArray[Any]) -> "Learner":
        ...

    def set_params(self, **params: Any) -> "Learner":
        ...


class DecayingLearner(Learner, Protocol):
    learning_rate: float

    def decay(self, X: NDArray[Any], *, decay_rate: Optional[float] = None) -> None:
        ...


@runtime_checkable
class BanditProtocol(Protocol):
    """Protocol for Bandits.

    Each Bandit must implement the following methods:
    - `choose_and_pull`
    - `update`
    - `pull`
    - `sample`

    """

    arms: Dict[str, "Arm"]
    policy: Callable[..., "Arm"]
    last_arm_pulled: Optional["Arm"]
    rng: Union[np.random.Generator, int, None]
    _contextual: bool

    def __init__(
        *args: Any, rng: Union[np.random.Generator, int, None] = None, **kwargs: Any
    ) -> None:
        ...

    def pull(self, X: Optional[ArrayLike] = None, **kwargs: Any) -> ActionToken:
        ...

    def sample(
        self, X: Optional[ArrayLike] = None, size: int = 1, **kwargs: Any
    ) -> ArrayLike:
        ...

    def update(
        self, X: Optional[ArrayLike], y: Optional[ArrayLike] = None, **kwargs: Any
    ) -> None:
        ...

    @property
    def arm_to_update(self) -> Arm:
        ...
