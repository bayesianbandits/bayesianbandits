from typing import Any, Callable, Dict, Optional, Protocol, Union, runtime_checkable

import numpy as np
from numpy.typing import ArrayLike, NDArray


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


@runtime_checkable
class ArmProtocol(Protocol):
    """Protocol for Arms and Bandits. Bandits themselves can be used as arms
    in other bandits, so both must implement the same minimal interface.

    Each Arm or Bandit must implement the following methods:
    - `pull`
    - `sample`
    - `update`

    """

    learner: Optional[Learner]

    def pull(self, X: Optional[ArrayLike] = None) -> None:
        ...

    def sample(self, X: Optional[ArrayLike] = None, size: int = 1) -> ArrayLike:
        ...

    def update(self, X: Optional[ArrayLike], y: Optional[ArrayLike] = None) -> None:
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

    arms: Dict[str, ArmProtocol]
    policy: Callable[..., ArmProtocol]
    last_arm_pulled: Optional[ArmProtocol]
    rng: Union[np.random.Generator, int, None]

    def __init__(
        *args: Any, rng: Union[np.random.Generator, int, None] = None, **kwargs: Any
    ) -> None:
        ...

    def pull(self, X: Optional[ArrayLike] = None, **kwargs: Any) -> None:
        ...

    def sample(
        self, X: Optional[ArrayLike] = None, size: int = 1, **kwargs: Any
    ) -> ArrayLike:
        ...

    def update(
        self, X: Optional[ArrayLike], y: Optional[ArrayLike] = None, **kwargs: Any
    ) -> None:
        ...
