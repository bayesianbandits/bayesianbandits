from __future__ import annotations

from typing import (
    Any,
    Optional,
    Protocol,
    Union,
)

import numpy as np
from numpy.typing import NDArray


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
    ) -> NDArray[np.float64]: ...

    def partial_fit(self, X: NDArray[Any], y: NDArray[Any]) -> "Learner": ...

    def set_params(self, **params: Any) -> "Learner": ...


class DecayingLearner(Learner, Protocol):
    learning_rate: float

    def decay(self, X: NDArray[Any], *, decay_rate: Optional[float] = None) -> None: ...
