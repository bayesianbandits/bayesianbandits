"""
Base classes for Bayesian bandit policies.
"""

from typing import (
    Generic,
    List,
    Optional,
)

import numpy as np
from numpy.typing import NDArray

from .._arm import Arm, ContextType, TokenType


class PolicyDefaultUpdate(Generic[ContextType, TokenType]):
    def update(
        self,
        arm: Arm[ContextType, TokenType],
        X: ContextType,
        y: NDArray[np.float64],
        all_arms: List[Arm[ContextType, TokenType]],
        rng: np.random.Generator,
        sample_weight: Optional[NDArray[np.float64]] = None,
    ) -> None:
        """Default update implementation that simply updates the arm."""
        arm.update(X, y, sample_weight=sample_weight)
