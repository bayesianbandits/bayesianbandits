"""Arm featurizers for shared model bandits.

This module provides the ArmFeaturizer interface and implementations for
transforming context features based on action tokens in shared model bandits.
"""

from abc import ABC, abstractmethod
from typing import Any, Generic, Sequence, Sized

from ._arm import TokenType


__all__ = [
    "ArmFeaturizer",
]


class ArmFeaturizer(ABC, Generic[TokenType]):
    """Abstract base class for vectorized arm feature transformation.

    Transforms features for multiple arms in a single call to enable
    efficient batched processing in shared model bandits.

    Type Parameters
    ---------------
    TokenType : type
        The type of action tokens this featurizer accepts.
    """

    @abstractmethod
    def transform(self, X: Sized, *, action_tokens: Sequence[TokenType]) -> Any:
        """Transform features for all arms in a single vectorized call.

        Parameters
        ----------
        X : Sized
            Input context features. Implementations typically expect array-like
            of shape (n_contexts, n_features).
        action_tokens : sequence of TokenType, length n_arms
            Action tokens for each arm to featurize.

        Returns
        -------
        X_transformed : array-like of shape (n_contexts * n_arms, n_features_out)
            Stacked features where rows are ordered as:
            - Rows 0:n_contexts are for action_tokens[0]
            - Rows n_contexts:2*n_contexts are for action_tokens[1]
            - etc.

            The specific type depends on the implementation and input type.

        Notes
        -----
        The output is "stacked" such that each context is repeated for each arm,
        with arm-specific features appended. This allows a single model to process
        all arm-context pairs efficiently.

        Examples
        --------
        >>> import numpy as np
        >>> # Discrete action tokens
        >>> X = np.array([[1, 2], [3, 4]])  # 2 contexts, 2 features
        >>> action_tokens = [0, 1, 2]  # 3 arms
        >>> # After transform: 6 rows (2 contexts * 3 arms)
        >>> # Rows 0-1: context features + arm 0 features
        >>> # Rows 2-3: context features + arm 1 features
        >>> # Rows 4-5: context features + arm 2 features
        """
