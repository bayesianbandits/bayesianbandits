"""One-hot arm featurizer for discrete actions."""

from typing import Any, Iterable, Sequence

import numpy as np
from numpy.typing import NDArray

from .._arm_featurizer import ArmFeaturizer

__all__ = ["OneHotArmFeaturizer"]


class OneHotArmFeaturizer(ArmFeaturizer[int]):
    """Vectorized one-hot encoding for discrete actions.

    This featurizer appends one-hot encoded action features to the context
    features for each arm. Useful for discrete action spaces where each
    action is represented by an integer index.

    Parameters
    ----------
    n_actions : int
        Total number of possible actions. Action tokens should be
        integers in the range [0, n_actions).

    Attributes
    ----------
    n_actions : int
        The total number of possible actions.

    Examples
    --------
    >>> import numpy as np
    >>>
    >>> # Create featurizer for 3 possible actions
    >>> featurizer = OneHotArmFeaturizer(n_actions=3)
    >>>
    >>> # Context features
    >>> X = np.array([[1.0, 2.0], [3.0, 4.0]])  # 2 contexts, 2 features
    >>>
    >>> # Transform with action tokens [0, 2]
    >>> X_transformed = featurizer.transform(X, action_tokens=[0, 2])
    >>> X_transformed.shape
    (4, 5)
    >>>
    >>> # First two rows are for action 0 (context features + one-hot for action 0)
    >>> X_transformed[:2]
    array([[1., 2., 1., 0., 0.],
           [3., 4., 1., 0., 0.]])
    >>>
    >>> # Last two rows are for action 2 (context features + one-hot for action 2)
    >>> X_transformed[2:]
    array([[1., 2., 0., 0., 1.],
           [3., 4., 0., 0., 1.]])
    """

    def __init__(self, n_actions: int):
        """Initialize the one-hot arm featurizer.

        Parameters
        ----------
        n_actions : int
            Total number of possible actions.

        Raises
        ------
        ValueError
            If n_actions is not positive.
        """
        if n_actions <= 0:
            raise ValueError(f"n_actions must be positive, got {n_actions}")
        self.n_actions = n_actions

    def transform(
        self, X: Iterable[Any], *, action_tokens: Sequence[int]
    ) -> NDArray[np.floating]:
        """Transform features by appending one-hot encoded actions.

        Parameters
        ----------
        X : Iterable[Any]
            Input context features, typically array-like of shape
            (n_contexts, n_features).
        action_tokens : sequence of int
            Action indices for each arm. Each token should be an
            integer in [0, n_actions).

        Returns
        -------
        X_transformed : ndarray of shape (n_contexts * n_arms, n_features + n_actions)
            Stacked features with one-hot encoded actions appended.

        Raises
        ------
        ValueError
            If any action token is out of bounds.
        """
        X = np.asarray(X)
        n_contexts = X.shape[0]
        n_arms = len(action_tokens)

        # Handle empty case
        if n_arms == 0:
            return np.empty((0, X.shape[1] + self.n_actions), dtype=np.float64)

        # Validate action tokens
        action_array = np.asarray(action_tokens)
        if np.any(action_array < 0) or np.any(action_array >= self.n_actions):
            raise ValueError(
                f"All action tokens must be in [0, {self.n_actions}), "
                f"got {action_tokens}"
            )

        # Create one-hot encoding block
        one_hot_block = np.zeros((n_arms, self.n_actions))
        one_hot_block[np.arange(n_arms), action_array] = 1

        # Repeat one-hot for each context
        one_hot_repeated = np.repeat(one_hot_block, n_contexts, axis=0)

        # Tile contexts for each arm
        X_tiled = np.tile(X, (n_arms, 1))

        # Concatenate context and one-hot features
        return np.hstack([X_tiled, one_hot_repeated]).astype(np.float64)
