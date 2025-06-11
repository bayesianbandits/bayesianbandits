"""Function-based arm featurizer wrapper."""

from typing import Callable, Sequence, Sized

import numpy as np
from numpy.typing import NDArray

from .._arm import TokenType
from .._arm_featurizer import ArmFeaturizer

__all__ = ["FunctionArmFeaturizer"]


class FunctionArmFeaturizer(ArmFeaturizer[TokenType]):
    """Create ArmFeaturizer from a user-provided function.

    Parameters
    ----------
    func : callable
        Function with signature (X, action_tokens) -> X_transformed.
        Must return a 3D array of shape (n_contexts, n_features_out, n_arms)
        where n_features_out includes both original and arm-specific features.

    Examples
    --------
    >>> import numpy as np
    >>> # One-hot encoding example
    >>> def one_hot_featurizer(X, action_tokens):
    ...     X = np.asarray(X)
    ...     n_contexts, n_features = X.shape
    ...     n_arms = len(action_tokens)
    ...     n_actions = 5  # total possible actions
    ...
    ...     # Create 3D array: (n_contexts, n_features + n_actions, n_arms)
    ...     X_3d = np.zeros((n_contexts, n_features + n_actions, n_arms))
    ...
    ...     # Broadcast context features to all arms
    ...     X_3d[:, :n_features, :] = X[:, :, np.newaxis]
    ...
    ...     # Add one-hot encoding for each arm
    ...     for i, token in enumerate(action_tokens):
    ...         X_3d[:, n_features + token, i] = 1
    ...
    ...     return X_3d
    >>>
    >>> featurizer = FunctionArmFeaturizer(one_hot_featurizer)
    >>> X = np.array([[1, 2], [3, 4]])
    >>> result = featurizer.transform(X, action_tokens=[0, 2])
    >>> result.shape
    (4, 7)

    >>> # Continuous action features example
    >>> def continuous_featurizer(X, action_tokens):
    ...     X = np.asarray(X)
    ...     n_contexts, n_features = X.shape
    ...     n_arms = len(action_tokens)
    ...
    ...     # Create output array with space for polynomial features
    ...     X_3d = np.zeros((n_contexts, n_features + 3, n_arms))
    ...
    ...     # Broadcast context features
    ...     X_3d[:, :n_features, :] = X[:, :, np.newaxis]
    ...
    ...     # Add polynomial features for each arm
    ...     for i, token in enumerate(action_tokens):
    ...         X_3d[:, n_features, i] = token
    ...         X_3d[:, n_features + 1, i] = token**2
    ...         X_3d[:, n_features + 2, i] = np.sin(2*np.pi*token)
    ...
    ...     return X_3d
    >>>
    >>> featurizer = FunctionArmFeaturizer(continuous_featurizer)
    """

    def __init__(
        self, func: Callable[[Sized, Sequence[TokenType]], NDArray[np.floating]]
    ):
        """Initialize the function-based arm featurizer.

        Parameters
        ----------
        func : callable
            Function that takes (X, action_tokens) and returns a 3D array.
        """
        self.func = func

    def transform(
        self, X: Sized, *, action_tokens: Sequence[TokenType]
    ) -> NDArray[np.floating]:
        """Transform features using the provided function.

        Parameters
        ----------
        X : Sized
            Input context features.
        action_tokens : sequence of TokenType
            Action tokens for each arm.

        Returns
        -------
        X_transformed : ndarray
            Stacked features in 2D format.

        Raises
        ------
        ValueError
            If the function output is not 3D or has incorrect dimensions.
        """
        result = self.func(X, action_tokens)

        # Validate 3D output
        if result.ndim != 3:
            raise ValueError(
                f"Expected 3D output from function with shape "
                f"(n_contexts, n_features_out, n_arms), got shape {result.shape}"
            )

        _, n_features_out, n_arms = result.shape

        if n_arms != len(action_tokens):
            raise ValueError(
                f"3D output has {n_arms} arms in last dimension, "
                f"but {len(action_tokens)} action_tokens provided"
            )

        # Reshape from (n_contexts, n_features, n_arms) to stacked format
        # First transpose to (n_arms, n_contexts, n_features)
        # Then reshape to (n_arms * n_contexts, n_features)
        result_transposed = result.transpose(2, 0, 1)
        return result_transposed.reshape(-1, n_features_out)
