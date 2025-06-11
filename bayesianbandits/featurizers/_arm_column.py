"""Arm column featurizer that appends arm tokens as a column."""

import importlib.util
from typing import Any, Sequence, Sized, TypeVar

import numpy as np

from .._arm_featurizer import ArmFeaturizer

__all__ = ["ArmColumnFeaturizer"]

HAS_PANDAS = importlib.util.find_spec("pandas") is not None

T = TypeVar("T")


class ArmColumnFeaturizer(ArmFeaturizer[T]):
    """Appends arm tokens as a column to context features.

    This featurizer simply adds the arm token as an additional column to each
    context's features. This allows users to then apply standard sklearn
    transformers (OneHotEncoder, PolynomialFeatures, etc.) to create the
    desired feature representation.

    The output can be used with sklearn's ColumnTransformer to apply different
    transformations to context features vs. the arm column.

    Parameters
    ----------
    column_name : str, default="arm_token"
        Name of the column to add containing the arm tokens.

    Examples
    --------
    >>> import numpy as np
    >>> from sklearn.compose import ColumnTransformer
    >>> from sklearn.preprocessing import OneHotEncoder, PolynomialFeatures
    >>>
    >>> # Create featurizer
    >>> featurizer = ArmColumnFeaturizer()
    >>>
    >>> # Context features
    >>> X = np.array([[1.0, 2.0], [3.0, 4.0]])  # 2 contexts, 2 features
    >>>
    >>> # Example 1: Discrete arm tokens with one-hot encoding
    >>> X_with_arms = featurizer.transform(X, action_tokens=[0, 1, 2])
    >>> X_with_arms.shape
    (6, 3)
    >>>
    >>> # Apply sklearn transformers
    >>> transformer = ColumnTransformer([
    ...     ('context', 'passthrough', [0, 1]),
    ...     ('arm', OneHotEncoder(sparse_output=False), [2])
    ... ])
    >>> X_final = transformer.fit_transform(X_with_arms)
    >>> # Result has 2 context features + 3 one-hot encoded arm features
    >>> X_final.shape
    (6, 5)

    >>> # Example 2: Continuous arm values with polynomial features
    >>> X_with_arms = featurizer.transform(X, action_tokens=[0.1, 0.5, 0.9])
    >>> transformer = ColumnTransformer([
    ...     ('context', 'passthrough', [0, 1]),
    ...     ('arm', PolynomialFeatures(degree=2, include_bias=False), [2])
    ... ])
    >>> X_final = transformer.fit_transform(X_with_arms)
    >>> # Result has 2 context features + 2 polynomial features (x, x^2)
    >>> X_final.shape
    (6, 4)

    >>> # Example 3: With pandas DataFrame (preserves DataFrame structure)
    >>> import pandas as pd
    >>> df = pd.DataFrame({'feature1': [1.0, 3.0], 'feature2': [2.0, 4.0]})
    >>> df_with_arms = featurizer.transform(df, action_tokens=['A', 'B', 'C'])
    >>> # Returns DataFrame with 'arm_token' column added
    >>> list(df_with_arms.columns)
    ['feature1', 'feature2', 'arm_token']

    >>> # Example 4: Using custom column name
    >>> featurizer = ArmColumnFeaturizer(column_name='product_id')
    >>> df_with_arms = featurizer.transform(df, action_tokens=['A', 'B', 'C'])
    >>> list(df_with_arms.columns)
    ['feature1', 'feature2', 'product_id']

    """

    def __init__(self, column_name: str = "arm_token"):
        """Initialize the ArmColumnFeaturizer.

        Parameters
        ----------
        column_name : str, default="arm_token"
            Name of the column to add containing the arm tokens.
        """
        self.column_name = column_name

    def transform(self, X: Sized, *, action_tokens: Sequence[T]) -> Any:
        """Append arm tokens as a column to context features.

        Parameters
        ----------
        X : Sized
            Input context features. Can be a pandas DataFrame, numpy array,
            or other array-like structure of shape (n_contexts, n_features).
        action_tokens : sequence of T
            Action tokens for each arm. Can be any type (int, float, str, etc.).

        Returns
        -------
        X_transformed : DataFrame or ndarray
            If X is a pandas DataFrame, returns a DataFrame with arm tokens
            appended as a new column. Otherwise, returns an ndarray of shape
            (n_contexts * n_arms, n_features + 1).
        """
        n_arms = len(action_tokens)

        # Handle pandas DataFrames
        if HAS_PANDAS:
            import pandas as pd  # type: ignore[import]

            if isinstance(X, pd.DataFrame):
                n_contexts = len(X)

                if n_arms == 0:
                    # Return empty DataFrame with arm column
                    result = pd.DataFrame(columns=list(X.columns) + [self.column_name])  # type: ignore[misc]
                    # Preserve dtypes from original DataFrame
                    for col in X.columns:
                        result[col] = result[col].astype(X[col].dtype)  # type: ignore[misc]
                    return result

                # Tile the DataFrame for each arm
                X_tiled = pd.concat([X] * n_arms, ignore_index=True)

                # Create arm column by repeating each token n_contexts times
                # Convert to list to ensure numpy can handle arbitrary types
                arm_column = np.repeat(np.array(action_tokens), n_contexts)

                # Add arm column to DataFrame
                X_tiled[self.column_name] = arm_column

                return X_tiled

        # Handle all other array-like inputs
        X_array = np.asarray(X)
        n_contexts = X_array.shape[0]

        if n_arms == 0:
            # Return empty array with proper shape
            return np.empty((0, X_array.shape[1] + 1), dtype=X_array.dtype)

        # Tile contexts for each arm
        X_tiled = np.tile(X_array, (n_arms, 1))

        # Create arm column by repeating each token n_contexts times
        # Convert to list to ensure numpy can handle arbitrary types
        arm_column = np.repeat(np.array(action_tokens), n_contexts)[:, np.newaxis]

        # Concatenate context features and arm column
        # Let numpy figure out the appropriate dtype
        return np.hstack([X_tiled, arm_column])
