"""Continuous arm featurizer for real-valued actions."""

from typing import Any, Iterable, Sequence

import numpy as np
from numpy.typing import NDArray

from .._arm_featurizer import ArmFeaturizer

__all__ = ["ContinuousArmFeaturizer"]


class ContinuousArmFeaturizer(ArmFeaturizer[float]):
    """Vectorized polynomial features for continuous actions.
    
    This featurizer appends polynomial features of continuous action values
    to the context features. Optionally includes interaction terms between
    context features and the action value.
    
    Parameters
    ----------
    degree : int, default=2
        Maximum degree of polynomial features to create from the action value.
        For example, degree=2 includes both action and action^2.
    include_interaction : bool, default=False
        Whether to include interaction terms between context features and
        the action value (i.e., context_i * action).
        
    Attributes
    ----------
    degree : int
        The maximum polynomial degree.
    include_interaction : bool
        Whether interaction terms are included.
        
    Examples
    --------
    >>> import numpy as np
    >>> 
    >>> # Create featurizer with degree 2
    >>> featurizer = ContinuousArmFeaturizer(degree=2)
    >>> 
    >>> # Context features
    >>> X = np.array([[1.0, 2.0], [3.0, 4.0]])  # 2 contexts, 2 features
    >>> 
    >>> # Transform with continuous action tokens
    >>> X_transformed = featurizer.transform(X, action_tokens=[0.1, 0.5, 0.9])
    >>> X_transformed.shape
    (6, 4)
    >>> 
    >>> # First two rows are for action 0.1 (context features + [0.1, 0.1^2])
    >>> X_transformed[:2]
    array([[1.  , 2.  , 0.1 , 0.01],
           [3.  , 4.  , 0.1 , 0.01]])
    >>> 
    >>> # With interactions (context features + action + context*action)
    >>> featurizer_interact = ContinuousArmFeaturizer(degree=1, include_interaction=True)
    >>> X_transformed = featurizer_interact.transform(X, action_tokens=[0.5])
    >>> X_transformed
    array([[1. , 2. , 0.5, 0.5, 1. ],
           [3. , 4. , 0.5, 1.5, 2. ]])
    """
    
    def __init__(self, degree: int = 2, include_interaction: bool = False):
        """Initialize the continuous arm featurizer.
        
        Parameters
        ----------
        degree : int, default=2
            Maximum polynomial degree for action features.
        include_interaction : bool, default=False
            Whether to include context-action interactions.
            
        Raises
        ------
        ValueError
            If degree is not positive.
        """
        if degree <= 0:
            raise ValueError(f"degree must be positive, got {degree}")
        self.degree = degree
        self.include_interaction = include_interaction
        
    def transform(
        self, 
        X: Iterable[Any], 
        *, 
        action_tokens: Sequence[float]
    ) -> NDArray[np.floating]:
        """Transform features by appending polynomial action features.
        
        Parameters
        ----------
        X : Iterable[Any]
            Input context features, typically array-like of shape
            (n_contexts, n_features).
        action_tokens : sequence of float
            Continuous action values for each arm.
            
        Returns
        -------
        X_transformed : ndarray
            Stacked features with polynomial action features appended.
            Shape is (n_contexts * n_arms, n_features + n_poly_features).
            If include_interaction is True, also includes n_features
            interaction terms.
        """
        X = np.asarray(X)
        n_contexts = X.shape[0]
        n_arms = len(action_tokens)
        
        # Handle empty case
        if n_arms == 0:
            n_poly_features = self.degree
            n_interaction_features = X.shape[1] if self.include_interaction else 0
            total_features = X.shape[1] + n_poly_features + n_interaction_features
            return np.empty((0, total_features), dtype=np.float64)
        
        # Build polynomial features for each action
        poly_features = []
        for d in range(1, self.degree + 1):
            # Shape: (n_arms, 1)
            poly_features.append(np.array(action_tokens)[:, np.newaxis] ** d)
        
        # Stack polynomial features: (n_arms, degree)
        poly_block = np.hstack(poly_features)
        
        # Repeat for each context: (n_arms * n_contexts, degree)
        poly_repeated = np.repeat(poly_block, n_contexts, axis=0)
        
        # Tile contexts for each arm
        X_tiled = np.tile(X, (n_arms, 1))
        
        # Base features + polynomial features
        result = np.hstack([X_tiled, poly_repeated]).astype(np.float64)
        
        # Add interaction terms if requested
        if self.include_interaction:
            # Create interaction terms between context and action
            # action_tokens_repeated: (n_arms * n_contexts, 1)
            action_repeated = np.repeat(action_tokens, n_contexts)[:, np.newaxis]
            interactions = X_tiled * action_repeated
            result = np.hstack([result, interactions]).astype(np.float64)
            
        return result