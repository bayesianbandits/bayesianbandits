"""Pipeline implementation for Bayesian bandits.

Enables sklearn transformers to work with Bayesian learners,
supporting workflows with DataFrames, dicts, and other input types.
"""

from typing import Any, Generic, List, Optional, Tuple, Union

import numpy as np
from numpy.typing import NDArray
from sklearn.pipeline import Pipeline as SklearnPipeline
from typing_extensions import Self

from ._arm import ContextType


class Pipeline(Generic[ContextType]):
    """Pipeline that bridges sklearn transformers with Bayesian learners.

    Implements the Learner protocol, allowing arbitrary input types to be
    transformed through sklearn transformers before reaching the final
    Bayesian learner. This enables the standard contextual bandit formulation
    where multiple arms share a single model with arm-specific features.

    Parameters
    ----------
    steps : list of tuples
        List of (name, transform/estimator) tuples. All steps except the
        last must be sklearn transformers. The last step must be a
        bayesianbandits learner (implementing partial_fit and sample).

    Examples
    --------
    **Standard Contextual Bandit with Shared Model** (most common pattern):

    >>> from functools import partial
    >>> from sklearn.preprocessing import FunctionTransformer, StandardScaler
    >>> from bayesianbandits import (
    ...     Arm, ContextualAgent, ThompsonSampling, NormalRegressor, Pipeline
    ... )
    >>>
    >>> # Single shared model for all arms (like LinUCB, Li et al. 2010)
    >>> shared_model = NormalRegressor(alpha=1.0, beta=1.0)
    >>>
    >>> # Product catalog with features
    >>> products = {
    ...     'iPhone': {'price': 999, 'category': 'electronics', 'brand_tier': 3},
    ...     'Shoes': {'price': 89, 'category': 'fashion', 'brand_tier': 2},
    ...     'Book': {'price': 15, 'category': 'media', 'brand_tier': 1},
    ... }
    >>>
    >>> # Create arms with product-specific features
    >>> arms = []
    >>> for product_id, features in products.items():
    ...     # Each arm augments context with its specific features
    ...     def add_product_features(X, prod_features):
    ...         # X shape: (n_users, n_user_features)
    ...         n_samples = len(X)
    ...         product_array = np.array([
    ...             prod_features['price'] / 1000,  # Normalize price
    ...             prod_features['brand_tier'] / 3,  # Normalize tier
    ...         ])
    ...         # Concatenate: [user_features, product_features]
    ...         return np.c_[X, np.tile(product_array, (n_samples, 1))]
    ...
    ...     pipeline = Pipeline([
    ...         ('add_product', FunctionTransformer(
    ...             partial(add_product_features, prod_features=features)
    ...         )),
    ...         ('scale', StandardScaler()),
    ...         ('model', shared_model)  # Shared across all products!
    ...     ])
    ...     arms.append(Arm(product_id, learner=pipeline))
    >>>
    >>> # The agent learns user preferences across all products jointly
    >>> agent = ContextualAgent(arms, ThompsonSampling())
    >>>
    >>> # User context: [age, income, days_since_last_purchase]
    >>> users = np.array([
    ...     [25, 50000, 7],
    ...     [45, 120000, 30],
    ... ])
    >>> recommendations = agent.pull(users)  # Personalized for each user

    **DataFrame Input with Feature Engineering**:

    >>> import pandas as pd
    >>> from sklearn.feature_extraction import FeatureHasher
    >>>
    >>> def enrich_user_context(df, item_features):
    ...     \"\"\"Add item features and compute interactions.\"\"\"
    ...     df = df.copy()
    ...
    ...     # Add item features
    ...     for key, value in item_features.items():
    ...         df[f'item_{key}'] = value
    ...
    ...     # Compute interactions
    ...     df['price_to_income_ratio'] = item_features['price'] / df['income']
    ...     df['is_premium_user_and_item'] = (
    ...         (df['user_tier'] == 'premium') & item_features['is_premium']
    ...     )
    ...     df['category_match'] = df['preferred_category'] == item_features['category']
    ...
    ...     return df
    >>>
    >>> # Shared model learns what features matter across all items
    >>> shared_model = NormalRegressor(alpha=0.1, beta=1.0)
    >>>
    >>> # Create item-specific pipelines
    >>> item_features = {'price': 49.99, 'category': 'electronics', 'is_premium': True}
    >>>
    >>> pipeline = Pipeline([
    ...     ('enrich', FunctionTransformer(
    ...         partial(enrich_user_context, item_features=item_features)
    ...     )),
    ...     ('hash', FeatureHasher(n_features=128, input_type='dict')),
    ...     ('model', shared_model)
    ... ])

    **Dict Input for A/B Testing Variants**:

    >>> from sklearn.feature_extraction import DictVectorizer
    >>>
    >>> # Testing different website layouts with shared learning
    >>> shared_model = NormalRegressor(alpha=1.0, beta=1.0)
    >>>
    >>> layouts = {
    ...     'layout_A': {'button_color': 'blue', 'header_size': 'large'},
    ...     'layout_B': {'button_color': 'green', 'header_size': 'medium'},
    ...     'layout_C': {'button_color': 'blue', 'header_size': 'medium'},
    ... }
    >>>
    >>> arms = []
    >>> for layout_id, layout_features in layouts.items():
    ...     def combine_context(user_dicts, layout):
    ...         \"\"\"Combine user and layout features.\"\"\"
    ...         combined = []
    ...         for user in user_dicts:
    ...             combined_dict = user.copy()
    ...             combined_dict.update({f'layout_{k}': v for k, v in layout.items()})
    ...             combined.append(combined_dict)
    ...         return combined
    ...
    ...     pipeline = Pipeline([
    ...         ('combine', FunctionTransformer(
    ...             partial(combine_context, layout=layout_features)
    ...         )),
    ...         ('vectorize', DictVectorizer()),
    ...         ('model', shared_model)  # Learn user preferences across layouts
    ...     ])
    ...     arms.append(Arm(layout_id, learner=pipeline))

    **Non-stationary Environments with Sparse Features**:

    >>> from scipy.sparse import csc_array, hstack
    >>>
    >>> # E-commerce with changing user preferences
    >>> shared_model = NormalRegressor(
    ...     alpha=1.0,
    ...     beta=1.0,
    ...     learning_rate=0.99,  # Gradual forgetting
    ...     sparse=True  # Efficient for high-dimensional features
    ... )
    >>>
    >>> def add_sparse_product_features(X, product_one_hot):
    ...     \"\"\"Add sparse product indicators to user features.\"\"\"
    ...     n_users = X.shape[0]
    ...     product_features = csc_array(
    ...         np.tile(product_one_hot.toarray(), (n_users, 1))
    ...     )
    ...     return hstack([X, product_features], format='csc')
    >>>
    >>> # Product one-hot encoding (e.g., 10000 products)
    >>> product_idx = 42
    >>> product_one_hot = csc_array(([1], ([0], [product_idx])), shape=(1, 10000))
    >>>
    >>> pipeline = Pipeline([
    ...     ('add_product', FunctionTransformer(
    ...         partial(add_sparse_product_features, product_one_hot=product_one_hot)
    ...     )),
    ...     ('model', shared_model)
    ... ])

    Notes
    -----
    The Pipeline class is essential for implementing standard contextual
    bandit algorithms from the literature, where a single model learns
    jointly across all arms. This provides several benefits:

    1. **Better cold-start**: New arms benefit from patterns learned on others
    2. **Parameter efficiency**: One model instead of K independent models
    3. **Standard formulation**: Matches papers like LinUCB, LinTS, etc.
    4. **Flexible features**: Easy to add arm-specific or interaction features

    See Also
    --------
    sklearn.pipeline.Pipeline : The underlying sklearn Pipeline
    sklearn.preprocessing.FunctionTransformer : For custom transformations
    bayesianbandits.Arm : Arms that use these pipelines as learners
    """

    def __init__(self, steps: List[Tuple[str, Any]]) -> None:
        if not steps:
            raise ValueError("Pipeline cannot be empty")

        *transformer_steps, (final_name, final_estimator) = steps

        # Validate final step has required methods
        if not (
            hasattr(final_estimator, "partial_fit")
            and hasattr(final_estimator, "sample")
            and hasattr(final_estimator, "predict")
            and hasattr(final_estimator, "decay")
        ):
            raise ValueError(
                f"Final step '{final_name}' must be a bayesianbandits learner "
                f"implementing partial_fit, sample, predict, and decay methods"
            )

        # Create sklearn pipeline for transformers
        if transformer_steps:
            self._transformers = SklearnPipeline(transformer_steps)
        else:
            self._transformers = None

        self._final_estimator = final_estimator
        self._final_name = final_name
        self._is_fitted = False

    @property
    def random_state(self) -> Union[np.random.Generator, int, None]:
        """Get random state from final estimator."""
        return getattr(self._final_estimator, "random_state", None)

    @random_state.setter
    def random_state(self, value: Union[np.random.Generator, int, None]) -> None:
        """Propagate random state to final estimator."""
        if hasattr(self._final_estimator, "random_state"):
            self._final_estimator.random_state = value

    def _apply_transformers(self, X: ContextType, fit: bool = False) -> Any:
        """Apply transformers to input data."""
        if self._transformers is None:
            return X

        if not self._is_fitted or fit:
            # First time or explicit fit request
            result = self._transformers.fit_transform(X)
            self._is_fitted = True
            return result
        else:
            return self._transformers.transform(X)

    def sample(self, X: ContextType, size: int = 1) -> NDArray[np.float64]:
        """Sample from the posterior predictive distribution.

        Parameters
        ----------
        X : ContextType
            Input data (DataFrame, dict, array, etc.)
        size : int, default=1
            Number of samples to draw

        Returns
        -------
        samples : NDArray[np.float64]
            Samples from the posterior
        """
        X_transformed = self._apply_transformers(X, fit=False)
        return self._final_estimator.sample(X_transformed, size)

    def partial_fit(
        self,
        X: ContextType,
        y: NDArray[np.float64],
        sample_weight: Optional[NDArray[np.float64]] = None,
    ) -> Self:
        """Update the learner with new data.

        Parameters
        ----------
        X : ContextType
            Input data (DataFrame, dict, array, etc.)
        y : NDArray[np.float64]
            Target values
        sample_weight : NDArray[np.float64], optional
            Sample weights

        Returns
        -------
        self : Pipeline
            Returns self for method chaining
        """
        X_transformed = self._apply_transformers(X, fit=True)
        self._final_estimator.partial_fit(X_transformed, y, sample_weight)
        return self

    def decay(self, X: ContextType, *, decay_rate: Optional[float] = None) -> None:
        """Decay the learner's parameters.

        Parameters
        ----------
        X : ContextType
            Input data (DataFrame, dict, array, etc.)
        decay_rate : float, optional
            Rate of decay
        """
        X_transformed = self._apply_transformers(X, fit=False)
        self._final_estimator.decay(X_transformed, decay_rate=decay_rate)

    def predict(self, X: ContextType) -> NDArray[np.float64]:
        """Predict expected values.

        Parameters
        ----------
        X : ContextType
            Input data (DataFrame, dict, array, etc.)

        Returns
        -------
        predictions : NDArray[np.float64]
            Expected values
        """
        X_transformed = self._apply_transformers(X, fit=False)
        return self._final_estimator.predict(X_transformed)

    def __repr__(self) -> str:
        """String representation."""
        steps_repr = [
            f"('{name}', {estimator.__class__.__name__})"
            for name, estimator in self.get_steps()
        ]
        return f"Pipeline({', '.join(steps_repr)})"

    def get_steps(self) -> List[Tuple[str, Any]]:
        """Get all steps including transformers and final estimator."""
        steps = []
        if self._transformers is not None:
            steps.extend(self._transformers.steps)
        steps.append((self._final_name, self._final_estimator))
        return steps
