"""Learner-implementing Pipeline for Bayesian bandits.

This module implements sklearn-compatible pipelines that implement the Learner
protocol. These pipelines can be used as learners within Arms, enabling sklearn
preprocessing to be applied to enriched features (e.g., after ArmFeaturizer
transforms contexts with arm-specific information).
"""

from typing import Any, Dict, Generic, List, Optional, Tuple, TypeVar, Union, cast

import numpy as np
from numpy.typing import NDArray
from typing_extensions import Self

from .._arm import Learner

X_contra = TypeVar("X_contra", contravariant=True)


class LearnerPipeline(Generic[X_contra]):
    """Pipeline that implements the Learner protocol with generic input type.

    Enables sklearn transformers to work with Bayesian learners by implementing
    the standard Learner interface (partial_fit, sample, predict, decay). This
    pipeline is designed to be used as a learner within Arms, particularly for
    LipschitzContextualAgent where enriched features (context + arm info) need
    sklearn preprocessing before reaching the final Bayesian learner.

    Type Parameters
    ---------------
    X_contra : type
        The contravariant input type that this pipeline accepts. Common types
        include np.ndarray, list, pd.DataFrame, or Any.

    Parameters
    ----------
    steps : List[Tuple[str, Any]]
        List of (name, transformer) tuples for preprocessing steps. All
        transformers must be either stateless or already fitted on historical data.
    learner : Any
        Final learner that implements the Learner protocol (partial_fit, sample,
        predict, decay methods).

    Examples
    --------
    **Post-Featurization Preprocessing with LipschitzContextualAgent**:

    >>> import numpy as np
    >>> from sklearn.preprocessing import StandardScaler
    >>> from sklearn.decomposition import PCA
    >>> from bayesianbandits import (
    ...     LipschitzContextualAgent, ThompsonSampling, NormalRegressor,
    ...     ArmColumnFeaturizer, Arm
    ... )
    >>> from bayesianbandits.pipelines import LearnerPipeline
    >>>
    >>> # After ArmFeaturizer enriches context with arm features,
    >>> # we want to standardize and reduce dimensionality
    >>> # Pre-fit transformers on historical data
    >>> scaler = StandardScaler()
    >>> pca = PCA(n_components=2)
    >>> # Fit on dummy historical enriched features (context + arm_features)
    >>> dummy_features = np.random.randn(100, 3)  # 2 context + 1 arm feature
    >>> _ = scaler.fit(dummy_features)
    >>> _ = pca.fit(scaler.transform(dummy_features))
    >>>
    >>> learner_pipeline = LearnerPipeline[np.ndarray](
    ...     steps=[
    ...         ('standardize', scaler),  # Pre-fitted scaler
    ...         ('reduce_dims', pca),     # Pre-fitted PCA
    ...     ],
    ...     learner=NormalRegressor(alpha=1.0, beta=1.0)  # Final learner
    ... )
    >>>
    >>> # Use numeric arm featurizer to avoid string conversion issues
    >>> from bayesianbandits.featurizers import FunctionArmFeaturizer
    >>> def numeric_featurizer(X, action_tokens):
    ...     n_contexts, n_features = X.shape
    ...     n_arms = len(action_tokens)
    ...     result = np.zeros((n_contexts, n_features + 1, n_arms))
    ...     for i, token in enumerate(action_tokens):
    ...         result[:, :-1, i] = X  # Original features
    ...         result[:, -1, i] = i   # Numeric arm ID
    ...     return result
    >>>
    >>> # Use in LipschitzContextualAgent - all arms share this learner
    >>> arms = [Arm(f'product_{i}', learner=learner_pipeline) for i in range(5)]
    >>> agent = LipschitzContextualAgent(
    ...     arms=arms,
    ...     policy=ThompsonSampling(),
    ...     arm_featurizer=FunctionArmFeaturizer(numeric_featurizer),
    ...     learner=learner_pipeline  # Shared learner with preprocessing
    ... )
    >>>
    >>> # Data flow:
    >>> # Raw context -> ArmFeaturizer -> [context + arm_features]
    >>> #            -> StandardScaler -> PCA (2D) -> NormalRegressor
    >>> user_context = np.array([[25, 50000]])  # [age, income]
    >>> recommendations = agent.pull(user_context)
    >>> len(recommendations)  # Should return number of contexts
    1

    **High-Dimensional Features with Dimensionality Reduction**:

    >>> from sklearn.decomposition import PCA
    >>>
    >>> # After arm featurization, we have high-dimensional features
    >>> # Reduce dimensionality before learning for efficiency
    >>> pca = PCA(n_components=10)
    >>> _ = pca.fit(np.random.randn(100, 20))  # Pre-fit on dummy historical data
    >>> high_dim_learner = LearnerPipeline[np.ndarray](
    ...     steps=[('reduce_dims', pca)],  # Pre-fitted PCA
    ...     learner=NormalRegressor(alpha=0.1, beta=1.0)
    ... )
    >>>
    >>> # Useful when ArmFeaturizer creates high-dimensional encodings
    >>> # e.g., interaction features between context and many arms

    **Different Input Types with Type Parameters**:

    >>> from sklearn.feature_extraction import DictVectorizer
    >>> from typing import List, Dict
    >>>
    >>> # Pipeline for list of dicts input - pre-fit vectorizer
    >>> vectorizer = DictVectorizer()
    >>> _ = vectorizer.fit([{'a': 1, 'b': 2}, {'a': 3, 'c': 4}])  # Fit on dummy data
    >>> dict_pipeline = LearnerPipeline[List[Dict[str, float]]](
    ...     steps=[('vectorize', vectorizer)],  # Pre-fitted vectorizer
    ...     learner=NormalRegressor(alpha=1.0, beta=1.0)
    ... )
    >>>
    >>> # Pipeline for numpy arrays - pre-fit scaler
    >>> scaler = StandardScaler()
    >>> _ = scaler.fit(np.random.randn(50, 5))  # Fit on dummy historical data
    >>> array_pipeline = LearnerPipeline[np.ndarray](
    ...     steps=[('scale', scaler)],  # Pre-fitted scaler
    ...     learner=NormalRegressor(alpha=1.0, beta=1.0)
    ... )

    **Stateless Transformations**:

    >>> from sklearn.preprocessing import FunctionTransformer
    >>>
    >>> def log_transform(X):
    ...     # Apply log transformation to certain features
    ...     X_log = X.copy()
    ...     X_log[:, 0] = np.log1p(X_log[:, 0])  # Log-transform first feature
    ...     return X_log
    >>>
    >>> # Stateless transformations don't need fitting
    >>> stateless_learner = LearnerPipeline[np.ndarray](
    ...     steps=[('log_transform', FunctionTransformer(log_transform))],
    ...     learner=NormalRegressor(alpha=1.0, beta=1.0)
    ... )  # No fitting needed for stateless transformers

    Notes
    -----
    The LearnerPipeline is specifically designed for the workflow where:

    1. **Raw context** is processed by ArmFeaturizer to create enriched features
    2. **Enriched features** need sklearn preprocessing (standardization, PCA, etc.)
    3. **Preprocessed features** are used by the final Bayesian learner

    This enables sophisticated feature engineering on the post-featurization
    feature space, which is not possible with agent-level preprocessing alone.

    **Important**: All sklearn transformers must be either stateless (like
    FunctionTransformer) or already fitted on historical data. No fitting
    occurs during online operation to maintain performance and predictability.

    The pipeline correctly implements all methods of the Learner protocol:
    - `partial_fit`: Transforms input and updates final learner
    - `sample`: Transforms input and samples from final learner
    - `predict`: Transforms input and predicts with final learner
    - `decay`: Transforms input and decays final learner
    - `random_state`: Delegates to final learner

    See Also
    --------
    bayesianbandits.pipelines.AgentPipeline : For agent-level preprocessing
    bayesianbandits.LipschitzContextualAgent : Uses shared learners efficiently
    sklearn.pipeline.Pipeline : Inspiration for the interface
    """

    def __init__(self, steps: List[Tuple[str, Any]], learner: Learner[Any]) -> None:
        # Validate steps (can be empty)
        if steps:
            names, _ = zip(*steps)
            # Validate names are unique
            if len(set(names)) != len(names):
                raise ValueError("Step names must be unique")

        # Validate learner has required Learner methods
        required_methods = ["partial_fit", "sample", "predict", "decay"]
        missing_methods = [
            method for method in required_methods if not hasattr(learner, method)
        ]
        if missing_methods:
            raise ValueError(
                f"Learner must implement the Learner protocol. "
                f"Missing methods: {missing_methods}"
            )

        # Store transformer steps and learner separately
        self.steps = steps  # Transformer steps only
        self._learner: Learner[Any] = learner

    @property
    def random_state(self) -> Union[np.random.Generator, int, None]:
        """Get random state from learner."""
        return getattr(self._learner, "random_state", None)

    @random_state.setter
    def random_state(self, value: Union[np.random.Generator, int, None]) -> None:
        """Propagate random state to learner."""
        if hasattr(self._learner, "random_state"):
            self._learner.random_state = value

    def _apply_transformers(self, X: X_contra) -> Any:
        """Apply transformers to input data.

        Transformers must be either stateless or already fitted.
        """
        if not self.steps:
            return X

        # Apply each transformer in sequence
        result = cast(Any, X)  # Cast to Any for sklearn calls
        for name, transformer in self.steps:
            try:
                result = transformer.transform(result)
            except Exception as e:
                # Provide helpful error for common case
                if hasattr(e, "args") and "not fitted" in str(e).lower():
                    raise RuntimeError(
                        f"Transformer '{name}' is not fitted. In online learning, "
                        f"all transformers must be either stateless or pre-fitted "
                        f"before use. Common stateless transformers include "
                        f"FunctionTransformer. Stateful transformers like "
                        f"StandardScaler must be fit on historical data before "
                        f"creating the pipeline."
                    ) from e
                raise
        return result

    def sample(self, X: X_contra, size: int = 1) -> NDArray[np.float64]:
        """Sample from the posterior predictive distribution.

        Parameters
        ----------
        X : X_contra
            Input data (enriched features from ArmFeaturizer)
        size : int, default=1
            Number of samples to draw

        Returns
        -------
        samples : NDArray[np.float64]
            Samples from the posterior
        """
        X_transformed = self._apply_transformers(X)
        return self._learner.sample(X_transformed, size)

    def partial_fit(
        self,
        X: X_contra,
        y: NDArray[np.float64],
        sample_weight: Optional[NDArray[np.float64]] = None,
    ) -> Self:
        """Update the learner with new data.

        Parameters
        ----------
        X : X_contra
            Input data (enriched features from ArmFeaturizer)
        y : NDArray[np.float64]
            Target values
        sample_weight : NDArray[np.float64], optional
            Sample weights

        Returns
        -------
        self : LearnerPipeline
            Returns self for method chaining
        """
        X_transformed = self._apply_transformers(X)
        self._learner.partial_fit(X_transformed, y, sample_weight)
        return self

    def decay(self, X: X_contra, *, decay_rate: Optional[float] = None) -> None:
        """Decay the learner's parameters.

        Parameters
        ----------
        X : X_contra
            Input data (enriched features from ArmFeaturizer)
        decay_rate : float, optional
            Rate of decay
        """
        X_transformed = self._apply_transformers(X)
        self._learner.decay(X_transformed, decay_rate=decay_rate)

    def predict(self, X: X_contra) -> NDArray[np.float64]:
        """Predict expected values.

        Parameters
        ----------
        X : X_contra
            Input data (enriched features from ArmFeaturizer)

        Returns
        -------
        predictions : NDArray[np.float64]
            Expected values
        """
        X_transformed = self._apply_transformers(X)
        return self._learner.predict(X_transformed)

    @property
    def named_steps(self) -> Dict[str, Any]:
        """Access pipeline transformer steps by name."""
        return dict(self.steps)

    @property
    def learner(self) -> Learner[Any]:
        """Access the final learner."""
        return self._learner

    def __repr__(self) -> str:
        """String representation."""
        steps_repr = [
            f"('{name}', {transformer.__class__.__name__})"
            for name, transformer in self.steps
        ]
        learner_repr = f"learner={self._learner.__class__.__name__}"
        if steps_repr:
            return f"LearnerPipeline(steps=[{', '.join(steps_repr)}], {learner_repr})"
        else:
            return f"LearnerPipeline(steps=[], {learner_repr})"

    def __len__(self) -> int:
        """Number of steps in the pipeline."""
        return len(self.steps)

    def __getitem__(self, ind: Union[int, str]) -> Any:
        """Get a step by index or name."""
        if isinstance(ind, str):
            return self.named_steps[ind]
        return self.steps[ind]
