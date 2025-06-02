"""Pipeline implementation for Bayesian bandits.

Enables sklearn transformers to work with Bayesian learners,
supporting workflows with DataFrames, dicts, and other input types.
"""

from typing import Any, Dict, Generic, List, Optional, Tuple, Union

import numpy as np
from numpy.typing import NDArray
from typing_extensions import Self

from ._arm import ContextType, Learner


class Pipeline(Generic[ContextType]):
    """Pipeline for composing pre-fitted transformers with Bayesian learners.

    Implements the Learner protocol, allowing arbitrary input types to be
    transformed through sklearn transformers before reaching the final
    Bayesian learner. All transformers must be either stateless or pre-fitted
    before use to ensure consistent feature transformations during online learning.

    As a performance optimization, policies will recognize if the final
    estimator in a Pipeline is shared across multiple arms, and will sample
    from them in a batched manner to reduce overhead.

    Parameters
    ----------
    steps : list of tuples
        List of (name, transform/estimator) tuples. All steps except the
        last must be sklearn transformers that are either stateless or
        pre-fitted. The last step must be a bayesianbandits learner.

    Raises
    ------
    ValueError
        If pipeline is empty or final step doesn't implement required methods
    RuntimeError
        If a stateful transformer is not fitted when first used

    Examples
    --------
    **Standard Usage with Pre-fitted Transformers**:

    >>> from sklearn.preprocessing import StandardScaler
    >>> from sklearn.feature_extraction import FeatureHasher
    >>> from bayesianbandits import Pipeline, NormalRegressor
    >>>
    >>> # Pre-fit transformers on historical data
    >>> historical_data = np.array([[1.0, 2.0], [3.0, 4.0]])
    >>> scaler = StandardScaler()
    >>> scaler.fit(historical_data)
    StandardScaler()
    >>>
    >>> pipeline = Pipeline([
    ...     ('scale', scaler),  # Pre-fitted
    ...     ('model', NormalRegressor(alpha=1.0, beta=1.0))
    ... ])
    >>>
    >>> # Use for online learning
    >>> X = np.array([[1.0, 2.0], [3.0, 4.0]])
    >>> y = np.array([1.0, 2.0])
    >>> pipeline.partial_fit(X, y)
    Pipeline([('scale', StandardScaler), ('model', NormalRegressor)])
    >>> predictions = pipeline.predict(X)

    **Complex Feature Engineering with Shared Model**:

    >>> from functools import partial
    >>> from sklearn.preprocessing import FunctionTransformer
    >>>
    >>> # Shared model across different products
    >>> shared_model = NormalRegressor(alpha=1.0, beta=1.0)
    >>>
    >>> # Create product-specific pipelines
    >>> def add_product_features(X, product_id, price):
    ...     n_samples = len(X)
    ...     return np.c_[X,
    ...                   np.full(n_samples, product_id),
    ...                   np.full(n_samples, price)]
    >>>
    >>> pipeline_a = Pipeline([
    ...     ('add_features', FunctionTransformer(
    ...         partial(add_product_features, product_id=1, price=9.99)
    ...     )),
    ...     ('model', shared_model)  # Shared!
    ... ])
    >>>
    >>> pipeline_b = Pipeline([
    ...     ('add_features', FunctionTransformer(
    ...         partial(add_product_features, product_id=2, price=14.99)
    ...     )),
    ...     ('model', shared_model)  # Same model!
    ... ])

    **Working with DataFrames and Dicts**:

    >>> import pandas as pd
    >>> from sklearn.feature_extraction import DictVectorizer
    >>>
    >>> # Pre-fit vectorizer on all possible feature combinations
    >>> vectorizer = DictVectorizer()
    >>> vectorizer.fit([{'user': 'A', 'item': 'X'},
    ...                 {'user': 'B', 'item': 'Y'}])
    DictVectorizer()
    >>>
    >>> # Pipeline for dict input
    >>> dict_pipeline = Pipeline([
    ...     ('vectorize', vectorizer),
    ...     ('model', NormalRegressor(alpha=1.0, beta=1.0, sparse=True))
    ... ])
    >>>
    >>> # Pipeline for DataFrame input
    >>> def df_to_dict(df):
    ...     return df.to_dict('records')
    >>>
    >>> df_pipeline = Pipeline([
    ...     ('to_dict', FunctionTransformer(df_to_dict)),
    ...     ('vectorize', vectorizer),
    ...     ('model', NormalRegressor(alpha=1.0, beta=1.0, sparse=True))
    ... ])

    **Integration with sklearn.pipeline.Pipeline**:

    >>> from sklearn.pipeline import Pipeline as SklearnPipeline
    >>> from sklearn.compose import ColumnTransformer
    >>> from bayesianbandits import BayesianGLM
    >>>
    >>> # Define numeric and categorical columns
    >>> numeric_columns = ['age', 'income']
    >>> categorical_columns = ['category_a', 'category_b']
    >>>
    >>> # Use sklearn.Pipeline for complex transformations
    >>> preprocessor = ColumnTransformer([
    ...     ('num', SklearnPipeline([...]), numeric_columns),
    ...     ('cat', SklearnPipeline([...]), categorical_columns)
    ... ])
    >>>
    >>> # Our Pipeline for the bandit learner
    >>> bandit_pipeline = Pipeline([
    ...     ('preprocess', preprocessor),
    ...     ('model', BayesianGLM(alpha=1.0, link='logit'))
    ... ])

    Notes
    -----
    This Pipeline is specifically designed for online learning with bandits:

    1. **No transformer fitting**: Transformers are never fitted during pipeline
       execution, ensuring consistent features across all updates.

    2. **Learner protocol**: Implements only the methods needed for bandits:
       partial_fit, predict, sample, decay, and random_state.

    3. **Type preservation**: Maintains input type through generic typing,
       supporting DataFrames, dicts, arrays, and custom types.

    4. **Composition friendly**: Can wrap sklearn.pipeline.Pipeline or other
       meta-estimators as transformers.

    For pure transformation pipelines without a learner, use sklearn.pipeline.Pipeline.
    This implementation is optimized specifically for the bandit learning use case.

    See Also
    --------
    sklearn.pipeline.Pipeline : For general transformation pipelines
    sklearn.preprocessing.FunctionTransformer : For custom stateless transforms
    bayesianbandits.Arm : Arms that use pipelines as learners
    """

    def __init__(self, steps: List[Tuple[str, Any]]) -> None:
        self.steps = steps
        self._validate_steps()

    def _validate_steps(self) -> None:
        """Validate the pipeline steps."""
        if not self.steps:
            raise ValueError("Pipeline cannot be empty")

        names, _ = zip(*self.steps)

        # Validate names
        if len(set(names)) != len(names):
            raise ValueError("Step names must be unique")

        # Validate final step is a learner
        final_name, final_estimator = self.steps[-1]
        required_methods = {"partial_fit", "sample", "predict", "decay"}
        missing = required_methods - set(dir(final_estimator))
        if missing:
            raise ValueError(
                f"Final step '{final_name}' must be a bayesianbandits learner "
                f"implementing {', '.join(sorted(missing))}"
            )

        # Store transformers and learner separately
        self._transformers = self.steps[:-1]
        self._learner: Learner[Any] = final_estimator

    @property
    def named_steps(self) -> Dict[str, Any]:
        """Access pipeline steps by name."""
        return dict(self.steps)

    def _transform(self, X: ContextType) -> Any:
        """Apply all transformers to input data.

        Transformers must be either stateless or pre-fitted.
        No fitting occurs during transformation.
        """
        result = X

        for name, transformer in self._transformers:
            try:
                result = transformer.transform(result)
            except Exception as e:
                # Provide helpful error for common case
                if hasattr(e, "args") and "not fitted" in str(e).lower():
                    raise RuntimeError(
                        f"Transformer '{name}' is not fitted. In online learning, "
                        f"all transformers must be either stateless or pre-fitted "
                        f"before use. Common stateless transformers include "
                        f"FunctionTransformer, FeatureHasher, and HashingVectorizer. "
                        f"Stateful transformers like StandardScaler must be fit on "
                        f"historical data before creating the pipeline."
                    ) from e
                raise

        return result

    def transform(self, X: ContextType) -> Any:
        """Apply all transformers, returning transformed data without final prediction.

        This enables batched prediction when multiple pipelines share the same
        final model.

        Returns
        -------
        X_transformed : Any
            The result of applying all transformers in sequence
        """
        return self._transform(X)

    @property
    def final_estimator(self) -> Learner[Any]:
        """Get the final estimator in the pipeline."""
        return self._learner

    # Learner protocol implementation
    @property
    def random_state(self) -> Union[np.random.Generator, int, None]:
        """Get random state from the learner."""
        return self._learner.random_state

    @random_state.setter
    def random_state(self, value: Union[np.random.Generator, int, None]) -> None:
        """Set random state on the learner."""
        self._learner.random_state = value

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
        X_transformed = self._transform(X)
        return self._learner.sample(X_transformed, size)

    def partial_fit(
        self,
        X: ContextType,
        y: NDArray[np.float64],
        sample_weight: Optional[NDArray[np.float64]] = None,
    ) -> Self:
        """Update the learner with new data.

        Note: Only the learner is updated. Transformers remain unchanged.

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
        X_transformed = self._transform(X)
        self._learner.partial_fit(X_transformed, y, sample_weight)
        return self

    def decay(self, X: ContextType, *, decay_rate: Optional[float] = None) -> None:
        """Decay the learner's parameters.

        Parameters
        ----------
        X : ContextType
            Input data (DataFrame, dict, array, etc.)
        decay_rate : float, optional
            Rate of decay. If None, uses the learner's default.
        """
        X_transformed = self._transform(X)
        self._learner.decay(X_transformed, decay_rate=decay_rate)

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
        X_transformed = self._transform(X)
        return self._learner.predict(X_transformed)

    def __repr__(self) -> str:
        """String representation."""
        steps_repr = [
            f"('{name}', {estimator.__class__.__name__})"
            for name, estimator in self.steps
        ]
        return f"Pipeline([{', '.join(steps_repr)}])"

    def __len__(self) -> int:
        """Number of steps in the pipeline."""
        return len(self.steps)

    def __getitem__(self, ind: Union[int, str]) -> Any:
        """Get a step by index or name.

        Parameters
        ----------
        ind : int or str
            If int, returns the (name, estimator) tuple at that position.
            If str, returns the estimator with that name.

        Returns
        -------
        step : tuple or estimator
            The requested step

        Raises
        ------
        IndexError
            If integer index is out of range
        KeyError
            If string name is not found
        """
        if isinstance(ind, str):
            return self.named_steps[ind]
        return self.steps[ind]
