"""Tests for simplified Pipeline implementation."""

from __future__ import annotations

from functools import partial

import numpy as np
import pytest
from sklearn.base import BaseEstimator, TransformerMixin
from sklearn.compose import ColumnTransformer
from sklearn.feature_extraction import DictVectorizer, FeatureHasher
from sklearn.pipeline import FeatureUnion
from sklearn.pipeline import Pipeline as SklearnPipeline
from sklearn.preprocessing import (
    FunctionTransformer,
    PolynomialFeatures,
    StandardScaler,
)

from bayesianbandits import (
    Arm,
    BayesianGLM,
    ContextualAgent,
    GammaRegressor,
    NormalInverseGammaRegressor,
    NormalRegressor,
    ThompsonSampling,
)
from bayesianbandits.pipeline import Pipeline


class MockTransformer(BaseEstimator, TransformerMixin):
    """Mock transformer for testing."""

    def __init__(self, fitted=False):
        self.fitted = fitted

    def fit(self, X, y=None): ...

    def transform(self, X):
        if not self.fitted:
            raise RuntimeError("Transformer not fitted")


class TestPipelineConstruction:
    """Test Pipeline construction and validation."""

    def test_basic_construction(self):
        """Test basic pipeline construction."""
        pipeline = Pipeline(
            [
                ("double", FunctionTransformer(lambda x: x * 2)),
                ("model", NormalRegressor(alpha=1.0, beta=1.0)),
            ]
        )

        assert len(pipeline) == 2
        assert hasattr(pipeline, "_learner")
        assert hasattr(pipeline, "_transformers")
        assert len(pipeline._transformers) == 1

    def test_single_step_pipeline(self):
        """Test pipeline with only a learner."""
        pipeline = Pipeline([("model", NormalRegressor(alpha=1.0, beta=1.0))])

        assert len(pipeline) == 1
        assert len(pipeline._transformers) == 0
        assert pipeline._learner is not None

    def test_empty_pipeline_error(self):
        """Test empty pipeline raises error."""
        with pytest.raises(ValueError, match="Pipeline cannot be empty"):
            Pipeline([])

    def test_invalid_final_step_error(self):
        """Test invalid final step raises error."""
        with pytest.raises(ValueError, match="must be a bayesianbandits learner"):
            Pipeline(
                [("transform", FunctionTransformer()), ("invalid", StandardScaler())]
            )

    def test_duplicate_names_error(self):
        """Test duplicate step names raise error."""
        with pytest.raises(ValueError, match="Step names must be unique"):
            Pipeline(
                [
                    ("transform", FunctionTransformer()),
                    ("transform", FunctionTransformer()),  # Duplicate!
                    ("model", NormalRegressor(alpha=1.0, beta=1.0)),
                ]
            )

    def test_named_steps_access(self):
        """Test accessing steps by name."""
        transform1 = FunctionTransformer(lambda x: x * 2)
        transform2 = FunctionTransformer(lambda x: x + 1)
        model = NormalRegressor(alpha=1.0, beta=1.0)

        pipeline = Pipeline(
            [("double", transform1), ("add_one", transform2), ("model", model)]
        )

        # Test named_steps property
        assert pipeline.named_steps["double"] is transform1
        assert pipeline.named_steps["add_one"] is transform2
        assert pipeline.named_steps["model"] is model

        # Test __getitem__ with string
        assert pipeline["double"] is transform1
        assert pipeline["model"] is model

        # Test __getitem__ with int
        assert pipeline[0] == ("double", transform1)
        assert pipeline[2] == ("model", model)

        # Test invalid access
        with pytest.raises(KeyError):
            _ = pipeline["invalid"]

        with pytest.raises(IndexError):
            _ = pipeline[10]


class TestTransformers:
    """Test transformer behavior."""

    def test_stateless_transformers(self):
        """Test pipeline with stateless transformers."""
        pipeline = Pipeline(
            [
                ("double", FunctionTransformer(lambda x: x * 2)),
                ("square", FunctionTransformer(lambda x: x**2)),
                ("model", NormalRegressor(alpha=1.0, beta=1.0)),
            ]
        )

        X = np.array([[1.0], [2.0], [3.0]])
        y = np.array([1.0, 4.0, 9.0])

        # Should work without any pre-fitting
        pipeline.partial_fit(X, y)
        predictions = pipeline.predict(X)
        assert predictions.shape == (3,)

        # Check transformations were applied
        # X -> X*2 -> (X*2)^2 = 4*X^2
        # So for X=2, transformed should be 16
        assert hasattr(pipeline._learner, "coef_")

    def test_pre_fitted_transformers(self):
        """Test pipeline with pre-fitted transformers."""
        # Pre-fit scaler
        scaler = StandardScaler()
        historical_data = np.random.randn(100, 2)
        scaler.fit(historical_data)

        # Pre-fit polynomial features
        poly = PolynomialFeatures(degree=2, include_bias=False)
        poly.fit(scaler.transform(historical_data))

        pipeline = Pipeline(
            [
                ("scale", scaler),
                ("poly", poly),
                ("model", NormalRegressor(alpha=1.0, beta=1.0)),
            ]
        )

        X = np.array([[1.0, 2.0], [3.0, 4.0], [5.0, 6.0]])
        y = np.array([1.0, 2.0, 3.0])

        pipeline.partial_fit(X, y)
        predictions = pipeline.predict(X)
        assert predictions.shape == (3,)

    def test_not_fitted_transformer_error(self):
        """Test helpful error when transformer not fitted."""
        pipeline = Pipeline(
            [
                ("scale", StandardScaler()),  # Not fitted!
                ("model", NormalRegressor(alpha=1.0, beta=1.0)),
            ]
        )

        X = np.array([[1.0], [2.0]])
        y = np.array([1.0, 2.0])

        with pytest.raises(RuntimeError) as exc_info:
            pipeline.partial_fit(X, y)

        assert "not fitted" in str(exc_info.value)
        assert "StandardScaler" in str(exc_info.value)
        assert "FunctionTransformer" in str(exc_info.value)

    def test_custom_transformer_error(self):
        """Test error handling with custom transformer."""
        pipeline = Pipeline(
            [
                ("mock", MockTransformer(fitted=False)),
                ("model", NormalRegressor(alpha=1.0, beta=1.0)),
            ]
        )

        X = np.array([[1.0], [2.0]])

        with pytest.raises(RuntimeError) as exc_info:
            pipeline.predict(X)

        assert "not fitted" in str(exc_info.value)

    def test_feature_hasher_stateless(self):
        """Test FeatureHasher works without fitting."""
        pipeline = Pipeline(
            [
                ("hash", FeatureHasher(n_features=100, input_type="dict")),
                ("model", NormalRegressor(alpha=1.0, beta=1.0, sparse=True)),
            ]
        )

        X = [{"user": "A", "item": 1}, {"user": "B", "item": 2}]
        y = np.array([1.0, 2.0])

        # Should work without fitting
        pipeline.partial_fit(X, y)
        predictions = pipeline.predict(X)
        assert predictions.shape == (2,)


class TestLearnerProtocol:
    """Test implementation of Learner protocol."""

    def test_partial_fit(self):
        """Test partial_fit updates the model."""
        pipeline = Pipeline(
            [
                ("identity", FunctionTransformer()),
                ("model", NormalRegressor(alpha=1.0, beta=1.0)),
            ]
        )

        # First batch
        X1 = np.array([[1.0], [2.0]])
        y1 = np.array([1.0, 2.0])
        pipeline.partial_fit(X1, y1)
        pred1 = pipeline.predict(X1)

        # Second batch
        X2 = np.array([[3.0], [4.0]])
        y2 = np.array([3.0, 4.0])
        pipeline.partial_fit(X2, y2)
        pred2 = pipeline.predict(X1)

        # Predictions should change
        assert not np.allclose(pred1, pred2)

    def test_sample_method(self):
        """Test posterior sampling."""
        pipeline = Pipeline(
            [
                ("scale", FunctionTransformer(lambda x: x / 10)),
                ("model", NormalInverseGammaRegressor()),
            ]
        )

        X = np.array([[10.0], [20.0], [30.0]])
        y = np.array([1.0, 2.0, 3.0])

        pipeline.partial_fit(X, y)

        # Test different sample sizes
        samples1 = pipeline.sample(X, size=1)
        assert samples1.shape == (1, 3)

        samples10 = pipeline.sample(X, size=10)
        assert samples10.shape == (10, 3)

        # Samples should be stochastic
        samples_a = pipeline.sample(X, size=5)
        samples_b = pipeline.sample(X, size=5)
        assert not np.allclose(samples_a, samples_b)

    def test_sample_weight_propagation(self):
        """Test sample weights are propagated."""
        pipeline = Pipeline(
            [
                ("identity", FunctionTransformer()),
                ("model", NormalRegressor(alpha=1.0, beta=1.0)),
            ]
        )

        X = np.array([[1.0], [2.0], [10.0]])  # Outlier
        y = np.array([1.0, 2.0, 1.0])

        # Fit with low weight on outlier
        weights = np.array([1.0, 1.0, 0.0001])
        pipeline.partial_fit(X, y, sample_weight=weights)

        # Prediction should be closer to linear trend
        pred = pipeline.predict(np.array([[5.0]]))
        assert 3.0 < pred[0] < 6.0  # Not pulled toward outlier

    def test_decay_method(self):
        """Test decay propagation."""
        model = NormalRegressor(alpha=1.0, beta=1.0, learning_rate=0.9)
        pipeline = Pipeline([("identity", FunctionTransformer()), ("model", model)])

        X = np.array([[1.0], [2.0]])
        y = np.array([1.0, 2.0])

        pipeline.partial_fit(X, y)
        initial_precision = model.cov_inv_.copy()

        # Apply decay
        pipeline.decay(X, decay_rate=0.5)

        # Precision should decrease
        assert np.all(model.cov_inv_ < initial_precision)

    def test_random_state_property(self):
        """Test random_state getter and setter."""
        model = NormalRegressor(alpha=1.0, beta=1.0, random_state=42)
        pipeline = Pipeline([("transform", FunctionTransformer()), ("model", model)])

        # Test getter
        assert pipeline.random_state == 42

        # Test setter
        pipeline.random_state = 123
        assert pipeline.random_state == 123
        assert model.random_state == 123

        # Test with Generator
        rng = np.random.default_rng(456)
        pipeline.random_state = rng
        assert pipeline.random_state is rng
        assert model.random_state is rng


class TestComplexInputTypes:
    """Test Pipeline with various input types."""

    def test_dict_input(self):
        """Test pipeline with dict input."""
        # Pre-fit vectorizer
        vectorizer = DictVectorizer()
        historical_dicts = [
            {"user": "A", "item": "X"},
            {"user": "A", "item": "Y"},
            {"user": "B", "item": "X"},
            {"user": "B", "item": "Y"},
        ]
        vectorizer.fit(historical_dicts)

        pipeline = Pipeline(
            [
                ("vectorize", vectorizer),
                ("model", NormalRegressor(alpha=1.0, beta=1.0, sparse=True)),
            ]
        )

        X = [{"user": "A", "item": "X"}, {"user": "B", "item": "Y"}]
        y = np.array([1.0, 2.0])

        pipeline.partial_fit(X, y)
        predictions = pipeline.predict(X)
        assert predictions.shape == (2,)

    def test_dataframe_input(self):
        """Test pipeline with DataFrame input."""
        pd = pytest.importorskip("pandas")

        def extract_features(df):
            """Extract numeric columns as array."""
            return df[["num1", "num2"]].values

        pipeline = Pipeline(
            [
                ("extract", FunctionTransformer(extract_features)),
                ("model", NormalRegressor(alpha=1.0, beta=1.0)),
            ]
        )

        df = pd.DataFrame(
            {
                "num1": [1.0, 2.0, 3.0],
                "num2": [4.0, 5.0, 6.0],
                "text": ["a", "b", "c"],  # Ignored
            }
        )
        y = np.array([1.0, 2.0, 3.0])

        pipeline.partial_fit(df, y)
        predictions = pipeline.predict(df)
        assert predictions.shape == (3,)

    def test_sparse_input(self):
        """Test pipeline with sparse matrices."""
        from scipy.sparse import csr_matrix

        # Convert sparse to CSC format (required by our models)
        def to_csc(X):
            return X.tocsc() if hasattr(X, "tocsc") else X

        pipeline = Pipeline(
            [
                ("to_csc", FunctionTransformer(to_csc)),
                ("model", NormalRegressor(alpha=1.0, beta=1.0, sparse=True)),
            ]
        )

        X = csr_matrix([[1, 0, 0], [0, 1, 0], [0, 0, 1]])
        y = np.array([1.0, 2.0, 3.0])

        pipeline.partial_fit(X, y)
        predictions = pipeline.predict(X)
        assert predictions.shape == (3,)


class TestIntegrationScenarios:
    """Test real-world integration scenarios."""

    def test_sklearn_pipeline_as_transformer(self):
        """Test using sklearn.Pipeline as a transformer."""
        # Pre-fit transformers
        scaler = StandardScaler()
        poly = PolynomialFeatures(degree=2, include_bias=False)

        historical = np.random.randn(100, 2)
        scaler.fit(historical)
        poly.fit(scaler.transform(historical))

        # Create sklearn pipeline
        sklearn_pipe = SklearnPipeline([("scale", scaler), ("poly", poly)])

        # Use in our pipeline
        pipeline = Pipeline(
            [
                ("preprocess", sklearn_pipe),
                ("model", NormalRegressor(alpha=1.0, beta=1.0)),
            ]
        )

        X = np.random.randn(10, 2)
        y = np.random.randn(10)

        pipeline.partial_fit(X, y)
        predictions = pipeline.predict(X)
        assert predictions.shape == (10,)

    def test_column_transformer_integration(self):
        """Test integration with ColumnTransformer."""
        # Pre-fit transformers
        num_scaler = StandardScaler()
        num_scaler.fit(np.random.randn(100, 2))

        # Create preprocessing with sklearn
        preprocessor = ColumnTransformer(
            [
                ("num", num_scaler, [0, 1]),
                ("cat", FeatureHasher(n_features=10, input_type="string"), [2]),
            ]
        )

        # Fit on historical data
        historical = [[1.0, 2.0, "cat"], [3.0, 4.0, "dog"], [5.0, 6.0, "cat"]]
        preprocessor.fit(historical)  # type: ignore

        # Our pipeline uses pre-fitted preprocessor
        pipeline = Pipeline(
            [
                ("preprocess", preprocessor),
                ("model", NormalRegressor(alpha=1.0, beta=1.0, sparse=True)),
            ]
        )

        # Test with mixed data
        import pandas as pd

        df = pd.DataFrame(
            {
                "num1": [1.0, 2.0, 3.0],
                "num2": [4.0, 5.0, 6.0],
                "cat": ["cat", "dog", "bird"],
            }
        )
        y = np.array([1.0, 2.0, 3.0])

        pipeline.partial_fit(df.values, y)
        predictions = pipeline.predict(df.values)
        assert predictions.shape == (3,)

    def test_feature_union_integration(self):
        """Test integration with FeatureUnion."""
        # Create feature extractors
        log_transform = FunctionTransformer(np.log1p)
        sqrt_transform = FunctionTransformer(np.sqrt)

        # Pre-fit polynomial
        poly = PolynomialFeatures(degree=2, include_bias=False)
        poly.fit(np.random.rand(100, 1))

        # Combine with FeatureUnion
        features = FeatureUnion(
            [("log", log_transform), ("sqrt", sqrt_transform), ("poly", poly)]
        )

        # Fit on historical data
        features.fit(np.random.rand(100, 1))

        # Use in pipeline
        pipeline = Pipeline(
            [("features", features), ("model", NormalRegressor(alpha=1.0, beta=1.0))]
        )

        X = np.array([[1.0], [4.0], [9.0]])
        y = np.array([1.0, 2.0, 3.0])

        pipeline.partial_fit(X, y)
        predictions = pipeline.predict(X)
        assert predictions.shape == (3,)

    def test_multi_armed_bandit_scenario(self):
        """Test realistic bandit scenario with shared model."""
        # Pre-fit shared scaler
        scaler = StandardScaler()
        scaler.fit(np.random.randn(1000, 5))

        # Shared model
        shared_model = BayesianGLM(alpha=1.0, link="logit")

        # Products with features
        products = {
            "A": {"price": 10.0, "quality": 0.8},
            "B": {"price": 15.0, "quality": 0.9},
            "C": {"price": 5.0, "quality": 0.6},
        }

        # Create arms
        arms = []
        for product_id, features in products.items():

            def add_product_context(X, prod_features):
                n_users = len(X)
                product_array = np.array(
                    [
                        prod_features["price"] / 20.0,  # Normalize
                        prod_features["quality"],
                    ]
                )
                return np.c_[X, np.tile(product_array, (n_users, 1))]

            pipeline = Pipeline(
                [
                    (
                        "add_product",
                        FunctionTransformer(
                            partial(add_product_context, prod_features=features)
                        ),
                    ),
                    ("scale", scaler),
                    ("model", shared_model),
                ]
            )

            arms.append(Arm(product_id, learner=pipeline))

        # Create agent
        agent = ContextualAgent(arms, ThompsonSampling())

        # Simulate interactions
        user_contexts = np.random.randn(10, 3)
        actions = agent.pull(user_contexts)
        assert len(actions) == 10

        # Update with binary rewards
        rewards = np.random.binomial(1, 0.7, size=10).astype(float)
        agent.update(user_contexts, rewards)

        # Shared model should be updated
        assert hasattr(shared_model, "coef_")

    def test_custom_transformer_chain(self):
        """Test complex custom transformation chain."""

        # Chain of transformations
        def add_interactions(X):
            """Add interaction features."""
            # X has shape (n, 2)
            return np.c_[X, X[:, 0] * X[:, 1]]

        def add_ratios(X):
            """Add ratio features."""
            # X now has shape (n, 3)
            return np.c_[X, X[:, 0] / (X[:, 1] + 1e-8)]

        def add_logs(X):
            """Add log features."""
            # X now has shape (n, 4)
            return np.c_[X, np.log1p(np.abs(X))]

        pipeline = Pipeline(
            [
                ("interactions", FunctionTransformer(add_interactions)),
                ("ratios", FunctionTransformer(add_ratios)),
                ("logs", FunctionTransformer(add_logs)),
                ("model", NormalRegressor(alpha=1.0, beta=1.0)),
            ]
        )

        X = np.array([[1.0, 2.0], [3.0, 4.0], [5.0, 6.0]])
        y = np.array([1.0, 2.0, 3.0])

        pipeline.partial_fit(X, y)
        predictions = pipeline.predict(X)
        assert predictions.shape == (3,)


class TestAllEstimators:
    """Test all bayesianbandits estimators work as final step."""

    def test_all_estimators(self):
        """Test each estimator type."""
        estimators = [
            (
                NormalRegressor(alpha=1.0, beta=1.0),
                np.array([[1.0], [2.0]]),
                np.array([1.0, 2.0]),
            ),
            (
                NormalInverseGammaRegressor(),
                np.array([[1.0], [2.0]]),
                np.array([1.0, 2.0]),
            ),
            (
                BayesianGLM(alpha=1.0, link="logit"),
                np.array([[1.0], [2.0]]),
                np.array([0.0, 1.0]),
            ),
            (
                BayesianGLM(alpha=1.0, link="log"),
                np.array([[1.0], [2.0]]),
                np.array([1.0, 2.0]),
            ),
            (
                GammaRegressor(alpha=1.0, beta=1.0),
                np.array([[1], [1], [1]]),
                np.array([1, 2, 3]),
            ),
        ]

        for estimator, X, y in estimators:
            pipeline = Pipeline(
                [("identity", FunctionTransformer()), ("model", estimator)]
            )

            # Test all methods
            pipeline.partial_fit(X, y)
            predictions = pipeline.predict(X)
            assert predictions.shape == (len(X),)

            samples = pipeline.sample(X, size=5)
            assert samples.shape == (5, len(X))

            pipeline.decay(X, decay_rate=0.9)


class TestErrorHandling:
    """Test error handling and edge cases."""

    def test_transform_error_propagation(self):
        """Test that transformer errors are propagated."""

        def failing_transform(X):
            raise ValueError("Custom error message")

        pipeline = Pipeline(
            [
                ("fail", FunctionTransformer(failing_transform)),
                ("model", NormalRegressor(alpha=1.0, beta=1.0)),
            ]
        )

        with pytest.raises(ValueError, match="Custom error message"):
            pipeline.predict(np.array([[1.0]]))

    def test_repr(self):
        """Test string representation."""
        pipeline = Pipeline(
            [
                ("transform", FunctionTransformer()),
                ("model", NormalRegressor(alpha=1.0, beta=1.0)),
            ]
        )

        repr_str = repr(pipeline)
        assert "Pipeline" in repr_str
        assert "FunctionTransformer" in repr_str
        assert "NormalRegressor" in repr_str

    def test_len(self):
        """Test __len__ method."""
        pipeline = Pipeline(
            [
                ("a", FunctionTransformer()),
                ("b", FunctionTransformer()),
                ("model", NormalRegressor(alpha=1.0, beta=1.0)),
            ]
        )
        assert len(pipeline) == 3
