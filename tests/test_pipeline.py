"""Tests for Pipeline implementation."""

from __future__ import annotations

from functools import partial
from typing import Dict, List

import numpy as np
import pytest
from numpy.typing import NDArray
from scipy.sparse import csc_array
from sklearn.feature_extraction import DictVectorizer
from sklearn.preprocessing import FunctionTransformer, StandardScaler

from bayesianbandits import (
    Arm,
    ContextualAgent,
    NormalInverseGammaRegressor,
    NormalRegressor,
    ThompsonSampling,
)
from bayesianbandits.pipeline import Pipeline


class TestPipeline:
    """Test Pipeline functionality."""

    def test_basic_pipeline(self):
        """Test basic pipeline with array input."""
        pipeline = Pipeline(
            [
                ("scaler", StandardScaler()),
                ("model", NormalRegressor(alpha=1.0, beta=1.0)),
            ]
        )

        X = np.array([[1.0], [2.0], [3.0]])
        y = np.array([1.0, 2.0, 3.0])

        # Test partial_fit
        pipeline.partial_fit(X, y)

        # Test predict
        predictions = pipeline.predict(X)
        assert predictions.shape == (3,)

        # Test sample
        samples = pipeline.sample(X, size=10)
        assert samples.shape == (10, 3)

    def test_get_steps(self):
        """Test getting steps from the pipeline."""
        pipeline = Pipeline(
            [
                ("scaler", StandardScaler()),
                ("model", NormalRegressor(alpha=1.0, beta=1.0)),
            ]
        )

        steps = pipeline.get_steps()
        assert len(steps) == 2
        assert steps[0][0] == "scaler"
        assert steps[1][0] == "model"

    def test_get_steps_no_transformers(self):
        """Test getting steps when no transformers are present."""
        pipeline = Pipeline([("model", NormalRegressor(alpha=1.0, beta=1.0))])

        steps = pipeline.get_steps()
        assert len(steps) == 1
        assert steps[0][0] == "model"

    def test_dataframe_input(self):
        """Test pipeline with DataFrame input."""
        pytest.importorskip("pandas")
        import pandas as pd

        pipeline = Pipeline(
            [
                ("select", FunctionTransformer(lambda df: df[["feature"]].values)),
                ("model", NormalRegressor(alpha=1.0, beta=1.0)),
            ]
        )

        df = pd.DataFrame({"feature": [1.0, 2.0, 3.0], "other": ["a", "b", "c"]})
        y = np.array([1.0, 2.0, 3.0])

        pipeline.partial_fit(df, y)
        predictions = pipeline.predict(df)
        assert predictions.shape == (3,)

    def test_dict_input(self):
        """Test pipeline with dict input."""
        pipeline = Pipeline(
            [
                ("vectorizer", DictVectorizer()),
                ("model", NormalRegressor(alpha=1.0, beta=1.0, sparse=True)),
            ]
        )

        X = [
            {"user": "A", "item": "X"},
            {"user": "B", "item": "Y"},
            {"user": "A", "item": "Y"},
        ]
        y = np.array([1.0, 2.0, 1.5])

        pipeline.partial_fit(X, y)
        predictions = pipeline.predict(X)
        assert predictions.shape == (3,)

    def test_shared_model(self):
        """Test multiple pipelines sharing the same model."""
        shared_model = NormalRegressor(alpha=1.0, beta=1.0)

        # Create two pipelines with different preprocessing but same model
        pipeline1 = Pipeline([("scale", StandardScaler()), ("model", shared_model)])

        pipeline2 = Pipeline(
            [("scale", StandardScaler(with_mean=False)), ("model", shared_model)]
        )

        X = np.array([[1.0], [2.0], [3.0]])
        y = np.array([1.0, 2.0, 3.0])

        # Train through first pipeline
        pipeline1.partial_fit(X, y)

        # Model should be updated
        assert hasattr(shared_model, "coef_")
        initial_coef = shared_model.coef_.copy()

        # Train through second pipeline
        pipeline2.partial_fit(X * 2, y * 2)  # type: ignore

        # Shared model should be updated
        assert not np.allclose(shared_model.coef_, initial_coef)

    def test_random_state_propagation(self):
        """Test that random_state is properly propagated."""
        model = NormalRegressor(alpha=1.0, beta=1.0, random_state=42)
        pipeline = Pipeline([("scaler", StandardScaler()), ("model", model)])

        # Check getter
        assert pipeline.random_state == 42

        # Check setter
        pipeline.random_state = 123
        assert model.random_state == 123
        assert pipeline.random_state == 123

    def test_validation_errors(self):
        """Test validation of pipeline steps."""
        # Empty pipeline
        with pytest.raises(ValueError, match="Pipeline cannot be empty"):
            Pipeline([])

        # Invalid final step
        with pytest.raises(ValueError, match="must be a bayesianbandits learner"):
            Pipeline(
                [
                    ("scaler", StandardScaler()),
                    ("invalid", StandardScaler()),  # Not a learner
                ]
            )

    def test_integration_with_bandit(self):
        """Test pipeline integration with ContextualAgent."""
        shared_model = NormalInverseGammaRegressor()

        # Create arms with different preprocessing
        arms: List[Arm[NDArray[np.float64], str]] = []
        for i in range(3):
            pipeline: Pipeline[NDArray[np.float64]] = Pipeline(
                [
                    (
                        "add_bias",
                        FunctionTransformer(
                            lambda x, bias=i: np.column_stack(
                                [x, np.full(len(x), bias)]
                            )
                        ),
                    ),
                    ("model", shared_model),
                ]
            )
            arms.append(Arm(f"arm_{i}", learner=pipeline))

        # Create agent
        agent = ContextualAgent(arms, ThompsonSampling())

        # Pull and update
        X = np.array([[1.0], [2.0]])
        actions = agent.pull(X)
        assert len(actions) == 2

        y = np.array([1.0, 2.0])
        agent.update(X, y)

    def test_decay(self):
        """Test decay propagation."""
        model = NormalRegressor(alpha=1.0, beta=1.0, learning_rate=0.9)
        pipeline = Pipeline([("scaler", StandardScaler()), ("model", model)])

        X = np.array([[1.0], [2.0]])
        y = np.array([1.0, 2.0])

        # Fit first
        pipeline.partial_fit(X, y)
        initial_cov_inv = model.cov_inv_.copy()

        # Decay
        pipeline.decay(X, decay_rate=0.5)

        # Precision should decrease (variance increase)
        assert np.all(model.cov_inv_ < initial_cov_inv)

    def test_no_transformer_pipeline(self):
        """Test pipeline with only a final estimator."""
        pipeline = Pipeline([("model", NormalRegressor(alpha=1.0, beta=1.0))])

        X = np.array([[1.0], [2.0]])
        y = np.array([1.0, 2.0])

        pipeline.partial_fit(X, y)
        predictions = pipeline.predict(X)
        assert predictions.shape == (2,)

    def test_sample_weight_propagation(self):
        """Test that sample weights are properly propagated."""
        model = NormalRegressor(alpha=1.0, beta=1.0)
        pipeline = Pipeline([("scaler", StandardScaler()), ("model", model)])

        X = np.array([[1.0], [2.0], [3.0]])
        y = np.array([1.0, 2.0, 10.0])
        weights = np.array([1.0, 1.0, 0.1])  # Downweight outlier

        pipeline.partial_fit(X, y, sample_weight=weights)

        # With low weight on outlier, prediction should be closer to 1-2 range
        pred = pipeline.predict(np.array([[2.5]]))
        assert pred[0] < 5.0  # Would be ~5.5 without weights


class TestPipelineInputTypes:
    """Test Pipeline with different input types and FunctionTransformer."""

    def test_array_with_static_features(self) -> None:
        """Test adding static features to numpy arrays."""
        # Define item features
        item_price: float = 9.99
        item_category: int = 2

        def add_item_features(X: NDArray[np.float64]) -> NDArray[np.float64]:
            """Add item price and category to each row."""
            n_samples = len(X)
            price_col = np.full((n_samples, 1), item_price)
            category_col = np.full((n_samples, 1), item_category)
            return np.column_stack([X, price_col, category_col])

        pipeline: Pipeline[NDArray[np.float64]] = Pipeline(
            [
                ("add_features", FunctionTransformer(add_item_features)),
                ("model", NormalRegressor(alpha=1.0, beta=1.0)),
            ]
        )

        X: NDArray[np.float64] = np.array([[0.5], [1.0], [1.5]])
        y: NDArray[np.float64] = np.array([10.0, 15.0, 20.0])

        pipeline.partial_fit(X, y)
        predictions: NDArray[np.float64] = pipeline.predict(X)

        assert predictions.shape == (3,)
        assert isinstance(predictions, np.ndarray)

    def test_array_with_interactions(self) -> None:
        """Test computing interaction features."""
        item_features = {"price": 10.0, "discount": 0.2}

        def add_interactions(
            X: NDArray[np.float64], features: Dict[str, float]
        ) -> NDArray[np.float64]:
            """Add features and compute price-sensitive interaction."""
            # X has columns: [user_income, days_since_purchase]
            user_income = X[:, 0]
            days_since = X[:, 1]

            # Compute interactions
            price_sensitivity = user_income / features["price"]
            urgency_discount = np.where(days_since > 30, features["discount"], 0)

            return np.column_stack(  # type:ignore[return]
                [
                    X,
                    price_sensitivity,
                    urgency_discount,
                    np.full(len(X), features["price"]),
                    np.full(len(X), features["discount"]),
                ]
            )

        pipeline: Pipeline[NDArray[np.float64]] = Pipeline(
            [
                (
                    "add_features",
                    FunctionTransformer(
                        partial(add_interactions, features=item_features)
                    ),
                ),
                ("model", NormalRegressor(alpha=1.0, beta=1.0)),
            ]
        )

        X: NDArray[np.float64] = np.array(
            [
                [50000, 45],  # Low income, long time
                [150000, 5],  # High income, recent
            ]
        )
        y: NDArray[np.float64] = np.array([1.0, 0.0])  # First bought, second didn't

        pipeline.partial_fit(X, y)
        assert pipeline.predict(X).shape == (2,)

    def test_sparse_array_features(self) -> None:
        """Test adding features to sparse arrays."""
        from scipy.sparse import hstack

        item_one_hot = csc_array([[0, 0, 1, 0, 0]])  # Item category 3

        def add_sparse_features(X: csc_array) -> csc_array:
            """Add sparse item features to each row."""
            n_samples = X.shape[0]  # type: ignore
            item_features = csc_array(np.tile(item_one_hot.toarray(), (n_samples, 1)))
            return hstack([X, item_features], format="csc")

        pipeline: Pipeline[csc_array] = Pipeline(
            [
                ("add_features", FunctionTransformer(add_sparse_features)),
                ("model", NormalRegressor(alpha=1.0, beta=1.0, sparse=True)),
            ]
        )

        X = csc_array([[1, 0], [0, 1], [1, 1]])
        y = np.array([1.0, 2.0, 3.0])

        pipeline.partial_fit(X, y)
        predictions = pipeline.predict(X)
        assert predictions.shape == (3,)

    def test_dataframe_features(self) -> None:
        """Test adding features to DataFrames."""
        pytest.importorskip("pandas")
        import pandas as pd

        item_info = {
            "brand": "BrandA",
            "price": 29.99,
            "in_stock": True,
            "avg_rating": 4.5,
        }

        def add_item_to_df(df: pd.DataFrame) -> pd.DataFrame:
            """Add item features and compute interactions."""
            df = df.copy()

            # Add static features
            for key, value in item_info.items():
                df[f"item_{key}"] = value

            # Add interactions
            df["can_afford"] = df["user_balance"] >= item_info["price"]
            df["brand_loyalty"] = df["preferred_brand"] == item_info["brand"]

            return df

        pipeline: Pipeline[pd.DataFrame] = Pipeline(
            [
                ("add_features", FunctionTransformer(add_item_to_df)),
                (
                    "vectorize",
                    FunctionTransformer(
                        lambda df: df.select_dtypes(include=[np.number]).values
                    ),
                ),
                ("model", NormalRegressor(alpha=1.0, beta=1.0)),
            ]
        )

        users = pd.DataFrame(
            {
                "user_balance": [100.0, 20.0, 50.0],
                "preferred_brand": ["BrandA", "BrandB", "BrandA"],
            }
        )
        y = np.array([1.0, 0.0, 1.0])  # Who bought

        pipeline.partial_fit(users, y)
        predictions = pipeline.predict(users)

        assert predictions.shape == (3,)

    def test_dict_list_features(self) -> None:
        """Test adding features to list of dicts."""
        from sklearn.feature_extraction import DictVectorizer

        product = {"sku": "ABC123", "category": "electronics", "price_tier": "premium"}

        def enrich_user_dict(
            users: List[Dict[str, str]], product_info: Dict[str, str]
        ) -> List[Dict[str, str]]:
            """Add product info to each user dict."""
            enriched = []
            for user in users:
                user_copy = user.copy()
                # Add product features
                for key, value in product_info.items():
                    user_copy[f"product_{key}"] = value
                # Add interaction
                user_copy["matches_interest"] = (  # type: ignore
                    user.get("interest") == product_info["category"]
                )
                enriched.append(user_copy)
            return enriched

        pipeline: Pipeline[List[Dict[str, str]]] = Pipeline(
            [
                (
                    "add_features",
                    FunctionTransformer(
                        partial(enrich_user_dict, product_info=product)
                    ),
                ),
                ("vectorize", DictVectorizer()),
                ("model", NormalRegressor(alpha=1.0, beta=1.0, sparse=True)),
            ]
        )

        users: List[Dict[str, str]] = [
            {"user_id": "U1", "interest": "electronics"},
            {"user_id": "U2", "interest": "sports"},
            {"user_id": "U3", "interest": "electronics"},
        ]
        y = np.array([1.0, 0.0, 1.0])

        pipeline.partial_fit(users, y)
        predictions = pipeline.predict(users)
        assert predictions.shape == (3,)

    def test_shared_model_different_inputs(self) -> None:
        """Test shared model with pipelines using different input types."""
        pytest.importorskip("pandas")
        import pandas as pd

        shared_model = NormalRegressor(alpha=1.0, beta=1.0)

        # Pipeline 1: DataFrame input for product A
        def add_product_a(df: pd.DataFrame) -> NDArray[np.float64]:
            df = df.copy()
            df["product_price"] = 19.99
            df["product_type"] = 1
            return df[["user_age", "product_price", "product_type"]].values

        pipeline_a: Pipeline[pd.DataFrame] = Pipeline(
            [("transform", FunctionTransformer(add_product_a)), ("model", shared_model)]
        )

        # Pipeline 2: Dict input for product B
        def add_product_b(users: List[Dict[str, float]]) -> NDArray[np.float64]:
            result = []
            for user in users:
                result.append([user["age"], 29.99, 2])  # price=29.99, type=2
            return np.array(result)

        pipeline_b: Pipeline[List[Dict[str, float]]] = Pipeline(
            [("transform", FunctionTransformer(add_product_b)), ("model", shared_model)]
        )

        # Train through pipeline A
        df_users = pd.DataFrame({"user_age": [25.0, 35.0]})
        pipeline_a.partial_fit(df_users, np.array([1.0, 0.0]))

        # Train through pipeline B (updates shared model)
        dict_users: List[Dict[str, float]] = [{"age": 45.0}, {"age": 55.0}]
        pipeline_b.partial_fit(dict_users, np.array([1.0, 1.0]))

        # Both pipelines should work for prediction
        pred_a = pipeline_a.predict(df_users)
        pred_b = pipeline_b.predict(dict_users)

        assert pred_a.shape == (2,)
        assert pred_b.shape == (2,)
