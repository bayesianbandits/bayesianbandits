"""Tests for LearnerPipeline implementation."""

from __future__ import annotations

import numpy as np
from numpy.typing import NDArray
import pytest
from sklearn.preprocessing import StandardScaler, FunctionTransformer
from sklearn.decomposition import PCA

from bayesianbandits import (
    NormalRegressor,
    Arm,
    LipschitzContextualAgent,
    ThompsonSampling,
)
from bayesianbandits.pipelines import LearnerPipeline


class MockLearner:
    """Mock learner implementing the Learner protocol."""

    def __init__(self):
        self.partial_fit_calls = []
        self.sample_calls = []
        self.predict_calls = []
        self.decay_calls = []
        self.random_state = None

    def partial_fit(self, X, y, sample_weight=None):
        self.partial_fit_calls.append((X, y, sample_weight))
        return self

    def sample(self, X, size=1):
        self.sample_calls.append((X, size))
        return np.random.randn(len(X), size).squeeze()

    def predict(self, X):
        self.predict_calls.append(X)
        return np.zeros(len(X))

    def decay(self, X, *, decay_rate=None):
        self.decay_calls.append((X, decay_rate))


class TestLearnerPipelineInit:
    """Test LearnerPipeline initialization."""

    def test_duplicate_step_names_error(self):
        """Test that duplicate step names raise ValueError."""
        mock_learner = MockLearner()
        with pytest.raises(ValueError, match="Step names must be unique"):
            LearnerPipeline(
                steps=[("dup", StandardScaler()), ("dup", StandardScaler())],
                learner=mock_learner
            )

    def test_valid_initialization_with_transformers(self):
        """Test initialization with transformers."""
        mock_learner = MockLearner()
        pipeline = LearnerPipeline(
            steps=[("scale", StandardScaler())],
            learner=mock_learner
        )

        assert len(pipeline.steps) == 1
        assert pipeline.learner is mock_learner

    def test_missing_learner_methods_error(self):
        """Test that learner without Learner protocol raises ValueError."""

        class BadLearner:
            def partial_fit(self, X, y):
                pass

            # Missing sample, predict, decay

        with pytest.raises(
            ValueError, match="Missing methods: \\['sample', 'predict', 'decay'\\]"
        ):
            LearnerPipeline(steps=[], learner=BadLearner())

    def test_valid_initialization(self):
        """Test valid initialization."""
        mock_learner = MockLearner()
        pipeline = LearnerPipeline(
            steps=[("scale", StandardScaler())],
            learner=mock_learner
        )

        assert len(pipeline.steps) == 1
        assert pipeline.learner is mock_learner


class TestLearnerPipelineTransformers:
    """Test transformer behavior."""

    def test_stateless_transformers(self):
        """Test stateless transformers work correctly."""

        def double_transform(X):
            return X * 2

        mock_learner = MockLearner()
        pipeline = LearnerPipeline(steps=[("double", FunctionTransformer(double_transform))], learner=mock_learner)

        X = np.array([[1, 2], [3, 4]])
        y = np.array([1, 2])

        # Should transform without any fitting
        pipeline.partial_fit(X, y)

        # Check that data was doubled before reaching learner
        received_X, received_y, _ = mock_learner.partial_fit_calls[0]
        np.testing.assert_array_equal(received_X, X * 2)
        np.testing.assert_array_equal(received_y, y)

    def test_pre_fitted_transformers(self):
        """Test pre-fitted transformers work correctly."""
        scaler = StandardScaler()
        X_train = np.array([[1, 2], [3, 4], [5, 6]])
        scaler.fit(X_train)  # Pre-fit

        mock_learner = MockLearner()
        pipeline = LearnerPipeline(steps=[("scale", scaler)], learner=mock_learner)

        X = np.array([[2, 3]])
        y = np.array([1])

        pipeline.partial_fit(X, y)

        # Should use pre-fitted scaler
        received_X, _, _ = mock_learner.partial_fit_calls[0]
        expected_X = scaler.transform(X)
        np.testing.assert_array_almost_equal(received_X, expected_X)

    def test_multiple_transformers(self):
        """Test multiple transformers work correctly."""

        def add_one(X):
            return X + 1

        def multiply_two(X):
            return X * 2

        mock_learner = MockLearner()
        pipeline = LearnerPipeline(steps=[("add", FunctionTransformer(add_one)), ("multiply", FunctionTransformer(multiply_two))], learner=mock_learner)

        X = np.array([[1, 2]])
        y = np.array([1])

        pipeline.partial_fit(X, y)

        # Should apply transformations in sequence: (X + 1) * 2
        received_X, _, _ = mock_learner.partial_fit_calls[0]
        expected_X = (X + 1) * 2
        np.testing.assert_array_equal(received_X, expected_X)

    def test_no_transformers(self):
        """Test pipeline with no transformer steps (only learner)."""
        mock_learner = MockLearner()
        pipeline = LearnerPipeline(steps=[], learner=mock_learner)

        X = np.array([[1, 2], [3, 4]])
        y = np.array([1, 2])

        # Should pass data through unchanged when no transformers
        pipeline.partial_fit(X, y)

        # Check that data was passed through unchanged
        received_X, received_y, _ = mock_learner.partial_fit_calls[0]
        np.testing.assert_array_equal(received_X, X)
        np.testing.assert_array_equal(received_y, y)

        # Test other methods also pass data through unchanged
        pipeline.sample(X, size=2)
        sample_X, size = mock_learner.sample_calls[0]
        np.testing.assert_array_equal(sample_X, X)
        assert size == 2

        pipeline.predict(X)
        predict_X = mock_learner.predict_calls[0]
        np.testing.assert_array_equal(predict_X, X)

        pipeline.decay(X, decay_rate=0.9)
        decay_X, decay_rate = mock_learner.decay_calls[0]
        np.testing.assert_array_equal(decay_X, X)
        assert decay_rate == 0.9

    def test_transformer_not_fitted_error(self):
        """Test helpful error message when transformer is not fitted."""
        from sklearn.preprocessing import StandardScaler
        
        mock_learner = MockLearner()
        # Create pipeline with unfitted transformer
        pipeline = LearnerPipeline(steps=[("unfitted_scaler", StandardScaler())], learner=mock_learner)

        X = np.array([[1, 2], [3, 4]])
        y = np.array([1, 2])

        # Should provide helpful error message
        with pytest.raises(RuntimeError) as exc_info:
            pipeline.partial_fit(X, y)

        error_msg = str(exc_info.value)
        assert "unfitted_scaler" in error_msg
        assert "not fitted" in error_msg
        assert "stateless or pre-fitted" in error_msg
        assert "FunctionTransformer" in error_msg


class TestLearnerPipelineInterface:
    """Test Learner protocol implementation."""

    def setup_method(self):
        """Set up test pipeline."""
        self.mock_learner = MockLearner()
        # Pre-fit the scaler for testing
        scaler = StandardScaler()
        scaler.fit(np.random.randn(100, 2))  # Fit on dummy data
        self.pipeline = LearnerPipeline(steps=[("scale", scaler)], learner=self.mock_learner)

    def test_sample_method(self):
        """Test sample method delegates correctly."""
        X = np.array([[1, 2], [3, 4]])

        # Train the pipeline first
        self.pipeline.partial_fit(X, np.array([1, 2]))

        # Now sample
        self.pipeline.sample(X, size=5)

        assert len(self.mock_learner.sample_calls) == 1
        received_X, size = self.mock_learner.sample_calls[0]
        assert size == 5
        # X should be transformed
        assert received_X.shape == X.shape

    def test_predict_method(self):
        """Test predict method delegates correctly."""
        X = np.array([[1, 2], [3, 4]])

        # Train the pipeline first
        self.pipeline.partial_fit(X, np.array([1, 2]))

        # Now predict
        self.pipeline.predict(X)

        assert len(self.mock_learner.predict_calls) == 1
        received_X = self.mock_learner.predict_calls[0]
        assert received_X.shape == X.shape

    def test_decay_method(self):
        """Test decay method delegates correctly."""
        X = np.array([[1, 2]])

        # Train the pipeline first
        self.pipeline.partial_fit(X, np.array([1]))

        # Now decay
        self.pipeline.decay(X, decay_rate=0.9)

        assert len(self.mock_learner.decay_calls) == 1
        received_X, decay_rate = self.mock_learner.decay_calls[0]
        assert decay_rate == 0.9
        assert received_X.shape == X.shape

    def test_random_state_property(self):
        """Test random_state property delegation."""
        # Test getting random_state
        self.mock_learner.random_state = 42  # type: ignore
        assert self.pipeline.random_state == 42

        # Test setting random_state
        self.pipeline.random_state = 123
        assert self.mock_learner.random_state == 123


class TestLearnerPipelineIntegration:
    """Test integration with real components."""

    def test_with_normal_regressor(self):
        """Test pipeline with real NormalRegressor."""
        # Pre-fit the scaler
        scaler = StandardScaler()
        scaler.fit(np.random.randn(50, 3))  # Fit on dummy data

        pipeline = LearnerPipeline(
            steps=[("scale", scaler)],
            learner=NormalRegressor(alpha=1.0, beta=1.0)
        )

        # Generate training data
        X = np.random.randn(20, 3)
        y = np.random.randn(20)

        # Train
        pipeline.partial_fit(X, y)

        # Test all methods work
        samples = pipeline.sample(X[:5], size=10)
        assert samples.shape == (10, 5)  # (size, n_samples)

        predictions = pipeline.predict(X[:5])
        assert predictions.shape == (5,)

        # Decay should work
        pipeline.decay(X[:5], decay_rate=0.95)

    def test_with_arm_and_lipschitz_agent(self):
        """Test LearnerPipeline used as learner in LipschitzContextualAgent."""
        from bayesianbandits.featurizers import FunctionArmFeaturizer

        # Create a numeric-only arm featurizer to avoid string conversion issues
        def numeric_arm_featurizer(X, action_tokens):
            """Add numeric arm features instead of string tokens."""
            n_contexts, n_features = X.shape
            n_arms = len(action_tokens)

            # Create 3D array: (n_contexts, n_features + 1, n_arms)
            result = np.zeros((n_contexts, n_features + 1, n_arms))

            for i, token in enumerate(action_tokens):
                result[:, :-1, i] = X  # Original features
                result[:, -1, i] = int(token.split("_")[1])  # Numeric arm ID

            return result

        # Create shared learner pipeline that works with numeric data
        # Pre-fit the scaler
        scaler = StandardScaler()
        scaler.fit(
            np.random.randn(50, 11)
        )  # Fit on dummy data (10 context + 1 arm feature)

        shared_learner: LearnerPipeline[NDArray[np.float64]] = LearnerPipeline(
            steps=[("scale", scaler)],  # Pre-fitted scaler
            learner=NormalRegressor(alpha=1.0, beta=1.0)
        )

        # Create arms that all share this learner
        arms = [Arm(f"item_{i}", learner=shared_learner) for i in range(5)]

        # Create agent with numeric arm featurizer
        agent = LipschitzContextualAgent(
            arms=arms,
            policy=ThompsonSampling(),
            arm_featurizer=FunctionArmFeaturizer(numeric_arm_featurizer),
            learner=shared_learner,
        )

        # Generate context (will be enriched by ArmFeaturizer)
        context = np.random.randn(3, 10)  # 3 contexts, 10 features each

        # Pull recommendations
        recommendations = agent.pull(context)
        assert len(recommendations) == 3

        # Update with rewards
        rewards = np.array([1.0, 0.5, 0.8])
        agent.update(context, rewards)

        # Should work without errors

    def test_complex_pipeline(self):
        """Test complex pipeline with multiple transformers."""
        # Pre-fit transformers
        scaler = StandardScaler()
        pca = PCA(n_components=5)

        # Fit on dummy high-dimensional data
        dummy_data = np.random.randn(100, 20)
        scaler.fit(dummy_data)
        pca.fit(scaler.transform(dummy_data))

        pipeline = LearnerPipeline(
            steps=[("scale", scaler), ("pca", pca)],
            learner=NormalRegressor(alpha=0.1, beta=1.0)
        )

        # High-dimensional data
        X = np.random.randn(50, 20)
        y = np.random.randn(50)

        # Should handle the full pipeline
        pipeline.partial_fit(X, y)

        # Test inference
        X_test = np.random.randn(10, 20)
        samples = pipeline.sample(X_test, size=1)
        assert samples.shape == (1, 10)  # (size, n_samples)

        predictions = pipeline.predict(X_test)
        assert predictions.shape == (10,)


class TestLearnerPipelineProperties:
    """Test pipeline properties and methods."""

    def test_named_steps_property(self):
        """Test named_steps property."""
        scaler = StandardScaler()
        learner = MockLearner()
        pipeline = LearnerPipeline(steps=[("scale", scaler)], learner=learner)

        named_steps = pipeline.named_steps
        assert named_steps["scale"] is scaler
        # No learner in named_steps - it's accessed via .learner property

    def test_steps_property(self):
        """Test steps property."""
        scaler = StandardScaler()
        learner = MockLearner()
        pipeline = LearnerPipeline(steps=[("scale", scaler)], learner=learner)

        steps = pipeline.steps
        assert len(steps) == 1  # Only transformer steps
        assert steps[0] == ("scale", scaler)

    def test_len_method(self):
        """Test __len__ method."""
        pipeline = LearnerPipeline(steps=[("scale", StandardScaler())], learner=MockLearner())
        assert len(pipeline) == 1  # Only transformer steps

    def test_getitem_method(self):
        """Test __getitem__ method."""
        scaler = StandardScaler()
        learner = MockLearner()
        pipeline = LearnerPipeline(steps=[("scale", scaler)], learner=learner)

        # Access by index
        assert pipeline[0] == ("scale", scaler)

        # Access by name (only transformer steps)
        assert pipeline["scale"] is scaler

    def test_repr_method(self):
        """Test __repr__ method."""
        pipeline = LearnerPipeline(steps=[("scale", StandardScaler())], learner=MockLearner())

        repr_str = repr(pipeline)
        assert "LearnerPipeline" in repr_str
        assert "StandardScaler" in repr_str
        assert "MockLearner" in repr_str
