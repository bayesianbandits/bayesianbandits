"""Tests for agent-wrapping AgentPipeline implementation."""

from __future__ import annotations

import numpy as np
import pytest
from sklearn.base import BaseEstimator, TransformerMixin
from sklearn.feature_extraction import DictVectorizer
from sklearn.preprocessing import FunctionTransformer, StandardScaler

from bayesianbandits import (
    Agent,
    Arm,
    ContextualAgent,
    EpsilonGreedy,
    NormalRegressor,
    ThompsonSampling,
    UpperConfidenceBound,
)
from bayesianbandits.pipelines import (
    AgentPipeline,
    ContextualAgentPipeline,
    NonContextualAgentPipeline,
)
from bayesianbandits.pipelines._agent import (
    _transform_data,
    _validate_steps,
)


def make_arms(tokens, learner_args=None):
    """Helper to create arms with proper learner args."""
    if learner_args is None:
        learner_args = {"alpha": 1.0, "beta": 1.0}
    return [Arm(token, learner=NormalRegressor(**learner_args)) for token in tokens]  # type: ignore


class MockTransformer(BaseEstimator, TransformerMixin):
    """Mock transformer for testing."""

    def __init__(self, fitted=False):
        self.fitted = fitted

    def fit(self, X, y=None):
        return self

    def transform(self, X):
        if not self.fitted:
            raise RuntimeError("Transformer not fitted")
        return X * 2


class TestValidateSteps:
    """Test step validation utilities."""

    def test_validate_empty_steps(self):
        """Test validation with empty steps."""
        with pytest.raises(ValueError, match="Pipeline steps cannot be empty"):
            _validate_steps([])

    def test_validate_duplicate_names(self):
        """Test validation with duplicate step names."""
        steps = [
            ("transform", FunctionTransformer()),
            ("transform", FunctionTransformer()),  # Duplicate
        ]
        with pytest.raises(ValueError, match="Step names must be unique"):
            _validate_steps(steps)

    def test_validate_valid_steps(self):
        """Test validation with valid steps."""
        steps = [
            ("step1", FunctionTransformer()),
            ("step2", StandardScaler()),
        ]
        # Should not raise
        _validate_steps(steps)


class TestTransformData:
    """Test data transformation utilities."""

    def test_transform_single_step(self):
        """Test transformation with single step."""
        steps = [("double", FunctionTransformer(lambda x: x * 2))]
        X = np.array([[1], [2], [3]])
        result = _transform_data(X, steps)
        expected = np.array([[2], [4], [6]])
        np.testing.assert_array_equal(result, expected)

    def test_transform_multiple_steps(self):
        """Test transformation with multiple steps."""
        steps = [
            ("double", FunctionTransformer(lambda x: x * 2)),
            ("add_one", FunctionTransformer(lambda x: x + 1)),
        ]
        X = np.array([[1], [2]])
        result = _transform_data(X, steps)
        expected = np.array([[3], [5]])  # (x * 2) + 1
        np.testing.assert_array_equal(result, expected)

    def test_transform_not_fitted_error(self):
        """Test helpful error when transformer not fitted."""
        steps = [("mock", MockTransformer(fitted=False))]
        X = np.array([[1], [2]])

        with pytest.raises(RuntimeError) as exc_info:
            _transform_data(X, steps)

        assert "not fitted" in str(exc_info.value)
        assert "mock" in str(exc_info.value)
        assert "FunctionTransformer" in str(exc_info.value)


class TestContextualAgentPipeline:
    """Test ContextualAgentPipeline class."""

    def test_basic_construction(self):
        """Test basic contextual pipeline construction."""
        arms = make_arms(range(3))
        agent = ContextualAgent(arms, ThompsonSampling())
        steps = [("double", FunctionTransformer(lambda x: x * 2))]

        pipeline = ContextualAgentPipeline(steps, agent)

        assert len(pipeline) == 1
        assert pipeline.named_steps["double"] is not None
        assert pipeline._agent is agent

    def test_empty_steps_error(self):
        """Test empty steps raise error."""
        arms = make_arms(range(3))
        agent = ContextualAgent(arms, ThompsonSampling())

        with pytest.raises(ValueError, match="Pipeline steps cannot be empty"):
            ContextualAgentPipeline([], agent)

    def test_pull_without_top_k(self):
        """Test pull method without top_k."""
        arms = make_arms(range(3))
        agent = ContextualAgent(arms, ThompsonSampling(), random_seed=42)
        steps = [("identity", FunctionTransformer())]

        pipeline = ContextualAgentPipeline(steps, agent)

        X = np.array([[1.0, 2.0], [3.0, 4.0]])
        actions = pipeline.pull(X)

        assert len(actions) == 2
        assert all(isinstance(action, int) for action in actions)

    def test_pull_with_top_k(self):
        """Test pull method with top_k."""
        arms = make_arms(range(5))
        agent = ContextualAgent(arms, ThompsonSampling(), random_seed=42)
        steps = [("identity", FunctionTransformer())]

        pipeline = ContextualAgentPipeline(steps, agent)

        X = np.array([[1.0, 2.0], [3.0, 4.0]])
        action_lists = pipeline.pull(X, top_k=3)

        assert len(action_lists) == 2
        assert all(len(actions) == 3 for actions in action_lists)

    def test_update(self):
        """Test update method."""
        arms = make_arms(range(3))
        agent = ContextualAgent(arms, ThompsonSampling(), random_seed=42)
        steps = [("scale", FunctionTransformer(lambda x: x / 10))]

        pipeline = ContextualAgentPipeline(steps, agent)

        X = np.array([[10.0, 20.0]])
        y = np.array([1.0])

        # Pull to set arm_to_update
        pipeline.pull(X)

        # Update should transform X before passing to agent
        pipeline.update(X, y)

        # Verify the learner was updated with transformed data
        arm_learner = pipeline.arm_to_update.learner
        assert hasattr(arm_learner, "coef_")

    def test_update_with_sample_weight(self):
        """Test update method with sample weights."""
        arms = make_arms(range(3))
        agent = ContextualAgent(arms, ThompsonSampling(), random_seed=42)
        steps = [("identity", FunctionTransformer())]

        pipeline = ContextualAgentPipeline(steps, agent)

        X = np.array([[1.0], [2.0]])
        y = np.array([1.0, 2.0])
        sample_weight = np.array([1.0, 0.1])

        # Pull to set arm_to_update
        pipeline.pull(X)

        # Should not raise
        pipeline.update(X, y, sample_weight=sample_weight)

    def test_decay(self):
        """Test decay method."""
        arms = make_arms(range(3))
        agent = ContextualAgent(arms, ThompsonSampling())
        steps = [("identity", FunctionTransformer())]

        pipeline = ContextualAgentPipeline(steps, agent)

        X = np.array([[1.0]])

        # Should not raise
        pipeline.decay(X, decay_rate=0.5)

    def test_transform(self):
        """Test direct transform method."""
        arms = make_arms(range(3))
        agent = ContextualAgent(arms, ThompsonSampling())
        steps = [("double", FunctionTransformer(lambda x: x * 2))]

        pipeline = ContextualAgentPipeline(steps, agent)

        X = np.array([[1], [2]])
        result = pipeline.transform(X)
        expected = np.array([[2], [4]])
        np.testing.assert_array_equal(result, expected)

    def test_delegation_methods(self):
        """Test that agent methods are properly delegated."""
        arms = make_arms(range(3))
        agent = ContextualAgent(arms, ThompsonSampling(), random_seed=42)
        steps = [("identity", FunctionTransformer())]

        pipeline = ContextualAgentPipeline(steps, agent)

        # Test property access
        assert pipeline.arms is agent.arms
        assert pipeline.policy is agent.policy
        assert pipeline.rng is agent.rng

        # Test arm access
        assert pipeline.arm(0) is agent.arm(0)

        # Test select_for_update
        result = pipeline.select_for_update(1)
        assert result is pipeline
        assert pipeline.arm_to_update is agent.arm_to_update

        # Test add_arm
        new_arm = Arm(99, learner=NormalRegressor(alpha=1.0, beta=1.0))
        pipeline.add_arm(new_arm)
        assert new_arm in agent.arms

        # Test remove_arm
        pipeline.remove_arm(99)
        assert new_arm not in agent.arms

    def test_indexing(self):
        """Test step indexing."""
        arms = make_arms(range(3))
        agent = ContextualAgent(arms, ThompsonSampling())
        transform1 = FunctionTransformer(lambda x: x * 2)
        transform2 = StandardScaler()
        steps = [("double", transform1), ("scale", transform2)]

        pipeline = ContextualAgentPipeline(steps, agent)

        # Test string indexing
        assert pipeline["double"] is transform1
        assert pipeline["scale"] is transform2

        # Test integer indexing
        assert pipeline[0] == ("double", transform1)
        assert pipeline[1] == ("scale", transform2)

        # Test invalid access
        with pytest.raises(KeyError):
            _ = pipeline["invalid"]

        with pytest.raises(IndexError):
            _ = pipeline[10]

    def test_repr(self):
        """Test string representation."""
        arms = make_arms(range(3))
        agent = ContextualAgent(arms, ThompsonSampling())
        steps = [("transform", FunctionTransformer())]

        pipeline = ContextualAgentPipeline(steps, agent)
        repr_str = repr(pipeline)

        assert "ContextualAgentPipeline" in repr_str
        assert "FunctionTransformer" in repr_str


class TestNonContextualAgentPipeline:
    """Test NonContextualAgentPipeline class."""

    def test_basic_construction(self):
        """Test basic non-contextual pipeline construction."""
        arms = make_arms(range(3))
        agent = Agent(arms, ThompsonSampling())
        steps = [("identity", FunctionTransformer())]

        pipeline = NonContextualAgentPipeline(steps, agent)

        assert len(pipeline) == 1
        assert pipeline._agent is agent

    def test_empty_steps_allowed(self):
        """Test empty steps are allowed for non-contextual."""
        arms = make_arms(range(3))
        agent = Agent(arms, ThompsonSampling())

        # Should not raise
        pipeline = NonContextualAgentPipeline([], agent)
        assert len(pipeline) == 0

    def test_pull_without_top_k(self):
        """Test pull method without top_k."""
        arms = make_arms(range(3))
        agent = Agent(arms, ThompsonSampling(), random_seed=42)
        steps = []

        pipeline = NonContextualAgentPipeline(steps, agent)

        actions = pipeline.pull()

        assert len(actions) == 1
        assert isinstance(actions[0], int)

    def test_pull_with_top_k(self):
        """Test pull method with top_k."""
        arms = make_arms(range(5))
        agent = Agent(arms, ThompsonSampling(), random_seed=42)
        steps = []

        pipeline = NonContextualAgentPipeline(steps, agent)

        action_lists = pipeline.pull(top_k=3)

        assert len(action_lists) == 1
        assert len(action_lists[0]) == 3

    def test_update(self):
        """Test update method."""
        arms = make_arms(range(3))
        agent = Agent(arms, ThompsonSampling(), random_seed=42)
        steps = []

        pipeline = NonContextualAgentPipeline(steps, agent)

        y = np.array([1.0, 2.0])

        # Pull to set arm_to_update
        pipeline.pull()

        # Should not raise
        pipeline.update(y)

    def test_update_with_sample_weight(self):
        """Test update method with sample weights."""
        arms = make_arms(range(3))
        agent = Agent(arms, ThompsonSampling(), random_seed=42)
        steps = []

        pipeline = NonContextualAgentPipeline(steps, agent)

        y = np.array([1.0, 2.0])
        sample_weight = np.array([1.0, 0.1])

        # Pull to set arm_to_update
        pipeline.pull()

        # Should not raise
        pipeline.update(y, sample_weight=sample_weight)

    def test_decay(self):
        """Test decay method."""
        arms = make_arms(range(3))
        agent = Agent(arms, ThompsonSampling())
        steps = []

        pipeline = NonContextualAgentPipeline(steps, agent)

        # Should not raise
        pipeline.decay(decay_rate=0.5)

    def test_delegation_methods(self):
        """Test that agent methods are properly delegated."""
        arms = make_arms(range(3))
        agent = Agent(arms, ThompsonSampling(), random_seed=42)
        steps = []

        pipeline = NonContextualAgentPipeline(steps, agent)

        # Test property access
        assert pipeline.arms is agent.arms
        assert pipeline.policy is agent.policy
        assert pipeline.rng is agent.rng

        # Test arm access
        assert pipeline.arm(0) is agent.arm(0)

        # Test select_for_update
        result = pipeline.select_for_update(1)
        assert result is pipeline
        assert pipeline.arm_to_update is agent.arm_to_update

    def test_repr(self):
        """Test string representation."""
        arms = make_arms(range(3))
        agent = Agent(arms, ThompsonSampling())
        steps = []

        pipeline = NonContextualAgentPipeline(steps, agent)
        repr_str = repr(pipeline)

        assert "NonContextualAgentPipeline" in repr_str


class TestAgentPipelineFactory:
    """Test AgentPipeline factory function."""

    def test_contextual_agent_dispatch(self):
        """Test factory dispatches to ContextualAgentPipeline for ContextualAgent."""
        arms = make_arms(range(3))
        agent = ContextualAgent(arms, ThompsonSampling())
        steps = [("identity", FunctionTransformer())]

        pipeline = AgentPipeline(steps, agent)

        assert isinstance(pipeline, ContextualAgentPipeline)
        assert pipeline._agent is agent

    def test_agent_dispatch(self):
        """Test factory dispatches to NonContextualAgentPipeline for Agent."""
        arms = make_arms(range(3))
        agent = Agent(arms, ThompsonSampling())
        steps = [("identity", FunctionTransformer())]

        pipeline = AgentPipeline(steps, agent)

        assert isinstance(pipeline, NonContextualAgentPipeline)
        assert pipeline._agent is agent

    def test_factory_preserves_functionality(self):
        """Test factory-created pipelines work correctly."""
        # Test contextual
        arms = make_arms(range(3))
        contextual_agent = ContextualAgent(arms, ThompsonSampling(), random_seed=42)
        steps = [("scale", FunctionTransformer(lambda x: x / 10))]

        contextual_pipeline = AgentPipeline(steps, contextual_agent)

        X = np.array([[10.0, 20.0]])
        actions = contextual_pipeline.pull(X)
        assert len(actions) == 1

        # Test non-contextual
        arms = make_arms(range(3))
        agent = Agent(arms, ThompsonSampling(), random_seed=42)

        noncontextual_pipeline = AgentPipeline([], agent)

        actions = noncontextual_pipeline.pull()
        assert len(actions) == 1

    def test_factory_isinstance_check(self):
        """Test factory function's isinstance logic for dispatch."""
        arms = make_arms(range(3))

        # Test that Agent gets NonContextualAgentPipeline
        agent = Agent(arms, ThompsonSampling())
        pipeline = AgentPipeline([("identity", FunctionTransformer())], agent)
        assert isinstance(pipeline, NonContextualAgentPipeline)

        # Test that ContextualAgent gets ContextualAgentPipeline
        contextual_agent = ContextualAgent(arms, ThompsonSampling())
        contextual_pipeline = AgentPipeline(
            [("identity", FunctionTransformer())], contextual_agent
        )
        assert isinstance(contextual_pipeline, ContextualAgentPipeline)


class TestTransformationFlow:
    """Test transformation pipeline flow."""

    def test_complex_transformation_chain(self):
        """Test complex sequence of transformations."""
        arms = make_arms(range(3))
        agent = ContextualAgent(arms, EpsilonGreedy(epsilon=0.1), random_seed=42)

        steps = [
            ("double", FunctionTransformer(lambda x: x * 2)),
            ("add_one", FunctionTransformer(lambda x: x + 1)),
            ("square", FunctionTransformer(lambda x: x**2)),
        ]

        pipeline = ContextualAgentPipeline(steps, agent)

        X = np.array([[1.0], [2.0]])
        # Transform: x -> 2x -> 2x+1 -> (2x+1)^2
        # For x=1: 1 -> 2 -> 3 -> 9
        # For x=2: 2 -> 4 -> 5 -> 25

        result = pipeline.transform(X)
        expected = np.array([[9.0], [25.0]])
        np.testing.assert_array_equal(result, expected)

        # Test full pipeline
        actions = pipeline.pull(X)
        assert len(actions) == 2

    def test_sklearn_transformer_integration(self):
        """Test integration with sklearn transformers."""
        # Pre-fit scaler
        scaler = StandardScaler()
        historical_data = np.array([[1.0, 2.0], [3.0, 4.0], [5.0, 6.0]])
        scaler.fit(historical_data)

        arms = make_arms(range(3))
        agent = ContextualAgent(arms, ThompsonSampling(), random_seed=42)

        steps = [("scale", scaler)]
        pipeline = ContextualAgentPipeline(steps, agent)

        X = np.array([[2.0, 3.0], [4.0, 5.0]])

        # Should work with pre-fitted transformer
        actions = pipeline.pull(X)
        assert len(actions) == 2

        y = np.array([1.0, 2.0])
        pipeline.update(X, y)

    def test_dict_vectorizer_integration(self):
        """Test integration with DictVectorizer."""
        # Pre-fit vectorizer
        vectorizer = DictVectorizer()
        historical_dicts = [
            {"user": "A", "item": "X"},
            {"user": "B", "item": "Y"},
        ]
        vectorizer.fit(historical_dicts)

        arms = [
            Arm(i, learner=NormalRegressor(alpha=1.0, beta=1.0, sparse=True))
            for i in range(3)
        ]
        agent = ContextualAgent(arms, ThompsonSampling(), random_seed=42)

        steps = [("vectorize", vectorizer)]
        pipeline = ContextualAgentPipeline(steps, agent)

        X = [{"user": "A", "item": "X"}, {"user": "B", "item": "Y"}]

        actions = pipeline.pull(X)
        assert len(actions) == 2

        y = np.array([1.0, 2.0])
        pipeline.update(X, y)


class TestIntegrationScenarios:
    """Test realistic integration scenarios."""

    @pytest.mark.parametrize(
        "policy_class", [ThompsonSampling, EpsilonGreedy, UpperConfidenceBound]
    )
    def test_recommendation_system_scenario(self, policy_class):
        """Test realistic recommendation system scenario."""
        # Create product arms
        product_arms = make_arms([f"product_{i}" for i in range(5)])

        # Create agent with preprocessing
        agent = ContextualAgent(product_arms, policy_class(), random_seed=42)

        # Pre-fit scaler on historical user data
        scaler = StandardScaler()
        historical_users = np.random.randn(1000, 3)  # user features
        scaler.fit(historical_users)

        # Create pipeline with preprocessing
        steps = [("user_scaler", scaler)]
        pipeline = AgentPipeline(steps, agent)

        # Simulate user interactions
        n_users = 20
        user_contexts = np.random.randn(n_users, 3)

        # Pull recommendations
        recommendations = pipeline.pull(user_contexts)
        assert len(recommendations) == n_users
        assert all(rec.startswith("product_") for rec in recommendations)

        # Simulate rewards and update
        rewards = np.random.beta(2, 5, size=n_users)  # Realistic reward distribution
        pipeline.update(user_contexts, rewards)

        # Verify models were updated
        for arm in pipeline.arms:
            assert hasattr(arm.learner, "coef_")

    def test_ab_testing_scenario(self):
        """Test A/B testing scenario with preprocessing."""
        # Create treatment arms
        arms = make_arms(["control", "treatment"])
        agent = Agent(arms, EpsilonGreedy(epsilon=0.1), random_seed=42)

        # No preprocessing needed for A/B test
        pipeline = AgentPipeline([], agent)

        # Run A/B test
        n_experiments = 100
        rewards = []

        for _ in range(n_experiments):
            assignment = pipeline.pull()[0]

            # Simulate reward based on assignment
            if assignment == "treatment":
                reward = np.random.normal(0.12, 0.1)  # Better performance
            else:
                reward = np.random.normal(0.10, 0.1)  # Baseline

            rewards.append(reward)
            pipeline.update(np.array([reward]))

        # Check that we collected data
        assert len(rewards) == n_experiments

    def test_feature_engineering_pipeline(self):
        """Test complex feature engineering pipeline."""
        arms = make_arms(range(3))
        agent = ContextualAgent(arms, ThompsonSampling(), random_seed=42)

        # Complex feature engineering
        def add_interactions(X):
            """Add interaction terms."""
            # X shape: (n_samples, 2)
            interactions = X[:, 0:1] * X[:, 1:2]  # x1 * x2
            return np.c_[X, interactions]

        def add_polynomials(X):
            """Add polynomial features."""
            # X shape: (n_samples, 3) after interactions
            squared = X**2
            return np.c_[X, squared]

        def normalize_features(X):
            """Simple normalization."""
            return X / (np.linalg.norm(X, axis=1, keepdims=True) + 1e-8)

        steps = [
            ("interactions", FunctionTransformer(add_interactions)),
            ("polynomials", FunctionTransformer(add_polynomials)),
            ("normalize", FunctionTransformer(normalize_features)),
        ]

        pipeline = ContextualAgentPipeline(steps, agent)

        # Test with raw features
        X = np.array([[1.0, 2.0], [3.0, 4.0], [5.0, 6.0]])

        # Check transformation
        transformed = pipeline.transform(X)
        assert transformed.shape[1] == 6  # 2 original + 1 interaction + 3 squared

        # Test full pipeline
        actions = pipeline.pull(X)
        assert len(actions) == 3

        y = np.array([1.0, 2.0, 3.0])
        pipeline.update(X, y)


class TestErrorHandling:
    """Test error handling and edge cases."""

    def test_transformer_error_propagation(self):
        """Test that transformer errors are properly propagated."""

        def failing_transform(X):
            raise ValueError("Custom transformation error")

        arms = make_arms(range(3))
        agent = ContextualAgent(arms, ThompsonSampling())

        steps = [("fail", FunctionTransformer(failing_transform))]
        pipeline = ContextualAgentPipeline(steps, agent)

        X = np.array([[1.0]])

        with pytest.raises(ValueError, match="Custom transformation error"):
            pipeline.pull(X)

    def test_not_fitted_transformer_helpful_error(self):
        """Test helpful error message for not fitted transformers."""
        arms = make_arms(range(3))
        agent = ContextualAgent(arms, ThompsonSampling())

        steps = [("scaler", StandardScaler())]  # Not fitted!
        pipeline = ContextualAgentPipeline(steps, agent)

        X = np.array([[1.0], [2.0]])

        with pytest.raises(RuntimeError) as exc_info:
            pipeline.pull(X)

        error_msg = str(exc_info.value)
        assert "not fitted" in error_msg
        assert "scaler" in error_msg
        assert "FunctionTransformer" in error_msg

    def test_invalid_agent_type(self):
        """Test error handling for invalid agent types."""
        # This would be caught by type checker, but test runtime behavior
        steps = [("identity", FunctionTransformer())]

        # Mock object that doesn't have the expected interface
        class MockAgent:
            pass

        mock_agent = MockAgent()

        # The factory function should handle invalid agent types
        # In practice, this would be a type error at development time
        # Since isinstance check won't match, it will try to create ContextualAgentPipeline
        # which will fail on first attribute access
        pipeline = AgentPipeline(steps, mock_agent)  # type: ignore
        # The error will happen when trying to use the agent
        with pytest.raises(AttributeError):
            _ = pipeline.arms


class TestCoverage:
    """Test edge cases for complete coverage."""

    def test_contextual_pipeline_policy_setter(self):
        """Test policy setter on contextual pipeline."""
        arms = make_arms(range(3))
        agent = ContextualAgent(arms, ThompsonSampling())
        pipeline = ContextualAgentPipeline([("identity", FunctionTransformer())], agent)

        new_policy = EpsilonGreedy(epsilon=0.2)
        pipeline.policy = new_policy
        assert pipeline.policy is new_policy
        assert agent.policy is new_policy

    def test_noncontextual_pipeline_policy_setter(self):
        """Test policy setter on non-contextual pipeline."""
        arms = make_arms(range(3))
        agent = Agent(arms, ThompsonSampling())
        pipeline = NonContextualAgentPipeline([], agent)

        new_policy = EpsilonGreedy(epsilon=0.2)
        pipeline.policy = new_policy
        assert pipeline.policy is new_policy

    def test_contextual_pipeline_with_sample_weights(self):
        """Test contextual pipeline update with sample weights."""
        arms = make_arms(range(3))
        agent = ContextualAgent(arms, ThompsonSampling(), random_seed=42)
        pipeline = ContextualAgentPipeline([("identity", FunctionTransformer())], agent)

        X = np.array([[1.0], [2.0], [3.0]])
        y = np.array([1.0, 2.0, 3.0])
        sample_weight = np.array([1.0, 0.5, 0.1])

        pipeline.pull(X)
        pipeline.update(X, y, sample_weight=sample_weight)

    def test_contextual_pipeline_decay_with_rate(self):
        """Test contextual pipeline decay with explicit rate."""
        arms = make_arms(range(3))
        agent = ContextualAgent(arms, ThompsonSampling())
        pipeline = ContextualAgentPipeline([("identity", FunctionTransformer())], agent)

        X = np.array([[1.0]])
        pipeline.decay(X, decay_rate=0.7)

    def test_noncontextual_pipeline_decay_with_rate(self):
        """Test non-contextual pipeline decay with explicit rate."""
        arms = make_arms(range(3))
        agent = Agent(arms, ThompsonSampling())
        pipeline = NonContextualAgentPipeline([], agent)

        pipeline.decay(decay_rate=0.7)

    def test_pipeline_len_and_getitem_edge_cases(self):
        """Test pipeline length and indexing edge cases."""
        arms = make_arms(range(3))
        agent = ContextualAgent(arms, ThompsonSampling())

        # Empty pipeline (though not allowed by validation)
        # We'll test NonContextualAgentPipeline which allows empty steps
        arms2 = make_arms(range(3))
        agent2 = Agent(arms2, ThompsonSampling())
        empty_pipeline = NonContextualAgentPipeline([], agent2)

        assert len(empty_pipeline) == 0
        assert empty_pipeline.named_steps == {}

        # Test negative indexing
        steps = [("a", FunctionTransformer()), ("b", FunctionTransformer())]
        pipeline = ContextualAgentPipeline(steps, agent)

        assert pipeline[-1] == steps[-1]
        assert pipeline[-2] == steps[-2]

    def test_uncovered_functionality(self):
        """Test functionality to improve code coverage."""
        arms = make_arms(range(3))

        # Test NonContextualAgentPipeline with non-empty steps (edge case)
        agent = Agent(arms, ThompsonSampling(), random_seed=42)
        steps = [("identity", FunctionTransformer())]
        pipeline = NonContextualAgentPipeline(steps, agent)

        # Test all delegation methods on NonContextualAgentPipeline
        assert len(pipeline.arms) == 3
        assert pipeline.arm(0) is not None
        pipeline.select_for_update(1)
        assert pipeline.arm_to_update is not None

        # Test add/remove arm
        new_arm = Arm(99, learner=NormalRegressor(alpha=1.0, beta=1.0))
        pipeline.add_arm(new_arm)
        pipeline.remove_arm(99)

        # Test indexing on NonContextualAgentPipeline
        assert pipeline["identity"] is not None
        assert pipeline[0] == ("identity", steps[0][1])

        # Test with invalid string key
        with pytest.raises(KeyError):
            _ = pipeline["nonexistent"]

        # Test with invalid index
        with pytest.raises(IndexError):
            _ = pipeline[10]
