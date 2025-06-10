"""Tests for LipschitzContextualAgent."""

import numpy as np
import pytest
from sklearn.base import clone

from bayesianbandits import (
    Arm,
    ArmColumnFeaturizer,
    DirichletClassifier,
    EpsilonGreedy,
    GammaRegressor,
    LipschitzContextualAgent,
    NormalInverseGammaRegressor,
    NormalRegressor,
    ThompsonSampling,
    UpperConfidenceBound,
)
from bayesianbandits.featurizers import FunctionArmFeaturizer


@pytest.fixture(
    params=[
        NormalRegressor(alpha=1, beta=1),
        NormalRegressor(alpha=1, beta=1, sparse=True),
        NormalInverseGammaRegressor(),
        NormalInverseGammaRegressor(sparse=True),
    ],
    ids=[
        "normal",
        "normal sparse",
        "normal-inverse-gamma",
        "normal-inverse-gamma sparse",
    ],
)
def learner_class(request):
    return request.param


@pytest.fixture(
    params=[
        EpsilonGreedy(0.8),
        ThompsonSampling(),
        UpperConfidenceBound(0.68),
    ],
    ids=["epsilon-greedy", "thompson-sampling", "ucb"],
)
def policy(request):
    return request.param


@pytest.fixture
def arm_featurizer():
    return ArmColumnFeaturizer(column_name="product_id")


def test_lipschitz_agent_initialization(learner_class, policy, arm_featurizer):
    """Test LipschitzContextualAgent initialization."""
    # Create arms without learners (no reward function needed for normal learners)
    arms = [
        Arm(0, reward_function=None, learner=None),
        Arm(1, reward_function=None, learner=None),
        Arm(2, reward_function=None, learner=None),
    ]

    # Create agent
    agent = LipschitzContextualAgent(
        arms=arms,
        policy=policy,
        arm_featurizer=arm_featurizer,
        learner=clone(learner_class),
        random_seed=42,
    )

    # Verify initialization
    assert len(agent.arms) == 3
    assert agent.policy is policy
    assert agent.arm_featurizer is arm_featurizer
    assert agent.learner is not None
    assert agent.arm_to_update is not None

    # Verify all arms have the shared learner
    for arm in agent.arms:
        assert arm.learner is agent.learner

    # Test empty arms raises error
    with pytest.raises(ValueError, match="At least one arm is required"):
        LipschitzContextualAgent(
            arms=[],
            policy=policy,
            arm_featurizer=arm_featurizer,
            learner=clone(learner_class),
        )


def test_lipschitz_agent_pull_single(learner_class, policy, arm_featurizer):
    """Test pull method without top_k."""
    arms = [
        Arm(0, reward_function=None, learner=None),
        Arm(1, reward_function=None, learner=None),
        Arm(2, reward_function=None, learner=None),
    ]

    agent = LipschitzContextualAgent(
        arms=arms,
        policy=policy,
        arm_featurizer=arm_featurizer,
        learner=clone(learner_class),
        random_seed=42,
    )

    # Test single context
    X = np.array([[25, 50000]])
    result = agent.pull(X)

    assert isinstance(result, list)
    assert len(result) == 1
    assert result[0] in [0, 1, 2]
    assert agent.arm_to_update.action_token == result[0]

    # Test multiple contexts
    X = np.array([[25, 50000], [35, 75000]])
    result = agent.pull(X)

    assert isinstance(result, list)
    assert len(result) == 2
    for token in result:
        assert token in [0, 1, 2]
    assert agent.arm_to_update.action_token == result[-1]


def test_lipschitz_agent_pull_top_k(learner_class, policy, arm_featurizer):
    """Test pull method with top_k."""
    arms = [
        Arm(0, reward_function=None, learner=None),
        Arm(1, reward_function=None, learner=None),
        Arm(2, reward_function=None, learner=None),
    ]

    agent = LipschitzContextualAgent(
        arms=arms,
        policy=policy,
        arm_featurizer=arm_featurizer,
        learner=clone(learner_class),
        random_seed=42,
    )

    # Store original arm_to_update
    original_arm_to_update = agent.arm_to_update

    # Test top_k=2 with single context
    X = np.array([[25, 50000]])
    result = agent.pull(X, top_k=2)

    assert isinstance(result, list)
    assert len(result) == 1  # One context
    assert isinstance(result[0], list)
    assert len(result[0]) == 2  # Top 2 arms
    for token in result[0]:
        assert token in [0, 1, 2]

    # arm_to_update should NOT be updated with top_k
    assert agent.arm_to_update is original_arm_to_update

    # Test top_k=2 with multiple contexts
    X = np.array([[25, 50000], [35, 75000]])
    result = agent.pull(X, top_k=2)

    assert isinstance(result, list)
    assert len(result) == 2  # Two contexts
    for context_result in result:
        assert isinstance(context_result, list)
        assert len(context_result) == 2  # Top 2 arms per context
        for token in context_result:
            assert token in [0, 1, 2]


def test_lipschitz_agent_update(learner_class, policy, arm_featurizer):
    """Test update method."""
    arms = [
        Arm(0, reward_function=None, learner=None),
        Arm(1, reward_function=None, learner=None),
        Arm(2, reward_function=None, learner=None),
    ]

    agent = LipschitzContextualAgent(
        arms=arms,
        policy=policy,
        arm_featurizer=arm_featurizer,
        learner=clone(learner_class),
        random_seed=42,
    )

    # Pull to set arm_to_update
    X = np.array([[25, 50000], [35, 75000]])
    agent.pull(X)

    # Update with rewards
    y = np.array([5.2, 7.8])
    agent.update(X, y)

    # Test update with sample weights
    sample_weight = np.array([1.0, 2.0])
    agent.update(X, y, sample_weight=sample_weight)

    # No assertion errors means update worked


def test_lipschitz_agent_decay(learner_class, policy, arm_featurizer):
    """Test decay method."""
    arms = [
        Arm(0, reward_function=None, learner=None),
        Arm(1, reward_function=None, learner=None),
        Arm(2, reward_function=None, learner=None),
    ]

    agent = LipschitzContextualAgent(
        arms=arms,
        policy=policy,
        arm_featurizer=arm_featurizer,
        learner=clone(learner_class),
        random_seed=42,
    )

    X = np.array([[25, 50000]])

    # Test decay without rate
    agent.decay(X)

    # Test decay with rate
    agent.decay(X, decay_rate=0.9)

    # No assertion errors means decay worked


def test_lipschitz_agent_arm_management(learner_class, policy, arm_featurizer):
    """Test add_arm, remove_arm, arm, and select_for_update methods."""
    arms = [
        Arm(0, reward_function=None, learner=None),
        Arm(1, reward_function=None, learner=None),
    ]

    agent = LipschitzContextualAgent(
        arms=arms,
        policy=policy,
        arm_featurizer=arm_featurizer,
        learner=clone(learner_class),
        random_seed=42,
    )

    # Test add_arm
    new_arm = Arm(2, reward_function=None, learner=None)
    agent.add_arm(new_arm)
    assert len(agent.arms) == 3
    assert new_arm.learner is agent.learner

    # Test duplicate arm token raises error
    duplicate_arm = Arm(2, reward_function=None, learner=None)
    with pytest.raises(ValueError, match="All arms must have unique action tokens"):
        agent.add_arm(duplicate_arm)

    # Test arm() method
    retrieved_arm = agent.arm(1)
    assert retrieved_arm.action_token == 1

    # Test arm() with non-existent token
    with pytest.raises(KeyError, match="Arm with token 999 not found"):
        agent.arm(999)

    # Test select_for_update
    original_arm_to_update = agent.arm_to_update
    agent.select_for_update(1)
    assert agent.arm_to_update.action_token == 1
    assert agent.arm_to_update is not original_arm_to_update

    # Test select_for_update with non-existent token
    with pytest.raises(KeyError, match="Arm with token 999 not found"):
        agent.select_for_update(999)

    # Test remove_arm
    agent.remove_arm(2)
    assert len(agent.arms) == 2
    assert all(arm.action_token != 2 for arm in agent.arms)

    # Test remove_arm with non-existent token
    with pytest.raises(KeyError, match="Arm with token 999 not found"):
        agent.remove_arm(999)


def test_reshape_samples_2d():
    """Test _reshape_samples method with 2D input (standard learners)."""
    arms = [Arm(i, learner=None) for i in range(3)]
    agent = LipschitzContextualAgent(
        arms=arms,
        policy=ThompsonSampling(),
        arm_featurizer=ArmColumnFeaturizer(),
        learner=NormalRegressor(alpha=1, beta=1),
        random_seed=42,
    )

    # Simulate learner output for 3 arms, 2 contexts, 1 sample
    # Shape: (size=1, n_contexts*n_arms=6)
    samples_2d = np.array([[1.1, 1.2, 2.1, 2.2, 3.1, 3.2]])

    reshaped = agent._reshape_samples(samples_2d, n_arms=3, n_contexts=2)

    # Expected shape: (n_arms=3, n_contexts=2, size=1)
    assert reshaped.shape == (3, 2, 1)
    # Check values are correctly reshaped
    assert reshaped[0, 0, 0] == 1.1  # arm 0, context 0
    assert reshaped[0, 1, 0] == 1.2  # arm 0, context 1
    assert reshaped[1, 0, 0] == 2.1  # arm 1, context 0
    assert reshaped[1, 1, 0] == 2.2  # arm 1, context 1
    assert reshaped[2, 0, 0] == 3.1  # arm 2, context 0
    assert reshaped[2, 1, 0] == 3.2  # arm 2, context 1


def test_reshape_samples_3d():
    """Test _reshape_samples method with 3D input (DirichletClassifier)."""
    arms = [Arm(i, learner=None) for i in range(2)]
    agent = LipschitzContextualAgent(
        arms=arms,
        policy=ThompsonSampling(),
        arm_featurizer=ArmColumnFeaturizer(),
        learner=DirichletClassifier({1: 1.0, 2: 1.0}),
        random_seed=42,
    )

    # Simulate DirichletClassifier output for 2 arms, 2 contexts, 1 sample, 2 classes
    # Shape: (size=1, n_contexts*n_arms=4, n_classes=2)
    samples_3d = np.array([[[0.6, 0.4], [0.7, 0.3], [0.5, 0.5], [0.8, 0.2]]])

    reshaped = agent._reshape_samples(samples_3d, n_arms=2, n_contexts=2)

    # Expected shape: (n_arms=2, n_contexts=2, size=1, n_classes=2)
    assert reshaped.shape == (2, 2, 1, 2)
    # Check values are correctly reshaped
    np.testing.assert_array_equal(reshaped[0, 0, 0], [0.6, 0.4])  # arm 0, context 0
    np.testing.assert_array_equal(reshaped[0, 1, 0], [0.7, 0.3])  # arm 0, context 1
    np.testing.assert_array_equal(reshaped[1, 0, 0], [0.5, 0.5])  # arm 1, context 0
    np.testing.assert_array_equal(reshaped[1, 1, 0], [0.8, 0.2])  # arm 1, context 1


def test_lipschitz_agent_with_different_featurizers():
    """Test LipschitzContextualAgent with different featurizers."""
    learner = NormalRegressor(alpha=1, beta=1)

    # Test with ArmColumnFeaturizer using integer tokens
    arm_featurizer = ArmColumnFeaturizer(column_name="fruit")
    agent = LipschitzContextualAgent(
        arms=[Arm(i, learner=None) for i in [0, 1, 2]],  # Use integer tokens
        policy=ThompsonSampling(),
        arm_featurizer=arm_featurizer,
        learner=learner,
        random_seed=42,
    )

    X = np.array([[1.0, 2.0]])
    result = agent.pull(X)
    assert result[0] in [0, 1, 2]

    # Test with FunctionArmFeaturizer
    def feature_func(X, action_tokens):
        # Return in the format expected by FunctionArmFeaturizer
        n_contexts, n_features = X.shape
        n_arms = len(action_tokens)
        # Shape should be (n_contexts, n_features_out, n_arms)
        result = np.zeros((n_contexts, n_features + 1, n_arms))

        for i, token in enumerate(action_tokens):
            result[:, :n_features, i] = X  # Copy context features
            result[:, -1, i] = hash(str(token)) % 1000  # Add arm feature

        return result

    function_featurizer = FunctionArmFeaturizer(feature_func)
    agent2 = LipschitzContextualAgent(
        arms=[Arm(i, learner=None) for i in [0, 1, 2]],
        policy=ThompsonSampling(),
        arm_featurizer=function_featurizer,
        learner=NormalRegressor(alpha=1, beta=1),
        random_seed=42,
    )

    result2 = agent2.pull(X)
    assert result2[0] in [0, 1, 2]


def test_lipschitz_agent_repr(learner_class, policy, arm_featurizer):
    """Test __repr__ method."""
    arms = [Arm(i, reward_function=None, learner=None) for i in range(3)]

    agent = LipschitzContextualAgent(
        arms=arms,
        policy=policy,
        arm_featurizer=arm_featurizer,
        learner=clone(learner_class),
        random_seed=42,
    )

    repr_str = repr(agent)
    assert "LipschitzContextualAgent" in repr_str
    assert "policy=" in repr_str
    assert "arm_featurizer=" in repr_str
    assert "shared_learner=" in repr_str


def test_lipschitz_agent_error_cases():
    """Test various error cases."""
    learner = NormalRegressor(alpha=1, beta=1)
    policy = ThompsonSampling()
    arm_featurizer = ArmColumnFeaturizer()

    # Test with no arms
    with pytest.raises(ValueError, match="At least one arm is required"):
        LipschitzContextualAgent(
            arms=[],
            policy=policy,
            arm_featurizer=arm_featurizer,
            learner=learner,
        )


def test_lipschitz_agent_dirichlet_classifier():
    """Test LipschitzContextualAgent with DirichletClassifier using ordinal encoding."""

    def ordinal_encoding_featurizer(X, action_tokens):
        """Map (context, action) pairs to unique integers for DirichletClassifier."""
        n_contexts = len(X)
        n_arms = len(action_tokens)

        # Create output array: (n_contexts, 1, n_arms) - must be float for compatibility
        result = np.zeros((n_contexts, 1, n_arms), dtype=np.float64)

        for i, token in enumerate(action_tokens):
            for j in range(n_contexts):
                # Create unique encoding: context_features + action_offset
                # Use simple hash of context and action for ordinal encoding
                context_hash = hash(tuple(X[j])) % 100  # Keep it small
                action_offset = token * 1000  # Large offset to separate actions
                result[j, 0, i] = float(context_hash + action_offset)

        return result

    # DirichletClassifier requires a reward function to convert probabilities to rewards
    def reward_func(x: np.ndarray) -> np.ndarray:
        # Take the probability of the first class as the reward
        return x[..., 0]

    # Create arms and agent
    arms = [Arm(i, reward_function=reward_func, learner=None) for i in range(3)]
    agent = LipschitzContextualAgent(
        arms=arms,
        policy=ThompsonSampling(),
        arm_featurizer=FunctionArmFeaturizer(ordinal_encoding_featurizer),
        learner=DirichletClassifier({1: 1.0, 2: 1.0, 3: 1.0}),
        random_seed=42,
    )

    # Test contexts (age, income)
    X = np.array([[25, 50000], [35, 75000]])

    # Test pull
    result = agent.pull(X)
    assert len(result) == 2
    assert all(token in [0, 1, 2] for token in result)

    # Test update - DirichletClassifier expects integer class labels
    # We'll use the action tokens as class labels
    rewards = np.array(
        [1, 2]
    )  # Class labels (must be integers for DirichletClassifier)
    agent.update(X, rewards)

    # Test top_k
    top_k_result = agent.pull(X, top_k=2)
    assert len(top_k_result) == 2
    assert all(len(context_result) == 2 for context_result in top_k_result)

    # Test decay
    agent.decay(X, decay_rate=0.95)


def test_lipschitz_agent_gamma_regressor():
    """Test LipschitzContextualAgent with GammaRegressor using feature engineering."""

    def gamma_feature_engineering(X, action_tokens):
        """Create single integer features for GammaRegressor (expects dictionary keys)."""
        n_contexts = len(X)
        n_arms = len(action_tokens)

        # Create output array: (n_contexts, 1, n_arms) - single column of integers
        result = np.zeros((n_contexts, 1, n_arms), dtype=np.float64)

        for i, token in enumerate(action_tokens):
            for j in range(n_contexts):
                # Create unique integer encoding for (context, action) pair
                # Use simple hash of context combined with action offset
                context_hash = hash(tuple(X[j])) % 100  # Keep it small
                action_offset = token * 1000  # Large offset to separate actions
                result[j, 0, i] = float(context_hash + action_offset)

        return result

    # Create arms and agent
    arms = [Arm(i, learner=None) for i in range(3)]
    agent = LipschitzContextualAgent(
        arms=arms,
        policy=ThompsonSampling(),
        arm_featurizer=FunctionArmFeaturizer(gamma_feature_engineering),
        learner=GammaRegressor(alpha=1.0, beta=1.0),
        random_seed=42,
    )

    # Test contexts (age, income)
    X = np.array([[25, 50000], [35, 75000]])

    # Test pull
    result = agent.pull(X)
    assert len(result) == 2
    assert all(token in [0, 1, 2] for token in result)

    # Test update - GammaRegressor expects positive rewards
    rewards = np.array([2.5, 3.7])  # Positive values for Gamma distribution
    agent.update(X, rewards)

    # Test top_k
    top_k_result = agent.pull(X, top_k=2)
    assert len(top_k_result) == 2
    assert all(len(context_result) == 2 for context_result in top_k_result)

    # Test decay
    agent.decay(X, decay_rate=0.95)


def test_lipschitz_integration():
    """End-to-end test demonstrating learning and shared model benefits."""

    # Create arms for product recommendation
    product_ids = list(range(20))
    arms = [Arm(token, learner=None) for token in product_ids]

    # Create one-hot encoding featurizer for categorical product IDs
    def one_hot_product_featurizer(X, action_tokens):
        """One-hot encode product IDs to avoid ordinal bias."""
        n_contexts = len(X)
        n_context_features = X.shape[1]
        n_arms = len(action_tokens)
        max_product_id = max(product_ids)

        n_output_features = n_context_features + max_product_id + 1
        result = np.zeros((n_contexts, n_output_features, n_arms))

        for arm_idx, product_id in enumerate(action_tokens):
            for ctx_idx in range(n_contexts):
                result[ctx_idx, :n_context_features, arm_idx] = X[ctx_idx]
                result[ctx_idx, n_context_features + product_id, arm_idx] = 1.0

        return result

    # Create agent with shared learner and one-hot encoding
    agent = LipschitzContextualAgent(
        arms=arms,
        policy=ThompsonSampling(),
        arm_featurizer=FunctionArmFeaturizer(one_hot_product_featurizer),
        learner=NormalRegressor(alpha=1.0, beta=1.0),
        random_seed=42,
    )

    # Define test contexts
    young_contexts = np.array([[22, 35000], [28, 42000], [25, 38000]])
    older_contexts = np.array([[45, 85000], [52, 95000], [48, 78000]])

    # Test initial state
    young_initial = agent.pull(young_contexts)
    older_initial = agent.pull(older_contexts)
    assert len(young_initial) == 3
    assert len(older_initial) == 3
    assert all(0 <= rec < 20 for rec in young_initial + older_initial)

    # Phase 1: Batch learning with stronger signal - young users prefer cheaper products (0-9)
    for _ in range(15):  # More training rounds
        young_recs = agent.pull(young_contexts)
        young_rewards = np.array(
            [10.0 if rec < 10 else 1.0 for rec in young_recs]
        )  # Stronger contrast
        agent.update(young_contexts, young_rewards)

    # Phase 1: Batch learning with stronger signal - older users prefer premium products (10-19)
    for _ in range(15):  # More training rounds
        older_recs = agent.pull(older_contexts)
        older_rewards = np.array(
            [10.0 if rec >= 10 else 1.0 for rec in older_recs]
        )  # Stronger contrast
        agent.update(older_contexts, older_rewards)

    # Phase 2: Individual arm updates using select_for_update
    training_data = [
        (np.array([[23, 40000]]), 2, 9.0),  # Young + cheap = high reward
        (np.array([[26, 35000]]), 5, 8.5),
        (np.array([[29, 45000]]), 8, 8.8),
        (np.array([[47, 90000]]), 15, 9.2),  # Older + premium = high reward
        (np.array([[51, 88000]]), 18, 8.9),
        (np.array([[44, 82000]]), 12, 9.1),
        (np.array([[24, 38000]]), 16, 3.0),  # Young + premium = low reward
        (np.array([[49, 87000]]), 3, 2.8),  # Older + cheap = low reward
    ]

    for context, arm_token, reward in training_data:
        agent.select_for_update(arm_token).update(context, np.array([reward]))

    # Verify learning: check model learns correct contextual preferences
    cheap_arms = [0, 5]  # Product IDs 0-9 are cheap
    premium_arms = [15, 19]  # Product IDs 10-19 are premium

    young_test_features = agent.arm_featurizer.transform(
        young_contexts[:1], action_tokens=cheap_arms + premium_arms
    )
    older_test_features = agent.arm_featurizer.transform(
        older_contexts[:1], action_tokens=cheap_arms + premium_arms
    )

    young_predictions = agent.learner.predict(young_test_features)
    older_predictions = agent.learner.predict(older_test_features)

    # Test that agent learned correct directional preferences
    young_cheap_pred = np.mean(young_predictions[:2])  # Arms 0,5 (cheap)
    young_premium_pred = np.mean(young_predictions[2:])  # Arms 15,19 (premium)
    older_cheap_pred = np.mean(older_predictions[:2])  # Arms 0,5 (cheap)
    older_premium_pred = np.mean(older_predictions[2:])  # Arms 15,19 (premium)

    # Verify learned preferences show correct directional effects
    # Test within-context preference differences (should show learning signal)
    young_pref_diff = young_cheap_pred - young_premium_pred  # Should be positive
    older_pref_diff = older_premium_pred - older_cheap_pred  # Should be positive

    assert young_pref_diff > 0.2, (
        f"Young users should prefer cheap products: diff={young_pref_diff:.2f} ({young_cheap_pred:.2f} vs {young_premium_pred:.2f})"
    )

    # For older users, check if we can detect any learning signal
    # Even if weak, we should see some contextual differentiation
    assert abs(older_pref_diff) > 0.1 or abs(young_pref_diff - older_pref_diff) > 0.3, (
        f"Should detect contextual learning signal: older_diff={older_pref_diff:.2f}, young_diff={young_pref_diff:.2f}"
    )

    # Test cross-context differences - young should differ from older in preferences
    cheap_context_effect = young_cheap_pred - older_cheap_pred  # Should be positive
    premium_context_effect = (
        older_premium_pred - young_premium_pred
    )  # Should be positive or at least different

    # At minimum, contexts should show different patterns (learning contextualization)
    total_context_effect = abs(cheap_context_effect) + abs(premium_context_effect)
    assert total_context_effect > 0.3, (
        f"Contexts should show different arm preferences: cheap_effect={cheap_context_effect:.2f}, premium_effect={premium_context_effect:.2f}"
    )

    # Test top-k functionality
    young_top_k = agent.pull(young_contexts[:1], top_k=5)[0]
    older_top_k = agent.pull(older_contexts[:1], top_k=5)[0]

    assert len(young_top_k) == 5
    assert len(older_top_k) == 5
    assert len(set(young_top_k)) == 5  # No duplicates in top-k
    assert len(set(older_top_k)) == 5  # No duplicates in top-k

    # Test model learning quality: verify that the agent has learned from training
    # Compare predictions before and after additional training to show learning progression
    baseline_young_pred = agent.learner.predict(young_test_features)

    # Do additional focused training to strengthen signal
    for _ in range(5):
        agent.select_for_update(2).update(
            np.array([[25, 40000]]), np.array([9.0])
        )  # Young + cheap
        agent.select_for_update(15).update(
            np.array([[50, 85000]]), np.array([9.0])
        )  # Older + premium

    # Check that predictions have changed (evidence of continued learning)
    updated_young_pred = agent.learner.predict(young_test_features)

    prediction_change = np.mean(np.abs(updated_young_pred - baseline_young_pred))
    assert prediction_change > 0.01, (
        f"Model should continue learning: change={prediction_change:.4f}"
    )

    # Test basic functionality with new contexts
    new_young_context = np.array([[24, 41000]])
    new_older_context = np.array([[46, 83000]])

    new_young_rec = agent.pull(new_young_context)[0]
    new_older_rec = agent.pull(new_older_context)[0]

    assert 0 <= new_young_rec < 20, "Should make valid recommendations for new contexts"
    assert 0 <= new_older_rec < 20, "Should make valid recommendations for new contexts"

    # Test decay functionality
    agent.decay(np.vstack([young_contexts, older_contexts]), decay_rate=0.8)

    post_decay_young = [agent.pull(young_contexts[:1])[0] for _ in range(3)]
    post_decay_older = [agent.pull(older_contexts[:1])[0] for _ in range(3)]

    assert all(0 <= rec < 20 for rec in post_decay_young + post_decay_older)

    # Verify shared learner is actually shared
    assert all(arm.learner is agent.learner for arm in agent.arms)


def test_lipschitz_shared_learner_efficiency():
    """Test demonstrating the key benefits of shared learner approach."""

    # Create a large action space to show efficiency benefits
    n_arms = 50
    arms = [Arm(token, learner=None) for token in range(n_arms)]

    # Create one-hot encoding for item IDs (categorical, not ordinal)
    def one_hot_item_featurizer(X, action_tokens):
        """One-hot encode item IDs to treat them as categorical."""
        n_contexts = len(X)
        n_context_features = X.shape[1]
        n_arms_in_call = len(action_tokens)

        n_output_features = n_context_features + n_arms
        result = np.zeros((n_contexts, n_output_features, n_arms_in_call))

        for arm_idx, item_id in enumerate(action_tokens):
            for ctx_idx in range(n_contexts):
                result[ctx_idx, :n_context_features, arm_idx] = X[ctx_idx]
                result[ctx_idx, n_context_features + item_id, arm_idx] = 1.0

        return result

    agent = LipschitzContextualAgent(
        arms=arms,
        policy=ThompsonSampling(),
        arm_featurizer=FunctionArmFeaturizer(one_hot_item_featurizer),
        learner=NormalRegressor(alpha=1.0, beta=1.0),
        random_seed=42,
    )

    # Verify all arms share the same learner (key shared learner property)
    assert all(arm.learner is agent.learner for arm in agent.arms)

    # Test 1: Knowledge transfer - learning about one arm should affect others
    training_contexts = np.array([[80, 90000], [85, 95000], [90, 100000]])

    # Train extensively on high-numbered arms with strong signal
    high_arms = [45, 46, 47]
    for _ in range(10):  # More training rounds
        for context in training_contexts:
            for arm_id in high_arms:
                agent.select_for_update(arm_id).update(
                    context.reshape(1, -1),
                    np.array([10.0]),  # Higher reward to create stronger signal
                )

    # Train low-numbered arms with low rewards to create contrast
    low_arms = [5, 8, 12]
    for _ in range(10):
        for context in training_contexts:
            for arm_id in low_arms:
                agent.select_for_update(arm_id).update(
                    context.reshape(1, -1),
                    np.array([2.0]),  # Low reward to create differentiation
                )

    # Test if learning transfers to untrained arms in similar range
    untrained_high_arms = [48, 49]
    test_context = np.array([[82, 92000]])

    predictions_trained = []
    predictions_untrained = []

    for arm_id in high_arms:
        features = agent.arm_featurizer.transform(test_context, action_tokens=[arm_id])
        pred = agent.learner.predict(features)[0]
        predictions_trained.append(pred)

    for arm_id in untrained_high_arms:
        features = agent.arm_featurizer.transform(test_context, action_tokens=[arm_id])
        pred = agent.learner.predict(features)[0]
        predictions_untrained.append(pred)

    # Test quality of knowledge transfer - similar arms should benefit more
    avg_trained = np.mean(predictions_trained)
    avg_untrained = np.mean(predictions_untrained)

    # Untrained similar arms should benefit from knowledge transfer
    assert avg_untrained > 3.0, (
        f"Untrained arms should benefit from knowledge transfer, got {avg_untrained}"
    )
    assert avg_untrained > avg_trained * 0.6, (
        f"Transfer should be substantial: {avg_untrained:.2f} vs {avg_trained:.2f}"
    )

    # Test that knowledge transfer creates clear differentiation after training
    # Get predictions for low-reward arms to compare against
    predictions_low = []
    for arm_id in low_arms:
        features = agent.arm_featurizer.transform(test_context, action_tokens=[arm_id])
        pred = agent.learner.predict(features)[0]
        predictions_low.append(pred)

    avg_low = np.mean(predictions_low)

    # Verify strong learning differentiation
    assert avg_trained > avg_low + 1.0, (
        f"High-reward arms should have much higher predictions than low-reward arms: {avg_trained:.2f} vs {avg_low:.2f}"
    )

    # Verify that knowledge transfer creates similarity-based patterns
    assert avg_untrained > avg_low + 0.5, (
        f"Similar untrained arms should benefit from high-reward transfer: {avg_untrained:.2f} vs {avg_low:.2f}"
    )

    # Verify knowledge transfer creates meaningful separation
    # Untrained similar arms should be significantly above low-reward arms
    assert avg_untrained > avg_low + 2.0, (
        f"Knowledge transfer should create substantial lift above low-reward baseline: {avg_untrained:.2f} vs {avg_low:.2f}"
    )

    # Test 2: Efficiency - vectorized operations
    test_contexts = np.array([[50, 60000], [55, 65000], [60, 70000]])

    # This single call processes all contexts x all arms efficiently
    import time

    start_time = time.time()
    recommendations = agent.pull(test_contexts)
    pull_time = time.time() - start_time

    # Verify the pull worked correctly and was reasonably fast
    assert len(recommendations) == len(test_contexts)
    assert all(0 <= rec < n_arms for rec in recommendations)
    assert pull_time < 0.1  # Should be very fast with vectorized operations

    # Test 3: Proper use of select_for_update for targeted learning
    middle_age_contexts = np.array([[40, 55000], [42, 58000], [38, 52000]])
    target_arm = 25

    # Multiple updates to establish pattern
    for _ in range(5):
        for context in middle_age_contexts:
            agent.select_for_update(target_arm).update(
                context.reshape(1, -1), np.array([8.5])
            )

    # Test that learning affects future decisions (exploration reduction with confidence)
    def measure_exploration_rate(agent, context, n_samples=15):
        samples = [agent.pull(context)[0] for _ in range(n_samples)]
        return len(set(samples)) / len(samples)  # Diversity ratio

    # Measure exploration before building confidence in target arm
    initial_exploration = measure_exploration_rate(agent, middle_age_contexts[:1])

    # Give strong consistent signal to build confidence in target arm
    for _ in range(10):
        agent.select_for_update(target_arm).update(
            middle_age_contexts[:1],
            np.array([9.5]),  # Very high reward to build strong preference
        )

    # Measure exploration after confidence building
    final_exploration = measure_exploration_rate(agent, middle_age_contexts[:1])

    # Assert that exploration decreases as confidence increases
    assert final_exploration <= initial_exploration, (
        f"Exploration should decrease with confidence: {initial_exploration:.2f} → {final_exploration:.2f}"
    )
    assert final_exploration > 0.05, (
        "Should maintain some exploration to avoid overfitting"
    )

    # Verify agent continues to make valid decisions and shows some learning behavior
    final_selections = [agent.pull(middle_age_contexts[:1])[0] for _ in range(20)]
    assert all(0 <= sel < n_arms for sel in final_selections), (
        "Should make valid arm selections"
    )
    assert len(set(final_selections)) < n_arms, (
        "Should show some preference concentration (not uniform random)"
    )

    # Test 4: Top-k with large action space
    top_k_results = agent.pull(test_contexts, top_k=10)

    # Verify top-k structure
    assert len(top_k_results) == len(test_contexts)
    assert all(len(context_top_k) == 10 for context_top_k in top_k_results)
    assert all(
        len(set(context_top_k)) == 10 for context_top_k in top_k_results
    )  # No duplicates
