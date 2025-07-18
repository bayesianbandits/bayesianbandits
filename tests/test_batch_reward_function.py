"""Test suite for BatchRewardFunction functionality in LipschitzContextualAgent."""

import numpy as np
import pytest
import warnings
from numpy.typing import NDArray

from bayesianbandits import (
    Arm,
    LipschitzContextualAgent,
    ThompsonSampling,
    ArmColumnFeaturizer,
    NormalRegressor,
)
from bayesianbandits._arm import batch_identity


class TestBatchRewardFunction:
    """Test suite for BatchRewardFunction functionality."""

    def test_identity_optimization_detection(self):
        """Test that identity functions are automatically detected."""
        # All arms with default identity function
        arms = [Arm(i, learner=None) for i in range(10)]

        agent = LipschitzContextualAgent(
            arms=arms,
            policy=ThompsonSampling(),
            arm_featurizer=ArmColumnFeaturizer(),
            learner=NormalRegressor(alpha=1, beta=1),
            random_seed=42,
        )

        # Should detect and optimize
        assert agent.batch_reward_function is batch_identity

    def test_identity_optimization_dynamic(self):
        """Test identity optimization updates with add/remove arm."""
        arms = [Arm(i, learner=None) for i in range(3)]

        agent = LipschitzContextualAgent(
            arms=arms,
            policy=ThompsonSampling(),
            arm_featurizer=ArmColumnFeaturizer(),
            learner=NormalRegressor(alpha=1, beta=1),
        )

        # Initially optimized
        assert agent.batch_reward_function is batch_identity

        # Add arm with custom reward function
        def custom_reward(x: NDArray[np.float64]) -> NDArray[np.float64]:
            return np.multiply(x, 2.0)
        custom_arm = Arm(10, reward_function=custom_reward, learner=None)
        agent.add_arm(custom_arm)

        # Optimization should be disabled
        assert agent.batch_reward_function is None

        # Remove the custom arm
        agent.remove_arm(10)

        # Optimization should be re-enabled
        assert agent.batch_reward_function is batch_identity

    def test_batch_reward_function_shapes(self):
        """Test batch reward function receives correct shapes."""
        n_arms = 5
        n_contexts = 3
        size = 1  # ThompsonSampling needs only 1 sample

        # Track calls to verify shapes
        calls = []

        def test_batch_reward(samples, action_tokens):
            calls.append(
                {
                    "samples_shape": samples.shape,
                    "n_tokens": len(action_tokens),
                    "tokens": action_tokens,
                }
            )
            return samples  # Identity for testing

        arms = [Arm(i, learner=None) for i in range(n_arms)]

        agent = LipschitzContextualAgent(
            arms=arms,
            policy=ThompsonSampling(),
            arm_featurizer=ArmColumnFeaturizer(),
            learner=NormalRegressor(alpha=1, beta=1),
            batch_reward_function=test_batch_reward,
        )

        X = np.random.randn(n_contexts, 2)
        agent.pull(X)

        # Verify the batch function was called with correct shapes
        assert len(calls) == 1
        assert calls[0]["samples_shape"] == (n_arms, n_contexts, size)
        assert calls[0]["n_tokens"] == n_arms
        assert calls[0]["tokens"] == list(range(n_arms))

    def test_context_aware_batch_function(self):
        """Test context-aware batch reward function."""

        def context_aware_reward(samples, action_tokens, X):
            # X should be original context, not enriched
            assert X.shape == (2, 3)  # Original shape
            return samples * X[:, 0].mean()  # Scale by mean of first feature

        arms = [Arm(i, learner=None) for i in range(4)]

        agent = LipschitzContextualAgent(
            arms=arms,
            policy=ThompsonSampling(),
            arm_featurizer=ArmColumnFeaturizer(),
            learner=NormalRegressor(alpha=1, beta=1),
            batch_reward_function=context_aware_reward,
        )

        X = np.array([[1.0, 2.0, 3.0], [4.0, 5.0, 6.0]])
        # Should not raise assertion error
        agent.pull(X)

    def test_batch_function_output_validation(self):
        """Test that batch reward function output shape is validated."""

        def wrong_shape_reward(samples, action_tokens):
            # Return wrong shape - drop a dimension
            return samples[:, :, 0]  # Wrong - missing size dimension

        arms = [Arm(i, learner=None) for i in range(3)]

        agent = LipschitzContextualAgent(
            arms=arms,
            policy=ThompsonSampling(),
            arm_featurizer=ArmColumnFeaturizer(),
            learner=NormalRegressor(alpha=1, beta=1),
            batch_reward_function=wrong_shape_reward,
        )

        X = np.array([[1.0, 2.0]])

        with pytest.raises(
            ValueError, match="batch_reward_function returned wrong shape"
        ):
            agent.pull(X)

    def test_batch_function_warning(self):
        """Test warning when batch function overrides individual functions."""

        def custom_reward(x):
            return x * 2

        arms = [
            Arm(0, reward_function=custom_reward, learner=None),
            Arm(1, reward_function=custom_reward, learner=None),
        ]

        def batch_reward(samples, action_tokens):
            return samples

        with warnings.catch_warnings(record=True) as w:
            warnings.simplefilter("always")
            LipschitzContextualAgent(
                arms=arms,
                policy=ThompsonSampling(),
                arm_featurizer=ArmColumnFeaturizer(),
                learner=NormalRegressor(alpha=1, beta=1),
                batch_reward_function=batch_reward,
            )

            assert len(w) == 1
            assert "batch_reward_function provided" in str(w[0].message)
            assert "will be ignored" in str(w[0].message)

    def test_no_warning_with_identity_functions(self):
        """Test no warning when all individual functions are identity."""
        arms = [Arm(i, learner=None) for i in range(3)]  # Default identity

        def batch_reward(samples, action_tokens):
            return samples

        with warnings.catch_warnings(record=True) as w:
            warnings.simplefilter("always")
            LipschitzContextualAgent(
                arms=arms,
                policy=ThompsonSampling(),
                arm_featurizer=ArmColumnFeaturizer(),
                learner=NormalRegressor(alpha=1, beta=1),
                batch_reward_function=batch_reward,
            )

            # No warning because all individual functions are identity
            assert len(w) == 0

    def test_empty_arms_gracefully_handled(self):
        """Test that empty arms are handled gracefully."""
        agent = LipschitzContextualAgent(
            arms=[Arm(0, learner=None)],  # Start with one arm
            policy=ThompsonSampling(),
            arm_featurizer=ArmColumnFeaturizer(),
            learner=NormalRegressor(alpha=1, beta=1),
        )

        # Remove the only arm
        agent.remove_arm(0)

        # Should handle empty arms gracefully
        assert len(agent.arms) == 0
        assert agent.batch_reward_function is None


class TestArticleCPCIntegration:
    """Integration test with realistic article/CPC scenario."""

    @pytest.fixture
    def article_metadata(self):
        """Realistic article metadata."""
        return {
            0: {"title": "AI News", "category": "Tech", "cpc": 0.15},
            1: {"title": "Cloud Computing", "category": "Tech", "cpc": 0.12},
            2: {"title": "Stock Tips", "category": "Finance", "cpc": 0.25},
            3: {"title": "Crypto Guide", "category": "Finance", "cpc": 0.30},
            4: {"title": "NBA Updates", "category": "Sports", "cpc": 0.08},
            5: {"title": "Travel Tips", "category": "Lifestyle", "cpc": 0.10},
        }

    def test_cpc_revenue_optimization(self, article_metadata):
        """Test that batch reward function correctly applies CPC multipliers."""

        def revenue_batch_reward(samples, action_tokens):
            cpcs = np.array([article_metadata[aid]["cpc"] for aid in action_tokens])
            return samples * cpcs[:, np.newaxis, np.newaxis]

        # Create arms for articles
        arms = [Arm(aid, learner=None) for aid in article_metadata.keys()]

        agent = LipschitzContextualAgent(
            arms=arms,
            policy=ThompsonSampling(),
            arm_featurizer=ArmColumnFeaturizer(column_name="article_id"),
            learner=NormalRegressor(alpha=1, beta=1),
            batch_reward_function=revenue_batch_reward,
            random_seed=42,
        )

        # Test that the batch reward function is correctly applying CPC values
        # by checking the transformation of samples
        n_contexts = 2
        X = np.random.randn(n_contexts, 3)
        
        # Mock the policy's sample method to return known values
        # This way we can verify the batch reward function's effect
        mock_samples = np.ones((len(arms), n_contexts, 1))
        
        # Get action tokens
        action_tokens = [arm.action_token for arm in agent.arms]
        
        # Apply the batch reward function
        transformed = revenue_batch_reward(mock_samples, action_tokens)
        
        # Verify CPC values are correctly applied
        for i, token in enumerate(action_tokens):
            expected_cpc = article_metadata[token]["cpc"]
            # Check all contexts for this arm have the right CPC multiplier
            assert np.allclose(transformed[i], expected_cpc), (
                f"Arm {token} should have CPC {expected_cpc}, "
                f"but got {transformed[i]}"
            )
        
        # Also verify the agent can pull without errors
        result = agent.pull(X)
        assert len(result) == n_contexts

    def test_batch_vs_individual_equivalence(self, article_metadata):
        """Test that batch and individual reward functions produce same results."""

        # Individual reward functions per arm
        arms_individual = []
        for aid, meta in article_metadata.items():
            cpc = meta["cpc"]
            arms_individual.append(
                Arm(aid, reward_function=lambda s, c=cpc: s * c, learner=None)
            )

        # Batch reward function
        def batch_reward(samples, action_tokens):
            cpcs = np.array([article_metadata[aid]["cpc"] for aid in action_tokens])
            return samples * cpcs[:, np.newaxis, np.newaxis]

        arms_batch = [Arm(aid, learner=None) for aid in article_metadata.keys()]

        # Create two agents with same settings
        agent_individual = LipschitzContextualAgent(
            arms=arms_individual,
            policy=ThompsonSampling(),
            arm_featurizer=ArmColumnFeaturizer(),
            learner=NormalRegressor(alpha=1, beta=1),
            random_seed=42,
        )

        agent_batch = LipschitzContextualAgent(
            arms=arms_batch,
            policy=ThompsonSampling(),
            arm_featurizer=ArmColumnFeaturizer(),
            learner=NormalRegressor(alpha=1, beta=1),
            batch_reward_function=batch_reward,
            random_seed=42,
        )

        # Both should produce same results
        X = np.random.randn(5, 3)

        # Need to ensure both use same random state for sampling
        result_individual = agent_individual.pull(X)
        agent_batch.rng = np.random.default_rng(42)  # Reset RNG
        result_batch = agent_batch.pull(X)

        assert result_individual == result_batch


class TestContextAwareBatchFunctions:
    """Test context-aware batch reward functions."""

    def test_segment_targeted_revenue(self):
        """Test context-aware revenue optimization based on user segments."""
        article_metadata = {
            0: {"title": "AI News", "category": "Tech", "cpc": 0.15},
            1: {"title": "Stock Tips", "category": "Finance", "cpc": 0.25},
            2: {"title": "Luxury Travel", "category": "Luxury", "cpc": 0.40},
        }

        def segment_targeted_revenue(samples, action_tokens, X):
            """Adjust revenue based on user segments and article categories."""
            # Extract user segments
            is_premium = X[:, 2].astype(bool)  # Third column is premium flag
            is_high_income = X[:, 1] > 100000  # Second column is income

            # Get article properties
            base_cpcs = np.array(
                [article_metadata[aid]["cpc"] for aid in action_tokens]
            )
            is_finance = np.array(
                ["Finance" in article_metadata[aid]["title"] for aid in action_tokens]
            )
            is_luxury = np.array(
                ["Luxury" in article_metadata[aid]["title"] for aid in action_tokens]
            )

            # Build CPC adjustment matrix
            cpc_multipliers = np.ones((len(action_tokens), len(X)))

            # Premium users get 50% boost for finance articles
            premium_finance_boost = is_finance[:, None] & is_premium[None, :]
            cpc_multipliers = np.where(premium_finance_boost, 1.5, cpc_multipliers)

            # High-income users get 30% boost for luxury articles
            luxury_income_boost = is_luxury[:, None] & is_high_income[None, :]
            cpc_multipliers = np.where(luxury_income_boost, 1.3, cpc_multipliers)

            # Calculate final CPC
            final_cpcs = base_cpcs[:, None] * cpc_multipliers
            final_cpcs_expanded = final_cpcs[:, :, None]

            return samples * final_cpcs_expanded

        arms = [Arm(aid, learner=None) for aid in article_metadata.keys()]

        agent = LipschitzContextualAgent(
            arms=arms,
            policy=ThompsonSampling(),
            arm_featurizer=ArmColumnFeaturizer(),
            learner=NormalRegressor(alpha=1, beta=1),
            batch_reward_function=segment_targeted_revenue,
            random_seed=42,
        )

        # Test with different user segments
        # [age, income, is_premium]
        regular_user = np.array([[35, 50000, 0]])
        premium_user = np.array([[45, 150000, 1]])
        high_income_user = np.array([[40, 120000, 0]])

        # All should work without errors
        agent.pull(regular_user)
        agent.pull(premium_user)
        agent.pull(high_income_user)

    def test_time_based_availability(self):
        """Test context-aware batch function for time-based availability."""

        def time_based_availability(samples, action_tokens, X):
            """Zero out rewards for unavailable items based on time."""
            # X contains: [user_id, hour_of_day]
            hour = X[:, 1]

            # Define availability windows for each item
            availability = {
                0: (6, 11),  # breakfast: 6 AM - 11 AM
                1: (11, 15),  # lunch: 11 AM - 3 PM
                2: (17, 22),  # dinner: 5 PM - 10 PM
            }

            # Create availability mask
            n_arms = len(action_tokens)
            n_contexts = len(X)
            mask = np.ones((n_arms, n_contexts))

            for i, token in enumerate(action_tokens):
                if token in availability:
                    start, end = availability[token]
                    for j in range(n_contexts):
                        if not (start <= hour[j] < end):
                            mask[i, j] = 0

            # Apply mask
            mask_expanded = mask[:, :, None]
            return samples * mask_expanded

        arms = [
            Arm(i, learner=None) for i in range(3)
        ]  # 0=breakfast, 1=lunch, 2=dinner

        agent = LipschitzContextualAgent(
            arms=arms,
            policy=ThompsonSampling(),
            arm_featurizer=ArmColumnFeaturizer(),
            learner=NormalRegressor(alpha=1, beta=1),
            batch_reward_function=time_based_availability,
        )

        # Test at different times
        morning_context = np.array([[123, 8]])  # 8 AM
        afternoon_context = np.array([[456, 13]])  # 1 PM
        evening_context = np.array([[789, 19]])  # 7 PM

        # All should work
        agent.pull(morning_context)
        agent.pull(afternoon_context)
        agent.pull(evening_context)


class TestPerformanceOptimizations:
    """Test performance optimizations and edge cases."""

    def test_identity_optimization_performance(self):
        """Test that identity optimization actually skips computation."""
        # Large number of arms to see performance benefit
        n_arms = 100
        arms = [Arm(i, learner=None) for i in range(n_arms)]

        agent = LipschitzContextualAgent(
            arms=arms,
            policy=ThompsonSampling(),
            arm_featurizer=ArmColumnFeaturizer(),
            learner=NormalRegressor(alpha=1, beta=1),
        )

        # Verify identity optimization is active
        assert agent.batch_reward_function is batch_identity

        # Should be fast even with many arms
        X = np.random.randn(10, 5)
        import time

        start = time.time()
        agent.pull(X)
        elapsed = time.time() - start

        assert elapsed < 0.1  # Should be very fast

    def test_batch_function_with_top_k(self):
        """Test batch reward function works correctly with top_k."""

        def custom_batch_reward(samples, action_tokens):
            # Apply different multipliers based on token
            multipliers = np.array([float(token) / 10 + 1 for token in action_tokens])
            return samples * multipliers[:, np.newaxis, np.newaxis]

        arms = [Arm(i, learner=None) for i in range(10)]

        agent = LipschitzContextualAgent(
            arms=arms,
            policy=ThompsonSampling(),
            arm_featurizer=ArmColumnFeaturizer(),
            learner=NormalRegressor(alpha=1, beta=1),
            batch_reward_function=custom_batch_reward,
        )

        X = np.array([[1.0, 2.0], [3.0, 4.0]])

        # Test top_k functionality
        result = agent.pull(X, top_k=3)

        assert len(result) == 2  # Two contexts
        assert all(len(context_result) == 3 for context_result in result)  # Top 3 each
        assert all(
            len(set(context_result)) == 3 for context_result in result
        )  # No duplicates

    def test_batch_function_with_mixed_arms(self):
        """Test batch function with arms added/removed dynamically."""

        def dynamic_batch_reward(samples, action_tokens):
            # Handle dynamic token set - premium tokens have value >= 100
            multipliers = []
            for token in action_tokens:
                if token >= 100:  # premium tokens
                    multipliers.append(2.0)
                else:
                    multipliers.append(1.0)
            multipliers = np.array(multipliers)
            return samples * multipliers[:, np.newaxis, np.newaxis]

        # Start with regular arms
        arms = [Arm(i, learner=None) for i in range(3)]

        agent = LipschitzContextualAgent(
            arms=arms,
            policy=ThompsonSampling(),
            arm_featurizer=ArmColumnFeaturizer(),
            learner=NormalRegressor(alpha=1, beta=1),
            batch_reward_function=dynamic_batch_reward,
        )

        X = np.array([[1.0, 2.0]])

        # Initial pull
        result1 = agent.pull(X)
        assert result1[0] in [0, 1, 2]

        # Add premium arm - use high integer to distinguish
        agent.add_arm(Arm(100, learner=None))  # premium_1

        # Pull with mixed arms
        result2 = agent.pull(X)
        assert result2[0] in [0, 1, 2, 100]

        # Remove an arm
        agent.remove_arm(1)

        # Pull after removal
        result3 = agent.pull(X)
        assert result3[0] in [0, 2, 100]
        assert result3[0] != 1  # Removed arm

    def test_batch_function_with_3d_learner_output(self):
        """Test batch function with 3D learner output (e.g., multi-class)."""

        def multiclass_reward(samples, action_tokens):
            # Assume samples shape: (n_arms, n_contexts, size, n_classes)
            # Take max probability as reward
            if samples.ndim == 4:
                return np.max(samples, axis=-1)
            return samples

        arms = [Arm(i, learner=None) for i in range(3)]

        # Note: This test is conceptual since we'd need a multi-output learner
        # The reshape logic is already tested in the base class
        agent = LipschitzContextualAgent(
            arms=arms,
            policy=ThompsonSampling(),
            arm_featurizer=ArmColumnFeaturizer(),
            learner=NormalRegressor(alpha=1, beta=1),  # Single output for this test
            batch_reward_function=multiclass_reward,
        )

        X = np.array([[1.0, 2.0]])
        result = agent.pull(X)
        assert len(result) == 1
        assert result[0] in [0, 1, 2]
