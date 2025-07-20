"""Test the batch reward function examples from the documentation."""

import numpy as np

from bayesianbandits import (
    Arm,
    LipschitzContextualAgent,
    ThompsonSampling,
    ArmColumnFeaturizer,
    NormalRegressor,
)


class TestDocumentationExamples:
    """Test that the examples in the documentation actually work."""

    def test_basic_revenue_batch_reward(self):
        """Test the basic revenue batch reward function from docs."""
        # Pre-compute revenue array for all products (vectorized approach)
        n_products = 100
        product_revenues = np.random.uniform(
            0.5, 3.0, n_products
        )  # Revenue per product

        # Create vectorized batch reward function
        def revenue_batch_reward(samples, action_tokens):
            # Direct numpy indexing - fully vectorized
            multipliers = product_revenues[action_tokens]
            # Broadcast to match samples shape: (n_arms, n_contexts, size)
            return samples * multipliers[:, np.newaxis, np.newaxis]

        # Create arms
        arms = [Arm(i, learner=None) for i in range(n_products)]

        # Create agent with batch reward function
        agent = LipschitzContextualAgent(
            arms=arms,
            policy=ThompsonSampling(),
            arm_featurizer=ArmColumnFeaturizer(column_name="product_id"),
            learner=NormalRegressor(alpha=1, beta=1),
            batch_reward_function=revenue_batch_reward,
            random_seed=42,
        )

        # Test pulling
        X = np.array([[25, 50000], [35, 75000]])  # age, income
        selected_products = agent.pull(X)

        assert len(selected_products) == 2
        assert all(0 <= p < n_products for p in selected_products)

        # Test the batch reward function directly
        test_samples = np.ones((5, 2, 1))  # 5 arms, 2 contexts, size=1
        test_tokens = [0, 10, 20, 30, 40]
        result = revenue_batch_reward(test_samples, test_tokens)

        # Check shape
        assert result.shape == (5, 2, 1)

        # Check values - should be samples * revenue
        for i, token in enumerate(test_tokens):
            expected = product_revenues[token]
            assert np.allclose(result[i], expected)

    def test_gross_profit_batch_reward(self):
        """Test the gross profit context-aware batch reward function from docs."""
        # Context-aware: calculate gross profit from prices, costs, and taxes
        # Arms represent different price points
        price_points = np.array([9.99, 14.99, 19.99, 24.99, 29.99])
        arms = [Arm(i, learner=None) for i in range(len(price_points))]

        def gross_profit_reward(samples, action_tokens, X):
            # X contains: [customer_value, cost_per_unit, tax_rate]
            costs = X[:, 1]  # shape: (n_contexts,)
            tax_rates = X[:, 2]  # shape: (n_contexts,)

            # Get prices for selected arms
            prices = price_points[action_tokens]  # shape: (n_arms,)

            # Vectorized profit calculation for all (arm, context) pairs
            # Revenue after tax: price * (1 - tax_rate)
            # Gross profit: revenue_after_tax - cost
            revenue_after_tax = prices[:, np.newaxis] * (1 - tax_rates[np.newaxis, :])
            gross_profit = revenue_after_tax - costs[np.newaxis, :]

            # Apply profit multiplier to samples, clamping negative profits to 0
            profit_multiplier = np.maximum(gross_profit, 0)
            return samples * profit_multiplier[:, :, np.newaxis]

        # Create agent
        agent = LipschitzContextualAgent(
            arms=arms,
            policy=ThompsonSampling(),
            arm_featurizer=ArmColumnFeaturizer(),
            learner=NormalRegressor(alpha=1, beta=1),
            batch_reward_function=gross_profit_reward,
            random_seed=42,
        )

        # Test with different contexts
        X = np.array(
            [
                [100, 5.0, 0.1],  # customer_value=100, cost=5, tax=10%
                [200, 8.0, 0.2],  # customer_value=200, cost=8, tax=20%
                [50, 15.0, 0.05],  # customer_value=50, cost=15, tax=5%
            ]
        )

        selected_prices = agent.pull(X)
        assert len(selected_prices) == 3
        assert all(0 <= p < len(price_points) for p in selected_prices)

        # Test the batch reward function directly
        test_samples = np.ones((len(price_points), 3, 1))
        test_tokens = list(range(len(price_points)))
        result = gross_profit_reward(test_samples, test_tokens, X)

        # Check shape
        assert result.shape == (5, 3, 1)

        # Manually verify some calculations
        # For price 9.99, cost 5.0, tax 10%:
        # revenue_after_tax = 9.99 * (1 - 0.1) = 8.991
        # gross_profit = 8.991 - 5.0 = 3.991
        expected_profit_0_0 = 9.99 * (1 - 0.1) - 5.0
        assert np.isclose(result[0, 0, 0], expected_profit_0_0)

        # For price 29.99, cost 15.0, tax 5%:
        # revenue_after_tax = 29.99 * (1 - 0.05) = 28.4905
        # gross_profit = 28.4905 - 15.0 = 13.4905
        expected_profit_4_2 = 29.99 * (1 - 0.05) - 15.0
        assert np.isclose(result[4, 2, 0], expected_profit_4_2)

        # Check that negative profits are clamped to 0
        # High cost context where some prices would yield negative profit
        X_high_cost = np.array([[100, 25.0, 0.1]])  # cost > most prices after tax
        result_high_cost = gross_profit_reward(
            test_samples[:, :1, :], test_tokens, X_high_cost
        )

        # Lower prices should be clamped to 0
        for i in range(3):  # First 3 prices are too low
            revenue_after_tax = price_points[i] * (1 - 0.1)
            if revenue_after_tax < 25.0:
                assert result_high_cost[i, 0, 0] == 0.0

    def test_gross_profit_edge_cases(self):
        """Test edge cases for the gross profit function."""
        price_points = np.array([10.0, 20.0, 30.0])

        def gross_profit_reward(samples, action_tokens, X):
            costs = X[:, 1]
            tax_rates = X[:, 2]
            prices = price_points[action_tokens]
            revenue_after_tax = prices[:, np.newaxis] * (1 - tax_rates[np.newaxis, :])
            gross_profit = revenue_after_tax - costs[np.newaxis, :]
            profit_multiplier = np.maximum(gross_profit, 0)
            return samples * profit_multiplier[:, :, np.newaxis]

        # Test with 100% tax (no revenue)
        X_full_tax = np.array([[100, 5.0, 1.0]])  # 100% tax
        samples = np.ones((3, 1, 1))
        result = gross_profit_reward(samples, [0, 1, 2], X_full_tax)
        assert np.all(result == 0)  # All profits should be 0

        # Test with zero cost (maximum profit)
        X_zero_cost = np.array([[100, 0.0, 0.1]])  # No cost, 10% tax
        result = gross_profit_reward(samples, [0, 1, 2], X_zero_cost)
        for i in range(3):
            expected = price_points[i] * 0.9  # 90% of price
            assert np.isclose(result[i, 0, 0], expected)
