"""
Test suite for EXP3.A (Average-based anytime EXP3) algorithm.
"""

from typing import Any, List

import numpy as np
import pytest
from numpy.typing import NDArray
from scipy.sparse import csc_array  # type: ignore

from bayesianbandits import (
    EXP3A,
    Arm,
    ContextualAgent,
    NormalInverseGammaRegressor,
    NormalRegressor,
)


class TestEXP3ABasics:
    """Test basic functionality of EXP3A."""

    def test_initialization(self):
        """Test EXP3A initialization with different parameters."""
        # Default initialization
        policy = EXP3A()
        assert policy.gamma == 0.0
        assert policy.eta == 1.0
        assert policy.ix_gamma == 0.5
        assert policy.samples == 100

        # Custom initialization
        policy = EXP3A(gamma=0.2, eta=2.0, samples=50)
        assert policy.gamma == 0.2
        assert policy.eta == 2.0
        assert policy.samples == 50

    def test_gamma_less_than_zero(self):
        """Test that gamma cannot be negative."""
        with pytest.raises(ValueError, match="gamma must be non-negative"):
            EXP3A(gamma=-0.1)

    def test_eta_less_than_zero(self):
        """Test that eta cannot be negative."""
        with pytest.raises(ValueError, match="eta must be positive"):
            EXP3A(eta=-0.1)

    def test_ix_gamma_less_than_zero(self):
        """Test that ix_gamma cannot be negative."""
        with pytest.raises(ValueError, match="ix_gamma must be non-negative"):
            EXP3A(ix_gamma=-0.1)

    def test_repr(self):
        """Test string representation."""
        policy = EXP3A(gamma=0.15, eta=1.5, samples=200)
        assert repr(policy) == "EXP3A(gamma=0.15, eta=1.5, ix_gamma=0.75, samples=200)"


class TestEXP3AArmSelection:
    """Test arm selection mechanics with real arms."""

    def test_basic_selection(self):
        """Test basic arm selection favors better arms."""
        # Create arms with different priors to simulate different quality
        arms: List[Arm[NDArray[Any], str]] = []
        for i, (mu, precision) in enumerate([(0.2, 10), (0.5, 10), (0.8, 10)]):
            learner = NormalInverseGammaRegressor(mu=mu, lam=precision, a=5, b=1)
            arms.append(Arm(f"arm_{i}", learner=learner))

        policy = EXP3A(gamma=0.1, eta=2.0, samples=100)
        agent = ContextualAgent(arms, policy, random_seed=42)

        X = np.array([[1.0]])

        # Select arms multiple times
        selections: List[int] = []
        for _ in range(1000):
            action = agent.pull(X)
            # Map action token back to index
            idx = int(action[0].split("_")[1])
            selections.append(idx)
            # Don't update - just test selection based on priors

        # Best arm (index 2) should be selected most often
        counts = np.bincount(selections)
        assert counts[2] > counts[1] > counts[0]

        # But all arms should be selected at least gamma * n_trials / n_arms times
        min_selections = 0.1 * 1000 / 3 * 0.5  # With some margin
        assert all(c >= min_selections for c in counts)

    def test_forced_exploration(self):
        """Test that gamma parameter enforces minimum exploration."""
        # Create arms with very different priors
        arms: List[Arm[NDArray[Any], int]] = []
        for i, mu in enumerate([0.0, 0.0, 1.0]):
            learner = NormalInverseGammaRegressor(mu=mu, lam=100, a=10, b=1)
            arms.append(Arm(i, learner=learner))

        # High gamma forces exploration
        policy = EXP3A(gamma=0.3, eta=5.0)
        agent = ContextualAgent(arms, policy, random_seed=42)

        X = np.array([[1.0]])

        selections: List[int] = []
        for _ in range(1000):
            action = agent.pull(X)
            selections.append(action[0])

        counts = [selections.count(i) for i in range(3)]
        # Each arm should get at least gamma/K fraction
        min_prob = 0.3 / 3
        for count in counts:
            assert count >= min_prob * 1000 * 0.8  # 80% margin

    def test_temperature_effect(self):
        """Test that eta parameter controls selection sharpness."""
        # Arms with similar but different priors
        arms: List[Arm[NDArray[Any], int]] = []
        for i, mu in enumerate([0.4, 0.5, 0.6]):
            learner = NormalInverseGammaRegressor(mu=mu, lam=50, a=5, b=1)
            arms.append(Arm(i, learner=learner))

        # Low temperature - more uniform selection
        policy_low_temp = EXP3A(gamma=0.0, eta=0.1)
        agent_low = ContextualAgent(arms, policy_low_temp, random_seed=42)

        # High temperature - sharper selection
        policy_high_temp = EXP3A(gamma=0.0, eta=10.0)
        agent_high = ContextualAgent(arms, policy_high_temp, random_seed=43)

        X = np.array([[1.0]])

        # Collect selections
        low_temp_selections: List[int] = []
        high_temp_selections: List[int] = []
        for _ in range(1000):
            low_temp_selections.append(agent_low.pull(X)[0])
            high_temp_selections.append(agent_high.pull(X)[0])

        # High temperature should have more concentrated selections
        low_temp_counts = np.bincount(low_temp_selections)
        high_temp_counts = np.bincount(high_temp_selections)

        # Variance in selection counts should be higher for high temperature
        assert np.var(high_temp_counts) > np.var(low_temp_counts)

    def test_multiple_contexts(self):
        """Test selection with multiple contexts at once."""
        # Create arms with context-dependent rewards
        arms: List[Arm[NDArray[Any], int]] = []
        for i in range(3):
            learner = NormalRegressor(alpha=1.0, beta=1.0)
            # Initialize with different slopes
            X_init = np.array([[1.0], [2.0], [3.0]])
            y_init = np.array([0.1 * i, 0.5 * i, 0.9 * i])
            learner.fit(X_init, y_init)
            arms.append(Arm(i, learner=learner))

        policy = EXP3A(gamma=0.1, eta=2.0, samples=50)
        agent = ContextualAgent(arms, policy, random_seed=42)

        X = np.array([[1.0], [2.0], [3.0]])

        selected = agent.pull(X)
        assert len(selected) == 3
        assert all(action in range(3) for action in selected)


class TestEXP3AUpdates:
    """Test update mechanics and importance weighting with real arms."""

    def test_importance_weighting(self):
        """Test that importance weights are computed correctly."""
        # Create arms with known differences
        arm_bad = Arm(0, learner=NormalInverseGammaRegressor(mu=0.2, lam=100))
        arm_good = Arm(1, learner=NormalInverseGammaRegressor(mu=0.8, lam=100))
        arms = [arm_bad, arm_good]

        policy = EXP3A(gamma=0.1, eta=2.0, ix_gamma=0.0, samples=50)
        agent = ContextualAgent(arms, policy, random_seed=42)

        X = np.array([[1.0]])
        y = np.array([1.0])

        # Track the actual sample weights used
        original_update = arm_bad.learner.partial_fit  # type: ignore
        weights_bad: List[NDArray[np.float64]] = []

        def track_weights_bad(
            X: NDArray[Any] | csc_array,
            y: NDArray[np.float64],
            sample_weight: NDArray[np.float64] | None = None,
        ):
            if sample_weight is not None:
                weights_bad.append(sample_weight[0])
            return original_update(X, y, sample_weight)

        arm_bad.learner.partial_fit = track_weights_bad  # type: ignore

        # Update the bad arm multiple times
        for _ in range(10):
            agent.select_for_update(0).update(X, y)

        # The importance weight should be > 1 for the less likely arm
        assert all(w > 1.0 for w in weights_bad)

        # Now track weights for good arm
        original_update_good = arm_good.learner.partial_fit  # type: ignore
        weights_good: List[NDArray[np.float64]] = []

        def track_weights_good(
            X: NDArray[Any] | csc_array,
            y: NDArray[np.float64],
            sample_weight: NDArray[np.float64] | None = None,
        ):
            if sample_weight is not None:
                weights_good.append(sample_weight[0])
            return original_update_good(X, y, sample_weight)

        arm_good.learner.partial_fit = track_weights_good  # type: ignore

        # Update the good arm
        for _ in range(10):
            agent.select_for_update(1).update(X, y)

        # Better arm should have lower importance weight on average
        assert np.mean(weights_good) < np.mean(weights_bad)

    def test_update_preserves_unbiasedness(self):
        """Test that importance weighting preserves unbiased estimates."""
        # Create arms with true Bernoulli rewards
        true_means = [0.3, 0.5, 0.7]
        arms: List[Arm[NDArray[Any], int]] = []

        for i, _ in enumerate(true_means):
            # Start with uninformative prior
            learner = NormalInverseGammaRegressor(mu=0.5, lam=0.1, a=1, b=1)
            arms.append(Arm(i, learner=learner))

        policy = EXP3A(gamma=0.2, eta=1.0, samples=50)
        agent = ContextualAgent(arms, policy, random_seed=42)

        X = np.array([[1.0]])

        # Run many rounds
        for _ in range(2000):
            action = agent.pull(X)
            arm_idx = action[0]

            # Generate Bernoulli reward based on true mean
            reward = float(np.random.random() < true_means[arm_idx])
            agent.update(X, np.array([reward]))

        # Check that learned means converge to true means
        for i, (arm, true_mean) in enumerate(zip(arms, true_means)):
            assert arm.learner is not None, f"Arm {i} has no learner"
            learned_mean = arm.learner.predict(X)[0]
            # Should be close to true mean (within statistical error)
            assert abs(learned_mean - true_mean) < 0.1, (
                f"Arm {i}: learned {learned_mean:.3f} vs true {true_mean}"
            )


class TestEXP3AEdgeCases:
    """Test edge cases and numerical stability."""

    def test_single_arm(self):
        """Test with only one arm."""
        arm = Arm(0, learner=NormalInverseGammaRegressor())

        policy = EXP3A(gamma=0.1, eta=1.0)
        agent = ContextualAgent([arm], policy, random_seed=42)

        X = np.array([[1.0]])

        # Should always select the only arm
        for _ in range(10):
            action = agent.pull(X)
            assert action[0] == 0

        # Update should work normally
        agent.update(X, np.array([1.0]))

    def test_extreme_eta(self):
        """Test numerical stability with extreme eta values."""
        arms: List[Arm[NDArray[Any], int]] = []
        for i, mu in enumerate([0.1, 0.2, 0.9]):
            learner = NormalInverseGammaRegressor(mu=mu, lam=100)
            arms.append(Arm(i, learner=learner))

        # Very high eta
        policy_high = EXP3A(gamma=0.01, eta=100.0, samples=20)
        agent_high = ContextualAgent(arms, policy_high, random_seed=42)

        X = np.array([[1.0]])

        # Should not crash and should heavily favor best arm
        selections: List[int] = []
        for _ in range(100):
            action = agent_high.pull(X)
            selections.append(action[0])

        assert max(set(selections), key=selections.count) == 2

        # Very low eta - should be nearly uniform
        policy_low = EXP3A(gamma=0.0, eta=0.001, samples=20)
        agent_low = ContextualAgent(arms, policy_low, random_seed=42)

        selections_low: List[int] = []
        for _ in range(300):
            action = agent_low.pull(X)
            selections_low.append(action[0])

        counts = np.bincount(selections_low)
        # Should be relatively uniform
        assert max(counts) / min(counts) < 2.0

    def test_zero_gamma(self):
        """Test with gamma=0 (no forced exploration)."""
        arms: List[Arm[NDArray[Any], int]] = []
        for i, mu in enumerate([0.1, 0.9]):
            # Need higher 'a' for more concentrated prior (less heavy tails)
            learner = NormalInverseGammaRegressor(mu=mu, lam=100, a=10, b=1)
            arms.append(Arm(i, learner=learner))

        policy = EXP3A(gamma=0.0, eta=5.0, samples=50)
        agent = ContextualAgent(arms, policy, random_seed=42)

        X = np.array([[1.0]])

        # With no exploration, should strongly favor better arm
        selections: List[int] = []
        for _ in range(100):
            action = agent.pull(X)
            selections.append(action[0])

        # Almost all selections should be arm 1
        assert selections.count(1) > 95

    def test_all_arms_equal(self):
        """Test when all arms have equal expected rewards."""
        arms: List[Arm[NDArray[Any], int]] = []
        for i in range(4):
            learner = NormalInverseGammaRegressor(mu=0.5, lam=100)
            arms.append(Arm(i, learner=learner))

        policy = EXP3A(gamma=0.1, eta=1.0)
        agent = ContextualAgent(arms, policy, random_seed=42)

        X = np.array([[1.0]])

        # Should select roughly uniformly
        selections: List[int] = []
        for _ in range(400):
            action = agent.pull(X)
            selections.append(action[0])

        counts = np.bincount(selections)
        # Each arm should get roughly 1/4 of selections
        for count in counts:
            assert 70 < count < 130  # Allowing for randomness


class TestEXP3ANonStationarity:
    """Test behavior in non-stationary environments."""

    def test_adapts_to_change(self):
        """Test that EXP3A adapts when rewards change."""
        # Create two arms
        arms = [
            Arm(0, learner=NormalInverseGammaRegressor(learning_rate=0.9)),
            Arm(1, learner=NormalInverseGammaRegressor(learning_rate=0.9)),
        ]

        policy = EXP3A(gamma=0.1, eta=2.0, samples=20)
        agent = ContextualAgent(arms, policy, random_seed=42)

        X = np.array([[1.0]])

        # First phase: arm 1 is better
        for _ in range(100):
            action = agent.pull(X)
            arm_idx = action[0]
            # Arm 0 gets reward 0.2, arm 1 gets reward 0.8
            reward = 0.2 if arm_idx == 0 else 0.8
            agent.update(X, np.array([reward]))

        # Check arm 1 is preferred
        phase1_selections: List[int] = []
        for _ in range(50):
            action = agent.pull(X)
            phase1_selections.append(action[0])

        assert phase1_selections.count(1) > phase1_selections.count(0)

        # Second phase: reverse the rewards
        for _ in range(200):
            action = agent.pull(X)
            arm_idx = action[0]
            # Now arm 0 gets reward 0.9, arm 1 gets reward 0.1
            reward = 0.9 if arm_idx == 0 else 0.1
            agent.update(X, np.array([reward]))

        # Check that algorithm adapted
        phase2_selections: List[int] = []
        for _ in range(50):
            action = agent.pull(X)
            phase2_selections.append(action[0])

        # Should now prefer arm 0
        assert phase2_selections.count(0) > phase2_selections.count(1)


class TestEXP3AProperties:
    """Test theoretical properties of the algorithm."""

    def test_sublinear_regret(self):
        """Test that regret grows sublinearly (basic check)."""
        # Simple 2-arm bandit with separated rewards
        true_means = [0.3, 0.7]

        # Create arms with uninformative but not too diffuse priors
        arms: List[Arm[NDArray[Any], int]] = []
        for i in range(2):
            learner = NormalInverseGammaRegressor(mu=0.5, lam=1.0, a=2, b=2)
            arms.append(Arm(i, learner=learner))

        policy = EXP3A(eta=10.0, samples=100)
        agent = ContextualAgent(arms, policy, random_seed=42)

        X = np.array([[1.0]])

        # Track regret
        cumulative_regret: List[float] = []
        total_regret = 0.0
        optimal_mean = max(true_means)

        # Use a fixed random seed for reproducible rewards
        reward_rng = np.random.default_rng(123)

        for _ in range(2000):
            action = agent.pull(X)
            arm_idx = action[0]

            # Get actual reward (Bernoulli with true mean)
            reward = float(reward_rng.random() < true_means[arm_idx])

            # Calculate instant regret
            instant_regret = optimal_mean - true_means[arm_idx]
            total_regret += instant_regret
            cumulative_regret.append(total_regret)

            # Update
            agent.update(X, np.array([reward]))

        # Check that average regret decreases over time
        # (a necessary condition for sublinear regret)
        avg_regret_early = cumulative_regret[199] / 200
        avg_regret_mid = cumulative_regret[999] / 1000
        avg_regret_late = cumulative_regret[1999] / 2000

        # Should see consistent improvement
        assert avg_regret_mid < avg_regret_early
        assert avg_regret_late < avg_regret_mid * 0.95

        # Also check that total regret is sublinear
        # For sublinear regret, regret/t should decrease
        assert cumulative_regret[1999] / 2000 < cumulative_regret[999] / 1000

    def test_adversarial_robustness(self):
        """Test robustness against an adversarial reward sequence."""
        # Create 3 arms
        arms: List[Arm[NDArray[Any], int]] = []
        for i in range(3):
            learner = NormalInverseGammaRegressor(mu=0.5, lam=1.0, a=2, b=2)
            arms.append(Arm(i, learner=learner))

        policy = EXP3A(eta=2.0, samples=100)
        agent = ContextualAgent(arms, policy, random_seed=42)

        X = np.array([[1.0]])

        # Track which arm gets pulled
        pull_counts = [0, 0, 0]
        total_reward = 0.0

        # Adversarial strategy: give high reward to least-pulled arm
        for _ in range(500):
            action = agent.pull(X)
            arm_idx = action[0]
            pull_counts[arm_idx] += 1

            # Adversarial reward: penalize the most frequently pulled arm
            if arm_idx == np.argmax(pull_counts):
                reward = 0.0  # Bad reward for overused arm
            else:
                reward = 1.0  # Good reward for underused arms

            total_reward += reward
            agent.update(X, np.array([reward]))

        # Check that no arm dominates (adversary forces exploration)
        pull_proportions = np.array(pull_counts) / sum(pull_counts)
        assert np.max(pull_proportions) < 0.4  # No arm pulled > 40% of time
        assert np.min(pull_proportions) > 0.2  # All arms pulled > 20% of time

        # Should still achieve reasonable reward despite adversary
        assert (
            total_reward / 500 > 0.6
        )  # Randomly pulling each arm should yield ~0.67 average reward

    def test_adaptive_adversary(self):
        """Test against adversary that adapts to algorithm's beliefs."""
        arms = [Arm(i, learner=NormalInverseGammaRegressor()) for i in range(2)]

        policy = EXP3A(eta=1.0, ix_gamma=0.5)
        agent = ContextualAgent(arms, policy, random_seed=42)

        X = np.array([[1.0]])
        rewards_received: List[float] = []
        last_50_pulls: List[int] = []

        for t in range(200):
            beliefs = [arm.learner.predict(X)[0] for arm in arms]  # type: ignore

            action = agent.pull(X)
            arm_idx = action[0]

            if t >= 150:  # Track last 50 pulls
                last_50_pulls.append(arm_idx)

            # Adversary: make the arm that looks best actually give bad reward
            if arm_idx == np.argmax(beliefs):
                reward = 0.0
            else:
                reward = 1.0

            rewards_received.append(reward)
            agent.update(X, np.array([reward]))

        # Should converge to ~0.5 reward (adversary makes arms equivalent)
        assert (
            np.mean(rewards_received[-50:]) > 0.45
        )  # Randomly pulling should yield ~0.5 average reward

        # Should be exploring both arms in steady state
        last_50_proportions = np.bincount(last_50_pulls, minlength=2) / 50
        assert np.min(last_50_proportions) > 0.3  # Both arms pulled frequently

    def test_exp3a_top_k_return_type(self) -> None:
        """Test that EXP3A returns correct types with top_k."""
        arms = [Arm(i, None, learner=NormalInverseGammaRegressor()) for i in range(5)]
        policy = EXP3A(gamma=0.1, eta=1.0)
        agent = ContextualAgent(arms, policy, random_seed=42)
        X = np.array([[1.0], [2.0]])

        # Default behavior - returns List[TokenType]
        result_default = agent.pull(X)
        assert isinstance(result_default, list)
        assert len(result_default) == 2  # One per context
        assert all(isinstance(token, int) for token in result_default)

        # top_k=1 - returns List[List[TokenType]]
        result_k1 = agent.pull(X, top_k=1)
        assert isinstance(result_k1, list)
        assert len(result_k1) == 2  # One list per context
        assert all(isinstance(sublist, list) for sublist in result_k1)
        assert all(len(sublist) == 1 for sublist in result_k1)

        # top_k=3 - returns List[List[TokenType]]
        result_k3 = agent.pull(X, top_k=3)
        assert isinstance(result_k3, list)
        assert len(result_k3) == 2  # One list per context
        assert all(isinstance(sublist, list) for sublist in result_k3)
        assert all(len(sublist) == 3 for sublist in result_k3)

    def test_exp3a_no_duplicates(self) -> None:
        """Test that top_k doesn't select the same arm twice."""
        arms = [Arm(i, None, learner=NormalInverseGammaRegressor()) for i in range(10)]
        policy = EXP3A(gamma=0.1, eta=1.0)
        agent = ContextualAgent(arms, policy, random_seed=42)
        X = np.array([[1.0]] * 20)  # 20 contexts

        results = agent.pull(X, top_k=5)

        # Check no duplicates in any selection
        for result in results:
            assert len(result) == len(set(result))  # No duplicates
            assert len(result) == 5  # Correct number selected

    def test_exp3a_top_k_exceeds_arms(self) -> None:
        """Test behavior when top_k > number of arms."""
        arms = [Arm(i, None, learner=NormalInverseGammaRegressor()) for i in range(3)]
        policy = EXP3A(gamma=0.1, eta=1.0)
        agent = ContextualAgent(arms, policy, random_seed=42)
        X = np.array([[1.0]])

        # Request more arms than available
        results = agent.pull(X, top_k=5)

        # Should return all 3 arms
        assert len(results[0]) == 3
        assert set(results[0]) == {0, 1, 2}


class TestEXP3ASampleWeight:
    """Test EXP3A correctly handles sample_weight parameter."""

    def test_exp3a_multiplies_sample_weights(self):
        """Test that EXP3A multiplies importance weights with provided sample_weight."""
        arms = [
            Arm(0, learner=NormalInverseGammaRegressor(mu=0.2, lam=100)),
            Arm(1, learner=NormalInverseGammaRegressor(mu=0.8, lam=100)),
        ]

        policy = EXP3A(gamma=0.1, eta=2.0, ix_gamma=0.1)
        agent = ContextualAgent(arms, policy, random_seed=42)

        X = np.array([[1.0]])
        y = np.array([1.0])
        sample_weight = np.array([10_000.0])

        # Track the actual weights passed to learner
        weights_seen: List[float] = []
        original_update = arms[0].learner.partial_fit  # type: ignore

        def track_weights(
            X: NDArray[Any] | csc_array,
            y: NDArray[np.float64],
            sample_weight: NDArray[np.float64] | None = None,
        ):
            if sample_weight is not None:
                weights_seen.append(sample_weight[0])
            return original_update(X, y, sample_weight)

        arms[0].learner.partial_fit = track_weights  # type: ignore

        # Update with sample_weight
        agent.select_for_update(0).update(X, y, sample_weight=sample_weight)

        # Should have seen a weight
        assert len(weights_seen) == 1

        # The weight should be importance_weight * sample_weight
        # We don't know exact importance weight, but we multiplied it by a big number
        # so we can check it is greater than 1
        assert 1.0 < weights_seen[0]

        # Update without sample_weight
        weights_seen.clear()
        agent.select_for_update(0).update(X, y, sample_weight=None)

        # Should still see importance weight only
        assert len(weights_seen) == 1

    def test_exp3a_top_k_with_zero_probabilities(self):
        """Test EXP3A top_k when some arms have zero probability due to extreme parameters."""
        
        class FixedRewardArm(Arm):
            """Arm that returns fixed rewards for testing extreme probability scenarios."""
            def __init__(self, action_token: int, fixed_reward: float):
                super().__init__(action_token, learner=NormalInverseGammaRegressor())
                self.fixed_reward = fixed_reward
            
            def sample(self, X: NDArray[np.float64], size: int = 1) -> NDArray[np.float64]:
                return np.full((size, len(X)), self.fixed_reward)

        # Create scenario with extreme reward differences
        arms = [
            FixedRewardArm(0, 100.0),  # High reward -> high probability
            FixedRewardArm(1, 0.0),    # Low reward -> low/zero probability  
            FixedRewardArm(2, 0.0),    # Low reward -> low/zero probability
            FixedRewardArm(3, 0.0),    # Low reward -> low/zero probability
        ]

        # Extreme parameters that cause zero probabilities for most arms
        policy = EXP3A(eta=100.0, gamma=0.0, ix_gamma=0.0, samples=10)
        agent = ContextualAgent(arms, policy, random_seed=42)
        X = np.array([[1.0]])

        # This should not raise "Fewer non-zero entries in p than size" ValueError
        result = agent.pull(X, top_k=3)
        
        # Should return exactly 3 arms for the single context
        assert len(result) == 1  # One context
        assert len(result[0]) == 3  # Three arms selected
        
        # Should include the high-probability arm (arm 0)
        selected_arms = result[0]
        assert 0 in selected_arms, "High-probability arm should be selected"
        
        # Should have no duplicates
        assert len(set(selected_arms)) == 3, "Should have no duplicate arms"
        
        # Test edge case where top_k equals number of arms
        result_all = agent.pull(X, top_k=4)
        assert len(result_all[0]) == 4
        assert set(result_all[0]) == {0, 1, 2, 3}
        
        # Test with multiple contexts
        X_multi = np.array([[1.0], [2.0]])
        result_multi = agent.pull(X_multi, top_k=2)
        assert len(result_multi) == 2  # Two contexts
        assert all(len(context_result) == 2 for context_result in result_multi)
        
        # Each context should include the high-probability arm
        for context_result in result_multi:
            assert 0 in context_result, "High-probability arm should be in each context"
