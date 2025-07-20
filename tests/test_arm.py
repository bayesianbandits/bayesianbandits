from unittest.mock import MagicMock

import numpy as np
import pytest
from numpy.typing import ArrayLike
from typing_extensions import Literal

from bayesianbandits import (
    Arm,
)


@pytest.fixture
def action_token() -> MagicMock:
    return MagicMock(autospec=True)


@pytest.fixture
def reward_function() -> MagicMock:
    return MagicMock(autospec=True)


@pytest.fixture
def learner() -> MagicMock:
    return MagicMock(autospec=True)


class TestArm:
    def test_init(
        self,
        action_token: MagicMock,
        reward_function: MagicMock,
    ) -> None:
        arm = Arm(action_token=1, reward_function=reward_function)
        assert arm.learner is None

    def test_pull(
        self,
        action_token: MagicMock,
        reward_function: MagicMock,
        learner: MagicMock,
    ) -> None:
        arm = Arm(action_token=action_token, reward_function=reward_function)
        arm.learner = learner

        ret = arm.pull()

        assert ret == action_token

    @pytest.mark.parametrize("size", [1, 2, 3])
    def test_sample(
        self,
        size: Literal[1, 2, 3],
        action_token: MagicMock,
        reward_function: MagicMock,
        learner: MagicMock,
    ) -> None:
        def side_effect_func(*args: ArrayLike, **kwargs: int) -> ArrayLike:
            return np.array([[1.0, 1.0, 1.0]]).repeat(args[1], axis=0)  # type: ignore

        def reward_side_effect_func(*args: ArrayLike, **kwargs: int) -> ArrayLike:
            return np.take(args[0], 0, axis=-1)  # type: ignore

        arm = Arm(action_token=action_token, reward_function=reward_function)

        learner.sample = MagicMock(side_effect=side_effect_func)
        reward_function.side_effect = reward_side_effect_func
        arm.learner = learner

        X = np.array([[1.0]])
        samples = arm.sample(X=X, size=size)

        assert len(samples) == size

    def test_update(
        self,
        action_token: MagicMock,
        reward_function: MagicMock,
        learner: MagicMock,
    ) -> None:
        arm = Arm(action_token=action_token, reward_function=reward_function)
        arm.learner = learner
        arm.learner.partial_fit = MagicMock(autospec=True)
        X = np.array([[1.0]])
        arm.update(X, np.atleast_1d([1]))

        assert arm.learner.partial_fit.call_count == 1  # type: ignore

    def test_exception(
        self,
        action_token: MagicMock,
        reward_function: MagicMock,
    ) -> None:
        arm = Arm(action_token=action_token, reward_function=reward_function)

        with pytest.raises(ValueError):
            arm.pull()

        with pytest.raises(ValueError):
            X = np.array([[1.0]])
            arm.sample(X=X, size=1)


class TestContextAwareRewardFunctions:
    """Test context-aware reward functions functionality."""

    def test_context_aware_reward_function(self) -> None:
        """Test reward function that uses context."""

        def context_multiplier(samples: np.ndarray, X: np.ndarray) -> np.ndarray:
            # Multiply by first feature (e.g., age) - maintain same dimensions
            # samples shape: (size, n_contexts), X shape: (n_contexts, n_features)
            return samples * X[:, 0]  # Broadcasting: (1, 2) * (2,) -> (1, 2)

        from bayesianbandits import NormalRegressor

        arm = Arm(
            0,
            reward_function=context_multiplier,
            learner=NormalRegressor(alpha=1, beta=1),
        )
        X = np.array([[2.0, 5.0], [3.0, 6.0]])

        result = arm.sample(X, size=1)
        # Should have shape (size, n_contexts) = (1, 2)
        assert result.shape == (1, 2)

    def test_traditional_reward_function_compatibility(self) -> None:
        """Test that traditional reward functions still work."""
        identity = lambda samples: samples

        from bayesianbandits import NormalRegressor

        arm = Arm(0, reward_function=identity, learner=NormalRegressor(alpha=1, beta=1))
        X = np.array([[1.0, 2.0]])

        result = arm.sample(X, size=1)
        assert result.shape == (1, 1)

    def test_accepts_context_detection(self) -> None:
        """Test parameter detection utility."""
        from bayesianbandits._arm import _accepts_context

        traditional = lambda samples: samples
        context_aware = lambda samples, X: samples * 2

        assert not _accepts_context(traditional)
        assert _accepts_context(context_aware)

    def test_context_aware_with_complex_logic(self) -> None:
        """Test complex context-aware reward function."""

        def premium_user_reward(samples: np.ndarray, X: np.ndarray) -> np.ndarray:
            """Higher rewards for premium users in certain contexts."""
            # Handle single context for backward compatibility
            if X.shape[0] == 1:
                age, income, premium_status = X[0, 0], X[0, 1], X[0, 2]
                
                if premium_status and age > 35:
                    return samples * 2.0
                elif income > 100000:
                    return samples * 1.5
                else:
                    return samples
            
            # Handle multiple contexts properly
            result = np.zeros_like(samples)
            for i in range(X.shape[0]):
                age, income, premium_status = X[i, 0], X[i, 1], X[i, 2]
                
                if premium_status and age > 35:
                    result[:, i] = samples[:, i] * 2.0
                elif income > 100000:
                    result[:, i] = samples[:, i] * 1.5
                else:
                    result[:, i] = samples[:, i]
            
            return result

        from bayesianbandits import NormalRegressor

        arm = Arm(
            0,
            reward_function=premium_user_reward,
            learner=NormalRegressor(alpha=1, beta=1),
        )

        # Test premium user over 35
        X_premium = np.array([[40, 50000, 1]])  # age=40, income=50k, premium=1
        result = arm.sample(X_premium, size=1)
        assert result.shape == (1, 1)

        # Test high income non-premium
        X_high_income = np.array([[30, 150000, 0]])  # age=30, income=150k, premium=0
        result = arm.sample(X_high_income, size=1)
        assert result.shape == (1, 1)

    def test_fixed_reward_based_on_context(self) -> None:
        """Test fixed rewards based on context lookup."""

        def fixed_reward_table(samples: np.ndarray, X: np.ndarray) -> np.ndarray:
            """Fixed rewards based on context lookup."""
            reward_table = {
                (25, 50000): 5.0,
                (35, 75000): 8.0,
                (45, 100000): 10.0,
            }
            default_reward = 3.0
            
            # Handle single context
            if X.shape[0] == 1:
                context_key = tuple(X[0])
                reward_value = reward_table.get(context_key, default_reward)
                return np.full_like(samples, reward_value)
            
            # Handle multiple contexts
            result = np.zeros_like(samples)
            for i in range(X.shape[0]):
                context_key = tuple(X[i])
                reward_value = reward_table.get(context_key, default_reward)
                result[:, i] = reward_value
            
            return result

        from bayesianbandits import NormalRegressor

        arm = Arm(
            0,
            reward_function=fixed_reward_table,
            learner=NormalRegressor(alpha=1, beta=1),
        )

        # Test known context
        X_known = np.array([[25, 50000]])
        result = arm.sample(X_known, size=1)
        assert result.shape == (1, 1)

        # Test unknown context
        X_unknown = np.array([[20, 30000]])
        result = arm.sample(X_unknown, size=1)
        assert result.shape == (1, 1)

