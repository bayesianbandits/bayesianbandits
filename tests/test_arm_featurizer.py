"""Tests for arm featurizers."""

import numpy as np
import pytest

from bayesianbandits import (
    ArmColumnFeaturizer,
    ArmFeaturizer,
    FunctionArmFeaturizer,
)


class TestArmFeaturizerABC:
    """Test the ArmFeaturizer abstract base class."""

    def test_cannot_instantiate_abc(self):
        """Test that ArmFeaturizer ABC cannot be instantiated."""
        with pytest.raises(TypeError, match="Can't instantiate abstract class"):
            ArmFeaturizer()  # type: ignore[abstract]

    def test_inheritance_check(self):
        """Test that built-in featurizers inherit from ArmFeaturizer."""
        assert issubclass(ArmColumnFeaturizer, ArmFeaturizer)
        assert issubclass(FunctionArmFeaturizer, ArmFeaturizer)

        # Test instances
        arm_column = ArmColumnFeaturizer()
        assert isinstance(arm_column, ArmFeaturizer)

        def dummy_func(X, tokens):
            return np.zeros((2, 3, len(tokens)))

        dummy_func(
            np.zeros((2, 3)), ["a", "b", "c"]
        )  # Call to check function signature

        func_feat = FunctionArmFeaturizer(dummy_func)
        assert isinstance(func_feat, ArmFeaturizer)


class TestFunctionArmFeaturizer:
    """Test the FunctionArmFeaturizer wrapper."""

    def test_basic_3d_function(self):
        """Test basic 3D function output."""

        def simple_3d(X, action_tokens):
            X = np.asarray(X)
            n_contexts, n_features = X.shape
            n_arms = len(action_tokens)

            # Just broadcast context features
            result = np.zeros((n_contexts, n_features + 1, n_arms))
            result[:, :n_features, :] = X[:, :, np.newaxis]
            result[:, n_features, :] = 1  # Add constant feature

            return result

        featurizer = FunctionArmFeaturizer(simple_3d)
        X = np.array([[1, 2], [3, 4]])
        tokens = ["a", "b", "c"]

        result = featurizer.transform(X, action_tokens=tokens)

        assert result.shape == (6, 3)  # 2 contexts * 3 arms, 3 features
        # Check first arm's contexts
        np.testing.assert_array_equal(result[:2, :2], X)
        np.testing.assert_array_equal(result[:2, 2], [1, 1])
        # Check second arm's contexts
        np.testing.assert_array_equal(result[2:4, :2], X)
        np.testing.assert_array_equal(result[2:4, 2], [1, 1])

    def test_wrong_output_dimensions(self):
        """Test error handling for wrong dimensions."""

        # Function returns 2D instead of 3D
        def wrong_2d(X, tokens):
            return np.zeros((4, 5))

        featurizer = FunctionArmFeaturizer(wrong_2d)
        X = np.array([[1, 2], [3, 4]])

        with pytest.raises(ValueError, match="Expected 3D output"):
            featurizer.transform(X, action_tokens=[0, 1])

    def test_mismatched_arms_dimension(self):
        """Test error when arms dimension doesn't match tokens."""

        def wrong_arms(X, tokens):
            # Returns 5 arms but only 3 tokens provided
            return np.zeros((2, 3, 5))

        featurizer = FunctionArmFeaturizer(wrong_arms)
        X = np.array([[1, 2], [3, 4]])

        with pytest.raises(
            ValueError, match="3D output has 5 arms.*but 3 action_tokens"
        ):
            featurizer.transform(X, action_tokens=[0, 1, 2])

    def test_handles_list_input(self):
        """Test that list inputs are converted properly."""

        def simple_func(X, tokens):
            X = np.asarray(X)
            n_contexts, n_features = X.shape
            n_arms = len(tokens)
            return np.ones((n_contexts, n_features, n_arms))

        featurizer = FunctionArmFeaturizer(simple_func)
        X_list = [[1, 2], [3, 4]]

        result = featurizer.transform(X_list, action_tokens=[0, 1])
        assert result.shape == (4, 2)  # 2 contexts * 2 arms, 2 features


class TestIntegration:
    """Integration tests with various input types."""

    def test_with_lists(self):
        """Test that featurizers work with list inputs."""
        X_list = [[1.0, 2.0], [3.0, 4.0]]

        # ArmColumn
        arm_column = ArmColumnFeaturizer()
        result = arm_column.transform(X_list, action_tokens=[0, 1])
        assert result.shape == (4, 3)

        # Function
        def func(X, tokens):
            X = np.asarray(X)
            n_c, n_f = X.shape
            return np.ones((n_c, n_f, len(tokens)))

        function = FunctionArmFeaturizer(func)
        result = function.transform(X_list, action_tokens=["a", "b"])
        assert result.shape == (4, 2)

    def test_empty_arms(self):
        """Test behavior with empty action_tokens."""
        X = np.array([[1, 2], [3, 4]])

        # All featurizers should handle empty arms gracefully
        arm_column = ArmColumnFeaturizer()
        result = arm_column.transform(X, action_tokens=[])
        assert result.shape == (0, 3)  # 0 rows, but correct feature dimension

    def test_single_arm(self):
        """Test with single arm."""
        X = np.array([[1, 2], [3, 4], [5, 6]])

        # ArmColumn with single arm
        arm_column = ArmColumnFeaturizer()
        result = arm_column.transform(X, action_tokens=[2])
        assert result.shape == (3, 3)  # 3 contexts, 2 features + 1 arm column
        np.testing.assert_array_equal(result[:, :2], X)
        np.testing.assert_array_equal(result[:, 2], [2, 2, 2])
