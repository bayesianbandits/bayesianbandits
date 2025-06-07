"""Tests for arm featurizers."""

import numpy as np
import pytest

from bayesianbandits import (
    ArmFeaturizer,
    ContinuousArmFeaturizer,
    FunctionArmFeaturizer,
    OneHotArmFeaturizer,
)


class TestArmFeaturizerABC:
    """Test the ArmFeaturizer abstract base class."""
    
    def test_cannot_instantiate_abc(self):
        """Test that ArmFeaturizer ABC cannot be instantiated."""
        with pytest.raises(TypeError, match="Can't instantiate abstract class"):
            ArmFeaturizer()
    
    def test_inheritance_check(self):
        """Test that built-in featurizers inherit from ArmFeaturizer."""
        assert issubclass(OneHotArmFeaturizer, ArmFeaturizer)
        assert issubclass(ContinuousArmFeaturizer, ArmFeaturizer)
        assert issubclass(FunctionArmFeaturizer, ArmFeaturizer)
        
        # Test instances
        one_hot = OneHotArmFeaturizer(n_actions=3)
        assert isinstance(one_hot, ArmFeaturizer)
        
        continuous = ContinuousArmFeaturizer(degree=2)
        assert isinstance(continuous, ArmFeaturizer)
        
        def dummy_func(X, tokens):
            return np.zeros((2, 3, len(tokens)))
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
        tokens = ['a', 'b', 'c']
        
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
        
        with pytest.raises(ValueError, match="3D output has 5 arms.*but 3 action_tokens"):
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


class TestOneHotArmFeaturizer:
    """Test the OneHotArmFeaturizer."""
    
    def test_basic_one_hot(self):
        """Test basic one-hot encoding."""
        featurizer = OneHotArmFeaturizer(n_actions=4)
        X = np.array([[1.0, 2.0], [3.0, 4.0]])
        
        result = featurizer.transform(X, action_tokens=[0, 2, 3])
        
        assert result.shape == (6, 6)  # 2 contexts * 3 arms, 2 + 4 features
        
        # Check first arm (action 0)
        np.testing.assert_array_equal(result[:2, :2], X)  # Context features
        np.testing.assert_array_equal(result[:2, 2:], [[1, 0, 0, 0], [1, 0, 0, 0]])
        
        # Check second arm (action 2)
        np.testing.assert_array_equal(result[2:4, :2], X)
        np.testing.assert_array_equal(result[2:4, 2:], [[0, 0, 1, 0], [0, 0, 1, 0]])
        
        # Check third arm (action 3)
        np.testing.assert_array_equal(result[4:6, :2], X)
        np.testing.assert_array_equal(result[4:6, 2:], [[0, 0, 0, 1], [0, 0, 0, 1]])
    
    def test_invalid_n_actions(self):
        """Test error handling for invalid n_actions."""
        with pytest.raises(ValueError, match="n_actions must be positive"):
            OneHotArmFeaturizer(n_actions=0)
        
        with pytest.raises(ValueError, match="n_actions must be positive"):
            OneHotArmFeaturizer(n_actions=-1)
    
    def test_out_of_bounds_tokens(self):
        """Test error handling for out-of-bounds tokens."""
        featurizer = OneHotArmFeaturizer(n_actions=3)
        X = np.array([[1, 2]])
        
        # Token 3 is out of bounds for n_actions=3
        with pytest.raises(ValueError, match="All action tokens must be in"):
            featurizer.transform(X, action_tokens=[0, 1, 3])
        
        # Negative token
        with pytest.raises(ValueError, match="All action tokens must be in"):
            featurizer.transform(X, action_tokens=[-1, 0])
    
    def test_single_context_multiple_arms(self):
        """Test with single context and multiple arms."""
        featurizer = OneHotArmFeaturizer(n_actions=3)
        X = np.array([[5.0, 6.0]])  # Single context
        
        result = featurizer.transform(X, action_tokens=[0, 1, 2])
        
        assert result.shape == (3, 5)  # 1 context * 3 arms, 2 + 3 features
        
        # All rows should have same context features
        for i in range(3):
            np.testing.assert_array_equal(result[i, :2], [5.0, 6.0])
        
        # Check one-hot encodings
        np.testing.assert_array_equal(result[:, 2:], np.eye(3))


class TestContinuousArmFeaturizer:
    """Test the ContinuousArmFeaturizer."""
    
    def test_basic_polynomial(self):
        """Test basic polynomial features."""
        featurizer = ContinuousArmFeaturizer(degree=2)
        X = np.array([[1.0, 2.0], [3.0, 4.0]])
        
        result = featurizer.transform(X, action_tokens=[0.5, 1.0])
        
        assert result.shape == (4, 4)  # 2 contexts * 2 arms, 2 + 2 features
        
        # Check first arm (action 0.5)
        np.testing.assert_array_equal(result[:2, :2], X)
        np.testing.assert_array_almost_equal(result[:2, 2], [0.5, 0.5])
        np.testing.assert_array_almost_equal(result[:2, 3], [0.25, 0.25])
        
        # Check second arm (action 1.0)
        np.testing.assert_array_equal(result[2:4, :2], X)
        np.testing.assert_array_almost_equal(result[2:4, 2], [1.0, 1.0])
        np.testing.assert_array_almost_equal(result[2:4, 3], [1.0, 1.0])
    
    def test_degree_3(self):
        """Test higher degree polynomials."""
        featurizer = ContinuousArmFeaturizer(degree=3)
        X = np.array([[1.0, 2.0]])
        
        result = featurizer.transform(X, action_tokens=[0.5])
        
        assert result.shape == (1, 5)  # 1 context * 1 arm, 2 + 3 features
        np.testing.assert_array_equal(result[0, :2], [1.0, 2.0])
        np.testing.assert_array_almost_equal(result[0, 2:], [0.5, 0.25, 0.125])
    
    def test_with_interactions(self):
        """Test interaction terms."""
        featurizer = ContinuousArmFeaturizer(degree=1, include_interaction=True)
        X = np.array([[1.0, 2.0], [3.0, 4.0]])
        
        result = featurizer.transform(X, action_tokens=[0.5])
        
        assert result.shape == (2, 5)  # 2 contexts * 1 arm, 2 + 1 + 2 features
        
        # Original features
        np.testing.assert_array_equal(result[:, :2], X)
        # Action value
        np.testing.assert_array_equal(result[:, 2], [0.5, 0.5])
        # Interaction terms: X * action
        np.testing.assert_array_almost_equal(result[:, 3:], [[0.5, 1.0], [1.5, 2.0]])
    
    def test_invalid_degree(self):
        """Test error handling for invalid degree."""
        with pytest.raises(ValueError, match="degree must be positive"):
            ContinuousArmFeaturizer(degree=0)
        
        with pytest.raises(ValueError, match="degree must be positive"):
            ContinuousArmFeaturizer(degree=-1)
    
    def test_multiple_arms_with_interactions(self):
        """Test multiple arms with interaction terms."""
        featurizer = ContinuousArmFeaturizer(degree=2, include_interaction=True)
        X = np.array([[2.0, 3.0]])
        
        result = featurizer.transform(X, action_tokens=[0.1, 0.5, 0.9])
        
        # 1 context * 3 arms, 2 original + 2 poly + 2 interaction
        assert result.shape == (3, 6)
        
        # Check interactions for each arm
        for i, action in enumerate([0.1, 0.5, 0.9]):
            row = result[i]
            # Original features
            np.testing.assert_array_equal(row[:2], [2.0, 3.0])
            # Polynomial features  
            np.testing.assert_array_almost_equal(row[2:4], [action, action**2])
            # Interaction features
            np.testing.assert_array_almost_equal(row[4:], [2.0 * action, 3.0 * action])


class TestIntegration:
    """Integration tests with various input types."""
    
    def test_with_lists(self):
        """Test that all featurizers work with list inputs."""
        X_list = [[1.0, 2.0], [3.0, 4.0]]
        
        # OneHot
        one_hot = OneHotArmFeaturizer(n_actions=3)
        result = one_hot.transform(X_list, action_tokens=[0, 1])
        assert result.shape == (4, 5)
        
        # Continuous
        continuous = ContinuousArmFeaturizer(degree=2)
        result = continuous.transform(X_list, action_tokens=[0.5])
        assert result.shape == (2, 4)
        
        # Function
        def func(X, tokens):
            X = np.asarray(X)
            n_c, n_f = X.shape
            return np.ones((n_c, n_f, len(tokens)))
        
        function = FunctionArmFeaturizer(func)
        result = function.transform(X_list, action_tokens=['a', 'b'])
        assert result.shape == (4, 2)
    
    def test_empty_arms(self):
        """Test behavior with empty action_tokens."""
        X = np.array([[1, 2], [3, 4]])
        
        # All featurizers should handle empty arms gracefully
        one_hot = OneHotArmFeaturizer(n_actions=3)
        result = one_hot.transform(X, action_tokens=[])
        assert result.shape == (0, 5)  # 0 rows, but correct feature dimension
        
        continuous = ContinuousArmFeaturizer(degree=2)
        result = continuous.transform(X, action_tokens=[])
        assert result.shape == (0, 4)
    
    def test_single_arm(self):
        """Test with single arm."""
        X = np.array([[1, 2], [3, 4], [5, 6]])
        
        # OneHot with single arm
        one_hot = OneHotArmFeaturizer(n_actions=5)
        result = one_hot.transform(X, action_tokens=[2])
        assert result.shape == (3, 7)
        np.testing.assert_array_equal(result[:, -5:], [[0, 0, 1, 0, 0]] * 3)
        
        # Continuous with single arm
        continuous = ContinuousArmFeaturizer(degree=3)
        result = continuous.transform(X, action_tokens=[0.7])
        assert result.shape == (3, 5)
        np.testing.assert_array_almost_equal(
            result[:, 2:], 
            [[0.7, 0.49, 0.343]] * 3
        )