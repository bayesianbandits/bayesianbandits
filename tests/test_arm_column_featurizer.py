"""Tests for ArmColumnFeaturizer."""

import importlib.util
import numpy as np
import pytest

from bayesianbandits import ArmColumnFeaturizer

HAS_PANDAS = importlib.util.find_spec("pandas") is not None


class TestArmColumnFeaturizer:
    """Test the ArmColumnFeaturizer."""

    def test_basic_numpy_array(self):
        """Test basic functionality with numpy arrays."""
        featurizer = ArmColumnFeaturizer()
        X = np.array([[1.0, 2.0], [3.0, 4.0]])
        
        result = featurizer.transform(X, action_tokens=[0, 1, 2])
        
        assert result.shape == (6, 3)  # 2 contexts * 3 arms, 2 + 1 features
        assert isinstance(result, np.ndarray)
        
        # Check first arm (token 0)
        np.testing.assert_array_equal(result[:2, :2], X)
        np.testing.assert_array_equal(result[:2, 2], [0, 0])
        
        # Check second arm (token 1)
        np.testing.assert_array_equal(result[2:4, :2], X)
        np.testing.assert_array_equal(result[2:4, 2], [1, 1])
        
        # Check third arm (token 2)
        np.testing.assert_array_equal(result[4:6, :2], X)
        np.testing.assert_array_equal(result[4:6, 2], [2, 2])

    def test_string_tokens(self):
        """Test with string action tokens."""
        featurizer = ArmColumnFeaturizer()
        X = np.array([[1.0, 2.0], [3.0, 4.0]])
        
        result = featurizer.transform(X, action_tokens=["apple", "banana", "cherry"])
        
        assert result.shape == (6, 3)
        
        # Check that string tokens are properly repeated
        assert result[0, 2] == "apple"
        assert result[1, 2] == "apple"
        assert result[2, 2] == "banana"
        assert result[3, 2] == "banana"
        assert result[4, 2] == "cherry"
        assert result[5, 2] == "cherry"

    def test_float_tokens(self):
        """Test with float action tokens."""
        featurizer = ArmColumnFeaturizer()
        X = np.array([[1.0, 2.0], [3.0, 4.0]])
        
        result = featurizer.transform(X, action_tokens=[0.1, 0.5, 0.9])
        
        assert result.shape == (6, 3)
        
        # Check float tokens
        np.testing.assert_array_almost_equal(result[:2, 2], [0.1, 0.1])
        np.testing.assert_array_almost_equal(result[2:4, 2], [0.5, 0.5])
        np.testing.assert_array_almost_equal(result[4:6, 2], [0.9, 0.9])

    def test_empty_arms(self):
        """Test with empty action_tokens."""
        featurizer = ArmColumnFeaturizer()
        X = np.array([[1, 2], [3, 4]])
        
        result = featurizer.transform(X, action_tokens=[])
        
        assert result.shape == (0, 3)
        assert isinstance(result, np.ndarray)

    def test_single_context_multiple_arms(self):
        """Test with single context and multiple arms."""
        featurizer = ArmColumnFeaturizer()
        X = np.array([[5.0, 6.0]])  # Single context
        
        result = featurizer.transform(X, action_tokens=[10, 20, 30])
        
        assert result.shape == (3, 3)
        
        # All rows should have same context features
        for i in range(3):
            np.testing.assert_array_equal(result[i, :2], [5.0, 6.0])
        
        # Check arm tokens
        np.testing.assert_array_equal(result[:, 2], [10, 20, 30])

    def test_list_input(self):
        """Test with list input for X."""
        featurizer = ArmColumnFeaturizer()
        X_list = [[1.0, 2.0], [3.0, 4.0]]
        
        result = featurizer.transform(X_list, action_tokens=["a", "b"])
        
        assert result.shape == (4, 3)
        assert isinstance(result, np.ndarray)

    @pytest.mark.skipif(not HAS_PANDAS, reason="pandas not installed")
    def test_pandas_dataframe(self):
        """Test with pandas DataFrame input."""
        import pandas as pd
        
        featurizer = ArmColumnFeaturizer()
        X = pd.DataFrame({
            'feature1': [1.0, 3.0],
            'feature2': [2.0, 4.0]
        })
        
        result = featurizer.transform(X, action_tokens=[0, 1, 2])
        
        # Should return a DataFrame
        assert isinstance(result, pd.DataFrame)
        assert result.shape == (6, 3)
        assert list(result.columns) == ['feature1', 'feature2', 'arm_token']
        
        # Check values
        assert list(result['arm_token']) == [0, 0, 1, 1, 2, 2]
        assert list(result['feature1']) == [1.0, 3.0, 1.0, 3.0, 1.0, 3.0]
        assert list(result['feature2']) == [2.0, 4.0, 2.0, 4.0, 2.0, 4.0]

    @pytest.mark.skipif(not HAS_PANDAS, reason="pandas not installed")
    def test_pandas_empty_arms(self):
        """Test pandas DataFrame with empty arms."""
        import pandas as pd
        
        featurizer = ArmColumnFeaturizer()
        X = pd.DataFrame({
            'feature1': [1.0, 3.0],
            'feature2': [2.0, 4.0]
        })
        
        result = featurizer.transform(X, action_tokens=[])
        
        assert isinstance(result, pd.DataFrame)
        assert result.shape == (0, 3)
        assert list(result.columns) == ['feature1', 'feature2', 'arm_token']
        # Check dtypes are preserved
        assert result['feature1'].dtype == X['feature1'].dtype
        assert result['feature2'].dtype == X['feature2'].dtype

    def test_mixed_type_tokens(self):
        """Test with mixed type tokens (numpy converts to common type)."""
        featurizer = ArmColumnFeaturizer()
        X = np.array([[1.0, 2.0]])
        
        # Mixed types in tokens - numpy will convert to strings
        result = featurizer.transform(X, action_tokens=[1, "two", 3.0])
        
        assert result.shape == (3, 3)
        # When mixed types are provided, numpy converts all to strings
        assert str(result[0, 2]) == "1"
        assert str(result[1, 2]) == "two"
        assert str(result[2, 2]) == "3.0"


class TestArmColumnFeaturizerWithSklearn:
    """Test ArmColumnFeaturizer integration with sklearn transformers."""

    def test_with_column_transformer(self):
        """Test integration with sklearn ColumnTransformer."""
        from sklearn.compose import ColumnTransformer
        from sklearn.preprocessing import OneHotEncoder, StandardScaler
        
        featurizer = ArmColumnFeaturizer()
        X = np.array([[1.0, 2.0], [3.0, 4.0]])
        
        # Transform with discrete arm tokens
        X_with_arms = featurizer.transform(X, action_tokens=[0, 1, 2])
        
        # Apply ColumnTransformer
        transformer = ColumnTransformer([
            ('context', StandardScaler(), [0, 1]),
            ('arm', OneHotEncoder(sparse_output=False), [2])
        ])
        
        X_final = transformer.fit_transform(X_with_arms)
        
        # Should have 2 scaled features + 3 one-hot features
        assert X_final.shape == (6, 5)

    def test_with_polynomial_features(self):
        """Test with PolynomialFeatures for continuous arms."""
        from sklearn.compose import ColumnTransformer
        from sklearn.preprocessing import PolynomialFeatures
        
        featurizer = ArmColumnFeaturizer()
        X = np.array([[1.0, 2.0], [3.0, 4.0]])
        
        # Transform with continuous arm values
        X_with_arms = featurizer.transform(X, action_tokens=[0.1, 0.5, 0.9])
        
        # Apply polynomial features to arm column
        transformer = ColumnTransformer([
            ('context', 'passthrough', [0, 1]),
            ('arm', PolynomialFeatures(degree=2, include_bias=False), [2])
        ])
        
        X_final = transformer.fit_transform(X_with_arms)
        
        # Should have 2 context features + 2 polynomial features (x, x^2)
        assert X_final.shape == (6, 4)