import sys
from unittest.mock import Mock

import numpy as np
import pytest

from bayesianbandits._arm import batch_sample_arms, can_batch_arms, stack_features


# Mock imports for testing with/without optional dependencies
class MockPandasModule:
    """Mock pandas module for testing"""

    class DataFrame:
        def __init__(self, data):
            self.data = data
            self.shape = np.array(data).shape

    @staticmethod
    def concat(dfs, **kwargs):
        # Simple concat simulation
        return MockPandasModule.DataFrame(np.vstack([df.data for df in dfs]))


class MockSparseMatrix:
    """Mock sparse matrix for testing"""

    def __init__(self, data):
        self.data = np.asarray(data)
        self.shape = self.data.shape


def mock_issparse(x):
    return isinstance(x, MockSparseMatrix)


def mock_csr_matrix(x):
    return x if isinstance(x, MockSparseMatrix) else MockSparseMatrix(x)


def mock_sparse_vstack(matrices, format="csr"):
    return MockSparseMatrix(np.vstack([m.data for m in matrices]))


# Test fixtures
@pytest.fixture
def mock_arm_class():
    """Create a mock Arm class"""

    class Arm:
        def __init__(self, token, learner, reward_function=None):
            self.token = token
            self.learner = learner
            self.reward_function = reward_function or (lambda x: x)

    return Arm


@pytest.fixture
def mock_model():
    """Create a mock model with sample method"""
    model = Mock()
    model.sample = Mock(
        side_effect=lambda X, size=1: np.random.randn(size, X.shape[0])
        if size > 1
        else np.random.randn(X.shape[0])
    )
    return model


@pytest.fixture
def mock_learner(mock_model):
    """Create a mock learner with transform and final_estimator"""
    learner = Mock()
    learner.transform = Mock(side_effect=lambda X: X)  # Identity transform
    learner.final_estimator = mock_model
    return learner


class TestCanBatchArms:
    """Tests for can_batch_arms function"""

    def test_empty_arms(self):
        assert not can_batch_arms([])

    def test_missing_transform(self, mock_arm_class):
        learner = Mock(spec=[])  # No transform attribute
        arm = mock_arm_class(0, learner)
        assert not can_batch_arms([arm])

    def test_missing_final_estimator(self, mock_arm_class):
        learner = Mock(spec=["transform"])
        arm = mock_arm_class(0, learner)
        assert not can_batch_arms([arm])

    def test_different_models(self, mock_arm_class, mock_learner):
        model1 = Mock()
        model2 = Mock()

        learner1 = Mock()
        learner1.transform = Mock()
        learner1.final_estimator = model1

        learner2 = Mock()
        learner2.transform = Mock()
        learner2.final_estimator = model2

        arms = [mock_arm_class(0, learner1), mock_arm_class(1, learner2)]
        assert not can_batch_arms(arms)

    def test_same_model(self, mock_arm_class, mock_model):
        # All arms share the same model instance
        arms = []
        for i in range(5):
            learner = Mock()
            learner.transform = Mock()
            learner.final_estimator = mock_model  # Same instance
            arms.append(mock_arm_class(i, learner))

        assert can_batch_arms(arms)


class TestStackFeatures:
    """Tests for stack_features function"""

    def test_empty_list(self):
        with pytest.raises(ValueError, match="Empty feature list"):
            stack_features([])

    def test_single_element(self):
        X = np.array([[1, 2], [3, 4]])
        result = stack_features([X])
        assert np.array_equal(result, X)

    def test_numpy_arrays(self):
        arrays = [
            np.array([[1, 2], [3, 4]]),
            np.array([[5, 6], [7, 8]]),
            np.array([[9, 10], [11, 12]]),
        ]
        result = stack_features(arrays)
        expected = np.vstack(arrays)
        assert np.array_equal(result, expected)

    def test_incompatible_shapes(self):
        arrays = [
            np.array([[1, 2], [3, 4]]),
            np.array([[5, 6, 7], [8, 9, 10]]),  # Different column count
        ]
        with pytest.raises(ValueError, match="Incompatible shapes"):
            stack_features(arrays)

    @pytest.mark.skipif("pandas" not in sys.modules, reason="pandas not available")
    def test_pandas_dataframes(self):
        import pandas as pd

        dfs = [
            pd.DataFrame({"a": [1, 2], "b": [3, 4]}),
            pd.DataFrame({"a": [5, 6], "b": [7, 8]}),
        ]
        result = stack_features(dfs)
        expected = pd.concat(dfs, ignore_index=True)
        pd.testing.assert_frame_equal(result, expected)

    @pytest.mark.skipif("pandas" not in sys.modules, reason="pandas not available")
    def test_stack_dataframe_and_array(self):
        import pandas as pd

        df = pd.DataFrame({"a": [1, 2], "b": [3, 4]})
        array = np.array([[5, 6], [7, 8]])
        with pytest.raises(
            ValueError, match="Cannot stack mixed DataFrame and non-DataFrame objects"
        ):
            stack_features([df, array])

    @pytest.mark.skipif("scipy.sparse" not in sys.modules, reason="scipy not available")
    def test_sparse_arrays(self):
        from scipy.sparse import csr_matrix

        sparse_arrays = [
            csr_matrix([[1, 0, 2], [0, 3, 0]]),
            csr_matrix([[4, 0, 0], [0, 5, 6]]),
        ]
        result = stack_features(sparse_arrays)
        assert result.shape == (4, 3)

    @pytest.mark.skipif("scipy.sparse" not in sys.modules, reason="scipy not available")
    def test_mixed_sparse_and_dense(self):
        from scipy.sparse import csr_matrix

        sparse_array = csr_matrix([[1, 0, 2], [0, 3, 0]])
        dense_array = np.array([[4, 5, 6], [7, 8, 9]])
        with pytest.raises(
            ValueError, match="Cannot stack mixed sparse and dense arrays"
        ):
            stack_features([sparse_array, dense_array])


class TestBatchSampleArms:
    """Tests for batch_sample_arms function"""

    def test_non_batchable_arms(self, mock_arm_class):
        # Arms with different models
        arms = []
        for i in range(3):
            learner = Mock()
            learner.transform = Mock()
            learner.final_estimator = Mock()  # Different instance each time
            arms.append(mock_arm_class(i, learner))

        X = np.random.randn(5, 3)
        result = batch_sample_arms(arms, X)
        assert result is None

    def test_basic_batching(self, mock_arm_class, mock_model):
        # Setup arms with same model
        n_arms = 4
        arms = []
        for i in range(n_arms):
            learner = Mock()
            learner.transform = Mock(side_effect=lambda X: X)
            learner.final_estimator = mock_model
            arms.append(mock_arm_class(i, learner))

        # Test with 2D context
        n_contexts = 3
        X = np.random.randn(n_contexts, 5)

        # Test size=1
        result = batch_sample_arms(arms, X, size=1)
        assert result is not None
        assert result.shape == (n_arms, n_contexts)

        # Test size>1
        size = 5
        result = batch_sample_arms(arms, X, size=size)
        assert result is not None
        assert result.shape == (n_arms, n_contexts, size)

    def test_transform_error_handling(self, mock_arm_class, mock_model):
        # Arms where transform raises exception
        arms = []
        for i in range(3):
            learner = Mock()
            if i == 1:
                # Second arm's transform returns incompatible shape
                learner.transform = Mock(return_value=np.random.randn(10, 10))
            else:
                learner.transform = Mock(return_value=np.random.randn(5, 3))
            learner.final_estimator = mock_model
            arms.append(mock_arm_class(i, learner))

        X = np.random.randn(5, 3)
        result = batch_sample_arms(arms, X)
        assert result is None  # Should handle gracefully

    def test_reward_function_identity(self, mock_arm_class, mock_model):
        # All arms use same reward function (identity)
        n_arms = 3
        identity_func = lambda x: x
        arms = []
        for i in range(n_arms):
            learner = Mock()
            learner.transform = Mock(side_effect=lambda X: X)
            learner.final_estimator = mock_model
            arms.append(mock_arm_class(i, learner, identity_func))

        X = np.random.randn(5, 3)
        result = batch_sample_arms(arms, X)
        assert result is not None

        # Should not modify samples when all use identity
        mock_model.sample.assert_called_once()

    def test_reward_function_custom(self, mock_arm_class, mock_model):
        # Different reward functions
        n_arms = 3
        arms = []
        for i in range(n_arms):
            learner = Mock()
            learner.transform = Mock(side_effect=lambda X: X)
            learner.final_estimator = mock_model
            # Each arm squares its reward differently
            reward_func = lambda x, scale=i + 1: x * scale
            arms.append(mock_arm_class(i, learner, reward_func))

        X = np.random.randn(2, 3)
        result = batch_sample_arms(arms, X)
        assert result is not None

    def test_single_context(self, mock_arm_class, mock_model):
        # Test with single context (needs len attribute for new code)
        class SingleContext:
            def __init__(self, data):
                self.data = data

            def __len__(self):
                # Return 1 to indicate single context
                return 1

        arms = []
        for i in range(3):
            learner = Mock()
            learner.transform = Mock(return_value=np.array([1, 2, 3]))
            learner.final_estimator = mock_model
            arms.append(mock_arm_class(i, learner))

        X = SingleContext(np.array([1, 2, 3]))
        result = batch_sample_arms(arms, X)  # type: ignore[arg-type]
        assert result is not None
        assert result.shape == (3, 1)  # 3 arms, 1 context

    def test_edge_cases(self, mock_arm_class, mock_model):
        # Single arm
        learner = Mock()
        learner.transform = Mock(side_effect=lambda X: X)
        learner.final_estimator = mock_model
        arms = [mock_arm_class(0, learner)]

        X = np.random.randn(5, 3)
        result = batch_sample_arms(arms, X)
        assert result is not None
        assert result.shape == (1, 5)


class TestIntegration:
    """Integration tests with mocked pandas/scipy"""

    def test_with_mocked_pandas(self, monkeypatch):
        # Mock pandas availability
        mock_pd = MockPandasModule()

        # Mock importlib.util.find_spec to indicate pandas is available
        mock_spec = Mock()
        monkeypatch.setattr(
            "importlib.util.find_spec",
            lambda name: mock_spec if name == "pandas" else None,
        )

        # Mock the pandas import inside stack_features
        # We need to patch the module in sys.modules before the import happens
        original_modules = sys.modules.copy()
        sys.modules["pandas"] = mock_pd  # type: ignore[assignment]
        monkeypatch.setattr("sys.modules", sys.modules)

        # Force re-evaluation of HAS_PANDAS by reimporting the module
        import bayesianbandits._arm
        from importlib import reload

        reload(bayesianbandits._arm)
        from bayesianbandits._arm import stack_features

        try:
            # Create DataFrames
            dfs = [
                mock_pd.DataFrame([[1, 2], [3, 4]]),
                mock_pd.DataFrame([[5, 6], [7, 8]]),
            ]

            result = stack_features(dfs)
            assert hasattr(result, "data")
            assert result.data.shape == (4, 2)
        finally:
            # Restore original modules
            sys.modules.clear()
            sys.modules.update(original_modules)
            # Reload to restore original state
            reload(bayesianbandits._arm)

    def test_with_mocked_scipy(self, monkeypatch):
        # Mock scipy availability
        monkeypatch.setattr("bayesianbandits._arm.issparse", mock_issparse)
        monkeypatch.setattr("bayesianbandits._arm.csr_matrix", mock_csr_matrix)
        monkeypatch.setattr("bayesianbandits._arm.sparse_vstack", mock_sparse_vstack)

        # Create sparse matrices
        sparse_arrays = [
            MockSparseMatrix([[1, 0, 2], [0, 3, 0]]),
            MockSparseMatrix([[4, 0, 0], [0, 5, 6]]),
        ]

        result = stack_features(sparse_arrays)
        assert isinstance(result, MockSparseMatrix)
        assert result.shape == (4, 3)


# Performance tests
class TestPerformance:
    """Performance-related tests"""

    def test_large_batch(self, mock_arm_class, mock_model):
        # Test with many arms
        n_arms = 100
        arms = []
        shared_model = mock_model

        for i in range(n_arms):
            learner = Mock()
            learner.transform = Mock(side_effect=lambda X: X)
            learner.final_estimator = shared_model
            arms.append(mock_arm_class(i, learner))

        X = np.random.randn(50, 10)
        result = batch_sample_arms(arms, X)

        assert result is not None
        assert result.shape == (n_arms, 50)
        # Should only call sample once due to batching
        assert shared_model.sample.call_count == 1
