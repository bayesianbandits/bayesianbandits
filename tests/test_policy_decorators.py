from typing import Any, Callable
from unittest.mock import MagicMock
import numpy as np
from numpy import floating

import pytest

from bayesianbandits import Arm
from bayesianbandits._policy_decorators import (
    upper_confidence_bound,
    _compute_arm_mean,
    _compute_arm_upper_bound,
    _draw_one_sample,
)


@pytest.fixture
def mock_arm():
    """Mock arm with a learner."""
    arm = MagicMock(autospec=True)
    return arm


@pytest.mark.parametrize(
    "aggregation_func", [_compute_arm_mean, _compute_arm_upper_bound, _draw_one_sample]
)
def test_aggregation_functions_return_type(
    aggregation_func: Callable[..., floating[Any]], mock_arm: Arm
):
    """Test that aggregation functions return the correct type."""

    mock_arm.sample.return_value = np.array([1.0])

    assert isinstance(aggregation_func(mock_arm, None), float)


def test_compute_arm_mean(mock_arm: Arm):
    """Test that the mean of the posterior distribution is computed correctly."""

    mock_arm.sample.return_value = np.array([1.0, 2.0, 3.0])

    assert _compute_arm_mean(mock_arm, None) == 2.0


def test_compute_arm_upper_bound(mock_arm: Arm):
    """Test that the upper bound of the posterior distribution is computed correctly."""

    mock_arm.sample.return_value = np.array([1.0, 2.0, 3.0, 4.0, 5.0, 6.0])

    assert _compute_arm_upper_bound(mock_arm, None, alpha=0.8) == 5.0


def test_draw_one_sample(mock_arm: Arm):
    """Test that a sample is drawn correctly."""

    mock_arm.sample.return_value = np.array([1.0])

    assert _draw_one_sample(mock_arm, None) == 1.0


def test_upper_confidence_bound_bad_alpha():
    """Test that an error is raised if alpha is not in (0, 1)."""

    with pytest.raises(ValueError):
        upper_confidence_bound(alpha=0.0)

    with pytest.raises(ValueError):
        upper_confidence_bound(alpha=1.0)
