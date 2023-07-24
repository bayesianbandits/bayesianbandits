from typing import Dict
import numpy as np
from numpy.typing import NDArray

import pytest

from bayesianbandits import Arm, NormalInverseGammaRegressor
from bayesianbandits._typing import ArmProtocol
from bayesianbandits._policy_decorators import (
    upper_confidence_bound,
    thompson_sampling,
    epsilon_greedy,
    ArmChoicePolicy,
)


@pytest.fixture
def test_arms():
    """Mock arm with a learner."""
    arm1 = Arm(0, learner=NormalInverseGammaRegressor(mu=0, a=50, b=50))
    arm2 = Arm(1, learner=NormalInverseGammaRegressor(mu=1, a=50, b=50))
    arm3 = Arm(2, learner=NormalInverseGammaRegressor(mu=2, a=50, b=50))
    arm4 = Arm(3, learner=NormalInverseGammaRegressor(mu=300, a=50, b=50))

    return {
        "arm1": arm1,
        "arm2": arm2,
        "arm3": arm3,
        "arm4": arm4,
    }


@pytest.fixture(params=["len_1", "len_100"])
def X(request: pytest.FixtureRequest):
    """Mock data."""
    if request.param == "len_1":
        return np.random.normal(size=(1, 2))
    else:
        return np.random.normal(size=(100, 2))


@pytest.mark.parametrize(
    "policy",
    [upper_confidence_bound(), thompson_sampling(), epsilon_greedy()],
    ids=[
        "upper_confidence_bound",
        "thompson_sampling",
        "epsilon_greedy",
    ],
)
def test_policies(
    policy: ArmChoicePolicy, X: NDArray[np.float_], test_arms: Dict[str, ArmProtocol]
):
    """Test that the policies return an arm."""

    rng = np.random.default_rng(0)

    arms = policy(test_arms, X, rng)

    if isinstance(arms, list):
        for arm in arms:
            assert isinstance(arm, Arm)
    else:
        assert isinstance(arms, Arm)


def test_thompson_sampling_statistics(test_arms: Dict[str, ArmProtocol]):
    sampler = thompson_sampling()

    rng = np.random.default_rng(0)

    # Add intercept
    X = np.ones((100, 3))

    arms = sampler(test_arms, X, rng)

    assert isinstance(arms, list)

    # check that arm 4 is the most likely to be chosen
    counts = np.array([0, 0, 0, 0])

    for arm in arms:
        counts[arm.action_token] += 1

    assert counts[3] > counts[0]
    assert counts[3] > counts[1]
    assert counts[3] > counts[2]


def test_epsilon_greedy_statistics(test_arms: Dict[str, ArmProtocol]):
    sampler = epsilon_greedy(epsilon=0.01)

    rng = np.random.default_rng(0)

    # Add intercept
    X = np.ones((100, 3))

    arms = sampler(test_arms, X, rng)

    assert isinstance(arms, list)

    # check that arm 4 is the most likely to be chosen
    counts = np.array([0, 0, 0, 0])

    for arm in arms:
        counts[arm.action_token] += 1

    assert counts[3] > counts[0]
    assert counts[3] > counts[1]
    assert counts[3] > counts[2]


def test_ucb_statistics(test_arms: Dict[str, ArmProtocol]):
    sampler = upper_confidence_bound()

    rng = np.random.default_rng(0)

    # Add intercept
    X = np.ones((100, 3))

    arms = sampler(test_arms, X, rng)

    assert isinstance(arms, list)

    # check that arm 4 is the most likely to be chosen
    counts = np.array([0, 0, 0, 0])

    for arm in arms:
        counts[arm.action_token] += 1

    assert counts[3] > counts[0]
    assert counts[3] > counts[1]
    assert counts[3] > counts[2]


def test_upper_confidence_bound_bad_alpha():
    """Test that an error is raised if alpha is not in (0, 1)."""

    with pytest.raises(ValueError):
        upper_confidence_bound(alpha=0.0)

    with pytest.raises(ValueError):
        upper_confidence_bound(alpha=1.0)
