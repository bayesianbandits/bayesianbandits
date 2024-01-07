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
