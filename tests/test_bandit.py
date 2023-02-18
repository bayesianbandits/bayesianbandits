from typing import cast, Optional
from typing_extensions import Literal
from unittest.mock import MagicMock

import numpy as np
from numpy.typing import ArrayLike, NDArray

import pytest

from bayesianbandits import Arm


@pytest.fixture
def action_function() -> MagicMock:
    return MagicMock(autospec=True)


@pytest.fixture
def reward_function() -> MagicMock:
    return MagicMock(autospec=True)


@pytest.fixture
def learner() -> MagicMock:
    return MagicMock(autospec=True)


@pytest.mark.parametrize("X", [None, np.array([[1.0]])])
class TestArm:
    def test_init(
        self,
        X: Optional[NDArray[np.float_]],
        action_function: MagicMock,
        reward_function: MagicMock,
    ) -> None:
        arm = Arm(action_function=action_function, reward_function=reward_function)
        assert arm.learner is None

    def test_pull(
        self,
        X: Optional[NDArray[np.float_]],
        action_function: MagicMock,
        reward_function: MagicMock,
        learner: MagicMock,
    ) -> None:
        arm = Arm(action_function=action_function, reward_function=reward_function)
        arm.learner = learner

        arm.pull()

        arm.action_function = cast(MagicMock, arm.action_function)
        assert arm.action_function.call_count == 1

    @pytest.mark.parametrize("size", [1, 2, 3])
    def test_sample(
        self,
        X: Optional[NDArray[np.float_]],
        size: Literal[1, 2, 3],
        action_function: MagicMock,
        reward_function: MagicMock,
        learner: MagicMock,
    ) -> None:
        def side_effect_func(*args: ArrayLike, **kwargs: int) -> ArrayLike:
            return np.array([[1.0, 1.0, 1.0]]).repeat(args[1], axis=0)  # type: ignore

        def reward_side_effect_func(*args: ArrayLike, **kwargs: int) -> ArrayLike:
            return np.take(args[0], 0, axis=-1)  # type: ignore

        arm = Arm(action_function=action_function, reward_function=reward_function)

        learner.sample = MagicMock(side_effect=side_effect_func)
        reward_function.side_effect = reward_side_effect_func
        arm.learner = learner

        samples = arm.sample(X=X, size=size)

        assert len(samples) == size

    def test_update(
        self,
        X: Optional[NDArray[np.float_]],
        action_function: MagicMock,
        reward_function: MagicMock,
        learner: MagicMock,
    ) -> None:
        arm = Arm(action_function=action_function, reward_function=reward_function)
        arm.learner = learner
        arm.learner.partial_fit = MagicMock(autospec=True)
        if X is None:
            arm.update([1])
        else:
            arm.update(X, [1])

        assert arm.learner.partial_fit.call_count == 1  # type: ignore

    def test_mean(
        self,
        X: Optional[NDArray[np.float_]],
        action_function: MagicMock,
        reward_function: MagicMock,
        learner: MagicMock,
    ) -> None:
        def side_effect_func(*args: ArrayLike, **kwargs: int) -> ArrayLike:
            return np.array([[1.0, 1.0, 1.0]]).repeat(args[1], axis=0)  # type: ignore

        def reward_side_effect_func(*args: ArrayLike, **kwargs: int) -> ArrayLike:
            return np.take(args[0], 0, axis=-1)  # type: ignore

        arm = Arm(action_function=action_function, reward_function=reward_function)

        learner.sample = MagicMock(side_effect=side_effect_func)
        reward_function.side_effect = reward_side_effect_func
        arm.learner = learner

        mean = arm.mean(X=X)

        assert mean == 1.0

    def test_exception(
        self,
        X: Optional[NDArray[np.float_]],
        action_function: MagicMock,
        reward_function: MagicMock,
    ) -> None:
        arm = Arm(action_function=action_function, reward_function=reward_function)

        with pytest.raises(ValueError):
            arm.pull()

        with pytest.raises(ValueError):
            arm.sample(X=X, size=1)

        with pytest.raises(ValueError):
            if X is None:
                arm.update([1])
            else:
                arm.update(X, [1])

        with pytest.raises(ValueError):
            arm.mean(X=X)
