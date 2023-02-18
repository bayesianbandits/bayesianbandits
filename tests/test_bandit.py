from functools import partial
from typing import Callable, cast, Optional
from typing_extensions import Literal
from unittest.mock import MagicMock

import numpy as np
from numpy.typing import ArrayLike, NDArray
from sklearn.utils.validation import check_is_fitted
import pytest

from bayesianbandits import Arm, epsilon_greedy, bandit, DirichletClassifier
from bayesianbandits._typing import BanditProtocol, ArmProtocol


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


@pytest.fixture(params=["no types", "with types", "with extra types"])
def bandit_class(request: pytest.FixtureRequest) -> type:
    def reward_func(x: ArrayLike) -> ArrayLike:
        return np.take(x, 0, axis=-1)  # type: ignore

    def action_func(x: int) -> None:
        print(f"action{x}")

    if request.param == "no types":

        class Experiment:  # type: ignore
            arm1 = Arm(partial(action_func, 1), reward_func)
            arm2 = Arm(partial(action_func, 2), reward_func)

    elif request.param == "with types":

        class Experiment:  # type: ignore
            arm1: Arm = Arm(partial(action_func, 1), reward_func)
            arm2: Arm = Arm(partial(action_func, 2), reward_func)

    elif request.param == "with extra types":

        class Experiment:  # type: ignore
            arm1: Arm = Arm(partial(action_func, 1), reward_func)
            arm2: Arm = Arm(partial(action_func, 2), reward_func)

            extension_: int = 1

    else:
        raise ValueError("invalid param")

    return Experiment


@pytest.mark.parametrize("choice", [epsilon_greedy(0.5)])
@pytest.mark.parametrize("learner", [DirichletClassifier({"a": 1.0, "b": 1.0})])
class TestBanditDecorator:
    def test_init(
        self,
        choice: Callable[[BanditProtocol, Optional[ArrayLike]], ArmProtocol],
        learner: DirichletClassifier,
        bandit_class: type,
    ) -> None:
        bandit_decorator = bandit(choice=choice, learner=learner)

        klass = bandit_decorator(bandit_class)

        instance = klass()

        # check that all instance attributes got assigned properly
        assert hasattr(instance, "arm1")
        assert hasattr(instance, "arm2")

        assert isinstance(instance.arms, dict)
        assert len(instance.arms) == 2

        assert instance.last_arm_pulled is None

        assert isinstance(instance.rng, np.random.Generator)

        instance2 = klass()

        # check that learners got assigned and are unique between arms and instances
        assert instance.arms["arm1"].learner is not None
        assert instance.arms["arm1"].learner is not instance.arms["arm2"].learner
        assert instance.arms["arm1"].learner is not instance2.arms["arm1"].learner

        # check that the bandit's rng is shared with each learner
        assert instance.arms["arm1"].learner.random_state is instance.rng

        # check that arms are not shared between instances
        assert [id(arm) for arm in instance.arms.values()] != [
            id(arm) for arm in instance2.arms.values()
        ]

    def test_no_arms_exception(
        self,
        choice: Callable[[BanditProtocol, Optional[ArrayLike]], ArmProtocol],
        learner: DirichletClassifier,
    ) -> None:
        bandit_decorator = bandit(choice=choice, learner=learner)

        with pytest.raises(ValueError):

            @bandit_decorator
            class NoArms:  # type: ignore
                pass

    def test_pull(
        self,
        choice: Callable[[BanditProtocol, Optional[ArrayLike]], ArmProtocol],
        learner: DirichletClassifier,
        bandit_class: type,
    ) -> None:
        bandit_decorator = bandit(choice=choice, learner=learner)

        klass = bandit_decorator(bandit_class)

        instance = klass()

        instance.pull()

        # check that the last arm pulled is not None and is one of the arms
        assert instance.last_arm_pulled is not None
        assert instance.last_arm_pulled in instance.arms.values()

    @pytest.mark.parametrize("size", [1, 2, 3])
    def test_sample(
        self,
        choice: Callable[[BanditProtocol, Optional[ArrayLike]], ArmProtocol],
        learner: DirichletClassifier,
        bandit_class: type,
        size: int,
    ) -> None:
        bandit_decorator = bandit(choice=choice, learner=learner)

        klass = bandit_decorator(bandit_class)

        instance = klass()

        # check that the sample method returns the correct number of samples
        sample = instance.sample(size=size)

        assert len(np.array(sample)) == size

    def test_update(
        self,
        choice: Callable[[BanditProtocol, Optional[ArrayLike]], ArmProtocol],
        learner: DirichletClassifier,
        bandit_class: type,
    ) -> None:
        bandit_decorator = bandit(choice=choice, learner=learner)

        klass = bandit_decorator(bandit_class)

        instance = klass()

        instance.pull()
        instance.update(["a"])

        # check that the learner was updated with the correct reward
        assert instance.last_arm_pulled is not None
        assert check_is_fitted(instance.last_arm_pulled.learner, "n_features_") is None
