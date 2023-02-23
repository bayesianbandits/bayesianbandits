from functools import partial
from typing import Any, Callable, Dict, List, Optional, cast
from unittest.mock import MagicMock

import numpy as np
import pytest
from numpy.typing import ArrayLike, NDArray
from sklearn.utils.validation import check_is_fitted
from typing_extensions import Literal

from bayesianbandits import (
    Arm,
    DirichletClassifier,
    bandit,
    contextfree,
    delayed_reward,
    restless,
    epsilon_greedy,
    thompson_sampling,
    upper_confidence_bound,
)
from bayesianbandits._typing import ArmProtocol, BanditProtocol


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


@pytest.mark.parametrize("restless_decorator", [None, restless])
@pytest.mark.parametrize("delayed_decorator", [None, delayed_reward])
@pytest.mark.parametrize("X", [None, np.array([[2.0]])])
@pytest.mark.parametrize(
    "choice", [epsilon_greedy(0.5), thompson_sampling(), upper_confidence_bound(0.68)]
)
@pytest.mark.parametrize("learner", [DirichletClassifier({"a": 1.0, "b": 1.0})])
class TestBanditDecorator:
    def test_init(
        self,
        choice: Callable[[BanditProtocol, Optional[ArrayLike]], ArmProtocol],
        learner: DirichletClassifier,
        X: Optional[NDArray[np.float_]],
        delayed_decorator: Optional[Callable[[type], type]],
        restless_decorator: Optional[Callable[[type], type]],
        bandit_class: type,
    ) -> None:
        bandit_decorator = bandit(policy=choice, learner=learner)

        klass = bandit_decorator(bandit_class)

        if X is None:
            klass = contextfree(klass)

        if delayed_decorator:
            bandit_decorator = delayed_reward(klass)

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
        X: Optional[NDArray[np.float_]],
        delayed_decorator: Optional[Callable[[type], type]],
        restless_decorator: Optional[Callable[[type], type]],
    ) -> None:
        bandit_decorator = bandit(policy=choice, learner=learner)

        with pytest.raises(ValueError):

            @bandit_decorator
            class NoArms:  # type: ignore
                pass

    def test_pull(
        self,
        choice: Callable[[BanditProtocol, Optional[ArrayLike]], ArmProtocol],
        learner: DirichletClassifier,
        X: Optional[NDArray[np.float_]],
        delayed_decorator: Optional[Callable[[type], type]],
        restless_decorator: Optional[Callable[[type], type]],
        bandit_class: type,
    ) -> None:
        bandit_decorator = bandit(policy=choice, learner=learner)

        klass = bandit_decorator(bandit_class)

        if delayed_decorator:
            klass = delayed_decorator(klass)

        if restless_decorator:
            klass = restless_decorator(klass)

        if X is None:
            klass = contextfree(klass)

        instance = klass()

        pull_args: List[Any] = []
        pull_kwargs: Dict[str, Any] = {}
        if X is not None:
            pull_args.append(X)
        if delayed_decorator:
            pull_kwargs["unique_id"] = 1

        instance.pull(*pull_args, **pull_kwargs)

        # check that the last arm pulled is not None and is one of the arms
        assert instance.last_arm_pulled is not None
        assert instance.last_arm_pulled in instance.arms.values()

        if delayed_decorator:
            # check that the last arm pulled is in the cache
            cached_arm = instance.arms[instance.__cache__[1]]  # type: ignore
            assert cached_arm is instance.last_arm_pulled

    @pytest.mark.parametrize("size", [1, 2])
    def test_sample(
        self,
        choice: Callable[[BanditProtocol, Optional[ArrayLike]], ArmProtocol],
        learner: DirichletClassifier,
        X: Optional[NDArray[np.float_]],
        delayed_decorator: Optional[Callable[[type], type]],
        restless_decorator: Optional[Callable[[type], type]],
        bandit_class: type,
        size: int,
    ) -> None:
        bandit_decorator = bandit(policy=choice, learner=learner)

        klass = bandit_decorator(bandit_class)

        if X is None:
            klass = contextfree(klass)
        if restless_decorator:
            klass = restless_decorator(klass)
        if delayed_decorator:
            klass = delayed_decorator(klass)

        instance = klass()

        if X is None:
            # check that the sample method returns the correct number of samples
            sample = instance.sample(size=size)
        else:
            sample = instance.sample(X, size=size)

        assert len(np.array(sample)) == size

    def test_update(
        self,
        choice: Callable[[BanditProtocol, Optional[ArrayLike]], ArmProtocol],
        learner: DirichletClassifier,
        X: Optional[NDArray[np.float_]],
        delayed_decorator: Optional[Callable[[type], type]],
        restless_decorator: Optional[Callable[[type], type]],
        bandit_class: type,
    ) -> None:
        bandit_decorator = bandit(policy=choice, learner=learner)

        klass = bandit_decorator(bandit_class)

        if X is None:
            klass = contextfree(klass)
        if restless_decorator:
            klass = restless_decorator(klass)
        if delayed_decorator:
            klass = delayed_decorator(klass)

        instance = klass()

        pull_args: List[Any] = []
        pull_kwargs: Dict[str, Any] = {}
        if X is not None:
            pull_args.append(X)
        if delayed_decorator:
            pull_kwargs["unique_id"] = 1

        instance.pull(*pull_args, **pull_kwargs)
        instance.update(*pull_args, "a", **pull_kwargs)  # type: ignore

        # check that the learner was updated with the correct reward
        assert instance.last_arm_pulled is not None
        assert check_is_fitted(instance.last_arm_pulled.learner, "n_features_") is None
