from typing import Any, Callable, Dict, List, Optional, Type
import pytest
import numpy as np
from numpy.typing import ArrayLike, NDArray
from functools import partial
from sklearn.base import check_is_fitted
from bayesianbandits import (
    Arm,
    DirichletClassifier,
    GammaRegressor,
    NormalRegressor,
    NormalInverseGammaRegressor,
)
from bayesianbandits._policy_decorators import (
    epsilon_greedy,
    upper_confidence_bound,
    thompson_sampling,
)
from bayesianbandits._basebandit import Bandit, restless, contextual
from bayesianbandits._typing import Learner, BanditProtocol, ArmProtocol


@pytest.fixture(params=["dirichlet", "gamma", "normal", "normal-inverse-gamma"])
def learner_class(request: pytest.FixtureRequest) -> Learner:
    if request.param == "dirichlet":
        return DirichletClassifier({1: 1.0, 2: 1.0})
    elif request.param == "gamma":
        return GammaRegressor(alpha=1, beta=1)
    elif request.param == "normal":
        return NormalRegressor(alpha=1, beta=1)
    elif request.param == "normal-inverse-gamma":
        return NormalInverseGammaRegressor()
    else:
        raise ValueError("invalid param")


@pytest.fixture(
    params=["epsilon_greedy", "thompson_sampling", "upper_confidence_bound"]
)
def choice(
    request: pytest.FixtureRequest,
) -> Callable[[BanditProtocol, Optional[ArrayLike]], ArmProtocol]:
    if request.param == "epsilon_greedy":
        return epsilon_greedy(0.5)
    elif request.param == "thompson_sampling":
        return thompson_sampling()
    elif request.param == "upper_confidence_bound":
        return upper_confidence_bound(0.68)
    else:
        raise ValueError("invalid param")


@pytest.fixture(params=["delayed_reward", "not delayed"])
def delayed_reward(request: pytest.FixtureRequest) -> bool:
    if request.param == "delayed_reward":
        return True
    elif request.param == "not delayed":
        return False
    else:
        raise ValueError("invalid param")


@pytest.fixture(params=["no types", "with types", "with extra types"])
def bandit_class(
    request: pytest.FixtureRequest,
    learner_class: Learner,
    choice: Callable[[BanditProtocol, Optional[ArrayLike]], ArmProtocol],
    delayed_reward: bool,
) -> type:
    def reward_func(x: ArrayLike) -> ArrayLike:
        return np.take(x, 0, axis=-1)  # type: ignore

    def action_func(x: int) -> None:
        print(f"action{x}")

    if request.param == "no types":

        class Experiment(  # type: ignore
            Bandit, learner=learner_class, policy=choice, delayed_reward=delayed_reward
        ):
            arm1 = Arm(partial(action_func, 1), reward_func)
            arm2 = Arm(partial(action_func, 2), reward_func)

    elif request.param == "with types":

        class Experiment(  # type: ignore
            Bandit, learner=learner_class, policy=choice, delayed_reward=delayed_reward
        ):
            arm1: Arm = Arm(partial(action_func, 1), reward_func)
            arm2: Arm = Arm(partial(action_func, 2), reward_func)

    elif request.param == "with extra types":

        class Experiment(  # type: ignore
            Bandit, learner=learner_class, policy=choice, delayed_reward=delayed_reward
        ):
            arm1: Arm = Arm(partial(action_func, 1), reward_func)
            arm2: Arm = Arm(partial(action_func, 2), reward_func)

            extension_: int = 1

    else:
        raise ValueError("invalid param")

    return Experiment


@pytest.mark.parametrize("restless_decorator", [None, restless])
@pytest.mark.parametrize("X", [None, np.array([[2.0]])])
class TestBanditDecorator:
    def test_init(
        self,
        X: Optional[NDArray[np.float_]],
        restless_decorator: Optional[Callable[[type], type]],
        bandit_class: Type[Bandit],
    ) -> None:
        klass = bandit_class

        if X is not None:
            klass = contextual(klass)

        if restless_decorator is not None:
            klass = restless_decorator(klass)

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
        X: Optional[NDArray[np.float_]],
        restless_decorator: Optional[Callable[[type], type]],
        bandit_class: Type[Bandit],
    ) -> None:
        with pytest.raises(ValueError):

            class NoArms(
                Bandit,
                learner=DirichletClassifier({1: 1.0, 2: 1.0}),
                policy=epsilon_greedy(),
            ):
                pass

            NoArms()

    def test_pull(
        self,
        X: Optional[NDArray[np.float_]],
        restless_decorator,
        bandit_class: Type[Bandit],
    ) -> None:
        klass = bandit_class

        if X is not None:
            klass = contextual(klass)

        if restless_decorator is not None:
            klass = restless_decorator(klass)

        instance = klass()

        pull_args: List[Any] = []
        pull_kwargs: Dict[str, Any] = {}
        if X is not None:
            pull_args.append(X)
        if bandit_class._delayed_reward:
            pull_kwargs["unique_id"] = 1

        instance.pull(*pull_args, **pull_kwargs)

        # check that the last arm pulled is not None and is one of the arms
        assert instance.last_arm_pulled is not None
        assert instance.last_arm_pulled in instance.arms.values()

        if bandit_class._delayed_reward:
            # check that the last arm pulled is in the cache
            cached_arm = instance.arms[instance.cache[1]]  # type: ignore
            assert cached_arm is instance.last_arm_pulled

    @pytest.mark.parametrize("size", [1, 2])
    def test_sample(
        self,
        X: Optional[NDArray[np.float_]],
        restless_decorator: Optional[Callable[[type], type]],
        bandit_class: Type[Bandit],
        size: int,
    ) -> None:
        klass = bandit_class

        if X is not None:
            klass = contextual(klass)

        if restless_decorator is not None:
            klass = restless_decorator(klass)

        instance = klass()

        if X is None:
            # check that the sample method returns the correct number of samples
            sample = instance.sample(size=size)
        else:
            sample = instance.sample(X, size=size)

        assert len(np.array(sample)) == size

    def test_update(
        self,
        X: Optional[NDArray[np.float_]],
        restless_decorator: Optional[Callable[[type], type]],
        bandit_class: Type[Bandit],
    ) -> None:
        klass = bandit_class

        if X is not None:
            klass = contextual(klass)

        if restless_decorator is not None:
            klass = restless_decorator(klass)

        instance = klass()

        pull_args: List[Any] = []
        pull_kwargs: Dict[str, Any] = {}
        if X is not None:
            pull_args.append(X)
        if bandit_class._delayed_reward:
            pull_kwargs["unique_id"] = 1

        instance.pull(*pull_args, **pull_kwargs)
        instance.update(*pull_args, 1, **pull_kwargs)  # type: ignore

        # check that the learner was updated with the correct reward
        assert instance.last_arm_pulled is not None
        assert check_is_fitted(instance.last_arm_pulled.learner) is None

    def test_context_exceptions(
        self,
        X: Optional[NDArray[np.float_]],
        restless_decorator: Optional[Callable[[type], type]],
        bandit_class: Type[Bandit],
    ) -> None:
        klass = bandit_class

        if X is not None:
            klass = contextual(klass)

        if restless_decorator is not None:
            klass = restless_decorator(klass)

        instance = klass()

        pull_kwargs: Dict[str, Any] = {}

        if bandit_class._delayed_reward:
            pull_kwargs["unique_id"] = 1

        if X is None:
            # check that a ValueError is raised if the context is provided
            with pytest.raises(ValueError):
                instance.pull(np.array([[1]]), **pull_kwargs)

            with pytest.raises(ValueError):
                instance.sample(np.array([[1]]), **pull_kwargs)

            instance.pull(**pull_kwargs)
            with pytest.raises(ValueError):
                instance.update(np.array([[1]]), 1, **pull_kwargs)
        else:
            # check that a ValueError is raised if the context is not provided
            with pytest.raises(ValueError):
                instance.pull(**pull_kwargs)

            with pytest.raises(ValueError):
                instance.sample(**pull_kwargs)

            instance.pull(X, **pull_kwargs)
            with pytest.raises(ValueError):
                instance.update("a", **pull_kwargs)
