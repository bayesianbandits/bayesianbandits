from functools import partial
from typing import Any, Callable, Dict, List, Optional, Type, TypeVar, Union

import numpy as np
import pytest
import scipy.sparse as sp
from numpy.typing import NDArray
from sklearn.base import check_is_fitted  # type: ignore

from bayesianbandits import (
    Arm,
    DirichletClassifier,
    GammaRegressor,
    NormalInverseGammaRegressor,
    NormalRegressor,
)
from bayesianbandits._basebandit import (
    Bandit,
    DelayedRewardException,
    DelayedRewardWarning,
    contextual,
    restless,
)
from bayesianbandits._policy_decorators import (
    ArmChoicePolicy,
    epsilon_greedy,
    thompson_sampling,
    upper_confidence_bound,
)
from bayesianbandits._typing import Learner


@pytest.fixture(
    params=[
        "dirichlet",
        "gamma",
        "normal",
        "normal sparse",
        "normal-inverse-gamma",
        "normal-inverse-gamma sparse",
    ]
)
def learner_class(request: pytest.FixtureRequest) -> Learner:
    if request.param == "dirichlet":
        return DirichletClassifier({1: 1.0, 2: 1.0})
    elif request.param == "gamma":
        return GammaRegressor(alpha=1, beta=1)
    elif request.param == "normal":
        return NormalRegressor(alpha=1, beta=1)
    elif request.param == "normal sparse":
        return NormalRegressor(alpha=1, beta=1, sparse=True)
    elif request.param == "normal-inverse-gamma":
        return NormalInverseGammaRegressor()
    elif request.param == "normal-inverse-gamma sparse":
        return NormalInverseGammaRegressor(sparse=True)
    else:
        raise ValueError("invalid param")


@pytest.fixture(
    params=["epsilon_greedy", "thompson_sampling", "upper_confidence_bound"]
)
def choice(
    request: pytest.FixtureRequest,
) -> ArmChoicePolicy:
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
    choice: ArmChoicePolicy,
    delayed_reward: bool,
) -> type:
    if isinstance(learner_class, DirichletClassifier):

        def reward_func(x: NDArray[np.float_]) -> Union[NDArray[np.float_], np.float_]:
            return np.take(x, 0, axis=-1)  # type: ignore

    else:
        reward_func = None  # type: ignore

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


B = TypeVar("B", bound=Bandit)


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
        restless_decorator: Optional[Callable[[Type[Bandit]], Type[Bandit]]],
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
        if bandit_class._delayed_reward:  # type: ignore
            pull_kwargs["unique_id"] = 1

        instance.pull(*pull_args, **pull_kwargs)

        # check that the last arm pulled is not None and is one of the arms
        assert instance.last_arm_pulled is not None
        assert instance.last_arm_pulled in instance.arms.values()

        if bandit_class._delayed_reward:  # type: ignore
            # check that the last arm pulled is in the cache
            assert instance.cache is not None
            cached_arm = instance.arms[instance.cache[1]]
            assert cached_arm is instance.last_arm_pulled

    @pytest.mark.parametrize("size", [1, 2])
    def test_sample(
        self,
        X: Optional[NDArray[np.float_]],
        restless_decorator: Optional[Callable[[Type[Bandit]], Type[Bandit]]],
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
        restless_decorator: Optional[Callable[[Type[Bandit]], Type[Bandit]]],
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
        if bandit_class._delayed_reward:  # type: ignore
            pull_kwargs["unique_id"] = 1

        instance.pull(*pull_args, **pull_kwargs)
        instance.update(*pull_args, 1, **pull_kwargs)  # type: ignore

        # check that the learner was updated with the correct reward
        assert instance.last_arm_pulled is not None
        assert check_is_fitted(instance.last_arm_pulled.learner) is None  # type: ignore

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

        if bandit_class._delayed_reward:  # type: ignore
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


def test_bandit_arms_with_existing_learners() -> None:
    def reward_func(x: NDArray[np.float_]) -> Union[NDArray[np.float_], np.float_]:
        return np.take(x, 0, axis=-1)  # type: ignore

    learner_class = GammaRegressor(1, 2)  # set this to a non-Dirichlet learner

    class Experiment(  # type: ignore
        Bandit, learner=learner_class, policy=thompson_sampling(), delayed_reward=True
    ):
        arm1 = Arm(0, reward_func, learner=DirichletClassifier({1: 1.0, 2: 1.0}))
        arm2 = Arm(1, reward_func, learner=DirichletClassifier({1: 6.0, 2: 1.0}))

    instance = Experiment()

    assert isinstance(instance.arms["arm1"].learner, DirichletClassifier)
    assert instance.arms["arm1"].learner.alphas[1] == 1.0
    assert isinstance(instance.arms["arm2"].learner, DirichletClassifier)
    assert instance.arms["arm2"].learner.alphas[1] == 6.0


@pytest.mark.parametrize("sparse", [True, False])
def test_bandit_batch_pull_and_update(sparse: bool) -> None:
    class Experiment(
        Bandit,
        learner=NormalInverseGammaRegressor(sparse=sparse),
        policy=thompson_sampling(),
        delayed_reward=True,
    ):
        arm1 = Arm(0)
        arm2 = Arm(1)

    instance = Experiment(rng=0)

    instance.pull(unique_id=[1, 2, 3])

    instance.update([1, 2, 1], unique_id=[1, 2, 3])

    # 0.1 + 0.5 - one update. this test could break if the rng changes
    assert instance.arm1.learner.a_ == 0.1  # type: ignore
    # 0.1 + 0.5 + 0.5 - two updates. this test could break if the rng changes
    assert instance.arm2.learner.a_ == 1.6  # type: ignore


@pytest.mark.parametrize("sparse", [True, False])
def test_bandit_batch_pull_and_update_single(sparse: bool) -> None:
    class Experiment(
        Bandit,
        learner=NormalInverseGammaRegressor(sparse=sparse),
        policy=thompson_sampling(),
        delayed_reward=True,
    ):
        arm1 = Arm(0)
        arm2 = Arm(1)

    instance = Experiment(rng=0)

    instance.pull(unique_id=[1])
    instance.update([1], unique_id=[1])

    # 0.1 + 0.5 - one update. this test could break if the rng changes
    assert instance.arm2.learner.a_ == 0.6  # type: ignore


@pytest.mark.parametrize("sparse", [True, False])
def test_contextual_bandit_batch_pull_and_update(sparse: bool) -> None:
    @contextual
    class Experiment(
        Bandit,
        learner=NormalInverseGammaRegressor(sparse=sparse),
        policy=thompson_sampling(),
        delayed_reward=True,
    ):
        arm1 = Arm(0)
        arm2 = Arm(1)

    instance = Experiment(rng=0)

    X = np.array([[1, 2], [3, 4], [5, 6]])
    if sparse:
        X = sp.csc_array(X)

    instance.pull(X, unique_id=[1, 2, 3])
    instance.update(X, [1, 2, 1], unique_id=[1, 2, 3])

    # 0.1 + 0.5 - one update. this test could break if the rng changes
    assert instance.arm1.learner.a_ == 1.1  # type: ignore
    # 0.1 + 0.5 + 0.5 - two updates. this test could break if the rng changes
    assert instance.arm2.learner.a_ == 0.6  # type: ignore


@pytest.mark.parametrize("sparse", [True, False])
def test_contextual_bandit_batch_pull_and_update_single(sparse: bool) -> None:
    @contextual
    class Experiment(
        Bandit,
        learner=NormalInverseGammaRegressor(sparse=sparse),
        policy=thompson_sampling(),
        delayed_reward=True,
    ):
        arm1 = Arm(0)
        arm2 = Arm(1)

    instance = Experiment(rng=0)

    X = np.array([[1, 2]])
    if sparse:
        X = sp.csc_array(X)

    instance.pull(X, unique_id=[1])
    instance.update(X, [1], unique_id=[1])

    # 0.1 + 0.5 - one update. this test could break if the rng changes
    assert instance.arm2.learner.a_ == 0.6  # type: ignore


@pytest.mark.parametrize("sparse", [True, False])
def test_contextual_bandit_batch_pull_length_mismatch_exception(sparse: bool) -> None:
    @contextual
    class Experiment(
        Bandit,
        learner=NormalInverseGammaRegressor(sparse=sparse),
        policy=thompson_sampling(),
        delayed_reward=True,
    ):
        arm1 = Arm(0)
        arm2 = Arm(1)

    instance = Experiment(rng=0)

    X = np.array([[1, 2], [3, 4], [5, 6]])
    if sparse:
        X = sp.csc_array(X)

    with pytest.raises(ValueError):
        instance.pull(X, unique_id=[1, 2])

    instance.pull(X, unique_id=[1, 2, 3])

    with pytest.raises(ValueError):
        instance.update(X, [1, 2, 3], unique_id=[1, 2])

    with pytest.raises(ValueError):
        instance.update(X, [1, 2], unique_id=[1, 2, 3])


def test_contextual_bandit_batch_without_restless_exception() -> None:
    @contextual
    class Experiment(
        Bandit,
        learner=NormalInverseGammaRegressor(),
        policy=thompson_sampling(),
        delayed_reward=False,
    ):
        arm1 = Arm(0)
        arm2 = Arm(1)

    instance = Experiment(rng=0)

    X = np.array([[1, 2], [3, 4], [5, 6]])

    with pytest.raises(ValueError):
        instance.pull(X)


def test_delayed_reward_reused_unique_id_exception() -> None:
    class Experiment(
        Bandit,
        learner=NormalInverseGammaRegressor(),
        policy=thompson_sampling(),
        delayed_reward=True,
    ):
        arm1 = Arm(0)
        arm2 = Arm(1)

    instance = Experiment(rng=0)

    instance.pull(unique_id=1)

    with pytest.raises(DelayedRewardException):
        instance.pull(unique_id=1)


def test_delayed_reward_batch_reused_unique_id_exception() -> None:
    class Experiment(
        Bandit,
        learner=NormalInverseGammaRegressor(),
        policy=thompson_sampling(),
        delayed_reward=True,
    ):
        arm1 = Arm(0)
        arm2 = Arm(1)

    instance = Experiment(rng=0)

    instance.pull(unique_id=[1, 2, 3])

    with pytest.raises(DelayedRewardException):
        instance.pull(unique_id=[1, 2, 4])


def test_delayed_reward_update_unknown_unique_id_exception() -> None:
    class Experiment(
        Bandit,
        learner=NormalInverseGammaRegressor(),
        policy=thompson_sampling(),
        delayed_reward=True,
    ):
        arm1 = Arm(0)
        arm2 = Arm(1)

    instance = Experiment(rng=0)

    with pytest.raises(DelayedRewardException):
        instance.update(1, unique_id=1)


def test_delayed_reward_batch_update_unknown_unique_id_warning() -> None:
    class Experiment(
        Bandit,
        learner=NormalInverseGammaRegressor(),
        policy=thompson_sampling(),
        delayed_reward=True,
    ):
        arm1 = Arm(0)
        arm2 = Arm(1)

    instance = Experiment(rng=0)
    instance.pull(unique_id=3)

    # update two non-existent unique_ids
    with pytest.raises(DelayedRewardException):
        instance.update([1, 2], unique_id=[1, 2])

    assert instance.arm1.learner.a_ == 0.1  # type: ignore
    assert instance.arm2.learner.a_ == 0.1  # type: ignore


def test_delayed_reward_batch_update_known_and_unknown_unique_id_warning() -> None:
    class Experiment(
        Bandit,
        learner=NormalInverseGammaRegressor(),
        policy=thompson_sampling(),
        delayed_reward=True,
    ):
        arm1 = Arm(0)
        arm2 = Arm(1)

    instance = Experiment(rng=0)
    instance.pull(unique_id=1)

    # update one non-existent unique_id, one valid unique_id
    with pytest.warns(DelayedRewardWarning, match="1 unique_ids"):
        instance.update([1, 2], unique_id=[1, 2])

    assert instance.arm1.learner.a_ == 0.1  # type: ignore
    assert instance.arm2.learner.a_ == 0.6  # type: ignore


class UnhashableClass:
    __hash__ = None  # type: ignore


def test_delayed_reward_bad_unique_id_errors() -> None:
    class Experiment(
        Bandit,
        learner=NormalInverseGammaRegressor(),
        policy=thompson_sampling(),
        delayed_reward=True,
    ):
        arm1 = Arm(0)
        arm2 = Arm(1)

    instance = Experiment(rng=0)

    with pytest.raises(ValueError, match="hashable"):
        instance.pull(unique_id=[[1]])

    with pytest.raises(ValueError, match="hashable"):
        instance.pull(unique_id=[UnhashableClass()])

    with pytest.raises(ValueError, match="hashable"):
        instance.pull(unique_id=UnhashableClass())

    with pytest.raises(ValueError, match="hashable"):
        instance.pull(unique_id=[1, [2]])

    with pytest.raises(DelayedRewardException):
        instance.pull()

    with pytest.raises(DelayedRewardException):
        instance.update(1)
