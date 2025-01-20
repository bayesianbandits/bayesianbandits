from typing import Any, TypeVar, Union

import numpy as np
import pytest
from numpy.typing import NDArray
from sklearn.base import (
    check_is_fitted,  # type: ignore
    clone,
)

from bayesianbandits import (
    Agent,
    Arm,
    ContextualAgent,
    DirichletClassifier,
    EpsilonGreedy,
    GammaRegressor,
    NormalInverseGammaRegressor,
    NormalRegressor,
    ThompsonSampling,
    UpperConfidenceBound,
)
from bayesianbandits._typing import DecayingLearner
from bayesianbandits.api import PolicyProtocol


@pytest.fixture(
    params=[
        DirichletClassifier({1: 1.0, 2: 1.0}),
        GammaRegressor(alpha=1, beta=1),
        NormalRegressor(alpha=1, beta=1),
        NormalRegressor(alpha=1, beta=1, sparse=True),
        NormalInverseGammaRegressor(),
        NormalInverseGammaRegressor(sparse=True),
    ],
    ids=[
        "dirichlet",
        "gamma",
        "normal",
        "normal sparse",
        "normal-inverse-gamma",
        "normal-inverse-gamma sparse",
    ],
)
def learner_class(
    request: pytest.FixtureRequest,
) -> Union[
    DirichletClassifier, GammaRegressor, NormalRegressor, NormalInverseGammaRegressor
]:
    return request.param


@pytest.fixture(
    params=[
        EpsilonGreedy(0.8),
        ThompsonSampling(),
        UpperConfidenceBound(0.68),
    ]
)
def choice(
    request: pytest.FixtureRequest,
) -> PolicyProtocol[Any, Any]:
    return request.param


LT = TypeVar("LT", bound=DecayingLearner)


@pytest.fixture(params=[Agent, ContextualAgent])
def bandit_instance(
    request: pytest.FixtureRequest,
    learner_class: Union[
        DirichletClassifier,
        GammaRegressor,
        NormalRegressor,
        NormalInverseGammaRegressor,
    ],
    choice: PolicyProtocol[LT, int],
) -> Union[
    Agent[LT, int, PolicyProtocol[LT, int]],
    ContextualAgent[LT, int, PolicyProtocol[LT, int]],
]:
    if isinstance(learner_class, DirichletClassifier):

        def reward_func(
            x: NDArray[np.float64],
        ) -> Union[NDArray[np.float64], np.float64]:
            return x[..., 0].T

    else:
        reward_func = None  # type: ignore

    arms = [
        Arm(0, reward_func, learner=clone(learner_class)),
        Arm(1, reward_func, learner=clone(learner_class)),
    ]

    agent = request.param(arms, choice, random_seed=0)

    return agent


class TestBandits:
    def test_pull(
        self,
        bandit_instance: Union[
            Agent[DecayingLearner, int, PolicyProtocol[DecayingLearner, int]],
            ContextualAgent[DecayingLearner, int, PolicyProtocol[DecayingLearner, int]],
        ],
    ) -> None:
        if isinstance(bandit_instance, Agent):
            (token,) = bandit_instance.pull()

        else:
            (token,) = bandit_instance.pull(np.array([[2.0]]))

        assert bandit_instance.arm_to_update.action_token == token

    def test_batch_pull(
        self,
        bandit_instance: Union[
            Agent[DecayingLearner, int, PolicyProtocol[DecayingLearner, int]],
            ContextualAgent[DecayingLearner, int, PolicyProtocol[DecayingLearner, int]],
        ],
    ) -> None:
        if isinstance(bandit_instance, Agent):
            pytest.skip("Batch pull is not supported for non-contextual bandits")

        # if isinstance(bandit_instance.arm(0).learner, DirichletClassifier):
        #     pytest.xfail("DirichletClassifier does not support batch pull")

        (_, _, token3) = bandit_instance.pull(np.array([[2.0], [2.0], [2.0]]))

        assert bandit_instance.arm_to_update.action_token == token3

    def test_update(
        self,
        bandit_instance: Union[
            Agent[DecayingLearner, int, PolicyProtocol[DecayingLearner, int]],
            ContextualAgent[DecayingLearner, int, PolicyProtocol[DecayingLearner, int]],
        ],
    ) -> None:
        if isinstance(bandit_instance, Agent):
            bandit_instance.pull()
            bandit_instance.update(np.array([1.0]))
        else:
            bandit_instance.pull(np.array([[2.0]]))
            bandit_instance.update(np.array([[2.0]]), np.array([1.0]))

        # check that the learner was updated with the correct reward
        assert bandit_instance.arm_to_update is not None
        assert check_is_fitted(bandit_instance.arm_to_update.learner) is None  # type: ignore

    def test_decay(
        self,
        bandit_instance: Union[
            Agent[DecayingLearner, int, PolicyProtocol[DecayingLearner, int]],
            ContextualAgent[DecayingLearner, int, PolicyProtocol[DecayingLearner, int]],
        ],
    ) -> None:
        if isinstance(bandit_instance, Agent):
            bandit_instance.decay()
        else:
            bandit_instance.decay(np.array([[2.0]]), decay_rate=0.5)

        # Check that no exception is raised

    def test_arm(
        self,
        bandit_instance: Union[
            Agent[DecayingLearner, int, PolicyProtocol[DecayingLearner, int]],
            ContextualAgent[DecayingLearner, int, PolicyProtocol[DecayingLearner, int]],
        ],
    ) -> None:
        arm = bandit_instance.arm(0)
        assert arm is bandit_instance.arms[0]

        with pytest.raises(KeyError):
            bandit_instance.arm(3)

    def test_select_for_update(
        self,
        bandit_instance: Union[
            Agent[DecayingLearner, int, PolicyProtocol[DecayingLearner, int]],
            ContextualAgent[DecayingLearner, int, PolicyProtocol[DecayingLearner, int]],
        ],
    ) -> None:
        bandit_instance.select_for_update(1)
        assert bandit_instance.arm_to_update is bandit_instance.arms[1]

        with pytest.raises(KeyError):
            bandit_instance.select_for_update(3)

    def test_add_arm(
        self,
        bandit_instance: Union[
            Agent[DecayingLearner, int, PolicyProtocol[DecayingLearner, int]],
            ContextualAgent[DecayingLearner, int, PolicyProtocol[DecayingLearner, int]],
        ],
    ) -> None:
        arm_to_add = Arm(2, None, learner=clone(bandit_instance.arms[0].learner))  # type: ignore
        bandit_instance.add_arm(arm_to_add)
        assert len(bandit_instance.arms) == 3

        with pytest.raises(ValueError):
            bandit_instance.add_arm(arm_to_add)

    def test_remove_arm(
        self,
        bandit_instance: Union[
            Agent[DecayingLearner, int, PolicyProtocol[DecayingLearner, int]],
            ContextualAgent[DecayingLearner, int, PolicyProtocol[DecayingLearner, int]],
        ],
    ) -> None:
        bandit_instance.remove_arm(0)
        assert len(bandit_instance.arms) == 1

        with pytest.raises(KeyError):
            bandit_instance.remove_arm(0)

    def test_change_policy(
        self,
        bandit_instance: Union[
            Agent[DecayingLearner, int, PolicyProtocol[DecayingLearner, int]],
            ContextualAgent[DecayingLearner, int, PolicyProtocol[DecayingLearner, int]],
        ],
    ) -> None:
        bandit_instance.policy = ThompsonSampling()

        assert isinstance(bandit_instance.policy, ThompsonSampling)

    def test_check_rng(
        self,
        bandit_instance: Union[
            Agent[DecayingLearner, int, PolicyProtocol[DecayingLearner, int]],
            ContextualAgent[DecayingLearner, int, PolicyProtocol[DecayingLearner, int]],
        ],
    ) -> None:
        assert isinstance(bandit_instance.rng, np.random.Generator)

    def test_constructor_exceptions(
        self,
    ):
        with pytest.raises(ValueError):
            Agent([], EpsilonGreedy())  # type: ignore

        with pytest.raises(ValueError):
            Agent(
                [
                    Arm(0, None, learner=NormalInverseGammaRegressor()),
                    Arm(0, None, learner=NormalInverseGammaRegressor()),
                ],
                EpsilonGreedy(),
            )


def test_contextual_agent_update_mismatched_shapes() -> None:
    with pytest.raises(
        ValueError,
        match="The number of rows in `X` must match the number of rows in `y`.",
    ):
        ContextualAgent(
            [
                Arm(0, None, learner=NormalInverseGammaRegressor()),
                Arm(1, None, learner=NormalInverseGammaRegressor()),
            ],
            EpsilonGreedy(),
            random_seed=0,
        ).update(np.array([[1.0]]), np.array([1.0, 2.0]))
