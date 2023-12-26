from typing import TypeVar, Union

import numpy as np
import pytest
from numpy.typing import NDArray
from sklearn.base import (
    check_is_fitted,  # type: ignore
    clone,
)

from bayesianbandits import (
    Arm,
    DirichletClassifier,
    GammaRegressor,
    NormalInverseGammaRegressor,
    NormalRegressor,
)
from bayesianbandits.api import (
    ContextualAgent,
    Agent,
    Policy,
    EpsilonGreedy,
    ThompsonSampling,
    UpperConfidenceBound,
)
from bayesianbandits._typing import DecayingLearner


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
def learner_class(
    request: pytest.FixtureRequest,
) -> Union[
    DirichletClassifier, GammaRegressor, NormalRegressor, NormalInverseGammaRegressor
]:
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


@pytest.fixture(params=["EpsilonGreedy", "ThompsonSampling", "UpperConfidenceBound"])
def choice(
    request: pytest.FixtureRequest,
) -> Policy:
    if request.param == "EpsilonGreedy":
        return EpsilonGreedy(0.8)
    elif request.param == "ThompsonSampling":
        return ThompsonSampling()
    elif request.param == "UpperConfidenceBound":
        return UpperConfidenceBound(0.68)
    else:
        raise ValueError("invalid param")


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
    choice: Policy,
) -> Union[Agent[LT, int], ContextualAgent[LT, int]]:
    if isinstance(learner_class, DirichletClassifier):

        def reward_func(x: NDArray[np.float_]) -> Union[NDArray[np.float_], np.float_]:
            return np.take(x, 0, axis=-1)  # type: ignore

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
            Agent[DecayingLearner, int], ContextualAgent[DecayingLearner, int]
        ],
    ) -> None:
        if isinstance(bandit_instance, Agent):
            (token,) = bandit_instance.pull()

        else:
            (token,) = bandit_instance.pull(np.array([[2.0]]))

        assert bandit_instance.arm_to_update.action_token == token

    def test_update(
        self,
        bandit_instance: Union[
            Agent[DecayingLearner, int], ContextualAgent[DecayingLearner, int]
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
            Agent[DecayingLearner, int], ContextualAgent[DecayingLearner, int]
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
            Agent[DecayingLearner, int], ContextualAgent[DecayingLearner, int]
        ],
    ) -> None:
        bandit_instance.arm(0)
        assert bandit_instance.arm_to_update is bandit_instance.arms[0]

        with pytest.raises(KeyError):
            bandit_instance.arm(3)

    def test_add_arm(
        self,
        bandit_instance: Union[
            Agent[DecayingLearner, int], ContextualAgent[DecayingLearner, int]
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
            Agent[DecayingLearner, int], ContextualAgent[DecayingLearner, int]
        ],
    ) -> None:
        bandit_instance.remove_arm(0)
        assert len(bandit_instance.arms) == 1

        with pytest.raises(KeyError):
            bandit_instance.remove_arm(0)

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

        with pytest.raises(ValueError):
            Agent(
                [
                    Arm(0, None),  # type: ignore
                ],
                EpsilonGreedy(),
            )
