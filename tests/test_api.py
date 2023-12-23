from typing import Union

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
    ContextualMultiArmedBandit,
    MultiArmedBandit,
    Policy,
    epsilon_greedy,
    thompson_sampling,
    upper_confidence_bound,
)


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
def learner_class(request: pytest.FixtureRequest):
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
) -> Policy:
    if request.param == "epsilon_greedy":
        return epsilon_greedy(0.8)
    elif request.param == "thompson_sampling":
        return thompson_sampling()
    elif request.param == "upper_confidence_bound":
        return upper_confidence_bound(0.68)
    else:
        raise ValueError("invalid param")


@pytest.fixture(params=[MultiArmedBandit, ContextualMultiArmedBandit])
def bandit_instance(
    request: pytest.FixtureRequest,
    learner_class,
    choice: Policy,
):
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
        bandit_instance,
    ) -> None:
        if isinstance(bandit_instance, MultiArmedBandit):
            (token,) = bandit_instance.pull()

        else:
            (token,) = bandit_instance.pull(np.array([[2.0]]))

        assert bandit_instance.arm_to_update.action_token == token

    def test_update(
        self,
        bandit_instance,
    ) -> None:
        if isinstance(bandit_instance, MultiArmedBandit):
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
        bandit_instance,
    ) -> None:
        if isinstance(bandit_instance, MultiArmedBandit):
            bandit_instance.decay()
        else:
            bandit_instance.decay(np.array([[2.0]]), decay_rate=0.5)

        # Check that no exception is raised

    def test_arm(
        self,
        bandit_instance,
    ) -> None:
        bandit_instance.arm(0)
        assert bandit_instance.arm_to_update is bandit_instance.arms[0]

        with pytest.raises(KeyError):
            bandit_instance.arm(3)

    def test_add_arm(
        self,
        bandit_instance,
    ) -> None:
        arm_to_add = Arm(2, None, learner=clone(bandit_instance.arms[0].learner))
        bandit_instance.add_arm(arm_to_add)
        assert len(bandit_instance.arms) == 3

        with pytest.raises(ValueError):
            bandit_instance.add_arm(arm_to_add)

    def test_remove_arm(
        self,
        bandit_instance,
    ) -> None:
        bandit_instance.remove_arm(0)
        assert len(bandit_instance.arms) == 1

        with pytest.raises(KeyError):
            bandit_instance.remove_arm(0)

    def test_constructor_exceptions(
        self,
    ):
        with pytest.raises(ValueError):
            MultiArmedBandit([], epsilon_greedy())

        with pytest.raises(ValueError):
            MultiArmedBandit(
                [
                    Arm(0, None, learner=NormalInverseGammaRegressor()),
                    Arm(0, None, learner=NormalInverseGammaRegressor()),
                ],
                epsilon_greedy(),
            )

        with pytest.raises(ValueError):
            MultiArmedBandit(
                [
                    Arm(0, None),
                ],
                epsilon_greedy(),
            )
