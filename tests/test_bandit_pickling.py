import tempfile

import joblib
from numpy.testing import assert_almost_equal
import pytest
from typing import IO
import numpy as np
from numpy.typing import NDArray


from bayesianbandits import (
    Bandit,
    Arm,
    epsilon_greedy,
    GammaRegressor,
    DelayedRewardWarning,
)


@pytest.fixture(scope="module")
def temp_model_file():
    with tempfile.NamedTemporaryFile() as f:
        yield f


def reward_func(x: NDArray[np.float64]) -> NDArray[np.float64]:
    return x


class Agent(Bandit, learner=GammaRegressor(1, 1), policy=epsilon_greedy()):
    arm1 = Arm("action1", reward_func)
    arm2 = Arm("action2", reward_func)
    arm3 = Arm("action3", reward_func)


def test_pickle_and_load(temp_model_file: IO[bytes]):
    agent = Agent(rng=1)

    agent.pull()
    agent.update(1)
    assert_almost_equal(agent.arm1.learner.coef_[1], [2, 2])  # type: ignore

    joblib.dump(agent, temp_model_file)
    temp_model_file.seek(0)

    loaded = joblib.load(temp_model_file)

    assert_almost_equal(loaded.arm1.learner.coef_[1], agent.arm1.learner.coef_[1])


def test_remove_arm(temp_model_file: IO[bytes]):
    global Agent

    class Agent(Bandit, learner=GammaRegressor(1, 1), policy=epsilon_greedy()):
        arm1 = Arm("action1", reward_func)
        arm2 = Arm("action2", reward_func)

    temp_model_file.seek(0)
    loaded = joblib.load(temp_model_file)

    assert_almost_equal(loaded.arm1.learner.coef_[1], [2, 2])  # type: ignore

    assert not hasattr(loaded, "arm3")


def test_new_arm(temp_model_file: IO[bytes]):
    global Agent

    class Agent(Bandit, learner=GammaRegressor(1, 1), policy=epsilon_greedy()):
        arm1 = Arm("action1", reward_func)
        arm2 = Arm("action2", reward_func)
        arm4 = Arm("action3", reward_func)

    temp_model_file.seek(0)
    loaded = joblib.load(temp_model_file)

    assert_almost_equal(loaded.arm1.learner.coef_[1], [2, 2])  # type: ignore

    assert not hasattr(loaded, "arm3")

    assert hasattr(loaded, "arm4")
    assert loaded.arm4.learner.random_state == loaded.arm1.learner.random_state


class DelayedRewardAgent(
    Bandit, learner=GammaRegressor(1, 1), policy=epsilon_greedy(), delayed_reward=True
):
    arm1 = Arm("action1", reward_func)
    arm2 = Arm("action2", reward_func)
    arm3 = Arm("action3", reward_func)


def test_removed_arm_update_warning(temp_model_file: IO[bytes]):
    global DelayedRewardAgent

    agent = DelayedRewardAgent(rng=1)
    agent.pull(unique_id=1)
    temp_model_file.seek(0)
    joblib.dump(agent, temp_model_file)

    class DelayedRewardAgent(
        Bandit,
        learner=GammaRegressor(1, 1),
        policy=epsilon_greedy(),
        delayed_reward=True,
    ):
        arm2 = Arm("action1", reward_func)
        arm3 = Arm("action2", reward_func)

    temp_model_file.seek(0)
    loaded = joblib.load(temp_model_file)

    with pytest.warns(DelayedRewardWarning):
        loaded.update(1, unique_id=1)


def test_removed_arm_update_warning_batch():
    global DelayedRewardAgent

    with tempfile.NamedTemporaryFile() as f:
        agent = DelayedRewardAgent(rng=1)
        agent.pull(unique_id=1)
        agent.pull(unique_id=2)
        joblib.dump(agent, f)

        class DelayedRewardAgent(
            Bandit,
            learner=GammaRegressor(1, 1),
            policy=epsilon_greedy(),
            delayed_reward=True,
        ):
            arm1 = Arm("action1", reward_func)
            arm3 = Arm("action2", reward_func)

        f.seek(0)
        loaded = joblib.load(f)

    with pytest.warns(DelayedRewardWarning):
        loaded.update([1, 2], unique_id=[1, 2])
