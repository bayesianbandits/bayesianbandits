import joblib
from numpy.testing import assert_almost_equal


from bayesianbandits import Bandit, Arm, epsilon_greedy, GammaRegressor


def action1():
    print("action1")


def action2():
    print("action2")


def action3():
    print("action3")


def reward_func(x):
    return x


class Agent(Bandit, learner=GammaRegressor(1, 1), policy=epsilon_greedy()):
    arm1 = Arm(action1, reward_func)
    arm2 = Arm(action2, reward_func)
    arm3 = Arm(action3, reward_func)


def test_pickle_and_load():
    agent = Agent(rng=1)

    agent.pull()
    agent.update(1)
    assert_almost_equal(agent.arm1.learner.coef_[1], [2, 2])  # type: ignore

    joblib.dump(agent, "agent.pkl")

    loaded = joblib.load("agent.pkl")

    assert_almost_equal(loaded.arm1.learner.coef_[1], agent.arm1.learner.coef_[1])


def test_remove_arm():
    global Agent

    class Agent(Bandit, learner=GammaRegressor(1, 1), policy=epsilon_greedy()):
        arm1 = Arm(action1, reward_func)
        arm2 = Arm(action2, reward_func)

    loaded = joblib.load("agent.pkl")

    assert_almost_equal(loaded.arm1.learner.coef_[1], [2, 2])  # type: ignore

    assert not hasattr(loaded, "arm3")


def test_new_arm():
    global Agent

    class Agent(Bandit, learner=GammaRegressor(1, 1), policy=epsilon_greedy()):
        arm1 = Arm(action1, reward_func)
        arm2 = Arm(action2, reward_func)
        arm4 = Arm(action3, reward_func)

    loaded = joblib.load("agent.pkl")

    assert_almost_equal(loaded.arm1.learner.coef_[1], [2, 2])  # type: ignore

    assert not hasattr(loaded, "arm3")

    assert hasattr(loaded, "arm4")
