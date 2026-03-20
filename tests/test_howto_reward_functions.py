"""Validate code snippets from docs/howto/reward-functions.rst."""

import numpy as np
import pandas as pd
from sklearn.compose import ColumnTransformer
from sklearn.preprocessing import OneHotEncoder, StandardScaler

from bayesianbandits import (
    Arm,
    ArmColumnFeaturizer,
    BayesianGLM,
    ContextualAgent,
    LearnerPipeline,
    LipschitzContextualAgent,
    NormalRegressor,
    ThompsonSampling,
)
from bayesianbandits._arm import _accepts_context, is_identity_function


def make_profit_reward(revenue, cost):
    """Expected profit = P(convert) * revenue - cost."""

    def reward(samples):
        return samples * revenue - cost

    return reward


def test_identity_is_default():
    """Arm with no reward_function uses identity."""
    arm = Arm("test", learner=NormalRegressor(alpha=1.0, beta=1.0))
    assert is_identity_function(arm.reward_function)

    X = np.array([[1.0]])
    arm.learner.fit(X, np.array([1.0]))

    samples = arm.sample(X, size=5)
    assert samples.shape == (5, 1)
    assert isinstance(samples, np.ndarray)


def test_binary_outcome_profit_reward():
    """BayesianGLM with profit reward: reward = P(convert) * revenue - cost."""
    reward_fn = make_profit_reward(revenue=50.0, cost=10.0)

    learner = BayesianGLM(alpha=1.0, link="logit")
    Arm("campaign_A", reward_function=reward_fn, learner=learner)

    X = np.array([[1.0, 0.0]])
    # BayesianGLM logit samples are in [0, 1] -- probabilities
    raw_samples = learner.sample(X, size=100)
    assert raw_samples.shape == (100, 1)
    assert np.all((raw_samples >= 0) & (raw_samples <= 1))

    expected = raw_samples * 50.0 - 10.0
    actual = reward_fn(raw_samples)
    np.testing.assert_allclose(actual, expected)


def test_binary_outcome_contextual_agent():
    """ContextualAgent with BayesianGLM arms: pull selects, update gets raw 0/1."""
    arms = [
        Arm(
            "campaign_A",
            reward_function=make_profit_reward(revenue=50.0, cost=10.0),
            learner=BayesianGLM(alpha=1.0, link="logit"),
        ),
        Arm(
            "campaign_B",
            reward_function=make_profit_reward(revenue=20.0, cost=2.0),
            learner=BayesianGLM(alpha=1.0, link="logit"),
        ),
    ]
    agent = ContextualAgent(arms, ThompsonSampling(), random_seed=42)

    X = np.array([[1.0, 0.0, 25.0]])
    (action,) = agent.pull(X)
    assert action in ("campaign_A", "campaign_B")

    # Update with raw binary outcome
    agent.update(X, np.array([1]))

    # Learner should now be fitted
    updated_arm = agent.arm(action)
    assert hasattr(updated_arm.learner, "coef_")


def test_context_aware_reward():
    """Context-aware reward subtracts cost column from samples."""

    def net_profit_reward(samples, X):
        cost = X[:, -1]
        return samples - cost

    arms = [
        Arm(
            f"product_{i}",
            reward_function=net_profit_reward,
            learner=NormalRegressor(alpha=1.0, beta=1.0),
        )
        for i in range(3)
    ]
    agent = ContextualAgent(arms, ThompsonSampling(), random_seed=42)

    X = np.array([[5.0, 1.2]])
    (action,) = agent.pull(X)
    assert action in ("product_0", "product_1", "product_2")

    agent.update(X, y=np.array([6.5]))

    # Verify the function directly
    samples = np.array([[10.0]])
    context = np.array([[5.0, 3.0]])
    result = net_profit_reward(samples, context)
    np.testing.assert_allclose(result, np.array([[7.0]]))


def test_context_parameter_must_be_named_X():
    """Only functions with a parameter literally named X are context-aware."""

    def with_X(samples, X):
        return samples

    def with_context(samples, context):
        return samples

    def no_context(samples):
        return samples

    assert _accepts_context(with_X) is True
    assert _accepts_context(with_context) is False
    assert _accepts_context(no_context) is False


def test_nonlinear_utility():
    """Asymmetric loss penalizes undershoot 3x."""

    def asymmetric_loss(samples, target=10.0, penalty=3.0):
        diff = samples - target
        return np.where(diff < 0, penalty * diff, diff)

    samples = np.array([8.0, 10.0, 12.0])
    result = asymmetric_loss(samples)
    # 8 -> (8-10)*3 = -6, 10 -> 0, 12 -> 2
    np.testing.assert_allclose(result, np.array([-6.0, 0.0, 2.0]))


def test_batch_reward_function():
    """Batch margin function on LipschitzContextualAgent."""
    MARGINS = {
        "product_0": 0.3,
        "product_1": 0.5,
        "product_2": 0.8,
    }

    def margin_reward(samples, action_tokens):
        multipliers = np.array([MARGINS[t] for t in action_tokens])
        return samples * multipliers[:, np.newaxis, np.newaxis]

    arm_tokens = list(MARGINS.keys())

    sample_enriched = pd.DataFrame(
        {
            "score": [0.8, 0.6, 0.9] * 3,
            "arm": arm_tokens * 3,
        }
    )
    ct = ColumnTransformer(
        [
            ("num", StandardScaler(), ["score"]),
            ("arm", OneHotEncoder(sparse_output=False), ["arm"]),
        ]
    )
    ct.fit(sample_enriched)

    shared_learner = LearnerPipeline(
        steps=[("preprocess", ct)],
        learner=NormalRegressor(alpha=1.0, beta=1.0),
    )
    arms = [Arm(token) for token in arm_tokens]
    agent = LipschitzContextualAgent(
        arms=arms,
        policy=ThompsonSampling(),
        arm_featurizer=ArmColumnFeaturizer("arm"),
        learner=shared_learner,
        batch_reward_function=margin_reward,
        random_seed=42,
    )

    context = pd.DataFrame({"score": [0.7]})
    (action,) = agent.pull(context)
    assert action in arm_tokens

    # Verify the batch function directly
    samples = np.ones((3, 1, 10))
    result = margin_reward(samples, arm_tokens)
    assert result.shape == (3, 1, 10)
    np.testing.assert_allclose(result[0], 0.3)
    np.testing.assert_allclose(result[1], 0.5)
    np.testing.assert_allclose(result[2], 0.8)


def test_update_trains_on_raw_outcomes():
    """Learner sees raw y, not reward(y)."""

    def double_reward(samples):
        return samples * 2.0

    learner = NormalRegressor(alpha=1.0, beta=1.0)
    arm = Arm("test", reward_function=double_reward, learner=learner)

    X = np.array([[1.0]])
    y = np.array([5.0])
    arm.update(X, y)

    # coef_ = (X^T X + I)^{-1} X^T y = (1+1)^{-1} * 5 = 2.5
    # If reward were applied, it would be 5.0
    prediction = learner.predict(X)
    np.testing.assert_allclose(prediction, np.array([2.5]))
