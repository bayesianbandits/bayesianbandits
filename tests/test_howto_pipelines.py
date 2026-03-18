"""Validate code snippets from docs/howto/pipelines.rst."""

import numpy as np
import pandas as pd
from sklearn.compose import ColumnTransformer
from sklearn.feature_extraction import DictVectorizer
from sklearn.preprocessing import OneHotEncoder, StandardScaler

from bayesianbandits import (
    AgentPipeline,
    Arm,
    ArmColumnFeaturizer,
    ContextualAgent,
    LearnerPipeline,
    LipschitzContextualAgent,
    NormalRegressor,
    ThompsonSampling,
)


def test_dict_vectorizer_pipeline():
    """Dict/JSON input with DictVectorizer."""
    vectorizer = DictVectorizer(sparse=True)
    vectorizer.fit(
        [
            {"user_age": 25, "region": "US"},
            {"user_age": 40, "region": "EU"},
        ]
    )

    arms = [
        Arm(f"variant_{i}", learner=NormalRegressor(alpha=1.0, beta=1.0, sparse=True))
        for i in range(3)
    ]
    agent = ContextualAgent(arms, ThompsonSampling(), random_seed=42)

    pipeline = AgentPipeline(
        steps=[("vectorize", vectorizer)],
        final_agent=agent,
    )

    contexts = [{"user_age": 30, "region": "US"}]
    (action,) = pipeline.pull(contexts)
    assert action in ("variant_0", "variant_1", "variant_2")

    pipeline.update(contexts, y=np.array([1.0]))

    updated_arm = pipeline.arm(action)
    assert hasattr(updated_arm.learner, "coef_")


def test_dataframe_column_transformer_pipeline():
    """DataFrame input with ColumnTransformer."""
    sample_df = pd.DataFrame(
        {
            "age": [25, 40, 35],
            "region": ["US", "EU", "US"],
        }
    )
    ct = ColumnTransformer(
        [
            ("num", StandardScaler(), ["age"]),
            ("cat", OneHotEncoder(sparse_output=False), ["region"]),
        ]
    )
    ct.fit(sample_df)

    arms = [
        Arm(f"variant_{i}", learner=NormalRegressor(alpha=1.0, beta=1.0))
        for i in range(3)
    ]
    agent = ContextualAgent(arms, ThompsonSampling(), random_seed=42)
    pipeline = AgentPipeline(
        steps=[("preprocess", ct)],
        final_agent=agent,
    )

    df = pd.DataFrame({"age": [30], "region": ["US"]})
    (action,) = pipeline.pull(df)
    assert action in ("variant_0", "variant_1", "variant_2")

    pipeline.update(df, y=np.array([1.0]))


def test_standard_scaler_pipeline():
    """Scaling raw numeric features."""
    rng = np.random.default_rng(0)
    historical_features = rng.standard_normal((50, 3))

    scaler = StandardScaler()
    scaler.fit(historical_features)

    arms = [
        Arm(f"variant_{i}", learner=NormalRegressor(alpha=1.0, beta=1.0))
        for i in range(3)
    ]
    agent = ContextualAgent(arms, ThompsonSampling(), random_seed=42)
    pipeline = AgentPipeline(
        steps=[("scale", scaler)],
        final_agent=agent,
    )

    X = rng.standard_normal((5, 3))
    actions = pipeline.pull(X)
    assert len(actions) == 5

    y = rng.standard_normal(5)
    pipeline.update(X, y)


def test_learner_pipeline_lipschitz():
    """LearnerPipeline as shared learner in LipschitzContextualAgent."""
    arm_tokens = ["product_0", "product_1", "product_2"]

    # Pre-fit a ColumnTransformer on representative enriched data
    sample_enriched = pd.DataFrame(
        {
            "age": [25, 40, 35] * 3,
            "score": [0.8, 0.6, 0.9] * 3,
            "arm": arm_tokens * 3,
        }
    )
    ct = ColumnTransformer(
        [
            ("num", StandardScaler(), ["age", "score"]),
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
        random_seed=42,
    )

    context = pd.DataFrame({"age": [30, 25], "score": [0.7, 0.5]})
    actions = agent.pull(context)
    assert len(actions) == 2
    assert all(a in arm_tokens for a in actions)

    y = np.array([1.0, 0.5])
    agent.update(context, y)


def test_learner_pipeline_lipschitz_hybrid():
    """Hybrid design matrix: shared context coefficients + per-arm offsets."""
    arm_tokens = ["product_0", "product_1", "product_2"]

    sample_enriched = pd.DataFrame(
        {
            "age": [25, 40, 35] * 3,
            "score": [0.8, 0.6, 0.9] * 3,
            "arm": arm_tokens * 3,
        }
    )
    ct = ColumnTransformer(
        [
            ("num", "passthrough", ["age", "score"]),
            ("arm", OneHotEncoder(sparse_output=False), ["arm"]),
        ]
    )
    ct.fit(sample_enriched)

    # Design matrix: [age, score, is_product_0, is_product_1, is_product_2]
    transformed = ct.transform(sample_enriched)
    assert transformed.shape[1] == 5  # 2 context + 3 one-hot

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
        random_seed=42,
    )

    context = pd.DataFrame({"age": [30], "score": [0.7]})
    (action,) = agent.pull(context)
    assert action in arm_tokens

    agent.update(context, y=np.array([1.0]))


def test_pipeline_indexing():
    """Accessing pipeline internals by name and position."""
    vectorizer = DictVectorizer(sparse=True)
    vectorizer.fit([{"a": 1}])

    arms = [Arm(0, learner=NormalRegressor(alpha=1.0, beta=1.0, sparse=True))]
    agent = ContextualAgent(arms, ThompsonSampling())
    pipeline = AgentPipeline(
        steps=[("vectorize", vectorizer)],
        final_agent=agent,
    )

    # By name
    assert pipeline["vectorize"] is vectorizer

    # By position
    name, transformer = pipeline[0]
    assert name == "vectorize"
    assert transformer is vectorizer

    # Named steps
    assert "vectorize" in pipeline.named_steps


def test_agent_methods_forwarded():
    """Agent methods (add_arm, remove_arm, etc.) work through pipeline."""
    arms = [
        Arm("a", learner=NormalRegressor(alpha=1.0, beta=1.0)),
        Arm("b", learner=NormalRegressor(alpha=1.0, beta=1.0)),
    ]
    agent = ContextualAgent(arms, ThompsonSampling(), random_seed=42)
    pipeline = AgentPipeline(
        steps=[("scale", StandardScaler().fit(np.zeros((1, 2))))],
        final_agent=agent,
    )

    assert len(pipeline.arms) == 2

    new_arm = Arm("c", learner=NormalRegressor(alpha=1.0, beta=1.0))
    pipeline.add_arm(new_arm)
    assert len(pipeline.arms) == 3

    pipeline.remove_arm("c")
    assert len(pipeline.arms) == 2

    result = pipeline.select_for_update("a")
    assert result is pipeline  # returns self for chaining
