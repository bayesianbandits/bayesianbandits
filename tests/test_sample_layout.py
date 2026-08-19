"""The sampling layout contract.

Two guarantees, at the two boundaries where draws change hands:

**Learner boundary.** The ``(size, n)`` sampling entry points --
``sample``, ``sample_marginal``, ``sample_reward_space`` -- return
draw-contiguous arrays: ``samples.T`` is C-contiguous, i.e. each
prediction row's draws are adjacent in memory. This costs nothing (the
final projection is computed transposed, the same inner products the
other way round) and makes every downstream transposed reshape
contiguous for free. ``size == 1`` satisfies the contract in either
orientation, which is what lets Thompson sampling's exact BLAS call --
and hence its bit-for-bit draw stream -- stay untouched.

**Policy boundary.** Agents hand ``select`` an
``(n_arms, n_contexts, size)`` tensor whose draw axis is unit-stride.
For conforming learners this is the free consequence of the learner
contract; for a learner that only promises shape, the agent normalizes
with one slabbed copy rather than letting a policy discover that
draw-axis reductions run against the largest stride in the tensor.
"""

from typing import Any, List

import numpy as np
import pytest
from scipy.sparse import csc_array
from scipy.sparse import random as sparse_random

from bayesianbandits import (
    Arm,
    ArmColumnFeaturizer,
    BayesianGLM,
    ContextualAgent,
    LipschitzContextualAgent,
    NormalInverseGammaRegressor,
    NormalRegressor,
    ThompsonSampling,
)


def make_learner(kind: str, sparse: bool):
    if kind == "normal":
        return NormalRegressor(alpha=1.0, beta=1.0, sparse=sparse, random_state=7)
    if kind == "nig":
        return NormalInverseGammaRegressor(sparse=sparse, random_state=7)
    if kind == "glm_logit":
        return BayesianGLM(alpha=1.0, link="logit", sparse=sparse, random_state=7)
    if kind == "glm_log":
        return BayesianGLM(alpha=1.0, link="log", sparse=sparse, random_state=7)
    raise ValueError(kind)


def make_X(n_rows: int, n_feats: int, sparse: bool):
    if sparse:
        return csc_array(
            sparse_random(n_rows, n_feats, density=0.5, random_state=3, format="csc")
        )
    return np.random.default_rng(3).normal(size=(n_rows, n_feats))


def fit(learner, n_feats: int, sparse: bool):
    X = make_X(6, n_feats, sparse)
    y = np.arange(6, dtype=np.float64)
    learner.partial_fit(X, y)
    return learner


def assert_draw_contiguous(samples: np.ndarray, size: int, n_rows: int):
    assert samples.shape == (size, n_rows)
    assert samples.T.flags.c_contiguous


LEARNERS = ["normal", "nig", "glm_logit", "glm_log"]


class TestLearnerContract:
    """(size, n) entry points return draw-contiguous arrays."""

    @pytest.mark.parametrize("kind", LEARNERS)
    @pytest.mark.parametrize("sparse", [False, True])
    @pytest.mark.parametrize("size", [1, 7])
    @pytest.mark.parametrize("fitted", [True, False])
    def test_sample(self, kind: str, sparse: bool, size: int, fitted: bool):
        learner = make_learner(kind, sparse)
        if fitted:
            fit(learner, 5, sparse)
        X = make_X(4, 5, sparse)
        assert_draw_contiguous(learner.sample(X, size=size), size, 4)

    @pytest.mark.parametrize("kind", LEARNERS)
    @pytest.mark.parametrize("sparse", [False, True])
    @pytest.mark.parametrize("size", [1, 7])
    def test_sample_marginal(self, kind: str, sparse: bool, size: int):
        learner = fit(make_learner(kind, sparse), 5, sparse)
        X = make_X(4, 5, sparse)
        assert_draw_contiguous(learner.sample_marginal(X, size=size), size, 4)

    @pytest.mark.parametrize("kind", LEARNERS)
    @pytest.mark.parametrize("sparse", [False, True])
    @pytest.mark.parametrize("size", [1, 7])
    @pytest.mark.parametrize("block_size", [None, 2])
    def test_sample_reward_space(
        self, kind: str, sparse: bool, size: int, block_size: Any
    ):
        learner = fit(make_learner(kind, sparse), 5, sparse)
        X = make_X(4, 5, sparse)
        assert_draw_contiguous(
            learner.sample_reward_space(X, size, block_size=block_size), size, 4
        )

    @pytest.mark.parametrize("kind", ["normal", "nig", "glm_logit"])
    def test_sample_through_the_row_reduction(self, kind: str):
        # Two rows and many draws routes sample() through the row-side
        # support reduction; the contract must hold through that path too.
        learner = fit(make_learner(kind, True), 40, True)
        X = make_X(2, 40, True)
        samples = learner.sample(X, size=50)
        assert_draw_contiguous(samples, 50, 2)


class RecordingPolicy(ThompsonSampling):
    """Thompson sampling that records the stride of what select receives."""

    def __init__(self, samples_needed_override: int):
        super().__init__()
        self._samples_needed = samples_needed_override
        self.seen: List[Any] = []

    @property
    def samples_needed(self) -> int:
        return self._samples_needed

    def select(self, samples, arms, rng, top_k=None):  # type: ignore[override]
        self.seen.append((samples.shape, samples.strides[-1] == samples.itemsize))
        return [arms[0] for _ in range(samples.shape[1])]


class TestPolicyBoundary:
    """Agents hand select a tensor whose draw axis is unit-stride."""

    def test_shared_learner_agent(self):
        policy = RecordingPolicy(samples_needed_override=9)
        agent = LipschitzContextualAgent(
            arms=[Arm(i, reward_function=None, learner=None) for i in range(5)],
            policy=policy,
            arm_featurizer=ArmColumnFeaturizer(column_name="pid"),
            learner=NormalRegressor(alpha=1.0, beta=1.0),
            random_seed=0,
        )
        X = np.random.default_rng(0).normal(size=(3, 2))
        agent.pull(X)
        agent.update(X, np.zeros(3))
        agent.pull(X)
        assert policy.seen and all(ok for _, ok in policy.seen)
        assert all(shape == (5, 3, 9) for shape, _ in policy.seen)

    def test_per_arm_agent(self):
        policy = RecordingPolicy(samples_needed_override=9)
        agent = ContextualAgent(
            arms=[
                Arm(i, learner=NormalInverseGammaRegressor(random_state=i))
                for i in range(4)
            ],
            policy=policy,
            random_seed=0,
        )
        X = np.random.default_rng(0).normal(size=(3, 2))
        agent.pull(X)
        assert policy.seen and all(ok for _, ok in policy.seen)
        assert all(shape == (4, 3, 9) for shape, _ in policy.seen)

    def test_nonconforming_learner_is_normalized(self):
        class COrderNormal(NormalRegressor):
            """Violates the learner contract: C-ordered (size, n)."""

            def sample(self, X, size=1):
                return np.ascontiguousarray(super().sample(X, size))

        policy = RecordingPolicy(samples_needed_override=9)
        agent = LipschitzContextualAgent(
            arms=[Arm(i, reward_function=None, learner=None) for i in range(5)],
            policy=policy,
            arm_featurizer=ArmColumnFeaturizer(column_name="pid"),
            learner=COrderNormal(alpha=1.0, beta=1.0),
            random_seed=0,
        )
        X = np.random.default_rng(0).normal(size=(3, 2))
        agent.pull(X)
        assert policy.seen and all(ok for _, ok in policy.seen)


class TestValuesUnchangedByOrientation:
    """The flipped projections are the same inner products."""

    @pytest.mark.parametrize("kind", ["normal", "nig"])
    def test_weight_space_projection_matches_manual(self, kind: str):
        # Drawing weights with one seed and projecting must equal the
        # C-ordered product of the same weights with the same X. Dense
        # only: these shapes route the sparse models through the support
        # reduction, whose stream this replay would not match.
        sparse = False
        learner = fit(make_learner(kind, sparse), 5, sparse)
        X = make_X(4, 5, sparse)
        seen = learner.sample(X, size=7)

        twin = fit(make_learner(kind, sparse), 5, sparse)
        # consume the same stream, then project the C-ordered way
        from bayesianbandits._estimators import (
            multivariate_normal_sample_from_precision,
            multivariate_t_sample_from_precision,
        )

        if kind == "normal":
            w = np.atleast_2d(
                multivariate_normal_sample_from_precision(
                    twin.coef_,
                    twin._precision_factor,
                    size=7,
                    random_state=twin.random_state_,
                )
            )
        else:
            w = np.atleast_2d(
                multivariate_t_sample_from_precision(
                    twin.coef_, twin.shape_, 2.0 * twin.a_, 7, twin.random_state_
                )
            )
        np.testing.assert_allclose(seen, w @ X.T, rtol=1e-12, atol=1e-14)
