"""Agent wiring for ``CONTEXT_JOINT`` draws.

``LipschitzContextualAgent`` stacks arm features arm-major, but a
per-context joint block needs each context's arm rows adjacent. These
check that permutation and its inverse, the block size the agent asks
for, and the pipeline's forwarding and fallback.
"""

import numpy as np
import pytest
from numpy.testing import assert_allclose

from bayesianbandits import (
    Arm,
    ArmColumnFeaturizer,
    InformationDirectedSampling,
    LipschitzContextualAgent,
    NormalRegressor,
)
from bayesianbandits._arm import (
    _sample_context_major_blocks,
    _take_rows,
    resolve_reward_space_sampler,
)
from bayesianbandits.pipelines import LearnerPipeline


class TestAgentWiring:
    @pytest.mark.parametrize(
        ("n_arms", "n_ctx"),
        [(3, 2), (3, 1), (1, 3)],
        ids=["general", "one-ctx", "one-arm"],
    )
    def test_context_major_axes_are_restored(self, n_arms, n_ctx):
        """The sampler is handed context-major rows; the result must come
        back indexed ``(arm, context, draw)``. Exact, by construction.
        The single-arm and single-context cases skip the gather, so they
        take a different branch and are the common shapes in production."""
        size = 4
        # arm-major row ``a * n_ctx + c`` carries the payload ``a * 10 + c``
        X = np.array(
            [[a * 10 + c] for a in range(n_arms) for c in range(n_ctx)], dtype=float
        )

        def sampler(X_in, size):
            return np.tile(X_in[:, 0], (size, 1))

        out = _sample_context_major_blocks(sampler, X, n_arms, n_ctx, size)
        assert out.shape == (n_arms, n_ctx, size)
        for a in range(n_arms):
            for c in range(n_ctx):
                assert_allclose(out[a, c], a * 10 + c)

    def test_take_rows_uses_iloc_for_dataframes(self):
        """Plain ``X[indices]`` selects columns by label on a DataFrame,
        so positional row selection must go through ``iloc``."""
        pd = pytest.importorskip("pandas")
        df = pd.DataFrame(np.arange(9.0).reshape(3, 3), columns=["a", "b", "c"])
        assert _take_rows(df, np.array([2, 0, 1])).values.tolist() == [
            [6, 7, 8],
            [0, 1, 2],
            [3, 4, 5],
        ]

    def _make_lipschitz(self, learner, n_arms=4, seed=0):
        arms = [Arm(i, reward_function=None, learner=None) for i in range(n_arms)]
        return LipschitzContextualAgent(
            arms=arms,
            policy=InformationDirectedSampling(samples=500),
            arm_featurizer=ArmColumnFeaturizer(column_name="product_id"),
            learner=learner,
            random_seed=seed,
        )

    def test_lipschitz_agent_blocks_by_arm(self, monkeypatch):
        """One block per arm is what makes the draws context-joint."""
        rng = np.random.default_rng(0)
        d = 100
        agent = self._make_lipschitz(NormalRegressor(alpha=1.0, beta=1.0))
        X_train = rng.standard_normal((300, d))
        agent.pull(X_train[:1])
        agent.update(X_train, rng.standard_normal(300))

        calls = {"n": 0}
        original = type(agent.learner).sample_reward_space  # type: ignore[attr-defined]

        def counting(self, X, size=1, *, block_size=None):
            calls["n"] += 1
            assert block_size == len(agent.arms)
            return original(self, X, size, block_size=block_size)

        monkeypatch.setattr(type(agent.learner), "sample_reward_space", counting)
        assert len(agent.pull(rng.standard_normal((3, d)))) == 3
        assert calls["n"] == 1

    def test_pipeline_forwards_reward_space(self):
        rng = np.random.default_rng(0)
        model = NormalRegressor(alpha=1.0, beta=1.0, random_state=0)
        model.fit(rng.standard_normal((300, 101)), rng.standard_normal(300))
        pipeline = LearnerPipeline(steps=[], learner=model)
        X = rng.standard_normal((4, 101))
        model.random_state_ = np.random.default_rng(3)
        via_pipeline = pipeline.sample_reward_space(X, 1000, block_size=2)
        model.random_state_ = np.random.default_rng(3)
        assert_allclose(via_pipeline, model.sample_reward_space(X, 1000, block_size=2))

    def test_pipeline_falls_back_to_joint_sample(self):
        """A learner without ``sample_reward_space`` still gets draws, and
        they are fully joint, which is a superset of what was asked."""

        class PlainLearner:
            def sample(self, X, size=1):
                return np.zeros((size, X.shape[0]))

            def predict(self, X):
                return np.full(X.shape[0], 3.0)

            def partial_fit(self, X, y, sample_weight=None):
                return self

            def decay(self, X, *, decay_rate=None):
                self.decayed = True

        learner = PlainLearner()
        pipeline = LearnerPipeline(steps=[], learner=learner)  # type: ignore[arg-type]
        X = np.zeros((4, 3))
        assert pipeline.sample_reward_space(X, 7, block_size=2).shape == (7, 4)
        assert not pipeline._use_reward_space(4, 7, 2)

    def test_resolver_never_bypasses_a_custom_sample(self):
        """A subclass overriding ``sample`` without ``sample_reward_space``
        must keep its own sampling behavior."""

        class ClippedRegressor(NormalRegressor):
            def sample(self, X, size=1):
                return np.clip(super().sample(X, size), 0.0, None)

        rng = np.random.default_rng(0)
        est = ClippedRegressor(alpha=1.0, beta=1.0)
        est.fit(rng.standard_normal((300, 101)), rng.standard_normal(300))
        assert resolve_reward_space_sampler(est, 10, 1000) is None
        assert (est.sample(rng.standard_normal((10, 101)), size=50) >= 0.0).all()

    def test_resolver_skips_unfitted_models(self):
        est = NormalRegressor(alpha=1.0, beta=1.0)
        assert resolve_reward_space_sampler(est, 10, 1000) is None
