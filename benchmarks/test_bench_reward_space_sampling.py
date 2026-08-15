"""Benchmarks for ``sample_reward_space``: joint predictive draws in reward space.

Regimes:

- Many draws, few rows (``size=1000``, ``n=10``): reward-space should win
  decisively (expected ~5-6x dense at d=100, ~100x+ sparse at d=100k).
- Many rows (``n=320 > d=100``): weight-space wins even against blocked
  mode -- generating ``n*size`` normals costs more than ``d*size`` --
  and the resolver's cost model (calibrated against these numbers) must
  keep such calls on ``sample``. Blocked mode's role is bounding the QR
  cost so many-context batches stay in the win region whenever
  ``n_arms * n_contexts`` draws remain RNG-cheaper than ``d``-dim
  weight draws (always at sparse d=100k).
- ``size=1`` stays on the weight path via the resolver flop gate; the
  existing ``test_sample_1_*`` benchmarks in ``test_bench_estimators.py``
  guard that regime.

``sample`` here is the weight-space baseline (it never dispatches);
``sample_reward_space`` is the explicit fast path the IDS policy routes
through.
"""

import numpy as np

# -- normal dense 100, size=1000, n=10 (reward-space regime) ------------------


def test_sample_1000_normal_dense_100_weight(benchmark, normal_dense_100):
    est, X, _ = normal_dense_100
    benchmark(est.sample, X, size=1000)


def test_sample_1000_normal_dense_100_reward(benchmark, normal_dense_100):
    est, X, _ = normal_dense_100
    benchmark(est.sample_reward_space, X, 1000)


def test_sample_1000_normal_dense_100_reward_blocked(benchmark, normal_dense_100):
    est, X, _ = normal_dense_100
    benchmark(est.sample_reward_space, X, 1000, block_size=5)


# -- normal dense 100, n=320 (blocked mode rescues the many-rows regime) ------


def test_sample_1000_rows_320_normal_dense_100_weight(benchmark, normal_dense_100):
    est, _, _ = normal_dense_100
    X = np.random.default_rng(7).standard_normal((320, 100))
    benchmark(est.sample, X, size=1000)


def test_sample_1000_rows_320_normal_dense_100_reward_full(benchmark, normal_dense_100):
    est, _, _ = normal_dense_100
    X = np.random.default_rng(7).standard_normal((320, 100))
    benchmark(est.sample_reward_space, X, 1000)


def test_sample_1000_rows_320_normal_dense_100_reward_blocked(
    benchmark, normal_dense_100
):
    est, _, _ = normal_dense_100
    X = np.random.default_rng(7).standard_normal((320, 100))
    benchmark(est.sample_reward_space, X, 1000, block_size=10)


# -- normal sparse 100k, size=1000, n=10 (reward-space regime) ----------------


def test_sample_1000_normal_sparse_100k_weight(benchmark, normal_sparse_100k):
    est, X, _ = normal_sparse_100k
    benchmark(est.sample, X, size=1000)


def test_sample_1000_normal_sparse_100k_reward(benchmark, normal_sparse_100k):
    est, X, _ = normal_sparse_100k
    benchmark(est.sample_reward_space, X, 1000)


def test_sample_1000_normal_sparse_100k_reward_blocked(benchmark, normal_sparse_100k):
    est, X, _ = normal_sparse_100k
    benchmark(est.sample_reward_space, X, 1000, block_size=5)


# -- nig dense 100, size=1000, n=10 -------------------------------------------


def test_sample_1000_nig_dense_100_weight(benchmark, nig_dense_100):
    est, X, _ = nig_dense_100
    benchmark(est.sample, X, size=1000)


def test_sample_1000_nig_dense_100_reward(benchmark, nig_dense_100):
    est, X, _ = nig_dense_100
    benchmark(est.sample_reward_space, X, 1000)


def test_sample_1000_nig_dense_100_reward_blocked(benchmark, nig_dense_100):
    est, X, _ = nig_dense_100
    benchmark(est.sample_reward_space, X, 1000, block_size=5)


# -- glm_logit dense 100, size=1000, n=10 -------------------------------------


def test_sample_1000_glm_logit_dense_100_weight(benchmark, glm_logit_dense_100):
    est, X, _ = glm_logit_dense_100
    benchmark(est.sample, X, size=1000)


def test_sample_1000_glm_logit_dense_100_reward(benchmark, glm_logit_dense_100):
    est, X, _ = glm_logit_dense_100
    benchmark(est.sample_reward_space, X, 1000)


# -- end-to-end: IDS pull through LipschitzContextualAgent --------------------


def _ids_lipschitz_agent(learner, n_arms=10, d_ctx=99, n_train=300):
    from bayesianbandits import (
        Arm,
        ArmColumnFeaturizer,
        InformationDirectedSampling,
        LipschitzContextualAgent,
    )

    rng = np.random.default_rng(0)
    arms = [Arm(i, reward_function=None, learner=None) for i in range(n_arms)]
    agent = LipschitzContextualAgent(
        arms=arms,
        policy=InformationDirectedSampling(samples=1000),
        arm_featurizer=ArmColumnFeaturizer(column_name="product_id"),
        learner=learner,
        random_seed=0,
    )
    X_train = rng.standard_normal((n_train, d_ctx))
    agent.pull(X_train[:1])
    agent.update(X_train, rng.standard_normal(n_train))
    return agent, rng.standard_normal((8, d_ctx))


def test_ids_pull_lipschitz_dense_100(benchmark):
    from bayesianbandits import NormalRegressor

    agent, X = _ids_lipschitz_agent(NormalRegressor(alpha=1.0, beta=1.0))
    benchmark(agent.pull, X)


def test_ids_pull_lipschitz_dense_100_weight_space(benchmark):
    """Forced weight-space baseline for the pull above."""
    from bayesianbandits import NormalRegressor

    agent, X = _ids_lipschitz_agent(NormalRegressor(alpha=1.0, beta=1.0))
    agent.learner._use_reward_space = lambda *a, **k: False
    benchmark(agent.pull, X)
