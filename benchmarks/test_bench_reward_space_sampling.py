"""Benchmarks for joint predictive draws: weight space against reward space.

``sample`` reduces through the cheapest exact route on its own, so the
``_weight`` cases below patch the reduction out to keep a true weight-space
baseline. ``sample_reward_space`` is the explicit row-side route, in full
and blocked mode. The ``rows_320`` cases (``n_rows > d``) sit in the
region where weight space should win and the gate must keep ``sample``
there; ``size=1`` is covered by ``test_bench_estimators.py``.
"""

from contextlib import contextmanager
from unittest import mock

import numpy as np

from bayesianbandits import _estimators


@contextmanager
def _weight_space():
    with mock.patch.object(_estimators, "build_joint_reduction", return_value=None):
        yield


# -- normal dense 100, size=1000, n=10 (reward-space regime) ------------------


def test_sample_1000_normal_dense_100_weight(benchmark, normal_dense_100):
    est, X, _ = normal_dense_100
    with _weight_space():
        benchmark(est.sample, X, size=1000)


def test_sample_1000_normal_dense_100_reward(benchmark, normal_dense_100):
    est, X, _ = normal_dense_100
    benchmark(est.sample_reward_space, X, 1000)


def test_sample_1000_normal_dense_100_reward_blocked(benchmark, normal_dense_100):
    est, X, _ = normal_dense_100
    benchmark(est.sample_reward_space, X, 1000, block_size=5)


# -- normal dense 100, n=320 > d (weight space should win) --------------------


def test_sample_1000_rows_320_normal_dense_100_weight(benchmark, normal_dense_100):
    est, _, _ = normal_dense_100
    X = np.random.default_rng(7).standard_normal((320, 100))
    with _weight_space():
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
    with _weight_space():
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
    with _weight_space():
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
    with _weight_space():
        benchmark(est.sample, X, size=1000)


def test_sample_1000_glm_logit_dense_100_reward(benchmark, glm_logit_dense_100):
    est, X, _ = glm_logit_dense_100
    benchmark(est.sample_reward_space, X, 1000)
