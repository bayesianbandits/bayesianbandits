"""Benchmarks for iid marginal posterior predictive sampling.

``sample_marginal`` draws iid per-row marginal samples via one triangular
half-solve per row against the cached precision factor, so its per-draw
cost is independent of the feature count. It beats joint ``sample``
decisively whenever many draws are taken per row and the feature count
is nontrivial (the UCB/EXP3A regime, ``size=1000``): measured ~6x dense
at d=100, n=10 and ~175x sparse at d=100k. The ``rows_320`` pair shows
the honest flip side: at n >> d the marginal path generates n*size
normals versus weight-space's d*size, so joint ``sample`` can win
modestly for very low-dimensional models with many rows.

The existing ``test_sample_1_*`` benchmarks in ``test_bench_estimators.py``
guard the Thompson-sampling ``sample(size=1)`` hot path, which this
feature leaves untouched.
"""

import numpy as np

# -- normal dense 100, size=1000, n=10 ----------------------------------------


def test_sample_1000_normal_dense_100_joint(benchmark, normal_dense_100):
    est, X, _ = normal_dense_100
    benchmark(est.sample, X, size=1000)


def test_sample_1000_normal_dense_100_marginal(benchmark, normal_dense_100):
    est, X, _ = normal_dense_100
    benchmark(est.sample_marginal, X, size=1000)


# -- normal dense 100, n=320 rows (marginal path has no row cap) --------------


def test_sample_1000_rows_320_normal_dense_100_joint(benchmark, normal_dense_100):
    est, _, _ = normal_dense_100
    X = np.random.default_rng(7).standard_normal((320, 100))
    benchmark(est.sample, X, size=1000)


def test_sample_1000_rows_320_normal_dense_100_marginal(benchmark, normal_dense_100):
    est, _, _ = normal_dense_100
    X = np.random.default_rng(7).standard_normal((320, 100))
    benchmark(est.sample_marginal, X, size=1000)


# -- normal sparse 100k, size=1000, n=10 --------------------------------------


def test_sample_1000_normal_sparse_100k_joint(benchmark, normal_sparse_100k):
    est, X, _ = normal_sparse_100k
    benchmark(est.sample, X, size=1000)


def test_sample_1000_normal_sparse_100k_marginal(benchmark, normal_sparse_100k):
    est, X, _ = normal_sparse_100k
    benchmark(est.sample_marginal, X, size=1000)


# -- nig dense 100, size=1000, n=10 -------------------------------------------


def test_sample_1000_nig_dense_100_joint(benchmark, nig_dense_100):
    est, X, _ = nig_dense_100
    benchmark(est.sample, X, size=1000)


def test_sample_1000_nig_dense_100_marginal(benchmark, nig_dense_100):
    est, X, _ = nig_dense_100
    benchmark(est.sample_marginal, X, size=1000)


# -- glm_logit dense 100, size=1000, n=10 -------------------------------------


def test_sample_1000_glm_logit_dense_100_joint(benchmark, glm_logit_dense_100):
    est, X, _ = glm_logit_dense_100
    benchmark(est.sample, X, size=1000)


def test_sample_1000_glm_logit_dense_100_marginal(benchmark, glm_logit_dense_100):
    est, X, _ = glm_logit_dense_100
    benchmark(est.sample_marginal, X, size=1000)
