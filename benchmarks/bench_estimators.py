"""Benchmarks for estimator operations (sample, predict, partial_fit)."""

import numpy as np
import pytest

# -- normal dense 100 ---------------------------------------------------------


def test_sample_1_normal_dense_100(benchmark, normal_dense_100):
    est, X, _ = normal_dense_100
    benchmark(est.sample, X, size=1)


def test_sample_10_normal_dense_100(benchmark, normal_dense_100):
    est, X, _ = normal_dense_100
    benchmark(est.sample, X, size=10)


def test_predict_normal_dense_100(benchmark, normal_dense_100):
    est, X, _ = normal_dense_100
    benchmark(est.predict, X)


def test_partial_fit_normal_dense_100(benchmark, normal_dense_100_fresh):
    def run():
        est, X, _ = normal_dense_100_fresh()
        y = np.zeros(X.shape[0])
        est.partial_fit(X, y)

    benchmark(run)


# -- normal dense 1k ----------------------------------------------------------


def test_sample_1_normal_dense_1k(benchmark, normal_dense_1k):
    est, X, _ = normal_dense_1k
    benchmark(est.sample, X, size=1)


def test_sample_10_normal_dense_1k(benchmark, normal_dense_1k):
    est, X, _ = normal_dense_1k
    benchmark(est.sample, X, size=10)


def test_predict_normal_dense_1k(benchmark, normal_dense_1k):
    est, X, _ = normal_dense_1k
    benchmark(est.predict, X)


def test_partial_fit_normal_dense_1k(benchmark, normal_dense_1k_fresh):
    def run():
        est, X, _ = normal_dense_1k_fresh()
        y = np.zeros(X.shape[0])
        est.partial_fit(X, y)

    benchmark(run)


# -- normal sparse 1k ---------------------------------------------------------


def test_sample_1_normal_sparse_1k(benchmark, normal_sparse_1k):
    est, X, _ = normal_sparse_1k
    benchmark(est.sample, X, size=1)


def test_sample_10_normal_sparse_1k(benchmark, normal_sparse_1k):
    est, X, _ = normal_sparse_1k
    benchmark(est.sample, X, size=10)


def test_predict_normal_sparse_1k(benchmark, normal_sparse_1k):
    est, X, _ = normal_sparse_1k
    benchmark(est.predict, X)


def test_partial_fit_normal_sparse_1k(benchmark, normal_sparse_1k_fresh):
    def run():
        est, X, _ = normal_sparse_1k_fresh()
        y = np.zeros(X.shape[0])
        est.partial_fit(X, y)

    benchmark(run)


# -- normal sparse 1m ---------------------------------------------------------


@pytest.mark.slow
def test_sample_1_normal_sparse_1m(benchmark, normal_sparse_1m):
    est, X, _ = normal_sparse_1m
    benchmark(est.sample, X, size=1)


@pytest.mark.slow
def test_sample_10_normal_sparse_1m(benchmark, normal_sparse_1m):
    est, X, _ = normal_sparse_1m
    benchmark(est.sample, X, size=10)


@pytest.mark.slow
def test_predict_normal_sparse_1m(benchmark, normal_sparse_1m):
    est, X, _ = normal_sparse_1m
    benchmark(est.predict, X)


@pytest.mark.slow
def test_partial_fit_normal_sparse_1m(benchmark, normal_sparse_1m_fresh):
    def run():
        est, X, _ = normal_sparse_1m_fresh()
        y = np.zeros(X.shape[0])
        est.partial_fit(X, y)

    benchmark(run)


# -- glm_logit dense 100 ------------------------------------------------------


def test_sample_1_glm_logit_dense_100(benchmark, glm_logit_dense_100):
    est, X, _ = glm_logit_dense_100
    benchmark(est.sample, X, size=1)


def test_sample_10_glm_logit_dense_100(benchmark, glm_logit_dense_100):
    est, X, _ = glm_logit_dense_100
    benchmark(est.sample, X, size=10)


# -- glm_logit dense 1k -------------------------------------------------------


def test_sample_1_glm_logit_dense_1k(benchmark, glm_logit_dense_1k):
    est, X, _ = glm_logit_dense_1k
    benchmark(est.sample, X, size=1)


def test_sample_10_glm_logit_dense_1k(benchmark, glm_logit_dense_1k):
    est, X, _ = glm_logit_dense_1k
    benchmark(est.sample, X, size=10)


# -- glm_logit sparse 1k ------------------------------------------------------


def test_sample_1_glm_logit_sparse_1k(benchmark, glm_logit_sparse_1k):
    est, X, _ = glm_logit_sparse_1k
    benchmark(est.sample, X, size=1)


def test_sample_10_glm_logit_sparse_1k(benchmark, glm_logit_sparse_1k):
    est, X, _ = glm_logit_sparse_1k
    benchmark(est.sample, X, size=10)


# -- glm_logit sparse 1m ------------------------------------------------------


@pytest.mark.slow
def test_sample_1_glm_logit_sparse_1m(benchmark, glm_logit_sparse_1m):
    est, X, _ = glm_logit_sparse_1m
    benchmark(est.sample, X, size=1)


@pytest.mark.slow
def test_sample_10_glm_logit_sparse_1m(benchmark, glm_logit_sparse_1m):
    est, X, _ = glm_logit_sparse_1m
    benchmark(est.sample, X, size=10)
