"""Benchmarks for estimator operations (fit, sample, predict, partial_fit, decay)."""

import numpy as np
import pytest
from scipy.sparse import csc_array
from scipy.sparse import random as sparse_random

from bayesianbandits._eb_estimators import (
    EmpiricalBayesGLM,
    EmpiricalBayesNormalRegressor,
)
from bayesianbandits._estimators import (
    BayesianGLM,
    NormalInverseGammaRegressor,
    NormalRegressor,
)
from bayesianbandits._gaussian import RVGAApproximator

# -- normal dense 100 ---------------------------------------------------------


def test_fit_normal_dense_100(benchmark):
    rng = np.random.default_rng(42)
    X = rng.standard_normal((200, 100))
    y = rng.standard_normal(200)

    def run():
        est = NormalRegressor(alpha=1.0, beta=1.0, sparse=False)
        est.fit(X, y)

    benchmark(run)


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


def test_decay_normal_dense_100(benchmark, normal_dense_100_fresh):
    def run():
        est, X, _ = normal_dense_100_fresh()
        est.decay(X, decay_rate=0.99)

    benchmark(run)


# -- normal dense 1k ----------------------------------------------------------


def test_fit_normal_dense_1k(benchmark):
    rng = np.random.default_rng(42)
    X = rng.standard_normal((200, 1000))
    y = rng.standard_normal(200)

    def run():
        est = NormalRegressor(alpha=1.0, beta=1.0, sparse=False)
        est.fit(X, y)

    benchmark(run)


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


def test_decay_normal_dense_1k(benchmark, normal_dense_1k_fresh):
    def run():
        est, X, _ = normal_dense_1k_fresh()
        est.decay(X, decay_rate=0.99)

    benchmark(run)


# -- normal sparse 1k ---------------------------------------------------------


def test_fit_normal_sparse_1k(benchmark):
    rng = np.random.default_rng(42)
    X = csc_array(sparse_random(200, 1000, density=0.01, random_state=42))
    y = rng.standard_normal(200)

    def run():
        est = NormalRegressor(alpha=1.0, beta=1.0, sparse=True)
        est.fit(X, y)

    benchmark(run)


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


def test_decay_normal_sparse_1k(benchmark, normal_sparse_1k_fresh):
    def run():
        est, X, _ = normal_sparse_1k_fresh()
        est.decay(X, decay_rate=0.99)

    benchmark(run)


# -- normal sparse 100k -------------------------------------------------------


def test_fit_normal_sparse_100k(benchmark):
    X = csc_array(sparse_random(200, 100_000, density=0.0001, random_state=42))
    rng = np.random.default_rng(42)
    y = rng.standard_normal(200)

    def run():
        est = NormalRegressor(alpha=1.0, beta=1.0, sparse=True)
        est.fit(X, y)

    benchmark(run)


def test_sample_1_normal_sparse_100k(benchmark, normal_sparse_100k):
    est, X, _ = normal_sparse_100k
    benchmark(est.sample, X, size=1)


def test_sample_10_normal_sparse_100k(benchmark, normal_sparse_100k):
    est, X, _ = normal_sparse_100k
    benchmark(est.sample, X, size=10)


def test_predict_normal_sparse_100k(benchmark, normal_sparse_100k):
    est, X, _ = normal_sparse_100k
    benchmark(est.predict, X)


def test_partial_fit_normal_sparse_100k(benchmark, normal_sparse_100k_fresh):
    def run():
        est, X, _ = normal_sparse_100k_fresh()
        y = np.zeros(X.shape[0])
        est.partial_fit(X, y)

    benchmark(run)


def test_decay_normal_sparse_100k(benchmark, normal_sparse_100k_fresh):
    def run():
        est, X, _ = normal_sparse_100k_fresh()
        est.decay(X, decay_rate=0.99)

    benchmark(run)


# -- normal sparse 1m ---------------------------------------------------------


@pytest.mark.slow
def test_fit_normal_sparse_1m(benchmark):
    X = csc_array(sparse_random(200, 1_000_000, density=0.00001, random_state=42))
    rng = np.random.default_rng(42)
    y = rng.standard_normal(200)

    def run():
        est = NormalRegressor(alpha=1.0, beta=1.0, sparse=True)
        est.fit(X, y)

    benchmark(run)


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


@pytest.mark.slow
def test_decay_normal_sparse_1m(benchmark, normal_sparse_1m_fresh):
    def run():
        est, X, _ = normal_sparse_1m_fresh()
        est.decay(X, decay_rate=0.99)

    benchmark(run)


# -- glm_logit dense 100 ------------------------------------------------------


def test_fit_glm_logit_dense_100(benchmark):
    rng = np.random.default_rng(42)
    X = rng.standard_normal((200, 100))
    y = rng.integers(0, 2, size=200).astype(np.float64)

    def run():
        est = BayesianGLM(alpha=1.0, link="logit", sparse=False)
        est.fit(X, y)

    benchmark(run)


def test_sample_1_glm_logit_dense_100(benchmark, glm_logit_dense_100):
    est, X, _ = glm_logit_dense_100
    benchmark(est.sample, X, size=1)


def test_sample_10_glm_logit_dense_100(benchmark, glm_logit_dense_100):
    est, X, _ = glm_logit_dense_100
    benchmark(est.sample, X, size=10)


def test_predict_glm_logit_dense_100(benchmark, glm_logit_dense_100):
    est, X, _ = glm_logit_dense_100
    benchmark(est.predict, X)


def test_partial_fit_glm_logit_dense_100(benchmark, glm_logit_dense_100_fresh):
    def run():
        est, X, _ = glm_logit_dense_100_fresh()
        y = np.zeros(X.shape[0])
        est.partial_fit(X, y)

    benchmark(run)


def test_decay_glm_logit_dense_100(benchmark, glm_logit_dense_100_fresh):
    def run():
        est, X, _ = glm_logit_dense_100_fresh()
        est.decay(X, decay_rate=0.99)

    benchmark(run)


# -- glm_logit dense 1k -------------------------------------------------------


def test_fit_glm_logit_dense_1k(benchmark):
    rng = np.random.default_rng(42)
    X = rng.standard_normal((200, 1000))
    y = rng.integers(0, 2, size=200).astype(np.float64)

    def run():
        est = BayesianGLM(alpha=1.0, link="logit", sparse=False)
        est.fit(X, y)

    benchmark(run)


def test_sample_1_glm_logit_dense_1k(benchmark, glm_logit_dense_1k):
    est, X, _ = glm_logit_dense_1k
    benchmark(est.sample, X, size=1)


def test_sample_10_glm_logit_dense_1k(benchmark, glm_logit_dense_1k):
    est, X, _ = glm_logit_dense_1k
    benchmark(est.sample, X, size=10)


def test_predict_glm_logit_dense_1k(benchmark, glm_logit_dense_1k):
    est, X, _ = glm_logit_dense_1k
    benchmark(est.predict, X)


def test_partial_fit_glm_logit_dense_1k(benchmark, glm_logit_dense_1k_fresh):
    def run():
        est, X, _ = glm_logit_dense_1k_fresh()
        y = np.zeros(X.shape[0])
        est.partial_fit(X, y)

    benchmark(run)


def test_decay_glm_logit_dense_1k(benchmark, glm_logit_dense_1k_fresh):
    def run():
        est, X, _ = glm_logit_dense_1k_fresh()
        est.decay(X, decay_rate=0.99)

    benchmark(run)


# -- glm_logit sparse 1k ------------------------------------------------------


def test_fit_glm_logit_sparse_1k(benchmark):
    rng = np.random.default_rng(42)
    X = csc_array(sparse_random(200, 1000, density=0.01, random_state=42))
    y = rng.integers(0, 2, size=200).astype(np.float64)

    def run():
        est = BayesianGLM(alpha=1.0, link="logit", sparse=True)
        est.fit(X, y)

    benchmark(run)


def test_sample_1_glm_logit_sparse_1k(benchmark, glm_logit_sparse_1k):
    est, X, _ = glm_logit_sparse_1k
    benchmark(est.sample, X, size=1)


def test_sample_10_glm_logit_sparse_1k(benchmark, glm_logit_sparse_1k):
    est, X, _ = glm_logit_sparse_1k
    benchmark(est.sample, X, size=10)


def test_predict_glm_logit_sparse_1k(benchmark, glm_logit_sparse_1k):
    est, X, _ = glm_logit_sparse_1k
    benchmark(est.predict, X)


def test_partial_fit_glm_logit_sparse_1k(benchmark, glm_logit_sparse_1k_fresh):
    def run():
        est, X, _ = glm_logit_sparse_1k_fresh()
        y = np.zeros(X.shape[0])
        est.partial_fit(X, y)

    benchmark(run)


def test_decay_glm_logit_sparse_1k(benchmark, glm_logit_sparse_1k_fresh):
    def run():
        est, X, _ = glm_logit_sparse_1k_fresh()
        est.decay(X, decay_rate=0.99)

    benchmark(run)


# -- glm_logit sparse 100k ----------------------------------------------------


def test_fit_glm_logit_sparse_100k(benchmark):
    X = csc_array(sparse_random(200, 100_000, density=0.0001, random_state=42))
    rng = np.random.default_rng(42)
    y = rng.integers(0, 2, size=200).astype(np.float64)

    def run():
        est = BayesianGLM(alpha=1.0, link="logit", sparse=True)
        est.fit(X, y)

    benchmark(run)


def test_sample_1_glm_logit_sparse_100k(benchmark, glm_logit_sparse_100k):
    est, X, _ = glm_logit_sparse_100k
    benchmark(est.sample, X, size=1)


def test_predict_glm_logit_sparse_100k(benchmark, glm_logit_sparse_100k):
    est, X, _ = glm_logit_sparse_100k
    benchmark(est.predict, X)


def test_partial_fit_glm_logit_sparse_100k(benchmark, glm_logit_sparse_100k_fresh):
    def run():
        est, X, _ = glm_logit_sparse_100k_fresh()
        y = np.zeros(X.shape[0])
        est.partial_fit(X, y)

    benchmark(run)


def test_decay_glm_logit_sparse_100k(benchmark, glm_logit_sparse_100k_fresh):
    def run():
        est, X, _ = glm_logit_sparse_100k_fresh()
        est.decay(X, decay_rate=0.99)

    benchmark(run)


# -- glm_logit sparse 1m ------------------------------------------------------


@pytest.mark.slow
def test_fit_glm_logit_sparse_1m(benchmark):
    X = csc_array(sparse_random(200, 1_000_000, density=0.00001, random_state=42))
    rng = np.random.default_rng(42)
    y = rng.integers(0, 2, size=200).astype(np.float64)

    def run():
        est = BayesianGLM(alpha=1.0, link="logit", sparse=True)
        est.fit(X, y)

    benchmark(run)


@pytest.mark.slow
def test_sample_1_glm_logit_sparse_1m(benchmark, glm_logit_sparse_1m):
    est, X, _ = glm_logit_sparse_1m
    benchmark(est.sample, X, size=1)


@pytest.mark.slow
def test_sample_10_glm_logit_sparse_1m(benchmark, glm_logit_sparse_1m):
    est, X, _ = glm_logit_sparse_1m
    benchmark(est.sample, X, size=10)


@pytest.mark.slow
def test_predict_glm_logit_sparse_1m(benchmark, glm_logit_sparse_1m):
    est, X, _ = glm_logit_sparse_1m
    benchmark(est.predict, X)


@pytest.mark.slow
def test_partial_fit_glm_logit_sparse_1m(benchmark, glm_logit_sparse_1m_fresh):
    def run():
        est, X, _ = glm_logit_sparse_1m_fresh()
        y = np.zeros(X.shape[0])
        est.partial_fit(X, y)

    benchmark(run)


@pytest.mark.slow
def test_decay_glm_logit_sparse_1m(benchmark, glm_logit_sparse_1m_fresh):
    def run():
        est, X, _ = glm_logit_sparse_1m_fresh()
        est.decay(X, decay_rate=0.99)

    benchmark(run)


# -- eb_normal dense 100 ------------------------------------------------------


def test_fit_eb_normal_dense_100(benchmark):
    rng = np.random.default_rng(42)
    X = rng.standard_normal((200, 100))
    y = rng.standard_normal(200)

    def run():
        est = EmpiricalBayesNormalRegressor(
            alpha=1.0, beta=1.0, learning_rate=0.99999, sparse=False
        )
        est.fit(X, y)

    benchmark(run)


def test_sample_1_eb_normal_dense_100(benchmark, eb_normal_dense_100):
    est, X, _ = eb_normal_dense_100
    benchmark(est.sample, X, size=1)


def test_sample_10_eb_normal_dense_100(benchmark, eb_normal_dense_100):
    est, X, _ = eb_normal_dense_100
    benchmark(est.sample, X, size=10)


def test_predict_eb_normal_dense_100(benchmark, eb_normal_dense_100):
    est, X, _ = eb_normal_dense_100
    benchmark(est.predict, X)


def test_partial_fit_eb_normal_dense_100(benchmark, eb_normal_dense_100_fresh):
    rng = np.random.default_rng(99)

    def run():
        est, X, _ = eb_normal_dense_100_fresh()
        y = rng.standard_normal(X.shape[0])
        est.partial_fit(X, y)

    benchmark(run)


def test_decay_eb_normal_dense_100(benchmark, eb_normal_dense_100_fresh):
    def run():
        est, X, _ = eb_normal_dense_100_fresh()
        est.decay(X, decay_rate=0.99)

    benchmark(run)


# -- eb_normal dense 1k -------------------------------------------------------


def test_fit_eb_normal_dense_1k(benchmark):
    rng = np.random.default_rng(42)
    X = rng.standard_normal((200, 1000))
    y = rng.standard_normal(200)

    def run():
        est = EmpiricalBayesNormalRegressor(
            alpha=1.0, beta=1.0, learning_rate=0.99999, sparse=False
        )
        est.fit(X, y)

    benchmark(run)


def test_sample_1_eb_normal_dense_1k(benchmark, eb_normal_dense_1k):
    est, X, _ = eb_normal_dense_1k
    benchmark(est.sample, X, size=1)


def test_sample_10_eb_normal_dense_1k(benchmark, eb_normal_dense_1k):
    est, X, _ = eb_normal_dense_1k
    benchmark(est.sample, X, size=10)


def test_predict_eb_normal_dense_1k(benchmark, eb_normal_dense_1k):
    est, X, _ = eb_normal_dense_1k
    benchmark(est.predict, X)


def test_partial_fit_eb_normal_dense_1k(benchmark, eb_normal_dense_1k_fresh):
    rng = np.random.default_rng(99)

    def run():
        est, X, _ = eb_normal_dense_1k_fresh()
        y = rng.standard_normal(X.shape[0])
        est.partial_fit(X, y)

    benchmark(run)


def test_decay_eb_normal_dense_1k(benchmark, eb_normal_dense_1k_fresh):
    def run():
        est, X, _ = eb_normal_dense_1k_fresh()
        est.decay(X, decay_rate=0.99)

    benchmark(run)


# -- eb_normal sparse 1k ------------------------------------------------------


def test_fit_eb_normal_sparse_1k(benchmark):
    rng = np.random.default_rng(42)
    X = csc_array(sparse_random(200, 1000, density=0.01, random_state=42))
    y = rng.standard_normal(200)

    def run():
        est = EmpiricalBayesNormalRegressor(
            alpha=1.0, beta=1.0, learning_rate=0.99999, sparse=True
        )
        est.fit(X, y)

    benchmark(run)


def test_sample_1_eb_normal_sparse_1k(benchmark, eb_normal_sparse_1k):
    est, X, _ = eb_normal_sparse_1k
    benchmark(est.sample, X, size=1)


def test_sample_10_eb_normal_sparse_1k(benchmark, eb_normal_sparse_1k):
    est, X, _ = eb_normal_sparse_1k
    benchmark(est.sample, X, size=10)


def test_predict_eb_normal_sparse_1k(benchmark, eb_normal_sparse_1k):
    est, X, _ = eb_normal_sparse_1k
    benchmark(est.predict, X)


def test_partial_fit_eb_normal_sparse_1k(benchmark, eb_normal_sparse_1k_fresh):
    rng = np.random.default_rng(99)

    def run():
        est, X, _ = eb_normal_sparse_1k_fresh()
        y = rng.standard_normal(X.shape[0])
        est.partial_fit(X, y)

    benchmark(run)


def test_decay_eb_normal_sparse_1k(benchmark, eb_normal_sparse_1k_fresh):
    def run():
        est, X, _ = eb_normal_sparse_1k_fresh()
        est.decay(X, decay_rate=0.99)

    benchmark(run)


# -- eb_normal sparse 100k ----------------------------------------------------


def test_fit_eb_normal_sparse_100k(benchmark):
    rng = np.random.default_rng(42)
    X = csc_array(sparse_random(200, 100_000, density=0.001, random_state=42))
    true_coef = np.zeros(100_000)
    true_coef[:50] = rng.standard_normal(50) * 2.0
    y = np.asarray(X @ true_coef).ravel() + 0.5 * rng.standard_normal(200)

    def run():
        est = EmpiricalBayesNormalRegressor(
            alpha=1.0, beta=1.0, learning_rate=0.99999, sparse=True
        )
        est.fit(X, y)

    benchmark(run)


def test_sample_1_eb_normal_sparse_100k(benchmark, eb_normal_sparse_100k):
    est, X, _ = eb_normal_sparse_100k
    benchmark(est.sample, X, size=1)


def test_sample_10_eb_normal_sparse_100k(benchmark, eb_normal_sparse_100k):
    est, X, _ = eb_normal_sparse_100k
    benchmark(est.sample, X, size=10)


def test_predict_eb_normal_sparse_100k(benchmark, eb_normal_sparse_100k):
    est, X, _ = eb_normal_sparse_100k
    benchmark(est.predict, X)


def test_partial_fit_eb_normal_sparse_100k(benchmark, eb_normal_sparse_100k_fresh):
    rng = np.random.default_rng(99)

    def run():
        est, X, _ = eb_normal_sparse_100k_fresh()
        y = rng.standard_normal(X.shape[0])
        est.partial_fit(X, y)

    benchmark(run)


def test_decay_eb_normal_sparse_100k(benchmark, eb_normal_sparse_100k_fresh):
    def run():
        est, X, _ = eb_normal_sparse_100k_fresh()
        est.decay(X, decay_rate=0.99)

    benchmark(run)


# -- eb_glm dense_100 -----------------------------------------------------


def test_fit_eb_glm_dense_100(benchmark):
    rng = np.random.default_rng(42)
    X = rng.standard_normal((200, 100))
    true_coef = np.zeros(100)
    true_coef[:50] = rng.standard_normal(50) * 2.0
    y = rng.binomial(1, 1.0 / (1.0 + np.exp(-(X @ true_coef)))).astype(np.float64)

    def run():
        est = EmpiricalBayesGLM(
            alpha=1.0, link="logit", learning_rate=0.99999, sparse=False
        )
        est.fit(X, y)

    benchmark(run)


def test_sample_1_eb_glm_dense_100(benchmark, eb_glm_dense_100):
    est, X, _ = eb_glm_dense_100
    benchmark(est.sample, X, size=1)


def test_sample_10_eb_glm_dense_100(benchmark, eb_glm_dense_100):
    est, X, _ = eb_glm_dense_100
    benchmark(est.sample, X, size=10)


def test_predict_eb_glm_dense_100(benchmark, eb_glm_dense_100):
    est, X, _ = eb_glm_dense_100
    benchmark(est.predict, X)


def test_partial_fit_eb_glm_dense_100(benchmark, eb_glm_dense_100_fresh):
    rng = np.random.default_rng(99)

    def run():
        est, X, _ = eb_glm_dense_100_fresh()
        y = rng.integers(0, 2, size=X.shape[0]).astype(np.float64)
        est.partial_fit(X, y)

    benchmark(run)


def test_decay_eb_glm_dense_100(benchmark, eb_glm_dense_100_fresh):
    def run():
        est, X, _ = eb_glm_dense_100_fresh()
        est.decay(X, decay_rate=0.99)

    benchmark(run)


# -- eb_glm dense_1k ------------------------------------------------------


def test_fit_eb_glm_dense_1k(benchmark):
    rng = np.random.default_rng(42)
    X = rng.standard_normal((200, 1000))
    true_coef = np.zeros(1000)
    true_coef[:50] = rng.standard_normal(50) * 2.0
    y = rng.binomial(1, 1.0 / (1.0 + np.exp(-(X @ true_coef)))).astype(np.float64)

    def run():
        est = EmpiricalBayesGLM(
            alpha=1.0, link="logit", learning_rate=0.99999, sparse=False
        )
        est.fit(X, y)

    benchmark(run)


def test_sample_1_eb_glm_dense_1k(benchmark, eb_glm_dense_1k):
    est, X, _ = eb_glm_dense_1k
    benchmark(est.sample, X, size=1)


def test_sample_10_eb_glm_dense_1k(benchmark, eb_glm_dense_1k):
    est, X, _ = eb_glm_dense_1k
    benchmark(est.sample, X, size=10)


def test_predict_eb_glm_dense_1k(benchmark, eb_glm_dense_1k):
    est, X, _ = eb_glm_dense_1k
    benchmark(est.predict, X)


def test_partial_fit_eb_glm_dense_1k(benchmark, eb_glm_dense_1k_fresh):
    rng = np.random.default_rng(99)

    def run():
        est, X, _ = eb_glm_dense_1k_fresh()
        y = rng.integers(0, 2, size=X.shape[0]).astype(np.float64)
        est.partial_fit(X, y)

    benchmark(run)


def test_decay_eb_glm_dense_1k(benchmark, eb_glm_dense_1k_fresh):
    def run():
        est, X, _ = eb_glm_dense_1k_fresh()
        est.decay(X, decay_rate=0.99)

    benchmark(run)


# -- eb_glm sparse_1k -----------------------------------------------------


def test_fit_eb_glm_sparse_1k(benchmark):
    rng = np.random.default_rng(42)
    X = csc_array(sparse_random(200, 1000, density=0.01, random_state=42))
    true_coef = np.zeros(1000)
    true_coef[:50] = rng.standard_normal(50) * 2.0
    eta = np.asarray(X @ true_coef).ravel()
    y = rng.binomial(1, 1.0 / (1.0 + np.exp(-eta))).astype(np.float64)

    def run():
        est = EmpiricalBayesGLM(
            alpha=1.0, link="logit", learning_rate=0.99999, sparse=True
        )
        est.fit(X, y)

    benchmark(run)


def test_sample_1_eb_glm_sparse_1k(benchmark, eb_glm_sparse_1k):
    est, X, _ = eb_glm_sparse_1k
    benchmark(est.sample, X, size=1)


def test_sample_10_eb_glm_sparse_1k(benchmark, eb_glm_sparse_1k):
    est, X, _ = eb_glm_sparse_1k
    benchmark(est.sample, X, size=10)


def test_predict_eb_glm_sparse_1k(benchmark, eb_glm_sparse_1k):
    est, X, _ = eb_glm_sparse_1k
    benchmark(est.predict, X)


def test_partial_fit_eb_glm_sparse_1k(benchmark, eb_glm_sparse_1k_fresh):
    rng = np.random.default_rng(99)

    def run():
        est, X, _ = eb_glm_sparse_1k_fresh()
        y = rng.integers(0, 2, size=X.shape[0]).astype(np.float64)
        est.partial_fit(X, y)

    benchmark(run)


def test_decay_eb_glm_sparse_1k(benchmark, eb_glm_sparse_1k_fresh):
    def run():
        est, X, _ = eb_glm_sparse_1k_fresh()
        est.decay(X, decay_rate=0.99)

    benchmark(run)


# -- eb_glm sparse_100k ---------------------------------------------------


def test_fit_eb_glm_sparse_100k(benchmark):
    rng = np.random.default_rng(42)
    X = csc_array(sparse_random(200, 100000, density=0.001, random_state=42))
    true_coef = np.zeros(100000)
    true_coef[:50] = rng.standard_normal(50) * 2.0
    eta = np.asarray(X @ true_coef).ravel()
    y = rng.binomial(1, 1.0 / (1.0 + np.exp(-eta))).astype(np.float64)

    def run():
        est = EmpiricalBayesGLM(
            alpha=1.0, link="logit", learning_rate=0.99999, sparse=True
        )
        est.fit(X, y)

    benchmark(run)


def test_sample_1_eb_glm_sparse_100k(benchmark, eb_glm_sparse_100k):
    est, X, _ = eb_glm_sparse_100k
    benchmark(est.sample, X, size=1)


def test_sample_10_eb_glm_sparse_100k(benchmark, eb_glm_sparse_100k):
    est, X, _ = eb_glm_sparse_100k
    benchmark(est.sample, X, size=10)


def test_predict_eb_glm_sparse_100k(benchmark, eb_glm_sparse_100k):
    est, X, _ = eb_glm_sparse_100k
    benchmark(est.predict, X)


def test_partial_fit_eb_glm_sparse_100k(benchmark, eb_glm_sparse_100k_fresh):
    rng = np.random.default_rng(99)

    def run():
        est, X, _ = eb_glm_sparse_100k_fresh()
        y = rng.integers(0, 2, size=X.shape[0]).astype(np.float64)
        est.partial_fit(X, y)

    benchmark(run)


def test_decay_eb_glm_sparse_100k(benchmark, eb_glm_sparse_100k_fresh):
    def run():
        est, X, _ = eb_glm_sparse_100k_fresh()
        est.decay(X, decay_rate=0.99)

    benchmark(run)


# -- nig dense 100 ------------------------------------------------------------


def test_fit_nig_dense_100(benchmark):
    rng = np.random.default_rng(42)
    X = rng.standard_normal((200, 100))
    y = rng.standard_normal(200)

    def run():
        est = NormalInverseGammaRegressor(sparse=False)
        est.fit(X, y)

    benchmark(run)


def test_sample_1_nig_dense_100(benchmark, nig_dense_100):
    est, X, _ = nig_dense_100
    benchmark(est.sample, X, size=1)


def test_sample_10_nig_dense_100(benchmark, nig_dense_100):
    est, X, _ = nig_dense_100
    benchmark(est.sample, X, size=10)


def test_predict_nig_dense_100(benchmark, nig_dense_100):
    est, X, _ = nig_dense_100
    benchmark(est.predict, X)


def test_partial_fit_nig_dense_100(benchmark, nig_dense_100_fresh):
    def run():
        est, X, _ = nig_dense_100_fresh()
        y = np.zeros(X.shape[0])
        est.partial_fit(X, y)

    benchmark(run)


def test_decay_nig_dense_100(benchmark, nig_dense_100_fresh):
    def run():
        est, X, _ = nig_dense_100_fresh()
        est.decay(X, decay_rate=0.99)

    benchmark(run)


# -- nig dense 1k -------------------------------------------------------------


def test_fit_nig_dense_1k(benchmark):
    rng = np.random.default_rng(42)
    X = rng.standard_normal((200, 1000))
    y = rng.standard_normal(200)

    def run():
        est = NormalInverseGammaRegressor(sparse=False)
        est.fit(X, y)

    benchmark(run)


def test_sample_1_nig_dense_1k(benchmark, nig_dense_1k):
    est, X, _ = nig_dense_1k
    benchmark(est.sample, X, size=1)


def test_sample_10_nig_dense_1k(benchmark, nig_dense_1k):
    est, X, _ = nig_dense_1k
    benchmark(est.sample, X, size=10)


def test_predict_nig_dense_1k(benchmark, nig_dense_1k):
    est, X, _ = nig_dense_1k
    benchmark(est.predict, X)


def test_partial_fit_nig_dense_1k(benchmark, nig_dense_1k_fresh):
    def run():
        est, X, _ = nig_dense_1k_fresh()
        y = np.zeros(X.shape[0])
        est.partial_fit(X, y)

    benchmark(run)


def test_decay_nig_dense_1k(benchmark, nig_dense_1k_fresh):
    def run():
        est, X, _ = nig_dense_1k_fresh()
        est.decay(X, decay_rate=0.99)

    benchmark(run)


# -- nig sparse 1k ------------------------------------------------------------


def test_fit_nig_sparse_1k(benchmark):
    rng = np.random.default_rng(42)
    X = csc_array(sparse_random(200, 1000, density=0.01, random_state=42))
    y = rng.standard_normal(200)

    def run():
        est = NormalInverseGammaRegressor(sparse=True)
        est.fit(X, y)

    benchmark(run)


def test_sample_1_nig_sparse_1k(benchmark, nig_sparse_1k):
    est, X, _ = nig_sparse_1k
    benchmark(est.sample, X, size=1)


def test_sample_10_nig_sparse_1k(benchmark, nig_sparse_1k):
    est, X, _ = nig_sparse_1k
    benchmark(est.sample, X, size=10)


def test_predict_nig_sparse_1k(benchmark, nig_sparse_1k):
    est, X, _ = nig_sparse_1k
    benchmark(est.predict, X)


def test_partial_fit_nig_sparse_1k(benchmark, nig_sparse_1k_fresh):
    def run():
        est, X, _ = nig_sparse_1k_fresh()
        y = np.zeros(X.shape[0])
        est.partial_fit(X, y)

    benchmark(run)


def test_decay_nig_sparse_1k(benchmark, nig_sparse_1k_fresh):
    def run():
        est, X, _ = nig_sparse_1k_fresh()
        est.decay(X, decay_rate=0.99)

    benchmark(run)


# -- nig sparse 100k ----------------------------------------------------------


def test_fit_nig_sparse_100k(benchmark):
    X = csc_array(sparse_random(200, 100_000, density=0.0001, random_state=42))
    rng = np.random.default_rng(42)
    y = rng.standard_normal(200)

    def run():
        est = NormalInverseGammaRegressor(sparse=True)
        est.fit(X, y)

    benchmark(run)


def test_sample_1_nig_sparse_100k(benchmark, nig_sparse_100k):
    est, X, _ = nig_sparse_100k
    benchmark(est.sample, X, size=1)


def test_sample_10_nig_sparse_100k(benchmark, nig_sparse_100k):
    est, X, _ = nig_sparse_100k
    benchmark(est.sample, X, size=10)


def test_predict_nig_sparse_100k(benchmark, nig_sparse_100k):
    est, X, _ = nig_sparse_100k
    benchmark(est.predict, X)


def test_partial_fit_nig_sparse_100k(benchmark, nig_sparse_100k_fresh):
    def run():
        est, X, _ = nig_sparse_100k_fresh()
        y = np.zeros(X.shape[0])
        est.partial_fit(X, y)

    benchmark(run)


def test_decay_nig_sparse_100k(benchmark, nig_sparse_100k_fresh):
    def run():
        est, X, _ = nig_sparse_100k_fresh()
        est.decay(X, decay_rate=0.99)

    benchmark(run)


# -- rvga_logit dense 100 (compare with glm_logit dense 100 above) -----------


def test_fit_rvga_logit_dense_100(benchmark):
    rng = np.random.default_rng(42)
    X = rng.standard_normal((200, 100))
    y = rng.integers(0, 2, size=200).astype(np.float64)

    def run():
        est = BayesianGLM(
            alpha=1.0,
            link="logit",
            sparse=False,
            approximator=RVGAApproximator(),
        )
        est.fit(X, y)

    benchmark(run)


def test_sample_1_rvga_logit_dense_100(benchmark, rvga_logit_dense_100):
    est, X, _ = rvga_logit_dense_100
    benchmark(est.sample, X, size=1)


def test_sample_10_rvga_logit_dense_100(benchmark, rvga_logit_dense_100):
    est, X, _ = rvga_logit_dense_100
    benchmark(est.sample, X, size=10)


def test_predict_rvga_logit_dense_100(benchmark, rvga_logit_dense_100):
    est, X, _ = rvga_logit_dense_100
    benchmark(est.predict, X)


def test_partial_fit_rvga_logit_dense_100(benchmark, rvga_logit_dense_100_fresh):
    def run():
        est, X, _ = rvga_logit_dense_100_fresh()
        y = np.zeros(X.shape[0])
        est.partial_fit(X, y)

    benchmark(run)


def test_decay_rvga_logit_dense_100(benchmark, rvga_logit_dense_100_fresh):
    def run():
        est, X, _ = rvga_logit_dense_100_fresh()
        est.decay(X, decay_rate=0.99)

    benchmark(run)


# -- rvga_logit dense 1k (compare with glm_logit dense 1k above) -------------


def test_fit_rvga_logit_dense_1k(benchmark):
    rng = np.random.default_rng(42)
    X = rng.standard_normal((200, 1000))
    y = rng.integers(0, 2, size=200).astype(np.float64)

    def run():
        est = BayesianGLM(
            alpha=1.0,
            link="logit",
            sparse=False,
            approximator=RVGAApproximator(),
        )
        est.fit(X, y)

    benchmark(run)


def test_sample_1_rvga_logit_dense_1k(benchmark, rvga_logit_dense_1k):
    est, X, _ = rvga_logit_dense_1k
    benchmark(est.sample, X, size=1)


def test_sample_10_rvga_logit_dense_1k(benchmark, rvga_logit_dense_1k):
    est, X, _ = rvga_logit_dense_1k
    benchmark(est.sample, X, size=10)


def test_predict_rvga_logit_dense_1k(benchmark, rvga_logit_dense_1k):
    est, X, _ = rvga_logit_dense_1k
    benchmark(est.predict, X)


def test_partial_fit_rvga_logit_dense_1k(benchmark, rvga_logit_dense_1k_fresh):
    def run():
        est, X, _ = rvga_logit_dense_1k_fresh()
        y = np.zeros(X.shape[0])
        est.partial_fit(X, y)

    benchmark(run)


def test_decay_rvga_logit_dense_1k(benchmark, rvga_logit_dense_1k_fresh):
    def run():
        est, X, _ = rvga_logit_dense_1k_fresh()
        est.decay(X, decay_rate=0.99)

    benchmark(run)


# -- rvga_logit sparse 1k (compare with glm_logit sparse 1k above) -----------


def test_fit_rvga_logit_sparse_1k(benchmark):
    rng = np.random.default_rng(42)
    X = csc_array(sparse_random(200, 1000, density=0.01, random_state=42))
    y = rng.integers(0, 2, size=200).astype(np.float64)

    def run():
        est = BayesianGLM(
            alpha=1.0,
            link="logit",
            sparse=True,
            approximator=RVGAApproximator(),
        )
        est.fit(X, y)

    benchmark(run)


def test_sample_1_rvga_logit_sparse_1k(benchmark, rvga_logit_sparse_1k):
    est, X, _ = rvga_logit_sparse_1k
    benchmark(est.sample, X, size=1)


def test_sample_10_rvga_logit_sparse_1k(benchmark, rvga_logit_sparse_1k):
    est, X, _ = rvga_logit_sparse_1k
    benchmark(est.sample, X, size=10)


def test_predict_rvga_logit_sparse_1k(benchmark, rvga_logit_sparse_1k):
    est, X, _ = rvga_logit_sparse_1k
    benchmark(est.predict, X)


def test_partial_fit_rvga_logit_sparse_1k(benchmark, rvga_logit_sparse_1k_fresh):
    def run():
        est, X, _ = rvga_logit_sparse_1k_fresh()
        y = np.zeros(X.shape[0])
        est.partial_fit(X, y)

    benchmark(run)


def test_decay_rvga_logit_sparse_1k(benchmark, rvga_logit_sparse_1k_fresh):
    def run():
        est, X, _ = rvga_logit_sparse_1k_fresh()
        est.decay(X, decay_rate=0.99)

    benchmark(run)


# -- rvga_logit sparse 100k (compare with glm_logit sparse 100k above) -------


def test_fit_rvga_logit_sparse_100k(benchmark):
    X = csc_array(sparse_random(200, 100_000, density=0.0001, random_state=42))
    rng = np.random.default_rng(42)
    y = rng.integers(0, 2, size=200).astype(np.float64)

    def run():
        est = BayesianGLM(
            alpha=1.0,
            link="logit",
            sparse=True,
            approximator=RVGAApproximator(),
        )
        est.fit(X, y)

    benchmark(run)


def test_sample_1_rvga_logit_sparse_100k(benchmark, rvga_logit_sparse_100k):
    est, X, _ = rvga_logit_sparse_100k
    benchmark(est.sample, X, size=1)


def test_predict_rvga_logit_sparse_100k(benchmark, rvga_logit_sparse_100k):
    est, X, _ = rvga_logit_sparse_100k
    benchmark(est.predict, X)


def test_partial_fit_rvga_logit_sparse_100k(benchmark, rvga_logit_sparse_100k_fresh):
    def run():
        est, X, _ = rvga_logit_sparse_100k_fresh()
        y = np.zeros(X.shape[0])
        est.partial_fit(X, y)

    benchmark(run)


def test_decay_rvga_logit_sparse_100k(benchmark, rvga_logit_sparse_100k_fresh):
    def run():
        est, X, _ = rvga_logit_sparse_100k_fresh()
        est.decay(X, decay_rate=0.99)

    benchmark(run)


# -- rvga_logit sparse 1m (compare with glm_logit sparse 1m above) -----------


@pytest.mark.slow
def test_fit_rvga_logit_sparse_1m(benchmark):
    X = csc_array(sparse_random(200, 1_000_000, density=0.00001, random_state=42))
    rng = np.random.default_rng(42)
    y = rng.integers(0, 2, size=200).astype(np.float64)

    def run():
        est = BayesianGLM(
            alpha=1.0,
            link="logit",
            sparse=True,
            approximator=RVGAApproximator(),
        )
        est.fit(X, y)

    benchmark(run)


@pytest.mark.slow
def test_sample_1_rvga_logit_sparse_1m(benchmark, rvga_logit_sparse_1m):
    est, X, _ = rvga_logit_sparse_1m
    benchmark(est.sample, X, size=1)


@pytest.mark.slow
def test_sample_10_rvga_logit_sparse_1m(benchmark, rvga_logit_sparse_1m):
    est, X, _ = rvga_logit_sparse_1m
    benchmark(est.sample, X, size=10)


@pytest.mark.slow
def test_predict_rvga_logit_sparse_1m(benchmark, rvga_logit_sparse_1m):
    est, X, _ = rvga_logit_sparse_1m
    benchmark(est.predict, X)


@pytest.mark.slow
def test_partial_fit_rvga_logit_sparse_1m(benchmark, rvga_logit_sparse_1m_fresh):
    def run():
        est, X, _ = rvga_logit_sparse_1m_fresh()
        y = np.zeros(X.shape[0])
        est.partial_fit(X, y)

    benchmark(run)


@pytest.mark.slow
def test_decay_rvga_logit_sparse_1m(benchmark, rvga_logit_sparse_1m_fresh):
    def run():
        est, X, _ = rvga_logit_sparse_1m_fresh()
        est.decay(X, decay_rate=0.99)

    benchmark(run)


# -- rvga_logit sparse, many rows (exercises the minibatch/batch_size path) ---
# All other rvga fixtures use n_obs=200 (< batch_size=2048), so they only ever
# hit the single joint update. These fit 8192 rows, forcing the sequential
# minibatch loop. This is the path that scales with n_samples; a regression in
# its per-chunk overhead (or an accidental quadratic in the Gram) shows here.


def test_fit_rvga_logit_sparse_8k_rows(benchmark):
    n_obs, n_features = 8_192, 2_000
    X = csc_array(sparse_random(n_obs, n_features, density=0.005, random_state=42))
    rng = np.random.default_rng(42)
    y = rng.integers(0, 2, size=n_obs).astype(np.float64)

    def run():
        est = BayesianGLM(
            alpha=1.0,
            link="logit",
            sparse=True,
            approximator=RVGAApproximator(),  # default batch_size=2048 -> 4 chunks
        )
        est.fit(X, y)

    benchmark(run)


def test_fit_rvga_logit_sparse_8k_rows_joint(benchmark):
    """Same data, batch_size=None forces a single joint update with the full
    8192x8192 Gram. Pairs with the chunked fit above to make the batch path's
    memory/time advantage at large n_samples visible."""
    n_obs, n_features = 8_192, 2_000
    X = csc_array(sparse_random(n_obs, n_features, density=0.005, random_state=42))
    rng = np.random.default_rng(42)
    y = rng.integers(0, 2, size=n_obs).astype(np.float64)

    def run():
        est = BayesianGLM(
            alpha=1.0,
            link="logit",
            sparse=True,
            approximator=RVGAApproximator(batch_size=None),
        )
        est.fit(X, y)

    benchmark(run)


def test_partial_fit_rvga_logit_sparse_8k_rows(
    benchmark, rvga_logit_sparse_8k_rows_fresh
):
    def run():
        est, X, _ = rvga_logit_sparse_8k_rows_fresh()
        y = np.zeros(X.shape[0])
        est.partial_fit(X, y)

    benchmark(run)


def test_predict_rvga_logit_sparse_8k_rows(benchmark, rvga_logit_sparse_8k_rows):
    est, X, _ = rvga_logit_sparse_8k_rows
    benchmark(est.predict, X)


def test_sample_1_rvga_logit_sparse_8k_rows(benchmark, rvga_logit_sparse_8k_rows):
    est, X, _ = rvga_logit_sparse_8k_rows
    benchmark(est.sample, X, size=1)


def test_decay_rvga_logit_sparse_8k_rows(benchmark, rvga_logit_sparse_8k_rows_fresh):
    def run():
        est, X, _ = rvga_logit_sparse_8k_rows_fresh()
        est.decay(X, decay_rate=0.99)

    benchmark(run)
