"""Benchmarks for ``InformationDirectedSampling.select``.

The decision, not sampling, dominates an IDS pull at realistic arm
counts, so these run ``select`` on a pre-drawn tensor. The ``_view`` /
``_carray`` pairs measure the layout normalization (agents hand policies
a transposed view whose draw axis would otherwise carry the largest
stride); the ``96_arms`` cases stress the pair scan; ``top_k_4`` guards
the per-slot multiplier.
"""

import numpy as np
import pytest

from bayesianbandits import InformationDirectedSampling


class _Arm:
    """Stand-in: ``select`` only ever indexes the arm list."""


def _draws(n_arms, n_contexts, size, seed=0):
    """A pre-drawn tensor in the draw-major view layout."""
    drawn = np.random.default_rng(seed).standard_normal((size, n_arms, n_contexts))
    return drawn.transpose(1, 2, 0)


@pytest.fixture(scope="module")
def arms_96():
    return [_Arm() for _ in range(96)]


@pytest.fixture(scope="module")
def arms_20():
    return [_Arm() for _ in range(20)]


@pytest.fixture(scope="module")
def policy():
    return InformationDirectedSampling(samples=1000)


# -- 96 arms, 8 contexts, size=1000 -------------------------------------------


def test_select_96_arms_8_contexts_view(benchmark, policy, arms_96):
    samples = _draws(96, 8, 1000)
    rng = np.random.default_rng(0)
    benchmark(policy.select, samples, arms_96, rng)


def test_select_96_arms_8_contexts_carray(benchmark, policy, arms_96):
    samples = np.ascontiguousarray(_draws(96, 8, 1000))
    rng = np.random.default_rng(0)
    benchmark(policy.select, samples, arms_96, rng)


# -- 96 arms, 64 contexts, size=1000 (pair scan and copy both bite) -----------


def test_select_96_arms_64_contexts_view(benchmark, policy, arms_96):
    samples = _draws(96, 64, 1000)
    rng = np.random.default_rng(0)
    benchmark(policy.select, samples, arms_96, rng)


def test_select_96_arms_64_contexts_carray(benchmark, policy, arms_96):
    samples = np.ascontiguousarray(_draws(96, 64, 1000))
    rng = np.random.default_rng(0)
    benchmark(policy.select, samples, arms_96, rng)


# -- 20 arms, 8 contexts, size=1000 (the small end stays cheap) ---------------


def test_select_20_arms_8_contexts_view(benchmark, policy, arms_20):
    samples = _draws(20, 8, 1000)
    rng = np.random.default_rng(0)
    benchmark(policy.select, samples, arms_20, rng)


# -- slates -------------------------------------------------------------------


def test_select_96_arms_8_contexts_top_k_4(benchmark, policy, arms_96):
    samples = _draws(96, 8, 1000)
    rng = np.random.default_rng(0)
    benchmark(policy.select, samples, arms_96, rng, 4)
