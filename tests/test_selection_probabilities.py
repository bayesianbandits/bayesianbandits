"""Tests for ``select_with_probabilities``: the distribution a policy drew
its decision from, conditional on the samples it was handed."""

from typing import Any, List

import numpy as np
import pytest

from bayesianbandits import EXP3A, Arm

# Every policy reporting probabilities; the tests below run on each
POLICIES: List[Any] = [EXP3A(eta=1.0, gamma=0.1, samples=8)]


def _arms(n_arms: int) -> List[Arm[Any, int]]:
    return [Arm(i, learner=None) for i in range(n_arms)]


def _tokens(arms: List[Arm[Any, int]]) -> List[int]:
    return [arm.action_token for arm in arms]


@pytest.mark.parametrize("policy", POLICIES, ids=repr)
def test_rows_are_distributions(policy: Any) -> None:
    rng = np.random.default_rng(0)
    samples = rng.normal(size=(4, 5, policy.samples_needed))
    arms = _arms(4)

    chosen, probs = policy.select_with_probabilities(samples, arms, rng)

    assert probs.shape == (5, 4)
    assert np.all(probs >= 0)
    assert np.allclose(probs.sum(axis=1), 1.0)
    assert np.all(probs[np.arange(5), _tokens(chosen)] > 0)


@pytest.mark.parametrize("policy", POLICIES, ids=repr)
def test_agrees_with_select(policy: Any) -> None:
    samples = np.random.default_rng(0).normal(size=(4, 5, policy.samples_needed))
    arms = _arms(4)

    plain = policy.select(samples, arms, np.random.default_rng(1))
    chosen, _ = policy.select_with_probabilities(
        samples, arms, np.random.default_rng(1)
    )

    assert _tokens(chosen) == _tokens(plain)


@pytest.mark.parametrize("policy", POLICIES, ids=repr)
def test_ips_is_unbiased(policy: Any) -> None:
    """IPS with the reported propensities recovers a target policy's value.

    The logger sees noisy draws around fixed arm means; the target always
    plays the arm the logger likes least, so the importance weights are
    large and a wrong propensity (one-hot, say) shows up as bias.
    """
    rng = np.random.default_rng(2)
    means = np.array([1.0, 0.0, -1.0])
    n_rounds = 20_000
    samples = means[:, None, None] + rng.normal(
        size=(3, n_rounds, policy.samples_needed)
    )
    arms = _arms(3)

    chosen, probs = policy.select_with_probabilities(samples, arms, rng)
    played = np.array(_tokens(chosen))
    rewards = means[played] + rng.normal(size=n_rounds)

    target = 2
    terms = (played == target) * rewards / probs[np.arange(n_rounds), played]
    estimate = terms.mean()
    stderr = terms.std(ddof=1) / np.sqrt(n_rounds)

    assert abs(estimate - means[target]) < 4 * stderr
