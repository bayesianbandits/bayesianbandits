"""The policy/agent contract: policies state a need, agents pick a method.

What matters is that each policy declares a level it can actually
consume, and that the agent supplies something at least that strong.
The lattice itself is an ``IntEnum``, so its ordering needs no test.
"""

from unittest import mock

import numpy as np
import pytest

from bayesianbandits import (
    EXP3A,
    Arm,
    ArmColumnFeaturizer,
    DrawKind,
    EpsilonGreedy,
    InformationDirectedSampling,
    LipschitzContextualAgent,
    NormalRegressor,
    ThompsonSampling,
    UpperConfidenceBound,
)
from bayesianbandits._arm import resolve_reward_space_sampler


@pytest.mark.parametrize(
    ("policy", "expected"),
    [
        (UpperConfidenceBound(alpha=0.8), DrawKind.MARGINAL_ONLY),
        (EpsilonGreedy(epsilon=0.1), DrawKind.MARGINAL_ONLY),
        (EXP3A(eta=1.0), DrawKind.MARGINAL_ONLY),
        (InformationDirectedSampling(samples=50), DrawKind.CONTEXT_JOINT),
        (ThompsonSampling(), DrawKind.JOINT),
    ],
    ids=lambda p: type(p).__name__ if hasattr(p, "consumes") else str(p),
)
def test_each_policy_declares_what_it_consumes(policy, expected):
    """Understating this is a correctness bug: the agent will supply
    draws too weak for the statistics the policy reads."""
    assert policy.consumes is expected


def _lipschitz(policy, batch_reward_function=None, n_arms=3, d=40):
    arms = [Arm(i, reward_function=None, learner=None) for i in range(n_arms)]
    agent = LipschitzContextualAgent(
        arms=arms,
        policy=policy,
        arm_featurizer=ArmColumnFeaturizer(column_name="arm"),
        learner=NormalRegressor(alpha=1.0, beta=1.0),
        batch_reward_function=batch_reward_function,
        random_seed=0,
    )
    rng = np.random.default_rng(0)
    X = rng.standard_normal((20, d))
    agent.pull(X[:1])
    agent.update(X, rng.standard_normal(20))
    return agent, rng


class TestAgentSuppliesAtLeastWhatIsAsked:
    def test_marginal_only_policy_takes_the_marginal_path(self):
        agent, rng = _lipschitz(UpperConfidenceBound(alpha=0.8, samples=50))
        with mock.patch(
            "bayesianbandits.api.resolve_marginal_sampler"
        ) as resolve_marginal:
            resolve_marginal.side_effect = (
                lambda learner: learner.sample_marginal  # noqa: E731
            )
            agent.pull(rng.standard_normal((2, 40)))
        resolve_marginal.assert_called_once()

    def test_context_joint_policy_takes_the_blocked_path(self):
        agent, rng = _lipschitz(InformationDirectedSampling(samples=200))
        with mock.patch(
            "bayesianbandits.api.resolve_reward_space_sampler",
            wraps=resolve_reward_space_sampler,
        ) as resolve_blocks:
            agent.pull(rng.standard_normal((2, 40)))
        resolve_blocks.assert_called_once()
        # one block per arm, which is what makes the draws context-joint
        assert resolve_blocks.call_args.kwargs["block_size"] == len(agent.arms)

    def test_joint_policy_takes_neither_reduction_path(self):
        agent, rng = _lipschitz(ThompsonSampling())
        with (
            mock.patch("bayesianbandits.api.resolve_marginal_sampler") as marg,
            mock.patch("bayesianbandits.api.resolve_reward_space_sampler") as blocks,
        ):
            agent.pull(rng.standard_normal((2, 40)))
        marg.assert_not_called()
        blocks.assert_not_called()


class TestBatchRewardFunctionWidens:
    @staticmethod
    def _share_of_total(samples, action_tokens):
        return samples / samples.sum(axis=0, keepdims=True)

    def test_a_supplied_batch_reward_widens_past_marginal(self):
        """It runs before the policy and may combine arms within a draw,
        so the arms must be jointly distributed whatever the policy reads."""
        agent, rng = _lipschitz(
            UpperConfidenceBound(alpha=0.8, samples=50), self._share_of_total
        )
        with mock.patch(
            "bayesianbandits.api.resolve_marginal_sampler"
        ) as resolve_marginal:
            agent.pull(rng.standard_normal((2, 40)))
        resolve_marginal.assert_not_called()

    def test_widening_cannot_lower_a_joint_policy(self):
        agent, rng = _lipschitz(ThompsonSampling(), self._share_of_total)
        with (
            mock.patch("bayesianbandits.api.resolve_marginal_sampler") as marg,
            mock.patch("bayesianbandits.api.resolve_reward_space_sampler") as blocks,
        ):
            agent.pull(rng.standard_normal((2, 40)))
        marg.assert_not_called()
        blocks.assert_not_called()
