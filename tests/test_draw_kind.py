"""The policy/agent contract: policies state a need, agents pick a method.

``DrawKind`` is the whole interface between the two. These check the
lattice itself, that every shipped policy declares the weakest kind it
can actually consume, and that an agent supplies something at least
that strong.
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
from bayesianbandits._arm import batch_identity, resolve_reward_space_sampler

ALL_POLICIES = [
    UpperConfidenceBound(alpha=0.8, samples=50),
    EpsilonGreedy(epsilon=0.1, samples=50),
    EXP3A(eta=1.0, samples=50),
    ThompsonSampling(),
    InformationDirectedSampling(samples=50),
]


class TestLattice:
    def test_is_totally_ordered_from_weakest_to_strongest(self):
        assert DrawKind.MARGINAL_ONLY < DrawKind.CONTEXT_JOINT < DrawKind.JOINT

    def test_max_widens_and_never_narrows(self):
        """An agent combines requirements with ``max``, so the result is
        always at least as strong as either input."""
        for a in DrawKind:
            for b in DrawKind:
                assert max(a, b) >= a
                assert max(a, b) >= b

    def test_joint_is_the_safe_default(self):
        """``JOINT`` must be the strongest, so a policy that does not
        think about this gets correctness rather than speed."""
        assert max(DrawKind) is DrawKind.JOINT


class TestPolicyDeclarations:
    @pytest.mark.parametrize("policy", ALL_POLICIES, ids=lambda p: type(p).__name__)
    def test_every_policy_declares_a_draw_kind(self, policy):
        assert isinstance(policy.consumes, DrawKind)

    def test_policies_no_longer_carry_method_flags(self):
        """The old contract named sampling methods a policy would
        tolerate; the new one names what it consumes. Nothing should
        still be answering to the old names."""
        for policy in ALL_POLICIES:
            assert not hasattr(policy, "marginal_ok")
            assert not hasattr(policy, "reward_space_ok")

    def test_thompson_sampling_needs_everything_joint(self):
        """It compares arms drawn from one weight vector."""
        assert ThompsonSampling().consumes is DrawKind.JOINT

    @pytest.mark.parametrize(
        "policy",
        [
            UpperConfidenceBound(alpha=0.8),
            EpsilonGreedy(epsilon=0.1),
            EXP3A(eta=1.0),
        ],
        ids=lambda p: type(p).__name__,
    )
    def test_per_arm_scorers_need_only_marginals(self, policy):
        assert policy.consumes is DrawKind.MARGINAL_ONLY


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
    def test_identity_batch_reward_does_not_widen(self):
        """``batch_identity`` combines nothing, so a ``MARGINAL_ONLY``
        policy keeps the cheapest path."""
        agent, rng = _lipschitz(UpperConfidenceBound(alpha=0.8, samples=50))
        assert agent.batch_reward_function is batch_identity
        with mock.patch(
            "bayesianbandits.api.resolve_marginal_sampler"
        ) as resolve_marginal:
            resolve_marginal.side_effect = (
                lambda learner: learner.sample_marginal  # noqa: E731
            )
            agent.pull(rng.standard_normal((2, 40)))
        resolve_marginal.assert_called_once()

    def test_a_supplied_batch_reward_widens_past_marginal(self):
        def share_of_total(samples, action_tokens):
            return samples / samples.sum(axis=0, keepdims=True)

        agent, rng = _lipschitz(
            UpperConfidenceBound(alpha=0.8, samples=50), share_of_total
        )
        with mock.patch(
            "bayesianbandits.api.resolve_marginal_sampler"
        ) as resolve_marginal:
            agent.pull(rng.standard_normal((2, 40)))
        resolve_marginal.assert_not_called()

    def test_widening_cannot_lower_a_joint_policy(self):
        def share_of_total(samples, action_tokens):
            return samples / samples.sum(axis=0, keepdims=True)

        agent, rng = _lipschitz(ThompsonSampling(), share_of_total)
        with (
            mock.patch("bayesianbandits.api.resolve_marginal_sampler") as marg,
            mock.patch("bayesianbandits.api.resolve_reward_space_sampler") as blocks,
        ):
            agent.pull(rng.standard_normal((2, 40)))
        marg.assert_not_called()
        blocks.assert_not_called()
