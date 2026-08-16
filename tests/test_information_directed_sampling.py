"""
Test suite for the variance-based information-directed sampling (IDS) policy.
"""

from typing import List

import numpy as np
import pytest

from bayesianbandits import (
    Agent,
    Arm,
    ArmColumnFeaturizer,
    ContextualAgent,
    InformationDirectedSampling,
    LipschitzContextualAgent,
    NormalInverseGammaRegressor,
    NormalRegressor,
)
from bayesianbandits.policies._information_directed_sampling import (
    _optimal_two_point,
    _sample_information_ratio_minimizer,
    _vids_statistics,
)


def pairwise_grid_min_ratio(
    delta: np.ndarray, v: np.ndarray, n_grid: int = 2001
) -> float:
    """Reference: grid-search the information ratio over all pairs."""
    q = np.linspace(0.0, 1.0, n_grid)
    best = np.inf
    for a in range(len(delta)):
        for b in range(len(delta)):
            num = (q * delta[a] + (1 - q) * delta[b]) ** 2
            den = q * v[a] + (1 - q) * v[b]
            ratio = np.full_like(num, np.inf)
            np.divide(num, den, out=ratio, where=den > 0.0)
            ratio[num == 0.0] = 0.0
            best = min(best, ratio.min())
    return best


def simplex_min_ratio(delta: np.ndarray, v: np.ndarray, n_draws: int = 20_000) -> float:
    """Reference over the *whole* simplex, not just its edges.

    Russo & Van Roy's Prop. 6 says the minimizer is supported on at most
    two arms, which is what licenses the closed-form pair scan. Sampling
    full-support mixtures is a direct probe of that claim: if any of them
    beats the pair optimum, the scan is unsound.
    """
    rng = np.random.default_rng(0)
    w = rng.dirichlet(np.ones(len(delta)), size=n_draws)
    num = (w @ delta) ** 2
    den = w @ v
    ratio = np.full(n_draws, np.inf)
    np.divide(num, den, out=ratio, where=den > 0.0)
    ratio[num == 0.0] = 0.0
    return float(ratio.min())


class TestPairOptimizer:
    """Closed-form pair optimization against a brute-force grid."""

    @pytest.mark.parametrize("seed", range(3))
    @pytest.mark.parametrize("n_arms", [2, 5, 8])
    def test_matches_brute_force(self, seed: int, n_arms: int):
        rng = np.random.default_rng(seed)
        delta = rng.uniform(0.0, 2.0, size=(n_arms, 1))
        v = rng.uniform(0.0, 1.5, size=(n_arms, 1))
        alive = np.ones((n_arms, 1), dtype=bool)

        a_idx, b_idx, q, ratio = _optimal_two_point(delta, v, alive)
        reference = pairwise_grid_min_ratio(delta[:, 0], v[:, 0])

        # The closed form is exact, so it can only be at or below the grid
        # minimum; the grid can only overshoot by its resolution.
        assert ratio[0] <= reference + 1e-12
        assert reference - ratio[0] <= 1e-4 * max(1.0, reference)

    @pytest.mark.parametrize("seed", range(3))
    @pytest.mark.parametrize("n_arms", [3, 6])
    def test_no_full_support_mixture_beats_the_pair_optimum(
        self, seed: int, n_arms: int
    ):
        """The closed form scans pairs only. That is sound exactly because
        the minimizer is supported on at most two arms, so no mixture over
        the whole simplex may beat it."""
        rng = np.random.default_rng(1000 + seed)
        delta = rng.uniform(0.0, 2.0, size=(n_arms, 1))
        v = rng.uniform(0.0, 1.5, size=(n_arms, 1))
        alive = np.ones((n_arms, 1), dtype=bool)

        *_, ratio = _optimal_two_point(delta, v, alive)
        assert ratio[0] <= simplex_min_ratio(delta[:, 0], v[:, 0]) + 1e-12

    @pytest.mark.parametrize("seed", range(3))
    def test_reported_mixture_achieves_reported_ratio(self, seed: int):
        rng = np.random.default_rng(seed)
        delta = rng.uniform(0.0, 2.0, size=(4, 3))
        v = rng.uniform(0.0, 1.5, size=(4, 3))
        # context 2 has regret but no information gain anywhere: its
        # ratio is infinite, which the loop below must skip
        v[:, 2] = 0.0
        alive = np.ones((4, 3), dtype=bool)

        a_idx, b_idx, q, ratio = _optimal_two_point(delta, v, alive)
        assert not np.isfinite(ratio[2])
        for c in range(3):
            if not np.isfinite(ratio[c]):
                continue
            num = (q[c] * delta[a_idx[c], c] + (1 - q[c]) * delta[b_idx[c], c]) ** 2
            den = q[c] * v[a_idx[c], c] + (1 - q[c]) * v[b_idx[c], c]
            achieved = 0.0 if num == 0.0 else num / den
            assert achieved == pytest.approx(ratio[c], abs=1e-12)

    def test_zero_regret_arm_gives_zero_ratio(self):
        # An arm with zero expected regret is played outright: ratio 0.
        delta = np.array([[0.0], [0.7]])
        v = np.array([[0.0], [0.3]])
        alive = np.ones((2, 1), dtype=bool)
        a_idx, b_idx, q, ratio = _optimal_two_point(delta, v, alive)
        assert ratio[0] == 0.0
        played = a_idx[0] if q[0] == 1.0 else b_idx[0]
        assert played == 0 or q[0] not in (0.0, 1.0)

    def test_no_information_anywhere_is_infinite(self):
        # Positive regret everywhere, zero gain everywhere: nothing to buy.
        delta = np.array([[0.5], [0.9]])
        v = np.zeros((2, 1))
        alive = np.ones((2, 1), dtype=bool)
        *_, ratio = _optimal_two_point(delta, v, alive)
        assert np.isinf(ratio[0])

    def test_dead_arms_are_excluded(self):
        # The globally best pair involves arm 0; killing it must change the answer.
        delta = np.array([[0.0], [0.5], [0.6]])
        v = np.array([[1.0], [0.4], [0.5]])
        alive = np.array([[False], [True], [True]])
        a_idx, b_idx, q, ratio = _optimal_two_point(delta, v, alive)
        assert a_idx[0] != 0 and b_idx[0] != 0
        assert np.isfinite(ratio[0])

    def test_interior_mixture_sampling_frequencies(self):
        # delta/v chosen so the optimum is a strict interior mixture.
        delta = np.array([[0.1], [0.9]])
        v = np.array([[0.01], [1.0]])
        alive = np.ones((2, 1), dtype=bool)
        a_idx, b_idx, q, ratio = _optimal_two_point(delta, v, alive)
        assert 0.0 < q[0] < 1.0

        rng = np.random.default_rng(0)
        n_trials = 4_000
        picks = np.concatenate(
            [
                _sample_information_ratio_minimizer(delta, v, alive, rng)
                for _ in range(n_trials)
            ]
        )
        freq_a = np.mean(picks == a_idx[0])
        sigma = np.sqrt(q[0] * (1 - q[0]) / n_trials)
        assert abs(freq_a - q[0]) < 5 * sigma

    def test_resolved_posterior_falls_back_to_greedy(self):
        """With no information left to buy, every pair's ratio is
        infinite and the minimizer plays the lowest-regret alive arm
        rather than whatever pair (0, 0) the argmin happens to land on."""
        # positive regret, zero gain everywhere -> ratio inf
        delta = np.array([[0.5, 2.0], [0.2, 1.0], [0.9, 0.4]])
        v = np.zeros((3, 2))
        alive = np.ones((3, 2), dtype=bool)
        _, _, _, ratio = _optimal_two_point(delta, v, alive)
        assert not np.isfinite(ratio).any()

        rng = np.random.default_rng(0)
        picks = _sample_information_ratio_minimizer(delta, v, alive, rng)
        assert picks.tolist() == [1, 2]  # argmin delta per context

    def test_resolved_greedy_respects_dead_arms(self):
        """The greedy fallback must not resurrect an arm already taken
        by an earlier top_k slot, even when it has the lowest regret."""
        delta = np.array([[0.5], [0.2], [0.9]])
        v = np.zeros((3, 1))
        alive = np.array([[True], [False], [True]])  # best arm is dead
        picks = _sample_information_ratio_minimizer(
            delta, v, alive, np.random.default_rng(0)
        )
        assert picks.tolist() == [0]


class TestVidsStatistics:
    """Monte Carlo regret and information-gain estimates."""

    def test_hand_computed_two_arm_case(self):
        # arm 0 draws [2, 0], arm 1 draws [1, 1]: each optimal in one draw.
        samples = np.array([[[2.0, 0.0]], [[1.0, 1.0]]])
        alive = np.ones((2, 1), dtype=bool)
        delta, v = _vids_statistics(samples, alive)
        # rho* = mean(max) = 1.5, mu = [1, 1]
        np.testing.assert_allclose(delta, [[0.5], [0.5]])
        # E[theta_0 | A*] in {2, 0} with p = 1/2 each -> v_0 = 1
        # E[theta_1 | A*] = 1 always -> v_1 = 0
        np.testing.assert_allclose(v, [[1.0], [0.0]])

    def test_dominant_arm_has_zero_regret_and_no_information(self):
        # arm 0 wins every draw: posterior resolved, all gains vanish.
        samples = np.stack(
            [np.full((1, 50), 3.0), np.random.default_rng(0).normal(0.0, 0.5, (1, 50))]
        )
        alive = np.ones((2, 1), dtype=bool)
        delta, v = _vids_statistics(samples, alive)
        assert delta[0, 0] == 0.0
        np.testing.assert_allclose(v, 0.0, atol=1e-12)

    def test_masked_arm_is_excluded_from_argmax(self):
        # With arm 0 dead, the subgame is arms 1 vs 2.
        samples = np.array(
            [[[9.0, 9.0]], [[2.0, 0.0]], [[1.0, 1.0]]]
        )  # arm 0 dominates when alive
        alive = np.array([[False], [True], [True]])
        delta, v = _vids_statistics(samples, alive)
        np.testing.assert_allclose(delta[1:], [[0.5], [0.5]])
        np.testing.assert_allclose(v[1:], [[1.0], [0.0]])

    def test_correlated_draws_transfer_information(self):
        # Perfectly anti-correlated arms: learning who is optimal moves
        # both conditional means, so *both* arms carry information -- the
        # signature shared-learner effect. With theta ~ N(0, 1) the gains
        # have closed forms: E[theta | theta > 0] = sqrt(2/pi) gives
        # v = 2/pi jointly, while independent draws of the same marginals
        # give E[Y | Y < X] = -1/sqrt(pi) and v = 1/pi -- exactly half.
        rng = np.random.default_rng(3)
        theta = rng.normal(0.0, 1.0, size=100_000)
        joint = np.stack([theta[None, :], -theta[None, :]])
        alive = np.ones((2, 1), dtype=bool)
        _, v_joint = _vids_statistics(joint, alive)

        shuffled = joint.copy()
        rng.shuffle(shuffled[1, 0])
        _, v_indep = _vids_statistics(shuffled, alive)

        np.testing.assert_allclose(v_joint, 2 / np.pi, rtol=0.05)
        np.testing.assert_allclose(v_indep[1, 0], 1 / np.pi, rtol=0.05)


class TestSelect:
    """Policy-level selection semantics."""

    def make_arms(self, n: int) -> List[Arm]:
        return [Arm(i, learner=NormalInverseGammaRegressor()) for i in range(n)]

    def test_resolved_posterior_plays_greedy(self):
        # Constant samples with distinct means: no information anywhere,
        # fallback plays the best arm deterministically.
        samples = np.stack([np.full((1, 100), m) for m in [1.0, 3.0, 2.0]])
        arms = self.make_arms(3)
        policy = InformationDirectedSampling()
        rng = np.random.default_rng(0)
        for _ in range(10):
            assert policy.select(samples, arms, rng)[0] is arms[1]

    def test_pure_informative_arm_is_played(self):
        # arm 0 known to give 1.0; arm 1 is a coin flip between 2 and 0.
        # Regret is equal (0.5 each), all information sits on arm 1, so
        # IDS plays arm 1 with probability one.
        draws = np.tile([2.0, 0.0], 50)
        samples = np.stack([np.full((1, 100), 1.0), draws[None, :]])
        arms = self.make_arms(2)
        policy = InformationDirectedSampling()
        rng = np.random.default_rng(0)
        for _ in range(10):
            assert policy.select(samples, arms, rng)[0] is arms[1]

    def test_top_k_one_matches_base_policy(self):
        rng_state = np.random.default_rng(7)
        samples = rng_state.normal(size=(4, 3, 200))
        arms = self.make_arms(4)
        policy = InformationDirectedSampling()

        base = policy.select(samples, arms, np.random.default_rng(11))
        slates = policy.select(samples, arms, np.random.default_rng(11), top_k=1)
        assert [s[0] for s in slates] == base

    def test_top_k_prefix_consistency_and_uniqueness(self):
        samples = np.random.default_rng(5).normal(size=(5, 4, 300))
        arms = self.make_arms(5)
        policy = InformationDirectedSampling()

        two = policy.select(samples, arms, np.random.default_rng(3), top_k=2)
        three = policy.select(samples, arms, np.random.default_rng(3), top_k=3)
        for slate2, slate3 in zip(two, three):
            assert slate3[:2] == slate2
            assert len(set(id(a) for a in slate3)) == 3  # without replacement

    def test_top_k_at_least_n_arms_returns_all(self):
        samples = np.random.default_rng(2).normal(size=(3, 2, 100))
        arms = self.make_arms(3)
        policy = InformationDirectedSampling()
        slates = policy.select(samples, arms, np.random.default_rng(0), top_k=10)
        for slate in slates:
            assert sorted(a.action_token for a in slate) == [0, 1, 2]

    def test_top_k_resolved_posterior_sorts_by_mean(self):
        samples = np.stack([np.full((1, 50), m) for m in [1.0, 4.0, 2.0, 3.0]])
        arms = self.make_arms(4)
        policy = InformationDirectedSampling()
        (slate,) = policy.select(samples, arms, np.random.default_rng(0), top_k=4)
        assert [a.action_token for a in slate] == [1, 3, 2, 0]


class TestPolicyBasics:
    def test_initialization_and_repr(self):
        policy = InformationDirectedSampling()
        assert policy.samples == 1000
        assert policy.samples_needed == 1000
        assert repr(policy) == "InformationDirectedSampling(samples=1000)"

        policy = InformationDirectedSampling(samples=250)
        assert policy.samples_needed == 250


class TestAgentIntegration:
    """End-to-end pulls and updates through each agent type."""

    def test_agent_pull_update_cycle(self):
        arms = [Arm(i, learner=NormalInverseGammaRegressor()) for i in range(3)]
        agent = Agent(arms, InformationDirectedSampling(samples=200), random_seed=0)
        for reward in [1.0, 0.5, 2.0, 1.5]:
            (token,) = agent.pull()
            assert token in {0, 1, 2}
            agent.update(np.array([reward]))

        slates = agent.pull(top_k=2)
        assert len(slates[0]) == 2

    @pytest.mark.parametrize("sparse", [False, True])
    def test_contextual_agent_pull_update_cycle(self, sparse: bool):
        arms = [
            Arm(i, learner=NormalRegressor(alpha=1.0, beta=1.0, sparse=sparse))
            for i in range(3)
        ]
        agent = ContextualAgent(
            arms, InformationDirectedSampling(samples=200), random_seed=0
        )
        X = np.array([[1.0, 0.5], [0.2, 1.0]])
        tokens = agent.pull(X[:1])
        assert len(tokens) == 1
        agent.update(X[:1], np.array([1.0]))

        slates = agent.pull(X, top_k=2)
        assert len(slates) == 2 and all(len(s) == 2 for s in slates)

    def test_lipschitz_agent_shared_learner(self):
        # Shared learner: joint sampling path (consumes is not MARGINAL_ONLY).
        arms = [Arm(i, reward_function=None, learner=None) for i in range(4)]
        agent = LipschitzContextualAgent(
            arms=arms,
            policy=InformationDirectedSampling(samples=200),
            arm_featurizer=ArmColumnFeaturizer(column_name="product_id"),
            learner=NormalInverseGammaRegressor(),
            random_seed=0,
        )
        X = np.array([[1.0], [2.0], [0.5]])
        tokens = agent.pull(X)
        assert len(tokens) == 3
        agent.update(X, np.array([1.0, 0.0, 2.0]))

        slates = agent.pull(X, top_k=2)
        assert len(slates) == 3 and all(len(s) == 2 for s in slates)

    def test_concentrates_on_best_arm(self):
        # After clearly separated observations, IDS should exploit.
        arms = [Arm(i, learner=NormalInverseGammaRegressor()) for i in range(2)]
        agent = Agent(arms, InformationDirectedSampling(samples=500), random_seed=42)
        rng = np.random.default_rng(0)
        for _ in range(60):
            (token,) = agent.pull()
            reward = rng.normal(loc=2.0 if token == 1 else 0.0, scale=0.1)
            agent.update(np.array([reward]))

        pulls = [agent.pull()[0] for _ in range(20)]
        assert np.mean([t == 1 for t in pulls]) > 0.9
