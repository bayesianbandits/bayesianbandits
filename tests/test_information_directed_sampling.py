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
from bayesianbandits._blas_helpers import draw_contiguous
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

    def test_ragged_frontiers_are_solved_per_context(self):
        """One context with a large frontier next to one already resolved:
        each must get its own optimum (the ragged pair list must not mix
        pairs across contexts)."""
        n_arms = 8
        # context 0: everything on the frontier (delta falls as v rises);
        # context 1: arm 0 dominates outright
        delta = np.empty((n_arms, 2))
        v = np.empty((n_arms, 2))
        delta[:, 0] = np.linspace(1.0, 0.1, n_arms)
        v[:, 0] = np.linspace(0.1, 1.0, n_arms)
        delta[:, 1] = np.linspace(0.5, 1.2, n_arms)
        v[:, 1] = np.linspace(1.0, 0.2, n_arms)
        alive = np.ones((n_arms, 2), dtype=bool)
        a_idx, b_idx, q, ratio = _optimal_two_point(delta, v, alive)
        for c in range(2):
            reference = pairwise_grid_min_ratio(delta[:, c], v[:, c])
            assert ratio[c] <= reference + 1e-12
            assert reference - ratio[c] <= 1e-4 * max(1.0, reference)


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

    def test_never_optimal_arms_carry_no_conditioning_weight(self):
        """v sums over p(A* = b), which is zero for arms that never win a
        draw; the implementation drops those columns from the conditioning
        entirely. Statistics must match the unrestricted formula."""
        rng = np.random.default_rng(8)
        n_arms, n_draws = 40, 500
        # two contenders and 38 arms that never win
        samples = rng.normal(0.0, 0.1, size=(n_arms, 2, n_draws))
        samples[0] += 5.0
        samples[1] += 5.0
        alive = np.ones((n_arms, 2), dtype=bool)
        delta, v = _vids_statistics(samples, alive)

        # unrestricted reference: full one-hot conditioning
        masked = samples
        argmax_idx = masked.argmax(axis=0)
        onehot = (argmax_idx[:, :, None] == np.arange(n_arms)).astype(np.float64)
        counts = onehot.sum(axis=1).T
        p_opt = counts / n_draws
        cond = np.matmul(samples.transpose(1, 0, 2), onehot).transpose(1, 2, 0)
        mu_cond = cond / np.maximum(counts, 1.0)[None, :, :]
        means = samples.mean(axis=-1)
        v_ref = np.einsum("bc,abc->ac", p_opt, (mu_cond - means[:, None, :]) ** 2)
        np.testing.assert_allclose(v, v_ref, atol=1e-12)
        assert (counts > 0).sum(axis=0).max() <= 2  # the premise held

    def test_derived_means_keep_zero_regret_exact(self):
        """The means and E[max] are both recovered from the conditional-sum
        GEMM rather than by passes over the tensor. For an always-optimal
        arm the two recoveries read the same float, so its regret is
        *exactly* zero -- an ulp of regret with zero gain would flip its
        ratio from 0 (play it outright) to inf (posterior looks resolved).
        Sums whose draws don't reduce exactly under reordering probe this."""
        rng = np.random.default_rng(21)
        n_arms, n_draws = 7, 337
        samples = rng.normal(0.0, 1.0, size=(n_arms, 3, n_draws)) / 3.0
        samples[4] += 100.0  # arm 4 wins every draw in every context
        alive = np.ones((n_arms, 3), dtype=bool)
        delta, v = _vids_statistics(samples, alive)
        assert (delta[4] == 0.0).all()
        a_idx, b_idx, q, ratio = _optimal_two_point(delta, v, alive)
        assert (ratio == 0.0).all()

    def test_tied_optima_split_evenly(self):
        # Duplicate arms under a shared learner draw identical rewards, so
        # the argmax is a genuine tie. The optimal-arm indicator is built
        # by comparing against the max rather than by argmax, which marks
        # every tied arm: splitting the draw evenly across them keeps
        # p(A* = a) a probability and treats the twins symmetrically.
        twin = np.array([[[2.0, 0.0]], [[2.0, 0.0]], [[1.0, 1.0]]])
        alive = np.ones((3, 1), dtype=bool)
        delta, v = _vids_statistics(twin, alive)

        # rho* = mean(max) = 1.5 and mu = [1, 1, 1], so regret is uniform
        np.testing.assert_allclose(delta, [[0.5], [0.5], [0.5]])
        # each twin takes half of draw 0, so p = [1/4, 1/4, 1/2]. A twin's
        # conditional mean is 2 when either twin is optimal and 0 when
        # arm 2 is, a unit move either way; arm 2's is 1 throughout.
        np.testing.assert_allclose(v, [[1.0], [1.0], [0.0]])


class TestDrawContiguous:
    """The layout normalization that fronts ``select``."""

    @pytest.mark.parametrize("shape", [(7, 5, 61), (1, 1, 3), (4, 1, 130), (3, 9, 1)])
    def test_normalizes_a_draw_major_view(self, shape):
        n_arms, n_contexts, n_draws = shape
        drawn = np.random.default_rng(0).normal(size=(n_draws, n_arms, n_contexts))
        view = drawn.transpose(1, 2, 0)
        out = draw_contiguous(view)
        assert out.strides[-1] == out.itemsize
        np.testing.assert_array_equal(out, view)

    def test_contiguous_input_is_not_copied(self):
        samples = np.zeros((3, 2, 5))
        assert draw_contiguous(samples) is samples

    def test_draw_contiguous_view_is_not_copied(self):
        # The blocked reward-space route yields an (arm, context, draw)
        # view whose draw axis is contiguous but whose leading axes are
        # swapped in memory. That satisfies the layout contract as-is.
        drawn = np.zeros((4, 3, 11))  # (C, K, S) in memory
        view = drawn.transpose(1, 0, 2)
        assert not view.flags.c_contiguous
        assert draw_contiguous(view) is view

    def test_slabbing_covers_every_draw(self):
        # A shape whose slab size does not divide the draw count: the
        # final short slab must still be copied.
        from bayesianbandits import _blas_helpers

        drawn = np.random.default_rng(1).normal(size=(37, 4, 3))
        view = drawn.transpose(1, 2, 0)
        original = (_blas_helpers._COPY_SLAB_BYTES, _blas_helpers._MIN_SLAB_DRAWS)
        try:
            _blas_helpers._COPY_SLAB_BYTES = 4 * 3 * 8 * 5  # five draws per slab
            _blas_helpers._MIN_SLAB_DRAWS = 1
            out = _blas_helpers.draw_contiguous(view)
        finally:
            _blas_helpers._COPY_SLAB_BYTES, _blas_helpers._MIN_SLAB_DRAWS = original
        np.testing.assert_array_equal(out, view)


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

    def test_select_accepts_a_transposed_view(self):
        # The agent's draw tensor is a view, not a C-ordered array.
        drawn = np.random.default_rng(9).normal(size=(400, 4, 3))
        view = drawn.transpose(1, 2, 0)
        arms = self.make_arms(4)
        policy = InformationDirectedSampling()

        from_view = policy.select(view, arms, np.random.default_rng(2))
        from_copy = policy.select(
            np.ascontiguousarray(view), arms, np.random.default_rng(2)
        )
        assert from_view == from_copy

    def test_select_does_not_mutate_its_input(self):
        samples = np.random.default_rng(6).normal(size=(4, 2, 150))
        before = samples.copy()
        policy = InformationDirectedSampling()
        policy.select(samples, self.make_arms(4), np.random.default_rng(0), top_k=3)
        np.testing.assert_array_equal(samples, before)


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
