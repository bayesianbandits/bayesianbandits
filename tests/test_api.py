from typing import Any, List, Type, TypeVar, Union, cast

import numpy as np
import pytest
from numpy.typing import NDArray
from scipy.sparse import csc_array
from sklearn.base import (
    check_is_fitted,  # type: ignore
    clone,
)

from bayesianbandits import (
    Agent,
    Arm,
    ContextualAgent,
    DirichletClassifier,
    EpsilonGreedy,
    GammaRegressor,
    NormalInverseGammaRegressor,
    NormalRegressor,
    ThompsonSampling,
    UpperConfidenceBound,
)
from bayesianbandits._arm import Learner
from bayesianbandits.api import PolicyProtocol
from bayesianbandits.policies._base import _top_k_indices


@pytest.fixture(
    params=[
        DirichletClassifier({1: 1.0, 2: 1.0}),
        GammaRegressor(alpha=1, beta=1),
        NormalRegressor(alpha=1, beta=1),
        NormalRegressor(alpha=1, beta=1, sparse=True),
        NormalInverseGammaRegressor(),
        NormalInverseGammaRegressor(sparse=True),
    ],
    ids=[
        "dirichlet",
        "gamma",
        "normal",
        "normal sparse",
        "normal-inverse-gamma",
        "normal-inverse-gamma sparse",
    ],
)
def learner_class(
    request: pytest.FixtureRequest,
) -> Union[
    DirichletClassifier, GammaRegressor, NormalRegressor, NormalInverseGammaRegressor
]:
    return request.param


policies: List[PolicyProtocol[Any, Any]] = [
    EpsilonGreedy(0.8),
    ThompsonSampling(),
    UpperConfidenceBound(0.68),
]


@pytest.fixture(
    params=policies,
)
def choice(
    request: pytest.FixtureRequest,
) -> PolicyProtocol[Any, Any]:
    return request.param


LT = TypeVar("LT", bound=Learner[Any])


@pytest.fixture(params=[Agent, ContextualAgent])
def bandit_instance(
    request: pytest.FixtureRequest,
    learner_class: Union[
        DirichletClassifier,
        GammaRegressor,
        NormalRegressor,
        NormalInverseGammaRegressor,
    ],
    choice: PolicyProtocol[NDArray[np.float64], int],
) -> Union[
    Agent[int],
    ContextualAgent[NDArray[np.float64], int],
]:
    if isinstance(learner_class, DirichletClassifier):

        def reward_func(
            x: NDArray[np.float64],
        ) -> NDArray[np.float64]:
            return x[..., 0]

    else:
        reward_func = None  # type: ignore

    arms = [
        Arm(0, reward_func, learner=clone(learner_class)),
        Arm(1, reward_func, learner=clone(learner_class)),
    ]

    agent = request.param(arms, choice, random_seed=0)

    return agent


class TestBandits:
    def test_pull(
        self,
        bandit_instance: Union[
            Agent[int],
            ContextualAgent[NDArray[np.float64], int],
        ],
    ) -> None:
        if isinstance(bandit_instance, Agent):
            (token,) = bandit_instance.pull()

        else:
            (token,) = bandit_instance.pull(np.array([[2.0]]))

        assert bandit_instance.arm_to_update.action_token == token

    def test_batch_pull(
        self,
        bandit_instance: Union[
            Agent[int],
            ContextualAgent[NDArray[np.float64], int],
        ],
    ) -> None:
        if isinstance(bandit_instance, Agent):
            pytest.skip("Batch pull is not supported for non-contextual bandits")

        # if isinstance(bandit_instance.arm(0).learner, DirichletClassifier):
        #     pytest.xfail("DirichletClassifier does not support batch pull")

        (_, _, token3) = bandit_instance.pull(np.array([[2.0], [2.0], [2.0]]))

        assert bandit_instance.arm_to_update.action_token == token3

    def test_update(
        self,
        bandit_instance: Union[
            Agent[int],
            ContextualAgent[NDArray[np.float64], int],
        ],
    ) -> None:
        if isinstance(bandit_instance, Agent):
            bandit_instance.pull()
            bandit_instance.update(np.array([1.0]))
        else:
            bandit_instance.pull(np.array([[2.0]]))
            bandit_instance.update(np.array([[2.0]]), np.array([1.0]))

        # check that the learner was updated with the correct reward
        assert bandit_instance.arm_to_update is not None
        assert check_is_fitted(bandit_instance.arm_to_update.learner) is None  # type: ignore

    def test_decay(
        self,
        bandit_instance: Union[
            Agent[int],
            ContextualAgent[NDArray[np.float64], int],
        ],
    ) -> None:
        if isinstance(bandit_instance, Agent):
            bandit_instance.decay(decay_rate=0.5)
        else:
            bandit_instance.decay(decay_rate=0.5)

        # Check that no exception is raised

    def test_arm(
        self,
        bandit_instance: Union[
            Agent[int],
            ContextualAgent[NDArray[np.float64], int],
        ],
    ) -> None:
        arm = bandit_instance.arm(0)
        assert arm is bandit_instance.arms[0]

        with pytest.raises(KeyError):
            bandit_instance.arm(3)

    def test_select_for_update(
        self,
        bandit_instance: Union[
            Agent[int],
            ContextualAgent[NDArray[np.float64], int],
        ],
    ) -> None:
        bandit_instance.select_for_update(1)
        assert bandit_instance.arm_to_update is bandit_instance.arms[1]

        with pytest.raises(KeyError):
            bandit_instance.select_for_update(3)

    def test_add_arm(
        self,
        bandit_instance: Union[
            Agent[int],
            ContextualAgent[NDArray[np.float64], int],
        ],
    ) -> None:
        arm_to_add = Arm(2, None, learner=clone(bandit_instance.arms[0].learner))  # type: ignore
        bandit_instance.add_arm(arm_to_add)
        assert len(bandit_instance.arms) == 3

        with pytest.raises(ValueError):
            bandit_instance.add_arm(arm_to_add)

    def test_remove_arm(
        self,
        bandit_instance: Union[
            Agent[int],
            ContextualAgent[NDArray[np.float64], int],
        ],
    ) -> None:
        bandit_instance.remove_arm(0)
        assert len(bandit_instance.arms) == 1

        with pytest.raises(KeyError):
            bandit_instance.remove_arm(0)

    def test_change_policy(
        self,
        bandit_instance: Union[
            Agent[int],
            ContextualAgent[NDArray[np.float64], int],
        ],
    ) -> None:
        bandit_instance.policy = ThompsonSampling()

        assert isinstance(bandit_instance.policy, ThompsonSampling)

    def test_check_rng(
        self,
        bandit_instance: Union[
            Agent[int],
            ContextualAgent[NDArray[np.float64], int],
        ],
    ) -> None:
        assert isinstance(bandit_instance.rng, np.random.Generator)

    def test_constructor_exceptions(
        self,
    ):
        with pytest.raises(ValueError):
            Agent([], EpsilonGreedy())  # type: ignore

        with pytest.raises(ValueError):
            Agent(
                [
                    Arm(0, None, learner=NormalInverseGammaRegressor()),
                    Arm(0, None, learner=NormalInverseGammaRegressor()),
                ],
                EpsilonGreedy(),
            )


def test_contextual_agent_update_mismatched_shapes() -> None:
    with pytest.raises(
        ValueError, match="Found input variables with inconsistent numbers of samples: "
    ):
        ContextualAgent(
            [
                Arm(0, None, learner=NormalInverseGammaRegressor()),
                Arm(1, None, learner=NormalInverseGammaRegressor()),
            ],
            EpsilonGreedy(),
            random_seed=0,
        ).update(np.array([[1.0]]), np.array([1.0, 2.0]))


class TestTopK:
    """Test top_k functionality for policies."""

    def test_epsilon_greedy_top_k_return_type(self) -> None:
        """Test that EpsilonGreedy returns correct types with top_k."""
        arms = [Arm(i, None, learner=NormalInverseGammaRegressor()) for i in range(5)]
        agent = ContextualAgent(arms, EpsilonGreedy(epsilon=0.5), random_seed=42)
        X = np.array([[1.0], [2.0]])

        # Default behavior - returns List[TokenType]
        result_default = agent.pull(X)
        assert isinstance(result_default, list)
        assert len(result_default) == 2  # One per context
        assert all(isinstance(token, int) for token in result_default)

        # top_k=1 - returns List[List[TokenType]]
        result_k1 = agent.pull(X, top_k=1)
        assert isinstance(result_k1, list)
        assert len(result_k1) == 2  # One list per context
        assert all(isinstance(sublist, list) for sublist in result_k1)
        assert all(len(sublist) == 1 for sublist in result_k1)

        # top_k=3 - returns List[List[TokenType]]
        result_k3 = agent.pull(X, top_k=3)
        assert isinstance(result_k3, list)
        assert len(result_k3) == 2  # One list per context
        assert all(isinstance(sublist, list) for sublist in result_k3)
        assert all(len(sublist) == 3 for sublist in result_k3)

    def test_epsilon_greedy_no_duplicates(self) -> None:
        """Test that top_k doesn't select the same arm twice."""
        arms = [Arm(i, None, learner=NormalInverseGammaRegressor()) for i in range(10)]
        agent = ContextualAgent(arms, EpsilonGreedy(epsilon=0.5), random_seed=42)
        X = np.array([[1.0]] * 20)  # 20 contexts

        results = agent.pull(X, top_k=5)

        # Check no duplicates in any selection
        for result in results:
            assert len(result) == len(set(result))  # No duplicates
            assert len(result) == 5  # Correct number selected

    def test_epsilon_greedy_top_k_exceeds_arms(self) -> None:
        """Test behavior when top_k > number of arms."""
        arms = [Arm(i, None, learner=NormalInverseGammaRegressor()) for i in range(3)]
        agent = ContextualAgent(arms, EpsilonGreedy(epsilon=0.5), random_seed=42)
        X = np.array([[1.0]])

        # Request more arms than available
        results = agent.pull(X, top_k=5)

        # Should return all 3 arms
        assert len(results[0]) == 3
        assert set(results[0]) == {0, 1, 2}

    def test_epsilon_greedy_postprocess_with_top_k(self) -> None:
        """Test that epsilon-greedy correctly explores with top_k selection."""
        policy: PolicyProtocol[NDArray[np.float64] | csc_array, int] = EpsilonGreedy(
            epsilon=1.0
        )  # Always explore

        # Create arms for testing
        arms = [Arm(i, learner=NormalInverseGammaRegressor()) for i in range(4)]

        # Mock samples that would produce means like the original test
        # Shape: (n_arms, n_contexts, samples_needed)
        samples = np.array(
            [
                [[1.0], [2.0], [3.0]],  # Arm 0 samples across 3 contexts
                [[4.0], [5.0], [6.0]],  # Arm 1
                [[7.0], [8.0], [9.0]],  # Arm 2
                [[10.0], [11.0], [12.0]],  # Arm 3
            ]
        ).astype(np.float64)

        rng = np.random.default_rng(42)

        # Test that with epsilon=1.0 and top_k=2, we get exploration
        # We can't directly test the internal inf-setting, but we can test
        # that the selection behavior is different from greedy
        selected_arms = policy.select(samples, arms, rng, top_k=2)

        # Should return 2 arms per context
        assert len(selected_arms) == 3  # 3 contexts
        assert all(len(context_arms) == 2 for context_arms in selected_arms)

    @pytest.mark.parametrize(
        "policy",
        [ThompsonSampling(), UpperConfidenceBound(), EpsilonGreedy(epsilon=0.0)],
        ids=repr,
    )
    @pytest.mark.parametrize("top_k", [3, 5, 8])
    def test_top_k_slates_are_best_first(self, policy: Any, top_k: int) -> None:
        """Slots run best score first, ties to the lower arm index as
        ``argmax`` breaks them, so slot 0 is the ``top_k=None`` pick; past
        the arm count the slate is every arm in that order."""
        scores = np.array(
            [
                [1.0, 5.0, 0.0],
                [3.0, 4.0, 0.0],
                [3.0, 3.0, 0.0],
                [0.0, 2.0, 0.0],
                [2.0, 1.0, 0.0],
            ]
        )
        # constant draws: the mean, quantile and draw all equal the score
        samples = np.repeat(scores[:, :, None], 4, axis=2)
        arms = [Arm(i, learner=NormalInverseGammaRegressor()) for i in range(5)]
        rng = np.random.default_rng(0)

        slates = policy.select(samples, arms, rng, top_k=top_k)
        best = policy.select(samples, arms, rng)

        order = [[1, 2, 4, 0, 3], [0, 1, 2, 3, 4], [0, 1, 2, 3, 4]]
        assert [[a.action_token for a in s] for s in slates] == [
            o[:top_k] for o in order
        ]
        assert [s[0] for s in slates] == best

    @pytest.mark.parametrize(
        "policy",
        [ThompsonSampling(), UpperConfidenceBound(), EpsilonGreedy(epsilon=0.0)],
        ids=repr,
    )
    @pytest.mark.parametrize("n_arms", [40, 300])  # one sort; partition first
    def test_top_k_ties_follow_arm_index(self, policy: Any, n_arms: int) -> None:
        """On many tied scores (where ``argpartition`` left ties in any
        order) slots stay best first with ties by arm index."""
        scores = np.round(np.random.default_rng(3).normal(size=(n_arms, 6)), 1)
        samples = np.repeat(scores[:, :, None], 4, axis=2)
        arms = [Arm(i, learner=NormalInverseGammaRegressor()) for i in range(n_arms)]

        slates = policy.select(samples, arms, np.random.default_rng(0), top_k=10)

        for c, slate in enumerate(slates):
            idx = np.array([a.action_token for a in slate])
            v = scores[idx, c]
            assert (v[:-1] >= v[1:]).all()
            assert ((v[:-1] > v[1:]) | (idx[:-1] < idx[1:])).all()
            assert idx[0] == scores[:, c].argmax()

    def test_epsilon_greedy_slot_0_is_the_greedy_pick_on_near_ties(self) -> None:
        """Arms whose draws are permutations of each other have equal means
        up to rounding; both paths round the same way, so slot 0 agrees."""
        rng = np.random.default_rng(1)
        draws = rng.normal(size=(200, 7))
        perm = rng.permuted(draws, axis=1)
        samples = np.stack([draws, perm, draws - 1.0])
        arms = [Arm(i, learner=NormalInverseGammaRegressor()) for i in range(3)]
        policy: Any = EpsilonGreedy(epsilon=0.0, samples=7)

        slates = policy.select(samples, arms, np.random.default_rng(0), top_k=2)
        best = policy.select(samples, arms, np.random.default_rng(0))

        assert [s[0] for s in slates] == best

    @pytest.mark.parametrize("n_arms", [100, 200])  # one sort; partition first
    def test_top_k_ranks_nan_last(self, n_arms: int) -> None:
        """With fewer finite scores than ``top_k``, both routes put the finite
        arms first, then the NaN arms by index."""
        scores = np.full((n_arms, 1), np.nan)
        scores[50:55, 0] = [1.0, 2.0, 3.0, 4.0, 5.0]

        slate = _top_k_indices(scores, 10)

        assert slate.tolist() == [[54, 53, 52, 51, 50, 0, 1, 2, 3, 4]]

    def test_epsilon_greedy_explores_in_the_order_drawn(self) -> None:
        """An exploring context's slate is its random arms in draw order, so
        every slot is uniform; the generator is consumed as before."""
        samples = np.arange(18, dtype=np.float64).reshape(6, 3, 1)
        arms = [Arm(i, learner=NormalInverseGammaRegressor()) for i in range(6)]
        policy: Any = EpsilonGreedy(epsilon=1.0)

        slates = policy.select(samples, arms, np.random.default_rng(7), top_k=4)

        twin = np.random.default_rng(7)
        twin.random(size=3)
        drawn = [list(twin.choice(6, size=4, replace=False)) for _ in range(3)]
        assert [[a.action_token for a in s] for s in slates] == drawn

    def test_agent_top_k(self) -> None:
        """Test top_k functionality for non-contextual Agent."""
        arms = [Arm(i, None, learner=NormalInverseGammaRegressor()) for i in range(5)]
        agent = Agent(arms, EpsilonGreedy(epsilon=0.5), random_seed=42)

        # Default behavior
        result_default = agent.pull()
        assert isinstance(result_default, list)
        assert len(result_default) == 1

        # top_k behavior
        result_k3 = agent.pull(top_k=3)
        assert isinstance(result_k3, list)
        assert len(result_k3) == 1  # Still wrapped in outer list
        assert isinstance(result_k3[0], list)
        assert len(result_k3[0]) == 3  # Inner list has 3 items

    @pytest.mark.parametrize("policy_class", [ThompsonSampling, UpperConfidenceBound])
    def test_other_policies_top_k(
        self,
        policy_class: (
            Type[ThompsonSampling[NDArray[np.float64], int]]
            | Type[UpperConfidenceBound[NDArray[np.float64], int]]
        ),
    ) -> None:
        """Test that other policies also support top_k."""
        arms = [Arm(i, None, learner=NormalInverseGammaRegressor()) for i in range(5)]

        if policy_class is UpperConfidenceBound:
            policy = cast(
                UpperConfidenceBound[NDArray[np.float64], int], policy_class(alpha=0.68)
            )
        elif policy_class is ThompsonSampling:
            policy = cast(ThompsonSampling[NDArray[np.float64], int], policy_class())

        agent = ContextualAgent(arms, policy, random_seed=42)  # type: ignore
        X = np.array([[1.0], [2.0]])

        # Should work without errors
        result = agent.pull(X, top_k=3)
        assert len(result) == 2  # One per context
        assert all(len(sublist) == 3 for sublist in result)
        assert all(len(set(sublist)) == 3 for sublist in result)  # No duplicates


class TestSampleWeight:
    """Test sample_weight parameter is correctly passed through the API."""

    def test_contextual_agent_sample_weight_passthrough(self) -> None:
        """Test that ContextualAgent passes sample_weight to policy.update()."""
        arms = [
            Arm(0, None, learner=NormalInverseGammaRegressor()),
            Arm(1, None, learner=NormalInverseGammaRegressor()),
        ]
        agent = ContextualAgent(arms, ThompsonSampling(), random_seed=42)

        X = np.array([[1.0], [2.0]])
        y = np.array([1.0, 2.0])
        weights = np.array([0.5, 2.0])

        # Should not raise any errors
        agent.pull(X)
        agent.update(X, y, sample_weight=weights)

        # Also test with None (default)
        agent.update(X, y, sample_weight=None)
        agent.update(X, y)  # No sample_weight parameter

    def test_agent_sample_weight_passthrough(self) -> None:
        """Test that Agent passes sample_weight through to ContextualAgent."""
        arms = [
            Arm(0, None, learner=NormalInverseGammaRegressor()),
            Arm(1, None, learner=NormalInverseGammaRegressor()),
        ]
        agent = Agent(arms, EpsilonGreedy(epsilon=0.1), random_seed=42)

        y = np.array([1.0])
        weights = np.array([0.5])

        # Should not raise any errors
        agent.pull()
        agent.update(y, sample_weight=weights)

        # Also test with None (default)
        agent.update(y, sample_weight=None)
        agent.update(y)  # No sample_weight parameter

    def test_sample_weight_with_select_for_update_chain(self) -> None:
        """Test the position-biased feedback example from the issue."""
        arms = [
            Arm("job_1", None, learner=NormalInverseGammaRegressor()),
            Arm("job_2", None, learner=NormalInverseGammaRegressor()),
            Arm("job_3", None, learner=NormalInverseGammaRegressor()),
        ]
        agent = ContextualAgent(arms, EpsilonGreedy(epsilon=0.2), random_seed=42)

        X = np.array([[1.0, 2.0]])
        recommendations = agent.pull(X, top_k=3)
        clicked_jobs = {recommendations[0][0]}

        # This pattern from the issue should now work
        for rank, job_id in enumerate(recommendations[0]):
            weight = 1.0 if job_id in clicked_jobs else 0.1 / (1 + rank)

            agent.select_for_update(job_id).update(
                X,
                np.array([float(job_id in clicked_jobs)]),
                sample_weight=np.array([weight]),
            )
