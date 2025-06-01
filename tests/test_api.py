from typing import Any, TypeVar, Union

import numpy as np
import pytest
from numpy.typing import NDArray
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


@pytest.fixture(
    params=[
        EpsilonGreedy(0.8),
        ThompsonSampling(),
        UpperConfidenceBound(0.68),
    ]
)
def choice(
    request: pytest.FixtureRequest,
) -> PolicyProtocol[Any, Any]:
    return request.param


LT = TypeVar("LT", bound=Learner)


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
        ) -> Union[NDArray[np.float64], np.float64]:
            return x[..., 0].T

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
            bandit_instance.decay()
        else:
            bandit_instance.decay(np.array([[2.0]]), decay_rate=0.5)

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
        """Test that postprocess correctly sets multiple values to inf for exploration."""
        policy = EpsilonGreedy(epsilon=1.0)  # Always explore

        # Mock arm summary (means)
        arm_summary = np.array(
            [
                [1.0, 2.0, 3.0],  # Arm 0 means across 3 contexts
                [4.0, 5.0, 6.0],  # Arm 1
                [7.0, 8.0, 9.0],  # Arm 2
                [10.0, 11.0, 12.0],  # Arm 3
            ]
        )

        rng = np.random.default_rng(42)
        processed = policy.postprocess(arm_summary.copy(), rng, top_k=2)

        # With epsilon=1.0, all contexts should explore
        # Check that exactly 2 values per column are set to inf
        for col in range(processed.shape[1]):
            inf_count = np.sum(np.isinf(processed[:, col]))
            assert inf_count == 2

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
    def test_other_policies_top_k(self, policy_class) -> None:
        """Test that other policies also support top_k."""
        arms = [Arm(i, None, learner=NormalInverseGammaRegressor()) for i in range(5)]

        if policy_class == UpperConfidenceBound:
            policy = policy_class(alpha=0.68)
        else:
            policy = policy_class()

        agent = ContextualAgent(arms, policy, random_seed=42)
        X = np.array([[1.0], [2.0]])

        # Should work without errors
        result = agent.pull(X, top_k=3)
        assert len(result) == 2  # One per context
        assert all(len(sublist) == 3 for sublist in result)
        assert all(len(set(sublist)) == 3 for sublist in result)  # No duplicates
