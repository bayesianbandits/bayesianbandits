"""Tests for ``select_with_probabilities``: the distribution a policy drew
its decision from, conditional on the samples it was handed."""

import pickle
from typing import Any, Callable, Dict, List, Sequence, Sized, cast

import numpy as np
import pytest
import scipy.sparse as sp
from numpy.typing import NDArray
from sklearn.preprocessing import FunctionTransformer

from bayesianbandits import (
    EXP3A,
    Agent,
    AgentPipeline,
    Arm,
    ContextualAgent,
    EpsilonGreedy,
    GammaRegressor,
    InformationDirectedSampling,
    LipschitzContextualAgent,
    NormalInverseGammaRegressor,
    NormalRegressor,
    ThompsonSampling,
    UpperConfidenceBound,
)
from bayesianbandits._draw_kind import DrawKind
from bayesianbandits.featurizers import FunctionArmFeaturizer
from bayesianbandits.pipelines import LearnerPipeline
from bayesianbandits.policies._base import PolicyDefaultUpdate

# Policies putting mass on every arm, which IPS of a fixed target needs
FULL_SUPPORT: List[Any] = [
    EXP3A(eta=1.0, gamma=0.1, samples=8),
    EpsilonGreedy(epsilon=0.3, samples=8),
]
POLICIES = FULL_SUPPORT + [
    UpperConfidenceBound(alpha=0.68, samples=8),
    InformationDirectedSampling(samples=8),
]


def _arms(n_arms: int) -> List[Arm[Any, int]]:
    return [Arm(i, learner=None) for i in range(n_arms)]


def _tokens(arms: List[Arm[Any, int]]) -> List[int]:
    return [arm.action_token for arm in arms]


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
def test_calibrated(policy: Any) -> None:
    """Each arm's play count matches the sum of its reported probabilities."""
    rng = np.random.default_rng(3)
    n_rounds = 20_000
    means = np.array([1.0, 0.0, -1.0])
    samples = means[:, None, None] + rng.normal(
        size=(3, n_rounds, policy.samples_needed)
    )

    chosen, probs = policy.select_with_probabilities(samples, _arms(3), rng)
    played = np.array(_tokens(chosen))

    assert np.allclose(probs.sum(axis=1), 1.0)
    for arm in range(3):
        residual = np.sum(played == arm) - probs[:, arm].sum()
        scale = np.sqrt(np.sum(probs[:, arm] * (1 - probs[:, arm])))
        assert abs(residual) <= 4 * scale  # exact when the policy is deterministic


@pytest.mark.parametrize("policy", FULL_SUPPORT, ids=repr)
def test_ips_is_unbiased(policy: Any) -> None:
    """The target always plays the logger's least favourite arm, so the
    importance weights are large and a wrong propensity shows up as bias."""
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

    assert probs.shape == (n_rounds, 3)
    assert np.allclose(probs.sum(axis=1), 1.0)

    target = 2
    terms = (played == target) * rewards / probs[np.arange(n_rounds), played]
    estimate = terms.mean()
    stderr = terms.std(ddof=1) / np.sqrt(n_rounds)

    assert abs(estimate - means[target]) < 4 * stderr


def _learned_arms() -> List[Arm[Any, int]]:
    return [Arm(i, learner=NormalRegressor(alpha=1.0, beta=1.0)) for i in range(3)]


def _one_hot(X: Sized, action_tokens: Sequence[int]) -> NDArray[np.float64]:
    """Three arm indicators and nothing else: independent arm scores."""
    out = np.zeros((len(X), 3, len(action_tokens)))
    out[:, list(action_tokens), np.arange(len(action_tokens))] = 1.0
    return out


AGENTS: Dict[str, Callable[[Any, int], Any]] = {
    "Agent": lambda policy, seed: Agent(_learned_arms(), policy, random_seed=seed),
    "ContextualAgent": lambda policy, seed: ContextualAgent(
        _learned_arms(), policy, random_seed=seed
    ),
    "AgentPipeline": lambda policy, seed: AgentPipeline(
        [("double", FunctionTransformer(lambda x: 2 * x))],
        ContextualAgent(_learned_arms(), policy, random_seed=seed),
    ),
    "LipschitzContextualAgent": lambda policy, seed: LipschitzContextualAgent(
        _arms(3),
        policy,
        arm_featurizer=FunctionArmFeaturizer(_one_hot),
        learner=NormalRegressor(alpha=1.0, beta=1.0),
        random_seed=seed,
    ),
}


@pytest.mark.parametrize("build", AGENTS.values(), ids=AGENTS.keys())
def test_agents_agree_with_pull(build: Callable[[Any, int], Any]) -> None:
    policy = EpsilonGreedy(epsilon=0.3, samples=8)
    X = () if build is AGENTS["Agent"] else (np.ones((5, 2)),)

    plain = build(policy, 1).pull(*X)
    agent = build(policy, 1)
    tokens, probs = agent.pull_with_probabilities(*X)

    assert tokens == plain
    assert probs.shape == (len(tokens), 3)
    assert np.allclose(probs.sum(axis=1), 1.0)
    assert (probs[np.arange(len(tokens)), tokens] > 0).all()
    assert agent.arm_to_update.action_token == tokens[-1]


def test_thompson_sampling_select_raises() -> None:
    """The samples alone do not determine TS's propensity."""
    samples = np.random.default_rng(0).normal(size=(3, 2, 1))
    with pytest.raises(NotImplementedError, match="largest"):
        ThompsonSampling().select_with_probabilities(
            samples, _arms(3), np.random.default_rng(0)
        )


def _continuous(X: Sized, action_tokens: Sequence[int]) -> NDArray[np.float64]:
    """Two arm features and no arm-specific coefficients: the scores'
    covariance has rank two, below the number of arm differences."""
    out = np.zeros((len(X), 2, len(action_tokens)))
    for j, token in enumerate(action_tokens):
        out[:, 0, j] = 1.0 + token
        out[:, 1, j] = (token - 1.5) ** 2
    return out


def _ts() -> ThompsonSampling[Any, int]:
    return ThompsonSampling(propensity_samples=100_000)


THOMPSON_AGENTS: Dict[str, Callable[[], Any]] = {
    "ContextualAgent": lambda: ContextualAgent(_learned_arms(), _ts(), random_seed=5),
    "ContextualAgent-NIG": lambda: ContextualAgent(
        [Arm(i, learner=NormalInverseGammaRegressor()) for i in range(3)],
        _ts(),
        random_seed=5,
    ),
    "ContextualAgent-exp-reward": lambda: ContextualAgent(
        [
            Arm(i, learner=NormalRegressor(alpha=1.0, beta=1.0), reward_function=np.exp)
            for i in range(3)
        ],
        _ts(),
        random_seed=5,
    ),
    "ContextualAgent-Gamma": lambda: ContextualAgent(
        [
            # one group: GammaRegressor keys its posterior on a single column
            Arm(
                i,
                learner=LearnerPipeline(
                    steps=[
                        ("one", FunctionTransformer(lambda x: np.ones((len(x), 1))))
                    ],
                    learner=GammaRegressor(alpha=1.0, beta=1.0),
                ),
            )
            for i in range(3)
        ],
        _ts(),
        random_seed=5,
    ),
    "Lipschitz-one-hot": lambda: LipschitzContextualAgent(
        _arms(3),
        _ts(),
        arm_featurizer=FunctionArmFeaturizer(_one_hot),
        learner=NormalRegressor(alpha=1.0, beta=1.0),
        random_seed=5,
    ),
    "Lipschitz-continuous": lambda: LipschitzContextualAgent(
        _arms(4),
        _ts(),
        arm_featurizer=FunctionArmFeaturizer(_continuous),
        learner=NormalInverseGammaRegressor(),
        random_seed=5,
    ),
    "Lipschitz-batch-reward": lambda: LipschitzContextualAgent(
        _arms(3),
        _ts(),
        arm_featurizer=FunctionArmFeaturizer(_one_hot),
        learner=NormalRegressor(alpha=1.0, beta=1.0),
        batch_reward_function=lambda samples, tokens: samples,
        random_seed=5,
    ),
}


def _fit(agent: Any, X: NDArray[np.float64], rng: np.random.Generator) -> None:
    """A few positive rewards per arm, so the posteriors differ and overlap."""
    for arm in agent.arms:
        y = np.abs(rng.normal(loc=1 + arm.action_token / 4, size=len(X)))
        agent.select_for_update(arm.action_token).update(X, y)


def _argmax_frequencies(agent: Any, X: NDArray[np.float64], n_draws: int) -> Any:
    """How often each arm's draw is the largest, from the agent's own
    sampling path: ``(n_contexts, n_arms)``."""
    if isinstance(agent, LipschitzContextualAgent):
        samples = agent._sample_rewards(X, n_draws)
    else:
        samples = agent.policy._draw_samples(agent.arms, X, n_draws)
    picks = samples.argmax(axis=0)  # (n_contexts, n_draws)
    return np.stack(
        [np.bincount(row, minlength=len(agent.arms)) / n_draws for row in picks]
    )


@pytest.mark.parametrize("build", THOMPSON_AGENTS.values(), ids=THOMPSON_AGENTS.keys())
def test_thompson_sampling_probabilities(build: Callable[[], Any]) -> None:
    """The reported probabilities are the frequencies with which the
    agent's draws pick each arm, and the decision agrees with ``pull``."""
    rng = np.random.default_rng(0)
    X_fit = np.abs(rng.normal(size=(6, 2)))
    agent = build()
    _fit(agent, X_fit, rng)
    X = X_fit[:2]

    tokens, probs = agent.pull_with_probabilities(X)
    frequencies = _argmax_frequencies(agent, X, 200_000)

    assert probs.shape == (2, len(agent.arms))
    assert np.allclose(probs.sum(axis=1), 1.0)
    played = [_tokens(agent.arms).index(t) for t in tokens]
    assert (probs[[0, 1], played] >= 1 / 100_001).all()
    assert np.abs(probs - frequencies).max() < 6e-3
    assert agent.arm_to_update.action_token == tokens[-1]

    plain, with_probs = build(), build()
    _fit(plain, X_fit, np.random.default_rng(1))
    _fit(with_probs, X_fit, np.random.default_rng(1))
    assert plain.pull(X) == with_probs.pull_with_probabilities(X)[0]


@pytest.mark.parametrize(
    "split, calls",
    [
        (True, [(0, 2), (0, 2), (0, 1), (1, 2), (1, 2), (1, 1)]),
        (False, [(0, 1)] * 5),
    ],
)
def test_drawn_probabilities_count_in_chunks(
    monkeypatch: pytest.MonkeyPatch, split: bool, calls: List[Any]
) -> None:
    """Every draw counts once, the decision's included, whether contexts
    take their draws one at a time or all together in chunks."""
    from bayesianbandits.policies import _thompson_sampling

    monkeypatch.setattr(_thompson_sampling, "_PROPENSITY_CHUNK_ELEMS", 4)
    seen: List[Any] = []

    def draw(contexts: slice, size: int) -> NDArray[np.float64]:
        # two arms, two contexts: arm 1 wins context 0, arm 0 context 1
        seen.append((contexts.start, size))
        out = np.zeros((2, 2, size))
        out[1, 0] = out[0, 1] = 1.0
        return out[:, contexts]

    decision = np.array([[[1.0], [0.0]], [[0.0], [1.0]]])  # arm 0, then arm 1
    arms = _arms(2)
    chosen, probs = ThompsonSampling(
        propensity_samples=5
    ).select_with_drawn_probabilities(
        decision, arms, np.random.default_rng(0), draw, split_contexts=split
    )

    assert chosen == [arms[0], arms[1]]
    assert seen == calls
    assert np.allclose(probs, [[1 / 6, 5 / 6], [5 / 6, 1 / 6]])


@pytest.mark.parametrize("split", [True, False])
def test_drawn_probabilities_of_no_contexts(split: bool) -> None:
    draw: Any = lambda contexts, size: pytest.fail("nothing to draw")  # noqa: E731
    chosen, probs = ThompsonSampling().select_with_drawn_probabilities(
        np.zeros((3, 0, 1)), _arms(3), np.random.default_rng(0), draw, split
    )
    assert chosen == [] and probs.shape == (0, 3)


def test_policy_pickled_before_propensity_samples_loads() -> None:
    """The class default stands in for the attribute an old pickle lacks."""
    old = ThompsonSampling[Any, int].__new__(ThompsonSampling)
    loaded = pickle.loads(pickle.dumps(old))
    assert "propensity_samples" not in vars(loaded)

    agent = ContextualAgent(_learned_arms(), loaded, random_seed=0)
    assert "propensity_samples=10000" in repr(agent)
    _, probs = agent.pull_with_probabilities(np.ones((1, 2)))
    assert np.allclose(probs.sum(axis=1), 1.0)


@pytest.mark.parametrize("bad", [0, -1, 1.5, 1e4, True, "10"])
def test_propensity_samples_must_be_a_positive_integer(bad: Any) -> None:
    with pytest.raises(ValueError, match="positive integer"):
        ThompsonSampling(propensity_samples=bad)
    assert (
        ThompsonSampling(propensity_samples=cast(Any, np.int64(5))).propensity_samples
        == 5
    )


def test_sample_rewards_honours_an_explicit_marginal_request() -> None:
    """``MARGINAL_ONLY`` is 0, falsy; it must still override the policy."""
    agent = LipschitzContextualAgent(
        _arms(3),
        ThompsonSampling(),
        arm_featurizer=FunctionArmFeaturizer(_one_hot),
        learner=NormalRegressor(alpha=1.0, beta=1.0),
        random_seed=0,
    )
    learner: Any = agent.learner
    calls: List[int] = []
    marginal = learner.sample_marginal
    learner.sample_marginal = lambda X, size=1: calls.append(size) or marginal(X, size)

    agent._sample_rewards(np.ones((2, 2)), 4, DrawKind.MARGINAL_ONLY)
    assert calls == [4]


@pytest.mark.parametrize("fmt", ["coo", "csc"])
def test_thompson_sampling_probabilities_with_sparse_contexts(fmt: str) -> None:
    """Contexts are sliced through CSR, which COO cannot do itself."""
    arms = [
        Arm(i, learner=NormalRegressor(alpha=1.0, beta=1.0, sparse=True))
        for i in range(3)
    ]
    agent = ContextualAgent(
        arms, ThompsonSampling(propensity_samples=500), random_seed=0
    )
    rng = np.random.default_rng(0)
    X = sp.csc_array(rng.normal(size=(6, 4)) * (rng.random((6, 4)) < 0.5))
    for arm in agent.arms:
        agent.select_for_update(arm.action_token).update(X, rng.normal(size=6))

    tokens, probs = agent.pull_with_probabilities(X.asformat(fmt))
    assert len(tokens) == 6 and probs.shape == (6, 3)
    assert np.allclose(probs.sum(axis=1), 1.0)


def test_default_raises() -> None:
    with pytest.raises(NotImplementedError, match="PolicyDefaultUpdate does not"):
        PolicyDefaultUpdate[Any, int]().select_with_probabilities(
            np.zeros((2, 1, 1)), _arms(2), np.random.default_rng(0)
        )
