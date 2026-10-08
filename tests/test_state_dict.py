"""state_dict / load_state_dict: plain-data state for learners and agents."""

import json
from typing import Any, Callable

import numpy as np
import pytest
import scipy.sparse as sp
from sklearn.preprocessing import StandardScaler

from bayesianbandits import (
    Agent,
    AgentPipeline,
    Arm,
    ArmColumnFeaturizer,
    BayesianGLM,
    ContextualAgent,
    DirichletClassifier,
    EmpiricalBayesDirichletClassifier,
    EmpiricalBayesGammaRegressor,
    EmpiricalBayesGLM,
    EmpiricalBayesNormalRegressor,
    GammaRegressor,
    LearnerPipeline,
    LipschitzContextualAgent,
    NormalInverseGammaRegressor,
    NormalRegressor,
    StabilizedForgetting,
    ThompsonSampling,
)
from bayesianbandits._sparse_bayesian_linear_regression import SparseSolver

# A few IRLS steps per update is all the GLMs need here; whether they
# converged is beside the point of a round trip
pytestmark = pytest.mark.filterwarnings("ignore::sklearn.exceptions.ConvergenceWarning")


# ---- estimators -------------------------------------------------------------


def _scaled_normal(sparse: bool = False) -> LearnerPipeline:
    if sparse:
        history = sp.csc_array(
            sp.random(50, 30, density=0.15, random_state=9)  # type: ignore[call-arg]
        )
    else:
        history = np.random.default_rng(9).standard_normal((50, 6))
    scaler = StandardScaler(with_mean=not sparse).fit(history)
    learner = NormalRegressor(alpha=1.0, beta=1.0, sparse=sparse)
    return LearnerPipeline(steps=[("scale", scaler)], learner=learner)


def _is_sparse(est) -> bool:
    return getattr(getattr(est, "learner", est), "sparse", False)


# (factory, data kind); every built-in estimator, and a pipeline around one
DENSE = [
    pytest.param(lambda: NormalRegressor(alpha=1.0, beta=2.0), "real", id="Normal"),
    pytest.param(NormalInverseGammaRegressor, "real", id="NIG"),
    pytest.param(BayesianGLM, "binary", id="GLM"),
    pytest.param(EmpiricalBayesNormalRegressor, "real", id="EBNormal"),
    pytest.param(EmpiricalBayesGLM, "binary", id="EBGLM"),
    pytest.param(lambda: DirichletClassifier({0: 1, 1: 2, 2: 1}), "classes", id="Dir"),
    pytest.param(
        lambda: EmpiricalBayesDirichletClassifier({0: 1, 1: 2, 2: 1}),
        "classes",
        id="EBDir",
    ),
    pytest.param(lambda: GammaRegressor(alpha=2, beta=1), "counts", id="Gamma"),
    pytest.param(
        lambda: EmpiricalBayesGammaRegressor(alpha=2, beta=1), "counts", id="EBGamma"
    ),
    pytest.param(_scaled_normal, "real", id="LearnerPipeline"),
]
SPARSE = [
    pytest.param(
        lambda: NormalRegressor(alpha=1.0, beta=2.0, sparse=True), "real", id="Normal"
    ),
    pytest.param(lambda: NormalInverseGammaRegressor(sparse=True), "real", id="NIG"),
    pytest.param(lambda: BayesianGLM(sparse=True), "binary", id="GLM"),
    pytest.param(
        lambda: EmpiricalBayesNormalRegressor(sparse=True), "real", id="EBNormal"
    ),
    pytest.param(lambda: EmpiricalBayesGLM(sparse=True), "binary", id="EBGLM"),
    pytest.param(lambda: _scaled_normal(sparse=True), "real", id="LearnerPipeline"),
]


def _batch(kind: str, sparse: bool, seed: int) -> tuple[Any, np.ndarray]:
    rng = np.random.default_rng(seed)
    n = 40
    if kind in ("classes", "counts"):
        X = rng.integers(0, 4, size=(n, 1))
        y = rng.integers(0, 3, size=n) if kind == "classes" else rng.poisson(3.0, n)
        return X, y
    if sparse:
        X = sp.csc_array(
            sp.random(n, 30, density=0.15, random_state=seed)  # type: ignore[call-arg]
        )
        w = rng.standard_normal(30)
    else:
        X = rng.standard_normal((n, 6))
        w = rng.standard_normal(6)
    eta = np.asarray(X @ w).ravel() + 0.3 * rng.standard_normal(n)
    return X, (eta > 0).astype(np.float64) if kind == "binary" else eta


def _compare(original, restored, X, exact: bool) -> None:
    """``predict`` and a seeded ``sample`` agree."""
    check: Callable[..., None] = (
        np.testing.assert_array_equal
        if exact
        # A loaded estimator factors afresh, where the original may have
        # carried its factor over: a factor a decay scaled, or a SuperLU
        # refactorization of an unchanged pattern, differs from a fresh one
        # in the last bit. Pickle drops the factor alike.
        else lambda a, b: np.testing.assert_allclose(a, b, rtol=1e-9, atol=1e-12)
    )
    check(restored.predict(X), original.predict(X))
    for est in (original, restored):
        est.random_state = np.random.default_rng(1)
    check(restored.sample(X, size=5), original.sample(X, size=5))


def _assert_same_state(a, b) -> None:
    if isinstance(a, dict):
        assert a.keys() == b.keys()
        for key in a:
            _assert_same_state(a[key], b[key])
    elif isinstance(a, np.ndarray):
        np.testing.assert_array_equal(a, b)
    else:
        assert a == b


def _round_trip(factory, kind: str, sparse: bool, exact: bool) -> None:
    original = factory()
    original.partial_fit(*_batch(kind, sparse, seed=0))
    original.partial_fit(*_batch(kind, sparse, seed=1))
    X, _ = _batch(kind, sparse, seed=4)

    restored = factory()
    restored.load_state_dict(original.state_dict())
    _compare(original, restored, X, exact)

    for est in (original, restored):
        est.partial_fit(*_batch(kind, sparse, seed=2))
    _compare(original, restored, X, exact)

    for est in (original, restored):
        est.decay(decay_rate=0.9)
    _compare(original, restored, X, exact)


@pytest.mark.parametrize("factory, kind", DENSE)
def test_round_trip_matches_original_exactly(factory, kind):
    _round_trip(factory, kind, sparse=False, exact=True)


@pytest.mark.parametrize("factory, kind", DENSE)
def test_update_right_after_load_matches_exactly(factory, kind):
    """Nothing read in between: the update takes what the state holds."""
    original = factory()
    original.partial_fit(*_batch(kind, False, seed=0))
    original.partial_fit(*_batch(kind, False, seed=1))
    restored = factory()
    restored.load_state_dict(original.state_dict())
    for est in (original, restored):
        est.partial_fit(*_batch(kind, False, seed=2))
    _compare(original, restored, _batch(kind, False, seed=4)[0], exact=True)


@pytest.mark.parametrize("factory, kind", DENSE)
def test_state_saved_after_decay(factory, kind):
    """A decay scales the cached factor where it can instead of factoring
    again; a loaded estimator factors afresh, so its draws match the
    original's to the last bit, as after unpickling, and its predictions
    exactly."""
    original = factory()
    original.partial_fit(*_batch(kind, False, seed=0))
    X, _ = _batch(kind, False, seed=4)
    original.sample(X)  # caches the factor the decay then scales
    original.decay(decay_rate=0.9)

    restored = factory()
    restored.load_state_dict(original.state_dict())
    np.testing.assert_array_equal(restored.predict(X), original.predict(X))
    _compare(original, restored, X, exact=False)


@pytest.mark.parametrize("factory, kind", SPARSE)
def test_sparse_round_trip_matches_original(factory, kind, sparse_solver):
    """Exact under CHOLMOD, whose refactorization of an unchanged pattern
    is the fresh factorization; SuperLU's differs in the last bit."""
    exact = sparse_solver is SparseSolver.CHOLMOD
    _round_trip(factory, kind, sparse=True, exact=exact)


@pytest.mark.parametrize("factory, kind", DENSE + SPARSE)
def test_state_is_plain_data(factory, kind):
    est = factory()
    sparse = _is_sparse(est)
    est.partial_fit(*_batch(kind, sparse, seed=0))

    def check(value):
        if isinstance(value, dict):
            assert all(isinstance(k, str) for k in value)
            for v in value.values():
                check(v)
        elif isinstance(value, (list, tuple)):
            for v in value:
                check(v)
        else:
            assert (
                value is None
                or type(value) in (bool, int, float, str)
                or (isinstance(value, np.ndarray) and value.dtype.kind in "fiu")
            ), repr(value)

    state = est.state_dict()
    assert state["version"] == 1
    check(state)


@pytest.mark.parametrize("factory, kind", DENSE + SPARSE)
def test_state_survives_json(factory, kind):
    """Arrays as nested lists and tuples as lists, as a JSON codec gives
    them back, load to the same model."""
    est = factory()
    sparse = _is_sparse(est)
    est.partial_fit(*_batch(kind, sparse, seed=0))
    encoded = json.dumps(est.state_dict(), default=lambda a: a.tolist())

    restored = factory()
    restored.load_state_dict(json.loads(encoded))
    X, _ = _batch(kind, sparse, seed=4)
    np.testing.assert_array_equal(restored.predict(X), est.predict(X))


@pytest.mark.parametrize("factory, kind", DENSE + SPARSE)
def test_state_is_a_copy(factory, kind):
    est = factory()
    sparse = _is_sparse(est)
    est.partial_fit(*_batch(kind, sparse, seed=0))
    X, _ = _batch(kind, sparse, seed=4)
    state = est.state_dict()
    before = est.predict(X)

    restored = factory()
    restored.load_state_dict(state)
    # Updating either model leaves the state and the other model alone
    est.partial_fit(*_batch(kind, sparse, seed=1))
    np.testing.assert_array_equal(restored.predict(X), before)
    restored.partial_fit(*_batch(kind, sparse, seed=2))
    again = factory()
    again.load_state_dict(state)
    np.testing.assert_array_equal(again.predict(X), before)


@pytest.mark.parametrize("factory, kind", DENSE + SPARSE)
def test_unfitted_state_returns_to_a_fresh_estimator(factory, kind):
    """A state from before any fit holds the version and any tuned
    hyperparameters; loading it over a fit, tuning included, leaves an
    estimator that trains as a fresh one does."""
    blank = factory().state_dict()
    assert set(blank) <= {"version", "alpha", "beta", "classes_", "alphas"}

    sparse = _is_sparse(factory())
    est, fresh = factory(), factory()
    est.partial_fit(*_batch(kind, sparse, seed=0))
    est.load_state_dict(blank)
    _assert_same_state(est.state_dict(), blank)
    for model in (est, fresh):
        model.partial_fit(*_batch(kind, sparse, seed=1))
    _compare(fresh, est, _batch(kind, sparse, seed=4)[0], exact=True)


@pytest.mark.parametrize("factory, kind", DENSE + SPARSE)
def test_load_replaces_a_fit(factory, kind):
    """Loading into a fitted estimator leaves nothing of its own fit:
    caches included, it matches a fresh estimator loading the state."""
    sparse = _is_sparse(factory())
    source = factory()
    source.partial_fit(*_batch(kind, sparse, seed=0))
    state = source.state_dict()

    fresh, refit = factory(), factory()
    refit.partial_fit(*_batch(kind, sparse, seed=3))
    refit.sample(_batch(kind, sparse, seed=4)[0])
    for est in (fresh, refit):
        est.load_state_dict(state)
        est.partial_fit(*_batch(kind, sparse, seed=1))
    _compare(fresh, refit, _batch(kind, sparse, seed=4)[0], exact=True)


def test_prior_initialized_by_sampling_stays_unfloored():
    """A prior a ``sample`` initialized has absorbed nothing, so the first
    stabilized update does not floor it; the state carries that."""
    X, y = _batch("real", False, seed=0)

    def make():
        return NormalRegressor(
            alpha=1.0, beta=1.0, forgetting=StabilizedForgetting(0.9)
        )

    original = make()
    original.sample(X)
    restored = make()
    restored.load_state_dict(original.state_dict())
    for est in (original, restored):
        est.partial_fit(X, y)
    np.testing.assert_array_equal(restored.cov_inv_, original.cov_inv_)


def test_eb_normal_keeps_a_pending_mean():
    """After a MacKay correction the mean is left as an unsolved
    information vector; a restored model updates from that same vector."""
    original = EmpiricalBayesNormalRegressor()
    original.fit(*_batch("real", False, seed=0))
    original.partial_fit(*_batch("real", False, seed=1))
    state = original.state_dict()
    assert state["coef_"] is None and state["pending_eta"] is not None

    restored = EmpiricalBayesNormalRegressor()
    restored.load_state_dict(state)
    for est in (original, restored):
        est.partial_fit(*_batch("real", False, seed=2))
    np.testing.assert_array_equal(restored.coef_, original.coef_)


def test_eb_conjugate_hyperparameters_follow_the_tuned_prior():
    X, y = _batch("counts", False, seed=0)
    original = EmpiricalBayesGammaRegressor(alpha=2, beta=1)
    original.fit(X, y)
    restored = EmpiricalBayesGammaRegressor(alpha=2, beta=1)
    restored.load_state_dict(original.state_dict())
    assert (restored.alpha, restored.beta) == (original.alpha, original.beta)

    X, y = _batch("classes", False, seed=0)
    dirichlet = EmpiricalBayesDirichletClassifier({0: 1, 1: 2, 2: 1})
    dirichlet.fit(X, y)
    loaded = EmpiricalBayesDirichletClassifier({0: 1, 1: 2, 2: 1})
    loaded.load_state_dict(dirichlet.state_dict())
    assert loaded.alphas == dirichlet.alphas


def test_learner_generator_is_seeded_as_fit_seeds_it():
    X, y = _batch("real", False, seed=0)
    original = NormalRegressor(alpha=1.0, beta=1.0, random_state=3)
    original.fit(X, y)
    restored = NormalRegressor(alpha=1.0, beta=1.0, random_state=3)
    restored.load_state_dict(original.state_dict())
    np.testing.assert_array_equal(restored.sample(X), original.sample(X))


# ---- rejected states --------------------------------------------------------

NORMAL = (lambda: NormalRegressor(alpha=1.0, beta=1.0), "real", False)
SPARSE_NORMAL = (lambda: NormalRegressor(1.0, 1.0, sparse=True), "real", True)
NIG = (NormalInverseGammaRegressor, "real", False)
GAMMA = (lambda: GammaRegressor(alpha=1, beta=1), "counts", False)
EB_NORMAL = (EmpiricalBayesNormalRegressor, "real", False)


def _fitted_state(setup) -> dict:
    """The state of a model fit and then updated, which leaves an EB
    mean pending."""
    factory, kind, sparse = setup
    est = factory()
    est.partial_fit(*_batch(kind, sparse, seed=0))
    est.partial_fit(*_batch(kind, sparse, seed=1))
    return est.state_dict()


def _nested(state: dict, path: tuple) -> dict:
    for key in path[:-1]:
        state = state[key]
    return state


def _set(*path, to) -> Callable[[dict], None]:
    """Set the value at ``path``; to ``to(old)`` when ``to`` is callable."""

    def corrupt(state: dict) -> None:
        parent = _nested(state, path)
        parent[path[-1]] = to(parent[path[-1]]) if callable(to) else to

    return corrupt


def _drop(*path) -> Callable[[dict], None]:
    return lambda state: _nested(state, path).pop(path[-1])


# (setup, corruption, error, message)
MALFORMED = [
    pytest.param(NORMAL, _set("version", to=2), ValueError, "1, not 2", id="version"),
    pytest.param(NORMAL, _drop("prior_is_fresh"), ValueError, "missing", id="missing"),
    pytest.param(NORMAL, _set("x", to=1), ValueError, "unexpected", id="unexpected"),
    pytest.param(NORMAL, _set("coef_", to=lambda c: c[:-1]), ValueError, "shape"),
    pytest.param(NORMAL, _set("prior_is_fresh", to="no"), TypeError, "a bool"),
    pytest.param(NIG, _set("a_", to="x"), TypeError, "a number", id="float"),
    pytest.param(GAMMA, _set("n_features_", to=True), TypeError, "an integer"),
    pytest.param(GAMMA, _set("n_features_", to=1.5), TypeError, "an integer"),
    pytest.param(NIG, _set("cov_inv_", to=lambda m: m[:, :-1]), ValueError, "square"),
    pytest.param(SPARSE_NORMAL, _drop("cov_inv_", "shape"), ValueError, "the keys"),
    pytest.param(
        SPARSE_NORMAL,
        _set("cov_inv_", "shape", to=lambda s: (s[0], s[1] + 1)),
        ValueError,
        "square",
        id="csc-shape",
    ),
    pytest.param(
        SPARSE_NORMAL,
        _set("cov_inv_", "indices", to=lambda i: i.astype(float)),
        ValueError,
        "integer array",
        id="csc-index-type",
    ),
    pytest.param(
        SPARSE_NORMAL,
        _set("cov_inv_", "indices", to=lambda i: i + 1000),
        ValueError,
        "indices must be <",
        id="csc-index-range",
    ),
    pytest.param(
        GAMMA, _set("coef_", to=[1.0, 2.0]), ValueError, "'keys' and 'values'"
    ),
    pytest.param(
        GAMMA,
        _set("coef_", "keys", to=lambda k: [k[0]] * len(k)),
        ValueError,
        "unique",
        id="table-keys",
    ),
    pytest.param(
        EB_NORMAL,
        _set("coef_", to=lambda _: np.zeros(6)),
        ValueError,
        "exactly one of",
        id="eb-mean",
    ),
    pytest.param(
        EB_NORMAL, _set("eff_XTy", to=None), ValueError, "together", id="eb-running"
    ),
    pytest.param(EB_NORMAL, _set("alpha", to=None), TypeError, "number", id="eb-tuned"),
]


@pytest.mark.parametrize("setup, corrupt, error, match", MALFORMED)
def test_rejects_malformed(setup, corrupt, error, match):
    state = _fitted_state(setup)
    corrupt(state)
    with pytest.raises(error, match=match):
        setup[0]().load_state_dict(state)


def test_rejects_another_estimators_state():
    with pytest.raises(ValueError, match=r"unexpected keys \['a_', 'b_'\]"):
        NormalRegressor(alpha=1.0, beta=1.0).load_state_dict(_fitted_state(NIG))


def test_rejects_a_state_that_is_not_a_dict():
    with pytest.raises(TypeError, match="must be a dict"):
        NormalRegressor(alpha=1.0, beta=1.0).load_state_dict([1])  # type: ignore[arg-type]


def test_rejects_the_other_precision_form():
    with pytest.raises(TypeError, match="estimator is sparse"):
        NormalRegressor(1.0, 1.0, sparse=True).load_state_dict(_fitted_state(NORMAL))
    with pytest.raises(TypeError, match="estimator is dense"):
        NormalRegressor(1.0, 1.0).load_state_dict(_fitted_state(SPARSE_NORMAL))


def test_rejects_other_classes():
    classes = (lambda: DirichletClassifier({0: 1, 1: 1, 2: 1}), "classes", False)
    with pytest.raises(ValueError, match="classes"):
        DirichletClassifier({0: 1, 2: 1, 1: 1}).load_state_dict(_fitted_state(classes))


def test_rejected_state_leaves_the_estimator_alone():
    est = NormalRegressor(alpha=1.0, beta=1.0)
    est.fit(*_batch("real", False, seed=3))
    before = est.state_dict()
    state = _fitted_state(NORMAL)
    state["coef_"] = state["coef_"][:-1]
    with pytest.raises(ValueError):
        est.load_state_dict(state)
    _assert_same_state(est.state_dict(), before)


# ---- agents -----------------------------------------------------------------

X_CTX = np.array([[1.0, 0.5], [0.2, -1.0]])


def _contextual(tokens=("a", "b", "c"), seed=0) -> ContextualAgent:
    arms = [Arm(t, learner=NormalRegressor(alpha=1.0, beta=1.0)) for t in tokens]
    return ContextualAgent(arms, ThompsonSampling(), random_seed=seed)


def _non_contextual(tokens=("a", "b", "c"), seed=0) -> Agent:
    arms = [Arm(t, learner=GammaRegressor(alpha=1, beta=1)) for t in tokens]
    return Agent(arms, ThompsonSampling(), random_seed=seed)


def _lipschitz(tokens=(0, 1, 2), seed=0) -> LipschitzContextualAgent:
    return LipschitzContextualAgent(
        [Arm(t, learner=None) for t in tokens],
        ThompsonSampling(),
        ArmColumnFeaturizer(column_name="arm"),
        NormalRegressor(alpha=1.0, beta=1.0),
        random_seed=seed,
    )


def _pipeline(tokens=("a", "b", "c"), seed=0) -> AgentPipeline:
    return AgentPipeline(
        steps=[("scale", StandardScaler().fit(X_CTX))],
        final_agent=_contextual(tokens, seed),
    )


def _play(agent, rounds: int, rng: np.random.Generator) -> list:
    """Pull, then update the queued arm with a reward, ``rounds`` times."""
    contextual = not isinstance(agent, Agent)
    pulls = []
    for _ in range(rounds):
        pulls.append(agent.pull(X_CTX) if contextual else agent.pull())
        if contextual:
            agent.update(X_CTX, rng.standard_normal(2))
        else:
            agent.update(rng.poisson(2.0, 1).astype(float))
    return pulls


AGENTS = [
    pytest.param(_contextual, id="ContextualAgent"),
    pytest.param(_non_contextual, id="Agent"),
    pytest.param(_lipschitz, id="LipschitzContextualAgent"),
    pytest.param(_pipeline, id="AgentPipeline"),
]


def _learner_states(agent) -> dict:
    return {f"arm {arm.action_token!r}": arm.learner.state_dict() for arm in agent.arms}


@pytest.mark.parametrize("make", AGENTS)
def test_agent_round_trip_continues_the_original(make):
    original = make()
    _play(original, 5, np.random.default_rng(0))
    # Queue an arm without updating it, as a pull awaiting its reward would
    if isinstance(original, Agent):
        original.pull()
    else:
        original.pull(X_CTX)

    restored = make(seed=99)  # the generator comes from the state
    restored.load_state_dict(original.state_dict())
    assert restored.arm_to_update.action_token == original.arm_to_update.action_token
    assert _play(restored, 10, np.random.default_rng(1)) == _play(
        original, 10, np.random.default_rng(1)
    )


@pytest.mark.parametrize("make", AGENTS)
def test_agent_learners_keep_sharing_the_generator(make):
    original = make()
    _play(original, 3, np.random.default_rng(0))
    restored = make(seed=99)
    restored.load_state_dict(original.state_dict())
    for arm in restored.arms:
        assert arm.learner.random_state is restored.rng


@pytest.mark.parametrize("make", AGENTS)
def test_agent_arm_order_may_differ(make):
    original = make()
    _play(original, 5, np.random.default_rng(0))
    tokens = [arm.action_token for arm in original.arms]

    restored = make(tokens=tokens[::-1])
    restored.load_state_dict(original.state_dict())
    for token in tokens:
        _assert_same_state(
            restored.arm(token).learner.state_dict(),
            original.arm(token).learner.state_dict(),
        )


def test_lipschitz_state_stores_the_shared_learner_once():
    agent = _lipschitz()
    _play(agent, 3, np.random.default_rng(0))
    state = agent.state_dict()
    assert state["arms"] == [0, 1, 2]
    assert state["learner"].keys() == {"version", "coef_", "cov_inv_", "prior_is_fresh"}


def test_removed_queued_arm_is_not_restored():
    agent = _contextual()
    agent.pull(X_CTX)
    queued = agent.arm_to_update.action_token
    agent.remove_arm(queued)
    state = agent.state_dict()
    assert state["arm_to_update"] == []

    tokens = [arm.action_token for arm in agent.arms]
    restored = _contextual(tokens=tokens)
    restored.load_state_dict(state)
    assert restored.arm_to_update.action_token == tokens[0]


def test_queued_arm_whose_token_is_none():
    """``None`` is a token an arm may carry, so the queued arm is stored
    as ``[token]`` and a removed one as ``[]``."""
    agent = _contextual(tokens=("a", None))
    agent.select_for_update(None)
    restored = _contextual(tokens=("a", None))
    restored.load_state_dict(agent.state_dict())
    assert restored.arm_to_update is restored.arm(None)


def test_queued_arm_survives_reordered_arms():
    """A codec may reorder the arms' mapping, as sorted keys would."""
    agent = _contextual(tokens=("b", "a"))
    agent.select_for_update("b")
    state = agent.state_dict()
    state["arms"] = dict(sorted(state["arms"].items()))
    restored = _contextual(tokens=("b", "a"))
    restored.load_state_dict(state)
    assert restored.arm_to_update.action_token == "b"


@pytest.mark.parametrize("make", AGENTS)
def test_agent_token_mismatch_raises(make):
    original = make()
    tokens = [arm.action_token for arm in original.arms]
    other = make(tokens=[*tokens[:-1], "other"])
    with pytest.raises(ValueError, match="missing tokens"):
        other.load_state_dict(original.state_dict())


def test_agent_rejects_malformed():
    state = _contextual().state_dict()
    with pytest.raises(ValueError, match="version"):
        _contextual().load_state_dict({**state, "version": 2})
    with pytest.raises(TypeError, match="dict of learner states"):
        _contextual().load_state_dict({**state, "arms": list(state["arms"])})
    with pytest.raises(TypeError, match=r"must be \[token\] or \[\]"):
        _contextual().load_state_dict({**state, "arm_to_update": "a"})
    with pytest.raises(ValueError, match="queues arm 'z'"):
        _contextual().load_state_dict({**state, "arm_to_update": ["z"]})


@pytest.mark.parametrize("make", AGENTS)
def test_rejected_agent_state_changes_nothing(make):
    """Every learner and the generator are checked before any is written."""
    original = make()
    _play(original, 3, np.random.default_rng(0))
    state = original.state_dict()
    if "learner" in state:
        state["learner"] = {**state["learner"], "version": 2}
    else:
        last = list(state["arms"])[-1]
        state["arms"] = {**state["arms"], last: {**state["arms"][last], "version": 2}}

    target = make(seed=99)
    _play(target, 2, np.random.default_rng(5))
    queued = target.arm_to_update
    before = {"rng": target.rng.bit_generator.state, **_learner_states(target)}
    with pytest.raises(ValueError, match="version"):
        target.load_state_dict(state)
    _assert_same_state(
        {"rng": target.rng.bit_generator.state, **_learner_states(target)}, before
    )
    assert target.arm_to_update is queued


def test_rejected_generator_state_changes_nothing():
    original = _contextual()
    _play(original, 3, np.random.default_rng(0))
    state = original.state_dict()
    state["rng"] = np.random.Generator(np.random.MT19937(0)).bit_generator.state

    target = _contextual(seed=99)
    before = _learner_states(target)
    with pytest.raises(ValueError):
        target.load_state_dict(state)
    _assert_same_state(_learner_states(target), before)


def test_agent_with_custom_learner_fails_clearly():
    class Custom:
        random_state = None

    agent = ContextualAgent([Arm("a", learner=Custom())], ThompsonSampling())  # type: ignore[arg-type]
    with pytest.raises(AttributeError, match="state_dict"):
        agent.state_dict()


def test_agent_loads_a_custom_learner_through_its_own_methods():
    """A learner with only the two public methods round-trips too; it
    checks its state as it loads it."""

    class Mean:
        def __init__(self):
            self.random_state = None
            self.total, self.count = 0.0, 0

        def partial_fit(self, X, y, sample_weight=None):
            self.total += float(np.sum(y))
            self.count += len(y)

        def sample(self, X, size=1):
            mean = self.total / max(self.count, 1)
            return np.full((size, len(X)), mean)

        def predict(self, X):
            return self.sample(X)[0]

        def decay(self, forgetting=None, *, decay_rate=None, steps=1):
            pass

        def state_dict(self):
            return {"total": self.total, "count": self.count}

        def load_state_dict(self, state):
            self.total, self.count = state["total"], state["count"]

    def make():
        arms = [Arm(t, learner=Mean()) for t in ("a", "b")]  # type: ignore[arg-type]
        return ContextualAgent(arms, ThompsonSampling(), random_seed=0)

    original = make()
    original.select_for_update("b").update(X_CTX, np.array([1.0, 3.0]))
    restored = make()
    restored.load_state_dict(original.state_dict())
    learner: Any = restored.arm("b").learner
    assert learner.state_dict() == {"total": 4.0, "count": 2}
