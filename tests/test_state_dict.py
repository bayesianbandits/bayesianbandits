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
from bayesianbandits._estimators import _GAUSSIAN_VERSIONS
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
def test_update_right_after_load_matches(factory, kind):
    """Nothing read in between: the update takes what the state holds.
    Exact but for an EB Normal whose mean a MacKay correction left
    pending: the state holds it solved, and the original updates from
    the unsolved information vector, which differs in the last bit."""
    original = factory()
    original.partial_fit(*_batch(kind, False, seed=0))
    original.partial_fit(*_batch(kind, False, seed=1))
    restored = factory()
    restored.load_state_dict(original.state_dict())
    for est in (original, restored):
        est.partial_fit(*_batch(kind, False, seed=2))
    exact = not isinstance(original, EmpiricalBayesNormalRegressor)
    _compare(original, restored, _batch(kind, False, seed=4)[0], exact=exact)


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
    """Blocks of plain data, each versioned, and no class or attribute
    names."""
    est = factory()
    sparse = _is_sparse(est)
    est.partial_fit(*_batch(kind, sparse, seed=0))

    def check(value):
        if isinstance(value, dict):
            assert all(isinstance(k, str) and not k.endswith("_") for k in value)
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
    assert state["family"] in ("gaussian", "dirichlet", "gamma")
    assert all(state[name]["v"] == 1 for name in state if name != "family")
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
    assert set(blank) == {"family", "prior"}

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


def test_eb_state_has_no_eb_block_before_tuning():
    est = EmpiricalBayesNormalRegressor()
    est.sample(_batch("real", False, seed=0)[0])
    assert "posterior" in est.state_dict() and "eb" not in est.state_dict()


def test_saving_solves_a_pending_mean_but_leaves_the_model_alone():
    """A MacKay correction leaves the mean unsolved behind its
    information vector. The state holds the mean, and saving does not
    solve it on the model, whose next update takes the same path either
    way."""
    original = EmpiricalBayesNormalRegressor()
    original.fit(*_batch("real", False, seed=0))
    original.partial_fit(*_batch("real", False, seed=1))
    assert "_pending_eta" in original.__dict__
    state = original.state_dict()
    assert "_pending_eta" in original.__dict__
    np.testing.assert_array_equal(state["posterior"]["mean"], original.coef_)


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
    pytest.param(NORMAL, _set("family", to="gamma"), ValueError, "reads gaussian"),
    pytest.param(NORMAL, _set("x", to={}), ValueError, "unexpected blocks"),
    pytest.param(NORMAL, _set("posterior", to=[1]), TypeError, "must be a dict"),
    pytest.param(NORMAL, _drop("forgetting"), ValueError, "no forgetting block"),
    pytest.param(NORMAL, _drop("posterior"), ValueError, "has no posterior"),
    pytest.param(NORMAL, _set("posterior", "v", to=2), ValueError, "version 2"),
    pytest.param(NORMAL, _drop("forgetting", "fresh"), ValueError, "missing keys"),
    pytest.param(NORMAL, _set("prior", "x", to=1), ValueError, "unexpected keys"),
    pytest.param(
        NORMAL, _set("posterior", "mean", to=lambda c: c[:-1]), ValueError, "shape"
    ),
    pytest.param(NORMAL, _set("forgetting", "fresh", to="no"), TypeError, "a bool"),
    pytest.param(NORMAL, _set("prior", "alpha", to="x"), TypeError, "a number"),
    pytest.param(NIG, _set("noise", "a", to="x"), TypeError, "a number", id="noise"),
    pytest.param(
        NIG,
        _set("posterior", "precision", to=lambda m: m[:, :-1]),
        ValueError,
        "square",
        id="dense-square",
    ),
    pytest.param(
        SPARSE_NORMAL,
        _drop("posterior", "precision", "shape"),
        ValueError,
        "the keys",
        id="csc-keys",
    ),
    pytest.param(
        SPARSE_NORMAL,
        _set("posterior", "precision", "shape", to=lambda s: [s[0], s[1] + 1]),
        ValueError,
        "square",
        id="csc-shape",
    ),
    pytest.param(
        SPARSE_NORMAL,
        _set("posterior", "precision", "indices", to=lambda i: i.astype(float)),
        ValueError,
        "integer array",
        id="csc-index-type",
    ),
    pytest.param(
        SPARSE_NORMAL,
        _set("posterior", "precision", "indices", to=lambda i: i + 1000),
        ValueError,
        "indices must be <",
        id="csc-index-range",
    ),
    pytest.param(
        GAMMA,
        _set("posterior", "groups", to=lambda k: [k[0]] * len(k)),
        ValueError,
        "repeat",
        id="groups-repeat",
    ),
    pytest.param(
        GAMMA,
        _set("posterior", "beta", to=lambda b: b[:-1]),
        ValueError,
        "shape",
        id="groups-shape",
    ),
    pytest.param(
        EB_NORMAL,
        _set("eb", "updates_rejected", to=1.5),
        TypeError,
        "an integer",
        id="eb-int",
    ),
    pytest.param(
        EB_NORMAL,
        _set("eb", "updates_rejected", to=True),
        TypeError,
        "an integer",
        id="eb-int-bool",
    ),
    pytest.param(
        GAMMA,
        _set("posterior", "groups", to="ab"),
        TypeError,
        "must be a list",
        id="groups-type",
    ),
    pytest.param(
        EB_NORMAL,
        _set("forgetting", "prior_weight", to=None),
        ValueError,
        "no forgetting prior_weight",
        id="eb-prior-weight",
    ),
    pytest.param(
        EB_NORMAL,
        _set("eb", "xty", to=None),
        ValueError,
        "must hold the Normal",
        id="eb-statistics-partial",
    ),
    pytest.param(
        EB_NORMAL,
        _set("eb", "loglik", to=0.0),
        ValueError,
        "must hold the Normal",
        id="eb-statistics-mixed",
    ),
    pytest.param(
        NORMAL,
        _set("posterior", "mean", to=lambda c: np.full_like(c, np.nan)),
        ValueError,
        "finite",
        id="nan",
    ),
    pytest.param(NORMAL, _set("prior", "alpha", to=-1.0), ValueError, "positive"),
    pytest.param(
        EB_NORMAL,
        _set("forgetting", "prior_weight", to=float("nan")),
        ValueError,
        "finite",
        id="prior-weight-nan",
    ),
    pytest.param(
        EB_NORMAL,
        _set("eb", "effective_n", to=float("nan")),
        ValueError,
        "finite",
        id="effective-n-nan",
    ),
    pytest.param(
        EB_NORMAL, _set("eb", "yty", to=-1.0), ValueError, "non-negative", id="yty"
    ),
    pytest.param(NIG, _set("noise", "b", to=0.0), ValueError, "positive", id="noise-b"),
    pytest.param(
        NORMAL,
        _set("posterior", "precision", to=lambda m: np.triu(m) + np.eye(len(m))),
        ValueError,
        "symmetric",
        id="asymmetric",
    ),
    pytest.param(
        GAMMA,
        _set("posterior", "alpha", to=lambda a: -a),
        ValueError,
        "positive",
        id="gamma-negative",
    ),
]


@pytest.mark.parametrize("setup, corrupt, error, match", MALFORMED)
def test_rejects_malformed(setup, corrupt, error, match):
    state = _fitted_state(setup)
    corrupt(state)
    with pytest.raises(error, match=match):
        setup[0]().load_state_dict(state)


def test_rejects_a_state_that_is_not_a_dict():
    with pytest.raises(TypeError, match="must be a dict"):
        NormalRegressor(alpha=1.0, beta=1.0).load_state_dict([1])  # type: ignore[arg-type]


def test_rejects_other_classes():
    classes = (lambda: DirichletClassifier({0: 1, 1: 1, 2: 1}), "classes", False)
    with pytest.raises(ValueError, match="classes"):
        DirichletClassifier({0: 1, 1: 1, 3: 1}).load_state_dict(_fitted_state(classes))


def test_rejected_state_leaves_the_estimator_alone():
    est = NormalRegressor(alpha=1.0, beta=1.0)
    est.fit(*_batch("real", False, seed=3))
    before = est.state_dict()
    state = _fitted_state(NORMAL)
    state["posterior"]["mean"] = state["posterior"]["mean"][:-1]
    with pytest.raises(ValueError):
        est.load_state_dict(state)
    _assert_same_state(est.state_dict(), before)


# ---- states across a family -------------------------------------------------

GAUSSIAN = [
    pytest.param(lambda: NormalRegressor(alpha=1.0, beta=2.0), id="Normal"),
    pytest.param(NormalInverseGammaRegressor, id="NIG"),
    pytest.param(BayesianGLM, id="GLM"),
    pytest.param(EmpiricalBayesNormalRegressor, id="EBNormal"),
    pytest.param(EmpiricalBayesGLM, id="EBGLM"),
]


@pytest.mark.parametrize("target", GAUSSIAN)
@pytest.mark.parametrize("source", GAUSSIAN)
def test_any_gaussian_state_loads_into_any_gaussian_estimator(source, target):
    """The posterior carries over; what the target has no use for is set
    aside, and what it lacks starts from its own prior."""
    original = source()
    original.partial_fit(*_batch("binary", False, seed=0))
    state = original.state_dict()
    restored = target()
    restored.load_state_dict(state)
    np.testing.assert_array_equal(restored.coef_, state["posterior"]["mean"])
    restored.partial_fit(*_batch("binary", False, seed=1))


@pytest.mark.parametrize(
    "sparse", [False, True], ids=["dense-to-sparse", "sparse-to-dense"]
)
def test_dense_and_sparse_states_load_into_each_other(sparse):
    """The precision converts to the target's form. Dense and sparse
    arithmetic differ in the last bit, so the models agree to rounding."""
    rng = np.random.default_rng(0)
    X = rng.standard_normal((40, 6))
    y = X @ rng.standard_normal(6)
    as_input = (lambda A: sp.csc_array(A)) if sparse else (lambda A: A)
    from_input = (lambda A: A) if sparse else (lambda A: sp.csc_array(A))
    original = NormalRegressor(1.0, 1.0, sparse=not sparse)
    original.fit(from_input(X), y)
    restored = NormalRegressor(1.0, 1.0, sparse=sparse)
    restored.load_state_dict(original.state_dict())
    assert sp.issparse(restored.cov_inv_) == sparse
    np.testing.assert_array_equal(restored.predict(X), original.predict(X))
    original.partial_fit(from_input(X[:10]), y[:10])
    restored.partial_fit(as_input(X[:10]), y[:10])
    np.testing.assert_allclose(restored.predict(X), original.predict(X), rtol=1e-9)


@pytest.mark.parametrize("source", [BayesianGLM, NormalInverseGammaRegressor])
def test_load_does_not_depend_on_what_the_estimator_learned_before(source):
    """A state without beta, or without either hyperparameter, gives an
    EB Normal the ones it was built with, however it was tuned before."""
    original = source()
    original.partial_fit(*_batch("binary", False, seed=0))
    state = original.state_dict()
    fresh, tuned = EmpiricalBayesNormalRegressor(), EmpiricalBayesNormalRegressor()
    tuned.fit(*_batch("real", False, seed=5))
    for est in (fresh, tuned):
        est.load_state_dict(state)
        est.partial_fit(*_batch("real", False, seed=1))
    assert (tuned.alpha, tuned.beta) == (fresh.alpha, fresh.beta)
    _compare(fresh, tuned, _batch("real", False, seed=4)[0], exact=True)


def test_hyperparameters_stay_as_built_where_nothing_tunes_them():
    original = NormalRegressor(alpha=2.0, beta=3.0)
    original.fit(*_batch("real", False, seed=0))
    restored = NormalRegressor(alpha=5.0, beta=7.0)
    restored.load_state_dict(original.state_dict())
    assert (restored.alpha, restored.beta) == (5.0, 7.0)


def test_a_block_version_change_leaves_other_blocks_readable(monkeypatch):
    """Bumping the eb block's version rejects only states that carry one."""
    plain = _fitted_state(NORMAL)
    tuned = _fitted_state(EB_NORMAL)
    monkeypatch.setitem(_GAUSSIAN_VERSIONS, "eb", 2)
    NormalRegressor(alpha=1.0, beta=1.0).load_state_dict(plain)
    with pytest.raises(ValueError, match="eb is version 1"):
        EmpiricalBayesNormalRegressor().load_state_dict(tuned)


def test_normal_state_loads_into_eb_normal():
    """Without an eb block, empirical Bayes starts tuning from the
    loaded posterior and prior."""
    X, y = _batch("real", False, seed=0)
    original = NormalRegressor(alpha=2.0, beta=3.0)
    original.fit(X, y)
    restored = EmpiricalBayesNormalRegressor()
    restored.load_state_dict(original.state_dict())
    assert (restored.alpha, restored.beta) == (2.0, 3.0)
    np.testing.assert_array_equal(restored.predict(X), original.predict(X))
    restored.partial_fit(*_batch("real", False, seed=1))
    assert np.isfinite(restored.log_evidence_)


def test_classes_are_matched_by_label():
    X, y = _batch("classes", False, seed=0)
    original = DirichletClassifier({0: 1, 1: 2, 2: 1})
    original.fit(X, y)
    restored = DirichletClassifier({2: 1, 1: 2, 0: 1})
    restored.load_state_dict(original.state_dict())
    np.testing.assert_array_equal(
        restored.predict_proba(X)[:, ::-1], original.predict_proba(X)
    )


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
    assert state["learner"]["family"] == "gaussian"


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
    """A codec may reorder the arms."""
    agent = _contextual(tokens=("b", "a"))
    agent.select_for_update("b")
    state = agent.state_dict()
    state["arms"] = sorted(state["arms"], key=lambda pair: pair[0])
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
        _contextual().load_state_dict({**state, "v": 2})
    with pytest.raises(TypeError, match=r"list of \[token, state\] pairs"):
        _contextual().load_state_dict({**state, "arms": dict(state["arms"])})
    with pytest.raises(ValueError, match="repeat"):
        _contextual().load_state_dict({**state, "arms": state["arms"] * 2})
    with pytest.raises(TypeError, match=r"list of \[token, state\] pairs"):
        _contextual().load_state_dict({**state, "arms": [["a"]]})
    with pytest.raises(TypeError, match=r"must be \[token\] or \[\]"):
        _contextual().load_state_dict({**state, "arm_to_update": "a"})
    with pytest.raises(ValueError, match="queues arm 'z'"):
        _contextual().load_state_dict({**state, "arm_to_update": ["z"]})


@pytest.mark.parametrize(
    "make",
    [_contextual, _lipschitz],
    ids=["ContextualAgent", "LipschitzContextualAgent"],
)
def test_agent_state_survives_json_with_integer_tokens(make):
    original = make(tokens=(0, 1, 2))
    _play(original, 3, np.random.default_rng(0))
    original.pull(X_CTX)
    encoded = json.dumps(original.state_dict(), default=lambda a: a.tolist())

    restored = make(tokens=(0, 1, 2), seed=99)
    restored.load_state_dict(json.loads(encoded))
    assert restored.arm_to_update.action_token == original.arm_to_update.action_token
    assert _play(restored, 5, np.random.default_rng(1)) == _play(
        original, 5, np.random.default_rng(1)
    )


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
