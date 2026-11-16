"""Pickles from earlier releases: the ones that load convert to a
``state_dict``, and the ones that cannot load say so."""

from __future__ import annotations

import copy
import pickle
from pathlib import Path
from typing import Any, Callable, Dict

import numpy as np
import pandas as pd
import pytest
from numpy.testing import assert_allclose

from bayesianbandits import (
    Agent,
    Arm,
    ArmColumnFeaturizer,
    BayesianGLM,
    ContextualAgent,
    DirichletClassifier,
    EmpiricalBayesDirichletClassifier,
    EmpiricalBayesGammaRegressor,
    EmpiricalBayesGLM,
    EmpiricalBayesNormalRegressor,
    ExponentialForgetting,
    GammaRegressor,
    LipschitzContextualAgent,
    NormalInverseGammaRegressor,
    NormalRegressor,
    StabilizedForgetting,
    ThompsonSampling,
)

# Models trained and pickled by bayesianbandits 1.5.0 on _X, with their
# predictions on inputs(name)
PATH = Path(__file__).parent / "legacy_pickles" / "v1_5_0.pkl"

_X = np.random.default_rng(0).normal(size=(40, 3))
_GROUPED = ("gamma", "eb_gamma", "dirichlet", "eb_dirichlet")


def _frame(n: int) -> pd.DataFrame:
    return pd.DataFrame(_X[:n], columns=pd.Index(["a", "b", "c"]))


_EXP, _STAB = ExponentialForgetting(0.95), StabilizedForgetting(0.95)

# How each pickled model was built
BUILD: Dict[str, Callable[[], Any]] = {
    "normal": lambda: NormalRegressor(1.0, 1.0, forgetting=_EXP),
    "normal_sparse": lambda: NormalRegressor(1.0, 1.0, forgetting=_EXP, sparse=True),
    "nig": lambda: NormalInverseGammaRegressor(forgetting=_EXP),
    "nig_sparse": lambda: NormalInverseGammaRegressor(forgetting=_EXP, sparse=True),
    "eb_normal": lambda: EmpiricalBayesNormalRegressor(forgetting=_STAB),
    "eb_normal_sparse": lambda: EmpiricalBayesNormalRegressor(
        forgetting=_STAB, sparse=True
    ),
    "glm": lambda: BayesianGLM(forgetting=_EXP),
    "glm_sparse": lambda: BayesianGLM(forgetting=_EXP, sparse=True),
    "eb_glm": lambda: EmpiricalBayesGLM(forgetting=_STAB),
    "gamma": lambda: GammaRegressor(1.0, 1.0, forgetting=_EXP),
    "eb_gamma": lambda: EmpiricalBayesGammaRegressor(1.0, 1.0, forgetting=_STAB),
    "dirichlet": lambda: DirichletClassifier({1: 1.0, 2: 1.0}, forgetting=_EXP),
    "eb_dirichlet": lambda: EmpiricalBayesDirichletClassifier({1: 1.0, 2: 1.0}),
    "agent": lambda: Agent(
        [Arm(i, learner=GammaRegressor(1.0, 1.0)) for i in range(3)],
        ThompsonSampling(),
        random_seed=0,
    ),
    "contextual_agent": lambda: ContextualAgent(
        [Arm(i, learner=NormalRegressor(1.0, 1.0)) for i in range(3)],
        ThompsonSampling(),
        random_seed=0,
    ),
    "lipschitz_agent": lambda: LipschitzContextualAgent(
        [Arm(i) for i in range(3)],
        ThompsonSampling(),
        ArmColumnFeaturizer("arm"),
        EmpiricalBayesNormalRegressor(forgetting=_STAB),
        random_seed=0,
    ),
}


@pytest.fixture(scope="module")
def v1_5_0() -> Dict[str, Any]:
    with PATH.open("rb") as f:
        return pickle.load(f)


def _inputs(name: str) -> Any:
    return np.ones((1, 1)) if name in _GROUPED else _X[:5]


def _pulls(name: str, agent: Any) -> Any:
    if name == "agent":
        return [agent.pull() for _ in range(5)]
    context = _frame if name == "lipschitz_agent" else (lambda n: _X[:n])
    return [agent.pull(context(4)) for _ in range(5)]


@pytest.mark.parametrize("name", list(BUILD))
def test_v1_5_0_pickle_converts_to_state_dict(v1_5_0: Dict[str, Any], name: str):
    old = copy.deepcopy(v1_5_0["models"][name])
    new = BUILD[name]()
    new.load_state_dict(old.state_dict())

    if name.endswith("agent"):
        assert _pulls(name, new) == _pulls(name, old)
        return
    X = _inputs(name)
    assert_allclose(old.predict(X), v1_5_0["expected"][name], rtol=1e-10)
    assert_allclose(new.predict(X), old.predict(X), rtol=1e-10)
    old.random_state = np.random.default_rng(0)
    new.random_state = np.random.default_rng(0)
    assert_allclose(new.sample(X, size=3), old.sample(X, size=3))


def test_eb_normal_pickled_before_1_5_0_says_to_refit():
    est = EmpiricalBayesNormalRegressor().fit(_X, _X[:, 0])
    state = est.__getstate__()
    del state["_alpha0"]
    fresh: Any = object.__new__(EmpiricalBayesNormalRegressor)
    with pytest.raises(ValueError, match="before 1.5.0.*Refit"):
        fresh.__setstate__(state)
