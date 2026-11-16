"""Pickles from earlier releases: the ones that load convert to a
``state_dict``, and the ones that cannot load say so."""

from __future__ import annotations

import copy
import pickle
from typing import Any, Dict

import numpy as np
import pytest
from numpy.testing import assert_allclose

from bayesianbandits import EmpiricalBayesNormalRegressor
from tests.legacy_pickles.make_v1_5_0 import (
    _X,
    PATH,
    _frame,
    cases,
    inputs,
    probe,
)

CASES = cases()


@pytest.fixture(scope="module")
def v1_5_0() -> Dict[str, Any]:
    with PATH.open("rb") as f:
        return pickle.load(f)


def _pulls(name: str, agent: Any) -> Any:
    if name == "agent":
        return [agent.pull() for _ in range(5)]
    context = _frame if name == "lipschitz_agent" else (lambda n: _X[:n])
    return [agent.pull(context(4)) for _ in range(5)]


@pytest.mark.parametrize("name", list(CASES))
def test_v1_5_0_pickle_converts_to_state_dict(v1_5_0: Dict[str, Any], name: str):
    old = copy.deepcopy(v1_5_0["models"][name])
    expected = v1_5_0["expected"][name]
    if expected is not None:
        assert_allclose(probe(name, old), expected, rtol=1e-10)

    build, _ = CASES[name]
    new = build()
    new.load_state_dict(old.state_dict())

    if name.endswith("agent"):
        assert _pulls(name, new) == _pulls(name, old)
    else:
        assert_allclose(probe(name, new), probe(name, old), rtol=1e-10)
        old.random_state = np.random.default_rng(0)
        new.random_state = np.random.default_rng(0)
        X = inputs(name)
        assert_allclose(new.sample(X, size=3), old.sample(X, size=3))


def test_eb_normal_pickled_before_1_5_0_says_to_refit():
    est = EmpiricalBayesNormalRegressor().fit(_X, _X[:, 0])
    state = est.__getstate__()
    del state["_alpha0"]
    fresh: Any = object.__new__(EmpiricalBayesNormalRegressor)
    with pytest.raises(ValueError, match="before 1.5.0.*Refit"):
        fresh.__setstate__(state)
