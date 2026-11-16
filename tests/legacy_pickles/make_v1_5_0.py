"""Write ``v1_5_0.pkl``: models trained and pickled by bayesianbandits 1.5.0.

Run with 1.5.0 installed, from the repository root::

    python -m tests.legacy_pickles.make_v1_5_0

``cases`` uses only API that 1.5.0 and later share, so the tests build
the same models in code and load the pickled ones' ``state_dict`` into
them.
"""

from __future__ import annotations

import pickle
from pathlib import Path
from typing import Any, Callable, Dict, Tuple

import numpy as np
import pandas as pd

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

PATH = Path(__file__).with_name("v1_5_0.pkl")

_X = np.random.default_rng(0).normal(size=(40, 3))
_COUNTS = np.random.default_rng(1).poisson(2.0, size=40).astype(float)
_LABELS = np.random.default_rng(2).integers(1, 3, size=40)


def _linear(est: Any, binary: bool = False) -> Any:
    y = (_X[:, 0] > 0).astype(float) if binary else _X @ [1.0, -2.0, 0.5]
    est.fit(_X[:20], y[:20])
    est.partial_fit(_X[20:], y[20:])
    est.decay(est.forgetting, steps=2)
    return est


def _grouped(est: Any, y: Any) -> Any:
    groups = np.ones((40, 1))
    est.fit(groups[:20], y[:20])
    est.partial_fit(groups[20:], y[20:])
    return est


def _agent(agent: Any, context: Callable[[int], Any]) -> Any:
    rng = np.random.default_rng(3)
    for _ in range(10):
        X = context(4)
        agent.pull(X)
        agent.update(X, rng.normal(size=4))
    return agent


def _plain_agent(agent: Any) -> Any:
    rng = np.random.default_rng(3)
    for _ in range(10):
        agent.pull()
        agent.update(rng.poisson(2.0, size=1).astype(float))
    return agent


def _frame(n: int) -> pd.DataFrame:
    return pd.DataFrame(_X[:n], columns=pd.Index(["a", "b", "c"]))


def cases() -> Dict[str, Tuple[Callable[[], Any], Callable[[Any], Any]]]:
    """Name -> (build in code, train). Training runs only under 1.5.0."""
    exp, stab = ExponentialForgetting(0.95), StabilizedForgetting(0.95)
    return {
        "normal": (lambda: NormalRegressor(1.0, 1.0, forgetting=exp), _linear),
        "normal_sparse": (
            lambda: NormalRegressor(1.0, 1.0, forgetting=exp, sparse=True),
            _linear,
        ),
        "nig": (lambda: NormalInverseGammaRegressor(forgetting=exp), _linear),
        "nig_sparse": (
            lambda: NormalInverseGammaRegressor(forgetting=exp, sparse=True),
            _linear,
        ),
        "eb_normal": (
            lambda: EmpiricalBayesNormalRegressor(forgetting=stab),
            _linear,
        ),
        "eb_normal_sparse": (
            lambda: EmpiricalBayesNormalRegressor(forgetting=stab, sparse=True),
            _linear,
        ),
        "glm": (
            lambda: BayesianGLM(forgetting=exp),
            lambda e: _linear(e, binary=True),
        ),
        "glm_sparse": (
            lambda: BayesianGLM(forgetting=exp, sparse=True),
            lambda e: _linear(e, binary=True),
        ),
        "eb_glm": (
            lambda: EmpiricalBayesGLM(forgetting=stab),
            lambda e: _linear(e, binary=True),
        ),
        "gamma": (
            lambda: GammaRegressor(1.0, 1.0, forgetting=exp),
            lambda e: _grouped(e, _COUNTS),
        ),
        "eb_gamma": (
            lambda: EmpiricalBayesGammaRegressor(1.0, 1.0, forgetting=stab),
            lambda e: _grouped(e, _COUNTS),
        ),
        "dirichlet": (
            lambda: DirichletClassifier({1: 1.0, 2: 1.0}, forgetting=exp),
            lambda e: _grouped(e, _LABELS),
        ),
        "eb_dirichlet": (
            lambda: EmpiricalBayesDirichletClassifier({1: 1.0, 2: 1.0}),
            lambda e: _grouped(e, _LABELS),
        ),
        "agent": (
            lambda: Agent(
                [Arm(i, learner=GammaRegressor(1.0, 1.0)) for i in range(3)],
                ThompsonSampling(),
                random_seed=0,
            ),
            _plain_agent,
        ),
        "contextual_agent": (
            lambda: ContextualAgent(
                [Arm(i, learner=NormalRegressor(1.0, 1.0)) for i in range(3)],
                ThompsonSampling(),
                random_seed=0,
            ),
            lambda a: _agent(a, lambda n: _X[:n]),
        ),
        "lipschitz_agent": (
            lambda: LipschitzContextualAgent(
                [Arm(i) for i in range(3)],
                ThompsonSampling(),
                ArmColumnFeaturizer("arm"),
                EmpiricalBayesNormalRegressor(forgetting=stab),
                random_seed=0,
            ),
            lambda a: _agent(a, _frame),
        ),
    }


def inputs(name: str) -> Any:
    """What a model of case ``name`` predicts and samples on."""
    if name in ("gamma", "eb_gamma", "dirichlet", "eb_dirichlet"):
        return np.ones((1, 1))
    return _X[:5]


def probe(name: str, model: Any) -> Any:
    """The deterministic output the tests compare against 1.5.0's."""
    return None if name.endswith("agent") else model.predict(inputs(name))


def main() -> None:
    models: Dict[str, Any] = {}
    for name, (build, train) in cases().items():
        models[name] = train(build())
    expected = {name: probe(name, model) for name, model in models.items()}
    with PATH.open("wb") as f:
        pickle.dump({"models": models, "expected": expected}, f, protocol=4)


if __name__ == "__main__":
    main()
