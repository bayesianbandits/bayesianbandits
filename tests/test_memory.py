import numpy as np
import pytest
import scipy.sparse as sp

from bayesianbandits import (
    Arm,
    ContextualAgent,
    NormalInverseGammaRegressor,
    NormalRegressor,
    ThompsonSampling,
    memory_usage,
)
from bayesianbandits._sparse_bayesian_linear_regression import (
    CholmodSparseFactor,
    SparseSolver,
    SuperLUSparseFactor,
    create_sparse_factor,
)


def test_measuring_does_not_build_lazy_caches():
    """The walk reads ``__dict__`` rather than calling ``getattr``. Reach
    for an attribute instead and asking a model its size silently builds
    the factorization being measured."""
    est = NormalRegressor(alpha=1.0, beta=1.0)
    est.fit(np.eye(3), np.array([1.0, 2.0, 3.0]))
    est.__dict__.pop("_precision_factor", None)

    est.memory_usage  # noqa: B018 - the point is that this is inert

    assert "_precision_factor" not in est.__dict__


class TestSharedBuffersAreChargedOnce:
    """A total has to be what dropping the object would free, not a sum
    over references -- otherwise every shared learner and every cached
    factor double-counts its precision matrix."""

    def test_the_same_array_twice(self):
        arr = np.zeros(1000)
        usage = memory_usage([arr, arr])
        assert usage.parts["[0]"].total == arr.nbytes
        assert usage.parts["[1]"].total == 0

    def test_precision_shared_between_estimator_and_its_factor(self):
        est = NormalRegressor(alpha=1.0, beta=1.0, sparse=True)
        est.fit(sp.csc_array(sp.eye_array(50, format="csc")), np.ones(50))
        est._precision_factor  # noqa: B018 - materialize the factor

        factor_parts = est.memory_usage.parts["_precision_factor"].parts

        assert factor_parts["_precision"].total == 0
        assert "shared" in factor_parts["_precision"].note

    def test_rng_shared_by_every_arm_learner(self):
        arms = [Arm(i, learner=NormalInverseGammaRegressor()) for i in range(5)]
        agent = ContextualAgent(arms, ThompsonSampling(), random_seed=0)

        per_arm = [
            part.total for part in agent.memory_usage.parts["_arms"].parts.values()
        ]

        # One Generator across all five learners; only one arm pays for it.
        assert max(per_arm) - min(per_arm) < 2000

    def test_interned_scalars_are_exempt(self):
        """Python hands out one object for the same small float, so
        deduplicating those would fill a report with 0-byte rows."""
        usage = memory_usage({"a": 1.0, "b": 1.0})
        assert usage.parts["['a']"].total == usage.parts["['b']"].total > 0


class TestWalk:
    def test_a_view_is_charged_for_the_buffer_it_holds_alive(self):
        """SciPy's sparse addition leaves its result on an oversized
        allocation, which the logical ``nbytes`` would hide."""
        buffer = np.zeros(10_000)
        usage = memory_usage(buffer[:10])
        assert usage.total == buffer.nbytes
        assert "holding" in usage.note

    def test_sparse_arrays_reach_their_three_buffers(self):
        A = sp.csc_array(sp.random_array((100, 100), density=0.1, format="csc", rng=0))
        usage = memory_usage(A)
        assert usage.parts["data"].total == A.data.nbytes
        assert usage.parts["indices"].total == A.indices.nbytes
        assert usage.parts["indptr"].total == A.indptr.nbytes

    def test_closure_contents_are_followed(self):
        """A reward function can close over a lookup table."""
        table = np.zeros(1000)

        def reward(x):
            return x * table.sum()

        assert memory_usage(reward).total > table.nbytes

    def test_a_reference_cycle_terminates(self):
        a: dict = {}
        a["self"] = a
        assert memory_usage(a).total > 0


class TestCFactorizations:
    @pytest.fixture
    def precision(self) -> sp.csc_array:
        rng = np.random.default_rng(0)
        A = sp.random_array((300, 300), density=0.01, format="csc", rng=rng)
        return sp.csc_array(A + A.T + sp.eye_array(300) * 20)

    def test_cholmod_is_exact_because_its_factor_is_packed_simplicial(
        self, precision: sp.csc_array
    ):
        """The exactness claim rests on scikit-sparse finalizing every
        factor to packed simplicial form, where ``nzmax == nnz``. If a
        release stops doing that, ``is_super`` flips here and the report
        degrades to a labelled lower bound rather than a silent lie."""
        factor = create_sparse_factor(precision, solver=SparseSolver.CHOLMOD)
        assert isinstance(factor, CholmodSparseFactor)

        usage = memory_usage(factor._factor)

        assert not factor._factor.is_super
        assert usage.exact
        assert usage.total > int(factor._factor.nnz) * 8

    def test_superlu_is_labelled_a_lower_bound(self, precision: sp.csc_array):
        """SciPy exposes only the logical nonzero count, and SuperLU's
        supernodal ``L`` stores more than that."""
        factor = create_sparse_factor(precision, solver=SparseSolver.SUPERLU)
        assert isinstance(factor, SuperLUSparseFactor)

        usage = memory_usage(factor._lu)

        assert not usage.exact
        assert "lower bound" in usage.note
        assert usage.total > factor._lu.nnz * 8
