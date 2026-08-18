import sys

import numpy as np
import pytest
import scipy.sparse as sp

from bayesianbandits import (
    Arm,
    ContextualAgent,
    MemoryUsage,
    NormalInverseGammaRegressor,
    NormalRegressor,
    ThompsonSampling,
    memory_usage,
)
from bayesianbandits._memory import _cholmod_usage, _human, _is_cholmod_factor
from bayesianbandits._sparse_bayesian_linear_regression import (
    CholmodSparseFactor,
    SparseSolver,
    SuperLUSparseFactor,
    create_sparse_factor,
)


class TestMemoryUsage:
    def test_total_and_int(self):
        usage = MemoryUsage(1024, parts={"a": MemoryUsage(1024)})
        assert usage.total == 1024
        assert int(usage) == 1024

    def test_repr_flags_estimates(self):
        assert repr(MemoryUsage(2048)) == "MemoryUsage(2.0 KiB)"
        assert repr(MemoryUsage(2048, exact=False)) == "MemoryUsage(2.0 KiB, estimated)"

    def test_report_orders_by_size_and_respects_depth(self):
        usage = MemoryUsage(
            3000,
            parts={
                "small": MemoryUsage(1000, parts={"deep": MemoryUsage(1000)}),
                "big": MemoryUsage(2000),
            },
        )
        assert usage.report(max_depth=1).splitlines() == [
            "2.9 KiB",
            "├── big: 2.0 KiB",
            "└── small: 1000 B",
        ]
        assert "deep" in usage.report(max_depth=2)

    def test_report_min_bytes_hides_without_changing_totals(self):
        usage = MemoryUsage(
            1048, parts={"big": MemoryUsage(1000), "tiny": MemoryUsage(48)}
        )
        report = usage.report(min_bytes=100)
        assert "tiny" not in report
        assert report.startswith("1.0 KiB")

    @pytest.mark.parametrize(
        "n_bytes, expected",
        [(0, "0 B"), (512, "512 B"), (1024, "1.0 KiB"), (1024**2, "1.0 MiB")],
    )
    def test_human_units(self, n_bytes: int, expected: str):
        assert _human(n_bytes) == expected


class TestArrays:
    def test_owned_array_is_its_nbytes(self):
        assert memory_usage(np.zeros(1000)).total == 8000

    def test_view_is_charged_for_the_buffer_it_holds_alive(self):
        buffer = np.zeros(10_000)
        usage = memory_usage(buffer[:10])
        assert usage.total == buffer.nbytes
        assert "holding" in usage.note

    def test_shared_array_is_counted_once(self):
        arr = np.zeros(1000)
        usage = memory_usage([arr, arr])
        assert usage.parts["[0]"].total == 8000
        assert usage.parts["[1]"].total == 0
        assert usage.total < 2 * 8000

    def test_sparse_array_counts_its_three_buffers(self):
        A = sp.random_array((100, 100), density=0.1, format="csc", rng=0)
        A = sp.csc_array(A)
        usage = memory_usage(A)
        assert usage.parts["data"].total == A.data.nbytes
        assert usage.parts["indices"].total == A.indices.nbytes
        assert usage.parts["indptr"].total == A.indptr.nbytes
        assert usage.exact

    def test_interned_scalars_are_not_reported_as_shared(self):
        # Two attributes holding the same interned float must both be
        # charged, not silently zeroed as "shared".
        usage = memory_usage({"a": 1.0, "b": 1.0})
        assert usage.parts["['a']"].total == usage.parts["['b']"].total > 0


class TestObjects:
    def test_closure_contents_are_followed(self):
        table = np.zeros(1000)

        def reward(x):
            return x * table.sum()

        assert memory_usage(reward).total > 8000

    def test_modules_and_classes_are_not_counted(self):
        assert memory_usage(np).total == 0
        assert memory_usage(NormalRegressor).total == 0

    def test_opaque_object_falls_back_to_its_header(self):
        opaque = iter([])  # a list_iterator: no __dict__, no slots
        usage = memory_usage(opaque)
        assert usage.total == sys.getsizeof(opaque)
        assert not usage.exact
        assert "header only" in usage.note

    def test_cycles_terminate(self):
        a: dict = {}
        a["self"] = a
        assert memory_usage(a).total > 0


class TestEstimators:
    def test_dense_estimator_reports_its_posterior(self):
        est = NormalInverseGammaRegressor()
        est.fit(np.eye(2), np.array([1.0, 2.0]))
        usage = est.memory_usage
        assert usage.parts["coef_"].total == 16
        assert usage.parts["cov_inv_"].total == 32
        assert usage.exact

    def test_measuring_does_not_build_lazy_caches(self):
        est = NormalRegressor(alpha=1.0, beta=1.0)
        est.fit(np.eye(3), np.array([1.0, 2.0, 3.0]))
        est.__dict__.pop("_precision_factor", None)
        est.memory_usage  # noqa: B018 - the point is that this is inert
        assert "_precision_factor" not in est.__dict__

    def test_precision_is_charged_once_across_estimator_and_factor(self):
        est = NormalRegressor(alpha=1.0, beta=1.0, sparse=True)
        X = sp.csc_array(sp.eye_array(50, format="csc"))
        est.fit(X, np.ones(50))
        est._precision_factor  # noqa: B018 - materialize the factor
        usage = est.memory_usage
        factor_parts = usage.parts["_precision_factor"].parts
        assert factor_parts["_precision"].total == 0
        assert "shared" in factor_parts["_precision"].note

    def test_sparse_estimator_totals_exceed_the_dense_attributes(self):
        est = NormalRegressor(alpha=1.0, beta=1.0, sparse=True)
        X = sp.csc_array(sp.random_array((20, 500), density=0.05, format="csc", rng=0))
        est.fit(X, np.ones(20))
        est._precision_factor  # noqa: B018 - materialize the factor
        usage = est.memory_usage
        assert usage.total > usage.parts["cov_inv_"].total
        assert usage.total >= sum(part.total for part in usage.parts.values())


class TestAgents:
    def test_agent_totals_cover_every_arm(self):
        arms = [Arm(i, learner=NormalInverseGammaRegressor()) for i in range(3)]
        agent = ContextualAgent(arms, ThompsonSampling(), random_seed=0)
        X = np.array([[1.0, 2.0]])
        agent.pull(X)
        agent.update(X, np.array([1.0]))
        usage = agent.memory_usage
        assert set(usage.parts["_arms"].parts) == {"[0]", "[1]", "[2]"}
        assert usage.total > sum(
            arm.parts["learner"].total for arm in usage.parts["_arms"].parts.values()
        )

    def test_shared_rng_is_not_multiplied_across_arms(self):
        arms = [Arm(i, learner=NormalInverseGammaRegressor()) for i in range(5)]
        agent = ContextualAgent(arms, ThompsonSampling(), random_seed=0)
        per_arm = [
            part.total for part in agent.memory_usage.parts["_arms"].parts.values()
        ]
        # Every learner holds the same Generator; only one of them pays.
        assert max(per_arm) - min(per_arm) < 2000


class TestSparseFactors:
    @pytest.fixture
    def precision(self) -> sp.csc_array:
        rng = np.random.default_rng(0)
        A = sp.random_array((300, 300), density=0.01, format="csc", rng=rng)
        return sp.csc_array(A + A.T + sp.eye_array(300) * 20)

    def test_superlu_is_reported_as_a_lower_bound(self, precision: sp.csc_array):
        factor = create_sparse_factor(precision, solver=SparseSolver.SUPERLU)
        assert isinstance(factor, SuperLUSparseFactor)
        usage = memory_usage(factor._lu)
        assert usage.total == factor._lu.nnz * 12 + 6 * 301 * 4
        assert not usage.exact
        assert "lower bound" in usage.note

    def test_cholmod_is_exact_and_matches_its_array_lengths(
        self, precision: sp.csc_array
    ):
        factor = create_sparse_factor(precision, solver=SparseSolver.CHOLMOD)
        assert isinstance(factor, CholmodSparseFactor)
        cholmod_factor = factor._factor
        usage = memory_usage(cholmod_factor)
        assert usage.exact
        # scikit-sparse finalizes to packed simplicial, so nzmax == nnz
        # and the reported arrays are the ones CHOLMOD actually holds.
        n, nnz = int(cholmod_factor.N), int(cholmod_factor.nnz)
        index_size = cholmod_factor.itype.itemsize
        assert usage.total == nnz * (8 + index_size) + (3 * n + 1) * index_size
        assert not cholmod_factor.is_super

    def test_dense_factor_is_charged_for_its_triangle(self):
        est = NormalRegressor(alpha=1.0, beta=1.0)
        est.fit(np.eye(20), np.ones(20))
        usage = est._precision_factor.memory_usage
        assert usage.parts["_U"].total == 20 * 20 * 8
        assert usage.exact


class TestCholmodFallback:
    """A supernodal factor cannot come out of scikit-sparse, but the
    branch that would report one as a lower bound still has to work."""

    def test_supernodal_factor_is_reported_as_a_lower_bound(self):
        class FakeSupernodalFactor:
            N = 100
            nnz = 5000
            dtype = np.dtype(np.float64)
            itype = np.dtype(np.int32)
            is_super = True

        FakeSupernodalFactor.__module__ = "sksparse.cholmod"
        factor = FakeSupernodalFactor()
        assert _is_cholmod_factor(factor)
        usage = _cholmod_usage(factor)
        assert not usage.exact
        assert "lower bound" in usage.note
        assert usage.total == 5000 * 12 + 301 * 4
