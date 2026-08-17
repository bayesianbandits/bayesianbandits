from pathlib import Path
from unittest.mock import patch

import joblib
import numpy as np
import pytest
import scipy.sparse as sp
from numpy.testing import assert_allclose, assert_array_almost_equal
from scipy.stats import Covariance, multivariate_normal, multivariate_t

from bayesianbandits._estimators import multivariate_t_sample_from_covariance
from bayesianbandits._sparse_bayesian_linear_regression import (
    CholmodSparseFactor,
    DenseFactor,
    SparseSolver,
    create_sparse_factor,
    multivariate_normal_sample_from_sparse_precision,
    multivariate_t_sample_from_precision,
    scale_factor,
)


@pytest.mark.parametrize("solver", [SparseSolver.SUPERLU, SparseSolver.CHOLMOD])
@pytest.mark.parametrize("size", [1, 10])
def test_multivariate_normal_sample_from_sparse_precision_ill_conditioned_matrices(
    size, solver
):
    this_file_path = Path(__file__)
    test_data_dir = this_file_path.parent / "ill_conditioned_matrices"
    for file_path in test_data_dir.glob("*"):
        sparse_array = joblib.load(file_path)
        factor = create_sparse_factor(sparse_array, solver=solver)
        samples = multivariate_normal_sample_from_sparse_precision(
            mean=None, factor=factor, size=size
        )
        if size != 1:
            assert samples.shape == (size, sparse_array.shape[0])
        else:
            assert samples.shape == (sparse_array.shape[0],)


class TestSparseFactor:
    @pytest.fixture(scope="class")
    def precision_matrix(self):
        mat = sp.random(500, 500, 0.001) * 100
        return (
            (mat @ mat.T) + 100 * sp.diags(1 + np.random.gamma(1, 1, 500))
        ).toarray()

    def test_prec_must_be_sparse(self):
        with pytest.raises(TypeError):
            create_sparse_factor(np.eye(10))  # type: ignore

    @pytest.mark.parametrize(
        "matrix",
        [
            joblib.load(file)
            for file in Path(__file__).parent.glob("ill_conditioned_matrices/*")
        ],
    )
    def test_colorize_ill_conditioned_matrices(self, matrix):
        """
        These aren't actually going to be the same, but they should be close. We'll
        test by taking a large number of samples and checking that the variances are
        close.
        """
        superlu_factor = create_sparse_factor(
            sp.csc_array(matrix), solver=SparseSolver.SUPERLU
        )
        cholmod_factor = create_sparse_factor(
            sp.csc_array(matrix), solver=SparseSolver.CHOLMOD
        )

        random_samples = np.random.default_rng(0).normal(size=(1000, matrix.shape[0]))
        superlu_samples = superlu_factor.colorize(random_samples.T).T
        cholmod_samples = cholmod_factor.colorize(random_samples.T).T

        assert_allclose(
            superlu_samples.var(axis=0), cholmod_samples.var(axis=0), rtol=0.5
        )

    def test_superlu_solve_accepts_a_sparse_rhs(self, precision_matrix):
        """A sparse ``b`` cannot go through the cached triangular factor,
        so ``solve`` routes it to ``spsolve`` instead."""
        precision = sp.csc_array(precision_matrix)
        factor = create_sparse_factor(precision, solver=SparseSolver.SUPERLU)
        n = precision.shape[0]
        b = sp.csc_array(np.eye(n)[:, :2])

        got = factor.solve(b)
        got = np.asarray(got.todense() if sp.issparse(got) else got).reshape(n, 2)
        want = np.linalg.solve(np.asarray(precision_matrix), b.toarray())
        assert_allclose(got, want, atol=1e-8)

    def test_umfpack_and_superlu_errors_when_not_symmetric_and_positive_definite(
        self,
    ):
        matrix = np.array([[0.0, 2.0], [1.0, 0.0]])
        with pytest.raises(ValueError):
            create_sparse_factor(sp.csc_array(matrix), solver=SparseSolver.SUPERLU)

    @pytest.mark.parametrize("solver", [SparseSolver.SUPERLU, SparseSolver.CHOLMOD])
    def test_mvn_sampling_against_scipy(self, precision_matrix, solver):
        factor = create_sparse_factor(sp.csc_array(precision_matrix), solver=solver)
        scipy_cov = Covariance.from_precision(precision_matrix)

        sparse_samples = multivariate_normal_sample_from_sparse_precision(
            mean=None, factor=factor, size=80000, random_state=0
        )
        scipy_samples = multivariate_normal.rvs(
            mean=None,
            cov=scipy_cov,  # type: ignore
            size=80000,
            random_state=0,  # type: ignore
        )

        sparse_emp_cov = np.cov(sparse_samples.T)
        scipy_emp_cov = np.cov(scipy_samples.T)

        assert_allclose(sparse_emp_cov, scipy_emp_cov, atol=0.05)

    def test_mvt_sample_scipy_vs_bb(self, precision_matrix):
        scipy_cov = Covariance.from_precision(precision_matrix)

        rs_1 = np.random.default_rng(0)
        rs_2 = np.random.default_rng(0)

        # The way that multivariate_t uses the random state is different from
        # how we use it, so we're going to mock out the multivariate normal sampling
        # in our implementation (as our above tests show that it's correct) and
        # just test the t distribution sampling.
        with patch("bayesianbandits._estimators.multivariate_normal.rvs") as mock_mvn:
            mock_mvn.side_effect = lambda mean, shape, size, random_state: (
                random_state.multivariate_normal(
                    np.zeros(scipy_cov.shape[0]), shape, size=size
                )
            )
            sparse_samples = multivariate_t_sample_from_covariance(
                loc=None, shape=scipy_cov.covariance, size=100, random_state=rs_1, df=3
            )

        scipy_samples = multivariate_t.rvs(
            loc=None, shape=scipy_cov.covariance, size=100, random_state=rs_2, df=3
        )

        assert_array_almost_equal(sparse_samples, scipy_samples)

    def test_mvt_sample_tuple_size_and_empty(self, precision_matrix):
        """Tuple ``size`` and ``size=0`` follow numpy conventions
        (regression: a ``reshape(size, -1)`` once broke both)."""
        scipy_cov = Covariance.from_precision(precision_matrix)
        d = scipy_cov.shape[0]
        draws = multivariate_t_sample_from_covariance(
            loc=None, shape=scipy_cov, size=(2, 4), random_state=0, df=5
        )
        assert np.asarray(draws).shape == (2, 4, d)
        empty = multivariate_t_sample_from_covariance(
            loc=None, shape=scipy_cov, size=0, random_state=0, df=5
        )
        assert np.asarray(empty).shape == (0, d)

    @pytest.mark.parametrize("solver", [SparseSolver.SUPERLU, SparseSolver.CHOLMOD])
    def test_logdet_matches_dense(self, precision_matrix, solver):
        """logdet via sparse factor matches np.linalg.slogdet for both solvers."""
        factor = create_sparse_factor(sp.csc_array(precision_matrix), solver=solver)
        expected = float(np.linalg.slogdet(precision_matrix)[1])
        result = factor.logdet()
        assert_allclose(result, expected, atol=1e-6)

    @pytest.mark.parametrize("solver", [SparseSolver.SUPERLU, SparseSolver.CHOLMOD])
    def test_mvt_sampling_against_scipy(self, precision_matrix, solver):
        factor = create_sparse_factor(sp.csc_array(precision_matrix), solver=solver)
        scipy_cov = Covariance.from_precision(precision_matrix)

        sparse_samples = multivariate_t_sample_from_precision(
            loc=None, factor=factor, size=80000, random_state=0, df=300
        )
        scipy_samples = multivariate_t_sample_from_covariance(
            loc=None,
            shape=scipy_cov,
            size=80000,
            random_state=0,
            df=300,
        )

        sparse_emp_cov = np.cov(sparse_samples.T)
        scipy_emp_cov = np.cov(scipy_samples.T)

        assert_allclose(sparse_emp_cov, scipy_emp_cov, atol=0.05)


class TestRefactorize:
    """Test that refactorize() produces identical results to fresh factorization."""

    @pytest.fixture
    def spd_matrices(self):
        """Two SPD matrices with the same sparsity pattern but different values."""
        rng = np.random.default_rng(42)
        n = 200
        base = sp.random(n, n, density=0.01, random_state=rng)  # type: ignore[call-arg]  # scipy>=1.14 floor predates the rng= kwarg
        A1 = (base @ base.T) + 50 * sp.diags(1 + rng.gamma(1, 1, n))
        A2 = (base @ base.T) + 80 * sp.diags(1 + rng.gamma(1, 1, n))
        return sp.csc_array(A1), sp.csc_array(A2)

    @pytest.mark.parametrize("solver", [SparseSolver.SUPERLU, SparseSolver.CHOLMOD])
    def test_refactorize_solve(self, spd_matrices, solver):
        A1, A2 = spd_matrices
        b = np.random.default_rng(0).standard_normal(A1.shape[0])

        fresh = create_sparse_factor(A2, solver=solver)
        refactored = create_sparse_factor(A1, solver=solver).refactorize(A2)

        assert_allclose(refactored.solve(b), fresh.solve(b), rtol=1e-12)

    @pytest.mark.parametrize("solver", [SparseSolver.SUPERLU, SparseSolver.CHOLMOD])
    def test_refactorize_logdet(self, spd_matrices, solver):
        A1, A2 = spd_matrices

        fresh = create_sparse_factor(A2, solver=solver)
        refactored = create_sparse_factor(A1, solver=solver).refactorize(A2)

        assert_allclose(refactored.logdet(), fresh.logdet(), rtol=1e-12)

    @pytest.mark.parametrize("solver", [SparseSolver.SUPERLU, SparseSolver.CHOLMOD])
    def test_refactorize_colorize(self, spd_matrices, solver):
        A1, A2 = spd_matrices
        z = np.random.default_rng(1).standard_normal(A1.shape[0])

        fresh = create_sparse_factor(A2, solver=solver)
        refactored = create_sparse_factor(A1, solver=solver).refactorize(A2)

        assert_allclose(refactored.colorize(z), fresh.colorize(z), rtol=1e-12)

    def test_superlu_refactorize_rejects_a_nonsymmetric_matrix(self):
        """``refactorize`` re-runs the factorization, so it repeats the
        symmetry check that ``create_sparse_factor`` makes."""
        factor = create_sparse_factor(
            sp.csc_array(np.eye(2) * 3.0), solver=SparseSolver.SUPERLU
        )
        with pytest.raises(ValueError, match="symmetric"):
            factor.refactorize(sp.csc_array(np.array([[0.0, 2.0], [1.0, 0.0]])))

    def test_cholmod_refactorize_returns_same_object(self, spd_matrices):
        A1, A2 = spd_matrices
        factor = create_sparse_factor(A1, solver=SparseSolver.CHOLMOD)
        refactored = factor.refactorize(A2)
        assert refactored is factor

    def test_cholmod_refactorize_updates_precision(self, spd_matrices):
        A1, A2 = spd_matrices
        factor = create_sparse_factor(A1, solver=SparseSolver.CHOLMOD)
        factor.refactorize(A2)
        assert factor._precision is A2

    def test_scaled_factor_refactorize_unwraps(self, spd_matrices):
        """ScaledSparseFactor.refactorize delegates to inner factor."""
        A1, A2 = spd_matrices
        factor = create_sparse_factor(A1, solver=SparseSolver.CHOLMOD)
        scaled = scale_factor(factor, 2.0)
        refactored = scaled.refactorize(A2)
        assert isinstance(refactored, CholmodSparseFactor)
        assert refactored is factor


class TestHalfSolve:
    """half_solve is the transpose of the colorize operator.

    If M is the colorize map (M @ M.T = inv(P)), half_solve applies M.T,
    so B = half_solve(X.T) satisfies B.T @ B = X @ inv(P) @ X.T -- the
    projected covariance, computed with one triangular solve per column
    against the cached factor and without ever forming inv(P).
    """

    D = 300

    @pytest.fixture(scope="class")
    def precision(self):
        rng = np.random.default_rng(42)
        mat = sp.random(self.D, self.D, 0.01, random_state=1) * 10  # type: ignore[call-arg]  # scipy>=1.14 floor predates the rng= kwarg
        return sp.csc_array((mat @ mat.T) + sp.diags(1 + rng.gamma(1, 1, self.D)))

    @pytest.fixture(scope="class")
    def X(self):
        return np.random.default_rng(7).standard_normal((8, self.D))

    def _reference(self, precision, X):
        return X @ np.linalg.solve(precision.toarray(), X.T)

    def _assert_gram_matches(self, B, S_ref):
        sd = np.sqrt(np.diag(S_ref))
        assert np.abs((B.T @ B - S_ref) / np.outer(sd, sd)).max() < 1e-10

    @pytest.mark.parametrize("solver", [SparseSolver.SUPERLU, SparseSolver.CHOLMOD])
    def test_sparse_gram_identity(self, precision, X, solver):
        factor = create_sparse_factor(precision, solver=solver)
        B = factor.half_solve(X.T)
        assert B.shape == (self.D, 8)
        self._assert_gram_matches(B, self._reference(precision, X))

    @pytest.mark.parametrize("solver", [SparseSolver.SUPERLU, SparseSolver.CHOLMOD])
    def test_scaled_factor_gram_identity(self, precision, X, solver):
        """A factor scaled by s represents s*P, so the projected
        covariance shrinks by 1/s."""
        factor = scale_factor(create_sparse_factor(precision, solver=solver), 0.7)
        B = factor.half_solve(X.T)
        self._assert_gram_matches(B, self._reference(precision, X) / 0.7)

    def test_dense_gram_identity(self, precision, X):
        from scipy.linalg import cholesky

        prec = precision.toarray()
        factor = DenseFactor(_U=cholesky(prec, lower=False), _n_features=self.D)
        B = factor.half_solve(X.T)
        assert B.shape == (self.D, 8)
        self._assert_gram_matches(B, self._reference(precision, X))

    def test_dense_colorize_accepts_a_vector(self, precision):
        """Every factor in the ``PrecisionFactor`` union takes a 1-D ``z``
        (``test_refactorize_colorize`` pins it for the sparse ones), so a
        caller dispatching on the union must not break on the dense one.
        ``dtrsm`` needs a 2-D right-hand side, hence the explicit reshape
        the sparse implementations get for free."""
        from scipy.linalg import cholesky, solve_triangular

        dense = DenseFactor(
            _U=cholesky(precision.toarray(), lower=False), _n_features=self.D
        )
        z = np.random.default_rng(3).standard_normal(self.D)

        got = dense.colorize(z)
        assert got.shape == (self.D,)
        assert_allclose(got, dense.colorize(z[:, None])[:, 0], rtol=1e-15)
        assert_allclose(got, solve_triangular(dense._U, z, lower=False), rtol=1e-10)

    def test_dense_trace_inv_rejects_a_singular_factor(self, precision):
        """``dtrtri`` reports singularity through ``info`` and leaves the
        triangle untouched, so an unchecked result is ``U`` itself and
        ``trace_inv`` a plausible number for a broken factor."""
        from scipy.linalg import LinAlgError, cholesky

        U = cholesky(precision.toarray(), lower=False)
        U[1, 1] = 0.0
        with pytest.raises(LinAlgError, match=r"U\[1, 1\]"):
            DenseFactor(_U=U, _n_features=self.D).trace_inv()

    @pytest.mark.parametrize("solver", [SparseSolver.SUPERLU, SparseSolver.CHOLMOD])
    def test_single_column_rhs_keeps_2d_shape(self, precision, X, solver):
        factor = create_sparse_factor(precision, solver=solver)
        B = factor.half_solve(X.T[:, :1])
        assert B.shape == (self.D, 1)
