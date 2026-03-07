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
    SparseSolver,
    create_sparse_factor,
    multivariate_normal_sample_from_sparse_precision,
    multivariate_t_sample_from_sparse_precision,
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

    def test_umfpack_and_superlu_errors_when_not_symmetric_and_positive_definite(
        self,
    ):
        matrix = np.array([[0.0, 2.0], [1.0, 0.0]])
        with pytest.raises(ValueError):
            create_sparse_factor(sp.csc_array(matrix), solver=SparseSolver.SUPERLU)

    @pytest.mark.parametrize("solver", [SparseSolver.SUPERLU, SparseSolver.CHOLMOD])
    def test_mvn_sampling_against_scipy(self, precision_matrix, solver):
        factor = create_sparse_factor(
            sp.csc_array(precision_matrix), solver=solver
        )
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
            mock_mvn.side_effect = (
                lambda mean,
                shape,
                size,
                random_state: random_state.multivariate_normal(
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

    @pytest.mark.parametrize("solver", [SparseSolver.SUPERLU, SparseSolver.CHOLMOD])
    def test_mvt_sampling_against_scipy(self, precision_matrix, solver):
        factor = create_sparse_factor(
            sp.csc_array(precision_matrix), solver=solver
        )
        scipy_cov = Covariance.from_precision(precision_matrix)

        sparse_samples = multivariate_t_sample_from_sparse_precision(
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
