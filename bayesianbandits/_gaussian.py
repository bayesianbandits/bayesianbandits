from typing import Optional, Union, cast

import numpy as np
from numpy.typing import NDArray
from scipy.sparse import csc_array, csc_matrix
from scipy.sparse.linalg import splu

from ._sparse_bayesian_linear_regression import SparseSolver, solver


def compute_effective_weights(
    n_samples: int, sample_weight: Optional[NDArray[np.float64]], learning_rate: float
) -> NDArray[np.float64]:
    """Apply learning rate decay to sample weights."""
    if sample_weight is None:
        sample_weight = np.ones(n_samples, dtype=np.float64)
    else:
        sample_weight = np.asarray(sample_weight, dtype=np.float64)
        if sample_weight.shape[0] != n_samples:
            raise ValueError(
                f"sample_weight.shape[0]={sample_weight.shape[0]} should be "
                f"equal to n_samples={n_samples}"
            )

    if n_samples > 1:
        decay_factors = np.flip(np.power(learning_rate, np.arange(n_samples)))
        return cast(NDArray[np.float64], sample_weight * decay_factors)
    else:
        return sample_weight


def solve_precision_weighted_mean(
    precision: Union[NDArray[np.float64], csc_array],
    eta: NDArray[np.float64],
    sparse: bool,
) -> NDArray[np.float64]:
    """Solve precision @ mu = eta for mu."""
    if sparse:
        if solver == SparseSolver.CHOLMOD:
            from sksparse.cholmod import cholesky as cholmod_cholesky

            return cholmod_cholesky(csc_matrix(precision))(eta)
        else:
            lu = splu(
                precision,
                # These two settings tell SuperLU that we're decomposing a Hermitian
                # positive-definite matrix, so we only want to pivot on the diagonal.
                # This preserves the sparsity of the matrix better than the default,
                # which allows for off-diagonal pivoting. See SuperLU User Guide
                # for more details.
                diag_pivot_thresh=0.0,
                permc_spec="MMD_AT_PLUS_A",
                options=dict(SymmetricMode=True),
            )
            return lu.solve(eta)
    else:
        from scipy.linalg import solve

        return solve(precision, eta, check_finite=False, assume_a="pos")
