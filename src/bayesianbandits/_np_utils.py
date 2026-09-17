from __future__ import annotations

from typing import Any, Generator, Optional, Tuple, TypeVar

import numpy as np
from numpy.typing import NDArray

_T = TypeVar("_T", bound=Any)


def groupby_array(
    *arrays: NDArray[_T], by: NDArray[Any]
) -> Generator[Tuple[NDArray[_T], ...], None, None]:
    """Group arrays by a given array.

    Parameters
    ----------
    *arrays : array-like
        Arrays to be grouped.
    by : array-like
        Array to group by.

    Yields
    ------
    array-like
        Grouped arrays.

    Examples
    --------
    >>> import numpy as np
    >>> from bayesianbandits._np_utils import groupby_array
    >>> X = np.array([[1, 2, 3], [4, 5, 6], [7, 8, 9]])
    >>> y = np.array([1, 2, 3])
    >>> for group in groupby_array(X, y, by=y):
    ...     print(group)
    (array([[1, 2, 3]]), array([1]))
    (array([[4, 5, 6]]), array([2]))
    (array([[7, 8, 9]]), array([3]))

    """
    sort_keys = np.argsort(by, kind="stable")
    sorted_by = by[sort_keys]
    sorted_arrays = [array[sort_keys] for array in arrays]

    group_indexes = np.unique(sorted_by, return_index=True)[1][1:]
    split_indexes = np.split(np.arange(len(sorted_by)), group_indexes)

    for split in split_indexes:
        yield tuple(array[split] for array in sorted_arrays)


def validated_sample_weight(
    n_samples: int, sample_weight: Optional[NDArray[Any]]
) -> NDArray[np.float64]:
    """``sample_weight`` as float64 of length ``n_samples``, ones if absent.

    A weight is how many observations a row counts as, so it has to be
    finite and non-negative; zero is fine and drops the row. Nothing
    downstream checks: the conjugate models add the weights straight
    onto a Dirichlet or Gamma concentration, where a negative one gives
    a parameter vector that is not a distribution, and the linear
    models square them into a precision through ``sqrt``, where a
    negative or NaN one gives a NaN coefficient. Both used to happen in
    silence.

    Examples
    --------
    >>> import numpy as np
    >>> from bayesianbandits._np_utils import validated_sample_weight
    >>> validated_sample_weight(3, None)
    array([1., 1., 1.])
    >>> validated_sample_weight(2, np.array([2.0, 0.0]))
    array([2., 0.])
    >>> validated_sample_weight(2, np.array([1.0, -1.0]))
    Traceback (most recent call last):
    ValueError: sample_weight must be finite and non-negative; got -1.0 at index 1.
    """
    if sample_weight is None:
        return np.ones(n_samples, dtype=np.float64)
    weights = np.asarray(sample_weight, dtype=np.float64)
    if weights.shape[0] != n_samples:
        raise ValueError(
            f"sample_weight.shape[0]={weights.shape[0]} should be "
            f"equal to n_samples={n_samples}"
        )
    bad = ~(weights >= 0.0) | np.isinf(weights)  # NaN fails the >= too
    if bad.any():
        index = int(np.argmax(bad))
        raise ValueError(
            "sample_weight must be finite and non-negative; got "
            f"{weights[index]} at index {index}."
        )
    return weights
