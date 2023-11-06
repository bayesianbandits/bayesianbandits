from __future__ import annotations

from typing import Any, Generator, Tuple, TypeVar

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
