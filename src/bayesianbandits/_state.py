"""Plain-data state for ``state_dict`` and ``load_state_dict``.

A state describes a posterior, not the object holding it: a ``family``
(``gaussian``, ``dirichlet`` or ``gamma``) and blocks such as ``prior``
and ``posterior``, each with its own version ``v``, in dicts, lists,
numpy arrays and Python scalars. No class or attribute names appear, so
any estimator of a family loads a state another wrote, and a change to
one block bumps only that block's version. Construction (forgetting
rules, policies, reward functions, and hyperparameters empirical Bayes
does not tune) stays in code, and caches such as factorizations are
rebuilt on use.
"""

from __future__ import annotations

import numbers
import operator
from collections.abc import Collection, Mapping
from typing import Any, Dict, List, Optional, Tuple, cast

import numpy as np
from numpy.typing import NDArray
from scipy.sparse import csc_array, issparse

_CSC_KEYS = frozenset({"data", "indices", "indptr", "shape"})
_INT32_MAX = np.iinfo(np.int32).max
_SYMMETRY_RTOL = 1e-9


def versioned(version: int, **fields: Any) -> Dict[str, Any]:
    return {"v": version, **fields}


def check_block(
    value: Any, version: int, keys: Collection[str], where: str
) -> Mapping[str, Any]:
    """Raise unless ``value`` is a block at ``version``, the only one this
    release reads, with exactly ``keys`` besides ``v``."""
    if not isinstance(value, Mapping):
        raise TypeError(f"{where} must be a dict, not {type(value).__name__}.")
    if value.get("v") != version:
        raise ValueError(
            f"{where} is version {value.get('v')!r}; this release reads "
            f"version {version}."
        )
    present = set(value) - {"v"}
    missing = sorted(set(keys) - present)
    unexpected = sorted(present - set(keys), key=repr)
    if missing or unexpected:
        raise ValueError(
            f"{where} does not match: missing keys {missing}, unexpected keys "
            f"{unexpected}."
        )
    return value


def check_state(
    state: Any,
    family: str,
    owner: str,
    *,
    required: Collection[str],
    optional: Collection[str],
) -> Dict[str, Any]:
    """The blocks of a ``family`` state, ``None`` for absent optional
    ones. Raises if the state is of another family or has a block this
    family does not write."""
    if not isinstance(state, Mapping):
        raise TypeError(f"{owner} state must be a dict, not {type(state).__name__}.")
    if state.get("family") != family:
        raise ValueError(f"{owner} reads {family} states, not {state.get('family')!r}.")
    blocks = set(state) - {"family"}
    missing = sorted(set(required) - blocks)
    unexpected = sorted(blocks - set(required) - set(optional), key=repr)
    if missing or unexpected:
        raise ValueError(
            f"{owner} state does not match: missing blocks {missing}, "
            f"unexpected blocks {unexpected}."
        )
    return {name: state.get(name) for name in (*required, *optional)}


def check_tokens(found: Collection[Any], expected: Collection[Any], owner: str) -> None:
    """Raise unless the arm tokens of a state are those of the agent."""
    missing = [t for t in expected if t not in found]
    unexpected = [t for t in found if t not in expected]
    if missing or unexpected:
        raise ValueError(
            f"{owner} state does not match its arms: missing tokens {missing}, "
            f"unexpected tokens {unexpected}."
        )


def load_float(value: Any, where: str) -> float:
    if isinstance(value, (bool, np.bool_)) or not isinstance(value, numbers.Real):
        raise TypeError(f"{where} must be a number, not {value!r}.")
    return float(value)


def load_int(value: Any, where: str) -> int:
    if isinstance(value, (bool, np.bool_)):
        raise TypeError(f"{where} must be an integer, not {value!r}.")
    try:
        return operator.index(value)
    except TypeError:
        raise TypeError(f"{where} must be an integer, not {value!r}.") from None


def load_finite(value: Any, where: str, *, nonnegative: bool = False) -> float:
    number = load_float(value, where)
    if not np.isfinite(number) or (nonnegative and number < 0):
        kind = "non-negative and finite" if nonnegative else "finite"
        raise ValueError(f"{where} must be {kind}, not {number!r}.")
    return number


def load_positive(value: Any, where: str) -> float:
    number = load_float(value, where)
    if not 0.0 < number < np.inf:
        raise ValueError(f"{where} must be positive and finite, not {number!r}.")
    return number


def load_bool(value: Any, where: str) -> bool:
    if not isinstance(value, (bool, np.bool_)):
        raise TypeError(f"{where} must be a bool, not {value!r}.")
    return bool(value)


def load_optional(value: Any, load: Any, where: str) -> Any:
    """``load(value, where)``, or ``None`` for a value not set."""
    return None if value is None else load(value, where)


def optional_float(value: Any) -> Optional[float]:
    return None if value is None else float(value)


def load_array(
    value: Any, shape: Tuple[int, ...], where: str, *, positive: bool = False
) -> NDArray[np.float64]:
    """A finite float64 copy of ``value``, which must have ``shape``;
    ``-1`` leaves an axis free."""
    array = np.array(value, dtype=np.float64, order="F" if len(shape) == 2 else "K")
    if array.ndim != len(shape) or any(
        want not in (-1, got) for want, got in zip(shape, array.shape)
    ):
        expected = tuple("n" if n == -1 else n for n in shape)
        raise ValueError(f"{where} must have shape {expected}, not {array.shape}.")
    if not np.isfinite(array).all():
        raise ValueError(f"{where} must be finite.")
    if positive and not (array > 0).all():
        raise ValueError(f"{where} must be positive.")
    return array


def load_keys(value: Any, where: str) -> List[Any]:
    """A list of distinct group keys or class labels."""
    if isinstance(value, (str, bytes, Mapping)) or not isinstance(value, Collection):
        raise TypeError(f"{where} must be a list, not {type(value).__name__}.")
    keys = list(value)
    if len(set(keys)) != len(keys):
        raise ValueError(f"{where} must not repeat a key.")
    return keys


def precision_state(precision: Any) -> Any:
    """A precision in canonical form. A dense one is a symmetric array:
    the estimators keep only its upper triangle current, so that is
    mirrored down. A sparse one is its CSC ``data``, ``indices``,
    ``indptr`` and ``shape``, indices sorted and int32 where they fit; its
    values are kept as held, symmetric to rounding, since changing them
    would change its factorization."""
    if not issparse(precision):
        upper = np.triu(np.asarray(precision, dtype=np.float64))
        return upper + np.triu(upper, 1).T
    csc = csc_array(precision).sorted_indices()
    fits = max(csc.nnz, *cast("tuple[int, int]", csc.shape)) <= _INT32_MAX
    index = np.int32 if fits else np.int64
    return {
        "data": csc.data.astype(np.float64),
        "indices": csc.indices.astype(index),
        "indptr": csc.indptr.astype(index),
        "shape": [int(n) for n in cast("tuple[int, int]", csc.shape)],
    }


def load_precision(value: Any, sparse: bool, where: str) -> Any:
    """The square precision ``value`` describes, in the form the
    estimator keeps it whichever form the state holds: a Fortran-ordered
    array when dense, a ``csc_array`` when sparse. A sparse structure is
    checked in full before use."""
    if isinstance(value, Mapping):
        precision: Any = _load_csc(value, where)
        asymmetry = abs(precision - precision.T).max() if precision.nnz else 0.0
        scale = abs(precision).max() if precision.nnz else 0.0
    else:
        precision = load_array(value, (-1, -1), where)
        if precision.shape[0] != precision.shape[1]:
            raise ValueError(f"{where} must be square, not {precision.shape}.")
        asymmetry = np.abs(precision - precision.T).max(initial=0.0)
        scale = np.abs(precision).max(initial=0.0)
    # Sparse updates leave the precision symmetric only to rounding
    if asymmetry > _SYMMETRY_RTOL * scale:
        raise ValueError(f"{where} must be symmetric.")
    if isinstance(value, Mapping):
        return precision if sparse else np.asfortranarray(precision.toarray())
    return csc_array(precision) if sparse else precision


def _load_csc(value: Mapping[str, Any], where: str) -> csc_array:
    if set(value) != _CSC_KEYS:
        raise ValueError(f"{where} must have exactly the keys {sorted(_CSC_KEYS)}.")
    shape = tuple(load_int(m, f"{where} shape") for m in value["shape"])
    if len(shape) != 2 or shape[0] != shape[1]:
        raise ValueError(f"{where} must be square, not {shape}.")
    indices = np.array(value["indices"])
    indptr = np.array(value["indptr"])
    for name, index in (("indices", indices), ("indptr", indptr)):
        if index.ndim != 1 or not np.issubdtype(index.dtype, np.integer):
            raise ValueError(f"{where} {name} must be a 1-d integer array.")
    data = load_array(value["data"], (-1,), f"{where} data")
    precision = csc_array((data, indices, indptr), shape=shape)
    precision.check_format(full_check=True)
    return precision


def discard(obj: Any, names: Collection[str]) -> None:
    """Remove instance attributes, where set."""
    for name in names:
        obj.__dict__.pop(name, None)


#: How empirical Bayes reports its last tuning, as ``(eb block key,
#: attribute, loader)``; ``None`` in a state for what is not set yet
REPORT_FIELDS: Tuple[Tuple[str, str, Any], ...] = (
    ("log_evidence", "log_evidence_", load_float),
    ("iterations", "n_eb_iterations_", load_int),
    ("converged", "eb_converged_", load_bool),
)


REPORT_KEYS = tuple(key for key, _, _ in REPORT_FIELDS)


def report_state(obj: Any) -> Dict[str, Any]:
    return {
        key: load_optional(obj.__dict__.get(attribute), load, key)
        for key, attribute, load in REPORT_FIELDS
    }


def load_report(value: Mapping[str, Any], where: str) -> Dict[str, Any]:
    return {
        key: load_optional(value[key], load, f"{where} {key}")
        for key, _, load in REPORT_FIELDS
    }


def write_report(obj: Any, report: Optional[Mapping[str, Any]]) -> None:
    discard(obj, [attribute for _, attribute, _ in REPORT_FIELDS])
    for key, attribute, _ in REPORT_FIELDS:
        if report is not None and report[key] is not None:
            obj.__dict__[attribute] = report[key]
