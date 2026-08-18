"""Byte-level accounting for the state a bandit keeps between rounds.

A fitted agent's memory is spread across Python attributes, NumPy
buffers, SciPy sparse triples, and -- when a model is sparse -- a
factorization that lives entirely inside a C library. ``sys.getsizeof``
sees only the first of those: it reports the header of a ``csc_array``
and nothing about its data, and 144 bytes for a SuperLU object holding
tens of megabytes. This module walks the object graph instead, charging
each buffer once to whichever holder reaches it first.

The public surface is :func:`memory_usage`, which any object can be
passed to, and :class:`MemoryUsageMixin`, which gives a class a
``memory_usage`` property over its own state.

Notes
-----
**What is counted.** Instance state only: ``__dict__`` (which is where
``functools.cached_property`` puts its results, so a factorization that
has been built is reported and one that has not costs nothing), slots,
container elements, and closure cells. Classes and modules are not
counted -- they are shared by the whole process, not owned by the
object.

**Shared buffers are charged once.** A precision matrix is referenced
both by the estimator and by the factor built from it; a
:class:`~bayesianbandits.LipschitzContextualAgent` gives every arm the
same learner; :func:`~bayesianbandits._sparse_bayesian_linear_regression.scale_factor`
makes shallow copies that share one numeric factorization. Every one of
those is reported at its first sighting and as ``0`` afterwards, so a
total is the memory that would be freed by dropping the object, not the
sum over its references.

**Views are charged for their buffer.** A slice keeps its base array
alive, so ``arr[:10]`` from a 100 MB buffer is charged 100 MB -- the
memory that is actually held, not the part being looked at.

**C factorizations are computed from their own array lengths.** Neither
CHOLMOD nor SuperLU reports its allocation to Python -- both track it
internally, and neither wrapper surfaces the counter -- so each is
priced from the structure counts it does expose.

For CHOLMOD that is exact: scikit-sparse finalizes every factor to
packed simplicial form, which fixes each array's length in terms of
``nnz`` and ``N``. For SuperLU it is a lower bound, because SciPy
exposes only the logical nonzero count and SuperLU stores ``L`` as
dense supernode blocks that cover more than that -- 10-20% more on the
factorizations this library builds. Those reports carry ``exact=False``
and say so in :attr:`MemoryUsage.note`.
"""

from __future__ import annotations

import sys
from collections.abc import Mapping as AbcMapping
from dataclasses import dataclass, field
from types import BuiltinFunctionType, FunctionType, MethodType, ModuleType
from typing import Any, Dict, Iterable, Iterator, Mapping, Tuple

import numpy as np
from scipy.sparse import issparse

__all__ = ["MemoryUsage", "MemoryUsageMixin", "memory_usage"]


_UNITS = ("B", "KiB", "MiB", "GiB", "TiB", "PiB")


def _human(n_bytes: int) -> str:
    """Format a byte count the way a memory tool should read: exact
    below a kibibyte, one decimal above it."""
    size = float(n_bytes)
    for unit in _UNITS:
        if abs(size) < 1024.0 or unit == _UNITS[-1]:
            return f"{int(size)} {unit}" if unit == "B" else f"{size:.1f} {unit}"
        size /= 1024.0
    raise AssertionError("unreachable")  # pragma: no cover


@dataclass(frozen=True)
class MemoryUsage:
    """Bytes held by an object, and by each of its parts.

    Attributes
    ----------
    total : int
        Bytes retained, including everything in :attr:`parts`.
    exact : bool
        False if any component was estimated rather than measured --
        see the module notes on CHOLMOD and SuperLU. An estimate
        anywhere makes every total containing it inexact.
    parts : Mapping[str, MemoryUsage]
        Breakdown by attribute name, container index, or dictionary key.
        Empty for a leaf.
    note : str
        Why a number is what it is, when that is not obvious: the buffer
        a view holds alive, the second sighting of a shared array, the
        storage format behind an estimate.

    Examples
    --------
    >>> usage = MemoryUsage(
    ...     1536,
    ...     parts={"coef_": MemoryUsage(512), "cov_inv_": MemoryUsage(1024)},
    ... )
    >>> usage.total
    1536
    >>> print(usage)
    1.5 KiB
    ├── cov_inv_: 1.0 KiB
    └── coef_: 512 B

    Sizes are plain integers, so they compose with ordinary arithmetic:

    >>> int(usage) // 1024
    1
    """

    total: int
    exact: bool = True
    parts: Mapping[str, "MemoryUsage"] = field(default_factory=dict)
    note: str = ""

    def __int__(self) -> int:
        return self.total

    def __repr__(self) -> str:
        estimate = "" if self.exact else ", estimated"
        return f"MemoryUsage({_human(self.total)}{estimate})"

    def __str__(self) -> str:
        return self.report()

    def report(self, max_depth: int = 3, min_bytes: int = 0) -> str:
        """Render the breakdown as an indented tree, largest part first.

        Parameters
        ----------
        max_depth : int, default=3
            Levels of nesting to show. Deeper parts are summed into
            their ancestor's total, which is always the full figure --
            trimming the tree never changes a number.
        min_bytes : int, default=0
            Hide parts smaller than this. Useful on an agent, where the
            interesting rows are the posteriors and the noise is a few
            dozen bytes of hyperparameters per arm.

        Examples
        --------
        >>> usage = MemoryUsage(
        ...     2048,
        ...     parts={
        ...         "big": MemoryUsage(2000, parts={"inner": MemoryUsage(2000)}),
        ...         "small": MemoryUsage(48),
        ...     },
        ... )
        >>> print(usage.report(max_depth=1))
        2.0 KiB
        ├── big: 2.0 KiB
        └── small: 48 B
        >>> print(usage.report(min_bytes=100))
        2.0 KiB
        └── big: 2.0 KiB
            └── inner: 2.0 KiB
        """
        header = _human(self.total)
        if not self.exact:
            header += " (estimated)"
        if self.note:
            header += f"  [{self.note}]"
        lines = [header]
        lines.extend(self._rows(max_depth, min_bytes, prefix=""))
        return "\n".join(lines)

    def _rows(self, depth: int, min_bytes: int, prefix: str) -> Iterator[str]:
        if depth <= 0:
            return
        shown = [
            (name, part)
            for name, part in sorted(self.parts.items(), key=lambda kv: -kv[1].total)
            if part.total >= min_bytes
        ]
        for i, (name, part) in enumerate(shown):
            last = i == len(shown) - 1
            label = f"{name}: {_human(part.total)}"
            if part.note:
                label += f"  [{part.note}]"
            yield f"{prefix}{'└── ' if last else '├── '}{label}"
            yield from part._rows(
                depth - 1, min_bytes, prefix + ("    " if last else "│   ")
            )


class MemoryUsageMixin:
    """Gives a class a ``memory_usage`` property over its own state.

    Examples
    --------
    >>> import numpy as np
    >>> from bayesianbandits import NormalInverseGammaRegressor
    >>> est = NormalInverseGammaRegressor()
    >>> _ = est.fit(np.array([[1.0, 0.0], [0.0, 1.0]]), np.array([1.0, 2.0]))
    >>> est.memory_usage.parts["coef_"].total
    16

    The property is recomputed on access and never touches the data it
    measures, so it is safe to call in a hot loop or a metrics scrape.
    """

    __slots__ = ()

    @property
    def memory_usage(self) -> MemoryUsage:
        """Bytes retained by this object; see :class:`MemoryUsage`."""
        return memory_usage(self)


def memory_usage(obj: Any) -> MemoryUsage:
    """Bytes retained by ``obj`` and everything it holds alive.

    Parameters
    ----------
    obj : Any
        Any object. Estimators, agents, factors, arrays, sparse
        matrices, and plain containers are all understood; anything
        else falls back to its header size and is reported inexact.

    Returns
    -------
    MemoryUsage
        Total and per-part breakdown.

    Examples
    --------
    >>> import numpy as np
    >>> from bayesianbandits import memory_usage
    >>> memory_usage(np.zeros((100, 100))).total
    80000

    A buffer reachable twice is charged once, so a total is what
    dropping the object would free:

    >>> arr = np.zeros(1000)
    >>> memory_usage([arr, arr]).parts["[1]"].note
    'shared; counted at first sighting'
    """
    return _Accountant().measure(obj)


# Objects whose ``sys.getsizeof`` is the whole truth: no referents worth
# following, no hidden allocation behind a C pointer.
_LEAF_TYPES: Tuple[type, ...] = (
    bool,
    bytearray,
    bytes,
    complex,
    float,
    int,
    str,
    type(None),
    np.dtype,
    np.generic,
)

_SEQUENCE_TYPES: Tuple[type, ...] = (list, tuple, set, frozenset)

_UNCOUNTED_TYPES: Tuple[type, ...] = (ModuleType, type, BuiltinFunctionType)

# Immutables Python may hand out as one shared object; see ``_claim``.
_INTERNED_TYPES: Tuple[type, ...] = (bool, complex, float, int, type(None))


class _Accountant:
    """One walk of an object graph.

    Keeps a reference to everything it has seen, both to charge shared
    buffers once and to keep ``id`` from being recycled underneath the
    ledger by an intermediate that the walk itself dropped.
    """

    def __init__(self) -> None:
        self._seen: Dict[int, Any] = {}

    def measure(self, obj: Any) -> MemoryUsage:
        if self._claim(obj) is False:
            return MemoryUsage(0, note="shared; counted at first sighting")
        if isinstance(obj, np.ndarray):
            return self._ndarray(obj)
        if issparse(obj):
            return self._composite(
                obj,
                _vars_items(obj),
                overhead=_instance_overhead(obj),
                note="sparse array",
            )
        if _is_cholmod_factor(obj):
            return _cholmod_usage(obj)
        if _is_superlu(obj):
            return _superlu_usage(obj)
        if isinstance(obj, _LEAF_TYPES):
            return MemoryUsage(sys.getsizeof(obj))
        if isinstance(obj, _UNCOUNTED_TYPES):
            return MemoryUsage(0, note="shared by the process; not counted")
        if isinstance(obj, np.random.Generator):
            return self._composite(obj, [("bit_generator", obj.bit_generator)])
        if isinstance(obj, np.random.BitGenerator):
            return self._composite(obj, [("state", obj.state)])
        if isinstance(obj, AbcMapping):
            return self._composite(obj, _mapping_items(obj))
        if isinstance(obj, _SEQUENCE_TYPES):
            return self._composite(obj, _sequence_items(obj))
        if isinstance(obj, (FunctionType, MethodType)):
            return self._composite(obj, _callable_items(obj))
        state = list(_state_items(obj))
        if state or hasattr(obj, "__dict__"):
            return self._composite(obj, state, overhead=_instance_overhead(obj))
        return MemoryUsage(
            sys.getsizeof(obj), exact=False, note="opaque object; header only"
        )

    def _claim(self, obj: Any) -> bool:
        """Record ``obj`` as counted, returning False if it already was.

        Small immutables are exempt: Python interns them, so the same
        ``0`` or ``1.0`` really is one object across a dozen unrelated
        hyperparameters, and deduplicating those would fill a report
        with 0-byte rows that say "shared" about nothing.
        """
        if isinstance(obj, _INTERNED_TYPES):
            return True
        if id(obj) in self._seen:
            return False
        self._seen[id(obj)] = obj
        return True

    def _composite(
        self,
        obj: Any,
        items: Iterable[Tuple[str, Any]],
        overhead: int = 0,
        note: str = "",
    ) -> MemoryUsage:
        """Sum an object's own footprint and its named parts."""
        total = sys.getsizeof(obj) if overhead == 0 else overhead
        exact = True
        parts: Dict[str, MemoryUsage] = {}
        for name, value in items:
            part = self.measure(value)
            parts[name] = part
            total += part.total
            exact &= part.exact
        return MemoryUsage(total, exact, parts, note)

    def _ndarray(self, arr: np.ndarray) -> MemoryUsage:
        """An array is charged for the buffer it keeps alive.

        A view's ``nbytes`` is the window, not the allocation: the base
        stays alive as long as the view does, so the base is what a
        memory report has to name. When the base is not an array at all
        -- CHOLMOD and SuperLU both hand out read-only views onto their
        own storage -- the bytes belong to that C object and are counted
        there instead.
        """
        base = arr.base
        if base is None:
            return MemoryUsage(arr.nbytes)
        if isinstance(base, np.ndarray):
            owner = self.measure(base)
            if owner.total <= arr.nbytes:
                # Nothing surprising: a reshape, a transpose, or a base
                # already charged elsewhere. Report it without comment.
                return owner
            return MemoryUsage(
                owner.total,
                owner.exact,
                note=f"view holding a {_human(owner.total)} buffer alive",
            )
        held = self.measure(base)
        return MemoryUsage(
            held.total,
            held.exact,
            note="view of memory owned by " + type(base).__name__,
        )


def _instance_overhead(obj: Any) -> int:
    """Header plus attribute dictionary, without its contents."""
    overhead = sys.getsizeof(obj)
    instance_dict = getattr(obj, "__dict__", None)
    if instance_dict is not None:
        overhead += sys.getsizeof(instance_dict)
    return overhead


def _vars_items(obj: Any) -> Iterator[Tuple[str, Any]]:
    yield from vars(obj).items()


def _mapping_items(obj: Mapping[Any, Any]) -> Iterator[Tuple[str, Any]]:
    for key, value in obj.items():
        yield f"[{key!r}]", value


def _sequence_items(obj: Iterable[Any]) -> Iterator[Tuple[str, Any]]:
    for i, value in enumerate(obj):
        yield f"[{i}]", value


def _callable_items(obj: Any) -> Iterator[Tuple[str, Any]]:
    """A reward function's closure can hold a lookup table; follow it."""
    func = obj.__func__ if isinstance(obj, MethodType) else obj
    if isinstance(obj, MethodType):
        yield "__self__", obj.__self__
    for i, cell in enumerate(func.__closure__ or ()):
        try:
            yield f"closure[{i}]", cell.cell_contents
        except ValueError:  # pragma: no cover - empty cell in a recursive def
            continue
    for i, default in enumerate(func.__defaults__ or ()):
        yield f"default[{i}]", default


def _state_items(obj: Any) -> Iterator[Tuple[str, Any]]:
    """Instance attributes, from ``__dict__`` and from slots.

    Read out of ``__dict__`` rather than through ``getattr``, so that a
    ``cached_property`` that has never been evaluated is not evaluated
    here: measuring a model must not build the factorization it is
    measuring.
    """
    yield from getattr(obj, "__dict__", {}).items()
    seen: set[str] = set()
    for klass in type(obj).__mro__:
        for name in getattr(klass, "__slots__", ()) or ():
            if name in ("__dict__", "__weakref__") or name in seen:
                continue
            seen.add(name)
            try:
                yield name, getattr(obj, name)
            except AttributeError:
                continue


def _is_cholmod_factor(obj: Any) -> bool:
    """A ``sksparse.cholmod.CholeskyFactor``, duck-typed.

    Imported nowhere in this module: scikit-sparse is optional, and the
    check has to work whether or not it is installed.
    """
    return type(obj).__module__.startswith("sksparse") and all(
        hasattr(obj, attr) for attr in ("N", "nnz", "itype", "is_super")
    )


def _is_superlu(obj: Any) -> bool:
    """A ``scipy.sparse.linalg.SuperLU``, duck-typed. The class is not
    importable from SciPy's public API, so match on the object."""
    return type(obj).__name__ == "SuperLU" and all(
        hasattr(obj, attr) for attr in ("nnz", "shape", "perm_r")
    )


def _cholmod_usage(factor: Any) -> MemoryUsage:
    """Bytes CHOLMOD holds for a factor, computed from its own array
    lengths.

    Every array in a ``cholmod_factor`` has a length scikit-sparse
    exposes: ``x`` and ``i`` hold ``nzmax`` entries, ``p`` holds
    ``n + 1``, and ``Perm`` and ``ColCount`` hold ``n`` each. The one
    length not exposed directly is ``nzmax``, and for a *packed* factor
    it equals the nonzero count -- which ``nnz`` reports, summing
    ``ColCount``.

    That covers every factor scikit-sparse returns.
    :meth:`CholeskyFactor.factorize` sets ``final_super = False``,
    ``final_pack = True`` and ``final_monotonic = True`` before handing
    the factor back, so the result is packed and simplicial whatever
    ordering or supernodal mode produced it -- ``nzmax == nnz``, and
    this figure is exact rather than modelled. A factor from somewhere
    else that is still supernodal stores dense blocks covering more
    than ``nnz`` entries; that case is detected and reported as a lower
    bound.

    Excluded: the scratch workspace on the ``cholmod_common`` that
    scikit-sparse gives each factor. It is scratch rather than factor
    state, and runs to tens of bytes per feature against this figure's
    twelve per nonzero.
    """
    n = int(factor.N)
    nnz = int(factor.nnz)
    value_size = np.dtype(factor.dtype).itemsize
    index_size = np.dtype(factor.itype).itemsize
    # x[nzmax], i[nzmax], p[n + 1], Perm[n], ColCount[n]
    values = nnz * value_size
    indices = nnz * index_size
    structure = (3 * n + 1) * index_size
    supernodal = bool(factor.is_super)
    exact = not supernodal
    note = f"CHOLMOD factor, {nnz} nonzeros; excludes CHOLMOD scratch workspace"
    if supernodal:
        note = (
            f"CHOLMOD factor, {nnz} nonzeros, supernodal; "
            "lower bound, dense blocks store more than nnz"
        )
    return MemoryUsage(
        values + indices + structure,
        exact=exact,
        parts={
            "values": MemoryUsage(values, exact=exact),
            "indices": MemoryUsage(indices, exact=exact),
            "structure": MemoryUsage(structure, exact=exact),
        },
        note=note,
    )


def _superlu_usage(lu: Any) -> MemoryUsage:
    """Lower bound on the memory behind a SuperLU decomposition.

    This one really is a bound, not a measurement, and SciPy is the
    reason. ``nnz`` -- the nonzeros of ``L`` and ``U`` together -- is
    the only size the ``SuperLU`` object exposes; ``sys.getsizeof``
    returns its 144-byte header however large the factorization is, and
    ``lu.L`` builds a fresh copy rather than describing the one already
    there. SuperLU itself knows the answer exactly, in the
    ``mem_usage.for_lu`` that ``dQuerySpace`` fills in, but SciPy wraps
    neither that call nor the ``SCformat`` array lengths behind it.

    So the arrays are priced at what the logical nonzeros need: float64
    values, int32 row indices, and the six length-``n`` vectors that the
    supernodal ``L``, the compressed-column ``U``, and the two
    permutations keep between them. SuperLU stores ``L`` as dense
    supernode blocks, which cover more than the logical nonzeros, so the
    true figure runs above this one -- 10-20% above on the sparse
    symmetric factorizations this library builds, and further above as
    supernodes grow denser.
    """
    n = int(lu.shape[0])
    nnz = int(lu.nnz)
    values = nnz * 8
    indices = nnz * 4
    structure = 6 * (n + 1) * 4
    return MemoryUsage(
        values + indices + structure,
        exact=False,
        parts={
            "values": MemoryUsage(values, exact=False),
            "indices": MemoryUsage(indices, exact=False),
            "structure": MemoryUsage(structure, exact=False),
        },
        note=(
            f"SuperLU factorization, {nnz} nonzeros in L and U; "
            "lower bound, supernode padding not exposed by SciPy"
        ),
    )
