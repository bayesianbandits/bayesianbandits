"""Plain-data state for ``state_dict`` and ``load_state_dict``.

A state holds what fitting changed -- posteriors, tuned hyperparameters,
running statistics -- as dicts, lists, numpy arrays and Python scalars,
so it can be stored without pickle. Construction (priors, forgetting
rules, policies, reward functions) stays in code: a state loads into an
object built the same way. Caches such as factorizations are rebuilt on
use. Every state carries a ``version``, and a loader accepts only the
version it writes, so a format change fails loudly.
"""

from __future__ import annotations

import numbers
import operator
from collections.abc import Collection, Mapping
from typing import (
    Any,
    Callable,
    Dict,
    Hashable,
    List,
    Optional,
    Tuple,
    cast,
)

import numpy as np
from numpy.typing import NDArray
from scipy.sparse import csc_array, issparse

#: The version every learner and agent state is written at.
STATE_VERSION = 1

_CSC_KEYS = frozenset({"data", "indices", "indptr", "shape"})
_TABLE_KEYS = frozenset({"keys", "values"})


def check_state(
    state: Any,
    keys: Collection[str],
    owner: str,
    *,
    unfitted_keys: Optional[Collection[str]] = None,
) -> bool:
    """Raise unless ``state`` is a mapping at :data:`STATE_VERSION` with
    exactly ``keys`` besides ``version``, or, when given, exactly
    ``unfitted_keys``: the state of a learner whose prior was never
    initialized. Returns whether it has ``keys``."""
    if not isinstance(state, Mapping):
        raise TypeError(f"{owner} state must be a dict, not {type(state).__name__}.")
    version = state.get("version")
    if version != STATE_VERSION:
        raise ValueError(
            f"{owner} reads state version {STATE_VERSION}, not {version!r}."
        )
    present = set(state) - {"version"}
    if unfitted_keys is not None and present == set(unfitted_keys):
        return False
    missing = sorted(set(keys) - present)
    unexpected = sorted(present - set(keys), key=repr)
    if missing or unexpected:
        raise ValueError(
            f"{owner} state does not match: missing keys {missing}, "
            f"unexpected keys {unexpected}."
        )
    return True


def check_tokens(found: Collection[Any], expected: Collection[Any], owner: str) -> None:
    """Raise unless the arm tokens of a state are those of the agent."""
    missing = [t for t in expected if t not in found]
    unexpected = [t for t in found if t not in expected]
    if missing or unexpected:
        raise ValueError(
            f"{owner} state does not match its arms: missing tokens {missing}, "
            f"unexpected tokens {unexpected}."
        )


def check_together(
    restored: Mapping[str, Any], keys: Collection[str], owner: str
) -> None:
    """Raise unless ``keys`` are all set or all ``None``: statistics a fit
    starts together, and that a later update reads together."""
    unset = [key for key in keys if restored[key] is None]
    if unset and len(unset) != len(keys):
        raise ValueError(
            f"{owner} state sets some of {sorted(keys)} but not {sorted(unset)}; "
            "a fit sets them together."
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


def load_bool(value: Any, where: str) -> bool:
    if not isinstance(value, (bool, np.bool_)):
        raise TypeError(f"{where} must be a bool, not {value!r}.")
    return bool(value)


def load_optional(value: Any, load: Any, where: str) -> Any:
    """``load(value, where)``, or ``None`` for an attribute the
    estimator had not set yet."""
    return None if value is None else load(value, where)


def load_array(value: Any, shape: Tuple[int, ...], where: str) -> NDArray[np.float64]:
    """A float64 copy of ``value``, which must have ``shape``; ``-1``
    leaves an axis free."""
    array = np.array(value, dtype=np.float64, order="F" if len(shape) == 2 else "K")
    if array.ndim != len(shape) or any(
        want not in (-1, got) for want, got in zip(shape, array.shape)
    ):
        expected = tuple("n" if n == -1 else n for n in shape)
        raise ValueError(f"{where} must have shape {expected}, not {array.shape}.")
    return array


def precision_state(precision: Any) -> Any:
    """A copy of a precision matrix: an array when dense, its CSC
    ``data``, ``indices``, ``indptr`` and ``shape`` when sparse."""
    if not issparse(precision):
        return np.array(precision, dtype=np.float64, order="K")
    csc = csc_array(precision)
    return {
        "data": csc.data.copy(),
        "indices": csc.indices.copy(),
        "indptr": csc.indptr.copy(),
        "shape": tuple(int(n) for n in cast("tuple[int, int]", csc.shape)),
    }


def load_precision(value: Any, sparse: bool, where: str) -> Any:
    """The square precision ``value`` describes, in the form the
    estimator keeps it: a Fortran-ordered array when dense, a
    ``csc_array`` when sparse. The sparse structure is checked in full
    before use."""
    if not sparse:
        if isinstance(value, Mapping):
            raise TypeError(f"{where} is sparse, but the estimator is dense.")
        precision = load_array(value, (-1, -1), where)
        if precision.shape[0] != precision.shape[1]:
            raise ValueError(f"{where} must be square, not {precision.shape}.")
        return precision
    if not isinstance(value, Mapping):
        raise TypeError(
            f"{where} must be a dict of CSC data, indices, indptr and shape, "
            "since the estimator is sparse."
        )
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


def table_state(table: Mapping[Any, NDArray[Any]], width: int) -> Dict[str, Any]:
    """A grouped model's per-group parameters as a list of group keys
    and one ``(n_groups, width)`` array, in insertion order."""
    keys = list(table)
    values = np.empty((len(keys), width), dtype=np.float64)
    for row, key in enumerate(keys):
        values[row] = table[key]
    return {"keys": keys, "values": values}


def load_table(value: Any, width: int, where: str) -> List[Tuple[Hashable, Any]]:
    """The ``(key, parameters)`` pairs :func:`table_state` wrote, each
    parameter vector its own array."""
    if not isinstance(value, Mapping) or set(value) != _TABLE_KEYS:
        raise ValueError(f"{where} must be a dict with keys 'keys' and 'values'.")
    keys = list(value["keys"])
    if len(set(keys)) != len(keys):
        raise ValueError(f"{where} keys must be unique.")
    values = load_array(value["values"], (len(keys), width), f"{where} values")
    return [(key, values[row].copy()) for row, key in enumerate(keys)]


#: Scalars an estimator keeps besides its posterior, as ``(state key,
#: attribute, loader)``; ``None`` in the state while the attribute is unset.
ScalarFields = Tuple[Tuple[str, str, Any], ...]


def field_keys(fields: ScalarFields) -> Tuple[str, ...]:
    return tuple(key for key, _, _ in fields)


def copy_or_none(value: Any) -> Optional[NDArray[np.float64]]:
    """A float64 copy of an optional array attribute."""
    return None if value is None else np.array(value, dtype=np.float64)


def scalars_state(obj: Any, fields: ScalarFields) -> Dict[str, Any]:
    """``fields`` of ``obj`` as Python scalars, read past any property."""
    return {
        key: load_optional(obj.__dict__.get(attribute), load, key)
        for key, attribute, load in fields
    }


def load_scalars(
    state: Mapping[str, Any], fields: ScalarFields, owner: str
) -> Dict[str, Any]:
    return {
        key: load_optional(state[key], load, f"{owner} state {key!r}")
        for key, _, load in fields
    }


def set_scalars(obj: Any, fields: ScalarFields, restored: Mapping[str, Any]) -> None:
    for key, attribute, _ in fields:
        set_optional(obj, attribute, restored[key])


def set_optional(obj: Any, attribute: str, value: Any) -> None:
    """Set an instance attribute, or leave it unset for ``None``."""
    if value is not None:
        obj.__dict__[attribute] = value


def drop_fit(estimator: Any, kept: Collection[str] = ()) -> None:
    """Return ``estimator`` to how ``__init__`` left it: every instance
    attribute but the constructor parameters and ``kept`` goes, fitted
    state and caches alike."""
    keep = set(estimator.get_params(deep=False)) | set(kept)
    for name in [name for name in estimator.__dict__ if name not in keep]:
        del estimator.__dict__[name]


def stage_load(learner: Any, state: Any) -> Callable[[], None]:
    """``learner.load_state_dict(state)``, checked now and written when
    called, so an agent can check every arm before it changes any. A
    learner without ``_stage_load`` checks as it writes."""
    stage = getattr(learner, "_stage_load", None)
    if stage is not None:
        return stage(state)
    load = learner.load_state_dict
    return lambda: load(state)


def stage_generator(rng: np.random.Generator, state: Any) -> Callable[[], None]:
    """Restore ``rng`` in place, so whoever shares it keeps sharing it;
    the state is checked on a scratch bit generator of the same kind."""
    type(rng.bit_generator)().state = state
    return lambda: setattr(rng.bit_generator, "state", state)


class LearnerStateMixin:
    """``state_dict`` and ``load_state_dict`` for the built-in estimators.

    A class names the attribute that marks an initialized posterior and
    the keys of its state, and supplies ``_fitted_state``,
    ``_read_state`` (check and convert, writing nothing) and
    ``_write_state``. Hyperparameters that fitting tunes in place are
    state whether or not the posterior is initialized, since loading has
    to put them back too.
    """

    #: Attribute whose presence marks an initialized posterior
    _initialized_by: str
    #: Keys of an initialized posterior's state, besides ``version`` and
    #: ``_scalars``
    _state_keys: Tuple[str, ...] = ()
    #: Hyperparameters fitting tunes in place, in every state
    _tuned: Tuple[str, ...] = ()
    #: Scalars kept beside an initialized posterior, ``None`` while unset
    _scalars: ScalarFields = ()
    #: Instance attributes ``__init__`` sets besides its parameters
    _init_attributes: Tuple[str, ...] = ()

    def state_dict(self) -> Dict[str, Any]:
        """
        Return the learned state as plain data.

        The state holds the posterior, the hyperparameters and running
        statistics empirical Bayes tunes, and a ``version``. A dense
        precision is an array, a sparse one its CSC ``data``,
        ``indices``, ``indptr`` and ``shape``, and a grouped model's
        per-group parameters ``{"keys": [...], "values": array}``.
        Constructor arguments are not state: build the estimator in code
        and call :meth:`load_state_dict`. Before the prior is
        initialized, the state holds only the version and any tuned
        hyperparameters.

        Returns
        -------
        state : dict
            Dicts, lists, numpy arrays and Python scalars only. Arrays
            are copies.

        See Also
        --------
        load_state_dict : Restore a state into an estimator built the
            same way.
        """
        state = {"version": STATE_VERSION, **self._tuned_state()}
        if self._initialized_by in self.__dict__:
            state.update(self._fitted_state())
            state.update(scalars_state(self, self._scalars))
        return state

    def load_state_dict(self, state: Dict[str, Any]) -> None:
        """
        Restore a state from :meth:`state_dict`, replacing any fit.

        Built as the original was, the estimator then predicts, samples
        and updates as the original does. Factorizations are rebuilt on
        first use, so where the original had carried one over -- after
        ``decay``, or a sparse update under SuperLU -- draws can differ
        in the last bit, as after unpickling. The generator is not state:
        ``random_state_`` is seeded from ``random_state`` as ``fit``
        seeds it. The state is checked in full before anything is
        written, so a rejected state leaves the estimator as it was.

        Parameters
        ----------
        state : dict
            A state from :meth:`state_dict`.

        Raises
        ------
        ValueError
            If the state is of another version, its keys, shapes, sparse
            structure or classes do not match, or it sets only some of
            the running statistics a fit sets together.
        TypeError
            If a value has the wrong type, or the precision is sparse for
            a dense estimator or the reverse.
        """
        self._stage_load(state)()

    def _stage_load(self, state: Any) -> Callable[[], None]:
        owner = type(self).__name__
        fitted = check_state(
            state,
            self._tuned + self._state_keys + field_keys(self._scalars),
            owner,
            unfitted_keys=self._tuned,
        )
        tuned = self._read_tuned(state, owner)
        restored = None
        if fitted:
            restored = {
                **self._read_state(state, owner),
                **load_scalars(state, self._scalars, owner),
            }

        def commit() -> None:
            drop_fit(self, self._init_attributes)
            self._write_tuned(tuned)
            if restored is not None:
                self._restore_prior(restored)
                self._write_state(restored)
                set_scalars(self, self._scalars, restored)

        return commit

    def _tuned_state(self) -> Dict[str, Any]:
        return {name: float(getattr(self, name)) for name in self._tuned}

    def _read_tuned(self, state: Mapping[str, Any], owner: str) -> Dict[str, Any]:
        return {
            name: load_float(state[name], f"{owner} state {name!r}")
            for name in self._tuned
        }

    def _write_tuned(self, tuned: Mapping[str, Any]) -> None:
        for name, value in tuned.items():
            setattr(self, name, value)

    def _restore_prior(self, restored: Mapping[str, Any]) -> None:
        """Set up what ``fit`` does before its data -- the generator, a
        prior the state then overwrites -- from the tuned
        hyperparameters already written."""
        cast(Any, self)._initialize_prior()

    def _fitted_state(self) -> Dict[str, Any]:
        raise NotImplementedError

    def _read_state(self, state: Mapping[str, Any], owner: str) -> Dict[str, Any]:
        raise NotImplementedError

    def _write_state(self, restored: Mapping[str, Any]) -> None:
        raise NotImplementedError
