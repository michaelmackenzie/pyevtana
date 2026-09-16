"""Thin attribute-access views onto one object's fields.

**Why these do not index awkward.** Awkward is built for whole-array operations; a single
scalar read goes through layout wrapping, backend dispatch and error-context construction,
measured at ~41 us for ``arr[i]`` plus ~17 us for ``record[field]``. At a few hundred such
reads per track that alone was essentially all of the object loop's runtime. Proxies
therefore read Python lists produced once per batch and per field (see
:mod:`pyevtana.columns`), where the same read costs ~0.1 us.

The array path is untouched: ``collection.array()`` and ``batch.get()`` still hand back
awkward arrays with full ``vector`` behaviour.

Proxies are **not picklable** on purpose: they reference a live branch cache, so returning
one from a parallel worker would be a bug.
"""

from __future__ import annotations

from .missing import PyEvtAnaError
from .vectors import Vec3, is_vector_dict


class NotPicklable(PyEvtAnaError):
    """Raised when a live proxy is about to cross a process boundary."""


class SubRecord:
    """A struct nested inside another struct, e.g. ``SimInfo.prirel``."""

    __slots__ = ("_rec",)

    def __init__(self, record: dict):
        self._rec = record

    def __getattr__(self, name: str):
        try:
            return _wrap(self._rec[name])
        except KeyError:
            raise AttributeError(
                f"no field {name!r}; available: {', '.join(self._rec)}") from None

    def __getitem__(self, name):
        return _wrap(self._rec[name])

    @property
    def fields(self) -> list:
        return list(self._rec)

    def to_dict(self) -> dict:
        return dict(self._rec)

    def __repr__(self) -> str:
        return f"SubRecord({self._rec})"


def _wrap(value):
    """Give nested structs attribute access and turn vector dicts into :class:`Vec3`."""
    if type(value) is dict:
        if is_vector_dict(value):
            return Vec3(value["x"], value["y"], value["z"])
        return SubRecord(value)
    return value


class RecordProxy:
    """One object: a track, a hit, a cluster."""

    __slots__ = ("_coll", "_i")

    #: fields shown by ``repr``; subclasses override.
    _repr_fields: tuple[str, ...] = ()

    def __init__(self, coll, index: int):
        self._coll = coll
        self._i = index

    # -- data access --------------------------------------------------------------------

    def __getattr__(self, name: str):
        if name[0] == "_":
            raise AttributeError(name)
        try:
            value = self._coll._column(name)[self._i]
        except KeyError:
            raise AttributeError(
                f"{type(self).__name__} has no field {name!r}; available fields: "
                f"{', '.join(self.fields)}"
            ) from None
        return value if type(value) is not dict else _wrap(value)

    def __getitem__(self, name: str):
        return _wrap(self._coll._column(name)[self._i])

    @property
    def fields(self) -> list:
        return self._coll.fields

    def has(self, name: str) -> bool:
        """Is this field present in the file? (Older ntuples may lack newer members.)"""
        return name in self._coll.fields

    def get(self, name: str, default=None):
        if name not in self._coll.fields:
            return default
        return _wrap(self._coll._column(name)[self._i])

    @property
    def raw(self) -> dict:
        """Every field of this object as a plain dict."""
        return self.to_dict()

    @property
    def index(self) -> int:
        """Position within its collection -- the index other branches refer to."""
        return self._i

    @property
    def missing(self) -> bool:
        return False

    def to_dict(self) -> dict:
        return {f: self._coll._column(f)[self._i] for f in self._coll.fields}

    # -- plumbing -----------------------------------------------------------------------

    def _sibling(self, name: str, hint: str = ""):
        """Same event, different branch -- the basis of every navigation method."""
        if self._coll is None:
            raise PyEvtAnaError(
                f"{type(self).__name__} was built without a collection, so it cannot "
                f"navigate to {name!r}"
            )
        return self._coll._sibling(name, hint=hint)

    def __repr__(self) -> str:
        fields = self._coll.fields
        shown = [f for f in self._repr_fields if f in fields]
        body = " ".join(f"{f}={self._coll._column(f)[self._i]}" for f in shown)
        return f"<{type(self).__name__} {body}>" if body else f"<{type(self).__name__}>"

    def __reduce__(self):
        raise NotPicklable(
            f"{type(self).__name__} holds a live branch cache and cannot be pickled. "
            "Return plain results (histograms, arrays, counts) from Dataset.map() workers "
            "instead of proxy objects."
        )


class EventRecord(RecordProxy):
    """An event-level singleton such as ``evtinfo`` or ``hitcount``."""

    __slots__ = ()
