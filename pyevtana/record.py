"""Thin attribute-access views onto one normalized awkward record.

Proxies deliberately hold no data of their own -- just the record and a back-pointer to
the collection they came from, which is what the navigation methods (``track.hits()``,
``cluster.hits()``) use to reach sibling branches.  They are Python-level objects, so a
loop over millions of them is much slower than working on the arrays directly; every
collection therefore also exposes ``.array()`` (see :mod:`pyevtana.collection`).

Proxies are **not picklable** on purpose: they reference a live branch cache, so returning
one from a parallel worker would be a bug.  :mod:`pyevtana.parallel` says so explicitly
rather than letting ``multiprocessing`` produce a baffling error.
"""

from __future__ import annotations

from .missing import PyEvtAnaError


class NotPicklable(PyEvtAnaError):
    """Raised when a live proxy is about to cross a process boundary."""


class RecordProxy:
    """One object: a track, a hit, a cluster."""

    __slots__ = ("_rec", "_coll", "_i")

    #: fields shown by ``repr``; subclasses override.
    _repr_fields: tuple[str, ...] = ()

    def __init__(self, rec, coll=None, index: int = -1):
        self._rec = rec
        self._coll = coll
        self._i = index

    # -- data access --------------------------------------------------------------------

    def __getattr__(self, name: str):
        if name.startswith("_"):
            raise AttributeError(name)
        try:
            return self._rec[name]
        except Exception:
            raise AttributeError(
                f"{type(self).__name__} has no field {name!r}; available fields: "
                f"{', '.join(self.fields)}"
            ) from None

    def __getitem__(self, name: str):
        return self._rec[name]

    @property
    def fields(self) -> list[str]:
        try:
            return list(self._rec.fields)
        except Exception:
            return []

    def has(self, name: str) -> bool:
        """Is this field present in the file? (Older ntuples may lack newer members.)"""
        return name in self.fields

    def get(self, name: str, default=None):
        return self._rec[name] if name in self.fields else default

    @property
    def raw(self):
        """The underlying awkward record, for anything this wrapper does not cover."""
        return self._rec

    @property
    def index(self) -> int:
        """Position within its collection -- the index other branches refer to."""
        return self._i

    @property
    def missing(self) -> bool:
        return False

    def to_dict(self) -> dict:
        return {f: self._rec[f] for f in self.fields}

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
        shown = [f for f in self._repr_fields if f in self.fields]
        body = " ".join(f"{f}={self._rec[f]}" for f in shown)
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
