"""A sequence of :class:`~pyevtana.record.RecordProxy` over one event's slice of a branch.

Every collection also hands back the underlying awkward array, so the object loop and the
vectorized style can be mixed freely -- the object loop is for clarity and per-event logic,
``.array()`` is for the hot path.
"""

from __future__ import annotations

from typing import Iterator, Optional

from .record import NotPicklable, RecordProxy


class ObjectCollection:
    """The objects of one collection, in one event."""

    __slots__ = ("_arr", "_cls", "_name", "_batch", "_ievt", "_tag")

    def __init__(self, arr, cls: type, name: str, batch=None, ievt: int = -1, tag: str = ""):
        self._arr = arr
        self._cls = cls
        self._name = name
        self._batch = batch
        self._ievt = ievt
        self._tag = tag

    # -- sequence -----------------------------------------------------------------------

    def __len__(self) -> int:
        return len(self._arr)

    def __iter__(self) -> Iterator[RecordProxy]:
        cls, arr = self._cls, self._arr
        for i in range(len(arr)):
            yield cls(arr[i], self, i)

    def __getitem__(self, index):
        if isinstance(index, slice):
            return [self._cls(self._arr[i], self, i) for i in range(*index.indices(len(self._arr)))]
        if index < 0:
            index += len(self._arr)
        return self._cls(self._arr[index], self, index)

    def __bool__(self) -> bool:
        return len(self._arr) > 0

    def first(self) -> Optional[RecordProxy]:
        return self[0] if len(self._arr) else None

    # -- escape hatches -------------------------------------------------------------------

    def array(self):
        """The normalized awkward array for this event's objects."""
        return self._arr

    def to_list(self) -> list:
        return self._arr.to_list()

    def to_pandas(self):
        import awkward as ak

        return ak.to_dataframe(self._arr)

    @property
    def fields(self) -> list[str]:
        try:
            return list(self._arr.fields)
        except Exception:
            return []

    # -- identity -------------------------------------------------------------------------

    @property
    def name(self) -> str:
        return self._name

    @property
    def tag(self) -> str:
        return self._tag

    @property
    def missing(self) -> bool:
        return False

    @property
    def state(self):
        from .missing import BranchState

        return BranchState.LOADED

    def _sibling(self, name: str, hint: str = ""):
        """Another branch of the same event (used by navigation methods)."""
        if self._batch is None:
            from .missing import PyEvtAnaError

            raise PyEvtAnaError(f"{self._name} is detached from its file; cannot reach {name!r}")
        return self._batch.event_array(name, self._ievt, hint=hint)

    def __repr__(self) -> str:
        return f"<{self._cls.__name__}Collection {self._name} n={len(self._arr)}>"

    def __reduce__(self):
        raise NotPicklable(
            f"{type(self).__name__} holds a live branch cache and cannot be pickled; "
            "return .array() or a reduced result from Dataset.map() workers instead."
        )
