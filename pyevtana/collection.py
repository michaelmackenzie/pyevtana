"""A sequence of :class:`~pyevtana.record.RecordProxy` over one slice of a branch.

A collection is a window onto a :class:`~pyevtana.columns.Columns` -- an index prefix into
the per-field lists, plus a length, plus optionally an explicit index list for slices built
by following stored indices (``cluster.hits()``). Fields are converted only when something
reads them, so a loop that touches ``sid`` never pays to convert ``pos``.

The awkward array is always one call away: :meth:`ObjectCollection.array` re-derives it
from the batch cache, so the object loop and the vectorized style mix freely.
"""

from __future__ import annotations

from typing import Iterator, Optional

from .record import NotPicklable, RecordProxy


class ObjectCollection:
    """The objects of one collection, in one slice (usually one event)."""

    __slots__ = ("_cols", "_prefix", "_length", "_cls", "_name", "_batch", "_ievt",
                 "_tag", "_indices", "_local")

    def __init__(self, cols, prefix, length: int, cls: type, name: str, batch=None,
                 ievt: int = -1, tag: str = "", indices=None):
        self._cols = cols
        #: index chain into each field's nested list that reaches this slice
        self._prefix = prefix
        self._length = length
        self._cls = cls
        self._name = name
        self._batch = batch
        self._ievt = ievt
        self._tag = tag
        #: explicit positions within the slice, for index-followed collections
        self._indices = indices
        self._local: dict = {}

    # -- field access ---------------------------------------------------------------------

    def _column(self, name: str) -> list:
        """This slice's values for one field, converting the field on first use."""
        column = self._local.get(name)
        if column is None:
            column = self._cols.column(name)
            for index in self._prefix:
                column = column[index]
            if self._indices is not None:
                column = [column[i] for i in self._indices]
            self._local[name] = column
        return column

    @property
    def fields(self) -> list:
        return self._cols.fields

    # -- sequence -------------------------------------------------------------------------

    def __len__(self) -> int:
        return self._length

    def __iter__(self) -> Iterator[RecordProxy]:
        cls = self._cls
        for i in range(self._length):
            yield cls(self, i)

    def __getitem__(self, index):
        if isinstance(index, slice):
            return [self._cls(self, i) for i in range(*index.indices(self._length))]
        if index < 0:
            index += self._length
        if not 0 <= index < self._length:
            raise IndexError(f"{self._name} index {index} out of range ({self._length})")
        return self._cls(self, index)

    def __bool__(self) -> bool:
        return self._length > 0

    def first(self) -> Optional[RecordProxy]:
        return self._cls(self, 0) if self._length else None

    # -- escape hatches ---------------------------------------------------------------------

    def array(self):
        """The normalized awkward array for this slice."""
        import awkward as ak

        if self._batch is None or self._prefix is None:
            return ak.Array([])
        array = self._batch.get(self._name)
        for index in self._prefix:
            array = array[index]
        if self._indices is not None:
            array = array[self._indices]
        return array

    def to_list(self) -> list:
        return [{f: self._column(f)[i] for f in self.fields} for i in range(self._length)]

    def to_pandas(self):
        import awkward as ak

        return ak.to_dataframe(self.array())

    # -- identity ---------------------------------------------------------------------------

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
        """Another branch of the same event, as :class:`~pyevtana.columns.Columns`."""
        if self._batch is None:
            from .missing import PyEvtAnaError

            raise PyEvtAnaError(f"{self._name} is detached from its file; cannot reach {name!r}")
        return self._batch.columns(name, hint=hint)

    def __repr__(self) -> str:
        return f"<{self._cls.__name__}Collection {self._name} n={self._length}>"

    def __reduce__(self):
        raise NotPicklable(
            f"{type(self).__name__} holds a live branch cache and cannot be pickled; "
            "return .array() or a reduced result from Dataset.map() workers instead."
        )


def event_collection(cols, ievt: int, cls: type, name: str, batch, tag: str = "",
                     depth: int = 1) -> ObjectCollection:
    """The objects of one event (depth 1), or one event's whole branch (depth 0)."""
    length = cols.lengths(1)[ievt] if depth else len(cols.array)
    prefix = (ievt,) if depth else ()
    return ObjectCollection(cols, prefix, length, cls, name, batch, ievt, tag or name)


def nested_collection(cols, ievt: int, iobj: int, cls: type, name: str, batch,
                      tag: str = "") -> ObjectCollection:
    """The objects a depth-2 branch holds for one object (e.g. one track's hits)."""
    length = cols.lengths(2)[ievt][iobj]
    return ObjectCollection(cols, (ievt, iobj), length, cls, name, batch, ievt, tag)
