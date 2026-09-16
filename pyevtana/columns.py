"""Per-field Python views of a collection, converted on first use.

Converting a whole record array with ``ak.to_list`` builds a Python dict per object, which
is by far the most expensive part of it: for ``trksegs`` that is 4.60 us per segment,
against 0.05 us per segment for a single scalar field. Most analyses touch a handful of a
struct's fields -- the worked selection uses 3 of ``TrkSegInfo``'s 9 and 1 of
``LoopHelixInfo``'s 15 -- so converting field by field, on demand, avoids nearly all of it.

Each :class:`Columns` wraps one normalized awkward array and hands out nested Python lists
one field at a time. The awkward array stays available for ``.array()``.
"""

from __future__ import annotations

import awkward as ak

from .vectors import Vec3


class VectorColumn:
    """A converted vector field, kept as three component lists.

    Converting an ``XYZVectorF`` field with ``ak.to_list`` builds a ``{"x", "y", "z"}``
    dict per object, which costs about twice as much as converting the three components
    separately -- and for ``trksegs.mom`` that was the single largest conversion in a
    typical selection. The components are converted individually and a :class:`Vec3` is
    built only for the objects actually read.
    """

    __slots__ = ("x", "y", "z", "depth")

    def __init__(self, x, y, z, depth: int):
        self.x = x
        self.y = y
        self.z = z
        #: remaining list axes above the vector itself
        self.depth = depth

    def __getitem__(self, index):
        if self.depth <= 1:
            return Vec3(self.x[index], self.y[index], self.z[index])
        return VectorColumn(self.x[index], self.y[index], self.z[index], self.depth - 1)

    def __len__(self) -> int:
        return len(self.x)

    def __iter__(self):
        for i in range(len(self.x)):
            yield self[i]

    def __repr__(self) -> str:
        return f"<VectorColumn depth={self.depth} n={len(self.x)}>"


class Columns:
    """Lazily converted per-field views of one collection's array."""

    __slots__ = ("array", "_fields", "_columns", "_lengths")

    def __init__(self, array):
        self.array = array
        try:
            self._fields = list(array.fields)
        except Exception:
            self._fields = []
        self._columns: dict = {}
        self._lengths: dict = {}

    @property
    def fields(self) -> list:
        return self._fields

    def column(self, name: str):
        """The nested Python list for one field, converting it the first time."""
        column = self._columns.get(name)
        if column is None:
            if name not in self._fields:
                raise KeyError(name)
            sub = self.array[name]
            if _is_vector(sub):
                column = VectorColumn(ak.to_list(sub["x"]), ak.to_list(sub["y"]),
                                      ak.to_list(sub["z"]), sub.ndim)
            else:
                column = ak.to_list(sub)
            self._columns[name] = column
        return column

    def lengths(self, depth: int):
        """Object counts at ``depth`` (1 = per event, 2 = per event per track)."""
        cached = self._lengths.get(depth)
        if cached is None:
            cached = ak.to_list(ak.num(self.array, axis=depth))
            self._lengths[depth] = cached
        return cached

    def __repr__(self) -> str:
        return (f"<Columns {len(self._fields)} fields, "
                f"{len(self._columns)} converted: {sorted(self._columns)}>")


def _is_vector(array) -> bool:
    """True for a rebuilt ``XYZVectorF`` field (see :mod:`pyevtana.vectors`)."""
    try:
        return set(array.fields) == {"x", "y", "z"}
    except Exception:
        return False
