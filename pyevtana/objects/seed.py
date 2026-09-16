"""Pattern-recognition seeds: time clusters, line seeds, helices."""

from __future__ import annotations

from ..missing import MissingCollection
from ..record import RecordProxy


class _SeedBase(RecordProxy):
    """Shared handling for seeds whose hits are not (yet) written to the ntuple."""

    __slots__ = ()

    #: branch suffix that *would* hold this seed's hits, if the maker wrote one
    _hits_suffix = "hits"

    def hits(self):
        """Hits in this seed.

        ``EventNtupleTimeClusterInfo`` and ``LineSeedInfo`` currently store only ``nhits``
        and ``nStrawHits`` -- there is no per-hit branch and no hit-index vector, so there
        is nothing to return.  If ``EventNtupleMaker`` is later extended to write a
        ``<name>hits`` vector-of-vector branch (or a ``hits_`` index vector, mirroring
        ``CaloClusterInfo``), this starts working with no change here beyond one row in
        :mod:`pyevtana.schema`.
        """
        coll = self._coll
        name = (coll.name if coll is not None else "seed") + self._hits_suffix
        if coll is not None and coll._batch is not None and coll._batch.has(name):
            from ..collection import ObjectCollection

            arr = coll._sibling(name)
            return ObjectCollection(arr[self._i], RecordProxy, name, coll._batch, coll._ievt)
        return MissingCollection(
            name,
            hint=(f"this ntuple stores only nhits/nStrawHits for {coll.name if coll else 'seeds'}; "
                  f"EventNtupleMaker would need to write a {name!r} branch for hit-level access"),
        )


class TimeCluster(_SeedBase):
    __slots__ = ()
    _repr_fields = ("nhits", "t0", "ecalo")


class LineSeed(_SeedBase):
    __slots__ = ()
    _repr_fields = ("nhits", "t0", "cos")


class Helix(RecordProxy):
    __slots__ = ()
    _repr_fields = ("nch", "nsh", "mom")
