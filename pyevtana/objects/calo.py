"""Calorimeter objects.

Unlike tracks, calo objects are linked by **stored indices**, not by position:
``CaloClusterInfo::hits_`` is a vector of indices into the ``calohits`` branch, and
``CaloHitInfo::recoDigis_`` indexes ``calorecodigis``.  So ``cluster.hits()`` really does
follow a link, and it needs the target branch to have been read.
"""

from __future__ import annotations

from ..collection import ObjectCollection, event_collection
from ..missing import MissingRecord, is_missing
from ..record import RecordProxy


class _Linked(RecordProxy):
    """Shared index-following helpers."""

    __slots__ = ()

    def _follow_many(self, field: str, target: str, cls: type):
        """``field`` holds indices into ``target``."""
        coll = self._coll
        cols = coll._sibling(target)
        if is_missing(cols):
            return cols
        try:
            indices = [int(i) for i in coll._column(field)[self._i]]
        except KeyError:
            from ..missing import MissingCollection

            return MissingCollection(target, hint=f"this ntuple has no {field!r} field")
        return ObjectCollection(cols, (coll._ievt,), len(indices), cls, target,
                                coll._batch, coll._ievt, indices=indices)

    def _follow_one(self, field: str, target: str, cls: type):
        """``field`` holds a single index, ``-1`` meaning 'not associated'."""
        coll = self._coll
        try:
            index = int(coll._column(field)[self._i])
        except KeyError:
            return MissingRecord(target, hint=f"this ntuple has no {field!r} field")
        if index < 0:
            return MissingRecord(target, hint=f"{field} < 0: not associated")
        cols = coll._sibling(target)
        if is_missing(cols):
            return MissingRecord(cols.name, cols.state, cols.reason)
        parent = event_collection(cols, coll._ievt, cls, target, coll._batch)
        return cls(parent, index)


class CaloRecoDigi(_Linked):
    """One reconstructed digi. Links both up to its hit and down to its raw digi."""

    __slots__ = ()
    _repr_fields = ("eDep_", "time_")

    def hit(self):
        """The crystal hit this digi belongs to (``caloHitIdx_`` -> ``calohits``)."""
        return self._follow_one("caloHitIdx_", "calohits", CaloHit)

    def digi(self):
        """The raw digi behind it (``caloDigiIdx_`` -> ``calodigis``)."""
        return self._follow_one("caloDigiIdx_", "calodigis", CaloDigi)


class CaloDigi(_Linked):
    """One raw calorimeter digi."""

    __slots__ = ()
    _repr_fields = ("SiPMID_", "t0_", "diskID_")

    def recodigi(self):
        """The reco digi made from it (``caloRecoDigiIdx_`` -> ``calorecodigis``)."""
        return self._follow_one("caloRecoDigiIdx_", "calorecodigis", CaloRecoDigi)


class CaloHit(_Linked):
    """One calorimeter crystal hit."""

    __slots__ = ()
    _repr_fields = ("crystalId_", "eDep_", "time_")

    def recodigis(self):
        """Reco digis making up this hit (``recoDigis_`` -> ``calorecodigis``)."""
        return self._follow_many("recoDigis_", "calorecodigis", CaloRecoDigi)

    def cluster(self):
        """The cluster this hit belongs to, or a falsy ``MissingRecord`` if unclustered."""
        return self._follow_one("clusterIdx_", "caloclusters", CaloCluster)


class CaloCluster(_Linked):
    """One calorimeter cluster."""

    __slots__ = ()
    _repr_fields = ("diskID_", "energyDep_", "time_")

    def hits(self):
        """Crystal hits in this cluster (``hits_`` -> ``calohits``)."""
        return self._follow_many("hits_", "calohits", CaloHit)
