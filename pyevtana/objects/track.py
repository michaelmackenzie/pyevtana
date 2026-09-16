"""Tracks and the things that hang off them.

The companion branches are *positionally* aligned with the track branch -- ``trkhits`` is
``vector<vector<TrkStrawHitInfo>>`` whose outer index is the track index -- so
``track.hits()`` is a plain slice, with no index-following anywhere.
"""

from __future__ import annotations

from typing import Optional

import awkward as ak

from .. import schema as S
from ..collection import ObjectCollection
from ..missing import MissingRecord, SchemaMismatch, is_missing
from ..record import RecordProxy
from ..surfaces import surface_id, surface_name


class TrackSeg(RecordProxy):
    """The fit evaluated at one surface."""

    __slots__ = ()
    _repr_fields = ("sid", "time")

    @property
    def surface(self) -> str:
        """Human-readable surface name for ``sid`` (``"TT_Mid"``, ``"ST_Foils"``, ...)."""
        return surface_name(self.sid)

    def __repr__(self) -> str:
        return f"<TrackSeg {self.surface} t={self.time:.1f}>"


class StrawHit(RecordProxy):
    __slots__ = ()
    _repr_fields = ("plane", "panel", "straw", "edep")


class StrawMat(RecordProxy):
    __slots__ = ()
    _repr_fields = ("plane", "radlen", "doca")


class TrackCaloHit(RecordProxy):
    __slots__ = ()
    _repr_fields = ("did", "ctime", "edep")


class TrackMC(RecordProxy):
    """``TrkInfoMC``. Note the true PDG lives in the genealogy, ``track.mcsim()``."""

    __slots__ = ()
    _repr_fields = ("nhits", "nactive", "t0")


class TrackCaloHitMC(RecordProxy):
    """``CaloClusterInfoMC`` for the cluster on a track."""

    __slots__ = ()
    _repr_fields = ("nhits", "etot", "tavg")


class MVAResult(RecordProxy):
    __slots__ = ()
    _repr_fields = ("result",)


class Track(RecordProxy):
    """One Kalman fit hypothesis."""

    __slots__ = ()
    _repr_fields = ("pdg", "nactive", "chisq")

    # -- companion access -----------------------------------------------------------------

    def _companion(self, role: str, cls: type, *, collection: Optional[bool] = None):
        """Fetch a companion branch, sliced to this track.

        ``collection`` overrides the depth from the schema table (depth 2 -> a collection,
        depth 1 -> a single record).
        """
        coll = self._coll
        tag = coll.tag
        name = S.track_branch(tag, role)
        depth = S.track_depth(role)
        as_collection = (depth == 2) if collection is None else collection

        arr = coll._sibling(name, hint=S.absent_hint(role))
        if is_missing(arr):
            # Preserve the reason; just change the shape of the null object.
            if as_collection:
                return arr
            return MissingRecord(arr.name, arr.state, arr.reason)

        if as_collection:
            return ObjectCollection(arr[self._i], cls, name, coll._batch, coll._ievt, tag)

        # A depth-1 companion is read by track index, which assumes the maker pushed one
        # entry per track. Verify it instead of trusting it: `trkcalohitmc` does not obey
        # this (see calohitmc), and a future branch might not either.
        if len(arr) != len(coll._arr):
            raise SchemaMismatch(
                f"{name!r} has {len(arr)} entries but this event has {len(coll._arr)} "
                f"tracks, so it cannot be indexed by track. This branch is not "
                f"track-aligned; read it as a whole with event.collection({name!r})."
            )
        return cls(arr[self._i], ObjectCollection(arr, cls, name, coll._batch, coll._ievt, tag), self._i)

    # -- reco ------------------------------------------------------------------------------

    def hits(self):
        """Straw hits assigned to this track."""
        return self._companion("hits", StrawHit)

    def hitcalibs(self):
        return self._companion("hitcalibs", StrawHit)

    def mats(self):
        """Straw materials the fit crossed."""
        return self._companion("mats", StrawMat)

    def segs(self):
        """Fit results at each intersected surface."""
        return self._companion("segs", TrackSeg)

    def seg(self, surface, index: int = 0) -> Optional[TrackSeg]:
        """The first segment at ``surface`` (a name like ``"TT_Mid"``, or a raw sid).

        Returns ``None`` when the track has no intersection with that surface -- a normal
        outcome, not an error, so callers must check.
        """
        want = surface_id(surface)
        segs = self.segs()
        if is_missing(segs):
            return None
        seen = 0
        for seg in segs:
            if seg.sid == want:
                if seen == index:
                    return seg
                seen += 1
        return None

    def segpars(self, parametrization: Optional[str] = None):
        """LoopHelix / CentralHelix / KinematicLine parameters at each surface.

        With no argument, whichever parametrization this ntuple was made with.
        """
        roles = ([f"segpars_{parametrization}"] if parametrization
                 else ["segpars_lh", "segpars_ch", "segpars_kl"])
        result = None
        for role in roles:
            result = self._companion(role, RecordProxy)
            if not is_missing(result):
                return result
        return result

    def calohit(self):
        """The calorimeter cluster assigned to this track."""
        return self._companion("calohit", TrackCaloHit)

    def dtdt(self):
        return self._companion("dtdt", RecordProxy)

    def qual(self, leaf: str = "") -> MVAResult:
        """TrkQual MVA output; ``leaf`` picks a non-default one (e.g. ``"_bdt"``)."""
        return self._companion("qual" + leaf, MVAResult, collection=False)

    def pid(self, index: int = 0) -> MVAResult:
        """TrkPID MVA output; ``index`` selects among multiple configured tags."""
        role = "pid" if index == 0 else f"pid{index + 1}"
        return self._companion(role, MVAResult, collection=False)

    # -- MC --------------------------------------------------------------------------------

    def mc(self) -> TrackMC:
        """MC truth for this track."""
        return self._companion("mc", TrackMC, collection=False)

    def mcsim(self):
        """MC genealogy: the SimParticles behind this track, best match first."""
        from .mc import SimParticle

        return self._companion("mcsim", SimParticle)

    def mcvd(self):
        """MC steps at virtual detectors."""
        from .mc import MCStep

        return self._companion("mcvd", MCStep)

    def segsmc(self):
        """MC surface steps, the truth counterpart of :meth:`segs`."""
        from .mc import SurfaceStep

        return self._companion("segsmc", SurfaceStep)

    def hitsmc(self):
        """MC truth for the straw hits on this track."""
        return self._companion("hitsmc", RecordProxy)

    def calohitmc(self):
        """MC truth for this track's calorimeter cluster.

        ``trkcalohitmc`` is **not** track-aligned. ``EventNtupleMaker`` pushes an entry
        only for tracks that have a calo cluster (``EventNtupleMaker_module.cc``: the
        ``kseed.hasCaloCluster() && fillCaloTrackMatchMC()`` guard), and stores no index
        back to the track -- so on a typical file it is much shorter than the track list
        (140 entries for 375 tracks in the sample ntuple).

        The mapping is recovered by counting: the k-th entry belongs to the k-th track,
        in order, that has a calo cluster. A track has one exactly when its
        ``trkcalohit.did >= 0``, because the maker fills that field under the same
        ``hasCaloCluster()`` guard. The recovered count is checked against the actual
        length, so if the maker ever changes this you get an error rather than a
        silently wrong object.

        Returns a falsy ``MissingRecord`` for a track with no calo cluster.
        """
        coll = self._coll
        name = S.track_branch(coll.tag, "calohitmc")
        mc = coll._sibling(name, hint=S.absent_hint("calohitmc"))
        if is_missing(mc):
            return MissingRecord(mc.name, mc.state, mc.reason)

        calohit_name = S.track_branch(coll.tag, "calohit")
        calohit = coll._sibling(calohit_name)
        if is_missing(calohit):
            raise SchemaMismatch(
                f"{name!r} is not track-aligned, so mapping it onto tracks needs "
                f"{calohit_name!r}, which is {calohit.state.value}. Select it too, or "
                f"read {name!r} as a whole with event.collection({name!r})."
            )

        has_cluster = calohit["did"] >= 0
        if not bool(has_cluster[self._i]):
            return MissingRecord(name, hint="this track has no calo cluster")

        expected = int(ak.sum(has_cluster))
        if len(mc) != expected:
            raise SchemaMismatch(
                f"cannot map {name!r} onto tracks: {len(mc)} entries but {expected} "
                f"tracks in this event have a calo cluster. The maker's fill condition "
                f"for {name!r} no longer matches trkcalohit.did >= 0."
            )
        rank = int(ak.sum(has_cluster[:self._i]))
        return TrackCaloHitMC(
            mc[rank], ObjectCollection(mc, TrackCaloHitMC, name, coll._batch, coll._ievt,
                                       coll.tag), rank)

    # -- convenience -------------------------------------------------------------------------

    def has_companion(self, role: str) -> bool:
        """Is this companion branch present *and* selected?"""
        coll = self._coll
        if coll is None or coll._batch is None:
            return False
        return coll._batch.has(S.track_branch(coll.tag, role))
