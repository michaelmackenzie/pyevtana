"""Tracks and the things that hang off them.

The companion branches are *positionally* aligned with the track branch -- ``trkhits`` is
``vector<vector<TrkStrawHitInfo>>`` whose outer index is the track index -- so
``track.hits()`` is a plain slice, with no index-following anywhere.
"""

from __future__ import annotations

from typing import Optional

from .. import schema as S
from ..collection import ObjectCollection
from ..missing import MissingRecord, is_missing
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
        return self._companion("calohitmc", RecordProxy, collection=False)

    # -- convenience -------------------------------------------------------------------------

    def has_companion(self, role: str) -> bool:
        """Is this companion branch present *and* selected?"""
        coll = self._coll
        if coll is None or coll._batch is None:
            return False
        return coll._batch.has(S.track_branch(coll.tag, role))
