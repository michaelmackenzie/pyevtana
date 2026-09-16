"""MC-truth objects: SimParticles, MC steps, and MC surface steps."""

from __future__ import annotations

from ..record import RecordProxy
from ..surfaces import surface_name


class SimParticle(RecordProxy):
    """One entry of a ``SimInfo`` genealogy.

    ``rank`` orders particles by how many of the track's hits they produced, so rank 0 is
    the particle the track is "really" from.  There is no stored mother index in the
    ntuple, so no parent navigation is offered here -- use ``prirel`` / ``trkrel``.
    """

    __slots__ = ()
    _repr_fields = ("pdg", "rank", "nhits")

    @property
    def is_best_match(self) -> bool:
        return int(self.rank) == 0


class MCStep(RecordProxy):
    __slots__ = ()
    _repr_fields = ("vid", "pdg", "time")


class SurfaceStep(RecordProxy):
    """MC truth counterpart of a track segment."""

    __slots__ = ()
    _repr_fields = ("sid", "time", "edep")

    @property
    def surface(self) -> str:
        return surface_name(self.sid)
