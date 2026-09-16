"""CRV objects.

``CrvPulseInfoReco::crvHitIndex`` points *back* at the coincidence cluster that owns the
pulse (``-1`` = unclustered), so ``coinc.pulses()`` is a reverse lookup over the pulse
branch rather than a forward index.
"""

from __future__ import annotations

from ..collection import ObjectCollection, event_collection
from ..missing import MissingRecord, is_missing
from ..record import RecordProxy


class CrvPulse(RecordProxy):
    __slots__ = ()
    _repr_fields = ("barId", "SiPMId", "PEs", "time")

    def coinc(self):
        """The coincidence cluster containing this pulse, falsy if unclustered."""
        coll = self._coll
        try:
            index = int(coll._column("crvHitIndex")[self._i])
        except KeyError:
            return MissingRecord("crvcoincs", hint="this ntuple has no 'crvHitIndex' field")
        if index < 0:
            return MissingRecord("crvcoincs", hint="crvHitIndex < 0: pulse is unclustered")
        cols = coll._sibling("crvcoincs")
        if is_missing(cols):
            return MissingRecord(cols.name, cols.state, cols.reason)
        parent = event_collection(cols, coll._ievt, CrvCoinc, "crvcoincs", coll._batch)
        return CrvCoinc(parent, index)


class CrvDigi(RecordProxy):
    __slots__ = ()
    _repr_fields = ("barId", "SiPMId")


class CrvCoinc(RecordProxy):
    """A CRV coincidence cluster."""

    __slots__ = ()
    _repr_fields = ("sectorType", "PEs", "time", "nHits")

    def pulses(self):
        """Reco pulses belonging to this cluster (reverse lookup on ``crvHitIndex``)."""
        coll = self._coll
        cols = coll._sibling("crvpulses")
        if is_missing(cols):
            return cols
        pulses = event_collection(cols, coll._ievt, CrvPulse, "crvpulses", coll._batch)
        try:
            matched = [i for i, owner in enumerate(pulses._column("crvHitIndex"))
                       if owner == self._i]
        except KeyError:
            from ..missing import MissingCollection

            return MissingCollection("crvpulses", hint="this ntuple has no 'crvHitIndex' field")
        return ObjectCollection(cols, (coll._ievt,), len(matched), CrvPulse, "crvpulses",
                                coll._batch, coll._ievt, indices=matched)
