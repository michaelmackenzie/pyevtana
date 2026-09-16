"""CRV objects.

``CrvPulseInfoReco::crvHitIndex`` points *back* at the coincidence cluster that owns the
pulse (``-1`` = unclustered), so ``coinc.pulses()`` is a reverse lookup over the pulse
branch rather than a forward index.
"""

from __future__ import annotations

from ..collection import ObjectCollection
from ..missing import MissingRecord, is_missing
from ..record import RecordProxy


class CrvPulse(RecordProxy):
    __slots__ = ()
    _repr_fields = ("barId", "SiPMId", "PEs", "time")

    def coinc(self):
        """The coincidence cluster containing this pulse, falsy if unclustered."""
        coll = self._coll
        try:
            index = int(self._rec["crvHitIndex"])
        except Exception:
            return MissingRecord("crvcoincs", hint="this ntuple has no 'crvHitIndex' field")
        if index < 0:
            return MissingRecord("crvcoincs", hint="crvHitIndex < 0: pulse is unclustered")
        arr = coll._sibling("crvcoincs")
        if is_missing(arr):
            return MissingRecord(arr.name, arr.state, arr.reason)
        return CrvCoinc(arr[index], ObjectCollection(arr, CrvCoinc, "crvcoincs",
                                                     coll._batch, coll._ievt), index)


class CrvDigi(RecordProxy):
    __slots__ = ()
    _repr_fields = ("barId", "SiPMId")


class CrvCoinc(RecordProxy):
    """A CRV coincidence cluster."""

    __slots__ = ()
    _repr_fields = ("sectorType", "PEs", "time", "nHits")

    def pulses(self):
        """Reco pulses belonging to this cluster (reverse lookup on ``crvHitIndex``)."""
        import awkward as ak

        coll = self._coll
        arr = coll._sibling("crvpulses")
        if is_missing(arr):
            return arr
        try:
            mask = arr["crvHitIndex"] == self._i
        except Exception:
            from ..missing import MissingCollection

            return MissingCollection("crvpulses", hint="this ntuple has no 'crvHitIndex' field")
        return ObjectCollection(arr[mask], CrvPulse, "crvpulses", coll._batch, coll._ievt)
