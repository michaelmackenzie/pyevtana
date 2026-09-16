"""The per-event facade.

``event.Tracks("trk")`` is the entry point for everything: the argument is the FHiCL
``branchname`` of the track-fit config that produced the branch (``"trk"``, ``"de"``,
``"kl"``, ...).  When a file holds exactly one collection of a given kind the tag may be
omitted; when it holds several, omitting it is an error rather than a guess.
"""

from __future__ import annotations

from typing import Optional

from . import schema as S
from .collection import ObjectCollection, event_collection
from .missing import (AmbiguousCollection, BranchState, MissingRecord, is_missing,
                      missing_record, resolve_missing)
from .objects import (CaloCluster, CaloDigi, CaloHit, CaloRecoDigi, CrvCoinc, CrvDigi,
                      CrvPulse, Helix, LineSeed, MCStep, SimParticle, TimeCluster, Track)
from .record import EventRecord, RecordProxy


class Event:
    """One entry of the ntuple."""

    __slots__ = ("_batch", "_i")

    def __init__(self, batch, index: int):
        self._batch = batch
        self._i = index

    # -- identity ---------------------------------------------------------------------------

    @property
    def index(self) -> int:
        """Index within the file (not within the batch)."""
        return self._batch.start + self._i

    @property
    def source(self) -> str:
        return self._batch.reader.source

    @property
    def schema(self):
        return self._batch.reader.schema

    def has(self, name: str) -> bool:
        """Is this branch present in the file *and* selected by the branch filter?"""
        return self._batch.has(name)

    def state(self, name: str) -> BranchState:
        return self._batch.reader.schema.state(name)

    # -- event-level singletons ---------------------------------------------------------------

    def _singleton(self, name: str, cls: type = EventRecord):
        cols = self._batch.columns(name)
        if is_missing(cols):
            return missing_record(cols.name, cols.state, cols.reason)
        # a depth-0 branch is one record per event, so the event index selects it
        coll = event_collection(cols, self._i, cls, name, self._batch, depth=0)
        return cls(coll, self._i)

    @property
    def info(self) -> EventRecord:
        """``evtinfo``: run, subrun, event, nprotons, pbtime."""
        return self._singleton("evtinfo")

    @property
    def mc(self) -> EventRecord:
        """``evtinfomc``: event-level MC truth."""
        return self._singleton("evtinfomc")

    @property
    def hitcount(self) -> EventRecord:
        return self._singleton("hitcount")

    @property
    def lumistream(self) -> EventRecord:
        return self._singleton("lumistream")

    @property
    def crvsummary(self) -> EventRecord:
        return self._singleton("crvsummary")

    @property
    def crvsummarymc(self) -> EventRecord:
        return self._singleton("crvsummarymc")

    @property
    def run(self) -> int:
        return int(self.info.run)

    @property
    def subrun(self) -> int:
        return int(self.info.subrun)

    @property
    def event(self) -> int:
        return int(self.info.event)

    # -- triggers and counters -------------------------------------------------------------------

    def triggers(self) -> dict:
        """All ``trig_*`` results for this event, keyed without the prefix."""
        return {
            name[len(S.TRIGGER_PREFIX):]: bool(self._batch.leaf(name, self._i))
            for name in self._batch.reader.schema.triggers
        }

    def trigger(self, name: str) -> bool:
        """One trigger path, with or without the ``trig_`` prefix."""
        full = name if name.startswith(S.TRIGGER_PREFIX) else S.TRIGGER_PREFIX + name
        return bool(self._batch.leaf(full, self._i))

    def ntracks(self, tag: Optional[str] = None) -> int:
        """``tcnt.n<tag>`` -- the maker's own count, without reading the track branch."""
        tag = self._resolve_track_tag(tag, for_count=True)
        leaf = self._batch.reader.schema.counters.get(tag)
        if leaf is None:
            return len(self.Tracks(tag))
        return int(self._batch.leaf(leaf, self._i))

    # -- collections ---------------------------------------------------------------------------

    def _collection(self, name: str, cls: type, tag: str = "", hint: str = ""):
        cols = self._batch.columns(name, hint=hint)
        if is_missing(cols):
            return cols
        return event_collection(cols, self._i, cls, name, self._batch, tag)

    def _resolve(self, kind: str, tag: Optional[str], what: str) -> str:
        schema = self._batch.reader.schema
        if tag is not None:
            return tag
        names = schema.names_of_kind(kind)
        if len(names) == 1:
            return names[0]
        if not names:
            return what  # let the missing-branch policy produce the error
        raise AmbiguousCollection(
            f"this file has {len(names)} {what} collections ({', '.join(sorted(names))}); "
            f"pass one explicitly, e.g. event.{what}({names[0]!r})"
        )

    def _resolve_track_tag(self, tag: Optional[str], for_count: bool = False) -> str:
        schema = self._batch.reader.schema
        if tag is not None:
            return tag
        tags = schema.track_tags()
        if len(tags) == 1:
            return tags[0]
        if not tags:
            return "trk"
        raise AmbiguousCollection(
            f"this file has {len(tags)} track collections ({', '.join(sorted(tags))}); "
            f"pass one explicitly, e.g. event.Tracks({tags[0]!r})"
        )

    def Tracks(self, tag: Optional[str] = None):
        """Reconstructed tracks of one fit collection."""
        tag = self._resolve_track_tag(tag)
        return self._collection(tag, Track, tag=tag)

    def TimeClusters(self, name: Optional[str] = None):
        return self._collection(self._resolve("timecluster", name, "TimeClusters"), TimeCluster)

    def LineSeeds(self, name: Optional[str] = None):
        return self._collection(self._resolve("lineseed", name, "LineSeeds"), LineSeed)

    def Helices(self):
        return self._collection("helices", Helix)

    def CaloClusters(self):
        return self._collection("caloclusters", CaloCluster)

    def CaloHits(self):
        return self._collection("calohits", CaloHit)

    def CaloRecoDigis(self):
        return self._collection("calorecodigis", CaloRecoDigi)

    def CaloDigis(self):
        return self._collection("calodigis", CaloDigi)

    def CaloClustersMC(self):
        return self._collection("caloclustersmc", RecordProxy)

    def CaloHitsMC(self):
        return self._collection("calohitsmc", RecordProxy)

    def CrvCoincs(self):
        return self._collection("crvcoincs", CrvCoinc)

    def CrvCoincsMC(self):
        return self._collection("crvcoincsmc", RecordProxy)

    def CrvPulses(self):
        return self._collection("crvpulses", CrvPulse)

    def CrvDigis(self):
        return self._collection("crvdigis", CrvDigi)

    def Primaries(self):
        """``primary``: SimParticles of the event primary."""
        return self._collection("primary", SimParticle)

    def MCSteps(self, instance: Optional[str] = None):
        """``mcsteps_<instance>``; the instance may be omitted if there is only one."""
        name = instance if instance and instance.startswith("mcsteps") else (
            f"mcsteps_{instance}" if instance else None)
        return self._collection(self._resolve("mcstep", name, "MCSteps"), MCStep)

    # -- generic access -------------------------------------------------------------------------

    def collection(self, name: str, cls: type = RecordProxy):
        """Any collection by branch name, for anything the named accessors do not cover."""
        return self._collection(name, cls)

    def array(self, name: str):
        """The normalized awkward array for this event's slice of ``name``."""
        return self._batch.event_array(name, self._i)

    # -- lowercase aliases ------------------------------------------------------------------------

    tracks = Tracks
    timeclusters = TimeClusters
    lineseeds = LineSeeds
    helices = Helices
    caloclusters = CaloClusters
    calohits = CaloHits
    crvcoincs = CrvCoincs
    crvpulses = CrvPulses
    primaries = Primaries
    mcsteps = MCSteps

    def __repr__(self) -> str:
        try:
            return f"<Event run={self.run} subrun={self.subrun} event={self.event}>"
        except Exception:
            return f"<Event index={self.index}>"

    def __reduce__(self):
        from .record import NotPicklable

        raise NotPicklable(
            "Event holds a live branch cache and cannot be pickled. Return plain results "
            "(histograms, arrays, counts) from Dataset.map() workers."
        )
