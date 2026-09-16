"""Static description of the EventNtuple branch-naming scheme.

Everything here mirrors ``EventNtuple/src/EventNtupleMaker_module.cc``.  Keeping it as
data (rather than scattered string literals) means a maker change is a table edit.

Branch names are *derived*, not hardcoded: a track-fit config with FHiCL ``branchname``
``B`` produces ``B``, ``B+"segs"``, ``B+"hits"``, ... (see ``TRACK_COMPANIONS``), and the
tags themselves are discovered from the branch classes (see :mod:`pyevtana.discovery`).
"""

from __future__ import annotations

import re
from dataclasses import dataclass, field
from typing import Optional

# --------------------------------------------------------------------------------------
# typename parsing
# --------------------------------------------------------------------------------------

_VEC2 = re.compile(r"^(?:std::)?vector<\s*(?:std::)?vector<\s*mu2e::(\w+)\s*>\s*>$")
_VEC1 = re.compile(r"^(?:std::)?vector<\s*mu2e::(\w+)\s*>$")
_OBJ = re.compile(r"^mu2e::(\w+)$")


def parse_typename(typename: str) -> Optional[tuple[int, str]]:
    """``vector<vector<mu2e::TrkSegInfo>>`` -> ``(2, "TrkSegInfo")``.

    Returns ``None`` for anything that is not a mu2e info struct (plain ints, bools,
    anonymous structs, ...).  Depth is the number of nested vectors: 0 = one per event,
    1 = one per object, 2 = one vector per object.
    """
    tn = " ".join(typename.split())
    for depth, pattern in ((2, _VEC2), (1, _VEC1), (0, _OBJ)):
        m = pattern.match(tn)
        if m:
            return depth, m.group(1)
    return None


# --------------------------------------------------------------------------------------
# collection roles
# --------------------------------------------------------------------------------------


@dataclass(frozen=True)
class CompanionSpec:
    """One branch that hangs off a track collection with tag ``B``."""

    role: str
    suffix: str  # literal suffix appended to the tag, or "" for the tag branch itself
    struct: str  # expected mu2e struct name
    depth: int
    #: regex (applied to the part after the tag) for companions with variable names
    pattern: Optional[str] = None
    doc: str = ""

    def matches(self, remainder: str) -> bool:
        if self.pattern is not None:
            return re.fullmatch(self.pattern, remainder) is not None
        return remainder == self.suffix


#: EventNtupleMaker_module.cc:727-796.  ``depth`` 1 = one entry per track,
#: 2 = one vector per track (so ``trkhits[ievt][itrk]`` is that track's hit list).
TRACK_COMPANIONS: tuple[CompanionSpec, ...] = (
    CompanionSpec("info", "", "TrkInfo", 1, doc="the track itself"),
    CompanionSpec("segs", "segs", "TrkSegInfo", 2, doc="fit result at each surface"),
    CompanionSpec("segpars_lh", "segpars_lh", "LoopHelixInfo", 2, doc="LoopHelix parameters"),
    CompanionSpec("segpars_ch", "segpars_ch", "CentralHelixInfo", 2, doc="CentralHelix parameters"),
    CompanionSpec("segpars_kl", "segpars_kl", "KinematicLineInfo", 2, doc="KinematicLine parameters"),
    CompanionSpec("dtdt", "dtdt", "TrkDtDtInfo", 1, doc="dt/dt fit result"),
    CompanionSpec("calohit", "calohit", "TrkCaloHitInfo", 1, doc="calo cluster on the track"),
    CompanionSpec("hits", "hits", "TrkStrawHitInfo", 2, doc="straw hits on the track"),
    CompanionSpec("hitcalibs", "hitcalibs", "TrkStrawHitCalibInfo", 2, doc="straw hit calibrations"),
    CompanionSpec("mats", "mats", "TrkStrawMatInfo", 2, doc="straw materials crossed"),
    CompanionSpec("mc", "mc", "TrkInfoMC", 1, doc="MC truth for the track"),
    CompanionSpec("mcsim", "mcsim", "SimInfo", 2, doc="MC genealogy"),
    CompanionSpec("mcvd", "mcvd", "MCStepInfo", 2, doc="MC virtual detector steps"),
    CompanionSpec("calohitmc", "calohitmc", "CaloClusterInfoMC", 1, doc="MC truth for the calo hit"),
    CompanionSpec("hitsmc", "hitsmc", "TrkStrawHitInfoMC", 2, doc="MC truth for straw hits"),
    CompanionSpec("segsmc", "segsmc", "SurfaceStepInfo", 2, doc="MC surface steps"),
    # variable-suffix companions
    CompanionSpec("qual", "qual", "MVAResultInfo", 1, pattern=r"qual(_\w+)?", doc="TrkQual MVA output"),
    CompanionSpec("pid", "pid", "MVAResultInfo", 1, pattern=r"pid\d*", doc="TrkPID MVA output"),
    CompanionSpec("mcsic", "mcsic", "MCStepInfo", 2, pattern=r"mcsic_\w+", doc="extra MC step collection"),
    CompanionSpec("mcssi", "mcssi", "MCStepSummaryInfo", 2, pattern=r"mcssi_\w+", doc="extra MC step summary"),
)

#: Struct that identifies a track collection; its branch name is the FHiCL ``branchname``.
TRACK_STRUCT = "TrkInfo"

#: Structs whose branch name is FHiCL-configurable, so the collection must be found by class.
#: ``kind`` is the accessor family on :class:`~pyevtana.event.Event`.
CONFIGURABLE_COLLECTIONS: dict[str, str] = {
    "EventNtupleTimeClusterInfo": "timecluster",
    "LineSeedInfo": "lineseed",
}

#: Collections with fixed names (EventNtupleMaker_module.cc:818-890).
#: name -> (struct, depth, kind)
FIXED_COLLECTIONS: dict[str, tuple[str, int, str]] = {
    # calorimeter reco
    "caloclusters": ("CaloClusterInfo", 1, "calocluster"),
    "calohits": ("CaloHitInfo", 1, "calohit"),
    "calorecodigis": ("CaloRecoDigiInfo", 1, "calorecodigi"),
    "calodigis": ("CaloDigiInfo", 1, "calodigi"),
    # calorimeter MC
    "caloclustersmc": ("CaloClusterInfoMC", 1, "caloclustermc"),
    "calohitsmc": ("CaloHitInfoMC", 1, "calohitmc"),
    "calodigismc": ("CaloDigiMCInfo", 1, "calodigimc"),
    "calodigisim": ("SimInfo", 1, "sim"),
    "calomcsim": ("SimInfo", 1, "sim"),
    # CRV
    "crvcoincs": ("CrvHitInfoReco", 1, "crvcoinc"),
    "crvcoincsmc": ("CrvHitInfoMC", 1, "crvcoincmc"),
    "crvpulses": ("CrvPulseInfoReco", 1, "crvpulse"),
    "crvpulsesmc": ("CrvPulseInfoReco", 1, "crvpulse"),
    "crvdigis": ("CrvWaveformInfo", 1, "crvdigi"),
    "crvsummary": ("CrvSummaryReco", 0, "event"),
    "crvsummarymc": ("CrvSummaryMC", 0, "event"),
    # seeds
    "helices": ("HelixInfo", 1, "helix"),
    # event level
    "evtinfo": ("EventInfo", 0, "event"),
    "evtinfomc": ("EventInfoMC", 0, "event"),
    "hitcount": ("HitCount", 0, "event"),
    "lumistream": ("LumiStreamInfo", 0, "event"),
    "primary": ("SimInfo", 1, "sim"),
}

#: Collections whose name carries a variable instance/suffix.
#: regex -> (struct, depth, kind)
PATTERN_COLLECTIONS: tuple[tuple[str, tuple[str, int, str]], ...] = (
    (r"mcsteps_\w+", ("MCStepInfo", 1, "mcstep")),
    (r"crvcoincsmcplane(_\w+)?", ("CrvPlaneInfoMC", 1, "crvplanemc")),
)

#: Prefix for the per-path trigger result leaves.
TRIGGER_PREFIX = "trig_"
#: Prefix for the per-track-collection counters (``tcnt.n<branchname>``).
TCNT_PREFIX = "tcnt."

#: Track companions that are NOT indexed by track number. The maker pushes an entry only
#: under some condition, and stores no back-index, so the mapping has to be recovered.
#: role -> (gating companion, gating field, comparison) -- see Track.calohitmc.
NON_ALIGNED_COMPANIONS: dict[str, tuple[str, str, str]] = {
    "calohitmc": ("calohit", "did", ">=0"),
}

#: Object-to-object links that are *not* positional.  ``(field, target, kind)`` where kind is
#: "indices" (a vector of indices), "index" (a single index, -1 = none) or "backref"
#: (the target's field points back at us).
INDEX_LINKS: dict[str, tuple[str, str, str]] = {
    "caloclusters.hits": ("hits_", "calohits", "indices"),
    "calohits.recodigis": ("recoDigis_", "calorecodigis", "indices"),
    "calohits.cluster": ("clusterIdx_", "caloclusters", "index"),
    "calorecodigis.hit": ("caloHitIdx_", "calohits", "index"),
    "calorecodigis.digi": ("caloDigiIdx_", "calodigis", "index"),
    "calodigis.recodigi": ("caloRecoDigiIdx_", "calorecodigis", "index"),
    "crvpulses.coinc": ("crvHitIndex", "crvcoincs", "index"),
    "crvcoincs.pulses": ("crvHitIndex", "crvpulses", "backref"),
}

#: Why a branch is typically absent, used to make error messages actionable.
ABSENT_HINTS: dict[str, str] = {
    "hits": "EventNtupleMaker was run with trk.fillHits: false or fits[i].options.fillHits: false",
    "hitsmc": "requires both fillHits and fillMC",
    "hitcalibs": "EventNtupleMaker was run with trk.fillHitCalibs: false",
    "mats": "EventNtupleMaker was run with trk.fillHits: false",
    "mc": "EventNtupleMaker was run with fillMC: false (data, or MC branches disabled)",
    "mcsim": "EventNtupleMaker was run with fillMC: false",
    "mcvd": "EventNtupleMaker was run with fillMC: false",
    "segsmc": "EventNtupleMaker was run with fillMC: false",
    "calohitmc": "EventNtupleMaker was run with fillMC: false",
    "dtdt": "EventNtupleMaker was run with trk.fillTrkDtDt: false, or trkDtDtTag was empty",
    "pid": "no trkPIDTags were configured for this track collection",
    "qual": "no trkQualLeaves were configured for this track collection",
    "segpars_lh": "this ntuple was made with a different fittype (see trk.fittype)",
    "segpars_ch": "this ntuple was made with a different fittype (see trk.fittype)",
    "segpars_kl": "this ntuple was made with a different fittype (see trk.fittype)",
}


def absent_hint(role: str) -> str:
    return ABSENT_HINTS.get(role, "")


@dataclass
class TrkQualModel:
    """One row of the ``trkqual_metadata`` histogram."""

    leaf: str
    input_tag: str = ""
    model_version: str = ""


#: The default tree path inside an EventNtuple file.
DEFAULT_TREE = "EventNtuple/ntuple"
#: The per-subrun tree.
SUBRUN_TREE = "EventNtuple/subrunNtuple"


#: role -> (suffix, depth), derived from TRACK_COMPANIONS for quick lookup by the proxies.
TRACK_ROLES: dict[str, tuple[str, int]] = {
    spec.role: (spec.suffix, spec.depth) for spec in TRACK_COMPANIONS
}


def track_branch(tag: str, role_or_suffix: str) -> str:
    """Branch name a track companion would have, whether or not the file has it."""
    suffix, _ = TRACK_ROLES.get(role_or_suffix, (role_or_suffix, 1))
    return tag + suffix


def track_depth(role_or_suffix: str) -> int:
    """1 = one entry per track, 2 = a vector per track."""
    if role_or_suffix in TRACK_ROLES:
        return TRACK_ROLES[role_or_suffix][1]
    for spec in TRACK_COMPANIONS:
        if spec.matches(role_or_suffix):
            return spec.depth
    return 1
