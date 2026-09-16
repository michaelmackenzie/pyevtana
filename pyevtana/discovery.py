"""Work out what is actually in a file.

Discovery keys off the **C++ class** in each branch's ``typename``, not off branch names,
because track branch names (FHiCL ``branchname``) and time-cluster / line-seed names
(FHiCL ``names``) are all configurable.  Once the track tags are known, their companions
are claimed by suffix -- and a companion is only accepted if its struct is the one the
schema expects, so a schema change is reported rather than silently misread.

Discovery runs **per file**: files in one dataset may legitimately differ (data vs MC,
different reprocessings), so a dataset-wide schema is an intersection plus a diff.
"""

from __future__ import annotations

import re
from dataclasses import dataclass, field
from typing import Iterable, Optional

from . import schema as S
from .missing import BranchState, SchemaMismatch
from .select import Selector


def canonical(branch_name: str) -> str:
    """``"lumistream."`` -> ``"lumistream"``.

    ROOT keeps the trailing dot for split object branches and drops it for vector
    branches, so the dot cannot be part of the name users type.
    """
    return branch_name[:-1] if branch_name.endswith(".") else branch_name


@dataclass
class CollectionInfo:
    """One branch, plus everything needed to read a subset of it."""

    name: str                      # canonical name, what users pass around
    branch: str                    # actual name in the file (may carry a trailing dot)
    struct: str
    depth: int                     # 0 = per event, 1 = per object, 2 = vector per object
    kind: str                      # accessor family: track, calocluster, crvcoinc, ...
    leaves: list[str] = field(default_factory=list)    # readable sub-leaves ([] = unsplit)
    selected: list[str] = field(default_factory=list)  # subset the filter kept
    tag: str = ""                  # owning track tag, for track companions
    role: str = ""                 # companion role: info, hits, segs, mc, ...

    @property
    def split(self) -> bool:
        return bool(self.leaves)

    @property
    def state(self) -> BranchState:
        # Present in the file; LOADED unless the filter kept nothing.
        if self.split and not self.selected:
            return BranchState.NOT_SELECTED
        if not self.split and not self.selected:
            return BranchState.NOT_SELECTED
        return BranchState.LOADED

    def __repr__(self) -> str:
        return (f"<CollectionInfo {self.name} struct={self.struct} depth={self.depth} "
                f"{self.state.value} {len(self.selected)}/{len(self.leaves) or 1} leaves>")


@dataclass
class TrackCollection:
    """A track branch plus the companions that hang off it."""

    tag: str
    companions: dict[str, CollectionInfo] = field(default_factory=dict)

    @property
    def info(self) -> Optional[CollectionInfo]:
        return self.companions.get("info")

    def companion(self, role: str) -> Optional[CollectionInfo]:
        return self.companions.get(role)

    def branch_for(self, role: str) -> str:
        """The branch name a companion would have, whether or not it exists."""
        existing = self.companions.get(role)
        if existing is not None:
            return existing.name
        for spec in S.TRACK_COMPANIONS:
            if spec.role == role:
                return self.tag + spec.suffix
        return self.tag + role


@dataclass
class NtupleSchema:
    """Everything discovery learned about one file."""

    collections: dict[str, CollectionInfo] = field(default_factory=dict)
    tracks: dict[str, TrackCollection] = field(default_factory=dict)
    triggers: list[str] = field(default_factory=list)
    counters: dict[str, str] = field(default_factory=dict)   # tag -> tcnt leaf
    other: dict[str, str] = field(default_factory=dict)      # unclaimed branch -> typename
    mismatches: list[str] = field(default_factory=list)
    version: tuple[int, int, int] = (0, 0, 0)
    trkqual_models: list[S.TrkQualModel] = field(default_factory=list)
    source: str = ""

    # -- lookups ----------------------------------------------------------------------

    def get(self, name: str) -> Optional[CollectionInfo]:
        return self.collections.get(name)

    def state(self, name: str) -> BranchState:
        info = self.collections.get(name)
        return info.state if info is not None else BranchState.ABSENT

    def has(self, name: str) -> bool:
        return self.state(name) is BranchState.LOADED

    def of_kind(self, kind: str) -> list[CollectionInfo]:
        return [c for c in self.collections.values() if c.kind == kind]

    def names_of_kind(self, kind: str) -> list[str]:
        return [c.name for c in self.collections.values() if c.kind == kind]

    def track_tags(self) -> list[str]:
        return list(self.tracks)


# --------------------------------------------------------------------------------------
# the discovery pass
# --------------------------------------------------------------------------------------


def _leaves_of(branch) -> list[str]:
    try:
        return list(branch.keys())
    except Exception:
        return []


def _apply_filter(info: CollectionInfo, selector: Selector) -> None:
    if info.split:
        info.selected = [
            leaf for leaf in info.leaves if selector.matches(leaf, info.name)
        ]
    else:
        info.selected = [info.name] if selector.matches(info.name, info.name) else []


def discover(tree, selector: Optional[Selector] = None, source: str = "") -> NtupleSchema:
    """Inspect an open uproot TTree and return its :class:`NtupleSchema`."""
    selector = selector or Selector(None)
    result = NtupleSchema(source=source)

    raw: dict[str, tuple[str, object, str, Optional[tuple[int, str]]]] = {}
    for name, branch in tree.items(recursive=False):
        typename = branch.typename
        raw[canonical(name)] = (name, branch, typename, S.parse_typename(typename))

    claimed: set[str] = set()

    def make(canon: str, kind: str, tag: str = "", role: str = "") -> CollectionInfo:
        branch_name, branch, typename, parsed = raw[canon]
        depth, struct = parsed if parsed else (0, typename)
        info = CollectionInfo(
            name=canon, branch=branch_name, struct=struct, depth=depth, kind=kind,
            leaves=_leaves_of(branch), tag=tag, role=role,
        )
        _apply_filter(info, selector)
        result.collections[canon] = info
        claimed.add(canon)
        return info

    # 1. track collections, identified by class -----------------------------------------
    tags = sorted(
        (c for c, (_, _, _, p) in raw.items() if p == (1, S.TRACK_STRUCT)),
        key=len, reverse=True,
    )

    for tag in tags:
        track = TrackCollection(tag=tag)
        result.tracks[tag] = track
        for canon in sorted(raw):
            if canon in claimed or not canon.startswith(tag):
                continue
            remainder = canon[len(tag):]
            parsed = raw[canon][3]
            for spec in S.TRACK_COMPANIONS:
                if not spec.matches(remainder):
                    continue
                if parsed != (spec.depth, spec.struct):
                    # Right name, wrong contents: report it, do not guess.
                    got = parsed[1] if parsed else raw[canon][2]
                    result.mismatches.append(
                        f"{canon}: expected {spec.struct} at depth {spec.depth} for role "
                        f"{spec.role!r}, found {got}"
                    )
                    continue
                track.companions[spec.role if spec.pattern is None else _role_key(spec, remainder)] = \
                    make(canon, "track", tag=tag, role=spec.role)
                break

    # 2. per-track counters --------------------------------------------------------------
    for canon in raw:
        if canon.startswith(S.TCNT_PREFIX + "n"):
            tag = canon[len(S.TCNT_PREFIX) + 1:]
            if tag in result.tracks:
                result.counters[tag] = canon
                claimed.add(canon)

    # 3. configurable-name collections, identified by class --------------------------------
    for canon, (_, _, _, parsed) in sorted(raw.items()):
        if canon in claimed or parsed is None:
            continue
        kind = S.CONFIGURABLE_COLLECTIONS.get(parsed[1])
        if kind is not None:
            make(canon, kind)

    # 4. fixed-name collections ----------------------------------------------------------
    for canon, (_, _, typename, parsed) in sorted(raw.items()):
        if canon in claimed:
            continue
        expected = S.FIXED_COLLECTIONS.get(canon)
        if expected is None:
            continue
        struct, depth, kind = expected
        if parsed is not None and parsed != (depth, struct):
            result.mismatches.append(
                f"{canon}: expected {struct} at depth {depth}, found {parsed[1]} at depth {parsed[0]}"
            )
            continue
        make(canon, kind)

    # 5. pattern-named collections --------------------------------------------------------
    for canon, (_, _, _, parsed) in sorted(raw.items()):
        if canon in claimed or parsed is None:
            continue
        for pattern, (struct, depth, kind) in S.PATTERN_COLLECTIONS:
            if re.fullmatch(pattern, canon) and parsed == (depth, struct):
                make(canon, kind)
                break

    # 6. triggers and leftovers ------------------------------------------------------------
    for canon, (_, _, typename, _) in sorted(raw.items()):
        if canon in claimed:
            continue
        if canon.startswith(S.TRIGGER_PREFIX):
            result.triggers.append(canon)
            claimed.add(canon)
        else:
            result.other[canon] = typename

    return result


def _role_key(spec: S.CompanionSpec, remainder: str) -> str:
    """Variable-suffix companions keep their distinguishing part: ``qual``, ``qual_bdt``."""
    return remainder


def read_metadata(directory, schema: NtupleSchema) -> NtupleSchema:
    """Fill in ntuple version and TrkQual provenance from the sibling histograms."""
    try:
        hist = directory["version"]
        values = [int(v) for v in hist.values()]
        labels = [str(x) for x in (hist.axis().labels() or [])]
        by_label = dict(zip(labels, values))
        schema.version = (
            by_label.get("major", values[0] if len(values) > 0 else 0),
            by_label.get("minor", values[1] if len(values) > 1 else 0),
            by_label.get("patch", values[2] if len(values) > 2 else 0),
        )
    except Exception:
        pass

    try:
        labels = directory["trkqual_metadata"].axis().labels() or []
        for label in labels:
            leaf, _, rest = str(label).partition(":")
            model = S.TrkQualModel(leaf=leaf.strip())
            for piece in rest.split(";"):
                key, _, value = piece.partition("=")
                key, value = key.strip().lower(), value.strip()
                if key == "input tag":
                    model.input_tag = value
                elif key == "model version":
                    model.model_version = value
            schema.trkqual_models.append(model)
    except Exception:
        pass

    return schema
