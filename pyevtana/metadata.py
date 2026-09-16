"""Run/subrun bookkeeping: the numbers an analysis needs to normalize its result.

These live in the sibling ``subrunNtuple`` tree and in the ``n_proc_events`` histogram,
not in the event tree, so they are read separately from the event loop.

One trap worth stating plainly: these totals must be summed over **every** file in the
dataset, including any that :meth:`Dataset.map` skipped via ``on_error='skip'``.  A
normalization computed from the files that happened to read successfully, divided into a
yield computed from those same files, is fine; a normalization computed from all files
while some were skipped -- or the reverse -- is silently wrong.  :class:`DatasetSummary`
therefore reports the files it could not read rather than quietly leaving them out.
"""

from __future__ import annotations

import os
from dataclasses import dataclass, field
from typing import Optional, Sequence

import uproot

from . import schema as S


@dataclass
class DatasetSummary:
    """Totals over a set of files, with the failures kept visible."""

    files: list[str] = field(default_factory=list)
    unreadable: list[tuple[str, str]] = field(default_factory=list)
    runs: set = field(default_factory=set)
    subruns: set = field(default_factory=set)
    proc_events: int = 0
    gen_events: int = 0
    cosmic_livetime: float = 0.0

    @property
    def complete(self) -> bool:
        """False if any file could not be read -- do not normalize with this if False."""
        return not self.unreadable

    @property
    def n_files(self) -> int:
        return len(self.files)

    @property
    def n_subruns(self) -> int:
        return len(self.subruns)

    def __repr__(self) -> str:
        state = "complete" if self.complete else f"INCOMPLETE ({len(self.unreadable)} unreadable)"
        return (f"<DatasetSummary {self.n_files} files, {self.n_subruns} subruns, "
                f"proc={self.proc_events}, gen={self.gen_events}, "
                f"livetime={self.cosmic_livetime:g}s, {state}>")


def read_subruns(path: str, tree_path: str = S.SUBRUN_TREE):
    """The per-subrun tree of one file as an awkward array, or ``None`` if absent."""
    with uproot.open(path) as handle:
        try:
            tree = handle[tree_path]
        except Exception:
            return None
        return tree.arrays()


def summarize(paths: Sequence[str], tree_path: str = S.SUBRUN_TREE,
              on_error: str = "raise") -> DatasetSummary:
    """Sum run/subrun bookkeeping over a list of files."""
    summary = DatasetSummary()
    for path in paths:
        try:
            arrays = read_subruns(path, tree_path)
        except Exception as error:
            if on_error == "raise":
                raise
            summary.unreadable.append((path, str(error)))
            continue
        summary.files.append(path)
        if arrays is None or len(arrays) == 0:
            continue
        fields = arrays.fields
        if "run" in fields:
            summary.runs.update(int(r) for r in arrays["run"])
        if "run" in fields and "subrun" in fields:
            summary.subruns.update(
                (int(r), int(s)) for r, s in zip(arrays["run"], arrays["subrun"])
            )
        for name, attr in (("procEventCount", "proc_events"),
                           ("genEventCount", "gen_events")):
            if name in fields:
                setattr(summary, attr, getattr(summary, attr) + int(sum(arrays[name])))
        if "cosmicLivetime" in fields:
            summary.cosmic_livetime += float(sum(arrays["cosmicLivetime"]))
    return summary


def n_proc_events(path: str, directory: str = "EventNtuple") -> Optional[int]:
    """The ``n_proc_events`` histogram total, a cross-check on ``procEventCount``."""
    with uproot.open(path) as handle:
        try:
            return int(sum(handle[f"{directory}/n_proc_events"].values()))
        except Exception:
            return None
