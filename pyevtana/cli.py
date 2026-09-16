"""``pyevtana-describe`` -- print what is actually in a file.

Shows every collection with its struct, depth and branch state, so that a surprising
``MissingBranch`` or ``BranchNotSelected`` can be diagnosed without guessing.
"""

from __future__ import annotations

import argparse
import sys

from . import schema as S
from .discovery import NtupleSchema
from .missing import BranchState
from .select import Selector

_MARK = {
    BranchState.LOADED: "+",
    BranchState.NOT_SELECTED: "-",
    BranchState.ABSENT: " ",
}


def format_schema(schema: NtupleSchema, selector: Selector = None) -> str:
    out: list[str] = []
    add = out.append

    add(f"file      : {schema.source or '?'}")
    add(f"version   : {'.'.join(str(v) for v in schema.version)}")
    if selector is not None:
        add(f"branches  : {selector!r}")
    add("")
    add("legend: [+] loaded   [-] present but excluded by branches=   [ ] absent")
    add("sizes are compressed bytes; unsplit branches must be read whole")
    add("")

    for tag, track in sorted(schema.tracks.items()):
        counter = schema.counters.get(tag, "-")
        add(f"track collection {tag!r}   (count leaf: {counter})")
        seen = set()
        for spec in S.TRACK_COMPANIONS:
            for role, info in sorted(track.companions.items()):
                if info.role != spec.role or role in seen:
                    continue
                seen.add(role)
                add(f"  [{_MARK[info.state]}] {info.name:<24} {info.struct:<24} "
                    f"depth={info.depth}  {_leafcount(info)}")
            if spec.role not in {i.role for i in track.companions.values()}:
                name = S.track_branch(tag, spec.role)
                hint = S.absent_hint(spec.role)
                add(f"  [ ] {name:<24} {spec.struct:<24} depth={spec.depth}"
                    + (f"  ({hint})" if hint else ""))
        add("")

    by_kind: dict[str, list] = {}
    for info in schema.collections.values():
        if info.kind == "track":
            continue
        by_kind.setdefault(info.kind, []).append(info)
    if by_kind:
        add("other collections")
        for kind in sorted(by_kind):
            for info in sorted(by_kind[kind], key=lambda i: i.name):
                add(f"  [{_MARK[info.state]}] {info.name:<24} {info.struct:<24} "
                    f"depth={info.depth}  kind={kind}  {_leafcount(info)}")
        add("")

    if schema.trkqual_models:
        add("TrkQual models")
        for model in schema.trkqual_models:
            add(f"  {model.leaf:<16} tag={model.input_tag:<22} version={model.model_version}")
        add("")

    if schema.triggers:
        add(f"triggers  : {len(schema.triggers)} paths "
            f"(e.g. {', '.join(t[len(S.TRIGGER_PREFIX):] for t in schema.triggers[:3])} ...)")
    if schema.other:
        add(f"unclaimed : {', '.join(sorted(schema.other))}")
    if schema.mismatches:
        add("")
        add("SCHEMA MISMATCHES (branch exists but does not hold the expected struct):")
        for problem in schema.mismatches:
            add(f"  ! {problem}")

    return "\n".join(out)


def _humanize(nbytes: int) -> str:
    if not nbytes:
        return " " * 9
    if nbytes >= 1e6:
        return f"{nbytes / 1e6:6.1f} MB"
    return f"{nbytes / 1e3:6.1f} kB"


def _leafcount(info) -> str:
    """Leaf count plus size, so the I/O cost of touching a branch is visible.

    An unsplit branch is all-or-nothing: ROOT cannot split a ``vector<vector<T>>``, so
    reading one field of it costs the whole branch. On a typical EventNtuple that is where
    the read time goes -- ``trksegpars_lh`` is 32 MB and an analysis often wants one of
    its fifteen fields.
    """
    size = _humanize(info.nbytes)
    if not info.split:
        return f"{size}  (unsplit: reading any field reads all {len(info.leaves) or ''}".rstrip() + " fields)"
    return f"{size}  {len(info.selected)}/{len(info.leaves)} leaves"


def main(argv=None) -> int:
    parser = argparse.ArgumentParser(
        prog="pyevtana-describe",
        description="Print the EventNtuple schema pyevtana discovers in a file.")
    parser.add_argument("file", help="an EventNtuple ROOT file")
    parser.add_argument("--tree", default=S.DEFAULT_TREE, help="tree path inside the file")
    parser.add_argument("--branches", nargs="*", default=None,
                        help="branch filter to apply, to see what it would select")
    args = parser.parse_args(argv)

    from .reader import Dataset

    dataset = Dataset(args.file, tree=args.tree, branches=args.branches, on_missing="empty")
    print(format_schema(dataset.schema, dataset._selector))
    return 0


if __name__ == "__main__":
    sys.exit(main())
