#!/usr/bin/env python3
"""Is pyevtana's weaker parallel scaling caused by how it splits up its reads?

pyevtana reads lazily: one `arrays()` call per branch, per batch. On one EventNtuple file
with this branch list that is 11 x 5 = 55 separate calls, where pyfitter issues one.
uproot coalesces basket reads only *within* a single call, so splitting them up means many
small scattered requests instead of a few large sequential ones -- which costs little when
one process has the disk to itself and a lot when twenty are competing.

Four shapes, at 1 and N workers:
  per-branch/batched    what pyevtana does today
  per-branch/whole-file one batch per file, still one call per branch
  coalesced/batched     one call for all branches, per batch
  coalesced/whole-file  one call for everything -- what pyfitter does
"""

import argparse
import os
import sys
import time

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

import common
import pyfitter_selection
from pyevtana import Dataset

BRANCHES = pyfitter_selection.PYFITTER_BRANCHES


def _names(reader):
    """Leaf names for a coalesced read: split branches by leaf, unsplit by branch."""
    out = []
    for name in BRANCHES:
        info = reader.schema.get(name)
        if info is None:
            continue
        out.extend(info.selected if info.split else [info.branch])
    return out


def per_branch_batched(chunk):
    n = 0
    for batch in chunk.batches():
        n += len(batch)
        for name in BRANCHES:
            batch.get(name)
    return n


def per_branch_whole(chunk):
    reader = chunk.reader
    tree = reader.tree
    n = tree.num_entries
    for name in BRANCHES:
        info = reader.schema.get(name)
        b = tree[info.branch]
        b.arrays(info.selected) if info.split else b.array()
    return n


def coalesced_batched(chunk):
    reader = chunk.reader
    tree = reader.tree
    names = _names(reader)
    n = 0
    for batch in chunk.batches():
        n += len(batch)
        tree.arrays(names, entry_start=batch.start, entry_stop=batch.stop)
    return n


def coalesced_whole(chunk):
    reader = chunk.reader
    tree = reader.tree
    tree.arrays(_names(reader))
    return tree.num_entries


SHAPES = [("per-branch/batched", per_branch_batched),
          ("per-branch/whole-file", per_branch_whole),
          ("coalesced/batched", coalesced_batched),
          ("coalesced/whole-file", coalesced_whole)]


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--files-per-worker", type=int, default=3)
    parser.add_argument("--jobs", type=int, nargs="+", default=[1, 10])
    args = parser.parse_args()

    biggest = common.files(max(args.jobs) * args.files_per_worker)
    print(f"warming {len(biggest)} files once ... ", end="", flush=True)
    print(f"{common.warm_cache(biggest, workers=12):.1f} s\n")

    print(f"{'workers':>8}  {'read shape':<24}{'wall':>9}{'cpu/wall':>10}{'MB/s':>9}{'ev/s':>9}")
    print("-" * 70)
    for jobs in args.jobs:
        paths = common.files(jobs * args.files_per_worker)
        n_events = common.count_events(paths)
        mbytes = common.compressed_bytes(paths, BRANCHES) / 1e6
        for label, fn in SHAPES:
            dataset = Dataset(paths, branches=BRANCHES)
            backend = "serial" if jobs == 1 else "process"
            with common.Timer() as timer:
                dataset.map(fn, workers=jobs, backend=backend, reduce=sum)
            print(f"{jobs:>8}  {label:<24}{timer.wall:>8.1f}s"
                  f"{timer.parallel_efficiency:>10.2f}{mbytes / timer.wall:>9.1f}"
                  f"{n_events / timer.wall:>9.0f}", flush=True)
        print()


if __name__ == "__main__":
    main()
