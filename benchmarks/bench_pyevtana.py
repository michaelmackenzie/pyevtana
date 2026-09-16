#!/usr/bin/env python3
"""Time pyevtana on pyfitter's selection.

Runs `pyfitter_selection.py` -- a cut-for-cut reimplementation of pyfitter's
`AnaProcessor` selection -- over the same files, reading the same branch list, so the
timing is a like-for-like comparison. `validate_against_pyfitter.py` checks the two agree
stage by stage.

Modes:
  objects  the full selection through pyevtana's object API
  read     the same branches read and normalized, no analysis -- the floor both share

    python3 benchmarks/bench_pyevtana.py --files 6 --jobs 1
"""

import argparse
import json
import os
import sys

HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.dirname(HERE)
sys.path.insert(0, ROOT)
sys.path.insert(0, HERE)

import common
import pyfitter_selection
from pyevtana import Dataset

#: read exactly what pyfitter reads, so neither side has an I/O advantage
BRANCHES = pyfitter_selection.PYFITTER_BRANCHES


def worker(chunk):
    """Full selection over one partition. Returns plain, picklable results."""
    return pyfitter_selection.run(chunk)


def worker_read(chunk):
    """Read and normalize the same branches, no analysis: the shared floor."""
    n_events = 0
    for batch in chunk.batches():
        n_events += len(batch)
        for name in BRANCHES:
            batch.get(name)
    return {"n_events": n_events, "stages": [], "events_at": [], "selected": 0}


def merge(results):
    results = [r for r in results if r]
    if not results:
        return {"n_events": 0, "events_at": [], "selected": 0, "stages": []}
    events_at = [sum(v) for v in zip(*(r["events_at"] for r in results))] \
        if results[0]["events_at"] else []
    return {"n_events": sum(r["n_events"] for r in results),
            "events_at": events_at,
            "selected": sum(r["selected"] for r in results),
            "stages": results[0]["stages"]}


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--files", type=int, default=6)
    parser.add_argument("--skip", type=int, default=0)
    parser.add_argument("--jobs", type=int, default=1)
    parser.add_argument("--mode", default="objects", choices=["objects", "read"])
    parser.add_argument("--backend", default="process",
                        choices=["serial", "thread", "process"])
    parser.add_argument("--no-warm", action="store_true")
    parser.add_argument("--json", default=None)
    args = parser.parse_args()

    paths = common.files(args.files, args.skip)
    n_events = common.count_events(paths)
    nbytes = common.compressed_bytes(paths, BRANCHES)

    if not args.no_warm:
        warm = common.warm_cache(paths)
        print(f"cache warm-up: {warm:.1f} s over {len(paths)} files")

    dataset = Dataset(paths, branches=BRANCHES)
    backend = "serial" if args.jobs == 1 else args.backend
    function = worker if args.mode == "objects" else worker_read

    with common.Timer() as timer:
        result = dataset.map(function, workers=args.jobs, backend=backend, reduce=merge)

    row = common.report(f"pyevtana {args.mode} (jobs={args.jobs}, backend={backend})",
                        n_events, timer,
                        extra=f"branches    : {nbytes / 1e6:.1f} MB compressed, "
                              f"{nbytes / 1e6 / timer.wall:.1f} MB/s")
    if result["events_at"]:
        print(f"  selected   : {result['events_at'][-1]} events after "
              f"{len(result['stages'])} cuts, {result['selected']} tracks")

    row.update(framework="pyevtana", mode=args.mode, jobs=args.jobs, backend=backend,
               nfiles=len(paths), mbytes=nbytes / 1e6,
               selected=result["events_at"][-1] if result["events_at"] else 0)
    if args.json:
        with open(args.json, "a") as handle:
            handle.write(json.dumps(row) + "\n")


if __name__ == "__main__":
    main()
