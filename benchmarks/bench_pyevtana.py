#!/usr/bin/env python3
"""Time pyevtana on the CeMLeadingLog selection.

Runs the selection from examples/07_signal_selection.py, which is the same analysis
pyfitter's AnaProcessor performs, so the two timings are comparable.

    python3 benchmarks/bench_pyevtana.py --files 2 --jobs 1
"""

import argparse
import importlib
import json
import os
import sys

HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.dirname(HERE)
sys.path.insert(0, ROOT)
sys.path.insert(0, os.path.join(ROOT, "examples"))

import common
from pyevtana import Dataset

selection = importlib.import_module("07_signal_selection")

#: exactly the branches the selection touches; everything else is never read
BRANCHES = ["evtinfo", "trk", "trksegs", "trksegpars_lh", "trkqual", "trkpid",
            "trkcalohit", "crvcoincs"]


def worker(chunk):
    """Full object-loop selection over one partition. Returns picklable results."""
    n_events, stages, events_at, tracks_in, _ = selection.run(chunk)
    return {"n_events": n_events, "events_at": events_at, "tracks": dict(tracks_in)}


def worker_arrays(chunk):
    """Read and normalize the same branches, then cut with awkward instead of objects.

    This is the floor for pyevtana's array path: it measures the I/O and normalization
    layer plus a vectorized per-track selection, with no per-object Python at all. The
    gap between this and `worker` is what the object API costs.
    """
    import awkward as ak

    n_events = 0
    selected = 0
    for batch in chunk.batches():
        n_events += len(batch)
        trk = batch.get("trk")
        qual = batch.get("trkqual")
        pid = batch.get("trkpid")
        segs = batch.get("trksegs")
        pars = batch.get("trksegpars_lh")
        batch.get("trkcalohit")
        batch.get("crvcoincs")
        batch.get("evtinfo")

        keep = ((trk.status >= 0) & (trk.goodfit == 1) & (trk.pdg == 11)
                & (trk.nactive >= 20) & (trk.nstup + trk.nstdown > 0)
                & (~trk.opainter) & (~trk.tsdainter)
                & (qual.result > 0.155) & (pid.result > 0.54))
        mid = segs[segs.sid == 1]                      # TT_Mid
        keep = keep & (ak.sum(mid.mom.z > 0, axis=-1) > 0)
        selected += int(ak.sum(keep))
        _ = ak.sum(pars.t0err < 0.85)
    return {"n_events": n_events, "events_at": [selected], "tracks": {"selected": selected}}


def merge(results):
    results = [r for r in results if r]
    if not results:
        return {"n_events": 0, "events_at": [], "tracks": {}}
    total = {"n_events": sum(r["n_events"] for r in results),
             "events_at": [sum(v) for v in zip(*(r["events_at"] for r in results))],
             "tracks": {}}
    for r in results:
        for key, value in r["tracks"].items():
            total["tracks"][key] = total["tracks"].get(key, 0) + value
    return total


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--files", type=int, default=2)
    parser.add_argument("--skip", type=int, default=0)
    parser.add_argument("--jobs", type=int, default=1)
    parser.add_argument("--backend", default="process",
                        choices=["serial", "thread", "process"])
    parser.add_argument("--mode", default="objects", choices=["objects", "arrays"])
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

    function = worker if args.mode == "objects" else worker_arrays
    with common.Timer() as timer:
        result = dataset.map(function, workers=args.jobs, backend=backend, reduce=merge)

    row = common.report(f"pyevtana {args.mode} (jobs={args.jobs}, backend={backend})",
                        n_events, timer,
                        extra=f"branches    : {nbytes / 1e6:.1f} MB compressed, "
                              f"{nbytes / 1e6 / timer.wall:.1f} MB/s")
    print(f"  selected   : {result['tracks'].get('selected', 0)} tracks, "
          f"{result['events_at'][-1] if result['events_at'] else 0} events")

    row.update(framework="pyevtana", mode=args.mode, jobs=args.jobs,
               backend=backend, nfiles=len(paths),
               mbytes=nbytes / 1e6, selected=result["tracks"].get("selected", 0))
    if args.json:
        with open(args.json, "a") as handle:
            handle.write(json.dumps(row) + "\n")


if __name__ == "__main__":
    main()
