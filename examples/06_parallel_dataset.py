#!/usr/bin/env python3
"""Processing a dataset in parallel, one partition per file.

The worker function is module-level on purpose: it has to be picklable to run in a worker
process.  It must also return plain data -- Event/Track proxies hold a live branch cache
and refuse to be pickled.

    python3 examples/06_parallel_dataset.py [files...] --workers 4 --backend process
"""

from collections import Counter

from _common import parse

from pyevtana import Dataset
from pyevtana.accumulate import merge_dicts


def analyze(chunk):
    """Runs once per partition. Returns plain, mergeable results."""
    pdgs = Counter()
    momenta = []
    for event in chunk:
        for track in event.Tracks():
            if track.nactive < 20:
                continue
            mid = track.seg("TT_Mid")
            if mid is None:
                continue
            pdgs[int(track.pdg)] += 1
            momenta.append(float(mid.mom.mag))
    return {"pdgs": pdgs, "n": len(momenta), "sum_p": sum(momenta)}


if __name__ == "__main__":
    args = parse(__doc__,
                 workers={"type": int, "default": 4},
                 backend={"default": "process",
                          "choices": ["serial", "thread", "process"]},
                 max_entries={"type": int, "default": 0,
                              "help": "split files into partitions of this many entries"})

    dataset = Dataset(args.files, branches=["evtinfo", "trk", "trksegs"])
    print(f"{dataset}\n{len(dataset.paths)} file(s), {dataset.num_entries} events")

    result = dataset.map(
        analyze,
        workers=args.workers,
        backend=args.backend,
        reduce=merge_dicts,
        max_entries=args.max_entries or None,
        on_error="skip",
        progress=True,
    )

    print(f"\nselected tracks : {result['n']}")
    print(f"mean momentum   : {result['sum_p'] / max(result['n'], 1):.3f} MeV/c")
    print(f"by pdg          : {dict(sorted(result['pdgs'].items()))}")

    if dataset.errors:
        print(f"\n{len(dataset.errors)} file(s) failed and were skipped:")
        for path, message in dataset.errors:
            print(f"  {path}: {message.splitlines()[-1]}")

    # Normalization must cover every file, including any that were skipped.
    summary = dataset.summary()
    print(f"\nbookkeeping     : {summary}")
    if not summary.complete:
        print("  WARNING: some files could not be read; do not normalize with this")
