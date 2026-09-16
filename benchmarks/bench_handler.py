#!/usr/bin/env python3
"""Does the uproot source handler explain pyevtana's weaker parallel scaling?

MemmapSource is much faster single-process -- one open instead of 364, no syscall per read
-- but memory-mapped reads are served by page faults, and page-fault handling on a network
filesystem need not scale across processes the way explicit pread() does. This reads the
same branches with each handler at several worker counts.

The cache is warmed once over the largest file set (the smaller sets are prefixes of it),
and the two handlers are measured back to back at each worker count, so neither is
favoured by drift.

    python3 benchmarks/bench_handler.py --files-per-worker 3 --jobs 1 5 10 20
"""

import argparse
import os
import sys

import uproot

HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.dirname(HERE)
sys.path.insert(0, ROOT)
sys.path.insert(0, HERE)

import common
import pyfitter_selection
from pyevtana import Dataset

BRANCHES = pyfitter_selection.PYFITTER_BRANCHES
HANDLERS = [("memmap", uproot.source.file.MemmapSource), ("uproot-default", None)]


def read_only(chunk):
    n = 0
    for batch in chunk.batches():
        n += len(batch)
        for name in BRANCHES:
            batch.get(name)
    return n


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--files-per-worker", type=int, default=3)
    parser.add_argument("--jobs", type=int, nargs="+", default=[1, 5, 10, 20])
    args = parser.parse_args()

    biggest = common.files(max(args.jobs) * args.files_per_worker)
    print(f"warming {len(biggest)} files once ...", flush=True)
    print(f"  {common.warm_cache(biggest, workers=12):.1f} s\n")

    print(f"{'workers':>8}{'files':>7}{'events':>9}  "
          f"{'handler':<16}{'wall':>9}{'cpu/wall':>10}{'events/s':>11}{'speed-up':>10}")
    print("-" * 82)
    base = {}
    for jobs in args.jobs:
        paths = common.files(jobs * args.files_per_worker)
        n_events = common.count_events(paths)
        for name, handler in HANDLERS:
            dataset = Dataset(paths, branches=BRANCHES, handler=handler)
            backend = "serial" if jobs == 1 else "process"
            with common.Timer() as timer:
                dataset.map(read_only, workers=jobs, backend=backend, reduce=sum)
            rate = n_events / timer.wall
            base.setdefault(name, rate)
            print(f"{jobs:>8}{len(paths):>7}{n_events:>9}  {name:<16}{timer.wall:>8.1f}s"
                  f"{timer.parallel_efficiency:>10.2f}{rate:>11.0f}"
                  f"{rate / base[name]:>9.1f}x", flush=True)
        print()


if __name__ == "__main__":
    main()
