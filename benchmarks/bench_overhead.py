#!/usr/bin/env python3
"""Is the remaining gap pyevtana's per-file machinery rather than the read itself?

Both arms do the *identical* uproot read -- one `tree.arrays()` per file for the same leaf
list. Arm A goes through a plain ProcessPoolExecutor; arm B goes through
`Dataset.map`, which additionally opens its own FileReader per partition and runs schema
discovery on each file. Any difference is pyevtana's overhead, not I/O.
"""

import argparse
import os
import sys
import time
from concurrent.futures import ProcessPoolExecutor

import uproot

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

import common
import pyfitter_selection
from pyevtana import Dataset

BRANCHES = pyfitter_selection.PYFITTER_BRANCHES
LEAVES = None          # filled in main(), then inherited by forked workers
HANDLER = None


def plain_read(path):
    f = uproot.open(path, handler=HANDLER) if HANDLER else uproot.open(path)
    tree = f["EventNtuple/ntuple"]
    tree.arrays(LEAVES)
    n = tree.num_entries
    f.close()
    return n


def via_dataset(chunk):
    tree = chunk.reader.tree                 # forces open() + discovery
    tree.arrays(LEAVES)
    return tree.num_entries


def discovery_cost(paths, handler):
    from pyevtana.discovery import discover, read_metadata
    from pyevtana.select import Selector
    total = 0.0
    for p in paths[:5]:
        f = uproot.open(p, handler=handler) if handler else uproot.open(p)
        t0 = time.perf_counter()
        schema = discover(f["EventNtuple/ntuple"], Selector(BRANCHES))
        read_metadata(f["EventNtuple"], schema)
        total += time.perf_counter() - t0
        f.close()
    return total / 5


def main():
    global LEAVES, HANDLER
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--files", type=int, default=60)
    parser.add_argument("--jobs", type=int, default=20)
    parser.add_argument("--handler", default="memmap", choices=["memmap", "default"])
    args = parser.parse_args()

    HANDLER = uproot.source.file.MemmapSource if args.handler == "memmap" else None
    paths = common.files(args.files)
    dataset = Dataset(paths, branches=BRANCHES,
                      handler=HANDLER if args.handler == "memmap" else None)
    reader = dataset.reader(paths[0])
    LEAVES = []
    for name in BRANCHES:
        info = reader.schema.get(name)
        LEAVES.extend(info.selected if info.split else [info.branch])
    dataset.close()

    mbytes = common.compressed_bytes(paths, BRANCHES) / 1e6
    n_events = common.count_events(paths)
    common.warm_cache(paths, workers=12)
    print(f"{len(paths)} files, {n_events} events, {mbytes:.0f} MB, "
          f"{args.jobs} workers, handler={args.handler}")
    print(f"schema discovery costs {1000 * discovery_cost(paths, HANDLER):.0f} ms per file\n")

    print(f"{'arm':<34}{'wall':>9}{'cpu/wall':>10}{'MB/s':>9}{'ev/s':>9}")
    print("-" * 71)

    with common.Timer() as timer:
        with ProcessPoolExecutor(max_workers=args.jobs) as pool:
            list(pool.map(plain_read, paths))
    print(f"{'A: plain pool + uproot.arrays':<34}{timer.wall:>8.1f}s"
          f"{timer.parallel_efficiency:>10.2f}{mbytes / timer.wall:>9.1f}"
          f"{n_events / timer.wall:>9.0f}", flush=True)

    dataset = Dataset(paths, branches=BRANCHES,
                      handler=HANDLER if args.handler == "memmap" else None)
    with common.Timer() as timer:
        dataset.map(via_dataset, workers=args.jobs, backend="process", reduce=sum)
    print(f"{'B: Dataset.map + same arrays()':<34}{timer.wall:>8.1f}s"
          f"{timer.parallel_efficiency:>10.2f}{mbytes / timer.wall:>9.1f}"
          f"{n_events / timer.wall:>9.0f}")


if __name__ == "__main__":
    main()
