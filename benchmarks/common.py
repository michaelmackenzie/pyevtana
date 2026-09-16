"""Shared harness for the pyevtana vs pyfitter/pyutils speed comparison.

Both frameworks are driven the same way: build a file list of N files, warm the page
cache, then time only the processing phase (imports and file-list construction excluded).
Warming matters -- these files live on /pnfs, and a cold first read measures dCache rather
than either framework.
"""

import os
import resource
import time

FILELIST = ("/exp/mu2e/app/users/mmackenz/main/Mu2eEvtAna/file_lists/"
            "nts.mu2e.CeMLeadingLogMix1BB.MDC2025au_best_v1_1.root.files")

#: branches pyfitter's AnaProcessor configures, used for cache warming so that every
#: framework starts from the same page-cache state
WARM_BRANCHES = ["evtinfo", "trk", "trkqual", "trkpid", "trksegs", "trksegsmc",
                 "trksegpars_lh", "trkmcsim", "trkmc", "crvcoincs", "caloclusters"]


def files(n: int, skip: int = 0) -> list:
    with open(FILELIST) as handle:
        paths = [line.strip() for line in handle if line.strip() and not line.startswith("#")]
    return paths[skip:skip + n]


def write_filelist(paths: list, path: str) -> str:
    with open(path, "w") as handle:
        handle.write("\n".join(paths) + "\n")
    return path


def count_events(paths: list) -> int:
    import uproot

    total = 0
    for p in paths:
        with uproot.open(p) as handle:
            total += handle["EventNtuple/ntuple"].num_entries
    return total


def compressed_bytes(paths: list, branches: list) -> int:
    """Compressed size of a branch set, so throughput can be quoted per byte as well."""
    import uproot

    def size(branch):
        return branch.compressed_bytes + sum(size(c) for c in branch.branches)

    total = 0
    for p in paths:
        with uproot.open(p) as handle:
            tree = handle["EventNtuple/ntuple"]
            for name in branches:
                try:
                    total += size(tree[name])
                except Exception:
                    pass
    return total


def warm_cache(paths: list, branches=None, workers: int = 8) -> float:
    """Read the branches once so the comparison measures CPU, not dCache latency."""
    import concurrent.futures

    import uproot

    branches = branches or WARM_BRANCHES
    start = time.time()

    def read(path):
        with uproot.open(path) as handle:
            tree = handle["EventNtuple/ntuple"]
            for name in branches:
                try:
                    tree[name].array()
                except Exception:
                    pass

    with concurrent.futures.ThreadPoolExecutor(max_workers=workers) as pool:
        list(pool.map(read, paths))
    return time.time() - start


class Timer:
    """Wall time plus CPU time of this process and all its children."""

    def __enter__(self):
        self.t0 = time.perf_counter()
        self.c0 = _cpu()
        return self

    def __exit__(self, *exc):
        self.wall = time.perf_counter() - self.t0
        self.cpu = _cpu() - self.c0

    @property
    def parallel_efficiency(self):
        return self.cpu / self.wall if self.wall else 0.0


def _cpu() -> float:
    me = resource.getrusage(resource.RUSAGE_SELF)
    kids = resource.getrusage(resource.RUSAGE_CHILDREN)
    return me.ru_utime + me.ru_stime + kids.ru_utime + kids.ru_stime


def report(label: str, n_events: int, timer: Timer, extra: str = "") -> dict:
    rate = n_events / timer.wall if timer.wall else 0.0
    print(f"\n=== {label} ===")
    print(f"  events     : {n_events}")
    print(f"  wall       : {timer.wall:8.2f} s")
    print(f"  cpu        : {timer.cpu:8.2f} s  (cpu/wall = {timer.parallel_efficiency:.2f})")
    print(f"  throughput : {rate:8.1f} events/s")
    if extra:
        print(f"  {extra}")
    return {"label": label, "events": n_events, "wall": timer.wall, "cpu": timer.cpu,
            "rate": rate}
