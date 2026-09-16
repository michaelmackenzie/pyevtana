#!/usr/bin/env python3
"""Check that the pyevtana benchmark selection reproduces pyfitter's, stage by stage.

Both report a per-event cumulative cut flow -- the number of events still holding at least
one surviving track -- so they can be compared directly. Any stage that differs is a real
difference in the selection, and the benchmark is only a fair speed comparison once every
stage matches.

    python3 benchmarks/validate_against_pyfitter.py --files 2
"""

import argparse
import os
import sys
import tempfile

HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.dirname(HERE)
PYFITTER = "/exp/mu2e/app/users/mmackenz/main/pyfitter"
sys.path.insert(0, HERE)
sys.path.insert(0, ROOT)

import common
import pyfitter_selection
from pyevtana import Dataset

# pyfitter's own 22 named switches; st_boundary is not among them (it reuses has_st's
# switch), so it is excluded here but still appears in the cut flow.
CUT_NAMES = [n for n, _ in pyfitter_selection.CUTS if n != "st_boundary"] + [
    "within_t0_540", "within_t0_640", "signal_region"]
SWITCHES = [True] * 19 + [False] * 3


def pyfitter_flow(paths):
    sys.path.insert(0, PYFITTER)
    os.chdir(PYFITTER)
    from process import AnaProcessor

    with tempfile.NamedTemporaryFile("w", suffix=".txt", delete=False) as handle:
        handle.write("\n".join(paths) + "\n")
        listfile = handle.name
    try:
        processor = AnaProcessor(file_list_path=listfile, jobs=1,
                                 cuts=dict(zip(CUT_NAMES, SWITCHES)), location="local")
        results = processor.execute()
    finally:
        os.unlink(listfile)

    totals = {}
    order = []
    for result in results or []:
        for row in result["cut_stats"]:
            if row["name"] == "No cuts":
                continue
            if row["name"] not in totals:
                order.append(row["name"])
            totals[row["name"]] = totals.get(row["name"], 0) + row["events_passing"]
    return order, totals


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--files", type=int, default=2)
    args = parser.parse_args()

    paths = common.files(args.files)
    order, pyf = pyfitter_flow(paths)

    os.chdir(ROOT)
    result = pyevtana_run(paths)
    mine = dict(zip(result["stages"], result["events_at"]))

    print(f"\nper-event cumulative cut flow, {result['n_events']} events, {len(paths)} file(s)\n")
    print(f"  {'stage':<32} {'pyfitter':>10} {'pyevtana':>10} {'diff':>8}")
    print(f"  {'-' * 32} {'-' * 10} {'-' * 10} {'-' * 8}")
    mismatches = 0
    for name in order:
        a, b = pyf.get(name, 0), mine.get(name)
        if b is None:
            print(f"  {name:<32} {a:>10} {'MISSING':>10}")
            mismatches += 1
            continue
        flag = "" if a == b else "  <-- differs"
        mismatches += (a != b)
        print(f"  {name:<32} {a:>10} {b:>10} {b - a:>8}{flag}")
    print(f"\n  {mismatches} stage(s) differ" if mismatches else "\n  all stages match")
    return 1 if mismatches else 0


def pyevtana_run(paths):
    dataset = Dataset(paths, branches=pyfitter_selection.BRANCHES)
    return pyfitter_selection.run(dataset)


if __name__ == "__main__":
    sys.exit(main())
