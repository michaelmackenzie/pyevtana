#!/usr/bin/env python3
"""Time pyfitter's AnaProcessor (which is built on pyutils) on the same files.

    python3 benchmarks/bench_pyfitter.py --files 2 --jobs 1
"""

import argparse
import json
import os
import sys
import tempfile

HERE = os.path.dirname(os.path.abspath(__file__))
PYFITTER = "/exp/mu2e/app/users/mmackenz/main/pyfitter"
sys.path.insert(0, HERE)
sys.path.insert(0, PYFITTER)

import common

#: the selection pyfitter applies by default (process.py), all cuts through within_t0_475
#: pyfitter's 22 named switches, in its own order (process.py). `st_boundary` is not a
#: named switch -- it reuses has_st's -- but it does appear in the resulting cut flow.
CUT_NAMES = [
    "has_a_track", "is_good_track", "has_trk_front_seg", "is_reco_electron_or_positron",
    "has_downstream", "charge_selection", "or_trigger", "upstream_veto",
    "no_multi_trk_veto", "good_trkpid", "pz_over_pt", "has_st", "no_opa", "good_trkqual",
    "has_hits", "within_t0err", "no_crv_veto", "in_mom_range", "within_t0_475",
    "within_t0_540", "within_t0_640", "signal_region",
]
SWITCHES = [True] * 19 + [False] * 3


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--files", type=int, default=2)
    parser.add_argument("--skip", type=int, default=0)
    parser.add_argument("--jobs", type=int, default=1)
    parser.add_argument("--no-warm", action="store_true")
    parser.add_argument("--json", default=None)
    args = parser.parse_args()

    paths = common.files(args.files, args.skip)
    n_events = common.count_events(paths)
    nbytes = common.compressed_bytes(paths, common.WARM_BRANCHES)

    if not args.no_warm:
        warm = common.warm_cache(paths)
        print(f"cache warm-up: {warm:.1f} s over {len(paths)} files")

    os.chdir(PYFITTER)                      # pyfitter reads relative paths at import
    from process import AnaProcessor

    with tempfile.NamedTemporaryFile("w", suffix=".txt", delete=False) as handle:
        handle.write("\n".join(paths) + "\n")
        listfile = handle.name

    try:
        processor = AnaProcessor(file_list_path=listfile, jobs=args.jobs,
                                 cuts=dict(zip(CUT_NAMES, SWITCHES)), location="local")
        with common.Timer() as timer:
            results = processor.execute()
    finally:
        os.unlink(listfile)

    # the last entry of pyfitter's cut flow is the per-event count after every cut,
    # which is what pyevtana's events_at[-1] reports
    selected = 0
    flow = None
    for result in results or []:
        try:
            stats = result["cut_stats"]
            selected += stats[-1]["events_passing"]
            if flow is None:
                flow = [c["name"] for c in stats]
        except Exception:
            pass

    row = common.report(f"pyfitter/pyutils (jobs={args.jobs})", n_events, timer,
                        extra=f"branches    : {nbytes / 1e6:.1f} MB compressed, "
                              f"{nbytes / 1e6 / timer.wall:.1f} MB/s")
    print(f"  selected   : {selected} events after {len(flow or [])} cuts")

    row.update(framework="pyfitter", jobs=args.jobs, backend="process",
               nfiles=len(paths), mbytes=nbytes / 1e6, selected=selected)
    row["mode"] = "arrays"
    if args.json:
        with open(args.json, "a") as handle:
            handle.write(json.dumps(row) + "\n")


if __name__ == "__main__":
    main()
