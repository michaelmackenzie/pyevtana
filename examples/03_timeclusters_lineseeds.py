#!/usr/bin/env python3
"""Time clusters and line seeds, whose branch names come from fhicl.

These collections are found by branch *class*, so whatever `timeclusters.names` and
`lineseeds.names` were set to, pyevtana picks them up.  Pass the name explicitly when a
file has more than one.

Note: `EventNtupleTimeClusterInfo` and `LineSeedInfo` currently store only `nhits` /
`nStrawHits` -- there is no per-hit branch -- so `.hits()` reports what the maker would
have to write rather than inventing something.

    python3 examples/03_timeclusters_lineseeds.py [file.root]
"""

from _common import parse

from pyevtana import Dataset, is_missing

args = parse(__doc__, max_events={"type": int, "default": 10})

dataset = Dataset(args.files, on_missing="empty")
schema = dataset.schema

print("time-cluster collections:", schema.names_of_kind("timecluster") or "(none in this file)")
print("line-seed collections   :", schema.names_of_kind("lineseed") or "(none in this file)")
print()

for event in dataset:
    if event.index >= args.max_events:
        break
    for name in schema.names_of_kind("timecluster"):
        for cluster in event.TimeClusters(name):
            print(f"evt {event.event} {name}: nhits={cluster.nhits} "
                  f"nStrawHits={cluster.nStrawHits} t0={cluster.t0:8.1f} "
                  f"pos=({cluster.pos.x:.1f}, {cluster.pos.y:.1f}, {cluster.pos.z:.1f}) "
                  f"ecalo={cluster.ecalo:.2f}")
            hits = cluster.hits()
            if is_missing(hits):
                print(f"    hits unavailable -- {hits.reason}")

    for name in schema.names_of_kind("lineseed"):
        for seed in event.LineSeeds(name):
            print(f"evt {event.event} {name}: nhits={seed.nhits} t0={seed.t0:8.1f} "
                  f"d0={seed.d0:8.2f} phi0={seed.phi0:6.3f} cos={seed.cos:6.3f}")
