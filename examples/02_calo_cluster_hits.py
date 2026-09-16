#!/usr/bin/env python3
"""Calorimeter clusters and their crystal hits.

Calo objects are linked by stored indices rather than by position, so `cluster.hits()`
follows `CaloClusterInfo::hits_` into the `calohits` branch -- and `hit.cluster()` follows
`clusterIdx_` back again.

    python3 examples/02_calo_cluster_hits.py [file.root]
"""

from _common import parse

from pyevtana import Dataset

args = parse(__doc__, max_events={"type": int, "default": 20})

clusters_seen = hits_seen = 0

for event in Dataset(args.files, on_missing="empty"):
    if event.index >= args.max_events:
        break
    for cluster in event.CaloClusters():
        clusters_seen += 1
        hits = cluster.hits()
        hits_seen += len(hits)
        print(f"evt {event.event}: disk {cluster.diskID_} "
              f"E={cluster.energyDep_:7.3f} MeV t={cluster.time_:8.1f} ns "
              f"({len(hits)} crystals)")
        for hit in hits:
            back = hit.cluster()            # falsy MissingRecord if unclustered
            print(f"    crystal {hit.crystalId_:>4} E={hit.eDep_:7.3f} "
                  f"t={hit.time_:8.1f}  -> back-ref ok: {bool(back) and back.index == cluster.index}")

print(f"\n{clusters_seen} clusters, {hits_seen} crystal hits")
