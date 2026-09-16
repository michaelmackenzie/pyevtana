#!/usr/bin/env python3
"""Turn the scaling study's JSONL into a table: throughput, speed-up, efficiency."""

import json
import sys
from collections import defaultdict

rows = [json.loads(l) for l in open(sys.argv[1]) if l.strip()]
by = defaultdict(dict)
for r in rows:
    key = r["framework"] + ("/" + r["mode"] if r.get("mode") else "")
    by[key][r["jobs"]] = r

order = ["pyfitter/arrays", "pyevtana/read", "pyevtana/objects"]
names = {"pyfitter/arrays": "pyfitter / pyutils", "pyevtana/read": "pyevtana read-only",
         "pyevtana/objects": "pyevtana objects"}

jobs = sorted({r["jobs"] for r in rows})
print(f"\n{'configuration':<20} " + "".join(f"{j:>12}" for j in jobs) + "   (workers)")
print("-" * (20 + 12 * len(jobs)))
for key in order:
    if key not in by:
        continue
    print(f"{names[key]:<20} " + "".join(
        f"{by[key][j]['rate']:>11.0f} " if j in by[key] else f"{'-':>12}" for j in jobs)
        + "   events/s")
print()
for key in order:
    if key not in by or 1 not in by[key]:
        continue
    base = by[key][1]["rate"]
    print(f"{names[key]:<20} " + "".join(
        f"{by[key][j]['rate'] / base:>10.1f}x " if j in by[key] else f"{'-':>12}" for j in jobs)
        + "   speed-up vs 1 worker")
print()
for key in order:
    if key not in by or 1 not in by[key]:
        continue
    base = by[key][1]["rate"]
    print(f"{names[key]:<20} " + "".join(
        f"{100 * by[key][j]['rate'] / (base * j):>10.0f}% " if j in by[key] else f"{'-':>12}"
        for j in jobs) + "   parallel efficiency")
print("\nper-point detail:")
for key in order:
    for j in jobs:
        r = by.get(key, {}).get(j)
        if r:
            print(f"  {names[key]:<20} jobs={j:<3} files={r['nfiles']:<4} "
                  f"events={r['events']:>7}  wall={r['wall']:7.1f}s  "
                  f"cpu/wall={r['cpu'] / r['wall']:5.2f}  {r['rate']:7.0f} ev/s")
