#!/usr/bin/env python3
"""The basic object loop: tracks, their segments, and their hits.

    python3 examples/01_track_loop.py [file.root]
"""

from _common import parse

from pyevtana import Dataset

args = parse(__doc__, max_events={"type": int, "default": 5})

for event in Dataset(args.files):
    if event.index >= args.max_events:
        break
    print(f"run {event.run} subrun {event.subrun} event {event.event}: "
          f"{len(event.Tracks())} track fits")

    for track in event.Tracks("trk"):
        mid = track.seg("TT_Mid")           # surfaces by name, not by sid
        if mid is None:
            continue                        # this fit never reached the tracker middle
        print(f"  pdg={track.pdg:>4} nactive={track.nactive:>3} "
              f"chisq/ndof={track.chisq / max(track.ndof, 1):5.2f} "
              f"p={mid.mom.mag:7.3f} MeV/c  t={mid.time:8.1f} ns  "
              f"trkqual={track.qual().result:.3f}")

        hits = track.hits()
        active = [hit for hit in hits if hit.dactive]
        print(f"      {len(hits)} straw hits ({len(active)} active), "
              f"planes {sorted({int(h.plane) for h in hits})}")

        best = track.mcsim()[0]             # MC genealogy, best match first
        print(f"      MC: pdg={best.pdg} rank={best.rank} "
              f"p_origin={best.mom.mag:.3f} MeV/c")
