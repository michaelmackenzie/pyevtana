#!/usr/bin/env python3
"""Mixing the object loop with array work.

The object loop is for clarity and per-event logic; awkward is for the hot path.  Every
collection exposes `.array()`, and `Dataset.array(name)` concatenates a whole dataset, so
you can move between the two without changing how the data is named.

    python3 examples/04_hybrid_vectorized.py [file.root]
"""

import awkward as ak

from _common import parse

from pyevtana import Dataset
from pyevtana.surfaces import surface_id

args = parse(__doc__)
dataset = Dataset(args.files)

# --- vectorized: momentum at the tracker middle, for every track in the dataset --------
segs = dataset.array("trksegs")
at_mid = segs[segs.sid == surface_id("TT_Mid")]
momenta = ak.flatten(at_mid.mom.mag, axis=None)
print(f"vectorized: {len(momenta)} TT_Mid intersections, "
      f"mean p = {ak.mean(momenta):.3f} MeV/c, "
      f"{ak.sum((momenta > 100) & (momenta < 110))} in 100-110 MeV/c")

# --- object loop: the same number, written the way you would think about it ------------
count = total = 0
for event in dataset:
    for track in event.Tracks():
        mid = track.seg("TT_Mid")
        if mid is not None:
            count += 1
            total += float(mid.mom.mag)
print(f"object loop: {count} TT_Mid intersections, mean p = {total / count:.3f} MeV/c")

# --- per-event arrays, when you want one event at a time but not one object at a time ---
event = next(iter(dataset))
tracks = event.Tracks()
print(f"\nfirst event: {len(tracks)} tracks, pdgs {tracks.array().pdg.to_list()}")
print(f"as a table:\n{tracks.to_pandas()[['pdg', 'nactive', 'chisq']]}")
