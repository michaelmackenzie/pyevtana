#!/usr/bin/env python3
"""Reading less: the `branches=` filter, and what it saves.

Hit-level branches dominate an EventNtuple, so an analysis that never looks at them should
never pay for them.  Branches are read on demand anyway; `branches=` makes the intent
explicit and turns an accidental hit-branch read into a loud error instead of a slow job.

    python3 examples/05_slim_io.py [file.root]
"""

from _common import parse

from pyevtana import Dataset
from pyevtana.missing import BranchNotSelected

args = parse(__doc__)
path = args.files[0]


def compressed_bytes(branch):
    return branch.compressed_bytes + sum(compressed_bytes(c) for c in branch.branches)


# --- what the file is actually made of --------------------------------------------------
import uproot

tree = uproot.open(path)["EventNtuple/ntuple"]
sizes = sorted(((compressed_bytes(b), n) for n, b in tree.items(recursive=False)), reverse=True)
total = sum(size for size, _ in sizes)
print(f"{path}\ntotal compressed: {total / 1e6:.3f} MB")
for size, name in sizes[:6]:
    print(f"  {name:<18} {size / 1e6:7.3f} MB  ({100 * size / total:4.1f}%)")

# --- a momentum analysis needs almost none of it -----------------------------------------
wanted = ["evtinfo", "trk.pdg", "trk.nactive", "trksegs"]
needed = 0
for name in ("evtinfo", "trksegs"):
    needed += compressed_bytes(tree[name])
for leaf in ("trk.pdg", "trk.nactive"):
    needed += compressed_bytes(tree["trk"][leaf])
print(f"\na momentum analysis needs {needed / 1e6:.3f} MB "
      f"({100 * needed / total:.1f}% of the file)")

# --- and asking for anything else is an error, not a silent slow read ---------------------
dataset = Dataset(path, branches=wanted)
event = next(iter(dataset))
print(f"\nread {len(event.Tracks())} tracks from the slimmed dataset; "
      f"fields available: {event.Tracks().fields}")
try:
    event.Tracks()[0].hits()
except BranchNotSelected as error:
    print(f"\nasking for hits says so plainly:\n  {error}")

# --- glob form, for excluding the expensive branches rather than listing the cheap ones ----
lean = Dataset(path, branches=["*", "!trkhits*", "!trkmats", "!*mc*"])
print(f"\nwith branches=['*', '!trkhits*', '!trkmats', '!*mc*']:")
for name in ("trk", "trksegs", "trkhits", "trkmats", "trkmcsim"):
    print(f"  {name:<12} {lean.schema.state(name).value}")
