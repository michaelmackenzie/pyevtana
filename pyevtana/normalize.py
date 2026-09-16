"""Make every collection look the same, whatever ROOT did to it on the way out.

Three things are fixed here:

1. **Prefixes.**  With ``splitlevel = 99`` a depth-1 branch is split, so uproot hands back
   field names carrying the branch name (``trk.pdg``); depth-2 branches are unsplit and
   come back clean (``plane``, ``panel``).  Object branches written with a trailing dot
   (``lumistream.``) keep the prefix too.  All prefixes are stripped.

2. **Shape.**  A split depth-1 branch arrives as a *record of jagged arrays*
   (``{"trk.pdg": var * int32, ...}``), not a jagged array of records.  It is restructured
   with ``ak.zip`` so that every collection ends up as ``evt * [var *]^depth record`` and
   ``trk[ievt][itrk].pdg`` works like ``trkhits[ievt][itrk][ihit].plane``.

3. **Vectors.**  ``XYZVectorF`` members are rebuilt (see :mod:`pyevtana.vectors`).

Field access is driven by what the file actually contains, so an older ntuple missing a
recently-added struct member degrades to "no such field" for that member rather than
breaking the whole collection.
"""

from __future__ import annotations

import awkward as ak

from . import vectors
from .vectors import COMPONENTS, COORD


def strip_prefix(field: str, prefix: str) -> str:
    return field[len(prefix) + 1:] if prefix and field.startswith(prefix + ".") else field


def _group_flat_vectors(names: list[str]) -> tuple[dict[str, dict[str, str]], list[str]]:
    """Split ``["pos.fCoordinates.fX", "time"]`` into vector groups and plain fields."""
    groups: dict[str, dict[str, str]] = {}
    plain: list[str] = []
    for name in names:
        head, sep, tail = name.partition(f".{COORD}.")
        if sep and tail in COMPONENTS:
            groups.setdefault(head, {})[tail] = name
        else:
            plain.append(name)
    # A group is only a vector if all three components are there.
    incomplete = [g for g, parts in groups.items() if len(parts) != len(COMPONENTS)]
    for g in incomplete:
        plain.extend(groups.pop(g).values())
    return groups, plain


def normalize_split(array, prefix: str, depth: int):
    """Depth-1 split branch: strip prefixes, rebuild vectors, zip into records."""
    renamed = {strip_prefix(f, prefix): f for f in array.fields}
    groups, plain = _group_flat_vectors(list(renamed))

    contents: dict[str, object] = {}
    for name in plain:
        contents[name] = array[renamed[name]]
    for name, parts in groups.items():
        contents[name] = vectors.build(
            name, *(array[renamed[parts[c]]] for c in COMPONENTS)
        )

    if not contents:
        return array
    # depth_limit=depth+1 keeps the per-event axis and zips the object axis into records.
    return ak.zip(contents, depth_limit=depth + 1)


def normalize_record(array):
    """Depth-2 (or any unsplit) branch: rebuild nested ``fCoordinates`` records."""
    try:
        fields = array.fields
    except Exception:
        return array
    if not fields:
        return array

    contents: dict[str, object] = {}
    changed = False
    for name in fields:
        value = array[name]
        if vectors.is_coord_record(value):
            contents[name] = vectors.from_nested(name, value)
            changed = True
        else:
            contents[name] = value
    if not changed:
        return array
    # ndim counts list axes only (records are not a dimension), which is exactly the
    # depth ak.zip must rebuild at: evt * var * var * record -> depth_limit=3.
    return ak.zip(contents, depth_limit=array.ndim)


def normalize(array, *, prefix: str, depth: int, split: bool):
    """Entry point: return ``evt * [var *]^depth record`` with clean field names."""
    if split:
        return normalize_split(array, prefix, depth)
    return normalize_record(array)
