"""Make every collection look the same, whatever ROOT did to it on the way out.

Struct *fields* are never enumerated here or anywhere else in the package -- they come
from the file, so adding a member to (say) ``TrkInfo.hh`` needs no change to pyevtana.
What this module does is repair the *names and shapes* ROOT gives those fields, which
differ by how deeply the member sits and by its C++ type:

1. **Prefixes.**  With ``splitlevel = 99`` a depth-1 branch is split, so uproot hands back
   field names carrying the branch name (``trk.pdg``); depth-2 branches are unsplit and
   come back clean (``plane``, ``panel``).  Branches written with a trailing dot
   (``lumistream.``) keep the prefix; ones written without it (``evtinfo``,
   ``crvsummary``) do not.  All prefixes are stripped.

2. **Shape.**  A split depth-1 branch arrives as a *record of jagged arrays*
   (``{"trk.pdg": var * int32, ...}``), not a jagged array of records.  It is restructured
   so every collection ends up as ``evt * [var *]^depth record`` and ``trk[ievt][itrk].pdg``
   works like ``trkhits[ievt][itrk][ihit].plane``.

3. **Nested members.**  A struct inside a struct is flattened by ROOT into dotted leaf
   names (``prel._rel``, ``cog_.fCoordinates.fX``), which are not reachable as Python
   attributes.  Any dotted name is re-nested into a sub-record, so ``obj.prel._rel`` works
   at depth 1 exactly as it already did at depth 2.  A group that turns out to be an
   ``XYZVectorF`` (a lone ``fCoordinates`` holding ``fX``/``fY``/``fZ``) becomes a
   ``vector`` object instead, giving ``.mag``, ``.pt``, ``.rho``.

4. **Fixed-size arrays.**  ROOT puts the dimension in the leaf name
   (``PEsPerLayer[4]``); the dimension is dropped so the field is ``PEsPerLayer``.

5. **Path-form duplicates.**  Some sub-branches appear twice, once as a name and once as a
   ``parent/child`` path (``pos`` and ``pos/pos.fCoordinates.fX``).  The path form is
   dropped and the structured version kept.

Because all of this is driven by what the file actually contains, an older ntuple missing a
recently-added struct member simply does not have that field, rather than breaking the
whole collection.
"""

from __future__ import annotations

import re

import awkward as ak

from . import vectors
from .vectors import COMPONENTS, COORD

#: ROOT appends the dimension of a fixed-size array to the leaf name: ``PEsPerLayer[4]``.
_ARRAY_DIM = re.compile(r"\[\d+\]$")


def strip_prefix(field: str, prefix: str) -> str:
    return field[len(prefix) + 1:] if prefix and field.startswith(prefix + ".") else field


def clean_field_name(field: str, prefix: str) -> str:
    """Leaf name in the file -> the name a user should see."""
    return _ARRAY_DIM.sub("", strip_prefix(field, prefix))


def is_path_form(field: str) -> bool:
    """``pos/pos.fCoordinates.fX`` duplicates a sub-branch already reachable by name."""
    return "/" in field


def fields_of(array) -> list:
    try:
        return list(array.fields or [])
    except Exception:
        return []


def _expand(name: str, array, out: list) -> None:
    """Flatten a (possibly nested) record into ``(dotted_name, leaf_array)`` pairs."""
    fields = fields_of(array)
    if not fields:
        out.append((name, array))
        return
    for field in fields:
        if is_path_form(field):
            continue
        sub = clean_field_name(field, name)
        _expand(f"{name}.{sub}" if sub else name, array[field], out)


def _is_vector_group(group: dict) -> bool:
    inner = group.get(COORD)
    return (len(group) == 1 and isinstance(inner, dict)
            and all(component in inner for component in COMPONENTS))


def _assemble(pairs: list, depth_limit: int) -> dict:
    """Turn dotted names back into nested records (and vectors where appropriate)."""
    tree: dict = {}
    for name, array in pairs:
        head, _, rest = name.partition(".")
        if rest:
            branch = tree.setdefault(head, {})
            if isinstance(branch, dict):
                branch[rest] = array
        else:
            tree.setdefault(head, array)

    contents: dict = {}
    for key, value in tree.items():
        if not isinstance(value, dict):
            contents[key] = value
            continue
        group = _assemble_group(value)
        if _is_vector_group(group):
            inner = group[COORD]
            contents[key] = vectors.build(key, *(inner[c] for c in COMPONENTS))
        else:
            contents[key] = ak.zip(_flatten_group(group), depth_limit=depth_limit)
    return contents


def _assemble_group(value: dict) -> dict:
    """One level of nesting, recursively, without zipping yet."""
    tree: dict = {}
    for name, array in value.items():
        head, _, rest = name.partition(".")
        if rest:
            branch = tree.setdefault(head, {})
            if isinstance(branch, dict):
                branch[rest] = array
        else:
            tree.setdefault(head, array)
    return {k: (_assemble_group(v) if isinstance(v, dict) else v) for k, v in tree.items()}


def _flatten_group(group: dict) -> dict:
    """Zip a nested group into records, innermost first."""
    out: dict = {}
    for key, value in group.items():
        if isinstance(value, dict):
            if _is_vector_group({COORD: value.get(COORD)} if COORD in value else value):
                inner = value[COORD] if COORD in value else value
                out[key] = vectors.build(key, *(inner[c] for c in COMPONENTS))
            elif all(component in value for component in COMPONENTS) and key == COORD:
                out[key] = ak.zip({c: value[c] for c in COMPONENTS})
            else:
                out[key] = ak.zip(_flatten_group(value))
        else:
            out[key] = value
    return out


def normalize_split(array, prefix: str, depth: int):
    """Depth-0/1 split branch: clean names, re-nest, rebuild vectors, zip into records."""
    pairs: list = []
    for field in array.fields:
        if is_path_form(field):
            continue
        _expand(clean_field_name(field, prefix), array[field], pairs)
    if not pairs:
        return array
    # depth_limit = depth + 1 keeps the per-event axis and zips the object axis into records.
    return ak.zip(_assemble(pairs, depth + 1), depth_limit=depth + 1)


def normalize_record(array):
    """Depth-2 (or any unsplit) branch: rebuild nested ``fCoordinates`` records."""
    fields = fields_of(array)
    if not fields:
        return array

    contents: dict = {}
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
    # ndim counts list axes only (records are not a dimension), which is the depth ak.zip
    # must rebuild at: evt * var * var * record -> depth_limit=3.
    return ak.zip(contents, depth_limit=array.ndim)


def normalize(array, *, prefix: str, depth: int, split: bool):
    """Entry point: return ``evt * [var *]^depth record`` with clean field names."""
    if split:
        return normalize_split(array, prefix, depth)
    return normalize_record(array)
