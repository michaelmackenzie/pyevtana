"""ROOT ``XYZVectorF`` members -> usable ``vector`` objects.

ROOT flattens an ``XYZVectorF`` differently depending on how deep it sits:

* depth 1 (split): three sibling leaves ``pos.fCoordinates.fX/fY/fZ``
* depth 2 (unsplit): a nested record ``pos -> fCoordinates -> {fX, fY, fZ}``

Both are rebuilt here into a single field, so ``seg.mom.mag``, ``seg.mom.pt`` and
``seg.pos.z`` read the same whatever the depth.
"""

from __future__ import annotations

import awkward as ak
import vector

vector.register_awkward()

COORD = "fCoordinates"
COMPONENTS = ("fX", "fY", "fZ")

#: Fields whose name means "this is a momentum", so ``.pt`` / ``.p`` make sense.
MOMENTUM_FIELDS = frozenset({"mom", "endmom", "momentum", "startmom", "mommc"})


def vector_name(field: str) -> str:
    return "Momentum3D" if field.lower() in MOMENTUM_FIELDS else "Vector3D"


def build(field: str, x, y, z):
    """Zip three components into a ``vector``-aware awkward record."""
    return ak.zip({"x": x, "y": y, "z": z}, with_name=vector_name(field))


def is_coord_record(array) -> bool:
    """True for a nested ``{fCoordinates: {fX, fY, fZ}}`` record."""
    try:
        return array.fields == [COORD] and all(c in array[COORD].fields for c in COMPONENTS)
    except Exception:
        return False


def from_nested(field: str, array):
    coords = array[COORD]
    return build(field, coords[COMPONENTS[0]], coords[COMPONENTS[1]], coords[COMPONENTS[2]])
