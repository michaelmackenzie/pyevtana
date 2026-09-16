"""ROOT ``XYZVectorF`` members -> usable ``vector`` objects.

ROOT flattens an ``XYZVectorF`` differently depending on how deep it sits:

* depth 1 (split): three sibling leaves ``pos.fCoordinates.fX/fY/fZ``
* depth 2 (unsplit): a nested record ``pos -> fCoordinates -> {fX, fY, fZ}``

Both are rebuilt here into a single field, so ``seg.mom.mag``, ``seg.mom.pt`` and
``seg.pos.z`` read the same whatever the depth.
"""

from __future__ import annotations

import math

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


class Vec3:
    """A 3-vector for the object path.

    The object API reads Python-native data (see :mod:`pyevtana.record`), where an
    ``XYZVectorF`` arrives as a plain ``{"x", "y", "z"}`` dict with no vector behaviour.
    Wrapping it in this class restores ``.mag``, ``.pt``, ``.rho``, ``.phi``, ``.theta``
    and ``.eta`` at a fraction of the cost of building a ``vector`` object per access;
    ``to_vector()`` hands back a real one for anything not covered here.

    Array-path code is unaffected: ``collection.array()`` still returns awkward arrays
    with the full ``vector`` behaviour registered.
    """

    __slots__ = ("x", "y", "z")

    def __init__(self, x, y, z):
        self.x = x
        self.y = y
        self.z = z

    @property
    def mag2(self):
        return self.x * self.x + self.y * self.y + self.z * self.z

    @property
    def mag(self):
        return math.sqrt(self.mag2)

    @property
    def rho2(self):
        return self.x * self.x + self.y * self.y

    @property
    def rho(self):
        return math.sqrt(self.rho2)

    #: momentum alias for :attr:`rho`
    @property
    def pt(self):
        return math.sqrt(self.rho2)

    @property
    def p(self):
        return self.mag

    @property
    def phi(self):
        return math.atan2(self.y, self.x)

    @property
    def theta(self):
        return math.atan2(self.rho, self.z)

    @property
    def eta(self):
        rho = self.rho
        if rho == 0.0:
            return math.inf if self.z > 0 else -math.inf
        return math.asinh(self.z / rho)

    def to_vector(self):
        """A full ``vector`` object, for operations this class does not implement."""
        return vector.obj(x=self.x, y=self.y, z=self.z)

    def to_dict(self) -> dict:
        return {"x": self.x, "y": self.y, "z": self.z}

    def __getitem__(self, key):
        return getattr(self, key)

    def __iter__(self):
        return iter((self.x, self.y, self.z))

    @property
    def fields(self) -> list:
        return ["x", "y", "z"]

    def __repr__(self) -> str:
        return f"Vec3(x={self.x:.4g}, y={self.y:.4g}, z={self.z:.4g})"


def is_vector_dict(value) -> bool:
    """True for the ``{"x", "y", "z"}`` dict a rebuilt vector becomes in Python."""
    return len(value) == 3 and "x" in value and "y" in value and "z" in value
