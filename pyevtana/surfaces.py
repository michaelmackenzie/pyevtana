"""SurfaceId names <-> values.

Generated from ``Offline/DataProducts/inc/SurfaceId.hh`` (mu2e::SurfaceIdDetail::enum_type).
Used by ``track.seg("TT_Mid")`` and friends so analyses name surfaces instead of
remembering integers.  Regenerate if the Offline enum changes; unknown names raise rather
than silently matching nothing.
"""

from __future__ import annotations

from .missing import PyEvtAnaError

#: Surface name -> sid value, exactly as Offline defines it.
SURFACE_IDS: dict[str, int] = {
    "unknown": -1,
    "TT_Front": 0,
    "TT_Mid": 1,
    "TT_Back": 2,
    "TT_Inner": 3,
    "TT_Outer": 4,
    "DS_Front": 80,
    "DS_Back": 81,
    "DS_Inner": 82,
    "DS_Outer": 83,
    "IPA_Legacy": 84,
    "DS_CryoInner": 85,
    "DS_CryoOuter": 86,
    "DS_ShieldInner": 87,
    "DS_ShieldOuter": 88,
    "DS_Coil": 89,
    "IPA": 90,
    "IPA_Front": 91,
    "IPA_Back": 92,
    "OPA": 95,
    "TSDA": 96,
    "ST_Front": 100,
    "ST_Back": 101,
    "ST_Inner": 102,
    "ST_Outer": 103,
    "ST_Foils": 104,
    "ST_Wires": 105,
    "TCRV": 200,
    "CRV_EX": 201,
    "CRV_T1": 202,
    "CRV_T2": 203,
    "CRV_T3": 204,
    "CRV_T4": 205,
    "CRV_T5": 206,
    "CRV_R1": 210,
    "CRV_R2": 211,
    "CRV_R3": 212,
    "CRV_R4": 213,
    "CRV_R5": 214,
    "CRV_R6": 215,
    "CRV_L1": 220,
    "CRV_L2": 221,
    "CRV_L3": 222,
    "CRV_E1": 230,
    "CRV_E2": 231,
    "CRV_U": 240,
    "CRV_D1": 250,
    "CRV_D2": 251,
    "CRV_D3": 252,
    "CRV_D4": 253,
    "CRV_C1": 260,
    "CRV_C2": 261,
    "CRV_C3": 262,
    "CRV_C4": 263,
    "CRV_M1": 270,
    "CRV_M2": 271,
    "CRV_M3": 272,
    "CRV_M4": 273,
    "CRV_M5": 274,
    "CRV_M6": 275,
    "CRV_M7": 276,
    "CRV_M8": 277,
    "CRV_StrongBack": 280,
    "DS_HatchConcrete": 300,
}

#: sid value -> name (first name wins for aliased values).
SURFACE_NAMES: dict[int, str] = {}
for _name, _value in SURFACE_IDS.items():
    SURFACE_NAMES.setdefault(_value, _name)


class UnknownSurface(PyEvtAnaError):
    """A surface name that Offline does not define."""


def surface_id(surface) -> int:
    """Accept a name or a raw sid and return the sid."""
    if isinstance(surface, int):
        return surface
    try:
        return SURFACE_IDS[surface]
    except KeyError:
        close = [n for n in SURFACE_IDS if n.lower().startswith(str(surface).lower()[:3])]
        hint = f" Did you mean one of: {', '.join(sorted(close)[:8])}?" if close else ""
        raise UnknownSurface(f"unknown surface {surface!r}.{hint}") from None


def surface_name(sid: int) -> str:
    return SURFACE_NAMES.get(int(sid), f"sid={int(sid)}")
