"""Branch availability: the three states, the policy, and the null objects.

A branch can be unavailable for two quite different reasons, and conflating them makes
for useless error messages:

``LOADED``        in the file and selected.
``NOT_SELECTED``  in the file, but excluded by the ``branches=`` filter (deliberate I/O
                  reduction).  Asking for it is a bug in *your* script, so it always
                  raises -- silently widening the filter would defeat the point of it.
``ABSENT``        not in the file at all (maker config, slimmed ntuple, older version).
                  Governed by ``on_missing``.
"""

from __future__ import annotations

import enum
import warnings
from typing import Iterator, Optional


class BranchState(enum.Enum):
    LOADED = "LOADED"
    NOT_SELECTED = "NOT_SELECTED"
    ABSENT = "ABSENT"


class PyEvtAnaError(Exception):
    """Base class for every error this package raises deliberately."""


class MissingBranch(PyEvtAnaError):
    """A branch is not present in the file."""


class BranchNotSelected(PyEvtAnaError):
    """A branch is present in the file but was excluded by the branch filter."""


class SchemaMismatch(PyEvtAnaError):
    """A branch exists but does not hold the struct the schema expects."""


class AmbiguousCollection(PyEvtAnaError):
    """A tag was omitted but more than one collection could have been meant."""


ON_MISSING_MODES = ("strict", "empty", "warn")

_warned: set[str] = set()


def resolve_missing(name: str, state: BranchState, on_missing: str, *,
                    hint: str = "", where: str = "", factory=None):
    """Apply the ``on_missing`` policy to an unavailable branch.

    Returns a null object when the policy allows it; raises otherwise.  ``factory`` builds
    the null object (a :class:`MissingCollection` by default).
    """
    if state is BranchState.NOT_SELECTED:
        # Always an error, even under on_missing="empty".  An explicit branch filter is a
        # promise about I/O; quietly returning nothing would hide a real mistake.
        raise BranchNotSelected(
            f"branch {name!r} is present in {where or 'the file'} but was excluded by your "
            f"branches= filter. Add {name!r} (or its collection) to branches= to read it."
        )

    detail = f" ({hint})" if hint else ""
    message = f"branch {name!r} is not present in {where or 'the file'}{detail}"

    if on_missing == "strict":
        raise MissingBranch(
            message + ". Pass on_missing='empty' to treat absent branches as empty instead."
        )
    if on_missing == "warn" and name not in _warned:
        _warned.add(name)
        warnings.warn(message, RuntimeWarning, stacklevel=3)
    elif on_missing not in ON_MISSING_MODES:
        raise ValueError(f"on_missing must be one of {ON_MISSING_MODES}, got {on_missing!r}")

    return (factory or MissingCollection)(name, state, hint)


class _Missing:
    __slots__ = ("_name", "_state", "_hint")

    def __init__(self, name: str, state: BranchState = BranchState.ABSENT, hint: str = ""):
        self._name = name
        self._state = state
        self._hint = hint

    @property
    def name(self) -> str:
        return self._name

    @property
    def state(self) -> BranchState:
        return self._state

    @property
    def reason(self) -> str:
        return self._hint

    @property
    def missing(self) -> bool:
        return True

    def __bool__(self) -> bool:
        return False

    def _describe(self) -> str:
        detail = f": {self._hint}" if self._hint else ""
        return f"{self._name}: {self._state.value}{detail}"


class MissingCollection(_Missing):
    """Stands in for an absent collection: empty, falsy, and says why."""

    __slots__ = ()

    def __len__(self) -> int:
        return 0

    def __iter__(self) -> Iterator:
        return iter(())

    def __getitem__(self, index):
        raise IndexError(f"{self._name} is {self._state.value}; it has no elements")

    def array(self):
        import awkward as ak

        return ak.Array([])

    def to_list(self) -> list:
        return []

    def __repr__(self) -> str:
        return f"<MissingCollection {self._describe()}>"


class MissingRecord(_Missing):
    """Stands in for an absent single object: every field reads back as ``None``."""

    __slots__ = ()

    def __getattr__(self, item: str) -> None:
        if item.startswith("_"):
            raise AttributeError(item)
        return None

    def __getitem__(self, item) -> None:
        return None

    @property
    def fields(self) -> list:
        return []

    def __repr__(self) -> str:
        return f"<MissingRecord {self._describe()}>"


def missing_record(name: str, state: BranchState = BranchState.ABSENT, hint: str = "") -> MissingRecord:
    return MissingRecord(name, state, hint)


def is_missing(obj) -> bool:
    """True for the null objects above; safe to call on anything."""
    return isinstance(obj, _Missing)
