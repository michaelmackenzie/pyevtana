"""The ``branches=`` filter: which branches (or individual leaves) get read.

Reading less is the single largest performance lever in this package -- on a typical
EventNtuple the hit-level branches are roughly two thirds of the file, so an analysis that
never touches them should never pay for them.

Spec forms, all accepted by :class:`Selector`:

``None``                    everything in the file (read lazily, on demand).
``["evtinfo", "trk"]``      collection names -- takes all of that collection's leaves.
``["trk.pdg"]``             individual leaves of a split branch.
``["trk*", "!trkhits*"]``   fnmatch globs; a leading ``!`` excludes.
``callable(name, typename)``  arbitrary predicate.

Includes are unioned first (everything, if no include patterns were given), then every
``!`` pattern is subtracted.
"""

from __future__ import annotations

import fnmatch
from typing import Callable, Iterable, Optional, Sequence, Union

SelectSpec = Union[None, str, Sequence[str], Callable[[str, str], bool]]


class Selector:
    """Decides whether a leaf is selected, by leaf name or by owning collection name."""

    __slots__ = ("_includes", "_excludes", "_predicate", "_spec")

    def __init__(self, spec: SelectSpec = None):
        self._spec = spec
        self._includes: list[str] = []
        self._excludes: list[str] = []
        self._predicate: Optional[Callable[[str, str], bool]] = None

        if spec is None:
            return
        if callable(spec):
            self._predicate = spec
            return
        if isinstance(spec, str):
            spec = [spec]
        for pattern in spec:
            if not isinstance(pattern, str):
                raise TypeError(f"branch patterns must be strings, got {pattern!r}")
            if pattern.startswith("!"):
                self._excludes.append(pattern[1:])
            else:
                self._includes.append(pattern)

    @property
    def selects_everything(self) -> bool:
        return self._spec is None

    @property
    def picklable(self) -> bool:
        """A callable spec may not survive being sent to a worker process."""
        return self._predicate is None or getattr(self._predicate, "__qualname__", "<lambda>").find("<") < 0

    def matches(self, leaf: str, collection: Optional[str] = None, typename: str = "") -> bool:
        """Is this leaf selected?

        ``collection`` lets a pattern name the collection as a whole -- ``"trk"`` selects
        every leaf of the ``trk`` branch, not only a leaf literally called ``trk``.
        """
        if self._predicate is not None:
            return bool(self._predicate(leaf, typename))
        names = [leaf] if collection is None else [leaf, collection]

        if self._includes:
            included = any(
                fnmatch.fnmatchcase(name, pattern)
                for pattern in self._includes
                for name in names
            )
        else:
            included = True
        if not included:
            return False
        return not any(
            fnmatch.fnmatchcase(name, pattern)
            for pattern in self._excludes
            for name in names
        )

    def filter(self, leaves: Iterable[str], collection: Optional[str] = None) -> list[str]:
        return [leaf for leaf in leaves if self.matches(leaf, collection)]

    def __repr__(self) -> str:
        if self.selects_everything:
            return "<Selector: everything>"
        if self._predicate is not None:
            return "<Selector: callable>"
        return f"<Selector includes={self._includes} excludes={self._excludes}>"

    # Selectors travel to worker processes, so keep them plainly picklable.
    def __getstate__(self):
        return {"spec": self._spec}

    def __setstate__(self, state):
        self.__init__(state["spec"])
