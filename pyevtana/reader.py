"""Files, batches, and the lazy branch cache.

Nothing is read until an event asks for it.  The first time a loop reaches
``event.Tracks("trk")``, the cache reads *that collection's* branches for the current
entry window, normalizes them once, and memoizes them for the rest of the window; a loop
that never calls ``track.hits()`` never reads ``trkhits``.  On a typical EventNtuple the
hit-level branches are about two thirds of the file, so this is the biggest single lever
on how long an analysis takes.
"""

from __future__ import annotations

import glob as _glob
import os
from typing import Iterator, Optional, Sequence, Union

import awkward as ak
import uproot

from . import schema as S
from .discovery import NtupleSchema, discover, read_metadata
from .missing import (BranchState, MissingBranch, MissingCollection, ON_MISSING_MODES,
                      PyEvtAnaError, resolve_missing)
from .columns import Columns
from .normalize import normalize
from .select import SelectSpec, Selector

PathLike = Union[str, os.PathLike]


def default_handler(path: str):
    """uproot source handler for a path.

    uproot's default source reopens the file for every read request -- 364 opens while
    reading eight branches of one EventNtuple, which measured 1.9x slower than mapping the
    file once. For an ordinary filesystem path (including NFS and /pnfs) ``MemmapSource``
    opens it once; remote URLs keep uproot's default, which is what they need.
    """
    if "://" in str(path):
        return None
    return uproot.source.file.MemmapSource


# --------------------------------------------------------------------------------------
# file resolution
# --------------------------------------------------------------------------------------


def resolve_files(files: Union[PathLike, Sequence[PathLike]]) -> list[str]:
    """Expand globs, read ``.txt`` filelists, flatten sequences; keep order, drop repeats."""
    if isinstance(files, (str, os.PathLike)):
        files = [files]
    out: list[str] = []
    for entry in files:
        entry = os.fspath(entry)
        if entry.endswith(".txt") and os.path.exists(entry):
            with open(entry) as handle:
                for line in handle:
                    line = line.strip()
                    if line and not line.startswith("#"):
                        out.append(line)
        elif any(ch in entry for ch in "*?[") and "://" not in entry:
            matched = sorted(_glob.glob(entry))
            if not matched:
                raise FileNotFoundError(f"no files matched {entry!r}")
            out.extend(matched)
        else:
            out.append(entry)
    seen, unique = set(), []
    for path in out:
        if path not in seen:
            seen.add(path)
            unique.append(path)
    if not unique:
        raise ValueError("no input files")
    return unique


# --------------------------------------------------------------------------------------
# the per-batch cache
# --------------------------------------------------------------------------------------


class Batch:
    """One entry window of one file, plus the branches read so far."""

    __slots__ = ("reader", "start", "stop", "_cache", "_objects", "_leaves",
                 "_collections")

    def __init__(self, reader: "FileReader", start: int, stop: int):
        self.reader = reader
        self.start = start
        self.stop = stop
        self._cache: dict = {}
        self._objects: dict = {}
        self._leaves: dict = {}
        #: event-level collections, reused across the objects of one event
        self._collections: dict = {}

    def __len__(self) -> int:
        return self.stop - self.start

    # -- availability -------------------------------------------------------------------

    def has(self, name: str) -> bool:
        return self.reader.schema.state(name) is BranchState.LOADED

    # -- reading -------------------------------------------------------------------------

    def get(self, name: str, hint: str = ""):
        """The normalized array for the whole window: ``evt * [var *]^depth record``."""
        cached = self._cache.get(name)
        if cached is not None:
            return cached

        schema = self.reader.schema
        info = schema.get(name)
        if info is None or info.state is not BranchState.LOADED:
            state = BranchState.ABSENT if info is None else info.state
            result = resolve_missing(
                name, state, self.reader.on_missing,
                hint=hint or (S.absent_hint(info.role) if info is not None else ""),
                where=self.reader.source,
            )
            self._cache[name] = result
            return result

        branch = self.reader.tree[info.branch]
        if info.split:
            raw = branch.arrays(info.selected, entry_start=self.start, entry_stop=self.stop,
                                **self.reader.io_kwargs)
        else:
            raw = branch.array(entry_start=self.start, entry_stop=self.stop,
                               **self.reader.io_kwargs)
        array = normalize(raw, prefix=info.name, depth=info.depth, split=info.split)
        self._cache[name] = array
        return array

    def event_array(self, name: str, index: int, hint: str = ""):
        """``get(name)`` sliced to one event; missing branches pass straight through."""
        array = self.get(name, hint=hint)
        if isinstance(array, MissingCollection):
            return array
        return array[index]

    def columns(self, name: str, hint: str = ""):
        """The collection as lazily converted per-field Python lists.

        The object API reads through here rather than indexing awkward per object: a
        scalar read costs ~0.1 us against Python lists versus ~58 us against awkward, and
        fields are converted only when something reads them. :meth:`get` still serves the
        array path, so nothing about ``.array()`` changes.
        """
        cached = self._objects.get(name)
        if cached is not None:
            return cached

        array = self.get(name, hint=hint)
        if isinstance(array, MissingCollection):
            self._objects[name] = array
            return array
        columns = Columns(array)
        self._objects[name] = columns
        return columns

    def leaf(self, name: str, index: int):
        """A standalone scalar leaf (``trig_*``, ``tcnt.n*``), cached per window.

        Converted to a Python list on first use, for the same reason the collections are:
        indexing an awkward array one element at a time costs tens of microseconds, and
        these are read once per event -- a trigger decision per event was measured at
        ~40% of the per-event cost before this.
        """
        values = self._leaves.get(name)
        if values is None:
            try:
                values = self.reader.tree[name].array(
                    entry_start=self.start, entry_stop=self.stop, **self.reader.io_kwargs)
            except Exception:
                raise MissingBranch(
                    f"leaf {name!r} is not present in {self.reader.source}") from None
            values = ak.to_list(values)
            self._leaves[name] = values
        return values[index]

    # -- iteration --------------------------------------------------------------------------

    def events(self) -> Iterator:
        from .event import Event

        for i in range(self.stop - self.start):
            yield Event(self, i)


# --------------------------------------------------------------------------------------
# one file
# --------------------------------------------------------------------------------------


class FileReader:
    """An open file: its schema, and batches over it."""

    __slots__ = ("path", "tree_path", "selector", "on_missing", "io_kwargs", "handler",
                 "_file", "_tree", "_schema")

    def __init__(self, path: str, tree_path: str, selector: Selector, on_missing: str,
                 io_kwargs: Optional[dict] = None, handler="auto"):
        self.path = path
        self.tree_path = tree_path
        self.selector = selector
        self.on_missing = on_missing
        self.io_kwargs = io_kwargs or {}
        self.handler = handler
        self._file = None
        self._tree = None
        self._schema: Optional[NtupleSchema] = None

    @property
    def source(self) -> str:
        return os.path.basename(self.path)

    def open(self) -> "FileReader":
        if self._tree is None:
            handler = default_handler(self.path) if self.handler == "auto" else self.handler
            try:
                self._file = uproot.open(self.path, handler=handler) if handler \
                    else uproot.open(self.path)
            except Exception:
                # any handler-specific failure falls back to uproot's own choice
                self._file = uproot.open(self.path)
            try:
                self._tree = self._file[self.tree_path]
            except Exception:
                raise PyEvtAnaError(
                    f"{self.path} has no tree {self.tree_path!r}; "
                    f"top-level keys are {self._file.keys(recursive=False)}"
                ) from None
            self._schema = discover(self._tree, self.selector, source=self.source)
            parent = self._file[self.tree_path.rsplit("/", 1)[0]] if "/" in self.tree_path else self._file
            read_metadata(parent, self._schema)
        return self

    def close(self) -> None:
        if self._file is not None:
            try:
                self._file.close()
            except Exception:
                pass
        self._file = self._tree = None

    @property
    def tree(self):
        return self.open()._tree

    @property
    def schema(self) -> NtupleSchema:
        self.open()
        return self._schema

    @property
    def num_entries(self) -> int:
        return self.tree.num_entries

    def check_required(self, required: Sequence[str]) -> None:
        missing = [name for name in required if self.schema.state(name) is BranchState.ABSENT]
        if missing:
            raise MissingBranch(
                f"{self.path} is missing required branch(es): {', '.join(missing)}"
            )
        not_selected = [name for name in required
                        if self.schema.state(name) is BranchState.NOT_SELECTED]
        if not_selected:
            raise PyEvtAnaError(
                f"required branch(es) {', '.join(not_selected)} were excluded by the "
                f"branches= filter; required= and branches= disagree"
            )

    def selected_leaves(self) -> list:
        """Every leaf this reader may actually read, for sizing batches."""
        names: list = []
        for info in self.schema.collections.values():
            if info.state is not BranchState.LOADED:
                continue
            names.extend(info.selected if info.split else [info.branch])
        return names

    def entries_per_batch(self, step_size) -> int:
        """Entries per batch, sized from the branches actually being read.

        ``num_entries_for`` defaults to the whole tree, which badly undersizes batches
        whenever a filter is in play: on an EventNtuple where 71 MB of 449 MB is selected,
        a "100 MB" step came out as 1830 entries instead of 11653, so a file was read in
        five batches rather than one. Each extra batch means another read call per branch,
        and those small scattered reads cost far more under concurrency than one large
        sequential one.
        """
        if isinstance(step_size, int):
            return max(1, step_size)
        try:
            names = self.selected_leaves()
            if names:
                return max(1, int(self.tree.num_entries_for(step_size, expressions=names)))
            return max(1, int(self.tree.num_entries_for(step_size)))
        except Exception:
            try:
                return max(1, int(self.tree.num_entries_for(step_size)))
            except Exception:
                return 10000

    def batches(self, step_size, start: int = 0, stop: Optional[int] = None) -> Iterator[Batch]:
        stop = self.num_entries if stop is None else min(stop, self.num_entries)
        step = self.entries_per_batch(step_size)
        for begin in range(start, stop, step):
            yield Batch(self, begin, min(begin + step, stop))


# --------------------------------------------------------------------------------------
# the dataset
# --------------------------------------------------------------------------------------


class Dataset:
    """A set of EventNtuple files, iterated event by event.

    ``branches`` is the I/O-reduction lever (see :mod:`pyevtana.select`); ``on_missing``
    decides what happens when a branch is not in the file at all (see
    :mod:`pyevtana.missing`).  Branches excluded by ``branches`` always raise when asked
    for, whatever ``on_missing`` says -- an explicit filter is a promise about I/O.
    """

    def __init__(self, files, tree: str = S.DEFAULT_TREE, branches: SelectSpec = None,
                 required: Optional[Sequence[str]] = None, on_missing: str = "strict",
                 step_size: Union[str, int] = "100 MB", io_threads: int = 0,
                 handler="auto"):
        if on_missing not in ON_MISSING_MODES:
            raise ValueError(f"on_missing must be one of {ON_MISSING_MODES}, got {on_missing!r}")
        self.paths = resolve_files(files)
        self.tree_path = tree
        self.branches = branches
        self.required = list(required or ())
        self.on_missing = on_missing
        self.step_size = step_size
        self.io_threads = io_threads
        self.handler = handler
        self.errors: list[tuple[str, str]] = []
        self._selector = Selector(branches)
        self._readers: dict[str, FileReader] = {}
        self._io_executor = None

    # -- plumbing ---------------------------------------------------------------------------

    @property
    def io_kwargs(self) -> dict:
        """uproot's decompression executor, built once and never pickled.

        Decompression releases the GIL, so these threads help even inside a single
        worker; they are orthogonal to the ``backend`` choice in :meth:`map`.
        """
        if not self.io_threads or self.io_threads <= 1:
            return {}
        if self._io_executor is None:
            self._io_executor = uproot.ThreadPoolExecutor(self.io_threads)
        return {"decompression_executor": self._io_executor}

    def reader(self, path: str) -> FileReader:
        reader = self._readers.get(path)
        if reader is None:
            reader = FileReader(path, self.tree_path, self._selector, self.on_missing,
                                self.io_kwargs, self.handler)
            reader.open()
            if self.required:
                reader.check_required(self.required)
            self._readers[path] = reader
        return reader

    def close(self) -> None:
        for reader in self._readers.values():
            reader.close()
        self._readers.clear()

    def __enter__(self) -> "Dataset":
        return self

    def __exit__(self, *exc) -> None:
        self.close()

    # -- iteration ----------------------------------------------------------------------------

    def __iter__(self) -> Iterator:
        for path in self.paths:
            for batch in self.reader(path).batches(self.step_size):
                yield from batch.events()

    def batches(self) -> Iterator[Batch]:
        for path in self.paths:
            yield from self.reader(path).batches(self.step_size)

    @property
    def num_entries(self) -> int:
        return sum(self.reader(path).num_entries for path in self.paths)

    # -- schema ----------------------------------------------------------------------------------

    @property
    def schema(self) -> NtupleSchema:
        """Schema of the first file. Use :meth:`schema_diff` for heterogeneous datasets."""
        return self.reader(self.paths[0]).schema

    def schemas(self) -> dict:
        return {path: self.reader(path).schema for path in self.paths}

    def schema_diff(self) -> dict:
        """Collections that are not present in every file, ``name -> [files missing it]``."""
        per_file = {path: set(self.reader(path).schema.collections) for path in self.paths}
        everything: set[str] = set().union(*per_file.values()) if per_file else set()
        diff: dict[str, list[str]] = {}
        for name in sorted(everything):
            absent = [os.path.basename(p) for p, names in per_file.items() if name not in names]
            if absent:
                diff[name] = absent
        return diff

    def has(self, name: str) -> bool:
        return self.schema.has(name)

    def describe(self, stream=None) -> str:
        from .cli import format_schema

        text = format_schema(self.schema, self._selector)
        if stream is not None:
            print(text, file=stream)
        return text

    # -- whole-dataset arrays ----------------------------------------------------------------------

    def array(self, name: str):
        """Concatenate one collection's normalized array across every file."""
        import awkward as ak

        pieces = []
        for batch in self.batches():
            piece = batch.get(name)
            if isinstance(piece, MissingCollection):
                continue
            pieces.append(piece)
        if not pieces:
            return ak.Array([])
        return ak.concatenate(pieces)

    def to_pandas(self, name: str):
        import awkward as ak

        return ak.to_dataframe(self.array(name))

    # -- bookkeeping ------------------------------------------------------------------------------

    def summary(self, on_error: Optional[str] = None):
        """Run/subrun totals for normalization, summed over every file.

        Reports unreadable files rather than silently omitting them -- see
        :mod:`pyevtana.metadata`.
        """
        from .metadata import summarize

        parent = self.tree_path.rsplit("/", 1)[0] if "/" in self.tree_path else ""
        subrun_tree = f"{parent}/subrunNtuple" if parent else "subrunNtuple"
        return summarize(self.paths, subrun_tree,
                         on_error=on_error or ("skip" if self.on_missing != "strict" else "raise"))

    # -- parallel ------------------------------------------------------------------------------------

    def partitions(self, max_entries: Optional[int] = None):
        from .parallel import build_partitions

        return build_partitions(self, max_entries)

    def map(self, function, **kwargs):
        from .parallel import map_dataset

        return map_dataset(self, function, **kwargs)

    # -- pickling: config travels, open handles do not -------------------------------------------------

    def __getstate__(self) -> dict:
        # Open files and thread pools do not survive a process boundary; the config does.
        state = self.__dict__.copy()
        state["_readers"] = {}
        state["_selector"] = None
        state["_io_executor"] = None
        return state

    def __setstate__(self, state: dict) -> None:
        self.__dict__.update(state)
        self._readers = {}
        self._io_executor = None
        self._selector = Selector(self.branches)

    def __repr__(self) -> str:
        return (f"<Dataset {len(self.paths)} file(s) tree={self.tree_path!r} "
                f"on_missing={self.on_missing!r} branches={self._selector!r}>")
