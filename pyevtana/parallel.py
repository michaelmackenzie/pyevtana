"""Run an analysis over a dataset in parallel.

The unit of work is a ``(file, entry_start, entry_stop)`` partition.  By default there is
one partition per file, which is the useful granularity for a many-file dataset; large
files are split by ``max_entries`` so that a dataset which is really one big file still
parallelizes.

**Backends are not interchangeable.**  The object loop is ordinary Python and therefore
GIL-bound, so it only scales with ``backend="process"``.  Array work (``.array()`` plus
awkward) spends its time in compiled code and in uproot decompression, both of which
release the GIL, so ``backend="thread"`` is better there -- no pickling, no process
start-up.  The default is ``"serial"`` so that nothing becomes concurrent by surprise.

**Workers must return plain data.**  ``Event``, ``Track`` and the other proxies hold a live
branch cache and refuse to be pickled; return histograms, arrays, counts or dicts instead.
"""

from __future__ import annotations

import os
import sys
import traceback
from concurrent.futures import ProcessPoolExecutor, ThreadPoolExecutor, as_completed
from dataclasses import dataclass
from typing import Callable, Iterator, Optional, Sequence

from .missing import PyEvtAnaError

BACKENDS = ("serial", "thread", "process")
ON_ERROR = ("raise", "skip", "collect")


@dataclass(frozen=True)
class Partition:
    """A contiguous entry range of one file."""

    path: str
    start: int
    stop: int
    index: int = 0

    @property
    def num_entries(self) -> int:
        return self.stop - self.start

    def __repr__(self) -> str:
        return f"<Partition {os.path.basename(self.path)}[{self.start}:{self.stop}]>"


@dataclass
class PartitionError:
    """A partition that failed, kept rather than lost when ``on_error != 'raise'``."""

    partition: Partition
    message: str

    def __repr__(self) -> str:
        return f"<PartitionError {self.partition}: {self.message.splitlines()[-1]}>"


class Chunk:
    """What a worker function receives: the same event API, over one partition.

    A chunk opens its **own** file handle rather than sharing the dataset's, so that
    concurrent partitions -- threads especially -- never touch each other's reader state.
    Re-running discovery per partition costs only a metadata read.
    """

    __slots__ = ("dataset", "partition", "_reader")

    def __init__(self, dataset, partition: Partition):
        self.dataset = dataset
        self.partition = partition
        self._reader = None

    @property
    def reader(self):
        if self._reader is None:
            from .reader import FileReader
            from .select import Selector

            reader = FileReader(self.partition.path, self.dataset.tree_path,
                                Selector(self.dataset.branches), self.dataset.on_missing,
                                self.dataset.io_kwargs)
            reader.open()
            if self.dataset.required:
                reader.check_required(self.dataset.required)
            self._reader = reader
        return self._reader

    def close(self) -> None:
        if self._reader is not None:
            self._reader.close()
            self._reader = None

    def __iter__(self) -> Iterator:
        for batch in self.batches():
            yield from batch.events()

    def batches(self) -> Iterator:
        return self.reader.batches(self.dataset.step_size,
                                   start=self.partition.start, stop=self.partition.stop)

    @property
    def path(self) -> str:
        return self.partition.path

    @property
    def num_entries(self) -> int:
        return self.partition.num_entries

    @property
    def schema(self):
        return self.reader.schema

    def __len__(self) -> int:
        return self.partition.num_entries

    def __repr__(self) -> str:
        return f"<Chunk {self.partition}>"


def plan_partitions(dataset, max_entries: Optional[int] = None,
                    on_error: str = "raise") -> tuple[list[Partition], list[PartitionError]]:
    """Partition the dataset, tolerating files that cannot even be opened.

    Working out where the partitions are means reading each file's entry count, so a
    truncated or unreadable file fails *here*, before any worker runs.  With
    ``on_error != "raise"`` that file becomes a recorded failure instead of killing the
    whole job -- which is the point of ``on_error="skip"``.
    """
    partitions: list[Partition] = []
    failures: list[PartitionError] = []
    for path in dataset.paths:
        try:
            total = dataset.reader(path).num_entries
        except Exception:
            if on_error == "raise":
                raise
            failures.append(PartitionError(Partition(path, 0, 0, len(partitions)),
                                           traceback.format_exc()))
            continue
        if max_entries is None or max_entries <= 0 or total <= max_entries:
            partitions.append(Partition(path, 0, total, len(partitions)))
        else:
            for start in range(0, total, max_entries):
                partitions.append(
                    Partition(path, start, min(start + max_entries, total), len(partitions))
                )
    return partitions, failures


def build_partitions(dataset, max_entries: Optional[int] = None) -> list[Partition]:
    """One partition per file, unless ``max_entries`` splits the big ones."""
    return plan_partitions(dataset, max_entries, on_error="raise")[0]


def _run_partition(task):
    """Executed in the worker. Module-level so it is picklable."""
    dataset, partition, function, on_error = task
    chunk = Chunk(dataset, partition)
    try:
        return partition.index, function(chunk)
    except Exception:
        if on_error == "raise":
            raise
        return partition.index, PartitionError(partition, traceback.format_exc())
    finally:
        # Close this chunk's own handle only -- the dataset may still be in use by
        # sibling threads sharing this process.
        try:
            chunk.close()
        except Exception:
            pass


def map_dataset(dataset, function: Callable, *, workers: Optional[int] = None,
                backend: str = "serial", reduce: Optional[Callable] = None,
                on_error: str = "raise", ordered: bool = False, progress: bool = False,
                max_entries: Optional[int] = None, mp_context: Optional[str] = None):
    """Call ``function(chunk)`` once per partition and collect the results.

    Returns the list of per-partition results, or ``reduce(results)`` when ``reduce`` is
    given.  Failures -- including files that could not be opened at all -- are collected
    into ``dataset.errors`` unless ``on_error='raise'``.
    """
    if backend not in BACKENDS:
        raise ValueError(f"backend must be one of {BACKENDS}, got {backend!r}")
    if on_error not in ON_ERROR:
        raise ValueError(f"on_error must be one of {ON_ERROR}, got {on_error!r}")

    partitions, failures = plan_partitions(dataset, max_entries, on_error)
    if backend == "process" and not _picklable(function):
        raise PyEvtAnaError(
            f"{getattr(function, '__name__', function)!r} cannot be pickled, so it cannot "
            "run in a worker process. Use a module-level function (not a lambda, a closure "
            "or a local def), or backend='thread'."
        )

    total = len(partitions)
    collected: list[tuple[int, object]] = []

    def note(done: int) -> None:
        if progress:
            print(f"\rpyevtana: {done}/{total} partitions", end="", file=sys.stderr, flush=True)

    if backend == "serial" or total == 1 or workers == 1:
        for done, partition in enumerate(partitions, start=1):
            collected.append(_run_partition((dataset, partition, function, on_error)))
            note(done)
    else:
        count = workers or min(total, os.cpu_count() or 1)
        if backend == "process":
            import multiprocessing

            # "fork" (the Linux default) keeps worker functions defined in __main__ or a
            # notebook usable; pass mp_context="spawn" if a library in the parent is
            # unhappy about being forked.
            pool = lambda n: ProcessPoolExecutor(                      # noqa: E731
                max_workers=n,
                mp_context=multiprocessing.get_context(mp_context) if mp_context else None,
            )
        else:
            pool = lambda n: ThreadPoolExecutor(max_workers=n)          # noqa: E731
        with pool(count) as executor:
            futures = [
                executor.submit(_run_partition, (dataset, partition, function, on_error))
                for partition in partitions
            ]
            for done, future in enumerate(as_completed(futures), start=1):
                collected.append(future.result())
                note(done)
    if progress:
        print(file=sys.stderr)

    if ordered or backend == "serial":
        collected.sort(key=lambda pair: pair[0])

    results = []
    for _, result in collected:
        if isinstance(result, PartitionError):
            failures.append(result)
        else:
            results.append(result)

    dataset.errors = [(f.partition.path, f.message) for f in failures]
    if failures and on_error == "collect":
        return (reduce(results) if reduce else results), failures
    return reduce(results) if reduce else results


def _picklable(function) -> bool:
    import pickle

    try:
        pickle.dumps(function)
        return True
    except Exception:
        return False
