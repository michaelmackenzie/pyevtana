"""Ways to merge the per-partition results that :func:`pyevtana.parallel.map_dataset` returns.

Each takes the list of worker results and returns one object.  ``reduce=add`` covers most
cases (histograms, numpy arrays, anything with ``__add__``); note that the builtin ``sum``
does *not*, because it starts from ``0`` and ``0 + hist`` is an error.
"""

from __future__ import annotations

import functools
import operator
from collections import Counter
from typing import Iterable, Sequence


def add(results: Sequence):
    """Fold with ``+``, with no zero element -- works for hist, numpy, Counter."""
    results = [r for r in results if r is not None]
    if not results:
        return None
    return functools.reduce(operator.add, results)


def concat(results: Sequence):
    """Concatenate awkward arrays, preserving partition order."""
    import awkward as ak

    pieces = [r for r in results if r is not None and len(r)]
    if not pieces:
        return ak.Array([])
    return ak.concatenate(pieces)


def merge_counters(results: Sequence) -> Counter:
    total: Counter = Counter()
    for result in results:
        if result:
            total.update(result)
    return total


def merge_dicts(results: Sequence) -> dict:
    """Merge dicts of addable values (counts, histograms) key by key."""
    total: dict = {}
    for result in results:
        for key, value in (result or {}).items():
            total[key] = value if key not in total else total[key] + value
    return total


def chain(results: Iterable) -> list:
    """Flatten a list of lists."""
    return [item for result in results if result for item in result]
