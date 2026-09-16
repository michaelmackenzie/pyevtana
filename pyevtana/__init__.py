"""pyevtana -- an object-oriented analysis layer for the Mu2e EventNtuple.

    from pyevtana import Dataset

    for event in Dataset("nts....root"):
        for track in event.Tracks("trk"):
            mid = track.seg("TT_Mid")
            if mid is None or track.nactive < 20:
                continue
            print(event.run, event.event, mid.mom.mag, track.mcsim()[0].pdg)
            for hit in track.hits():
                print("  ", hit.plane, hit.panel, hit.straw, hit.edep)

The collection tag is the FHiCL ``branchname`` of the track-fit config that wrote the
branch, and collections are discovered from branch *classes*, so configurable names
(tracks, time clusters, line seeds) are picked up automatically.

Two things worth knowing before writing an analysis:

* Branches are read on demand.  Passing ``branches=[...]`` to :class:`Dataset` restricts
  I/O further, which is the largest single lever on runtime.
* Proxies are Python objects, so a loop over millions of hits is far slower than awkward.
  Every collection also exposes ``.array()``; use the object loop for clarity and
  per-event logic, arrays for the hot path, and :meth:`Dataset.map` to run either in
  parallel over files.
"""

from .collection import ObjectCollection
from .discovery import CollectionInfo, NtupleSchema, discover
from .event import Event
from .missing import (AmbiguousCollection, BranchNotSelected, BranchState, MissingBranch,
                      MissingCollection, MissingRecord, PyEvtAnaError, SchemaMismatch,
                      is_missing)
from .parallel import Chunk, Partition, PartitionError
from .reader import Batch, Dataset, FileReader, resolve_files
from .record import RecordProxy
from .select import Selector
from .surfaces import SURFACE_IDS, surface_id, surface_name

__version__ = "0.1.0"

__all__ = [
    "Dataset", "Event", "Batch", "FileReader", "Chunk", "Partition", "PartitionError",
    "ObjectCollection", "RecordProxy", "Selector",
    "NtupleSchema", "CollectionInfo", "discover", "describe",
    "BranchState", "PyEvtAnaError", "MissingBranch", "BranchNotSelected",
    "SchemaMismatch", "AmbiguousCollection", "MissingCollection", "MissingRecord",
    "is_missing", "resolve_files",
    "SURFACE_IDS", "surface_id", "surface_name",
    "__version__",
]


def describe(path: str, tree: str = None, branches=None) -> str:
    """Return (and print) the schema pyevtana discovers in ``path``."""
    from . import schema as _schema
    from .cli import format_schema

    dataset = Dataset(path, tree=tree or _schema.DEFAULT_TREE, branches=branches,
                      on_missing="empty")
    text = format_schema(dataset.schema, dataset._selector)
    print(text)
    return text
