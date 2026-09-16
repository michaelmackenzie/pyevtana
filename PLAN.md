# `pyevtana` — an object-oriented analysis layer for the EventNtuple

**Location:** `/exp/mu2e/app/users/mmackenz/main/pyevtana`
**Reference source:** `/exp/mu2e/app/users/mmackenz/main/EventNtuple` (`src/EventNtupleMaker_module.cc`, `inc/*.hh`)

---

## Status (2026-09-16)

**Built and tested: all phases. 122 tests pass** (`PYTHONPATH=.:tests python3 -m unittest
discover -s tests -t tests`), stdlib `unittest` because the `rootana 2.5.0` environment has
no pytest. See [README.md](README.md) for usage.

| phase | state |
|---|---|
| 0 skeleton | done — `pyproject.toml`, `setup.sh` |
| 1 schema + discovery + branch states | done — `schema.py`, `discovery.py`, `select.py`, `cli.py` |
| 2 normalization + vectors | done — `normalize.py`, `vectors.py` |
| 3 reader + Event + Track + missing-branch policy | done — `reader.py`, `event.py`, `record.py`, `collection.py`, `missing.py`, `objects/track.py` |
| 4 parallel `map` | done — `parallel.py`, `accumulate.py` |
| 5 calo + CRV navigation | done — `objects/calo.py`, `objects/crv.py` |
| 6 time clusters, line seeds, helices, MC steps | **code done, not validated on a real file** — see below |
| 7 metadata, subruns, docs | done — `metadata.py`, `README.md`, `examples/` |
| 8 worked analysis example | done — `examples/07_signal_selection.py`, regression-tested |

**Validated against real data.** Everything above is exercised against
`/exp/mu2e/app/users/mmackenz/main/nts.owner.description.version.sequencer.root`
(EventNtuple v6.13.1, single `trk` collection, LoopHelix fit, 100 events).
`tests/test_against_uproot.py` walks **every field of every collection** and compares it to
a direct uproot read, NaN-aware and recursing into nested structs.

**Not validated against real data: time clusters and line seeds.** No file on this node has
those branches, and regenerating one (`mu2e -c EventNtuple/fcl/from_mcs-Run1B.fcl -s <mcs
.art>`) was outside what was asked for here. The naming rules they depend on *are* tested,
against a stub tree (`tests/stubs.py`, `tests/test_schema_naming.py`): configurable
collection names found by class, four side-by-side track collections (`de`/`ue`/`dm`/`um`)
with no branch claimed twice, prefix-ambiguous tags (`de` vs `dem`), multiple TrkQual/TrkPID
leaves, `mcsteps_*` and `crvcoincsmcplane_*`, and deliberate schema mismatches. What remains
unproven is the end-to-end read of a real `EventNtupleTimeClusterInfo` / `LineSeedInfo`
branch — worth one run against a Run1B ntuple before relying on it.

**Maintenance burden vs. the C++ headers.** Struct *fields* are never enumerated — they
come from the file, so adding a member to any info struct needs no change here, including
new nested structs, fixed-size arrays and `XYZVectorF` members (pinned by
`tests/test_normalize.py::TestNewFieldsNeedNoCodeChange`, which normalizes invented members
appearing in no header). What does need a table row is a new *branch* or a new *struct
type*: one line in `TRACK_COMPANIONS`, `FIXED_COLLECTIONS` or `CONFIGURABLE_COLLECTIONS`.
README.md has the full breakdown.

**Four normalization bugs found and fixed while answering that question**, all in the
depth-1 split path, all from having special-cased `fCoordinates` instead of handling nested
names generically: fixed-size arrays kept ROOT's dimension in the name (`PEsPerLayer[4]`),
nested struct members stayed dotted (`prel._rel`, so unreachable as an attribute and
inconsistent with depth 2), ROOT's path-form duplicates leaked through as a field called
`pos/pos`, and `event.collection()` on those was unusable. `normalize.py` now re-nests any
dotted name generically.

**One real correctness bug found and fixed: `trkcalohitmc` is not track-aligned.** The maker
pushes an entry only for tracks with a calo cluster (140 of 375 in the sample file) and
stores no back-index, so indexing it by track number returned the wrong object.
`Track.calohitmc()` now recovers the mapping by counting tracks with `trkcalohit.did >= 0`
and verifies the count, and every depth-1 companion is length-checked at access time so a
future branch with this shape fails loudly. All 140 entries verified against the raw branch.

**Two things found while building, neither a pyevtana bug.** `trksegpars_lh.raderr` is
genuinely `NaN` for 184 of 6772 entries in the sample file. And ROOT prefixes the leaves of
some split branches with the branch name (`trk.pdg`, `lumistream.nTrackerHits`) but not
others (`evtinfo`, `crvsummary`, written without a trailing dot) — the normalizer derives
the mapping from the real leaf names rather than assuming either.

**Defaults chosen where §11 left a decision open** (all easy to change):
`on_missing="strict"`; capitalized `event.Tracks(...)` canonical with lowercase aliases;
tag auto-selected when exactly one collection of that kind exists, `AmbiguousCollection`
when several. Question 1 (extending the maker for time-cluster hits) and question 5
(upstreaming) are untouched and still yours.

---

## 1. Goal

Give the EventNtuple a Python API in which the *physics objects* are first-class, so that an
analysis reads the way the data model actually looks:

```python
from pyevtana import Dataset

for event in Dataset("nts....root"):
    for track in event.Tracks("trk"):
        if track.nactive < 20:
            continue
        mid = track.seg("TT_Mid")
        print(event.info.run, event.info.event, mid.mom.mag, track.mc().pdg)
        for hit in track.hits():
            print("  ", hit.plane, hit.panel, hit.straw, hit.edep, hit.rdrift)

    for cluster in event.CaloClusters():
        for hit in cluster.hits():          # follows CaloClusterInfo::hits_ into `calohits`
            print(hit.crystalId_, hit.time_)
```

The collection tag passed to `event.Tracks(tag)` is exactly the FHiCL `branchname` of the
`trk.fits` entry that produced it (`"trk"`, `"de"`, `"ue"`, `"dm"`, `"um"`, `"kl"`, ...), and the
same for the fhicl-configurable `timeclusters.names` / `lineseeds.names`.

This is deliberately **complementary to Mu2e `pyutils`**, which is array-oriented (awkward masks,
`plot_1D`, ...). `pyevtana` is the per-event/per-object view, and every collection exposes its
underlying awkward array so the two can be mixed (§7).

---

## 2. The naming scheme we are encoding

All of this is read straight out of `EventNtupleMaker_module.cc::beginJob()`. For a track-fit
config with `branchname` **`B`** (`EventNtupleMaker_module.cc:727-796`):

| role | branch | struct | depth | gated by |
|---|---|---|---|---|
| track info | `B` | `TrkInfo` | 1 | always |
| segments | `B`+`segs` | `TrkSegInfo` | 2 | always |
| segment pars | `B`+`segpars_lh` / `_ch` / `_kl` | `LoopHelixInfo` / `CentralHelixInfo` / `KinematicLineInfo` | 2 | `fittype` |
| dt/dt fit | `B`+`dtdt` | `TrkDtDtInfo` | 1 | `trk.fillTrkDtDt` |
| calo hit | `B`+`calohit` | `TrkCaloHitInfo` | 1 | always |
| track qual | `B`+`qual`+`<leafname>` | `MVAResultInfo` | 1 | `trkQualLeaves` |
| track PID | `B`+`pid`, `B`+`pid2`, ... | `MVAResultInfo` | 1 | `trkPIDTags` |
| straw hits | `B`+`hits` | `TrkStrawHitInfo` | 2 | `fillHits` |
| hit calibs | `B`+`hitcalibs` | `TrkStrawHitCalibInfo` | 2 | `fillHitCalibs` |
| materials | `B`+`mats` | `TrkStrawMatInfo` | 2 | `fillHits` |
| track MC | `B`+`mc` | `TrkInfoMC` | 1 | `fillMC` |
| MC genealogy | `B`+`mcsim` | `SimInfo` | 2 | `fillMC` |
| MC VD steps | `B`+`mcvd` | `MCStepInfo` | 2 | `fillMC` |
| calo hit MC | `B`+`calohitmc` | `CaloClusterInfoMC` | 1 | `fillMC` |
| straw hit MC | `B`+`hitsmc` | `TrkStrawHitInfoMC` | 2 | `fillMC` + `fillHits` |
| MC surface steps | `B`+`segsmc` | `SurfaceStepInfo` | 2 | `fillMC` |
| extra MC steps | `B`+`mcsic_<inst>` / `B`+`mcssi_<inst>` | `MCStepInfo` / `MCStepSummaryInfo` | 2 | extra step colls |
| count | `tcnt.n`+`B` | `int` | 0 | always |

"depth 1" = one entry per track in the event; "depth 2" = one *vector* per track, i.e. the outer
index is the track index. That is what makes `track.hits()` a pure slice — **no index-following
needed**: `trkhits[ievt][itrk]` is the hit list of `trk[ievt][itrk]`. Verified on a real file
(4 tracks -> 4 hit lists, 6 tracks -> 6 hit lists).

Other collections (`EventNtupleMaker_module.cc:798-890`):

- **Time clusters** — branch name from `timeclusters.names` + `"."`, struct `EventNtupleTimeClusterInfo`.
- **Line seeds** — branch name from `lineseeds.names` + `"."`, struct `LineSeedInfo`.
- **Calo** — `caloclusters`, `calohits`, `calorecodigis`, `calodigis`; MC `caloclustersmc`,
  `calohitsmc`, `calodigismc`, `calodigisim`, `calomcsim`.
- **CRV** — `crvsummary`, `crvcoincs`, `crvcoincsmc`, `crvcoincsmcplane[_<suffix>]`, `crvpulses`,
  `crvpulsesmc`, `crvdigis`, `crvcosmic`.
- **Helices** — `helices`.
- **MC steps** — `mcsteps_<instance>`.
- **Event level** — `evtinfo`, `evtinfomc`, `hitcount`, `tcnt.*`, `lumistream.`, `primary.`,
  `evtwt.*`, `trig_<path>`.
- **Subrun tree** — `subrunNtuple` with `srinfo`, `genEventCount`, `procEventCount`, `cosmicLivetime`.
- **Metadata** — `version` (TH1I, axis labels `major/minor/patch`), `trkqual_metadata` (TH1I whose
  axis labels carry `leaf: input tag = ...; model version = ...`), `n_proc_events` (TH1I).

### Object-to-object links that are *not* positional

| from | field | to |
|---|---|---|
| `CaloClusterInfo` | `hits_` (`vector<int>`) | indices into `calohits` |
| `CaloHitInfo` | `recoDigis_` (`vector<int>`) | indices into `calorecodigis` |
| `CaloHitInfo` | `clusterIdx_` | back-index into `caloclusters` |
| `CrvPulseInfoReco` | `crvHitIndex` | back-index into `crvcoincs` (`-1` = unclustered) |
| `SimInfo` | `index`, `rank`, `prirel`, `trkrel` | position within the same `mcsim` vector |
| `TrkSegInfo` / `SurfaceStepInfo` | `sid`, `sindex` | `SurfaceId` enum (`TT_Mid`, `ST_Foils`, ...) |

**Gap to flag up front:** `EventNtupleTimeClusterInfo` and `LineSeedInfo` currently store only
`nhits` / `nStrawHits` — there is **no per-hit branch and no hit-index list**, so a literal
`cluster.hits()` for time clusters / line seeds cannot be implemented from the ntuple as it stands
today. The plan handles this by (a) making `nhits` available as normal, (b) having `.hits()` raise a
`NoSuchBranch` error that names exactly what the maker would have to write, and (c) putting the
hit-linking behind a single mixin so that adding a `<name>hits` vector-of-vector branch (or a
`hits_` index vector, mirroring `CaloClusterInfo`) to the maker makes `.hits()` start working with
no changes to `pyevtana` beyond one schema table row. Whether to extend the maker is a separate
decision — see §9.

---

## 3. Two awkward-layout facts the package must absorb

Measured on `/exp/mu2e/app/users/mmackenz/main/nts.owner.description.version.sequencer.root`
(`splitlevel = 99`):

1. **Depth-1 vector branches are split**, so uproot returns a record whose *field names carry the
   branch prefix*: `trk` -> `{"trk.status": var*int32, "trk.pdg": ..., ...}`. Depth-2 branches are
   not split and return clean nested records: `trkhits` -> `var * var * struct[{plane, panel, ...}]`.
   Single-object branches written with a trailing dot (`lumistream.`) also keep the prefix
   (`lumistream.nCaloHitsD0`), while those without (`evtinfo`) come back clean (`event`, `run`, ...).
2. **`XYZVectorF` members are flattened differently by depth**: depth-1 gives three sibling leaves
   `pos.fCoordinates.fX/fY/fZ`; depth-2 gives a nested record `pos.fCoordinates.{fX,fY,fZ}`.

So a **normalization pass** is mandatory, and it is the one piece of real engineering here:

- strip any `"<branch>."` prefix from field names;
- collapse `<name>.fCoordinates.{fX,fY,fZ}` (sibling *or* nested) into a single field `<name>`
  built as a `vector` 3D object, so `seg.mom.mag`, `seg.mom.rho`, `seg.pos.z` all work;
- leave `std::array<...>` members (`CrvHitInfoReco.PEsPerLayer`, `TrkStrawHitInfo.tot`) as fixed-size
  awkward lists.

After normalization `track.status` and `hit.plane` work identically regardless of split depth,
which is the whole point.

---

## 4. Package layout

```
pyevtana/
├── PLAN.md                     # this file
├── README.md                   # quick reference + examples
├── pyproject.toml              # installable (`pip install -e .`), deps: uproot, awkward, vector, numpy
├── setup.sh                    # source-able: setupmu2e-art.sh + pyenv rootana 2.5.0 + PYTHONPATH
├── pyevtana/
│   ├── __init__.py             # Dataset, Event, describe, __version__
│   ├── schema.py               # the tables of §2, as data
│   ├── discovery.py            # TTree -> NtupleSchema (class-driven, see §5)
│   ├── select.py               # branch/leaf filter spec: globs, !excludes, callables (§6)
│   ├── reader.py               # Dataset / FileReader / BranchCache (lazy batched uproot reads)
│   ├── missing.py              # BranchState, MissingCollection/MissingRecord, on_missing (§7)
│   ├── parallel.py             # partitioning, map/reduce, process|thread|serial backends (§8)
│   ├── accumulate.py           # picklable result mergers: hist, Counter, awkward concat
│   ├── normalize.py            # the §3 normalization pass
│   ├── vectors.py              # XYZVectorF <-> vector.obj / VectorAwkward
│   ├── record.py               # RecordProxy base: __getattr__, __repr__, .raw, .fields
│   ├── collection.py           # ObjectCollection: sequence + .array() / .to_pandas() escape hatch
│   ├── event.py                # Event facade: Tracks(), TimeClusters(), CaloClusters(), ...
│   ├── objects/
│   │   ├── track.py            # Track, TrackSeg, StrawHit, TrackMC
│   │   ├── seed.py             # TimeCluster, LineSeed, Helix
│   │   ├── calo.py             # CaloCluster, CaloHit, CaloRecoDigi, CaloDigi (+ MC)
│   │   ├── crv.py              # CrvCoinc, CrvPulse, CrvDigi (+ MC)
│   │   └── mc.py               # SimParticle, MCStep, SurfaceStep
│   ├── surfaces.py             # SurfaceId enum <-> name, parsed from Offline/DataProducts/inc/SurfaceId.hh
│   ├── metadata.py             # ntuple version, trkqual metadata, subrun tree, POT bookkeeping
│   └── cli.py                  # `pyevtana-describe <file>` -> prints the discovered schema
├── tests/
│   ├── test_discovery.py
│   ├── test_normalize.py
│   ├── test_navigation.py
│   ├── test_missing_branches.py  # ABSENT vs NOT_SELECTED, all three on_missing modes
│   ├── test_parallel.py          # serial == process == thread; error isolation; partitioning
│   └── test_against_uproot.py    # every object-loop value == direct uproot read
└── examples/
    ├── 01_track_loop.py
    ├── 02_calo_cluster_hits.py
    ├── 03_timeclusters_lineseeds.py
    ├── 04_hybrid_vectorized.py
    ├── 05_slim_io.py             # branches= filter, and what it saves
    └── 06_parallel_dataset.py    # ds.map over a multi-file dataset
```

---

## 5. Discovery: by branch class, not by hardcoded name

Because track branch names, time-cluster names and line-seed names are all FHiCL-configurable,
discovery keys off the **C++ class in `branch.typename`**, then attaches companions by suffix.
(This mirrors the decision already taken for the rooutil C++ side, so the two stay consistent.)

```
for name, branch in tree.items(recursive=False):
    vector<mu2e::TrkInfo>                     -> track collection, tag = name
    vector<mu2e::EventNtupleTimeClusterInfo>  -> time-cluster collection, tag = name
    vector<mu2e::LineSeedInfo>                -> line-seed collection, tag = name
    vector<mu2e::HelixInfo>                   -> helix collection
    ... (calo / CRV / mcsteps_* by their own classes)
```

then for each track tag `B`, walk the §2 table, and for each expected companion check both that
`B+suffix` exists **and** that its `typename` is the expected struct — a mismatch is a schema
change and is reported, not silently ignored. The variable suffixes (`qual<leaf>`, `pid<N>`,
`mcsic_<inst>`, `mcssi_<inst>`, `crvcoincsmcplane_<suffix>`, `mcsteps_<inst>`) are matched by regex
against the remaining unclaimed branches.

The result is an `NtupleSchema`: which collections exist, which companions each has, the ntuple
version, and the `trkqual` leaf -> (input tag, model version) map. `pyevtana-describe file.root`
prints it, which doubles as the debugging tool when a file does not have what you expected.

Ambiguity note: deriving tags from `TrkInfo` branches rather than from string parsing is what keeps
`"de"`/`"dm"`/`"um"` safe — otherwise `"de"`+`"mc"` and a hypothetical branch `"dem"`+`"c"` are
indistinguishable.

---

## 6. Reading model — lazy, batched, and branch-selective

```python
Dataset(files,
        tree="EventNtuple/ntuple",
        branches=None,          # I/O reduction: see below
        required=None,          # fail fast at open time
        on_missing="strict",    # strict | empty | warn
        step_size="100 MB")
```

- Accepts a single path, a glob, a list, or a `.txt` filelist; local and `/pnfs`/xrootd paths alike.
- Iterates in entry batches; `Event` is a lightweight cursor `(cache, local_index)` and object
  proxies are `(cache, ievt, iobj)` with `__slots__`. Nothing is copied out of awkward until a leaf
  is actually read.
- **Branches are read on demand, per batch.** The first time a loop asks for `event.Tracks("trk")`,
  the `BranchCache` issues `tree.arrays(<that collection's branches>, entry_start, entry_stop)`,
  normalizes it (§3), and memoizes it for the rest of the batch. A loop that never calls
  `track.hits()` never reads `trkhits`.

### Why this matters — measured on the local 100-event file

| branch | compressed | share of file |
|---|---|---|
| `trkhits` | 2.360 MB | 42.4 % |
| `trkmats` | 1.418 MB | 25.5 % |
| `trksegpars_lh` | 0.390 MB | 7.0 % |
| `trkhitsmc` | 0.359 MB | 6.5 % |
| `trksegs` | 0.291 MB | 5.2 % |
| everything else | 0.746 MB | 13.4 % |
| **total** | **5.564 MB** | |

A momentum-spectrum analysis that touches only `trk`, `trksegs` and `evtinfo` reads ~6 % of the
file. Hit-level branches are two thirds of it. So on-demand reading is not a nicety — it is the
single largest performance lever in the package, ahead of parallelism.

### Leaf-level selection

Depth-1 branches are split, so their leaves are individually readable — verified:
`tree.arrays(["trk.pdg", "trk.nactive"])` works, and `trk.pdg` is 0.43 kB of the 31.6 kB `trk`
branch. `branches=` therefore accepts leaf granularity, not just branch granularity:

```python
Dataset(files, branches=["evtinfo", "trk.pdg", "trk.nactive", "trksegs"])   # explicit
Dataset(files, branches=["evtinfo", "trk*", "!trkhits*", "!trkmats"])        # globs, ! excludes
Dataset(files, branches=lambda name, typename: ...)                          # callable
Dataset(files, collections=["trk", "caloclusters"])                          # collection-level
```

Exclusions are applied after includes, so `["trk*", "!trkhits*"]` means what it looks like.
Depth-2 branches (`trkhits`, `trksegs`, ...) are unsplit, so they are all-or-nothing — the package
should say so in the `describe` output rather than silently ignoring a leaf-level request for one.

---

## 7. Missing and dropped branches — a first-class case

Branches go missing for two quite different reasons, and conflating them produces bad error
messages. `pyevtana` tracks three states for every leaf and collection:

| state | meaning | cause |
|---|---|---|
| `LOADED` | in the file and selected | normal |
| `NOT_SELECTED` | in the file, excluded by `branches=` | deliberate I/O reduction (§6) |
| `ABSENT` | not in the file at all | maker config (`fillHits: false`, `fillMC: false`, no CRV, no calo), a stripped/slimmed ntuple, or an older ntuple version |

### Behaviour

- **`ABSENT`** is governed by `on_missing`:
  - `"strict"` (default) — raise `MissingBranch`, naming the branch, the collection, the file, and
    the most likely maker switch (e.g. *"`trkhits` absent; `EventNtupleMaker` was run with
    `trk.fillHits: false` or `fits[i].options.fillHits: false`"*).
  - `"empty"` — return null objects: `MissingCollection` has `len() == 0`, iterates empty, is
    falsy, and `.array()` gives a correctly-typed empty awkward array; `MissingRecord` returns
    `None` for any field and is falsy. So `for hit in track.hits():` is a no-op instead of a
    traceback. This is the mode for running one script over mixed MC/data or full/slimmed datasets.
  - `"warn"` — behaves as `"empty"` but warns once per branch per process.
- **`NOT_SELECTED`** always raises `BranchNotSelected`, even under `on_missing="empty"`, and the
  message says *"present in the file but excluded by your `branches=` filter; add `trkhits` to
  read it."* Rationale: an explicit filter is a promise about I/O, and silently widening it — or
  silently returning empty — would defeat the exact thing you asked for. The one exception is when
  no filter was given at all, in which case everything in the file is implicitly selected and the
  cache just reads it on demand.

### Guard rails so failures are early and loud

- `required=["trk", "trksegs", "trkmc"]` is checked once at open time against every file, so a
  missing branch fails before a long loop rather than 40 minutes in.
- Introspection everywhere: `dataset.schema`, `dataset.available()`, `event.has("trkhits")`,
  `track.has("hits")`, `collection.state`, and `pyevtana-describe file.root` printing all three
  states side by side.
- `MissingCollection` / `MissingRecord` carry the *reason*, so `repr(track.hits())` prints
  `<MissingCollection trkhits: ABSENT (fillHits false?)>` rather than an empty list.

### Heterogeneous datasets and schema evolution

- **Discovery is per file, not per dataset** (§5). Files in one `Dataset` may legitimately differ
  (data vs MC, reprocessings, different `branchname` sets).
- `dataset.schema` reports the *intersection* plus an explicit diff of what varies across files;
  `on_missing` then governs per-file behaviour at access time.
- Normalization must not assume a struct has every field the current headers declare. Field access
  is driven by what the file actually contains, cross-checked against the `version` TH1I
  (`major/minor/patch`), so an older ntuple missing a recently-added member degrades to `ABSENT`
  for that member rather than crashing the whole collection.
- A branch whose `typename` does not match the expected struct is reported as a schema mismatch,
  never silently reinterpreted.

---

## 8. Parallel processing

The unit of work is a **`(file, entry_start, entry_stop)` partition**. By default there is one
partition per file — file-level parallelism, as you suggested — but large files are split by
`max_entries_per_partition` so that a dataset which is one big file still parallelizes, and a
dataset of a thousand small files does not create a thousand tasks.

```python
ds = Dataset(files, branches=[...])

results = ds.map(analyze,                 # called once per partition
                 workers=8,
                 backend="process",       # process | thread | serial
                 reduce=merge,            # optional; else a list of per-partition results
                 on_error="raise",        # raise | skip | collect
                 ordered=False,
                 progress=True)
```

`analyze(chunk)` receives a partition-scoped iterable of `Event`s — the *same* API as the serial
loop, so a function written against `for event in ds:` moves to `ds.map` unchanged:

```python
def analyze(chunk):
    h = Hist.new.Reg(100, 95, 115).Double()
    for event in chunk:
        for track in event.Tracks("trk"):
            if track.nactive < 20:
                continue
            h.fill(track.seg("TT_Mid").mom.mag)
    return h                      # picklable; merged with `+`

spectrum = ds.map(analyze, workers=8, reduce=sum)
```

### Design points

- **Backend choice is not cosmetic.** The object loop is Python-level and GIL-bound, so it needs
  `ProcessPoolExecutor` to scale. Pure-array work (`.array()` + awkward) spends most of its time in
  compiled code and in uproot decompression, both of which release the GIL, so `backend="thread"`
  is the better choice there — cheaper startup, no pickling. `backend="serial"` runs in-process and
  is the default, so nothing becomes parallel by surprise and debugging stays sane.
- **Return values must be picklable; proxies are not.** `Event`, `Track`, `StrawHit` etc. hold a
  live branch cache and are explicitly not serializable. The worker function must return plain
  results — histograms, numpy/awkward arrays, counters, dicts. This is a documented contract, and
  `map` raises a clear error rather than a `PicklingError` from deep inside `multiprocessing` if a
  proxy escapes.
- **Accumulator helpers** for the common merges: `hist`/numpy histogram addition, `Counter` merge,
  awkward concatenation, and a small `Accumulator` protocol (`+`) so custom results just work with
  `reduce=sum`.
- **Intra-file threading is free and orthogonal**: uproot's `decompression_executor` /
  `interpretation_executor` are exposed as `Dataset(..., io_threads=N)` and apply in the serial
  path too.
- **Error isolation.** `on_error="skip"` drops a bad file (unreadable, truncated, missing a
  `required` branch) and records it in `ds.errors` rather than losing the whole job; `"collect"`
  returns them alongside results. Default stays `"raise"`.
- **Determinism.** `ordered=True` returns per-partition results in dataset order, which matters when
  the result is a concatenated array rather than a commutative merge.
- **Composes with §6 and §7**: each worker applies the same branch filter, so parallelism multiplies
  the I/O saving rather than replacing it. Per-file discovery happens inside the worker, so a
  heterogeneous dataset does not force a serial pre-scan.
- **Not in scope**: a dask/distributed backend or grid submission. The `map`/`reduce` shape is
  chosen so a dask backend could be added later without changing user code, but building it is not
  part of this plan.

---

## 9. Two tiers: object loop *and* array escape hatch

Object proxies are Python-level, so a per-hit loop over millions of events will be far slower than
awkward. `pyevtana` therefore never hides the arrays:

```python
trks = event.Tracks("trk")
trks.array()                          # normalized awkward record for this event
Dataset(f).Tracks("trk").array()      # whole-dataset concatenated awkward array
Dataset(f).Tracks("trk").to_pandas()  # flat table
```

Recommended use, documented in the README: object loop for prototyping, event displays, printing,
and per-event logic that genuinely needs branching; `.array()` for the hot path; `ds.map` on top of
either. The package should state this honestly rather than pretend the object loop is free.

The package depends only on `uproot`, `awkward`, `vector` and `numpy` — **not** on `pyutils`
(your call). Anything produced here is a plain awkward array, so feeding it to `pyutils`' plotting
or selection helpers remains a one-liner for whoever wants that.

---

## 10. Implementation phases

Each phase ends in a working, tested state.

**Phase 0 — skeleton (small).** Directory, `pyproject.toml`, `setup.sh`, `__init__.py`, pytest
wired to the local 100-event file
`/exp/mu2e/app/users/mmackenz/main/nts.owner.description.version.sequencer.root`.

**Phase 1 — schema + discovery + branch states.** `schema.py` tables, `discovery.py`, the
`LOADED`/`NOT_SELECTED`/`ABSENT` model of §7, `pyevtana-describe`. Test: on the local file,
discovery finds exactly one track collection `trk` with companions `segs, segpars_lh, dtdt,
calohit, qual, qual_bdt, pid, hits, mats, mc, mcsim, mcvd, calohitmc, hitsmc, segsmc`, the calo and
CRV collections, and reports ntuple version + both trkqual models. Test that a deliberately
filtered-out branch reports `NOT_SELECTED` and a genuinely absent one reports `ABSENT`.

**Phase 2 — normalization + vectors.** `normalize.py`, `vectors.py`, tolerant of fields the file
does not have. Test: for both split depths, field names are prefix-free and `pos`/`mom` are usable
`vector` objects; values match a direct uproot read leaf-by-leaf.

**Phase 3 — reader + Event + Track + missing-branch policy.** `reader.py`, `event.py`, `record.py`,
`collection.py`, `missing.py`, `objects/track.py`. Delivers `event.Tracks(tag)`, `track.hits()`,
`.segs()`, `.seg("TT_Mid")`, `.mc()`, `.mcsim()`, `.qual()`, `.pid()`, plus
`evtinfo`/`evtinfomc`/`hitcount`/triggers, and all three `on_missing` modes. Tests: object-loop
values identical to direct uproot for every leaf over all 100 events; a `branches=`-restricted run
reads strictly fewer bytes (assert against `uproot`'s reported baskets) and raises
`BranchNotSelected` on the excluded ones; an `on_missing="empty"` run over a file with no MC
completes and yields empty collections.

**Phase 4 — parallel `map`.** `parallel.py`: partitioning, the three backends, `reduce`,
`on_error`, `ordered`, progress, accumulator helpers, and the not-picklable-proxy guard. Tests:
`serial` and `process` backends give identical results on the local file; a deliberately corrupt
path is skipped under `on_error="skip"`; partitioning a single file into N chunks reproduces the
whole-file answer.

**Phase 5 — calo + CRV navigation.** `objects/calo.py`, `objects/crv.py`, including the
index-following links of §2 (`cluster.hits()`, `hit.recodigis()`, `hit.cluster()`,
`coinc.pulses()` via a per-event reverse index on `crvpulses.crvHitIndex`).

**Phase 6 — time clusters, line seeds, helices, MC steps.** `objects/seed.py`, `objects/mc.py`,
`surfaces.py`. Validated against a Run1B ntuple regenerated per the known recipe
(`mu2e -c EventNtuple/fcl/from_mcs-Run1B.fcl -s <mcs .art>`), since the local file has no
time-cluster/line-seed branches.

**Phase 7 — metadata, subruns, docs.** `metadata.py` (version, POT / `procEventCount` /
`cosmicLivetime` bookkeeping for normalization — note this must be summed across *all* files
including ones skipped by `on_error`, or normalizations go silently wrong), README, examples, and a
short section in `EventNtuple/tutorial/` if you want it discoverable from the main repo.

---

## 11. Open decisions for you

1. **Time-cluster / line-seed hits.** Leave `.hits()` unimplemented-with-a-clear-error (plan as
   written), or extend `EventNtupleMaker` to write per-cluster hit information so it can work?
   The latter is a change to the maker and to the branch schema, and overlaps the Run1B rooutil
   work — out of scope unless you say otherwise.
2. **Method naming.** `event.Tracks("trk")` as you wrote it, capitalized, with lowercase aliases
   (`event.tracks(...)`) — or lowercase primary? Plan currently assumes your capitalized form is
   canonical.
3. **Default tag.** Should `event.Tracks()` with no argument auto-select when exactly one track
   collection exists (convenient, and the common case), or always require the tag (explicit)?
   Plan assumes auto-select-when-unambiguous, error when ambiguous.
4. **Default `on_missing`.** Plan defaults to `"strict"` so mistakes surface immediately. If most of
   your running is over mixed data/MC or slimmed ntuples, `"empty"` may be the better default.
5. **Upstreaming.** Private to `main/`, or eventually a PR to `Mu2e/EventNtuple` (alongside
   `helper/ntuplehelper.py`) / a standalone Mu2e repo? Affects how strictly the schema tables track
   the maker.

*Resolved:* standalone package, no `pyutils` dependency (§9).
