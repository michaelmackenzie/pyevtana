# pyevtana

An object-oriented analysis layer for the Mu2e **EventNtuple**. Physics objects are
first-class, so an analysis reads the way the data model actually looks:

```python
from pyevtana import Dataset

for event in Dataset("nts....root"):
    for track in event.Tracks("trk"):
        mid = track.seg("TT_Mid")
        if mid is None or track.nactive < 20:
            continue
        print(event.run, event.event, mid.mom.mag, track.mcsim()[0].pdg)
        for hit in track.hits():
            print("  ", hit.plane, hit.panel, hit.straw, hit.edep)
```

It is standalone — it needs only `uproot`, `awkward`, `vector` and `numpy`. Everything it
produces is a plain awkward array, so passing results to `pyutils` (or anything else) is a
one-liner if you want that.

See [PLAN.md](PLAN.md) for the design and the reasoning behind it.

## Setup

```bash
source /exp/mu2e/app/users/mmackenz/main/pyevtana/setup.sh
```

which is `setupmu2e-art.sh` + `pyenv rootana 2.5.0` + this directory on `PYTHONPATH`.

## What's in a file

```bash
python3 -m pyevtana.cli nts....root          # or pyevtana-describe once installed
```

prints every collection with its struct, depth and state:

```
legend: [+] loaded   [-] present but excluded by branches=   [ ] absent

track collection 'trk'   (count leaf: tcnt.ntrk)
  [+] trk                      TrkInfo                  depth=1  37/37 leaves
  [+] trkhits                  TrkStrawHitInfo          depth=2  (unsplit)
  [ ] trkhitcalibs             TrkStrawHitCalibInfo     depth=2  (EventNtupleMaker was run with trk.fillHitCalibs: false)
```

This is the first thing to run when a branch is not where you expected it.

## Collections

The tag passed to `event.Tracks(tag)` is the FHiCL `branchname` of the track-fit config
that wrote the branch (`"trk"`, `"de"`, `"ue"`, `"dm"`, `"um"`, `"kl"`, ...). Collections
are discovered from branch **classes**, not names, so configurable names — tracks, time
clusters, line seeds — are picked up whatever they were called. Omit the tag when a file
has exactly one collection of that kind; with several, omitting it is an error rather than
a guess.

| accessor | branch |
|---|---|
| `event.Tracks(tag)` | the track-fit collections |
| `event.TimeClusters(name)`, `event.LineSeeds(name)`, `event.Helices()` | seeds |
| `event.CaloClusters()`, `.CaloHits()`, `.CaloRecoDigis()`, `.CaloDigis()` | calorimeter |
| `event.CrvCoincs()`, `.CrvPulses()`, `.CrvDigis()` | CRV |
| `event.MCSteps(instance)`, `event.Primaries()` | MC |
| `event.info`, `.mc`, `.hitcount`, `.lumistream`, `.crvsummary` | event-level singletons |
| `event.trigger(name)`, `event.triggers()`, `event.ntracks(tag)` | trigger and counters |
| `event.collection(name)` | anything not covered above |

`track.seg(surface)` returns the first crossing; `track.segs_at(surface)` returns them all,
which is what you want when a track crosses a surface more than once.

Lowercase aliases (`event.tracks(...)`) exist for all of them.

### Navigating

Track companions are *positionally* aligned — `trkhits` is a `vector<vector<>>` indexed by
track — so these are pure slices:

```python
track.hits()      track.mats()      track.hitcalibs()   track.segs()
track.seg("TT_Mid")                 track.segpars()     track.calohit()
track.mc()        track.mcsim()     track.mcvd()        track.segsmc()
track.hitsmc()    track.qual()      track.qual("_bdt")  track.pid()
```

Calo and CRV objects are linked by **stored indices**, so these follow a link and need the
target branch to have been read:

```python
cluster.hits()          # CaloClusterInfo::hits_ -> calohits
hit.recodigis()         # CaloHitInfo::recoDigis_ -> calorecodigis
hit.cluster()           # clusterIdx_ -> caloclusters (falsy if unclustered)
recodigi.hit()          # caloHitIdx_ -> calohits
coinc.pulses()          # reverse lookup on CrvPulseInfoReco::crvHitIndex
pulse.coinc()           # crvHitIndex -> crvcoincs (falsy if unclustered)
```

Surfaces are named, not numbered: `track.seg("TT_Mid")`, `seg.surface`, and
`pyevtana.surface_id("ST_Foils")`. Names come from `Offline/DataProducts/inc/SurfaceId.hh`.

**`EventNtupleTimeClusterInfo` and `LineSeedInfo` store only `nhits`/`nStrawHits` today** —
there is no per-hit branch — so `timecluster.hits()` reports what the maker would have to
write instead of inventing something. If a `<name>hits` branch is added later it starts
working with one row of `pyevtana/schema.py`.

## Reading less

Hit-level branches dominate an EventNtuple. On the sample file, `trkhits` + `trkmats` are
**68% of 5.56 MB**, and a momentum analysis touching only `trk`, `trksegs` and `evtinfo`
reads about 6% of it. Branches are read on demand anyway, so a loop that never calls
`track.hits()` never reads `trkhits`; `branches=` makes that explicit and turns an
accidental read into a loud error rather than a slow job.

```python
Dataset(files, branches=["evtinfo", "trk.pdg", "trk.nactive", "trksegs"])  # leaf-level
Dataset(files, branches=["trk*", "!trkhits*", "!trkmats"])                 # globs, ! excludes
Dataset(files, branches=lambda name, typename: ...)                        # callable
```

Includes are unioned first (everything, if no include patterns), then `!` patterns are
subtracted. Split branches can be cut to individual leaves; unsplit ones (`trkhits`,
`trksegs`) are all-or-nothing.

## Missing branches

Three states, because "you excluded it" and "the file never had it" deserve different
answers:

| state | meaning |
|---|---|
| `LOADED` | in the file and selected |
| `NOT_SELECTED` | in the file, excluded by `branches=` |
| `ABSENT` | not in the file (maker config, slimmed ntuple, older version) |

`ABSENT` follows `on_missing`:

- **`"strict"`** (default) — raise `MissingBranch`, naming the branch *and* the likely
  maker switch.
- **`"empty"`** — return null objects. `MissingCollection` has `len() == 0`, iterates
  empty and is falsy, so `for hit in track.hits():` is a no-op; `MissingRecord` returns
  `None` for any field and is falsy. This is the mode for one script over mixed MC/data or
  full/slimmed datasets.
- **`"warn"`** — as `"empty"`, plus one warning per branch.

`NOT_SELECTED` **always** raises `BranchNotSelected`, even under `"empty"`. Silently
widening your filter, or silently returning nothing, would defeat the I/O reduction you
asked for.

```python
Dataset(files, required=["trk", "trksegs"])   # checked at open time, per file, not 40 min in
event.has("trkhits")                          # present and selected?
track.has_companion("hits")
dataset.schema.state("trkhits")               # BranchState
dataset.schema_diff()                         # what varies across a heterogeneous dataset
```

Discovery runs **per file**, so files in one dataset may legitimately differ.

## Arrays

The object loop is Python-level, so a loop over millions of hits is far slower than
awkward. Nothing is hidden:

```python
event.Tracks().array()          # this event, normalized
event.Tracks().to_pandas()
dataset.array("trksegs")        # whole dataset, concatenated
```

Normalization is what makes these usable: field-name prefixes are stripped (`trk.pdg` →
`pdg`), split branches are restructured into jagged records, and `XYZVectorF` members are
rebuilt as `vector` objects so `seg.mom.mag`, `seg.mom.pt` and `seg.pos.z` work at any
depth.

Use the object loop for clarity and per-event logic, arrays for the hot path.

## Parallel processing

The unit of work is a `(file, entry_start, entry_stop)` partition — one per file by
default, with big files split by `max_entries` so a single-file dataset still parallelizes.

```python
from pyevtana.accumulate import add, merge_counters

def analyze(chunk):            # module-level: it must be picklable
    total = 0
    for event in chunk:        # same API as the serial loop
        total += len(event.Tracks())
    return total               # plain data only

result = dataset.map(analyze, workers=8, backend="process", reduce=sum,
                     on_error="skip", progress=True)
```

- **`backend`**: `"serial"` (default, nothing parallel by surprise), `"thread"`,
  `"process"`. The object loop is GIL-bound and only scales with `"process"`; array work
  is better with `"thread"` (no pickling, and uproot decompression releases the GIL).
- **Workers must return plain data.** `Event`/`Track` hold a live branch cache and refuse
  to pickle; `map` rejects an unpicklable worker up front rather than letting
  `multiprocessing` produce a baffling error.
- **`on_error="skip"`** survives a file that cannot even be opened, recording it in
  `dataset.errors`.
- `reduce=add` folds with `+` (histograms, numpy); `sum` does not, because it starts at 0.
- `io_threads=N` on `Dataset` enables uproot's decompression threads, in serial too.

## Normalization bookkeeping

```python
summary = dataset.summary()       # runs, subruns, procEventCount, genEventCount, livetime
if not summary.complete:          # some file could not be read
    ...
```

These totals must cover **every** file, including any `on_error="skip"` dropped — a
normalization from all files divided into a yield from a subset is silently wrong, so
`DatasetSummary` reports what it could not read instead of quietly omitting it.

## Examples

```
examples/01_track_loop.py            tracks, segments, hits
examples/02_calo_cluster_hits.py     index-following into calohits
examples/03_timeclusters_lineseeds.py  collections with fhicl-configurable names
examples/04_hybrid_vectorized.py     object loop and .array() side by side
examples/05_slim_io.py               branches= and what it saves
examples/06_parallel_dataset.py      Dataset.map over a multi-file dataset
examples/07_signal_selection.py      a full analysis selection with a per-event cut flow
```

`07_signal_selection.py` is the worked one: it histograms p, pT, cos(theta), t and cluster
energy for downstream electron tracks (pz at `TT_Mid` > 0, |fit PDG| == 11) in three sets
-- all, p > 90 MeV/c, and passing a 16-cut selection -- and prints the selection as a
**per-event** cut flow (events holding at least one surviving track at each stage). It also
shows a trap worth knowing: a track can cross `TT_Front` twice, and the first crossing is
the upstream-going one, so direction-sensitive quantities must use `track.segs_at(surface)`
rather than `track.seg(surface)`.

## Performance

Measured on one file of `CeMLeadingLogMix1BB` (8212 events, ~3.9 tracks/event, ~17
segments/track), warm page cache, running the full selection of
`examples/07_signal_selection.py`. Reproduce with `benchmarks/`.

| | events/s |
|---|---|
| object loop, as first written | 47.7 |
| object loop, now | **1510** |
| pyevtana array path | 2824 |
| pyfitter / pyutils, same files | 1841 |

The object loop is **31.7x faster** than the first implementation, with byte-identical
physics. Where the time goes now:

| component | share |
|---|---|
| uproot read + normalization | 45% |
| per-field `to_list` | 14% |
| Python object loop | 41% |

### What was slow, and what fixed it

**Awkward scalar access, ~58 us per field read.** `arr[i]` costs ~41 us and
`record[field]` a further ~17 us: awkward builds an error context, dispatches a backend
and re-wraps the layout on *every* scalar read. At a few hundred reads per track that was
essentially the whole runtime. Proxies now read Python lists converted once per batch
(`columns.py`), where the same read is ~0.13 us.

**Converting fields nothing reads.** `ak.to_list` on a record array builds a dict per
object -- 4.60 us per segment, against 0.05 us for one scalar field. Conversion is now per
field on first use: this selection converts **15 of 76** available fields.

**Vector fields.** `XYZVectorF` converted to `{"x","y","z"}` dicts cost about twice what
the three components cost separately, and `trksegs.mom` was the single largest conversion.
`VectorColumn` keeps the components and builds a `Vec3` only for objects actually read;
`Vec3` itself is 30x cheaper to construct than a `vector` object (0.51 vs 15.3 us) and
gives the same `.mag`, `.pt`, `.rho`, `.phi`, `.theta`, `.eta`.

**uproot's default source reopens the file.** Reading eight branches of one EventNtuple
triggered **364 opens** of the same file. `Dataset` now passes `MemmapSource` for ordinary
filesystem paths, which opens it once and is **1.9x faster on the read** -- worth knowing
for any uproot-based framework, not just this one. Remote URLs keep uproot's default.
Override with `Dataset(..., handler=...)`.

**What did not help:** `io_threads` (uproot's decompression executor) made this workload
*slower* at 2, 4 and 8 threads -- the pool overhead exceeds the gain when the branches are
small and the cache is warm. Normalization is not worth optimizing either: it is 4.3% of
read+normalize and 1.9% of the whole run, and it defers nothing -- a normalized array
converts to Python marginally *faster* than the raw one, because `ak.zip` builds a view
over the existing buffers rather than copying.

**Where the read time actually goes: unsplit branches.** ROOT cannot split a
`vector<vector<T>>`, so every depth-2 branch is all-or-nothing -- reading one field of
`trksegpars_lh` costs the whole 32 MB branch, and on this selection that single field
(`t0err`) is 0.56 s of the 1.56 s read. `trksegs` and `trksegpars_lh` together are 85% of
it. Nothing in pyevtana can avoid that, so `pyevtana-describe` now prints each branch's
compressed size and marks the unsplit ones, making the cost of touching one visible before
you write the loop.

### Writing a fast analysis on top of this

The two biggest wins in `examples/07_signal_selection.py` were in the analysis, not the
framework:

- **Fill histograms in bulk.** `Hist.fill(scalar)` costs ~33 us per value; filling the
  same values as one array is ~5000x cheaper per value for identical contents. Filling one
  value at a time was costing more than reading the data. Accumulate into a list, fill
  once (see `Collector`).
- **Scan segments once.** A track's segments are the most expensive thing to walk, so
  build the per-event picture in a single pass rather than re-deriving it, and cache
  derived quantities on the candidate (`cached_property`) -- `seg.mom` builds a fresh
  vector on every access.

Beyond that: pass `branches=` so nothing unused is read, use `track.seg()`/`segs_at()`
rather than iterating `segs()` yourself, and drop to `.array()` for anything that is a
whole-dataset reduction.

## Maintenance: what tracks the C++ headers, and what doesn't

**Struct fields are never enumerated in this package.** `RecordProxy.__getattr__` reads
straight from the awkward record, whose fields come from the file, so:

| change in EventNtuple | what pyevtana needs |
|---|---|
| add / rename / remove a member of `TrkInfo.hh` (or any info struct) | **nothing** |
| add a member of a *new* nested type, a fixed-size array, an `XYZVectorF` | **nothing** — handled generically |
| a file older than the headers, missing a recently-added member | **nothing** — that field is simply absent |
| add a new companion branch, e.g. `<tag>newthing` | one row in `TRACK_COMPANIONS` (`schema.py`) |
| add a new fixed-name collection, e.g. `calonewthings` | one row in `FIXED_COLLECTIONS` |
| add a collection with a fhicl-configurable name | one row in `CONFIGURABLE_COLLECTIONS`, keyed by struct |

`tests/test_normalize.py::TestNewFieldsNeedNoCodeChange` pins the first three rows by
normalizing invented members (`brandNewScalar`, `brandNewArray[3]`,
`brandNewNested._alpha`, `brandNewVec.fCoordinates.*`) that appear in no header.

Field names *do* appear in four places, none of which is a per-struct list:

1. **`_repr_fields`** on each proxy — display hints only. A stale one prints less in
   `repr()`; nothing else breaks.
2. **`vectors.MOMENTUM_FIELDS`** — which names get `Momentum3D` (with `.pt`, `.p`) rather
   than `Vector3D`. A new momentum member named something else still works as a vector,
   just without the momentum aliases.
3. **Link fields** in `objects/calo.py` and `objects/crv.py` (`hits_`, `recoDigis_`,
   `clusterIdx_`, `caloHitIdx_`, `crvHitIndex`, ...), tabulated in `schema.INDEX_LINKS`.
   These are genuine couplings: rename one in Offline and that navigation method breaks,
   loudly.
4. **`schema.NON_ALIGNED_COMPANIONS`** — see below.

### One branch that is not track-aligned

`trkcalohitmc` is the exception to "companion index == track index". `EventNtupleMaker`
pushes an entry only for tracks that have a calo cluster and stores no back-index, so it is
shorter than the track list (140 entries for 375 tracks in the sample file). `track.calohitmc()`
recovers the mapping by counting — the k-th entry belongs to the k-th track with
`trkcalohit.did >= 0`, which the maker fills under the same `hasCaloCluster()` guard — and
**checks the recovered count against the actual length**, so if the maker's condition ever
changes you get a `SchemaMismatch` rather than a silently wrong object. A track with no
cluster gets a falsy `MissingRecord`.

Every other depth-1 companion is verified to be track-aligned at access time by the same
guard, so a future branch with this shape fails loudly instead of returning the wrong row.

## Tests

```bash
source setup.sh
cd /exp/mu2e/app/users/mmackenz/main/pyevtana
PYTHONPATH=.:tests python3 -m unittest discover -s tests -t tests -v
```

122 tests, stdlib `unittest` (the `rootana` environment has no pytest; pytest collects them
too if you have it). `test_against_uproot.py` walks every field of every collection and
compares it against a direct uproot read.
