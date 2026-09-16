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

## Tests

```bash
source setup.sh
cd /exp/mu2e/app/users/mmackenz/main/pyevtana
PYTHONPATH=.:tests python3 -m unittest discover -s tests -t tests -v
```

89 tests, stdlib `unittest` (the `rootana` environment has no pytest; pytest collects them
too if you have it). `test_against_uproot.py` walks every field of every collection and
compares it against a direct uproot read.
