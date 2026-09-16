"""The strongest correctness check: every value pyevtana serves equals a direct read.

Normalization renames fields, restructures split branches and rebuilds vectors, so this
walks every collection in the file and compares field by field against plain uproot.
"""

import unittest

import awkward as ak

from pyevtana import Dataset
from pyevtana.missing import BranchState
from pyevtana.vectors import COMPONENTS, COORD

from base import NtupleTestCase


def _raw_names(collection):
    """Map cleaned field path -> leaf name in the file.

    ROOT prefixes the leaves of some split branches with the branch name (``trk.pdg``,
    ``lumistream.nTrackerHits``) and not others (``evtinfo``, ``crvsummary``, written
    without a trailing dot); it also appends array dimensions (``PEsPerLayer[4]``) and
    emits path-form duplicates (``pos/pos.fCoordinates.fX``). The mapping is derived with
    the same cleaning the normalizer uses, rather than assumed.
    """
    from pyevtana.normalize import clean_field_name, is_path_form

    if not collection.split:
        return None
    return {clean_field_name(leaf, collection.name): leaf
            for leaf in collection.selected if not is_path_form(leaf)}


def _fields_of(array):
    try:
        return list(array.fields or [])
    except Exception:
        return []


def _navigate(array, path):
    """Follow a dotted path into an unsplit (nested-record) array."""
    for part in path.split("."):
        array = array[part]
    return array


def _resolve(raw, names, key):
    """Raw array for a cleaned field path, in a split branch.

    Usually the path is a leaf name. Sometimes ROOT gives a sub-branch instead, whose
    value is itself a record with dotted field names (``crvsummarymc.pos``), so fall back
    to the nearest ancestor that is a leaf and index into it.
    """
    if key in names:
        return raw[names[key]]
    parts = key.split(".")
    for i in range(len(parts) - 1, 0, -1):
        head = ".".join(parts[:i])
        if head in names:
            array = raw[names[head]]
            rest = ".".join(parts[i:])
            for candidate in (key, rest, f"{head}.{rest}"):
                if candidate in _fields_of(array):
                    return array[candidate]
    raise KeyError(f"no raw leaf for {key!r}; available: {sorted(names)[:8]}")


def assert_leaves_equal(case, got, expect, label):
    """Compare two leaf arrays, treating NaN as equal to NaN.

    Some float members are genuinely NaN in the file (184 of ``trksegpars_lh.raderr``
    here), and NaN never compares equal to itself.
    """
    import numpy as np

    flat_got = ak.to_numpy(ak.flatten(got, axis=None))
    flat_expect = ak.to_numpy(ak.flatten(expect, axis=None))
    case.assertEqual(flat_got.shape, flat_expect.shape, label)
    if flat_got.dtype.kind == "f":
        same = (flat_got == flat_expect) | (np.isnan(flat_got) & np.isnan(flat_expect))
        case.assertTrue(bool(np.all(same)), label)
    else:
        case.assertTrue(bool(np.all(flat_got == flat_expect)), label)


def compare(case, got, raw, names, path, counter):
    """Walk a normalized array against the raw one, whatever nesting the normalizer built.

    Handles the three shapes normalization produces: rebuilt vectors (compared against
    the original fX/fY/fZ leaves), re-nested sub-records such as ``prel`` (compared field
    by field -- numpy has no ``==`` for an ``MCRelationship``), and plain leaves.
    """
    fields = _fields_of(got)

    if set(fields) == {"x", "y", "z"}:
        for axis, component in zip("xyz", COMPONENTS):
            key = f"{path}.{COORD}.{component}"
            expect = (_resolve(raw, names, key) if names
                      else _navigate(raw, path)[COORD][component])
            assert_leaves_equal(case, got[axis], expect, f"{path}.{axis}")
            counter[0] += 1
        return

    if fields:
        for field in fields:
            compare(case, got[field], raw, names, f"{path}.{field}" if path else field,
                    counter)
        return

    expect = _resolve(raw, names, path) if names else _navigate(raw, path)
    assert_leaves_equal(case, got, expect, path)
    counter[0] += 1


class TestEveryCollectionMatchesUproot(NtupleTestCase):
    @classmethod
    def setUpClass(cls):
        super().setUpClass()
        cls.dataset = Dataset(cls.path)
        cls.batch = next(cls.dataset.batches())
        cls.n = len(cls.batch)

    def test_every_leaf_of_every_collection(self):
        checked_collections = 0
        counter = [0]

        for info in self.dataset.schema.collections.values():
            if info.state is not BranchState.LOADED:
                continue
            got = self.batch.get(info.name)
            if info.split:
                raw = self.tree[info.branch].arrays(info.selected, entry_stop=self.n)
            else:
                raw = self.tree[info.branch].array(entry_stop=self.n)

            checked_collections += 1
            names = _raw_names(info)
            for field in got.fields:
                compare(self, got[field], raw, names, field, counter)

        self.assertGreater(checked_collections, 10)
        self.assertGreater(counter[0], 100)

    def test_object_loop_agrees_with_arrays(self):
        """What the proxies serve is what the arrays hold, event by event."""
        raw_pdg = self.tree["trk"].arrays(["trk.pdg"], entry_stop=self.n)["trk.pdg"]
        raw_hits = self.tree["trkhits"].array(entry_stop=self.n)

        total_hits = 0
        for i, event in enumerate(self.batch.events()):
            tracks = event.Tracks("trk")
            self.assertEqual(len(tracks), len(raw_pdg[i]))
            for j, track in enumerate(tracks):
                self.assertEqual(track.pdg, raw_pdg[i][j])
                hits = track.hits()
                self.assertEqual(len(hits), len(raw_hits[i][j]))
                for k, hit in enumerate(hits):
                    self.assertEqual(hit.edep, raw_hits[i][j][k].edep)
                    total_hits += 1
        self.assertGreater(total_hits, 1000)

    def test_dataset_array_concatenates_every_event(self):
        whole = self.dataset.array("trk")
        self.assertEqual(len(whole), self.tree.num_entries)
        raw = self.tree["trk"].arrays(["trk.pdg"])["trk.pdg"]
        self.assertTrue(ak.all(ak.num(whole.pdg, axis=1) == ak.num(raw, axis=1)))

    def test_event_identity(self):
        raw = self.tree["evtinfo"].arrays(entry_stop=self.n)
        for i, event in enumerate(self.batch.events()):
            self.assertEqual(event.run, raw.run[i])
            self.assertEqual(event.event, raw.event[i])
            self.assertEqual(event.index, i)

    def test_trigger_values(self):
        name = self.dataset.schema.triggers[0]
        raw = self.tree[name].array(entry_stop=self.n)
        short = name[len("trig_"):]
        for i, event in enumerate(self.batch.events()):
            self.assertEqual(event.trigger(short), bool(raw[i]))
            self.assertEqual(event.triggers()[short], bool(raw[i]))

    def test_counter_leaf_matches_track_count(self):
        for event in self.batch.events():
            self.assertEqual(event.ntracks("trk"), len(event.Tracks("trk")))


if __name__ == "__main__":
    unittest.main()
