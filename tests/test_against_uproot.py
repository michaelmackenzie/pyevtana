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
    """Map normalized field name -> leaf name in the file.

    ROOT prefixes the leaves of some split branches with the branch name (``trk.pdg``,
    ``lumistream.nTrackerHits``) and not others (``evtinfo`` and ``crvsummary``, written
    without a trailing dot), so the mapping is derived from the real leaf names rather
    than assumed.
    """
    from pyevtana.normalize import strip_prefix

    if not collection.split:
        return None
    return {strip_prefix(leaf, collection.name): leaf for leaf in collection.selected}


def _fields_of(array):
    try:
        return list(array.fields or [])
    except Exception:
        return []


def assert_arrays_equal(case, got, expect, label):
    """Compare two arrays, walking into nested structs, treating NaN as equal to NaN.

    Two wrinkles this has to handle, both properties of the data rather than of pyevtana:
    some members are themselves structs (``SimInfo.prirel`` is an ``MCRelationship``) and
    numpy has no ``==`` for those, so records are compared field by field; and some float
    members are genuinely NaN in the file (184 of ``trksegpars_lh.raderr`` here), which
    never compares equal to itself.
    """
    import numpy as np

    fields = _fields_of(got)
    if fields:
        for field in fields:
            assert_arrays_equal(case, got[field], expect[field], f"{label}.{field}")
        return

    flat_got = ak.to_numpy(ak.flatten(got, axis=None))
    flat_expect = ak.to_numpy(ak.flatten(expect, axis=None))
    case.assertEqual(flat_got.shape, flat_expect.shape, label)
    if flat_got.dtype.kind == "f":
        same = (flat_got == flat_expect) | (np.isnan(flat_got) & np.isnan(flat_expect))
        case.assertTrue(bool(np.all(same)), label)
    else:
        case.assertTrue(bool(np.all(flat_got == flat_expect)), label)


class TestEveryCollectionMatchesUproot(NtupleTestCase):
    @classmethod
    def setUpClass(cls):
        super().setUpClass()
        cls.dataset = Dataset(cls.path)
        cls.batch = next(cls.dataset.batches())
        cls.n = len(cls.batch)

    def test_every_leaf_of_every_collection(self):
        checked_collections = 0
        checked_fields = 0

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
                value = got[field]
                if set(getattr(value, "fields", []) or []) == {"x", "y", "z"}:
                    # rebuilt vector: compare against the original components
                    if info.split:
                        for component, axis in zip(COMPONENTS, "xyz"):
                            expect = raw[names[f"{field}.{COORD}.{component}"]]
                            assert_arrays_equal(self, value[axis], expect,
                                                f"{info.name}.{field}.{axis}")
                            checked_fields += 1
                    else:
                        coords = raw[field][COORD]
                        for component, axis in zip(COMPONENTS, "xyz"):
                            assert_arrays_equal(self, value[axis], coords[component],
                                                f"{info.name}.{field}.{axis}")
                            checked_fields += 1
                    continue

                expect = raw[names[field] if names else field]
                assert_arrays_equal(self, value, expect, f"{info.name}.{field}")
                checked_fields += 1

        self.assertGreater(checked_collections, 10)
        self.assertGreater(checked_fields, 100)

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
