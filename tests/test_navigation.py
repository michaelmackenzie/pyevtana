"""Object navigation: positional companions, index links, and reverse links."""

import unittest

import awkward as ak

from pyevtana import Dataset, is_missing
from pyevtana.surfaces import surface_id

from base import NtupleTestCase


class TestTrackNavigation(NtupleTestCase):
    """`trkhits` is vector<vector<>> indexed by track, so hits() is a pure slice."""

    @classmethod
    def setUpClass(cls):
        super().setUpClass()
        cls.raw_trk = cls.tree["trk"].arrays(["trk.pdg", "trk.nactive"], entry_stop=20)
        cls.raw_hits = cls.tree["trkhits"].array(entry_stop=20)
        cls.raw_segs = cls.tree["trksegs"].array(entry_stop=20)
        cls.events = []
        for event in Dataset(cls.path):
            cls.events.append(event)
            if len(cls.events) == 20:
                break

    def test_track_count_and_fields_match_file(self):
        for i, event in enumerate(self.events):
            tracks = event.Tracks("trk")
            self.assertEqual(len(tracks), len(self.raw_trk["trk.pdg"][i]))
            for j, track in enumerate(tracks):
                self.assertEqual(track.pdg, self.raw_trk["trk.pdg"][i][j])
                self.assertEqual(track.nactive, self.raw_trk["trk.nactive"][i][j])

    def test_hits_are_aligned_with_their_track(self):
        for i, event in enumerate(self.events):
            for j, track in enumerate(event.Tracks("trk")):
                hits = track.hits()
                expect = self.raw_hits[i][j]
                self.assertEqual(len(hits), len(expect))
                for k, hit in enumerate(hits):
                    self.assertEqual(hit.plane, expect[k].plane)
                    self.assertEqual(hit.straw, expect[k].straw)

    def test_segs_are_aligned_with_their_track(self):
        for i, event in enumerate(self.events):
            for j, track in enumerate(event.Tracks("trk")):
                self.assertEqual(len(track.segs()), len(self.raw_segs[i][j]))

    def test_seg_lookup_by_surface_name(self):
        found = 0
        for event in self.events:
            for track in event.Tracks("trk"):
                seg = track.seg("TT_Mid")
                if seg is None:
                    continue
                found += 1
                self.assertEqual(seg.sid, surface_id("TT_Mid"))
                self.assertEqual(seg.surface, "TT_Mid")
                self.assertGreater(seg.mom.mag, 0)
        self.assertGreater(found, 0, "no track intersected TT_Mid in 20 events")

    def test_seg_returns_none_for_unintersected_surface(self):
        track = self.events[0].Tracks("trk")[0]
        self.assertIsNone(track.seg("CRV_M8"))

    def test_depth1_companions_are_single_records(self):
        track = self.events[0].Tracks("trk")[0]
        self.assertFalse(is_missing(track.mc()))
        self.assertIn("nhits", track.mc().fields)
        self.assertIn("result", track.qual().fields)
        # the two configured TrkQual leaves are distinguishable
        self.assertNotEqual(track.qual().result, track.qual("_bdt").result)

    def test_mcsim_is_ranked(self):
        track = self.events[0].Tracks("trk")[0]
        best = track.mcsim()[0]
        self.assertEqual(int(best.rank), 0)
        self.assertTrue(best.is_best_match)

    def test_segpars_picks_the_available_parametrization(self):
        track = self.events[0].Tracks("trk")[0]
        pars = track.segpars()
        self.assertFalse(is_missing(pars))
        self.assertEqual(len(pars), len(track.segs()))

    def test_unknown_field_names_the_alternatives(self):
        track = self.events[0].Tracks("trk")[0]
        with self.assertRaises(AttributeError) as caught:
            track.no_such_field
        self.assertIn("available fields", str(caught.exception))


class TestCaloNavigation(NtupleTestCase):
    """Calo objects are linked by stored indices, not by position."""

    @classmethod
    def setUpClass(cls):
        super().setUpClass()
        cls.raw = cls.tree.arrays(["caloclusters.hits_", "calohits.crystalId_"], entry_stop=100)
        cls.events = list(Dataset(cls.path))

    def test_cluster_hits_follow_stored_indices(self):
        checked = 0
        for i, event in enumerate(self.events):
            for j, cluster in enumerate(event.CaloClusters()):
                indices = self.raw["caloclusters.hits_"][i][j]
                hits = cluster.hits()
                self.assertEqual(len(hits), len(indices))
                for k, hit in enumerate(hits):
                    expect = self.raw["calohits.crystalId_"][i][int(indices[k])]
                    self.assertEqual(hit.crystalId_, expect)
                    checked += 1
        self.assertGreater(checked, 0, "no calo cluster hits were checked")

    def test_hit_cluster_backref_round_trips(self):
        for event in self.events:
            for cluster in event.CaloClusters():
                for hit in cluster.hits():
                    back = hit.cluster()
                    if back:
                        self.assertEqual(back.index, cluster.index)

    def test_unassociated_link_is_falsy_not_an_exception(self):
        """clusterIdx_ < 0 means 'no cluster'; that is data, not an error."""
        from pyevtana.missing import MissingRecord

        for event in self.events:
            for hit in event.CaloHits():
                if int(hit.clusterIdx_) < 0:
                    result = hit.cluster()
                    self.assertIsInstance(result, MissingRecord)
                    self.assertFalse(result)
                    self.assertIsNone(result.time_)
                    return
        self.skipTest("no unclustered calo hit in this file")


class TestSeedHits(NtupleTestCase):
    """Time clusters / line seeds store only nhits today; say so clearly."""

    def test_missing_hits_explains_what_the_maker_would_need(self):
        from pyevtana.objects.seed import TimeCluster

        dataset = Dataset(self.path, on_missing="empty")
        event = next(iter(dataset))
        # this file has no time clusters at all
        clusters = event.TimeClusters("timeclusters")
        self.assertTrue(is_missing(clusters))
        self.assertEqual(len(clusters), 0)


if __name__ == "__main__":
    unittest.main()
