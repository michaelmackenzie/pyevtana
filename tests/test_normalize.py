"""Normalized values are identical to a direct uproot read, at every split depth."""

import unittest

import awkward as ak

from pyevtana import Dataset

from base import NtupleTestCase


class TestNormalization(NtupleTestCase):
    @classmethod
    def setUpClass(cls):
        super().setUpClass()
        cls.batch = next(Dataset(cls.path).batches())
        cls.n = len(cls.batch)

    def test_depth1_split_is_restructured(self):
        """`trk` arrives as a record of jagged arrays; it must become jagged records."""
        arr = self.batch.get("trk")
        self.assertEqual(arr.ndim, 2)                 # evt * var * record
        self.assertIn("pdg", arr.fields)              # prefix stripped
        self.assertFalse([f for f in arr.fields if f.startswith("trk.")])

        raw = self.tree["trk"].arrays(["trk.pdg", "trk.nactive"], entry_stop=self.n)
        self.assertTrue(ak.all(arr.pdg == raw["trk.pdg"]))
        self.assertTrue(ak.all(arr.nactive == raw["trk.nactive"]))

    def test_depth2_fields_are_clean(self):
        hits = self.batch.get("trkhits")
        self.assertEqual(hits.ndim, 3)                # evt * var(trk) * var(hit) * record
        raw = self.tree["trkhits"].array(entry_stop=self.n)
        self.assertTrue(ak.all(hits.plane == raw.plane))
        self.assertTrue(ak.all(hits.edep == raw.edep))

    def test_depth0_singleton(self):
        info = self.batch.get("evtinfo")
        raw = self.tree["evtinfo"].arrays(entry_stop=self.n)
        self.assertTrue(ak.all(info.run == raw.run))
        self.assertTrue(ak.all(info.event == raw.event))

    def test_prefixed_singleton_is_stripped(self):
        """`lumistream.` keeps its prefix in the file; users should never see it."""
        lumi = self.batch.get("lumistream")
        self.assertIn("nTrackerHits", lumi.fields)
        self.assertFalse([f for f in lumi.fields if f.startswith("lumistream.")])

    def test_nested_vector_rebuilt(self):
        """depth 2: XYZVectorF is a nested {fCoordinates:{fX,fY,fZ}} record."""
        vec = self.batch.get("trksegs").mom
        self.assertEqual(set(vec.fields), {"x", "y", "z"})
        raw = self.tree["trksegs"].array(entry_stop=self.n)["mom"]["fCoordinates"]
        self.assertTrue(ak.all(vec.x == raw["fX"]))
        self.assertTrue(ak.all(vec.z == raw["fZ"]))

    def test_flat_vector_rebuilt(self):
        """depth 1: the same member is three sibling leaves."""
        vec = self.batch.get("trkcalohit").mom
        self.assertEqual(set(vec.fields), {"x", "y", "z"})
        raw = self.tree["trkcalohit"].arrays(entry_stop=self.n)
        self.assertTrue(ak.all(vec.x == raw["trkcalohit.mom.fCoordinates.fX"]))

    def test_vector_behaviour_is_live(self):
        segs = self.batch.get("trksegs")
        mag = ak.flatten(segs.mom.mag, axis=None)
        self.assertGreater(len(mag), 0)
        self.assertTrue(ak.all(mag >= 0))
        # momentum fields get Momentum3D (.pt), positions get Vector3D (.rho)
        self.assertGreater(len(ak.flatten(segs.mom.pt, axis=None)), 0)
        self.assertGreater(len(ak.flatten(segs.pos.rho, axis=None)), 0)

    def test_magnitude_matches_hand_computation(self):
        segs = self.batch.get("trksegs")
        raw = self.tree["trksegs"].array(entry_stop=self.n)["mom"]["fCoordinates"]
        expect = (raw["fX"] ** 2 + raw["fY"] ** 2 + raw["fZ"] ** 2) ** 0.5
        self.assertTrue(ak.all(abs(segs.mom.mag - expect) < 1e-3))


if __name__ == "__main__":
    unittest.main()
