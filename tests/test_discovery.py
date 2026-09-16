"""Discovery finds the right collections, by class rather than by name."""

import unittest

from pyevtana import Dataset
from pyevtana.missing import BranchState
from pyevtana.schema import parse_typename

from base import NtupleTestCase


class TestTypenameParsing(unittest.TestCase):
    def test_depths(self):
        self.assertEqual(parse_typename("vector<mu2e::TrkInfo>"), (1, "TrkInfo"))
        self.assertEqual(parse_typename("std::vector<std::vector<mu2e::TrkSegInfo>>"),
                         (2, "TrkSegInfo"))
        self.assertEqual(parse_typename("mu2e::EventInfo"), (0, "EventInfo"))

    def test_non_structs_are_ignored(self):
        self.assertIsNone(parse_typename("int32_t"))
        self.assertIsNone(parse_typename("struct {int32_t nwts; float PBIWeight;}"))


class TestDiscovery(NtupleTestCase):
    @classmethod
    def setUpClass(cls):
        super().setUpClass()
        cls.schema = Dataset(cls.path).schema

    def test_track_tag_found_by_class(self):
        self.assertEqual(self.schema.track_tags(), ["trk"])

    def test_companions(self):
        companions = self.schema.tracks["trk"].companions
        for role in ["info", "segs", "segpars_lh", "dtdt", "calohit", "hits", "mats",
                     "mc", "mcsim", "mcvd", "calohitmc", "hitsmc", "segsmc", "pid"]:
            self.assertIn(role, companions, f"missing companion {role}")

    def test_both_trkqual_leaves_kept_apart(self):
        companions = self.schema.tracks["trk"].companions
        self.assertEqual(companions["qual"].name, "trkqual")
        self.assertEqual(companions["qual_bdt"].name, "trkqual_bdt")

    def test_companion_struct_and_depth(self):
        companions = self.schema.tracks["trk"].companions
        self.assertEqual(companions["hits"].struct, "TrkStrawHitInfo")
        self.assertEqual(companions["hits"].depth, 2)
        self.assertEqual(companions["mc"].struct, "TrkInfoMC")
        self.assertEqual(companions["mc"].depth, 1)

    def test_absent_companions_are_absent(self):
        # made with fillHitCalibs false, and a LoopHelix (not KinematicLine) fit
        self.assertIs(self.schema.state("trkhitcalibs"), BranchState.ABSENT)
        self.assertIs(self.schema.state("trksegpars_kl"), BranchState.ABSENT)
        self.assertIs(self.schema.state("trksegpars_lh"), BranchState.LOADED)

    def test_other_collections(self):
        for name in ["evtinfo", "evtinfomc", "hitcount", "caloclusters", "calohits",
                     "crvcoincs", "lumistream"]:
            self.assertTrue(self.schema.has(name), name)

    def test_counters_and_triggers(self):
        self.assertEqual(self.schema.counters["trk"], "tcnt.ntrk")
        self.assertGreater(len(self.schema.triggers), 10)

    def test_no_schema_mismatches(self):
        self.assertEqual(self.schema.mismatches, [])

    def test_metadata(self):
        self.assertGreaterEqual(self.schema.version[0], 6)
        leaves = {m.leaf: m for m in self.schema.trkqual_models}
        self.assertEqual(leaves["trkqual"].input_tag, "TrkQualAll:ANN")
        self.assertEqual(leaves["trkqual_bdt"].model_version, "BDT1_v2.0")


if __name__ == "__main__":
    unittest.main()
