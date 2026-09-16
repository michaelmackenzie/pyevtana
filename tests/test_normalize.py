"""Normalized values are identical to a direct uproot read, at every split depth."""

import unittest

import awkward as ak

from pyevtana import Dataset
from pyevtana.normalize import normalize

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


class TestFieldDiscoveryIsAutomatic(NtupleTestCase):
    """Fields come from the file, never from a list in pyevtana.

    These pin the four ways ROOT mangles a member name, all handled generically so that a
    new member of any of these types needs no change here.
    """

    @classmethod
    def setUpClass(cls):
        super().setUpClass()
        cls.batch = next(Dataset(cls.path).batches())

    def test_exposed_fields_are_exactly_the_files_leaves(self):
        """Nothing added, nothing dropped, for a plain struct."""
        got = set(self.batch.get("trk").fields)
        raw = {k[len("trk."):] for k in self.tree["trk"].keys()}
        self.assertEqual(got, raw)

    def test_fixed_size_array_loses_its_dimension(self):
        """ROOT names the leaf `PEsPerLayer[4]`, which is not a Python attribute."""
        coincs = self.batch.get("crvcoincs")
        self.assertIn("PEsPerLayer", coincs.fields)
        self.assertNotIn("PEsPerLayer[4]", coincs.fields)
        # and it is still a length-4 list per object
        self.assertEqual(str(coincs.PEsPerLayer.type).count("4 * float32"), 1)

    def test_nested_struct_member_is_a_record_at_depth_1(self):
        """`MCRelationship` is flattened by ROOT to `prel._rel`; re-nest it."""
        mc = self.batch.get("trkcalohitmc")
        self.assertIn("prel", mc.fields)
        self.assertEqual(set(mc["prel"].fields), {"_rel", "_rem"})
        self.assertNotIn("prel._rel", mc.fields)

    def test_nested_struct_is_consistent_between_depths(self):
        """The same member must read the same way whether split or not."""
        depth1 = self.batch.get("calomcsim")      # split
        depth2 = self.batch.get("trkmcsim")       # unsplit
        self.assertEqual(set(depth1["prirel"].fields), set(depth2["prirel"].fields))

    def test_path_form_duplicate_is_dropped(self):
        """`crvsummarymc` has both `pos` and `pos/pos.fCoordinates.fX`."""
        summary = self.batch.get("crvsummarymc")
        self.assertIn("pos", summary.fields)
        self.assertFalse([f for f in summary.fields if "/" in f])
        self.assertEqual(set(summary["pos"].fields), {"x", "y", "z"})

    def test_nested_vector_member_becomes_a_vector(self):
        """`cog_` sits one level down and still gets vector behaviour."""
        clusters = self.batch.get("caloclusters")
        self.assertEqual(set(clusters["cog_"].fields), {"x", "y", "z"})
        self.assertIsNotNone(clusters["cog_"].rho)

    def test_a_field_the_file_lacks_is_simply_absent(self):
        """CaloClusterInfo.hh declares e1_/e9_/secondMoment_; this ntuple predates them."""
        clusters = self.batch.get("caloclusters")
        for newer in ("e1_", "e9_", "e25_", "secondMoment_"):
            self.assertNotIn(newer, clusters.fields)
        # the rest of the collection is unaffected
        self.assertIn("energyDep_", clusters.fields)

    def test_unknown_field_raises_with_the_available_names(self):
        from pyevtana import Dataset as _Dataset

        event = next(iter(_Dataset(self.path)))
        cluster = event.CaloClusters()[0]
        with self.assertRaises(AttributeError) as caught:
            cluster.e9_
        self.assertIn("available fields", str(caught.exception))


class TestNewFieldsNeedNoCodeChange(unittest.TestCase):
    """Normalization is driven by the data, so invented field names work unchanged.

    This is the answer to "does a new member in TrkInfo.hh need maintaining here?" -- the
    names below appear in no header and in no pyevtana table, and they come through with
    the right names, nesting and vector behaviour.
    """

    def _split_branch(self):
        """A depth-1 split branch as uproot would deliver it, with made-up members."""
        return ak.Array({
            "trk.pdg": [[11, -11], [13]],
            "trk.brandNewScalar": [[1.5, 2.5], [3.5]],
            "trk.brandNewArray[3]": [[[1, 2, 3], [4, 5, 6]], [[7, 8, 9]]],
            "trk.brandNewNested._alpha": [[1, 2], [3]],
            "trk.brandNewNested._beta": [[4, 5], [6]],
            "trk.brandNewVec.fCoordinates.fX": [[3.0, 0.0], [1.0]],
            "trk.brandNewVec.fCoordinates.fY": [[4.0, 0.0], [0.0]],
            "trk.brandNewVec.fCoordinates.fZ": [[0.0, 5.0], [0.0]],
        })

    def test_new_scalar_field(self):
        out = normalize(self._split_branch(), prefix="trk", depth=1, split=True)
        self.assertIn("brandNewScalar", out.fields)
        self.assertEqual(out[0, 0].brandNewScalar, 1.5)

    def test_new_fixed_size_array_field(self):
        out = normalize(self._split_branch(), prefix="trk", depth=1, split=True)
        self.assertIn("brandNewArray", out.fields)
        self.assertNotIn("brandNewArray[3]", out.fields)
        self.assertEqual(out[0, 1].brandNewArray.to_list(), [4, 5, 6])

    def test_new_nested_struct_field(self):
        out = normalize(self._split_branch(), prefix="trk", depth=1, split=True)
        self.assertIn("brandNewNested", out.fields)
        self.assertEqual(set(out["brandNewNested"].fields), {"_alpha", "_beta"})
        self.assertEqual(out[0, 1].brandNewNested._beta, 5)

    def test_new_vector_field_gets_vector_behaviour(self):
        out = normalize(self._split_branch(), prefix="trk", depth=1, split=True)
        self.assertEqual(set(out["brandNewVec"].fields), {"x", "y", "z"})
        self.assertAlmostEqual(float(out[0, 0].brandNewVec.mag), 5.0, places=5)

    def test_a_removed_field_just_disappears(self):
        """Dropping a member from the file must not disturb the others."""
        data = self._split_branch()
        without = ak.Array({f: data[f] for f in data.fields if f != "trk.brandNewScalar"})
        out = normalize(without, prefix="trk", depth=1, split=True)
        self.assertNotIn("brandNewScalar", out.fields)
        self.assertIn("pdg", out.fields)

    def test_unsplit_branch_with_a_new_nested_vector(self):
        """The depth-2 path, where ROOT gives a nested record instead."""
        data = ak.Array([[{"newMom": {"fCoordinates": {"fX": 3.0, "fY": 4.0, "fZ": 0.0}},
                          "newScalar": 7}]])
        out = normalize(data, prefix="whatever", depth=2, split=False)
        self.assertEqual(set(out["newMom"].fields), {"x", "y", "z"})
        self.assertAlmostEqual(float(out[0, 0].newMom.mag), 5.0, places=5)
        self.assertEqual(out[0, 0].newScalar, 7)
