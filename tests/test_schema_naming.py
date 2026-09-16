"""The naming scheme, tested against configurations no local file exercises.

The sample ntuple has a single `trk` collection and no time clusters or line seeds, so
these use a stub tree (see stubs.py) to check the rules that matter for Run1B-style files.
"""

import unittest

from pyevtana.discovery import discover
from pyevtana.missing import BranchState
from pyevtana.select import Selector

from stubs import StubTree, obj, track_branches, vec, vecvec


def build(branches):
    return discover(StubTree(branches), Selector(None), source="stub")


class TestMultipleTrackCollections(unittest.TestCase):
    """prolog.fcl's separate-branch mode writes de/ue/dm/um side by side."""

    def setUp(self):
        branches = {"evtinfo": (obj("EventInfo"), ["event", "run"])}
        for tag in ("de", "ue", "dm", "um"):
            branches.update(track_branches(tag))
        self.schema = build(branches)

    def test_all_tags_found(self):
        self.assertEqual(sorted(self.schema.track_tags()), ["de", "dm", "ue", "um"])

    def test_companions_attach_to_the_right_tag(self):
        for tag in ("de", "ue", "dm", "um"):
            companions = self.schema.tracks[tag].companions
            self.assertEqual(companions["info"].name, tag)
            self.assertEqual(companions["hits"].name, f"{tag}hits")
            self.assertEqual(companions["mc"].name, f"{tag}mc")

    def test_no_branch_is_claimed_twice(self):
        owners = {}
        for tag, track in self.schema.tracks.items():
            for info in track.companions.values():
                self.assertNotIn(info.name, owners,
                                 f"{info.name} claimed by both {owners.get(info.name)} and {tag}")
                owners[info.name] = tag

    def test_counters_per_tag(self):
        self.assertEqual(self.schema.counters,
                         {"de": "tcnt.nde", "dm": "tcnt.ndm",
                          "ue": "tcnt.nue", "um": "tcnt.num"})


class TestPrefixAmbiguity(unittest.TestCase):
    """Tags that are prefixes of each other must not steal each other's companions."""

    def test_longer_tag_wins_its_own_branches(self):
        branches = {}
        branches.update(track_branches("de"))
        branches.update(track_branches("dem"))
        schema = build(branches)
        self.assertEqual(sorted(schema.track_tags()), ["de", "dem"])
        self.assertEqual(schema.tracks["dem"].companions["info"].name, "dem")
        self.assertEqual(schema.tracks["dem"].companions["hits"].name, "demhits")
        # "dem" is a TrkInfo branch, so it is a tag in its own right and must not be
        # mistaken for de + "m".
        de_names = {i.name for i in schema.tracks["de"].companions.values()}
        self.assertNotIn("dem", de_names)


class TestConfigurableNames(unittest.TestCase):
    """Time-cluster and line-seed branch names come from fhicl, so find them by class."""

    def setUp(self):
        self.schema = build({
            "evtinfo": (obj("EventInfo"), ["event"]),
            "timeclusters": (vec("EventNtupleTimeClusterInfo"), ["timeclusters.nhits"]),
            "tcTrigger": (vec("EventNtupleTimeClusterInfo"), ["tcTrigger.nhits"]),
            "lineseeds": (vec("LineSeedInfo"), ["lineseeds.nhits"]),
            "cosmicSeeds": (vec("LineSeedInfo"), ["cosmicSeeds.nhits"]),
            "helices": (vec("HelixInfo"), ["helices.nhits"]),
        })

    def test_found_regardless_of_name(self):
        self.assertEqual(sorted(self.schema.names_of_kind("timecluster")),
                         ["tcTrigger", "timeclusters"])
        self.assertEqual(sorted(self.schema.names_of_kind("lineseed")),
                         ["cosmicSeeds", "lineseeds"])

    def test_helices_found(self):
        self.assertTrue(self.schema.has("helices"))


class TestPatternedCollections(unittest.TestCase):
    def test_mcsteps_and_crv_planes(self):
        schema = build({
            "mcsteps_virtualdetector": (vec("MCStepInfo"), ["mcsteps_virtualdetector.vid"]),
            "mcsteps_protonabsorber": (vec("MCStepInfo"), ["mcsteps_protonabsorber.vid"]),
            "crvcoincsmcplane": (vec("CrvPlaneInfoMC"), ["crvcoincsmcplane.x"]),
            "crvcoincsmcplane_top": (vec("CrvPlaneInfoMC"), ["crvcoincsmcplane_top.x"]),
        })
        self.assertEqual(sorted(schema.names_of_kind("mcstep")),
                         ["mcsteps_protonabsorber", "mcsteps_virtualdetector"])
        self.assertEqual(sorted(schema.names_of_kind("crvplanemc")),
                         ["crvcoincsmcplane", "crvcoincsmcplane_top"])

    def test_extra_mc_step_companions(self):
        branches = track_branches("trk")
        branches["trkmcsic_protonabsorber"] = vecvec("MCStepInfo")
        branches["trkmcssi_protonabsorber"] = vecvec("MCStepSummaryInfo")
        schema = build(branches)
        companions = schema.tracks["trk"].companions
        self.assertIn("mcsic_protonabsorber", companions)
        self.assertIn("mcssi_protonabsorber", companions)


class TestMultipleQualLeaves(unittest.TestCase):
    def test_each_qual_leaf_is_separate(self):
        schema = build(track_branches("trk", quals=("", "_bdt", "_ann2"), pids=2))
        companions = schema.tracks["trk"].companions
        for role, name in [("qual", "trkqual"), ("qual_bdt", "trkqual_bdt"),
                           ("qual_ann2", "trkqual_ann2"), ("pid", "trkpid"),
                           ("pid2", "trkpid2")]:
            self.assertIn(role, companions)
            self.assertEqual(companions[role].name, name)


class TestSchemaMismatch(unittest.TestCase):
    """A branch with the right name but the wrong struct is reported, never guessed at."""

    def test_wrong_struct_is_reported_not_claimed(self):
        branches = track_branches("trk")
        branches["trkhits"] = vecvec("CaloHitInfo")      # wrong struct
        schema = build(branches)
        self.assertNotIn("hits", schema.tracks["trk"].companions)
        self.assertTrue(any("trkhits" in m for m in schema.mismatches))
        self.assertIn("TrkStrawHitInfo", schema.mismatches[0])

    def test_wrong_depth_is_reported(self):
        branches = track_branches("trk")
        branches["trkmc"] = vecvec("TrkInfoMC")          # should be depth 1
        schema = build(branches)
        self.assertNotIn("mc", schema.tracks["trk"].companions)
        self.assertTrue(any("trkmc" in m for m in schema.mismatches))


class TestSelectionOnStub(unittest.TestCase):
    def test_collection_level_include(self):
        schema = discover(StubTree(track_branches("trk")), Selector(["trk", "trkhits"]))
        self.assertIs(schema.state("trk"), BranchState.LOADED)
        self.assertIs(schema.state("trkhits"), BranchState.LOADED)
        self.assertIs(schema.state("trksegs"), BranchState.NOT_SELECTED)

    def test_exclusion_after_include(self):
        schema = discover(StubTree(track_branches("trk")), Selector(["trk*", "!trkhits"]))
        self.assertIs(schema.state("trksegs"), BranchState.LOADED)
        self.assertIs(schema.state("trkhits"), BranchState.NOT_SELECTED)

    def test_callable_spec(self):
        schema = discover(StubTree(track_branches("trk")),
                          Selector(lambda name, typename: name.startswith("trkseg")))
        self.assertIs(schema.state("trksegs"), BranchState.LOADED)
        self.assertIs(schema.state("trkhits"), BranchState.NOT_SELECTED)


if __name__ == "__main__":
    unittest.main()
