"""Dropped and absent branches: the three states, and the policy for each.

The distinction that matters: a branch you *excluded* is a different situation from a
branch the file never had, and they get different treatment.
"""

import unittest
import warnings

from pyevtana import Dataset, is_missing
from pyevtana.missing import (BranchNotSelected, BranchState, MissingBranch,
                              MissingCollection, MissingRecord)
from pyevtana.record import NotPicklable

from base import NtupleTestCase

#: present in the sample file
PRESENT = "trkhits"
#: not present: made with fillHitCalibs false
ABSENT = "trkhitcalibs"


class TestAbsentBranches(NtupleTestCase):
    def test_strict_raises_with_an_actionable_hint(self):
        event = next(iter(Dataset(self.path, on_missing="strict")))
        track = event.Tracks("trk")[0]
        with self.assertRaises(MissingBranch) as caught:
            track.hitcalibs()
        message = str(caught.exception)
        self.assertIn(ABSENT, message)
        self.assertIn("fillHitCalibs", message)       # names the maker switch
        self.assertIn("on_missing='empty'", message)  # names the way out

    def test_empty_returns_a_falsy_empty_collection(self):
        event = next(iter(Dataset(self.path, on_missing="empty")))
        track = event.Tracks("trk")[0]
        result = track.hitcalibs()
        self.assertIsInstance(result, MissingCollection)
        self.assertEqual(len(result), 0)
        self.assertFalse(result)
        self.assertTrue(is_missing(result))
        # the whole point: looping over it is a no-op rather than a traceback
        self.assertEqual([h for h in result], [])

    def test_empty_keeps_the_reason_visible(self):
        event = next(iter(Dataset(self.path, on_missing="empty")))
        result = event.Tracks("trk")[0].hitcalibs()
        self.assertIs(result.state, BranchState.ABSENT)
        self.assertIn("fillHitCalibs", repr(result))

    def test_absent_single_record_is_falsy_and_yields_none(self):
        event = next(iter(Dataset(self.path, on_missing="empty")))
        # trksegpars_kl is absent (this is a LoopHelix ntuple)
        result = event.Tracks("trk")[0].segpars("kl")
        self.assertTrue(is_missing(result))

    def test_warn_mode_warns_once_then_behaves_like_empty(self):
        import pyevtana.missing as missing_module

        missing_module._warned.clear()
        event = next(iter(Dataset(self.path, on_missing="warn")))
        track = event.Tracks("trk")[0]
        with warnings.catch_warnings(record=True) as caught:
            warnings.simplefilter("always")
            first = track.hitcalibs()
        self.assertEqual(len(caught), 1)
        self.assertEqual(len(first), 0)

    def test_missing_record_fields_read_back_as_none(self):
        record = MissingRecord("trkmc", BranchState.ABSENT, "fillMC false")
        self.assertIsNone(record.pdg)
        self.assertFalse(record)
        self.assertEqual(record.fields, [])

    def test_event_has_reports_availability(self):
        event = next(iter(Dataset(self.path, on_missing="empty")))
        self.assertTrue(event.has(PRESENT))
        self.assertFalse(event.has(ABSENT))
        self.assertIs(event.state(ABSENT), BranchState.ABSENT)

    def test_track_has_companion(self):
        event = next(iter(Dataset(self.path, on_missing="empty")))
        track = event.Tracks("trk")[0]
        self.assertTrue(track.has_companion("hits"))
        self.assertFalse(track.has_companion("hitcalibs"))


class TestDroppedBranches(NtupleTestCase):
    """Branches excluded by `branches=` are a different failure from absent ones."""

    def test_excluded_branch_always_raises_even_under_empty(self):
        dataset = Dataset(self.path, branches=["evtinfo", "trk", "trksegs"],
                          on_missing="empty")
        event = next(iter(dataset))
        track = event.Tracks("trk")[0]
        with self.assertRaises(BranchNotSelected) as caught:
            track.hits()
        message = str(caught.exception)
        self.assertIn("trkhits", message)
        self.assertIn("branches=", message)
        # and it does NOT pretend the branch is absent
        self.assertNotIn("not present", message)

    def test_state_distinguishes_excluded_from_absent(self):
        dataset = Dataset(self.path, branches=["evtinfo", "trk"])
        schema = dataset.schema
        self.assertIs(schema.state("trkhits"), BranchState.NOT_SELECTED)
        self.assertIs(schema.state("trkhitcalibs"), BranchState.ABSENT)
        self.assertIs(schema.state("trk"), BranchState.LOADED)

    def test_leaf_level_selection(self):
        """Split branches can be cut down to individual leaves."""
        dataset = Dataset(self.path, branches=["trk.pdg", "trk.nactive"])
        info = dataset.schema.get("trk")
        self.assertEqual(sorted(info.selected), ["trk.nactive", "trk.pdg"])
        batch = next(dataset.batches())
        self.assertEqual(sorted(batch.get("trk").fields), ["nactive", "pdg"])

    def test_glob_and_exclusion(self):
        dataset = Dataset(self.path, branches=["trk*", "!trkhits*", "!trkmats"])
        schema = dataset.schema
        self.assertIs(schema.state("trk"), BranchState.LOADED)
        self.assertIs(schema.state("trksegs"), BranchState.LOADED)
        self.assertIs(schema.state("trkhits"), BranchState.NOT_SELECTED)
        self.assertIs(schema.state("trkhitsmc"), BranchState.NOT_SELECTED)
        self.assertIs(schema.state("trkmats"), BranchState.NOT_SELECTED)
        self.assertIs(schema.state("evtinfo"), BranchState.NOT_SELECTED)


class TestLaziness(NtupleTestCase):
    """A loop that never touches a branch must never read it."""

    def test_only_touched_branches_are_read(self):
        dataset = Dataset(self.path)
        batch = next(dataset.batches())
        self.assertEqual(batch._cache, {})
        for event in batch.events():
            for track in event.Tracks("trk"):
                _ = track.nactive
        self.assertEqual(set(batch._cache), {"trk"})
        self.assertNotIn("trkhits", batch._cache)

    def test_reading_hits_adds_exactly_that_branch(self):
        dataset = Dataset(self.path)
        batch = next(dataset.batches())
        for event in batch.events():
            for track in event.Tracks("trk"):
                for _ in track.hits():
                    pass
        self.assertEqual(set(batch._cache), {"trk", "trkhits"})


class TestRequired(NtupleTestCase):
    """`required=` fails at open time, not 40 minutes into a loop."""

    def test_missing_required_branch_fails_fast(self):
        with self.assertRaises(MissingBranch) as caught:
            Dataset(self.path, required=["trk", ABSENT]).schema
        self.assertIn(ABSENT, str(caught.exception))

    def test_required_present_is_fine(self):
        dataset = Dataset(self.path, required=["trk", "trksegs", "evtinfo"])
        self.assertTrue(dataset.schema.has("trk"))

    def test_required_conflicting_with_filter_is_reported(self):
        from pyevtana.missing import PyEvtAnaError

        with self.assertRaises(PyEvtAnaError) as caught:
            Dataset(self.path, branches=["evtinfo"], required=["trkhits"]).schema
        self.assertIn("disagree", str(caught.exception))


class TestProxiesRefusePickling(NtupleTestCase):
    def test_event_and_track_are_not_picklable(self):
        import pickle

        event = next(iter(Dataset(self.path)))
        with self.assertRaises(NotPicklable):
            pickle.dumps(event)
        with self.assertRaises(NotPicklable):
            pickle.dumps(event.Tracks("trk"))


if __name__ == "__main__":
    unittest.main()
