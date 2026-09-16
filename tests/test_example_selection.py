"""Regression test for examples/07_signal_selection.py.

The example encodes a real analysis selection, so it is worth pinning: a cut flow must
never increase, the three track sets must nest, and the cuts must actually bite.
"""

import importlib
import os
import sys
import unittest

from pyevtana import Dataset

from base import NtupleTestCase

sys.path.insert(0, os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))),
                                "examples"))
selection = importlib.import_module("07_signal_selection")


class TestSelectionExample(NtupleTestCase):
    @classmethod
    def setUpClass(cls):
        super().setUpClass()
        dataset = Dataset(cls.path)
        (cls.n_events, cls.stages, cls.events_at,
         cls.tracks_in, cls.histograms) = selection.run(dataset)

    def test_cut_flow_is_monotonic(self):
        """Each stage is a subset of the one before it."""
        for earlier, later, stage in zip(self.events_at, self.events_at[1:],
                                         self.stages[1:]):
            self.assertLessEqual(later, earlier, f"count rose at {stage!r}")

    def test_cut_flow_never_exceeds_the_event_count(self):
        for count, stage in zip(self.events_at, self.stages):
            self.assertLessEqual(count, self.n_events, stage)

    def test_one_stage_per_cut_plus_preselection(self):
        self.assertEqual(len(self.stages), len(selection.CUTS) + 1)
        self.assertEqual(len(self.events_at), len(self.stages))

    def test_the_selection_actually_bites(self):
        """A selection that removes nothing, or everything, is not being tested."""
        self.assertGreater(self.events_at[0], 0)
        self.assertGreater(self.events_at[-1], 0)
        self.assertLess(self.events_at[-1], self.events_at[0])

    def test_track_sets_nest(self):
        self.assertLessEqual(self.tracks_in["selected"], self.tracks_in["all"])
        self.assertLessEqual(self.tracks_in["p > 90"], self.tracks_in["all"])

    def test_every_observable_is_filled_for_every_set(self):
        for name, _, _, _, _, _ in selection.OBSERVABLES:
            for label in selection.SETS:
                self.assertIn((name, label), self.histograms)
                self.assertGreater(self.histograms[(name, label)].values().sum(), 0,
                                   f"{name} / {label} is empty")

    def test_selected_tracks_are_in_the_signal_window(self):
        """Everything surviving must satisfy the momentum window the selection imposes."""
        hist = self.histograms[("p", "selected")]
        centers, counts = hist.axes[0].centers, hist.values()
        filled = centers[counts > 0]
        self.assertGreater(len(filled), 0)
        # the cut is on p at TT_Front; p at TT_Mid is close but not identical
        self.assertGreater(filled.min(), 95)
        self.assertLess(filled.max(), 115)

    def test_front_segment_choice_matters(self):
        """The first TT_Front crossing is the upstream one for reflected tracks.

        If this stops being true the example's use of segs_at() could be simplified --
        and if it silently changed, seg("TT_Front") would quietly return the wrong leg.
        """
        wrong_leg = 0
        for event in Dataset(self.path):
            for track in event.Tracks():
                fronts = track.segs_at("TT_Front")
                if len(fronts) > 1 and fronts[0].mom.z < 0:
                    wrong_leg += 1
        self.assertGreater(wrong_leg, 0)


if __name__ == "__main__":
    unittest.main()
