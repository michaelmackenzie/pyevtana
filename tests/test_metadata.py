"""Run/subrun bookkeeping, and graceful degradation of links whose target is absent."""

import os
import tempfile
import unittest

from pyevtana import Dataset, is_missing
from pyevtana.metadata import n_proc_events, summarize

from base import NtupleTestCase


class TestSummary(NtupleTestCase):
    def test_totals_match_the_file(self):
        summary = Dataset(self.path).summary()
        self.assertTrue(summary.complete)
        self.assertEqual(summary.n_files, 1)
        self.assertEqual(summary.proc_events, self.tree.num_entries)
        self.assertEqual(summary.runs, {1430})
        self.assertEqual(summary.subruns, {(1430, 0)})

    def test_cross_check_against_the_histogram(self):
        self.assertEqual(n_proc_events(self.path), Dataset(self.path).summary().proc_events)

    def test_totals_add_up_over_files(self):
        with tempfile.TemporaryDirectory() as tmp:
            for name in ("a.root", "b.root"):
                os.symlink(self.path, os.path.join(tmp, name))
            summary = Dataset(os.path.join(tmp, "*.root")).summary()
            self.assertEqual(summary.proc_events, 2 * self.tree.num_entries)

    def test_unreadable_file_is_reported_not_silently_dropped(self):
        """A normalization computed from a partial read must be visibly partial."""
        with tempfile.TemporaryDirectory() as tmp:
            good = os.path.join(tmp, "good.root")
            bad = os.path.join(tmp, "bad.root")
            os.symlink(self.path, good)
            with open(bad, "w") as handle:
                handle.write("not a root file")
            summary = summarize([good, bad], "EventNtuple/subrunNtuple", on_error="skip")
            self.assertFalse(summary.complete)
            self.assertEqual(len(summary.unreadable), 1)
            self.assertEqual(summary.proc_events, self.tree.num_entries)


class TestAbsentLinkTargets(NtupleTestCase):
    """This file has no calorecodigis/calodigis, so those links have nothing to point at."""

    def test_link_to_an_absent_branch_degrades(self):
        dataset = Dataset(self.path, on_missing="empty")
        self.assertFalse(dataset.schema.has("calorecodigis"))
        for event in dataset:
            for cluster in event.CaloClusters():
                for hit in cluster.hits():
                    digis = hit.recodigis()
                    self.assertTrue(is_missing(digis))
                    self.assertEqual(len(digis), 0)
                    return
        self.skipTest("no calo clusters in this file")

    def test_link_to_an_absent_branch_raises_under_strict(self):
        from pyevtana.missing import MissingBranch

        dataset = Dataset(self.path, on_missing="strict")
        for event in dataset:
            for cluster in event.CaloClusters():
                for hit in cluster.hits():
                    with self.assertRaises(MissingBranch):
                        hit.recodigis()
                    return
        self.skipTest("no calo clusters in this file")


if __name__ == "__main__":
    unittest.main()
