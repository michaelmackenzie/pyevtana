"""Partitioning and parallel execution.

Worker functions are module-level on purpose: a lambda or a local `def` cannot be pickled,
so it cannot run in a worker process, and `map` says so rather than letting
multiprocessing produce a baffling error.
"""

import os
import tempfile
import unittest
from collections import Counter

from pyevtana import Dataset
from pyevtana.accumulate import add, merge_counters
from pyevtana.missing import PyEvtAnaError
from pyevtana.parallel import Partition, build_partitions

from base import NtupleTestCase


# -- module-level workers ----------------------------------------------------------------


def count_tracks(chunk):
    """Total tracks in a partition -- a plain, picklable result."""
    return sum(len(event.Tracks("trk")) for event in chunk)


def count_pdgs(chunk):
    counts = Counter()
    for event in chunk:
        for track in event.Tracks("trk"):
            counts[int(track.pdg)] += 1
    return counts


def momentum_sum(chunk):
    total = 0.0
    for event in chunk:
        for track in event.Tracks("trk"):
            seg = track.seg("TT_Mid")
            if seg is not None:
                total += float(seg.mom.mag)
    return total


def boom(chunk):
    raise RuntimeError("deliberate failure")


class TestPartitioning(NtupleTestCase):
    def test_one_partition_per_file_by_default(self):
        """A whole-file partition leaves `stop` open so the file need not be opened."""
        parts = build_partitions(Dataset(self.path), None)
        self.assertEqual(len(parts), 1)
        self.assertEqual(parts[0].start, 0)
        self.assertIsNone(parts[0].stop)

    def test_default_partitioning_opens_no_files(self):
        """Opening files to partition means schema discovery for each, serially, in the
        parent -- ~0.4 s per EventNtuple, which at 60 files cost more than the parallel
        read that followed. The worker opens its own file anyway."""
        dataset = Dataset(self.path)
        build_partitions(dataset, None)
        self.assertEqual(dataset._readers, {},
                         "partitioning opened files in the parent process")

    def test_large_file_is_split(self):
        """A dataset that is really one big file must still parallelize."""
        parts = build_partitions(Dataset(self.path), 30)
        self.assertEqual(len(parts), 4)
        self.assertEqual([(p.start, p.stop) for p in parts],
                         [(0, 30), (30, 60), (60, 90), (90, 100)])
        self.assertEqual(sum(p.num_entries for p in parts), 100)

    def test_partitions_cover_every_entry_exactly_once(self):
        for max_entries in (7, 30, 1000):
            parts = build_partitions(Dataset(self.path), max_entries)
            covered = [i for p in parts for i in range(p.start, p.stop)]
            self.assertEqual(covered, list(range(100)), f"max_entries={max_entries}")
        # the default whole-file partition covers everything by construction
        parts = build_partitions(Dataset(self.path), None)
        self.assertEqual([(p.start, p.stop) for p in parts], [(0, None)])


class TestMap(NtupleTestCase):
    @classmethod
    def setUpClass(cls):
        super().setUpClass()
        cls.expected_tracks = sum(len(e.Tracks("trk")) for e in Dataset(cls.path))

    def test_serial_matches_a_plain_loop(self):
        result = Dataset(self.path).map(count_tracks, reduce=sum)
        self.assertEqual(result, self.expected_tracks)

    def test_split_partitions_reproduce_the_whole_file_answer(self):
        result = Dataset(self.path).map(count_tracks, reduce=sum, max_entries=17)
        self.assertEqual(result, self.expected_tracks)

    def test_thread_backend_matches_serial(self):
        result = Dataset(self.path).map(count_tracks, backend="thread", workers=4,
                                        reduce=sum, max_entries=25)
        self.assertEqual(result, self.expected_tracks)

    def test_process_backend_matches_serial(self):
        result = Dataset(self.path).map(count_tracks, backend="process", workers=2,
                                        reduce=sum, max_entries=25)
        self.assertEqual(result, self.expected_tracks)

    def test_reduce_helpers(self):
        counts = Dataset(self.path).map(count_pdgs, reduce=merge_counters, max_entries=25)
        self.assertGreater(counts[11], 0)
        self.assertEqual(sum(counts.values()), self.expected_tracks)

    def test_add_helper_folds_without_a_zero(self):
        total = Dataset(self.path).map(momentum_sum, reduce=add, max_entries=25)
        single = Dataset(self.path).map(momentum_sum, reduce=add)
        self.assertAlmostEqual(total, single, places=3)

    def test_ordered_results_follow_dataset_order(self):
        results = Dataset(self.path).map(count_tracks, backend="thread", workers=4,
                                         max_entries=25, ordered=True)
        serial = Dataset(self.path).map(count_tracks, max_entries=25)
        self.assertEqual(results, serial)

    def test_unpicklable_worker_is_rejected_up_front(self):
        with self.assertRaises(PyEvtAnaError) as caught:
            Dataset(self.path).map(lambda chunk: 0, backend="process", workers=2,
                                   max_entries=25)
        self.assertIn("pickle", str(caught.exception).lower())


class TestErrorIsolation(NtupleTestCase):
    def test_raise_is_the_default(self):
        with self.assertRaises(RuntimeError):
            Dataset(self.path).map(boom)

    def test_skip_keeps_going_and_records_the_failure(self):
        dataset = Dataset(self.path)
        results = dataset.map(boom, on_error="skip", max_entries=50)
        self.assertEqual(results, [])
        self.assertEqual(len(dataset.errors), 2)
        self.assertIn("deliberate failure", dataset.errors[0][1])

    def test_collect_returns_results_and_failures(self):
        dataset = Dataset(self.path)
        results, failures = dataset.map(boom, on_error="collect", max_entries=50)
        self.assertEqual(results, [])
        self.assertEqual(len(failures), 2)

    def test_unreadable_file_is_skipped(self):
        with tempfile.TemporaryDirectory() as tmp:
            good = os.path.join(tmp, "good.root")
            bad = os.path.join(tmp, "bad.root")
            os.symlink(self.path, good)
            with open(bad, "w") as handle:
                handle.write("not a root file")
            dataset = Dataset([good, bad], on_missing="empty")
            results = dataset.map(count_tracks, on_error="skip")
            self.assertEqual(len(results), 1)
            self.assertEqual(len(dataset.errors), 1)


class TestMultiFile(NtupleTestCase):
    """File-level parallelism, the case the partitioner is built around."""

    def test_two_files_give_two_partitions_and_double_the_tracks(self):
        with tempfile.TemporaryDirectory() as tmp:
            first = os.path.join(tmp, "a.root")
            second = os.path.join(tmp, "b.root")
            os.symlink(self.path, first)
            os.symlink(self.path, second)

            dataset = Dataset([first, second])
            self.assertEqual(len(build_partitions(dataset, None)), 2)
            self.assertEqual(dataset.num_entries, 200)

            single = Dataset(self.path).map(count_tracks, reduce=sum)
            both = dataset.map(count_tracks, backend="thread", workers=2, reduce=sum)
            self.assertEqual(both, 2 * single)

    def test_glob_resolution(self):
        with tempfile.TemporaryDirectory() as tmp:
            for name in ("a.root", "b.root"):
                os.symlink(self.path, os.path.join(tmp, name))
            dataset = Dataset(os.path.join(tmp, "*.root"))
            self.assertEqual(len(dataset.paths), 2)

    def test_filelist_resolution(self):
        with tempfile.TemporaryDirectory() as tmp:
            listing = os.path.join(tmp, "files.txt")
            with open(listing, "w") as handle:
                handle.write(f"# a comment\n{self.path}\n")
            dataset = Dataset(listing)
            self.assertEqual(dataset.paths, [self.path])


if __name__ == "__main__":
    unittest.main()
