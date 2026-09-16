"""Shared fixtures for the test suite.

Written with stdlib ``unittest`` so it runs in the standard Mu2e ``rootana`` environment,
which has no pytest.  Run with::

    source setup.sh
    python3 -m unittest discover -s tests -v

pytest, if you have it in another environment, collects these TestCase classes too.
"""

import os
import unittest

#: A real EventNtuple: v6.13.1, single `trk` collection, LoopHelix fit, 100 events.
NTUPLE = "/exp/mu2e/app/users/mmackenz/main/nts.owner.description.version.sequencer.root"

HAVE_NTUPLE = os.path.exists(NTUPLE)
requires_ntuple = unittest.skipUnless(HAVE_NTUPLE, f"test ntuple not available: {NTUPLE}")


@requires_ntuple
class NtupleTestCase(unittest.TestCase):
    """Base class giving every test the sample file and a raw uproot handle."""

    path = NTUPLE

    @classmethod
    def setUpClass(cls):
        import uproot

        cls.file = uproot.open(NTUPLE)
        cls.tree = cls.file["EventNtuple/ntuple"]

    @classmethod
    def tearDownClass(cls):
        cls.file.close()
