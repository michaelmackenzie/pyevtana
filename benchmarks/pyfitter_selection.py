"""pyevtana implementation of pyfitter's AnaProcessor selection, cut for cut.

The benchmark compares processing *speed*, so both frameworks must do the same work. This
reproduces the 19 active cuts of `pyfitter/analyze.py::define_cuts` with pyfitter's own
definitions, including the ones that differ from how one might naturally write them:

* **downstream** is the sign of pz on the *earliest-time* TT_Front segment, not pz at
  TT_Mid.
* the **upstream veto** compares ``dt = t_this - t_partner`` against ``[40, 110]`` (this
  track later than the partner), with times taken from the *first* TT_Front segment, and
  a partner is any track that is not downstream.
* the **multi-track veto** uses the *mean* TT_Front time, not the first, and counts
  coincident downstream e+/e- tracks within ``|dt| < 150``.
* **good_trkpid** additionally requires some calorimeter cluster energy in the event.
* **tanDip** is read from ``trksegpars_lh`` at the first TT_Front segment.
* the surface counts use segment counts (``ST_Foils`` > 0, ``OPA`` == 0), matching
  ``pyutils.pyselect.has_ST``/``has_OPA``.
* ``in_mom_range`` and ``within_t0_475`` require *every* TT_Front segment to pass, which is
  what ``ak.all(~at_trk_front | cond)`` means.

The cut flow is per-event and cumulative -- an event counts at a stage if any track still
passes -- which is exactly what ``pyutils.pycut.create_cut_flow`` reports, so the two can
be compared stage by stage. `validate_against_pyfitter.py` does that.
"""

from statistics import fmean

from pyevtana import is_missing
from pyevtana.surfaces import surface_id

TT_FRONT = surface_id("TT_Front")
TT_MID = surface_id("TT_Mid")
ST_FOILS = surface_id("ST_Foils")
OPA = surface_id("OPA")
#: pyfitter's st_boundary cut (Track_t::STBoundary) -- any stopping-target bounding surface
ST_BOUNDARY = {surface_id(n) for n in ("ST_Front", "ST_Back", "ST_Inner", "ST_Outer")}

#: branches this selection reads
BRANCHES = ["trk", "trkqual", "trkpid", "trksegs", "trksegpars_lh", "crvcoincs",
            "caloclusters"]

#: pyfitter's own branch list, for an I/O-matched comparison
PYFITTER_BRANCHES = ["evtinfo", "trk", "trkqual", "trkpid", "trksegs", "trksegsmc",
                     "trksegpars_lh", "trkmcsim", "trkmc", "crvcoincs", "caloclusters"]

MOM_LO, MOM_HI = 100.0, 115.0          # AnaProcessor defaults
TRIGGERS = ("cpr_TrkDe_80m70p", "apr_TrkDe_80m70p")


class TrackView:
    """Everything the selection needs from one track, computed once."""

    __slots__ = ("index", "pdg", "status", "goodfit", "nactive", "qual", "pid",
                 "front_times", "front_pz", "front_mom_mag", "mid_t0err", "tandip",
                 "n_st", "n_opa", "st_boundary", "is_good", "is_downstream",
                 "first_front_t", "mean_front_t")

    def __init__(self, track):
        self.index = track.index
        self.pdg = int(track.pdg)
        self.status = int(track.status)
        self.goodfit = int(track.goodfit)
        self.nactive = int(track.nactive)
        self.qual = float(track.qual().result)
        self.pid = float(track.pid().result)

        segs = track.segs()
        sids = segs._column("sid")
        times = segs._column("time")
        moms = segs._column("mom")

        front = [i for i, sid in enumerate(sids) if sid == TT_FRONT]
        self.front_times = [float(times[i]) for i in front]
        self.front_pz = [float(moms[i].z) for i in front]
        self.front_mom_mag = [moms[i].mag for i in front]
        self.n_st = sum(1 for sid in sids if sid == ST_FOILS)
        self.n_opa = sum(1 for sid in sids if sid == OPA)
        self.st_boundary = any(sid in ST_BOUNDARY for sid in sids)

        pars = track.segpars()
        mid = [i for i, sid in enumerate(sids) if sid == TT_MID]
        if is_missing(pars):
            self.mid_t0err = []
            self.tandip = -100.0
        else:
            t0err = pars._column("t0err")
            self.mid_t0err = [float(t0err[i]) for i in mid]
            self.tandip = float(pars._column("tanDip")[front[0]]) if front else -100.0

        # pyfitter: the earliest-time front segment decides the direction
        if front:
            earliest = min(self.front_times)
            self.is_downstream = any(pz > 0 for t, pz in zip(self.front_times, self.front_pz)
                                     if t == earliest)
            self.first_front_t = self.front_times[0]
            self.mean_front_t = fmean(self.front_times)
        else:
            self.is_downstream = False
            self.first_front_t = 0.0
            self.mean_front_t = 0.0

        self.is_good = self.status >= 0 and self.goodfit != 0


class EventView:
    """Per-event quantities plus the per-track views."""

    __slots__ = ("tracks", "trigger", "crv_times", "has_calo_energy")

    def __init__(self, event):
        self.tracks = [TrackView(t) for t in event.Tracks()]
        self.trigger = any(event.trigger(name) for name in TRIGGERS)
        self.crv_times = ([float(c.time) for c in event.CrvCoincs()]
                          if event.has("crvcoincs") else [])
        self.has_calo_energy = (
            any(c.energyDep_ > 0 for c in event.CaloClusters())
            if event.has("caloclusters") else False)


# -- the cuts, in pyfitter's order ---------------------------------------------------------


def has_a_track(t, e):
    return bool(e.tracks)


def is_good_track(t, e):
    return t.status >= 0 and t.goodfit != 0


def has_trk_front_seg(t, e):
    return bool(t.front_times)


def is_reco_electron_or_positron(t, e):
    return abs(t.pdg) == 11


def has_downstream(t, e):
    return t.is_downstream


def charge_selection(t, e):
    return t.pdg == 11


def or_trigger(t, e):
    return e.trigger


def upstream_veto(t, e):
    """No good non-downstream partner with t_this - t_partner in [40, 110] ns."""
    if not (t.is_downstream and t.is_good):
        return True
    for other in e.tracks:
        if other.index == t.index or other.is_downstream or not other.is_good:
            continue
        dt = t.first_front_t - other.first_front_t
        if 40.0 <= dt <= 110.0:
            return False
    return True


def no_multi_trk_veto(t, e):
    """No other downstream e+/e- track within |dt| < 150 ns of the mean front time."""
    mine = t.is_downstream and abs(t.pdg) == 11 and t.is_good
    if not mine:
        return True
    for other in e.tracks:
        if other.index == t.index:
            continue
        if not (other.is_downstream and abs(other.pdg) == 11 and other.is_good):
            continue
        if abs(t.mean_front_t - other.mean_front_t) < 150.0:
            return False
    return True


def good_trkpid(t, e):
    return t.pid > 0.54 and e.has_calo_energy


def pz_over_pt(t, e):
    return 0.575 < t.tandip < 0.85


def st_boundary(t, e):
    """Any stopping-target bounding surface intersection (Track_t::STBoundary)."""
    return t.st_boundary


def has_st(t, e):
    return t.n_st > 0


def no_opa(t, e):
    return t.n_opa == 0


def good_trkqual(t, e):
    return t.qual > 0.155


def has_hits(t, e):
    return t.nactive >= 20


def within_t0err(t, e):
    return all(err < 0.85 for err in t.mid_t0err)


def no_crv_veto(t, e):
    for time in t.front_times:
        for crv in e.crv_times:
            if 0 < time - crv < 150:
                return False
    return True


def in_mom_range(t, e):
    return all(MOM_LO < mag < MOM_HI for mag in t.front_mom_mag)


def within_t0_475(t, e):
    return all(475 < time < 1650 for time in t.front_times)


CUTS = [
    ("has_a_track", has_a_track),
    ("is_good_track", is_good_track),
    ("has_trk_front_seg", has_trk_front_seg),
    ("is_reco_electron_or_positron", is_reco_electron_or_positron),
    ("has_downstream", has_downstream),
    ("charge_selection", charge_selection),
    ("or_trigger", or_trigger),
    ("upstream_veto", upstream_veto),
    ("no_multi_trk_veto", no_multi_trk_veto),
    ("good_trkpid", good_trkpid),
    ("pz_over_pt", pz_over_pt),
    ("st_boundary", st_boundary),
    ("has_st", has_st),
    ("no_opa", no_opa),
    ("good_trkqual", good_trkqual),
    ("has_hits", has_hits),
    ("within_t0err", within_t0err),
    ("no_crv_veto", no_crv_veto),
    ("in_mom_range", in_mom_range),
    ("within_t0_475", within_t0_475),
]


def run(dataset):
    """Per-event cumulative cut flow, the same quantity pyfitter's create_cut_flow reports."""
    stages = [name for name, _ in CUTS]
    events_at = [0] * len(stages)
    n_events = 0
    selected = 0

    for event in dataset:
        n_events += 1
        view = EventView(event)
        surviving = view.tracks
        for i, (_, cut) in enumerate(CUTS):
            surviving = [t for t in surviving if cut(t, view)]
            if surviving:
                events_at[i] += 1
        selected += len(surviving)

    return {"n_events": n_events, "stages": stages, "events_at": events_at,
            "selected": selected}
