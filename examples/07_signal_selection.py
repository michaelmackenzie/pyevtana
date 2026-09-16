#!/usr/bin/env python3
"""Histogram observables for downstream electron tracks, with a per-event cut flow.

Fills p, pT, cos(theta), t and calorimeter cluster energy for three track sets:

  * all      -- every downstream electron track (pz at TT_Mid > 0, |fit PDG| == 11)
  * p > 90   -- the same, above 90 MeV/c
  * selected -- the same, passing the full selection below

and prints the selection as a **per-event** cut flow: at each stage, the number of events
still holding at least one surviving track.

    python3 examples/07_signal_selection.py [files...] [--plot spectrum.png]

One subtlety this example has to get right: a track can cross ``TT_Front`` twice. In the
sample file 164 of 375 tracks do, and the *first* crossing is always the upstream-going
one -- so ``track.seg("TT_Front")`` would return the wrong leg for a downstream analysis.
Every "front" quantity below is taken from the crossing with pz > 0, via ``segs_at``.
"""

from collections import Counter
from dataclasses import dataclass, field
from functools import cached_property
from typing import Optional

from _common import parse

from pyevtana import Dataset, is_missing

#: the event passes if either of these fired
TRIGGERS = ("apr_TrkDe_80m70p", "cpr_TrkDe_80m70p")


def charge(pdg: int) -> int:
    """Charge of a charged lepton from its PDG code (11 = e-, -11 = e+).

    The ntuple stores no charge, only the PDG of the fit hypothesis.
    """
    return -1 if pdg > 0 else 1


# --------------------------------------------------------------------------------------
# candidates
# --------------------------------------------------------------------------------------


@dataclass
class Candidate:
    """A downstream electron track, with the segments the selection needs.

    Every derived quantity is a ``cached_property``: the cut flow and the histograms both
    read ``p``, ``pt`` and the front-segment momentum, and each read of ``seg.mom`` builds
    a fresh vector, so computing them once per track matters.
    """

    track: object
    index: int
    mid: object                     # TT_Mid segment (defines "downstream")
    front: Optional[object]         # downstream-going TT_Front crossing, if any
    t0err: float                    # fit t0 uncertainty at TT_Mid

    @cached_property
    def mom(self):
        return self.mid.mom

    @cached_property
    def front_mom(self):
        return self.front.mom if self.front is not None else None

    @cached_property
    def p(self) -> float:
        return float(self.mom.mag)

    @cached_property
    def pt(self) -> float:
        return float(self.mom.pt)

    @cached_property
    def cos(self) -> float:
        return float(self.mom.z) / self.p

    @cached_property
    def t(self) -> float:
        return float(self.mid.time)

    @cached_property
    def front_t(self) -> Optional[float]:
        return float(self.front.time) if self.front is not None else None

    @cached_property
    def qual(self) -> float:
        return float(self.track.qual().result)

    @cached_property
    def pid(self) -> float:
        return float(self.track.pid().result)

    @cached_property
    def ecalo(self) -> Optional[float]:
        """Calorimeter cluster energy, or None when the track has no cluster."""
        calohit = self.track.calohit()
        return float(calohit.edep) if int(calohit.did) >= 0 else None


class EventContext:
    """Per-event quantities the cross-track and CRV cuts need."""

    __slots__ = ("trigger", "crv_times", "electrons")

    def __init__(self, trigger, crv_times, electrons):
        self.trigger = trigger
        self.crv_times = crv_times
        self.electrons = electrons


def scan(event):
    """One pass over the tracks, producing both the context and the candidates.

    Scanning a track's segments is the most expensive thing done per track, so the
    upstream/downstream TT_Front legs and the TT_Mid crossing are found once here and
    shared, rather than re-derived by a separate preselection pass.
    """
    electrons = []
    candidates = []

    for track in event.Tracks():
        if abs(int(track.pdg)) != 11:
            continue
        fronts = track.segs_at("TT_Front")
        mids = track.segs_at("TT_Mid")
        up_fronts, down_fronts = [], []
        for seg in fronts:
            (down_fronts if seg.mom.z > 0 else up_fronts).append(seg)
        downstream = bool(mids) and mids[0].mom.z > 0

        electrons.append({
            "index": track.index,
            "up_fronts": up_fronts,
            "down_fronts": down_fronts,
            "downstream": downstream,
        })
        if not downstream:
            continue

        mid = mids[0]
        # segpars share the segment indexing, so the fit t0 error at TT_Mid is the
        # segpars entry at the same position
        pars = track.segpars()
        t0err = float(pars[mid.index].t0err) if not is_missing(pars) else float("inf")
        candidates.append(Candidate(track=track, index=track.index, mid=mid,
                                    front=down_fronts[0] if down_fronts else None,
                                    t0err=t0err))

    context = EventContext(
        trigger=any(event.trigger(name) for name in TRIGGERS),
        crv_times=([float(c.time) for c in event.CrvCoincs()]
                   if event.has("crvcoincs") else []),
        electrons=electrons,
    )
    return context, candidates


def preselect(event) -> list:
    """Downstream electron tracks: pz at TT_Mid > 0 and |fit PDG| == 11."""
    return scan(event)[1]


# --------------------------------------------------------------------------------------
# the selection, in order
# --------------------------------------------------------------------------------------


def cut_fit_status(c, ctx):
    return int(c.track.status) >= 0 and int(c.track.goodfit) == 1


def cut_charge(c, ctx):
    return charge(int(c.track.pdg)) < 0


def cut_trigger(c, ctx):
    return ctx.trigger


def cut_no_upstream_partner(c, ctx):
    """No upstream-moving electron track arriving 40-110 ns after this one at TT_Front."""
    if c.front is None:
        return False
    mine = c.front_t
    for other in ctx.electrons:
        for seg in other["up_fronts"]:
            if 40 < float(seg.time) - mine < 110:
                return False
    return True


def cut_no_downstream_partner(c, ctx):
    """No *other* downstream electron track within +-150 ns of this one at TT_Front."""
    if c.front is None:
        return False
    mine = c.front_t
    for other in ctx.electrons:
        if other["index"] == c.index or not other["downstream"]:
            continue
        for seg in other["down_fronts"]:
            if -150 < float(seg.time) - mine < 150:
                return False
    return True


def cut_pid(c, ctx):
    return c.pid > 0.54


def cut_tandip_front(c, ctx):
    """Polar-angle window at TT_Front: 0.575 < tan(dip) < 0.85, i.e. 49.6 to 60.1 deg."""
    if c.front is None:
        return False
    mom = c.front_mom
    return mom.pt != 0 and 0.575 < mom.z / mom.pt < 0.85


def cut_st_intersections(c, ctx):
    """TrkInfo counts stopping-target foil intersections up- and downstream separately."""
    return int(c.track.nstup) + int(c.track.nstdown) > 0


def cut_no_opa(c, ctx):
    return not bool(c.track.opainter)


def cut_no_tsda(c, ctx):
    return not bool(c.track.tsdainter)


def cut_trkqual(c, ctx):
    return c.qual > 0.155


def cut_nactive(c, ctx):
    return int(c.track.nactive) >= 20


def cut_t0err(c, ctx):
    return c.t0err < 0.85


def cut_crv_veto(c, ctx):
    """No CRV cluster 0-150 ns before the track reaches the tracker front."""
    if c.front is None:
        return False
    return not any(0 < c.front_t - t < 150 for t in ctx.crv_times)


def cut_p_front(c, ctx):
    return c.front is not None and 100 < c.front_mom.mag < 110


def cut_t_front(c, ctx):
    return c.front is not None and 540 < c.front_t < 1650


CUTS = [
    ("fit status >= 0 and goodfit == 1", cut_fit_status),
    ("charge < 0", cut_charge),
    ("apr or cpr TrkDe_80m70p trigger", cut_trigger),
    ("no upstream e- at dt_front in (40, 110)", cut_no_upstream_partner),
    ("no other downstream e- within |dt_front| < 150", cut_no_downstream_partner),
    ("PID > 0.54", cut_pid),
    ("0.575 < tan(dip) = pz/pT (TT_Front) < 0.85", cut_tandip_front),
    ("N(ST foil intersections) > 0", cut_st_intersections),
    ("N(OPA intersections) == 0", cut_no_opa),
    ("N(TSdA intersections) == 0", cut_no_tsda),
    ("trkqual > 0.155", cut_trkqual),
    ("N(active hits) >= 20", cut_nactive),
    ("t0 error (TT_Mid) < 0.85", cut_t0err),
    ("no CRV cluster 0 < dt < 150 ns", cut_crv_veto),
    ("100 < p (TT_Front) < 110", cut_p_front),
    ("540 < t (TT_Front) < 1650", cut_t_front),
]

# --------------------------------------------------------------------------------------
# observables and histograms
# --------------------------------------------------------------------------------------

OBSERVABLES = [
    ("p", "p at TT_Mid [MeV/c]", 120, 80, 120, lambda c: c.p),
    ("pt", "pT at TT_Mid [MeV/c]", 100, 0, 110, lambda c: c.pt),
    ("cos", "cos(theta) = pz/p at TT_Mid", 100, 0, 1, lambda c: c.cos),
    ("t", "t at TT_Mid [ns]", 120, 400, 1700, lambda c: c.t),
    ("ecalo", "cluster energy [MeV]", 120, 0, 120, lambda c: c.ecalo),
]

SETS = ["all", "p > 90", "selected"]


def make_histograms():
    from hist import Hist

    return {
        (name, label): Hist.new.Reg(bins, lo, hi, name=name, label=axis).Double()
        for name, axis, bins, lo, hi, _ in OBSERVABLES
        for label in SETS
    }


class Collector:
    """Gather observable values, then fill each histogram once.

    ``Hist.fill(scalar)`` costs ~33 us per value -- filling one value at a time was by far
    the most expensive thing this example did, several times the cost of reading the data.
    Filling the same values as one array is ~5000x cheaper per value and gives identical
    contents, so values are accumulated in lists and the histograms are built at the end.
    """

    __slots__ = ("values",)

    def __init__(self):
        self.values = {(name, label): []
                       for name, _, _, _, _, _ in OBSERVABLES for label in SETS}

    def add(self, label, candidate):
        for name, _, _, _, _, value_of in OBSERVABLES:
            value = value_of(candidate)
            if value is not None:      # ecalo is None when the track has no cluster
                self.values[(name, label)].append(value)

    def histograms(self):
        import numpy as np

        histograms = make_histograms()
        for key, values in self.values.items():
            if values:
                histograms[key].fill(np.asarray(values, dtype=float))
        return histograms


# --------------------------------------------------------------------------------------


def run(dataset):
    """Fill the histograms and the per-event cut flow. Returns everything report() needs."""
    collector = Collector()
    stages = ["preselected: downstream e- track"] + [label for label, _ in CUTS]
    events_at = [0] * len(stages)
    tracks_in = Counter()
    n_events = 0

    for event in dataset:
        n_events += 1
        context, candidates = scan(event)

        for candidate in candidates:
            tracks_in["all"] += 1
            collector.add("all", candidate)
            if candidate.p > 90:
                tracks_in["p > 90"] += 1
                collector.add("p > 90", candidate)

        # per-event cut flow: an event counts at a stage if >= 1 track is still alive
        if candidates:
            events_at[0] += 1
        surviving = candidates
        for i, (_, cut) in enumerate(CUTS, start=1):
            surviving = [c for c in surviving if cut(c, context)]
            if surviving:
                events_at[i] += 1

        for candidate in surviving:
            tracks_in["selected"] += 1
            collector.add("selected", candidate)

    return n_events, stages, events_at, tracks_in, collector.histograms()


def check_supported(dataset):
    needed = ["trk", "trksegs", "trksegpars_lh", "trkcalohit", "trkqual", "trkpid"]
    absent = [n for n in needed if not dataset.schema.has(n)]
    if absent:
        raise SystemExit(f"this file cannot support the selection; missing: {absent}\n"
                         f"run pyevtana-describe on it to see what it has")


def main():
    args = parse(__doc__, plot={"default": None, "help": "write a PNG of the spectra"})

    dataset = Dataset(args.files)
    check_supported(dataset)
    n_events, stages, events_at, tracks_in, histograms = run(dataset)
    report(n_events, stages, events_at, tracks_in, histograms)
    if args.plot:
        draw(histograms, args.plot)


def report(n_events, stages, events_at, tracks_in, histograms):
    width = max(len(s) for s in stages)
    print(f"\n{n_events} events read\n")
    print("per-event cut flow (events with at least one surviving downstream e- track)\n")
    print(f"  {'stage':<{width}}  {'events':>7}  {'rel':>7}  {'abs':>7}")
    print(f"  {'-' * width}  {'-' * 7}  {'-' * 7}  {'-' * 7}")
    previous = None
    for stage, count in zip(stages, events_at):
        rel = "-" if previous is None else f"{100 * count / previous:6.1f}%" if previous else "  n/a "
        absolute = f"{100 * count / n_events:6.1f}%" if n_events else "  n/a "
        print(f"  {stage:<{width}}  {count:>7}  {rel:>7}  {absolute:>7}")
        previous = count

    print(f"\ntracks per set: " + ", ".join(f"{s}={tracks_in[s]}" for s in SETS))
    print("\nhistogram summary (mean +- rms of filled entries)\n")
    print(f"  {'observable':<30}" + "".join(f"{s:>26}" for s in SETS))
    for name, axis, _, _, _, _ in OBSERVABLES:
        row = f"  {axis:<30}"
        for label in SETS:
            h = histograms[(name, label)]
            row += f"{summarize(h):>26}"
        print(row)


def summarize(histogram) -> str:
    import numpy as np

    counts = histogram.values()
    centers = histogram.axes[0].centers
    total = counts.sum()
    if total == 0:
        return "empty"
    mean = float((counts * centers).sum() / total)
    rms = float(np.sqrt((counts * (centers - mean) ** 2).sum() / total))
    return f"n={int(total)} {mean:.3g} +- {rms:.3g}"


def draw(histograms, path):
    try:
        import matplotlib
        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
    except ImportError:
        print(f"matplotlib not available; skipping {path}")
        return

    fig, axes = plt.subplots(1, len(OBSERVABLES), figsize=(4 * len(OBSERVABLES), 3.4))
    for ax, (name, axis_label, _, _, _, _) in zip(axes, OBSERVABLES):
        for label in SETS:
            h = histograms[(name, label)]
            ax.step(h.axes[0].centers, h.values(), where="mid", label=label)
        ax.set_xlabel(axis_label)
        ax.set_ylabel("tracks")
        ax.set_yscale("log")
        ax.legend(fontsize="small")
    fig.tight_layout()
    fig.savefig(path, dpi=110)
    print(f"\nwrote {path}")


if __name__ == "__main__":
    main()
