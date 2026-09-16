#!/usr/bin/env python3
"""Plot the three-way framework comparison from a run_scaling.sh JSONL.

    python3 benchmarks/plot_frameworks.py results.jsonl -o frameworks.png

Reads recorded results only; it never runs a benchmark.
"""

import argparse
import json
import os
from collections import defaultdict

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

HERE = os.path.dirname(os.path.abspath(__file__))

# Categorical slots 1-3 of the validated default palette, fixed order, never cycled.
BLUE, ORANGE, AQUA = "#2a78d6", "#eb6834", "#1baf7a"
INK, INK_2, GRID, REF = "#0b0b0b", "#52514e", "#dcdcd8", "#9a9a94"

SERIES = [
    ("pyevtana/read", AQUA, "pyevtana read-only (floor)"),
    ("pyfitter/arrays", ORANGE, "pyfitter / pyutils (vectorized)"),
    ("pyevtana/objects", BLUE, "pyevtana object loop"),
]


def style(ax, xlabel, ylabel, title):
    ax.set_title(title, fontsize=11, color=INK, pad=10, loc="left", fontweight="medium")
    ax.set_xlabel(xlabel, fontsize=9.5, color=INK_2)
    ax.set_ylabel(ylabel, fontsize=9.5, color=INK_2)
    ax.grid(True, which="major", color=GRID, linewidth=0.7, zorder=0)
    ax.set_axisbelow(True)
    for side in ("top", "right"):
        ax.spines[side].set_visible(False)
    for side in ("left", "bottom"):
        ax.spines[side].set_color(GRID)
    ax.tick_params(colors=INK_2, labelsize=9, length=3)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("results")
    parser.add_argument("-o", "--out", default=os.path.join(HERE, "frameworks.png"))
    args = parser.parse_args()

    rows = [json.loads(l) for l in open(args.results) if l.strip()]
    by = defaultdict(dict)
    for r in rows:
        by[r["framework"] + ("/" + r["mode"] if r.get("mode") else "")][r["jobs"]] = r
    ticks = sorted({r["jobs"] for r in rows})

    fig, axes = plt.subplots(1, 3, figsize=(15.5, 4.9))
    fig.patch.set_facecolor("#fcfcfb")
    for ax in axes:
        ax.set_facecolor("#fcfcfb")

    # -- throughput ------------------------------------------------------------------------
    ax = axes[0]
    anchor = by["pyfitter/arrays"][ticks[0]]["rate"]
    ax.plot(ticks, [anchor * w for w in ticks], color=REF, linewidth=1.5, linestyle="--",
            zorder=2, label="linear from pyfitter 1-worker rate")
    for key, color, label in SERIES:
        pts = sorted(by[key])
        y = [by[key][j]["rate"] for j in pts]
        ax.plot(pts, y, color=color, linewidth=2, marker="o", markersize=7,
                markeredgecolor="white", markeredgewidth=1.4, label=label, zorder=3)
        ax.annotate(f"{y[-1]:,.0f}", (pts[-1], y[-1]), textcoords="offset points",
                    xytext=(8, -3), fontsize=8.5, color=color, fontweight="medium")
    ax.set_xscale("log"); ax.set_yscale("log")
    ax.set_xticks(ticks); ax.set_xticklabels(ticks); ax.minorticks_off()
    yt = [2000, 5000, 10000, 20000, 50000]
    ax.set_yticks(yt); ax.set_yticklabels([f"{v // 1000}k" for v in yt])
    style(ax, "worker processes", "events / s", "Throughput")
    ax.legend(frameon=False, fontsize=8.5, labelcolor=INK_2, loc="upper left")

    # -- speed-up --------------------------------------------------------------------------
    ax = axes[1]
    ax.plot(ticks, ticks, color=REF, linewidth=1.5, linestyle="--", zorder=2, label="ideal")
    for i, (key, color, label) in enumerate(SERIES):
        pts = sorted(by[key])
        base = by[key][pts[0]]["rate"]
        y = [by[key][j]["rate"] / base for j in pts]
        ax.plot(pts, y, color=color, linewidth=2, marker="o", markersize=7,
                markeredgecolor="white", markeredgewidth=1.4, label=label, zorder=3)
        # read-only and the object loop land within 0.2x of each other; stagger the labels
        ax.annotate(f"{y[-1]:.1f}x", (pts[-1], y[-1]), textcoords="offset points",
                    xytext=(8, 5 if i == 0 else (-9 if i == 2 else -3)),
                    fontsize=8.5, color=color, fontweight="medium")
    ax.set_xticks(ticks)
    style(ax, "worker processes", "speed-up vs 1 worker", "Parallel speed-up")
    ax.legend(frameon=False, fontsize=8.5, labelcolor=INK_2, loc="upper left")

    # -- utilisation -----------------------------------------------------------------------
    ax = axes[2]
    ax.plot(ticks, ticks, color=REF, linewidth=1.5, linestyle="--", zorder=2,
            label="ideal (all workers busy)")
    for key, color, label in SERIES:
        pts = sorted(by[key])
        y = [by[key][j]["cpu"] / by[key][j]["wall"] for j in pts]
        ax.plot(pts, y, color=color, linewidth=2, marker="o", markersize=7,
                markeredgecolor="white", markeredgewidth=1.4, label=label, zorder=3)
        ax.annotate(f"{y[-1]:.1f}", (pts[-1], y[-1]), textcoords="offset points",
                    xytext=(8, -3), fontsize=8.5, color=color, fontweight="medium")
    ax.set_xticks(ticks)
    style(ax, "worker processes", "CPU seconds per wall second", "Worker utilisation")
    ax.legend(frameon=False, fontsize=8.5, labelcolor=INK_2, loc="upper left")

    nfiles = by["pyfitter/arrays"][ticks[0]]["nfiles"]
    fig.suptitle("Same selection, same branches: pyfitter vs pyevtana",
                 fontsize=13, color=INK, x=0.006, ha="left", y=0.99, fontweight="semibold")
    footnote = "\n".join([
        f"{nfiles} files (~25,000 events) per worker, warm page cache, 48-core host. "
        "pyfitter and the pyevtana object loop run the identical 20-cut selection \u2014 every stage of the",
        "per-event cut flow agrees exactly (validate_against_pyfitter.py) \u2014 and read the identical "
        "branch list. pyevtana read-only performs no analysis and is the floor both share.",
        "Speed-up is relative to each series' own 1-worker rate, so a faster series can show a "
        "smaller factor while still finishing first.",
    ])
    fig.text(0.006, 0.012, footnote, fontsize=8, color=INK_2, ha="left", va="bottom")
    fig.tight_layout(rect=[0, 0.10, 1, 0.945])
    fig.savefig(args.out, dpi=130, facecolor=fig.get_facecolor())
    print(f"wrote {args.out}")


if __name__ == "__main__":
    main()
