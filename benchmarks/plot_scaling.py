#!/usr/bin/env python3
"""Plot the parallel-scaling measurements recorded in scaling_results.json.

Reads recorded numbers only -- it never runs a benchmark, so the figure can be
regenerated without touching the cluster.

    python3 benchmarks/plot_scaling.py [-o scaling.png]
"""

import argparse
import json
import os

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

HERE = os.path.dirname(os.path.abspath(__file__))

# Categorical slots 1-3 of the validated default palette; assigned in fixed order,
# never cycled. Reference lines are neutral gray, not a categorical hue.
BLUE, ORANGE, AQUA = "#2a78d6", "#eb6834", "#1baf7a"
INK, INK_2, GRID, REF = "#0b0b0b", "#52514e", "#dcdcd8", "#9a9a94"

STAGES = [("before fixes", BLUE), ("after batch-size fix", ORANGE),
          ("after partitioning fix", AQUA)]
FINAL = [("after partitioning fix", AQUA, "pyevtana (MemmapSource)"),
         ("uproot default handler", BLUE, "pyevtana (uproot default)"),
         ("pyfitter / pyutils", ORANGE, "pyfitter / pyutils")]


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


def line(ax, x, y, color, label, dashed=False):
    ax.plot(x, y, color=color, linewidth=2, marker="o", markersize=7,
            linestyle="--" if dashed else "-",
            markeredgecolor="white", markeredgewidth=1.4, label=label, zorder=3)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("-o", "--out", default=os.path.join(HERE, "scaling.png"))
    args = parser.parse_args()

    data = json.load(open(os.path.join(HERE, "scaling_results.json")))
    series = data["series"]
    mb_per_file, fpw = data["mb_per_file"], data["files_per_worker"]
    ticks = [1, 5, 10, 20]

    fig, axes = plt.subplots(1, 3, figsize=(15.5, 4.9))
    fig.patch.set_facecolor("#fcfcfb")
    for ax in axes:
        ax.set_facecolor("#fcfcfb")

    # -- 1. throughput, and how the two fixes moved it -----------------------------------
    ax = axes[0]
    base = series["after partitioning fix"]["events_per_s"][0]
    ax.plot(ticks, [base * w for w in ticks], color=REF, linewidth=1.5, linestyle="--",
            zorder=2, label="linear from fixed 1-worker rate")
    for name, color in STAGES:
        s = series[name]
        line(ax, s["workers"], s["events_per_s"], color, name)
        ax.annotate(f"{s['events_per_s'][-1]:,.0f}", (s["workers"][-1], s["events_per_s"][-1]),
                    textcoords="offset points", xytext=(8, -3), fontsize=8.5, color=color,
                    fontweight="medium")
    ax.set_xscale("log"); ax.set_yscale("log")
    ax.set_xticks(ticks); ax.set_xticklabels(ticks)
    yt = [2000, 5000, 10000, 20000, 50000]
    ax.set_yticks(yt)
    ax.set_yticklabels([f"{v // 1000}k" for v in yt])
    ax.minorticks_off()
    style(ax, "worker processes", "events / s", "Throughput (read-only)")
    ax.legend(frameon=False, fontsize=8.5, labelcolor=INK_2, loc="upper left")

    # -- 2. are the workers actually working? ---------------------------------------------
    ax = axes[1]
    ax.plot(ticks, ticks, color=REF, linewidth=1.5, linestyle="--", zorder=2,
            label="ideal (all workers busy)")
    for name, color in STAGES:
        s = series[name]
        line(ax, s["workers"], s["cpu_over_wall"], color, name)
        ax.annotate(f"{s['cpu_over_wall'][-1]:.1f}", (s["workers"][-1], s["cpu_over_wall"][-1]),
                    textcoords="offset points", xytext=(8, -3), fontsize=8.5, color=color,
                    fontweight="medium")
    ax.set_xticks(ticks)
    style(ax, "worker processes", "CPU seconds per wall second",
          "Worker utilisation — the diagnostic")
    ax.legend(frameon=False, fontsize=8.5, labelcolor=INK_2, loc="upper left")

    # -- 3. aggregate read bandwidth, comparable across workloads --------------------------
    ax = axes[2]
    for name, color, label in FINAL:
        s = series[name]
        mb = [w * fpw * mb_per_file / wall for w, wall in zip(s["workers"], s["wall"])]
        partial = len(s["workers"]) < len(ticks)
        line(ax, s["workers"], mb, color,
             label + (" (2 points measured)" if partial else ""), dashed=partial)
        ax.annotate(f"{mb[-1]:.0f}", (s["workers"][-1], mb[-1]), textcoords="offset points",
                    xytext=(8, -3), fontsize=8.5, color=color, fontweight="medium")
    ax.set_xticks(ticks)
    style(ax, "worker processes", "MB / s (compressed, aggregate)",
          "Read bandwidth — same bytes, either framework")
    ax.legend(frameon=False, fontsize=8.5, labelcolor=INK_2, loc="upper left")

    fig.suptitle("pyevtana parallel scaling: a serial prologue, not an I/O ceiling",
                 fontsize=13, color=INK, x=0.006, ha="left", y=0.99, fontweight="semibold")
    fig.text(0.006, 0.015,
             f"{fpw} files (~24,950 events) per worker, warm page cache, 48-core host.  "
             "pyevtana series are read-and-normalize only; pyfitter runs its full selection, so "
             "it is compared on bandwidth, where the two are equivalent (identical files and "
             "branch list).\n"
             "Cause: the partitioner opened every file in the parent process — ~0.4 s of schema "
             "discovery each, serially — so the serial fraction grew with the worker count and "
             "throughput flattened. Whole-file partitions no longer open anything.",
             fontsize=8, color=INK_2, ha="left", va="bottom", wrap=True)
    fig.tight_layout(rect=[0, 0.075, 1, 0.945])
    fig.savefig(args.out, dpi=130, facecolor=fig.get_facecolor())
    print(f"wrote {args.out}")


if __name__ == "__main__":
    main()
