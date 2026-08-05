#!/usr/bin/env python3
"""Render matched MPCC scaling time and audit-success curves."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.lines import Line2D


SOLVERS = {
    "ContactIPM": {"color": "#0072B2", "marker": "o"},
    "CRISP": {"color": "#D55E00", "marker": "s"},
}


def render(summaries: list[dict], output_prefix: Path) -> None:
    summaries = sorted(summaries, key=lambda item: item["problem"])
    figure, axes = plt.subplots(
        2,
        len(summaries),
        figsize=(4.1 * len(summaries), 5.5),
        sharex="col",
        gridspec_kw={"height_ratios": (1.35, 1.0)},
        squeeze=False,
    )
    for column, summary in enumerate(summaries):
        time_axis = axes[0][column]
        success_axis = axes[1][column]
        points = sorted(summary["points"], key=lambda point: point["nodes"])
        for solver, style in SOLVERS.items():
            timed = [
                point
                for point in points
                if point["solvers"][solver]["eligible"] > 0
            ]
            time_axis.plot(
                [point["total_complementarity_pairs"] for point in timed],
                [
                    point["solvers"][solver][
                        "median_eligible_wall_time_seconds"
                    ]
                    for point in timed
                ],
                color=style["color"],
                marker=style["marker"],
                linewidth=1.4,
                markersize=4.5,
            )
            success_axis.plot(
                [point["total_complementarity_pairs"] for point in points],
                [
                    point["solvers"][solver]["eligible"]
                    / point["solvers"][solver]["attempted"]
                    for point in points
                ],
                color=style["color"],
                marker=style["marker"],
                linewidth=1.4,
                markersize=4.5,
            )
        for point in points:
            speedup = point["median_paired_speedup_crisp_over_contactipm"]
            if speedup is None:
                continue
            contact_time = point["solvers"]["ContactIPM"][
                "median_eligible_wall_time_seconds"
            ]
            crisp_time = point["solvers"]["CRISP"][
                "median_eligible_wall_time_seconds"
            ]
            time_axis.annotate(
                f"{speedup:.2g}x",
                (
                    point["total_complementarity_pairs"],
                    max(contact_time, crisp_time),
                ),
                xytext=(0, 5),
                textcoords="offset points",
                ha="center",
                fontsize=6.5,
            )
        label = (
            "Push Box"
            if summary["problem"] == "push_box"
            else f"Push T segment {summary['push_t_segment']}"
        )
        time_axis.set_title(label)
        time_axis.set_yscale("log")
        time_axis.set_ylabel("Eligible median wall time (s)")
        time_axis.grid(True, which="both", linewidth=0.35, alpha=0.45)
        success_axis.set_ylim(-0.05, 1.05)
        success_axis.set_yticks((0.0, 0.5, 1.0))
        success_axis.set_ylabel("Converged + audited fraction")
        success_axis.set_xlabel("Total complementarity pairs")
        success_axis.grid(True, linewidth=0.35, alpha=0.45)
    legend = [
        Line2D(
            [0],
            [0],
            color=style["color"],
            marker=style["marker"],
            linewidth=1.4,
            label=solver,
        )
        for solver, style in SOLVERS.items()
    ]
    figure.legend(
        handles=legend,
        loc="outside lower center",
        ncol=2,
        frameon=False,
        fontsize=8,
    )
    figure.tight_layout(rect=(0.0, 0.07, 1.0, 1.0))
    output_prefix.parent.mkdir(parents=True, exist_ok=True)
    figure.savefig(output_prefix.with_suffix(".pdf"), bbox_inches="tight")
    figure.savefig(output_prefix.with_suffix(".png"), dpi=240, bbox_inches="tight")
    plt.close(figure)


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("summaries", nargs="+", type=Path)
    parser.add_argument("--output-prefix", type=Path, required=True)
    args = parser.parse_args()
    summaries = [
        json.loads(path.read_text(encoding="utf-8")) for path in args.summaries
    ]
    problems = {summary["problem"] for summary in summaries}
    if problems != {"push_box", "push_t"}:
        parser.error("provide exactly one Push Box and one Push T summary")
    render(summaries, args.output_prefix)
    print(f"Scaling figure: {args.output_prefix.with_suffix('.pdf').resolve()}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
