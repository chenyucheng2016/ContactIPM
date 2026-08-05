#!/usr/bin/env python3
"""Render publication-ready tracking-effort Pareto curves."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.lines import Line2D


SOLVERS = {
    "ContactIPM": {"color": "#0072B2"},
    "CRISP": {"color": "#D55E00"},
}


def render(summary: dict, output_prefix: Path) -> None:
    problems = summary["problems"]
    figure, axes = plt.subplots(
        1, len(problems), figsize=(4.0 * len(problems), 3.25), squeeze=False
    )
    for axis, problem in zip(axes[0], problems):
        problem_points = [
            point for point in summary["points"] if point["problem"] == problem
        ]
        for solver, style in SOLVERS.items():
            points = [point for point in problem_points if point["solver"] == solver]
            front = sorted(
                (point for point in points if point["pareto_efficient"]),
                key=lambda point: point["effort_cost"],
            )
            if front:
                axis.plot(
                    [point["effort_cost"] for point in front],
                    [point["tracking_cost"] for point in front],
                    color=style["color"],
                    linewidth=1.4,
                    zorder=1,
                )
            for point in points:
                valid = point["solver_converged"] and point["successful"]
                quality_only = point["successful"] and not point["solver_converged"]
                if valid:
                    marker = "o"
                    facecolor = style["color"]
                elif quality_only:
                    marker = "o"
                    facecolor = "none"
                else:
                    marker = "x"
                    facecolor = style["color"]
                axis.scatter(
                    point["effort_cost"],
                    point["tracking_cost"],
                    marker=marker,
                    s=35,
                    color=style["color"],
                    facecolors=facecolor,
                    linewidths=1.2,
                    zorder=2,
                )
                if point["effort_scale"] in (
                    min(summary["effort_scales"]),
                    1.0,
                    max(summary["effort_scales"]),
                ):
                    axis.annotate(
                        f"{point['effort_scale']:g}",
                        (point["effort_cost"], point["tracking_cost"]),
                        xytext=(3, 3),
                        textcoords="offset points",
                        fontsize=6.5,
                    )
        label = "Push Box" if problem == "push_box" else (
            f"Push T segment {summary['push_t_segment']}"
        )
        axis.set_title(label)
        axis.set_xscale("log")
        axis.set_yscale("log")
        axis.grid(True, which="both", linewidth=0.35, alpha=0.45)
        axis.set_xlabel("Unscaled force-effort cost")
    axes[0][0].set_ylabel("Unscaled tracking cost")
    legend = [
        Line2D([0], [0], color=style["color"], linewidth=1.4, label=solver)
        for solver, style in SOLVERS.items()
    ]
    legend.extend(
        [
            Line2D([0], [0], color="#555555", marker="o", linestyle="none",
                   label="Converged + audit pass"),
            Line2D([0], [0], color="#555555", marker="o", markerfacecolor="none",
                   linestyle="none", label="Audit pass, solver exit failed"),
            Line2D([0], [0], color="#555555", marker="x", linestyle="none",
                   label="Audit failed"),
        ]
    )
    figure.legend(
        handles=legend,
        loc="outside lower center",
        ncol=3,
        frameon=False,
        fontsize=7,
    )
    figure.tight_layout(rect=(0.0, 0.13, 1.0, 1.0))
    output_prefix.parent.mkdir(parents=True, exist_ok=True)
    figure.savefig(output_prefix.with_suffix(".pdf"), bbox_inches="tight")
    figure.savefig(output_prefix.with_suffix(".png"), dpi=240, bbox_inches="tight")
    plt.close(figure)


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("summary", type=Path)
    parser.add_argument("--output-prefix", type=Path)
    args = parser.parse_args()
    summary = json.loads(args.summary.read_text(encoding="utf-8"))
    output_prefix = args.output_prefix or args.summary.with_name("pareto")
    render(summary, output_prefix)
    print(f"Pareto figure: {output_prefix.with_suffix('.pdf').resolve()}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
