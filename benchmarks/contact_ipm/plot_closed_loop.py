#!/usr/bin/env python3
"""Plot representative trajectories from the Push Box closed-loop suite."""

from __future__ import annotations

import argparse
import csv
import json
import math
from pathlib import Path
from typing import Any

import matplotlib.pyplot as plt
from matplotlib.colors import to_rgba
from matplotlib.patches import Polygon


REPRESENTATIVE_GROUPS = ("nominal", "combined")
PATH_DRAW_ORDER = ("combined", "nominal")
BOX_HALF_LENGTH = 0.5
BOX_HALF_WIDTH = 0.25
COLORS = {
    "nominal": "#0072B2",
    "combined": "#009E73",
}
LABELS = {
    "nominal": "nominal",
    "combined": "combined disturbance",
}
LINE_STYLES = {
    "nominal": "--",
    "combined": "-.",
}


def draw_box_pose(
    axis: Any,
    row: dict[str, float],
    color: str,
    linestyle: str,
    zorder: int,
    fill_alpha: float = 0.12,
) -> None:
    x = row["x"]
    y = row["y"]
    theta = row["theta"]
    cosine = math.cos(theta)
    sine = math.sin(theta)
    corners = [
        (-BOX_HALF_LENGTH, -BOX_HALF_WIDTH),
        (BOX_HALF_LENGTH, -BOX_HALF_WIDTH),
        (BOX_HALF_LENGTH, BOX_HALF_WIDTH),
        (-BOX_HALF_LENGTH, BOX_HALF_WIDTH),
    ]
    world_corners = [
        (x + cosine * px - sine * py, y + sine * px + cosine * py)
        for px, py in corners
    ]
    axis.add_patch(
        Polygon(
            world_corners,
            closed=True,
            facecolor=to_rgba(color, fill_alpha),
            edgecolor=color,
            linestyle=linestyle,
            linewidth=1.0,
            zorder=zorder,
        )
    )
    axis.plot(
        [x, x + BOX_HALF_LENGTH * cosine],
        [y, y + BOX_HALF_LENGTH * sine],
        color=color,
        linestyle=linestyle,
        linewidth=1.0,
        zorder=zorder + 0.1,
    )


def halfway_path_index(rows: list[dict[str, float]]) -> int:
    cumulative = [0.0]
    for previous, current in zip(rows, rows[1:]):
        cumulative.append(
            cumulative[-1]
            + math.hypot(current["x"] - previous["x"], current["y"] - previous["y"])
        )
    halfway = 0.5 * cumulative[-1]
    return min(range(len(rows)), key=lambda index: abs(cumulative[index] - halfway))


def read_trajectory(path: Path) -> list[dict[str, float]]:
    with path.open(newline="", encoding="utf-8") as stream:
        return [
            {key: float(value) for key, value in row.items()}
            for row in csv.DictReader(stream)
        ]


def representative_rollouts(payload: dict[str, Any]) -> list[dict[str, Any]]:
    representatives = []
    for group in REPRESENTATIVE_GROUPS:
        representatives.append(
            next(rollout for rollout in payload["rollouts"]
                 if rollout["group"] == group)
        )
    return representatives


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("input", type=Path)
    parser.add_argument("trajectory_dir", type=Path)
    parser.add_argument("--output-prefix", type=Path, required=True)
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    payload = json.loads(args.input.read_text(encoding="utf-8"))
    config = payload["configuration"]
    target_x, target_y, target_angle = config["target"]
    representatives = representative_rollouts(payload)

    trajectories: dict[str, list[dict[str, float]]] = {}
    for rollout in representatives:
        filename = (
            f"rollout_{int(rollout['index']):02d}_{rollout['group']}.csv"
        )
        trajectories[rollout["group"]] = read_trajectory(
            args.trajectory_dir / filename
        )

    plt.rcParams.update(
        {
            "font.size": 9,
            "axes.grid": True,
            "grid.alpha": 0.25,
            "figure.dpi": 140,
        }
    )
    figure, axes = plt.subplots(2, 2, figsize=(8.0, 6.2))

    path_axis = axes[0, 0]
    for group in PATH_DRAW_ORDER:
        rows = trajectories[group]
        path_axis.plot(
            [row["x"] for row in rows],
            [row["y"] for row in rows],
            color=COLORS[group],
            linestyle=LINE_STYLES[group],
            linewidth=2.4 if group == "nominal" else 1.8,
            zorder=4 if group == "nominal" else 2,
            label=LABELS[group],
        )
        for index in (0, halfway_path_index(rows)):
            draw_box_pose(
                path_axis,
                rows[index],
                COLORS[group],
                LINE_STYLES[group],
                3 if group == "nominal" else 1,
            )
    draw_box_pose(
        path_axis,
        {"x": target_x, "y": target_y, "theta": target_angle},
        "black",
        "--",
        1,
        0.0,
    )
    nominal_start = trajectories["nominal"][0]
    path_axis.scatter(
        [nominal_start["x"]],
        [nominal_start["y"]],
        marker="o",
        facecolors="white",
        edgecolors="black",
        s=28,
        zorder=6,
        label="start",
    )
    path_axis.scatter(
        [target_x], [target_y], marker="*", color="black", s=90, label="target"
    )

    path_axis.set_xlabel("box x [m]")
    path_axis.set_ylabel("box y [m]")
    path_axis.set_title("(a) Closed-loop paths")
    path_axis.axis("equal")
    path_axis.legend(frameon=False, fontsize=7, loc="upper left")

    error_axis = axes[0, 1]
    for group in PATH_DRAW_ORDER:
        rows = trajectories[group]
        times = [row["time"] for row in rows]
        translation = [
            math.hypot(row["x"] - target_x, row["y"] - target_y)
            for row in rows
        ]
        angular = [abs(row["theta"] - target_angle) for row in rows]
        error_axis.plot(
            times,
            translation,
            color=COLORS[group],
            linestyle=LINE_STYLES[group],
            linewidth=2.0 if group == "nominal" else 1.6,
            zorder=4 if group == "nominal" else 2,
            label=f"{LABELS[group]} distance",
        )
        error_axis.plot(
            times,
            angular,
            color=COLORS[group],
            linewidth=1.2,
            linestyle=":",
            zorder=4 if group == "nominal" else 2,
            label=f"{LABELS[group]} angle",
        )
    error_axis.axhline(
        config["translation_tolerance"],
        color="0.35",
        linestyle=":",
        linewidth=1,
        label="goal tolerance",
    )
    error_axis.set_xlabel("closed-loop time [s]")
    error_axis.set_ylabel("target error [m or rad]")
    error_axis.set_title("(b) Distance and angle to target")
    error_axis.legend(frameon=False, fontsize=6.5, ncol=2)

    timing_axis = axes[1, 0]
    all_times = sorted(
        1000.0 * value
        for rollout in payload["rollouts"]
        for value in rollout["solve_times_s"]
    )
    warm_times = sorted(
        1000.0 * value
        for rollout in payload["rollouts"]
        for value in rollout["solve_times_s"][1:]
    )
    for label, values, color in (
        ("all solves", all_times, "#7A5195"),
        ("shifted warm starts", warm_times, "#EF5675"),
    ):
        timing_axis.plot(
            values,
            [100.0 * (index + 1) / len(values)
             for index in range(len(values))],
            color=color,
            linewidth=1.8,
            label=label,
        )
    timing_axis.axvline(
        1000.0 * config["control_dt"],
        color="black",
        linestyle="--",
        linewidth=1.2,
        label="control deadline",
    )
    timing_axis.set_xscale("log")
    timing_axis.set_xlabel("solve time [ms]")
    timing_axis.set_ylabel("empirical CDF [%]")
    timing_axis.set_title("(c) ContactIPM latency distribution")
    timing_axis.legend(frameon=False, fontsize=8)

    contact_axis = axes[1, 1]
    for group in PATH_DRAW_ORDER:
        rows = trajectories[group]
        contact_axis.plot(
            [row["time"] for row in rows],
            [max(row["physical_mpcc"], 1e-16) for row in rows],
            color=COLORS[group],
            linestyle=LINE_STYLES[group],
            linewidth=2.0 if group == "nominal" else 1.5,
            zorder=4 if group == "nominal" else 2,
            label=LABELS[group],
        )
    contact_axis.axhline(
        config["solver_mpcc_tolerance"],
        color="black",
        linestyle="--",
        linewidth=1.2,
        label="physical tolerance",
    )
    contact_axis.set_yscale("log")
    contact_axis.set_xlabel("time [s]")
    contact_axis.set_ylabel("max physical product")
    contact_axis.set_title("(d) Applied complementarity")
    contact_axis.legend(frameon=False, fontsize=8)

    figure.tight_layout()
    args.output_prefix.parent.mkdir(parents=True, exist_ok=True)
    png_path = args.output_prefix.with_suffix(".png")
    pdf_path = args.output_prefix.with_suffix(".pdf")
    figure.savefig(png_path, bbox_inches="tight")
    figure.savefig(pdf_path, bbox_inches="tight")
    print(png_path)
    print(pdf_path)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
