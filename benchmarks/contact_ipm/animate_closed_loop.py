#!/usr/bin/env python3
"""Render compact animations for representative closed-loop Push Box runs."""

from __future__ import annotations

import argparse
import csv
import json
import math
from pathlib import Path
from typing import Any

import matplotlib.pyplot as plt
from matplotlib.animation import FuncAnimation, PillowWriter


GROUPS = ("nominal", "combined")
COLORS = {
    "nominal": "#0072B2",
    "combined": "#009E73",
}
DISPLAY_LABELS = {
    "nominal": "nominal",
    "combined": "disturbed motion",
}
OUTPUT_STEMS = {
    "nominal": "nominal",
    "combined": "disturbed_motion",
}
HALF_LENGTH = 0.5
HALF_WIDTH = 0.25


def read_trajectory(path: Path) -> list[dict[str, float]]:
    with path.open(newline="", encoding="utf-8") as stream:
        return [
            {key: float(value) for key, value in row.items()}
            for row in csv.DictReader(stream)
        ]


def box_polygon(x: float, y: float, theta: float) -> tuple[list[float], list[float]]:
    corners = [
        (-HALF_LENGTH, -HALF_WIDTH),
        (HALF_LENGTH, -HALF_WIDTH),
        (HALF_LENGTH, HALF_WIDTH),
        (-HALF_LENGTH, HALF_WIDTH),
        (-HALF_LENGTH, -HALF_WIDTH),
    ]
    cosine = math.cos(theta)
    sine = math.sin(theta)
    return (
        [x + cosine * px - sine * py for px, py in corners],
        [y + sine * px + cosine * py for px, py in corners],
    )


def representative(payload: dict[str, Any], group: str) -> dict[str, Any]:
    return next(item for item in payload["rollouts"] if item["group"] == group)


def render(
    rows: list[dict[str, float]],
    group: str,
    target: list[float],
    output: Path,
) -> None:
    color = COLORS[group]
    figure, axis = plt.subplots(figsize=(5.0, 4.2), dpi=110)
    target_x, target_y = box_polygon(*target)
    axis.plot(target_x, target_y, color="black", linestyle="--", linewidth=1.5)
    axis.scatter([target[0]], [target[1]], marker="*", color="black", s=80)

    xs = [row["x"] for row in rows] + [target[0]]
    ys = [row["y"] for row in rows] + [target[1]]
    padding = 0.75
    axis.set_xlim(min(xs) - padding, max(xs) + padding)
    axis.set_ylim(min(ys) - padding, max(ys) + padding)
    axis.set_aspect("equal")
    axis.grid(alpha=0.25)
    axis.set_xlabel("box x [m]")
    axis.set_ylabel("box y [m]")

    path_line, = axis.plot([], [], color=color, linewidth=1.8)
    box_line, = axis.plot([], [], color=color, linewidth=2.5)
    contact_point, = axis.plot([], [], marker="o", color=color, markersize=5)
    force_line, = axis.plot([], [], color=color, linewidth=2.0)

    def update(frame: int) -> tuple[Any, ...]:
        row = rows[frame]
        path_line.set_data(
            [sample["x"] for sample in rows[: frame + 1]],
            [sample["y"] for sample in rows[: frame + 1]],
        )
        polygon_x, polygon_y = box_polygon(row["x"], row["y"], row["theta"])
        box_line.set_data(polygon_x, polygon_y)

        cosine = math.cos(row["theta"])
        sine = math.sin(row["theta"])
        contact_x = (
            row["x"] + cosine * row["contact_x"] - sine * row["contact_y"]
        )
        contact_y = (
            row["y"] + sine * row["contact_x"] + cosine * row["contact_y"]
        )
        contact_point.set_data([contact_x], [contact_y])

        body_force_x = row["lambda1"] + row["lambda3"]
        body_force_y = row["lambda2"] + row["lambda4"]
        world_force_x = cosine * body_force_x - sine * body_force_y
        world_force_y = sine * body_force_x + cosine * body_force_y
        force_norm = math.hypot(world_force_x, world_force_y)
        force_scale = 0.45 / max(force_norm, 1e-12)
        force_line.set_data(
            [contact_x, contact_x + force_scale * world_force_x],
            [contact_y, contact_y + force_scale * world_force_y],
        )
        axis.set_title(
            f"{DISPLAY_LABELS[group]} | t={row['time']:.1f} s | "
            f"solve={1000.0 * row['solve_time_s']:.1f} ms\n"
            f"physical product={row['physical_mpcc']:.2e}"
        )
        return path_line, box_line, contact_point, force_line

    animation = FuncAnimation(
        figure, update, frames=len(rows), interval=150, blit=False
    )
    output.parent.mkdir(parents=True, exist_ok=True)
    animation.save(output, writer=PillowWriter(fps=7))
    plt.close(figure)
    print(output)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("input", type=Path)
    parser.add_argument("trajectory_dir", type=Path)
    parser.add_argument("--output-dir", type=Path, required=True)
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    payload = json.loads(args.input.read_text(encoding="utf-8"))
    target = list(payload["configuration"]["target"])
    for group in GROUPS:
        rollout = representative(payload, group)
        path = args.trajectory_dir / (
            f"rollout_{int(rollout['index']):02d}_{group}.csv"
        )
        render(
            read_trajectory(path),
            group,
            target,
            args.output_dir / f"closed_loop_{OUTPUT_STEMS[group]}.gif",
        )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
