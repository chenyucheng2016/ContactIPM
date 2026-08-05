#!/usr/bin/env python3
"""Render an anonymous ICRA-ready Push Box combined-disturbance clip."""

from __future__ import annotations

import argparse
import bisect
import csv
import json
import math
from pathlib import Path
from typing import Any

import imageio_ffmpeg
import matplotlib

matplotlib.use("Agg")

import matplotlib.pyplot as plt
from matplotlib.animation import FFMpegWriter
from matplotlib.patches import Polygon


FPS = 30
FIGURE_DPI = 120
FIGURE_SIZE = (16, 9)
PLAYBACK_RATE = 0.5
OPENING_HOLD_S = 1.0
CLOSING_HOLD_S = 1.5
BOX_HALF_LENGTH = 0.5
BOX_HALF_WIDTH = 0.25
BOX_COLOR = "#009E73"
CONTACT_COLOR = "#D55E00"
PATH_COLOR = "#0072B2"
NOMINAL_COLOR = "#666666"


def read_trajectory(path: Path) -> list[dict[str, float]]:
    with path.open(newline="", encoding="utf-8") as stream:
        return [
            {key: float(value) for key, value in row.items()}
            for row in csv.DictReader(stream)
        ]


def box_corners(x: float, y: float, theta: float) -> list[tuple[float, float]]:
    cosine = math.cos(theta)
    sine = math.sin(theta)
    return [
        (x + cosine * px - sine * py, y + sine * px + cosine * py)
        for px, py in (
            (-BOX_HALF_LENGTH, -BOX_HALF_WIDTH),
            (BOX_HALF_LENGTH, -BOX_HALF_WIDTH),
            (BOX_HALF_LENGTH, BOX_HALF_WIDTH),
            (-BOX_HALF_LENGTH, BOX_HALF_WIDTH),
        )
    ]


def force_magnitude(row: dict[str, float]) -> float:
    return math.hypot(
        row["lambda1"] + row["lambda3"],
        row["lambda2"] + row["lambda4"],
    )


def choose_rollout(
    payload: dict[str, Any], requested_index: int | None
) -> dict[str, Any]:
    combined = [
        rollout
        for rollout in payload["rollouts"]
        if rollout["group"] == "combined" and rollout["task_success"]
    ]
    if requested_index is not None:
        return next(
            rollout for rollout in combined
            if int(rollout["index"]) == requested_index
        )
    return max(
        combined,
        key=lambda rollout: math.hypot(
            float(rollout["kick"][0]), float(rollout["kick"][1])
        ),
    )


def state_at_time(
    rows: list[dict[str, float]],
    sample_times: list[float],
    time_s: float,
    reset_time: float,
    reset: list[float],
) -> tuple[float, float, float]:
    if time_s <= sample_times[0]:
        return rows[0]["x"], rows[0]["y"], rows[0]["theta"]
    if time_s >= sample_times[-1]:
        return rows[-1]["x"], rows[-1]["y"], rows[-1]["theta"]

    upper = bisect.bisect_right(sample_times, time_s)
    lower = upper - 1
    lower_time = sample_times[lower]
    upper_time = sample_times[upper]
    fraction = (time_s - lower_time) / (upper_time - lower_time)
    lower_state = [rows[lower][name] for name in ("x", "y", "theta")]
    upper_state = [rows[upper][name] for name in ("x", "y", "theta")]

    if lower_time < reset_time <= upper_time and time_s < reset_time:
        upper_state = [
            value - reset[index] for index, value in enumerate(upper_state)
        ]
    return tuple(
        lower_state[index]
        + fraction * (upper_state[index] - lower_state[index])
        for index in range(3)
    )


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("input", type=Path)
    parser.add_argument("trajectory_dir", type=Path)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--rollout-index", type=int)
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    payload = json.loads(args.input.read_text(encoding="utf-8"))
    rollout = choose_rollout(payload, args.rollout_index)
    rollout_index = int(rollout["index"])
    trajectory_path = (
        args.trajectory_dir / f"rollout_{rollout_index:02d}_combined.csv"
    )
    rows = read_trajectory(trajectory_path)
    sample_times = [row["time"] for row in rows]
    nominal_rollout = next(
        reference
        for reference in payload["rollouts"]
        if reference["group"] == "nominal"
        and int(reference["group_index"]) == 0
    )
    nominal_index = int(nominal_rollout["index"])
    nominal_path = (
        args.trajectory_dir / f"rollout_{nominal_index:02d}_nominal.csv"
    )
    nominal_rows = read_trajectory(nominal_path)
    nominal_times = [row["time"] for row in nominal_rows]
    config = payload["configuration"]
    target = [float(value) for value in config["target"]]
    reset = [float(value) for value in rollout["kick"]]
    reset_time = (int(rollout["kick_step"]) + 1) * float(config["control_dt"])

    matplotlib.rcParams["animation.ffmpeg_path"] = (
        imageio_ffmpeg.get_ffmpeg_exe()
    )
    plt.rcParams.update(
        {
            "font.size": 12,
            "axes.grid": True,
            "grid.alpha": 0.22,
            "axes.facecolor": "#FAFAFA",
            "figure.facecolor": "#F5F5F5",
        }
    )

    figure = plt.figure(figsize=FIGURE_SIZE, dpi=FIGURE_DPI)
    grid = figure.add_gridspec(
        3,
        2,
        width_ratios=(1.55, 1.0),
        height_ratios=(1.0, 0.68, 0.92),
        left=0.055,
        right=0.965,
        top=0.86,
        bottom=0.075,
        wspace=0.20,
        hspace=0.52,
    )
    workspace = figure.add_subplot(grid[:, 0])
    error_axis = figure.add_subplot(grid[0, 1])
    evidence_grid = grid[1, 1].subgridspec(1, 2, wspace=0.42)
    noise_axis = figure.add_subplot(evidence_grid[0, 0])
    control_axis = figure.add_subplot(evidence_grid[0, 1])
    status_axis = figure.add_subplot(grid[2, 1])
    status_axis.axis("off")

    figure.suptitle(
        "ContactIPM closed-loop Push Box",
        fontsize=25,
        fontweight="semibold",
        y=0.965,
    )
    figure.text(
        0.5,
        0.91,
        (
            f"Combined disturbance vs nominal reference  |  "
            f"rollout {rollout_index}  |  "
            f"10 Hz feedback  |  {PLAYBACK_RATE:.1f}x real time"
        ),
        ha="center",
        fontsize=14,
        color="#404040",
    )

    all_x = (
        [row["x"] for row in rows]
        + [row["x"] for row in nominal_rows]
        + [target[0]]
    )
    all_y = (
        [row["y"] for row in rows]
        + [row["y"] for row in nominal_rows]
        + [target[1]]
    )
    padding = 0.85
    workspace.set_xlim(min(all_x) - padding, max(all_x) + padding)
    workspace.set_ylim(min(all_y) - padding, max(all_y) + padding)
    workspace.set_aspect("equal")
    workspace.set_xlabel("box x [m]")
    workspace.set_ylabel("box y [m]")
    workspace.set_title("Executed motion and contact action", fontsize=16)

    workspace.add_patch(
        Polygon(
            box_corners(*target),
            closed=True,
            fill=False,
            edgecolor="#202020",
            linestyle="--",
            linewidth=2.0,
            label="target pose",
        )
    )
    workspace.scatter(
        [target[0]],
        [target[1]],
        marker="*",
        color="#202020",
        s=130,
        zorder=6,
    )
    workspace.scatter(
        [rows[0]["x"]],
        [rows[0]["y"]],
        facecolors="white",
        edgecolors="#202020",
        s=60,
        zorder=6,
        label="initial pose",
    )
    workspace.plot(
        [row["x"] for row in nominal_rows],
        [row["y"] for row in nominal_rows],
        color=NOMINAL_COLOR,
        linestyle=(0, (5, 3)),
        linewidth=2.1,
        label="nominal closed-loop reference",
        zorder=2,
    )
    nominal_pose_indices = sorted(
        set(range(0, len(nominal_rows), 4)) | {len(nominal_rows) - 1}
    )
    for index in nominal_pose_indices:
        nominal_row = nominal_rows[index]
        workspace.add_patch(
            Polygon(
                box_corners(
                    nominal_row["x"],
                    nominal_row["y"],
                    nominal_row["theta"],
                ),
                closed=True,
                facecolor=NOMINAL_COLOR,
                edgecolor=NOMINAL_COLOR,
                linewidth=1.0,
                alpha=0.10,
                zorder=1,
            )
        )
    trail, = workspace.plot(
        [],
        [],
        color=BOX_COLOR,
        linewidth=2.8,
        label="combined-disturbance motion",
        zorder=3,
    )
    box = Polygon(
        box_corners(rows[0]["x"], rows[0]["y"], rows[0]["theta"]),
        closed=True,
        facecolor=BOX_COLOR,
        edgecolor="#006B50",
        linewidth=2.5,
        alpha=0.78,
        zorder=5,
    )
    workspace.add_patch(box)
    heading, = workspace.plot([], [], color="#003D2E", linewidth=2.5, zorder=6)
    contact_point, = workspace.plot(
        [],
        [],
        marker="o",
        color=CONTACT_COLOR,
        markersize=8,
        linestyle="none",
        label="contact point",
        zorder=7,
    )
    force_arrow = workspace.quiver(
        [0.0],
        [0.0],
        [0.0],
        [0.0],
        angles="xy",
        scale_units="xy",
        scale=1.0,
        color=CONTACT_COLOR,
        width=0.008,
        zorder=7,
        label="contact force direction",
    )

    event_row_index = min(
        range(len(rows)),
        key=lambda index: abs(rows[index]["time"] - reset_time),
    )
    event_post = [
        rows[event_row_index]["x"],
        rows[event_row_index]["y"],
        rows[event_row_index]["theta"],
    ]
    event_pre = [
        event_post[index] - reset[index] for index in range(3)
    ]
    reset_arrow = workspace.annotate(
        "",
        xy=(event_post[0], event_post[1]),
        xytext=(event_pre[0], event_pre[1]),
        arrowprops={
            "arrowstyle": "->",
            "color": "#CC7A00",
            "linewidth": 3.0,
        },
        zorder=8,
    )
    reset_arrow.set_visible(False)
    reset_label = workspace.text(
        event_post[0] + 0.08,
        event_post[1] + 0.08,
        "state reset",
        color="#8A5200",
        fontsize=11,
        fontweight="semibold",
        visible=False,
        zorder=8,
    )
    workspace.legend(
        loc="upper left",
        frameon=True,
        framealpha=0.92,
        fontsize=10,
    )

    distances = [
        math.hypot(row["x"] - target[0], row["y"] - target[1])
        for row in rows
    ]
    angles = [abs(row["theta"] - target[2]) for row in rows]
    nominal_distances = [
        math.hypot(row["x"] - target[0], row["y"] - target[1])
        for row in nominal_rows
    ]
    nominal_angles = [
        abs(row["theta"] - target[2]) for row in nominal_rows
    ]
    error_axis.set_xlim(0.0, sample_times[-1] + 0.05)
    error_axis.set_ylim(
        0.0,
        1.08
        * max(
            max(distances),
            max(angles),
            max(nominal_distances),
            max(nominal_angles),
        ),
    )
    error_axis.set_xlabel("closed-loop time [s]")
    error_axis.set_ylabel("target error [m or rad]")
    error_axis.set_title("Feedback recovery", fontsize=16)
    error_axis.plot(
        nominal_times,
        nominal_distances,
        color=NOMINAL_COLOR,
        linestyle=(0, (5, 3)),
        linewidth=1.8,
        label="nominal position",
    )
    error_axis.plot(
        nominal_times,
        nominal_angles,
        color=NOMINAL_COLOR,
        linestyle=":",
        linewidth=1.8,
        label="nominal angle",
    )
    error_axis.axhline(
        float(config["translation_tolerance"]),
        color="#555555",
        linestyle=":",
        linewidth=1.5,
        label="goal tolerance",
    )
    distance_line, = error_axis.plot(
        [],
        [],
        color=BOX_COLOR,
        linewidth=2.5,
        label="disturbed position",
    )
    angle_line, = error_axis.plot(
        [],
        [],
        color=PATH_COLOR,
        linestyle="--",
        linewidth=2.2,
        label="disturbed angle",
    )
    distance_marker, = error_axis.plot(
        [],
        [],
        marker="o",
        color=BOX_COLOR,
        markersize=6,
        linestyle="none",
    )
    angle_marker, = error_axis.plot(
        [],
        [],
        marker="o",
        color=PATH_COLOR,
        markersize=6,
        linestyle="none",
    )
    error_axis.axvline(
        reset_time,
        color="#CC7A00",
        linestyle="-.",
        linewidth=1.5,
        label="state reset",
    )
    error_axis.legend(
        loc="upper right",
        frameon=True,
        framealpha=0.92,
        fontsize=9,
        ncol=2,
    )

    noise_position_mm = [
        1000.0
        * math.hypot(
            row["measured_x"] - row["x"],
            row["measured_y"] - row["y"],
        )
        for row in rows
    ]
    noise_angle_mrad = [
        1000.0 * abs(row["measured_theta"] - row["theta"])
        for row in rows
    ]
    noise_ceiling = max(noise_position_mm + noise_angle_mrad)
    noise_axis.set_xlim(0.0, sample_times[-1] + 0.05)
    noise_axis.set_ylim(0.0, 1.12 * max(noise_ceiling, 1e-6))
    noise_axis.set_xlabel("time [s]", fontsize=9)
    noise_axis.set_ylabel("residual [mm, mrad]", fontsize=9)
    noise_axis.set_title("Measurement noise seen by MPC", fontsize=11)
    noise_axis.tick_params(labelsize=8)
    noise_axis.axvline(
        reset_time,
        color="#CC7A00",
        linestyle="-.",
        linewidth=1.0,
    )
    noise_position_line, = noise_axis.plot(
        [],
        [],
        color=BOX_COLOR,
        linewidth=1.8,
        label="position",
    )
    noise_angle_line, = noise_axis.plot(
        [],
        [],
        color=PATH_COLOR,
        linestyle="--",
        linewidth=1.6,
        label="angle",
    )
    noise_axis.legend(
        loc="upper right",
        frameon=True,
        framealpha=0.92,
        fontsize=7.5,
    )

    combined_forces = [force_magnitude(row) for row in rows]
    nominal_forces = [force_magnitude(row) for row in nominal_rows]
    control_axis.set_xlim(0.0, sample_times[-1] + 0.05)
    control_axis.set_ylim(
        0.0,
        1.08 * max(max(combined_forces), max(nominal_forces), 1e-6),
    )
    control_axis.set_xlabel("time [s]", fontsize=9)
    control_axis.set_ylabel("force norm", fontsize=9)
    control_axis.set_title("Control response", fontsize=11)
    control_axis.tick_params(labelsize=8)
    control_axis.axvline(
        reset_time,
        color="#CC7A00",
        linestyle="-.",
        linewidth=1.0,
    )
    control_axis.plot(
        nominal_times,
        nominal_forces,
        color=NOMINAL_COLOR,
        linestyle=(0, (5, 3)),
        linewidth=1.6,
        label="nominal",
    )
    combined_force_line, = control_axis.plot(
        [],
        [],
        color=CONTACT_COLOR,
        linewidth=1.9,
        label="disturbed",
    )
    control_axis.legend(
        loc="upper right",
        frameon=True,
        framealpha=0.92,
        fontsize=7.5,
    )

    initial = [float(value) for value in rollout["initial_offset"]]
    mass_scale = float(rollout["mass_scale"])
    friction_scale = float(rollout["friction_scale"])
    disturbance_text = (
        "COMBINED DISTURBANCE\n"
        f"Initial pose   dx={initial[0]:+.3f} m, "
        f"dy={initial[1]:+.3f} m, dtheta={initial[2]:+.3f} rad\n"
        f"Model          mass x{mass_scale:.3f}, "
        f"friction x{friction_scale:.3f}\n"
        "Measurement    sigma_xy=0.001 m, sigma_theta=0.001 rad\n"
        f"State reset    dx={reset[0]:+.3f} m, "
        f"dy={reset[1]:+.3f} m, dtheta={reset[2]:+.3f} rad\n"
        f"               applied at t={reset_time:.1f} s"
    )
    status_axis.text(
        0.0,
        1.0,
        disturbance_text,
        transform=status_axis.transAxes,
        ha="left",
        va="top",
        fontsize=9.5,
        linespacing=1.4,
        family="monospace",
    )
    phase_text = status_axis.text(
        0.0,
        0.42,
        "",
        transform=status_axis.transAxes,
        ha="left",
        va="top",
        fontsize=12,
        fontweight="semibold",
        color="#006B50",
    )
    metrics_text = status_axis.text(
        0.0,
        0.31,
        "",
        transform=status_axis.transAxes,
        ha="left",
        va="top",
        fontsize=10.5,
        linespacing=1.35,
    )
    motion_duration = sample_times[-1] / PLAYBACK_RATE
    total_duration = OPENING_HOLD_S + motion_duration + CLOSING_HOLD_S
    total_frames = int(math.ceil(total_duration * FPS))

    def frame_time(frame: int) -> float:
        video_time = frame / FPS
        if video_time <= OPENING_HOLD_S:
            return 0.0
        motion_time = video_time - OPENING_HOLD_S
        return min(sample_times[-1], motion_time * PLAYBACK_RATE)

    def update(frame: int) -> None:
        time_s = frame_time(frame)
        x, y, theta = state_at_time(
            rows, sample_times, time_s, reset_time, reset
        )
        sample_index = max(
            0, min(bisect.bisect_right(sample_times, time_s) - 1, len(rows) - 1)
        )
        row = rows[sample_index]

        trail_rows = rows[: sample_index + 1]
        trail_x = [sample["x"] for sample in trail_rows]
        trail_y = [sample["y"] for sample in trail_rows]
        if not trail_x or abs(trail_x[-1] - x) + abs(trail_y[-1] - y) > 1e-12:
            trail_x.append(x)
            trail_y.append(y)
        trail.set_data(trail_x, trail_y)
        box.set_xy(box_corners(x, y, theta))
        heading.set_data(
            [x, x + BOX_HALF_LENGTH * math.cos(theta)],
            [y, y + BOX_HALF_LENGTH * math.sin(theta)],
        )

        cosine = math.cos(theta)
        sine = math.sin(theta)
        contact_x = (
            x + cosine * row["contact_x"] - sine * row["contact_y"]
        )
        contact_y = (
            y + sine * row["contact_x"] + cosine * row["contact_y"]
        )
        contact_point.set_data([contact_x], [contact_y])
        body_force_x = row["lambda1"] + row["lambda3"]
        body_force_y = row["lambda2"] + row["lambda4"]
        world_force_x = cosine * body_force_x - sine * body_force_y
        world_force_y = sine * body_force_x + cosine * body_force_y
        force_norm = math.hypot(world_force_x, world_force_y)
        force_scale = 0.55 / max(force_norm, 1e-12)
        force_arrow.set_offsets([[contact_x, contact_y]])
        force_arrow.set_UVC(
            [force_scale * world_force_x],
            [force_scale * world_force_y],
        )

        current_distance = math.hypot(x - target[0], y - target[1])
        current_angle = abs(theta - target[2])
        plot_times = sample_times[: sample_index + 1]
        plot_distances = distances[: sample_index + 1]
        plot_angles = angles[: sample_index + 1]
        if not plot_times or abs(plot_times[-1] - time_s) > 1e-12:
            plot_times = plot_times + [time_s]
            plot_distances = plot_distances + [current_distance]
            plot_angles = plot_angles + [current_angle]
        distance_line.set_data(plot_times, plot_distances)
        angle_line.set_data(plot_times, plot_angles)
        distance_marker.set_data([time_s], [current_distance])
        angle_marker.set_data([time_s], [current_angle])
        noise_position_line.set_data(
            sample_times[: sample_index + 1],
            noise_position_mm[: sample_index + 1],
        )
        noise_angle_line.set_data(
            sample_times[: sample_index + 1],
            noise_angle_mrad[: sample_index + 1],
        )
        combined_force_line.set_data(
            sample_times[: sample_index + 1],
            combined_forces[: sample_index + 1],
        )

        reset_visible = time_s >= reset_time
        reset_arrow.set_visible(reset_visible)
        reset_label.set_visible(reset_visible)
        if time_s < reset_time:
            phase = "TRACKING - combined disturbance active"
        elif time_s < reset_time + 0.35:
            phase = "STATE RESET APPLIED - feedback recovery"
        else:
            phase = "RECOVERY - target hold"
        phase_text.set_text(phase)
        solve_kind = "cold start" if sample_index == 0 else "shifted warm"
        metrics_text.set_text(
            f"time             {time_s:5.2f} s\n"
            f"target error     {current_distance:7.4f} m, "
            f"{current_angle:7.4f} rad\n"
            f"measurement      {noise_position_mm[sample_index]:5.2f} mm, "
            f"{noise_angle_mrad[sample_index]:5.2f} mrad\n"
            f"contact force    {combined_forces[sample_index]:7.2f}\n"
            f"ContactIPM       {1000.0 * row['solve_time_s']:7.2f} ms"
            f"  [{solve_kind}]\n"
            f"physical product {row['physical_mpcc']:.2e}"
        )

    args.output.parent.mkdir(parents=True, exist_ok=True)
    writer = FFMpegWriter(
        fps=FPS,
        codec="libx264",
        bitrate=3500,
        metadata={
            "title": "ContactIPM closed-loop Push Box combined disturbance",
            "comment": "Anonymous ICRA 2027 accompanying-video clip",
        },
        extra_args=[
            "-pix_fmt",
            "yuv420p",
            "-movflags",
            "+faststart",
            "-profile:v",
            "high",
        ],
    )
    with writer.saving(figure, str(args.output), FIGURE_DPI):
        for frame in range(total_frames):
            update(frame)
            writer.grab_frame(facecolor=figure.get_facecolor())
            if frame % FPS == 0:
                print(
                    f"rendered {frame // FPS:02d}/{math.ceil(total_duration):02d} s",
                    flush=True,
                )
    plt.close(figure)
    print(
        f"selected rollout {rollout_index}: largest combined translational "
        f"state reset ({math.hypot(reset[0], reset[1]):.4f} m)"
    )
    print(args.output)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
