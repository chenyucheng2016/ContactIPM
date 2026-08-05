#!/usr/bin/env python3
"""Independent physical/task audit for the IMPACT-parameter comparison."""

from __future__ import annotations

import argparse
import json
import math
from pathlib import Path
from typing import Callable, Iterable


FEASIBILITY_TOL = 1e-5
POSE_POSITION_TOL = 1e-2
POSE_ANGLE_TOL = 1e-2
CART_POSITION_TOL = 1e-2
CART_VELOCITY_TOL = 1e-2


def _max_abs(values: Iterable[float]) -> float:
    return max((abs(value) for value in values), default=0.0)


def _squared_norm(values: Iterable[float]) -> float:
    return sum(value * value for value in values)


def _read_numeric_rows(path: Path) -> list[list[float]]:
    rows: list[list[float]] = []
    for line in path.read_text(encoding="utf-8").splitlines():
        stripped = line.strip()
        if not stripped or stripped.startswith("#"):
            continue
        rows.append([float(value) for value in stripped.split()])
    return rows


def _matrix_after(
    lines: list[str], marker: str, expected_rows: int, width: int
) -> list[list[float]]:
    start = next(
        (index + 1 for index, line in enumerate(lines) if line.startswith(marker)),
        None,
    )
    if start is None:
        raise ValueError(f"missing trajectory section: {marker}")
    rows: list[list[float]] = []
    for line in lines[start:]:
        stripped = line.strip()
        if not stripped:
            if rows:
                break
            continue
        if stripped.startswith("#"):
            if rows:
                break
            continue
        row = [float(value) for value in stripped.split()]
        if len(row) != width:
            raise ValueError(
                f"{marker} row has width {len(row)}, expected {width}"
            )
        rows.append(row)
    if len(rows) != expected_rows:
        raise ValueError(
            f"{marker} has {len(rows)} rows, expected {expected_rows}"
        )
    return rows


def load_trajectory(
    path: Path, solver: str, state_dim: int, control_dim: int, horizon: int
) -> tuple[list[list[float]], list[list[float]]]:
    if solver == "impact":
        lines = path.read_text(encoding="utf-8").splitlines()
        states = _matrix_after(
            lines, "# State Trajectory", horizon + 1, state_dim
        )
        controls = _matrix_after(
            lines, "# Control Trajectory", horizon, control_dim
        )
        return states, controls
    if solver != "contactipm":
        raise ValueError(f"unknown solver: {solver}")
    rows = _read_numeric_rows(path)
    expected_width = state_dim + control_dim
    if len(rows) != horizon + 1:
        raise ValueError(
            f"ContactIPM trajectory has {len(rows)} rows, expected {horizon + 1}"
        )
    if any(len(row) != expected_width for row in rows):
        raise ValueError(
            f"ContactIPM trajectory rows must have width {expected_width}"
        )
    states = [row[:state_dim] for row in rows]
    controls = [row[state_dim:] for row in rows[:horizon]]
    return states, controls


def _base_metrics(
    states: list[list[float]],
    controls: list[list[float]],
    start: list[float],
    goal: list[float],
    step: Callable[[list[float], list[float]], list[float]],
) -> dict[str, float]:
    dynamics_defect = 0.0
    for index, control in enumerate(controls):
        predicted = step(states[index], control)
        dynamics_defect = max(
            dynamics_defect,
            _max_abs(
                predicted[axis] - states[index + 1][axis]
                for axis in range(len(predicted))
            ),
        )
    return {
        "initial_state_error": _max_abs(
            states[0][axis] - start[axis] for axis in range(len(start))
        ),
        "dynamics_defect": dynamics_defect,
        "total_tracking_error": sum(
            _squared_norm(state[axis] - goal[axis] for axis in range(len(goal)))
            for state in states
        ),
    }


def audit_push_box(
    states: list[list[float]],
    controls: list[list[float]],
    start: list[float],
    goal: list[float],
) -> dict[str, float | bool]:
    dt = 0.05
    inverse_drag = 1.0 / (0.5 * 0.1 * 9.81)

    def step(state: list[float], control: list[float]) -> list[float]:
        cx, cy, l1, l2, l3, l4 = control
        del cx
        force_x = l2 + l4
        force_y = l1 + l3
        cosine = math.cos(state[2])
        sine = math.sin(state[2])
        return [
            state[0] + dt * inverse_drag * (force_x * cosine - force_y * sine),
            state[1] + dt * inverse_drag * (force_x * sine + force_y * cosine),
            state[2]
            + dt
            * inverse_drag
            / (0.5 * 0.5)
            * (-cy * force_x + control[0] * force_y),
        ]

    metrics = _base_metrics(states, controls, start, goal, step)
    side_violation = 0.0
    complementarity = 0.0
    force_effort = 0.0
    peak_force = 0.0
    control_cost = 0.0
    for control in controls:
        cx, cy, l1, l2, l3, l4 = control
        g = [l1, l2, -l3, -l4, l1, l1, l1, l2, l2, -l3]
        h = [
            cy + 0.4,
            cx + 0.3,
            0.4 - cy,
            0.3 - cx,
            l2,
            -l3,
            -l4,
            -l3,
            -l4,
            -l4,
        ]
        side_violation = max(
            side_violation,
            max((max(0.0, -value) for value in g + h), default=0.0),
        )
        complementarity = max(
            complementarity,
            max((abs(left * right) for left, right in zip(g, h)), default=0.0),
        )
        force_sq = l1 * l1 + l2 * l2 + l3 * l3 + l4 * l4
        force_effort += force_sq
        peak_force = max(peak_force, math.sqrt(force_sq))
        control_cost += 0.001 * _squared_norm(control)

    terminal_error = [states[-1][i] - goal[i] for i in range(3)]
    translation_error = math.hypot(terminal_error[0], terminal_error[1])
    angular_error = abs(terminal_error[2])
    terminal_cost = 100.0 * _squared_norm(terminal_error)
    metrics.update(
        {
            "side_violation": side_violation,
            "equality_violation": 0.0,
            "complementarity": complementarity,
            "translation_error": translation_error,
            "angular_error": angular_error,
            "force_effort": force_effort,
            "peak_force": peak_force,
            "control_cost": control_cost,
            "terminal_cost": terminal_cost,
            "objective": control_cost + terminal_cost,
        }
    )
    return _finish(metrics, translation_error <= POSE_POSITION_TOL
                   and angular_error <= POSE_ANGLE_TOL)


def _push_t_terms(control: list[float]) -> tuple[
    list[float], list[float], list[float], list[float], list[float]
]:
    cx, cy = control[:2]
    lambdas = control[2:10]
    v = control[10:17]
    w = control[17:24]
    unit = 0.05
    dc = 2.6429
    absolute = [v[i] + w[i] for i in range(7)]
    gaps = [
        (4.0 - dc) * unit - cy,
        absolute[0] + absolute[1] + absolute[2] - unit,
        absolute[0] + absolute[2] + absolute[3] - 1.5 * unit,
        absolute[2] + absolute[3] + absolute[4] - 3.0 * unit,
        absolute[3] + absolute[4] + absolute[5] - unit,
        absolute[2] + absolute[4] + absolute[5] - 3.0 * unit,
        absolute[2] + absolute[5] + absolute[6] - 1.5 * unit,
        absolute[1] + absolute[2] + absolute[6] - unit,
    ]
    force_magnitudes = [
        -lambdas[0],
        -lambdas[1],
        lambdas[2],
        -lambdas[3],
        lambdas[4],
        lambdas[5],
        lambdas[6],
        lambdas[7],
    ]
    g = list(v) + force_magnitudes
    h = list(w) + gaps
    for left in range(8):
        for right in range(left + 1, 8):
            g.append(force_magnitudes[left])
            h.append(force_magnitudes[right])
    equality = [
        (v[0] - w[0]) - (cx - 2.0 * unit),
        (v[1] - w[1]) - (cy - (4.0 - dc) * unit),
        (v[2] - w[2]) - (cy - (3.0 - dc) * unit),
        (v[3] - w[3]) - (cx - 0.5 * unit),
        (v[4] - w[4]) - (cy + dc * unit),
        (v[5] - w[5]) - (cx + 0.5 * unit),
        (v[6] - w[6]) - (cx + 2.0 * unit),
    ]
    inequality = [
        -2.0 * unit - cx,
        cx - 2.0 * unit,
        -dc * unit - cy,
        cy - (4.0 - dc) * unit,
    ]
    return g, h, equality, inequality, lambdas


def audit_push_t(
    states: list[list[float]],
    controls: list[list[float]],
    start: list[float],
    goal: list[float],
) -> dict[str, float | bool]:
    dt = 0.05
    inverse_drag = 1.0 / (0.4 * 0.1 * 9.8)

    def step(state: list[float], control: list[float]) -> list[float]:
        cx, cy = control[:2]
        lambdas = control[2:10]
        force_y = sum(lambdas[0::2])
        force_x = sum(lambdas[1::2])
        cosine = math.cos(state[2])
        sine = math.sin(state[2])
        return [
            state[0] + dt * inverse_drag * (force_x * cosine - force_y * sine),
            state[1] + dt * inverse_drag * (force_x * sine + force_y * cosine),
            state[2]
            + dt
            * inverse_drag
            / (0.4 * 2.8 * 0.05)
            * (-cy * force_x + cx * force_y),
        ]

    metrics = _base_metrics(states, controls, start, goal, step)
    side_violation = 0.0
    equality_violation = 0.0
    complementarity = 0.0
    force_effort = 0.0
    peak_force = 0.0
    control_cost = 0.0
    for control in controls:
        g, h, equality, inequality, lambdas = _push_t_terms(control)
        side_violation = max(
            side_violation,
            max((max(0.0, -value) for value in g + h), default=0.0),
            max((max(0.0, value) for value in inequality), default=0.0),
        )
        equality_violation = max(equality_violation, _max_abs(equality))
        complementarity = max(
            complementarity,
            max((abs(left * right) for left, right in zip(g, h)), default=0.0),
        )
        force_sq = _squared_norm(lambdas)
        force_effort += force_sq
        peak_force = max(peak_force, math.sqrt(force_sq))
        control_cost += 0.01 * _squared_norm(control)

    terminal_error = [states[-1][i] - goal[i] for i in range(3)]
    translation_error = math.hypot(terminal_error[0], terminal_error[1])
    angular_error = abs(terminal_error[2])
    terminal_cost = 100.0 * _squared_norm(terminal_error)
    metrics.update(
        {
            "side_violation": side_violation,
            "equality_violation": equality_violation,
            "complementarity": complementarity,
            "translation_error": translation_error,
            "angular_error": angular_error,
            "force_effort": force_effort,
            "peak_force": peak_force,
            "control_cost": control_cost,
            "terminal_cost": terminal_cost,
            "objective": control_cost + terminal_cost,
        }
    )
    return _finish(metrics, translation_error <= POSE_POSITION_TOL
                   and angular_error <= POSE_ANGLE_TOL)


def audit_cart_transport(
    states: list[list[float]],
    controls: list[list[float]],
    start: list[float],
    goal: list[float],
) -> dict[str, float | bool]:
    dt = 0.02

    def step(state: list[float], control: list[float]) -> list[float]:
        friction_force, active_force = control[:2]
        return [
            state[0] + dt * state[2],
            state[1] + dt * state[3],
            state[2] + dt * friction_force / 0.1,
            state[3] + dt * (active_force - friction_force) / 0.2,
        ]

    metrics = _base_metrics(states, controls, start, goal, step)
    side_violation = 0.0
    equality_violation = 0.0
    complementarity = 0.0
    force_effort = 0.0
    peak_force = 0.0
    control_cost = 0.0
    friction_limit = 0.2 * 0.1 * 9.81
    for state, control in zip(states, controls):
        friction_force, active_force, v, w = control
        equality = state[2] - state[3] - v + w
        g = [v, w, w]
        h = [w, friction_limit - friction_force,
             friction_force + friction_limit]
        inequalities = [
            friction_force - friction_limit,
            -friction_force - friction_limit,
            state[1] - state[0] - 1.0,
            state[0] - state[1] - 1.0,
        ]
        side_violation = max(
            side_violation,
            max((max(0.0, -value) for value in g + h), default=0.0),
            max((max(0.0, value) for value in inequalities), default=0.0),
        )
        equality_violation = max(equality_violation, abs(equality))
        complementarity = max(
            complementarity,
            max((abs(left * right) for left, right in zip(g, h)), default=0.0),
        )
        force_sq = friction_force * friction_force + active_force * active_force
        force_effort += force_sq
        peak_force = max(peak_force, math.sqrt(force_sq))
        control_cost += 1e-6 * _squared_norm(control)

    terminal_error = [states[-1][i] - goal[i] for i in range(4)]
    position_error = math.hypot(terminal_error[0], terminal_error[1])
    velocity_error = math.hypot(terminal_error[2], terminal_error[3])
    terminal_cost = 5000.0 * _squared_norm(terminal_error)
    metrics.update(
        {
            "side_violation": side_violation,
            "equality_violation": equality_violation,
            "complementarity": complementarity,
            "position_error": position_error,
            "velocity_error": velocity_error,
            "force_effort": force_effort,
            "peak_force": peak_force,
            "control_cost": control_cost,
            "terminal_cost": terminal_cost,
            "objective": control_cost + terminal_cost,
        }
    )
    return _finish(metrics, position_error <= CART_POSITION_TOL
                   and velocity_error <= CART_VELOCITY_TOL)


def _finish(
    metrics: dict[str, float], task_success: bool
) -> dict[str, float | bool]:
    feasible = (
        metrics["initial_state_error"] <= FEASIBILITY_TOL
        and metrics["dynamics_defect"] <= FEASIBILITY_TOL
        and metrics["side_violation"] <= FEASIBILITY_TOL
        and metrics["equality_violation"] <= FEASIBILITY_TOL
        and metrics["complementarity"] <= FEASIBILITY_TOL
    )
    return {
        **metrics,
        "feasible": feasible,
        "task_success": task_success,
        "audited_success": feasible and task_success,
    }


PROBLEMS = {
    "push_box": (3, 6, 50, audit_push_box),
    "push_t": (3, 24, 50, audit_push_t),
    "cart_transport": (4, 4, 300, audit_cart_transport),
}


def audit(
    problem: str,
    solver: str,
    trajectory: Path,
    start: list[float],
    goal: list[float],
) -> dict[str, float | bool | str]:
    state_dim, control_dim, horizon, evaluator = PROBLEMS[problem]
    if len(start) != state_dim or len(goal) != state_dim:
        raise ValueError(f"{problem} requires {state_dim}-element start and goal")
    states, controls = load_trajectory(
        trajectory, solver, state_dim, control_dim, horizon
    )
    return {
        "problem": problem,
        "solver": solver,
        "trajectory": str(trajectory),
        **evaluator(states, controls, start, goal),
    }


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--problem", choices=sorted(PROBLEMS), required=True)
    parser.add_argument("--solver", choices=["contactipm", "impact"], required=True)
    parser.add_argument("--trajectory", type=Path, required=True)
    parser.add_argument("--start", type=float, nargs="+", required=True)
    parser.add_argument("--goal", type=float, nargs="+", required=True)
    args = parser.parse_args()
    result = audit(
        args.problem, args.solver, args.trajectory, args.start, args.goal
    )
    print(json.dumps(result, indent=2, sort_keys=True))
    return 0 if result["audited_success"] else 1


if __name__ == "__main__":
    raise SystemExit(main())
