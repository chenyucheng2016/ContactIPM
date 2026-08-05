"""Independent audits of ContactIPM and CRISP raw benchmark trajectories."""

from __future__ import annotations

import json
import math
from pathlib import Path


MANIFEST_PATH = Path(__file__).with_name("source_cases.json")
CONTACT_ACTIVITY_TOLERANCE = 1e-8


def _manifest(problem: str) -> dict:
    document = json.loads(MANIFEST_PATH.read_text(encoding="utf-8"))
    return document["problems"][problem]


def _case(problem: str, overrides: dict | None) -> dict:
    case = _manifest(problem)
    if overrides:
        case.update(overrides)
    return case


def _load(path: Path, rows: int, columns: int, started_at: float) -> list[list[float]]:
    if not path.is_file() or path.stat().st_mtime < started_at - 1.0:
        raise RuntimeError(f"fresh trajectory missing: {path}")
    trajectory = [
        [float(value) for value in line.split()]
        for line in path.read_text(encoding="utf-8").splitlines()
        if line.strip()
    ]
    if len(trajectory) != rows or any(len(row) != columns for row in trajectory):
        raise RuntimeError(
            f"invalid trajectory shape {len(trajectory)}x"
            f"{len(trajectory[0]) if trajectory else 0}, expected {rows}x{columns}: "
            f"{path}"
        )
    if any(not math.isfinite(value) for row in trajectory for value in row):
        raise RuntimeError(f"non-finite trajectory value: {path}")
    return trajectory


def _result(instance: dict) -> dict:
    return {
        "successful_instances": int(instance["success"]),
        "feasible_instances": int(instance["feasible"]),
        "task_successful_instances": int(instance["task_success"]),
        "attempted_instances": 1,
        "worst_equality": instance["max_equality"],
        "worst_side_violation": instance["max_side_violation"],
        "worst_physical_mpcc": instance["physical_mpcc"],
        "worst_position_error": instance.get(
            "position_error", instance.get("translation_error", 0.0)
        ),
        "worst_angular_error": instance.get("angular_error", 0.0),
        "worst_velocity_error": instance.get("velocity_error", 0.0),
        "objective": instance["objective"],
        "instances": [instance],
    }


def audit_cartpole(
    directory: Path, solver: str, started_at: float, case_override: dict | None = None
) -> dict:
    case = _case("cartpole_soft_walls", case_override)
    p = case["parameters"]
    t = case["task_tolerances"]
    trajectory = _load(
        directory / f"{solver}_cartpole_soft_walls.txt",
        case["nodes"],
        len(case["row_layout"]),
        started_at,
    )
    initial = case["initial_state"]
    target = case["target_state"]
    equality = max(
        abs(trajectory[0][index] - initial[index]) for index in range(4)
    )
    side_violation = 0.0
    product = 0.0
    for step, row in enumerate(trajectory[:-1]):
        x, theta, x_dot, theta_dot, active, left, right = row
        denominator = (
            -p["pole_mass"] * math.cos(theta) ** 2
            + p["cart_mass"]
            + p["pole_mass"]
        )
        x_ddot = (
            right
            - left
            + active
            + left * math.cos(theta) ** 2
            - right * math.cos(theta) ** 2
            - p["gravity"] * p["pole_mass"] * math.cos(theta) * math.sin(theta)
            + p["pole_length"]
            * p["pole_mass"]
            * theta_dot**2
            * math.sin(theta)
        ) / denominator
        theta_ddot = -(
            left * p["cart_mass"] * math.cos(theta)
            - right * p["cart_mass"] * math.cos(theta)
            + p["pole_mass"] * active * math.cos(theta)
            - p["gravity"] * p["pole_mass"] ** 2 * math.sin(theta)
            - p["gravity"]
            * p["cart_mass"]
            * p["pole_mass"]
            * math.sin(theta)
            + p["pole_length"]
            * p["pole_mass"] ** 2
            * theta_dot**2
            * math.cos(theta)
            * math.sin(theta)
        ) / (p["pole_length"] * p["pole_mass"] * denominator)
        following = trajectory[step + 1]
        dt = case["dt"]
        equality = max(
            equality,
            abs(following[0] - x - following[2] * dt),
            abs(following[1] - theta - following[3] * dt),
            abs(following[2] - x_dot - x_ddot * dt),
            abs(following[3] - theta_dot - theta_ddot * dt),
        )
        left_gap = (
            p["left_wall"]
            - x
            - p["pole_length"] * math.sin(theta)
            + left / p["left_stiffness"]
        )
        right_gap = (
            p["right_wall"]
            + x
            + p["pole_length"] * math.sin(theta)
            + right / p["right_stiffness"]
        )
        side_violation = max(side_violation, -left, -right, -left_gap, -right_gap)
        product = max(product, abs(left * left_gap), abs(right * right_gap))
    terminal = trajectory[-1]
    position_error = abs(terminal[0] - target[0])
    angular_error = abs(
        math.atan2(
            math.sin(terminal[1] - target[1]),
            math.cos(terminal[1] - target[1]),
        )
    )
    velocity_error = abs(terminal[2] - target[2])
    angular_velocity_error = abs(terminal[3] - target[3])
    terminal_tracking_cost = sum(
        weight * (terminal[index] - target[index]) ** 2
        for index, weight in enumerate(case["terminal_weights"])
    )
    control_effort_cost = sum(
        case["control_weights"][0] * row[4] ** 2 for row in trajectory[:-1]
    )
    objective = terminal_tracking_cost + control_effort_cost
    feasible = (
        equality <= t["equality"]
        and side_violation <= t["side"]
        and product <= t["physical_complementarity"]
    )
    task_success = (
        position_error < t["position"]
        and angular_error < t["angle"]
        and velocity_error < t["velocity"]
        and angular_velocity_error < t["angular_velocity"]
    )
    contact_forces = [math.hypot(row[5], row[6]) for row in trajectory[:-1]]
    return _result(
        {
            "success": feasible and task_success,
            "feasible": feasible,
            "task_success": task_success,
            "max_equality": equality,
            "max_side_violation": side_violation,
            "physical_mpcc": product,
            "feasibility_ratios": {
                "equality": equality / t["equality"],
                "side": side_violation / t["side"],
                "physical_complementarity": (
                    product / t["physical_complementarity"]
                ),
            },
            "position_error": position_error,
            "angular_error": angular_error,
            "velocity_error": velocity_error,
            "angular_velocity_error": angular_velocity_error,
            "objective": objective,
            "objective_components": {
                "terminal_tracking": terminal_tracking_cost,
                "control_effort": control_effort_cost,
            },
            "actuation": {
                "peak_control_force": max(abs(row[4]) for row in trajectory[:-1]),
                "peak_contact_force_norm": max(contact_forces),
                "contact_impulse": case["dt"] * sum(contact_forces),
                "active_contact_steps": sum(
                    force > CONTACT_ACTIVITY_TOLERANCE for force in contact_forces
                ),
            },
        }
    )


def audit_push_box(
    directory: Path, solver: str, started_at: float, case_override: dict | None = None
) -> dict:
    case = _case("push_box", case_override)
    p = case["parameters"]
    t = case["task_tolerances"]
    trajectory = _load(
        directory / f"{solver}_push_box.txt",
        case["nodes"],
        len(case["row_layout"]),
        started_at,
    )
    initial = case["initial_state"]
    equality = max(
        abs(trajectory[0][index] - initial[index]) for index in range(3)
    )
    side_violation = 0.0
    product = 0.0
    force_sign = (1.0, 1.0, -1.0, -1.0)
    for step, row in enumerate(trajectory[:-1]):
        px, py, theta, cx, cy = row[:5]
        force = row[5:9]
        vx = (
            math.cos(theta) * (force[1] + force[3])
            - math.sin(theta) * (force[0] + force[2])
        ) / (p["friction"] * p["mass"] * p["gravity"])
        vy = (
            math.sin(theta) * (force[1] + force[3])
            + math.cos(theta) * (force[0] + force[2])
        ) / (p["friction"] * p["mass"] * p["gravity"])
        radius = math.hypot(p["half_length"], p["half_width"])
        omega = (-cy * (force[1] + force[3]) + cx * (force[0] + force[2])) / (
            p["friction"]
            * p["mass"]
            * p["gravity"]
            * p["rotation_scale"]
            * radius
        )
        following = trajectory[step + 1]
        equality = max(
            equality,
            abs(following[0] - px - case["dt"] * vx),
            abs(following[1] - py - case["dt"] * vy),
            abs(following[2] - theta - case["dt"] * omega),
        )
        signed_force = [
            force_sign[index] * force[index] for index in range(4)
        ]
        gaps = [
            cy + p["half_width"],
            cx + p["half_length"],
            p["half_width"] - cy,
            p["half_length"] - cx,
        ]
        side_violation = max(
            side_violation, *(-value for value in signed_force + gaps)
        )
        product = max(
            product,
            *(abs(signed_force[index] * gaps[index]) for index in range(4)),
            *(
                abs(signed_force[first] * signed_force[second])
                for first in range(4)
                for second in range(first + 1, 4)
            ),
        )
    if "target_state" in case:
        target = case["target_state"]
    else:
        target_angle = (
            case["target_segment"] * 2.0 * math.pi / case["target_segments"]
        )
        target = [
            case["target_radius"] * math.cos(target_angle),
            case["target_radius"] * math.sin(target_angle),
            target_angle,
        ]
    terminal = trajectory[-1]
    translation_error = math.hypot(
        terminal[0] - target[0], terminal[1] - target[1]
    )
    angular_error = abs(terminal[2] - target[2])
    terminal_tracking_cost = sum(
        weight * (terminal[index] - target[index]) ** 2
        for index, weight in enumerate(case["terminal_weights"])
    )
    force_effort_cost = sum(
        sum(
            weight * row[5 + index] ** 2
            for index, weight in enumerate(case["force_weights"])
        )
        for row in trajectory[:-1]
    )
    objective = terminal_tracking_cost + force_effort_cost
    feasible = (
        equality <= t["equality"]
        and side_violation <= t["side"]
        and product <= t["physical_complementarity"]
    )
    task_success = (
        translation_error < t["translation"] and angular_error < t["angle"]
    )
    force_norms = [
        math.sqrt(sum(value * value for value in row[5:9]))
        for row in trajectory[:-1]
    ]
    modes = []
    for row in trajectory[:-1]:
        signed_force = [
            force_sign[index] * row[5 + index] for index in range(4)
        ]
        peak = max(signed_force)
        modes.append(
            signed_force.index(peak) if peak > CONTACT_ACTIVITY_TOLERANCE else -1
        )
    return _result(
        {
            "success": feasible and task_success,
            "feasible": feasible,
            "task_success": task_success,
            "max_equality": equality,
            "max_side_violation": side_violation,
            "physical_mpcc": product,
            "feasibility_ratios": {
                "equality": equality / t["equality"],
                "side": side_violation / t["side"],
                "physical_complementarity": (
                    product / t["physical_complementarity"]
                ),
            },
            "translation_error": translation_error,
            "angular_error": angular_error,
            "objective": objective,
            "objective_components": {
                "terminal_tracking": terminal_tracking_cost,
                "force_effort": force_effort_cost,
            },
            "actuation": {
                "peak_contact_force_norm": max(force_norms),
                "contact_impulse": case["dt"] * sum(force_norms),
                "active_contact_steps": sum(mode >= 0 for mode in modes),
                "dominant_contact_mode_changes": sum(
                    current != previous
                    for previous, current in zip(modes, modes[1:])
                ),
            },
        }
    )


def audit_transport(
    directory: Path, solver: str, started_at: float, case_override: dict | None = None
) -> dict:
    case = _case("transport", case_override)
    p = case["parameters"]
    t = case["task_tolerances"]
    trajectory = _load(
        directory / f"{solver}_transport.txt",
        case["nodes"],
        len(case["row_layout"]),
        started_at,
    )
    equality = max(
        abs(trajectory[0][index] - case["initial_state"][index])
        for index in range(6)
    )
    side_violation = 0.0
    product = 0.0
    friction_limit = p["friction"] * p["payload_mass"] * p["gravity"]
    for step, row in enumerate(trajectory):
        x1, x2, v1, v2, positive, negative, force, active = row
        side = [
            positive,
            negative,
            friction_limit - force,
            force + friction_limit,
            x1 - x2 + p["half_cart_length"],
            p["half_cart_length"] - (x1 - x2),
        ]
        side_violation = max(side_violation, *(-value for value in side))
        product = max(
            product,
            abs(positive * negative),
            abs(negative * (friction_limit - force)),
            abs(positive * (force + friction_limit)),
        )
        if step + 1 == len(trajectory):
            continue
        following = trajectory[step + 1]
        equality = max(
            equality,
            abs(following[0] - x1 - following[2] * case["dt"]),
            abs(following[1] - x2 - following[3] * case["dt"]),
            abs(following[2] - v1 - force / p["payload_mass"] * case["dt"]),
            abs(
                following[3]
                - v2
                - (active - force) / p["cart_mass"] * case["dt"]
            ),
            abs(v1 - v2 - positive + negative),
        )
    target = case["target_state"]
    terminal = trajectory[-1]
    translation_error = math.hypot(
        terminal[0] - target[0], terminal[1] - target[1]
    )
    velocity_error = math.hypot(
        terminal[2] - target[2], terminal[3] - target[3]
    )
    terminal_tracking_cost = sum(
        weight * (terminal[index] - target[index]) ** 2
        for index, weight in enumerate(case["terminal_weights"])
    )
    control_effort_cost = sum(
        case["control_weights"][0] * row[6] ** 2
        + case["control_weights"][1] * row[7] ** 2
        for row in trajectory[:-1]
    )
    objective = terminal_tracking_cost + control_effort_cost
    feasible = (
        equality <= t["equality"]
        and side_violation <= t["side"]
        and product <= t["physical_complementarity"]
    )
    task_success = (
        translation_error < t["translation"] and velocity_error < t["velocity"]
    )
    return _result(
        {
            "success": feasible and task_success,
            "feasible": feasible,
            "task_success": task_success,
            "max_equality": equality,
            "max_side_violation": side_violation,
            "physical_mpcc": product,
            "feasibility_ratios": {
                "equality": equality / t["equality"],
                "side": side_violation / t["side"],
                "physical_complementarity": (
                    product / t["physical_complementarity"]
                ),
            },
            "translation_error": translation_error,
            "velocity_error": velocity_error,
            "objective": objective,
            "objective_components": {
                "terminal_tracking": terminal_tracking_cost,
                "control_effort": control_effort_cost,
            },
            "actuation": {
                "peak_friction_force": max(abs(row[6]) for row in trajectory[:-1]),
                "peak_control_force": max(abs(row[7]) for row in trajectory[:-1]),
            },
        }
    )


def audit_push_t(
    directory: Path,
    solver: str,
    started_at: float,
    segments: list[int] | None = None,
    case_override: dict | None = None,
) -> dict:
    case = _case("push_t", case_override)
    p = case["parameters"]
    t = case["task_tolerances"]
    unit = p["unit"]
    dc = p["dc"]
    force_sign = (-1.0, -1.0, 1.0, -1.0, 1.0, 1.0, 1.0, 1.0)
    audited_segments = []
    requested_segments = (
        range(case["initial_cases"]["count"]) if segments is None else segments
    )
    for segment in requested_segments:
        path = directory / f"{solver}_push_t_seg_{segment:02d}.txt"
        trajectory = _load(
            path, case["nodes"], len(case["row_layout"]), started_at
        )
        equality = 0.0
        side_violation = 0.0
        product = 0.0
        running_tracking_cost = 0.0
        terminal_tracking_cost = 0.0
        force_effort_cost = 0.0
        force_norms = []
        modes = []
        target_x, target_y, target_angle = case["target_pose"]
        for step, row in enumerate(trajectory):
            px, py, theta, cx, cy = row[:5]
            v = [row[5 + 2 * contact] for contact in range(7)]
            w = [row[6 + 2 * contact] for contact in range(7)]
            force = row[19:27]
            split = (
                cx - 2.0 * unit - v[0] + w[0],
                cy - (4.0 - dc) * unit - v[1] + w[1],
                cy - (3.0 - dc) * unit - v[2] + w[2],
                cx - 0.5 * unit - v[3] + w[3],
                cy + dc * unit - v[4] + w[4],
                cx + 0.5 * unit - v[5] + w[5],
                cx + 2.0 * unit - v[6] + w[6],
                row[27] - math.cos(theta),
                row[28] - math.sin(theta),
            )
            equality = max(equality, *(abs(value) for value in split))
            pose_cost = (
                (px - target_x) ** 2
                + (py - target_y) ** 2
                + (row[27] - math.cos(target_angle)) ** 2
                + (row[28] - math.sin(target_angle)) ** 2
            )
            if step == case["nodes"] - 1:
                terminal_tracking_cost += 100.0 * pose_cost
                continue
            running_tracking_cost += pose_cost
            force_effort_cost += sum(
                weight * value * value
                for weight, value in zip(case["force_weights"], force)
            )
            force_norms.append(math.sqrt(sum(value * value for value in force)))
            odd_sum = sum(force[0::2])
            even_sum = sum(force[1::2])
            inverse_drag = 1.0 / (p["friction"] * p["mass"] * p["gravity"])
            vx = inverse_drag * (
                math.cos(theta) * even_sum - math.sin(theta) * odd_sum
            )
            vy = inverse_drag * (
                math.sin(theta) * even_sum + math.cos(theta) * odd_sum
            )
            omega = (-cy * even_sum + cx * odd_sum) / (
                p["friction"]
                * p["mass"]
                * p["gravity"]
                * p["rotation_scale"]
                * (p["radius_units"] * unit)
            )
            following = trajectory[step + 1]
            equality = max(
                equality,
                abs(following[0] - px - case["dt"] * vx),
                abs(following[1] - py - case["dt"] * vy),
                abs(following[2] - theta - case["dt"] * omega),
            )
            signed_force = [
                force_sign[index] * force[index] for index in range(8)
            ]
            peak = max(signed_force)
            modes.append(
                signed_force.index(peak)
                if peak > CONTACT_ACTIVITY_TOLERANCE
                else -1
            )
            gaps = [
                (4.0 - dc) * unit - cy,
                v[0] + w[0] + v[1] + w[1] + v[2] + w[2] - unit,
                v[0] + w[0] + v[2] + w[2] + v[3] + w[3] - 1.5 * unit,
                v[2] + w[2] + v[3] + w[3] + v[4] + w[4] - 3.0 * unit,
                v[3] + w[3] + v[4] + w[4] + v[5] + w[5] - unit,
                v[2] + w[2] + v[4] + w[4] + v[5] + w[5] - 3.0 * unit,
                v[2] + w[2] + v[5] + w[5] + v[6] + w[6] - 1.5 * unit,
                v[1] + w[1] + v[2] + w[2] + v[6] + w[6] - unit,
            ]
            side_violation = max(
                side_violation,
                *(-value for value in v + w + signed_force + gaps),
            )
            product = max(
                product,
                *(abs(v[index] * w[index]) for index in range(7)),
                *(abs(signed_force[index] * gaps[index]) for index in range(8)),
                *(
                    abs(signed_force[first] * signed_force[second])
                    for first in range(8)
                    for second in range(first + 1, 8)
                ),
            )
        angle = (
            case["initial_cases"]["angle_period"]
            * segment
            / case["initial_cases"]["count"]
        )
        radius = case["initial_cases"]["radius_start"]
        if case["initial_cases"]["count"] > 1:
            radius += (
                case["initial_cases"]["radius_end"]
                - case["initial_cases"]["radius_start"]
            ) * segment / (case["initial_cases"]["count"] - 1)
        initial = trajectory[0]
        equality = max(
            equality,
            abs(initial[0] - radius * math.cos(angle)),
            abs(initial[1] - radius * math.sin(angle)),
            abs(initial[27] - math.cos(angle)),
            abs(initial[28] - math.sin(angle)),
        )
        terminal = trajectory[-1]
        position_error = math.hypot(
            terminal[0] - target_x, terminal[1] - target_y
        )
        angular_error = abs(
            math.atan2(
                math.sin(terminal[2] - target_angle),
                math.cos(terminal[2] - target_angle),
            )
        )
        feasible = (
            equality <= t["equality"]
            and side_violation <= t["side"]
            and product <= t["physical_complementarity"]
        )
        task_success = (
            position_error < t["position"] and angular_error < t["angle"]
        )
        objective = (
            running_tracking_cost + terminal_tracking_cost + force_effort_cost
        )
        audited_segments.append(
            {
                "segment": segment,
                "success": feasible and task_success,
                "feasible": feasible,
                "task_success": task_success,
                "max_equality": equality,
                "max_side_violation": side_violation,
                "physical_mpcc": product,
                "feasibility_ratios": {
                    "equality": equality / t["equality"],
                    "side": side_violation / t["side"],
                    "physical_complementarity": (
                        product / t["physical_complementarity"]
                    ),
                },
                "position_error": position_error,
                "angular_error": angular_error,
                "objective": objective,
                "objective_components": {
                    "running_tracking": running_tracking_cost,
                    "terminal_tracking": terminal_tracking_cost,
                    "force_effort": force_effort_cost,
                },
                "actuation": {
                    "peak_contact_force_norm": max(force_norms),
                    "contact_impulse": case["dt"] * sum(force_norms),
                    "active_contact_steps": sum(mode >= 0 for mode in modes),
                    "dominant_contact_mode_changes": sum(
                        current != previous
                        for previous, current in zip(modes, modes[1:])
                    ),
                },
            }
        )
    return {
        "successful_instances": sum(
            segment["success"] for segment in audited_segments
        ),
        "feasible_instances": sum(
            segment["feasible"] for segment in audited_segments
        ),
        "task_successful_instances": sum(
            segment["task_success"] for segment in audited_segments
        ),
        "attempted_instances": len(audited_segments),
        "worst_equality": max(
            segment["max_equality"] for segment in audited_segments
        ),
        "worst_side_violation": max(
            segment["max_side_violation"] for segment in audited_segments
        ),
        "worst_physical_mpcc": max(
            segment["physical_mpcc"] for segment in audited_segments
        ),
        "worst_position_error": max(
            segment["position_error"] for segment in audited_segments
        ),
        "worst_angular_error": max(
            segment["angular_error"] for segment in audited_segments
        ),
        "objective_sum": sum(
            segment["objective"] for segment in audited_segments
        ),
        "objective_component_sums": {
            name: sum(
                segment["objective_components"][name]
                for segment in audited_segments
            )
            for name in (
                "running_tracking",
                "terminal_tracking",
                "force_effort",
            )
        },
        "instances": audited_segments,
    }


AUDITORS = {
    "cartpole_soft_walls": audit_cartpole,
    "push_box": audit_push_box,
    "transport": audit_transport,
    "push_t": audit_push_t,
}


def audit(
    problem: str,
    directory: Path,
    solver: str,
    started_at: float,
    push_t_segment: int | None = None,
    case_override: dict | None = None,
) -> dict:
    if problem == "push_t":
        segments = None if push_t_segment is None else [push_t_segment]
        return audit_push_t(
            directory, solver.lower(), started_at, segments, case_override
        )
    return AUDITORS[problem](directory, solver.lower(), started_at, case_override)
