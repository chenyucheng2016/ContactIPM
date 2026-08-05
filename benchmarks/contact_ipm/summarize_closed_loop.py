#!/usr/bin/env python3
"""Summarize the deterministic Push Box closed-loop validation artifact."""

from __future__ import annotations

import argparse
import hashlib
import json
import math
from pathlib import Path
from typing import Any, Iterable


EXPECTED_GROUPS = {
    "nominal": 5,
    "initial_pose": 15,
    "model_mismatch": 10,
    "pose_kick": 10,
    "combined": 10,
}
GROUP_LABELS = {
    "nominal": "Nominal",
    "initial_pose": "Initial-pose perturbation",
    "model_mismatch": "Mass/friction mismatch",
    "pose_kick": "Isolated state-reset diagnostic",
    "combined": "Disturbed motion",
}


def quantile(values: Iterable[float], probability: float) -> float:
    ordered = sorted(float(value) for value in values)
    if not ordered:
        return 0.0
    index = probability * (len(ordered) - 1)
    lower = math.floor(index)
    upper = math.ceil(index)
    weight = index - lower
    return ordered[lower] * (1.0 - weight) + ordered[upper] * weight


def distribution(values: Iterable[float], scale: float = 1.0) -> dict[str, float]:
    samples = [scale * float(value) for value in values]
    return {
        "median": quantile(samples, 0.5),
        "p90": quantile(samples, 0.9),
        "p99": quantile(samples, 0.99),
        "max": max(samples, default=0.0),
    }


def sum_field(rollouts: list[dict[str, Any]], field: str) -> int:
    return sum(int(rollout[field]) for rollout in rollouts)


def maximum(rollouts: list[dict[str, Any]], field: str) -> float:
    return max((float(rollout[field]) for rollout in rollouts), default=0.0)


def group_summary(name: str, rollouts: list[dict[str, Any]]) -> dict[str, Any]:
    return {
        "group": name,
        "rollouts": len(rollouts),
        "successes": sum(bool(item["task_success"]) for item in rollouts),
        "steps": distribution(item["steps"] for item in rollouts),
        "final_translation_error_m": distribution(
            item["final_translation_error"] for item in rollouts
        ),
        "final_angular_error_rad": distribution(
            item["final_angular_error"] for item in rollouts
        ),
        "primary_failures": sum_field(rollouts, "primary_failures"),
        "cold_fallbacks": sum_field(rollouts, "cold_fallbacks"),
        "unrecovered_failures": sum_field(rollouts, "unrecovered_failures"),
        "invalid_plans": sum_field(rollouts, "invalid_plans"),
        "deadline_misses_all": sum_field(rollouts, "deadline_misses_all"),
        "deadline_misses_warm": sum_field(rollouts, "deadline_misses_warm"),
        "max_plan_physical_mpcc": maximum(
            rollouts, "max_plan_physical_mpcc"
        ),
        "max_one_step_model_error": maximum(
            rollouts, "max_one_step_model_error"
        ),
    }


def summarize(payload: dict[str, Any], source_sha256: str) -> dict[str, Any]:
    if payload.get("benchmark") != "push_box_closed_loop":
        raise ValueError("input is not a push_box_closed_loop artifact")
    rollouts = list(payload.get("rollouts", []))
    if not rollouts:
        raise ValueError("input contains no rollouts")

    all_times = [
        float(value)
        for rollout in rollouts
        for value in rollout["solve_times_s"]
    ]
    warm_times = [
        float(value)
        for rollout in rollouts
        for value in rollout["solve_times_s"][1:]
    ]
    groups: dict[str, list[dict[str, Any]]] = {}
    for rollout in rollouts:
        groups.setdefault(str(rollout["group"]), []).append(rollout)

    configuration = dict(payload["configuration"])
    required_hold = int(configuration["goal_hold_steps"])
    if any(
        rollout["task_success"]
        and int(rollout["goal_hold_steps"]) < required_hold
        for rollout in rollouts
    ):
        raise ValueError("successful rollout does not satisfy the goal hold gate")
    deadline = float(configuration["control_dt"])
    recomputed_all_misses = sum(value > deadline for value in all_times)
    recomputed_warm_misses = sum(value > deadline for value in warm_times)
    recorded_all_misses = sum_field(rollouts, "deadline_misses_all")
    recorded_warm_misses = sum_field(rollouts, "deadline_misses_warm")
    if (recomputed_all_misses, recomputed_warm_misses) != (
        recorded_all_misses,
        recorded_warm_misses,
    ):
        raise ValueError("recorded and recomputed deadline misses disagree")

    return {
        "schema_version": 1,
        "benchmark": payload["benchmark"],
        "seed": payload["seed"],
        "source_sha256": source_sha256,
        "configuration": configuration,
        "overall": {
            "rollouts": len(rollouts),
            "successes": sum(
                bool(rollout["task_success"]) for rollout in rollouts
            ),
            "solve_count": len(all_times),
            "warm_solve_count": len(warm_times),
            "minimum_goal_hold_steps": min(
                int(rollout["goal_hold_steps"]) for rollout in rollouts
            ),
            "timing_all_ms": distribution(all_times, 1000.0),
            "timing_warm_ms": distribution(warm_times, 1000.0),
            "deadline_ms": 1000.0 * deadline,
            "deadline_misses_all": recorded_all_misses,
            "deadline_misses_warm": recorded_warm_misses,
            "primary_failures": sum_field(rollouts, "primary_failures"),
            "cold_fallbacks": sum_field(rollouts, "cold_fallbacks"),
            "unrecovered_failures": sum_field(
                rollouts, "unrecovered_failures"
            ),
            "invalid_plans": sum_field(rollouts, "invalid_plans"),
            "final_translation_error_m": distribution(
                rollout["final_translation_error"] for rollout in rollouts
            ),
            "final_angular_error_rad": distribution(
                rollout["final_angular_error"] for rollout in rollouts
            ),
            "max_plan_dynamics_defect": maximum(
                rollouts, "max_plan_dynamics_defect"
            ),
            "max_plan_side_violation": maximum(
                rollouts, "max_plan_side_violation"
            ),
            "max_plan_physical_mpcc": maximum(
                rollouts, "max_plan_physical_mpcc"
            ),
            "max_applied_side_violation": maximum(
                rollouts, "max_applied_side_violation"
            ),
            "max_applied_physical_mpcc": maximum(
                rollouts, "max_applied_physical_mpcc"
            ),
            "max_one_step_model_error": maximum(
                rollouts, "max_one_step_model_error"
            ),
        },
        "groups": [
            group_summary(name, groups[name])
            for name in EXPECTED_GROUPS
            if name in groups
        ],
    }


def validate_suite(summary: dict[str, Any]) -> None:
    observed = {
        group["group"]: group["rollouts"] for group in summary["groups"]
    }
    if observed != EXPECTED_GROUPS:
        raise ValueError(
            f"suite groups differ: expected {EXPECTED_GROUPS}, got {observed}"
        )
    if summary["overall"]["rollouts"] != sum(EXPECTED_GROUPS.values()):
        raise ValueError("suite does not contain 50 rollouts")


def fmt(value: float, digits: int = 3) -> str:
    return f"{value:.{digits}g}"


def markdown(summary: dict[str, Any]) -> str:
    overall = summary["overall"]
    config = summary["configuration"]
    success_rate = 100.0 * overall["successes"] / overall["rollouts"]
    warm_miss_rate = (
        100.0
        * overall["deadline_misses_warm"]
        / max(1, overall["warm_solve_count"])
    )
    lines = [
        "# ContactIPM closed-loop Push Box validation",
        "",
        f"Artifact SHA-256: `{summary['source_sha256']}`.",
        "",
    ]
    if "environment" in summary:
        environment = summary["environment"]
        lines.extend(
            [
                "Reference execution: "
                f"{environment.get('platform', 'unspecified platform')}, "
                f"{environment.get('cpu_model', 'unspecified CPU')}, "
                f"{environment.get('compiler', 'unspecified compiler')}, "
                f"CPU affinity {environment.get('cpu_affinity', 'unspecified')}.",
                "",
            ]
        )
    lines.extend(
        [
        (
            f"ContactIPM completed {overall['successes']}/{overall['rollouts']} "
            f"rollouts ({success_rate:.1f}%) with zero unrecovered failures "
            f"and zero invalid plans. Success requires the terminal pose to "
            f"remain within {config['translation_tolerance']} m and "
            f"{config['angular_tolerance']} rad for "
            f"{config['goal_hold_steps']} consecutive control steps."
        ),
        "",
        "## Robustness by scenario",
        "",
        (
            "| Scenario | Success | Steps median/max | Translation error "
            "median/max (m) | Angular error median/max (rad) | "
            "Primary failures | Cold fallbacks |"
        ),
        "|---|---:|---:|---:|---:|---:|---:|",
        ]
    )
    for group in summary["groups"]:
        translation = group["final_translation_error_m"]
        angular = group["final_angular_error_rad"]
        steps = group["steps"]
        lines.append(
            f"| {GROUP_LABELS.get(group['group'], group['group'])} | "
            f"{group['successes']}/{group['rollouts']} | "
            f"{fmt(steps['median'])}/{fmt(steps['max'])} | "
            f"{fmt(translation['median'])}/{fmt(translation['max'])} | "
            f"{fmt(angular['median'])}/{fmt(angular['max'])} | "
            f"{group['primary_failures']} | {group['cold_fallbacks']} |"
        )

    all_timing = overall["timing_all_ms"]
    warm_timing = overall["timing_warm_ms"]
    lines.extend(
        [
            "",
            "## Solver timing",
            "",
            "| Population | Solves | Median | P90 | P99 | Max | Deadline misses |",
            "|---|---:|---:|---:|---:|---:|---:|",
            (
                f"| All solves | {overall['solve_count']} | "
                f"{fmt(all_timing['median'])} ms | "
                f"{fmt(all_timing['p90'])} ms | "
                f"{fmt(all_timing['p99'])} ms | "
                f"{fmt(all_timing['max'])} ms | "
                f"{overall['deadline_misses_all']} |"
            ),
            (
                f"| Shifted warm starts | {overall['warm_solve_count']} | "
                f"{fmt(warm_timing['median'])} ms | "
                f"{fmt(warm_timing['p90'])} ms | "
                f"{fmt(warm_timing['p99'])} ms | "
                f"{fmt(warm_timing['max'])} ms | "
                f"{overall['deadline_misses_warm']} ({warm_miss_rate:.2f}%) |"
            ),
            "",
            (
                f"The control period is {fmt(overall['deadline_ms'])} ms. "
                "These measurements support mostly real-time-capable execution, "
                "not a hard real-time guarantee, because deadline misses remain."
            ),
            "",
            "## Physical and numerical audit",
            "",
            "| Metric | Worst value |",
            "|---|---:|",
            (
                "| Planned dynamics defect | "
                f"{fmt(overall['max_plan_dynamics_defect'])} |"
            ),
            (
                "| Planned side-constraint violation | "
                f"{fmt(overall['max_plan_side_violation'])} |"
            ),
            (
                "| Planned physical complementarity | "
                f"{fmt(overall['max_plan_physical_mpcc'])} |"
            ),
            (
                "| Applied side-constraint violation | "
                f"{fmt(overall['max_applied_side_violation'])} |"
            ),
            (
                "| Applied physical complementarity | "
                f"{fmt(overall['max_applied_physical_mpcc'])} |"
            ),
            (
                "| Maximum one-step plant/model state error | "
                f"{fmt(overall['max_one_step_model_error'])} |"
            ),
            "",
            (
                "Physical complementarity is audited on the original side-row "
                "products, independently of the elastic product slack and its "
                "log barrier."
            ),
            "",
            "## Reading the figure",
            "",
            (
                "Panels (a), (b), and (d) compare only the nominal run with the "
                "disturbed-motion run. Panel (a) overlays shaded box footprints "
                "at the initial and geometric-halfway samples, plus an unfilled "
                "target "
                "footprint; each center-to-front line shows the signed box heading. "
                "The dashed blue nominal path "
                "is drawn last so it remains visible where the paths overlap."
            ),
            "",
            (
                "Panel (b) plots distance and absolute angular error to the fixed "
                "target versus elapsed closed-loop time. Samples are 0.1 s apart; "
                "the errors decrease because feedback drives the measured pose "
                "toward that target. The horizontal dotted line is the common "
                "0.1 m/rad goal tolerance, and each curve ends after its rollout "
                "completes the ten-step goal hold. Panel (c) is the empirical "
                "solve-time CDF over all 1,104 solves, with the 100 ms deadline. "
                "Panel (d) audits the original physical complementarity product "
                "against its tolerance."
            ),
            "",
            (
                "Disturbed motion is the combined stress test. Its initial x/y "
                "offsets are independently uniformly sampled from "
                "[-0.075, 0.075] m and "
                "its initial yaw offset from [-0.075, 0.075] rad. Mass and "
                "friction are independently uniformly scaled by factors in "
                "[0.85, 1.15]. "
                "Every feedback measurement receives independent zero-mean "
                "Gaussian noise with 0.001 m standard deviation in x/y and "
                "0.001 rad in yaw. After the 1.5 s control update, one direct "
                "state reset adds independent uniform x/y offsets in "
                "[-0.1125, 0.1125] m "
                "and a yaw offset from [-0.09, 0.09] rad; it first appears in "
                "the recorded state at t=1.6 s. All draws use deterministic seed "
                "2027. It is therefore not a single-disturbance experiment. The "
                "10 isolated state-reset diagnostics remain in the 50-run "
                "aggregate but are not plotted."
            ),
            "",
            "## Scope",
            "",
            (
                "The controller uses the CRISP Push Box quasi-static model, a "
                "19-step forward-Euler prediction horizon, zero free-state and "
                "control guesses for the first solve (with stage zero fixed to "
                "the measurement), and shifted warm starts thereafter. The "
                "OCP retains the source terminal tracking and force effort and "
                "adds unit-weight stage pose tracking to prevent finite-horizon "
                "motion deferral. The plant uses ten RK4 substeps. Disturbed "
                "motion combines deterministic mass/friction mismatch, "
                "measurement noise, an initial-pose error, and one scheduled "
                "pose state reset."
            ),
            "",
            (
                "Because the source benchmark has pose but no momentum state, "
                "the scheduled reset is a direct pose/yaw state change rather "
                "than an inertial impulse. The model also has no penetration or "
                "friction-cone state to audit; this experiment demonstrates "
                "closed-loop complementarity handling, not high-fidelity contact "
                "physics."
            ),
            "",
        ]
    )
    return "\n".join(lines)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("input", type=Path)
    parser.add_argument("--output-prefix", type=Path)
    parser.add_argument("--require-suite", action="store_true")
    parser.add_argument("--platform")
    parser.add_argument("--cpu-model")
    parser.add_argument("--compiler")
    parser.add_argument("--cpu-affinity")
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    raw = args.input.read_bytes()
    payload = json.loads(raw)
    summary = summarize(payload, hashlib.sha256(raw).hexdigest())
    environment = {
        key: value
        for key, value in {
            "platform": args.platform,
            "cpu_model": args.cpu_model,
            "compiler": args.compiler,
            "cpu_affinity": args.cpu_affinity,
        }.items()
        if value
    }
    if environment:
        summary["environment"] = environment
    if args.require_suite:
        validate_suite(summary)
    prefix = args.output_prefix or args.input.with_suffix("")
    json_path = prefix.with_name(prefix.name + "_summary").with_suffix(".json")
    markdown_path = prefix.with_name(prefix.name + "_summary").with_suffix(".md")
    json_path.parent.mkdir(parents=True, exist_ok=True)
    json_path.write_text(
        json.dumps(summary, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    markdown_path.write_text(markdown(summary), encoding="utf-8")
    print(json_path)
    print(markdown_path)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
