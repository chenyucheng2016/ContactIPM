#!/usr/bin/env python3
"""Qualify sustained ContactIPM and convex-WBC MuJoCo execution.

The runner remains the source of raw measurements.  This harness independently
checks its machine-readable records, preserves every log, and keeps stress
results separate from the core paper verdict.
"""

from __future__ import annotations

import argparse
import csv
import hashlib
import json
import math
import os
import platform
import shlex
import subprocess
import sys
import time
from dataclasses import asdict, dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Callable, Iterable


SUMMARY_PREFIX = "real-time execution gates:"
TASK_PREFIX = "real-time task ledger event:"
EVENT_PREFIX = "real-time execution event:"
UNASSIGNED_EVENT_PREFIX = "real-time unassigned execution event:"
PHASE_PREFIX = "real-time execution phase timing:"
MISS_PREFIX = "real-time execution miss attribution:"

MINIMUM_RUNS_PER_CONDITION = 10
MINIMUM_DETERMINISTIC_RUNS_PER_CONDITION = 11
MINIMUM_RANDOM_SEEDS_PER_AMPLITUDE = 20
MINIMUM_DURATION_S = 20.0
MINIMUM_CONTACT_TASKS = 12
MINIMUM_GAIT_CYCLES = 3
MINIMUM_ROUTE_PROGRESS_M = 0.18
MINIMUM_DEADLINE_FRACTION = 0.999
MINIMUM_PLANNER_PUBLICATION_FRACTION = 0.99
MAXIMUM_PLANNER_P99_MS = 180.0
WBC_RATE_HZ = 500.0
WBC_RATE_TOLERANCE_FRACTION = 0.01

REQUIRED_PHASES = (
    "mj_step1",
    "state_reference_contact",
    "wbc_apply",
    "mj_step2",
    "housekeeping",
)
REQUIRED_MISS_SCOPES = (
    "complete_tick_work",
    "scheduled_tick_response",
)
BUILD_METADATA_FIELDS = (
    "schema_version",
    "source_git_revision",
    "source_git_tree",
    "source_worktree_clean",
    "source_worktree_status_sha256",
    "runner_sha256",
    "compiler_id",
    "compiler_version",
    "build_type",
    "cxx_flags",
    "fast_math",
    "ndebug",
    "mujoco_version",
    "mujoco_library_path",
    "mujoco_library_sha256",
)


@dataclass(frozen=True)
class TerrainCase:
    name: str
    terrain_class: str
    level: int
    arguments: tuple[str, ...]
    stress: bool = False
    random_amplitude: float | None = None

    @property
    def scope(self) -> str:
        return "stress" if self.stress else "core"

    def run_arguments(self, run_index: int, first_random_seed: int) -> list[str]:
        arguments = list(self.arguments)
        if self.random_amplitude is not None:
            arguments.extend((
                "--terrain-seed", str(first_random_seed + run_index),
            ))
        return arguments


@dataclass(frozen=True)
class GateFailure:
    gate_scope: str
    category: str
    gate: str
    expected: str
    observed: str


@dataclass
class ParsedEvidence:
    summary: dict[str, str]
    tasks: list[dict[str, str]]
    events: list[dict[str, str]]
    phases: dict[str, dict[str, str]]
    misses: dict[str, dict[str, str]]
    parse_errors: list[str]


def core_cases() -> list[TerrainCase]:
    return [
        TerrainCase("flat", "flat", 1, ("--terrain", "flat")),
        TerrainCase(
            "slope_up_10pct", "slope", 2,
            ("--terrain", "slope", "--slope-x", "0.10"),
        ),
        TerrainCase(
            "slope_down_10pct", "slope", 2,
            ("--terrain", "slope", "--slope-x", "-0.10"),
        ),
        TerrainCase(
            "cross_positive_10pct", "cross_slope", 3,
            ("--terrain", "cross-slope", "--slope-y", "0.10"),
        ),
        TerrainCase(
            "cross_negative_10pct", "cross_slope", 3,
            ("--terrain", "cross-slope", "--slope-y", "-0.10"),
        ),
        TerrainCase(
            "sinusoidal_02cm", "sinusoidal", 4,
            ("--terrain", "sinusoidal", "--terrain-amplitude", "0.02"),
        ),
        TerrainCase(
            "sinusoidal_04cm", "sinusoidal", 4,
            ("--terrain", "sinusoidal", "--terrain-amplitude", "0.04"),
        ),
        TerrainCase(
            "smooth_step_02cm", "smooth_step", 5,
            ("--terrain", "smooth_step", "--step-height", "0.02"),
        ),
        TerrainCase(
            "random_02cm", "random_smooth_02cm", 6,
            ("--terrain", "random_smooth", "--terrain-amplitude", "0.02"),
            random_amplitude=0.02,
        ),
        TerrainCase(
            "random_04cm", "random_smooth_04cm", 6,
            ("--terrain", "random_smooth", "--terrain-amplitude", "0.04"),
            random_amplitude=0.04,
        ),
    ]


def stress_cases() -> list[TerrainCase]:
    cross_grade = math.tan(math.radians(20.0))
    return [
        TerrainCase(
            "cross_positive_20deg", "cross_slope_20deg", 7,
            ("--terrain", "cross-slope", "--slope-y", str(cross_grade)),
            stress=True,
        ),
        TerrainCase(
            "cross_negative_20deg", "cross_slope_20deg", 7,
            ("--terrain", "cross-slope", "--slope-y", str(-cross_grade)),
            stress=True,
        ),
        TerrainCase(
            "sinusoidal_06cm", "sinusoidal_06cm", 7,
            ("--terrain", "sinusoidal", "--terrain-amplitude", "0.06"),
            stress=True,
        ),
        TerrainCase(
            "smooth_step_04cm", "smooth_step_04cm", 7,
            ("--terrain", "smooth_step", "--step-height", "0.04"),
            stress=True,
        ),
    ]


def campaign_cases(profile: str) -> list[TerrainCase]:
    if profile == "smoke":
        return [core_cases()[0]]
    return core_cases() + stress_cases()


def required_runs_for_case(case: TerrainCase, profile: str,
                           runs_per_condition: int) -> int:
    if profile == "development" and case.random_amplitude is not None:
        return max(runs_per_condition, MINIMUM_RANDOM_SEEDS_PER_AMPLITUDE)
    if profile == "development" and not case.stress:
        return max(runs_per_condition, MINIMUM_DETERMINISTIC_RUNS_PER_CONDITION)
    return runs_per_condition


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("runner", type=Path)
    parser.add_argument("scene", type=Path)
    parser.add_argument("output_directory", type=Path)
    parser.add_argument(
        "--profile", choices=("smoke", "development"), default="development",
    )
    parser.add_argument("--runs-per-condition", type=int)
    parser.add_argument("--duration", type=float, default=MINIMUM_DURATION_S)
    parser.add_argument("--first-random-seed", type=int, default=1000)
    parser.add_argument("--trial-timeout", type=float, default=600.0)
    parser.add_argument("--build-metadata", type=Path)
    parser.add_argument("--trace-execution", action="store_true")
    parser.add_argument("--continue-after-level-failure", action="store_true")
    parser.add_argument(
        "--plot-command-template",
        help="record a plot command; placeholders include {runs_csv} and {output_dir}",
    )
    parser.add_argument(
        "--video-command-template",
        help="record a video command; the campaign never claims or fabricates output",
    )
    args = parser.parse_args(argv)
    if args.runs_per_condition is None:
        args.runs_per_condition = (
            1 if args.profile == "smoke" else MINIMUM_RUNS_PER_CONDITION
        )
    if args.runs_per_condition < 1:
        parser.error("runs per condition must be positive")
    if args.duration <= 0.0:
        parser.error("duration must be positive")
    if args.first_random_seed < 0:
        parser.error("first random seed must be nonnegative")
    if args.trial_timeout <= 0.0:
        parser.error("trial timeout must be positive")
    return args


def parse_key_values(line: str, prefix: str) -> dict[str, str]:
    if not line.startswith(prefix):
        raise ValueError(f"record does not start with {prefix!r}")
    values: dict[str, str] = {}
    for field in line[len(prefix):].strip().split():
        if "=" not in field:
            continue
        key, value = field.split("=", 1)
        if key in values:
            raise ValueError(f"duplicate field {key!r} in {prefix!r} record")
        values[key] = value
    return values


def parse_execution_log(text: str) -> ParsedEvidence:
    summary: dict[str, str] | None = None
    tasks: list[dict[str, str]] = []
    events: list[dict[str, str]] = []
    phases: dict[str, dict[str, str]] = {}
    misses: dict[str, dict[str, str]] = {}
    parse_errors: list[str] = []
    for line_number, line in enumerate(text.splitlines(), start=1):
        try:
            if line.startswith(SUMMARY_PREFIX):
                if summary is not None:
                    raise ValueError("duplicate summary record")
                summary = parse_key_values(line, SUMMARY_PREFIX)
            elif line.startswith(TASK_PREFIX):
                tasks.append(parse_key_values(line, TASK_PREFIX))
            elif line.startswith(EVENT_PREFIX):
                events.append(parse_key_values(line, EVENT_PREFIX))
            elif line.startswith(UNASSIGNED_EVENT_PREFIX):
                events.append(parse_key_values(line, UNASSIGNED_EVENT_PREFIX))
            elif line.startswith(PHASE_PREFIX):
                record = parse_key_values(line, PHASE_PREFIX)
                phase = record.get("phase")
                if phase is None:
                    raise ValueError("phase timing record is missing phase")
                if phase in phases:
                    raise ValueError(f"duplicate phase timing record {phase!r}")
                phases[phase] = record
            elif line.startswith(MISS_PREFIX):
                record = parse_key_values(line, MISS_PREFIX)
                scope = record.get("scope")
                if scope is None:
                    raise ValueError("miss-attribution record is missing scope")
                if scope in misses:
                    raise ValueError(f"duplicate miss-attribution record {scope!r}")
                misses[scope] = record
        except ValueError as error:
            parse_errors.append(f"line {line_number}: {error}")
    return ParsedEvidence(
        summary or {}, tasks, events, phases, misses, parse_errors,
    )


def finite_number(values: dict[str, str], key: str) -> float | None:
    try:
        value = float(values[key])
    except (KeyError, TypeError, ValueError):
        return None
    return value if math.isfinite(value) else None


def exact_integer(values: dict[str, str], key: str) -> int | None:
    try:
        text = values[key]
        value = int(text)
    except (KeyError, TypeError, ValueError):
        return None
    return value if str(value) == text or text in (f"+{value}", f"-{abs(value)}") \
        else None


def evaluate_execution_evidence(
    evidence: ParsedEvidence,
    requested_duration_s: float = MINIMUM_DURATION_S,
) -> tuple[list[GateFailure], list[GateFailure]]:
    """Return (execution failures, paper-environment failures)."""
    execution: list[GateFailure] = []
    qualification: list[GateFailure] = []
    summary = evidence.summary

    def failure(
        target: list[GateFailure], category: str, gate: str,
        expected: str, observed: object,
    ) -> None:
        target.append(GateFailure(
            "execution" if target is execution else "qualification",
            category, gate, expected,
            "<missing>" if observed is None else str(observed),
        ))

    def require_string(
        key: str, expected: str, category: str = "configuration",
        target: list[GateFailure] = execution,
    ) -> None:
        observed = summary.get(key)
        if observed != expected:
            failure(target, category, key, expected, observed)

    def require_nonempty_string(
        key: str, category: str,
        target: list[GateFailure] = execution,
    ) -> str | None:
        observed = summary.get(key)
        if observed is None or not observed.strip():
            failure(target, category, key, "nonempty string", observed)
            return None
        return observed

    def require_int(
        key: str, predicate: Callable[[int], bool], expected: str,
        category: str, target: list[GateFailure] = execution,
    ) -> int | None:
        observed = exact_integer(summary, key)
        if observed is None or not predicate(observed):
            failure(target, category, key, expected, summary.get(key))
        return observed

    def require_float(
        key: str, predicate: Callable[[float], bool], expected: str,
        category: str, target: list[GateFailure] = execution,
    ) -> float | None:
        observed = finite_number(summary, key)
        if observed is None or not predicate(observed):
            failure(target, category, key, expected, summary.get(key))
        return observed

    if not summary:
        failure(execution, "evidence", "summary_record", "present", None)
        return execution, qualification

    for index, error in enumerate(evidence.parse_errors):
        failure(execution, "evidence", f"parse_error_{index}", "none", error)

    require_string("scheduling", "background_worker")
    require_int("wall_concurrent", lambda value: value == 1, "1", "configuration")
    require_int(
        "planner_updates_enabled", lambda value: value == 1, "1", "planner",
    )
    require_int("sustained_mode", lambda value: value == 1, "1", "protocol")
    require_float(
        "planner_horizon_s", lambda value: math.isclose(value, 2.5),
        "2.5", "configuration",
    )
    require_int("planner_knots", lambda value: value == 50, "50", "configuration")
    require_float(
        "planner_dt_s", lambda value: math.isclose(value, 0.05),
        "0.05", "configuration",
    )
    require_float(
        "planner_rate_hz", lambda value: math.isclose(value, 5.0),
        "5", "configuration",
    )
    require_float(
        "wbc_rate_hz", lambda value: math.isclose(value, WBC_RATE_HZ),
        "500", "configuration",
    )
    require_nonempty_string("mujoco_version", "configuration")
    require_float(
        "logical_wbc_rate_hz", lambda value: math.isclose(value, WBC_RATE_HZ),
        "500", "configuration",
    )
    require_float(
        "duration_s", lambda value: value + 1e-12 >= requested_duration_s,
        f">={requested_duration_s}", "protocol",
    )
    require_float(
        "wall_duration",
        lambda value: value + 1e-12 >= 0.99 * requested_duration_s,
        f">={0.99 * requested_duration_s}", "protocol",
    )
    require_int("duration_gate", lambda value: value == 1, "1", "protocol")

    wbc_ticks = require_int(
        "wbc_ticks",
        lambda value: value >= math.floor(
            0.99 * requested_duration_s * WBC_RATE_HZ
        ),
        f">={math.floor(0.99 * requested_duration_s * WBC_RATE_HZ)}",
        "wbc_timing",
    )
    for stem in (
        "wbc_compute", "complete_tick_work", "scheduled_tick_response",
    ):
        samples = require_int(
            f"{stem}_samples", lambda value: value > 0, ">0", "wbc_timing",
        )
        if samples is not None and wbc_ticks is not None and samples != wbc_ticks:
            failure(
                execution, "wbc_timing", f"{stem}_sample_coverage",
                f"{wbc_ticks}", samples,
            )
        within = require_int(
            f"{stem}_within_2ms", lambda value: value >= 0,
            ">=0", "wbc_timing",
        )
        fraction = require_float(
            f"{stem}_within_2ms_fraction",
            lambda value: value + 1e-15 >= MINIMUM_DEADLINE_FRACTION,
            f">={MINIMUM_DEADLINE_FRACTION}", "wbc_timing",
        )
        if samples is not None and within is not None:
            if within > samples:
                failure(
                    execution, "wbc_timing", f"{stem}_within_sample_bound",
                    f"<={samples}", within,
                )
            expected_fraction = within / samples if samples else math.nan
            if fraction is not None and not math.isclose(
                fraction, expected_fraction, rel_tol=0.0, abs_tol=5e-7
            ):
                failure(
                    execution, "wbc_timing", f"{stem}_fraction_consistency",
                    f"{expected_fraction:.9f}", fraction,
                )
        require_float(
            f"{stem}_p99_ms", lambda value: value <= 2.0,
            "<=2.0", "wbc_timing",
        )
        require_int(
            f"{stem}_gate", lambda value: value == 1, "1", "wbc_timing",
        )
    require_float(
        "actual_wall_tick_rate_hz",
        lambda value: abs(value - WBC_RATE_HZ) <=
        WBC_RATE_HZ * WBC_RATE_TOLERANCE_FRACTION,
        "495..505", "wbc_timing",
    )
    require_int("wall_rate_gate", lambda value: value == 1, "1", "wbc_timing")

    replan_opportunities = require_int(
        "replan_opportunities", lambda value: value > 0, ">0", "planner",
    )
    warm_attempts = require_int(
        "warm_attempts", lambda value: value > 0, ">0", "planner",
    )
    usable_updates = require_int(
        "usable_updates", lambda value: value >= 0, ">=0", "planner",
    )
    accepted_publications = require_int(
        "accepted_publications", lambda value: value >= 0, ">=0", "planner",
    )
    usable_fraction = require_float(
        "usable_fraction",
        lambda value: value + 1e-15 >= MINIMUM_PLANNER_PUBLICATION_FRACTION,
        f">={MINIMUM_PLANNER_PUBLICATION_FRACTION}", "planner",
    )
    publication_fraction = require_float(
        "accepted_publication_fraction",
        lambda value: value + 1e-15 >= MINIMUM_PLANNER_PUBLICATION_FRACTION,
        f">={MINIMUM_PLANNER_PUBLICATION_FRACTION}", "planner",
    )
    if replan_opportunities is not None:
        for key, value in (
            ("warm_attempts", warm_attempts),
            ("usable_updates", usable_updates),
            ("accepted_publications", accepted_publications),
        ):
            if value is not None and value > replan_opportunities:
                failure(
                    execution, "planner", f"{key}_bound",
                    f"<={replan_opportunities}", value,
                )
        for key, count, fraction in (
            ("usable_fraction", usable_updates, usable_fraction),
            ("accepted_publication_fraction", accepted_publications,
             publication_fraction),
        ):
            if count is not None and fraction is not None:
                expected_fraction = count / replan_opportunities
                if not math.isclose(
                    fraction, expected_fraction, rel_tol=0.0, abs_tol=5e-7
                ):
                    failure(
                        execution, "planner", f"{key}_consistency",
                        f"{expected_fraction:.9f}", fraction,
                    )
    require_int(
        "publication_rate_gate", lambda value: value == 1, "1", "planner",
    )
    require_string(
        "planner_latency_scope", "request_to_wbc_handoff", "planner",
    )
    require_float(
        "planner_p99_ms", lambda value: value <= MAXIMUM_PLANNER_P99_MS,
        f"<={MAXIMUM_PLANNER_P99_MS}", "planner",
    )
    require_int(
        "max_consecutive_fallbacks", lambda value: value <= 1, "<=1", "planner",
    )
    for key in ("fallbacks", "deadline_misses"):
        require_int(key, lambda value: value >= 0, ">=0", "planner")
    for key in (
        "invalid_publications", "stale_publications",
        "exact_hessian_analytic_calls", "exact_hessian_fd_calls",
        "expired_plan_ticks",
    ):
        require_int(key, lambda value: value == 0, "0", "planner")
    require_int("zero_hessian_gate", lambda value: value == 1, "1", "planner")

    commanded = require_int(
        "contact_tasks_commanded",
        lambda value: value >= MINIMUM_CONTACT_TASKS,
        f">={MINIMUM_CONTACT_TASKS}", "contact_execution",
    )
    completed = require_int(
        "contact_tasks_completed",
        lambda value: value >= MINIMUM_CONTACT_TASKS,
        f">={MINIMUM_CONTACT_TASKS}", "contact_execution",
    )
    if commanded is not None and completed is not None and completed != commanded:
        failure(
            execution, "contact_execution", "contact_task_completion",
            f"{commanded}/{commanded}", f"{completed}/{commanded}",
        )
    reported_cycles = require_int(
        "completed_gait_cycles", lambda value: value >= MINIMUM_GAIT_CYCLES,
        f">={MINIMUM_GAIT_CYCLES}", "contact_execution",
    )
    require_float(
        "route_progress_m", lambda value: value >= MINIMUM_ROUTE_PROGRESS_M,
        f">={MINIMUM_ROUTE_PROGRESS_M}", "contact_execution",
    )
    swing_events = require_int(
        "swing_events", lambda value: value >= MINIMUM_CONTACT_TASKS,
        f">={MINIMUM_CONTACT_TASKS}", "contact_execution",
    )
    if completed is not None and swing_events is not None and swing_events < completed:
        failure(
            execution, "contact_execution", "event_coverage",
            f">={completed}", swing_events,
        )
    require_int(
        "sustained_contact_execution", lambda value: value == 1,
        "1", "contact_execution",
    )
    require_string(
        "execution_evidence_scope", "sustained_multi_contact",
        "contact_execution",
    )
    require_int("missing_touchdown", lambda value: value == 0, "0", "contact_execution")
    require_float(
        "minimum_clearance", lambda value: value >= 0.02,
        ">=0.02", "contact_execution",
    )
    require_float(
        "maximum_landing_error", lambda value: value <= 0.03,
        "<=0.03", "contact_execution",
    )
    require_float(
        "maximum_slip", lambda value: value <= 0.02,
        "<=0.02", "contact_execution",
    )
    require_int(
        "saturated_joint_ticks", lambda value: value == 0,
        "0", "contact_execution",
    )
    require_int(
        "execution_gates_passed", lambda value: value == 1,
        "1", "execution",
    )
    require_int(
        "contact_execution_quality_gate", lambda value: value == 1,
        "1", "contact_execution",
    )
    require_int(
        "task_metadata_rejections", lambda value: value == 0,
        "0", "contact_execution",
    )
    accepted_transitions = require_int(
        "accepted_task_transitions", lambda value: value >= 0,
        ">=0", "contact_execution",
    )
    fresh_witnesses = require_int(
        "fresh_task_transition_witnesses", lambda value: value >= 0,
        ">=0", "contact_execution",
    )
    if (
        accepted_transitions is not None and fresh_witnesses is not None and
        fresh_witnesses != accepted_transitions
    ):
        failure(
            execution, "contact_execution", "fresh_transition_witness_coverage",
            str(accepted_transitions), fresh_witnesses,
        )

    pending = require_int(
        "pending_contact_tasks", lambda value: value == 0,
        "0 due-but-incomplete tasks", "contact_execution",
    )
    reported_censored = require_int(
        "right_censored_contact_tasks", lambda value: 0 <= value <= 1,
        "0 or 1", "contact_execution",
    )
    require_int(
        "unassigned_contact_events", lambda value: value == 0,
        "0", "contact_execution",
    )
    require_int(
        "task_ledger_mismatches", lambda value: value == 0,
        "0", "contact_execution",
    )
    require_int(
        "task_ledger_integrity_gate", lambda value: value == 1,
        "1", "contact_execution",
    )

    ledger: list[tuple[int, int, int, dict[str, str]]] = []
    seen_task_ids: set[int] = set()
    seen_sequences: set[int] = set()
    for record_index, record in enumerate(evidence.tasks):
        task_id = exact_integer(record, "task_id")
        sequence = exact_integer(record, "task_sequence")
        foot = exact_integer(record, "moving_foot")
        if task_id is None or task_id < 0:
            failure(
                execution, "task_ledger", f"task_{record_index}_id",
                "nonnegative integer", record.get("task_id"),
            )
        elif task_id in seen_task_ids:
            failure(
                execution, "task_ledger", f"task_{record_index}_id_unique",
                "unique", task_id,
            )
        else:
            seen_task_ids.add(task_id)
        if sequence is None or sequence < 0:
            failure(
                execution, "task_ledger", f"task_{record_index}_sequence",
                "nonnegative integer", record.get("task_sequence"),
            )
        elif sequence in seen_sequences:
            failure(
                execution, "task_ledger", f"task_{record_index}_sequence_unique",
                "unique", sequence,
            )
        else:
            seen_sequences.add(sequence)
        if foot is None or not 0 <= foot < 4:
            failure(
                execution, "task_ledger", f"task_{record_index}_moving_foot",
                "0..3", record.get("moving_foot"),
            )
        if task_id is None or sequence is None or foot is None:
            continue
        ledger.append((sequence, task_id, foot, record))

    ledger.sort(key=lambda item: item[0])
    due_count = 0
    completed_count = 0
    censored_count = 0
    completed_per_foot = [0, 0, 0, 0]
    gait_pattern: list[int] = []
    previous_expected_liftoff = -math.inf
    previous_planned_liftoff = -math.inf
    seen_event_indices: set[tuple[int, int]] = set()
    for ledger_index, (sequence, task_id, foot, record) in enumerate(ledger):
        if ledger_index > 0:
            previous_sequence, previous_id, _, _ = ledger[ledger_index - 1]
            if sequence != previous_sequence + 1:
                failure(
                    execution, "task_ledger", "task_sequence_contiguous",
                    str(previous_sequence + 1), sequence,
                )
            if task_id != previous_id + 1:
                failure(
                    execution, "task_ledger", "task_id_monotonic",
                    str(previous_id + 1), task_id,
                )
        if ledger_index < 4:
            gait_pattern.append(foot)
            if ledger_index == 3 and set(gait_pattern) != {0, 1, 2, 3}:
                failure(
                    execution, "task_ledger", "first_gait_cycle_foot_coverage",
                    "each foot exactly once", gait_pattern,
                )
        elif len(gait_pattern) == 4 and foot != gait_pattern[ledger_index % 4]:
            failure(
                execution, "task_ledger", f"task_{ledger_index}_gait_order",
                str(gait_pattern[ledger_index % 4]), foot,
            )
        target_x = finite_number(record, "target_x")
        if target_x is None:
            failure(
                execution, "task_ledger", f"task_{ledger_index}_target_x",
                "finite", record.get("target_x"),
            )

        due = exact_integer(record, "due")
        completed_task = exact_integer(record, "completed")
        censored = exact_integer(record, "right_censored")
        bound = exact_integer(record, "event_bound")
        missing = exact_integer(record, "missing")
        schedule_match = exact_integer(record, "schedule_match")
        for key, observed in (
            ("due", due), ("completed", completed_task),
            ("right_censored", censored), ("event_bound", bound),
            ("missing", missing), ("schedule_match", schedule_match),
        ):
            if observed not in (0, 1):
                failure(
                    execution, "task_ledger", f"task_{ledger_index}_{key}",
                    "0 or 1", record.get(key),
                )
        if any(value not in (0, 1) for value in (
            due, completed_task, censored, bound, missing, schedule_match,
        )):
            continue
        if censored:
            censored_count += 1
            if (
                due or completed_task or missing or not schedule_match or
                ledger_index != len(ledger) - 1
            ):
                failure(
                    execution, "task_ledger", f"task_{ledger_index}_censoring",
                    "only final not-due incomplete task may be censored",
                    json.dumps(record, sort_keys=True),
                )
        else:
            due_count += due
            completed_count += completed_task
            if not due or not completed_task or not bound or missing or not schedule_match:
                failure(
                    execution, "task_ledger", f"task_{ledger_index}_completion",
                    "due=1 completed=1 event_bound=1 missing=0 schedule_match=1",
                    json.dumps(record, sort_keys=True),
                )
            if completed_task:
                completed_per_foot[foot] += 1
        expected_liftoff = finite_number(record, "expected_liftoff")
        expected_touchdown = finite_number(record, "expected_touchdown")
        if (
            expected_liftoff is None or expected_touchdown is None or
            expected_liftoff < previous_expected_liftoff or
            expected_touchdown <= expected_liftoff
        ):
            failure(
                execution, "task_ledger", f"task_{ledger_index}_expected_timing",
                "ordered finite liftoff < touchdown",
                f"{record.get('expected_liftoff')},{record.get('expected_touchdown')}",
            )
        elif expected_liftoff is not None:
            previous_expected_liftoff = expected_liftoff
        if bound:
            event_index = exact_integer(record, "event_index")
            if event_index is None or event_index < 0:
                failure(
                    execution, "task_ledger", f"task_{ledger_index}_event_index",
                    "nonnegative integer", record.get("event_index"),
                )
            elif (foot, event_index) in seen_event_indices:
                failure(
                    execution, "task_ledger",
                    f"task_{ledger_index}_event_index_unique",
                    "unique", event_index,
                )
            else:
                seen_event_indices.add((foot, event_index))
        if bound and completed_task:
            event_times: dict[str, float | None] = {}
            for key, predicate, expected in (
                ("planned_liftoff", lambda value: value >= 0.0, ">=0"),
                ("measured_liftoff", lambda value: value >= 0.0, ">=0"),
                ("planned_touchdown", lambda value: value >= 0.0, ">=0"),
                ("measured_touchdown", lambda value: value >= 0.0, ">=0"),
                ("clearance", lambda value: value >= 0.02, ">=0.02"),
                ("landing_error", lambda value: 0.0 <= value <= 0.03, "0..0.03"),
                ("slip", lambda value: 0.0 <= value <= 0.02, "0..0.02"),
            ):
                observed = finite_number(record, key)
                event_times[key] = observed
                if observed is None or not predicate(observed):
                    failure(
                        execution, "task_ledger", f"task_{ledger_index}_{key}",
                        expected, record.get(key),
                    )
            planned_liftoff = event_times.get("planned_liftoff")
            measured_liftoff = event_times.get("measured_liftoff")
            planned_touchdown = event_times.get("planned_touchdown")
            measured_touchdown = event_times.get("measured_touchdown")
            if (
                planned_liftoff is not None and planned_touchdown is not None and
                (planned_touchdown <= planned_liftoff or
                 planned_liftoff < previous_planned_liftoff)
            ):
                failure(
                    execution, "task_ledger", f"task_{ledger_index}_planned_order",
                    "globally ordered liftoff < touchdown",
                    f"{planned_liftoff},{planned_touchdown}",
                )
            elif planned_liftoff is not None:
                previous_planned_liftoff = planned_liftoff
            if (
                measured_liftoff is not None and measured_touchdown is not None and
                measured_touchdown <= measured_liftoff
            ):
                failure(
                    execution, "task_ledger", f"task_{ledger_index}_measured_order",
                    "liftoff < touchdown",
                    f"{measured_liftoff},{measured_touchdown}",
                )
        elif bound:
            planned_liftoff = finite_number(record, "planned_liftoff")
            if planned_liftoff is None or planned_liftoff < 0.0:
                failure(
                    execution, "task_ledger", f"task_{ledger_index}_planned_liftoff",
                    ">=0 for a bound partial event", record.get("planned_liftoff"),
                )

    derived_cycles = min(completed_per_foot) if ledger else 0
    expected_transitions = max(0, len(ledger) - 1)
    for gate, expected, observed in (
        ("task_ledger_commanded", commanded, due_count),
        ("task_ledger_completed", completed, completed_count),
        ("task_ledger_pending", pending, due_count - completed_count),
        ("task_ledger_censored", reported_censored, censored_count),
        ("task_ledger_cycles", reported_cycles, derived_cycles),
        ("task_ledger_transitions", accepted_transitions, expected_transitions),
        ("task_ledger_row_count", swing_events, len(ledger)),
    ):
        if expected is None:
            failure(execution, "task_ledger", gate, "summary count present", None)
        elif observed != expected:
            failure(execution, "task_ledger", gate, str(expected), observed)

    for phase in REQUIRED_PHASES:
        record = evidence.phases.get(phase)
        if record is None:
            failure(execution, "timing_attribution", f"phase_{phase}", "present", None)
            continue
        for key in ("samples", "thread_cpu_samples"):
            observed = exact_integer(record, key)
            if observed is None or observed <= 0:
                failure(
                    execution, "timing_attribution", f"{phase}_{key}",
                    ">0", record.get(key),
                )
            elif wbc_ticks is not None and observed != wbc_ticks:
                failure(
                    execution, "timing_attribution", f"{phase}_{key}_coverage",
                    str(wbc_ticks), observed,
                )
        for key in (
            "wall_p50_ms", "wall_p90_ms", "wall_p99_ms", "wall_max_ms",
            "thread_cpu_p50_ms", "thread_cpu_p90_ms", "thread_cpu_p99_ms",
            "thread_cpu_max_ms",
        ):
            observed = finite_number(record, key)
            if observed is None or observed < 0.0:
                failure(
                    execution, "timing_attribution", f"{phase}_{key}",
                    "finite and >=0", record.get(key),
                )

    for scope in REQUIRED_MISS_SCOPES:
        record = evidence.misses.get(scope)
        if record is None:
            failure(
                execution, "timing_attribution", f"miss_scope_{scope}",
                "present", None,
            )
            continue
        counts = [exact_integer(record, key) for key in (
            "misses", "mujoco_physics", "wbc_or_control_work",
            "scheduling_or_blocking", "thread_cpu_time_unavailable",
        )]
        if any(value is None or value < 0 for value in counts):
            failure(
                execution, "timing_attribution", f"miss_counts_{scope}",
                "nonnegative integers", json.dumps(record, sort_keys=True),
            )
        elif counts[0] != sum(counts[1:]):
            failure(
                execution, "timing_attribution", f"miss_sum_{scope}",
                str(counts[0]), sum(counts[1:]),
            )

    require_string(
        "environment", "linux", "environment", qualification,
    )
    for key in (
        "separate_logical_cpus", "wbc_affinity_applied",
        "planner_scheduling_reported", "planner_affinity_applied",
        "logical_thread_isolation_applied", "wbc_priority_applied",
        "planner_priority_applied", "physical_core_isolation_verified",
        "realtime_priorities_applied", "paper_grade_scheduling_gate", "accepted",
    ):
        require_int(
            key, lambda value: value == 1, "1", "environment", qualification,
        )
    require_int(
        "environment_limited", lambda value: value == 0,
        "0", "environment", qualification,
    )
    require_string(
        "paper_grade_verdict", "paper_grade_pass", "environment", qualification,
    )
    return execution, qualification


def wilson_interval(successes: int, trials: int, z: float = 1.959963984540054) \
        -> tuple[float, float]:
    if trials <= 0:
        return math.nan, math.nan
    fraction = successes / trials
    denominator = 1.0 + z * z / trials
    center = (fraction + z * z / (2.0 * trials)) / denominator
    margin = z / denominator * math.sqrt(
        fraction * (1.0 - fraction) / trials + z * z / (4.0 * trials * trials)
    )
    return center - margin, center + margin


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def is_hex_digest(value: object, lengths: tuple[int, ...]) -> bool:
    return isinstance(value, str) and len(value) in lengths and all(
        character in "0123456789abcdef" for character in value.lower()
    )


def run_git(repo: Path, *arguments: str, binary: bool = False) -> str | bytes | None:
    completed = subprocess.run(
        ("git", *arguments), cwd=repo, capture_output=True,
        text=not binary, check=False,
    )
    return completed.stdout if completed.returncode == 0 else None


def repository_root(start: Path) -> Path | None:
    completed = subprocess.run(
        ("git", "rev-parse", "--show-toplevel"), cwd=start,
        capture_output=True, text=True, check=False,
    )
    return Path(completed.stdout.strip()).resolve() \
        if completed.returncode == 0 else None


def load_build_metadata(path: Path | None) -> tuple[dict[str, object] | None, list[str]]:
    if path is None:
        return None, ["build metadata file was not supplied"]
    try:
        metadata = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError) as error:
        return None, [f"could not read build metadata: {error}"]
    if not isinstance(metadata, dict):
        return None, ["build metadata must be a JSON object"]
    errors = [f"missing build metadata field {key}" for key in BUILD_METADATA_FIELDS
              if key not in metadata]
    if metadata.get("schema_version") != 2:
        errors.append("build metadata schema_version must be 2")
    for key in (
        "source_git_revision", "source_git_tree", "compiler_id",
        "compiler_version", "build_type", "cxx_flags", "mujoco_version",
        "mujoco_library_path",
    ):
        if not isinstance(metadata.get(key), str) or not metadata.get(key):
            errors.append(f"build metadata {key} must be a nonempty string")
    for key in ("source_worktree_clean", "fast_math", "ndebug"):
        if not isinstance(metadata.get(key), bool):
            errors.append(f"build metadata {key} must be boolean")
    if metadata.get("source_worktree_clean") is not True:
        errors.append("build metadata source_worktree_clean must be true")
    if not is_hex_digest(metadata.get("source_git_revision"), (40, 64)):
        errors.append("build metadata source_git_revision must be a git object id")
    if not is_hex_digest(metadata.get("source_git_tree"), (40, 64)):
        errors.append("build metadata source_git_tree must be a git object id")
    for key in (
        "source_worktree_status_sha256", "runner_sha256",
        "mujoco_library_sha256",
    ):
        if not is_hex_digest(metadata.get(key), (64,)):
            errors.append(f"build metadata {key} must be a SHA-256 digest")
    return metadata, errors


def collect_provenance(
    repo: Path | None, runner: Path, scene: Path, metadata_path: Path | None,
) -> dict[str, object]:
    provenance: dict[str, object] = {
        "captured_utc": datetime.now(timezone.utc).isoformat(),
        "runner_path": str(runner),
        "runner_sha256": sha256_file(runner),
        "runner_size_bytes": runner.stat().st_size,
        "scene_path": str(scene),
        "scene_sha256": sha256_file(scene),
        "scene_size_bytes": scene.stat().st_size,
        "platform": platform.platform(),
        "system": platform.system().lower(),
        "machine": platform.machine(),
        "processor": platform.processor(),
        "python_version": platform.python_version(),
        "cpu_count": os.cpu_count(),
    }
    if metadata_path is not None and metadata_path.is_file():
        provenance.update({
            "build_metadata_path": str(metadata_path),
            "build_metadata_sha256": sha256_file(metadata_path),
            "build_metadata_size_bytes": metadata_path.stat().st_size,
        })
    else:
        provenance["build_metadata_sha256"] = None
    if repo is None:
        provenance.update({
            "git_revision": "unknown",
            "git_tree": "unknown",
            "worktree_status": "unknown",
            "worktree_status_sha256": "unknown",
            "worktree_diff_sha256": "unknown",
        })
        return provenance
    revision = run_git(repo, "rev-parse", "HEAD")
    tree = run_git(repo, "rev-parse", "HEAD^{tree}")
    status = run_git(repo, "status", "--porcelain=v1")
    diff = run_git(repo, "diff", "--binary", "HEAD", "--", binary=True)
    provenance.update({
        "repository": str(repo),
        "git_revision": revision.strip() if isinstance(revision, str) else "unknown",
        "git_tree": tree.strip() if isinstance(tree, str) else "unknown",
        "worktree_status": (
            "clean" if status == "" else "dirty"
            if isinstance(status, str) else "unknown"
        ),
        "worktree_status_sha256": (
            hashlib.sha256(status.encode("utf-8")).hexdigest()
            if isinstance(status, str) else "unknown"
        ),
        "worktree_diff_sha256": (
            hashlib.sha256(diff).hexdigest()
            if isinstance(diff, bytes) else "unknown"
        ),
    })
    return provenance


def provenance_matches(initial: dict[str, object], final: dict[str, object]) -> bool:
    keys = (
        "runner_sha256", "runner_size_bytes", "scene_sha256", "scene_size_bytes",
        "build_metadata_sha256", "git_revision", "worktree_status_sha256",
        "git_tree", "worktree_diff_sha256", "mujoco_library_sha256",
        "platform", "machine",
        "processor", "cpu_count",
    )
    return all(
        initial.get(key) not in (None, "unknown") and
        initial.get(key) == final.get(key)
        for key in keys
    )


def write_csv(path: Path, rows: list[dict[str, object]], fields: Iterable[str]) -> None:
    fieldnames = list(fields)
    for row in rows:
        for key in row:
            if key not in fieldnames:
                fieldnames.append(key)
    temporary = path.with_suffix(path.suffix + ".tmp")
    with temporary.open("w", newline="", encoding="utf-8") as output:
        writer = csv.DictWriter(output, fieldnames=fieldnames, extrasaction="ignore")
        writer.writeheader()
        writer.writerows(rows)
    temporary.replace(path)


def condition_summaries(
    rows: list[dict[str, object]], expected_runs: dict[str, int] | int,
) -> list[dict[str, object]]:
    grouped: dict[str, list[dict[str, object]]] = {}
    for row in rows:
        grouped.setdefault(str(row["case"]), []).append(row)
    summaries: list[dict[str, object]] = []
    for case_name, members in sorted(grouped.items()):
        required_runs = expected_runs[case_name] \
            if isinstance(expected_runs, dict) else expected_runs
        execution_successes = sum(int(row["execution_pass"]) for row in members)
        qualification_successes = sum(
            int(row["qualification_pass"]) for row in members
        )
        execution_interval = wilson_interval(execution_successes, len(members))
        qualification_interval = wilson_interval(
            qualification_successes, len(members)
        )
        summaries.append({
            "case": case_name,
            "terrain_class": members[0]["terrain_class"],
            "scope": members[0]["scope"],
            "runs": len(members),
            "required_runs": required_runs,
            "execution_successes": execution_successes,
            "execution_fraction": execution_successes / len(members),
            "execution_wilson95_low": execution_interval[0],
            "execution_wilson95_high": execution_interval[1],
            "qualification_successes": qualification_successes,
            "qualification_fraction": qualification_successes / len(members),
            "qualification_wilson95_low": qualification_interval[0],
            "qualification_wilson95_high": qualification_interval[1],
            "execution_verdict": (
                "PASS" if len(members) >= required_runs and
                execution_successes == len(members) else "FAIL"
            ),
            "qualification_verdict": (
                "PASS" if len(members) >= required_runs and
                qualification_successes == len(members) else "FAIL"
            ),
        })
    return summaries


def deterministic_class_publications(
    rows: list[dict[str, object]],
) -> tuple[dict[str, int], list[str]]:
    totals: dict[str, int] = {}
    errors: list[str] = []
    random_classes = {
        case.terrain_class for case in core_cases()
        if case.random_amplitude is not None
    }
    for row in rows:
        terrain_class = str(row["terrain_class"])
        if row.get("scope") != "core" or terrain_class in random_classes:
            continue
        text = str(row.get("accepted_publications", ""))
        try:
            publications = int(text)
        except ValueError:
            errors.append(
                f"{row.get('case')} run {row.get('run')} has invalid "
                "accepted_publications",
            )
            continue
        if str(publications) != text or publications < 0:
            errors.append(
                f"{row.get('case')} run {row.get('run')} has invalid "
                "accepted_publications",
            )
            continue
        totals[terrain_class] = totals.get(terrain_class, 0) + publications
    return totals, errors


def artifact_hooks(
    args: argparse.Namespace, output_paths: dict[str, Path],
) -> list[dict[str, object]]:
    hooks: list[dict[str, object]] = []
    substitutions = {
        key: str(value.resolve()) for key, value in output_paths.items()
    }
    substitutions.update({
        "output_dir": str(args.output_directory.resolve()),
        "runner": str(args.runner.resolve()),
        "scene": str(args.scene.resolve()),
    })
    for kind, template in (
        ("plot", args.plot_command_template),
        ("video", args.video_command_template),
    ):
        if not template:
            hooks.append({"kind": kind, "status": "not_requested"})
            continue
        try:
            command_text = template.format(**substitutions)
            command = shlex.split(command_text, posix=os.name != "nt")
        except (KeyError, ValueError) as error:
            hooks.append({
                "kind": kind, "status": "invalid_template", "error": str(error),
            })
            continue
        record: dict[str, object] = {
            "kind": kind,
            "command": command,
            "status": "ready_not_run",
            "note": "No artifact is claimed until this hook succeeds.",
        }
        hooks.append(record)
    return hooks


def execute_run(
    args: argparse.Namespace, case: TerrainCase, run_index: int,
    log_directory: Path, expected_mujoco_version: str | None = None,
) -> tuple[dict[str, object], list[dict[str, object]],
           list[dict[str, object]], list[dict[str, object]],
           list[dict[str, object]], list[dict[str, object]]]:
    command = [
        str(args.runner), str(args.scene),
        *case.run_arguments(run_index, args.first_random_seed),
        "--realtime-replan-duration", str(args.duration),
        "--realtime-sustained",
    ]
    if args.trace_execution:
        command.append("--trace-execution")
    random_seed = (
        args.first_random_seed + run_index
        if case.random_amplitude is not None else 0
    )
    log_path = log_directory / f"{case.name}_run_{run_index + 1:02d}.log"
    started = datetime.now(timezone.utc).isoformat()
    wall_start = time.perf_counter()
    timed_out = False
    return_code: int | None = None
    output = ""
    try:
        completed = subprocess.run(
            command, capture_output=True, text=True,
            timeout=args.trial_timeout, check=False,
        )
        return_code = completed.returncode
        output = completed.stdout + completed.stderr
    except subprocess.TimeoutExpired as error:
        timed_out = True
        captured = error.stdout or ""
        if isinstance(captured, bytes):
            captured = captured.decode("utf-8", errors="replace")
        captured_error = error.stderr or ""
        if isinstance(captured_error, bytes):
            captured_error = captured_error.decode("utf-8", errors="replace")
        output = captured + captured_error
    wall_seconds = time.perf_counter() - wall_start
    log_path.write_text(output, encoding="utf-8")
    evidence = parse_execution_log(output)
    execution_failures, qualification_failures = evaluate_execution_evidence(
        evidence, args.duration,
    )
    if (
        expected_mujoco_version is not None and
        evidence.summary.get("mujoco_version") != expected_mujoco_version
    ):
        qualification_failures.append(GateFailure(
            "qualification", "provenance", "runtime_mujoco_version",
            expected_mujoco_version,
            evidence.summary.get("mujoco_version", "<missing>"),
        ))
    if timed_out:
        execution_failures.append(GateFailure(
            "execution", "process", "trial_timeout", "not timed out", "timed out",
        ))
    if return_code != 0:
        qualification_failures.append(GateFailure(
            "qualification", "process", "runner_return_code", "0", str(return_code),
        ))
    execution_pass = not execution_failures
    qualification_pass = execution_pass and not qualification_failures
    row: dict[str, object] = {
        "case": case.name,
        "terrain_class": case.terrain_class,
        "scope": case.scope,
        "level": case.level,
        "run": run_index + 1,
        "terrain_seed": random_seed,
        "command": json.dumps(command),
        "started_utc": started,
        "trial_wall_seconds": wall_seconds,
        "timed_out": int(timed_out),
        "return_code": "" if return_code is None else return_code,
        "log": str(log_path),
        "execution_pass": int(execution_pass),
        "qualification_pass": int(qualification_pass),
        "execution_failure_count": len(execution_failures),
        "qualification_failure_count": len(qualification_failures),
    }
    row.update(evidence.summary)
    failure_rows = [
        {
            "case": case.name,
            "terrain_class": case.terrain_class,
            "scope": case.scope,
            "run": run_index + 1,
            "terrain_seed": random_seed,
            **asdict(item),
            "log": str(log_path),
        }
        for item in execution_failures + qualification_failures
    ]
    task_rows = [
        {
            "case": case.name, "scope": case.scope, "run": run_index + 1,
            "terrain_seed": random_seed, **record,
        }
        for record in evidence.tasks
    ]
    event_rows = [
        {
            "case": case.name, "scope": case.scope, "run": run_index + 1,
            "terrain_seed": random_seed, **record,
        }
        for record in evidence.events
    ]
    phase_rows = [
        {
            "case": case.name, "scope": case.scope, "run": run_index + 1,
            "terrain_seed": random_seed, **record,
        }
        for record in evidence.phases.values()
    ]
    miss_rows = [
        {
            "case": case.name, "scope": case.scope, "run": run_index + 1,
            "terrain_seed": random_seed, **record,
        }
        for record in evidence.misses.values()
    ]
    return row, failure_rows, task_rows, event_rows, phase_rows, miss_rows


def main(argv: list[str] | None = None) -> int:
    args = parse_args(argv)
    args.runner = args.runner.resolve()
    args.scene = args.scene.resolve()
    args.output_directory = args.output_directory.resolve()
    if args.build_metadata is not None:
        args.build_metadata = args.build_metadata.resolve()
    if not args.runner.is_file() or not args.scene.is_file():
        raise SystemExit("runner and scene must both exist")
    args.output_directory.mkdir(parents=True, exist_ok=True)
    log_directory = args.output_directory / "logs"
    log_directory.mkdir(exist_ok=True)
    repo = repository_root(Path(__file__).resolve().parent)
    metadata, metadata_errors = load_build_metadata(args.build_metadata)
    initial_provenance = collect_provenance(
        repo, args.runner, args.scene, args.build_metadata,
    )
    qualified_mujoco_library: Path | None = None
    if metadata is not None:
        for metadata_key, provenance_key, label in (
            ("source_git_revision", "git_revision", "source revision"),
            ("source_git_tree", "git_tree", "source tree"),
            ("source_worktree_status_sha256", "worktree_status_sha256",
             "build-time source status"),
            ("runner_sha256", "runner_sha256", "runner binary"),
        ):
            if metadata.get(metadata_key) != initial_provenance.get(provenance_key):
                metadata_errors.append(
                    f"build metadata {label} does not match the qualification input"
                )
        library_text = metadata.get("mujoco_library_path")
        library_path = Path(str(library_text))
        if not library_path.is_absolute() and args.build_metadata is not None:
            library_path = args.build_metadata.parent / library_path
        qualified_mujoco_library = library_path.resolve()
        if not library_path.is_file():
            metadata_errors.append("build metadata MuJoCo library does not exist")
            initial_provenance["mujoco_library_sha256"] = "unknown"
        else:
            observed_library_sha256 = sha256_file(library_path)
            initial_provenance["mujoco_library_sha256"] = observed_library_sha256
            if observed_library_sha256 != metadata.get("mujoco_library_sha256"):
                metadata_errors.append(
                    "build metadata MuJoCo library hash does not match the library"
                )
    else:
        initial_provenance["mujoco_library_sha256"] = "unknown"

    rows: list[dict[str, object]] = []
    failures: list[dict[str, object]] = []
    tasks: list[dict[str, object]] = []
    events: list[dict[str, object]] = []
    phases: list[dict[str, object]] = []
    misses: list[dict[str, object]] = []
    skipped_cases: list[dict[str, object]] = []
    cases = campaign_cases(args.profile)
    required_runs = {
        case.name: required_runs_for_case(
            case, args.profile, args.runs_per_condition,
        )
        for case in cases
    }
    frozen_random_seeds = {
        case.name: [
            args.first_random_seed + index
            for index in range(required_runs[case.name])
        ]
        for case in cases if case.random_amplitude is not None
    }
    frozen_seed_payload = json.dumps(
        frozen_random_seeds, sort_keys=True, separators=(",", ":"),
    ).encode("utf-8")
    frozen_random_seed_sha256 = hashlib.sha256(frozen_seed_payload).hexdigest()
    stop_after_level = False
    for level in sorted({case.level for case in cases}):
        level_cases = [case for case in cases if case.level == level]
        if stop_after_level:
            skipped_cases.extend({
                "case": case.name, "scope": case.scope, "level": case.level,
                "reason": "previous execution gate level failed",
            } for case in level_cases)
            continue
        level_rows: list[dict[str, object]] = []
        for case in level_cases:
            case_runs = required_runs[case.name]
            for run_index in range(case_runs):
                (
                    row, run_failures, run_tasks, run_events,
                    run_phases, run_misses,
                ) = execute_run(
                    args, case, run_index, log_directory,
                    str(metadata["mujoco_version"]) if metadata is not None else None,
                )
                rows.append(row)
                level_rows.append(row)
                failures.extend(run_failures)
                tasks.extend(run_tasks)
                events.extend(run_events)
                phases.extend(run_phases)
                misses.extend(run_misses)
                print(
                    f"{case.name} run {run_index + 1}/{case_runs}: "
                    f"execution={'PASS' if row['execution_pass'] else 'FAIL'} "
                    f"qualification={'PASS' if row['qualification_pass'] else 'FAIL'}"
                )
        if (
            any(int(row["execution_pass"]) == 0 for row in level_rows)
            and not args.continue_after_level_failure
        ):
            stop_after_level = True

    output_paths = {
        "runs_csv": args.output_directory / "runs.csv",
        "failures_csv": args.output_directory / "failures.csv",
        "task_ledger_csv": args.output_directory / "task_ledger.csv",
        "events_csv": args.output_directory / "events.csv",
        "phase_timing_csv": args.output_directory / "phase_timing.csv",
        "miss_attribution_csv": args.output_directory / "miss_attribution.csv",
        "condition_summary_csv": args.output_directory / "condition_summary.csv",
        "campaign_json": args.output_directory / "campaign.json",
    }
    summaries = condition_summaries(rows, required_runs)
    deterministic_publications, deterministic_publication_errors = \
        deterministic_class_publications(rows)
    expected_deterministic_classes = {
        case.terrain_class for case in core_cases()
        if case.random_amplitude is None
    }
    deterministic_publication_pass = (
        not deterministic_publication_errors and
        set(deterministic_publications) == expected_deterministic_classes and
        all(value >= 1000 for value in deterministic_publications.values())
    )
    random_seed_sets = {
        case.name: sorted({
            int(row["terrain_seed"]) for row in rows
            if row["case"] == case.name
        })
        for case in core_cases() if case.random_amplitude is not None
    }
    random_seed_protocol_pass = (
        args.profile == "development" and
        all(
            len(seeds) >= MINIMUM_RANDOM_SEEDS_PER_AMPLITUDE and
            seeds == frozen_random_seeds.get(case_name, [])
            for case_name, seeds in random_seed_sets.items()
        ) and
        len(random_seed_sets) == len([
            case for case in core_cases() if case.random_amplitude is not None
        ])
    )
    for error in deterministic_publication_errors:
        failures.append({
            "case": "<campaign>", "terrain_class": "deterministic",
            "scope": "core", "run": "", "terrain_seed": "",
            "gate_scope": "qualification", "category": "planner",
            "gate": "deterministic_publication_record",
            "expected": "valid accepted publication count", "observed": error,
            "log": "",
        })
    for terrain_class in sorted(expected_deterministic_classes):
        observed = deterministic_publications.get(terrain_class, 0)
        if observed < 1000:
            failures.append({
                "case": "<campaign>", "terrain_class": terrain_class,
                "scope": "core", "run": "", "terrain_seed": "",
                "gate_scope": "qualification", "category": "planner",
                "gate": "deterministic_class_publications",
                "expected": ">=1000", "observed": observed, "log": "",
            })
    if args.profile == "development" and not random_seed_protocol_pass:
        failures.append({
            "case": "<campaign>", "terrain_class": "random_smooth",
            "scope": "core", "run": "", "terrain_seed": "",
            "gate_scope": "qualification", "category": "protocol",
            "gate": "frozen_random_seed_matrix",
            "expected": "20 distinct frozen seeds per amplitude",
            "observed": json.dumps(random_seed_sets, sort_keys=True), "log": "",
        })
    write_csv(output_paths["runs_csv"], rows, (
        "case", "terrain_class", "scope", "level", "run", "terrain_seed",
        "execution_pass", "qualification_pass", "return_code", "timed_out", "log",
    ))
    write_csv(output_paths["failures_csv"], failures, (
        "case", "terrain_class", "scope", "run", "terrain_seed", "gate_scope",
        "category", "gate", "expected", "observed", "log",
    ))
    write_csv(output_paths["task_ledger_csv"], tasks, (
        "case", "scope", "run", "terrain_seed", "task_sequence", "task_id",
        "moving_foot", "target_x", "expected_liftoff", "expected_touchdown",
        "event_bound", "event_index", "planned_liftoff", "measured_liftoff",
        "planned_touchdown", "measured_touchdown", "clearance",
        "landing_error", "slip", "due", "completed", "right_censored",
        "missing", "schedule_match",
    ))
    write_csv(output_paths["events_csv"], events, (
        "case", "scope", "run", "terrain_seed", "index", "unload",
        "planned_touchdown", "measured_touchdown", "clearance", "landing_error",
        "slip", "missing",
    ))
    write_csv(output_paths["phase_timing_csv"], phases, (
        "case", "scope", "run", "terrain_seed", "phase", "samples",
        "wall_p50_ms", "wall_p90_ms", "wall_p99_ms", "wall_max_ms",
        "thread_cpu_samples", "thread_cpu_p50_ms", "thread_cpu_p90_ms",
        "thread_cpu_p99_ms", "thread_cpu_max_ms",
    ))
    write_csv(output_paths["miss_attribution_csv"], misses, (
        "case", "scope", "run", "terrain_seed", "misses", "mujoco_physics",
        "wbc_or_control_work", "scheduling_or_blocking",
        "thread_cpu_time_unavailable",
    ))
    write_csv(output_paths["condition_summary_csv"], summaries, (
        "case", "terrain_class", "scope", "runs", "required_runs",
        "execution_successes", "execution_fraction", "execution_wilson95_low",
        "execution_wilson95_high", "qualification_successes",
        "qualification_fraction", "qualification_wilson95_low",
        "qualification_wilson95_high", "execution_verdict",
        "qualification_verdict",
    ))

    hooks = artifact_hooks(args, output_paths)
    (args.output_directory / "artifact_hooks.json").write_text(
        json.dumps(hooks, indent=2, sort_keys=True) + "\n", encoding="utf-8",
    )
    final_provenance = collect_provenance(
        repo, args.runner, args.scene, args.build_metadata,
    )
    final_provenance["mujoco_library_sha256"] = (
        sha256_file(qualified_mujoco_library)
        if qualified_mujoco_library is not None and
        qualified_mujoco_library.is_file() else "unknown"
    )
    stable_provenance = provenance_matches(initial_provenance, final_provenance)
    clean_provenance = (
        initial_provenance.get("worktree_status") == "clean" and
        final_provenance.get("worktree_status") == "clean"
    )
    metadata_complete = not metadata_errors
    protocol_qualified = (
        args.profile == "development" and
        args.runs_per_condition >= MINIMUM_RUNS_PER_CONDITION and
        args.duration + 1e-12 >= MINIMUM_DURATION_S and
        random_seed_protocol_pass and deterministic_publication_pass
    )
    core_summaries = [row for row in summaries if row["scope"] == "core"]
    stress_summaries = [row for row in summaries if row["scope"] == "stress"]
    expected_core_cases = len(core_cases()) if args.profile == "development" else 1
    expected_stress_cases = len(stress_cases()) if args.profile == "development" else 0
    core_execution_pass = (
        len(core_summaries) == expected_core_cases and
        all(row["execution_verdict"] == "PASS" for row in core_summaries)
    )
    core_qualification_pass = (
        len(core_summaries) == expected_core_cases and
        all(row["qualification_verdict"] == "PASS" for row in core_summaries)
    )
    stress_execution_pass = (
        len(stress_summaries) == expected_stress_cases and
        all(row["execution_verdict"] == "PASS" for row in stress_summaries)
    )
    if not protocol_qualified:
        architecture_verdict = "development_only_protocol"
    elif not metadata_complete:
        architecture_verdict = "development_only_invalid_build_attestation"
    elif initial_provenance.get("system") != "linux":
        architecture_verdict = "development_only_host_not_linux"
    elif not stable_provenance:
        architecture_verdict = "development_only_unstable_provenance"
    elif not clean_provenance:
        architecture_verdict = "development_only_dirty_worktree"
    elif not core_execution_pass:
        architecture_verdict = "core_execution_failed"
    elif not core_qualification_pass:
        environments = {str(row.get("environment", "missing")) for row in rows}
        architecture_verdict = (
            "development_only_wsl" if "wsl" in environments
            else "core_qualification_failed"
        )
    else:
        architecture_verdict = "architecture_qualification_pass"
    artifact_status = {str(hook["kind"]): hook["status"] for hook in hooks}
    artifact_evidence_complete = all(
        artifact_status.get(kind) == "succeeded" for kind in ("plot", "video")
    )
    if (
        architecture_verdict == "architecture_qualification_pass" and
        artifact_evidence_complete
    ):
        paper_verdict = "paper_grade_pass"
    elif architecture_verdict == "architecture_qualification_pass":
        paper_verdict = "architecture_only_artifacts_incomplete"
    else:
        paper_verdict = "not_ready_architecture_not_qualified"
    campaign = {
        "schema_version": 2,
        "profile": args.profile,
        "runs_per_condition": args.runs_per_condition,
        "required_runs_by_case": required_runs,
        "duration_s": args.duration,
        "trial_timeout_s": args.trial_timeout,
        "first_random_seed": args.first_random_seed,
        "frozen_random_seeds": frozen_random_seeds,
        "frozen_random_seed_sha256": frozen_random_seed_sha256,
        "random_seed_protocol_pass": random_seed_protocol_pass,
        "deterministic_class_publications": deterministic_publications,
        "deterministic_publication_protocol_pass": (
            deterministic_publication_pass
        ),
        "protocol_qualified": protocol_qualified,
        "minimums": {
            "runs_per_condition": MINIMUM_RUNS_PER_CONDITION,
            "deterministic_runs_per_condition": (
                MINIMUM_DETERMINISTIC_RUNS_PER_CONDITION
            ),
            "random_seeds_per_amplitude": MINIMUM_RANDOM_SEEDS_PER_AMPLITUDE,
            "deterministic_publications_per_class": 1000,
            "duration_s": MINIMUM_DURATION_S,
            "contact_tasks": MINIMUM_CONTACT_TASKS,
            "gait_cycles": MINIMUM_GAIT_CYCLES,
            "route_progress_m": MINIMUM_ROUTE_PROGRESS_M,
        },
        "core_execution_verdict": "PASS" if core_execution_pass else "FAIL",
        "core_qualification_verdict": (
            "PASS" if core_qualification_pass else "FAIL"
        ),
        "stress_verdict": "PASS" if stress_execution_pass else "FAIL",
        "stress_affects_architecture_verdict": False,
        "stress_affects_paper_verdict": False,
        "architecture_qualification_verdict": architecture_verdict,
        "artifact_evidence_complete": artifact_evidence_complete,
        "paper_grade_verdict": paper_verdict,
        "metadata": metadata,
        "metadata_errors": metadata_errors,
        "initial_provenance": initial_provenance,
        "final_provenance": final_provenance,
        "provenance_stable": stable_provenance,
        "provenance_clean": clean_provenance,
        "skipped_cases": skipped_cases,
        "condition_summaries": summaries,
        "runs": rows,
        "failure_count": len(failures),
        "artifact_hooks": hooks,
        "output_files": {key: str(path) for key, path in output_paths.items()},
    }
    output_paths["campaign_json"].write_text(
        json.dumps(campaign, indent=2, sort_keys=True) + "\n", encoding="utf-8",
    )
    print(
        f"core_execution={campaign['core_execution_verdict']} "
        f"core_qualification={campaign['core_qualification_verdict']} "
        f"stress={campaign['stress_verdict']} "
        f"architecture={architecture_verdict} paper={paper_verdict}"
    )
    return 0 if architecture_verdict == "architecture_qualification_pass" else 1


if __name__ == "__main__":
    raise SystemExit(main())
