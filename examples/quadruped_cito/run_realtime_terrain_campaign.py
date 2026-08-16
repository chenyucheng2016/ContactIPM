#!/usr/bin/env python3
"""Run the N=50 ContactIPM real-time benchmark over a frozen terrain matrix."""

from __future__ import annotations

import argparse
import csv
import hashlib
import json
import math
import platform
import subprocess
import sys
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Iterable


MINIMUM_ACCEPTANCE_UPDATES_PER_CLASS = 1000
MINIMUM_ACCEPTANCE_RANDOM_SEEDS = 20
MINIMUM_ACCEPTANCE_RANDOM_UPDATES_PER_SEED = 100
FIRST_HELD_OUT_RANDOM_SEED = 2000
LAST_HELD_OUT_RANDOM_SEED = (
    FIRST_HELD_OUT_RANDOM_SEED + MINIMUM_ACCEPTANCE_RANDOM_SEEDS - 1
)
REQUIRED_RANDOM_AMPLITUDES = (0.02, 0.04)
REALTIME_TIME_STEP_S = 0.05
REALTIME_SWING_LAST_STAGE = 39


@dataclass(frozen=True)
class Case:
    name: str
    terrain_class: str
    arguments: tuple[str, ...]
    held_out: bool = False
    terrain_seed: int | None = None
    stress: bool = False


def parse_csv_values(text: str) -> list[float]:
    return [float(item.strip()) for item in text.split(",") if item.strip()]


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def git_revision(repo: Path) -> str:
    completed = subprocess.run(
        ("git", "rev-parse", "HEAD"), cwd=repo, capture_output=True,
        text=True, check=False,
    )
    return completed.stdout.strip() if completed.returncode == 0 else "unknown"


def git_status_provenance(repo: Path) -> dict[str, object]:
    completed = subprocess.run(
        ("git", "status", "--porcelain=v1"), cwd=repo,
        capture_output=True, text=True, check=False,
    )
    if completed.returncode != 0:
        return {"worktree_status": "unknown"}
    status = completed.stdout
    tracked_diff = subprocess.run(
        ("git", "diff", "--binary", "HEAD", "--"), cwd=repo,
        capture_output=True, check=False,
    )
    return {
        "worktree_status": "dirty" if status else "clean",
        "worktree_status_entries": len(status.splitlines()),
        "worktree_status_sha256": hashlib.sha256(
            status.encode("utf-8")
        ).hexdigest(),
        "worktree_diff_sha256": (
            hashlib.sha256(tracked_diff.stdout).hexdigest()
            if tracked_diff.returncode == 0 else "unknown"
        ),
    }


def code_provenance(repo: Path, benchmark: Path) -> dict[str, object]:
    return {
        "git_revision": git_revision(repo),
        **git_status_provenance(repo),
        "benchmark_sha256": sha256_file(benchmark),
        "benchmark_size_bytes": benchmark.stat().st_size,
    }


def code_provenance_matches(
    initial: dict[str, object], final: dict[str, object]
) -> bool:
    stable_keys = (
        "git_revision", "worktree_diff_sha256", "benchmark_sha256",
        "benchmark_size_bytes",
    )
    return all(
        initial.get(key) not in (None, "unknown") and
        initial.get(key) == final.get(key)
        for key in stable_keys
    )


def ranges_overlap(first_a: int, count_a: int, first_b: int, count_b: int) -> bool:
    return first_a < first_b + count_b and first_b < first_a + count_a


def terrain_cases(
    slope_grades: Iterable[float],
    random_amplitudes: Iterable[float],
    seed_start: int,
    random_seeds: int,
) -> list[Case]:
    cases = [Case("flat", "flat", ("--terrain", "flat"))]
    for grade in slope_grades:
        grade_tag = f"{round(100.0 * abs(grade)):02d}pct"
        for sign, direction in ((1.0, "up"), (-1.0, "down")):
            signed = sign * grade
            cases.append(
                Case(
                    f"slope_{direction}_{grade_tag}",
                    "slope",
                    ("--terrain", "slope", "--slope-x", str(signed)),
                )
            )
            cases.append(
                Case(
                    f"cross_{direction}_{grade_tag}",
                    "cross_slope",
                    ("--terrain", "cross-slope", "--slope-y", str(signed)),
                )
            )
    cross_20_degree_grade = 0.36397023426620234
    for sign, direction in ((1.0, "positive"), (-1.0, "negative")):
        cases.append(
            Case(
                f"cross_{direction}_20deg_stress",
                "cross_slope_20deg_stress",
                (
                    "--terrain",
                    "cross-slope",
                    "--slope-y",
                    str(sign * cross_20_degree_grade),
                ),
                stress=True,
            )
        )
    for amplitude in (0.02, 0.04, 0.06):
        stress = amplitude == 0.06
        cases.append(
            Case(
                f"sinusoidal_{round(100 * amplitude):02d}cm",
                "sinusoidal_06cm_stress" if stress else "sinusoidal",
                ("--terrain", "sinusoidal", "--terrain-amplitude", str(amplitude)),
                stress=stress,
            )
        )
    for height in (0.02, 0.04):
        stress = height == 0.04
        cases.append(
            Case(
                f"smooth_step_{round(100 * height):02d}cm",
                "smooth_step_04cm_stress" if stress else "smooth_step",
                ("--terrain", "smooth_step", "--step-height", str(height)),
                stress=stress,
            )
        )
    for amplitude in random_amplitudes:
        for seed in range(seed_start, seed_start + random_seeds):
            held_out = (
                FIRST_HELD_OUT_RANDOM_SEED
                <= seed
                <= LAST_HELD_OUT_RANDOM_SEED
            )
            cases.append(
                Case(
                    f"random_{round(100 * amplitude):02d}cm_seed_{seed}",
                    f"random_smooth_{round(100 * amplitude):02d}cm",
                    (
                        "--terrain",
                        "random_smooth",
                        "--terrain-amplitude",
                        str(amplitude),
                        "--terrain-seed",
                        str(seed),
                    ),
                    held_out=held_out,
                    terrain_seed=seed,
                )
            )
    return cases


def smoke_cases() -> list[Case]:
    return [
        Case("flat", "flat", ("--terrain", "flat")),
        Case("slope_up_10pct", "slope", ("--terrain", "slope", "--slope-x", "0.10")),
        Case("cross_up_10pct", "cross_slope", ("--terrain", "cross-slope", "--slope-y", "0.10")),
        Case("sinusoidal_02cm", "sinusoidal", ("--terrain", "sinusoidal", "--terrain-amplitude", "0.02")),
        Case("smooth_step_02cm", "smooth_step", ("--terrain", "smooth_step", "--step-height", "0.02")),
        Case(
            "random_02cm_seed_1000",
            "random_smooth_02cm",
            ("--terrain", "random_smooth", "--terrain-amplitude", "0.02", "--terrain-seed", "1000"),
            terrain_seed=1000,
        ),
    ]


def contact_task_episode_updates(rate_hz: int) -> int:
    shift_steps = round(1.0 / (rate_hz * REALTIME_TIME_STEP_S))
    return math.ceil(REALTIME_SWING_LAST_STAGE / shift_steps)


def acceptance_case_errors(cases: list[Case]) -> list[str]:
    errors: list[str] = []
    names = [case.name for case in cases]
    if len(names) != len(set(names)):
        errors.append("case names must be unique")

    required_classes = {
        "flat",
        "slope",
        "cross_slope",
        "sinusoidal",
        "smooth_step",
        "random_smooth_02cm",
        "random_smooth_04cm",
    }
    classes = {case.terrain_class for case in cases}
    missing_classes = sorted(required_classes - classes)
    if missing_classes:
        errors.append(
            "acceptance matrix is missing terrain classes: "
            + ", ".join(missing_classes)
        )

    required_slopes = (
        ("slope", "--slope-x", 0.10),
        ("slope", "--slope-x", -0.10),
        ("cross_slope", "--slope-y", 0.10),
        ("cross_slope", "--slope-y", -0.10),
    )
    for terrain_class, option, required_value in required_slopes:
        found = False
        for case in cases:
            if case.stress or case.terrain_class != terrain_class:
                continue
            try:
                value = float(case.arguments[case.arguments.index(option) + 1])
            except (ValueError, IndexError):
                continue
            if math.isclose(value, required_value, rel_tol=0.0, abs_tol=1e-12):
                found = True
                break
        if not found:
            errors.append(
                "acceptance matrix requires core "
                f"{terrain_class} {required_value:+.2f}"
            )

    for terrain_class in ("random_smooth_02cm", "random_smooth_04cm"):
        members = [case for case in cases if case.terrain_class == terrain_class]
        seeds = [case.terrain_seed for case in members]
        if any(not case.held_out for case in members):
            errors.append(f"{terrain_class} contains a non-held-out case")
        if any(
            seed is None or seed < FIRST_HELD_OUT_RANDOM_SEED or
            seed > LAST_HELD_OUT_RANDOM_SEED
            for seed in seeds
        ):
            errors.append(
                f"{terrain_class} seeds must be in the frozen held-out range "
                f"[{FIRST_HELD_OUT_RANDOM_SEED}, "
                f"{LAST_HELD_OUT_RANDOM_SEED}]"
            )
        unique_seeds = {seed for seed in seeds if seed is not None}
        if len(unique_seeds) < MINIMUM_ACCEPTANCE_RANDOM_SEEDS:
            errors.append(
                f"{terrain_class} requires at least "
                f"{MINIMUM_ACCEPTANCE_RANDOM_SEEDS} unique held-out seeds"
            )
    return errors


def parse_key_values(line: str) -> dict[str, str]:
    values: dict[str, str] = {}
    for field in next(csv.reader([line]))[1:]:
        if "=" in field:
            key, value = field.split("=", 1)
            values[key] = value
    return values


def read_key_value_record(path: Path, record_type: str) -> dict[str, str]:
    record: dict[str, str] | None = None
    prefix = record_type + ","
    with path.open(newline="", encoding="utf-8") as stream:
        for line in stream:
            if line.startswith(prefix):
                record = parse_key_values(line.rstrip("\r\n"))
    if record is None:
        raise RuntimeError(f"missing {record_type} record in {path}")
    return record


def read_summary(path: Path) -> dict[str, str]:
    return read_key_value_record(path, "summary")


def case_argument(case: Case, option: str, default: str) -> str:
    try:
        index = case.arguments.index(option)
    except ValueError:
        return default
    return case.arguments[index + 1]


def benchmark_config_matches_case(
    config: dict[str, str], summary: dict[str, str], case: Case,
    updates: int, rate_hz: int, episode_updates: int,
) -> bool:
    expected_terrain = case_argument(case, "--terrain", "").replace("-", "_")
    expected_seed = case.terrain_seed if case.terrain_seed is not None else 0
    terrain_kind = "slope" if expected_terrain in ("slope", "cross_slope") \
        else expected_terrain
    expected_strings = {
        "terrain": expected_terrain,
        "terrain_kind": terrain_kind,
        "terrain_seed": str(expected_seed),
        "N": "50",
        "nx": "25",
        "nu": "28",
        "nc": "86",
        "rate_hz": str(rate_hz),
        "shift_steps": str(round(1.0 / (rate_hz * REALTIME_TIME_STEP_S))),
        "updates": str(updates),
        "episode_updates": str(episode_updates),
        "contact_task_episode_updates": str(contact_task_episode_updates(rate_hz)),
        "episode_reset": "standing",
        "contact_task_update_definition":
            "warm_update_in_complete_active_to_settled_episode",
        "recovery": "off",
        "exact_hessian": "off",
        "adaptive_exact_hessian": "off",
        "nonlinear_rollout": "off",
        "terrain_convention": "go1_foot_center",
    }
    if any(config.get(key) != value for key, value in expected_strings.items()):
        return False
    expected_numbers = {
        "offset": -0.017805846,
        "dt": REALTIME_TIME_STEP_S,
        "horizon_s": 2.5,
        "deadline_ms": 1000.0 / rate_hz,
        "measurement_offset": 0.0005,
        "slope_x": float(case_argument(case, "--slope-x", "0")),
        "slope_y": float(case_argument(case, "--slope-y", "0")),
        "amplitude": float(case_argument(case, "--terrain-amplitude", "0")),
        "step_height": float(case_argument(case, "--step-height", "0")),
        "riccati_reg_base": 1e-8,
    }
    terrain_numbers = {
        "wave_number_x": 0.0,
        "wave_number_y": 0.0,
        "phase_x": 0.0,
        "phase_y": 0.0,
        "secondary_amplitude": 0.0,
        "secondary_wave_number_x": 0.0,
        "secondary_wave_number_y": 0.0,
        "secondary_phase_x": 0.0,
        "secondary_phase_y": 0.0,
        "step_center_x": 0.0,
        "step_sharpness": 0.0,
    }
    if expected_terrain == "sinusoidal":
        terrain_numbers["wave_number_x"] = float(
            case_argument(case, "--wave-number-x", "4.0")
        )
        terrain_numbers["wave_number_y"] = float(
            case_argument(case, "--wave-number-y", "3.0")
        )
    elif expected_terrain == "smooth_step":
        terrain_numbers["step_center_x"] = float(
            case_argument(case, "--step-center-x", "0.23")
        )
        terrain_numbers["step_sharpness"] = float(
            case_argument(case, "--step-sharpness", "18.0")
        )
    elif expected_terrain == "random_smooth":
        state = expected_seed ^ 0x9E3779B9

        def unit() -> float:
            nonlocal state
            state = (1664525 * state + 1013904223) & 0xFFFFFFFF
            return ((state >> 8) & 0x00FFFFFF) / float(0x01000000)

        maximum_amplitude = float(
            case_argument(case, "--terrain-amplitude", "0")
        )
        terrain_numbers.update({
            "wave_number_x": 4.0 + 2.0 * unit(),
            "wave_number_y": 2.5 + 2.0 * unit(),
            "phase_x": 2.0 * math.pi * unit(),
            "phase_y": 2.0 * math.pi * unit(),
            "secondary_amplitude": 0.35 * maximum_amplitude,
            "secondary_wave_number_x": 7.0 + 3.0 * unit(),
            "secondary_wave_number_y": 4.0 + 2.0 * unit(),
            "secondary_phase_x": 2.0 * math.pi * unit(),
            "secondary_phase_y": 2.0 * math.pi * unit(),
        })
        expected_numbers["amplitude"] = 0.65 * maximum_amplitude
    expected_numbers.update(terrain_numbers)
    try:
        if any(
            not math.isclose(
                float(config[key]), value, rel_tol=1e-8, abs_tol=1e-9
            )
            for key, value in expected_numbers.items()
        ):
            return False
    except (KeyError, ValueError):
        return False
    fingerprint = config.get("terrain_fingerprint", "")
    return (
        len(fingerprint) == 16 and
        all(character in "0123456789abcdefABCDEF" for character in fingerprint) and
        config.get("terrain") == summary.get("terrain") and
        fingerprint == summary.get("terrain_fingerprint") and
        config.get("terrain_seed") == summary.get("terrain_seed")
    )


def numeric(summary: dict[str, str], key: str) -> float:
    try:
        return float(summary[key])
    except (KeyError, ValueError) as error:
        raise RuntimeError(f"invalid summary field {key!r}: {summary}") from error


def integer(row: dict[str, object], key: str) -> int:
    try:
        return int(row.get(key, 0))
    except (TypeError, ValueError):
        return 0


def floating(row: dict[str, object], key: str) -> float:
    if key not in row:
        return math.inf
    try:
        return float(row.get(key, 0.0))
    except (TypeError, ValueError):
        return math.inf


REQUIRED_INTEGER_EVIDENCE_FIELDS = (
    "updates", "completed", "episodes", "cold_failures",
    "completed_contact_tasks", "contact_task_updates", "unqualified_updates",
    "full_kkt", "task_pass", "published", "fallbacks", "deadline_misses",
    "invalid_publications", "exact_hessian_calls",
    "max_consecutive_fallbacks", "contact_task_protocol_qualified",
    "benchmark_config_valid", "return_code",
)


def row_has_required_evidence(row: dict[str, object]) -> bool:
    if row.get("verdict") not in ("PASS", "FAIL"):
        return False
    try:
        for key in REQUIRED_INTEGER_EVIDENCE_FIELDS:
            if key not in row:
                return False
            int(row[key])
        if "e2e_p99_ms" not in row or not math.isfinite(
            float(row["e2e_p99_ms"])
        ):
            return False
    except (TypeError, ValueError):
        return False
    return True


def class_summaries(
    rows: list[dict[str, object]],
    minimum_contact_task_updates: int = 0,
    minimum_random_seeds: int = 0,
) -> list[dict[str, object]]:
    grouped: dict[str, list[dict[str, object]]] = {}
    for row in rows:
        grouped.setdefault(str(row["terrain_class"]), []).append(row)
    output: list[dict[str, object]] = []
    for terrain_class, members in sorted(grouped.items()):
        scopes = {str(row.get("scope", "core")) for row in members}
        scope = next(iter(scopes)) if len(scopes) == 1 else "mixed"
        attempted = sum(integer(row, "updates") for row in members)
        completed = sum(integer(row, "completed") for row in members)
        contact_task_updates = sum(
            integer(row, "contact_task_updates") for row in members
        )
        completed_contact_tasks = sum(
            integer(row, "completed_contact_tasks") for row in members
        )
        unqualified_updates = sum(
            integer(row, "unqualified_updates") for row in members
        )
        full_kkt = sum(integer(row, "full_kkt") for row in members)
        task_pass = sum(integer(row, "task_pass") for row in members)
        published = sum(integer(row, "published") for row in members)
        cold_failures = sum(integer(row, "cold_failures") for row in members)
        invalid = sum(integer(row, "invalid_publications") for row in members)
        exact_hessian_calls = sum(
            integer(row, "exact_hessian_calls") for row in members
        )
        failed_cases = sum(
            row.get("verdict") != "PASS" or integer(row, "return_code") != 0
            for row in members
        )
        protocol_failures = sum(
            integer(row, "contact_task_protocol_qualified") != 1
            for row in members
        )
        evidence_failures = sum(
            not row_has_required_evidence(row) for row in members
        )
        held_out_members = [row for row in members if integer(row, "held_out") == 1]
        held_out_seeds = {
            integer(row, "terrain_seed") for row in held_out_members
        }
        held_out_fingerprints = {
            str(row.get("terrain_fingerprint", "")) for row in held_out_members
            if row.get("terrain_fingerprint")
        }
        minimum_usable = math.ceil(0.99 * attempted)
        is_random = terrain_class.startswith("random_smooth_")
        random_seed_gate = (
            not is_random
            or minimum_random_seeds == 0
            or (
                len(held_out_members) == len(members)
                and len(held_out_seeds) >= minimum_random_seeds
                and len(held_out_fingerprints) >= minimum_random_seeds
            )
        )
        worst_e2e_p99_ms = max(
            floating(row, "e2e_p99_ms") for row in members
        )
        worst_consecutive_fallbacks = max(
            integer(row, "max_consecutive_fallbacks") for row in members
        )
        accepted = (
            scope != "mixed"
            and failed_cases == 0
            and protocol_failures == 0
            and evidence_failures == 0
            and attempted >= minimum_contact_task_updates
            and completed == attempted
            and contact_task_updates == attempted
            and completed_contact_tasks > 0
            and unqualified_updates == 0
            and full_kkt >= minimum_usable
            and task_pass >= minimum_usable
            and published >= minimum_usable
            and cold_failures == 0
            and invalid == 0
            and exact_hessian_calls == 0
            and worst_e2e_p99_ms <= 180.0
            and worst_consecutive_fallbacks <= 1
            and random_seed_gate
        )
        output.append(
            {
                "terrain_class": terrain_class,
                "scope": scope,
                "cases": len(members),
                "failed_cases": failed_cases,
                "protocol_failures": protocol_failures,
                "evidence_failures": evidence_failures,
                "attempted_updates": attempted,
                "completed_updates": completed,
                "episodes": sum(integer(row, "episodes") for row in members),
                "completed_contact_tasks": completed_contact_tasks,
                "contact_task_updates": contact_task_updates,
                "unqualified_updates": unqualified_updates,
                "contact_task_coverage": (
                    contact_task_updates / attempted if attempted else 0.0
                ),
                "minimum_contact_task_updates": minimum_contact_task_updates,
                "cold_failures": cold_failures,
                "full_kkt_updates": full_kkt,
                "task_pass_updates": task_pass,
                "published_updates": published,
                "usable_rate": published / attempted if attempted else 0.0,
                "deadline_misses": sum(
                    integer(row, "deadline_misses") for row in members
                ),
                "invalid_publications": invalid,
                "exact_hessian_calls": exact_hessian_calls,
                "held_out_seed_count": len(held_out_seeds),
                "held_out_fingerprint_count": len(held_out_fingerprints),
                "minimum_held_out_seeds": (
                    minimum_random_seeds if is_random else 0
                ),
                "worst_e2e_p99_ms": worst_e2e_p99_ms,
                "worst_max_consecutive_fallbacks": worst_consecutive_fallbacks,
                "verdict": "PASS" if accepted else "FAIL",
            }
        )
    return output


def failed_verdict_counts(rows: list[dict[str, object]]) -> tuple[int, int]:
    core = sum(
        row.get("verdict") != "PASS" and row.get("scope", "core") != "stress"
        for row in rows
    )
    stress = sum(
        row.get("verdict") != "PASS" and row.get("scope") == "stress"
        for row in rows
    )
    return core, stress


def write_csv(path: Path, rows: list[dict[str, object]]) -> None:
    if not rows:
        return
    fieldnames: list[str] = []
    for row in rows:
        for key in row:
            if key not in fieldnames:
                fieldnames.append(key)
    with path.open("w", newline="", encoding="utf-8") as stream:
        writer = csv.DictWriter(stream, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)


def summary_qualifies_contact_task_protocol(
    summary: dict[str, str], expected_updates: int, episode_updates: int
) -> bool:
    try:
        return (
            int(summary["updates"]) == expected_updates
            and int(summary["completed"]) == expected_updates
            and int(summary["contact_task_episode_updates"]) == episode_updates
            and int(summary["completed_contact_tasks"])
            == expected_updates // episode_updates
            and int(summary["contact_task_updates"]) == expected_updates
            and int(summary["unqualified_updates"]) == 0
        )
    except (KeyError, TypeError, ValueError):
        return False


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--benchmark", required=True, type=Path)
    parser.add_argument("--output", required=True, type=Path)
    parser.add_argument("--profile", choices=("smoke", "development", "acceptance"), default="acceptance")
    parser.add_argument("--updates", type=int, choices=(100, 1000))
    parser.add_argument("--random-updates-per-seed", type=int)
    parser.add_argument("--episode-updates", type=int)
    parser.add_argument("--rate", type=int, choices=(5, 10), default=5)
    parser.add_argument("--seed-start", type=int)
    parser.add_argument("--random-seeds", type=int, default=20)
    parser.add_argument("--slope-grades", default="0.10")
    parser.add_argument("--random-amplitudes", default="0.02,0.04")
    parser.add_argument("--list", action="store_true", help="Print cases without executing them")
    args = parser.parse_args()

    seed_start = args.seed_start
    if seed_start is None:
        seed_start = (
            FIRST_HELD_OUT_RANDOM_SEED
            if args.profile == "acceptance"
            else 1000
        )
    if seed_start < 0 or args.random_seeds < 1:
        parser.error("seed-start must be nonnegative and random-seeds must be positive")
    if args.profile == "acceptance":
        if (
            seed_start != FIRST_HELD_OUT_RANDOM_SEED or
            args.random_seeds != MINIMUM_ACCEPTANCE_RANDOM_SEEDS
        ):
            parser.error(
                "acceptance profile requires the frozen random seed range "
                f"{FIRST_HELD_OUT_RANDOM_SEED}-"
                f"{LAST_HELD_OUT_RANDOM_SEED}"
            )
    elif ranges_overlap(
        seed_start, args.random_seeds, FIRST_HELD_OUT_RANDOM_SEED,
        MINIMUM_ACCEPTANCE_RANDOM_SEEDS,
    ):
        parser.error(
            "non-acceptance profiles may not consume the frozen held-out "
            f"random seed range {FIRST_HELD_OUT_RANDOM_SEED}-"
            f"{LAST_HELD_OUT_RANDOM_SEED}"
        )
    required_episode_updates = contact_task_episode_updates(args.rate)
    episode_updates = args.episode_updates or required_episode_updates
    maximum_episode_updates = 12 if args.rate == 5 else 25
    if episode_updates < 1 or episode_updates > maximum_episode_updates:
        parser.error(
            "episode-updates must keep the executed transition inside the "
            f"2.5 s horizon (1..{maximum_episode_updates} at {args.rate} Hz)"
        )
    if episode_updates != required_episode_updates:
        parser.error(
            "episode-updates must cover exactly one complete contact task "
            f"({required_episode_updates} at {args.rate} Hz)"
        )
    updates = args.updates or (1000 if args.profile == "acceptance" else 100)
    if updates % episode_updates != 0:
        parser.error("updates must contain an integer number of contact tasks")
    random_updates_per_seed = args.random_updates_per_seed
    if random_updates_per_seed is None:
        random_updates_per_seed = updates
        if args.profile == "acceptance":
            minimum_random_updates = max(
                MINIMUM_ACCEPTANCE_RANDOM_UPDATES_PER_SEED,
                math.ceil(
                    MINIMUM_ACCEPTANCE_UPDATES_PER_CLASS / args.random_seeds
                ),
            )
            random_updates_per_seed = (
                math.ceil(minimum_random_updates / episode_updates)
                * episode_updates
            )
    if (
        random_updates_per_seed <= 0
        or random_updates_per_seed % episode_updates != 0
    ):
        parser.error(
            "random-updates-per-seed must contain an integer number of "
            "contact tasks"
        )
    if (
        args.profile == "acceptance"
        and updates < MINIMUM_ACCEPTANCE_UPDATES_PER_CLASS
    ):
        parser.error(
            "acceptance profile requires at least "
            f"{MINIMUM_ACCEPTANCE_UPDATES_PER_CLASS} updates per "
            "deterministic case"
        )
    if (
        args.profile == "acceptance"
        and args.random_seeds < MINIMUM_ACCEPTANCE_RANDOM_SEEDS
    ):
        parser.error(
            "acceptance profile requires at least "
            f"{MINIMUM_ACCEPTANCE_RANDOM_SEEDS} unseen random seeds"
        )
    if (
        args.profile == "acceptance"
        and (
            random_updates_per_seed
            < MINIMUM_ACCEPTANCE_RANDOM_UPDATES_PER_SEED
            or random_updates_per_seed * args.random_seeds
            < MINIMUM_ACCEPTANCE_UPDATES_PER_CLASS
        )
    ):
        parser.error(
            "acceptance profile requires at least "
            f"{MINIMUM_ACCEPTANCE_RANDOM_UPDATES_PER_SEED} updates per random "
            "seed and 1000 aggregate updates per random terrain class"
        )
    slope_grades = parse_csv_values(args.slope_grades)
    random_amplitudes = parse_csv_values(args.random_amplitudes)
    if (
        not slope_grades
        or not random_amplitudes
        or any(not math.isfinite(value) or value <= 0.0 for value in slope_grades)
        or any(
            not math.isfinite(value) or value <= 0.0
            for value in random_amplitudes
        )
    ):
        parser.error("slope grades and random amplitudes must be finite and positive")
    if args.profile == "acceptance" and any(
        not any(math.isclose(value, required, abs_tol=1e-12)
                for value in random_amplitudes)
        for required in REQUIRED_RANDOM_AMPLITUDES
    ):
        parser.error("acceptance profile requires 0.02 and 0.04 m random terrain")
    if args.profile == "smoke":
        cases = smoke_cases()
    else:
        cases = terrain_cases(
            slope_grades,
            random_amplitudes,
            seed_start,
            args.random_seeds,
        )
    if args.profile == "acceptance":
        case_errors = acceptance_case_errors(cases)
        if case_errors:
            parser.error("; ".join(case_errors))
    if args.list:
        for case in cases:
            case_updates = (
                random_updates_per_seed
                if case.terrain_class.startswith("random_smooth_")
                else updates
            )
            scope = "stress" if case.stress else "core"
            print(
                case.name, f"scope={scope}", f"updates={case_updates}",
                " ".join(case.arguments),
            )
        return 0
    if not args.benchmark.is_file():
        parser.error(f"benchmark does not exist: {args.benchmark}")

    args.output.mkdir(parents=True, exist_ok=True)
    repo_root = Path(__file__).resolve().parents[2]
    benchmark_path = args.benchmark.resolve()
    initial_code_provenance = code_provenance(repo_root, benchmark_path)
    manifest = {
        "profile": args.profile,
        "rate_hz": args.rate,
        "updates_per_deterministic_case": updates,
        "random_updates_per_seed": random_updates_per_seed,
        "episode_updates": episode_updates,
        "contact_task_episode_updates": required_episode_updates,
        "seed_start": seed_start,
        "random_seeds": args.random_seeds,
        "case_count": len(cases),
        "core_case_count": sum(not case.stress for case in cases),
        "stress_case_count": sum(case.stress for case in cases),
        "benchmark": str(benchmark_path),
        "provenance": {
            **initial_code_provenance,
            "platform": platform.platform(),
            "machine": platform.machine(),
            "processor": platform.processor(),
            "python": sys.version.replace("\n", " "),
            "command": [str(Path(__file__).resolve()), *sys.argv[1:]],
        },
        "started_unix_s": time.time(),
        "acceptance": {
            "minimum_usable_rate": 0.99,
            "maximum_e2e_p99_ms": 180.0,
            "maximum_invalid_publications": 0,
            "maximum_exact_hessian_calls": 0,
            "maximum_consecutive_fallbacks": 1,
            "stress_cases_gate_headline_verdict": False,
            "minimum_contact_task_updates_per_class": (
                MINIMUM_ACCEPTANCE_UPDATES_PER_CLASS
                if args.profile == "acceptance"
                else 0
            ),
            "minimum_unique_held_out_random_seeds_per_class": (
                MINIMUM_ACCEPTANCE_RANDOM_SEEDS
                if args.profile == "acceptance"
                else 0
            ),
            "minimum_random_updates_per_seed": (
                MINIMUM_ACCEPTANCE_RANDOM_UPDATES_PER_SEED
                if args.profile == "acceptance"
                else 0
            ),
            "first_held_out_random_seed": FIRST_HELD_OUT_RANDOM_SEED,
        },
    }
    (args.output / "campaign.json").write_text(
        json.dumps(manifest, indent=2) + "\n", encoding="utf-8"
    )

    rows: list[dict[str, object]] = []
    core_case_failures = 0
    stress_case_failures = 0
    for index, case in enumerate(cases, start=1):
        case_updates = (
            random_updates_per_seed
            if case.terrain_class.startswith("random_smooth_")
            else updates
        )
        output_csv = args.output / f"{case.name}.csv"
        command = [
            str(args.benchmark),
            "--rate",
            str(args.rate),
            "--updates",
            str(case_updates),
            "--episode-updates",
            str(episode_updates),
            *case.arguments,
            "--csv",
            str(output_csv),
        ]
        print(f"[{index}/{len(cases)}] {case.name}", flush=True)
        start = time.monotonic()
        completed = subprocess.run(command, text=True, capture_output=True, check=False)
        wall_s = time.monotonic() - start
        stderr_path = args.output / f"{case.name}.stderr.txt"
        if completed.stderr:
            stderr_path.write_text(completed.stderr, encoding="utf-8")
        try:
            summary = read_summary(output_csv)
            benchmark_config = read_key_value_record(
                output_csv, "benchmark_config"
            )
        except (OSError, RuntimeError) as error:
            summary = {"verdict": "ERROR", "error": str(error)}
            benchmark_config = {}
        build_provenance = {
            f"build_{key}": benchmark_config.get(key, "missing")
            for key in (
                "compiler", "compiler_major", "compiler_minor",
                "compiler_patch", "fast_math", "ndebug", "cplusplus",
            )
        }
        benchmark_config_valid = benchmark_config_matches_case(
            benchmark_config, summary, case, case_updates, args.rate,
            episode_updates,
        )
        row: dict[str, object] = {
            "case": case.name,
            "terrain_class": case.terrain_class,
            "scope": "stress" if case.stress else "core",
            "held_out": int(case.held_out),
            "return_code": completed.returncode,
            "wall_s": f"{wall_s:.3f}",
            "benchmark_config_valid": int(benchmark_config_valid),
            **build_provenance,
            **summary,
        }
        protocol_qualified = summary_qualifies_contact_task_protocol(
            summary, case_updates, episode_updates
        )
        row["contact_task_protocol_qualified"] = int(protocol_qualified)
        rows.append(row)
        case_failed = (
            completed.returncode != 0
            or summary.get("verdict") != "PASS"
            or not protocol_qualified
            or not benchmark_config_valid
        )
        if case_failed:
            if case.stress:
                stress_case_failures += 1
            else:
                core_case_failures += 1
        write_csv(args.output / "runs.csv", rows)

    classes = class_summaries(
        rows,
        minimum_contact_task_updates=(
            MINIMUM_ACCEPTANCE_UPDATES_PER_CLASS
            if args.profile == "acceptance"
            else 0
        ),
        minimum_random_seeds=(
            MINIMUM_ACCEPTANCE_RANDOM_SEEDS
            if args.profile == "acceptance"
            else 0
        ),
    )
    write_csv(args.output / "terrain_classes.csv", classes)
    core_class_failures, stress_class_failures = failed_verdict_counts(classes)
    core_class_count = sum(row["scope"] != "stress" for row in classes)
    stress_class_count = len(classes) - core_class_count
    build_keys = (
        "build_compiler", "build_compiler_major", "build_compiler_minor",
        "build_compiler_patch", "build_fast_math", "build_ndebug",
        "build_cplusplus",
    )
    core_build_configurations = {
        tuple(str(row.get(key, "missing")) for key in build_keys)
        for row in rows if row.get("scope") != "stress"
    }
    build_provenance_valid = (
        len(core_build_configurations) == 1 and
        "missing" not in next(iter(core_build_configurations), ())
    )
    final_code_provenance = code_provenance(repo_root, benchmark_path)
    code_provenance_valid = code_provenance_matches(
        initial_code_provenance, final_code_provenance
    )
    clean_worktree_required = args.profile == "acceptance"
    clean_worktree_valid = (
        not clean_worktree_required or
        initial_code_provenance.get("worktree_status") == "clean"
    )
    manifest["finished_unix_s"] = time.time()
    manifest["failed_cases"] = core_case_failures
    manifest["total_failed_cases"] = (
        core_case_failures + stress_case_failures
    )
    manifest["failed_core_cases"] = core_case_failures
    manifest["failed_stress_cases"] = stress_case_failures
    manifest["failed_classes"] = core_class_failures
    manifest["total_failed_classes"] = (
        core_class_failures + stress_class_failures
    )
    manifest["failed_core_classes"] = core_class_failures
    manifest["failed_stress_classes"] = stress_class_failures
    manifest["build_provenance_valid"] = build_provenance_valid
    manifest["end_code_provenance"] = final_code_provenance
    manifest["code_provenance_valid"] = code_provenance_valid
    manifest["clean_worktree_required"] = clean_worktree_required
    manifest["clean_worktree_valid"] = clean_worktree_valid
    manifest["observed_core_build_configurations"] = [
        dict(zip(build_keys, values))
        for values in sorted(core_build_configurations)
    ]
    manifest["stress_verdict"] = (
        "PASS" if stress_case_failures == 0 and stress_class_failures == 0
        else "FAIL"
    )
    manifest["verdict"] = (
        "PASS" if core_case_failures == 0 and core_class_failures == 0 and
        build_provenance_valid and code_provenance_valid and
        clean_worktree_valid
        else "FAIL"
    )
    (args.output / "campaign.json").write_text(
        json.dumps(manifest, indent=2) + "\n", encoding="utf-8"
    )
    clean_worktree_verdict = "NOT_REQUIRED"
    if clean_worktree_required:
        clean_worktree_verdict = "PASS" if clean_worktree_valid else "FAIL"
    print(
        f"campaign {manifest['verdict']}: "
        f"{manifest['core_case_count'] - core_case_failures}/"
        f"{manifest['core_case_count']} core cases and "
        f"{core_class_count - core_class_failures}/{core_class_count} "
        "core classes passed; build/code provenance "
        f"{'PASS' if build_provenance_valid and code_provenance_valid else 'FAIL'}; "
        "clean acceptance worktree "
        f"{clean_worktree_verdict}; "
        f"stress {manifest['stress_verdict']}: "
        f"{manifest['stress_case_count'] - stress_case_failures}/"
        f"{manifest['stress_case_count']} cases and "
        f"{stress_class_count - stress_class_failures}/{stress_class_count} "
        "classes passed"
    )
    return 0 if manifest["verdict"] == "PASS" else 1


if __name__ == "__main__":
    sys.exit(main())
