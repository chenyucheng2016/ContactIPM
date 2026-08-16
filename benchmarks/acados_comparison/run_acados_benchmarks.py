#!/usr/bin/env python3
"""Run audited acados robustness or timing trials on the frozen paper cases."""

from __future__ import annotations

import argparse
import datetime as dt
import hashlib
import itertools
import json
import math
import os
import platform
import random
import statistics
import subprocess
import sys
from pathlib import Path
from typing import Any

import numpy as np


HERE = Path(__file__).resolve().parent
ROOT = HERE.parents[1]
CONTACT_DIR = ROOT / "benchmarks" / "contact_ipm"
IMPACT_DIR = ROOT / "benchmarks" / "impact_comparison"
sys.path.insert(0, str(HERE))
sys.path.insert(0, str(CONTACT_DIR))
sys.path.insert(0, str(IMPACT_DIR))

from acados_contact_benchmarks import make_benchmark, solve  # noqa: E402
import impact_audit  # noqa: E402
import trajectory_audit  # noqa: E402


CRISP_PROBLEMS = (
    "cartpole_soft_walls", "push_box", "transport", "push_t"
)
IMPACT_PROBLEMS = ("push_box", "push_t", "cart_transport")


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def git_revision(path: Path) -> str | None:
    result = subprocess.run(
        ["git", "-C", str(path), "rev-parse", "HEAD"],
        text=True,
        stdout=subprocess.PIPE,
        stderr=subprocess.DEVNULL,
        check=False,
    )
    return result.stdout.strip() if result.returncode == 0 else None


def _source_case(problem: str) -> dict[str, Any]:
    manifest = json.loads(
        (CONTACT_DIR / "source_cases.json").read_text(encoding="utf-8")
    )["problems"][problem]
    benchmark, _ = make_benchmark("crisp", problem)
    if problem == "push_t":
        segment = 8
        angle = 2.0 * math.pi * segment / 50.0
        radius = 0.25 + 0.25 * segment / 49.0
        return {
            "id": f"segment_{segment:02d}",
            "start": [
                radius * math.cos(angle), radius * math.sin(angle), angle
            ],
            "goal": benchmark.default_goal.tolist(),
            "segment": segment,
            "audit_override": None,
        }
    start = manifest["initial_state"][: benchmark.model.x.rows()]
    goal = manifest.get(
        "target_state", benchmark.default_goal.tolist()
    )[: benchmark.model.p.rows()]
    return {
        "id": "source",
        "start": start,
        "goal": goal,
        "segment": None,
        "audit_override": None,
    }


def _crisp_robustness_cases(problem: str) -> list[dict[str, Any]]:
    benchmark, _ = make_benchmark("crisp", problem)
    if problem == "push_t":
        cases = []
        for segment in range(50):
            angle = 2.0 * math.pi * segment / 50.0
            radius = 0.25 + 0.25 * segment / 49.0
            cases.append(
                {
                    "id": f"segment_{segment:02d}",
                    "start": [
                        radius * math.cos(angle),
                        radius * math.sin(angle),
                        angle,
                    ],
                    "goal": benchmark.default_goal.tolist(),
                    "segment": segment,
                    "audit_override": None,
                }
            )
        return cases
    suite = json.loads(
        (CONTACT_DIR / "validation_cases.json").read_text(encoding="utf-8")
    )["problems"][problem]
    cases = []
    for start_axis, goal_axis in itertools.product(
        suite["axes"]["initial_state"], suite["axes"]["target_state"]
    ):
        full_start = list(start_axis["value"])
        full_goal = list(goal_axis["value"])
        cases.append(
            {
                "id": f"{start_axis['id']}__{goal_axis['id']}",
                "start": full_start[: benchmark.model.x.rows()],
                "goal": full_goal[: benchmark.model.p.rows()],
                "segment": None,
                "audit_override": {
                    "initial_state": full_start,
                    "target_state": full_goal,
                },
            }
        )
    if len(cases) != suite["expected_count"]:
        raise RuntimeError(f"case count mismatch for {problem}")
    return cases


def load_cases(
    suite: str, problem: str, mode: str
) -> list[dict[str, Any]]:
    if suite == "crisp":
        return (
            [_source_case(problem)]
            if mode == "timing" else _crisp_robustness_cases(problem)
        )
    document = json.loads(
        (IMPACT_DIR / "impact_cases.json").read_text(encoding="utf-8")
    )
    cases = document["problems"][problem]["cases"]
    if mode == "timing":
        cases = cases[:1]
    return [
        {
            "id": case["id"],
            "start": case["start"],
            "goal": case["goal"],
            "segment": None,
            "audit_override": None,
        }
        for case in cases
    ]


def trajectory_path(
    suite: str, problem: str, segment: int | None, directory: Path
) -> Path:
    if suite == "crisp" and problem == "push_t":
        return directory / f"acados_push_t_seg_{segment:02d}.txt"
    return directory / f"acados_{problem}.txt"


def audit_trial(
    suite: str,
    problem: str,
    case: dict[str, Any],
    path: Path,
) -> dict[str, Any]:
    if suite == "crisp":
        result = trajectory_audit.audit(
            problem,
            path.parent,
            "acados",
            0.0,
            push_t_segment=case["segment"],
            case_override=case["audit_override"],
        )
        instance = result["instances"][0]
        return {
            "feasible": bool(instance["feasible"]),
            "task_success": bool(instance["task_success"]),
            "audited_success": bool(instance["success"]),
            "equality": float(instance["max_equality"]),
            "side_violation": float(instance["max_side_violation"]),
            "complementarity": float(instance["physical_mpcc"]),
            "objective": float(instance["objective"]),
            "position_error": float(
                instance.get(
                    "position_error", instance.get("translation_error", 0.0)
                )
            ),
            "angular_error": float(instance.get("angular_error", 0.0)),
            "velocity_error": float(instance.get("velocity_error", 0.0)),
            "raw": result,
        }
    result = impact_audit.audit(
        problem,
        "contactipm",
        path,
        list(case["start"]),
        list(case["goal"]),
    )
    return {
        "feasible": bool(result["feasible"]),
        "task_success": bool(result["task_success"]),
        "audited_success": bool(result["audited_success"]),
        "equality": float(
            max(result["initial_state_error"], result["dynamics_defect"],
                result["equality_violation"])
        ),
        "side_violation": float(result["side_violation"]),
        "complementarity": float(result["complementarity"]),
        "objective": float(result["objective"]),
        "position_error": float(
            result.get(
                "position_error", result.get("translation_error", 0.0)
            )
        ),
        "angular_error": float(result.get("angular_error", 0.0)),
        "velocity_error": float(result.get("velocity_error", 0.0)),
        "raw": result,
    }


def run_trial(
    suite: str,
    problem: str,
    case: dict[str, Any],
    directory: Path,
) -> dict[str, Any]:
    directory.mkdir(parents=True, exist_ok=False)
    benchmark, _ = make_benchmark(suite, problem)
    path = trajectory_path(suite, problem, case["segment"], directory)
    try:
        result = solve(
            benchmark,
            np.asarray(case["start"], dtype=float),
            np.asarray(case["goal"], dtype=float),
            path,
            quiet=True,
        )
        audit = audit_trial(suite, problem, case, path)
        error = None
    except Exception as exception:  # preserve the failed trial and continue
        result = {
            "status": None,
            "reported_success": False,
            "solver_seconds": None,
            "iterations": None,
            "objective": None,
            "residuals": None,
            "trajectory": str(path),
        }
        audit = None
        error = f"{type(exception).__name__}: {exception}"
    eligible = bool(
        result["reported_success"]
        and audit is not None
        and audit["audited_success"]
    )
    return {
        **result,
        "audit": audit,
        "error": error,
        "eligible": eligible,
    }


def distribution(values: list[float]) -> dict[str, float | int | None]:
    if not values:
        return {
            "count": 0, "median": None, "p90": None,
            "q1": None, "q3": None, "iqr": None,
        }
    q1, q3 = np.percentile(values, [25.0, 75.0])
    return {
        "count": len(values),
        "median": statistics.median(values),
        "p90": float(np.percentile(values, 90.0)),
        "q1": float(q1),
        "q3": float(q3),
        "iqr": float(q3 - q1),
    }


def summarize(trials: list[dict[str, Any]]) -> dict[str, Any]:
    summary: dict[str, Any] = {}
    for problem in sorted({trial["problem"] for trial in trials}):
        rows = [trial for trial in trials if trial["problem"] == problem]
        audits = [row["acados"]["audit"] for row in rows
                  if row["acados"]["audit"] is not None]
        eligible = [row for row in rows if row["acados"]["eligible"]]
        all_times = [
            row["acados"]["solver_seconds"] for row in rows
            if row["acados"]["solver_seconds"] is not None
        ]
        eligible_times = [
            row["acados"]["solver_seconds"] for row in eligible
        ]
        summary[problem] = {
            "trials": len(rows),
            "reported_converged": sum(
                row["acados"]["reported_success"] for row in rows
            ),
            "audited_feasible": sum(audit["feasible"] for audit in audits),
            "audited_task_success": sum(
                audit["task_success"] for audit in audits
            ),
            "audited_success": sum(
                audit["audited_success"] for audit in audits
            ),
            "eligible": len(eligible),
            "solver_seconds_all": distribution(all_times),
            "solver_seconds_eligible": distribution(eligible_times),
            "iterations_all": distribution([
                float(row["acados"]["iterations"]) for row in rows
                if row["acados"]["iterations"] is not None
            ]),
            "objective_audited_success": distribution([
                audit["objective"] for audit in audits
                if audit["audited_success"]
            ]),
            "worst_equality": max(
                (audit["equality"] for audit in audits), default=None
            ),
            "worst_side_violation": max(
                (audit["side_violation"] for audit in audits), default=None
            ),
            "worst_complementarity": max(
                (audit["complementarity"] for audit in audits), default=None
            ),
        }
    return summary


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--suite", choices=("crisp", "impact"), required=True)
    parser.add_argument(
        "--mode", choices=("robustness", "timing"), required=True
    )
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--problems", nargs="+")
    parser.add_argument("--case-limit", type=int)
    parser.add_argument("--repetitions", type=int, default=1)
    parser.add_argument("--warmups", type=int, default=0)
    parser.add_argument("--seed", type=int, default=2027)
    parser.add_argument("--cpu", type=int)
    args = parser.parse_args()
    allowed = CRISP_PROBLEMS if args.suite == "crisp" else IMPACT_PROBLEMS
    problems = args.problems or list(allowed)
    if any(problem not in allowed for problem in problems):
        parser.error(f"allowed problems: {', '.join(allowed)}")
    if args.case_limit is not None and args.case_limit < 1:
        parser.error("case-limit must be positive")
    if args.repetitions < 1 or args.warmups < 0:
        parser.error("repetitions must be positive and warmups nonnegative")
    if args.cpu is not None:
        os.sched_setaffinity(0, {args.cpu})

    raw_root = args.output.with_suffix("")
    raw_root = raw_root.parent / f"{raw_root.name}_raw"
    if raw_root.exists():
        raise FileExistsError(raw_root)
    raw_root.mkdir(parents=True)
    selected = {
        problem: load_cases(args.suite, problem, args.mode)[
            : args.case_limit
        ]
        for problem in problems
    }
    for warmup in range(args.warmups):
        for problem in problems:
            run_trial(
                args.suite,
                problem,
                selected[problem][0],
                raw_root / "warmups" / f"{warmup:02d}_{problem}",
            )

    jobs = [
        (problem, case, repetition)
        for problem in problems
        for case in selected[problem]
        for repetition in range(args.repetitions)
    ]
    rng = random.Random(args.seed)
    rng.shuffle(jobs)
    trials = []
    for index, (problem, case, repetition) in enumerate(jobs):
        trial = run_trial(
            args.suite,
            problem,
            case,
            raw_root
            / "trials"
            / f"{index:05d}_{problem}_{case['id']}_r{repetition:02d}",
        )
        trials.append(
            {
                "index": index,
                "problem": problem,
                "case_id": case["id"],
                "start": case["start"],
                "goal": case["goal"],
                "segment": case["segment"],
                "repetition": repetition,
                "acados": trial,
            }
        )
        print(
            f"{index + 1}/{len(jobs)} {problem}/{case['id']}: "
            f"status={trial['status']} audit="
            f"{trial['audit']['audited_success'] if trial['audit'] else False}"
        )

    acados_repository = Path(os.environ["ACADOS_SOURCE_DIR"])
    result = {
        "schema_version": 1,
        "created_utc": dt.datetime.now(dt.timezone.utc).isoformat(),
        "suite": args.suite,
        "mode": args.mode,
        "formulation": {
            "side_constraints": "a >= 0, b >= 0",
            "complementarity": "a*b <= 0 (exact with nonnegative sides)",
            "relaxation_used": False,
            "nlp_solver": "SQP",
            "qp_solver": "PARTIAL_CONDENSING_HPIPM",
            "hessian": (
                "exact cost Hessian; Gauss-Newton omission of dynamics/"
                "constraint second derivatives"
            ),
            "globalization": "MERIT_BACKTRACKING",
            "initialization": "matched to the corresponding competitor",
        },
        "protocol": {
            "seed": args.seed,
            "repetitions": args.repetitions,
            "warmups": args.warmups,
            "cpu_affinity": (
                sorted(os.sched_getaffinity(0))
                if hasattr(os, "sched_getaffinity") else None
            ),
            "solver_time": "acados time_tot; generation and compilation excluded",
            "eligibility": "acados status 0 and independent audited success",
            "paper_reported_times_used": False,
        },
        "case_source": {
            "path": str(
                CONTACT_DIR / "validation_cases.json"
                if args.suite == "crisp"
                else IMPACT_DIR / "impact_cases.json"
            ),
            "sha256": sha256(
                CONTACT_DIR / "validation_cases.json"
                if args.suite == "crisp"
                else IMPACT_DIR / "impact_cases.json"
            ),
        },
        "machine": {
            "platform": platform.platform(),
            "python": sys.version,
            "cpu_count": os.cpu_count(),
        },
        "revision": {
            "acados": git_revision(acados_repository),
        },
        "harness_source_sha256": {
            "runner": sha256(Path(__file__)),
            "formulation": sha256(HERE / "acados_contact_benchmarks.py"),
            "crisp_audit": sha256(CONTACT_DIR / "trajectory_audit.py"),
            "impact_audit": sha256(IMPACT_DIR / "impact_audit.py"),
        },
        "summary": summarize(trials),
        "trials": trials,
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(
        json.dumps(result, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    print(f"Wrote {args.output}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
