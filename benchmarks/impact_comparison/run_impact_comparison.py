#!/usr/bin/env python3
"""Run paired ContactIPM-versus-IMPACT local comparisons."""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import random
import re
import shutil
import subprocess
import sys
import time
from pathlib import Path
from typing import Any

import run_impact_audit as audit_entry

CONTACT_DIR = Path(__file__).resolve().parents[1] / "contact_ipm"
sys.path.insert(0, str(CONTACT_DIR))
from source_provenance import describe_source  # noqa: E402


EXECUTABLES = {
    "push_box": {
        "contactipm": "contact_impact_push_box",
        "impact": "impact/experiments/box/box_impact_multiple",
        "contact_trajectory": "contactipm_impact_push_box.txt",
    },
    "push_t": {
        "contactipm": "contact_impact_push_t",
        "impact": "impact/experiments/push_t/push_t_impact_multiple",
        "contact_trajectory": "contactipm_impact_push_t.txt",
    },
    "cart_transport": {
        "contactipm": "contact_impact_cart_transport",
        "impact": (
            "impact/experiments/cart_transporter/"
            "cart_transporter_impact_multiple"
        ),
        "contact_trajectory": "contactipm_impact_cart_transport.txt",
    },
}


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def extract(pattern: str, text: str, conversion: Any, default: Any = None) -> Any:
    match = re.search(pattern, text, flags=re.MULTILINE)
    return conversion(match.group(1)) if match else default


def parse_stdout(solver: str, stdout: str, returncode: int) -> dict[str, Any]:
    if solver == "contactipm":
        status = extract(r"^Status:\s+(.+)$", stdout, str, "")
        return {
            "reported_success": returncode == 0 and status.strip() == "Success",
            "reported_status": status.strip(),
            "solver_seconds": extract(
                r"^Solve time:\s+([0-9.eE+-]+)\s+s$", stdout, float
            ),
            "iterations": extract(r"^Iterations:\s+(\d+)", stdout, int),
            "reported_objective": extract(
                r"^Objective:\s+([0-9.eE+-]+)", stdout, float
            ),
            "initialization_state_error": extract(
                r"^Initialization state error:\s*([0-9.eE+-]+)",
                stdout,
                float,
            ),
            "initialization_control_max": extract(
                r"^Initialization control max:\s*([0-9.eE+-]+)",
                stdout,
                float,
            ),
        }
    converged = extract(r"^Converged:\s+(YES|NO)$", stdout, str, "NO")
    return {
        "reported_success": returncode == 0 and converged == "YES",
        "reported_status": converged,
        "solver_seconds": extract(
            r"^Solve time:\s+([0-9.eE+-]+)\s+seconds$", stdout, float
        ),
        "iterations": extract(
            r"^Total inner iterations:\s+(\d+)$", stdout, int
        ),
        "reported_objective": extract(
            r"^Objective value:\s+([0-9.eE+-]+)$", stdout, float
        ),
    }


def command_for(
    problem: str,
    solver: str,
    contact_build: Path,
    impact_build: Path,
    case: dict[str, Any],
    trial_dir: Path,
) -> tuple[list[str], Path]:
    start = [str(value) for value in case["start"]]
    goal = [str(value) for value in case["goal"]]
    spec = EXECUTABLES[problem]
    if solver == "contactipm":
        executable = contact_build / spec["contactipm"]
        trajectory = trial_dir / spec["contact_trajectory"]
        return [str(executable), *start, *goal], trajectory
    executable = impact_build / spec["impact"]
    trajectory = trial_dir / "impact_trajectory.txt"
    return [str(executable), *start, *goal, str(trajectory)], trajectory


def run_one(
    problem: str,
    solver: str,
    contact_build: Path,
    impact_build: Path,
    case: dict[str, Any],
    trial_dir: Path,
    base_environment: dict[str, str],
    timeout: float,
) -> dict[str, Any]:
    trial_dir.mkdir(parents=True, exist_ok=False)
    command, trajectory = command_for(
        problem, solver, contact_build, impact_build, case, trial_dir
    )
    environment = dict(base_environment)
    if solver == "contactipm":
        environment["CONTACT_BENCHMARK_TRAJECTORY_DIR"] = str(trial_dir)
    start_time = time.perf_counter()
    timed_out = False
    try:
        completed = subprocess.run(
            command,
            text=True,
            stdout=subprocess.PIPE,
            stderr=subprocess.STDOUT,
            env=environment,
            timeout=timeout,
            check=False,
        )
        stdout = completed.stdout
        returncode = completed.returncode
    except subprocess.TimeoutExpired as error:
        timed_out = True
        stdout = (
            error.stdout.decode() if isinstance(error.stdout, bytes)
            else (error.stdout or "")
        )
        returncode = 124
    wall_seconds = time.perf_counter() - start_time
    (trial_dir / "stdout.txt").write_text(stdout, encoding="utf-8")
    parsed = parse_stdout(solver, stdout, returncode)
    audit: dict[str, Any] | None = None
    audit_error: str | None = None
    if trajectory.exists():
        try:
            audit = audit_entry.core.audit(
                problem,
                solver,
                trajectory,
                list(case["start"]),
                list(case["goal"]),
            )
        except (OSError, ValueError) as error:
            audit_error = str(error)
    repository = Path(__file__).resolve().parents[2]
    contactipm_source = describe_source(
        repository, (repository / "benchmarks" / "impact_comparison",)
    )
    result = {
        "solver": solver,
        "command": command,
        "returncode": returncode,
        "timed_out": timed_out,
        "wall_seconds": wall_seconds,
        "trajectory": str(trajectory),
        "stdout": str(trial_dir / "stdout.txt"),
        "audit": audit,
        "audit_error": audit_error,
        **parsed,
    }
    result["eligible"] = bool(
        result["reported_success"]
        and audit is not None
        and audit["audited_success"]
    )
    return result


def warm_up(
    problems: list[str],
    cases: dict[str, Any],
    contact_build: Path,
    impact_build: Path,
    raw_root: Path,
    environment: dict[str, str],
    count: int,
    timeout: float,
) -> None:
    for warmup in range(count):
        for problem in problems:
            case = cases[problem]["cases"][0]
            for solver in ("contactipm", "impact"):
                run_one(
                    problem,
                    solver,
                    contact_build,
                    impact_build,
                    case,
                    raw_root
                    / "warmups"
                    / f"{warmup:02d}_{problem}_{solver}",
                    environment,
                    timeout,
                )


def git_metadata(repository: Path) -> dict[str, Any]:
    commit = subprocess.run(
        ["git", "-C", str(repository), "rev-parse", "HEAD"],
        text=True,
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
        check=False,
    )
    diff = subprocess.run(
        ["git", "-C", str(repository), "diff", "--ignore-space-at-eol", "--stat"],
        text=True,
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
        check=False,
    )
    return {
        "commit": commit.stdout.strip() if commit.returncode == 0 else None,
        "substantive_diff": diff.stdout.strip() if diff.returncode == 0 else None,
    }


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--contact-build", type=Path, required=True)
    parser.add_argument("--impact-build", type=Path, required=True)
    parser.add_argument("--impact-repository", type=Path, required=True)
    parser.add_argument(
        "--cases",
        type=Path,
        default=Path(__file__).with_name("impact_cases.json"),
    )
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument(
        "--problems", nargs="+", choices=sorted(EXECUTABLES),
        default=sorted(EXECUTABLES)
    )
    parser.add_argument("--case-limit", type=int, default=None)
    parser.add_argument("--repetitions", type=int, default=1)
    parser.add_argument("--warmups", type=int, default=0)
    parser.add_argument("--seed", type=int, default=2027)
    parser.add_argument("--threads", type=int, default=1)
    parser.add_argument("--cpu", type=int, default=None)
    parser.add_argument("--timeout", type=float, default=120.0)
    args = parser.parse_args()

    if args.repetitions < 1 or args.warmups < 0 or args.threads < 1:
        parser.error("repetitions/threads must be positive and warmups nonnegative")
    if args.case_limit is not None and args.case_limit < 1:
        parser.error("case-limit must be positive")
    if args.cpu is not None:
        os.sched_setaffinity(0, {args.cpu})

    suite = json.loads(args.cases.read_text(encoding="utf-8"))
    selected: dict[str, Any] = {}
    for problem in args.problems:
        block = suite["problems"][problem]
        selected[problem] = {
            **block,
            "cases": block["cases"][: args.case_limit],
        }

    raw_root = args.output.with_suffix("")
    raw_root = raw_root.parent / f"{raw_root.name}_raw"
    if raw_root.exists():
        raise FileExistsError(f"raw result directory already exists: {raw_root}")
    raw_root.mkdir(parents=True)

    environment = dict(os.environ)
    environment.update(
        {
            "OMP_NUM_THREADS": str(args.threads),
            "OPENBLAS_NUM_THREADS": str(args.threads),
            "MKL_NUM_THREADS": str(args.threads),
            "VECLIB_MAXIMUM_THREADS": str(args.threads),
            "NUMEXPR_NUM_THREADS": str(args.threads),
        }
    )
    casadi_library = environment.get("CONTACTIPM_IMPACT_CASADI_LIBRARY")
    if casadi_library:
        existing = environment.get("LD_LIBRARY_PATH", "")
        environment["LD_LIBRARY_PATH"] = (
            casadi_library if not existing else f"{casadi_library}:{existing}"
        )

    warm_up(
        args.problems,
        selected,
        args.contact_build,
        args.impact_build,
        raw_root,
        environment,
        args.warmups,
        args.timeout,
    )

    jobs = [
        (problem, case, repetition)
        for problem in args.problems
        for case in selected[problem]["cases"]
        for repetition in range(args.repetitions)
    ]
    rng = random.Random(args.seed)
    rng.shuffle(jobs)
    pairs: list[dict[str, Any]] = []
    for block_index, (problem, case, repetition) in enumerate(jobs):
        order = ["contactipm", "impact"]
        rng.shuffle(order)
        trial_results: dict[str, Any] = {}
        for solver in order:
            trial_results[solver] = run_one(
                problem,
                solver,
                args.contact_build,
                args.impact_build,
                case,
                raw_root
                / "trials"
                / f"{block_index:05d}_{problem}_{case['id']}_r{repetition:02d}_{solver}",
                environment,
                args.timeout,
            )
        pairs.append(
            {
                "block_index": block_index,
                "problem": problem,
                "case_id": case["id"],
                "start": case["start"],
                "goal": case["goal"],
                "repetition": repetition,
                "solver_order": order,
                "contactipm": trial_results["contactipm"],
                "impact": trial_results["impact"],
                "pair_eligible": (
                    trial_results["contactipm"]["eligible"]
                    and trial_results["impact"]["eligible"]
                ),
            }
        )

    executable_hashes: dict[str, dict[str, str]] = {}
    for problem in args.problems:
        spec = EXECUTABLES[problem]
        executable_hashes[problem] = {
            "contactipm": sha256(args.contact_build / spec["contactipm"]),
            "impact": sha256(args.impact_build / spec["impact"]),
        }
    result = {
        "schema_version": 1,
        "created_unix_seconds": time.time(),
        "comparison": "ContactIPM versus locally executed IMPACT",
        "contactipm_source_snapshot": contactipm_source["snapshot"],
        "contactipm_source_manifest": contactipm_source["manifest"],
        "paper_reported_times_used": False,
        "case_suite": {
            "path": str(args.cases),
            "sha256": sha256(args.cases),
            "seed": suite["seed"],
        },
        "run_protocol": {
            "seed": args.seed,
            "repetitions": args.repetitions,
            "warmups": args.warmups,
            "threads": args.threads,
            "cpu_affinity": (
                sorted(os.sched_getaffinity(0))
                if hasattr(os, "sched_getaffinity") else None
            ),
            "timeout_seconds": args.timeout,
            "adjacent_randomized_pairs": True,
        },
        "impact_repository": git_metadata(args.impact_repository),
        "executable_hashes": executable_hashes,
        "pairs": pairs,
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(
        json.dumps(result, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    print(f"Wrote {args.output}")
    print(f"Pairs: {len(pairs)}")
    print(f"Eligible pairs: {sum(pair['pair_eligible'] for pair in pairs)}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
