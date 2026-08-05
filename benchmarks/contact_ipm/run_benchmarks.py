#!/usr/bin/env python3
"""Run ContactIPM and CRISP benchmarks under one local protocol."""

from __future__ import annotations

import argparse
import datetime as dt
import hashlib
import itertools
import json
import os
import platform
import random
import re
import subprocess
import sys
import time
from pathlib import Path

from trajectory_audit import CONTACT_ACTIVITY_TOLERANCE, audit


CONTACTIPM_TARGETS = {
    "cartpole_soft_walls": "contact_cartpole_soft_walls",
    "push_box": "contact_push_box",
    "transport": "contact_transport",
    "push_t": "contact_push_t",
}

CRISP_TARGETS = {
    "cartpole_soft_walls": "pushbot_example",
    "push_box": "pushbox_example",
    "transport": "cartTransp_example",
    "push_t": "pushT_example",
}

CRISP_SOURCES = {
    "cartpole_soft_walls": "src/examples/pushbot/cpp/SolvePushbot.cpp",
    "push_box": "src/examples/pushbox/SolvePushbox.cpp",
    "transport": "src/examples/Transp/cpp/SolveTransp.cpp",
    "push_t": "src/examples/pushT/SolvePushT.cpp",
}

CONTACTIPM_SOURCES = {
    "cartpole_soft_walls": "benchmarks/contact_ipm/cartpole_soft_walls.cpp",
    "push_box": "benchmarks/contact_ipm/push_box.cpp",
    "transport": "benchmarks/contact_ipm/transport.cpp",
    "push_t": "benchmarks/contact_ipm/push_t.cpp",
}


def executable(build_dir: Path, target: str, subdir: str = "") -> Path:
    suffix = ".exe" if os.name == "nt" else ""
    path = build_dir / subdir / f"{target}{suffix}"
    if not path.is_file():
        raise FileNotFoundError(f"benchmark executable not found: {path}")
    return path.resolve()


def git_revision(repo: Path) -> str | None:
    result = subprocess.run(
        ["git", "rev-parse", "HEAD"],
        cwd=repo,
        capture_output=True,
        text=True,
        check=False,
    )
    return result.stdout.strip() if result.returncode == 0 else None


def git_tracked_dirty(repo: Path) -> bool | None:
    result = subprocess.run(
        [
            "git",
            "-c",
            "core.autocrlf=true",
            "status",
            "--porcelain",
            "--untracked-files=no",
        ],
        cwd=repo,
        capture_output=True,
        text=True,
        check=False,
    )
    return bool(result.stdout.strip()) if result.returncode == 0 else None


def git_tracked_patch(repo: Path) -> dict | None:
    arguments = ["git", "-c", "core.autocrlf=true"]
    patch = subprocess.run(
        arguments + ["diff", "--binary", "HEAD", "--"],
        cwd=repo,
        capture_output=True,
        check=False,
    )
    names = subprocess.run(
        arguments + ["diff", "--name-only", "HEAD", "--"],
        cwd=repo,
        capture_output=True,
        text=True,
        check=False,
    )
    if patch.returncode != 0 or names.returncode != 0:
        return None
    return {
        "sha256": hashlib.sha256(patch.stdout).hexdigest(),
        "files": [line for line in names.stdout.splitlines() if line],
    }


def sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def save(path: Path, document: dict) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(
        json.dumps(document, indent=2, sort_keys=True), encoding="utf-8"
    )
    temporary.replace(path)


def load_case_suite(path: Path | None, problems: list[str]) -> dict[str, list[dict]]:
    if path is None:
        return {problem: [{"id": "source", "overrides": {}}] for problem in problems}
    document = json.loads(path.read_text(encoding="utf-8"))
    if document.get("schema_version") != 1:
        raise ValueError("unsupported validation case-suite schema")

    suites = {}
    for problem in problems:
        specification = document.get("problems", {}).get(problem)
        if specification is None:
            suites[problem] = [{"id": "source", "overrides": {}}]
            continue
        axes = specification.get("axes", {})
        if not axes or any(name not in {"initial_state", "target_state"} for name in axes):
            raise ValueError(f"{problem}: invalid or empty case axes")
        names = list(axes)
        choices = []
        for name in names:
            axis = axes[name]
            if not axis:
                raise ValueError(f"{problem}.{name}: empty axis")
            for choice in axis:
                if not re.fullmatch(r"[a-z0-9_]+", choice.get("id", "")):
                    raise ValueError(f"{problem}.{name}: invalid case id")
                if not isinstance(choice.get("value"), list) or not choice["value"]:
                    raise ValueError(f"{problem}.{name}: invalid vector")
            choices.append(axis)
        cases = []
        for combination in itertools.product(*choices):
            case_id = "__".join(choice["id"] for choice in combination)
            overrides = {
                name: [float(value) for value in choice["value"]]
                for name, choice in zip(names, combination)
            }
            cases.append({"id": case_id, "overrides": overrides})
        if len(cases) != specification.get("expected_count"):
            raise ValueError(f"{problem}: expanded case count drifted")
        suites[problem] = cases
    return suites


def case_environment(
    overrides: dict, push_t_segment: int | None = None
) -> dict[str, str]:
    names = {
        "initial_state": "CONTACT_BENCHMARK_INITIAL_STATE",
        "target_state": "CONTACT_BENCHMARK_TARGET_STATE",
    }
    environment = {
        names[name]: ",".join(format(float(value), ".17g") for value in values)
        for name, values in overrides.items()
    }
    if push_t_segment is not None:
        environment["CONTACT_BENCHMARK_PUSH_T_SEGMENT"] = str(push_t_segment)
    return environment


def parse_cpu_affinity(value: str | None) -> list[int] | None:
    if value is None:
        return None
    if not hasattr(os, "sched_getaffinity"):
        raise ValueError("CPU affinity is unsupported on this platform")
    available = set(os.sched_getaffinity(0))
    if value == "auto":
        return [min(available)]
    selected: set[int] = set()
    for token in value.split(","):
        bounds = token.strip().split("-", 1)
        try:
            first = int(bounds[0])
            last = int(bounds[-1])
        except ValueError as error:
            raise ValueError(f"invalid CPU affinity token: {token!r}") from error
        if first < 0 or last < first:
            raise ValueError(f"invalid CPU affinity range: {token!r}")
        selected.update(range(first, last + 1))
    if not selected:
        raise ValueError("CPU affinity cannot be empty")
    unavailable = selected - available
    if unavailable:
        raise ValueError(f"CPUs unavailable to this process: {sorted(unavailable)}")
    return sorted(selected)


def read_first(path: Path) -> str | None:
    try:
        return path.read_text(encoding="utf-8").strip()
    except OSError:
        return None


def cpu_model() -> str | None:
    try:
        for line in Path("/proc/cpuinfo").read_text(encoding="utf-8").splitlines():
            if line.startswith("model name"):
                return line.split(":", 1)[1].strip()
    except OSError:
        pass
    return platform.processor() or None


def schedule_trials(
    commands: list[dict], repetitions: int, seed: int
) -> list[dict]:
    """Randomize blocks and solver order while keeping matched pairs adjacent."""
    rng = random.Random(seed)
    case_keys = list(
        dict.fromkeys(
            (entry["problem"], entry.get("case_id", "source"))
            for entry in commands
        )
    )
    blocks = []
    for repetition in range(repetitions):
        for problem, case_id in case_keys:
            entries = [
                entry
                for entry in commands
                if entry["problem"] == problem
                and entry.get("case_id", "source") == case_id
            ]
            rng.shuffle(entries)
            blocks.append((repetition, problem, case_id, entries))
    rng.shuffle(blocks)

    trials = []
    for block_index, (repetition, problem, case_id, entries) in enumerate(blocks):
        for order_in_block, entry in enumerate(entries):
            trials.append(
                {
                    "repetition": repetition,
                    "problem": problem,
                    "case_id": case_id,
                    "entry": entry,
                    "block_index": block_index,
                    "order_in_block": order_in_block,
                }
            )
    return trials


def run_process(
    command: list[str],
    cwd: str,
    environment: dict[str, str],
    cpu_affinity: list[int] | None,
) -> subprocess.CompletedProcess:
    kwargs = {}
    if cpu_affinity is not None:
        kwargs["preexec_fn"] = lambda: os.sched_setaffinity(0, cpu_affinity)
    return subprocess.run(
        command,
        cwd=cwd,
        env=environment,
        capture_output=True,
        text=True,
        check=False,
        **kwargs,
    )


def parse_contactipm_push_t(stdout: str) -> dict:
    patterns = {
        "successful_segments": r"Successful segments:\s+(\d+)/(\d+)",
        "total_iterations": r"Total iterations:\s+(\d+)",
        "worst_dynamics": r"Worst dynamics:\s+([^\s]+)",
        "worst_side_violation": r"Worst side violation:\s*([^\s]+)",
        "worst_physical_mpcc": r"Worst physical MPCC:\s+([^\s]+)",
        "worst_position_error": r"Worst position error:\s*([^\s]+)",
        "worst_angular_error": r"Worst angular error:\s+([^\s]+)",
    }
    metrics = {}
    for name, pattern in patterns.items():
        match = re.search(pattern, stdout)
        if not match:
            continue
        if name == "successful_segments":
            metrics[name] = int(match.group(1))
            metrics["attempted_segments"] = int(match.group(2))
        elif name == "total_iterations":
            metrics[name] = int(match.group(1))
        else:
            metrics[name] = float(match.group(1))
    return metrics


def main() -> int:
    parser = argparse.ArgumentParser(
        description=(
            "Run locally built ContactIPM and CRISP executables without using "
            "execution times reported in either paper."
        )
    )
    parser.add_argument("--contactipm-build", type=Path, required=True)
    parser.add_argument("--crisp-build", type=Path)
    parser.add_argument(
        "--output", type=Path, default=Path("contact_benchmark_results.json")
    )
    parser.add_argument(
        "--problems",
        nargs="+",
        choices=tuple(CONTACTIPM_TARGETS),
        default=tuple(CONTACTIPM_TARGETS),
    )
    parser.add_argument("--repetitions", type=int, default=1)
    parser.add_argument("--warmups", type=int, default=0)
    parser.add_argument("--threads", type=int, default=1)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument(
        "--cpu-affinity",
        help="Linux CPU list such as 2 or 2-3, or 'auto' for one available CPU",
    )
    parser.add_argument(
        "--case-suite", type=Path,
        help="Cartesian initial-state/target suite for robustness evaluation",
    )
    parser.add_argument("--push-t-segment", type=int)
    parser.add_argument(
        "--nodes",
        type=int,
        help="expected compile-time node count for one scaling-study problem",
    )
    parser.add_argument(
        "--dt", type=float,
        help="expected compile-time time step for a scaling study",
    )
    parser.add_argument("--dry-run", action="store_true")
    args = parser.parse_args()

    if args.repetitions < 1 or args.warmups < 0 or args.threads < 1:
        parser.error("repetitions and threads must be positive; warmups nonnegative")
    if args.push_t_segment is not None:
        if not 0 <= args.push_t_segment < 50:
            parser.error("--push-t-segment must be in [0, 49]")
    if args.nodes is not None:
        if args.nodes < 2:
            parser.error("--nodes must be at least 2")
        if len(args.problems) != 1:
            parser.error("--nodes requires exactly one selected problem")
        if args.problems[0] not in {"push_box", "push_t"}:
            parser.error("--nodes currently supports Push Box and Push T")
    if args.dt is not None:
        if args.dt <= 0.0:
            parser.error("--dt must be positive")
        if args.nodes is None:
            parser.error("--dt requires --nodes")

    try:
        cpu_affinity = parse_cpu_affinity(args.cpu_affinity)
    except ValueError as error:
        parser.error(str(error))

    repo = Path(__file__).resolve().parents[2]
    if args.push_t_segment is not None and args.crisp_build is not None:
        crisp_push_t_source = repo / "benchmarks" / "CRISP" / CRISP_SOURCES["push_t"]
        if "CONTACT_BENCHMARK_PUSH_T_SEGMENT" not in crisp_push_t_source.read_text(
            encoding="utf-8"
        ):
            parser.error(
                "instrument CRISP before requesting a paired Push T segment"
            )
    source_case_manifest = Path(__file__).with_name("source_cases.json").resolve()
    case_suite_path = args.case_suite.resolve() if args.case_suite else None
    try:
        case_suites = load_case_suite(case_suite_path, list(args.problems))
    except (OSError, ValueError, json.JSONDecodeError) as error:
        parser.error(str(error))

    commands: list[dict] = []
    for problem in args.problems:
        contact_executable = executable(
            args.contactipm_build, CONTACTIPM_TARGETS[problem]
        )
        contact_command = [str(contact_executable)]
        if problem == "push_t" and args.push_t_segment is not None:
            contact_command.append(str(args.push_t_segment))
        crisp_executable = (
            executable(args.crisp_build, CRISP_TARGETS[problem], "examples")
            if args.crisp_build is not None
            else None
        )
        for case in case_suites[problem]:
            audit_case = dict(case["overrides"])
            if args.nodes is not None:
                audit_case["nodes"] = args.nodes
            if args.dt is not None:
                audit_case["dt"] = args.dt
            common = {
                "problem": problem,
                "case_id": case["id"],
                "case": audit_case,
                "environment": case_environment(
                    case["overrides"],
                    args.push_t_segment if problem == "push_t" else None,
                ),
            }
            commands.append(
                {
                    **common,
                    "solver": "ContactIPM",
                    "command": contact_command,
                    "cwd": str(repo),
                    "executable_sha256": sha256(contact_executable),
                }
            )
            if crisp_executable is not None:
                commands.append(
                    {
                        **common,
                        "solver": "CRISP",
                        "command": [str(crisp_executable)],
                        "cwd": str(crisp_executable.parent),
                        "executable_sha256": sha256(crisp_executable),
                    }
                )

    document = {
        "schema_version": 5,
        "created_utc": dt.datetime.now(dt.timezone.utc).isoformat(),
        "protocol": {
            "timing": "process wall clock on this machine",
            "warmups": args.warmups,
            "repetitions": args.repetitions,
            "threads": args.threads,
            "random_seed": args.seed,
            "pairing": (
                "contiguous (repetition, problem) blocks with randomized "
                "block and solver order"
            ),
            "cpu_affinity": cpu_affinity,
            "paper_reported_times_used": False,
            "contactipm_environment": {
                name: value
                for name, value in sorted(os.environ.items())
                if name.startswith("CONTACTIPM_")
            },
            "benchmark_objective_environment": {
                name: os.environ[name]
                for name in (
                    "CONTACT_BENCHMARK_EFFORT_SCALE",
                    "CONTACT_BENCHMARK_TRACKING_SCALE",
                )
                if name in os.environ
            },
            "scaling": (
                {
                    "nodes": args.nodes,
                    "dt": args.dt,
                    "physical_duration": (
                        (args.nodes - 1) * args.dt
                        if args.dt is not None else None
                    ),
                    "total_complementarity_pairs": {
                        "push_box": 10 * (args.nodes - 1),
                        "push_t": 43 * (args.nodes - 1),
                    }.get(args.problems[0]),
                }
                if args.nodes is not None
                else None
            ),
            "source_case_manifest": str(source_case_manifest),
            "source_case_manifest_sha256": sha256(source_case_manifest),
            "case_suite": str(case_suite_path) if case_suite_path else None,
            "case_suite_sha256": (
                sha256(case_suite_path) if case_suite_path else None
            ),
            "trajectory_audit": "same independent audit for both solvers",
            "contact_activity_tolerance": CONTACT_ACTIVITY_TOLERANCE,
            "evaluation_order": [
                "physical_feasibility",
                "task_success",
                "task_errors",
                "objective_components",
                "computational_performance",
                "contact_behavior",
            ],
        },
        "machine": {
            "platform": platform.platform(),
            "processor": platform.processor(),
            "cpu_model": cpu_model(),
            "python": sys.version,
            "logical_cpu_count": os.cpu_count(),
            "available_cpu_affinity": (
                sorted(os.sched_getaffinity(0))
                if hasattr(os, "sched_getaffinity")
                else None
            ),
            "cpu_governor": read_first(
                Path("/sys/devices/system/cpu/cpu0/cpufreq/scaling_governor")
            ),
        },
        "revision": {
            "contactipm": git_revision(repo),
            "contactipm_tracked_dirty": git_tracked_dirty(repo),
            "crisp": git_revision(repo / "benchmarks" / "CRISP"),
            "crisp_tracked_dirty": (
                git_tracked_dirty(repo / "benchmarks" / "CRISP")
                if args.crisp_build is not None
                else None
            ),
            "crisp_tracked_patch": (
                git_tracked_patch(repo / "benchmarks" / "CRISP")
                if args.crisp_build is not None
                else None
            ),
            "contactipm_source_sha256": {
                problem: sha256(repo / CONTACTIPM_SOURCES[problem])
                for problem in args.problems
            },
            "crisp_source_sha256": {
                problem: sha256(
                    repo / "benchmarks" / "CRISP" / CRISP_SOURCES[problem]
                )
                for problem in args.problems
            }
            if args.crisp_build is not None
            else {},
        },
        "commands": commands,
        "results": [],
    }
    save(args.output, document)
    if args.dry_run:
        print(json.dumps(document, indent=2, sort_keys=True))
        return 0

    environment = os.environ.copy()
    for variable in ("OMP_NUM_THREADS", "OPENBLAS_NUM_THREADS", "MKL_NUM_THREADS"):
        environment[variable] = str(args.threads)
    environment["CONTACT_BENCHMARK_CARTPOLE_GUESS"] = str(
        repo
        / "benchmarks"
        / "CRISP"
        / "src"
        / "examples"
        / "pushbot"
        / "initial_guess_pushbot_example.txt"
    )

    for warmup in range(args.warmups):
        warmup_trials = schedule_trials(commands, 1, args.seed + warmup)
        for trial in warmup_trials:
            entry = trial["entry"]
            print(
                f"warmup {warmup + 1}/{args.warmups}: "
                f"{entry['solver']} {entry['problem']} {entry['case_id']}",
                flush=True,
            )
            warmup_environment = environment.copy()
            warmup_environment.update(entry["environment"])
            run_process(
                entry["command"],
                entry["cwd"],
                warmup_environment,
                cpu_affinity,
            )

    trials = schedule_trials(commands, args.repetitions, args.seed)
    for index, trial in enumerate(trials, start=1):
        repetition = trial["repetition"]
        entry = trial["entry"]
        print(
            f"trial {index}/{len(trials)}: {entry['solver']} "
            f"{entry['problem']} {entry['case_id']} repetition={repetition}",
            flush=True,
        )
        trajectory_dir = (
            args.output.resolve().parent
            / f"{args.output.stem}_trajectories"
            / (
                f"trial_{index:03d}_{entry['solver'].lower()}_"
                f"{entry['problem']}_{entry['case_id']}"
            )
        )
        trajectory_dir.mkdir(parents=True, exist_ok=True)
        trial_environment = environment.copy()
        trial_environment.update(entry["environment"])
        trial_environment["CONTACT_BENCHMARK_TRAJECTORY_DIR"] = str(
            trajectory_dir
        )
        started_at = time.time()
        start = time.perf_counter()
        result = run_process(
            entry["command"],
            entry["cwd"],
            trial_environment,
            cpu_affinity,
        )
        elapsed = time.perf_counter() - start
        metrics = {}
        try:
            metrics["trajectory_audit"] = audit(
                entry["problem"],
                trajectory_dir,
                entry["solver"],
                started_at,
                args.push_t_segment,
                entry["case"],
            )
        except RuntimeError as error:
            metrics["trajectory_audit_error"] = str(error)
        if entry["solver"] == "ContactIPM" and entry["problem"] == "push_t":
            metrics["summary"] = parse_contactipm_push_t(result.stdout)
        document["results"].append(
            {
                "solver": entry["solver"],
                "problem": entry["problem"],
                "case_id": entry["case_id"],
                "case": entry["case"],
                "repetition": repetition,
                "block_index": trial["block_index"],
                "order_in_block": trial["order_in_block"],
                "returncode": result.returncode,
                "wall_time_seconds": elapsed,
                "trajectory_directory": str(trajectory_dir),
                "metrics": metrics,
                "stdout": result.stdout,
                "stderr": result.stderr,
            }
        )
        save(args.output, document)

    process_failures = sum(
        result["returncode"] != 0 for result in document["results"]
    )
    audit_failures = sum(
        "trajectory_audit_error" in result["metrics"]
        or (
            result["metrics"]["trajectory_audit"]["successful_instances"]
            != result["metrics"]["trajectory_audit"]["attempted_instances"]
        )
        for result in document["results"]
    )
    print(
        f"completed {len(trials)} trials; process failures={process_failures}; "
        f"audit failures={audit_failures}"
    )
    print(f"results: {args.output.resolve()}")
    return 0 if process_failures == 0 and audit_failures == 0 else 1


if __name__ == "__main__":
    raise SystemExit(main())
