#!/usr/bin/env python3
"""Build and run matched horizon/complementarity-count scaling studies."""

from __future__ import annotations

import argparse
import json
import re
import shutil
import statistics
import subprocess
import sys
from pathlib import Path


PROBLEMS = {
    "push_box": {
        "contact_target": "contact_push_box",
        "crisp_target": "pushbox_example",
        "cmake_option": "NMPC_CONTACT_PUSH_BOX_NODES",
        "contact_width": (3, 6),
        "crisp_width": 9,
        "pairs_per_stage": 10,
        "physical_duration": 1.98,
        "dt_option": "NMPC_CONTACT_PUSH_BOX_DT",
    },
    "push_t": {
        "contact_target": "contact_push_t",
        "crisp_target": "pushT_example",
        "cmake_option": "NMPC_CONTACT_PUSH_T_NODES",
        "contact_width": (3, 17),
        "crisp_width": 29,
        "pairs_per_stage": 43,
        "physical_duration": 2.45,
        "dt_option": "NMPC_CONTACT_PUSH_T_DT",
    },
}


def save(path: Path, document: dict) -> None:
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(
        json.dumps(document, indent=2, sort_keys=True), encoding="utf-8"
    )
    temporary.replace(path)


def audit_passed(result: dict) -> bool:
    audit = result["metrics"].get("trajectory_audit")
    return (
        audit is not None
        and audit["successful_instances"] == audit["attempted_instances"]
    )


def iterations(result: dict) -> int | None:
    pattern = (
        r"Iterations:\s+(\d+)"
        if result["solver"] == "ContactIPM"
        else r"solved in\s+(\d+)\s+iterations"
    )
    match = re.search(pattern, result["stdout"])
    return int(match.group(1)) if match else None


def median_or_none(values: list[float]) -> float | None:
    return statistics.median(values) if values else None


def summarize_point(
    document: dict, problem: str, nodes: int, time_step: float
) -> dict:
    spec = PROBLEMS[problem]
    groups = {}
    eligible_by_key = {}
    for solver in ("ContactIPM", "CRISP"):
        rows = [row for row in document["results"] if row["solver"] == solver]
        eligible = [
            row for row in rows if row["returncode"] == 0 and audit_passed(row)
        ]
        eligible_by_key[solver] = {
            (row["repetition"], row.get("case_id", "source")): row
            for row in eligible
        }
        parsed_iterations = [
            value for row in eligible if (value := iterations(row)) is not None
        ]
        groups[solver] = {
            "attempted": len(rows),
            "solver_converged": sum(row["returncode"] == 0 for row in rows),
            "audit_passed": sum(audit_passed(row) for row in rows),
            "eligible": len(eligible),
            "eligible_wall_times_seconds": [
                row["wall_time_seconds"] for row in eligible
            ],
            "median_eligible_wall_time_seconds": median_or_none(
                [row["wall_time_seconds"] for row in eligible]
            ),
            "eligible_iterations": parsed_iterations,
            "median_eligible_iterations": median_or_none(parsed_iterations),
        }
    common_keys = sorted(
        set(eligible_by_key["ContactIPM"]) & set(eligible_by_key["CRISP"])
    )
    ratios = [
        eligible_by_key["CRISP"][key]["wall_time_seconds"]
        / eligible_by_key["ContactIPM"][key]["wall_time_seconds"]
        for key in common_keys
    ]
    state, control = spec["contact_width"]
    return {
        "problem": problem,
        "nodes": nodes,
        "intervals": nodes - 1,
        "dt": time_step,
        "physical_duration": time_step * (nodes - 1),
        "total_complementarity_pairs": spec["pairs_per_stage"] * (nodes - 1),
        "contactipm_source_primal_variables": nodes * state
        + (nodes - 1) * control,
        "crisp_source_primal_variables": nodes * spec["crisp_width"],
        "solvers": groups,
        "qualified_pairs": len(common_keys),
        "paired_speedups_crisp_over_contactipm": ratios,
        "median_paired_speedup_crisp_over_contactipm": median_or_none(ratios),
    }


def fmt(value: float | None, precision: int = 4) -> str:
    return "n/a" if value is None else f"{value:.{precision}g}"


def render_markdown(document: dict) -> str:
    lines = [
        f"# {document['problem'].replace('_', ' ').title()} scaling study",
        "",
        "Node count and time step are compiled into both solvers while total "
        "physical duration, physics, target, and per-stage contact geometry remain "
        "fixed. Build and discarded "
        "warmup/model-generation times are excluded. A timed run is eligible only "
        "when the solver exits successfully and the common trajectory audit passes. "
        "No execution time from either paper is used.",
        "",
        "The downloaded per-stage objective weights are unchanged, so both solvers "
        "solve the same discrete objective at each node count. The objective is not "
        "rescaled as a continuous-time quadrature across node counts; this study "
        "isolates solver scaling, not discretization-invariant optimal control cost.",
        "",
        "| Nodes | dt (s) | MPCC pairs | ContactIPM eligible | CRISP eligible | ContactIPM median (s) | CRISP median (s) | Paired speedup | ContactIPM iterations | CRISP iterations |",
        "|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|",
    ]
    for point in document["points"]:
        contact = point["solvers"]["ContactIPM"]
        crisp = point["solvers"]["CRISP"]
        speedup = point["median_paired_speedup_crisp_over_contactipm"]
        speedup_text = "n/a" if speedup is None else f"{speedup:.3g}x"
        lines.append(
            "| {nodes} | {dt} | {pairs} | {contact_ok}/{contact_n} | "
            "{crisp_ok}/{crisp_n} | {contact_time} | {crisp_time} | "
            "{speedup} | {contact_iterations} | {crisp_iterations} |".format(
                nodes=point["nodes"],
                dt=fmt(point["dt"], 5),
                pairs=point["total_complementarity_pairs"],
                contact_ok=contact["eligible"],
                contact_n=contact["attempted"],
                crisp_ok=crisp["eligible"],
                crisp_n=crisp["attempted"],
                contact_time=fmt(contact["median_eligible_wall_time_seconds"]),
                crisp_time=fmt(crisp["median_eligible_wall_time_seconds"]),
                speedup=speedup_text,
                contact_iterations=fmt(
                    contact["median_eligible_iterations"], 5
                ),
                crisp_iterations=fmt(crisp["median_eligible_iterations"], 5),
            )
        )
    lines.extend(
        [
            "",
            "ContactIPM source-variable counts reflect its exact elimination of "
            "Push T split equalities; CRISP counts retain the downloaded source "
            "variables. These counts exclude solver-internal slacks and duals.",
        ]
    )
    return "\n".join(lines) + "\n"


def run_checked(command: list[str], cwd: Path) -> None:
    print("+", " ".join(command), flush=True)
    completed = subprocess.run(command, cwd=cwd)
    if completed.returncode != 0:
        raise RuntimeError(
            f"command failed with status {completed.returncode}: {command[0]}"
        )


def reset_crisp_model_cache(build_root: Path, crisp_build: Path) -> None:
    cache = (crisp_build / "examples" / "model").resolve()
    root = build_root.resolve()
    if root not in cache.parents:
        raise RuntimeError(
            f"refusing to remove model cache outside build root: {cache}"
        )
    if cache.exists():
        shutil.rmtree(cache)


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--problem", choices=tuple(PROBLEMS), required=True)
    parser.add_argument("--nodes", nargs="+", type=int, required=True)
    parser.add_argument("--build-root", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--crisp-root", type=Path)
    parser.add_argument("--repetitions", type=int, default=5)
    parser.add_argument("--warmups", type=int, default=1)
    parser.add_argument("--threads", type=int, default=1)
    parser.add_argument("--seed", type=int, default=2027)
    parser.add_argument("--cpu-affinity", default="auto")
    parser.add_argument("--push-t-segment", type=int, default=8)
    parser.add_argument("--build-jobs", type=int, default=1)
    args = parser.parse_args()
    if (
        any(nodes < 2 for nodes in args.nodes)
        or len(set(args.nodes)) != len(args.nodes)
        or args.repetitions < 1
        or args.warmups < 0
        or args.threads < 1
        or args.build_jobs < 1
    ):
        parser.error("node counts and positive counts must be valid and unique")

    repo = Path(__file__).resolve().parents[2]
    crisp_root = (args.crisp_root or repo / "benchmarks" / "CRISP").resolve()
    crisp_source = crisp_root / "src" / "examples"
    scaling_marker = "// CONTACT_BENCHMARK_SCALING"
    source = (
        crisp_source / "pushbox" / "SolvePushbox.cpp"
        if args.problem == "push_box"
        else crisp_source / "pushT" / "SolvePushT.cpp"
    )
    if scaling_marker not in source.read_text(encoding="utf-8"):
        parser.error("run instrument_crisp.py before the scaling study")

    spec = PROBLEMS[args.problem]
    build_root = args.build_root.resolve()
    output_dir = args.output_dir.resolve()
    build_root.mkdir(parents=True, exist_ok=True)
    output_dir.mkdir(parents=True, exist_ok=True)
    runner = Path(__file__).with_name("run_benchmarks.py")
    points = []
    raw_artifacts = []
    revision = None
    machine = None
    for nodes in args.nodes:
        time_step = spec["physical_duration"] / (nodes - 1)
        time_step_text = format(time_step, ".17g")
        root = build_root / f"{args.problem}_nodes_{nodes}"
        contact_build = root / "contactipm"
        crisp_build = root / "crisp"
        run_checked(
            [
                "cmake",
                "-S",
                str(repo),
                "-B",
                str(contact_build),
                "-DCMAKE_BUILD_TYPE=Release",
                "-DNMPC_BUILD_EXAMPLES=OFF",
                "-DNMPC_BUILD_TESTS=OFF",
                "-DNMPC_BUILD_CONTACT_BENCHMARKS=ON",
                f"-D{spec['cmake_option']}={nodes}",
                f"-D{spec['dt_option']}={time_step_text}",
            ],
            repo,
        )
        run_checked(
            [
                "cmake",
                "--build",
                str(contact_build),
                "--target",
                spec["contact_target"],
                "-j",
                str(args.build_jobs),
            ],
            repo,
        )
        run_checked(
            [
                "cmake",
                "-S",
                str(crisp_root / "src"),
                "-B",
                str(crisp_build),
                "-DCMAKE_BUILD_TYPE=Release",
                (
                    f"-DCMAKE_CXX_FLAGS=-DCONTACT_BENCHMARK_NODES={nodes} "
                    f"-DCONTACT_BENCHMARK_DT={time_step_text}"
                ),
            ],
            repo,
        )
        run_checked(
            [
                "cmake",
                "--build",
                str(crisp_build),
                "--target",
                spec["crisp_target"],
                "-j",
                str(args.build_jobs),
            ],
            repo,
        )
        reset_crisp_model_cache(build_root, crisp_build)
        raw = output_dir / f"{args.problem}_nodes_{nodes}.json"
        command = [
            sys.executable,
            str(runner),
            "--contactipm-build",
            str(contact_build),
            "--crisp-build",
            str(crisp_build),
            "--output",
            str(raw),
            "--problems",
            args.problem,
            "--nodes",
            str(nodes),
            "--dt",
            time_step_text,
            "--repetitions",
            str(args.repetitions),
            "--warmups",
            str(args.warmups),
            "--threads",
            str(args.threads),
            "--seed",
            str(args.seed),
            "--cpu-affinity",
            args.cpu_affinity,
        ]
        if args.problem == "push_t":
            command.extend(("--push-t-segment", str(args.push_t_segment)))
        completed = subprocess.run(command, cwd=repo)
        if completed.returncode not in (0, 1):
            raise RuntimeError(
                f"benchmark runner failed with status {completed.returncode}"
            )
        raw_document = json.loads(raw.read_text(encoding="utf-8"))
        if revision is None:
            revision = raw_document["revision"]
            machine = raw_document["machine"]
        elif raw_document["revision"] != revision:
            raise RuntimeError("solver provenance changed during scaling study")
        points.append(
            summarize_point(raw_document, args.problem, nodes, time_step)
        )
        raw_artifacts.append(raw.name)

    summary = {
        "schema_version": 1,
        "description": "Matched compiled node/complementarity-count scaling study.",
        "problem": args.problem,
        "nodes": list(args.nodes),
        "push_t_segment": (
            args.push_t_segment if args.problem == "push_t" else None
        ),
        "protocol": {
            "repetitions": args.repetitions,
            "warmups": args.warmups,
            "threads": args.threads,
            "seed": args.seed,
            "cpu_affinity": args.cpu_affinity,
            "paper_reported_times_used": False,
            "fixed_physical_duration": True,
            "continuous_time_objective_rescaled": False,
        },
        "revision": revision,
        "machine": machine,
        "raw_artifacts": raw_artifacts,
        "points": points,
    }
    save(output_dir / "summary.json", summary)
    (output_dir / "summary.md").write_text(
        render_markdown(summary), encoding="utf-8"
    )
    print(f"Scaling summary: {(output_dir / 'summary.md').resolve()}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
