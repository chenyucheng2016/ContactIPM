#!/usr/bin/env python3
"""Run reproducible ContactIPM Push Box component ablations."""

from __future__ import annotations

import argparse
import json
import os
import statistics
import subprocess
import sys
from pathlib import Path


HERE = Path(__file__).resolve().parent
CONFIGURATIONS = [
    {"id": "full", "environment": {}},
    {
        "id": "no_recovery",
        "environment": {"CONTACTIPM_PUSH_RECOVERY": "0"},
    },
    {
        "id": "no_preconditioner",
        "environment": {"CONTACTIPM_PUSH_PRECONDITIONER": "0"},
    },
    {
        "id": "gauss_newton_primary",
        "environment": {"CONTACTIPM_PUSH_EXACT_HESSIAN": "0"},
    },
    {
        "id": "recovery_mu_0p01",
        "environment": {"CONTACTIPM_PUSH_RECOVERY_MU": "0.01"},
    },
    {
        "id": "recovery_mu_1",
        "environment": {"CONTACTIPM_PUSH_RECOVERY_MU": "1.0"},
    },
]


def summarize_artifact(document: dict) -> dict:
    rows = [
        result
        for result in document.get("results", [])
        if result.get("solver") == "ContactIPM"
    ]
    audits = [
        result.get("metrics", {}).get("trajectory_audit")
        for result in rows
    ]
    audits = [audit for audit in audits if audit is not None]
    instances = [
        instance
        for audit in audits
        for instance in audit.get("instances", [])
    ]
    objectives = [
        float(instance["objective"])
        for instance in instances
        if instance.get("success") and instance.get("objective") is not None
    ]
    times = [float(result["wall_time_seconds"]) for result in rows]
    return {
        "processes": len(rows),
        "process_exit_successful": sum(
            result.get("returncode") == 0 for result in rows
        ),
        "attempted": sum(audit.get("attempted_instances", 0) for audit in audits),
        "feasible": sum(audit.get("feasible_instances", 0) for audit in audits),
        "task_successful": sum(
            audit.get("task_successful_instances", 0) for audit in audits
        ),
        "successful": sum(
            audit.get("successful_instances", 0) for audit in audits
        ),
        "median_wall_time_seconds": statistics.median(times) if times else None,
        "total_wall_time_seconds": sum(times),
        "median_successful_objective": (
            statistics.median(objectives) if objectives else None
        ),
    }


def format_number(value: float | None) -> str:
    return "n/a" if value is None else f"{value:.6g}"


def render_markdown(summary: dict) -> str:
    lines = [
        "# Push Box solver-component ablation",
        "",
        "Each configuration is evaluated once on every one of the 25 frozen "
        "initial-state/target cases. This is robustness evidence, not repeated "
        "timing evidence.",
        "",
        "| Configuration | Environment overrides | Audited success | Feasible | "
        "Task success | Exit 0 | Median process wall (s) | Total wall (s) | "
        "Median objective on success |",
        "|---|---|---:|---:|---:|---:|---:|---:|---:|",
    ]
    for row in summary["configurations"]:
        metrics = row["metrics"]
        overrides = ", ".join(
            f"{name}={value}" for name, value in row["environment"].items()
        ) or "none"
        lines.append(
            f"| {row['id']} | {overrides} | "
            f"{metrics['successful']}/{metrics['attempted']} | "
            f"{metrics['feasible']}/{metrics['attempted']} | "
            f"{metrics['task_successful']}/{metrics['attempted']} | "
            f"{metrics['process_exit_successful']}/{metrics['processes']} | "
            f"{format_number(metrics['median_wall_time_seconds'])} | "
            f"{format_number(metrics['total_wall_time_seconds'])} | "
            f"{format_number(metrics['median_successful_objective'])} |"
        )
    lines.append("")
    return "\n".join(lines)


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--contactipm-build", type=Path, required=True)
    parser.add_argument(
        "--case-suite", type=Path, default=HERE / "validation_cases.json"
    )
    parser.add_argument(
        "--output-dir", type=Path, default=HERE / "results" / "push_box_ablation"
    )
    parser.add_argument("--cpu-affinity", default="auto")
    parser.add_argument("--seed", type=int, default=2027)
    args = parser.parse_args()

    args.output_dir.mkdir(parents=True, exist_ok=True)
    configuration_summaries = []
    for configuration in CONFIGURATIONS:
        output = args.output_dir / f"{configuration['id']}.json"
        command = [
            sys.executable,
            str(HERE / "run_benchmarks.py"),
            "--contactipm-build",
            str(args.contactipm_build),
            "--problems",
            "push_box",
            "--case-suite",
            str(args.case_suite),
            "--repetitions",
            "1",
            "--warmups",
            "0",
            "--threads",
            "1",
            "--seed",
            str(args.seed),
            "--cpu-affinity",
            args.cpu_affinity,
            "--output",
            str(output),
        ]
        environment = {
            name: value
            for name, value in os.environ.items()
            if not name.startswith("CONTACTIPM_")
        }
        environment.update(configuration["environment"])
        print(f"ablation: {configuration['id']}", flush=True)
        completed = subprocess.run(command, env=environment, check=False)
        if completed.returncode not in (0, 1):
            raise RuntimeError(
                f"{configuration['id']}: runner failed with "
                f"exit code {completed.returncode}"
            )
        document = json.loads(output.read_text(encoding="utf-8"))
        configuration_summaries.append(
            {
                **configuration,
                "artifact": str(output.resolve()),
                "metrics": summarize_artifact(document),
            }
        )

    summary = {
        "schema_version": 1,
        "problem": "push_box",
        "case_suite": str(args.case_suite.resolve()),
        "configurations": configuration_summaries,
    }
    json_output = args.output_dir / "summary.json"
    markdown_output = args.output_dir / "summary.md"
    json_output.write_text(
        json.dumps(summary, indent=2, sort_keys=True), encoding="utf-8"
    )
    markdown_output.write_text(render_markdown(summary), encoding="utf-8")
    print(f"summary JSON: {json_output.resolve()}")
    print(f"summary Markdown: {markdown_output.resolve()}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
