#!/usr/bin/env python3
"""Summarize audited, locally measured ContactIPM/CRISP benchmark trials."""

from __future__ import annotations

import argparse
import json
import math
import random
import statistics
from collections import defaultdict
from pathlib import Path


SOLVER_ORDER = {"ContactIPM": 0, "CRISP": 1}
PROBLEM_LABELS = {
    "cartpole_soft_walls": "Cartpole with Soft Walls",
    "push_box": "Push Box",
    "transport": "Transport",
    "push_t": "Push T",
}


def percentile(values: list[float], probability: float) -> float:
    ordered = sorted(values)
    if not ordered:
        raise ValueError("percentile of empty data")
    position = (len(ordered) - 1) * probability
    lower = math.floor(position)
    upper = math.ceil(position)
    if lower == upper:
        return ordered[lower]
    fraction = position - lower
    return ordered[lower] * (1.0 - fraction) + ordered[upper] * fraction


def describe(
    values: list[float], bootstrap_samples: int, rng: random.Random
) -> dict | None:
    if not values:
        return None
    medians = []
    for _ in range(bootstrap_samples):
        sample = [values[rng.randrange(len(values))] for _ in values]
        medians.append(statistics.median(sample))
    return {
        "count": len(values),
        "median": statistics.median(values),
        "q1": percentile(values, 0.25),
        "q3": percentile(values, 0.75),
        "iqr": percentile(values, 0.75) - percentile(values, 0.25),
        "p90": percentile(values, 0.90),
        "median_ci95": [percentile(medians, 0.025), percentile(medians, 0.975)],
    }


def audit_of(result: dict) -> dict | None:
    return result.get("metrics", {}).get("trajectory_audit")


def qualified(result: dict) -> bool:
    audit = audit_of(result)
    return bool(
        audit is not None
        and audit.get("successful_instances") == audit.get("attempted_instances")
    )


def aggregate_group(
    solver: str,
    problem: str,
    results: list[dict],
    bootstrap_samples: int,
    rng: random.Random,
) -> dict:
    audits = [audit for result in results if (audit := audit_of(result)) is not None]
    instances = [
        instance
        for audit in audits
        for instance in audit.get("instances", [])
    ]
    qualified_results = [result for result in results if qualified(result)]

    quality_instances = [
        instance for instance in instances if instance.get("success", True)
    ]
    component_values: dict[str, list[float]] = defaultdict(list)
    actuation_values: dict[str, list[float]] = defaultdict(list)
    objectives = []
    for instance in quality_instances:
        if instance.get("objective") is not None:
            objectives.append(float(instance["objective"]))
        for name, value in instance.get("objective_components", {}).items():
            component_values[name].append(float(value))
        for name, value in instance.get("actuation", {}).items():
            actuation_values[name].append(float(value))

    worst_fields = {
        "equality": "worst_equality",
        "side_violation": "worst_side_violation",
        "physical_mpcc": "worst_physical_mpcc",
        "position_error": "worst_position_error",
        "angular_error": "worst_angular_error",
        "velocity_error": "worst_velocity_error",
    }
    worst = {
        name: max((float(audit.get(field, 0.0)) for audit in audits), default=None)
        for name, field in worst_fields.items()
    }

    return {
        "solver": solver,
        "problem": problem,
        "trials": len(results),
        "process_successful_trials": sum(
            result.get("returncode") == 0 for result in results
        ),
        "qualified_trials": len(qualified_results),
        "audited_trials": len(audits),
        "instances": {
            "attempted": sum(audit.get("attempted_instances", 0) for audit in audits),
            "feasible": sum(audit.get("feasible_instances", 0) for audit in audits),
            "task_successful": sum(
                audit.get("task_successful_instances", 0) for audit in audits
            ),
            "successful": sum(
                audit.get("successful_instances", 0) for audit in audits
            ),
        },
        "timing_all_processes_seconds": describe(
            [float(result["wall_time_seconds"]) for result in results],
            bootstrap_samples,
            rng,
        ),
        "timing_qualified_seconds": describe(
            [float(result["wall_time_seconds"]) for result in qualified_results],
            bootstrap_samples,
            rng,
        ),
        "worst": worst,
        "objective_per_instance": describe(objectives, bootstrap_samples, rng),
        "objective_components": {
            name: describe(values, bootstrap_samples, rng)
            for name, values in sorted(component_values.items())
        },
        "actuation": {
            name: describe(values, bootstrap_samples, rng)
            for name, values in sorted(actuation_values.items())
        },
    }


def paired_comparisons(
    results: list[dict],
    bootstrap_samples: int,
    rng: random.Random,
    protocol_ready: bool,
    min_repetitions: int,
) -> list[dict]:
    indexed = {
        (
            result["problem"],
            result.get("case_id", "source"),
            int(result["repetition"]),
            result["solver"],
        ): result
        for result in results
    }
    comparisons = []
    observed_problems = {result["problem"] for result in results}
    ordered_problems = [
        problem for problem in PROBLEM_LABELS if problem in observed_problems
    ]
    ordered_problems.extend(sorted(observed_problems - set(PROBLEM_LABELS)))
    for problem in ordered_problems:
        case_repetitions = sorted(
            {
                (result.get("case_id", "source"), int(result["repetition"]))
                for result in results
                if result["problem"] == problem
            }
        )
        pairs = []
        for case_id, repetition in case_repetitions:
            contact = indexed.get((problem, case_id, repetition, "ContactIPM"))
            crisp = indexed.get((problem, case_id, repetition, "CRISP"))
            if contact is None or crisp is None:
                continue
            pairs.append((contact, crisp))
        qualified_pairs = [
            pair for pair in pairs if qualified(pair[0]) and qualified(pair[1])
        ]
        outcomes = {
            "both": sum(qualified(contact) and qualified(crisp) for contact, crisp in pairs),
            "contactipm_only": sum(
                qualified(contact) and not qualified(crisp) for contact, crisp in pairs
            ),
            "crisp_only": sum(
                not qualified(contact) and qualified(crisp) for contact, crisp in pairs
            ),
            "neither": sum(
                not qualified(contact) and not qualified(crisp)
                for contact, crisp in pairs
            ),
        }
        all_ratios = [
            float(crisp["wall_time_seconds"]) / float(contact["wall_time_seconds"])
            for contact, crisp in pairs
        ]
        qualified_ratios = [
            float(crisp["wall_time_seconds"]) / float(contact["wall_time_seconds"])
            for contact, crisp in qualified_pairs
        ]
        comparisons.append(
            {
                "problem": problem,
                "pairs": len(pairs),
                "qualified_pairs": len(qualified_pairs),
                "robustness_outcomes": outcomes,
                "all_process_speedup_crisp_over_contactipm": describe(
                    all_ratios, bootstrap_samples, rng
                ),
                "qualified_speedup_crisp_over_contactipm": describe(
                    qualified_ratios, bootstrap_samples, rng
                ),
                "publication_timing_ready": bool(
                    protocol_ready and len(qualified_pairs) >= min_repetitions
                ),
            }
        )
    return comparisons


def summarize(
    document: dict,
    bootstrap_samples: int,
    seed: int,
    min_repetitions: int,
) -> dict:
    protocol = document.get("protocol", {})
    issues = []
    if document.get("schema_version", 0) < 5:
        issues.append("schema version predates contiguous randomized pairing")
    if protocol.get("case_suite") is not None:
        issues.append(
            "multi-instance sweeps are robustness evidence, not repeated timing"
        )
    if protocol.get("contactipm_environment"):
        issues.append("ContactIPM environment overrides indicate an ablation run")
    if protocol.get("benchmark_objective_environment"):
        issues.append("objective-weight overrides indicate a Pareto study")
    if protocol.get("scaling"):
        issues.append("node-count override indicates a scaling study")
    if protocol.get("warmups", 0) < 1:
        issues.append("at least one warmup is required")
    if protocol.get("repetitions", 0) < min_repetitions:
        issues.append(f"fewer than {min_repetitions} measured repetitions")
    if protocol.get("cpu_affinity") is None:
        issues.append("CPU affinity was not explicitly pinned")
    if protocol.get("paper_reported_times_used") is not False:
        issues.append("timing provenance does not exclude paper-reported times")
    protocol_ready = not issues

    rng = random.Random(seed)
    grouped: dict[tuple[str, str], list[dict]] = defaultdict(list)
    for result in document.get("results", []):
        grouped[(result["solver"], result["problem"])].append(result)
    groups = [
        aggregate_group(solver, problem, rows, bootstrap_samples, rng)
        for (solver, problem), rows in sorted(
            grouped.items(),
            key=lambda item: (
                list(PROBLEM_LABELS).index(item[0][1]),
                SOLVER_ORDER.get(item[0][0], 99),
            ),
        )
    ]
    pairs = paired_comparisons(
        document.get("results", []),
        bootstrap_samples,
        rng,
        protocol_ready,
        min_repetitions,
    )
    return {
        "schema_version": 1,
        "source_schema_version": document.get("schema_version"),
        "source_created_utc": document.get("created_utc"),
        "protocol": protocol,
        "machine": document.get("machine", {}),
        "revision": document.get("revision", {}),
        "bootstrap": {"samples": bootstrap_samples, "seed": seed},
        "readiness": {
            "protocol_ready": protocol_ready,
            "minimum_repetitions": min_repetitions,
            "issues": issues,
            "all_problem_timings_ready": bool(pairs)
            and all(pair["publication_timing_ready"] for pair in pairs),
        },
        "groups": groups,
        "paired_comparisons": pairs,
    }


def fmt(value: float | None, precision: int = 3) -> str:
    if value is None:
        return "n/a"
    return f"{value:.{precision}g}"


def timing_cell(stats: dict | None) -> str:
    if stats is None:
        return "n/a"
    return (
        f"{stats['median']:.4f} "
        f"[{stats['q1']:.4f}, {stats['q3']:.4f}], p90 {stats['p90']:.4f}"
    )


def render_markdown(summary: dict, source: Path) -> str:
    readiness = summary["readiness"]
    revision = summary.get("revision", {})
    source_snapshot = revision.get(
        "contactipm_source_snapshot", revision.get("contactipm", "unknown")
    )
    lines = [
        "# Local contact benchmark summary",
        "",
        f"Source artifact: `{source.as_posix()}`",
        "",
        (
            "All execution times in this report were measured locally; no runtime "
            "reported in the CRISP paper is used."
        ),
        "",
        f"ContactIPM source snapshot: `{source_snapshot}`",
        "",
    ]
    if readiness["issues"]:
        lines.extend(
            [
                "This run is correctness/development evidence, not publication-ready "
                "timing evidence, because:",
                "",
                *[f"- {issue}" for issue in readiness["issues"]],
                "",
            ]
        )
    else:
        lines.extend(["The timing protocol meets the configured readiness gates.", ""])

    lines.extend(
        [
            "## Feasibility and task quality",
            "",
            "| Solver | Problem | Qualified trials | Audited success | Task success | Worst equality | Worst side | Worst MPCC | Worst position | Worst angle | Worst velocity |",
            "|---|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|",
        ]
    )
    for group in summary["groups"]:
        instances = group["instances"]
        worst = group["worst"]
        lines.append(
            "| {solver} | {problem} | {qualified}/{trials} | {success}/{attempted} | "
            "{task}/{attempted} | {eq} | {side} | {mpcc} | {position} | "
            "{angle} | {velocity} |".format(
                solver=group["solver"],
                problem=PROBLEM_LABELS.get(group["problem"], group["problem"]),
                qualified=group["qualified_trials"],
                trials=group["trials"],
                success=instances["successful"],
                attempted=instances["attempted"],
                task=instances["task_successful"],
                eq=fmt(worst["equality"]),
                side=fmt(worst["side_violation"]),
                mpcc=fmt(worst["physical_mpcc"]),
                position=fmt(worst["position_error"]),
                angle=fmt(worst["angular_error"]),
                velocity=fmt(worst["velocity_error"]),
            )
        )

    lines.extend(
        [
            "",
            "## Paired robustness outcomes",
            "",
            "Each initial-state/target pair is counted once using the common trajectory audit.",
            "",
            "| Problem | Both succeed | ContactIPM only | CRISP only | Neither |",
            "|---|---:|---:|---:|---:|",
        ]
    )
    for pair in summary["paired_comparisons"]:
        outcomes = pair["robustness_outcomes"]
        lines.append(
            f"| {PROBLEM_LABELS.get(pair['problem'], pair['problem'])} | "
            f"{outcomes['both']} | {outcomes['contactipm_only']} | "
            f"{outcomes['crisp_only']} | {outcomes['neither']} |"
        )

    group_index = {
        (group["problem"], group["solver"]): group for group in summary["groups"]
    }
    lines.extend(
        [
            "",
            "## Locally measured process wall time",
            "",
            "Values are median [Q1, Q3] and P90 in seconds over all measured processes. "
            "A paired speedup is reported only when both trajectories pass the common audit.",
            "",
            "| Problem | ContactIPM | CRISP | Qualified pairs | CRISP / ContactIPM paired speedup (95% CI) | Ready |",
            "|---|---:|---:|---:|---:|---:|",
        ]
    )
    for pair in summary["paired_comparisons"]:
        problem = pair["problem"]
        contact = group_index.get((problem, "ContactIPM"))
        crisp = group_index.get((problem, "CRISP"))
        speed = pair["qualified_speedup_crisp_over_contactipm"]
        if speed is None:
            speed_cell = "n/a"
        else:
            speed_cell = (
                f"{speed['median']:.3f}x "
                f"[{speed['median_ci95'][0]:.3f}, {speed['median_ci95'][1]:.3f}]"
            )
        contact_timing = contact["timing_all_processes_seconds"] if contact else None
        crisp_timing = crisp["timing_all_processes_seconds"] if crisp else None
        lines.append(
            f"| {PROBLEM_LABELS.get(problem, problem)} | "
            f"{timing_cell(contact_timing)} | "
            f"{timing_cell(crisp_timing)} | "
            f"{pair['qualified_pairs']}/{pair['pairs']} | {speed_cell} | "
            f"{'yes' if pair['publication_timing_ready'] else 'no'} |"
        )

    lines.extend(["", "## Objective decomposition", ""])
    lines.extend(
        [
            "Objective values are shown after feasibility and task metrics; they are not "
            "used as a standalone success criterion.",
            "",
            "| Solver | Problem | Objective per instance (median) | Component medians |",
            "|---|---|---:|---|",
        ]
    )
    for group in summary["groups"]:
        objective = group["objective_per_instance"]
        components = ", ".join(
            f"{name}={stats['median']:.6g}"
            for name, stats in group["objective_components"].items()
        ) or "n/a"
        lines.append(
            f"| {group['solver']} | {PROBLEM_LABELS.get(group['problem'], group['problem'])} | "
            f"{fmt(objective['median'] if objective else None, 6)} | {components} |"
        )
    lines.append("")
    return "\n".join(lines)


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("input", type=Path)
    parser.add_argument("--json-output", type=Path)
    parser.add_argument("--markdown-output", type=Path)
    parser.add_argument("--bootstrap-samples", type=int, default=10000)
    parser.add_argument("--seed", type=int, default=2027)
    parser.add_argument("--min-repetitions", type=int, default=20)
    args = parser.parse_args()
    if args.bootstrap_samples < 1 or args.min_repetitions < 1:
        parser.error("bootstrap samples and minimum repetitions must be positive")

    source = args.input.resolve()
    document = json.loads(source.read_text(encoding="utf-8"))
    summary = summarize(
        document, args.bootstrap_samples, args.seed, args.min_repetitions
    )
    json_output = args.json_output or source.with_name(source.stem + "_summary.json")
    markdown_output = args.markdown_output or source.with_name(
        source.stem + "_summary.md"
    )
    json_output.write_text(
        json.dumps(summary, indent=2, sort_keys=True), encoding="utf-8"
    )
    markdown_output.write_text(
        render_markdown(summary, args.input), encoding="utf-8"
    )
    print(f"summary JSON: {json_output.resolve()}")
    print(f"summary Markdown: {markdown_output.resolve()}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
