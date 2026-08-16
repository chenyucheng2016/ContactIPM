#!/usr/bin/env python3
"""Summarize paired ContactIPM-versus-IMPACT artifacts."""

from __future__ import annotations

import argparse
import json
import math
import random
import statistics
from pathlib import Path
from typing import Any, Iterable


SOLVERS = ("contactipm", "impact")
LABELS = {"contactipm": "ContactIPM", "impact": "IMPACT"}
PROBLEM_LABELS = {
    "push_box": "Push Box",
    "push_t": "Push T",
    "cart_transport": "Cart Transport",
}


def percentile(values: Iterable[float], probability: float) -> float:
    ordered = sorted(values)
    if not ordered:
        return math.nan
    position = probability * (len(ordered) - 1)
    lower = math.floor(position)
    upper = math.ceil(position)
    if lower == upper:
        return ordered[lower]
    alpha = position - lower
    return (1.0 - alpha) * ordered[lower] + alpha * ordered[upper]


def distribution(values: Iterable[float]) -> dict[str, float | int | None]:
    finite = [value for value in values if math.isfinite(value)]
    if not finite:
        return {
            "count": 0,
            "median": None,
            "q25": None,
            "q75": None,
            "p90": None,
        }
    return {
        "count": len(finite),
        "median": statistics.median(finite),
        "q25": percentile(finite, 0.25),
        "q75": percentile(finite, 0.75),
        "p90": percentile(finite, 0.90),
    }


def wilson(successes: int, total: int, z: float = 1.959963984540054) -> list[float]:
    if total == 0:
        return [math.nan, math.nan]
    proportion = successes / total
    denominator = 1.0 + z * z / total
    center = (proportion + z * z / (2.0 * total)) / denominator
    radius = (
        z
        * math.sqrt(
            proportion * (1.0 - proportion) / total
            + z * z / (4.0 * total * total)
        )
        / denominator
    )
    return [max(0.0, center - radius), min(1.0, center + radius)]


def bootstrap_median_ci(
    values: list[float], seed: int = 2027, draws: int = 10000
) -> list[float | None]:
    if not values:
        return [None, None]
    rng = random.Random(seed)
    bootstrapped = []
    for _ in range(draws):
        sample = [values[rng.randrange(len(values))] for _ in values]
        bootstrapped.append(statistics.median(sample))
    return [percentile(bootstrapped, 0.025), percentile(bootstrapped, 0.975)]


def success_summary(pairs: list[dict[str, Any]], solver: str) -> dict[str, Any]:
    successes = sum(bool(pair[solver]["eligible"]) for pair in pairs)
    total = len(pairs)
    return {
        "successes": successes,
        "total": total,
        "fraction": successes / total if total else None,
        "wilson_95": wilson(successes, total),
    }


def audit_distributions(
    pairs: list[dict[str, Any]], solver: str
) -> dict[str, dict[str, float | int | None]]:
    audits = [
        pair[solver]["audit"]
        for pair in pairs
        if pair[solver]["eligible"] and pair[solver]["audit"] is not None
    ]
    keys = sorted(
        {
            key
            for audit in audits
            for key, value in audit.items()
            if isinstance(value, (int, float)) and not isinstance(value, bool)
        }
    )
    return {
        key: distribution(float(audit[key]) for audit in audits if key in audit)
        for key in keys
    }


def timing_summary(pairs: list[dict[str, Any]]) -> dict[str, Any]:
    eligible = [pair for pair in pairs if pair["pair_eligible"]]
    result: dict[str, Any] = {
        "eligible_pairs": len(eligible),
        "total_pairs": len(pairs),
    }
    for field in ("solver_seconds", "wall_seconds"):
        result[field] = {
            solver: distribution(
                float(pair[solver][field])
                for pair in eligible
                if pair[solver][field] is not None
            )
            for solver in SOLVERS
        }
        ratios = [
            float(pair["impact"][field]) / float(pair["contactipm"][field])
            for pair in eligible
            if pair["impact"][field] is not None
            and pair["contactipm"][field] is not None
            and float(pair["contactipm"][field]) > 0.0
        ]
        result[f"{field}_impact_over_contactipm"] = {
            **distribution(ratios),
            "bootstrap_median_95": bootstrap_median_ci(ratios),
        }
    return result


def summarize(
    robustness: dict[str, Any], timing: dict[str, Any] | None
) -> dict[str, Any]:
    source_snapshot = robustness.get("contactipm_source_snapshot")
    if timing is not None and timing.get("contactipm_source_snapshot") != source_snapshot:
        raise ValueError("ContactIPM source snapshot differs between artifacts")
    problems = sorted({pair["problem"] for pair in robustness["pairs"]})
    result: dict[str, Any] = {
        "schema_version": 1,
        "paper_reported_times_used": False,
        "contactipm_source_snapshot": source_snapshot,
        "robustness_artifact": robustness.get("comparison"),
        "problems": {},
    }
    for problem in problems:
        robust_pairs = [
            pair for pair in robustness["pairs"] if pair["problem"] == problem
        ]
        timing_pairs = (
            [pair for pair in timing["pairs"] if pair["problem"] == problem]
            if timing is not None
            else []
        )
        result["problems"][problem] = {
            "success": {
                solver: success_summary(robust_pairs, solver)
                for solver in SOLVERS
            },
            "quality": {
                solver: audit_distributions(robust_pairs, solver)
                for solver in SOLVERS
            },
            "timing": timing_summary(timing_pairs) if timing_pairs else None,
        }
    return result


def number(value: float | None, digits: int = 4) -> str:
    if value is None or not math.isfinite(value):
        return "n/a"
    if value == 0.0:
        return "0"
    if abs(value) < 1e-3 or abs(value) >= 1e4:
        return f"{value:.3e}"
    return f"{value:.{digits}f}"


def success_cell(summary: dict[str, Any]) -> str:
    low, high = summary["wilson_95"]
    return (
        f"{summary['successes']}/{summary['total']} "
        f"({100.0 * summary['fraction']:.1f}%; "
        f"95% CI {100.0 * low:.1f}–{100.0 * high:.1f}%)"
    )


def dist_cell(summary: dict[str, Any] | None) -> str:
    if not summary or summary["median"] is None:
        return "n/a"
    return (
        f"{number(summary['median'])} "
        f"[{number(summary['q25'])}, {number(summary['q75'])}], "
        f"P90 {number(summary['p90'])}"
    )


def metric_row(
    label: str,
    problem: dict[str, Any],
    key: str,
) -> str:
    values = []
    for solver in SOLVERS:
        values.append(
            dist_cell(problem["quality"][solver].get(key))
        )
    return f"| {label} | {values[0]} | {values[1]} |"


def markdown(summary: dict[str, Any]) -> str:
    lines = [
        "# ContactIPM versus IMPACT: local parameter-faithful comparison",
        "",
        "All values come from local executions on the same machine. "
        "No execution times reported in Paper 3 are used.",
        "",
        "Robustness statistics use one solve per predeclared local instance. "
        "Timing statistics use separately repeated, adjacent randomized pairs. "
        "A run is successful only when the solver reports convergence and the "
        "independent dynamics, side, equality, complementarity, and task audit passes.",
        "",
    ]
    for problem_name, problem in summary["problems"].items():
        lines.extend(
            [
                f"## {PROBLEM_LABELS[problem_name]}",
                "",
                "| Metric | ContactIPM | IMPACT |",
                "|---|---:|---:|",
                (
                    "| Audited success | "
                    f"{success_cell(problem['success']['contactipm'])} | "
                    f"{success_cell(problem['success']['impact'])} |"
                ),
                metric_row(
                    "Total tracking error (median [IQR], P90)",
                    problem,
                    "total_tracking_error",
                ),
                metric_row(
                    "Objective (median [IQR], P90)", problem, "objective"
                ),
                metric_row(
                    "Control cost (median [IQR], P90)",
                    problem,
                    "control_cost",
                ),
                metric_row(
                    "Force effort (median [IQR], P90)",
                    problem,
                    "force_effort",
                ),
                metric_row(
                    "Peak force (median [IQR], P90)", problem, "peak_force"
                ),
                metric_row(
                    "Dynamics defect (median [IQR], P90)",
                    problem,
                    "dynamics_defect",
                ),
                metric_row(
                    "Complementarity (median [IQR], P90)",
                    problem,
                    "complementarity",
                ),
            ]
        )
        if problem_name in ("push_box", "push_t"):
            lines.extend(
                [
                    metric_row(
                        "Terminal translation error (median [IQR], P90)",
                        problem,
                        "translation_error",
                    ),
                    metric_row(
                        "Terminal angular error (median [IQR], P90)",
                        problem,
                        "angular_error",
                    ),
                ]
            )
        else:
            lines.extend(
                [
                    metric_row(
                        "Terminal position error (median [IQR], P90)",
                        problem,
                        "position_error",
                    ),
                    metric_row(
                        "Terminal velocity error (median [IQR], P90)",
                        problem,
                        "velocity_error",
                    ),
                ]
            )
        timing = problem["timing"]
        if timing:
            solver_ratio = timing["solver_seconds_impact_over_contactipm"]
            wall_ratio = timing["wall_seconds_impact_over_contactipm"]
            lines.extend(
                [
                    (
                        "| Solver-only time (s; median [IQR], P90) | "
                        f"{dist_cell(timing['solver_seconds']['contactipm'])} | "
                        f"{dist_cell(timing['solver_seconds']['impact'])} |"
                    ),
                    (
                        "| End-to-end time (s; median [IQR], P90) | "
                        f"{dist_cell(timing['wall_seconds']['contactipm'])} | "
                        f"{dist_cell(timing['wall_seconds']['impact'])} |"
                    ),
                    "",
                    (
                        f"Paired IMPACT/ContactIPM median solver-only ratio: "
                        f"{number(solver_ratio['median'])}× "
                        f"(bootstrap 95% CI "
                        f"{number(solver_ratio['bootstrap_median_95'][0])}–"
                        f"{number(solver_ratio['bootstrap_median_95'][1])})."
                    ),
                    "",
                    (
                        f"Paired IMPACT/ContactIPM median end-to-end ratio: "
                        f"{number(wall_ratio['median'])}× "
                        f"(bootstrap 95% CI "
                        f"{number(wall_ratio['bootstrap_median_95'][0])}–"
                        f"{number(wall_ratio['bootstrap_median_95'][1])})."
                    ),
                ]
            )
        lines.extend(
            [
                "",
                "These statistics jointly expose reliability, task quality, "
                "physical feasibility, and effort; the scalar objective is not "
                "used as a stand-alone solution-quality verdict.",
                "",
            ]
        )
    return "\n".join(lines)


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--robustness", type=Path, required=True)
    parser.add_argument("--timing", type=Path)
    parser.add_argument("--output-prefix", type=Path, required=True)
    args = parser.parse_args()
    robustness = json.loads(args.robustness.read_text(encoding="utf-8"))
    timing = (
        json.loads(args.timing.read_text(encoding="utf-8"))
        if args.timing
        else None
    )
    result = summarize(robustness, timing)
    args.output_prefix.parent.mkdir(parents=True, exist_ok=True)
    args.output_prefix.with_suffix(".json").write_text(
        json.dumps(result, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    args.output_prefix.with_suffix(".md").write_text(
        markdown(result) + "\n", encoding="utf-8"
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
