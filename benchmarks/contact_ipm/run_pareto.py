#!/usr/bin/env python3
"""Run matched tracking-effort sweeps and summarize physical tradeoffs."""

from __future__ import annotations

import argparse
import json
import os
import subprocess
import sys
from pathlib import Path


PROBLEM_LABELS = {"push_box": "Push Box", "push_t": "Push T segment 8"}


def scale_slug(value: float) -> str:
    return format(value, ".6g").replace("-", "m").replace(".", "p")


def save(path: Path, document: dict) -> None:
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(
        json.dumps(document, indent=2, sort_keys=True), encoding="utf-8"
    )
    temporary.replace(path)


def extract_points(document: dict, effort_scale: float, tracking_scale: float) -> list[dict]:
    points = []
    for result in document["results"]:
        audit = result["metrics"].get("trajectory_audit")
        if audit is None or audit["attempted_instances"] != 1:
            raise RuntimeError("Pareto runs require exactly one audited instance")
        instance = audit["instances"][0]
        components = instance.get("objective_components", {})
        tracking_cost = components.get("terminal_tracking", 0.0) + components.get(
            "running_tracking", 0.0
        )
        points.append(
            {
                "solver": result["solver"],
                "problem": result["problem"],
                "effort_scale": effort_scale,
                "tracking_scale": tracking_scale,
                "solver_converged": result["returncode"] == 0,
                "successful": bool(instance["success"]),
                "feasible": bool(instance["feasible"]),
                "task_successful": bool(instance["task_success"]),
                "position_error": instance.get(
                    "translation_error", instance.get("position_error")
                ),
                "angular_error": instance.get("angular_error", 0.0),
                "physical_mpcc": instance["physical_mpcc"],
                "tracking_cost": tracking_cost,
                "effort_cost": components.get("force_effort"),
                "objective_components": components,
                "actuation": instance.get("actuation", {}),
                "returncode": result["returncode"],
                "wall_time_seconds": result["wall_time_seconds"],
            }
        )
    return points


def mark_efficient(points: list[dict]) -> None:
    for point in points:
        point["pareto_efficient"] = False
        if (
            not point.get("solver_converged", True)
            or not point["successful"]
            or point["effort_cost"] is None
        ):
            continue
        competitors = [
            other
            for other in points
            if other["solver"] == point["solver"]
            and other["problem"] == point["problem"]
            and other.get("solver_converged", True)
            and other["successful"]
            and other["effort_cost"] is not None
        ]
        point["pareto_efficient"] = not any(
            other["effort_cost"] <= point["effort_cost"]
            and other["tracking_cost"] <= point["tracking_cost"]
            and (
                other["effort_cost"] < point["effort_cost"]
                or other["tracking_cost"] < point["tracking_cost"]
            )
            for other in competitors
        )


def fmt(value: float | None, precision: int = 4) -> str:
    return "n/a" if value is None else f"{value:.{precision}g}"


def render_markdown(document: dict) -> str:
    revision = document["revision"]
    source_snapshot = revision.get(
        "contactipm_source_snapshot", revision.get("contactipm", "unknown")
    )
    lines = [
        "# Matched tracking-effort Pareto study",
        "",
        "Both solvers optimize the same source objective with the listed positive "
        "effort multiplier and a fixed tracking multiplier. Reported tracking and "
        "effort costs use the unscaled common audit definitions, so points remain "
        "comparable across multipliers. Nondominance requires solver convergence "
        "and a successful common audit. Process times are diagnostic only.",
        "",
        f"ContactIPM source snapshot: `{source_snapshot}`",
        "",
    ]
    for problem in document["problems"]:
        label = PROBLEM_LABELS[problem]
        if problem == "push_t":
            label = f"Push T segment {document['push_t_segment']}"
        lines.extend(
            [
                f"## {label}",
                "",
                "| Solver | Effort scale | Solver exit | Audit | Effort cost | Tracking cost | Position error | Angle error | Peak force | Nondominated |",
                "|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|",
            ]
        )
        for point in sorted(
            (row for row in document["points"] if row["problem"] == problem),
            key=lambda row: (row["solver"], row["effort_scale"]),
        ):
            peak_force = point["actuation"].get("peak_contact_force_norm")
            lines.append(
                "| {solver} | {scale:g} | {exit} | {audit} | {effort} | {tracking} | "
                "{position} | {angle} | {peak} | {efficient} |".format(
                    solver=point["solver"],
                    scale=point["effort_scale"],
                    exit="converged" if point["solver_converged"] else "failed",
                    audit="pass" if point["successful"] else "fail",
                    effort=fmt(point["effort_cost"]),
                    tracking=fmt(point["tracking_cost"]),
                    position=fmt(point["position_error"]),
                    angle=fmt(point["angular_error"]),
                    peak=fmt(peak_force),
                    efficient="yes" if point["pareto_efficient"] else "no",
                )
            )
        lines.append("")
    return "\n".join(lines)


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--contactipm-build", type=Path, required=True)
    parser.add_argument("--crisp-build", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument(
        "--problems",
        nargs="+",
        choices=tuple(PROBLEM_LABELS),
        default=tuple(PROBLEM_LABELS),
    )
    parser.add_argument(
        "--effort-scales",
        nargs="+",
        type=float,
        default=(0.1, 0.25, 0.5, 1.0, 2.0, 4.0, 10.0),
    )
    parser.add_argument("--tracking-scale", type=float, default=1.0)
    parser.add_argument("--push-t-segment", type=int, default=8)
    parser.add_argument("--threads", type=int, default=1)
    parser.add_argument("--seed", type=int, default=2027)
    parser.add_argument("--cpu-affinity", default="auto")
    args = parser.parse_args()
    if args.tracking_scale <= 0.0 or any(value <= 0.0 for value in args.effort_scales):
        parser.error("objective scales must be positive")
    if len(set(args.effort_scales)) != len(args.effort_scales):
        parser.error("effort scales must be unique")

    repo = Path(__file__).resolve().parents[2]
    output_dir = args.output_dir.resolve()
    output_dir.mkdir(parents=True, exist_ok=True)
    runner = Path(__file__).with_name("run_benchmarks.py")
    points = []
    raw_artifacts = []
    first_document = None
    source_hashes = {"contactipm": {}, "crisp": {}}
    stable_revision_fields = (
        "contactipm_source_snapshot",
        "contactipm_source_manifest",
        "contactipm_tracked_dirty",
        "crisp",
        "crisp_tracked_dirty",
        "crisp_tracked_patch",
    )
    for problem in args.problems:
        for effort_scale in args.effort_scales:
            output = output_dir / f"{problem}_effort_{scale_slug(effort_scale)}.json"
            command = [
                sys.executable,
                str(runner),
                "--contactipm-build",
                str(args.contactipm_build.resolve()),
                "--crisp-build",
                str(args.crisp_build.resolve()),
                "--output",
                str(output),
                "--problems",
                problem,
                "--repetitions",
                "1",
                "--warmups",
                "0",
                "--threads",
                str(args.threads),
                "--seed",
                str(args.seed),
                "--cpu-affinity",
                args.cpu_affinity,
            ]
            if problem == "push_t":
                command.extend(("--push-t-segment", str(args.push_t_segment)))
            environment = os.environ.copy()
            environment["CONTACT_BENCHMARK_TRACKING_SCALE"] = format(
                args.tracking_scale, ".17g"
            )
            environment["CONTACT_BENCHMARK_EFFORT_SCALE"] = format(
                effort_scale, ".17g"
            )
            completed = subprocess.run(command, cwd=repo, env=environment)
            if completed.returncode not in (0, 1):
                raise RuntimeError(
                    f"benchmark runner failed with status {completed.returncode}"
                )
            document = json.loads(output.read_text(encoding="utf-8"))
            if len(document.get("results", [])) != 2:
                raise RuntimeError("incomplete matched Pareto point")
            if first_document is None:
                first_document = document
            elif any(
                document["revision"].get(field)
                != first_document["revision"].get(field)
                for field in stable_revision_fields
            ):
                raise RuntimeError("solver provenance changed during Pareto sweep")
            source_hashes["contactipm"].update(
                document["revision"].get("contactipm_source_sha256", {})
            )
            source_hashes["crisp"].update(
                document["revision"].get("crisp_source_sha256", {})
            )
            points.extend(
                extract_points(document, effort_scale, args.tracking_scale)
            )
            raw_artifacts.append(output.name)

    mark_efficient(points)
    if first_document is None:
        raise RuntimeError("no Pareto problems selected")
    revision = dict(first_document["revision"])
    revision["contactipm_source_sha256"] = source_hashes["contactipm"]
    revision["crisp_source_sha256"] = source_hashes["crisp"]
    summary = {
        "schema_version": 1,
        "description": "Matched objective-weight sweep; not a timing study.",
        "created_utc": first_document["created_utc"],
        "problems": list(args.problems),
        "effort_scales": list(args.effort_scales),
        "tracking_scale": args.tracking_scale,
        "push_t_segment": args.push_t_segment,
        "raw_artifacts": raw_artifacts,
        "revision": revision,
        "machine": first_document["machine"],
        "points": points,
    }
    save(output_dir / "summary.json", summary)
    (output_dir / "summary.md").write_text(
        render_markdown(summary).rstrip() + "\n", encoding="utf-8"
    )
    print(f"Pareto summary: {(output_dir / 'summary.md').resolve()}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
