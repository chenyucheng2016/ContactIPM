#!/usr/bin/env python3
"""Run paired ContactIPM, force-ablation, and fixed-schedule replays."""

import argparse
import csv
import math
import re
import subprocess
from pathlib import Path


CONDITIONS = (
    ("nominal", ()),
    ("mass_up_10", ("--mass-scale", "1.10")),
    ("mass_down_10", ("--mass-scale", "0.90")),
    ("friction_75", ("--friction-scale", "0.75")),
    ("lateral_push_2Ns", ("--push", "0", "20", "0", "0.90", "0.10")),
    ("combined", ("--mass-scale", "1.10", "--friction-scale", "0.75",
                  "--push", "0", "20", "0", "0.90", "0.10")),
)

REPLAY_RE = re.compile(
    r"base_position_rms=([0-9.eE+-]+).*saturation_fraction=([0-9.eE+-]+)")
FOOT_RE = re.compile(
    r"clearance=([0-9.eE+-]+) landing_error=([0-9.eE+-]+).*"
    r"touchdown_error=([0-9.eE+-]+) post_touchdown_slip=([0-9.eE+-]+) "
    r"accepted=([01])")
GATE_RE = re.compile(r"task_success=([01]) timing_success=([01])")


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("runner", type=Path)
    parser.add_argument("scene", type=Path)
    parser.add_argument("forward_plan", type=Path)
    parser.add_argument("reverse_plan", type=Path)
    parser.add_argument("output_csv", type=Path)
    parser.add_argument("--terrain", default="flat")
    parser.add_argument("--fixed-forward-plan", type=Path)
    parser.add_argument("--fixed-reverse-plan", type=Path)
    args = parser.parse_args()
    if ((args.fixed_forward_plan is None) !=
            (args.fixed_reverse_plan is None)):
        parser.error("both fixed-schedule plans must be provided together")
    return args


def metric_or_nan(match, group):
    return float(match.group(group)) if match else math.nan


def main():
    args = parse_args()
    paths = [args.runner, args.scene, args.forward_plan, args.reverse_plan]
    if args.fixed_forward_plan is not None:
        paths.extend((args.fixed_forward_plan, args.fixed_reverse_plan))
    for path in paths:
        if not path.is_file():
            raise SystemExit(f"missing input: {path}")

    args.output_csv.parent.mkdir(parents=True, exist_ok=True)
    log_directory = args.output_csv.with_suffix("")
    log_directory.mkdir(parents=True, exist_ok=True)
    rows = []
    plans = (("forward", args.forward_plan), ("reverse", args.reverse_plan))
    methods = []
    for ordering, plan in plans:
        methods.append(("cito", ordering, plan, "cito"))
        methods.append(("uniform_force", ordering, plan, "uniform"))
    if args.fixed_forward_plan is not None:
        fixed_plans = (("forward", args.fixed_forward_plan),
                       ("reverse", args.fixed_reverse_plan))
        for ordering, plan in fixed_plans:
            methods.append(("fixed_schedule", ordering, plan, "cito"))
    for method, ordering, plan, force_reference in methods:
        for condition, condition_args in CONDITIONS:
            command = [
                str(args.runner), str(args.scene),
                "--terrain", args.terrain,
                "--replay-plan", str(plan),
                "--force-reference", force_reference,
                *condition_args,
            ]
            completed = subprocess.run(
                command, text=True, stdout=subprocess.PIPE,
                stderr=subprocess.STDOUT, check=False)
            trial = f"{ordering}_{method}_{condition}"
            log_path = log_directory / f"{trial}.log"
            log_path.write_text(completed.stdout, encoding="utf-8")
            replay = REPLAY_RE.search(completed.stdout)
            gates = GATE_RE.search(completed.stdout)
            feet = FOOT_RE.findall(completed.stdout)
            clearances = [float(values[0]) for values in feet]
            landing_errors = [float(values[1]) for values in feet]
            touchdown_errors = [float(values[2]) for values in feet]
            slips = [float(values[3]) for values in feet]
            rows.append({
                "method": method,
                "ordering": ordering,
                "force_reference": force_reference,
                "condition": condition,
                "accepted": int(completed.returncode == 0),
                "task_success": int(gates.group(1)) if gates else 0,
                "timing_success": int(gates.group(2)) if gates else 0,
                "return_code": completed.returncode,
                "base_position_rms": metric_or_nan(replay, 1),
                "saturation_fraction": metric_or_nan(replay, 2),
                "minimum_clearance": min(clearances, default=math.nan),
                "maximum_landing_error": max(landing_errors, default=math.nan),
                "maximum_touchdown_error": max(touchdown_errors,
                                                 default=math.nan),
                "maximum_post_touchdown_slip": max(slips,
                                                    default=math.nan),
                "log": str(log_path),
            })
            print(f"{trial}: {'PASS' if completed.returncode == 0 else 'FAIL'}")

    with args.output_csv.open("w", newline="", encoding="utf-8") as output:
        writer = csv.DictWriter(output, fieldnames=rows[0].keys())
        writer.writeheader()
        writer.writerows(rows)
    print(f"wrote {len(rows)} paired trials to {args.output_csv}")


if __name__ == "__main__":
    main()
