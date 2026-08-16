#!/usr/bin/env python3
"""Run deterministic seeded ContactIPM replanning courses in MuJoCo."""

import argparse
import csv
import math
import re
import statistics
import subprocess
import time
from pathlib import Path


SOLVE_RE = re.compile(
    r"replanning solve: segment=(\d+) seed_policy=([^ ]+) seed=([^ ]+) "
    r"status=(.*?) solver_status=(.*?) audit_passed=([01]) "
    r"schedule_passed=([01]) iterations=(\d+) "
    r"solve_ms=([0-9.eE+-]+) objective=([0-9.eE+-]+) "
    r".*?optimized_fingerprint=(\d+).*?"
    r"schedule_deformation=(\d+)")
CANDIDATE_RE = re.compile(
    r"replanning candidate: segment=(\d+) seed=([^ ]+) status=(.*?) "
    r"solver_status=(.*?) audit_passed=([01]) schedule_passed=([01]) "
    r"objective=([0-9.eE+-]+) iterations=(\d+) "
    r"solve_ms=([0-9.eE+-]+) fingerprint=(\d+) "
    r"events=\((\d+),(\d+),(\d+),(\d+)\) "
    r"schedule_deformation=(\d+)")
COURSE_RE = re.compile(
    r"replanned course gates: segments=(\d+) seed_policy=([^ ]+) "
    r"swing_events=(\d+) expected_swing_events=(\d+) "
    r"base_displacement=([0-9.eE+-]+) "
    r"minimum_base_displacement=([0-9.eE+-]+) "
    r"final_base=\(([0-9.eE+-]+), ([0-9.eE+-]+), ([0-9.eE+-]+)\) "
    r"task_successful_segments=(\d+) timing_successful_segments=(\d+) "
    r"timing_required=([01]) accepted=([01])")
REPLAY_RE = re.compile(
    r"ContactIPM replay: segment=(\d+) .*?"
    r"base_position_rms=([0-9.eE+-]+) "
    r"saturation_fraction=([0-9.eE+-]+) "
    r"early_contact_activations=(\d+) "
    r"late_touchdown_search_ticks=(\d+) "
    r"final_base=\(([0-9.eE+-]+), ([0-9.eE+-]+), "
    r"([0-9.eE+-]+)\)")
EVENT_RE = re.compile(
    r"event segment=(\d+) foot=(\d+) index=(\d+): .*?"
    r"clearance=([0-9.eE+-]+) "
    r"landing_error=([0-9.eE+-]+).*?"
    r"planned_touchdown=([0-9.eE+-]+) "
    r"measured_touchdown=([0-9.eE+-]+) "
    r"touchdown_error=([0-9.eE+-]+) "
    r"post_touchdown_slip=([0-9.eE+-]+) accepted=([01]) "
    r"task_accepted=([01]) timing_accepted=([01]) "
    r"signed_touchdown_error=([0-9.eE+-]+)")
TOUCHDOWN_TRACE_RE = re.compile(
    r"touchdown trace: segment=(\d+) tick=(\d+) foot=(\d+) event=(\d+) "
    r"time=([0-9.eE+-]+) relative_time=([0-9.eE+-]+) "
    r"plan_mask=(0x[0-9a-f]+) support_mask=(0x[0-9a-f]+) "
    r"measured_mask=(0x[0-9a-f]+) plan_contact=([01]) "
    r"support_contact=([01]) measured_contact=([01]) "
    r"desired_gap=([0-9.eE+-]+) measured_gap=([0-9.eE+-]+) "
    r"desired_vz=([0-9.eE+-]+) measured_vz=([0-9.eE+-]+) "
    r"normal_force=([0-9.eE+-]+) contact_blend=([0-9.eE+-]+) "
    r"foot_position_error=([0-9.eE+-]+) "
    r"wrench_residual=([0-9.eE+-]+)")
TOUCHDOWN_CONTROL_TRACE_RE = re.compile(
    r"touchdown control trace: segment=(\d+) tick=(\d+) foot=(\d+) "
    r"event=(\d+) time=([0-9.eE+-]+) relative_time=([0-9.eE+-]+) "
    r"plan_mask=(0x[0-9a-f]+) support_mask=(0x[0-9a-f]+) "
    r"measured_mask=(0x[0-9a-f]+) "
    r"foot_position_error=\(([^)]+)\) "
    r"foot_velocity_error=\(([^)]+)\) "
    r"joint_position=\(([^)]+)\) joint_velocity=\(([^)]+)\) "
    r"requested_torque=\(([^)]+)\) predicted_torque=\(([^)]+)\) "
    r"applied_torque=\(([^)]+)\) torque_headroom=\(([^)]+)\) "
    r"saturated_mask=(0x[0-9a-f]+) swing_force=\(([^)]+)\) "
    r"jacobian_condition=([0-9.eE+-]+) "
    r"base_position_error=\(([^)]+)\) "
    r"base_velocity_error=\(([^)]+)\) "
    r"orientation_error_world=\(([^)]+)\) "
    r"angular_velocity_error_body=\(([^)]+)\) "
    r"wrench_residual=\(([^)]+)\)")
GATE_RE = re.compile(
    r"ContactIPM replay gates: segment=(\d+) moving_feet=(\d+) "
    r"swing_events=(\d+) task_success=([01]) timing_success=([01])")
STABILIZATION_RE = re.compile(
    r"replanning stabilization: duration=([0-9.eE+-]+) contacts=(\d+) "
    r"position_error=([0-9.eE+-]+) "
    r"orientation_error=([0-9.eE+-]+) "
    r"linear_speed=([0-9.eE+-]+) angular_speed=([0-9.eE+-]+) "
    r"max_foot_slip=([0-9.eE+-]+).*? accepted=([01])")


def parse_args(argv=None):
    parser = argparse.ArgumentParser()
    parser.add_argument("runner", type=Path)
    parser.add_argument("scene", type=Path)
    parser.add_argument("output_csv", type=Path)
    parser.add_argument("--terrain", default="random_smooth")
    parser.add_argument("--terrain-amplitude", type=float, default=0.02)
    parser.add_argument("--segments", type=int, default=2)
    parser.add_argument("--seed-policy", default="multistart")
    parser.add_argument("--first-seed", type=int, default=0)
    parser.add_argument("--seed-count", type=int, default=10)
    parser.add_argument("--trial-timeout", type=float, default=300.0)
    parser.add_argument("--require-timing", action="store_true")
    parser.add_argument("--disable-early-contact-feedback",
                        action="store_true")
    parser.add_argument("--fixed-stabilization-handoff",
                        action="store_true")
    parser.add_argument("--enable-late-touchdown-search",
                        action="store_true")
    parser.add_argument("--stabilization-duration", type=float, default=2.0)
    parser.add_argument("--maximum-stabilization-duration", type=float,
                        default=3.0)
    parser.add_argument("--trace-execution", action="store_true")
    parser.add_argument("--snapshot-directory", type=Path)
    args = parser.parse_args(argv)
    if args.terrain_amplitude <= 0.0:
        parser.error("terrain amplitude must be positive")
    if not 1 <= args.segments <= 32:
        parser.error("segments must be between 1 and 32")
    if args.first_seed < 0:
        parser.error("first seed must be nonnegative")
    if args.seed_count < 1:
        parser.error("seed count must be positive")
    if args.trial_timeout <= 0.0:
        parser.error("trial timeout must be positive")
    if args.stabilization_duration <= 0.0:
        parser.error("stabilization duration must be positive")
    if args.maximum_stabilization_duration < args.stabilization_duration:
        parser.error("maximum stabilization duration must not be shorter "
                     "than its minimum")
    return args


def float_or_nan(match, group):
    return float(match.group(group)) if match else math.nan


def int_or_zero(match, group):
    return int(match.group(group)) if match else 0


def maximum_or_nan(values):
    return max(values) if values else math.nan


def minimum_or_nan(values):
    return min(values) if values else math.nan


def observed_replay_errors(values):
    return [value for value in values
            if math.isfinite(value) and value < 1e100]


def finite_maximum(rows, field):
    values = [row[field] for row in rows if math.isfinite(row[field])]
    return maximum_or_nan(values)


def classify_failure(return_code, timed_out, accepted, requested_segments,
                     solves, course, gates, stabilizations, divergence_count):
    if accepted:
        return "success"
    if timed_out:
        return "timeout"
    for solve in solves:
        if solve[3] == "Success":
            continue
        if solve[4] == "Success" and solve[6] == "0":
            return "schedule_audit"
        if solve[4] == "Success":
            return "planner_audit"
        return "planner"
    if any(stabilization[7] == "0" for stabilization in stabilizations):
        return "stabilization"
    if any(gate[3] == "0" for gate in gates):
        return "execution"
    if any(gate[3] == "1" and gate[4] == "0" for gate in gates):
        return "timing"
    if course:
        if int(course.group(3)) != int(course.group(4)):
            return "execution"
        if float(course.group(5)) < float(course.group(6)):
            return "route_progress"
        return "course_gate"
    if not solves:
        return "setup"
    if len(solves) < requested_segments:
        return "execution" if divergence_count else "incomplete"
    return "unknown" if return_code == 0 else "execution"


def write_rows(path, rows):
    temporary_path = path.with_suffix(f"{path.suffix}.tmp")
    with temporary_path.open("w", newline="", encoding="utf-8") as output:
        writer = csv.DictWriter(output, fieldnames=rows[0].keys())
        writer.writeheader()
        writer.writerows(rows)
    temporary_path.replace(path)


def write_optional_rows(path, rows):
    if rows:
        write_rows(path, rows)


def write_summary(args, rows):
    planning_times = [row["planning_ms_total"] for row in rows]
    summary = {
        "terrain": args.terrain,
        "terrain_amplitude": args.terrain_amplitude,
        "segments": args.segments,
        "seed_policy": args.seed_policy,
        "trials": len(rows),
        "planner_success": sum(
            row["planner_successful_segments"] == args.segments
            for row in rows),
        "execution_success": sum(
            row["task_successful_segments"] == args.segments
            for row in rows),
        "strict_timing_success": sum(
            row["task_successful_segments"] == args.segments and
            row["timing_successful_segments"] == args.segments
            for row in rows),
        "accepted": sum(row["accepted"] for row in rows),
        "median_planning_ms_total": statistics.median(planning_times),
        "median_trial_wall_time_seconds": statistics.median(
            row["trial_wall_time_seconds"] for row in rows),
        "maximum_touchdown_error": finite_maximum(
            rows, "maximum_touchdown_error"),
        "maximum_landing_error": finite_maximum(
            rows, "maximum_landing_error"),
        "maximum_post_touchdown_slip": finite_maximum(
            rows, "maximum_post_touchdown_slip"),
    }
    summary_path = args.output_csv.with_name(
        f"{args.output_csv.stem}_summary.csv")
    with summary_path.open("w", newline="", encoding="utf-8") as output:
        writer = csv.DictWriter(output, fieldnames=summary.keys())
        writer.writeheader()
        writer.writerow(summary)
    print(
        "summary: "
        f"planner={summary['planner_success']}/{summary['trials']} "
        f"execution={summary['execution_success']}/{summary['trials']} "
        f"strict_timing={summary['strict_timing_success']}/"
        f"{summary['trials']} "
        f"accepted={summary['accepted']}/{summary['trials']} "
        f"median_planning_ms={summary['median_planning_ms_total']:.3f} "
        f"max_touchdown_error={summary['maximum_touchdown_error']:.6f} "
        f"max_slip={summary['maximum_post_touchdown_slip']:.6f}")
    print(f"wrote summary to {summary_path}")


def main():
    args = parse_args()
    for path in (args.runner, args.scene):
        if not path.is_file():
            raise SystemExit(f"missing input: {path}")

    args.output_csv.parent.mkdir(parents=True, exist_ok=True)
    log_directory = args.output_csv.with_suffix("")
    log_directory.mkdir(parents=True, exist_ok=True)
    rows = []
    event_rows = []
    touchdown_trace_rows = []
    touchdown_control_trace_rows = []
    for terrain_seed in range(args.first_seed,
                              args.first_seed + args.seed_count):
        command = [
            str(args.runner), str(args.scene),
            "--terrain", args.terrain,
            "--terrain-seed", str(terrain_seed),
            "--terrain-amplitude", str(args.terrain_amplitude),
            "--replan-count", str(args.segments),
            "--seed-policy", args.seed_policy,
        ]
        if not args.require_timing:
            command.append("--allow-timing-miss")
        if args.disable_early_contact_feedback:
            command.append("--disable-early-contact-feedback")
        if args.fixed_stabilization_handoff:
            command.append("--fixed-stabilization-handoff")
        if args.enable_late_touchdown_search:
            command.append("--enable-late-touchdown-search")
        command.extend(["--stabilization-duration",
                        str(args.stabilization_duration)])
        command.extend(["--maximum-stabilization-duration",
                        str(args.maximum_stabilization_duration)])
        if args.trace_execution:
            command.append("--trace-execution")
        if args.snapshot_directory:
            args.snapshot_directory.mkdir(parents=True, exist_ok=True)
            command.extend([
                "--write-boundary-snapshots",
                str(args.snapshot_directory / f"terrain_seed_{terrain_seed}"),
            ])
        timed_out = False
        trial_start = time.monotonic()
        try:
            completed = subprocess.run(
                command, text=True, stdout=subprocess.PIPE,
                stderr=subprocess.STDOUT, check=False,
                timeout=args.trial_timeout)
            output_text = completed.stdout
            return_code = completed.returncode
        except subprocess.TimeoutExpired as error:
            timed_out = True
            output_text = error.stdout or ""
            if isinstance(output_text, bytes):
                output_text = output_text.decode("utf-8", errors="replace")
            return_code = 124
        trial_wall_time = time.monotonic() - trial_start
        log_path = log_directory / f"terrain_seed_{terrain_seed}.log"
        log_path.write_text(output_text, encoding="utf-8")

        solves = SOLVE_RE.findall(output_text)
        candidates = CANDIDATE_RE.findall(output_text)
        course = COURSE_RE.search(output_text)
        replays = REPLAY_RE.findall(output_text)
        events = EVENT_RE.findall(output_text)
        touchdown_traces = TOUCHDOWN_TRACE_RE.findall(output_text)
        touchdown_control_traces = TOUCHDOWN_CONTROL_TRACE_RE.findall(
            output_text)
        gates = GATE_RE.findall(output_text)
        stabilizations = STABILIZATION_RE.findall(output_text)
        divergence_count = output_text.count("execution divergence:")
        selected_times = [float(solve[8]) for solve in solves]
        candidate_times = [float(candidate[8]) for candidate in candidates]
        fingerprints = {solve[10] for solve in solves}
        successful_candidates = sum(
            candidate[2] == "Success" for candidate in candidates)
        course_accepted = int(course.group(13)) if course else 0
        accepted = int(return_code == 0 and course_accepted == 1)
        planner_successful_segments = sum(
            solve[3] == "Success" for solve in solves)
        task_successful_segments = sum(gate[3] == "1" for gate in gates)
        timing_successful_segments = sum(gate[4] == "1" for gate in gates)
        clearances = [float(event[3]) for event in events]
        landing_errors = [float(event[4]) for event in events]
        touchdown_errors = [float(event[7]) for event in events]
        observed_landing_errors = observed_replay_errors(landing_errors)
        observed_touchdown_errors = observed_replay_errors(touchdown_errors)
        slips = [float(event[8]) for event in events]
        for event in events:
            event_rows.append({
                "terrain_seed": terrain_seed,
                "segment": int(event[0]),
                "foot": int(event[1]),
                "event": int(event[2]),
                "clearance": float(event[3]),
                "landing_error": float(event[4]),
                "planned_touchdown": float(event[5]),
                "measured_touchdown": float(event[6]),
                "touchdown_error": float(event[7]),
                "signed_touchdown_error": float(event[12]),
                "post_touchdown_slip": float(event[8]),
                "accepted": int(event[9]),
                "task_accepted": int(event[10]),
                "timing_accepted": int(event[11]),
            })
        for trace in touchdown_traces:
            touchdown_trace_rows.append({
                "terrain_seed": terrain_seed,
                "segment": int(trace[0]),
                "tick": int(trace[1]),
                "foot": int(trace[2]),
                "event": int(trace[3]),
                "time": float(trace[4]),
                "relative_time": float(trace[5]),
                "plan_mask": trace[6],
                "support_mask": trace[7],
                "measured_mask": trace[8],
                "plan_contact": int(trace[9]),
                "support_contact": int(trace[10]),
                "measured_contact": int(trace[11]),
                "desired_gap": float(trace[12]),
                "measured_gap": float(trace[13]),
                "desired_vz": float(trace[14]),
                "measured_vz": float(trace[15]),
                "normal_force": float(trace[16]),
                "contact_blend": float(trace[17]),
                "foot_position_error": float(trace[18]),
                "wrench_residual": float(trace[19]),
            })
        for trace in touchdown_control_traces:
            touchdown_control_trace_rows.append({
                "terrain_seed": terrain_seed,
                "segment": int(trace[0]),
                "tick": int(trace[1]),
                "foot": int(trace[2]),
                "event": int(trace[3]),
                "time": float(trace[4]),
                "relative_time": float(trace[5]),
                "plan_mask": trace[6],
                "support_mask": trace[7],
                "measured_mask": trace[8],
                "foot_position_error": trace[9],
                "foot_velocity_error": trace[10],
                "joint_position": trace[11],
                "joint_velocity": trace[12],
                "requested_torque": trace[13],
                "predicted_torque": trace[14],
                "applied_torque": trace[15],
                "torque_headroom": trace[16],
                "saturated_mask": trace[17],
                "swing_force": trace[18],
                "jacobian_condition": float(trace[19]),
                "base_position_error": trace[20],
                "base_velocity_error": trace[21],
                "orientation_error_world": trace[22],
                "angular_velocity_error_body": trace[23],
                "wrench_residual": trace[24],
            })
        planning_ms_total = (sum(candidate_times) if candidates
                             else sum(selected_times))
        failure_category = classify_failure(
            return_code, timed_out, accepted, args.segments, solves, course,
            gates, stabilizations, divergence_count)
        rows.append({
            "terrain": args.terrain,
            "terrain_seed": terrain_seed,
            "terrain_amplitude": args.terrain_amplitude,
            "requested_segments": args.segments,
            "seed_policy": args.seed_policy,
            "timing_required": int(args.require_timing),
            "early_contact_feedback": int(
                not args.disable_early_contact_feedback),
            "confirmed_stabilization": int(
                not args.fixed_stabilization_handoff),
            "late_touchdown_search": int(
                args.enable_late_touchdown_search),
            "stabilization_duration": args.stabilization_duration,
            "maximum_stabilization_duration": (
                args.maximum_stabilization_duration),
            "accepted": accepted,
            "failure_category": failure_category,
            "return_code": return_code,
            "timed_out": int(timed_out),
            "trial_timeout": args.trial_timeout,
            "trial_wall_time_seconds": trial_wall_time,
            "completed_segments": int_or_zero(course, 1),
            "planner_solve_count": len(solves),
            "planner_successful_segments": planner_successful_segments,
            "execution_segment_count": len(gates),
            "task_successful_segments": task_successful_segments,
            "timing_successful_segments": timing_successful_segments,
            "stabilization_count": len(stabilizations),
            "successful_stabilizations": sum(
                stabilization[7] == "1" for stabilization in stabilizations),
            "maximum_observed_stabilization_duration": maximum_or_nan(
                [float(stabilization[0])
                 for stabilization in stabilizations]),
            "divergence_count": divergence_count,
            "swing_events": int_or_zero(course, 3),
            "expected_swing_events": int_or_zero(course, 4),
            "base_displacement": float_or_nan(course, 5),
            "minimum_base_displacement": float_or_nan(course, 6),
            "final_base_x": float_or_nan(course, 7),
            "final_base_y": float_or_nan(course, 8),
            "final_base_z": float_or_nan(course, 9),
            "selected_solve_count": len(solves),
            "selected_solve_ms_total": sum(selected_times),
            "selected_solve_ms_max": max(selected_times, default=math.nan),
            "planning_ms_total": planning_ms_total,
            "selected_seeds": ";".join(solve[2] for solve in solves),
            "selected_statuses": ";".join(solve[3] for solve in solves),
            "selected_solver_statuses": ";".join(
                solve[4] for solve in solves),
            "selected_schedule_fingerprints": ";".join(
                solve[10] for solve in solves),
            "selected_schedule_deformations": ";".join(
                solve[11] for solve in solves),
            "candidate_solve_count": len(candidates),
            "successful_candidate_count": successful_candidates,
            "candidate_solve_ms_total": sum(candidate_times),
            "candidate_solver_failures": sum(
                candidate[3] != "Success" for candidate in candidates),
            "candidate_audit_rejections": sum(
                candidate[3] == "Success" and candidate[4] == "0"
                for candidate in candidates),
            "candidate_schedule_rejections": sum(
                candidate[3] == "Success" and candidate[5] == "0"
                for candidate in candidates),
            "candidate_statuses": ";".join(
                f"{candidate[0]}:{candidate[1]}:{candidate[2]}"
                for candidate in candidates),
            "candidate_event_counts": ";".join(
                f"{candidate[0]}:{candidate[1]}:"
                f"{candidate[10]}/{candidate[11]}/"
                f"{candidate[12]}/{candidate[13]}"
                for candidate in candidates),
            "candidate_schedule_deformations": ";".join(
                f"{candidate[0]}:{candidate[1]}:{candidate[14]}"
                for candidate in candidates),
            "candidates_with_extra_events": sum(
                candidate[3] == "Success" and
                any(int(events) != 1 for events in candidate[10:14])
                for candidate in candidates),
            "distinct_selected_schedules": len(fingerprints),
            "missing_touchdowns": (
                len(touchdown_errors) - len(observed_touchdown_errors)),
            "minimum_clearance": minimum_or_nan(clearances),
            "maximum_landing_error": maximum_or_nan(
                observed_landing_errors),
            "maximum_touchdown_error": maximum_or_nan(
                observed_touchdown_errors),
            "maximum_post_touchdown_slip": maximum_or_nan(slips),
            "maximum_base_position_rms": maximum_or_nan(
                [float(replay[1]) for replay in replays]),
            "maximum_saturation_fraction": maximum_or_nan(
                [float(replay[2]) for replay in replays]),
            "early_contact_activations": sum(
                int(replay[3]) for replay in replays),
            "late_touchdown_search_ticks": sum(
                int(replay[4]) for replay in replays),
            "maximum_stabilization_orientation_error": maximum_or_nan(
                [float(stabilization[3])
                 for stabilization in stabilizations]),
            "maximum_stabilization_foot_slip": maximum_or_nan(
                [float(stabilization[6])
                 for stabilization in stabilizations]),
            "log": str(log_path),
        })
        write_rows(args.output_csv, rows)
        write_optional_rows(
            args.output_csv.with_name(f"{args.output_csv.stem}_events.csv"),
            event_rows)
        write_optional_rows(
            args.output_csv.with_name(
                f"{args.output_csv.stem}_touchdown_trace.csv"),
            touchdown_trace_rows)
        write_optional_rows(
            args.output_csv.with_name(
                f"{args.output_csv.stem}_touchdown_control_trace.csv"),
            touchdown_control_trace_rows)
        print(
            f"terrain_seed_{terrain_seed}: "
            f"{'PASS' if accepted else 'FAIL'} "
            f"category={failure_category} "
            f"planner={planner_successful_segments}/{args.segments} "
            f"task={task_successful_segments}/{args.segments}")

    accepted_trials = sum(row["accepted"] for row in rows)
    print(f"wrote {len(rows)} trials to {args.output_csv}; "
          f"accepted={accepted_trials}/{len(rows)}")
    write_summary(args, rows)


if __name__ == "__main__":
    main()
