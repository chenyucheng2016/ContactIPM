#!/usr/bin/env python3
from __future__ import annotations

import importlib.util
import json
import math
import sys
import tempfile
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
CAMPAIGN = (
    ROOT / "examples" / "quadruped_cito" / "mujoco" /
    "run_sustained_realtime_campaign.py"
)


def load_campaign():
    spec = importlib.util.spec_from_file_location("sustained_campaign", CAMPAIGN)
    assert spec is not None and spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = module
    spec.loader.exec_module(module)
    return module


def qualified_summary(**overrides: str) -> dict[str, str]:
    summary = {
        "scheduling": "background_worker",
        "wall_concurrent": "1",
        "planner_updates_enabled": "1",
        "sustained_mode": "1",
        "planner_horizon_s": "2.5",
        "planner_knots": "50",
        "planner_dt_s": "0.05",
        "planner_rate_hz": "5",
        "wbc_rate_hz": "500",
        "mujoco_version": "3.3.5",
        "logical_wbc_rate_hz": "500",
        "duration_s": "20",
        "wall_duration": "20",
        "duration_gate": "1",
        "wbc_ticks": "10000",
        "actual_wall_tick_rate_hz": "500",
        "wall_rate_gate": "1",
        "replan_opportunities": "100",
        "warm_attempts": "100",
        "usable_updates": "100",
        "accepted_publications": "100",
        "usable_fraction": "1",
        "accepted_publication_fraction": "1",
        "publication_rate_gate": "1",
        "planner_latency_scope": "request_to_wbc_handoff",
        "planner_p99_ms": "100",
        "max_consecutive_fallbacks": "0",
        "fallbacks": "0",
        "deadline_misses": "0",
        "invalid_publications": "0",
        "stale_publications": "0",
        "exact_hessian_analytic_calls": "0",
        "exact_hessian_fd_calls": "0",
        "expired_plan_ticks": "0",
        "zero_hessian_gate": "1",
        "contact_tasks_commanded": "12",
        "contact_tasks_completed": "12",
        "completed_gait_cycles": "3",
        "route_progress_m": "0.20",
        "swing_events": "12",
        "sustained_contact_execution": "1",
        "execution_evidence_scope": "sustained_multi_contact",
        "missing_touchdown": "0",
        "minimum_clearance": "0.03",
        "maximum_landing_error": "0.01",
        "maximum_slip": "0.01",
        "saturated_joint_ticks": "0",
        "contact_execution_quality_gate": "1",
        "execution_gates_passed": "1",
        "task_metadata_rejections": "0",
        "accepted_task_transitions": "11",
        "fresh_task_transition_witnesses": "11",
        "pending_contact_tasks": "0",
        "right_censored_contact_tasks": "0",
        "unassigned_contact_events": "0",
        "task_ledger_mismatches": "0",
        "task_ledger_integrity_gate": "1",
        "environment": "linux",
        "environment_limited": "0",
        "separate_logical_cpus": "1",
        "wbc_affinity_applied": "1",
        "planner_scheduling_reported": "1",
        "planner_affinity_applied": "1",
        "logical_thread_isolation_applied": "1",
        "wbc_priority_applied": "1",
        "planner_priority_applied": "1",
        "physical_core_isolation_verified": "1",
        "realtime_priorities_applied": "1",
        "paper_grade_scheduling_gate": "1",
        "paper_grade_verdict": "paper_grade_pass",
        "accepted": "1",
    }
    for stem in (
        "wbc_compute", "complete_tick_work", "scheduled_tick_response",
    ):
        summary.update({
            f"{stem}_samples": "10000",
            f"{stem}_within_2ms": "10000",
            f"{stem}_within_2ms_fraction": "1",
            f"{stem}_p99_ms": "0.1",
            f"{stem}_gate": "1",
        })
    summary.update(overrides)
    return summary


def fixture_log(module, summary: dict[str, str] | None = None) -> str:
    values = qualified_summary() if summary is None else summary
    lines = [
        module.SUMMARY_PREFIX + " " +
        " ".join(f"{key}={value}" for key, value in values.items())
    ]
    for index in range(12):
        lines.append(
            module.TASK_PREFIX + " " +
            f"task_id={100 + index} task_sequence={index} "
            f"moving_foot={index % 4} target_x={0.1 * (index + 1):.3f} "
            f"expected_liftoff={float(index):.3f} "
            f"expected_touchdown={index + 0.5:.3f} "
            f"event_bound=1 event_index={index} "
            f"planned_liftoff={float(index):.3f} "
            f"measured_liftoff={index + 0.01:.3f} "
            f"planned_touchdown={index + 0.5:.3f} "
            f"measured_touchdown={index + 0.51:.3f} "
            "clearance=0.03 landing_error=0.01 slip=0.01 due=1 "
            "completed=1 right_censored=0 missing=0 schedule_match=1"
        )
    for phase in module.REQUIRED_PHASES:
        lines.append(
            module.PHASE_PREFIX + " " +
            f"phase={phase} samples=10000 wall_p50_ms=0.01 "
            "wall_p90_ms=0.02 wall_p99_ms=0.03 wall_max_ms=0.04 "
            "thread_cpu_samples=10000 thread_cpu_p50_ms=0.01 "
            "thread_cpu_p90_ms=0.02 thread_cpu_p99_ms=0.03 "
            "thread_cpu_max_ms=0.04"
        )
    for scope in module.REQUIRED_MISS_SCOPES:
        lines.append(
            module.MISS_PREFIX + " " +
            f"scope={scope} misses=0 mujoco_physics=0 "
            "wbc_or_control_work=0 scheduling_or_blocking=0 "
            "thread_cpu_time_unavailable=0"
        )
    return "\n".join(lines) + "\n"


def main() -> int:
    module = load_campaign()

    evidence = module.parse_execution_log(fixture_log(module))
    execution, qualification = module.evaluate_execution_evidence(evidence)
    if execution or qualification:
        raise AssertionError(
            f"qualified fixture was rejected: {execution + qualification}"
        )

    duplicate_summary = fixture_log(module) + fixture_log(module).splitlines()[0] + "\n"
    duplicate_execution, _ = module.evaluate_execution_evidence(
        module.parse_execution_log(duplicate_summary)
    )
    if not any(failure.category == "evidence" for failure in duplicate_execution):
        raise AssertionError("duplicate summary record was accepted")

    duplicate_key = fixture_log(module).replace(
        "task_id=100 task_sequence=0",
        "task_id=100 task_id=101 task_sequence=0",
    )
    duplicate_key_execution, _ = module.evaluate_execution_evidence(
        module.parse_execution_log(duplicate_key)
    )
    if not any(
        failure.category == "evidence" for failure in duplicate_key_execution
    ):
        raise AssertionError("duplicate key was accepted")

    duplicate_task = fixture_log(module).replace(
        "task_id=101 task_sequence=1", "task_id=100 task_sequence=1",
    )
    duplicate_task_execution, _ = module.evaluate_execution_evidence(
        module.parse_execution_log(duplicate_task)
    )
    if "task_1_id_unique" not in {
        failure.gate for failure in duplicate_task_execution
    }:
        raise AssertionError("duplicate task ledger id was accepted")

    broken_gait = fixture_log(module).replace(
        "task_id=104 task_sequence=4 moving_foot=0",
        "task_id=104 task_sequence=4 moving_foot=1",
    )
    broken_gait_execution, _ = module.evaluate_execution_evidence(
        module.parse_execution_log(broken_gait)
    )
    if "task_4_gait_order" not in {
        failure.gate for failure in broken_gait_execution
    }:
        raise AssertionError("non-repeating gait order was accepted")

    duplicate_event = fixture_log(module).replace(
        "task_id=104 task_sequence=4 moving_foot=0 target_x=0.500 "
        "expected_liftoff=4.000 expected_touchdown=4.500 event_bound=1 "
        "event_index=4",
        "task_id=104 task_sequence=4 moving_foot=0 target_x=0.500 "
        "expected_liftoff=4.000 expected_touchdown=4.500 event_bound=1 "
        "event_index=0",
    )
    duplicate_event_execution, _ = module.evaluate_execution_evidence(
        module.parse_execution_log(duplicate_event)
    )
    if "task_4_event_index_unique" not in {
        failure.gate for failure in duplicate_event_execution
    }:
        raise AssertionError("duplicate per-foot event binding was accepted")

    legacy = qualified_summary(
        execution_evidence_scope="single_contact_task",
        contact_tasks_commanded="1",
        contact_tasks_completed="1",
        swing_events="1",
        sustained_contact_execution="0",
    )
    del legacy["route_progress_m"]
    del legacy["completed_gait_cycles"]
    legacy_evidence = module.parse_execution_log(fixture_log(module, legacy))
    legacy_execution, _ = module.evaluate_execution_evidence(legacy_evidence)
    legacy_gates = {failure.gate for failure in legacy_execution}
    for gate in (
        "route_progress_m", "completed_gait_cycles",
        "execution_evidence_scope", "contact_tasks_commanded",
    ):
        if gate not in legacy_gates:
            raise AssertionError(f"legacy evidence did not fail {gate}")

    short_route = qualified_summary(route_progress_m="0.179")
    short_execution, _ = module.evaluate_execution_evidence(
        module.parse_execution_log(fixture_log(module, short_route))
    )
    if "route_progress_m" not in {failure.gate for failure in short_execution}:
        raise AssertionError("insufficient route progress was accepted")

    wsl = qualified_summary(
        environment="wsl", environment_limited="1",
        physical_core_isolation_verified="0", realtime_priorities_applied="0",
        paper_grade_scheduling_gate="0",
        paper_grade_verdict="environment_limited_wsl", accepted="0",
    )
    wsl_execution, wsl_qualification = module.evaluate_execution_evidence(
        module.parse_execution_log(fixture_log(module, wsl))
    )
    if wsl_execution:
        raise AssertionError("WSL environment incorrectly invalidated technical evidence")
    if not wsl_qualification:
        raise AssertionError("WSL environment was accepted as paper evidence")

    incomplete_log = fixture_log(module)
    incomplete_log = "\n".join(
        line for line in incomplete_log.splitlines()
        if "phase=mj_step2" not in line and
        "scope=scheduled_tick_response" not in line
    )
    incomplete_execution, _ = module.evaluate_execution_evidence(
        module.parse_execution_log(incomplete_log)
    )
    incomplete_gates = {failure.gate for failure in incomplete_execution}
    if "phase_mj_step2" not in incomplete_gates:
        raise AssertionError("missing phase attribution was accepted")
    if "miss_scope_scheduled_tick_response" not in incomplete_gates:
        raise AssertionError("missing deadline attribution was accepted")

    mismatched_misses = fixture_log(module).replace(
        "scope=complete_tick_work misses=0 mujoco_physics=0",
        "scope=complete_tick_work misses=2 mujoco_physics=1",
    )
    mismatch_execution, _ = module.evaluate_execution_evidence(
        module.parse_execution_log(mismatched_misses)
    )
    if "miss_sum_complete_tick_work" not in {
        failure.gate for failure in mismatch_execution
    }:
        raise AssertionError("inconsistent miss attribution was accepted")

    cases = module.campaign_cases("development")
    core = [case for case in cases if not case.stress]
    stress = [case for case in cases if case.stress]
    if len(core) != 10 or len(stress) != 4:
        raise AssertionError("unexpected core/stress matrix size")
    if {case.level for case in core} != {1, 2, 3, 4, 5, 6}:
        raise AssertionError("core gate ladder is incomplete")
    if any(not case.stress for case in stress):
        raise AssertionError("stress case leaked into core scope")
    random_case = next(case for case in core if case.name == "random_04cm")
    if random_case.run_arguments(3, 1000)[-1] != "1003":
        raise AssertionError("random run seed was not deterministic")
    if module.required_runs_for_case(random_case, "development", 10) != 20:
        raise AssertionError("qualification did not require 20 random seeds")
    if module.required_runs_for_case(core[0], "development", 10) != 11:
        raise AssertionError("qualification cannot reach 1000 deterministic updates")

    low, high = module.wilson_interval(10, 10)
    if not (0.72 < low < 0.73 and math.isclose(high, 1.0)):
        raise AssertionError("Wilson interval regression")

    summary_rows = module.condition_summaries([
        {
            "case": "flat", "terrain_class": "flat", "scope": "core",
            "execution_pass": 1, "qualification_pass": 1,
        }
        for _ in range(10)
    ] + [
        {
            "case": "stress", "terrain_class": "stress", "scope": "stress",
            "execution_pass": 0, "qualification_pass": 0,
        }
        for _ in range(10)
    ], 10)
    by_case = {row["case"]: row for row in summary_rows}
    if by_case["flat"]["qualification_verdict"] != "PASS":
        raise AssertionError("qualified core condition was rejected")
    if by_case["stress"]["execution_verdict"] != "FAIL":
        raise AssertionError("failed stress condition was accepted")

    initial = {
        "runner_sha256": "runner", "runner_size_bytes": 1,
        "scene_sha256": "scene", "scene_size_bytes": 2,
        "build_metadata_sha256": "metadata", "git_revision": "revision",
        "git_tree": "tree",
        "mujoco_library_sha256": "mujoco",
        "worktree_status_sha256": "status", "worktree_diff_sha256": "diff",
        "platform": "platform", "machine": "machine",
        "processor": "processor", "cpu_count": 4,
    }
    if not module.provenance_matches(initial, dict(initial)):
        raise AssertionError("stable provenance was rejected")
    if module.provenance_matches(initial, dict(initial, runner_sha256="changed")):
        raise AssertionError("changed runner provenance was accepted")
    if module.provenance_matches(initial, dict(initial, build_metadata_sha256=None)):
        raise AssertionError("missing build provenance was accepted")

    metadata, errors = module.load_build_metadata(None)
    if metadata is not None or not errors:
        raise AssertionError("missing build metadata was not development-only")
    with tempfile.TemporaryDirectory() as directory:
        path = Path(directory) / "build.json"
        path.write_text(json.dumps({
            "schema_version": 2,
            "source_git_revision": "1" * 40,
            "source_git_tree": "2" * 40,
            "source_worktree_clean": True,
            "source_worktree_status_sha256": "3" * 64,
            "runner_sha256": "4" * 64,
            "compiler_id": "GNU",
            "compiler_version": "15.2",
            "build_type": "Release",
            "cxx_flags": "-O3",
            "fast_math": True,
            "ndebug": True,
            "mujoco_version": "3.3.5",
            "mujoco_library_path": "libmujoco.so",
            "mujoco_library_sha256": "5" * 64,
        }), encoding="utf-8")
        metadata, errors = module.load_build_metadata(path)
        if metadata is None or errors:
            raise AssertionError(f"valid build metadata was rejected: {errors}")

    with tempfile.TemporaryDirectory() as directory:
        directory_path = Path(directory)
        fake_scene = directory_path / "fixture_runner.py"
        fake_scene.write_text(
            "print(" + repr(fixture_log(module)) + ")\n", encoding="utf-8",
        )
        output = directory_path / "campaign"
        return_code = module.main([
            sys.executable, str(fake_scene), str(output),
            "--profile", "smoke", "--duration", "20",
        ])
        if return_code == 0:
            raise AssertionError("smoke fixture was mislabeled as paper evidence")
        manifest = json.loads(
            (output / "campaign.json").read_text(encoding="utf-8")
        )
        if manifest["core_execution_verdict"] != "PASS":
            raise AssertionError("fixture campaign did not preserve execution PASS")
        if manifest["architecture_qualification_verdict"] != \
                "development_only_protocol":
            raise AssertionError("smoke campaign escaped the architecture gate")
        if manifest["paper_grade_verdict"] != \
                "not_ready_architecture_not_qualified":
            raise AssertionError("smoke campaign escaped the paper-readiness gate")
        if manifest["stress_affects_paper_verdict"]:
            raise AssertionError("stress results affect the headline verdict")
        run = manifest["runs"][0]
        command = json.loads(run["command"])
        if "--realtime-sustained" not in command:
            raise AssertionError("campaign did not request sustained execution")
        for filename in (
            "runs.csv", "failures.csv", "task_ledger.csv", "events.csv",
            "phase_timing.csv",
            "miss_attribution.csv", "condition_summary.csv",
            "artifact_hooks.json",
        ):
            if not (output / filename).is_file():
                raise AssertionError(f"campaign output is missing {filename}")

    source = CAMPAIGN.read_text(encoding="utf-8")
    for required in (
        "stress_affects_paper_verdict\": False",
        "development_only_invalid_build_attestation",
        "architecture_qualification_pass",
        "previous execution gate level failed",
        "No artifact is claimed until this hook succeeds.",
    ):
        if required not in source:
            raise AssertionError(f"missing fail-closed campaign marker: {required}")
    print("sustained real-time campaign protocol tests passed")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
