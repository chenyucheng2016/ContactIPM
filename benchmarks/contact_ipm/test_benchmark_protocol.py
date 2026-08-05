#!/usr/bin/env python3
"""Regression tests for the paired benchmark protocol and summary statistics."""

from __future__ import annotations

import random
import tempfile
import unittest
from pathlib import Path
from unittest import mock

from run_benchmarks import (
    case_environment,
    git_tracked_dirty,
    load_case_suite,
    schedule_trials,
)
from run_contact_ablations import summarize_artifact
from run_pareto import mark_efficient
from run_scaling import summarize_point
from summarize_benchmarks import percentile, render_markdown, summarize
from trajectory_audit import audit_push_t


def result(solver: str, repetition: int, elapsed: float, successful: bool) -> dict:
    success_count = int(successful)
    return {
        "solver": solver,
        "problem": "push_box",
        "repetition": repetition,
        "returncode": 0,
        "wall_time_seconds": elapsed,
        "metrics": {
            "trajectory_audit": {
                "attempted_instances": 1,
                "feasible_instances": 1,
                "task_successful_instances": 1,
                "successful_instances": success_count,
                "worst_equality": 1e-8,
                "worst_side_violation": 0.0,
                "worst_physical_mpcc": 1e-7,
                "worst_position_error": 0.01,
                "worst_angular_error": 0.02,
                "worst_velocity_error": 0.0,
                "instances": [
                    {
                        "success": successful,
                        "objective": 3.0,
                        "objective_components": {
                            "terminal_tracking": 1.0,
                            "force_effort": 2.0,
                        },
                        "actuation": {"peak_contact_force_norm": 4.0},
                    }
                ],
            }
        },
    }


class ProtocolTests(unittest.TestCase):
    def test_schedule_keeps_randomized_pairs_adjacent(self) -> None:
        commands = [
            {"solver": solver, "problem": problem}
            for problem in ("push_box", "transport")
            for solver in ("ContactIPM", "CRISP")
        ]
        trials = schedule_trials(commands, repetitions=3, seed=2027)
        self.assertEqual(len(trials), 12)
        self.assertEqual(trials, schedule_trials(commands, 3, 2027))
        for offset in range(0, len(trials), 2):
            pair = trials[offset : offset + 2]
            self.assertEqual(pair[0]["block_index"], pair[1]["block_index"])
            self.assertEqual(pair[0]["problem"], pair[1]["problem"])
            self.assertEqual(pair[0]["repetition"], pair[1]["repetition"])
            self.assertEqual(
                {trial["entry"]["solver"] for trial in pair},
                {"ContactIPM", "CRISP"},
            )
            self.assertEqual([trial["order_in_block"] for trial in pair], [0, 1])

    def test_validation_suite_expands_cartesian_axes(self) -> None:
        path = Path(__file__).with_name("validation_cases.json")
        suites = load_case_suite(
            path, ["cartpole_soft_walls", "push_box", "transport"]
        )
        self.assertEqual(len(suites["cartpole_soft_walls"]), 15)
        self.assertEqual(len(suites["push_box"]), 25)
        self.assertEqual(len(suites["transport"]), 15)
        for cases in suites.values():
            self.assertEqual(len({case["id"] for case in cases}), len(cases))
            self.assertTrue(
                all(set(case["overrides"]) == {"initial_state", "target_state"}
                    for case in cases)
            )

    @mock.patch("run_benchmarks.subprocess.run")
    def test_dirty_check_normalizes_cross_platform_line_endings(self, run) -> None:
        run.return_value.returncode = 0
        run.return_value.stdout = ""
        self.assertFalse(git_tracked_dirty(Path(".")))
        self.assertEqual(
            run.call_args.args[0][:3], ["git", "-c", "core.autocrlf=true"]
        )
    def test_push_t_segment_is_shared_with_both_solvers(self) -> None:
        self.assertEqual(
            case_environment({}, push_t_segment=23),
            {"CONTACT_BENCHMARK_PUSH_T_SEGMENT": "23"},
        )

    def test_percentile_uses_linear_interpolation(self) -> None:
        self.assertAlmostEqual(percentile([1.0, 2.0, 3.0, 4.0], 0.25), 1.75)
        self.assertAlmostEqual(percentile([1.0, 2.0, 3.0, 4.0], 0.90), 3.7)

    def test_summary_excludes_failed_audit_from_paired_speedup(self) -> None:
        document = {
            "schema_version": 5,
            "created_utc": "2027-01-01T00:00:00+00:00",
            "protocol": {
                "warmups": 1,
                "repetitions": 2,
                "cpu_affinity": [0],
                "paper_reported_times_used": False,
            },
            "machine": {},
            "revision": {},
            "results": [
                result("ContactIPM", 0, 1.0, True),
                result("CRISP", 0, 2.0, True),
                result("ContactIPM", 1, 3.0, True),
                result("CRISP", 1, 6.0, False),
            ],
        }
        summary = summarize(document, bootstrap_samples=100, seed=7, min_repetitions=2)
        pair = summary["paired_comparisons"][0]
        self.assertEqual(pair["pairs"], 2)
        self.assertEqual(pair["qualified_pairs"], 1)
        self.assertEqual(
            pair["robustness_outcomes"],
            {"both": 1, "contactipm_only": 1, "crisp_only": 0, "neither": 0},
        )
        self.assertAlmostEqual(
            pair["all_process_speedup_crisp_over_contactipm"]["median"], 2.0
        )
        self.assertAlmostEqual(
            pair["qualified_speedup_crisp_over_contactipm"]["median"], 2.0
        )
        self.assertFalse(pair["publication_timing_ready"])
        self.assertTrue(summary["readiness"]["protocol_ready"])

    def test_common_audit_is_authoritative_over_process_exit_gate(self) -> None:
        contact = result("ContactIPM", 0, 1.0, True)
        crisp = result("CRISP", 0, 2.0, True)
        contact["returncode"] = 1
        document = {
            "schema_version": 5,
            "protocol": {
                "warmups": 1,
                "repetitions": 1,
                "cpu_affinity": [0],
                "paper_reported_times_used": False,
            },
            "results": [contact, crisp],
        }
        summary = summarize(document, bootstrap_samples=100, seed=7, min_repetitions=1)
        contact_group = next(
            group for group in summary["groups"] if group["solver"] == "ContactIPM"
        )
        self.assertEqual(contact_group["process_successful_trials"], 0)
        self.assertEqual(contact_group["qualified_trials"], 1)
        self.assertEqual(summary["paired_comparisons"][0]["qualified_pairs"], 1)

    def test_summary_pairs_same_repetition_across_cases(self) -> None:
        rows = []
        for case_id, contact_time, crisp_time in (
            ("case_a", 1.0, 2.0),
            ("case_b", 2.0, 6.0),
        ):
            contact = result("ContactIPM", 0, contact_time, True)
            crisp = result("CRISP", 0, crisp_time, True)
            contact["case_id"] = crisp["case_id"] = case_id
            rows.extend((contact, crisp))
        document = {
            "schema_version": 5,
            "protocol": {
                "warmups": 1,
                "repetitions": 1,
                "cpu_affinity": [0],
                "paper_reported_times_used": False,
                "case_suite": "validation_cases.json",
            },
            "results": rows,
        }
        summary = summarize(document, bootstrap_samples=100, seed=7, min_repetitions=1)
        pair = summary["paired_comparisons"][0]
        self.assertEqual(pair["pairs"], 2)
        self.assertEqual(pair["qualified_pairs"], 2)
        self.assertAlmostEqual(
            pair["qualified_speedup_crisp_over_contactipm"]["median"], 2.5
        )
        self.assertFalse(summary["readiness"]["protocol_ready"])

    def test_environment_override_marks_ablation_not_timing(self) -> None:
        document = {
            "schema_version": 5,
            "protocol": {
                "warmups": 1,
                "repetitions": 1,
                "cpu_affinity": [0],
                "paper_reported_times_used": False,
                "contactipm_environment": {"CONTACTIPM_PUSH_RECOVERY": "0"},
            },
            "results": [
                result("ContactIPM", 0, 1.0, True),
                result("CRISP", 0, 2.0, True),
            ],
        }
        summary = summarize(document, bootstrap_samples=100, seed=7, min_repetitions=1)
        self.assertFalse(summary["readiness"]["protocol_ready"])
        self.assertIn("ablation", " ".join(summary["readiness"]["issues"]).lower())

    def test_objective_override_marks_pareto_not_timing(self) -> None:
        document = {
            "schema_version": 5,
            "protocol": {
                "warmups": 1,
                "repetitions": 20,
                "cpu_affinity": [0],
                "paper_reported_times_used": False,
                "benchmark_objective_environment": {
                    "CONTACT_BENCHMARK_EFFORT_SCALE": "2.0"
                },
            },
            "results": [
                result("ContactIPM", 0, 1.0, True),
                result("CRISP", 0, 2.0, True),
            ],
        }
        summary = summarize(
            document, bootstrap_samples=100, seed=7, min_repetitions=1
        )
        self.assertFalse(summary["readiness"]["protocol_ready"])
        self.assertIn("pareto", " ".join(summary["readiness"]["issues"]).lower())

    def test_node_override_marks_scaling_not_timing(self) -> None:
        document = {
            "schema_version": 5,
            "protocol": {
                "warmups": 1,
                "repetitions": 20,
                "cpu_affinity": [0],
                "paper_reported_times_used": False,
                "scaling": {
                    "nodes": 50,
                    "total_complementarity_pairs": 490,
                },
            },
            "results": [
                result("ContactIPM", 0, 1.0, True),
                result("CRISP", 0, 2.0, True),
            ],
        }
        summary = summarize(
            document, bootstrap_samples=100, seed=7, min_repetitions=1
        )
        self.assertFalse(summary["readiness"]["protocol_ready"])
        self.assertIn("scaling", " ".join(summary["readiness"]["issues"]).lower())

    def test_pareto_front_excludes_dominated_success(self) -> None:
        points = [
            {
                "solver": "ContactIPM",
                "problem": "push_box",
                "successful": True,
                "effort_cost": 1.0,
                "tracking_cost": 3.0,
            },
            {
                "solver": "ContactIPM",
                "problem": "push_box",
                "successful": True,
                "effort_cost": 2.0,
                "tracking_cost": 2.0,
            },
            {
                "solver": "ContactIPM",
                "problem": "push_box",
                "successful": True,
                "effort_cost": 3.0,
                "tracking_cost": 4.0,
            },
        ]
        mark_efficient(points)
        self.assertEqual(
            [point["pareto_efficient"] for point in points], [True, True, False]
        )

    def test_pareto_front_excludes_nonconverged_solution(self) -> None:
        points = [
            {
                "solver": "ContactIPM",
                "problem": "push_box",
                "solver_converged": False,
                "successful": True,
                "effort_cost": 1.0,
                "tracking_cost": 1.0,
            }
        ]
        mark_efficient(points)
        self.assertFalse(points[0]["pareto_efficient"])

    def test_scaling_requires_convergence_and_audit(self) -> None:
        contact = result("ContactIPM", 0, 1.0, True)
        crisp = result("CRISP", 0, 2.0, True)
        contact["returncode"] = 1
        contact["stdout"] = "Iterations: 12 total"
        crisp["stdout"] = "Optimization problem solved in 20 iterations."
        point = summarize_point(
            {"results": [contact, crisp]}, "push_box", nodes=50, time_step=1.98 / 49
        )
        self.assertEqual(point["total_complementarity_pairs"], 490)
        self.assertEqual(point["solvers"]["ContactIPM"]["audit_passed"], 1)
        self.assertEqual(point["solvers"]["ContactIPM"]["eligible"], 0)
        self.assertEqual(point["solvers"]["CRISP"]["eligible"], 1)
        self.assertEqual(point["qualified_pairs"], 0)

    def test_push_t_audit_accepts_scaling_case_override(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            trajectory = Path(directory) / "contactipm_push_t_seg_08.txt"
            trajectory.write_text(
                "\n".join(" ".join(["0"] * 29) for _ in range(25)),
                encoding="utf-8",
            )
            metrics = audit_push_t(
                Path(directory),
                "contactipm",
                0.0,
                segments=[8],
                case_override={"nodes": 25, "dt": 2.45 / 24},
            )
        self.assertEqual(metrics["attempted_instances"], 1)

    def test_ablation_summary_uses_common_audit(self) -> None:
        contact = result("ContactIPM", 0, 1.5, True)
        metrics = summarize_artifact({"results": [contact]})
        self.assertEqual(metrics["successful"], 1)
        self.assertEqual(metrics["process_exit_successful"], 1)
        self.assertEqual(metrics["attempted"], 1)
        self.assertEqual(metrics["median_wall_time_seconds"], 1.5)
        self.assertEqual(metrics["median_successful_objective"], 3.0)

        contact["returncode"] = 1
        metrics = summarize_artifact({"results": [contact]})
        self.assertEqual(metrics["successful"], 1)
        self.assertEqual(metrics["process_exit_successful"], 0)

    def test_markdown_handles_missing_solver_result(self) -> None:
        document = {
            "schema_version": 5,
            "protocol": {
                "warmups": 1,
                "repetitions": 1,
                "cpu_affinity": [0],
                "paper_reported_times_used": False,
            },
            "results": [result("ContactIPM", 0, 1.0, True)],
        }
        summary = summarize(document, bootstrap_samples=100, seed=7, min_repetitions=1)
        markdown = render_markdown(summary, Path("incomplete.json"))
        self.assertIn("| Push Box | 1.0000", markdown)
        self.assertIn("| n/a | 0/0 |", markdown)

if __name__ == "__main__":
    unittest.main()
