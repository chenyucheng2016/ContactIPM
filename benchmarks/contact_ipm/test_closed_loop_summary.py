#!/usr/bin/env python3
"""Regression tests for closed-loop validation aggregation."""

from __future__ import annotations

import unittest

from summarize_closed_loop import EXPECTED_GROUPS, quantile, summarize, validate_suite


def rollout(index: int, times: list[float]) -> dict:
    return {
        "index": index,
        "group": "nominal",
        "task_success": True,
        "steps": len(times),
        "goal_hold_steps": 10,
        "solve_times_s": times,
        "primary_failures": 0,
        "cold_fallbacks": 0,
        "unrecovered_failures": 0,
        "invalid_plans": 0,
        "deadline_misses_all": sum(value > 0.1 for value in times),
        "deadline_misses_warm": sum(value > 0.1 for value in times[1:]),
        "final_translation_error": 0.01,
        "final_angular_error": 0.02,
        "max_plan_dynamics_defect": 1e-7,
        "max_plan_side_violation": 1e-9,
        "max_plan_physical_mpcc": 1e-6,
        "max_applied_side_violation": 1e-9,
        "max_applied_physical_mpcc": 1e-6,
        "max_one_step_model_error": 0.1,
    }


class ClosedLoopSummaryTests(unittest.TestCase):
    def test_quantile_uses_linear_interpolation(self) -> None:
        self.assertAlmostEqual(quantile([1, 2, 3, 4], 0.9), 3.7)

    def test_summary_separates_cold_and_warm_deadline_misses(self) -> None:
        payload = {
            "benchmark": "push_box_closed_loop",
            "seed": 2027,
            "configuration": {"control_dt": 0.1, "goal_hold_steps": 10},
            "rollouts": [
                rollout(0, [0.12, 0.02]),
                rollout(1, [0.05, 0.11]),
            ],
        }
        result = summarize(payload, "digest")
        overall = result["overall"]
        self.assertEqual(overall["deadline_misses_all"], 2)
        self.assertEqual(overall["deadline_misses_warm"], 1)
        self.assertEqual(overall["solve_count"], 4)
        self.assertEqual(overall["warm_solve_count"], 2)
        self.assertAlmostEqual(overall["timing_all_ms"]["median"], 80.0)

        payload["rollouts"][0]["goal_hold_steps"] = 9
        with self.assertRaises(ValueError):
            summarize(payload, "digest")

    def test_suite_validation_checks_group_cardinalities(self) -> None:
        summary = {
            "overall": {"rollouts": sum(EXPECTED_GROUPS.values())},
            "groups": [
                {"group": name, "rollouts": count}
                for name, count in EXPECTED_GROUPS.items()
            ],
        }
        validate_suite(summary)
        summary["groups"][0]["rollouts"] -= 1
        with self.assertRaises(ValueError):
            validate_suite(summary)


if __name__ == "__main__":
    unittest.main()
