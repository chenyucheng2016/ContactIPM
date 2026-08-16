#!/usr/bin/env python3

import importlib.util
import math
import unittest
from pathlib import Path


SCRIPT = (Path(__file__).parents[1] / "examples" / "quadruped_cito" /
          "mujoco" / "run_replanning_sweep.py")
SPEC = importlib.util.spec_from_file_location("run_replanning_sweep", SCRIPT)
SWEEP = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(SWEEP)


class ReplanningSweepTest(unittest.TestCase):

    def test_fixed_stabilization_handoff_flag_is_supported(self):
        arguments = SWEEP.parse_args(
            ["sim", "scene", "results.csv",
             "--fixed-stabilization-handoff"])
        self.assertTrue(arguments.fixed_stabilization_handoff)

    def test_snapshot_directory_is_supported(self):
        arguments = SWEEP.parse_args(
            ["sim", "scene", "results.csv",
             "--snapshot-directory", "snapshots"])
        self.assertEqual(arguments.snapshot_directory, Path("snapshots"))

    def test_missing_replay_measurements_are_not_numeric_maxima(self):
        values = [0.012, 1e300, math.nan, 0.276]
        self.assertEqual(SWEEP.observed_replay_errors(values), [0.012, 0.276])

    def test_feedback_replay_metrics_are_parsed(self):
        line = (
            "ContactIPM replay: segment=2 horizon=100 duration=6.000 "
            "base_position_rms=0.012000 saturation_fraction=0.001000 "
            "early_contact_activations=3 late_touchdown_search_ticks=17 "
            "final_base=(0.100000, 0.000000, 0.250000)")
        replay = SWEEP.REPLAY_RE.search(line)
        self.assertIsNotNone(replay)
        self.assertEqual(replay.groups(),
                         ("2", "0.012000", "0.001000", "3", "17",
                          "0.100000", "0.000000", "0.250000"))

    def test_adaptive_stabilization_duration_is_parsed(self):
        line = (
            "replanning stabilization: duration=2.048 contacts=4 "
            "position_error=0.011094 orientation_error=0.087895 "
            "linear_speed=0.003036 angular_speed=0.001299 "
            "max_foot_slip=0.004920 target=(0.1, 0.0, 0.25) accepted=1")
        stabilization = SWEEP.STABILIZATION_RE.search(line)
        self.assertIsNotNone(stabilization)
        self.assertEqual(stabilization.groups(),
                         ("2.048", "4", "0.011094", "0.087895",
                          "0.003036", "0.001299", "0.004920", "1"))

    def test_signed_event_metric_is_parsed(self):
        line = (
            "  event segment=3 foot=0 index=0: unload=1 "
            "clearance=0.046041 landing_error=0.037171 "
            "planned_liftoff=3.002000 measured_liftoff=3.024000 "
            "planned_touchdown=4.082000 measured_touchdown=3.826000 "
            "touchdown_error=0.256000 post_touchdown_slip=0.007089 "
            "accepted=0 task_accepted=0 timing_accepted=0 "
            "signed_touchdown_error=-0.256000")
        event = SWEEP.EVENT_RE.search(line)
        self.assertIsNotNone(event)
        self.assertEqual(event.groups(),
                         ("3", "0", "0", "0.046041", "0.037171",
                          "4.082000", "3.826000", "0.256000",
                          "0.007089", "0", "0", "0", "-0.256000"))

    def test_touchdown_window_trace_is_parsed(self):
        line = (
            "touchdown trace: segment=3 tick=1900 foot=0 event=0 "
            "time=3.802000 relative_time=-0.280000 plan_mask=0x6 "
            "support_mask=0x6 measured_mask=0xe plan_contact=0 "
            "support_contact=0 measured_contact=0 desired_gap=0.004000 "
            "measured_gap=0.002000 desired_vz=-0.030000 "
            "measured_vz=-0.020000 normal_force=0.000000 "
            "contact_blend=0.000000 foot_position_error=0.003000 "
            "wrench_residual=20.000000")
        trace = SWEEP.TOUCHDOWN_TRACE_RE.search(line)
        self.assertIsNotNone(trace)
        self.assertEqual(trace.group(1), "3")
        self.assertEqual(trace.group(6), "-0.280000")
        self.assertEqual(trace.group(20), "20.000000")

    def test_touchdown_control_trace_is_parsed(self):
        line = (
            "touchdown control trace: segment=3 tick=1900 foot=0 event=0 "
            "time=3.802000 relative_time=-0.280000 plan_mask=0x6 "
            "support_mask=0x6 measured_mask=0xe "
            "foot_position_error=(0.1,0.2,0.3) "
            "foot_velocity_error=(0.4,0.5,0.6) "
            "joint_position=(1,2,3) joint_velocity=(4,5,6) "
            "requested_torque=(7,8,9) predicted_torque=(7,8,9) "
            "applied_torque=(7,8,9) torque_headroom=(10,11,12) "
            "saturated_mask=0x0 swing_force=(13,14,15) "
            "jacobian_condition=3.500000 "
            "base_position_error=(0.01,0.02,0.03) "
            "base_velocity_error=(0.04,0.05,0.06) "
            "orientation_error_world=(0.07,0.08,0.09) "
            "angular_velocity_error_body=(0.1,0.2,0.3) "
            "wrench_residual=(1,2,3,4,5,6)")
        trace = SWEEP.TOUCHDOWN_CONTROL_TRACE_RE.search(line)
        self.assertIsNotNone(trace)
        self.assertEqual(trace.group(1), "3")
        self.assertEqual(trace.group(18), "0x0")
        self.assertEqual(trace.group(20), "3.500000")
        self.assertEqual(trace.group(25), "1,2,3,4,5,6")


if __name__ == "__main__":
    unittest.main()
