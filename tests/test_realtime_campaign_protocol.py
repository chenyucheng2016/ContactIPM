#!/usr/bin/env python3
from __future__ import annotations

import importlib.util
import subprocess
import sys
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
CAMPAIGN = ROOT / "examples" / "quadruped_cito" / "run_realtime_terrain_campaign.py"
BENCHMARK = ROOT / "examples" / "quadruped_cito" / "quadruped_cito_realtime_benchmark.cpp"


def load_campaign():
    spec = importlib.util.spec_from_file_location("realtime_campaign", CAMPAIGN)
    assert spec is not None and spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = module
    spec.loader.exec_module(module)
    return module


def rejected(*arguments: str) -> bool:
    completed = subprocess.run(
        [sys.executable, str(CAMPAIGN), *arguments],
        capture_output=True,
        text=True,
        check=False,
    )
    return completed.returncode != 0


def qualified_row(**overrides: object) -> dict[str, object]:
    row: dict[str, object] = {
        "terrain_class": "flat",
        "terrain_fingerprint": "flat",
        "terrain_seed": "0",
        "scope": "core",
        "held_out": "0",
        "return_code": "0",
        "benchmark_config_valid": "1",
        "contact_task_protocol_qualified": "1",
        "updates": "1000",
        "completed": "1000",
        "episodes": "100",
        "cold_failures": "0",
        "contact_task_episode_updates": "10",
        "completed_contact_tasks": "100",
        "contact_task_updates": "1000",
        "unqualified_updates": "0",
        "full_kkt": "1000",
        "task_pass": "1000",
        "published": "1000",
        "fallbacks": "0",
        "invalid_publications": "0",
        "exact_hessian_calls": "0",
        "deadline_misses": "0",
        "e2e_p99_ms": "1",
        "max_consecutive_fallbacks": "0",
        "verdict": "PASS",
    }
    row.update(overrides)
    return row


def main() -> int:
    module = load_campaign()
    provenance = {
        "git_revision": "abc", "worktree_diff_sha256": "diff",
        "benchmark_sha256": "binary", "benchmark_size_bytes": 123,
    }
    if not module.code_provenance_matches(provenance, dict(provenance)):
        raise AssertionError("identical code provenance was rejected")
    changed_binary = dict(provenance, benchmark_sha256="changed")
    if module.code_provenance_matches(provenance, changed_binary):
        raise AssertionError("changed benchmark binary was accepted")
    missing_diff = dict(provenance)
    del missing_diff["worktree_diff_sha256"]
    if module.code_provenance_matches(provenance, missing_diff):
        raise AssertionError("missing end-of-campaign provenance was accepted")
    flat_case = module.Case("flat", "flat", ("--terrain", "flat"))
    flat_config = {
        "terrain": "flat", "terrain_fingerprint": "flat-fingerprint",
        "terrain_kind": "flat", "terrain_seed": "0", "N": "50",
        "nx": "25", "nu": "28",
        "nc": "86", "rate_hz": "5", "shift_steps": "4",
        "updates": "100", "episode_updates": "10",
        "contact_task_episode_updates": "10", "episode_reset": "standing",
        "contact_task_update_definition":
            "warm_update_in_complete_active_to_settled_episode",
        "recovery": "off", "exact_hessian": "off",
        "adaptive_exact_hessian": "off", "nonlinear_rollout": "off",
        "terrain_convention": "go1_foot_center", "dt": "0.050",
        "horizon_s": "2.500", "deadline_ms": "200.000",
        "measurement_offset": "0.0005",
        "offset": "-0.017805846", "slope_x": "0", "slope_y": "0",
        "amplitude": "0", "wave_number_x": "0", "wave_number_y": "0",
        "phase_x": "0", "phase_y": "0", "secondary_amplitude": "0",
        "secondary_wave_number_x": "0", "secondary_wave_number_y": "0",
        "secondary_phase_x": "0", "secondary_phase_y": "0",
        "step_height": "0", "step_center_x": "0", "step_sharpness": "0",
        "riccati_reg_base": "1e-8",
    }
    flat_summary = {
        "terrain": "flat", "terrain_fingerprint": "0123456789abcdef",
        "terrain_seed": "0",
    }
    flat_config["terrain_fingerprint"] = flat_summary["terrain_fingerprint"]
    if not module.benchmark_config_matches_case(
        flat_config, flat_summary, flat_case, 100, 5, 10
    ):
        raise AssertionError("matching benchmark configuration was rejected")
    wrong_horizon = dict(flat_config, horizon_s="3.000")
    if module.benchmark_config_matches_case(
        wrong_horizon, flat_summary, flat_case, 100, 5, 10
    ):
        raise AssertionError("wrong executed horizon was accepted")
    wrong_terrain = dict(flat_config, terrain="sinusoidal")
    if module.benchmark_config_matches_case(
        wrong_terrain, flat_summary, flat_case, 100, 5, 10
    ):
        raise AssertionError("wrong executed terrain was accepted")
    missing_fingerprint_config = dict(flat_config)
    missing_fingerprint_summary = dict(flat_summary)
    del missing_fingerprint_config["terrain_fingerprint"]
    del missing_fingerprint_summary["terrain_fingerprint"]
    if module.benchmark_config_matches_case(
        missing_fingerprint_config, missing_fingerprint_summary,
        flat_case, 100, 5, 10
    ):
        raise AssertionError("missing terrain fingerprint was accepted")

    sine_case = module.Case(
        "sine", "sinusoidal",
        ("--terrain", "sinusoidal", "--terrain-amplitude", "0.02"),
    )
    sine_config = dict(
        flat_config, terrain="sinusoidal", terrain_kind="sinusoidal",
        amplitude="0.02", wave_number_x="4", wave_number_y="3",
    )
    sine_summary = dict(flat_summary, terrain="sinusoidal")
    if not module.benchmark_config_matches_case(
        sine_config, sine_summary, sine_case, 100, 5, 10
    ):
        raise AssertionError("matching sinusoidal configuration was rejected")
    if module.benchmark_config_matches_case(
        dict(sine_config, wave_number_x="5"), sine_summary,
        sine_case, 100, 5, 10
    ):
        raise AssertionError("wrong sinusoidal frequency was accepted")

    step_case = module.Case(
        "step", "smooth_step",
        ("--terrain", "smooth_step", "--step-height", "0.02"),
    )
    step_config = dict(
        flat_config, terrain="smooth_step", terrain_kind="smooth_step",
        step_height="0.02", step_center_x="0.23", step_sharpness="18",
    )
    step_summary = dict(flat_summary, terrain="smooth_step")
    if not module.benchmark_config_matches_case(
        step_config, step_summary, step_case, 100, 5, 10
    ):
        raise AssertionError("matching smooth-step configuration was rejected")
    if module.benchmark_config_matches_case(
        dict(step_config, step_center_x="0.30"), step_summary,
        step_case, 100, 5, 10
    ):
        raise AssertionError("wrong smooth-step location was accepted")

    random_case = module.Case(
        "random", "random_smooth_02cm",
        ("--terrain", "random_smooth", "--terrain-amplitude", "0.02",
         "--terrain-seed", "1000"),
        terrain_seed=1000,
    )
    random_config = dict(
        flat_config, terrain="random_smooth", terrain_kind="random_smooth",
        terrain_seed="1000", amplitude="0.013",
        wave_number_x="4.63981426", wave_number_y="3.89541674",
        phase_x="2.03629433", phase_y="0.0258567142",
        secondary_amplitude="0.007",
        secondary_wave_number_x="7.4117291",
        secondary_wave_number_y="5.47614348",
        secondary_phase_x="0.655722483",
        secondary_phase_y="3.34648844",
    )
    random_summary = dict(
        flat_summary, terrain="random_smooth", terrain_seed="1000"
    )
    if not module.benchmark_config_matches_case(
        random_config, random_summary, random_case, 100, 5, 10
    ):
        raise AssertionError("matching seeded random terrain was rejected")
    rows = [qualified_row(exact_hessian_calls="1")]
    if module.class_summaries(rows)[0]["verdict"] != "FAIL":
        raise AssertionError("class accepted nonzero exact-Hessian calls")
    for missing in (
        "e2e_p99_ms", "max_consecutive_fallbacks", "exact_hessian_calls",
        "invalid_publications", "cold_failures", "published",
    ):
        missing_row = qualified_row()
        del missing_row[missing]
        if module.class_summaries([missing_row])[0]["verdict"] != "FAIL":
            raise AssertionError(f"class accepted missing evidence {missing}")
    if module.class_summaries(
        [qualified_row(contact_task_updates="999", unqualified_updates="1")]
    )[0]["verdict"] != "FAIL":
        raise AssertionError("class accepted an unqualified warm update")
    if module.class_summaries(
        [qualified_row(updates="500", completed="500",
                       completed_contact_tasks="50",
                       contact_task_updates="500", full_kkt="500",
                       task_pass="500", published="500")],
        minimum_contact_task_updates=1000,
    )[0]["verdict"] != "FAIL":
        raise AssertionError("class accepted fewer than 1000 task updates")

    split_rows = [
        qualified_row(
            updates="500", completed="500", episodes="50",
            completed_contact_tasks="50", contact_task_updates="500",
            full_kkt="500", task_pass="500", published="500",
            terrain_fingerprint=f"flat-{index}",
        )
        for index in range(2)
    ]
    if module.class_summaries(
        split_rows, minimum_contact_task_updates=1000
    )[0]["verdict"] != "PASS":
        raise AssertionError("class did not aggregate 1000 qualified updates")

    random_rows = [
        qualified_row(
            terrain_class="random_smooth_02cm",
            terrain_seed=str(2000 + index), held_out="1",
            terrain_fingerprint=f"random-{index}", updates="100",
            completed="100", episodes="10", completed_contact_tasks="10",
            contact_task_updates="100", full_kkt="100", task_pass="100",
            published="100",
        )
        for index in range(20)
    ]
    if module.class_summaries(
        random_rows, minimum_contact_task_updates=1000,
        minimum_random_seeds=20,
    )[0]["verdict"] != "PASS":
        raise AssertionError("qualified 20-seed random class was rejected")
    random_rows[-1]["terrain_fingerprint"] = random_rows[0]["terrain_fingerprint"]
    if module.class_summaries(
        random_rows, minimum_contact_task_updates=1000,
        minimum_random_seeds=20,
    )[0]["verdict"] != "FAIL":
        raise AssertionError("duplicate held-out terrain was accepted")

    valid_cases = module.terrain_cases([0.10], [0.02, 0.04], 2000, 20)
    if module.acceptance_case_errors(valid_cases):
        raise AssertionError("valid acceptance matrix was rejected")
    weak_slope_cases = module.terrain_cases(
        [0.0001], [0.02, 0.04], 2000, 20
    )
    if not module.acceptance_case_errors(weak_slope_cases):
        raise AssertionError("acceptance matrix allowed trivial core slopes")
    reused_cases = module.terrain_cases([0.10], [0.02, 0.04], 1000, 20)
    if not module.acceptance_case_errors(reused_cases):
        raise AssertionError("development seeds were accepted as held out")

    cases_by_name = {case.name: case for case in valid_cases}
    expected_stress_classes = {
        "cross_positive_20deg_stress": "cross_slope_20deg_stress",
        "cross_negative_20deg_stress": "cross_slope_20deg_stress",
        "sinusoidal_06cm": "sinusoidal_06cm_stress",
        "smooth_step_04cm": "smooth_step_04cm_stress",
    }
    for name, terrain_class in expected_stress_classes.items():
        case = cases_by_name[name]
        if not case.stress or case.terrain_class != terrain_class:
            raise AssertionError(f"stress case {name} was not separated")
    if cases_by_name["sinusoidal_04cm"].stress or cases_by_name[
        "smooth_step_02cm"
    ].stress:
        raise AssertionError("a core terrain was classified as stress")

    failed_stress_class = module.class_summaries([
        qualified_row(
            terrain_class="sinusoidal_06cm_stress", scope="stress",
            verdict="FAIL", return_code="1",
        )
    ])[0]
    if failed_stress_class["scope"] != "stress":
        raise AssertionError("stress class summary lost its scope")
    if module.failed_verdict_counts([
        module.class_summaries([qualified_row()])[0], failed_stress_class,
    ]) != (0, 1):
        raise AssertionError("stress failure contaminated the core verdict")

    summary = {key: str(value) for key, value in qualified_row().items()}
    if not module.summary_qualifies_contact_task_protocol(summary, 1000, 10):
        raise AssertionError("qualified benchmark summary was rejected")
    del summary["completed_contact_tasks"]
    if module.summary_qualifies_contact_task_protocol(summary, 1000, 10):
        raise AssertionError("stale benchmark summary was accepted")

    common = ("--benchmark", str(BENCHMARK), "--output", str(ROOT / "tmp"),
              "--profile", "acceptance", "--list")
    if not rejected(*common, "--updates", "100", "--random-seeds", "20"):
        raise AssertionError("acceptance profile allowed fewer than 1000 updates")
    if not rejected(*common, "--updates", "1000", "--random-seeds", "1"):
        raise AssertionError("acceptance profile allowed fewer than 20 seeds")
    if not rejected(*common, "--updates", "1000", "--random-seeds", "20",
                    "--seed-start", "1000"):
        raise AssertionError("acceptance profile reused development seeds")
    if not rejected(*common, "--updates", "1000", "--random-seeds", "20",
                    "--seed-start", "3000"):
        raise AssertionError("acceptance profile changed the frozen seed range")
    if not rejected(*common, "--updates", "1000", "--random-seeds", "20",
                    "--random-amplitudes", "0.02"):
        raise AssertionError("acceptance profile omitted random 4 cm terrain")
    if not rejected(*common, "--updates", "1000", "--random-seeds", "20",
                    "--episode-updates", "9"):
        raise AssertionError("acceptance profile allowed a partial contact task")
    if not rejected(*common, "--updates", "1000", "--random-seeds", "20",
                    "--random-updates-per-seed", "10"):
        raise AssertionError("acceptance profile undersampled each random seed")
    if not rejected(*common, "--updates", "1000", "--random-seeds", "20",
                    "--slope-grades", "0.0001"):
        raise AssertionError("acceptance profile allowed trivial core slopes")
    if rejected(*common, "--updates", "1000", "--random-seeds", "20"):
        raise AssertionError("valid acceptance protocol was rejected")
    development = (
        "--benchmark", str(BENCHMARK), "--output", str(ROOT / "tmp"),
        "--profile", "development", "--list", "--seed-start", "2000",
        "--random-seeds", "20",
    )
    if not rejected(*development):
        raise AssertionError("development profile consumed frozen held-out seeds")
    listed = subprocess.run(
        [sys.executable, str(CAMPAIGN), *common, "--updates", "1000",
         "--random-seeds", "20"],
        capture_output=True, text=True, check=False,
    )
    random_lines = [
        line for line in listed.stdout.splitlines() if line.startswith("random_")
    ]
    if len(random_lines) != 40 or any("updates=100" not in line for line in random_lines):
        raise AssertionError("acceptance workload is not 100 updates per random seed")
    if "flat scope=core updates=1000" not in listed.stdout:
        raise AssertionError("deterministic acceptance workload is not 1000 updates")
    if "slope_up_20pct" in listed.stdout or "cross_up_20pct" in listed.stdout:
        raise AssertionError("default acceptance matrix added unrequested 20% core slopes")
    for name in expected_stress_classes:
        if not any(
            line.startswith(name + " ") and "scope=stress" in line
            for line in listed.stdout.splitlines()
        ):
            raise AssertionError(f"stress case {name} was not reported separately")

    benchmark_source = BENCHMARK.read_text(encoding="utf-8")
    if "exact_hessian_calls == 0" not in benchmark_source:
        raise AssertionError("benchmark acceptance no longer gates curvature calls")
    if "contact_task_updates == completed_updates" not in benchmark_source:
        raise AssertionError("benchmark acceptance no longer gates task coverage")
    if "completed_contact_tasks > 0" not in benchmark_source:
        raise AssertionError("benchmark no longer requires a completed contact task")
    if "contact_task_updates += required_contact_task_updates" not in benchmark_source:
        raise AssertionError("benchmark task counter is not transition-qualified")
    if benchmark_source.count("result.published &&") < 2:
        raise AssertionError("benchmark credits an unpublished contact-task witness")
    if "warm_update_in_complete_active_to_" not in benchmark_source:
        raise AssertionError("benchmark does not define contact-task update semantics")
    for field in ("compiler=%s", "fast_math=%d", "ndebug=%d", "cplusplus=%ld"):
        if field not in benchmark_source:
            raise AssertionError(f"benchmark is missing build provenance {field}")
    campaign_source = CAMPAIGN.read_text(encoding="utf-8")
    for field in (
        "benchmark_sha256", "git_revision", "worktree_diff_sha256",
        "end_code_provenance", "code_provenance_valid",
        "clean_worktree_required", "clean_worktree_valid", "platform",
        "command",
    ):
        if field not in campaign_source:
            raise AssertionError(f"campaign is missing provenance {field}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
