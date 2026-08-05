#!/usr/bin/env python3
"""Fail when frozen benchmark instances drift from either solver adapter."""

from __future__ import annotations

import hashlib
import json
from pathlib import Path


HERE = Path(__file__).resolve().parent
ROOT = HERE.parents[1]
MANIFEST = json.loads((HERE / "source_cases.json").read_text(encoding="utf-8"))
VALIDATION = json.loads(
    (HERE / "validation_cases.json").read_text(encoding="utf-8")
)


def require(path: Path, snippets: list[str]) -> None:
    text = path.read_text(encoding="utf-8")
    missing = [snippet for snippet in snippets if snippet not in text]
    if missing:
        raise RuntimeError(f"{path}: source-case drift, missing {missing!r}")


def main() -> int:
    cases = MANIFEST["problems"]
    expected_cases = {
        "cartpole_soft_walls": {
            "nodes": 100,
            "dt": 0.02,
            "initial_state": [0.0, 3.14159, 0.0, 2.8],
            "target_state": [0.0, 0.0, 0.0, 0.0],
        },
        "push_box": {
            "nodes": 100,
            "dt": 0.02,
            "target_segment": 12,
            "target_segments": 18,
            "target_radius": 3.0,
        },
        "transport": {
            "nodes": 200,
            "dt": 0.02,
            "initial_state": [3.5, 3.0, -4.0, -4.0, 0.0, 0.0],
            "target_state": [-0.5, 0.0, -2.0, -2.0, 0.0, 0.0],
        },
        "push_t": {
            "nodes": 50,
            "dt": 0.05,
            "target_pose": [0.01, 0.01, 0.01],
            "initial_cases": {
                "count": 50,
                "angle_start": 0.0,
                "angle_period": 2.0 * 3.141592653589793,
                "radius_start": 0.25,
                "radius_end": 0.5,
            },
        },
    }
    expected_validation_counts = {
        "cartpole_soft_walls": 15, "push_box": 25, "transport": 15
    }
    for problem, count in expected_validation_counts.items():
        if VALIDATION["problems"][problem]["expected_count"] != count:
            raise RuntimeError(f"{problem}: validation case count drifted")

    for problem, expected in expected_cases.items():
        for field, value in expected.items():
            if cases[problem][field] != value:
                raise RuntimeError(
                    f"{problem}.{field}: {cases[problem][field]!r} != {value!r}"
                )
    guess = ROOT / cases["cartpole_soft_walls"]["initial_guess"]["path"]
    digest = hashlib.sha256(guess.read_bytes()).hexdigest()
    expected = cases["cartpole_soft_walls"]["initial_guess"]["sha256"]
    if digest != expected:
        raise RuntimeError(f"cartpole initial guess hash {digest} != {expected}")
    values = [float(value) for value in guess.read_text().split()]
    if len(values) != 700 or values[:4] != [0.0, 3.14159, 0.0, 2.8]:
        raise RuntimeError("cartpole initial guess shape or first state drifted")

    require(
        HERE / "cartpole_soft_walls.cpp",
        [
            "constexpr int HORIZON = 99;",
            "constexpr double DT = 0.02;",
            "constexpr double LENGTH = 0.8;",
            "constexpr double STIFF_LEFT = 200.0;",
            "load_crisp_guess(guess_file, *problem)",
            "CONTACT_BENCHMARK_INITIAL_STATE",
            "CONTACT_BENCHMARK_TARGET_STATE",
        ],
    )
    require(
        ROOT / "benchmarks/CRISP/src/examples/pushbot/cpp/SolvePushbot.cpp",
        [
            "const double dt = 0.02;",
            "const size_t N = 100;",
            "const double l = 0.8;",
            "const double k1 = 200.0;",
            "CONTACT_BENCHMARK_CARTPOLE_GUESS",
            "CONTACT_BENCHMARK_INITIAL_STATE",
            "CONTACT_BENCHMARK_TARGET_STATE",
        ],
    )
    require(
        HERE / "push_box.cpp",
        [
            "#define CONTACT_BENCHMARK_NODES 100",
            "constexpr int HORIZON = CONTACT_BENCHMARK_NODES - 1;",
            "#define CONTACT_BENCHMARK_DT 0.02",
            "constexpr double DT = CONTACT_BENCHMARK_DT;",
            "const double angle = 12.0 * 2.0 * PI / 18.0;",
            "target[0] = 3.0 * std::cos(angle);",
            "CONTACT_BENCHMARK_INITIAL_STATE",
            "CONTACT_BENCHMARK_TARGET_STATE",
            "problem->stages[k].x.zero();",
            "problem->stages[k].u.zero();",
        ],
    )
    require(
        ROOT / "benchmarks/CRISP/src/examples/pushbox/SolvePushbox.cpp",
        [
            "#define CONTACT_BENCHMARK_DT 0.02",
            "const scalar_t dt = CONTACT_BENCHMARK_DT;",
            "#define CONTACT_BENCHMARK_NODES 100",
            "const size_t N = CONTACT_BENCHMARK_NODES;",
            "size_t num_segments = 18;",
            "scalar_t theta = 12 * 2 * M_PI / num_segments;",
            "xInitialGuess.setZero();",
            "CONTACT_BENCHMARK_INITIAL_STATE",
            "CONTACT_BENCHMARK_TARGET_STATE",
        ],
    )
    require(
        HERE / "transport.cpp",
        [
            "constexpr int HORIZON = 199;",
            "constexpr double DT = 0.02;",
            "constexpr double GRAVITY = 9.81;",
            "problem->x0[0] = 3.5;",
            "return k < HORIZON ? 6 : 2;",
            "CONTACT_BENCHMARK_INITIAL_STATE",
            "CONTACT_BENCHMARK_TARGET_STATE",
            "problem->stages[k].x.zero();",
            "problem->stages[k].u.zero();",
        ],
    )
    require(
        ROOT / "benchmarks/CRISP/src/examples/Transp/cpp/SolveTransp.cpp",
        [
            "const scalar_t dt = 0.02;",
            "const size_t N = 200;",
            "xInitialGuess.setZero();",
            "xInitial<< x2_initial + 0.5, x2_initial, -4.0, -4.0, 0.0, 0.0;",
            "xFinal << x2_final - 0.5, x2_final, -2.0, -2.0, 0.0, 0.0;",
            "CONTACT_BENCHMARK_INITIAL_STATE",
            "CONTACT_BENCHMARK_TARGET_STATE",
        ],
    )
    require(
        HERE / "push_t.cpp",
        [
            "#define CONTACT_BENCHMARK_NODES 50",
            "constexpr int HORIZON = CONTACT_BENCHMARK_NODES - 1;",
            "#define CONTACT_BENCHMARK_DT 0.05",
            "constexpr double DT = CONTACT_BENCHMARK_DT;",
            "constexpr int NUM_SEGMENTS = 50;",
            "RADIUS_MIN + (RADIUS_MAX - RADIUS_MIN) * fraction",
            "problem->stages[k].x.zero();",
            "problem->stages[k].u.zero();",
        ],
    )
    require(
        ROOT / "benchmarks/CRISP/src/examples/pushT/SolvePushT.cpp",
        [
            "#define CONTACT_BENCHMARK_DT 0.05",
            "const scalar_t dt = CONTACT_BENCHMARK_DT;",
            "#define CONTACT_BENCHMARK_NODES 50",
            "const size_t N = CONTACT_BENCHMARK_NODES;",
            "const size_t numSegments = 50;",
            "const scalar_t radius_min = 0.25;",
            "const scalar_t radius_max = 0.5;",
            "xInitialGuess.setZero();",
            "CONTACT_BENCHMARK_PUSH_T_SEGMENT",
        ],
    )
    print("source-case parity verified for all four benchmarks")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
