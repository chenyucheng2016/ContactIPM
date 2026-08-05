#!/usr/bin/env python3
"""Generate the predeclared local IMPACT-parameter comparison suite."""

from __future__ import annotations

import argparse
import json
import math
import random
from pathlib import Path


SEED = 2027
COUNT = 50


def pose_cases(
    rng: random.Random,
    official_goal: list[float],
    radius_range: tuple[float, float],
    orientation_range: tuple[float, float],
) -> list[dict[str, object]]:
    cases: list[dict[str, object]] = [
        {
            "id": "official_00",
            "start": [0.0, 0.0, 0.0],
            "goal": official_goal,
        }
    ]
    for index in range(1, COUNT):
        radius = rng.uniform(*radius_range)
        direction = rng.uniform(-math.pi, math.pi)
        cases.append(
            {
                "id": f"local_{index:02d}",
                "start": [0.0, 0.0, 0.0],
                "goal": [
                    radius * math.cos(direction),
                    radius * math.sin(direction),
                    rng.uniform(*orientation_range),
                ],
            }
        )
    return cases


def cart_cases(rng: random.Random) -> list[dict[str, object]]:
    cases: list[dict[str, object]] = [
        {
            "id": "official_00",
            "start": [0.0, 0.0, 0.0, 0.0],
            "goal": [1.0, 0.0, 0.0, 0.0],
        }
    ]
    for index in range(1, COUNT):
        start_center = rng.uniform(-1.0, 1.0)
        start_relative = rng.uniform(-0.4, 0.4)
        goal_center = start_center + rng.uniform(-2.0, 2.0)
        goal_relative = rng.uniform(-0.4, 0.4)
        cases.append(
            {
                "id": f"local_{index:02d}",
                "start": [
                    start_center + 0.5 * start_relative,
                    start_center - 0.5 * start_relative,
                    0.0,
                    0.0,
                ],
                "goal": [
                    goal_center + 0.5 * goal_relative,
                    goal_center - 0.5 * goal_relative,
                    0.0,
                    0.0,
                ],
            }
        )
    return cases


def generate() -> dict[str, object]:
    rng = random.Random(SEED)
    return {
        "schema_version": 1,
        "seed": SEED,
        "description": (
            "Predeclared local suite because the IMPACT paper/repository does "
            "not publish its original random instances or sampling distribution."
        ),
        "problems": {
            "push_box": {
                "sampling": {
                    "fixed_start": [0.0, 0.0, 0.0],
                    "goal_radius": [0.25, 2.0],
                    "goal_direction": [-math.pi, math.pi],
                    "goal_orientation": [-1.5, 1.5],
                },
                "cases": pose_cases(
                    rng, [0.1, 0.1, 1.0], (0.25, 2.0), (-1.5, 1.5)
                ),
            },
            "push_t": {
                "sampling": {
                    "fixed_start": [0.0, 0.0, 0.0],
                    "goal_radius": [0.1, 0.75],
                    "goal_direction": [-math.pi, math.pi],
                    "goal_orientation": [-1.5, 1.5],
                },
                "cases": pose_cases(
                    rng, [0.05, 0.05, 1.5708], (0.1, 0.75), (-1.5, 1.5)
                ),
            },
            "cart_transport": {
                "sampling": {
                    "zero_start_and_goal_velocities": True,
                    "start_center": [-1.0, 1.0],
                    "start_payload_cart_offset": [-0.4, 0.4],
                    "goal_center_displacement": [-2.0, 2.0],
                    "goal_payload_cart_offset": [-0.4, 0.4],
                },
                "cases": cart_cases(rng),
            },
        },
    }


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--output",
        type=Path,
        default=Path(__file__).with_name("impact_cases.json"),
    )
    args = parser.parse_args()
    args.output.write_text(
        json.dumps(generate(), indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
