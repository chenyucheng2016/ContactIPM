#!/usr/bin/env python3
"""Verify the provenance and fairness invariants of an IMPACT comparison."""

from __future__ import annotations

import argparse
import hashlib
import json
import math
import subprocess
from pathlib import Path
from typing import Any


SOURCE_PATHS = {
    "push_box": Path("experiments/box/box_impact_multiple.cpp"),
    "push_t": Path("experiments/push_t/push_t_impact_multiple.cpp"),
    "cart_transport": Path(
        "experiments/cart_transporter/cart_transporter_impact_multiple.cpp"
    ),
}


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def git(repository: Path, *arguments: str) -> str:
    completed = subprocess.run(
        ["git", "-C", str(repository), *arguments],
        check=False,
        text=True,
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
    )
    if completed.returncode != 0:
        raise RuntimeError(completed.stderr.strip())
    return completed.stdout.strip()


def verify_repository(repository: Path, manifest: dict[str, Any]) -> None:
    expected_commit = manifest["competitor"]["commit"]
    actual_commit = git(repository, "rev-parse", "HEAD")
    assert actual_commit == expected_commit, (actual_commit, expected_commit)

    numstat = git(repository, "diff", "--ignore-space-at-eol", "--numstat")
    substantive = []
    for line in numstat.splitlines():
        added, deleted, path = line.split("\t", maxsplit=2)
        if added != "0" or deleted != "0":
            substantive.append(path)
    assert not substantive, f"substantive IMPACT edits: {substantive}"

    for problem, relative_path in SOURCE_PATHS.items():
        expected = manifest["problems"][problem]["source_sha256"]
        actual = sha256(repository / relative_path)
        assert actual == expected, (problem, actual, expected)

    multiple_shooting = (
        repository / "impact_solver/src/multiple_shooting.cpp"
    ).read_text(encoding="utf-8")
    assert "Eigen::MatrixXd::Zero(nu, config.horizon)" in multiple_shooting
    assert "(1.0 - a) * config.x_0 + a * config.x_goal" in multiple_shooting
    assert "x_init.col(k) = config.x_0" in multiple_shooting
    cart_source = (
        repository
        / "experiments/cart_transporter/cart_transporter_impact_multiple.cpp"
    ).read_text(encoding="utf-8")
    assert "config.use_constant_state_init = true" in cart_source


def verify_artifact(path: Path, timing: bool) -> None:
    artifact = json.loads(path.read_text(encoding="utf-8"))
    assert artifact["paper_reported_times_used"] is False
    assert artifact["run_protocol"]["adjacent_randomized_pairs"] is True
    if timing:
        protocol = artifact["run_protocol"]
        assert protocol["warmups"] >= 1
        assert protocol["repetitions"] >= 20
        assert protocol["threads"] == 1
        assert len(protocol["cpu_affinity"]) == 1
        assert all(pair["pair_eligible"] for pair in artifact["pairs"])

    for pair in artifact["pairs"]:
        contact = pair["contactipm"]
        assert contact["initialization_state_error"] <= 1e-12
        assert contact["initialization_control_max"] <= 1e-12
        for solver in ("contactipm", "impact"):
            result = pair[solver]
            if result["reported_objective"] is None or result["audit"] is None:
                continue
            reported = float(result["reported_objective"])
            audited = float(result["audit"]["objective"])
            tolerance = 1e-10 if solver == "contactipm" else 1e-5
            assert math.isclose(
                reported, audited, rel_tol=tolerance, abs_tol=tolerance
            ), (pair["problem"], pair["case_id"], solver, reported, audited)


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--impact-repository", type=Path, required=True)
    parser.add_argument(
        "--manifest",
        type=Path,
        default=Path(__file__).with_name("impact_source_cases.json"),
    )
    parser.add_argument("--robustness", type=Path, required=True)
    parser.add_argument("--timing", type=Path, required=True)
    args = parser.parse_args()

    manifest = json.loads(args.manifest.read_text(encoding="utf-8"))
    verify_repository(args.impact_repository, manifest)
    verify_artifact(args.robustness, timing=False)
    verify_artifact(args.timing, timing=True)
    print("IMPACT comparison verification passed")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
