"""Backward-compatible wrapper for the former CRISP-only Push T audit."""

from pathlib import Path

from trajectory_audit import audit_push_t as _audit_push_t


def audit_push_t(directory: Path, started_at: float) -> dict:
    result = _audit_push_t(directory, "crisp", started_at)
    result["successful_segments"] = result["successful_instances"]
    result["attempted_segments"] = result["attempted_instances"]
    result["segments"] = result["instances"]
    return result