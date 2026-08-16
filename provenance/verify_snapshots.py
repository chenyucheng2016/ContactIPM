#!/usr/bin/env python3
"""Verify the identity-neutral source snapshots and current SRBD source set."""

from __future__ import annotations

import base64
import hashlib
import json
import subprocess
import tempfile
from pathlib import Path


PROVENANCE_DIR = Path(__file__).resolve().parent
REPOSITORY_ROOT = PROVENANCE_DIR.parent


def sha256(data: bytes) -> str:
    return hashlib.sha256(data).hexdigest()


def run_git(*args: str, binary: bool = False) -> bytes | str:
    result = subprocess.run(
        ["git", *args], check=True, capture_output=True
    )
    return result.stdout if binary else result.stdout.decode("utf-8")


def read_manifest(path: Path) -> dict[str, str]:
    entries: dict[str, str] = {}
    for line_number, line in enumerate(path.read_text(encoding="utf-8").splitlines(), 1):
        digest, separator, relative_path = line.partition("  ")
        if separator != "  " or len(digest) != 64 or not relative_path:
            raise RuntimeError(f"Malformed manifest line {path}:{line_number}")
        entries[relative_path] = digest
    return entries


def verify_snapshot(
    repository: Path, label: str, manifest_path: Path
) -> int:
    expected = read_manifest(manifest_path)
    tree_output = run_git(
        "-C", str(repository), "ls-tree", "-r", "-z", "--name-only", label,
        binary=True,
    )
    actual_paths = {
        item.decode("utf-8") for item in tree_output.split(b"\0") if item
    }
    if actual_paths != set(expected):
        missing = sorted(set(expected) - actual_paths)
        extra = sorted(actual_paths - set(expected))
        raise RuntimeError(f"{label} file-set mismatch: missing={missing}, extra={extra}")

    for relative_path, digest in expected.items():
        blob = run_git(
            "-C", str(repository), "cat-file", "blob",
            f"{label}:{relative_path}", binary=True,
        )
        if sha256(blob) != digest:
            raise RuntimeError(f"{label} hash mismatch: {relative_path}")
    return len(expected)


def main() -> None:
    index = json.loads((PROVENANCE_DIR / "source-snapshots.json").read_text())
    armored_path = PROVENANCE_DIR / index["bundle_base64"]
    encoded = "".join(armored_path.read_text(encoding="ascii").split())
    bundle = base64.b64decode(encoded, validate=True)
    if sha256(bundle) != index["decoded_bundle_sha256"]:
        raise RuntimeError("Decoded source bundle SHA-256 mismatch")

    snapshot_file_count = 0
    with tempfile.TemporaryDirectory(prefix="contactipm-source-verify-") as temp:
        temp_path = Path(temp)
        bundle_path = temp_path / "source-snapshots.bundle"
        repository = temp_path / "repository.git"
        bundle_path.write_bytes(bundle)
        run_git("init", "--bare", "--quiet", str(repository))
        run_git(
            "-C",
            str(repository),
            "fetch",
            "--quiet",
            str(bundle_path),
            "refs/tags/*:refs/tags/*",
        )

        actual_labels = set(
            run_git(
                "-C", str(repository), "for-each-ref",
                "--format=%(refname:short)", "refs/tags",
            ).splitlines()
        )
        if actual_labels != set(index["snapshots"]):
            raise RuntimeError("Decoded source bundle tag set mismatch")

        for label, entry in sorted(index["snapshots"].items()):
            manifest_path = PROVENANCE_DIR / entry["manifest"]
            if sha256(manifest_path.read_bytes()) != entry["manifest_sha256"]:
                raise RuntimeError(f"Manifest SHA-256 mismatch: {label}")
            snapshot_file_count += verify_snapshot(repository, label, manifest_path)

    srbd_manifest_path = PROVENANCE_DIR / index["current_srbd_source_manifest"]
    if sha256(srbd_manifest_path.read_bytes()) != index["current_srbd_source_manifest_sha256"]:
        raise RuntimeError("Current SRBD manifest SHA-256 mismatch")
    srbd_entries = read_manifest(srbd_manifest_path)
    for relative_path, digest in srbd_entries.items():
        data = (REPOSITORY_ROOT / relative_path).read_bytes().replace(b"\r\n", b"\n")
        if sha256(data) != digest:
            raise RuntimeError(f"Current SRBD source hash mismatch: {relative_path}")

    print(
        f"Verified {len(index['snapshots'])} source snapshots "
        f"({snapshot_file_count} files) and {len(srbd_entries)} current SRBD files."
    )


if __name__ == "__main__":
    main()
