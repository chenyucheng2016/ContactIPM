"""Git-independent source provenance for ContactIPM benchmark artifacts."""

from __future__ import annotations

import hashlib
from pathlib import Path


def describe_source(
    repository: Path, extra_roots: tuple[Path, ...] = ()
) -> dict[str, object]:
    files = [repository / "CMakeLists.txt"]
    for root in (
        repository / "include" / "nmpc",
        repository / "benchmarks" / "contact_ipm",
        *extra_roots,
    ):
        files.extend(
            path
            for path in root.rglob("*")
            if path.is_file()
            and "results" not in path.relative_to(repository).parts
            and path.name != "README.md"
            and path.suffix not in {".pyc", ".pyo"}
        )

    manifest = {
        path.relative_to(repository).as_posix(): hashlib.sha256(
            path.read_bytes()
        ).hexdigest()
        for path in sorted(set(files))
    }
    payload = "".join(
        f"{digest}  {path}\n" for path, digest in sorted(manifest.items())
    ).encode("utf-8")
    return {
        "snapshot": f"sha256:{hashlib.sha256(payload).hexdigest()}",
        "manifest": manifest,
    }
