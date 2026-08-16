import hashlib
import tempfile
import unittest
from pathlib import Path

from source_provenance import describe_source


class SourceProvenanceTest(unittest.TestCase):
    def test_snapshot_is_content_addressed_and_excludes_results(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            (root / "include" / "nmpc").mkdir(parents=True)
            (root / "benchmarks" / "contact_ipm" / "results").mkdir(
                parents=True
            )
            (root / "CMakeLists.txt").write_text("project(example)\n")
            header = root / "include" / "nmpc" / "solver.hpp"
            header.write_bytes(b"first\n")
            (root / "benchmarks" / "contact_ipm" / "runner.py").write_text(
                "pass\n"
            )
            (root / "benchmarks" / "contact_ipm" / "results" / "run.json").write_text(
                "{}\n"
            )

            first = describe_source(root)
            self.assertEqual(len(first["snapshot"]), len("sha256:") + 64)
            self.assertNotIn(
                "benchmarks/contact_ipm/results/run.json", first["manifest"]
            )

            header.write_bytes(b"second\n")
            second = describe_source(root)
            self.assertNotEqual(first["snapshot"], second["snapshot"])
            self.assertEqual(
                second["manifest"]["include/nmpc/solver.hpp"],
                hashlib.sha256(b"second\n").hexdigest(),
            )


if __name__ == "__main__":
    unittest.main()
