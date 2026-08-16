# Identity-neutral source provenance

Historical result files name the exact ContactIPM source state as
`contactipm-src-01` through `contactipm-src-17`. These labels replace
repository-specific object identifiers without changing any measurement,
configuration, seed, command, third-party revision, or existing SHA-256 hash.

The source-only snapshots are stored as the ASCII-armored
`contactipm-source-snapshots.bundle.b64`. Decode it to recover a Git bundle
containing 17 independent, single-commit refs with generic metadata and no
original repository history. ASCII armoring keeps the binary bundle intact
when an anonymous archive transports repository files as text.
Generated results, prose, media, and repository metadata are excluded; source,
build files, benchmark inputs, runners, and tests are retained. Per-file SHA-256
manifests and the decoded bundle digest are indexed by `source-snapshots.json`.

Verify every snapshot and the current SRBD source set with:

```bash
python3 provenance/verify_snapshots.py
```

To inspect and verify a snapshot, for example `contactipm-src-02`:

```bash
base64 --decode provenance/contactipm-source-snapshots.bundle.b64 \
  > /tmp/contactipm-source-snapshots.bundle
git -c core.autocrlf=false clone \
  /tmp/contactipm-source-snapshots.bundle /tmp/contactipm-sources
git -C /tmp/contactipm-sources checkout contactipm-src-02
(cd /tmp/contactipm-sources && sha256sum -c \
  "$OLDPWD/provenance/manifests/contactipm-src-02.sha256")
```

The CRISP, IMPACT, acados, and MuJoCo Menagerie object identifiers remain in
the repository because they pin independent third-party dependencies. Current
SRBD code is present directly in the anonymous source tree and does not require
a historical snapshot. Its paper-run source set is recorded in
`srbd-closed-loop-source.sha256`.
