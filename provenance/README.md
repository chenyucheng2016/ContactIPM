# Identity-neutral source provenance

Historical result files name the exact ContactIPM source state as
`contactipm-src-01` through `contactipm-src-17`. These labels replace
repository-specific object identifiers without changing any measurement,
configuration, seed, command, third-party revision, or existing SHA-256 hash.

The source-only snapshots are stored in
`contactipm-source-snapshots.bundle`. The bundle contains 17 independent,
single-commit refs with generic metadata and no original repository history.
Generated results, prose, media, and repository metadata are excluded; source,
build files, benchmark inputs, runners, and tests are retained. Per-file SHA-256
manifests and the bundle digest are indexed by `source-snapshots.json`.

To inspect and verify a snapshot, for example `contactipm-src-02`:

```bash
git clone provenance/contactipm-source-snapshots.bundle /tmp/contactipm-sources
git -C /tmp/contactipm-sources checkout contactipm-src-02
(cd /tmp/contactipm-sources && sha256sum -c \
  "$OLDPWD/provenance/manifests/contactipm-src-02.sha256")
```

The CRISP, IMPACT, acados, and MuJoCo Menagerie object identifiers remain in
the repository because they pin independent third-party dependencies. Current
SRBD code is present directly in the anonymous source tree and does not require
a historical snapshot. Its paper-run source set is recorded in
`srbd-closed-loop-source.sha256`.
