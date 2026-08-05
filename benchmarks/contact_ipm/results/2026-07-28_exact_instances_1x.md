# Exact-instance audit milestone - 2026-07-28

This is a correctness and reproducibility milestone, not a
publication-quality timing study. All execution times were measured locally by
the common runner; no execution time reported in the CRISP paper was used.
The JSON records base revision `cf7fe92` because the run preceded its commit;
implementation commit `075fadb` captures the measured source tree exactly.

## What is now identical

- The four problems come from `benchmarks/CRISP/src/examples`.
- `source_cases.json` freezes the nodes, time steps, physical parameters,
  targets, initial-condition generators, variable layouts, and audit gates.
- Cartpole uses the exact CRISP initial-guess file, verified by SHA-256
  `470c6b28b09c87200498f1ef8895a11d783df1d4f0646cb1ef749360e8003e5d`.
- Push Box and Transport use the source zero guesses and source initial/target
  states. Push T uses the source zero guess and all 50 source-generated cases.
- CRISP source instrumentation only selects the frozen cartpole guess and emits
  raw trajectories. It does not alter objectives, constraints, solver
  parameters, or solve calls.
- The same independent auditor evaluates both solvers. Neither solver's printed
  status or residual is accepted as correctness evidence.

The frozen manifest hash for this run is
`f712a425b48c6246238253cc45a53f69d927a8d0ede86f32950c248ef5f1231e`.
The JSON artifact also records the CRISP revision and hashes of all four
instrumented source files.

## Common raw audit

The gates are maximum equality defect `<= 1e-5`, side violation `<= 1e-8`,
physical complementarity product `<= 1e-5`, plus each source task's terminal
tolerances.

| Problem | ContactIPM | CRISP | ContactIPM objective | CRISP objective |
|---|---:|---:|---:|---:|
| Cartpole with Soft Walls | 1/1 | 1/1 | 2.720176 | 2.720172 |
| Push Box | 1/1 | 1/1 | 22.199436 | 6.930783 |
| Transport | 1/1 | 1/1 | 2.964113 | 2.983455 |
| Push T | 50/50 | 27/50 | 1171.890 total | 846.242 total |

For Push T, the total-objective columns are not a fair quality comparison
because CRISP fails 23 cases. On the 27 cases where both trajectories pass all
gates, ContactIPM's summed objective is 366.268 and CRISP's is 346.985.
Therefore this milestone establishes better ContactIPM robustness, but not
universal objective dominance.

Worst audited residuals:

| Solver/problem | Equality | Side violation | Physical product |
|---|---:|---:|---:|
| ContactIPM Cartpole | 5.00e-16 | 0 | 1.00e-6 |
| CRISP Cartpole | 9.04e-8 | 5.58e-15 | 9.80e-15 |
| ContactIPM Push Box | 4.58e-16 | 0 | 9.80e-7 |
| CRISP Push Box | 2.79e-7 | 9.75e-14 | 6.09e-13 |
| ContactIPM Transport | 1.26e-8 | 0 | 9.50e-7 |
| CRISP Transport | 1.17e-14 | 3.96e-15 | 2.74e-12 |
| ContactIPM Push T | 6.58e-6 | 8.50e-9 | 4.56e-6 |
| CRISP Push T | 2.00 | 4.69e-11 | 4.48e-4 |

All eight solver processes returned zero. The runner nevertheless returned
nonzero because the CRISP Push T trajectory audit failed, demonstrating that
process status alone is insufficient.

## Timing scope

The JSON contains one locally measured wall-clock observation per
solver/problem pair. A single observation is useful for smoke testing only and
must not be used as a paper statistic. The next timing study should run from
clean pinned commits, pin CPU affinity and frequency policy, perform warmups,
use randomized interleaving, collect enough repetitions for confidence
intervals, and compare only trials passing the common audit.

## Verification

- Both instrumented suites compiled in WSL Release builds.
- Source-case parity check passed all four benchmarks.
- Python syntax checks passed for the benchmark tools.
- All 10 CTest regressions passed.
- Machine-readable artifact:
  [2026-07-28_exact_instances_1x.json](2026-07-28_exact_instances_1x.json).
