# Push Box generic-recovery ablation interpretation

## Provenance and protocol

- ContactIPM source snapshot: `contactipm-src-02`
- ContactIPM tracked worktree: clean in every raw artifact
- Cases: 25 frozen Push Box initial-state/target pairs from `validation_cases.json`
- Repetitions: one per case and configuration
- Evaluation: the same independent trajectory audit used for both solvers

This experiment measures robustness across cases. Its process times are
diagnostic only, not publication timing statistics.

## Results

| Configuration | Audited success | Feasible | Task success | Exit 0 |
|---|---:|---:|---:|---:|
| Full generic solver | 24/25 | 25/25 | 24/25 | 24/25 |
| No MPCC recovery | 3/25 | 4/25 | 11/25 | 2/25 |
| No preconditioner | 21/25 | 22/25 | 23/25 | 20/25 |
| Gauss-Newton primary solve | 25/25 | 25/25 | 25/25 | 25/25 |
| Recovery `mu=0.01` | 23/25 | 24/25 | 24/25 | 21/25 |
| Recovery `mu=1.0` | 23/25 | 23/25 | 24/25 | 22/25 |

The complete descriptive table is in [summary.md](summary.md).

## Interpretation

1. Generic MPCC recovery is essential. Removing it reduces audited success
   from 24/25 to 3/25 and physical feasibility from 25/25 to 4/25.
2. The generic preconditioner contributes robustness: disabling it loses three
   audited successes and three feasible trajectories.
3. Gauss-Newton primary curvature is best on this Push Box sweep, reaching
   25/25. It is not adopted as a generic default because its all-benchmark
   cross-check produced 49/50 solver-success statuses on Push T, even though
   all 50 trajectories passed the independent audit. Exact curvature retains
   50/50 solver and audit success there.
4. Moving the recovery barrier in either tested direction is worse than the
   default: both `mu=0.01` and `mu=1.0` reach only 23/25 audited successes.
5. Median objective on audited successes remains between 18.58 and 18.79.
   The large success-rate changes are feasibility/task robustness changes, not
   evidence from the scalar objective alone.

## Solver-policy decision

Keep exact primary curvature and the five-phase generic recovery portfolio as
the problem-independent default. Do not add a Push-Box-specific curvature
branch. Report Gauss-Newton as an ablation and revisit automatic curvature
selection only if a problem-independent diagnostic can be validated across all
benchmarks.
