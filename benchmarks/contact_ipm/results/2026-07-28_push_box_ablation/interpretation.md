# Push Box solver-component ablation interpretation

## Provenance and protocol

- ContactIPM source snapshot: `contactipm-src-14`
- ContactIPM tracked worktree: clean in every raw artifact
- Cases: the 25 frozen Push Box initial-state/target cases in
  `validation_cases.json`
- Repetitions: one per case and configuration
- Evaluation: the same independent trajectory audit used for both ContactIPM
  and CRISP

This experiment measures robustness across cases. Its process times are
diagnostic only, not publication timing statistics.

## Results

| Configuration | Audited success | Feasible | Task success | Exit 0 |
|---|---:|---:|---:|---:|
| Full solver | 21/25 | 22/25 | 22/25 | 21/25 |
| No MPCC recovery | 3/25 | 4/25 | 11/25 | 2/25 |
| No preconditioner | 18/25 | 19/25 | 21/25 | 18/25 |
| Gauss-Newton primary solve | 24/25 | 25/25 | 24/25 | 24/25 |
| Recovery `mu=0.01` | 21/25 | 22/25 | 21/25 | 20/25 |
| Recovery `mu=1.0` | 25/25 | 25/25 | 25/25 | 24/25 |

The complete table, including descriptive wall times and median objectives, is
in [summary.md](summary.md).

## Interpretation

1. Generic MPCC recovery is essential on this suite. Removing it reduces
   audited success from 21/25 to 3/25, mainly through loss of physical
   feasibility.
2. The generic preconditioner helps robustness, adding three audited successes
   relative to disabling it.
3. Exact primary curvature is not beneficial on this Push Box sweep.
   Gauss-Newton raises audited success from 21/25 to 24/25. This result alone is
   not sufficient to change the global solver default; the candidate must be
   checked on Cartpole, Transport, and Push T.
4. Recovery is sensitive to its conservative barrier. Raising recovery
   `mu` from the default 0.1 to 1.0 gives 25/25 audited success; lowering it to
   0.01 gives no audited-success improvement.
5. The median objective among audited successes stays in the narrow range
   18.58--18.79. The large success-rate changes therefore reflect
   feasibility/task robustness rather than a change in the scalar objective.

One `mu=1.0` case exits nonzero after reaching the iteration limit, but the
independent audit accepts the trajectory: equality error `2.36e-16`, zero side
violation, physical complementarity `6.69e-7`, translation error `0.03443`,
and angular error `0.01319`. Both the audited 25/25 result and the
solver-reported 24/25 result must be retained. The status mismatch is a
convergence-termination diagnostic, not evidence that the physical trajectory
failed.

## Next validation decision

The next experiment should cross-validate the two promising changes
(Gauss-Newton primary curvature and recovery `mu=1.0`) on all four benchmarks
before adopting either as a generic default. A combined configuration should
also be tested to determine whether their gains are additive.