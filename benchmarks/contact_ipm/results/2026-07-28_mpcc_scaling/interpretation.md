# Duration-fixed MPCC scaling interpretation

## Scope and protocol

This study varies compiled horizon length while keeping total physical
duration, dynamics, target, and per-stage contact geometry fixed. Push Box uses
10 complementarity pairs per interval over 1.98 s; Push T segment 8 uses 43
pairs per interval over 2.45 s. The time step is therefore recomputed as
duration divided by the number of intervals for both solvers.

Each point uses one discarded warmup and five randomized adjacent solver
pairs, one pinned logical CPU, and one thread. Build time and CRISP
model-generation time are excluded. A measured process is eligible only when
the solver exits successfully and the same independent trajectory audit passes
dynamics, side feasibility, physical complementarity, and terminal task
tolerances. No execution time reported by either paper is used.

The downloaded per-stage objective weights are unchanged, so both solvers
receive the same discrete objective at each node count. The objective is not
rescaled as a continuous-time quadrature across node counts. This isolates
solver scaling on the downloaded formulations; it is not an
objective-discretization convergence study.

## Results

| Problem | Nodes | Complementarity pairs | ContactIPM eligible | CRISP eligible | ContactIPM median (s) | CRISP median (s) | Paired CRISP / ContactIPM |
|---|---:|---:|---:|---:|---:|---:|---:|
| Push Box | 25 | 240 | 0/5 | 5/5 | n/a | 0.215 | n/a |
| Push Box | 50 | 490 | 0/5 | 5/5 | n/a | 0.309 | n/a |
| Push Box | 75 | 740 | 5/5 | 5/5 | 0.135 | 0.235 | 1.59x |
| Push Box | 100 | 990 | 5/5 | 5/5 | 0.243 | 0.554 | 2.25x |
| Push Box | 150 | 1490 | 5/5 | 0/5 | 2.531 | n/a | n/a |
| Push T segment 8 | 25 | 1032 | 5/5 | 5/5 | 0.315 | 1.101 | 3.84x |
| Push T segment 8 | 35 | 1462 | 5/5 | 5/5 | 0.361 | 3.897 | 10.9x |
| Push T segment 8 | 50 | 2107 | 5/5 | 5/5 | 0.741 | 10.49 | 13.9x |
| Push T segment 8 | 75 | 3182 | 5/5 | 0/5 | 1.099 | n/a | n/a |
| Push T segment 8 | 100 | 4257 | 5/5 | 0/5 | 3.026 | n/a | n/a |

The scaling results separate eligible median time from
converged-and-audited fraction. Raw Push Box and Push T artifacts are in the
sibling `2026-07-28_scaling_push_box_5x` and
`2026-07-28_scaling_push_t_5x` directories.

## Feasibility interpretation

Push Box ContactIPM fails at 25 nodes on both task and physical
complementarity, and at 50 nodes on physical complementarity despite meeting
the task threshold. Both methods pass at 75 and 100 nodes, where ContactIPM is
1.59x and 2.25x faster in paired eligible runs. At 150 nodes, ContactIPM remains
physically feasible and task successful; CRISP remains physically feasible but
misses the translation threshold with 0.1155 m error.

Both methods pass Push T at 25, 35, and 50 nodes. ContactIPM's paired speedup
increases from 3.84x to 13.9x over 1032 to 2107 complementarity pairs.
ContactIPM remains feasible and task successful at 75 and 100 nodes. CRISP is
physically feasible at both sizes but misses the task gate, with terminal
position errors of 0.1437 m and 0.1751 m.

These failures are retained as outcomes and their process times are not used
in paired speedup claims. Solver exit status alone is not treated as success.

## Provenance and claim limits

- Push Box records clean ContactIPM revision `97833fe`; Push T records clean
  revision `a230e13`, which fixes scaled Push T audit routing without changing
  solver binaries.
- Both record CRISP revision `d429c06` and the exact tracked instrumentation
  patch hash `93679cc1d7c6bbe7ac9973f5bc14bf3aba6c26e18f804757ecc2cf37e581cc48`.
- All 10 raw artifacts contain source and executable hashes. The corrected
  Push T artifacts contain zero trajectory-audit errors.
- This scales horizon length and total complementarity count while keeping the
  number of contact pairs per stage fixed. It does not independently scale the
  spatial contact-candidate set.
- Five repetitions establish the development trend but are insufficient for
  final paper timing statistics. Publication results require at least 20
  interleaved pairs at the predeclared node counts, with IQR, P90, and paired
  confidence intervals.
