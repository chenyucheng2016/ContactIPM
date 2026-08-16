# Local contact benchmark summary

Source artifact: `benchmarks/contact_ipm/results/2026-07-28_generic_preconditioner_exact_1x.json`

All execution times in this report were measured locally; no runtime reported in the CRISP paper is used.

ContactIPM source snapshot: `contactipm-src-08`

This run is correctness/development evidence, not publication-ready timing evidence, because:

- schema version predates contiguous randomized pairing
- at least one warmup is required
- fewer than 20 measured repetitions
- CPU affinity was not explicitly pinned

## Feasibility and task quality

| Solver | Problem | Qualified trials | Audited success | Task success | Worst equality | Worst side | Worst MPCC | Worst position | Worst angle | Worst velocity |
|---|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| ContactIPM | Cartpole with Soft Walls | 1/1 | 1/1 | 1/1 | 5.55e-16 | 0 | 1.18e-06 | 0.00241 | 0.0298 | 0.00479 |
| CRISP | Cartpole with Soft Walls | 1/1 | 1/1 | 1/1 | 9.04e-08 | 5.58e-15 | 9.8e-15 | 0.00238 | 0.0298 | 0.0048 |
| ContactIPM | Push Box | 1/1 | 1/1 | 1/1 | 2.91e-16 | 0 | 5.67e-07 | 0.0335 | 0.0135 | 0 |
| CRISP | Push Box | 1/1 | 1/1 | 1/1 | 2.79e-07 | 9.75e-14 | 6.09e-13 | 0.0505 | 0.0171 | 0 |
| ContactIPM | Transport | 1/1 | 1/1 | 1/1 | 1.27e-08 | 0 | 9.5e-07 | 3.38e-05 | 0 | 0.462 |
| CRISP | Transport | 1/1 | 1/1 | 1/1 | 1.17e-14 | 3.96e-15 | 2.74e-12 | 3.55e-05 | 0 | 0.465 |
| ContactIPM | Push T | 1/1 | 50/50 | 50/50 | 7.64e-06 | 8.28e-10 | 1.16e-06 | 0.0575 | 0.0215 | 0 |
| CRISP | Push T | 0/1 | 27/50 | 43/50 | 2 | 4.69e-11 | 0.000448 | 0.388 | 0.267 | 0 |

## Locally measured process wall time

Values are median [Q1, Q3] and P90 in seconds over all measured processes. A paired speedup is reported only when both trajectories pass the common audit.

| Problem | ContactIPM | CRISP | Qualified pairs | CRISP / ContactIPM paired speedup (95% CI) | Ready |
|---|---:|---:|---:|---:|---:|
| Cartpole with Soft Walls | 0.0237 [0.0237, 0.0237], p90 0.0237 | 0.1961 [0.1961, 0.1961], p90 0.1961 | 1/1 | 8.258x [8.258, 8.258] | no |
| Push Box | 0.1814 [0.1814, 0.1814], p90 0.1814 | 0.4858 [0.4858, 0.4858], p90 0.4858 | 1/1 | 2.678x [2.678, 2.678] | no |
| Transport | 0.0748 [0.0748, 0.0748], p90 0.0748 | 0.3626 [0.3626, 0.3626], p90 0.3626 | 1/1 | 4.847x [4.847, 4.847] | no |
| Push T | 59.8183 [59.8183, 59.8183], p90 59.8183 | 366.8725 [366.8725, 366.8725], p90 366.8725 | 0/1 | n/a | no |

## Objective decomposition

Objective values are shown after feasibility and task metrics; they are not used as a standalone success criterion.

| Solver | Problem | Objective per instance (median) | Component medians |
|---|---|---:|---|
| ContactIPM | Cartpole with Soft Walls | 2.72019 | control_effort=2.6208, terminal_tracking=0.0993929 |
| CRISP | Cartpole with Soft Walls | 2.72017 | control_effort=2.62123, terminal_tracking=0.0989437 |
| ContactIPM | Push Box | 18.7871 | force_effort=18.6568, terminal_tracking=0.130309 |
| CRISP | Push Box | 6.93078 | force_effort=6.64645, terminal_tracking=0.284329 |
| ContactIPM | Transport | 2.96411 | control_effort=0.829089, terminal_tracking=2.13502 |
| CRISP | Transport | 2.98346 | control_effort=0.823621, terminal_tracking=2.15983 |
| ContactIPM | Push T | 20.9776 | force_effort=6.33779, running_tracking=13.1746, terminal_tracking=0.0118006 |
| CRISP | Push T | 7.23128 | force_effort=2.95576, running_tracking=4.90239, terminal_tracking=0.0837748 |
