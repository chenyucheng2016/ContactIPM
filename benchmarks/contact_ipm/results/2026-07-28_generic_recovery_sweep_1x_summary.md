# Local contact benchmark summary

Source artifact: `benchmarks/contact_ipm/results/2026-07-28_generic_recovery_sweep_1x.json`

All execution times in this report were measured locally; no runtime reported in the CRISP paper is used.

ContactIPM source snapshot: `contactipm-src-02`

This run is correctness/development evidence, not publication-ready timing evidence, because:

- multi-instance sweeps are robustness evidence, not repeated timing
- at least one warmup is required
- fewer than 20 measured repetitions

## Feasibility and task quality

| Solver | Problem | Qualified trials | Audited success | Task success | Worst equality | Worst side | Worst MPCC | Worst position | Worst angle | Worst velocity |
|---|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| ContactIPM | Cartpole with Soft Walls | 15/15 | 15/15 | 15/15 | 5e-16 | 0 | 9.98e-07 | 0.00629 | 0.0427 | 0.00531 |
| CRISP | Cartpole with Soft Walls | 15/15 | 15/15 | 15/15 | 1.66e-07 | 2.46e-14 | 5.21e-14 | 0.00265 | 0.0325 | 0.00531 |
| ContactIPM | Push Box | 24/25 | 24/25 | 24/25 | 4.44e-16 | 0 | 1e-06 | 1.18 | 0.472 | 0 |
| CRISP | Push Box | 19/25 | 19/25 | 19/25 | 4.51e-07 | 4.71e-13 | 4.09e-12 | 0.358 | 0.147 | 0 |
| ContactIPM | Transport | 8/15 | 8/15 | 8/15 | 2.54e-07 | 0 | 1e-06 | 0.000252 | 0 | 1.97 |
| CRISP | Transport | 8/15 | 8/15 | 8/15 | 3.28e-13 | 3.21e-13 | 2.74e-12 | 0.063 | 0 | 1.97 |

## Paired robustness outcomes

Each initial-state/target pair is counted once using the common trajectory audit.

| Problem | Both succeed | ContactIPM only | CRISP only | Neither |
|---|---:|---:|---:|---:|
| Cartpole with Soft Walls | 15 | 0 | 0 | 0 |
| Push Box | 19 | 5 | 0 | 1 |
| Transport | 8 | 0 | 0 | 7 |

## Locally measured process wall time

Values are median [Q1, Q3] and P90 in seconds over all measured processes. A paired speedup is reported only when both trajectories pass the common audit.

| Problem | ContactIPM | CRISP | Qualified pairs | CRISP / ContactIPM paired speedup (95% CI) | Ready |
|---|---:|---:|---:|---:|---:|
| Cartpole with Soft Walls | 0.0216 [0.0194, 0.0347], p90 0.1039 | 0.1717 [0.1327, 0.1921], p90 0.1986 | 15/15 | 6.460x [4.445, 9.623] | no |
| Push Box | 0.2988 [0.1835, 1.4074], p90 1.7018 | 0.4231 [0.3154, 0.4801], p90 0.5466 | 19/25 | 1.585x [0.341, 2.575] | no |
| Transport | 0.1420 [0.1139, 0.2696], p90 0.3335 | 0.4061 [0.1511, 0.6001], p90 0.7272 | 8/15 | 0.822x [0.337, 1.055] | no |

## Objective decomposition

Objective values are shown after feasibility and task metrics; they are not used as a standalone success criterion.

| Solver | Problem | Objective per instance (median) | Component medians |
|---|---|---:|---|
| ContactIPM | Cartpole with Soft Walls | 2.72017 | control_effort=2.62127, terminal_tracking=0.0989025 |
| CRISP | Cartpole with Soft Walls | 2.72017 | control_effort=2.62123, terminal_tracking=0.0989437 |
| ContactIPM | Push Box | 18.6819 | force_effort=18.5519, terminal_tracking=0.13047 |
| CRISP | Push Box | 8.19694 | force_effort=7.87744, terminal_tracking=0.241992 |
| ContactIPM | Transport | 0.792137 | control_effort=0.72255, terminal_tracking=0.0695878 |
| CRISP | Transport | 0.793676 | control_effort=0.72314, terminal_tracking=0.0705366 |
