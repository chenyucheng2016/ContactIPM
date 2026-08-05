# Local contact benchmark summary

Source artifact: `benchmarks/contact_ipm/results/2026-07-28_initial_target_sweep_1x.json`

All execution times in this report were measured locally; no runtime reported in the CRISP paper is used.

ContactIPM revision: `73a4a68dd39be639dcf9b0c85123523aaa9c80d0`

This run is correctness/development evidence, not publication-ready timing evidence, because:

- multi-instance sweeps are robustness evidence, not repeated timing
- at least one warmup is required
- fewer than 20 measured repetitions

## Feasibility and task quality

| Solver | Problem | Qualified trials | Audited success | Task success | Worst equality | Worst side | Worst MPCC | Worst position | Worst angle | Worst velocity |
|---|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| ContactIPM | Cartpole with Soft Walls | 15/15 | 15/15 | 15/15 | 9.63e-07 | 0 | 1.67e-06 | 0.0063 | 0.0426 | 0.00531 |
| CRISP | Cartpole with Soft Walls | 15/15 | 15/15 | 15/15 | 1.66e-07 | 2.46e-14 | 5.21e-14 | 0.00265 | 0.0325 | 0.00531 |
| ContactIPM | Push Box | 21/25 | 21/25 | 22/25 | 1.81e-06 | 5.77e-07 | 7.68e-06 | 1.18 | 0.472 | 0 |
| CRISP | Push Box | 19/25 | 19/25 | 19/25 | 4.51e-07 | 4.71e-13 | 4.09e-12 | 0.358 | 0.147 | 0 |
| ContactIPM | Transport | 6/15 | 6/15 | 8/15 | 8.61e-07 | 2.52e-06 | 9.88e-06 | 0.000259 | 0 | 1.97 |
| CRISP | Transport | 8/15 | 8/15 | 8/15 | 3.28e-13 | 3.21e-13 | 2.74e-12 | 0.063 | 0 | 1.97 |

## Paired robustness outcomes

Each initial-state/target pair is counted once using the common trajectory audit.

| Problem | Both succeed | ContactIPM only | CRISP only | Neither |
|---|---:|---:|---:|---:|
| Cartpole with Soft Walls | 15 | 0 | 0 | 0 |
| Push Box | 17 | 4 | 2 | 2 |
| Transport | 6 | 0 | 2 | 7 |

## Locally measured process wall time

Values are median [Q1, Q3] and P90 in seconds over all measured processes. A paired speedup is reported only when both trajectories pass the common audit.

| Problem | ContactIPM | CRISP | Qualified pairs | CRISP / ContactIPM paired speedup (95% CI) | Ready |
|---|---:|---:|---:|---:|---:|
| Cartpole with Soft Walls | 0.0292 [0.0246, 0.0418], p90 0.1612 | 0.2206 [0.1832, 0.2452], p90 0.2583 | 15/15 | 7.106x [4.430, 8.555] | no |
| Push Box | 0.3482 [0.2139, 0.6140], p90 0.6609 | 0.4877 [0.3514, 0.5798], p90 0.7558 | 17/25 | 1.564x [0.966, 2.391] | no |
| Transport | 0.1898 [0.1583, 0.2723], p90 0.7771 | 0.4678 [0.1457, 0.6045], p90 0.9438 | 6/15 | 0.578x [0.177, 1.603] | no |

## Objective decomposition

Objective values are shown after feasibility and task metrics; they are not used as a standalone success criterion.

| Solver | Problem | Objective per instance (median) | Component medians |
|---|---|---:|---|
| ContactIPM | Cartpole with Soft Walls | 2.74413 | control_effort=2.64387, terminal_tracking=0.100261 |
| CRISP | Cartpole with Soft Walls | 2.72017 | control_effort=2.62123, terminal_tracking=0.0989437 |
| ContactIPM | Push Box | 18.7171 | force_effort=18.587, terminal_tracking=0.130346 |
| CRISP | Push Box | 8.19694 | force_effort=7.87744, terminal_tracking=0.241992 |
| ContactIPM | Transport | 1.145 | control_effort=0.738609, terminal_tracking=0.406389 |
| CRISP | Transport | 0.793676 | control_effort=0.72314, terminal_tracking=0.0705366 |
