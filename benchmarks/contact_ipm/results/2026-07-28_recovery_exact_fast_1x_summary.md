# Local contact benchmark summary

Source artifact: `benchmarks/contact_ipm/results/2026-07-28_recovery_exact_fast_1x.json`

All execution times in this report were measured locally; no runtime reported in the CRISP paper is used.

ContactIPM revision: `8bd659f5c9c1287ce65d57db1bbc12c1fa3fdad4`

This run is correctness/development evidence, not publication-ready timing evidence, because:

- at least one warmup is required
- fewer than 20 measured repetitions

## Feasibility and task quality

| Solver | Problem | Qualified trials | Audited success | Task success | Worst equality | Worst side | Worst MPCC | Worst position | Worst angle | Worst velocity |
|---|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| ContactIPM | Cartpole with Soft Walls | 1/1 | 1/1 | 1/1 | 4.44e-16 | 0 | 9.96e-07 | 0.00238 | 0.0298 | 0.00479 |
| CRISP | Cartpole with Soft Walls | 1/1 | 1/1 | 1/1 | 9.04e-08 | 5.58e-15 | 9.8e-15 | 0.00238 | 0.0298 | 0.0048 |
| ContactIPM | Push Box | 1/1 | 1/1 | 1/1 | 2.91e-16 | 0 | 5.67e-07 | 0.0335 | 0.0135 | 0 |
| CRISP | Push Box | 1/1 | 1/1 | 1/1 | 2.79e-07 | 9.75e-14 | 6.09e-13 | 0.0505 | 0.0171 | 0 |
| ContactIPM | Transport | 1/1 | 1/1 | 1/1 | 1.61e-07 | 0 | 9.89e-07 | 3.38e-05 | 0 | 0.462 |
| CRISP | Transport | 1/1 | 1/1 | 1/1 | 1.17e-14 | 3.96e-15 | 2.74e-12 | 3.55e-05 | 0 | 0.465 |

## Paired robustness outcomes

Each initial-state/target pair is counted once using the common trajectory audit.

| Problem | Both succeed | ContactIPM only | CRISP only | Neither |
|---|---:|---:|---:|---:|
| Cartpole with Soft Walls | 1 | 0 | 0 | 0 |
| Push Box | 1 | 0 | 0 | 0 |
| Transport | 1 | 0 | 0 | 0 |

## Locally measured process wall time

Values are median [Q1, Q3] and P90 in seconds over all measured processes. A paired speedup is reported only when both trajectories pass the common audit.

| Problem | ContactIPM | CRISP | Qualified pairs | CRISP / ContactIPM paired speedup (95% CI) | Ready |
|---|---:|---:|---:|---:|---:|
| Cartpole with Soft Walls | 0.0275 [0.0275, 0.0275], p90 0.0275 | 0.2320 [0.2320, 0.2320], p90 0.2320 | 1/1 | 8.427x [8.427, 8.427] | no |
| Push Box | 0.2312 [0.2312, 0.2312], p90 0.2312 | 0.7105 [0.7105, 0.7105], p90 0.7105 | 1/1 | 3.072x [3.072, 3.072] | no |
| Transport | 0.1702 [0.1702, 0.1702], p90 0.1702 | 0.3598 [0.3598, 0.3598], p90 0.3598 | 1/1 | 2.114x [2.114, 2.114] | no |

## Objective decomposition

Objective values are shown after feasibility and task metrics; they are not used as a standalone success criterion.

| Solver | Problem | Objective per instance (median) | Component medians |
|---|---|---:|---|
| ContactIPM | Cartpole with Soft Walls | 2.72017 | control_effort=2.62127, terminal_tracking=0.0989025 |
| CRISP | Cartpole with Soft Walls | 2.72017 | control_effort=2.62123, terminal_tracking=0.0989437 |
| ContactIPM | Push Box | 18.7871 | force_effort=18.6568, terminal_tracking=0.130309 |
| CRISP | Push Box | 6.93078 | force_effort=6.64645, terminal_tracking=0.284329 |
| ContactIPM | Transport | 2.96439 | control_effort=0.829006, terminal_tracking=2.13538 |
| CRISP | Transport | 2.98346 | control_effort=0.823621, terminal_tracking=2.15983 |
