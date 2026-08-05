# Local contact benchmark summary

Source artifact: `benchmarks/contact_ipm/results/2026-07-28_generic_recovery_exact_fast_1x.json`

All execution times in this report were measured locally; no runtime reported in the CRISP paper is used.

ContactIPM revision: `2c6fb1020780538b06f8a106d18780250c8390e0`

This run is correctness/development evidence, not publication-ready timing evidence, because:

- at least one warmup is required
- fewer than 20 measured repetitions

## Feasibility and task quality

| Solver | Problem | Qualified trials | Audited success | Task success | Worst equality | Worst side | Worst MPCC | Worst position | Worst angle | Worst velocity |
|---|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| ContactIPM | Cartpole with Soft Walls | 1/1 | 1/1 | 1/1 | 5e-16 | 0 | 9.96e-07 | 0.00238 | 0.0298 | 0.00479 |
| CRISP | Cartpole with Soft Walls | 1/1 | 1/1 | 1/1 | 9.04e-08 | 5.58e-15 | 9.8e-15 | 0.00238 | 0.0298 | 0.0048 |
| ContactIPM | Push Box | 1/1 | 1/1 | 1/1 | 2.12e-16 | 0 | 5e-07 | 0.0335 | 0.0135 | 0 |
| CRISP | Push Box | 1/1 | 1/1 | 1/1 | 2.79e-07 | 9.75e-14 | 6.09e-13 | 0.0505 | 0.0171 | 0 |
| ContactIPM | Transport | 1/1 | 1/1 | 1/1 | 1.7e-07 | 0 | 9.51e-07 | 3.39e-05 | 0 | 0.462 |
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
| Cartpole with Soft Walls | 0.0249 [0.0249, 0.0249], p90 0.0249 | 0.1887 [0.1887, 0.1887], p90 0.1887 | 1/1 | 7.577x [7.577, 7.577] | no |
| Push Box | 0.2118 [0.2118, 0.2118], p90 0.2118 | 0.4551 [0.4551, 0.4551], p90 0.4551 | 1/1 | 2.148x [2.148, 2.148] | no |
| Transport | 0.1604 [0.1604, 0.1604], p90 0.1604 | 0.3846 [0.3846, 0.3846], p90 0.3846 | 1/1 | 2.397x [2.397, 2.397] | no |

## Objective decomposition

Objective values are shown after feasibility and task metrics; they are not used as a standalone success criterion.

| Solver | Problem | Objective per instance (median) | Component medians |
|---|---|---:|---|
| ContactIPM | Cartpole with Soft Walls | 2.72017 | control_effort=2.62127, terminal_tracking=0.0989025 |
| CRISP | Cartpole with Soft Walls | 2.72017 | control_effort=2.62123, terminal_tracking=0.0989437 |
| ContactIPM | Push Box | 18.7871 | force_effort=18.6568, terminal_tracking=0.130302 |
| CRISP | Push Box | 6.93078 | force_effort=6.64645, terminal_tracking=0.284329 |
| ContactIPM | Transport | 2.9646 | control_effort=0.829653, terminal_tracking=2.13495 |
| CRISP | Transport | 2.98346 | control_effort=0.823621, terminal_tracking=2.15983 |
