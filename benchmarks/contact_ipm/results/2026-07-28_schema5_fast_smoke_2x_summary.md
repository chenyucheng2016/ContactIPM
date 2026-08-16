# Local contact benchmark summary

Source artifact: `benchmarks/contact_ipm/results/2026-07-28_schema5_fast_smoke_2x.json`

All execution times in this report were measured locally; no runtime reported in the CRISP paper is used.

ContactIPM source snapshot: `contactipm-src-08`

This run is correctness/development evidence, not publication-ready timing evidence, because:

- fewer than 20 measured repetitions

## Feasibility and task quality

| Solver | Problem | Qualified trials | Audited success | Task success | Worst equality | Worst side | Worst MPCC | Worst position | Worst angle | Worst velocity |
|---|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| ContactIPM | Cartpole with Soft Walls | 2/2 | 2/2 | 2/2 | 5.55e-16 | 0 | 1.18e-06 | 0.00241 | 0.0298 | 0.00479 |
| CRISP | Cartpole with Soft Walls | 2/2 | 2/2 | 2/2 | 9.04e-08 | 5.58e-15 | 9.8e-15 | 0.00238 | 0.0298 | 0.0048 |
| ContactIPM | Push Box | 2/2 | 2/2 | 2/2 | 2.91e-16 | 0 | 5.67e-07 | 0.0335 | 0.0135 | 0 |
| CRISP | Push Box | 2/2 | 2/2 | 2/2 | 2.79e-07 | 9.75e-14 | 6.09e-13 | 0.0505 | 0.0171 | 0 |
| ContactIPM | Transport | 2/2 | 2/2 | 2/2 | 1.27e-08 | 0 | 9.5e-07 | 3.38e-05 | 0 | 0.462 |
| CRISP | Transport | 2/2 | 2/2 | 2/2 | 1.17e-14 | 3.96e-15 | 2.74e-12 | 3.55e-05 | 0 | 0.465 |

## Locally measured process wall time

Values are median [Q1, Q3] and P90 in seconds over all measured processes. A paired speedup is reported only when both trajectories pass the common audit.

| Problem | ContactIPM | CRISP | Qualified pairs | CRISP / ContactIPM paired speedup (95% CI) | Ready |
|---|---:|---:|---:|---:|---:|
| Cartpole with Soft Walls | 0.0404 [0.0383, 0.0425], p90 0.0438 | 0.2909 [0.2800, 0.3018], p90 0.3083 | 2/2 | 7.226x [7.009, 7.444] | no |
| Push Box | 0.2813 [0.2812, 0.2813], p90 0.2813 | 0.6659 [0.6593, 0.6725], p90 0.6765 | 2/2 | 2.367x [2.321, 2.414] | no |
| Transport | 0.1202 [0.1170, 0.1234], p90 0.1253 | 0.5458 [0.5298, 0.5619], p90 0.5716 | 2/2 | 4.540x [4.514, 4.567] | no |

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
