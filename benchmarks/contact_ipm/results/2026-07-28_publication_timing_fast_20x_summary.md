# Local contact benchmark summary

Source artifact: `benchmarks/contact_ipm/results/2026-07-28_publication_timing_fast_20x.json`

All execution times in this report were measured locally; no runtime reported in the CRISP paper is used.

ContactIPM source snapshot: `contactipm-src-11`

The timing protocol meets the configured readiness gates.

## Feasibility and task quality

| Solver | Problem | Qualified trials | Audited success | Task success | Worst equality | Worst side | Worst MPCC | Worst position | Worst angle | Worst velocity |
|---|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| ContactIPM | Cartpole with Soft Walls | 20/20 | 20/20 | 20/20 | 4.44e-16 | 0 | 9.96e-07 | 0.00238 | 0.0298 | 0.00479 |
| CRISP | Cartpole with Soft Walls | 20/20 | 20/20 | 20/20 | 9.04e-08 | 5.58e-15 | 9.8e-15 | 0.00238 | 0.0298 | 0.0048 |
| ContactIPM | Push Box | 20/20 | 20/20 | 20/20 | 2.91e-16 | 0 | 5.67e-07 | 0.0335 | 0.0135 | 0 |
| CRISP | Push Box | 20/20 | 20/20 | 20/20 | 2.79e-07 | 9.75e-14 | 6.09e-13 | 0.0505 | 0.0171 | 0 |
| ContactIPM | Transport | 20/20 | 20/20 | 20/20 | 1.61e-07 | 0 | 9.89e-07 | 3.38e-05 | 0 | 0.462 |
| CRISP | Transport | 20/20 | 20/20 | 20/20 | 1.17e-14 | 3.96e-15 | 2.74e-12 | 3.55e-05 | 0 | 0.465 |

## Paired robustness outcomes

Each initial-state/target pair is counted once using the common trajectory audit.

| Problem | Both succeed | ContactIPM only | CRISP only | Neither |
|---|---:|---:|---:|---:|
| Cartpole with Soft Walls | 20 | 0 | 0 | 0 |
| Push Box | 20 | 0 | 0 | 0 |
| Transport | 20 | 0 | 0 | 0 |

## Locally measured process wall time

Values are median [Q1, Q3] and P90 in seconds over all measured processes. A paired speedup is reported only when both trajectories pass the common audit.

| Problem | ContactIPM | CRISP | Qualified pairs | CRISP / ContactIPM paired speedup (95% CI) | Ready |
|---|---:|---:|---:|---:|---:|
| Cartpole with Soft Walls | 0.0288 [0.0262, 0.0345], p90 0.0362 | 0.2095 [0.1816, 0.2514], p90 0.2709 | 20/20 | 7.666x [6.459, 8.791] | yes |
| Push Box | 0.2191 [0.1982, 0.2451], p90 0.2887 | 0.5261 [0.4751, 0.5903], p90 0.6385 | 20/20 | 2.271x [2.148, 2.563] | yes |
| Transport | 0.1934 [0.1713, 0.2199], p90 0.2746 | 0.4001 [0.3689, 0.4717], p90 0.5265 | 20/20 | 2.144x [2.037, 2.222] | yes |

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
