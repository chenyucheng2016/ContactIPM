# Local contact benchmark summary

Source artifact: `benchmarks/contact_ipm/results/2026-07-28_publication_timing_push_t_seg08_20x.json`

All execution times in this report were measured locally; no runtime reported in the CRISP paper is used.

ContactIPM source snapshot: `contactipm-src-11`

The timing protocol meets the configured readiness gates.

## Feasibility and task quality

| Solver | Problem | Qualified trials | Audited success | Task success | Worst equality | Worst side | Worst MPCC | Worst position | Worst angle | Worst velocity |
|---|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| ContactIPM | Push T | 20/20 | 20/20 | 20/20 | 3.47e-17 | 0 | 1e-06 | 0.00146 | 2.39e-05 | 0 |
| CRISP | Push T | 20/20 | 20/20 | 20/20 | 3.48e-06 | 7.49e-15 | 1.57e-07 | 0.0667 | 0.0251 | 0 |

## Paired robustness outcomes

Each initial-state/target pair is counted once using the common trajectory audit.

| Problem | Both succeed | ContactIPM only | CRISP only | Neither |
|---|---:|---:|---:|---:|
| Push T | 20 | 0 | 0 | 0 |

## Locally measured process wall time

Values are median [Q1, Q3] and P90 in seconds over all measured processes. A paired speedup is reported only when both trajectories pass the common audit.

| Problem | ContactIPM | CRISP | Qualified pairs | CRISP / ContactIPM paired speedup (95% CI) | Ready |
|---|---:|---:|---:|---:|---:|
| Push T | 0.6902 [0.6455, 0.7285], p90 0.7927 | 5.1942 [4.9395, 5.2899], p90 5.4642 | 20/20 | 7.533x [7.224, 7.715] | yes |

## Objective decomposition

Objective values are shown after feasibility and task metrics; they are not used as a standalone success criterion.

| Solver | Problem | Objective per instance (median) | Component medians |
|---|---|---:|---|
| ContactIPM | Push T | 4.29128 | force_effort=1.73482, running_tracking=2.55624, terminal_tracking=0.000212205 |
| CRISP | Push T | 5.6465 | force_effort=2.07146, running_tracking=3.06793, terminal_tracking=0.507112 |
