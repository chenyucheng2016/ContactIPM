# Local contact benchmark summary

Source artifact: `benchmarks/contact_ipm/results/2026-07-28_generic_recovery_exact_push_t_1x.json`

All execution times in this report were measured locally; no runtime reported in the CRISP paper is used.

ContactIPM revision: `2c6fb1020780538b06f8a106d18780250c8390e0`

This run is correctness/development evidence, not publication-ready timing evidence, because:

- at least one warmup is required
- fewer than 20 measured repetitions

## Feasibility and task quality

| Solver | Problem | Qualified trials | Audited success | Task success | Worst equality | Worst side | Worst MPCC | Worst position | Worst angle | Worst velocity |
|---|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| ContactIPM | Push T | 1/1 | 50/50 | 50/50 | 4.97e-06 | 1.92e-09 | 1e-06 | 0.0579 | 0.00697 | 0 |
| CRISP | Push T | 0/1 | 27/50 | 43/50 | 2 | 4.69e-11 | 0.000448 | 0.388 | 0.267 | 0 |

## Paired robustness outcomes

Each initial-state/target pair is counted once using the common trajectory audit.

| Problem | Both succeed | ContactIPM only | CRISP only | Neither |
|---|---:|---:|---:|---:|
| Push T | 0 | 1 | 0 | 0 |

## Locally measured process wall time

Values are median [Q1, Q3] and P90 in seconds over all measured processes. A paired speedup is reported only when both trajectories pass the common audit.

| Problem | ContactIPM | CRISP | Qualified pairs | CRISP / ContactIPM paired speedup (95% CI) | Ready |
|---|---:|---:|---:|---:|---:|
| Push T | 38.1723 [38.1723, 38.1723], p90 38.1723 | 360.4330 [360.4330, 360.4330], p90 360.4330 | 0/1 | n/a | no |

## Objective decomposition

Objective values are shown after feasibility and task metrics; they are not used as a standalone success criterion.

| Solver | Problem | Objective per instance (median) | Component medians |
|---|---|---:|---|
| ContactIPM | Push T | 29.0335 | force_effort=9.85642, running_tracking=19.0355, terminal_tracking=0.00164436 |
| CRISP | Push T | 7.23128 | force_effort=2.95576, running_tracking=4.90239, terminal_tracking=0.0837748 |
