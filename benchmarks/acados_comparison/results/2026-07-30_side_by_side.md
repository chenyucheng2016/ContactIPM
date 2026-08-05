# Contact benchmark comparison

All execution times below were measured locally; no time reported in a paper is
used. CRISP-parameter and IMPACT-parameter tasks are reported separately because
their physical parameters differ.

Strict success means solver-reported convergence **and** an independently
audited feasible, task-completing trajectory. The acados baseline uses full SQP
with `PARTIAL_CONDENSING_HPIPM` and exact complementarity constraints
`a >= 0`, `b >= 0`, `a*b <= 0`; it does not use a complementarity relaxation.

## CRISP-parameter robustness

| Problem | Trials | ContactIPM strict | CRISP strict | acados converged | acados feasible | acados task | acados strict |
|---|---:|---:|---:|---:|---:|---:|---:|
| Cartpole soft walls | 15 | 15/15 | 15/15 | 9/15 | 11/15 | 10/15 | 9/15 |
| Push Box | 25 | 24/25 | 19/25 | 19/25 | 25/25 | 0/25 | 0/25 |
| Transport | 15 | 8/15 | 8/15 | 0/15 | 14/15 | 0/15 | 0/15 |
| Push T | 50 | 50/50 | 27/50 | 0/50 | 1/50 | 8/50 | 0/50 |

The decomposition exposes the contact-initiation failure: acados often returns
a feasible no-contact stationary solution for Push Box, but the box does not
move to its target. Fast termination of such a run is not counted as success.

## CRISP-parameter source-case timing

Median [P90] solver time over 20 eligible repetitions after one warmup:

| Problem | ContactIPM | CRISP | acados | Source-case objective |
|---|---:|---:|---:|---|
| Cartpole soft walls | 27.2 [29.9] ms | 240.2 [260.8] ms | 952.4 [1060.2] ms | 2.720171 / 2.720172 / 2.720171 |
| Push Box | 261.4 [334.7] ms | 593.1 [665.8] ms | no eligible solve | 18.7871 / 6.9308 / -- |
| Transport | 196.7 [230.4] ms | 498.8 [547.6] ms | no eligible solve | 2.96460 / 2.98346 / -- |
| Push T, segment 8 | 788.5 [897.2] ms | 5528.5 [5887.2] ms | no eligible solve | 4.28988 / 5.64650 / -- |

The objective columns are ordered ContactIPM / CRISP / acados. ContactIPM is
8.8x, 2.3x, 2.5x, and 7.0x faster than CRISP on the four eligible source cases.
It is 35x faster than acados on Cartpole; no speedup is claimed where acados has
no eligible solve.

For diagnostic completeness, the ineligible acados median termination times
were 595.3 ms for Push Box, 55.3 ms for Transport, and 15.2 ms for Push T.
These are not valid solve times.

## CRISP-parameter solution quality

Medians over independently audited successful robustness cases:

| Problem | Solver | Objective | Terminal position error | Terminal angle/velocity error | Complementarity |
|---|---|---:|---:|---:|---:|
| Cartpole | ContactIPM | 2.72017 | 2.38e-3 | angle 2.98e-2 | 9.94e-7 |
|  | CRISP | 2.72017 | 2.38e-3 | angle 2.98e-2 | 1.25e-14 |
|  | acados | 2.70841 | 2.35e-3 | angle 2.97e-2 | 1.96e-11 |
| Push Box | ContactIPM | 18.6819 | 3.35e-2 | angle 1.35e-2 | 7.77e-7 |
|  | CRISP | 8.19694 | 4.43e-2 | angle 2.13e-2 | 6.14e-13 |
|  | acados | -- | -- | -- | -- |
| Transport | ContactIPM | 0.79214 | 4.96e-6 | velocity 6.96e-2 | 9.41e-7 |
|  | CRISP | 0.79368 | 5.06e-6 | velocity 7.13e-2 | 6.01e-14 |
|  | acados | -- | -- | -- | -- |
| Push T | ContactIPM | 29.0335 | 4.05e-3 | angle 7.05e-5 | 1.00e-6 |
|  | CRISP | 7.23128 | 2.89e-2 | angle 8.55e-6 | 9.26e-15 |
|  | acados | -- | -- | -- | -- |

This table prevents a larger scalar objective from being mistaken for a worse
trajectory. ContactIPM's Push Box and Push T solutions use more effort, but
have better median terminal position tracking and substantially higher success.

## IMPACT-parameter robustness

| Problem | Trials | ContactIPM strict | IMPACT strict | acados converged | acados feasible | acados task | acados strict |
|---|---:|---:|---:|---:|---:|---:|---:|
| Push Box | 50 | 50/50 | 49/50 | 43/50 | 50/50 | 0/50 | 0/50 |
| Push T | 50 | 50/50 | 48/50 | 2/50 | 31/50 | 3/50 | 1/50 |
| Cart Transport | 50 | 50/50 | 50/50 | 0/50 | 50/50 | 26/50 | 0/50 |

ContactIPM is the only solver with 100% strict success on all three
IMPACT-parameter sweeps. Cart Transport also shows why solver status is kept
separate: acados produced 26 audited-valid trajectories but never declared
convergence within 300 SQP iterations.

## IMPACT-parameter source-case timing

Median [P90] solver time over 20 eligible repetitions after one warmup:

| Problem | ContactIPM | IMPACT | acados | Source-case objective |
|---|---:|---:|---:|---|
| Push Box | 266.6 [321.0] ms | 59.9 [82.0] ms | no eligible solve | 0.038858 / 0.007947 / -- |
| Push T | 222.6 [245.5] ms | 664.5 [788.2] ms | no eligible solve | 0.086733 / 0.053888 / -- |
| Cart Transport | 36.6 [41.5] ms | 180.0 [201.5] ms | no eligible solve | 0.003329 / 0.000500 / -- |

IMPACT is 4.5x faster on Push Box. ContactIPM is 3.0x faster on Push T and
4.9x faster on Cart Transport. The ineligible acados median termination times
were 1.43 ms, 1900.2 ms, and 2064.0 ms respectively; they are diagnostics, not
valid solve-time comparisons.

## IMPACT-parameter solution quality

Medians over independently audited successful robustness cases:

| Problem | Solver | Objective | Terminal position error | Terminal angle error | Complementarity |
|---|---|---:|---:|---:|---:|
| Push Box | ContactIPM | 0.011557 | 3.63e-5 | 2.08e-5 | 9.98e-7 |
|  | IMPACT | 0.011950 | 2.83e-5 | 1.62e-5 | 3.15e-6 |
|  | acados | -- | -- | -- | -- |
| Push T | ContactIPM | 0.045281 | 2.15e-4 | 7.66e-5 | 1.00e-6 |
|  | IMPACT | 0.038389 | 5.34e-5 | 8.81e-5 | 1.10e-6 |
|  | acados | 0.038609 | 2.46e-4 | 5.28e-5 | 1.94e-10 |
| Cart Transport | ContactIPM | 1.69e-4 | 5.45e-8 | -- | 9.98e-7 |
|  | IMPACT | 2.72e-4 | 1.78e-5 | -- | 2.93e-6 |
|  | acados (audited valid, not converged) | 3.71e-6 | 3.22e-9 | -- | 8.67e-11 |

The single successful acados Push T case and the 26 audited-valid acados Cart
Transport cases are shown honestly, while the strict-success table preserves
the termination requirement.

## Evidence

- `2026-07-30_crisp_robustness.json`
- `2026-07-30_crisp_timing_20x.json`
- `2026-07-30_impact_robustness.json`
- `2026-07-30_impact_timing_20x.json`
- `../../contact_ipm/results/2026-07-28_generic_recovery_sweep_1x_summary.json`
- `../../contact_ipm/results/2026-07-28_generic_recovery_exact_push_t_1x_summary.json`
- `../../contact_ipm/results/2026-07-28_generic_recovery_publication_timing_fast_20x_summary.json`
- `../../contact_ipm/results/2026-07-28_generic_recovery_publication_timing_push_t_seg08_20x_summary.json`
- `../../impact_comparison/results/2026-07-29_robustness_50_summary.json`
- `../../impact_comparison/results/2026-07-29_timing_20x.json`
