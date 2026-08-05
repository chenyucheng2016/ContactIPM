# ContactIPM closed-loop Push Box validation

Artifact SHA-256: `f9330b2362f104c13b7c14b76e2ef17b9f09a695a8992da2d96278bf7aee54d2`.

Reference execution: Ubuntu 20.04 WSL2, Intel Core Ultra 7 155H, GCC 9.4 Release, CPU affinity CPU 0.

ContactIPM completed 50/50 rollouts (100.0%) with zero unrecovered failures and zero invalid plans. Success requires the terminal pose to remain within 0.1 m and 0.1 rad for 10 consecutive control steps.

## Robustness by scenario

| Scenario | Success | Steps median/max | Translation error median/max (m) | Angular error median/max (rad) | Primary failures | Cold fallbacks |
|---|---:|---:|---:|---:|---:|---:|
| Nominal | 5/5 | 19/19 | 0.000153/0.000153 | 0.000161/0.000161 | 0 | 0 |
| Initial-pose perturbation | 15/15 | 19/21 | 0.00157/0.0102 | 8.68e-05/0.000876 | 0 | 0 |
| Mass/friction mismatch | 10/10 | 20.5/24 | 0.000523/0.00279 | 9.73e-05/0.000969 | 0 | 0 |
| Isolated state-reset diagnostic | 10/10 | 26/31 | 0.000361/0.00732 | 8e-05/0.000801 | 0 | 0 |
| Disturbed motion | 10/10 | 25/27 | 0.00136/0.0119 | 0.000621/0.00215 | 0 | 0 |

## Solver timing

| Population | Solves | Median | P90 | P99 | Max | Deadline misses |
|---|---:|---:|---:|---:|---:|---:|
| All solves | 1104 | 2.08 ms | 11.5 ms | 60.8 ms | 317 ms | 7 |
| Shifted warm starts | 1054 | 2.01 ms | 7.74 ms | 47.4 ms | 290 ms | 2 (0.19%) |

The control period is 100 ms. These measurements support mostly real-time-capable execution, not a hard real-time guarantee, because deadline misses remain.

## Physical and numerical audit

| Metric | Worst value |
|---|---:|
| Planned dynamics defect | 8.62e-07 |
| Planned side-constraint violation | 7.3e-09 |
| Planned physical complementarity | 1.01e-06 |
| Applied side-constraint violation | 7.02e-09 |
| Applied physical complementarity | 1e-06 |
| Maximum one-step plant/model state error | 2.91 |

Physical complementarity is audited on the original side-row products, independently of the elastic product slack and its log barrier.

## Reading the figure

Panels (a), (b), and (d) compare only the nominal run with the disturbed-motion run. Panel (a) overlays shaded box footprints at the initial and geometric-halfway samples, plus an unfilled target footprint; each center-to-front line shows the signed box heading. The dashed blue nominal path is drawn last so it remains visible where the paths overlap.

Panel (b) plots distance and absolute angular error to the fixed target versus elapsed closed-loop time. Samples are 0.1 s apart; the errors decrease because feedback drives the measured pose toward that target. The horizontal dotted line is the common 0.1 m/rad goal tolerance, and each curve ends after its rollout completes the ten-step goal hold. Panel (c) is the empirical solve-time CDF over all 1,104 solves, with the 100 ms deadline. Panel (d) audits the original physical complementarity product against its tolerance.

Disturbed motion is the combined stress test. Its initial x/y offsets are independently uniformly sampled from [-0.075, 0.075] m and its initial yaw offset from [-0.075, 0.075] rad. Mass and friction are independently uniformly scaled by factors in [0.85, 1.15]. Every feedback measurement receives independent zero-mean Gaussian noise with 0.001 m standard deviation in x/y and 0.001 rad in yaw. After the 1.5 s control update, one direct state reset adds independent uniform x/y offsets in [-0.1125, 0.1125] m and a yaw offset from [-0.09, 0.09] rad; it first appears in the recorded state at t=1.6 s. All draws use deterministic seed 2027. It is therefore not a single-disturbance experiment. The 10 isolated state-reset diagnostics remain in the 50-run aggregate but are not plotted.

## Scope

The controller uses the CRISP Push Box quasi-static model, a 19-step forward-Euler prediction horizon, zero free-state and control guesses for the first solve (with stage zero fixed to the measurement), and shifted warm starts thereafter. The OCP retains the source terminal tracking and force effort and adds unit-weight stage pose tracking to prevent finite-horizon motion deferral. The plant uses ten RK4 substeps. Disturbed motion combines deterministic mass/friction mismatch, measurement noise, an initial-pose error, and one scheduled pose state reset.

Because the source benchmark has pose but no momentum state, the scheduled reset is a direct pose/yaw state change rather than an inertial impulse. The model also has no penetration or friction-cone state to audit; this experiment demonstrates closed-loop complementarity handling, not high-fidelity contact physics.
