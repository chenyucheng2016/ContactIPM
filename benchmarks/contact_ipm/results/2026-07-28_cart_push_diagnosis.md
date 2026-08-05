# Cartpole and Push Box diagnosis milestone

This is a one-repetition correctness and diagnosis run, not a publication-quality timing study. Both methods ran locally on the same WSL machine and no execution time reported in the CRISP paper was used. The machine-readable schema-version 4 artifact is [`2026-07-28_cart_push_scorecard_1x.json`](2026-07-28_cart_push_scorecard_1x.json).

The comparison is feasibility-first: physical feasibility, task success, task errors, objective components, computational performance, then contact behavior. A larger scalar objective is not treated as task failure.

## Exact-instance scorecard

| Problem | Solver | Feasible / task | Position error | Angle error | Dynamics defect | Physical MPCC | Tracking cost | Effort cost | Total objective | Peak force | Mode changes | Wall time |
|---|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| Cartpole | ContactIPM | 1/1 / 1/1 | 0.002375 | 0.029795 | 5.00e-16 | 9.94e-7 | 0.098868 | 2.621303 | 2.720171 | 10.11 control | n/a | 0.0207 s |
| Cartpole | CRISP | 1/1 / 1/1 | 0.002377 | 0.029824 | 9.04e-8 | 9.80e-15 | 0.098944 | 2.621229 | 2.720172 | 10.11 control | n/a | 0.1579 s |
| Push Box | ContactIPM | 1/1 / 1/1 | 0.033503 | 0.013466 | 2.22e-16 | 9.58e-7 | 0.130380 | 18.659618 | 18.789998 | 14.63 contact | 0 | 0.1806 s |
| Push Box | CRISP | 1/1 / 1/1 | 0.050505 | 0.017103 | 2.79e-7 | 6.09e-13 | 0.284329 | 6.646455 | 6.930783 | 8.30 contact | 3 | 0.4138 s |

The wall times above are single observations and must not be used as final paper statistics.

## Cartpole diagnosis

The previous fixed Jacobi-preconditioned run required 276 iterations. Its diagnostic trace had mean accepted primal step 0.051, with 237 iterations below step 0.1. Disabling the fixed preconditioner increased the mean accepted step to 0.276 and reduced convergence to 46 iterations while preserving every audit gate and essentially the same objective and terminal state. Starting with full exact finite-difference curvature failed at the 300-iteration cap, so exact curvature is not the appropriate fix for this case.

The selected default therefore leaves the already well-scaled Cartpole formulation unpreconditioned. Diagnostic environment overrides remain available for reproducing the ablation; they do not affect the default benchmark protocol.

## Push Box diagnosis

The earlier ContactIPM trajectory was feasible and tracked better than CRISP, but used 22.087 force-effort cost, peak force norm 26.84, and 43 dominant-mode changes. CRISP used 6.646 force-effort cost, peak force 8.30, and 3 mode changes. This identified a contact-mode basin/chattering issue rather than a task or physical-feasibility failure.

Starting the elastic log-barrier continuation at 1e-4 reduced ContactIPM to 18.660 effort cost and eliminated dominant-mode changes while retaining better terminal tracking and all physical gates. This is the selected zero-initial-guess default.

As a diagnostic only, ContactIPM initialized from CRISP's feasible trajectory converged in 16 iterations to objective 6.711605, translation error 0.019415, angle error 0.005157, dynamics defect 1.79e-14, and physical MPCC 2.26e-7. This proves the solver can preserve and improve the low-effort contact basin. The cross-warm-start result is excluded from fair benchmark timing and baseline quality claims because it uses the competitor trajectory.