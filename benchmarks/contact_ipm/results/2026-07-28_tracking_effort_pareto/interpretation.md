# Tracking-effort Pareto interpretation

This experiment scales the same source objective in ContactIPM and the
instrumented CRISP checkout. The tracking multiplier is fixed at 1 and the
force-effort multiplier is swept over 0.1, 0.25, 0.5, 1, 2, 4, and 10.
The default multiplier of 1 preserves the original benchmark objective
exactly. Reported tracking and effort components are recomputed by the common
trajectory audit without the sweep multiplier, so they are comparable between
points. The one-shot process times in the raw files are diagnostic only.

## Main results

| Problem | Solver | Solver-converged | Audit pass | Converged and audited |
|---|---|---:|---:|---:|
| Push Box | ContactIPM | 4/7 | 5/7 | 4/7 |
| Push Box | CRISP | 7/7 | 5/7 | 5/7 |
| Push T segment 8 | ContactIPM | 7/7 | 7/7 | 7/7 |
| Push T segment 8 | CRISP | 7/7 | 2/7 | 2/7 |

Push Box shows a genuine tracking-effort tradeoff rather than a scalar-cost
winner. In the shared valid range from 0.25 through 2, ContactIPM uses more
force and reaches smaller terminal errors; CRISP uses less force with larger
terminal errors. Both methods remain physically feasible at weights 4 and 10
but fail the common terminal-task gate. At weight 0.1, ContactIPM reaches the
iteration limit even though its returned trajectory passes the independent
audit. That point is retained in the table but excluded from Pareto-front
eligibility.

Push T is the stronger robustness result. ContactIPM converges and passes the
common audit at all seven weights. CRISP passes only at weights 0.25 and 1. Its
0.1 and 0.5 trajectories meet the task gate but exceed the physical
complementarity tolerance, with products 3.43e-5 and 2.93e-5. Its weights 2,
4, and 10 are physically feasible but fail the terminal-task gate. At the
original weight 1, ContactIPM also has both lower unscaled force effort
(1.732 versus 2.071) and lower tracking cost (2.558 versus 3.575).

The curves are nonmonotone in places because both solvers address a nonconvex
MPCC and can enter different local basins as the weight changes. The report
therefore retains every point and marks nondominance only when both solver
termination and the independent audit succeed.

## Scope

This is objective-sensitivity evidence on one fixed Push Box instance and one
fixed Push T instance. It supports the claim that the default objective gap
cannot be interpreted as solution quality by itself, and it demonstrates a
wide valid tradeoff range on Push T. It does not replace multi-instance
robustness, repeated timing, horizon/contact-count scaling, or a harder
closed-loop contact task.

See [the generated table](summary.md) and
[machine-readable summary](summary.json).
