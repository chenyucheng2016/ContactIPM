# Push T scaling study

Node count and time step are compiled into both solvers while total physical duration, physics, target, and per-stage contact geometry remain fixed. Build and discarded warmup/model-generation times are excluded. A timed run is eligible only when the solver exits successfully and the common trajectory audit passes. No execution time from either paper is used.

The downloaded per-stage objective weights are unchanged, so both solvers solve the same discrete objective at each node count. The objective is not rescaled as a continuous-time quadrature across node counts; this study isolates solver scaling, not discretization-invariant optimal control cost.

| Nodes | dt (s) | MPCC pairs | ContactIPM eligible | CRISP eligible | ContactIPM median (s) | CRISP median (s) | Paired speedup | ContactIPM iterations | CRISP iterations |
|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| 25 | 0.10208 | 1032 | 5/5 | 5/5 | 0.3152 | 1.101 | 3.84x | n/a | n/a |
| 35 | 0.072059 | 1462 | 5/5 | 5/5 | 0.3608 | 3.897 | 10.9x | n/a | n/a |
| 50 | 0.05 | 2107 | 5/5 | 5/5 | 0.7406 | 10.49 | 13.9x | n/a | n/a |
| 75 | 0.033108 | 3182 | 5/5 | 0/5 | 1.099 | n/a | n/a | n/a | n/a |
| 100 | 0.024747 | 4257 | 5/5 | 0/5 | 3.026 | n/a | n/a | n/a | n/a |

ContactIPM source-variable counts reflect its exact elimination of Push T split equalities; CRISP counts retain the downloaded source variables. These counts exclude solver-internal slacks and duals.
