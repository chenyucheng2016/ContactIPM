# Push Box scaling study

Node count and time step are compiled into both solvers while total physical duration, physics, target, and per-stage contact geometry remain fixed. Build and discarded warmup/model-generation times are excluded. A timed run is eligible only when the solver exits successfully and the common trajectory audit passes. No execution time from either paper is used.

The downloaded per-stage objective weights are unchanged, so both solvers solve the same discrete objective at each node count. The objective is not rescaled as a continuous-time quadrature across node counts; this study isolates solver scaling, not discretization-invariant optimal control cost.

| Nodes | dt (s) | MPCC pairs | ContactIPM eligible | CRISP eligible | ContactIPM median (s) | CRISP median (s) | Paired speedup | ContactIPM iterations | CRISP iterations |
|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| 25 | 0.0825 | 240 | 0/5 | 5/5 | n/a | 0.2153 | n/a | n/a | 135 |
| 50 | 0.040408 | 490 | 0/5 | 5/5 | n/a | 0.309 | n/a | n/a | 83 |
| 75 | 0.026757 | 740 | 5/5 | 5/5 | 0.1354 | 0.2348 | 1.59x | 231 | 38 |
| 100 | 0.02 | 990 | 5/5 | 5/5 | 0.2427 | 0.5536 | 2.25x | 282 | 77 |
| 150 | 0.013289 | 1490 | 5/5 | 0/5 | 2.531 | n/a | n/a | 1956 | n/a |

ContactIPM source-variable counts reflect its exact elimination of Push T split equalities; CRISP counts retain the downloaded source variables. These counts exclude solver-internal slacks and duals.
