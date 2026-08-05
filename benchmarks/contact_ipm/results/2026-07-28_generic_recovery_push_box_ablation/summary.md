# Push Box solver-component ablation

Each configuration is evaluated once on every one of the 25 frozen initial-state/target cases. This is robustness evidence, not repeated timing evidence.

| Configuration | Environment overrides | Audited success | Feasible | Task success | Exit 0 | Median process wall (s) | Total wall (s) | Median objective on success |
|---|---|---:|---:|---:|---:|---:|---:|---:|
| full | none | 24/25 | 25/25 | 24/25 | 24/25 | 0.288625 | 18.0009 | 18.6819 |
| no_recovery | CONTACTIPM_PUSH_RECOVERY=0 | 3/25 | 4/25 | 11/25 | 2/25 | 0.159392 | 4.07489 | 18.5769 |
| no_preconditioner | CONTACTIPM_PUSH_PRECONDITIONER=0 | 21/25 | 22/25 | 23/25 | 20/25 | 0.202913 | 10.3674 | 18.79 |
| gauss_newton_primary | CONTACTIPM_PUSH_EXACT_HESSIAN=0 | 25/25 | 25/25 | 25/25 | 25/25 | 0.199691 | 5.59979 | 18.6467 |
| recovery_mu_0p01 | CONTACTIPM_PUSH_RECOVERY_MU=0.01 | 23/25 | 24/25 | 24/25 | 21/25 | 0.409845 | 18.5549 | 18.6405 |
| recovery_mu_1 | CONTACTIPM_PUSH_RECOVERY_MU=1.0 | 23/25 | 23/25 | 24/25 | 22/25 | 0.317177 | 19.1515 | 18.6482 |
