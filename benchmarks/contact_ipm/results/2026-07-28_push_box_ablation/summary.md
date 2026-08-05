# Push Box solver-component ablation

Each configuration is evaluated once on every one of the 25 frozen initial-state/target cases. This is robustness evidence, not repeated timing evidence.

| Configuration | Environment overrides | Audited success | Feasible | Task success | Exit 0 | Median process wall (s) | Total wall (s) | Median objective on success |
|---|---|---:|---:|---:|---:|---:|---:|---:|
| full | none | 21/25 | 22/25 | 22/25 | 21/25 | 0.303164 | 8.33341 | 18.7171 |
| no_recovery | CONTACTIPM_PUSH_RECOVERY=0 | 3/25 | 4/25 | 11/25 | 2/25 | 0.179428 | 4.62412 | 18.5769 |
| no_preconditioner | CONTACTIPM_PUSH_PRECONDITIONER=0 | 18/25 | 19/25 | 21/25 | 18/25 | 0.188341 | 5.49925 | 18.6203 |
| gauss_newton_primary | CONTACTIPM_PUSH_EXACT_HESSIAN=0 | 24/25 | 25/25 | 24/25 | 24/25 | 0.218417 | 7.15754 | 18.6427 |
| recovery_mu_0p01 | CONTACTIPM_PUSH_RECOVERY_MU=0.01 | 21/25 | 22/25 | 21/25 | 20/25 | 0.384267 | 10.3986 | 18.7171 |
| recovery_mu_1 | CONTACTIPM_PUSH_RECOVERY_MU=1.0 | 25/25 | 25/25 | 25/25 | 24/25 | 0.414411 | 9.90741 | 18.7886 |
