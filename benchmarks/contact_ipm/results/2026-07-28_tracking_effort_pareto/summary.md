# Matched tracking-effort Pareto study

Both solvers optimize the same source objective with the listed positive effort multiplier and a fixed tracking multiplier. Reported tracking and effort costs use the unscaled common audit definitions, so points remain comparable across multipliers. Nondominance requires solver convergence and a successful common audit. Process times are diagnostic only.

ContactIPM source snapshot: `contactipm-src-16`

## Push Box

| Solver | Effort scale | Solver exit | Audit | Effort cost | Tracking cost | Position error | Angle error | Peak force | Nondominated |
|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| CRISP | 0.1 | converged | pass | 6.735 | 0.09878 | 0.02973 | 0.01018 | 8.359 | yes |
| CRISP | 0.25 | converged | pass | 6.71 | 0.1421 | 0.03566 | 0.01223 | 8.344 | yes |
| CRISP | 0.5 | converged | pass | 6.543 | 0.6265 | 0.07513 | 0.02493 | 8.24 | yes |
| CRISP | 1 | converged | pass | 6.646 | 0.2843 | 0.05051 | 0.0171 | 8.305 | yes |
| CRISP | 2 | converged | pass | 6.621 | 0.3546 | 0.0564 | 0.0191 | 8.289 | yes |
| CRISP | 4 | converged | fail | 6.224 | 2.571 | 0.1529 | 0.04831 | 8.039 | no |
| CRISP | 10 | converged | fail | 5.935 | 5.572 | 0.2259 | 0.06845 | 7.859 | no |
| ContactIPM | 0.1 | failed | pass | 18.91 | 0.02444 | 0.01557 | 0.001434 | 17.08 | no |
| ContactIPM | 0.25 | converged | pass | 18.85 | 0.008209 | 0.008419 | 0.003347 | 14.7 | yes |
| ContactIPM | 0.5 | converged | pass | 18.79 | 0.03275 | 0.01681 | 0.006706 | 14.68 | yes |
| ContactIPM | 1 | converged | pass | 18.66 | 0.1303 | 0.03348 | 0.01348 | 14.62 | yes |
| ContactIPM | 2 | converged | pass | 18.4 | 0.5172 | 0.06666 | 0.027 | 14.49 | yes |
| ContactIPM | 4 | failed | fail | 17.9 | 2.022 | 0.1313 | 0.0547 | 14.29 | no |
| ContactIPM | 10 | failed | fail | 16.48 | 11.91 | 0.3147 | 0.1414 | 13.67 | no |

## Push T segment 8

| Solver | Effort scale | Solver exit | Audit | Effort cost | Tracking cost | Position error | Angle error | Peak force | Nondominated |
|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| CRISP | 0.1 | converged | fail | 4.826 | 1.545 | 0.0001079 | 4.354e-05 | 17.64 | no |
| CRISP | 0.25 | converged | pass | 4.269 | 1.792 | 0.002516 | 0.00104 | 17.65 | yes |
| CRISP | 0.5 | converged | fail | 2.914 | 2.18 | 0.01152 | 0.00162 | 13.25 | no |
| CRISP | 1 | converged | pass | 2.071 | 3.575 | 0.06666 | 0.02505 | 9.884 | yes |
| CRISP | 2 | converged | fail | 1.325 | 6.977 | 0.1638 | 0.06795 | 7.781 | no |
| CRISP | 4 | converged | fail | 0.8392 | 8.328 | 0.1702 | 0.07 | 5.052 | no |
| CRISP | 10 | converged | fail | 0.4998 | 10.45 | 0.1743 | 0.07225 | 3.07 | no |
| ContactIPM | 0.1 | converged | pass | 6.795 | 1.104 | 0.001872 | 1.905e-05 | 25.24 | yes |
| ContactIPM | 0.25 | converged | pass | 5.429 | 1.408 | 0.002223 | 0.0001195 | 22.21 | yes |
| ContactIPM | 0.5 | converged | pass | 2.706 | 1.744 | 0.0002985 | 9.359e-06 | 12.17 | yes |
| ContactIPM | 1 | converged | pass | 1.732 | 2.558 | 0.00115 | 6.988e-05 | 8.826 | yes |
| ContactIPM | 2 | converged | pass | 1.274 | 3.042 | 0.006044 | 0.001128 | 6.502 | yes |
| ContactIPM | 4 | converged | pass | 0.9299 | 4.108 | 0.01439 | 0.001232 | 4.856 | yes |
| ContactIPM | 10 | converged | pass | 0.6905 | 9.353 | 0.05565 | 0.0009409 | 4.885 | yes |
