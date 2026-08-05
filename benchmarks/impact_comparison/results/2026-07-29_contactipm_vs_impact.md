# ContactIPM versus IMPACT: local parameter-faithful comparison

All values come from local executions on the same machine. No execution times reported in Paper 3 are used.

Robustness statistics use one solve per predeclared local instance. Timing statistics use separately repeated, adjacent randomized pairs. A run is successful only when the solver reports convergence and the independent dynamics, side, equality, complementarity, and task audit passes.

## Cart Transport

| Metric | ContactIPM | IMPACT |
|---|---:|---:|
| Audited success | 50/50 (100.0%; 95% CI 92.9–100.0%) | 50/50 (100.0%; 95% CI 92.9–100.0%) |
| Total tracking error (median [IQR], P90) | 4837.2749 [3880.3633, 5308.1801], P90 5741.8749 | 280.4719 [137.8679, 797.1581], P90 1368.8230 |
| Objective (median [IQR], P90) | 1.686e-04 [1.257e-04, 5.345e-04], P90 8.875e-04 | 2.719e-04 [1.016e-04, 6.701e-04], P90 0.0022 |
| Control cost (median [IQR], P90) | 1.669e-04 [1.149e-04, 4.581e-04], P90 8.459e-04 | 2.675e-04 [1.012e-04, 6.667e-04], P90 0.0022 |
| Force effort (median [IQR], P90) | 154.8426 [108.1479, 439.4672], P90 809.6345 | 254.0451 [93.6759, 640.9515], P90 2161.9714 |
| Peak force (median [IQR], P90) | 4.3689 [3.2463, 13.9586], P90 16.9581 | 7.0336 [4.5993, 11.6486], P90 25.9679 |
| Dynamics defect (median [IQR], P90) | 2.220e-16 [2.220e-16, 2.220e-16], P90 4.441e-16 | 1.846e-08 [3.049e-09, 3.542e-08], P90 4.884e-08 |
| Complementarity (median [IQR], P90) | 9.979e-07 [9.952e-07, 9.992e-07], P90 9.996e-07 | 2.926e-06 [1.639e-06, 6.165e-06], P90 8.356e-06 |
| Terminal position error (median [IQR], P90) | 5.450e-08 [2.857e-08, 7.465e-08], P90 1.017e-07 | 1.775e-05 [7.316e-06, 3.486e-05], P90 5.125e-05 |
| Terminal velocity error (median [IQR], P90) | 8.044e-09 [2.429e-09, 2.154e-08], P90 4.685e-08 | 9.916e-09 [3.563e-09, 2.310e-08], P90 6.318e-08 |
| Solver-only time (s; median [IQR], P90) | 0.0366 [0.0334, 0.0385], P90 0.0415 | 0.1799 [0.1704, 0.1953], P90 0.2015 |
| End-to-end time (s; median [IQR], P90) | 0.0881 [0.0791, 0.0928], P90 0.0959 | 0.3677 [0.3526, 0.3873], P90 0.4207 |

Paired IMPACT/ContactIPM median solver-only ratio: 4.9078× (bootstrap 95% CI 4.7166–5.2545).

Paired IMPACT/ContactIPM median end-to-end ratio: 4.1916× (bootstrap 95% CI 3.9981–4.6539).

These statistics jointly expose reliability, task quality, physical feasibility, and effort; the scalar objective is not used as a stand-alone solution-quality verdict.

## Push Box

| Metric | ContactIPM | IMPACT |
|---|---:|---:|
| Audited success | 50/50 (100.0%; 95% CI 92.9–100.0%) | 49/50 (98.0%; 95% CI 89.5–99.6%) |
| Total tracking error (median [IQR], P90) | 38.8010 [20.6295, 57.6643], P90 85.0543 | 42.0388 [26.8116, 65.3952], P90 87.9542 |
| Objective (median [IQR], P90) | 0.0116 [0.0093, 0.0138], P90 0.0163 | 0.0120 [0.0105, 0.0135], P90 0.0155 |
| Control cost (median [IQR], P90) | 0.0116 [0.0093, 0.0138], P90 0.0163 | 0.0120 [0.0105, 0.0135], P90 0.0155 |
| Force effort (median [IQR], P90) | 5.3512 [3.0248, 7.8547], P90 10.5796 | 3.7937 [3.0759, 5.7708], P90 7.0513 |
| Peak force (median [IQR], P90) | 0.4443 [0.3830, 0.5227], P90 1.2898 | 0.4359 [0.3356, 0.5131], P90 0.8125 |
| Dynamics defect (median [IQR], P90) | 2.776e-17 [3.469e-18, 5.551e-17], P90 1.221e-16 | 1.267e-08 [5.810e-09, 2.784e-08], P90 1.001e-07 |
| Complementarity (median [IQR], P90) | 9.981e-07 [9.861e-07, 9.992e-07], P90 1.000e-06 | 3.148e-06 [1.863e-06, 4.950e-06], P90 5.876e-06 |
| Terminal translation error (median [IQR], P90) | 3.629e-05 [2.382e-05, 4.811e-05], P90 6.854e-05 | 2.828e-05 [1.443e-05, 3.752e-05], P90 4.564e-05 |
| Terminal angular error (median [IQR], P90) | 2.078e-05 [1.027e-05, 3.878e-05], P90 7.917e-05 | 1.618e-05 [9.092e-06, 2.408e-05], P90 2.887e-05 |
| Solver-only time (s; median [IQR], P90) | 0.2666 [0.2473, 0.2853], P90 0.3210 | 0.0599 [0.0543, 0.0681], P90 0.0820 |
| End-to-end time (s; median [IQR], P90) | 0.3158 [0.2879, 0.3378], P90 0.3871 | 0.1780 [0.1646, 0.1971], P90 0.2369 |

Paired IMPACT/ContactIPM median solver-only ratio: 0.2240× (bootstrap 95% CI 0.2169–0.2423).

Paired IMPACT/ContactIPM median end-to-end ratio: 0.5804× (bootstrap 95% CI 0.5320–0.6212).

These statistics jointly expose reliability, task quality, physical feasibility, and effort; the scalar objective is not used as a stand-alone solution-quality verdict.

## Push T

| Metric | ContactIPM | IMPACT |
|---|---:|---:|
| Audited success | 50/50 (100.0%; 95% CI 92.9–100.0%) | 48/50 (96.0%; 95% CI 86.5–98.9%) |
| Total tracking error (median [IQR], P90) | 12.0717 [6.6110, 21.2543], P90 37.9459 | 13.4867 [7.8406, 23.9066], P90 35.0088 |
| Objective (median [IQR], P90) | 0.0453 [0.0382, 0.0532], P90 0.0694 | 0.0384 [0.0352, 0.0449], P90 0.0490 |
| Control cost (median [IQR], P90) | 0.0453 [0.0382, 0.0531], P90 0.0694 | 0.0384 [0.0351, 0.0449], P90 0.0490 |
| Force effort (median [IQR], P90) | 1.4714 [0.8476, 2.4296], P90 3.6168 | 0.8690 [0.6284, 1.1274], P90 1.3126 |
| Peak force (median [IQR], P90) | 0.4093 [0.2501, 0.8093], P90 1.1991 | 0.2447 [0.2111, 0.2663], P90 0.2934 |
| Dynamics defect (median [IQR], P90) | 5.551e-17 [1.388e-17, 1.943e-16], P90 8.097e-07 | 3.908e-08 [7.671e-09, 1.124e-07], P90 2.001e-07 |
| Complementarity (median [IQR], P90) | 9.998e-07 [9.993e-07, 9.999e-07], P90 1.000e-06 | 1.101e-06 [9.217e-07, 1.172e-06], P90 1.246e-06 |
| Terminal translation error (median [IQR], P90) | 2.145e-04 [1.062e-04, 3.524e-04], P90 6.481e-04 | 5.339e-05 [3.334e-05, 1.126e-04], P90 1.284e-04 |
| Terminal angular error (median [IQR], P90) | 7.659e-05 [3.681e-05, 1.531e-04], P90 2.496e-04 | 8.809e-05 [6.259e-05, 9.584e-05], P90 1.044e-04 |
| Solver-only time (s; median [IQR], P90) | 0.2226 [0.2055, 0.2410], P90 0.2455 | 0.6645 [0.6250, 0.7193], P90 0.7882 |
| End-to-end time (s; median [IQR], P90) | 0.2713 [0.2486, 0.2937], P90 0.3029 | 0.8907 [0.8454, 0.9377], P90 1.0327 |

Paired IMPACT/ContactIPM median solver-only ratio: 2.9600× (bootstrap 95% CI 2.6538–3.2005).

Paired IMPACT/ContactIPM median end-to-end ratio: 3.2658× (bootstrap 95% CI 3.0063–3.5207).

These statistics jointly expose reliability, task quality, physical feasibility, and effort; the scalar objective is not used as a stand-alone solution-quality verdict.

