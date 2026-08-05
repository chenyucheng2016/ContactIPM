# CRISP contact-implicit benchmark reproductions

These benchmarks are transcribed from the downloaded CRISP examples, not
reconstructed only from the paper:

| Adapter | CRISP source |
|---|---|
| `cartpole_soft_walls.cpp` | `benchmarks/CRISP/src/examples/pushbot/cpp/SolvePushbot.cpp` |
| `push_box.cpp` | `benchmarks/CRISP/src/examples/pushbox/SolvePushbox.cpp` |
| `transport.cpp` | `benchmarks/CRISP/src/examples/Transp/cpp/SolveTransp.cpp` |
| `push_t.cpp` | `benchmarks/CRISP/src/examples/pushT/SolvePushT.cpp` |

The adapters replace each hand-written product inequality with marked
complementarity metadata. ContactIPM generates the elastic product row, its
positive slack, and its log-barrier/primal-dual equations. The physical
complementarity product remains a separate convergence gate, so this is not
merely acceptance of a relaxed product constraint.

`ContactIPM::solve_mpcc_with_recovery` supplies a bounded,
problem-independent recovery policy for marked complementarity problems. It
tries the configured solve, two geometric barrier-continuation stages, then
restores the original primal guess for a conservative-barrier Gauss-Newton
solve and a final tightening solve. `ContactIPMParams::mpcc_recovery_mu`
controls the conservative barrier and `mpcc_recovery_max_iters` bounds each
recovery stage. Every attempted phase is included in the returned iteration
count and measured wall time. All four adapters invoke this same solver-level
policy; none selects retries using benchmark-specific task metrics.

CRISP's Transport and Push T models include per-node algebraic split equalities
of the form `h - v + w = 0`. The adapters eliminate them exactly with
`w = v - h`, then retain `v >= 0`, `w >= 0`, and `v*w = 0` as marked elastic
MPCC rows. This preserves the original equality and avoids approximating it
with narrow state bounds. All metrics are evaluated on the raw
multiple-shooting solution; there is no dynamics projection or split-variable
polishing after a solve.

## Build and run

Run from the repository root in Ubuntu or WSL. Fresh-machine prerequisites and
end-to-end CRISP/IMPACT instructions are in
[`REPRODUCIBILITY.md`](../../REPRODUCIBILITY.md).

```bash
cmake -S . -B build-contact \
  -DCMAKE_BUILD_TYPE=Release \
  -DNMPC_BUILD_EXAMPLES=OFF \
  -DNMPC_BUILD_TESTS=ON \
  -DNMPC_BUILD_CONTACT_BENCHMARKS=ON
cmake --build build-contact -j2
(cd build-contact && ctest --output-on-failure)

./build-contact/contact_cartpole_soft_walls
./build-contact/contact_push_box
./build-contact/contact_transport
./build-contact/contact_push_t
```

The subshell CTest invocation works with the CTest 3.16 version used for the
reported WSL experiments.

## Closed-loop Push Box validation

`contact_push_box_closed_loop` applies ContactIPM in receding-horizon feedback
to the audited CRISP Push Box complementarity model. The first solve uses zero
free-state/control guesses, with stage zero fixed to the measurement; later
solves use only the solver's shifted previous solution. The OCP retains source
terminal tracking and force effort, and adds unit-weight stage pose tracking to
prevent finite-horizon motion deferral. A 19-step, 0.1 s prediction horizon uses
forward Euler, while the simulated plant uses ten RK4 substeps. The reader-facing
disturbed
motion uses independent uniform initial x/y offsets in `[-0.075, 0.075]` m and
yaw offset in `[-0.075, 0.075]` rad; independent uniform mass and friction
scale factors in `[0.85, 1.15]`;
adds zero-mean Gaussian measurement noise with standard deviations `0.001` m
in x/y and `0.001` rad in yaw; and, after the 1.5 s control update, adds one
direct state reset with independent uniform x/y offsets in
`[-0.1125, 0.1125]` m and yaw offset in
`[-0.09, 0.09]` rad. The reset first appears in the recorded state at 1.6 s.
All draws use deterministic seed 2027.

Run and summarize the fixed 50-rollout suite with:

```bash
cmake --build build-contact -j2 --target contact_push_box_closed_loop
taskset -c 0 ./build-contact/contact_push_box_closed_loop \
  --mode suite --seed 2027 \
  --output closed_loop_push_box.json \
  --trajectory-dir closed_loop_push_box_trajectories
python3 benchmarks/contact_ipm/summarize_closed_loop.py \
  closed_loop_push_box.json --require-suite
python3 benchmarks/contact_ipm/plot_closed_loop.py \
  closed_loop_push_box.json closed_loop_push_box_trajectories \
  --output-prefix closed_loop_push_box_validation
python3 benchmarks/contact_ipm/animate_closed_loop.py \
  closed_loop_push_box.json closed_loop_push_box_trajectories \
  --output-dir closed_loop_push_box_media
```

Render the anonymous 1080p/30 fps H.264 combined-disturbance clip with:

```bash
python3 -m pip install imageio-ffmpeg
python3 benchmarks/contact_ipm/render_closed_loop_video.py \
  closed_loop_push_box.json closed_loop_push_box_trajectories \
  --output contactipm_push_box_combined_disturbance.mp4
```

Unless `--rollout-index` is supplied, the renderer selects the successful
combined-disturbance rollout with the largest translational state reset. The
reference video therefore uses rollout 49. It overlays nominal rollout 0,
plots the measured-minus-true state residual supplied to MPC, and compares the
nominal and disturbed contact-force norms. It labels the required
zero-initialized first solve separately from shifted-warm-start solves.

The [reference summary](results/2026-07-29_closed_loop_push_box_50_summary.md)
reports 50/50 task successes, zero solver failures/fallbacks/invalid plans,
2.08/11.5/60.8 ms median/P90/P99 over 1,104 solves, and two warm-started
100 ms deadline misses. Worst planned physical complementarity is
`1.01e-6`; worst terminal translation and angular errors are 0.0119 m and
0.00215 rad. The commands above regenerate the validation figure and two GIFs
from the raw artifact and ignored trajectory CSVs. The generated visuals compare only the nominal run with disturbed motion.

Panel (a) uses shaded box footprints at the initial and geometric-halfway poses,
with center-to-front heading lines to show orientation along both paths. In panel (b), the horizontal axis is elapsed
closed-loop time, sampled every 0.1 s; translation and absolute angular error
fall as feedback drives the measured pose toward the fixed target. Each curve
ends after that rollout completes the ten-step goal hold.

This is a quasi-static benchmark: its state contains pose but no velocity or
momentum. Accordingly, the scheduled reset inside disturbed motion directly changes
`[x, y, theta]`; it is not an external force or impulse. The model has no penetration or
friction-cone state. The experiment supports closed-loop complementarity
handling and feedback robustness, not a
high-fidelity contact-physics claim. Timing is machine-dependent and is not a
hard real-time guarantee.

Run one Push T initial condition while developing:

```bash
./build-contact/contact_push_t 23
```

Record a machine-readable local run:

```bash
python3 benchmarks/contact_ipm/run_benchmarks.py \
  --contactipm-build build-contact \
  --output contact_benchmark_results.json
```

For an exact-instance comparison with the downloaded CRISP checkout, first add
the deterministic trajectory exports and rebuild CRISP:

```bash
python3 benchmarks/contact_ipm/instrument_crisp.py \
  --crisp-root benchmarks/CRISP
python3 benchmarks/contact_ipm/verify_source_parity.py
cmake -S benchmarks/CRISP/src -B build-crisp \
  -DCMAKE_BUILD_TYPE=Release
cmake --build build-crisp -j2 --target \
  pushbot_example pushbox_example cartTransp_example pushT_example
python3 benchmarks/contact_ipm/run_benchmarks.py \
  --contactipm-build build-contact \
  --crisp-build build-crisp \
  --warmups 1 --repetitions 20 --threads 1 \
  --cpu-affinity auto --seed 2027 \
  --output contact_benchmark_results.json
python3 benchmarks/contact_ipm/summarize_benchmarks.py \
  contact_benchmark_results.json
```

Run the development initial-state/target robustness suite separately:

```bash
python3 benchmarks/contact_ipm/run_benchmarks.py \
  --contactipm-build build-contact \
  --crisp-build build-crisp \
  --problems cartpole_soft_walls push_box transport \
  --case-suite benchmarks/contact_ipm/validation_cases.json \
  --warmups 0 --repetitions 1 --threads 1 \
  --cpu-affinity auto --seed 2027 \
  --output initial_target_sweep.json
```

This suite varies only initial states and targets. Horizons and physical
parameters remain frozen because those quantities are compiled into the
downloaded CRISP generated problems; horizon/parameter scaling is a distinct
experiment and must rebuild both solvers at each setting. A case-suite run is
robustness evidence and is deliberately excluded from publication timing
readiness.

For repeated Push T timing, pass a fixed `--push-t-segment` after running
`instrument_crisp.py`. The runner supplies that segment to ContactIPM and the
instrumented CRISP loop, so both processes solve exactly one identical source
instance. The full 50-segment run is retained separately as reliability
evidence. Segment 8 is the current predeclared timing instance: it passes the
common audit for both solvers and was near the median CRISP solve time among
CRISP-successful segments in the prior complete sweep.

`source_cases.json` freezes the physical problem, initial guess, target, node
layout, and audit tolerances. The parity check includes the SHA-256 hash of the
cartpole initial-guess file and fails on source/adapter drift. The CRISP
instrumentation is idempotent and changes only raw trajectory output,
controlled initial/target inputs, the cartpole guess path, and optional Push T
loop selection; it does not alter solver settings or formulations.

Every timed trial writes to a new directory. Schema version 5 keeps each
solver pair adjacent, randomizes both block order and solver order, records CPU
affinity and executable/source hashes, and never imports paper-reported
execution times.

Tracked-dirty checks normalize Windows line endings, and each artifact records
the SHA-256 digest and file list of the CRISP tracked patch. This makes the two
Eigen include-path compatibility edits and the benchmark instrumentation
auditable without treating CRLF conversion as a source change.

`trajectory_audit.py` then applies
the same independent dynamics, side-feasibility, physical complementarity,
terminal-task, and objective calculations to both solvers. Schema version 4
also separates feasibility from task success, decomposes tracking and effort
costs, and reports peak force, contact impulse, active-contact steps, and
contact-mode changes. Results are interpreted in that order; objective only
ranks solutions after feasibility and task quality are visible. Process exit
codes and solver-reported residuals are retained but are not treated as audit
evidence. The runner uses only measurements collected on the current machine;
it never imports execution times reported in either paper.

Set `CONTACTIPM_DIAG_CSV` to record per-iteration solver diagnostics. The
problem-independent ablation controls are `CONTACTIPM_EXACT_HESSIAN`,
`CONTACTIPM_PRECONDITIONER`, `CONTACTIPM_MPCC_RECOVERY`, and
`CONTACTIPM_MPCC_RECOVERY_MU`. Cartpole and Push Box retain their legacy
`CONTACTIPM_CART_*` and `CONTACTIPM_PUSH_*` aliases for reproducibility.
Passing a 100-by-9 raw trajectory as the optional Push Box command-line
argument supplies a controlled initial guess for basin diagnostics; it is not
used by the fair default run.

Run the fixed 25-case Push Box solver-component ablation with:

```bash
python3 benchmarks/contact_ipm/run_contact_ablations.py \
  --contactipm-build build-contact \
  --output-dir benchmarks/contact_ipm/results/push_box_ablation
```

The six configurations isolate recovery, preconditioning, primary exact
curvature, and recovery-barrier sensitivity. Every configuration is evaluated
with the same independent trajectory audit. The single observation per case is
robustness evidence; its process times are not publication timing statistics.

The [current Push Box ablation](results/2026-07-28_generic_recovery_push_box_ablation/interpretation.md)
shows that removing generic recovery drops audited success from 24/25 to 3/25
and disabling the generic preconditioner drops it to 21/25. A Gauss-Newton
primary solve reaches 25/25 on Push Box, but a cross-benchmark trial produced
only 49/50 solver-success statuses on Push T despite 50/50 audited trajectories.
Exact curvature therefore remains the problem-independent default; no
benchmark-specific curvature switch is used.

Run the matched tracking-effort study with:

```bash
python3 benchmarks/contact_ipm/run_pareto.py \
  --contactipm-build build-contact \
  --crisp-build build-crisp \
  --output-dir benchmarks/contact_ipm/results/tracking_effort_pareto \
  --problems push_box push_t \
  --effort-scales 0.1 0.25 0.5 1 2 4 10 \
  --tracking-scale 1 --push-t-segment 8 \
  --threads 1 --seed 2027 --cpu-affinity auto
python3 benchmarks/contact_ipm/plot_pareto.py \
  benchmarks/contact_ipm/results/tracking_effort_pareto/summary.json
```

Both solvers receive the same objective multipliers. The audit reports
unscaled tracking and force-effort components, so points remain comparable
across weights. A point is marked nondominated only when the solver exits
successfully and its trajectory passes the common physical and task audit.
These one-shot runs are quality evidence, not timing measurements.

For duration-fixed horizon/contact-count scaling, compile and run both methods
through the dedicated driver, then render the combined timing and audit-success
figure:

```bash
python3 benchmarks/contact_ipm/run_scaling.py \
  --problem push_box --nodes 25 50 75 100 150 \
  --build-root build-scaling \
  --output-dir benchmarks/contact_ipm/results/scaling_push_box \
  --repetitions 5 --warmups 1 --threads 1 --cpu-affinity auto
python3 benchmarks/contact_ipm/run_scaling.py \
  --problem push_t --nodes 25 35 50 75 100 \
  --build-root build-scaling \
  --output-dir benchmarks/contact_ipm/results/scaling_push_t \
  --push-t-segment 8 \
  --repetitions 5 --warmups 1 --threads 1 --cpu-affinity auto
python3 benchmarks/contact_ipm/plot_scaling.py \
  benchmarks/contact_ipm/results/scaling_push_box/summary.json \
  benchmarks/contact_ipm/results/scaling_push_t/summary.json \
  --output-prefix benchmarks/contact_ipm/results/mpcc_scaling/scaling
```

The time step is chosen to preserve each source problem's physical duration.
Only solver-converged, independently audited runs contribute timing. Per-stage
source objective weights remain unchanged, so the study is not a
continuous-time objective-discretization convergence test.

## Current WSL development baseline

The solver passes all 13 C++ regression tests, and the benchmark harness passes
all 17 Python protocol tests.
All four adapters expose complementarity pairs through the common constraint
metadata and call the same bounded `solve_mpcc_with_recovery` policy. The
[three fast exact-instance artifact](results/2026-07-28_generic_recovery_exact_fast_1x.json)
and [Push T exact-instance artifact](results/2026-07-28_generic_recovery_exact_push_t_1x.json)
record ContactIPM success on Cartpole, Push Box, Transport, and all 50 Push T
segments. CRISP passes the first three source instances, 27/50 Push T segments
under the physical audit, and 43/50 Push T task gates. These one-repetition
runs are correctness evidence, not timing evidence.

The [current 55-case sweep](results/2026-07-28_generic_recovery_sweep_1x_summary.md)
records ContactIPM versus CRISP audited success of 15/15 versus 15/15 on
Cartpole, 24/25 versus 19/25 on Push Box, and 8/15 versus 8/15 on Transport.
The paired Push Box outcomes are 19 both, five ContactIPM-only, zero CRISP-only,
and one neither; Transport has eight both and seven neither. Both solvers are
physically feasible on all 55 cases, so remaining sweep failures are task-level
robustness failures. Overall audited success is 47/55 versus 42/55. This is
breadth/correctness evidence, not publication timing.

The current audited timing artifacts combine the robustness and quality audits with 20-pair local
timing on revision `2c6fb10`. Median paired CRISP/ContactIPM speedups are
8.868x on Cartpole, 2.168x on Push Box, 2.337x on Transport, and 6.920x on the
predeclared Push T segment 8. All 80 pairs pass the common audit and every
timing-readiness gate. Earlier publication-timing artifacts remain useful
historical records but predate the unified recovery implementation and are not
used as the current timing result.

The [matched tracking-effort interpretation](results/2026-07-28_tracking_effort_pareto/interpretation.md)
shows that the Push Box objective gap is a force-versus-tracking tradeoff, not a
quality failure. On Push T segment 8, ContactIPM converges and passes the common
audit at all 7 weights, versus 2/7 for CRISP. The study records revision
`dde3b44`; default-weight audit metrics are unchanged from the original source
objectives. Process times from this one-shot weight sweep are not used.

The [duration-fixed MPCC scaling interpretation](results/2026-07-28_mpcc_scaling/interpretation.md)
reports five-pair development sweeps from 240 to 1490 Push Box complementarity pairs
and 1032 to 4257 Push T pairs. ContactIPM is faster at every jointly eligible
point and remains task successful at the largest tested sizes where CRISP
misses the terminal gate. ContactIPM fails the two coarsest Push Box grids, so
the result is not claimed as uniform dominance. A 20-pair rerun is still
required for final timing statistics.

`summarize_benchmarks.py` reports audited success, task/physics metrics,
objective components, median/IQR/P90 wall time, bootstrap confidence
intervals, and paired speedups only when both runs pass. It marks results as
underpowered unless the run uses warmups, explicit CPU affinity, schema-version
5 pairing, and at least 20 repetitions. The
[two-repetition schema-v5 smoke summary](results/2026-07-28_schema5_fast_smoke_2x_summary.md)
validates pairing, affinity, hashing, and reporting on the three fast problems;
it is intentionally marked underpowered.

## Historical WSL development baselines

The exact contact-curvature milestone at implementation commit `bd6666c`
passes all 10 WSL regression tests and all four ContactIPM task gates. Push T
passes 50/50 source initial conditions with worst dynamics `6.580e-6`, side
violation `8.499e-9`, and physical complementarity product `4.564e-6`.

A one-warmup, three-repetition randomized/interleaved run was then collected
locally for both solvers with one thread and seed 2027. Median wall times favor
ContactIPM on Push Box (1.70x), Transport (4.91x), and Push T (5.82x), while
CRISP is faster on Cartpole (ContactIPM/CRISP = 2.83x). No execution time from
the CRISP paper is used.

The historical result below predates the exact-instance exporter for the three
single-instance CRISP examples. Use a schema-version 4 runner artifact for new
comparisons; it contains the common raw audit for all four problems.

The first schema-version 3 exact-instance audit is in the
[2026-07-28 milestone report](results/2026-07-28_exact_instances_1x.md) and
[machine-readable artifact](results/2026-07-28_exact_instances_1x.json).
ContactIPM passed all 50 Push T cases while CRISP passed 27; both solvers passed
Cartpole, Push Box, and Transport. This one-repetition run is correctness
evidence, not publication timing statistics.

The feasibility-first Cartpole and Push Box follow-up is in the
[diagnosis report](results/2026-07-28_cart_push_diagnosis.md) and
[schema-version 4 artifact](results/2026-07-28_cart_push_scorecard_1x.json).


See the [development report](results/2026-07-27_wsl_interleaved_3x.md) and
[machine-readable artifact](results/2026-07-27_wsl_interleaved_3x.json). These
three-repetition WSL measurements are development evidence, not final
publication statistics.
