# Quadruped CITO

This directory contains the quadruped SRBD model, the continuous ContactIPM
planner-in-the-loop experiment, focused audits, and optional developer bridges.
It does not reuse the prescribed schedule in `centroidal_trot_nmpc.cpp`.

The intended control stack is:

```text
terrain model -> 5 Hz ContactIPM high-level planner -> ContactPlan
             -> high-rate convex whole-body controller -> robot/simulator
```

ContactIPM produces time-indexed body motion, contact decisions, footholds,
swing clearance, and ground-reaction-force references. The validated closed
loop in this repository currently ends at an independently integrated SRBD
plant. The convex-WBC and MuJoCo files are development components, not evidence
for articulated closed-loop execution. The next integration step is to feed the
rolling ContactIPM plan to the convex controller and validate that combined
stack in MuJoCo.

## Continuous SRBD feedback experiment

`quadruped_cito_srbd_closed_loop` performs one cold solve and then maintains one
persistent SRBD plant for 200 warm updates. The planner uses `N=50`,
`dt=0.05` s, and a 2.5 s horizon. At each 5 Hz update, four planned stages are
integrated before the resulting 25-state measurement initializes the next
solve. There are no episode resets.

```bash
./build-srbd-repro/quadruped_cito_srbd_closed_loop \
  --output-dir /tmp/contactipm_srbd_02cm \
  --updates 200 --terrain-amplitude 0.02 \
  --wave-number-x 4 --wave-number-y 3 --deadline-ms 0
```

The accepted reference run publishes 200/200 warm plans with zero fallback,
completes all 24 scheduled transitions over 40 s, and passes the independent
rolling contact audit. See the root
[`REPRODUCIBILITY.md`](../../REPRODUCIBILITY.md#8-continuous-srbd-closed-loop-validation)
for the complete build, test, output, and claim boundaries.

## Implemented scope

- 25-state SRBD model: base position, unit quaternion, world linear velocity,
  body angular velocity, and four world-frame foot positions.
- 28 controls: four world-frame ground-reaction forces, four world-frame foot
  velocities, and one nonnegative motion envelope per foot.
- Contact-implicit constraints for terrain gap/normal force, friction pyramid,
  force-motion complementarity, tangential swing clearance, and leg reach.
- A force-motion formulation that complements normal force with the motion
  envelope. Linear component-wise velocity bounds connect the envelope to foot
  velocity, avoiding the zero derivative and stance-repelling log barrier of a
  squared-speed complementarity row.
- Schedule-free terrain clearance envelopes of the form
  `|tangent_velocity| <= clearance_rate * (gap + tolerance)`. They require a
  moving foot to gain clearance without selecting a swing foot or lift-off time
  in advance.
- An optional schedule-free support margin requiring every pair of feet to
  carry a minimum combined normal force. When enabled, any one foot may swing,
  but two feet cannot be unloaded together; CITO still selects the foot and
  timing.
- Dimensionless contact rows. Gap, force, speed, and reach are normalized before
  entering the common MPCC relaxation.
- Plane, analytic sinusoidal height-field, and differentiable step terrain,
  including exact first derivatives of the changing terrain frame.
- A fixed-size planner-to-controller contract and time sampler in
  `quadruped_cito_plan.hpp`. State references are interpolated, while the
  direct-transcription force and velocity controls are held over their stage.
- An endpoint-preserving swing-reference generator that uses CITO liftoff and
  touchdown times, touchdown locations, and maximum planned clearance without
  feeding direct-transcription velocity discontinuities into Cartesian
  derivative feedback.
- An instantaneous convex force-redistribution controller in
  `quadruped_cito_wbc.hpp`. It tracks the planned net wrench and force
  feed-forward subject to per-foot friction cones, then maps stance forces with
  `J^T R^T f` and swing Cartesian PD commands through `J^T`.
- Regressions for derivatives, standing equilibrium, mesh scaling, terrain
  difficulty, two-foot schedule sensitivity, plan sampling, convex force
  allocation, and joint-torque mapping.
- A calibrated MuJoCo Menagerie Go1 SRBD description (12.743448 kg composite
  mass, composite CoM/inertia, and measured home foot geometry), plus one
  analytic terrain descriptor shared by CITO and MuJoCo's sampled height field.
- Native Go1 plan initialization by free-base/leg IK, measured-contact stance
  confirmation with contact-loss recovery, and per-moving-foot replay gates for
  clearance, landing, touchdown timing, slip, base tracking, and actuator
  saturation.
- Event-level replay metrics that evaluate every liftoff/touchdown pair, rather
  than only the final swing of each foot.
- Native 100-node, 6 s flat and sinusoidal-terrain plans with a terminal-support
  audit, both rear-foot contact orderings, and a matched fixed-schedule plan
  exported before ContactIPM optimization.
- A 100-node, 6 s all-leg plan with 5.3 cm base travel and four 8 cm steps. The
  same 20 N support margin and convex WBC pass strict flat and sinusoidal MuJoCo
  replays without a prescribed gait schedule.

The transition regression is a functionality test, not a performance result.
It directly solves a 20-node, 1.6 s plan for an 8 cm front-foot transition on a
2.5 cm-amplitude sinusoidal height field. No contact schedule is supplied. The
test independently requires at least 6 cm of achieved displacement, 2 cm of
terrain clearance, a genuinely unloaded moving phase, dynamics defect below
`2e-5`, inequality violation below `1e-6`, and MPCC product below `1e-4`.

The same 1.6 s physical problem is discretized at 20, 40, 60, and 80 nodes to
separate mesh scaling from increasing task duration. A separate 50-node, 3 s
case moves two feet. It is solved from both possible swing orders with the same
terminal objective. Both orders remain valid local solutions, demonstrating
that the schedule is not encoded as a constraint but also showing that the
current nonconvex solve is warm-start dependent.

The corresponding 100-node case holds the completed maneuver in four-foot
support through 6 s. On the 2 cm sinusoidal terrain, the forward and reverse
warm starts converge to the requested opposite contact orders in 32 and 39
iterations. Their local development solve times were 6.2 s and 12.5 s;
repeated release-build timing distributions are still required before using
these numbers as a solver comparison.

The terrain gates currently include a 6 cm sinusoidal field and a 20 degree
cross-slope as direct solves. The differentiable 4 cm step currently exhausts
the bounded recovery portfolio in this build and remains an unresolved stress
case, not evidence of routine step-terrain performance.

One local, single-run mesh snapshot from the regression build is shown below.
These timings are diagnostics, not paper-quality performance measurements:

| Nodes | Fixed problem storage | Iterations | Solve time | Dynamics defect | MPCC product |
|---:|---:|---:|---:|---:|---:|
| 20 | 1.42 MiB | 26 | 0.28 s | `9.8e-9` | `1.0e-5` |
| 40 | 2.77 MiB | 29 | 2.61 s | `2.0e-7` | `1.0e-5` |
| 60 | 4.12 MiB | 27 | 0.78 s | `9.7e-11` | `1.0e-5` |
| 80 | 5.47 MiB | 30 | 8.93 s | `8.2e-6` | `1.1e-5` |

The non-monotonic times are a warning to collect repeated release-build timing
distributions before making scaling claims. The MuJoCo-labeled suite currently
takes about 199 s; the single-foot plan exports dominate that time. The separate
smooth-step recovery stress test currently reaches its 2,500-iteration limit in
about 160--175 s.

## Planner-to-controller contract

All quantities use SI units. Positions, velocities, forces, and terrain normals
are in the world frame. Base angular velocity is in the body frame. The
quaternion is ordered `(w, x, y, z)` and rotates body vectors into the world
frame.

Each `ContactPlanStage` contains:

- stage start time and duration;
- desired base pose and velocity;
- desired foot position and velocity;
- feed-forward ground-reaction force and local terrain normal;
- the terrain gap, normal force, and a thresholded contact flag.

The continuous CITO solution remains the source of truth. The contact flag is
only an execution interface, classified by default with a 2 mm maximum gap and
5 N minimum normal force. The controller should use the continuous force and
motion references for transitions instead of treating this flag as an
additional planning variable.

`sample_contact_plan` explicitly distinguishes active, pre-start, and expired
plans. An expired plan produces zero force/velocity commands and must be
handled by the eventual simulator bridge instead of silently extrapolating an
old plan.

## Convex whole-body execution boundary

`ConvexWholeBodyController` is an instantaneous controller, not another MPC.
At each control tick it:

1. combines the CITO feed-forward wrench with proportional-derivative base
   feedback;
2. solves a convex quadratic force-redistribution problem by projected
   gradient iterations with exact projections onto capped friction cones;
3. blends stance force control and swing Cartesian tracking based on planned
   normal force; and
4. maps the resulting body-frame foot commands through caller-supplied leg
   Jacobians.

The input contains measured base/foot state and four body-frame foot Jacobians.
The output contains world-frame contact forces and three joint torques per leg.
Robot dynamics, actuator limits, collision geometry, state estimation, and
contact sensing are intentionally not guessed here; they are responsibilities
of the next simulator-specific adapter. Its required conventions, logging, and
acceptance sequence are specified in `SIMULATOR_HANDOFF.md`.

This follows the force/torque interface in Di Carlo et al., [*Dynamic
Locomotion in the MIT Cheetah 3 Through Convex Model-Predictive
Control*](https://dspace.mit.edu/bitstream/handle/1721.1/138000/convex_mpc_2fix.pdf),
and the associated
[`mit-biomimetics/Cheetah-Software`](https://github.com/mit-biomimetics/Cheetah-Software)
project. ContactIPM replaces the prescribed gait and short-horizon force MPC;
the current-step convex force correction and joint-level mappings remain in the
execution layer.

## Development milestones

1. **Core formulation (implemented).** Exact first-derivative tests, standing
   solve, contact transitions, and planner output extraction.
2. **Horizon and terrain gates (implemented).** Required 20/40/60/80-node mesh
   tests, a 3 s two-foot problem, smooth-field/cross-slope cases, and an explicit
   recovery-only smooth-step stress case.
3. **Execution layer (implemented without robot assumptions).** Plan sampling,
   current-step convex force correction, friction projection, stance-force
   torque mapping, and swing-foot tracking.
4. **Native high-fidelity simulator bridge (implemented).** The pinned MuJoCo
   Menagerie Go1 adapter supplies calibrated state, foot Jacobians, bias
   torques, actuator limits, contact events, and direct IK initialization from
   the unmodified CITO plan. Flat, 2 cm sinusoidal, 10% slope, and both 3 s
   two-foot orderings pass closed-loop replay.
5. **Initial paired robustness layer (implemented).** Runtime plant-mass,
   terrain-friction, and timed-push perturbations use the same frozen plan. A
   uniform stance-force mode provides a force-reference ablation, and the
   pre-optimization feasible seed provides a fixed-schedule, terrain-projected
   baseline under the same WBC.
6. **Long-horizon high-fidelity bridge (implemented).** Both 100-node, 6 s
   contact orderings solve and replay on the shared sinusoidal terrain. The
   planner audit requires the moving feet to remain loaded, stationary, and on
   the terrain during the terminal support window.
7. **Initial all-leg route (implemented).** A 100-node, 6 s problem advances the
   base and all four feet. A schedule-free pairwise force margin prevents
   zero-margin diagonal support, and the resulting plans pass flat and shared
   sinusoidal-terrain MuJoCo replay.
8. **Measured-state continuous replanning (implemented).** The all-leg problem
   is now a callable planner. MuJoCo can execute an arbitrary number of
   independently solved 100-node plans from the same live robot state without a
   physics reset. The original two-plan flat gate remains, the sinusoidal gate
   uses audited multi-start after a forced reverse seed proved basin-sensitive,
   and a four-plan flat course now requires four successful solves, 16 accepted
   swing events, and at least 16 cm of net base travel.
9. **Audited seed exploration and asymmetric terrain (implemented as
   integration gates).** Four explicit contact-order seeds expose distinct
   optimized schedule fingerprints and seed-to-solution timing changes. A
   sequential multi-start policy accepts only solutions with exactly one
   unload/reload event per foot, then minimizes total seed-to-solution schedule
   deformation with tracking objective as the tie-breaker. A deterministic
   seeded two-mode analytic terrain is shared exactly with the MuJoCo height
   field; its two-plan gate requires eight accepted swing events.
10. **Repeated-contact scaling diagnostic (implemented, not accepted).** A
   220-node, 13.2 s experimental problem requests two swings per foot. The
   current solver stops after 118 iterations with line-search failure despite
   small candidate residuals (`5.58e-5` dynamics and `1.15e-5` MPCC). A prior
   dynamics-projected diagnostic produced excessive swing excursions and failed
   MuJoCo replay, so failed/projected iterates are not registered as passing
   tests or paper results.
11. **Seeded uneven-terrain development sweep (implemented).** The frozen
    10-seed, 2 cm random-smooth campaign now achieves planner feasibility in
    10/10 trials, physical two-segment completion in 8/10, and strict touchdown
    timing in 8/10. Terrain-relative swing shaping and plan-only multi-start
    ranking remove the previous planning exhaustion and pass the predeclared
    8/10 advancement gate without relaxing replay thresholds.
12. **Event-aware repeated handoff (implemented as a development result).**
    Confirmed early touchdown now promotes the measured foot to WBC support
    without changing the CITO plan used for audit and metrics. Replanning waits
    at least 2 s, then requires 25 consecutive acceptable stabilization samples,
    with a 3 s cap. The original seed-0 failure now completes all four segments.
13. **Longer-course development boundary (measured, not yet accepted).** On the
    frozen random-smooth seeds 0--4, all five trials obtain four audited plans
    when touchdown timing is recorded rather than used to terminate the course.
    Four of five complete the physical route, but only two of five satisfy the
    unchanged strict timing gate. Paper-scale evaluation remains deferred until
    touchdown-timing consistency improves without relaxing that gate.

## MuJoCo Go1 bridge

The optional MuJoCo bridge lives in `examples/quadruped_cito/mujoco`. It maps
the pinned Menagerie Go1 model to `WholeBodyState` and
`ConvexWBCCommand` by object name, not by model-array position. Its acceptance
test checks 100 random foot-Jacobian configurations, foot-velocity mapping,
joint-torque transmission, and foot-contact force direction before any
controller tuning. See `mujoco/DEPENDENCIES.md` for the pinned WSL setup and
build commands.

Standalone plan replay uses no footprint, mass, height, or terrain
transformations. CITO evaluates the same analytic surface used to populate
MuJoCo's height field. Swing references preserve CITO's planned clearance
relative to that surface while smoothing horizontal motion; terrain slope is
included in the vertical velocity reference. At a continuous replanning
boundary, the SRBD state keeps the measured base pose and velocity and measured
foot horizontal positions.
Stance-foot height is projected from MuJoCo's compliant, penetrated contact site
onto CITO's rigid foot-center contact manifold; the correction is printed and
the MuJoCo state is not modified. The flat, sinusoidal, and 10%
slope single-foot replays achieve 3.26--3.49 cm clearance, 1.64--1.82 cm
terminal landing error, and 60--62 ms touchdown error without actuator
saturation. Both native 50-node, 3 s rear-foot orderings pass: nominal base RMS
is 1.12--1.14 cm, maximum landing error is 9.6 mm, and maximum touchdown error
is 54 ms.

The deterministic development robustness matrix contains both orderings under
nominal dynamics, +/-10% mass, 25% friction reduction, a 2 N s lateral push,
and the combined +10% mass/friction/push condition. The full CITO-force
controller passes 12/12; the uniform-force ablation passes 2/12. This matrix is
a regression and controller-tuning result, not yet the randomized statistical
evaluation required for a paper claim.

The 100-node sinusoidal-terrain matrix adds the matched fixed-schedule baseline.
All three methods complete the task in 12/12 deterministic trials under the
same six conditions. With the current WBC and force-thresholded touchdown
detector, the strict one-knot timing gate passes 12/12 for ContactIPM, 5/12 for
the uniform-force ablation, and 10/12 for the fixed schedule. ContactIPM's
maximum touchdown error is 38 ms; the fixed schedule reaches 66 ms, and one
uniform-force case never establishes touchdown. This is evidence of execution
timing accuracy, not yet a stability or terrain-traversal success-rate
advantage.

The first all-leg development plan moves each foot about 8 cm and advances the
base 5.3 cm over 6 s. With the 20 N schedule-free support margin, CITO discovers
the four lift-off times and the unchanged plan passes both flat and sinusoidal
MuJoCo replay. On the smooth replay, maximum touchdown error is 46 ms, maximum
landing error is 9.9 mm, maximum post-touchdown slip is 6.6 mm, and actuator
saturation is zero. This is a deterministic integration gate, not yet a terrain
traversal result.

The continuous-replanning runner supports one to 32 independently solved
100-node segments and explicit forward, reverse, left-first, right-first,
cycling, and multi-start seed policies. Every boundary uses the measured state
after an all-stance WBC phase. The default boundary waits at least 2 s and
then requires all four contacts and the existing pose, velocity, orientation, and
slip gates to hold for 25 consecutive 2 ms samples; it gives up after 3 s. The
accepted duration and rigid-contact height correction are printed.
The four-segment flat audited multi-start task gate passes all 16 swing events and advances the
base 25.4 cm without a physics reset. The deterministic random-smooth seed-7
multi-start gate passes all eight events over two segments and advances 9.4 cm.
Its worst touchdown occurs at the registered 122 ms continuous-handoff limit.
These are integration cases, not a terrain-distribution robustness claim.

The schedule audit records the complete seeded and optimized contact masks,
64-bit schedule fingerprints, and per-foot lift-off/touchdown changes. It rejects
otherwise converged trajectories with an extra unload/reload event. Multi-start
solves all four seeds sequentially and reports every candidate. Among fully
audited candidates it selects the least total schedule deformation, using CITO
tracking objective only as a tie-breaker. Its total planning cost is therefore
the sum of candidate solve times, not the selected candidate's solve time alone.
It is currently an offline reliability mechanism, not a real-time planner
claim.

Across the frozen random-smooth development seeds 0--9, 80 multi-start
candidates were attempted. Four candidates ended in a raw solver failure and 26
otherwise solver-successful candidates were rejected by the independent
schedule audit; all 26 contained an extra per-foot contact event. All 20 segment
planning requests returned an audited plan. Eight of ten two-segment trials pass
both the physical task and strict timing gates. The two failures occur during
second-segment execution, with early terrain contact rather than planner
exhaustion. Median total multi-start planning time is 166.0 s per trial and the
maximum is 286.5 s. The worst observed touchdown error, landing error, and
post-touchdown slip are respectively 442 ms, 4.01 cm, and 2.16 cm; all maxima
include failed trials. These seeds were used for development and the result is
not final evaluation evidence.

The original four-segment random-smooth seed-0 probe exposed the repeated-handoff
failure: one foot contacted 276 ms early, two feet slipped 3.05 cm and 14.26 cm,
and one foot never established touchdown. Per-tick tracing isolated the causal
mismatch between the planned support mask and measured load-bearing contacts.
The execution layer now confirms a swing-foot contact for three samples after
minimum clearance has been observed, then gives that measured foot a zero-force
support reference. The immutable CITO reference still drives all plan metrics.
A bounded late-touchdown search is retained only as an opt-in ablation because
it worsened seed 0; it is disabled by default.

With early-contact feedback and confirmed adaptive stabilization, seed 0 passes
all four segments and all 16 swing events with strict timing. Its maximum
touchdown error is 86 ms, maximum landing error is 1.51 cm, maximum slip is
1.13 cm, and actuator saturation is zero. One early-contact promotion occurs.
In the matched ablation with early-contact feedback disabled, confirmed
stabilization still gives physical 4/4 completion, but strict timing fails with
a 174 ms maximum touchdown error and maximum slip rises to 1.81 cm. Thus the
handoff gate is the principal route-completion change on this seed, while
measured-contact promotion improves timing and execution margin.
Across the frozen five-seed development set, planner feasibility is 5/5 and
physical four-segment completion is 4/5. Strict timing is only 2/5: the failed
timing maxima are 172--256 ms against the unchanged 122 ms continuous-handoff
tolerance. No completed trial has a missing touchdown or large-slip recurrence.
This is a development boundary and causal implementation result, not a
terrain-distribution robustness claim.

The uncensored rerun at commit `c76c210` reproduces those exact aggregate
counts. Event-centered traces cover only the 300 ms before and after each
planned touchdown and record the planned, WBC-support, and measured contact
masks together with terrain gaps, vertical velocities, contact force, contact
blend, foot error, and base-wrench residual. The first strict violations are:

| Seed | Segment | Foot | Signed error | Desired/measured gap before contact | Desired/measured vertical speed | Outcome |
| --- | ---: | ---: | ---: | ---: | ---: | --- |
| 1 | 2 | 0 | -168 ms | 10.0/1.8 mm | -0.100/-0.174 m/s | Segment 2 task pass; segment 3 repeats at -256 ms and 3.72 cm landing error, so physical completion is 3/4 |
| 3 | 3 | 0 | -212 ms | 11.2/1.6 mm | -0.092/-0.186 m/s | Physical completion 4/4; timing 3/4 |
| 4 | 3 | 0 | -172 ms | 8.9/1.5 mm | -0.087/-0.168 m/s | Physical completion 4/4; timing 3/4 |

The sign convention is measured minus planned touchdown time, so all three are
early. Each first failure is front-left foot 0: the measured foot is already
7--10 mm closer to terrain, descends about 1.7--2.0 times faster than its
reference, has 2.0--2.7 cm position error, and has neither contact force nor WBC
contact blend immediately before impact. This rules out contact-detection lag
and reference/terrain intersection as the dominant mechanism. Seed 1's fourth
segment is the same mismatch at larger landing error, not an independent
failure.

One bounded landing-phase correction was evaluated without changing the solver,
schedule, terrain, or acceptance thresholds: add vertical swing damping only in
the final 2.5 cm when the measured foot leads and descends faster than its
reference. Across seeds 0--4 it achieved planner-complete 4/5, physical
completion 3/5, and strict timing 1/5. It improved seed 1 from physical 3/4 and
timing 2/4 to physical 4/4 and timing 3/4, but its last front-left touchdown
remained 214 ms early. Seed 0 regressed to physical 3/4, seeds 3 and 4 achieved
only 0/2 strict timing trials, and seed 4 regressed to physical 2/4 with planner
3/4. The correction was therefore rejected and is not present in the
controller.

The paired five-seed causal ablations are:

| Configuration | Planner-complete | Physical completion | Strict timing | Maximum landing error | Maximum slip |
| --- | ---: | ---: | ---: | ---: | ---: |
| Default early feedback + confirmed handoff | 5/5 | 4/5 | 2/5 | 3.72 cm | 1.48 cm |
| Landing-phase correction enabled | 4/5 | 3/5 | 1/5 | 3.27 cm | 1.56 cm |
| Early-contact feedback disabled | 4/5 | 2/5 | 0/5 | 3.70 cm | 67.85 cm |
| Historical fixed 2 s handoff | 4/5 | 3/5 | 3/5 | 3.65 cm | 1.43 cm |

Thus measured early-contact promotion is causally important for both physical
completion and timing on this development set. The confirmed handoff trades one
strict trial for planner and physical reliability; it remains the default, but
the five seeds are too few for a statistical claim. The landing correction and
late-touchdown search remain rejected. Because the unchanged default fails the
required strict-timing gate of at least 4/5, the controller is not frozen and
paper-scale robustness or evaluation-seed experiments must not begin yet.

## Experiment standard

### Trial protocol

- Keep 10 development seeds separate from evaluation seeds.
- Use at least 50 paired evaluation seeds per reported condition. Every method
  receives the same terrain, initial state, model perturbation, and disturbance.
- Set one wall-clock planning budget per experiment family. A timeout or solver
  failure counts as a failed trial; do not omit it from timing statistics.
- Treat physical route completion as the primary outcome. Report touchdown
  timing compliance separately; do not convert a task-successful traversal into
  a primary failure solely because a foot lands outside the diagnostic timing
  window.
- Save the terrain seed, full configuration, planned trajectory, contact events,
  solver statistics, controller diagnostics, and simulator state history.
- Define success before running the final data: reach at least 90% of the route,
  with no non-foot terrain collision, unrecovered body attitude beyond 60
  degrees, or controller/planner timeout.

The 3 s deterministic development matrix can be reproduced with:

```bash
python3 examples/quadruped_cito/mujoco/run_robustness.py \
  build-mujoco/quadruped_cito_mujoco \
  build-mujoco/go1_cito_terrain_scene.xml \
  build-mujoco/quadruped_cito_multifoot_forward.plan \
  build-mujoco/quadruped_cito_multifoot_reverse.plan \
  build-mujoco/quadruped_cito_robustness.csv
```

It writes one log per paired trial and a CSV containing pass/fail, base RMS,
clearance, landing/touchdown error, slip, and saturation. The CSV separates
task success from the stricter touchdown-timing gate.

After running the `QuadrupedCITOMujocoLongSmooth` CTests, reproduce the 6 s
three-method matrix with:

```bash
python3 examples/quadruped_cito/mujoco/run_robustness.py \
  build-mujoco/quadruped_cito_mujoco \
  build-mujoco/go1_cito_terrain_scene.xml \
  build-mujoco/quadruped_cito_long_smooth_forward.plan \
  build-mujoco/quadruped_cito_long_smooth_reverse.plan \
  build-mujoco/quadruped_cito_long_smooth_methods.csv \
  --terrain smooth \
  --fixed-forward-plan \
    build-mujoco/quadruped_cito_long_smooth_fixed_forward.plan \
  --fixed-reverse-plan \
    build-mujoco/quadruped_cito_long_smooth_fixed_reverse.plan
```

Build-directory outputs are intentionally not source artifacts.

Run a development sweep over deterministic analytic terrain seeds with:

```bash
python3 examples/quadruped_cito/mujoco/run_replanning_sweep.py \
  build-mujoco/quadruped_cito_mujoco \
  build-mujoco/go1_cito_terrain_scene.xml \
  build-mujoco/quadruped_cito_replanning_development.csv \
  --terrain random_smooth --terrain-amplitude 0.02 \
  --segments 2 --seed-policy multistart \
  --first-seed 0 --seed-count 10 \
  --trial-timeout 300
```

The sweep stores one complete log per seed and separates selected-candidate
solve time from total multi-start solve time. A failed solve, replay, or missing
course gate remains a failed CSV row. The CSV is atomically checkpointed after
every completed seed. By default, physical task success is the primary
acceptance gate and strict touchdown timing is a separate column; pass
`--require-timing` for a timing-gated regression campaign. The 300 s timeout
caps the complete solve-and-replay trial. Do not use these ten development seeds
as the final paper evaluation set.

### Terrain ladder

Advance only after the preceding level is reliable:

1. flat ground at 0.5, 1.0, 1.5, and 2.0 m/s;
2. smooth height fields with 2, 4, and 6 cm amplitude and 0.3--0.8 m wavelength;
3. ramps/cross-slopes at 10, 20, and 30 degrees;
4. steps and block fields with 4, 8, and 12 cm height changes;
5. stepping stones with 5, 10, and 15 cm gaps plus randomized lateral offsets;
6. mixed courses combining height, gap, slope, and low-friction patches.

Use progressive gates rather than tuning on the hardest course. Require 100
consecutive flat-ground trials before uneven-terrain evaluation, then require at
least 90% success at one difficulty before advancing to the next.

### Robustness sweeps

Apply both isolated and combined perturbations:

- mass +/-15% and principal inertia +/-10%;
- friction sampled over 0.4--0.9 while the planner uses a fixed nominal value;
- state-estimation noise and 0--20 ms observation/control latency;
- actuator strength +/-10%;
- lateral and longitudinal pushes at multiple gait phases;
- terrain-height bias and local missing/corrupted height samples.

### Comparisons and ablations

Use the same simulator, estimator, actuator model, and WBC for every method:

- fixed trot schedule with nominal footholds;
- fixed schedule with terrain-projected/optimized footholds;
- short-horizon ContactIPM schedule;
- proposed long-horizon ContactIPM schedule;
- proposed method without planned force feed-forward;
- proposed method without replanning, if replanning is included in the claim.

For a solver-oriented result, also run a general sparse NLP/contact-implicit
baseline on a smaller matched subset where its runtime is tractable.

### Required metrics

The primary result is paired traversal success versus terrain difficulty. Also
report:

- route completion, fall location, and time-to-failure;
- base pose/velocity RMS and peak tracking errors;
- touchdown position/time errors, foot slip distance, and unintended contacts;
- planned versus realized contact sequence and normal-force tracking error;
- actuator torque, speed, power, saturation fraction, and WBC infeasible cycles;
- CITO convergence rate, wall time, iterations, recovery usage, dynamics defect,
  inequality violation, and MPCC product;
- median, interquartile range, and 95th percentile for timing quantities.

Report Wilson 95% intervals for success rates and paired bootstrap intervals for
differences between methods. Publish failure cases and per-trial data alongside
aggregate plots.

## Immediate acceptance gates

- `ctest --test-dir build-quadruped --output-on-failure` passes completely.
- Analytic dynamics and constraint Jacobians agree with finite differences to
  the existing regression tolerances.
- Every planned trajectory is independently audited after solve; solver status
  alone is insufficient.
- Mesh-scaling, multi-foot, terrain, plan-sampling, and convex-WBC regressions
  pass with solve mode, iterations, time, problem storage, and independent
  residuals printed by the relevant executable.
- The MuJoCo convention, 60 s standing, three native single-foot terrain
  replays, both two-foot orderings, combined perturbation replay, and expected
  uniform-force ablation failure are automated. No plan-aware SRBD MPC is
  inserted between ContactIPM and the convex WBC.
