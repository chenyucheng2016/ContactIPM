# High-Fidelity Simulator Handoff

This contract is implemented by the optional MuJoCo Menagerie Go1 bridge. Go1
mass, composite CoM/inertia, and home foot geometry are calibrated directly;
CITO and MuJoCo consume the same terrain descriptor. Native single-foot and
two-foot plans pass without replay-time coordinate transformations or a second
SRBD MPC.

## Required adapter inputs

At every 250--500 Hz control tick, populate `WholeBodyState` with:

- base position and linear velocity in the world frame;
- quaternion `(w, x, y, z)` rotating body vectors into the world frame;
- base angular velocity in the body frame;
- each foot position and velocity in the world frame;
- each 3-by-3 leg Jacobian in the body frame, with
  `foot_velocity_body = J * joint_velocity`;
- each leg's joint-space gravity/Coriolis bias torque. Stance actuation uses
  `tau = tau_bias - J^T R^T f` because the CITO force is the ground reaction
  acting on the robot;
- optional swing-leg inverse-dynamics feed-forward torque. It may be zero for
  the first ideal-tracking tests.

The adapter must also provide:

- the terrain height/normal query used to construct the CITO problem;
- measured foot contacts and normal forces for event diagnostics;
- joint positions, velocities, applied torques, and actuator saturation flags;
- a monotonic control clock aligned to the plan start time.

All leg and joint ordering must be mapped explicitly. The planner uses
front-left, front-right, rear-left, rear-right and three joints per leg.

## Required adapter outputs

Apply the three joint torques per leg in `ConvexWBCCommand`. Log the associated
world-frame contact-force command, contact blend, force-solver iterations, and
wrench residual. Do not silently clip torque commands: report both requested
and applied torque and a saturation flag.

## Initial simulator acceptance sequence

1. **Convention checks.** At 100 random configurations, compare adapter foot
   positions and Jacobian directional derivatives against finite differences.
   Require maximum position derivative error below `1e-5` in SI units.
2. **Torque sign check.** With the robot fixed, apply small isolated joint
   torques and verify stance actuation agrees with
   `tau_bias - J^T R^T f`.
3. **Standing replay.** Use ideal state and flat terrain for 60 s. Require no
   fall, no NaNs, no WBC infeasible tick, and less than 2 cm RMS base-position
   error.
4. **Single-foot replay.** Execute the 20-node transition on flat, smooth, and
   10% sloped
   terrain. Require the planned foot to unload, clear terrain by at least 2 cm,
   land within 3 cm and one planning knot plus one simulator tick of plan, and
   slip less than 2 cm after touchdown.
5. **Two-foot replay.** Execute both valid 3 s warm-start orderings. Require the
   realized contact order to match the selected CITO solution and complete
   without an unplanned body contact.
6. **Terrain gates.** The native bridge currently automates a 2 cm analytic
   sinusoid and 10% slope. Larger fields, cross-slopes, and the recovery-only
   smooth step remain progression gates rather than completed execution claims.

The ideal-state gates now pass, so the bridge exposes deterministic mass,
friction, and timed-push perturbations. Measured collision confirms stance
activation; when scheduled touchdown is late, swing tracking continues until
contact instead of dropping the foot task at the planned event.

## Closed-loop robustness sequence

After ideal replay, add one perturbation family at a time:

- mass `+/-15%` and principal inertia `+/-10%`;
- friction mismatch over `0.4--0.9`;
- observation/control delay from 0 to 20 ms;
- state and terrain-height noise;
- actuator strength `+/-10%`;
- phase-randomized lateral and longitudinal pushes;
- early and late touchdown events.

Early contact should switch that leg to measured stance handling. Late contact
should continue the swing reference with bounded extension and trigger a replan
or controlled stop after a fixed timeout. These policies depend on simulator
contact data and therefore do not belong in the current simulator-independent
module.

## Logging contract

Save one record per control tick containing:

- plan time/status and sampled stage/phase;
- desired and measured base/foot state;
- planned, corrected, and measured contact forces;
- planned and measured contact flags and event times;
- requested/applied joint torque, joint speed, power, and saturation;
- WBC iterations and wrench residual;
- CITO status, iterations, solve time, dynamics defect, inequality violation,
  and MPCC product;
- terrain/configuration seed and complete controller parameters.

The adapter is complete only when these logs are sufficient to reproduce every
reported success, failure, timing value, and contact-event error.
