# OCS2: Contact Constraints & Switched Times in the IPM-based Nonlinear MPC Solver

A code-grounded walkthrough of how OCS2 formulates and solves optimal control problems with
**(a) contact/hybrid constraints** and **(b) switched (event) times**, and how both are integrated
into the **multiple-shooting interior-point (IPM) solver** in `ocs2_ipm`.

---

## 1. The OCP data structure: where contact/switch hooks live

Everything is declared in one struct, `OptimalControlProblem`
(`ocs2_oc/include/ocs2_oc/oc_problem/OptimalControlProblem.h`). The problem is **explicitly
partitioned by location along the horizon**:

- **Intermediate** nodes (continuous-flow intervals): `costPtr`, `equalityConstraintPtr`,
  `inequalityConstraintPtr`, `stateInequalityConstraintPtr`, …
- **Pre-jump (event)** nodes: `preJumpCostPtr`, `preJumpEqualityConstraintPtr`,
  `preJumpInequalityConstraintPtr`, …
- **Final** node: `finalCostPtr`, `finalEqualityConstraintPtr`, `finalInequalityConstraintPtr`.

This tri-partition is exactly what lets the solver treat "an event happens here" as a structurally
distinct node type rather than a special case buried inside the dynamics.

Constraints come in four interface families in `ocs2_core/constraint/`:

- `StateConstraint` (state-only) and `StateInputConstraint` (state+input)
- Each carries a `ConstraintOrder` (`Linear` or `Quadratic`, `ConstraintOrder.h`)
- Each implements `isActive(time)`, `getNumConstraints(time)`, `getValue(...)`,
  `getLinearApproximation(...)`, and optionally `getQuadraticApproximation(...)`.

Note `isActive(time)` — this is the primary mechanism by which **contact-dependent constraints
turn on/off per mode** (see §4).

---

## 2. Switched systems & event/switch times — the formulation layer

### 2a. The `ModeSchedule`

A switched system is a sequence of modes separated by **event times**
(`ocs2_core/reference/ModeSchedule.h`):

```
eventTimes   : size N-1      (the switch times)
modeSequence : size N        (mode index for each interval)
modeAtTime(t): lower-count rule if t == an event time
```

The mode number itself encodes the contact configuration. For the quadruped
(`MotionPhaseDefinition.h`), each mode is a 4-bit bitmask of which feet are in stance
(`stanceLeg2ModeNumber`), with `FLY=0` (all swing) and `STANCE=15` (all contact). So "switched
system" and "contact schedule" are the **same object**: a change in the contact pattern is a new
mode, and the boundary between two contact patterns is an event time.

### 2b. The dynamics interface has jump maps & guard surfaces

`ControlledSystemBase` / `SystemDynamicsBase` (`ocs2_core/dynamics/`) provide the hybrid-system
primitives:

- `computeFlowMap(t,x,u)` — continuous dynamics within a mode.
- `computeJumpMap(t,x)` — the **state reset** at an event (e.g. an impulse/velocity reset on
  contact, or identity if the state is continuous).
- `computeGuardSurfaces(t,x)` — scalar surfaces whose zero-crossing *detects* an event (used only
  by state-triggered rollout).
- `jumpMapLinearApproximation(t,x)`, `guardSurfacesLinearApproximation(t,x,u)` — derivatives the
  optimizer needs.

Crucially, **the multiple-shooting solvers do not use guard surfaces or root-finding** — they
consume the *given* `ModeSchedule` (event times are inputs from the `ReferenceManager`, not
decision variables in `ocs2_ipm`). Guard surfaces are only used by the simulation/rollout path.

---

## 3. Switched times inside the IPM solver

### 3a. Time discretization injects event times as special nodes

`IpmSolver::runImpl` (`ocs2_ipm/src/IpmSolver.cpp:211`) starts every solve by pulling the event
times from the reference manager and building an annotated grid:

```cpp
const auto& eventTimes = this->getReferenceManager().getModeSchedule().eventTimes;
const auto timeDiscretization = timeDiscretizationWithEvents(initTime, finalTime, settings_.dt, eventTimes);
```

`timeDiscretizationWithEvents` (`TimeDiscretization.cpp`) walks the horizon in steps of `dt`, but:

1. Whenever a step would cross an event time, it **snaps the node to the event time** and tags it
   `Event::PreEvent`.
2. Each `PreEvent` node is immediately duplicated as a `PostEvent` node at the same timestamp.
3. Sub-`dt_min` intervals are merged to avoid degenerate slivers.

So at an event, the grid looks like:

```
... t_k  [PreEvent t_e]  [PostEvent t_e]  t_{k+1} ...
```

The `AnnotatedTime::Event` enum (`None / PreEvent / PostEvent`) and helpers
`getIntervalStart/End/Duration` apply `±weakEpsilon` shifts so that the interval on each side of an
event is well-defined and interpolations don't straddle a discontinuity.

### 3b. Event nodes = jump-map equality constraints

The QP is assembled per-node in `setupQuadraticSubproblem` (`IpmSolver.cpp:606`). The dispatch is
the heart of the switched-system treatment:

```cpp
if (time[i].event == AnnotatedTime::Event::PreEvent) {
  auto result = multiple_shooting::setupEventNode(ocpDefinition, time[i].time, x[i], x[i + 1]);
  ...
} else {
  // normal intermediate node with flow-map dynamics
  auto result = multiple_shooting::setupIntermediateNode(..., ti, dt, x[i], x[i + 1], u[i]);
}
```

In `setupEventNode` (`Transcription.cpp:156`), the **dynamics equality constraint is the linearized
jump map** instead of the integrated flow map:

```cpp
dynamics = ocp.dynamicsPtr->jumpMapLinearApproximation(t, x);
dynamics.f -= x_next;            // residual: dx_{k+1} = jumpMap(x) - x_next
dynamics.dfdu.setZero(x.size(), 0);  // no input at an event node
```

So a switch is enforced as a multiple-shooting defect `x⁺ = jumpMap(x⁻)` connecting the pre- and
post-event states, with **no input variable** at that node (input arrays are sized 0 there — see
`incrementTrajectory`'s `if (dv[i].size() > 0)` guard in `Helpers.h`). Event nodes also evaluate
`preJumpCostPtr`, `preJumpEqualityConstraintPtr`, and `preJumpInequalityConstraintPtr`.

The terminal node is handled by `setupTerminalNode`, and the initial node has its state-only
inequality constraints explicitly disabled (`result.stateIneqConstraints.setZero(0,…)` at
`IpmSolver.cpp:660`) since `x[0]` is fixed.

### 3c. Trajectory spreading: warm-starting across moving event times

Because MPC re-solves every cycle with a possibly-shifted gait schedule, the solver warm-starts
from the previous solution and reconciles it with the new event times via **Trajectory Spreading**
(`TrajectorySpreading.h`, used in `IpmSolver::runImpl`):

```cpp
if (!primalSolution_.timeTrajectory_.empty()) {
  std::ignore = trajectorySpread(oldModeSchedule, newModeSchedule, primalSolution_);
}
...
if (!slackIneqTrajectory_.timeTrajectory.empty()) {
  std::ignore = trajectorySpread(oldModeSchedule, newModeSchedule, slackIneqTrajectory_);
  std::ignore = trajectorySpread(oldModeSchedule, newModeSchedule, dualIneqTrajectory_);
}
```

It matches old↔new event times, erases the trajectory tail past the first mismatched mode, and
**spreads (holds constant) the last valid value** across the inserted/removed mode interval —
applied uniformly to states, inputs, slacks, and duals. `initializeSlackDualTrajectory`
(`IpmSolver.cpp:390`) additionally uses `findIntersectionToExtendableInterval` to interpolate the
cached dual/slack solution only over times where the old mode schedule is still valid, and
re-initializes (via IPOPT-style bounds, §5) elsewhere.

---

## 4. Contact constraints in practice — the switched-model example

The clearest illustration is the ANYmal/quadruped stack. `QuadrupedPointfootInterface.cpp:40`
adds, **per leg**, three state-input equality constraints:

```cpp
for (int i = 0; i < NUM_CONTACT_POINTS; i++) {
  problemPtr_->equalityConstraintPtr->add(footName + "_ZeroForce", createZeroForceConstraint(i));
  problemPtr_->equalityConstraintPtr->add(footName + "_EENormal",  createFootNormalConstraint(i));
  problemPtr_->equalityConstraintPtr->add(footName + "_EEVel",     createEndEffectorVelocityConstraint(i));
}
```

The contact-awareness is entirely in `isActive(time)`:

- **`ZeroForceConstraint`** (`ZeroForceConstraint.cpp`): `isActive = !getContactFlags(time)[leg]`.
  When a leg is in **swing**, it forces that leg's 3 contact-force input components to zero
  (`dfdu` = identity on those rows). In stance it is inactive, so the optimizer is free to produce
  ground-reaction forces.
- **`FootNormalConstraint`** (`FootNormalConstraint.h`): a hybrid
  `A_p·position + A_v·velocity + b = 0` in the terrain-normal direction — active in *both* stance
  (pin foot to terrain) and swing (track the swing trajectory's normal component).
- **`FrictionConeConstraint` / `FrictionConeCost`** (`constraint/FrictionConeConstraint.h`): the
  second-order cone `μ(Fz+gripper) − √(Fx²+Fy²+regularization) ≥ 0`, with analytic local
  derivatives (`ConeLocalDerivatives`) and a `hessianDiagonalShift` to keep the quadratic
  approximation strictly convex. It's added as a **soft cost** (`createFrictionConeCost`, penalized
  by a relaxed-barrier) rather than a hard inequality.

The contact schedule itself is produced by `SwitchedModelModeScheduleManager` (a `ReferenceManager`
subclass). Its `modifyReferences` (`SwitchedModelModeScheduleManager.cpp:20`) advances the
`GaitSchedule` to the current time and emits a `ModeSchedule` covering the horizon; the
`SwingTrajectoryPlanner` then computes swing-foot reference trajectories from that schedule. Because
`isActive(time)` queries `getContactFlags(time) → modeNumber2StanceLeg(modeAtTime(time))`, **the
same static OCP definition correctly activates/deactivates constraints as the solver walks the time
grid through different modes** — no reformulation is needed when the gait changes.

This is the central design idea: **contact is not modeled as a separate solver feature; it is modes
+ time-activated constraints + a jump map**, all expressed through the generic OCP interface.

---

## 5. The IPM treatment of inequality constraints (the "interior point" part)

The IPM handles **inequality** constraints (state-only, state-input, pre-jump, terminal) via
slack/dual variables and a logarithmic barrier. Equalities (including the contact equality
constraints above and the dynamics/jump-map defects) go directly into the HPIPM QP.

### 5a. Condensing inequality constraints into the Lagrangian

For each node, after building the LQ approximation, `ipm::condenseIneqConstraints`
(`IpmHelpers.cpp:37`) eliminates the slack/dual directions from the KKT system by folding their
effect into the Lagrangian's gradient and Hessian:

```cpp
const vector_t condensingLinearCoeff    = (dual.array()*ineq.f.array() - μ) / slack.array();
const vector_t condensingQuadraticCoeff = dual.cwiseQuotient(slack);
lagrangian.dfdx  -= ineq.dfdxᵀ·dual;
lagrangian.dfdx  += ineq.dfdxᵀ·condensingLinearCoeff;
lagrangian.dfdxx += ineq.dfdxᵀ·diag(condensingQuadraticCoeff)·ineq.dfdx;
// (and the dfdu / dfduu / dfdux terms when an input is present)
```

This is the standard primal-dual IPM condensing (Wächter–Biegler / IPOPT form). Equality constraints
and dynamics are *not* condensed — they remain as explicit linear constraints handed to HPIPM. Both
state-only and state-input inequalities are condensed at intermediate nodes; only state-only at
event and terminal nodes.

### 5b. Slack/dual Newton directions and step sizes

After HPIPM solves the condensed QP for `δx, δu` (`getOCPSolution`, `IpmSolver.cpp:470`), the
slack/dual directions are **recovered** in closed form (`IpmHelpers.cpp`):

```cpp
δslack = ineq.f − slack + ineq.dfdx·δx + ineq.dfdu·δu;        // retrieveSlackDirection
δdual  = (dual ⊙ (slack + δslack) − μ) / (−slack);            // retrieveDualDirection
```

Step sizes use the **fraction-to-boundary** rule (`fractionToBoundaryStepSize`, default margin
`0.995`) so slacks and duals never leave the positive orthant. There are separate primal and dual
step sizes; `usePrimalStepSizeForDual` (default true) couples them by taking the min.

### 5c. Barrier schedule, initialization, convergence

- The barrier `μ` starts at `initialBarrierParameter` (1e-2) and is driven toward
  `targetBarrierParameter` (1e-4). `updateBarrierParameter` reduces μ (linear `×0.2` or superlinear
  `μ^1.5`, whichever is smaller) only once the merit change and constraint violation fall below
  `barrierReductionCostTol`/`barrierReductionConstraintTol`.
- Slack/dual are initialized IPOPT-style (`IpmInitialization.cpp` + `IpmSettings.h`):
  `initializeSlackVariable` pushes the constraint value to be strictly positive with
  `initialSlackLowerBound` (1e-4) and `initialSlackMarginRate` (0.01); duals initialized from
  `slack, μ` similarly. There are dedicated initializers for intermediate, terminal, **and event**
  nodes (`initializeEventSlackVariable` queries `preJumpInequalityConstraintPtr`).
- Convergence (`checkConvergence`) accepts on iteration cap, tiny step (`alpha_min`),
  merit/feasibility tolerance, or primal-step tolerance once μ has hit target.
- **Filter line search** (`FilterLinesearch`, `takePrimalStep`) accepts steps using a bi-criteria
  (constraint violation vs. merit) filter, with Armijo backtracking (`alpha_decay=0.5`). The
  barrier-perturbed complementary-slackness residual `(slackᵢ·dualᵢ − μ)` is folded into the
  dual-feasibility SSE used by the filter.

### 5d. No special "contact" code in the solver

A key observation: **nothing in `ocs2_ipm` mentions contacts, feet, friction, or gaits.** The
solver is generic. All contact-specific behavior enters through:

1. the `ModeSchedule`/event times → the time grid and jump-map nodes, and
2. constraint objects whose `isActive(time)` flips with the contact flags.

The only structural accommodations to *switched/hybrid* systems are the `PreEvent/PostEvent` node
type, the jump-map dynamics in `setupEventNode`, the `±weakEpsilon` interval bookkeeping, and the
trajectory-spreading warm start.

---

## 6. How switched *times* could become decision variables (the rollout path)

In `ocs2_ipm` the event times are **fixed inputs** for each solve (the gait schedule is decided
outside the QP). The codebase *does* contain machinery to treat event times as arising from state
conditions — but it lives on the **rollout/simulation** side, not the IPM side:

- `StateTriggeredRollout` (`StateTriggeredRollout.cpp`) integrates the flow map, evaluates
  `computeGuardSurfaces` each step, throws an event when a surface crosses zero, then uses a
  `RootFinder` (bracket + bisect/Newton variants) to refine the crossing time, applies
  `computeJumpMap`, and appends to `modeSchedule.eventTimes` / `postEventIndices`. This is the
  classic guard-surface/hybrid-time integration scheme.

So OCS2 supports two philosophies: **time-triggered** (event times prescribed by a gait schedule —
what the MPC solvers use) and **state-triggered** (events discovered during rollout via guard
surfaces). The DDP family can optimize over event times as parameters; the multiple-shooting
IPM/SQP/SLP solvers use the prescribed-schedule form.

---

## TL;DR

- **Contact** is modeled as a **mode schedule** (which feet are in stance per time interval) plus
  **time-activated constraints** (`isActive(time)` reads contact flags) and a **jump map** for state
  resets. Friction cones enter as soft (relaxed-barrier) costs; stance/swing force and foot-position
  requirements enter as state-input equality/inequality constraints that toggle per mode. There is
  no contact-specific code in the solver.
- **Switched times** enter the IPM solver as a **fixed event-time set** from the
  `ReferenceManager`. `timeDiscretizationWithEvents` inserts paired `PreEvent/PostEvent` grid nodes;
  `setupEventNode` turns each event into a **jump-map equality defect** (no input there), and
  `±weakEpsilon` interval rules keep interpolations well-defined. Trajectory spreading warm-starts
  states/inputs/slacks/duals when the gait shifts between MPC cycles.
- **The IPM** is a multiple-shooting primal-dual interior-point method: equalities (dynamics, jump
  maps, contact-equality constraints) are handed to HPIPM as a structured QP; inequalities get
  slack/dual variables, are **condensed** into the Lagrangian with the barrier `μ`, and their Newton
  directions + fraction-to-boundary steps are recovered post-QP. A filter line search on (merit,
  constraint-violation) accepts steps, and μ is driven to its target. Event, intermediate, and
  terminal nodes each have their own initialization/transcription/metrics paths, so the hybrid
  structure is handled uniformly but explicitly throughout.
