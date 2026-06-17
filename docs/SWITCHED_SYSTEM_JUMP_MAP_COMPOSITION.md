# Switched System Support via Jump-Map Composition

## Why This Design

An earlier draft proposed replacing flow dynamics with the jump map at pre-event nodes (setting B=0). That approach has three fatal problems:

1. **Physics deletion.** `x_{k+1} = x_k` at pre-event stages discards the real RK4 integration step. OCS2's identity jump is only valid because it pairs zero-duration PreEvent/PostEvent nodes — our fixed-size `StageData[N+1]` cannot represent zero-duration nodes.

2. **Riccati B=0 trap.** At `nmpc_riccati.hpp:134`, `reg = max(reg_base, 1e-10 * s_max_abs)` collapses to `1e-12` when `S=0`, the LDLT floor silently "succeeds," and gains explode to `~1e12` with no `KKT_SINGULAR` flag. Survivable today only because the trot keeps `Quu = 13*Rf*I > 0`.

3. **Ill-posed QP.** Zeroing B at k=11 removes a real causal DoF (the contact force that sets the COM velocity entering the next phase) and leaves `u_11` in a cost-only subproblem disconnected from continuity.

---

## Core Design: Compose Jump with Flow (flow-then-jump)

At a pre-event stage, model the interval as **flow-then-jump**:

```
x_{k+1} = jumpMap( f(x_k, u_k, dt, phase=k) )

defect:  c_k = jumpMap(x_flow_post) - x_{k+1}      where x_flow_post = f(x_k, u_k, dt)
A_k = J_jump(x_flow_post) * A_flow                  (chain rule)
B_k = J_jump(x_flow_post) * B_flow
```

This:
- **Preserves real integration** — u_k affects defect through B_flow. No causal DoF removed, no ill-posedness.
- **Keeps B != 0 everywhere** — S = Quu + B'PB stays well-conditioned. No Riccati changes needed.
- **Reduces to current behavior for identity jump** — Jg=I means A_k=A_flow, B_k=B_flow, c_k=f-x_{k+1}. Zero risk to pendulum/chain/quadrotor.
- **Extends to real resets** — velocity impulses on footstrike use the same chain-rule composition.

This matches OCS2's intent (`jumpMapLinearApproximation` composed with the integrated flow) without requiring dynamic grid resizing.

---

## Task 1: Add JumpModel + Metadata (`nmpc_problem.hpp`)

Add a separate optional interface (mirror OCS2's `computeFlowMap`/`computeJumpMap` split — do not pollute `DynamicsModel`):

```cpp
template <int NX>
struct JumpModel {
    virtual ~JumpModel() = default;
    // x_post = g(x_flow_post)  where x_flow_post = f(x_k,u_k,dt)
    virtual Status jump_map(const Vec<NX>& x_flow_post, Vec<NX>& x_post) = 0;
    // J = dg/dx evaluated AT x_flow_post (NOT at x_k)
    virtual Status jump_jacobian(const Vec<NX>& x_flow_post, Mat<NX,NX>& J) = 0;
};
```

In `NMPCProblem` (after `stages[HORIZON+1]`):

```cpp
bool is_pre_event[HORIZON + 1] = {};        // false everywhere by default
JumpModel<NX>* jump_model = nullptr;        // null = no jumps (all benchmarks today)
```

Compile-time-sized, no heap. Note the Jacobian is evaluated at `x_flow_post`, not `x_k` — this corrects an ambiguity in the original plan's signature.

---

## Task 2: Compose Jump in `evaluate_model()` (`nmpc_ipm_paper.hpp` lines 867-879)

```cpp
// Dynamics defect: c_k = f(x_k,u_k) - x_{k+1}, then compose jump if pre-event
prob_->dynamics->discrete_step(s[k].x, s[k].u, prob_->dt, fk, k);
prob_->dynamics->linearize(s[k].x, s[k].u, prob_->dt, s[k].A, s[k].B, k);

if (prob_->is_pre_event[k] && prob_->jump_model) {
    Vec<NX> x_post;  Mat<NX,NX> Jg;
    prob_->jump_model->jump_map(fk, x_post);            // g evaluated at flow output
    prob_->jump_model->jump_jacobian(fk, Jg);
    // c = g(f) - x_{k+1}
    for (int i = 0; i < NX; ++i) s[k].c[i] = x_post[i] - s[k+1].x[i];
    // A = Jg * A_flow ,  B = Jg * B_flow   (in-place: A <- Jg*A, B <- Jg*B)
    s[k].A = mat_mul(Jg, s[k].A);
    s[k].B = mat_mul(Jg, s[k].B);
} else {
    for (int i = 0; i < NX; ++i) s[k].c[i] = fk[i] - s[k+1].x[i];
}
```

Add a small `mat_mul` helper if one doesn't exist — check `nmpc_core.hpp` first; `Mat` is column-major. Cost (`Qxx`/`Quu`/`Qux`/`qx`/`qu`) and constraints are unchanged — they already use the existing `stage_cost(x_k,u_k,k)` / `evaluate(x_k,u_k,k)` interfaces, which is correct because the outgoing contact config at stage k is `get_phase(k)`.

---

## Task 3: Compose Jump in `IPMTrialEvaluator::evaluate` (line ~2178)

The trial evaluator recomputes the nonlinear defect per line-search alpha. Apply the same composition so the filter's theta (constraint violation) is consistent:

```cpp
sv->prob_->dynamics->discrete_step(xk_t, uk_t, sv->prob_->dt, fk, k);
if (sv->prob_->is_pre_event[k] && sv->prob_->jump_model) {
    Vec<NX> gj;  sv->prob_->jump_model->jump_map(fk, gj);
    for (int i = 0; i < NX; ++i) out_theta += std::fabs(gj[i] - xkp1_t[i]);
} else {
    for (int i = 0; i < NX; ++i) out_theta += std::fabs(fk[i] - xkp1_t[i]);
}
```

SOC (`compute_soc`) swaps `riccati_stages_[k].c`; since that `c` is now the composed defect from Task 2, SOC is automatically consistent — no change needed there.

---

## Task 4: Centroidal Trot IdentityJump + Warm-Start Fix (`centroidal_trot_nmpc.cpp`)

The trot's real jump is identity (continuous state), so this stage is a correctness/extensibility validation, not a convergence driver:

```cpp
struct IdentityJump : JumpModel<NX> {
    Status jump_map(const Vec<NX>& x_flow_post, Vec<NX>& x_post) override {
        x_post = x_flow_post; return Status::SUCCESS;  // g(x)=x
    }
    Status jump_jacobian(const Vec<NX>&, Mat<NX,NX>& J) override {
        J.set_identity(); return Status::SUCCESS;
    }
};
// in main():
IdentityJump jump;
prob.jump_model = &jump;
prob.is_pre_event[11] = prob.is_pre_event[24] = prob.is_pre_event[36] = true;
```

**Expected result: identical trajectory and iteration count to current (~151 iters).** This is the success criterion — it proves the composition is wired correctly without changing physics. If iterations change, something is wrong.

### Warm-Start Fix (lines 559-574)

Add the jump after the flow rollout so the initial guess respects the jump map:

```cpp
dyn.discrete_step(prob.stages[k].x, prob.stages[k].u, DT, nx, k);
if (prob.is_pre_event[k] && prob.jump_model)
    prob.jump_model->jump_map(nx, nx);   // in-place for identity; composes for real jumps
prob.stages[k+1].x = nx;
```

---

## Task 5: Optional Defensive Guard in `nmpc_riccati.hpp`

Because the composition keeps B != 0, the Riccati needs no change for this plan to be correct. However, add a defensive guard at line ~134 so future direct-B=0 setups fail loudly instead of corrupting silently:

```cpp
if (s_max_abs < 1e-12) {
    // S is identically zero — control is fully decoupled. Skip elimination
    // (treat du as fixed) rather than dividing by the reg floor.
    // [documented escape hatch; not hit by the composed-jump design]
}
```

This closes the trap the original plan would have walked into. Implement only if you agree it's worth the defensive code; it's strictly optional for the trot.

---

## Task 6: Validation

1. Build and run pendulum, chain_mass, quadrotor — must be bit-identical (`jump_model=nullptr`, `is_pre_event` all false means no code path changes).
2. Run centroidal trot with `IdentityJump` at k=11/24/36 — must match current iterations (~151) within +/-1-2. This proves the plumbing.
3. Verify linear KKT residuals stay at machine precision (existing diagnostic `compute_linear_kkt_residual`).
4. *(Optional extensibility demo)* A tiny synthetic test with a non-identity Jg (e.g. negate one velocity component) to confirm `A_k = Jg * A_flow` composes correctly and the solve still converges.

---

## Files Touched

1. `include/nmpc/nmpc_problem.hpp` — add `JumpModel<NX>`, `is_pre_event[]`, `jump_model*`
2. `include/nmpc/nmpc_ipm_paper.hpp` — compose jump in `evaluate_model()` and `IPMTrialEvaluator::evaluate`
3. `examples/centroidal_trot_nmpc.cpp` — IdentityJump, wire is_pre_event, warm-start fix
4. `include/nmpc/nmpc_riccati.hpp` — optional defensive B=0 guard only

---

## Why This Is the Right Scope for "Generic Switched-System Support"

The composition lays down the exact machinery future non-identity jumps need — a `JumpModel` with `jump_map` + `jump_jacobian`, applied at the correct point (post-flow, with chain-ruled Jacobians), threaded through both the model evaluation and the line-search trial evaluator, and respected by the warm-start. Adding a real footstrike impulse later is then just writing a non-identity `JumpModel` subclass — zero solver changes. The original plan's shortcut (replace flow with identity jump, zero B) would have had to be undone before any of that could work.
