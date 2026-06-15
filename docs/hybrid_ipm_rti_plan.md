
# Hybrid IPM/RTI Implementation

## Solver Mode Switch

This is a **solver-level mode switch**, not a minor optimization toggle:

- **Off** (`hybrid_rti_enabled = false`): Run the baseline full-horizon IPM exactly as before.
- **On** (`hybrid_rti_enabled = true`): One outer tail condensation sweep produces frozen terminal cost `(P_M, p_M)`, then IPM iterates on active stages `0..M-1` only. This is an **approximate real-time mode** — it solves a *different* optimization problem (barrier-softened tail + frozen linearization) than the exact full-horizon IPM.

The tail's `(P_M, p_M)` is computed **once per MPC sample**, outside the IPM iteration loop.

**Reference**: Zanelli et al., "A partially tightened real-time iteration scheme" (CDC 2017).

---

## Two Active Ranges

Two distinct ranges govern loop bounds throughout the solver:

- **State/dynamics range** `0..N_active_`: stages with valid `x`, `dx`, `A`, `B`, `c`. This is `N_active_ + 1` nodes.
- **Constraint/slack/dual range** `0..K`: stages with real, IPM-maintained `s`, `λ`, `ds_`, `dλ_`.

```cpp
const int K = tail_computed_ ? (N_active_ - 1) : N_active_;
```

When `tail_computed_`, node `N_active_` is a pure terminal node from the tail condensation — no constraints, no slacks (Zanelli eq. 13). When `tail_computed_` is false, node `N_active_` is the true terminal stage with `evaluate_terminal` constraints.

**Verification** (non-hybrid): `N_active_ = HORIZON`, `tail_computed_ = false`, `K = HORIZON`. Every bound below collapses to existing behavior. ✓

**Verification** (hybrid): `tail_computed_ = true`, `K = N_active_ - 1`. Node `N_active_` excluded from constraint/slack handling but still gets its state updated. ✓

---

## Task 1: Create TailSolver Module

**New file**: `include/nmpc/nmpc_tail_solver.hpp`

The TailSolver reads pre-populated stage data from `prob_->stages[]` (output of `evaluate_model()`) and performs:
1. Barrier penalty addition: `-τ·Σlog(-g_j)` modifies gradient and Hessian
2. Riccati backward sweep over tail stages `[M, HORIZON-1]` producing `(P_M, p_M)`
3. Stores feedback gains for forward expansion

### Class structure
```cpp
template<int NX, int NU, int NC, int HORIZON>
class TailSolver {
public:
    struct TerminalCost {
        SymMat<NX> P;
        Vec<NX>    p;
        bool       success;  // explicit feasibility flag
    };

    TerminalCost compute(StageData<NX,NU,NC> stages[], int M, double tau);
    void forward_expansion(const Vec<NX>& x_M, StageData<NX,NU,NC> stages[], int M);

private:
    // Feedback gains in DELTA coordinates: du = -K·dx - d
    Mat<NU, NX> K_stored_[HORIZON];
    Vec<NU>     d_stored_[HORIZON];
    // Linearization points for forward expansion
    Vec<NX>     x_bar_[HORIZON];
    Vec<NU>     u_bar_[HORIZON];

    // Workspace (no heap allocation)
    SymMat<NX> P_curr_, P_next_;
    Vec<NX>    p_curr_, p_next_;

    bool add_barrier_penalty(/* reads stages[k].d, .Cx, .Cu; modifies Qxx/qx/Quu/qu/Qux */);
    bool backward_step_tail(/* A, B, c, Qxx, Quu, Qux, qx, qu, P_next → P_curr, K, d */);
};
```

### Key implementation details

- **Barrier penalty**: Reads `stages[k].d[j]` (constraint value at linearization point). If any `d[j] >= 0`, returns `success=false`. Gradient adds `(τ/|g_j|)·C_{j,:}`, Hessian adds `(τ/g_j²)·C_{j,:}^T·C_{j,:}`. First-order barrier Hessian exact for linear constraints.
- **Column-by-column LDLT solve**: `SymMat::ldlt_solve` takes `Vec<N>&` only, so `K = S⁻¹G` solved column-by-column.
- **LDLT retry with escalating regularization**: `backward_step_tail` must follow the same pattern as `RiccatiSolver::backward_lhs` — up to 6 attempts with `reg *= 10.0`. Factorization failure → `success=false` (triggering fallback).
- **Mat → SymMat conversion**: Stage data uses `Mat<NX,NX>` for `Qxx`. After barrier penalty, convert via `SymMat::copy_lower_from(Mat)`.
- **Index mapping**: `K_stored_[k - M]`, `d_stored_[k - M]`, `x_bar_[k - M]`, `u_bar_[k - M]` for stage `k`.

### Forward expansion gain convention

`backward_step_tail` produces gains in **delta coordinates** (matching `compute_Pk`/`compute_pk` in nmpc_riccati.hpp): `du_k = -K_k · dx_k - d_k`

`forward_expansion` uses the same convention with saved linearization points:
```cpp
dx_k = x_k - x_bar_[idx]
du_k = -K_stored_[idx] · dx_k - d_stored_[idx]
u_k  = u_bar_[idx] + du_k
```

---

## Task 2: Add Hybrid Parameters to PaperIPMParams

**File**: `include/nmpc/nmpc_ipm_paper.hpp` (in `PaperIPMParams`, ~line 74)

```cpp
// === Hybrid IPM/RTI (Partial Tightening) ===
bool   hybrid_rti_enabled = false;   // mode switch: off = full IPM, on = hybrid
int    active_horizon = 0;            // M: active stages [0, M). 0 = use HORIZON
double tau_tail = 0.1;                // barrier parameter for tail penalty
```

**Guard**: `active_horizon=0` means "use full horizon." Do NOT `clamp(0, 1, HORIZON)` — that silently gives M=1.

---

## Task 3: Modify Riccati Recursion — Add N_active Only

**File**: `include/nmpc/nmpc_riccati.hpp`

### Simplification: no P_terminal/p_terminal pointer plumbing

The KKT builder (`build_kkt_lhs`/`build_kkt_rhs` in Task 4) writes `P_tail_`/`p_tail_` directly into `riccati_stages_[N_active_]` when `tail_computed_`. Riccati functions need **only** `N_active`.

### 3.1 `backward_lhs`
```cpp
static Status backward_lhs(Stage stages[], WS& ws, double reg_base, double& reg_used,
                           int N_active = HORIZON)
```
Change `const int N = HORIZON` → `const int N = N_active`. Everything else unchanged.

### 3.2 `backward_rhs`
```cpp
static Status backward_rhs(Stage stages[], WS& ws, int N_active = HORIZON)
```
Change `const int N = HORIZON` → `const int N = N_active`.

### 3.3 `forward`
```cpp
static Status forward(Stage stages[], WS& ws, Vec<NX>& dx0, int N_active = HORIZON)
```

### 3.4 `backward` (convenience wrapper)
```cpp
static Status backward(Stage stages[], WS& ws, double reg_base, double& reg_used,
                       int N_active = HORIZON)
```

---

## Task 4: Modify KKT Builder — Inject Tail Cost at Terminal Node

**File**: `include/nmpc/nmpc_ipm_paper.hpp`

### 4.1 `build_kkt_lhs()` (~line 1234)

Loop bound: `k = 0..N_active_`. At `k == N_active_` when `tail_computed_`:
- Write `P_tail_` into `riccati_stages_[k].Qxx` (SymMat → Mat)
- Zero `Quu`, `Qux`, `A`, `B`, `c`
- **Skip** constraint-barrier-Hessian block (avoids reading stale `s[N_active_].s[j]`/`.λ[j]`)

When `!tail_computed_`: existing logic unchanged.

### 4.2 `build_kkt_rhs()` (~line 1287)

At `k == N_active_` when `tail_computed_`:
- Write `p_tail_` into `riccati_stages_[k].qx`
- Zero `qu`
- **Skip** constraint correction block

### 4.3 `solve_kkt_lhs()` / `solve_kkt_rhs_and_forward()`

Pass `N_active_` only — no pointer args needed.

---

## Task 5: Propagate N_active and K Through ALL Solver Methods

**File**: `include/nmpc/nmpc_ipm_paper.hpp`

### 5.1 Method loop-bound table

| Method | Loop bound | Notes |
|--------|-----------|-------|
| `evaluate_model()` | **No change** — all stages | Must evaluate ALL stages for tail solver input |
| `build_kkt_lhs()` | `k = 0..N_active_` | Tail cost injected at terminal |
| `build_kkt_rhs()` | `k = 0..N_active_` | Tail gradient injected at terminal |
| `recover_inequality_steps()` | `k = 0..K` | Tail stages have no slacks/duals |
| `compute_kkt_residuals()` | `k = 0..K`, with `if (k < N_active_)` for dynamics | Only active stages have barrier variables |
| `compute_ftb_limits()` | `k = 0..K` | Only stages with real `s`/`λ` |
| `apply_primal_dual_step()` `.x` | `k = 0..N_active_` | **Includes terminal state** |
| `apply_primal_dual_step()` `.u` | `k < N_active_` | No control at terminal |
| `apply_primal_dual_step()` `.s`/`.λ` | `k = 0..K` | Only stages with real slacks |
| `compute_objective()` | `k < N_active_` + terminal replacement | See Task 6 |
| `cost_directional_derivative()` | `k ≤ N_active_` (qx/dx), `k < N_active_` (qu/du) | Match Riccati range |
| `compute_theta()` | dyn: `k < N_active_`; ineq: `k = 0..K` | Two sub-ranges |
| `sz_complement()` | `k = 0..K` | Only stages with real slacks |
| Inertia check (S pivots) | `k = 0..N_active_-1` | Riccati stages with S_fact |
| Inertia check (P terminal) | `riccati_ws_.P[N_active_]` | **NOT `P[HORIZON]`** |
| Linear KKT residual | `k = 0..N_active_` | Terminal at N_active_ |

### 5.2 `apply_primal_dual_step` — detailed fix

**Critical**: `.x` at `k = N_active_` MUST be updated. If skipped:
- **Non-hybrid**: terminal state `x_N` never updated → dynamics defect can never be zero → **breaks all existing examples**
- **Hybrid**: `forward_expansion` receives stale `x_M`

### 5.3 IPMTrialEvaluator — 5 methods

**`evaluate()`**: dynamics `k < N_active_`, terminal cost with quadratic replacement when `tail_computed_`, constraints `k = 0..K`, barrier `k = 0..K`

**`current_phi()`**: barrier loop `k = 0..K`, `compute_objective()` handles terminal

**`compute_Dphi()`**: Hessian form `k ≤ N_active_` for `.Qxx`, `k < N_active_` for `.Quu`/`.Qux`

**`compute_soc()`**: save/restore `k ≤ N_active_` (dx), `k < N_active_` (du), `k ≤ K` (ds/dlam); trial dynamics `k < N_active_`; `backward_rhs`/`forward` with `N_active_`; `recover_inequality_steps` with K

---

## Task 6: `compute_objective()` — Terminal Cost Replacement

When `tail_computed_`, terminal cost is a **replacement**, not an addition:
```cpp
if (tail_computed_) {
    // ψ(x_M) = 0.5 * x_M^T P_tail x_M + p_tail^T x_M
    const auto& xM = s[N_active_].x;
    double quad = 0.0;
    for (int i = 0; i < NX; ++i)
        for (int j = 0; j < NX; ++j)
            quad += xM[i] * P_tail_(i, j) * xM[j];
    obj += 0.5 * quad + p_tail_.dot(xM);
} else {
    obj += prob_->cost->terminal_cost(s[N_active_].x);
}
```

---

## Task 7: Main Solve Loop Integration

**File**: `include/nmpc/nmpc_ipm_paper.hpp`, `solve()` method (~line 159)

### 7.1 Add data members
```cpp
TailSolver<NX, NU, NC, HORIZON> tail_solver_;
SymMat<NX> P_tail_;
Vec<NX>    p_tail_;
bool       tail_computed_ = false;
int        N_active_ = HORIZON;
```

### 7.2 Modify `solve()`
```cpp
// Determine active horizon (guard active_horizon=0)
if (params_.hybrid_rti_enabled
    && params_.active_horizon > 0
    && params_.active_horizon < HORIZON)
    N_active_ = params_.active_horizon;
else
    N_active_ = HORIZON;

evaluate_model();  // ALL stages

tail_computed_ = false;
if (N_active_ < HORIZON) {
    auto terminal = tail_solver_.compute(prob_->stages, N_active_, params_.tau_tail);
    if (terminal.success) {
        P_tail_ = terminal.P; p_tail_ = terminal.p; tail_computed_ = true;
    } else {
        N_active_ = HORIZON;  // fallback
    }
}

// ... existing Mehrotra loop (all methods use N_active_/K) ...

// Forward expansion after convergence
if (tail_computed_)
    tail_solver_.forward_expansion(prob_->stages[N_active_].x, prob_->stages, N_active_);
```

### 7.3 Inertia check fix (~line 260)
`riccati_ws_.P[HORIZON]` → `riccati_ws_.P[N_active_]` (P[HORIZON] is uninitialized in hybrid mode)

---

## Task 8: Unit Tests

**New file**: `tests/test_tail_solver.cpp`

| # | Test | Verification |
|---|------|-------------|
| 1 | M = HORIZON | `tail_computed_ = false`, identical to standard IPM |
| 2 | M = 1 (max tail) | Converges, P_tail PSD |
| 3 | Barrier penalty | Hessian = `C^T·diag(τ/g²)·C` |
| 4 | Disabled mode | `N_active_ = HORIZON`, identical to baseline |
| 5 | Infeasible tail | `success=false`, fallback |
| 6 | LDLT failure in tail | Retry exhausted → `success=false` |
| 7 | Forward expansion | `stages[HORIZON].x` correct, delta-coord gains |
| 8 | Non-hybrid regression | `K = HORIZON`, all bounds match original |
| 9 | `active_horizon=0` guard | `hybrid=true, M=0` → `N_active_ = HORIZON` |

---

## Task 9: Integration Benchmarks

| Scenario | M Values | Metrics |
|----------|----------|---------|
| Pendulum | N, N/2, 5 | Cost, iters, time, constraint violation |
| Quadrotor | N, N/2, 10 | Speedup ∝ (N-M)/N |
| Chain mass | N, N/2, 15 | Tail cost captures long-horizon |

τ sensitivity: `τ = 1.0, 0.1, 0.01, 0.001`. Expect gap ~ O(τ).

---

## Task 10: CMakeLists.txt Update

Add `tests/test_tail_solver.cpp` to test targets.

---

## Implementation Order

1. **Phase 1**: `nmpc_tail_solver.hpp` — TailSolver with `compute()` and `forward_expansion()`
2. **Phase 2**: `nmpc_riccati.hpp` — `N_active` parameter only (no pointer args)
3. **Phase 3**: `nmpc_ipm_paper.hpp` — params, data members, `build_kkt_lhs/rhs` (inject tail at terminal), ALL loop-bound methods (Task 5 table), `IPMTrialEvaluator` (5 methods), `compute_objective`, inertia check, solve loop
4. **Phase 4**: `CMakeLists.txt` + `test_tail_solver.cpp`
5. **Phase 5**: Integration benchmarks

---

## Known Pitfalls

| Pitfall | Resolution |
|---------|-----------|
| `ldlt_solve` takes `Vec` only | Column-by-column solve for `K = S⁻¹G` |
| LDLT failure in tail | Retry with escalating `reg` (6 attempts, ×10), `success=false` on exhaustion |
| VLA `K_gain[N-M]` | Fixed-size `K_stored_[HORIZON]` arrays |
| `Mat` vs `SymMat` | `Mat` for stage data, `copy_lower_from` to `SymMat` |
| `stages_` doesn't exist | Use `prob_->stages` or `riccati_stages_` |
| `stages[HORIZON].x` not set | Forward expansion sets it after loop |
| Re-evaluating model in tail | Read pre-populated `prob_->stages[k]` |
| Brittle `is_nonzero()` | Explicit `bool success` in `TerminalCost` |
| `SymMat::set_zero()` doesn't exist | Use `SymMat::zero()` |
| `Status::OK` doesn't exist | Use `Status::SUCCESS` |
| Two active ranges conflated | State/dynamics `0..N_active_`, constraint/slack/dual `0..K` |
| `apply_primal_dual_step` `.x` bound | Must include `k = N_active_` |
| `P[HORIZON]` in inertia check | Use `P[N_active_]` |
| `IPMTrialEvaluator` hardcoded | All 5 methods need N_active_/K treatment |
| `compute_objective()` terminal | Replacement (quadratic ψ(x_M)), not addition |
| Forward expansion gains | Delta coordinates: `du = -K·(x - x̄) - d` |
| `active_horizon=0` default | Guard explicitly, do NOT clamp to 1 |
| Pointer plumbing | Write directly into `riccati_stages_[N_active_]` |
