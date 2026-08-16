#pragma once
/**
 * @file    nmpc_problem.hpp
 * @brief   NMPC problem definition – multiple-shooting transcription.
 *
 * Problem:
 *   min   Σ_{k=0}^{N-1} ℓ_k(x_k, u_k)  +  V_N(x_N)
 *   s.t.  x_{k+1} = f_k(x_k, u_k)       k=0..N-1    (dynamics)
 *         g_k(x_k, u_k) ≤ 0              k=0..N-1    (path constraints)
 *         g_N(x_N) ≤ 0                                 (terminal constraints)
 *         x_0 = x̄ (given)
 *
 * The user provides callable objects for  f, ℓ, g  and their derivatives.
 */

#include "nmpc_core.hpp"

namespace nmpc {

// ─────────────────────────────────────────────────────────────────────────────
//  Stage-wise data:  state + control + slack + multipliers
// ─────────────────────────────────────────────────────────────────────────────

template <int NX, int NU, int NC>
struct StageData {
    Vec<NX> x;       // state
    Vec<NU> u;       // control  (terminal stage: unused)
    Vec<NC> s;       // slack for inequalities
    Vec<NC> lambda;  // dual multiplier for g(x,u)+s=0

    // Bound multipliers (for variable bounds, populated during solve)
    // These are separate from NC coupling constraints — they handle
    // simple bounds like |u| <= 1 without slack/multiplier coupling.
    Vec<NU> z_L_u, z_U_u;   // lower/upper bound multipliers for controls
    Vec<NX> z_L_x, z_U_x;   // lower/upper bound multipliers for states

    // Sensitivity matrices for Riccati (populated by Hessian evaluation)
    Mat<NX, NX> Qxx;  // ∂²L/∂x²
    Mat<NU, NU> Quu;  // ∂²L/∂u²
    Mat<NU, NX> Qux;  // ∂²L/∂u∂x
    Mat<NX, NU> Qxu;  // = Qux^T

    Vec<NX> qx;       // ∂L/∂x
    Vec<NU> qu;       // ∂L/∂u

    // Dynamics linearization:  x_{k+1} ≈ A_k x_k + B_k u_k + c_k
    Mat<NX, NX> A;
    Mat<NX, NU> B;
    Vec<NX>     c;    // f(x̄,ū) - A x̄ - B ū

    // Constraint linearization:  g(x,u) ≈ C_k^x x + C_k^u u + d_k
    Mat<NC, NX> Cx;
    Mat<NC, NU> Cu;
    Vec<NC>     d;    // g(x̄,ū) - Cx x̄ - Cu ū

    // Cost at current linearization point
    double cost = 0.0;
};

// ─────────────────────────────────────────────────────────────────────────────
//  Problem dimensions  (compile-time)
// ─────────────────────────────────────────────────────────────────────────────

template <int NX_, int NU_, int NC_, int HORIZON_>
struct ProblemDimensions {
    static constexpr int NX      = NX_;
    static constexpr int NU      = NU_;
    static constexpr int NC      = NC_;
    static constexpr int HORIZON = HORIZON_;
    static constexpr int NZ      = NX + NU;   // primal vars per stage
};

// ─────────────────────────────────────────────────────────────────────────────
//  Abstract interfaces for dynamics, cost, constraints
//
//  The user derives from these and provides the actual physics.
//  All methods return Status; no exceptions.
// ─────────────────────────────────────────────────────────────────────────────

template <int NX, int NU>
struct DynamicsModel {
    virtual ~DynamicsModel() = default;

    // Continuous-time:  ẋ = f_c(x, u)
    // Default implementation: user overrides this OR discrete_step.
    virtual Status continuous(const Vec<NX>& /*x*/, const Vec<NU>& /*u*/,
                              Vec<NX>& xdot) {
        xdot.zero();
        return Status::SUCCESS;
    }

    // Discrete-time:  x_next = f_d(x, u, dt)
    virtual Status discrete_step(const Vec<NX>& x, const Vec<NU>& u,
                                 double dt, Vec<NX>& x_next) = 0;

    // Jacobians of discrete dynamics:  A = ∂f/∂x,  B = ∂f/∂u
    virtual Status linearize(const Vec<NX>& x, const Vec<NU>& u,
                             double dt,
                             Mat<NX, NX>& A, Mat<NX, NU>& B) = 0;

    // ── k-aware overloads for time-varying dynamics ──────────────────
    //
    // Override these when the dynamics depend on the stage index k
    // (e.g. time-varying contact schedules, reference trajectories).
    // Default implementations delegate to the k-free versions above,
    // so existing clients require no changes.

    virtual Status discrete_step(const Vec<NX>& x, const Vec<NU>& u,
                                 double dt, Vec<NX>& x_next, int /*k*/) {
        return discrete_step(x, u, dt, x_next);
    }

    virtual Status linearize(const Vec<NX>& x, const Vec<NU>& u,
                             double dt,
                             Mat<NX, NX>& A, Mat<NX, NU>& B, int /*k*/) {
        return linearize(x, u, dt, A, B);
    }

    // ── Optional: analytic adjoint (contracted) Hessian ─────────────────
    //
    // For exact-Newton steps the solver needs the second-order curvature of
    // the dynamics contracted with the costate p (= p_{k+1}):
    //     Σ_m p[m] · ∇²_zz f_m(x, u),   z = (x, u).
    // When a model returns true from provides_adjoint_hessian(), the solver
    // calls adjoint_hessian() instead of finite-differencing discrete_step().
    // The method must ADD its contribution into the (pre-populated) blocks:
    //     Hxx += ∂²/∂x²,  Hux += ∂²/(∂u ∂x),  Huu += ∂²/∂u².
    // Default: not provided → solver falls back to finite differences.
    virtual bool provides_adjoint_hessian() const { return false; }

    virtual Status adjoint_hessian(const Vec<NX>& /*x*/, const Vec<NU>& /*u*/,
                                   double /*dt*/, int /*k*/,
                                   const Vec<NX>& /*p*/,
                                   Mat<NX, NX>& /*Hxx*/, Mat<NU, NX>& /*Hux*/,
                                   Mat<NU, NU>& /*Huu*/) {
        return Status::SUCCESS;
    }
};

template <int NX, int NU>
struct CostModel {
    virtual ~CostModel() = default;

    // Running cost  ℓ_k(x, u)
    virtual double stage_cost(const Vec<NX>& x, const Vec<NU>& u, int k) = 0;

    // Terminal cost  V_N(x)
    virtual double terminal_cost(const Vec<NX>& x) = 0;

    // Gradient & Hessian of stage cost
    virtual Status stage_gradient(const Vec<NX>& x, const Vec<NU>& u, int k,
                                  Vec<NX>& qx, Vec<NU>& qu) = 0;

    virtual Status stage_hessian(const Vec<NX>& x, const Vec<NU>& u, int k,
                                 Mat<NX, NX>& Qxx, Mat<NU, NU>& Quu,
                                 Mat<NU, NX>& Qux) = 0;

    // Gradient & Hessian of terminal cost
    virtual Status terminal_gradient(const Vec<NX>& x,
                                     Vec<NX>& qx) = 0;

    virtual Status terminal_hessian(const Vec<NX>& x,
                                    Mat<NX, NX>& Qxx) = 0;
};

template <int NX, int NU, int NC>
struct ConstraintModel {
    virtual ~ConstraintModel() = default;

    // Number of user-supplied one-sided rows g(x,u) <= 0 at stage k.
    // The solver appends one relaxed product row for every complementarity
    // pair, so num_constraints(k) + num_complementarity_pairs(k) must be <= NC.
    virtual int num_constraints(int k) const { return NC; }

    // Optional MPCC metadata. A pair (a, b) refers to two user-supplied rows
    // g_a <= 0 and g_b <= 0, representing nonnegative quantities
    // y_a = -g_a and y_b = -g_b with 0 <= y_a perpendicular to y_b >= 0.
    // The solver recognizes the pair and appends
    //
    //     y_a * y_b - theta(mu) + s_c = 0,  s_c > 0,
    //
    // with s_c included in the solver's log barrier and primal-dual system.
    // The default keeps existing constraint models source-compatible.
    virtual int num_complementarity_pairs(int k) const { return 0; }

    virtual bool complementarity_pair(int /*k*/, int /*pair*/,
                                      int& /*first_row*/,
                                      int& /*second_row*/) const {
        return false;
    }

    // Constraint value:  g_k(x, u)
    virtual Status evaluate(const Vec<NX>& x, const Vec<NU>& u, int k,
                            Vec<NC>& g) = 0;

    // Terminal constraint:  g_N(x)
    virtual Status evaluate_terminal(const Vec<NX>& x, Vec<NC>& g) = 0;

    // Jacobians of stage constraint
    virtual Status jacobian(const Vec<NX>& x, const Vec<NU>& u, int k,
                            Mat<NC, NX>& Cx, Mat<NC, NU>& Cu) = 0;

    // Optional fused value/Jacobian path. The default preserves existing
    // models; terrain-heavy models can override it to share sampled geometry.
    virtual Status evaluate_with_jacobian(
        const Vec<NX>& x, const Vec<NU>& u, int k, Vec<NC>& g,
        Mat<NC, NX>& Cx, Mat<NC, NU>& Cu) {
        Status st = evaluate(x, u, k, g);
        return st == Status::SUCCESS ? jacobian(x, u, k, Cx, Cu) : st;
    }

    // Jacobian of terminal constraint
    virtual Status jacobian_terminal(const Vec<NX>& x,
                                     Mat<NC, NX>& Cx) = 0;

    virtual Status evaluate_terminal_with_jacobian(
        const Vec<NX>& x, Vec<NC>& g, Mat<NC, NX>& Cx) {
        Status st = evaluate_terminal(x, g);
        return st == Status::SUCCESS ? jacobian_terminal(x, Cx) : st;
    }

    // ── Optional: analytic adjoint (contracted) Hessian ─────────────────
    //
    // Second-order curvature of the constraints contracted with the
    // multiplier λ:  Σ_j λ[j] · ∇²_zz g_j(x, u),  z = (x, u).  Same block
    // convention and ADD semantics as DynamicsModel::adjoint_hessian.
    // Default: not provided → solver falls back to finite differences.
    virtual bool provides_adjoint_hessian() const { return false; }

    virtual Status adjoint_hessian(const Vec<NX>& /*x*/, const Vec<NU>& /*u*/,
                                   int /*k*/, const Vec<NC>& /*lambda*/,
                                   Mat<NX, NX>& /*Hxx*/, Mat<NU, NX>& /*Hux*/,
                                   Mat<NU, NU>& /*Huu*/) {
        return Status::SUCCESS;
    }

    virtual Status adjoint_hessian_terminal(const Vec<NX>& /*x*/,
                                            const Vec<NC>& /*lambda*/,
                                            Mat<NX, NX>& /*Hxx*/) {
        return Status::SUCCESS;
    }
};

// ─────────────────────────────────────────────────────────────────────────────
//  Problem specification – bundles models + dimensions + settings
// ─────────────────────────────────────────────────────────────────────────────

template <int NX, int NU, int NC, int HORIZON>
struct NMPCProblem {
    using Dims = ProblemDimensions<NX, NU, NC, HORIZON>;

    DynamicsModel<NX, NU>*    dynamics    = nullptr;
    CostModel<NX, NU>*        cost        = nullptr;
    ConstraintModel<NX, NU, NC>* constraints = nullptr;

    double dt = 0.05;  // discretization timestep

    // Initial state
    Vec<NX> x0;

    // Variable bounds (separate from coupling constraints)
    // Use -1e20 / 1e20 to indicate "free" (no bound).
    // These are handled via log-barrier directly on variables — no slacks,
    // no multipliers in the NC constraint channel. This matches IPOPT's
    // x_L/x_U formulation and avoids slack-primal desynchronization.
    Vec<NX> x_lb, x_ub;    // state bounds (-1e20 = free)
    Vec<NU> u_lb, u_ub;    // control bounds (-1e20 = free)
    int n_bound_u = 0;      // number of controls with a finite bound
    int n_bound_x = 0;      // number of states with a finite bound

    // Initial guess for primal variables (all stages)
    StageData<NX, NU, NC> stages[HORIZON + 1];  // 0..N-1 + terminal
    // Terminal stage uses x only, with terminal constraints

    // Warm-start data from previous solve
    bool warm_start_enabled = true;

    // Check that user provided required models
    Status validate() const {
        if (!dynamics)    return Status::BAD_ARGUMENT;
        if (!cost)        return Status::BAD_ARGUMENT;
        // constraints can be nullptr (unconstrained case)
        return Status::SUCCESS;
    }

    // Initialize bound arrays to "free" (no bounds)
    void init_bounds_free() {
        for (int i = 0; i < NX; ++i) {
            x_lb[i] = -1e20; x_ub[i] = 1e20;
        }
        for (int i = 0; i < NU; ++i) {
            u_lb[i] = -1e20; u_ub[i] = 1e20;
        }
        n_bound_x = 0;
        n_bound_u = 0;
    }
};

} // namespace nmpc
