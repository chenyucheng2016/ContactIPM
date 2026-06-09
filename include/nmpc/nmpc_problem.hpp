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

    // Number of active constraints at stage k (can be < NC)
    virtual int num_constraints(int k) const { return NC; }

    // Constraint value:  g_k(x, u)
    virtual Status evaluate(const Vec<NX>& x, const Vec<NU>& u, int k,
                            Vec<NC>& g) = 0;

    // Terminal constraint:  g_N(x)
    virtual Status evaluate_terminal(const Vec<NX>& x, Vec<NC>& g) = 0;

    // Jacobians of stage constraint
    virtual Status jacobian(const Vec<NX>& x, const Vec<NU>& u, int k,
                            Mat<NC, NX>& Cx, Mat<NC, NU>& Cu) = 0;

    // Jacobian of terminal constraint
    virtual Status jacobian_terminal(const Vec<NX>& x,
                                     Mat<NC, NX>& Cx) = 0;
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
};

} // namespace nmpc
