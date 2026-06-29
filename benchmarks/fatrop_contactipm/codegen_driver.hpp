#pragma once
/**
 * @file    codegen_driver.hpp
 * @brief   Generic driver that wraps casadi-generated C functions into
 *          ContactIPM's DynamicsModel / CostModel / ConstraintModel interfaces.
 *
 * The casadi CodeGenerator produces C functions with signature:
 *   int func(const double** arg, double** res, long long* iw, double* w, int mem);
 *
 * This header provides:
 *   - CodegenDynamics<NX, NU>:  RK4 integration + FD Jacobians
 *   - CodegenCost<NX, NU>:      stage/terminal cost + gradient/Hessian
 *   - CodegenConstraints<NX, NU, NC>: path/terminal constraints + Jacobian
 *
 * Usage:
 *   #include "codegen_driver.hpp"
 *   // Declare extern C functions from generated code
 *   extern "C" {
 *     int f_expl(const double** arg, double** res, ...);
 *     int l_stage(const double** arg, double** res, ...);
 *     // ...
 *   }
 *   CodegenDynamics<4,1> dyn;
 *   dyn.set_functions(f_expl, f_jac_x, f_jac_u);
 */

#include "nmpc/nmpc_problem.hpp"
#include <cmath>
#include <cstdio>
#include <cstring>

namespace nmpc {

// ── Casadi function pointer type ──────────────────────────────────────────
using CasadiFunc = int (*)(const double** arg, double** res,
                           long long* iw, double* w, int mem);

// ── Helper: call a casadi C function ──────────────────────────────────────
// CasADi sparse codegen only writes structurally nonzero entries.
// We must zero the output buffer first so structural zeros are correct.
inline void casadi_call(CasadiFunc f,
                        const double* in0, const double* in1,
                        double* out0, int out_size) {
    std::memset(out0, 0, out_size * sizeof(double));
    const double* arg[2] = {in0, in1};
    double* res[1] = {out0};
    f(arg, res, nullptr, nullptr, 0);
}

inline void casadi_call_1in(CasadiFunc f,
                             const double* in0,
                             double* out0, int out_size) {
    std::memset(out0, 0, out_size * sizeof(double));
    const double* arg[1] = {in0};
    double* res[1] = {out0};
    f(arg, res, nullptr, nullptr, 0);
}

// ─────────────────────────────────────────────────────────────────────────
//  CodegenDynamics: RK4 integration + exact sensitivity Jacobians
// ─────────────────────────────────────────────────────────────────────────
template <int NX, int NU>
struct CodegenDynamics : DynamicsModel<NX, NU> {
    CasadiFunc f_expl_fn = nullptr;    // f(x,u) -> xdot
    CasadiFunc f_jac_x_fn = nullptr;   // df/dx(x,u)  [NX×NX dense]
    CasadiFunc f_jac_u_fn = nullptr;   // df/du(x,u)  [NX×NU dense]

    // For free-time problems: index of T in augmented state (-1 = fixed time)
    int free_time_idx = -1;

    void set_functions(CasadiFunc f, CasadiFunc jac_x, CasadiFunc jac_u) {
        f_expl_fn = f;
        f_jac_x_fn = jac_x;
        f_jac_u_fn = jac_u;
    }

    // Evaluate effective continuous dynamics: xdot_eff = F(x, u)
    // For fixed-time: F = f(x,u).  For free-time: F = T * f(x,u).
    void eval_ode(const Vec<NX>& x, const Vec<NU>& u, Vec<NX>& xdot) {
        casadi_call(f_expl_fn, x.data, u.data, xdot.data, NX);
        if (free_time_idx >= 0) {
            double T = x[free_time_idx];
            for (int i = 0; i < NX; ++i) xdot[i] *= T;
        }
    }

    // Evaluate effective continuous Jacobians dF/dx and dF/du
    // For free-time: dF/dx = T * df/dx (physical part) + f(x,u)*e_T^T
    void eval_jac(const Vec<NX>& x, const Vec<NU>& u,
                  Mat<NX, NX>& Fx, Mat<NX, NU>& Fu) {
        casadi_call(f_jac_x_fn, x.data, u.data, Fx.data, NX*NX);
        casadi_call(f_jac_u_fn, x.data, u.data, Fu.data, NX*NU);
        if (free_time_idx >= 0) {
            double T = x[free_time_idx];
            // Scale df/dx and df/du by T
            for (int i = 0; i < NX * NX; ++i) Fx.data[i] *= T;
            for (int i = 0; i < NX * NU; ++i) Fu.data[i] *= T;
            // Add df/dT column: dF/dT = f(x,u) (physical ODE before T-scaling)
            Vec<NX> fval;
            casadi_call(f_expl_fn, x.data, u.data, fval.data, NX);
            for (int i = 0; i < NX; ++i)
                Fx(i, free_time_idx) += fval[i]; // += because df/dx already has T-scaled part
        }
    }

    // RK4 integration of continuous dynamics
    Status discrete_step(const Vec<NX>& x, const Vec<NU>& u,
                         double dt, Vec<NX>& x_next) override {
        Vec<NX> k1, k2, k3, k4, tmp;
        eval_ode(x, u, k1);
        for (int i = 0; i < NX; ++i) tmp[i] = x[i] + 0.5 * dt * k1[i];
        eval_ode(tmp, u, k2);
        for (int i = 0; i < NX; ++i) tmp[i] = x[i] + 0.5 * dt * k2[i];
        eval_ode(tmp, u, k3);
        for (int i = 0; i < NX; ++i) tmp[i] = x[i] + dt * k3[i];
        eval_ode(tmp, u, k4);
        for (int i = 0; i < NX; ++i)
            x_next[i] = x[i] + (dt / 6.0) * (k1[i] + 2*k2[i] + 2*k3[i] + k4[i]);
        return Status::SUCCESS;
    }

    // Discrete Jacobians via RK4 sensitivity integration.
    // Uses exact continuous Jacobians (f_jac_x, f_jac_u) from CasADi codegen,
    // propagated through the RK4 formula to get discrete Jacobians.
    // This avoids O(eps^2) FD truncation error and (NX+NU) extra integrations.
    Status linearize(const Vec<NX>& x, const Vec<NU>& u, double dt,
                     Mat<NX, NX>& A, Mat<NX, NU>& B) override {
        // RK4 stages: positions and slopes
        Vec<NX> k1, k2, k3, k4, tmp;
        Mat<NX, NX> Fx1, Fx2, Fx3, Fx4;
        Mat<NX, NU> Fu1, Fu2, Fu3, Fu4;

        // Stage 1: at (x, u)
        eval_ode(x, u, k1);
        eval_jac(x, u, Fx1, Fu1);

        // Stage 2: at (x + dt/2 * k1, u)
        for (int i = 0; i < NX; ++i) tmp[i] = x[i] + 0.5 * dt * k1[i];
        eval_ode(tmp, u, k2);
        eval_jac(tmp, u, Fx2, Fu2);

        // Stage 3: at (x + dt/2 * k2, u)
        for (int i = 0; i < NX; ++i) tmp[i] = x[i] + 0.5 * dt * k2[i];
        eval_ode(tmp, u, k3);
        eval_jac(tmp, u, Fx3, Fu3);

        // Stage 4: at (x + dt * k3, u)
        for (int i = 0; i < NX; ++i) tmp[i] = x[i] + dt * k3[i];
        eval_ode(tmp, u, k4);
        eval_jac(tmp, u, Fx4, Fu4);

        // Chain sensitivities through RK4.
        // S1x = Fx1, S1u = Fu1
        // For stages i=2,3,4:
        //   Si_x = Fx_i * (I + c_i*dt * S_{i-1}_x)   (cumulative x-sensitivity)
        //   Si_u = Fu_i + c_i*dt * Fx_i * S_{i-1}_u   (cumulative u-sensitivity)
        // Then A = I + (dt/6)(S1x + 2*S2x + 2*S3x + S4x)
        //      B =    (dt/6)(S1u + 2*S2u + 2*S3u + S4u)

        // Accumulate A and B directly (weighted sum of S_i)
        A.set_identity();
        B.zero();

        // Stage 1 contributions: weight = dt/6
        double w1 = dt / 6.0;
        for (int r = 0; r < NX; ++r)
            for (int c = 0; c < NX; ++c)
                A(r, c) += w1 * Fx1(r, c);
        for (int r = 0; r < NX; ++r)
            for (int c = 0; c < NU; ++c)
                B(r, c) += w1 * Fu1(r, c);

        // Compute S2_x, S2_u and accumulate (weight = dt/3)
        // S2_x = Fx2 * (I + dt/2 * Fx1)  →  S2_x(r,c) = Fx2(r,c) + dt/2 * sum_m Fx2(r,m)*Fx1(m,c)
        // S2_u = Fu2 + dt/2 * Fx2 * Fu1  →  S2_u(r,c) = Fu2(r,c) + dt/2 * sum_m Fx2(r,m)*Fu1(m,c)
        Mat<NX, NX> Sx_prev;
        Mat<NX, NU> Su_prev;
        double h2 = 0.5 * dt;

        // S2_x = Fx2 + h2 * Fx2 * Fx1
        for (int r = 0; r < NX; ++r)
            for (int c = 0; c < NX; ++c) {
                double sum = Fx2(r, c);
                for (int m = 0; m < NX; ++m)
                    sum += h2 * Fx2(r, m) * Fx1(m, c);
                Sx_prev(r, c) = sum;
            }
        // S2_u = Fu2 + h2 * Fx2 * Fu1
        for (int r = 0; r < NX; ++r)
            for (int c = 0; c < NU; ++c) {
                double sum = Fu2(r, c);
                for (int m = 0; m < NX; ++m)
                    sum += h2 * Fx2(r, m) * Fu1(m, c);
                Su_prev(r, c) = sum;
            }
        // Accumulate: weight = dt/3 (= 2 * dt/6)
        double w2 = dt / 3.0;
        for (int r = 0; r < NX; ++r)
            for (int c = 0; c < NX; ++c)
                A(r, c) += w2 * Sx_prev(r, c);
        for (int r = 0; r < NX; ++r)
            for (int c = 0; c < NU; ++c)
                B(r, c) += w2 * Su_prev(r, c);

        // S3_x = Fx3 + h2 * Fx3 * S2_x
        Mat<NX, NX> Sx3;
        Mat<NX, NU> Su3;
        for (int r = 0; r < NX; ++r)
            for (int c = 0; c < NX; ++c) {
                double sum = Fx3(r, c);
                for (int m = 0; m < NX; ++m)
                    sum += h2 * Fx3(r, m) * Sx_prev(m, c);
                Sx3(r, c) = sum;
            }
        for (int r = 0; r < NX; ++r)
            for (int c = 0; c < NU; ++c) {
                double sum = Fu3(r, c);
                for (int m = 0; m < NX; ++m)
                    sum += h2 * Fx3(r, m) * Su_prev(m, c);
                Su3(r, c) = sum;
            }
        // Accumulate: weight = dt/3
        for (int r = 0; r < NX; ++r)
            for (int c = 0; c < NX; ++c)
                A(r, c) += w2 * Sx3(r, c);
        for (int r = 0; r < NX; ++r)
            for (int c = 0; c < NU; ++c)
                B(r, c) += w2 * Su3(r, c);

        // S4_x = Fx4 + dt * Fx4 * S3_x
        Mat<NX, NX> Sx4;
        Mat<NX, NU> Su4;
        double h4 = dt;
        for (int r = 0; r < NX; ++r)
            for (int c = 0; c < NX; ++c) {
                double sum = Fx4(r, c);
                for (int m = 0; m < NX; ++m)
                    sum += h4 * Fx4(r, m) * Sx3(m, c);
                Sx4(r, c) = sum;
            }
        for (int r = 0; r < NX; ++r)
            for (int c = 0; c < NU; ++c) {
                double sum = Fu4(r, c);
                for (int m = 0; m < NX; ++m)
                    sum += h4 * Fx4(r, m) * Su3(m, c);
                Su4(r, c) = sum;
            }
        // Accumulate: weight = dt/6
        for (int r = 0; r < NX; ++r)
            for (int c = 0; c < NX; ++c)
                A(r, c) += w1 * Sx4(r, c);
        for (int r = 0; r < NX; ++r)
            for (int c = 0; c < NU; ++c)
                B(r, c) += w1 * Su4(r, c);

        return Status::SUCCESS;
    }
};

// ─────────────────────────────────────────────────────────────────────────
//  CodegenCost: stage + terminal cost with gradients and Hessians
// ─────────────────────────────────────────────────────────────────────────
template <int NX, int NU>
struct CodegenCost : CostModel<NX, NU> {
    CasadiFunc l_stage_fn = nullptr;
    CasadiFunc l_stage_grad_x_fn = nullptr;
    CasadiFunc l_stage_grad_u_fn = nullptr;
    CasadiFunc l_stage_hess_xx_fn = nullptr;
    CasadiFunc l_stage_hess_uu_fn = nullptr;
    CasadiFunc l_stage_hess_ux_fn = nullptr;

    CasadiFunc l_term_fn = nullptr;
    CasadiFunc l_term_grad_fn = nullptr;
    CasadiFunc l_term_hess_fn = nullptr;

    double stage_cost(const Vec<NX>& x, const Vec<NU>& u, int) override {
        double out[1];
        casadi_call(l_stage_fn, x.data, u.data, out, 1);
        return out[0];
    }

    double terminal_cost(const Vec<NX>& x) override {
        double out[1];
        casadi_call_1in(l_term_fn, x.data, out, 1);
        return out[0];
    }

    Status stage_gradient(const Vec<NX>& x, const Vec<NU>& u, int,
                          Vec<NX>& qx, Vec<NU>& qu) override {
        casadi_call(l_stage_grad_x_fn, x.data, u.data, qx.data, NX);
        casadi_call(l_stage_grad_u_fn, x.data, u.data, qu.data, NU);
        return Status::SUCCESS;
    }

    Status stage_hessian(const Vec<NX>& x, const Vec<NU>& u, int,
                         Mat<NX, NX>& Qxx, Mat<NU, NU>& Quu,
                         Mat<NU, NX>& Qux) override {
        // Use exact CasADi-generated Hessians (dense output, column-major)
        casadi_call(l_stage_hess_xx_fn, x.data, u.data, Qxx.data, NX*NX);
        casadi_call(l_stage_hess_uu_fn, x.data, u.data, Quu.data, NU*NU);
        casadi_call(l_stage_hess_ux_fn, x.data, u.data, Qux.data, NU*NX);
        return Status::SUCCESS;
    }

    Status terminal_gradient(const Vec<NX>& x, Vec<NX>& qx) override {
        casadi_call_1in(l_term_grad_fn, x.data, qx.data, NX);
        return Status::SUCCESS;
    }

    Status terminal_hessian(const Vec<NX>& x, Mat<NX, NX>& Qxx) override {
        // Use exact CasADi-generated terminal Hessian (dense output)
        casadi_call_1in(l_term_hess_fn, x.data, Qxx.data, NX*NX);
        return Status::SUCCESS;
    }
};

// ─────────────────────────────────────────────────────────────────────────
//  CodegenConstraints: path + terminal constraints with Jacobians
// ─────────────────────────────────────────────────────────────────────────
template <int NX, int NU, int NC>
struct CodegenConstraints : ConstraintModel<NX, NU, NC> {
    CasadiFunc g_path_fn = nullptr;
    CasadiFunc g_path_jac_x_fn = nullptr;
    CasadiFunc g_path_jac_u_fn = nullptr;

    CasadiFunc g_term_fn = nullptr;
    CasadiFunc g_term_jac_fn = nullptr;

    Status evaluate(const Vec<NX>& x, const Vec<NU>& u, int,
                    Vec<NC>& g) override {
        casadi_call(g_path_fn, x.data, u.data, g.data, NC);
        return Status::SUCCESS;
    }

    Status evaluate_terminal(const Vec<NX>& x, Vec<NC>& g) override {
        casadi_call_1in(g_term_fn, x.data, g.data, NC);
        return Status::SUCCESS;
    }

    Status jacobian(const Vec<NX>& x, const Vec<NU>& u, int,
                    Mat<NC, NX>& Cx, Mat<NC, NU>& Cu) override {
        // Use exact CasADi-generated Jacobians (dense, column-major)
        casadi_call(g_path_jac_x_fn, x.data, u.data, Cx.data, NC * NX);
        casadi_call(g_path_jac_u_fn, x.data, u.data, Cu.data, NC * NU);
        return Status::SUCCESS;
    }

    Status jacobian_terminal(const Vec<NX>& x, Mat<NC, NX>& Cx) override {
        // Use exact CasADi-generated terminal constraint Jacobian
        casadi_call_1in(g_term_jac_fn, x.data, Cx.data, NC * NX);
        return Status::SUCCESS;
    }
};

} // namespace nmpc
