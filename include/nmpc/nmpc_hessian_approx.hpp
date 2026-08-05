#pragma once
/**
 * @file    nmpc_hessian_approx.hpp
 * @brief   Hessian approximation for non-least-squares NMPC costs.
 *
 * Provides two approaches:
 *   1. Gauss-Newton: for least-squares costs, skip 2nd-order terms.
 *      H_GN = Jᵀ·R⁻¹·J  where J is the residual Jacobian.
 *      Guaranteed PSD — no safeguard needed.
 *
 *   2. Powell's damped BFGS: for general nonlinear costs.
 *      B⁺ = B - (B·s)(B·s)ᵀ/(sᵀB·s) + γ·γᵀ/(sᵀγ)
 *      with Powell damping: θ·γ + (1-θ)·B·s when sᵀγ < 0.2·sᵀB·s.
 *      Guaranteed PSD through curvature condition.
 *
 * Both produce a positive semi-definite Hessian block for each stage,
 * ensuring the Riccati backwards pass encounters well-conditioned
 * Schur complement matrices.
 *
 * Dependencies: nmpc_core.hpp only.
 */

#include "nmpc_core.hpp"
#include <cmath>

namespace nmpc {

// ─────────────────────────────────────────────────────────────────────────────
//  Gauss-Newton Hessian approximation
// ─────────────────────────────────────────────────────────────────────────────

// Computes H = Jᵀ·R·J where J is N_RES x N_VAR and R is N_RES x N_RES diagonal.
// For least-squares cost: f(x) = ½·||r(x)||²_R = ½·r(x)ᵀ·R·r(x)
//    ∇f = Jᵀ·R·r
//    ∇²f ≈ Jᵀ·R·J  (drop r·∇²r term — valid when residuals are small)
//
// Template: N_VAR = NX+NU, N_RES = dimension of residual vector.
template <int N_VAR, int N_RES>
struct GaussNewtonHessian {

    // Compute H ≈ Jᵀ·diag(r_weights)·J
    // Input:  J[N_RES][N_VAR] — residual Jacobian (row-major layout)
    //         r_weights[N_RES] — diagonal of R matrix
    // Output: H[N_VAR][N_VAR] — symmetric approximate Hessian
    static void compute(const Mat<N_RES, N_VAR>& J,
                        const Vec<N_RES>& r_weights,
                        SymMat<N_VAR>& H)
    {
        H.zero();
        for (int i = 0; i < N_RES; ++i) {
            double wi = r_weights[i];
            if (wi < 1e-14) continue;
            // Rank-1 update: H += w_i · J_iᵀ · J_i
            for (int r = 0; r < N_VAR; ++r) {
                double jr = J(i, r);
                if (std::fabs(jr) < 1e-30) continue;
                for (int c = 0; c <= r; ++c) {
                    H(r, c) += wi * jr * J(i, c);
                }
            }
        }
    }
};

// ─────────────────────────────────────────────────────────────────────────────
//  Powell's damped BFGS update for a single stage Hessian block
// ─────────────────────────────────────────────────────────────────────────────

// Maintains a PSD Hessian approximation B (SymMat<N>) via BFGS updates.
// Powell damping ensures sᵀγ > 0 (curvature condition) even for nonconvex problems.
//
// Standard BFGS:  γ = ∇f⁺ − ∇f,  s = x⁺ − x
//   B⁺ = B − (B·s)(B·s)ᵀ/(sᵀB·s) + γ·γᵀ/(sᵀγ)
//
// Powell damping: if sᵀγ < κ·sᵀB·s (insufficient curvature):
//   ω = θ·γ + (1−θ)·B·s   where θ = (1−κ)·sᵀB·s / (sᵀB·s − sᵀγ)
//   Then use ω in place of γ in the BFGS formula.

template <int N>
struct PowellBFGS {
    static constexpr double kappa = 0.2;  // curvature threshold

    // Apply one BFGS update (with Powell damping if needed).
    // Input:  B (current Hessian approx, modified in-place)
    //         s = x⁺ − x  (step)
    //         y = ∇f⁺ − ∇f  (gradient difference)
    // Returns true if update was applied, false if skipped (numerically unstable).
    static bool update(SymMat<N>& B,
                       const Vec<N>& s,
                       const Vec<N>& y)
    {
        // Compute sᵀB·s
        double sBs = 0.0;
        for (int i = 0; i < N; ++i) {
            double Bi_s = 0.0;
            for (int j = 0; j < N; ++j)
                Bi_s += B(i, j) * s[j];
            sBs += s[i] * Bi_s;
        }

        // Compute sᵀγ
        double sTy = 0.0;
        for (int i = 0; i < N; ++i)
            sTy += s[i] * y[i];

        // Powell damping: ensure sᵀω ≥ κ·sᵀB·s
        Vec<N> omega;
        if (sTy < kappa * sBs) {
            // Damp: ω = θ·γ + (1−θ)·B·s
            double theta = (1.0 - kappa) * sBs / (sBs - sTy + 1e-14);
            for (int i = 0; i < N; ++i) {
                double Bs_i = 0.0;
                for (int j = 0; j < N; ++j) Bs_i += B(i, j) * s[j];
                omega[i] = theta * y[i] + (1.0 - theta) * Bs_i;
            }
        } else {
            // Curvature condition satisfied — use γ directly
            for (int i = 0; i < N; ++i) omega[i] = y[i];
        }

        // Recompute sᵀω after damping
        double sTo = 0.0;
        for (int i = 0; i < N; ++i) sTo += s[i] * omega[i];
        if (sTo < 1e-14) return false;  // numerically degenerate

        // BFGS: B⁺ = B − (B·s)(B·s)ᵀ/(sᵀB·s) + ω·ωᵀ/(sᵀω)
        // Compute B·s
        Vec<N> Bs;
        for (int i = 0; i < N; ++i) {
            double sum = 0.0;
            for (int j = 0; j < N; ++j) sum += B(i, j) * s[j];
            Bs[i] = sum;
        }

        // Rank-1 subtract: B − (B·s)(B·s)ᵀ/sBs
        double inv_sBs = 1.0 / (sBs + 1e-14);
        for (int i = 0; i < N; ++i) {
            for (int j = 0; j <= i; ++j) {
                B(i, j) -= inv_sBs * Bs[i] * Bs[j];
            }
        }

        // Rank-1 add: B + ω·ωᵀ/sTo
        double inv_sTo = 1.0 / (sTo + 1e-14);
        for (int i = 0; i < N; ++i) {
            for (int j = 0; j <= i; ++j) {
                B(i, j) += inv_sTo * omega[i] * omega[j];
            }
        }

        return true;
    }
};

} // namespace nmpc
