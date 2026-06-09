#pragma once
/**
 * @file    nmpc_preconditioner.hpp
 * @brief   Block-diagonal Cholesky preconditioner for the per-stage Hessian.
 *
 * Factorizes the FINAL Hessian blocks (cost + barrier) at each stage:
 *
 *     Qxx_k = Lx_k · Lx_k^T,    Quu_k = Lu_k · Lu_k^T
 *
 * Provides L, L^{-1}, and scaling utilities for the solver to apply
 * left/right preconditioning to the KKT system.
 */

#include "nmpc_core.hpp"
#include "nmpc_problem.hpp"
#include "nmpc_riccati.hpp"
#include <cmath>
#include <cstdio>

namespace nmpc {

template <int NX, int NU, int HORIZON>
class HessianPreconditioner {
public:
    using Stage = StageData<NX, NU, 1>;  // NC irrelevant for preconditioner

    HessianPreconditioner() = default;

    // ═══════════════════════════════════════════════════════════════════
    //  Compute Cholesky factors of the final Hessian blocks (Qxx, Quu).
    //  Call AFTER build_kkt_lhs() has filled stages with cost + barrier.
    // ═══════════════════════════════════════════════════════════════════

    template <int NC_>
    void compute(const StageData<NX, NU, NC_> stages[]) {
        for (int k = 0; k <= HORIZON; ++k) {
            factorize_spd(stages[k].Qxx, Lx_[k], inv_Lx_[k]);
            if (k < HORIZON) {
                factorize_spd(stages[k].Quu, Lu_[k], inv_Lu_[k]);
            }
        }
        computed_ = true;
    }

    // ═══════════════════════════════════════════════════════════════════
    //  Accessors
    // ═══════════════════════════════════════════════════════════════════

    /// Cholesky factor L such that H = L·L^T
    const Mat<NX, NX>& Lx(int k) const { return Lx_[k]; }
    const Mat<NU, NU>& Lu(int k) const { return Lu_[k]; }

    /// Inverse of L (lower triangular)
    const Mat<NX, NX>& inv_Lx(int k) const { return inv_Lx_[k]; }
    const Mat<NU, NU>& inv_Lu(int k) const { return inv_Lu_[k]; }

    bool is_computed() const { return computed_; }

    // ═══════════════════════════════════════════════════════════════════
    //  Scaling utilities — apply L^{-1}, L^{-T}, L, L^T to vectors
    // ═══════════════════════════════════════════════════════════════════

    /// v ← Lx_k^{-1} · v
    void scale_x(int k, Vec<NX>& v) const { apply_left(v, inv_Lx_[k]); }
    /// v ← Lu_k^{-1} · v
    void scale_u(int k, Vec<NU>& v) const { apply_left(v, inv_Lu_[k]); }

    /// v ← Lx_k^{-T} · v
    void scale_x_invT(int k, Vec<NX>& v) const { apply_left_transpose(v, inv_Lx_[k]); }
    /// v ← Lu_k^{-T} · v
    void scale_u_invT(int k, Vec<NU>& v) const { apply_left_transpose(v, inv_Lu_[k]); }

    /// v ← Lx_k · v
    void unscale_x(int k, Vec<NX>& v) const { apply_left(v, Lx_[k]); }
    /// v ← Lu_k · v
    void unscale_u(int k, Vec<NU>& v) const { apply_left(v, Lu_[k]); }

    /// v ← Lx_k^T · v
    void unscale_x_T(int k, Vec<NX>& v) const { apply_left_transpose(v, Lx_[k]); }
    /// v ← Lu_k^T · v
    void unscale_u_T(int k, Vec<NU>& v) const { apply_left_transpose(v, Lu_[k]); }

    // ═══════════════════════════════════════════════════════════════════
    //  Matrix scaling: congruence and sandwich products
    // ═══════════════════════════════════════════════════════════════════

    /// Symmetric congruence: M ← T · M · T^T
    template <int N>
    void sandwich_symmetric(Mat<N, N>& M, const Mat<N, N>& T) const {
        Mat<N, N> tmp;
        tmp.zero();
        for (int r = 0; r < N; ++r)
            for (int c = 0; c < N; ++c) {
                double sum = 0.0;
                for (int m = 0; m < N; ++m)
                    sum += T(r, m) * M(m, c);
                tmp(r, c) = sum;
            }
        M.zero();
        for (int r = 0; r < N; ++r)
            for (int c = 0; c < N; ++c) {
                double sum = 0.0;
                for (int m = 0; m < N; ++m)
                    sum += tmp(r, m) * T(c, m);
                M(r, c) = sum;
            }
    }

    /// Rectangular sandwich: M ← T_left · M · T_right^T
    template <int NR, int NC, int NL>
    void sandwich_rectangular(Mat<NR, NC>& M,
                               const Mat<NR, NR>& T_left,
                               const Mat<NC, NC>& T_right) const {
        Mat<NR, NC> tmp;
        tmp.zero();
        for (int r = 0; r < NR; ++r)
            for (int c = 0; c < NC; ++c) {
                double sum = 0.0;
                for (int m = 0; m < NL; ++m)
                    sum += T_left(r, m) * M(m, c);
                tmp(r, c) = sum;
            }
        M.zero();
        for (int r = 0; r < NR; ++r)
            for (int c = 0; c < NC; ++c) {
                double sum = 0.0;
                for (int m = 0; m < NC; ++m)
                    sum += tmp(r, m) * T_right(c, m);
                M(r, c) = sum;
            }
    }

    /// v ← T · v
    template <int N>
    void apply_left(Vec<N>& v, const Mat<N, N>& T) const {
        Vec<N> tmp;
        for (int i = 0; i < N; ++i) {
            double sum = 0.0;
            for (int j = 0; j < N; ++j)
                sum += T(i, j) * v[j];
            tmp[i] = sum;
        }
        v = tmp;
    }

    /// v ← T^T · v
    template <int N>
    void apply_left_transpose(Vec<N>& v, const Mat<N, N>& T) const {
        Vec<N> tmp;
        for (int i = 0; i < N; ++i) {
            double sum = 0.0;
            for (int j = 0; j < N; ++j)
                sum += T(j, i) * v[j];
            tmp[i] = sum;
        }
        v = tmp;
    }

    // ═══════════════════════════════════════════════════════════════════
    //  Diagnostics
    // ═══════════════════════════════════════════════════════════════════

    // ═══════════════════════════════════════════════════════════════════
    //  Solver integration methods
    // ═══════════════════════════════════════════════════════════════════

    /// Transform Riccati stages in-place for the scaled KKT system.
    /// Under variable change x̃ = Lx^{-T}·x, ũ = Lu^{-T}·u:
    ///   Hessian: Qxx→I, Quu→I, Qux→0
    ///   Dynamics: A→Lx^{-1}·A·Lx, B→Lx^{-1}·B·Lu, c→Lx^{-1}·c
    template <int NC_>
    void transform_stages(StageData<NX, NU, NC_> stages[]) const {
        for (int k = 0; k <= HORIZON; ++k) {
            // Hessian blocks: sandwich with L^{-1}
            sandwich_symmetric(stages[k].Qxx, inv_Lx_[k]);
            if (k < HORIZON) {
                sandwich_symmetric(stages[k].Quu, inv_Lu_[k]);
                sandwich_rectangular<NU, NX, NX>(stages[k].Qux,
                                                  inv_Lu_[k], inv_Lx_[k]);
                // Dynamics: left by inv_Lx, right by Lx
                sandwich_rectangular<NX, NX, NX>(stages[k].A,
                                                  inv_Lx_[k], Lx_[k]);
                sandwich_rectangular<NX, NU, NX>(stages[k].B,
                                                  inv_Lx_[k], Lu_[k]);
            }
            // c ← inv_Lx · c
            apply_left(stages[k].c, inv_Lx_[k]);
        }
    }

    /// Scale gradient vectors: v ← L^{-T}·v
    template <int NC_>
    void scale_rhs(StageData<NX, NU, NC_> stages[]) const {
        for (int k = 0; k <= HORIZON; ++k) {
            scale_x_invT(k, stages[k].qx);
            if (k < HORIZON)
                scale_u_invT(k, stages[k].qu);
        }
    }

    /// Unscale Riccati steps: v ← L^T·v
    void unscale_riccati_steps(RiccatiWorkspace<NX, NU, HORIZON>& ws) const {
        for (int k = 0; k <= HORIZON; ++k) {
            unscale_x_T(k, ws.dx[k]);
            if (k < HORIZON)
                unscale_u_T(k, ws.du[k]);
        }
    }

    /// Scale initial state residual: v ← Lx^T·v
    void scale_dx0(Vec<NX>& dx0) const {
        unscale_x_T(0, dx0);
    }

    /// Estimate condition number from the Hessian diagonal ratios.
    template <int NC_>
    double condition_estimate(const StageData<NX, NU, NC_> stages[]) const {
        double max_diag = 0.0, min_diag = 1e100;
        for (int k = 0; k <= HORIZON; ++k) {
            for (int i = 0; i < NX; ++i) {
                double d = stages[k].Qxx(i, i);
                if (d > max_diag) max_diag = d;
                if (d > 1e-14 && d < min_diag) min_diag = d;
            }
            if (k < HORIZON) {
                for (int i = 0; i < NU; ++i) {
                    double d = stages[k].Quu(i, i);
                    if (d > max_diag) max_diag = d;
                    if (d > 1e-14 && d < min_diag) min_diag = d;
                }
            }
        }
        if (min_diag > 1e99) min_diag = 1e-14;
        return max_diag / min_diag;
    }

private:
    // ═══════════════════════════════════════════════════════════════════
    //  Cholesky factorization via LDLT → L_chol = L_ldlt · sqrt(D).
    //  Falls back to Jacobi (diagonal sqrt) on failure.
    // ═══════════════════════════════════════════════════════════════════

    template <int N>
    void factorize_spd(const Mat<N, N>& M, Mat<N, N>& L_out, Mat<N, N>& inv_L_out) {
        SymMat<N> S;
        for (int c = 0; c < N; ++c)
            for (int r = c; r < N; ++r)
                S(r, c) = M(r, c);

        bool ok = S.ldlt_factorize(1e-14);

        if (!ok) {
            L_out.zero();
            inv_L_out.zero();
            for (int i = 0; i < N; ++i) {
                double d = M(i, i);
                double s = (d > 1e-14) ? std::sqrt(d) : 1.0;
                L_out(i, i) = s;
                inv_L_out(i, i) = 1.0 / s;
            }
            return;
        }

        L_out.zero();
        for (int j = 0; j < N; ++j) {
            double sqrt_d = std::sqrt(std::max(S(j, j), 1e-14));
            L_out(j, j) = sqrt_d;
            for (int i = j + 1; i < N; ++i)
                L_out(i, j) = S(i, j) * sqrt_d;
        }

        compute_triangular_inverse(L_out, inv_L_out);
    }

    template <int N>
    void compute_triangular_inverse(const Mat<N, N>& L, Mat<N, N>& inv_L) {
        inv_L.zero();
        for (int col = 0; col < N; ++col) {
            double x[N] = {};
            x[col] = 1.0;
            for (int i = col; i < N; ++i) {
                double sum = x[i];
                for (int j = col; j < i; ++j)
                    sum -= L(i, j) * x[j];
                x[i] = sum / L(i, i);
            }
            for (int i = col; i < N; ++i)
                inv_L(i, col) = x[i];
        }
    }

    // ═══════════════════════════════════════════════════════════════════
    //  Data
    // ═══════════════════════════════════════════════════════════════════

    Mat<NX, NX> Lx_[HORIZON + 1];       // Cholesky factor: Qxx = Lx · Lx^T
    Mat<NU, NU> Lu_[HORIZON];            // Cholesky factor: Quu = Lu · Lu^T
    Mat<NX, NX> inv_Lx_[HORIZON + 1];   // Lx^{-1}
    Mat<NU, NU> inv_Lu_[HORIZON];        // Lu^{-1}
    bool computed_ = false;
};

} // namespace nmpc
