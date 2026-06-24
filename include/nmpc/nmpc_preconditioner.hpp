#pragma once
/**
 * @file    nmpc_preconditioner.hpp
 * @brief   Diagonal Jacobi preconditioner for the per-stage Hessian.
 *
 * Computes scaling factors from the Hessian diagonal ONCE per MPC solve:
 *
 *     Lx_i = sqrt(max(Qxx_ii, floor))    floor = (1e-4 * max_j(sqrt(Qxx_jj)))^2
 *     Lu_i = sqrt(max(Quu_ii, floor))
 *
 * Every Newton iteration, transform_qp() applies the FIXED scaling to fresh
 * derivatives from evaluate_model().  The Riccati solver sees only scaled data
 * and is completely unaware of scaling (IPOPT-style separation).
 *
 * Convention: dx = inv_Lx · dx̂, i.e., dx̂ = Lx · dx.
 *
 * Transform table:
 *   Qxx ← inv_Lx · Qxx · inv_Lx     qx ← inv_Lx · qx
 *   Quu ← inv_Lu · Quu · inv_Lu     qu ← inv_Lu · qu
 *   Qux ← inv_Lu · Qux · inv_Lx
 *   A   ← Lx_{k+1} · A · inv_Lx_k   c ← Lx_{k+1} · c
 *   B   ← Lx_{k+1} · B · inv_Lu_k
 *   Cx  ← Cx · inv_Lx               Cu ← Cu · inv_Lu
 *
 * Recovery:
 *   Primal:  dx = inv_Lx · dx̂,  du = inv_Lu · dû
 *   Dual:    ν = Lx · ν̂  (costate),  λ = λ̂  (constraint multiplier, invariant)
 */

#include "nmpc_core.hpp"
#include "nmpc_problem.hpp"
#include "nmpc_riccati.hpp"
#include <cmath>
#include <cstdio>
#include <algorithm>

namespace nmpc {

template <int NX, int NU, int HORIZON>
class HessianPreconditioner {
public:
    HessianPreconditioner() {
        // Initialize to identity (no scaling) so inv_Lx/inv_Lu return 1.0
        // when preconditioner is not computed.
        for (int k = 0; k <= HORIZON; ++k) {
            for (int i = 0; i < NX; ++i) {
                Lx_[k][i] = 1.0;
                inv_Lx_[k][i] = 1.0;
            }
            if (k < HORIZON) {
                for (int i = 0; i < NU; ++i) {
                    Lu_[k][i] = 1.0;
                    inv_Lu_[k][i] = 1.0;
                }
            }
        }
    }

    // ═══════════════════════════════════════════════════════════════════
    //  Compute diagonal scaling from Hessian diagonal.
    //  Call ONCE per MPC solve (outside Newton loop).
    //  Lx, Lu are FIXED for the entire solve.
    // ═══════════════════════════════════════════════════════════════════

    template <int NC_>
    void compute(const StageData<NX, NU, NC_> stages[]) {
        for (int k = 0; k <= HORIZON; ++k) {
            // Find max diagonal for relative floor
            double max_d = 0.0;
            for (int i = 0; i < NX; ++i)
                max_d = std::max(max_d, stages[k].Qxx(i, i));
            double floor_val = 1e-8 * max_d;  // (1e-4 * sqrt(max))^2 = 1e-8 * max

            for (int i = 0; i < NX; ++i) {
                double d = std::max(stages[k].Qxx(i, i), floor_val);
                Lx_[k][i] = std::sqrt(d);
                inv_Lx_[k][i] = 1.0 / Lx_[k][i];
            }

            if (k < HORIZON) {
                double max_du = 0.0;
                for (int i = 0; i < NU; ++i)
                    max_du = std::max(max_du, stages[k].Quu(i, i));
                double floor_u = 1e-8 * max_du;

                for (int i = 0; i < NU; ++i) {
                    double d = std::max(stages[k].Quu(i, i), floor_u);
                    Lu_[k][i] = std::sqrt(d);
                    inv_Lu_[k][i] = 1.0 / Lu_[k][i];
                }
            }
        }
        computed_ = true;
    }

    // ═══════════════════════════════════════════════════════════════════
    //  Accessors
    // ═══════════════════════════════════════════════════════════════════

    const Vec<NX>& Lx(int k) const { return Lx_[k]; }
    const Vec<NU>& Lu(int k) const { return Lu_[k]; }
    const Vec<NX>& inv_Lx(int k) const { return inv_Lx_[k]; }
    const Vec<NU>& inv_Lu(int k) const { return inv_Lu_[k]; }
    bool is_computed() const { return computed_; }

    // Array accessors for debug exposure (returns pointer to stage 0)
    const Vec<NX>* debug_Lx() const { return Lx_; }
    const Vec<NU>* debug_Lu() const { return Lu_; }
    const Vec<NX>* debug_inv_Lx() const { return inv_Lx_; }
    const Vec<NU>* debug_inv_Lu() const { return inv_Lu_; }

    // ═══════════════════════════════════════════════════════════════════
    //  Transform QP derivatives to scaled space.
    //  Call every Newton iteration AFTER evaluate_model().
    //  Modifies ONLY derivatives (Q, q, A, B, c, Cx, Cu), NOT primal/dual.
    // ═══════════════════════════════════════════════════════════════════

    template <int NC_>
    void transform_qp(StageData<NX, NU, NC_> stages[]) const {
        for (int k = 0; k <= HORIZON; ++k) {
            const Vec<NX>& ilx = inv_Lx_[k];

            // Qxx ← inv_Lx · Qxx · inv_Lx  (element-wise: Qxx(i,j) *= inv_Lx[i]*inv_Lx[j])
            for (int r = 0; r < NX; ++r)
                for (int c = 0; c < NX; ++c)
                    stages[k].Qxx(r, c) *= ilx[r] * ilx[c];

            // qx ← inv_Lx · qx
            for (int i = 0; i < NX; ++i)
                stages[k].qx[i] *= ilx[i];

            if (k < HORIZON) {
                const Vec<NU>& ilu = inv_Lu_[k];
                const Vec<NX>& ilx_next = inv_Lx_[k + 1];
                const Vec<NX>& lx_next = Lx_[k + 1];

                // Quu ← inv_Lu · Quu · inv_Lu
                for (int r = 0; r < NU; ++r)
                    for (int c = 0; c < NU; ++c)
                        stages[k].Quu(r, c) *= ilu[r] * ilu[c];

                // Qux ← inv_Lu · Qux · inv_Lx
                for (int r = 0; r < NU; ++r)
                    for (int c = 0; c < NX; ++c)
                        stages[k].Qux(r, c) *= ilu[r] * ilx[c];

                // Qxu = Qux^T — recompute from transformed Qux
                for (int r = 0; r < NX; ++r)
                    for (int c = 0; c < NU; ++c)
                        stages[k].Qxu(r, c) = stages[k].Qux(c, r);

                // qu ← inv_Lu · qu
                for (int i = 0; i < NU; ++i)
                    stages[k].qu[i] *= ilu[i];

                // A ← Lx_{k+1} · A · inv_Lx_k
                //   A(i,j) *= Lx_{k+1}[i] * inv_Lx_k[j]
                for (int r = 0; r < NX; ++r)
                    for (int c = 0; c < NX; ++c)
                        stages[k].A(r, c) *= lx_next[r] * ilx[c];

                // B ← Lx_{k+1} · B · inv_Lu_k
                for (int r = 0; r < NX; ++r)
                    for (int c = 0; c < NU; ++c)
                        stages[k].B(r, c) *= lx_next[r] * ilu[c];

                // Constraint Jacobians: column scaling only.
                // d is NOT scaled — the linearization convention is
                //   g(x̄,ū) + Cx·Δx + Cu·Δu + s + ds = 0
                // where Δx,Δu are step directions.  Since Cx·inv_Lx·(Lx·dx_phys)=Cx·dx_phys,
                // using scaled Cx with scaled dx̂ (before recovery) gives the correct physical ds.
                if (NC_ > 0) {
                    // Cx ← Cx · inv_Lx_k  (each column j scaled by inv_Lx[j])
                    for (int r = 0; r < NC_; ++r)
                        for (int c = 0; c < NX; ++c)
                            stages[k].Cx(r, c) *= ilx[c];

                    // Cu ← Cu · inv_Lu_k
                    for (int r = 0; r < NC_; ++r)
                        for (int c = 0; c < NU; ++c)
                            stages[k].Cu(r, c) *= ilu[c];
                }
            }

            // c ← Lx_{k+1} · c  (use next stage's Lx for dynamics)
            // For terminal stage (k == HORIZON), no dynamics — but c is unused.
            if (k < HORIZON) {
                const Vec<NX>& lx_next = Lx_[k + 1];
                for (int i = 0; i < NX; ++i)
                    stages[k].c[i] *= lx_next[i];
            }

            // Terminal stage: scale Cx only (no Cu, no d — d is physical)
            if (k == HORIZON && NC_ > 0) {
                for (int r = 0; r < NC_; ++r)
                    for (int c = 0; c < NX; ++c)
                        stages[k].Cx(r, c) *= ilx[c];
            }
        }
    }

    // ═══════════════════════════════════════════════════════════════════
    //  Recover physical primal step from scaled step.
    //  dx = inv_Lx · dx̂,  du = inv_Lu · dû
    // ═══════════════════════════════════════════════════════════════════

    void recover_primal_step(RiccatiWorkspace<NX, NU, HORIZON>& ws) const {
        for (int k = 0; k <= HORIZON; ++k) {
            for (int i = 0; i < NX; ++i)
                ws.dx[k][i] *= inv_Lx_[k][i];
            if (k < HORIZON) {
                for (int i = 0; i < NU; ++i)
                    ws.du[k][i] *= inv_Lu_[k][i];
            }
        }
    }

    // ═══════════════════════════════════════════════════════════════════
    //  Recover physical dual variables from scaled duals.
    //  Costate:    ν = Lx · ν̂   (stored in ws.p)
    //  Constraint: λ = λ̂        (invariant — no scaling needed)
    // ═══════════════════════════════════════════════════════════════════

    void recover_dual_step(RiccatiWorkspace<NX, NU, HORIZON>& ws) const {
        for (int k = 0; k <= HORIZON; ++k) {
            for (int i = 0; i < NX; ++i)
                ws.p[k][i] *= Lx_[k][i];
            // λ is invariant — no operation needed
        }
    }

    // ═══════════════════════════════════════════════════════════════════
    //  Scale initial state residual for forward pass.
    //  dx̂0 = Lx · dx0
    // ═══════════════════════════════════════════════════════════════════

    void scale_dx0(Vec<NX>& dx0) const {
        for (int i = 0; i < NX; ++i)
            dx0[i] *= Lx_[0][i];
    }

    // ═══════════════════════════════════════════════════════════════════
    //  Diagnostics: condition number estimate from Hessian diagonal
    // ═══════════════════════════════════════════════════════════════════

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
    //  Data — diagonal scaling stored as vectors (not full matrices)
    // ═══════════════════════════════════════════════════════════════════

    Vec<NX> Lx_[HORIZON + 1];      // sqrt of Qxx diagonal (with floor)
    Vec<NU> Lu_[HORIZON];           // sqrt of Quu diagonal (with floor)
    Vec<NX> inv_Lx_[HORIZON + 1];  // 1 / Lx
    Vec<NU> inv_Lu_[HORIZON];       // 1 / Lu
    bool computed_ = false;
};

} // namespace nmpc
