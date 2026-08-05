#pragma once
/**
 * @file    nmpc_preconditioner.hpp
 * @brief   Horizon-consistent diagonal preconditioner for NMPC coordinates.
 *
 * Computes one horizon-wide scaling per physical state/control coordinate ONCE
 * per MPC solve. Cost curvature defines the scale when present; reduced-system
 * log-barrier curvature is a fallback only for structurally unpenalized,
 * persistently dynamics-coupled contact/auxiliary coordinates:
 *
 *     D_i = max_k H_{k,ii},                         if D_i > 0
 *     D_i = max_k [C^T diag(lambda / s) C]_{k,ii}, otherwise if persistent
 *     L_i = sqrt(D_i), with identity for a structural zero
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
        double xcost[NX] = {};
        double ucost[NU] = {};
        double xbarrier[NX] = {};
        double ubarrier[NU] = {};
        double udynamics_max[NU] = {};
        double udynamics_min[NU];
        for (int i = 0; i < NU; ++i) udynamics_min[i] = 1e100;

        for (int k = 0; k <= HORIZON; ++k) {
            double stage_xbarrier[NX] = {};
            double stage_ubarrier[NU] = {};
            for (int i = 0; i < NX; ++i)
                xcost[i] = std::max(
                    xcost[i], std::max(stages[k].Qxx(i, i), 0.0));
            if (k < HORIZON)
                for (int i = 0; i < NU; ++i) {
                    ucost[i] = std::max(
                        ucost[i], std::max(stages[k].Quu(i, i), 0.0));
                    double column_norm = 0.0;
                    for (int r = 0; r < NX; ++r)
                        column_norm = std::max(
                            column_norm, std::fabs(stages[k].B(r, i)));
                    udynamics_max[i] = std::max(udynamics_max[i], column_norm);
                    udynamics_min[i] = std::min(udynamics_min[i], column_norm);
                }

            // Barrier curvature supplies units only when the objective has no
            // curvature for that coordinate. It must not override a known cost
            // scale merely because one contact is nearly active at one node.
            for (int j = 0; j < NC_; ++j) {
                if (stages[k].s[j] <= 1e-14 || stages[k].lambda[j] <= 0.0)
                    continue;
                const double ratio = stages[k].lambda[j] / stages[k].s[j];
                for (int i = 0; i < NX; ++i)
                    stage_xbarrier[i] += ratio * stages[k].Cx(j, i)
                                                * stages[k].Cx(j, i);
                if (k < HORIZON)
                    for (int i = 0; i < NU; ++i)
                        stage_ubarrier[i] += ratio * stages[k].Cu(j, i)
                                                    * stages[k].Cu(j, i);
            }
            for (int i = 0; i < NX; ++i)
                xbarrier[i] = std::max(xbarrier[i], stage_xbarrier[i]);
            if (k < HORIZON)
                for (int i = 0; i < NU; ++i)
                    ubarrier[i] = std::max(ubarrier[i], stage_ubarrier[i]);
        }

        // Repeated OCP variables keep the same physical units at every node.
        // A horizon-wide scale avoids injecting artificial L_{k+1}/L_k
        // factors into the dynamics.
        for (int k = 0; k <= HORIZON; ++k) {
            for (int i = 0; i < NX; ++i) {
                const double d = xcost[i] > 1e-14 ? xcost[i]
                    : (xbarrier[i] > 1e-14 ? xbarrier[i] : 1.0);
                Lx_[k][i] = std::sqrt(d);
                inv_Lx_[k][i] = 1.0 / Lx_[k][i];
            }
            if (k < HORIZON)
                for (int i = 0; i < NU; ++i) {
                    const double d = ucost[i] > 1e-14 ? ucost[i]
                        : (udynamics_max[i] > 1e-14
                           && udynamics_min[i] >= 1e-2 * udynamics_max[i]
                           && ubarrier[i] > 1e-14
                            ? ubarrier[i] : 1.0);
                    Lu_[k][i] = std::sqrt(d);
                    inv_Lu_[k][i] = 1.0 / Lu_[k][i];
                }
        }
        computed_ = true;
    }

    /**
     * Recompute scaling from the physical Hessian plus the variable-bound
     * curvature used by the condensed primal-dual KKT system. Near an active
     * lower bound,
     *
     *   d_L = x - x_L ~= mu / z_L,
     *   d2b/dx2 = z_L / d_L ~= mu / d_L^2.
     *
     * Updating L_i = sqrt(max(H_ii + d2b_i, floor)) therefore keeps the
     * diagonal near one in scaled coordinates as mu falls.
     */
    template <int NC_>
    void compute_bound_aware(const StageData<NX, NU, NC_> stages[],
                             const Vec<NX>& x_lb, const Vec<NX>& x_ub,
                             const Vec<NU>& u_lb, const Vec<NU>& u_ub,
                             double /*mu*/, double distance_floor) {
        for (int k = 0; k <= HORIZON; ++k) {
            double xdiag[NX];
            double max_dx = 0.0;
            for (int i = 0; i < NX; ++i) {
                double d = std::max(stages[k].Qxx(i, i), 0.0);
                if (x_lb[i] > -1e19) {
                    double dist = std::max(stages[k].x[i] - x_lb[i], 0.0)
                                + distance_floor;
                    d += stages[k].z_L_x[i] / dist;
                }
                if (x_ub[i] < 1e19) {
                    double dist = std::max(x_ub[i] - stages[k].x[i], 0.0)
                                + distance_floor;
                    d += stages[k].z_U_x[i] / dist;
                }
                xdiag[i] = d;
                max_dx = std::max(max_dx, d);
            }
            const bool xzero = max_dx < 1e-14;
            const double xfloor = xzero ? 1.0 : 1e-8 * max_dx;
            for (int i = 0; i < NX; ++i) {
                double d = xzero ? 1.0 : std::max(xdiag[i], xfloor);
                Lx_[k][i] = std::sqrt(d);
                inv_Lx_[k][i] = 1.0 / Lx_[k][i];
            }

            if (k < HORIZON) {
                double udiag[NU];
                double max_du = 0.0;
                for (int i = 0; i < NU; ++i) {
                    double d = std::max(stages[k].Quu(i, i), 0.0);
                    if (u_lb[i] > -1e19) {
                        double dist = std::max(stages[k].u[i] - u_lb[i], 0.0)
                                    + distance_floor;
                        d += stages[k].z_L_u[i] / dist;
                    }
                    if (u_ub[i] < 1e19) {
                        double dist = std::max(u_ub[i] - stages[k].u[i], 0.0)
                                    + distance_floor;
                        d += stages[k].z_U_u[i] / dist;
                    }
                    udiag[i] = d;
                    max_du = std::max(max_du, d);
                }
                const bool uzero = max_du < 1e-14;
                const double ufloor = uzero ? 1.0 : 1e-8 * max_du;
                for (int i = 0; i < NU; ++i) {
                    double d = uzero ? 1.0 : std::max(udiag[i], ufloor);
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
