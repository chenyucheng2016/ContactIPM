#pragma once
/**
 * @file    nmpc_riccati.hpp
 * @brief   Riccati backward-forward recursion for structured KKT solve.
 *
 * Exploits the banded/temporal structure of the multiple-shooting NLP.
 * Each SQP subproblem is an equality-constrained QP with dynamics constraints.
 * The Riccati recursion solves this in O(N·(nx+nu)³) time and O(N·nx²) memory.
 *
 * Key numerical safeguards:
 *   - Levenberg-Marquardt regularization on the control Hessian S matrix
 *   - Pivot monitoring on LDL^T factorization
 *   - Condition number estimation with adaptive regularization
 */

#include "nmpc_core.hpp"
#include "nmpc_problem.hpp"

namespace nmpc {

// ─────────────────────────────────────────────────────────────────────────────
//  Riccati workspace – allocated once per solver construction
// ─────────────────────────────────────────────────────────────────────────────

template <int NX, int NU, int HORIZON>
struct RiccatiWorkspace {
    // Backward-pass storage: P_k matrices and p_k vectors at each stage
    SymMat<NX> P[HORIZON + 1];   // cost-to-go Hessian
    Vec<NX>     p[HORIZON + 1];   // cost-to-go gradient

    // Feedback gains
    Mat<NU, NX> K[HORIZON];      // Δu = -K Δx - d
    Vec<NU>     d[HORIZON];       // feedforward term

    // Per-stage scratch
    SymMat<NU> S;                 // Quu + B^T P B  (Schur complement)
    SymMat<NU> S_fact[HORIZON];   // S factorization per stage (for RHS reuse)
    Mat<NU, NX> Qux_plus_BtPA;   // Qux + B^T P A

    Mat<NX, NU> BtP;             // B^T P
    Mat<NU, NX> K_times_S;       // K^T S (temporary for P update)

    // Per-stage cached values (stored in backward_lhs, reused in backward_rhs/compute_pk)
    Mat<NX, NU> BtP_stages[HORIZON];
    Mat<NU, NX> Qux_plus_BtPA_stages[HORIZON];

    // Forward pass
    Vec<NX> dx[HORIZON + 1];     // state step
    Vec<NU> du[HORIZON];         // control step

    // Regularization
    double reg_S      = 1e-8;    // base regularization for S
    double reg_adaptive = 1e-8;  // current adapted value
};

// ─────────────────────────────────────────────────────────────────────────────
//  Riccati recursion solver  (static methods, no state – all through workspace)
// ─────────────────────────────────────────────────────────────────────────────

template <int NX, int NU, int NC, int HORIZON>
struct RiccatiSolver {
    using Dims  = ProblemDimensions<NX, NU, NC, HORIZON>;
    using Stage = StageData<NX, NU, NC>;
    using WS    = RiccatiWorkspace<NX, NU, HORIZON>;

    // ═════════════════════════════════════════════════════════════════
    //  Phase 1: backward LHS  (factorize S, compute P_k, K_k)
    //  These depend only on Hessian (Qxx,Quu,Qux) and dynamics (A,B,c).
    //  Called once per Newton iteration — reuse for predictor+corrector.
    // ═════════════════════════════════════════════════════════════════

    static Status backward_lhs(Stage stages[], WS& ws,
                                double reg_base, double& reg_used)
    {
        const int N = HORIZON;

        ws.P[N].copy_lower_from(stages[N].Qxx);
        for (int i = 0; i < NX; ++i)
            ws.P[N](i, i) += reg_base * std::max(std::fabs(ws.P[N](i, i)), 1e-14);  // scale-invariant
        reg_used = reg_base;

        for (int k = N - 1; k >= 0; --k) {
            Stage& s = stages[k];
            const SymMat<NX>& P_next = ws.P[k + 1];

            // S = Quu + B^T P_{k+1} B
            ws.S.zero();
            for (int i = 0; i < NU; ++i)
                for (int j = 0; j <= i; ++j)
                    ws.S(i, j) = s.Quu(i, j);

            for (int r = 0; r < NU; ++r)
                for (int c = 0; c < NX; ++c) {
                    double sum = 0.0;
                    for (int m = 0; m < NX; ++m)
                        sum += s.B(m, r) * P_next(m, c);
                    ws.BtP(r, c) = sum;
                }
            for (int r = 0; r < NU; ++r)
                for (int c = 0; c <= r; ++c) {
                    double sum = 0.0;
                    for (int m = 0; m < NX; ++m)
                        sum += ws.BtP(r, m) * s.B(m, c);
                    ws.S(r, c) += sum;
                }

            // Qup = Qux + B^T P A
            for (int r = 0; r < NU; ++r)
                for (int c = 0; c < NX; ++c) {
                    double sumPA = 0.0;
                    for (int m = 0; m < NX; ++m)
                        sumPA += ws.BtP(r, m) * s.A(m, c);
                    ws.Qux_plus_BtPA(r, c) = s.Qux(r, c) + sumPA;
                }

            // Cache BtP and Qux_plus_BtPA for this stage
            ws.BtP_stages[k] = ws.BtP;
            ws.Qux_plus_BtPA_stages[k] = ws.Qux_plus_BtPA;

            // Regularize & factorize S
            // Scale-aware regularization floor
            // Use max diagonal (not max abs entry) for scale invariance:
            // max_diag(inv_Lu·S·inv_Lu) = inv_Lu^2 · max_diag(S) which scales correctly.
            double s_max_diag = 0.0;
            for (int i = 0; i < NU; ++i)
                s_max_diag = std::max(s_max_diag, std::fabs(ws.S(i, i)));

            double min_diag = 1e100;
            for (int i = 0; i < NU; ++i)
                if (ws.S(i, i) < min_diag) min_diag = ws.S(i, i);
            double reg = reg_base;
            if (min_diag < 0.0)
                reg = std::max(reg, -min_diag * 1.1);
            // NOTE: removed "reg = max(reg, 1e-10 * s_max_diag)" floor.
            // That floor was NOT scale-invariant: s_max_diag changes under
            // diagonal scaling, so the relative perturbation reg differed
            // between physical and scaled spaces, breaking the similarity
            // transform property of the preconditioner.

            // Save original S for clean restore on each retry
            SymMat<NU> S_save = ws.S;

            bool factored = false;
            for (int attempt = 0; attempt < 6; ++attempt) {
                ws.S = S_save;  // full restore before each attempt
                // Scale-invariant regularization: reg * |S(i,i)| instead of reg * 1
                // This ensures reg·diag(S) transforms correctly under diagonal scaling.
                for (int i = 0; i < NU; ++i)
                    ws.S(i, i) += reg * std::max(std::fabs(S_save(i, i)), 1e-14);
                double d_min = 0.0;
                if (ws.S.ldlt_factorize(1e-14, &d_min)) { factored = true; break; }
                // Layer 2A: informed bump from actual indefiniteness magnitude
                reg = std::max(reg * 10.0, 2.0 * std::fabs(d_min) + 1e-12);
            }
            if (!factored) return Status::KKT_SINGULAR;
            if (reg > reg_used) reg_used = reg;

            // Save factorized S for later RHS solves (backward_rhs uses S_fact[k])
            ws.S_fact[k] = ws.S;

            // K = S^{-1} * Qup   (NU×NX)
            for (int col = 0; col < NX; ++col) {
                Vec<NU> rhs;
                for (int r = 0; r < NU; ++r) rhs[r] = ws.Qux_plus_BtPA(r, col);
                ws.S_fact[k].ldlt_solve(rhs);
                for (int r = 0; r < NU; ++r) ws.K[k](r, col) = rhs[r];
            }

            // P_k = Qxx + A^T P A - Qup^T K
            compute_Pk(ws, s, P_next, k, reg_base);
        }

        return Status::SUCCESS;
    }

    // ═════════════════════════════════════════════════════════════════
    //  Phase 2: backward RHS  (compute p_k, d_k from qx, qu)
    //  Only depends on linear terms qx/qu — cheap, O(N·(nx²+nu²)).
    //  Re-run after changing RHS (e.g. corrector step in IPM).
    //  Requires backward_lhs() already called (P_k, K_k, S factorization).
    // ═════════════════════════════════════════════════════════════════

    static Status backward_rhs(Stage stages[], WS& ws)
    {
        const int N = HORIZON;

        ws.p[N] = stages[N].qx;

        for (int k = N - 1; k >= 0; --k) {
            Stage& s = stages[k];
            const Vec<NX>& p_next = ws.p[k + 1];
            const SymMat<NX>& P_next = ws.P[k + 1];

            // Load cached BtP = B_k^T · P_{k+1} from backward_lhs
            ws.BtP = ws.BtP_stages[k];

            // d = S^{-1} * (qu + B^T p_{k+1} + B^T P_{k+1} c)
            // Use the factorization stored from backward_lhs
            {
                Vec<NU> rhs_vec;
                for (int r = 0; r < NU; ++r) {
                    double btp = 0.0;
                    for (int m = 0; m < NX; ++m) btp += s.B(m, r) * p_next[m];
                    double btpc = 0.0;
                    for (int m = 0; m < NX; ++m) btpc += ws.BtP(r, m) * s.c[m];
                    rhs_vec[r] = s.qu[r] + btp + btpc;
                }
                ws.S_fact[k].ldlt_solve(rhs_vec);
                ws.d[k] = rhs_vec;
            }

            compute_pk(ws, s, p_next, k);
        }

        return Status::SUCCESS;
    }

    // ── Combined backward (LHS + RHS) for convenience ─────────────

    static Status backward(Stage stages[], WS& ws,
                            double reg_base, double& reg_used)
    {
        Status st = backward_lhs(stages, ws, reg_base, reg_used);
        if (st != Status::SUCCESS) return st;
        return backward_rhs(stages, ws);
    }

    // ── Forward pass ────────────────────────────────────────────────────

    static Status forward(Stage stages[], WS& ws,
                          Vec<NX>& dx0)  // dx0 = x̄ - x0 (initial state residual)
    {
        const int N = HORIZON;

        ws.dx[0] = dx0;

        for (int k = 0; k < N; ++k) {
            // Δu_k = -K_k Δx_k - d_k
            for (int r = 0; r < NU; ++r) {
                double sum = 0.0;
                for (int c = 0; c < NX; ++c)
                    sum += ws.K[k](r, c) * ws.dx[k][c];
                ws.du[k][r] = -sum - ws.d[k][r];
            }

            // Δx_{k+1} = A_k Δx_k + B_k Δu_k + c_k
            ws.dx[k + 1].zero();
            for (int r = 0; r < NX; ++r) {
                double ax = 0.0, bu = 0.0;
                for (int c = 0; c < NX; ++c)
                    ax += stages[k].A(r, c) * ws.dx[k][c];
                for (int c = 0; c < NU; ++c)
                    bu += stages[k].B(r, c) * ws.du[k][c];
                ws.dx[k + 1][r] = ax + bu + stages[k].c[r];
            }
        }

        return Status::SUCCESS;
    }

private:
    // ── P_k = Qxx + A^T P_{k+1} A - Qup^T * K ──────────────────────────

    static void compute_Pk(WS& ws, const Stage& s,
                           const SymMat<NX>& P_next, int k, double reg)
    {
        SymMat<NX>& Pk = ws.P[k];

        Pk.copy_lower_from(s.Qxx);

        for (int r = 0; r < NX; ++r) {
            for (int c = 0; c <= r; ++c) {
                double val = 0.0;
                for (int i = 0; i < NX; ++i) {
                    double pi_row = 0.0;
                    for (int j = 0; j < NX; ++j)
                        pi_row += P_next(i, j) * s.A(j, c);
                    val += s.A(i, r) * pi_row;
                }
                Pk(r, c) += val;
            }
        }

        for (int r = 0; r < NX; ++r) {
            for (int c = 0; c <= r; ++c) {
                double sub = 0.0;
                for (int m = 0; m < NU; ++m)
                    sub += ws.Qux_plus_BtPA(m, r) * ws.K[k](m, c);
                Pk(r, c) -= sub;
            }
        }

        for (int i = 0; i < NX; ++i)
            Pk(i, i) += reg * std::max(std::fabs(Pk(i, i)), 1e-14);  // scale-invariant
    }

    // ── p_k = qx + A^T (p_{k+1} + P_{k+1} * c) - Qup^T * d ─────────────

    static void compute_pk(WS& ws, const Stage& s,
                           const Vec<NX>& p_next, int k)
    {
        Vec<NX>& pk = ws.p[k];

        pk = s.qx;

        Vec<NX> temp;
        for (int i = 0; i < NX; ++i) {
            double pc = 0.0;
            for (int j = 0; j < NX; ++j)
                pc += ws.P[k + 1](i, j) * s.c[j];
            temp[i] = p_next[i] + pc;
        }
        for (int i = 0; i < NX; ++i) {
            double at = 0.0;
            for (int j = 0; j < NX; ++j)
                at += s.A(j, i) * temp[j];
            pk[i] += at;
        }

        // p_k = qx + A^T (p_{k+1} + P_{k+1} * c) - Qup^T * d
        // Use cached Qup = Qux + B^T P_{k+1} A from backward_lhs
        const auto& Qup = ws.Qux_plus_BtPA_stages[k];
        for (int i = 0; i < NX; ++i) {
            double sub = 0.0;
            for (int m = 0; m < NU; ++m) {
                sub += Qup(m, i) * ws.d[k][m];
            }
            pk[i] -= sub;
        }
    }
};

} // namespace nmpc
