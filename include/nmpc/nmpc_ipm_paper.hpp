#pragma once
/**
 * @file    nmpc_ipm_paper.hpp
 * @brief   Paper-aligned primal-dual IPM for convex multistage NLP.
 *
 * Follows: Domahidi, Zgraggen, Zeilinger, Morari, Jones (CDC 2012)
 *   "Efficient Interior Point Methods for Multistage Problems
 *    Arising in Receding Horizon Control"
 *
 * Algorithm:  Primal-dual IPM applied directly to the barrier subproblem
 *   of a convex NLP, using Mehrotra predictor-corrector search directions
 *   and Riccati recursion to exploit the multistage KKT structure.
 *
 * The three key ideas from the paper:
 *   1. Mehrotra predictor-corrector: two KKT solves per iteration
 *      (affine predictor → centering correction)
 *   2. Block elimination of inequality slacks/duals from the KKT
 *   3. Riccati recursion (equivalent to block LDL^T) on the reduced
 *      multistage-banded KKT system → O(N·(nx+nu)³) per iteration
 */

#include "nmpc_core.hpp"
#include "nmpc_problem.hpp"
#include "nmpc_riccati.hpp"
#include "nmpc_barrier_manager.hpp"
#include "nmpc_filter_ls.hpp"
#include "nmpc_preconditioner.hpp"

namespace nmpc {

// ─────────────────────────────────────────────────────────────────────────────
//  Paper-aligned IPM parameters
// ─────────────────────────────────────────────────────────────────────────────

struct PaperIPMParams {
    // Barrier
    double mu_init       = 1.0;
    double mu_min        = 5e-4;    // barrier parameter floor

    // Mehrotra predictor-corrector
    double centering     = 0.1;     // minimum σ ∈ (0,1)

    // Single-loop convergence — per-condition tolerances
    int    max_iters     = 100;     // total Newton iterations
    double tol_primal    = 1e-6;    // primal infeasibility tolerance
    double tol_compl     = 1e-6;    // complementarity tolerance (|s·λ − μ|)
    double tol_ineq      = 1e-8;    // inequality tolerance (allow small negative s/λ)
    double tol_stat      = 0.5;     // stationarity tolerance (‖∇L‖∞ including costates)

    // Fraction-to-boundary
    double tau           = 0.999;    // tighter: 0.99 (was 0.995)

    // ── Globalization (line-search) ──
    int    max_ls_iters   = 20;
    double armijo_c       = 1e-4;

    // === Globalization: Second-Order Correction ===
    int    soc_max         = 4;       // max SOC attempts per iteration
    double kappa_soc       = 0.99;    // SOC stall threshold: abort if theta_soc > kappa_soc * theta_prev

    // === Complementarity Safeguard (HPIPM-style) ===
    double m_safe          = 0.5;    // enforce lambda_j * s_j >= m_safe * mu


    // === Barrier Update (σ-modulation) ===
    double kappa_eps       = 10.0;    // E_mu <= kappa_eps * mu → barrier solved
    double sigma_exp_easy   = 1.5;    // easy subproblems: σ^1.5 (more aggressive)
    double sigma_exp_normal = 1.0;    // standard Mehrotra
    double sigma_exp_hard   = 0.5;    // hard subproblems: σ^0.5 (conservative)
    int    fast_threshold   = 2;      // ≤ this many iters → easy
    int    slow_threshold   = 4;      // ≥ this many iters → hard
    int    max_same_mu     = 30;      // force mu reduction after this many iterations at same mu

    // === Nonlinear KKT Iterative Refinement (Shamanskii chord) ===
    bool   enable_refinement  = false;   // enable direction refinement before line search
    int    max_refine_iters   = 5;       // max chord passes
    double refine_tol         = 1e-6;    // nonlinear residual tolerance
    double refine_diverge_fac = 1.5;     // divergence guard: stop if ||r|| > fac * prev

    // Output
    int    verbosity     = 0;

    // Preconditioning
    bool   enable_preconditioner = false;  // diagonal Jacobi preconditioner
};

// ─────────────────────────────────────────────────────────────────────────────
//  Linear KKT residual diagnostics
// ─────────────────────────────────────────────────────────────────────────────

struct KKTLinearResiduals {
    // Per-equation-type max residuals (∞-norm)
    double max_dyn_res;       // dynamics: ||Δx_{k+1} - A·Δx_k - B·Δu_k - c_k||_∞
    double max_feas_res;      // constraint feasibility: ||Δs + C·Δz + g + s||_∞
    double max_stat_x_res;    // x-stationarity (full KKT, Bellman equation)
    double max_stat_u_res;    // u-stationarity
    double max_stat_term_res; // terminal x-stationarity
    double max_comp_res;      // complementarity: ||s·Δλ + λ·Δs + sλ + cross - σμ||_∞

    // Riccati-specific (pure reduced system, excluding barrier terms)
    double max_riccati_x_res; // Riccati x-stationarity: qx + H̄^xx·Δx + (H̄^ux)^T·Δu - A^T·ν_{k+1} + ν_k
    double max_riccati_u_res; // Riccati u-stationarity: qu + H̄^uu·Δu + H̄^ux·Δx + B^T·ν_{k+1}

    // RHS norms for relative residual computation
    double rhs_dyn_norm;      // max ||c_k||_∞
    double rhs_feas_norm;     // max ||g + s||_∞
    double rhs_stat_x_norm;   // max ||q̃^x_k||_∞
    double rhs_stat_u_norm;   // max ||q̃^u_k||_∞
    double rhs_stat_term_norm;
    double rhs_comp_norm;     // max |σμ|

    // Relative residuals: res / max(rhs_norm, 1e-14)
    double rel_dyn_res;
    double rel_feas_res;
    double rel_stat_x_res;
    double rel_stat_u_res;
    double rel_stat_term_res;
    double rel_comp_res;

    // Aggregates
    double max_abs_res;       // max over all linear KKT equations
    double max_rel_res;       // max relative residual
    int    worst_stage;       // stage with worst residual
    int    worst_eq_type;     // 0=dyn, 1=stat_x, 2=stat_u, 3=term, 4=comp, 5=ricc_x, 6=ricc_u, 7=feas

    // Quality assessment
    bool   is_well_solved()  const { return max_rel_res < 1e-6; }
    bool   is_acceptable()   const { return max_rel_res < 1e-3; }
    bool   is_poor()         const { return max_rel_res > 1e-1; }

    const char* quality_label() const {
        if (is_well_solved()) return "WELL_SOLVED";
        if (is_acceptable())  return "ACCEPTABLE";
        if (is_poor())        return "POOR";
        return "MARGINAL";
    }
};

// ─────────────────────────────────────────────────────────────────────────────
//  Paper-aligned IPM Solver  (for convex multistage NLP)
// ─────────────────────────────────────────────────────────────────────────────

template <int NX, int NU, int NC, int HORIZON>
class PaperIPMSolver {
public:
    using Prob  = NMPCProblem<NX, NU, NC, HORIZON>;
    using Stage = StageData<NX, NU, NC>;
    using WS    = RiccatiWorkspace<NX, NU, HORIZON>;
    using Ricc  = RiccatiSolver<NX, NU, NC, HORIZON>;
    using Prec  = HessianPreconditioner<NX, NU, HORIZON>;

    PaperIPMSolver() = default;

    // ── Step snapshot for regression testing ──────────────────────
    // Captures the Newton step and globalization data from the last
    // iteration.  Used by test_precond_invariance to verify that
    // baseline and preconditioned solves produce identical steps.
    struct StepSnapshot {
        const Vec<NX>* dx;   // [HORIZON+1]  primal x-step (physical)
        const Vec<NU>* du;   // [HORIZON]    primal u-step (physical)
        const Vec<NC>* ds;   // [HORIZON+1]  slack step
        const Vec<NC>* dlambda; // [HORIZON+1]  multiplier step
        double alpha_p;      // primal step size
        double alpha_lambda; // dual step size
        double sigma;        // Mehrotra centering parameter
        double mu;           // barrier parameter
    };
    StepSnapshot get_step_snapshot() const {
        return { riccati_ws_.dx, riccati_ws_.du, ds_, dlambda_,
                 last_alpha_p_, alpha_lambda_, sigma_, mu_ };
    }

    // ── Diagnostic: expose Riccati internals for invariance debugging ──
    struct RiccatiDiag {
        Vec<NU> d_feedforward;     // d[0] (feedforward term, scaled space)
        Vec<NU> S_diag;            // S_fact[0] diagonal (scaled space)
        Vec<NX> p_terminal;        // p[N] (scaled space)
        Vec<NX> p_stage0;          // p[0] (scaled space)
        Vec<NU> rhs_d;             // RHS for d[0]: qu + B^T*p[1] + B^T*P[1]*c
        double reg_used;
        double sigma;
        double mu;
    };
    RiccatiDiag get_riccati_diag() const {
        RiccatiDiag r;
        r.d_feedforward = riccati_ws_.d[0];
        for (int i = 0; i < NU; ++i)
            r.S_diag[i] = riccati_ws_.S_fact[0](i, i);
        r.p_terminal = riccati_ws_.p[HORIZON];
        r.p_stage0 = riccati_ws_.p[0];
        // Reconstruct RHS for d[0]: qu[0] + B[0]^T*p[1] + B[0]^T*P[1]*c[0]
        const auto& s0 = riccati_stages_[0];
        const auto& p1 = riccati_ws_.p[1];
        const auto& P1 = riccati_ws_.P[1];
        for (int r = 0; r < NU; ++r) {
            double btp = 0.0;
            for (int m = 0; m < NX; ++m) btp += s0.B(m, r) * p1[m];
            double btpc = 0.0;
            for (int m = 0; m < NX; ++m)
                for (int n = 0; n < NX; ++n)
                    btpc += s0.B(m, r) * P1(m, n) * s0.c[n];
            r.rhs_d[r] = s0.qu[r] + btp + btpc;
        }
        r.reg_used = reg_used_;
        r.sigma = sigma_;
        r.mu = mu_;
        return r;
    }

    Status init(const PaperIPMParams& params = PaperIPMParams{}) {
        params_ = params;
        mu_     = params_.mu_init;

        // Auto-derive stationarity tolerance from mu_min if not set
        if (params_.tol_stat < 0.0) {
            params_.tol_stat = 100.0 * params_.mu_min;
            if (params_.verbosity >= 1)
                printf("  [auto] tol_stat = %.1e  (100 * mu_min)\n", params_.tol_stat);
        }

        return Status::SUCCESS;
    }

    // ── Main solve (true single-loop IPM) ─────────────────────────

    Status solve(Prob& problem, SolverStats& out_stats) {
        Status st = problem.validate();
        if (st != Status::SUCCESS) return st;

        prob_ = &problem;
        initialize_from_problem();
        evaluate_model();          // ensure first KKT uses fresh data

        // Compute preconditioner scaling ONCE per MPC solve (outside Newton loop)
        if (params_.enable_preconditioner) {
            prec_.compute(prob_->stages);
            // Note: transform_qp() is called inside the loop before compute_kkt_residuals
            // so that convergence checks use scaled residuals.
        }

        mu_ = params_.mu_init;
        cond_estimate_ = 1.0;
        has_costates_ = false;

        // Initialize barrier strategy
        {
            BarrierUpdateParams bup;
            bup.mu_init     = params_.mu_init;
            bup.mu_min      = params_.mu_min;
            bup.kappa_eps   = params_.kappa_eps;
            bup.sigma_exp_easy   = params_.sigma_exp_easy;
            bup.sigma_exp_normal = params_.sigma_exp_normal;
            bup.sigma_exp_hard   = params_.sigma_exp_hard;
            bup.fast_threshold   = params_.fast_threshold;
            bup.slow_threshold   = params_.slow_threshold;
            bup.max_same_mu = params_.max_same_mu;
            bup.m_safe      = params_.m_safe;
            bup.verbosity   = 0;  // solver loop handles logging
            barrier_strategy_.reset(bup);
        }

        // Initialize filter line search
        {
            FilterLSParams fp;
            fp.soc_max    = params_.soc_max;
            fp.kappa_soc  = params_.kappa_soc;
            fp.alpha_min  = 1e-14;
            fp.eta_phi    = params_.armijo_c;
            fp.verbosity  = params_.verbosity;
            filter_ls_.init(fp);
        }

        // Complementarity safeguard on initial iterates
        if (barrier_strategy_.m_safe() > 0.0) {
            if (params_.verbosity >= 2)
                printf("  [init: sz_complement(m_safe=%.3e, mu=%.2e)]\n",
                       barrier_strategy_.m_safe(), mu_);
            sz_complement(barrier_strategy_.m_safe());
        }

        int iter;

        // Per-iteration state tracking
        double alpha_p = 1.0, alpha_d = 1.0;
        int    ls_iters = 1;
        bool   accepted = false;
        bool   ls_failed = false;  // line search failure flag
        bool   model_evaluated = true;  // evaluate_model() already done before loop
        // Stagnation detection: break when stationarity stops improving.
        double stat_best = 1e100;
        int    stagnation_count = 0;
        int    final_status_at_break = -1;  // -1=unknown, 0=stagnation, 1=converged
        constexpr int STAGNATION_LIMIT = 30;

        for (iter = 0; iter < params_.max_iters; ++iter) {

            // 1. Evaluate model at current iterate (skip if already evaluated)
            if (!model_evaluated) {
                st = evaluate_model();
                if (st != Status::SUCCESS) return st;
            }
            model_evaluated = false;
            stages_scaled_ = false;

            // 2. Apply FIXED preconditioner scaling to derivatives (before KKT build)
            // After this, prob_->stages derivatives are in SCALED space.
            if (params_.enable_preconditioner) {
                prec_.transform_qp(prob_->stages);
                stages_scaled_ = true;
            }

            // 3. Compute KKT residuals (uses SCALED derivatives after transform_qp)
            compute_kkt_residuals();

            // TEMPORARY: physical stationarity diagnostic (no convergence decisions)
            if (params_.enable_preconditioner && params_.verbosity >= 2
                && (iter % 5 == 0 || barrier_strategy_.at_minimum(mu_))) {
                // Save scaled residuals
                double s_stat = stat_inf_, s_prim = primal_inf_, s_compl = compl_inf_;
                // Evaluate physical model and compute physical KKT
                evaluate_model();
                stages_scaled_ = false;
                compute_kkt_residuals();
                double p_stat = stat_inf_, p_prim = primal_inf_, p_compl = compl_inf_;
                // Restore scaled stages and residuals
                prec_.transform_qp(prob_->stages);
                stages_scaled_ = true;
                compute_kkt_residuals();
                stat_inf_ = s_stat; primal_inf_ = s_prim; compl_inf_ = s_compl;
                printf("  [PHYS] stat=%.4e prim=%.4e compl=%.4e | scaled stat=%.4e\n",
                       p_stat, p_prim, p_compl, s_stat);
            }

            // 4. Convergence check: KKT satisfied AND μ at minimum
            if (barrier_strategy_.at_minimum(mu_) && kkt_converged()) {
                final_status_at_break = 1;
                break;
            }

            // 3b. Stagnation detection is now handled post-Riccati (after
            //     compute_post_riccati_stationarity) where costates are
            //     consistent with the current model evaluation.
            if (barrier_strategy_.at_minimum(mu_) && params_.verbosity >= 3) {
                bool primal_ok  = (primal_inf_ <= params_.tol_primal);
                bool compl_ok   = (compl_inf_  <= params_.tol_compl);
                bool ineq_ok    = (ineq_viol_  >= -params_.tol_ineq);
                bool stat_ok    = (stat_inf_   <= params_.tol_stat);
                printf("  [conv: primal_ok=%d(%.2e<=%.2e) compl_ok=%d(%.2e<=%.2e) ineq_ok=%d(%.2e>=-%.2e) stat_ok=%d(%.2e<=%.2e)]\n",
                       primal_ok, primal_inf_, params_.tol_primal,
                       compl_ok, compl_inf_, params_.tol_compl,
                       ineq_ok, ineq_viol_, params_.tol_ineq,
                       stat_ok, stat_inf_, params_.tol_stat);
            }

            // 4. ── PREDICTOR: build LHS once, factorize ────────────
            build_kkt_lhs();
            st = solve_kkt_lhs();
            if (st != Status::SUCCESS) return st;

            // ── Inertia check: D pivots of S_fact[k] ──────────────
            if (params_.verbosity >= 2) {
                double min_pivot = 1e100, max_pivot = 0.0;
                int worst_k = -1;
                for (int k = 0; k < HORIZON; ++k) {
                    for (int i = 0; i < NU; ++i) {
                        double d = riccati_ws_.S_fact[k](i, i);
                        if (d < min_pivot) { min_pivot = d; worst_k = k; }
                        if (d > max_pivot) max_pivot = d;
                    }
                }
                // Also check terminal P[N] diagonal
                double min_pN = 1e100;
                for (int i = 0; i < NX; ++i) {
                    double d = riccati_ws_.P[HORIZON](i, i);
                    if (d < min_pN) min_pN = d;
                }
                const char* flag = (min_pivot < 1e-10) ? " WARN" : "";
                printf("  [inertia] S pivots=[%.2e, %.2e] worst_k=%d"
                       " P[N]_min=%.2e%s\n",
                       min_pivot, max_pivot, worst_k, min_pN, flag);
            }

            // ── Affine predictor (σ=0) ──────────────────────────
            build_kkt_rhs();
            st = solve_kkt_rhs_and_forward();
            if (st != Status::SUCCESS) return st;

            // Compute ds/dλ BEFORE recovering primal step to physical space.
            // recover_inequality_steps uses scaled Cx/Cu with scaled dx̂/dû from Riccati:
            //   Cx_scaled·dx̂ = (Cx·inv_Lx)·(Lx·dx_phys) = Cx·dx_phys  → correct physical ds.
            recover_inequality_steps(0.0);

            // Recover physical step from scaled Riccati solution
            if (params_.enable_preconditioner) {
                prec_.recover_primal_step(riccati_ws_);
                prec_.recover_dual_step(riccati_ws_);
            }

            // Copy predictor steps (avoids redundant save_predictor_step)
            for (int k = 0; k <= HORIZON; ++k) {
                ds_aff_[k]      = ds_[k];
                dlambda_aff_[k] = dlambda_[k];
            }

            // ── Compute affine FTB limits (for σ and barrier FSM) ────
            {
                double ap_aff = 1.0, ad_aff = 1.0;
                for (int k = 0; k <= HORIZON; ++k)
                    for (int j = 0; j < NC; ++j) {
                        if (ds_aff_[k][j] < -1e-16) {
                            double a = -params_.tau * prob_->stages[k].s[j] / ds_aff_[k][j];
                            if (a < ap_aff) ap_aff = a;
                        }
                        if (dlambda_aff_[k][j] < -1e-16) {
                            double a = -params_.tau * prob_->stages[k].lambda[j] / dlambda_aff_[k][j];
                            if (a < ad_aff) ad_aff = a;
                        }
                    }
                aff_alpha_p_ = ap_aff;
                aff_alpha_d_ = ad_aff;
            }

            // Adaptive σ scheduling with FTB-bottleneck override
            sigma_ = compute_adaptive_sigma();

            // 5. ── CORRECTOR: reuse LHS, update RHS with centering ──
            // Save sigma/mu for invariance debugging
            debug_sigma_ = sigma_;
            debug_mu_ = mu_;
            debug_primal_inf_ = primal_inf_;
            debug_compl_inf_ = compl_inf_;

            build_kkt_rhs();

            // Save PRISTINE Riccati stages right after KKT build (before SOC/LS)
            for (int kk = 0; kk <= HORIZON; ++kk) {
                debug_pristine_stages_[kk] = riccati_stages_[kk];
            }

            // Save Riccati internals for invariance debugging
            // NOTE: save BEFORE corrector forward — d[0] is from predictor here
            // We'll re-save after corrector below
            debug_P_term_ = riccati_ws_.P[HORIZON];
            debug_S_fact0_ = riccati_ws_.S_fact[0];

            st = solve_kkt_rhs_and_forward();
            if (st != Status::SUCCESS) return st;

            // Re-save Riccati internals AFTER corrector forward pass
            // d[0] and K[0] are now from the corrector
            for (int i = 0; i < NU; ++i) {
                debug_d0_[i] = riccati_ws_.d[0][i];
                for (int j = 0; j < NX; ++j)
                    debug_K0_(i, j) = riccati_ws_.K[0](i, j);
            }

            // ── Nonlinear KKT iterative refinement (DISABLED) ────
            // if (params_.enable_refinement) {
            //     refine_newton_direction();
            // }

            // Compute ds/dλ BEFORE recovering primal step (same as predictor).
            recover_inequality_steps(sigma_);

            // ── True scaled-space linear KKT residual (before recovery) ──
            // At this point ws.dx, ws.du, ws.p are ALL in scaled space.
            // This measures ||K̂·ẑ - r̂|| directly.
            if (params_.verbosity >= 2) {
                double max_ricc_u = 0.0, max_ricc_x = 0.0;
                double min_S_diag = 1e100, max_S_diag = 0.0;
                int worst_u_k = -1, worst_u_i = -1;
                double max_reg_perturb = 0.0;
                for (int kk = 0; kk < HORIZON; ++kk) {
                    const auto& rs = riccati_stages_[kk];
                    const auto& dx_k = riccati_ws_.dx[kk];
                    const auto& du_k = riccati_ws_.du[kk];
                    const auto& dx_k1 = riccati_ws_.dx[kk+1];
                    // ν_k = p_k + P_k·dx_k (scaled)
                    Vec<NX> nu_k;
                    for (int i = 0; i < NX; ++i) {
                        nu_k[i] = riccati_ws_.p[kk][i];
                        for (int j = 0; j < NX; ++j)
                            nu_k[i] += riccati_ws_.P[kk](i,j) * dx_k[j];
                    }
                    Vec<NX> nu_next;
                    for (int i = 0; i < NX; ++i) {
                        nu_next[i] = riccati_ws_.p[kk+1][i];
                        for (int j = 0; j < NX; ++j)
                            nu_next[i] += riccati_ws_.P[kk+1](i,j) * dx_k1[j];
                    }
                    // u-stationarity: rs.qu + H̄uu·du + H̄ux·dx + B^T·ν_{k+1}
                    for (int i = 0; i < NU; ++i) {
                        double r = rs.qu[i];
                        for (int j = 0; j < NU; ++j) r += rs.Quu(i,j) * du_k[j];
                        for (int j = 0; j < NX; ++j) r += rs.Qux(i,j) * dx_k[j];
                        for (int j = 0; j < NX; ++j) r += rs.B(j,i) * nu_next[j];
                        if (std::fabs(r) > max_ricc_u) {
                            max_ricc_u = std::fabs(r);
                            worst_u_k = kk; worst_u_i = i;
                        }
                    }
                    // Regularization perturbation estimate:
                    // Riccati solves (S + reg·D)·du = rhs, so the residual
                    // of the UNREGULARIZED equation is ≈ reg·D·du
                    for (int i = 0; i < NU; ++i) {
                        double perturb = reg_used_ * std::max(std::fabs(riccati_ws_.S_fact[kk](i,i)), 1e-14) * std::fabs(du_k[i]);
                        max_reg_perturb = std::max(max_reg_perturb, perturb);
                    }
                    // x-stationarity: rs.qx + H̄xx·dx + H̄ux^T·du + A^T·ν_{k+1} - ν_k
                    for (int i = 0; i < NX; ++i) {
                        double r = rs.qx[i];
                        for (int j = 0; j < NX; ++j) r += rs.Qxx(i,j) * dx_k[j];
                        for (int j = 0; j < NU; ++j) r += rs.Qux(j,i) * du_k[j];
                        for (int j = 0; j < NX; ++j) r += rs.A(j,i) * nu_next[j];
                        r -= nu_k[i];
                        max_ricc_x = std::max(max_ricc_x, std::fabs(r));
                    }
                    // S diagonal stats
                    for (int i = 0; i < NU; ++i) {
                        double d = riccati_ws_.S_fact[kk](i, i);
                        min_S_diag = std::min(min_S_diag, d);
                        max_S_diag = std::max(max_S_diag, d);
                    }
                }
                // Terminal stationarity
                {
                    const auto& rs = riccati_stages_[HORIZON];
                    const auto& dx_k = riccati_ws_.dx[HORIZON];
                    Vec<NX> nu_k;
                    for (int i = 0; i < NX; ++i) {
                        nu_k[i] = riccati_ws_.p[HORIZON][i];
                        for (int j = 0; j < NX; ++j)
                            nu_k[i] += riccati_ws_.P[HORIZON](i,j) * dx_k[j];
                    }
                    for (int i = 0; i < NX; ++i) {
                        double r = rs.qx[i];
                        for (int j = 0; j < NX; ++j) r += rs.Qxx(i,j) * dx_k[j];
                        r -= nu_k[i];
                        max_ricc_x = std::max(max_ricc_x, std::fabs(r));
                    }
                }
                double reg_S_ratio = reg_used_ / std::max(max_S_diag, 1e-14);
                printf("  [linKKT-scaled] ricc_x=%.3e ricc_u=%.3e(worst k=%d i=%d)"
                       " reg_perturb=%.3e"
                       " S_diag=[%.2e,%.2e] reg=%.1e reg/S=%.3e\n",
                       max_ricc_x, max_ricc_u, worst_u_k, worst_u_i,
                       max_reg_perturb,
                       min_S_diag, max_S_diag, reg_used_, reg_S_ratio);
                // ── Term decomposition at worst u node ──────────
                if (worst_u_k >= 0) {
                    const auto& rs = riccati_stages_[worst_u_k];
                    const auto& dx_k = riccati_ws_.dx[worst_u_k];
                    const auto& du_k = riccati_ws_.du[worst_u_k];
                    const auto& dx_k1 = riccati_ws_.dx[worst_u_k+1];
                    Vec<NX> nu_next;
                    for (int i = 0; i < NX; ++i) {
                        nu_next[i] = riccati_ws_.p[worst_u_k+1][i];
                        for (int j = 0; j < NX; ++j)
                            nu_next[i] += riccati_ws_.P[worst_u_k+1](i,j) * dx_k1[j];
                    }
                    int wi = worst_u_i;
                    double t_qu = rs.qu[wi];
                    double t_Quu_du = 0.0;
                    for (int j = 0; j < NU; ++j) t_Quu_du += rs.Quu(wi,j) * du_k[j];
                    double t_Qux_dx = 0.0;
                    for (int j = 0; j < NX; ++j) t_Qux_dx += rs.Qux(wi,j) * dx_k[j];
                    double t_Bt_nu = 0.0;
                    for (int j = 0; j < NX; ++j) t_Bt_nu += rs.B(j,wi) * nu_next[j];
                    printf("  [res-decomp] k=%d i=%d:"
                           " qu=%.4e Quu·du=%.4e Qux·dx=%.4e Bᵀν=%.4e"
                           " |du|=%.4e |dx|=%.4e\n",
                           worst_u_k, wi,
                           t_qu, t_Quu_du, t_Qux_dx, t_Bt_nu,
                           std::fabs(du_k[wi]), dx_k.norm_inf());
                }
            }

            // ── Post-Riccati stationarity: use ORIGINAL Riccati costates ────
            // MUST be called BEFORE recover_dual_step(), which converts p[k]
            // from scaled to physical space.
            compute_post_riccati_stationarity();

            // Save corrector step BEFORE recovery (scaled space for precond)
            for (int kk = 0; kk <= HORIZON; ++kk) {
                debug_scaled_dx_[kk] = riccati_ws_.dx[kk];
                debug_scaled_p_[kk]  = riccati_ws_.p[kk];
                if (kk < HORIZON)
                    debug_scaled_du_[kk] = riccati_ws_.du[kk];
            }

            // Recover physical step from scaled Riccati solution
            if (params_.enable_preconditioner) {
                prec_.recover_primal_step(riccati_ws_);
                prec_.recover_dual_step(riccati_ws_);
            }

            // Save corrector step AFTER recovery (physical space for both)
            for (int kk = 0; kk <= HORIZON; ++kk) {
                debug_phys_dx_[kk] = riccati_ws_.dx[kk];
                debug_phys_p_[kk]  = riccati_ws_.p[kk];
                if (kk < HORIZON)
                    debug_phys_du_[kk] = riccati_ws_.du[kk];
            }
            has_costates_ = true;  // costates now valid for stationarity check

            // ── Step-norm diagnostic: physical Newton step magnitudes ──
            if (params_.verbosity >= 2) {
                double dx_inf = 0.0, du_inf = 0.0, p_inf = 0.0;
                for (int kk = 0; kk <= HORIZON; ++kk) {
                    for (int i = 0; i < NX; ++i)
                        dx_inf = std::max(dx_inf, std::fabs(riccati_ws_.dx[kk][i]));
                    if (kk < HORIZON)
                        for (int i = 0; i < NU; ++i)
                            du_inf = std::max(du_inf, std::fabs(riccati_ws_.du[kk][i]));
                    for (int i = 0; i < NX; ++i)
                        p_inf = std::max(p_inf, std::fabs(riccati_ws_.p[kk][i]));
                }
                printf("  [step] ||dx||=%.3e ||du||=%.3e ||p||=%.3e reg=%.1e\n",
                       dx_inf, du_inf, p_inf, reg_used_);
            }

            // ── Early convergence check with post-Riccati stationarity ──
            // The primal/complementarity residuals from compute_kkt_residuals
            // are still valid (same point).  Only stat_inf_ was updated.
            if (barrier_strategy_.at_minimum(mu_) && kkt_converged()) {
                final_status_at_break = 1;
                break;
            }

            // ── Post-Riccati stagnation detection ──────────────
            const int stag_limit = STAGNATION_LIMIT;
            if (stat_inf_ < stat_best * 0.999) {
                stat_best = stat_inf_;
                stagnation_count = 0;
            } else {
                ++stagnation_count;
                if (stagnation_count >= stag_limit && barrier_strategy_.at_minimum(mu_)
                    && !kkt_converged()) {
                    final_status_at_break = 0;
                    if (params_.verbosity >= 1)
                        printf("  [stagnation: stat=%.3e unchanged for %d iters - terminating\n",
                               stat_inf_, STAGNATION_LIMIT);
                    break;
                }
            }

            // ── Conditional Mehrotra cross-term gate ────────────────
            // Compute the predicted average complementarity at the full
            // corrector step (alpha=1).  If the cross-term pushed it above
            // 2*mu, the second-order correction is hurting -- recompute the
            // corrector globally with the cross-term dropped (pure linear).
            // The result (cross_term_accepted_) is fed to the barrier FSM to
            // bias mu-reduction aggressiveness.
            cross_term_accepted_ = true;  // default: healthy (no constraints / sigma=0)
            if (sigma_ > 0.0 && prob_->constraints) {
                double mu_with = 0.0;
                int cnt = 0;
                for (int k = 0; k <= HORIZON; ++k) {
                    for (int j = 0; j < NC; ++j) {
                        double sp = prob_->stages[k].s[j]      + ds_[k][j];
                        double lp = prob_->stages[k].lambda[j] + dlambda_[k][j];
                        mu_with += std::max(sp, 0.0) * std::max(lp, 0.0);
                        ++cnt;
                    }
                }
                if (cnt > 0) mu_with /= cnt;
                cross_term_accepted_ = (mu_with <= 2.0 * mu_);
                if (!cross_term_accepted_) {
                    if (params_.verbosity >= 2)
                        printf("  [mehrotra: cross-term rejected, mu_with=%.3e > 2*mu=%.3e\n",
                               mu_with, 2.0 * mu_);
                    // Recompute dlambda ONLY (ds unchanged by cross-term).
                    // Cannot re-call recover_inequality_steps because dx/du
                    // are already physical (recovered). Just strip the cross term.
                    for (int kk = 0; kk <= HORIZON; ++kk) {
                        const Stage& stg = prob_->stages[kk];
                        for (int jj = 0; jj < NC; ++jj) {
                            if (stg.s[jj] > 1e-14) {
                                double sl = stg.s[jj] * stg.lambda[jj];
                                dlambda_[kk][jj] = (sigma_ * mu_ - sl
                                                    - stg.lambda[jj] * ds_[kk][jj]) / stg.s[jj];
                            }
                        }
                    }
                }
            }

            // ── Check linearized constraint residual BEFORE step ────────
            if (params_.verbosity >= 2 && iter <= 2) {
                double max_lin_res = 0.0;
                int worst_k = -1, worst_j = -1;
                for (int kk = 0; kk <= HORIZON; ++kk) {
                    const Stage& stg = prob_->stages[kk];
                    for (int jj = 0; jj < NC; ++jj) {
                        double C_dz = 0.0;
                        for (int i = 0; i < NX; ++i)
                            C_dz += stg.Cx(jj, i) * riccati_ws_.dx[kk][i];
                        if (kk < HORIZON)
                            for (int i = 0; i < NU; ++i)
                                C_dz += stg.Cu(jj, i) * riccati_ws_.du[kk][i];
                        double lin_res = std::fabs(stg.d[jj] + stg.s[jj] + C_dz + ds_[kk][jj]);
                        if (lin_res > max_lin_res) {
                            max_lin_res = lin_res;
                            worst_k = kk; worst_j = jj;
                        }
                    }
                }
                printf("  [lin-con-pre] BEFORE step: worst k=%d j=%d |g+s+Cdz+ds|=%.4e\n",
                       worst_k, worst_j, max_lin_res);
            }

            // ── ds-ftb diagnostic ────────────────────────────────
            if (params_.verbosity >= 2 && iter <= 3) {
                // Find FTB bottleneck: constraint with largest |ds|/s ratio where ds < 0
                double worst_ratio = 0.0;
                int worst_k = -1, worst_j = -1;
                for (int kk = 0; kk <= HORIZON; ++kk)
                    for (int jj = 0; jj < NC; ++jj)
                        if (ds_[kk][jj] < -1e-16) {
                            double r = -ds_[kk][jj] / (prob_->stages[kk].s[jj] + 1e-14);
                            if (r > worst_ratio) {
                                worst_ratio = r;
                                worst_k = kk; worst_j = jj;
                            }
                        }

                // Also compute max |du| and barrier force magnitude
                double max_du = 0.0;
                double max_barrier = 0.0;
                double max_p = 0.0;  // costate
                double max_qx = 0.0; // state cost gradient
                for (int kk = 0; kk <= HORIZON; ++kk) {
                    for (int i = 0; i < NX; ++i) {
                        if (std::fabs(riccati_ws_.p[kk][i]) > max_p)
                            max_p = std::fabs(riccati_ws_.p[kk][i]);
                        if (std::fabs(prob_->stages[kk].qx[i]) > max_qx)
                            max_qx = std::fabs(prob_->stages[kk].qx[i]);
                    }
                    if (kk < HORIZON) {
                        for (int i = 0; i < NU; ++i)
                            if (std::fabs(riccati_ws_.du[kk][i]) > max_du)
                                max_du = std::fabs(riccati_ws_.du[kk][i]);
                        for (int jj = 0; jj < NC; ++jj) {
                            double barr = mu_ / prob_->stages[kk].s[jj];
                            if (barr > max_barrier) max_barrier = barr;
                        }
                    }
                }
                // Max cost gradient
                double max_qu = 0.0;
                for (int kk = 0; kk < HORIZON; ++kk)
                    for (int i = 0; i < NU; ++i)
                        if (std::fabs(prob_->stages[kk].qu[i]) > max_qu)
                            max_qu = std::fabs(prob_->stages[kk].qu[i]);

                if (worst_k >= 0) {
                    const Stage& stg = prob_->stages[worst_k];
                    double g_val = stg.d[worst_j];
                    double s_val = stg.s[worst_j];
                    double g_plus_s = g_val + s_val;
                    double C_dz = 0.0;
                    for (int i = 0; i < NX; ++i)
                        C_dz += stg.Cx(worst_j, i) * riccati_ws_.dx[worst_k][i];
                    if (worst_k < HORIZON)
                        for (int i = 0; i < NU; ++i)
                            C_dz += stg.Cu(worst_j, i) * riccati_ws_.du[worst_k][i];

                    printf("  [ds-ftb] s=%.2e ds=%+.2e s_full=%+.2e | |qu|=%.2e |qx|=%.2e |p|=%.2e |du|=%.2e barr=%.2e\n",
                           s_val, ds_[worst_k][worst_j], s_val + ds_[worst_k][worst_j],
                           max_qu, max_qx, max_p, max_du, max_barrier);
                }
            }

            // ── Linear KKT residual check (always computed) ────
            linear_kkt_res_ = compute_linear_kkt_residual();

            // ── KKT residual log (unified with compute_linear_kkt_residual) ───
            if (params_.verbosity >= 2) {
                printf("  [kkt_res] dyn=%.2e feas=%.2e ricc_u=%.2e ricc_x=%.2e"
                       " stat_x=%.2e stat_u=%.2e"
                       " term=%.2e comp=%.2e reg=%.2e"
                       " | rel=%.2e(%s)\n",
                       linear_kkt_res_.max_dyn_res, linear_kkt_res_.max_feas_res,
                       linear_kkt_res_.max_riccati_u_res, linear_kkt_res_.max_riccati_x_res,
                       linear_kkt_res_.max_stat_x_res, linear_kkt_res_.max_stat_u_res,
                       linear_kkt_res_.max_stat_term_res, linear_kkt_res_.max_comp_res,
                       reg_used_,
                       linear_kkt_res_.max_rel_res, linear_kkt_res_.quality_label());

                // ── Costate vs residual correlation diagnostic ────────────
                {
                    double max_p = 0.0;
                    double max_P = 0.0;
                    double max_dx = 0.0;
                    for (int kk = 0; kk <= HORIZON; ++kk) {
                        for (int i = 0; i < NX; ++i) {
                            double pv = std::fabs(riccati_ws_.p[kk][i]);
                            if (pv > max_p) max_p = pv;
                            double dxv = std::fabs(riccati_ws_.dx[kk][i]);
                            if (dxv > max_dx) max_dx = dxv;
                        }
                        for (int i = 0; i < NX; ++i)
                            for (int j = 0; j < NX; ++j) {
                                double Pv = std::fabs(riccati_ws_.P[kk](i,j));
                                if (Pv > max_P) max_P = Pv;
                            }
                    }
                    double nu_max = max_p + max_P * max_dx;
                    double ratio = (nu_max > 1e-14) ? linear_kkt_res_.max_riccati_x_res / nu_max : 0.0;
                    printf("  [costate-corr] |p|=%.2e |P|=%.2e |dx|=%.2e |ν|_est=%.2e | ricc_x=%.2e ratio=%.1f%%\n",
                           max_p, max_P, max_dx, nu_max, linear_kkt_res_.max_riccati_x_res, 100.0 * ratio);
                }

                // ── Independent p_k verification ──────────────────────────
                // Recompute p_k from scratch and compare with stored value
                if (iter <= 2) {
                    double max_pk_err = 0.0;
                    int worst_pk_k = -1;
                    for (int kk = 0; kk < HORIZON; ++kk) {
                        const Stage& stg = prob_->stages[kk];
                        const auto& P_next = riccati_ws_.P[kk + 1];
                        const auto& p_next = riccati_ws_.p[kk + 1];

                        // Recompute p_k = qx + A^T(p_{k+1} + P_{k+1}·c) - Qup^T·d
                        for (int i = 0; i < NX; ++i) {
                            double pk_check = stg.qx[i];
                            // A^T(p_{k+1} + P_{k+1}·c)
                            for (int j = 0; j < NX; ++j) {
                                double temp_j = p_next[j];
                                for (int m = 0; m < NX; ++m)
                                    temp_j += P_next(j, m) * stg.c[m];
                                pk_check += stg.A(j, i) * temp_j;
                            }
                            // - Qup^T·d
                            for (int m = 0; m < NU; ++m) {
                                double qup_mi = stg.Qux(m, i);
                                for (int a = 0; a < NX; ++a) {
                                    double btp = 0.0;
                                    for (int b = 0; b < NX; ++b)
                                        btp += stg.B(b, m) * P_next(b, a);
                                    qup_mi += btp * stg.A(a, i);
                                }
                                pk_check -= qup_mi * riccati_ws_.d[kk][m];
                            }
                            double err = std::fabs(pk_check - riccati_ws_.p[kk][i]);
                            if (err > max_pk_err) {
                                max_pk_err = err;
                                worst_pk_k = kk;
                            }
                        }
                    }
                    double max_pk_stored = 0.0;
                    for (int i = 0; i < NX; ++i)
                        if (std::fabs(riccati_ws_.p[worst_pk_k][i]) > max_pk_stored)
                            max_pk_stored = std::fabs(riccati_ws_.p[worst_pk_k][i]);
                    printf("  [pk-verify] max|p_k_err|=%.4e at k=%d |p_stored|=%.2e rel_err=%.2e%%\n",
                           max_pk_err, worst_pk_k, max_pk_stored,
                           100.0 * max_pk_err / (max_pk_stored + 1e-30));

                    // ── x-stationarity term breakdown at worst stage (k=0) ────
                    {
                        const int kk = 0;
                        const Stage& stg = prob_->stages[kk];
                        const auto& dx_k  = riccati_ws_.dx[kk];
                        const auto& du_k  = riccati_ws_.du[kk];
                        const auto& dx_k1 = riccati_ws_.dx[kk + 1];
                        const auto& P_next = riccati_ws_.P[kk + 1];
                        const auto& p_next = riccati_ws_.p[kk + 1];
                        const auto& P_k = riccati_ws_.P[kk];
                        const auto& p_k = riccati_ws_.p[kk];

                        // Compute ν_{k+1} = p_{k+1} + P_{k+1}·dx_{k+1}
                        Vec<NX> nu_next;
                        for (int i = 0; i < NX; ++i) {
                            nu_next[i] = p_next[i];
                            for (int j = 0; j < NX; ++j)
                                nu_next[i] += P_next(i, j) * dx_k1[j];
                        }
                        // Compute ν_k = p_k + P_k·dx_k
                        Vec<NX> nu_k;
                        for (int i = 0; i < NX; ++i) {
                            nu_k[i] = p_k[i];
                            for (int j = 0; j < NX; ++j)
                                nu_k[i] += P_k(i, j) * dx_k[j];
                        }
                        // A^T·ν_{k+1}
                        Vec<NX> At_nu;
                        for (int i = 0; i < NX; ++i) {
                            At_nu[i] = 0.0;
                            for (int j = 0; j < NX; ++j)
                                At_nu[i] += stg.A(j, i) * nu_next[j];
                        }
                        // Qxx·dx
                        Vec<NX> Qxx_dx;
                        for (int i = 0; i < NX; ++i) {
                            Qxx_dx[i] = 0.0;
                            for (int j = 0; j < NX; ++j)
                                Qxx_dx[i] += stg.Qxx(i, j) * dx_k[j];
                        }
                        // Qux^T·du
                        Vec<NX> Quxt_du;
                        for (int i = 0; i < NX; ++i) {
                            Quxt_du[i] = 0.0;
                            for (int j = 0; j < NU; ++j)
                                Quxt_du[i] += stg.Qux(j, i) * du_k[j];
                        }

                        // Find worst component
                        int worst_i = 0;
                        double max_res = 0.0;
                        for (int i = 0; i < NX; ++i) {
                            double res = std::fabs(stg.qx[i] + Qxx_dx[i] + Quxt_du[i] - At_nu[i] + nu_k[i]);
                            if (res > max_res) { max_res = res; worst_i = i; }
                        }
                        int wi = worst_i;
                        printf("  [x-stat-break] k=0 i=%d: qx=%.3e Qxx·dx=%.3e Qux^T·du=%.3e -A^T·ν=%.3e +ν_k=%.3e | res=%.3e\n",
                               wi, stg.qx[wi], Qxx_dx[wi], Quxt_du[wi], -At_nu[wi], nu_k[wi], max_res);
                        printf("  [x-stat-break]   A^T·ν_{k+1}=%.3e ν_k=%.3e qx+A^T·ν-ν_k=%.3e (should=0 if Riccati costate = KKT costate)\n",
                               At_nu[wi], nu_k[wi], stg.qx[wi] + At_nu[wi] - nu_k[wi]);
                    }
                }
            }

            // 6. Step acceptance: separate primal/dual fraction-to-boundary ──
            compute_ftb_limits(alpha_p, alpha_d);
            last_alpha_p_ = alpha_p;  // store for barrier FTB-bottleneck detection
            log_iteration(iter, sigma_, alpha_p, alpha_d);
            alpha_lambda_ = alpha_d;  // FTB step for λ (independent of line search)

            // ── FTB bottleneck diagnostic (periodic, verbose only) ──
            if (params_.verbosity >= 2 && iter % 25 == 0 && iter > 0) {
                // Find primal FTB bottleneck
                double worst_r = 0.0;
                int bkw = -1, bjw = -1;
                for (int kk = 0; kk <= HORIZON; ++kk)
                    for (int jj = 0; jj < NC; ++jj)
                        if (ds_[kk][jj] < -1e-16) {
                            double r = -ds_[kk][jj] / (prob_->stages[kk].s[jj] + 1e-14);
                            if (r > worst_r) { worst_r = r; bkw = kk; bjw = jj; }
                        }
                // Find dual FTB bottleneck
                double worst_rd = 0.0;
                int bkw_d = -1, bjw_d = -1;
                for (int kk = 0; kk <= HORIZON; ++kk)
                    for (int jj = 0; jj < NC; ++jj)
                        if (dlambda_[kk][jj] < -1e-16) {
                            double r = -dlambda_[kk][jj] / (prob_->stages[kk].lambda[jj] + 1e-14);
                            if (r > worst_rd) { worst_rd = r; bkw_d = kk; bjw_d = jj; }
                        }
                double bp_s = (bkw >= 0) ? prob_->stages[bkw].s[bjw] : -1;
                double bp_ds = (bkw >= 0) ? ds_[bkw][bjw] : 0;
                double bp_g = (bkw >= 0) ? prob_->stages[bkw].d[bjw] : 0;
                double bd_l = (bkw_d >= 0) ? prob_->stages[bkw_d].lambda[bjw_d] : -1;
                double bd_dl = (bkw_d >= 0) ? dlambda_[bkw_d][bjw_d] : 0;
                printf("  [ftb-diag] alpha_p=%.3f bottleneck: k=%d j=%d s=%.2e ds=%+.2e g=%.2e | alpha_d=%.3f bottleneck: k=%d j=%d lam=%.2e dlam=%+.2e\n",
                       alpha_p, bkw, bjw, bp_s, bp_ds, bp_g,
                       alpha_d, bkw_d, bjw_d, bd_l, bd_dl);
            }

            if (alpha_p < 1e-14 || alpha_lambda_ < 1e-14) {
                sz_complement();
                if (params_.verbosity >= 1)
                    printf("  [ftb-zero: ap=%.2e, alam=%.2e, sz_complement]\n",
                           alpha_p, alpha_lambda_);
                continue;
            }

            ls_iters = 1;
            accepted = false;
            double alpha = alpha_p;

            // ── Globalization: filter line search ────────────────
            evaluator_.bind(this);

            if (params_.verbosity >= 2) {
                // θ decomposition: dynamics vs inequality contributions
                double theta_dyn_0 = 0.0, theta_ineq_0 = 0.0;
                {
                    Stage* s = prob_->stages;
                    for (int k = 0; k < HORIZON; ++k)
                        for (int i = 0; i < NX; ++i) {
                            double c_phys = stages_scaled_ ? s[k].c[i] * prec_.inv_Lx(k + 1)[i] : s[k].c[i];
                            theta_dyn_0 += std::fabs(c_phys);
                        }
                    if (prob_->constraints) {
                        for (int k = 0; k <= HORIZON; ++k)
                            for (int j = 0; j < NC; ++j)
                                theta_ineq_0 += std::fabs(s[k].d[j] + s[k].s[j]);
                    } else {
                        for (int k = 0; k <= HORIZON; ++k)
                            for (int j = 0; j < NC; ++j)
                                theta_ineq_0 += std::fabs(s[k].s[j]);
                    }
                }
                printf("  [filter: theta_0=%.4e phi0=%.4e Dphi=%.3e"
                       " | th_dyn=%.4e th_ineq=%.4e | ap=%.4f ad=%.4f]\n",
                       evaluator_.current_theta(), evaluator_.current_phi(),
                       evaluator_.compute_Dphi(),
                       theta_dyn_0, theta_ineq_0, alpha_p, alpha_d);
            }

            LSResult ls_result = filter_ls_.search(evaluator_, alpha_p);

            if (ls_result.status == LSStatus::ACCEPTED) {
                accepted = true;
                alpha = ls_result.alpha;
                ls_iters = ls_result.ls_iters;
                if (ls_result.soc_used) out_stats.soc_steps++;

                // Evaluate trial θ,φ at accepted alpha for diagnostics
                if (params_.verbosity >= 2) {
                    double theta_trial, phi_trial;
                    evaluator_.evaluate(alpha, theta_trial, phi_trial);
                    printf("  [ls-accept] a=%.4e theta_trial=%.4e phi_trial=%.4e"
                           " soc=%d ls_iters=%d\n",
                           alpha, theta_trial, phi_trial,
                           ls_result.soc_used ? 1 : 0, ls_result.ls_iters);
                }

                // Apply the accepted step — the ONLY place variables are updated
                apply_primal_dual_step(alpha, alpha_lambda_);

                // Re-evaluate model at new point for accurate theta reporting
                st = evaluate_model();
                if (st != Status::SUCCESS) return st;
                model_evaluated = true;
                stages_scaled_ = false;

                if (params_.verbosity >= 1) {
                    printf("  [step: a=%.4f alam=%.4f ls=%d soc=%s cost=%.4e theta=%.4e]\n",
                           alpha, alpha_lambda_, ls_iters,
                           ls_result.soc_used ? "yes" : "no",
                           compute_objective(), compute_theta());
                }
            } else {
                alpha = ls_result.alpha;
                if (params_.verbosity >= 1)
                    printf("  [LS fail: a=%.3e alam=%.2e ls=%d] step too tiny\n",
                           alpha, alpha_lambda_, ls_result.ls_iters);
                ls_failed = true;
                break;
            }

            // Complementarity safeguard
            if (barrier_strategy_.m_safe() > 0.0) {
                if (params_.verbosity >= 3)
                    printf("  [safeguard: m_safe=%.3e]\n", barrier_strategy_.m_safe());
                sz_complement(barrier_strategy_.m_safe());
            }

            // Barrier update: reduce μ if subproblem solved, else hold
            {
                double E_mu = std::max(primal_inf_, compl_inf_);
                // FTB-bottleneck detection: primal step severely limited
                // while constraint violation is still significant.
                // This signals that the barrier force is too strong and
                // μ should be reduced more aggressively.
                double ftb_cons_thresh = barrier_strategy_.kappa_eps() * mu_;
                bool ftb_bottleneck = (last_alpha_p_ < 0.1)
                                   && (primal_inf_ > ftb_cons_thresh)
                                   && (aff_alpha_p_ < 0.1);
                bool mu_changed = barrier_strategy_.update(
                    mu_, primal_inf_, compl_inf_, sigma_, stat_inf_,
                    cross_term_accepted_, ftb_bottleneck);
                if (params_.verbosity >= 1) {
                    printf("  [barrier: E_mu=%.2e k*mu=%.2e %s mu=%.2e]\n",
                           E_mu, barrier_strategy_.kappa_eps() * mu_,
                           mu_changed ? "REDUCE" : "HOLD ", mu_);
                }
                if (mu_changed) {
                    filter_ls_.reset_filter();
                }
            }
        }

        // ── Solve summary ─────────────────────────────────────────
        // Re-evaluate for accurate final primal/complementarity stats.
        // Save the riccati-based stationarity (the convergence metric)
        // before re-evaluation overwrites it.
        double saved_stat = stat_inf_;
        evaluate_model();
        // Restore scaled stages so compute_linear_kkt_residual is consistent
        // with the scaled Riccati workspace (dx, du, P, p).
        if (params_.enable_preconditioner) {
            prec_.transform_qp(prob_->stages);
        }
        compute_kkt_residuals();
        stat_inf_ = saved_stat;  // restore Riccati-consistent stationarity

        out_stats.inner_iterations = iter;
        out_stats.barrier_param    = mu_;
        out_stats.primal_infeas    = primal_inf_;
        out_stats.dual_infeas      = stat_inf_;
        out_stats.complementarity  = compl_inf_;
        out_stats.condition_estimate = ineq_viol_;
        out_stats.cost             = compute_objective();

        // Use the convergence result from the loop-break point, NOT
        // from after re-evaluation.  Finite-difference Jacobians
        // can shift residuals slightly, turning a converged state
        // into a "Stagnation" false alarm.
        Status final_status;
        if (final_status_at_break >= 0)
            final_status = (final_status_at_break == 1) ? Status::SUCCESS : Status::STAGNATION;
        else if (ls_failed)
            final_status = Status::LINE_SEARCH_FAILURE;
        else
            final_status = Status::MAX_ITERATIONS;

        if (params_.verbosity >= 1) {
            printf("\n=== SOLVE COMPLETE ===\n");
            printf("Status:          %s\n", status_string(final_status));
            printf("Iterations:      %d\n", iter);
            printf("Final mu:        %.3e\n", mu_);
            printf("Primal inf:      %.3e  (tol=%.1e)  %s\n", primal_inf_, params_.tol_primal,
                   primal_inf_ <= params_.tol_primal ? "OK" : "FAIL");
            printf("Stationarity:    %.3e  (tol=%.1e)  %s\n", stat_inf_, params_.tol_stat,
                   stat_inf_ <= params_.tol_stat ? "OK" : "FAIL");
            printf("Complementarity: %.3e  (tol=%.1e)  %s\n", compl_inf_, params_.tol_compl,
                   compl_inf_ <= params_.tol_compl ? "OK" : "FAIL");
            printf("Ineq viol:       %.3e  (tol=%.1e)  %s\n", ineq_viol_, params_.tol_ineq,
                   ineq_viol_ >= -params_.tol_ineq ? "OK" : "FAIL");
            printf("SOC steps:       %d\n", out_stats.soc_steps);
            printf("Regularization:  %.1e\n", reg_used_);
            printf("Cost:            %.4f\n", compute_objective());

            // Convergence diagnosis
            bool p_ok = primal_inf_ <= params_.tol_primal;
            bool s_ok = stat_inf_   <= params_.tol_stat;
            bool c_ok = compl_inf_  <= params_.tol_compl;
            bool i_ok = ineq_viol_  >= -params_.tol_ineq;
            if (!(p_ok && s_ok && c_ok && i_ok)) {
                printf("\n── Convergence Diagnosis --\n");
                if (!s_ok) {
                    printf("  * STATIONARITY not satisfied: ||nabla L||_inf=%.3e >> tol=%.1e  (worst node k=%d)\n",
                                  stat_inf_, params_.tol_stat, stat_worst_node_);
                    printf("    component breakdown at node %d:\n", stat_worst_node_);
                    printf("      |grad_x|=%.2e  |grad_u|=%.2e  (cost gradient)\n",
                           stat_breakdown_[0], stat_breakdown_[1]);
                    printf("      |Cx^T·λ|=%.2e  |Cu^T·λ|=%.2e  (constraint dual)\n",
                           stat_breakdown_[2], stat_breakdown_[3]);
                    printf("      |costate_x|=%.2e  |costate_u|=%.2e  (dynamics costate)\n",
                           stat_breakdown_[4], stat_breakdown_[5]);
                }
                if (!p_ok) printf("  * PRIMAL FEASIBILITY not satisfied: %.3e >> tol=%.1e\n",
                                  primal_inf_, params_.tol_primal);
                if (!c_ok) printf("  * COMPLEMENTARITY not satisfied: %.3e >> tol=%.1e\n",
                                  compl_inf_, params_.tol_compl);
                if (!i_ok) printf("  * INEQUALITY violated: %.3e\n", ineq_viol_);
            }

            // ── Linear KKT solution quality ─────────────────────
            printf("\n── Linear KKT Solve Quality ──\n");
            printf("Quality:         %s\n", linear_kkt_res_.quality_label());
            printf("Max abs res:     %.2e  (worst stage=%d, eq=%d)\n",
                   linear_kkt_res_.max_abs_res,
                   linear_kkt_res_.worst_stage, linear_kkt_res_.worst_eq_type);
            printf("Max rel res:     %.2e\n", linear_kkt_res_.max_rel_res);
            printf("  dynamics:   abs=%.2e  rel=%.2e  (RHS_norm=%.2e)\n",
                   linear_kkt_res_.max_dyn_res, linear_kkt_res_.rel_dyn_res,
                   linear_kkt_res_.rhs_dyn_norm);
            printf("  feasibility:abs=%.2e  rel=%.2e  (RHS_norm=%.2e)\n",
                   linear_kkt_res_.max_feas_res, linear_kkt_res_.rel_feas_res,
                   linear_kkt_res_.rhs_feas_norm);
            printf("  stat_x:     abs=%.2e  rel=%.2e  (RHS_norm=%.2e)\n",
                   linear_kkt_res_.max_stat_x_res, linear_kkt_res_.rel_stat_x_res,
                   linear_kkt_res_.rhs_stat_x_norm);
            printf("  stat_u:     abs=%.2e  rel=%.2e  (RHS_norm=%.2e)\n",
                   linear_kkt_res_.max_stat_u_res, linear_kkt_res_.rel_stat_u_res,
                   linear_kkt_res_.rhs_stat_u_norm);
            printf("  term_stat:  abs=%.2e  rel=%.2e  (RHS_norm=%.2e)\n",
                   linear_kkt_res_.max_stat_term_res, linear_kkt_res_.rel_stat_term_res,
                   linear_kkt_res_.rhs_stat_term_norm);
            printf("  complement: abs=%.2e  rel=%.2e  (RHS_norm=%.2e)\n",
                   linear_kkt_res_.max_comp_res, linear_kkt_res_.rel_comp_res,
                   linear_kkt_res_.rhs_comp_norm);
            printf("  riccati_x:  abs=%.2e  (pure reduced KKT)\n",
                   linear_kkt_res_.max_riccati_x_res);
            printf("  riccati_u:  abs=%.2e  (pure reduced KKT)\n",
                   linear_kkt_res_.max_riccati_u_res);

            if (NU > 0)
                printf("First u* =       [%.3f]\n", prob_->stages[0].u[0]);
        }

        return final_status;
    }

private:
    // ═════════════════════════════════════════════════════════════════════
    //  Initialize barrier variables from problem data
    // ═════════════════════════════════════════════════════════════════════

    void initialize_from_problem() {
        const int N = HORIZON;
        Stage* s = prob_->stages;

        for (int k = 0; k <= N; ++k) {
            if (prob_->constraints) {
                if (k < N) {
                    prob_->constraints->evaluate(s[k].x, s[k].u, k, s[k].d);
                } else {
                    prob_->constraints->evaluate_terminal(s[k].x, s[k].d);
                }
                for (int j = 0; j < NC; ++j) {
                    s[k].s[j] = std::max(-s[k].d[j], mu_);  // linear floor, matches sz_complement safeguard
                    s[k].lambda[j] = mu_ / s[k].s[j];
                }
            } else {
                for (int j = 0; j < NC; ++j) {
                    s[k].s[j]      = 1.0;
                    s[k].lambda[j] = 1.0;
                }
            }
        }
    }

    // ═════════════════════════════════════════════════════════════════════
    //  Evaluate nonlinear model at current iterate
    // ═════════════════════════════════════════════════════════════════════

    Status evaluate_model() {
        const int N = HORIZON;
        Stage* s = prob_->stages;

        for (int k = 0; k < N; ++k) {
            // Dynamics defect: c_k = f(x_k, u_k) - x_{k+1}
            Vec<NX> fk;
            Status st = prob_->dynamics->discrete_step(s[k].x, s[k].u,
                                                        prob_->dt, fk, k);
            if (st != Status::SUCCESS) return st;
            for (int i = 0; i < NX; ++i)
                s[k].c[i] = fk[i] - s[k + 1].x[i];

            // Dynamics Jacobians
            st = prob_->dynamics->linearize(s[k].x, s[k].u, prob_->dt,
                                            s[k].A, s[k].B, k);
            if (st != Status::SUCCESS) return st;

            // Cost
            s[k].cost = prob_->cost->stage_cost(s[k].x, s[k].u, k);
            st = prob_->cost->stage_gradient(s[k].x, s[k].u, k,
                                              s[k].qx, s[k].qu);
            if (st != Status::SUCCESS) return st;
            st = prob_->cost->stage_hessian(s[k].x, s[k].u, k,
                                             s[k].Qxx, s[k].Quu, s[k].Qux);
            if (st != Status::SUCCESS) return st;

            // Constraints
            if (prob_->constraints) {
                st = prob_->constraints->evaluate(s[k].x, s[k].u, k, s[k].d);
                if (st != Status::SUCCESS) return st;
                st = prob_->constraints->jacobian(s[k].x, s[k].u, k,
                                                   s[k].Cx, s[k].Cu);
                if (st != Status::SUCCESS) return st;
            }

            // NaN guard
            if (!is_finite(s[k].x) || !is_finite(s[k].u))
                return Status::NAN_DETECTED;
        }

        // Terminal stage
        s[N].cost = prob_->cost->terminal_cost(s[N].x);
        Status st = prob_->cost->terminal_gradient(s[N].x, s[N].qx);
        if (st != Status::SUCCESS) return st;
        st = prob_->cost->terminal_hessian(s[N].x, s[N].Qxx);
        if (st != Status::SUCCESS) return st;

        if (prob_->constraints) {
            st = prob_->constraints->evaluate_terminal(s[N].x, s[N].d);
            if (st != Status::SUCCESS) return st;
            st = prob_->constraints->jacobian_terminal(s[N].x, s[N].Cx);
            if (st != Status::SUCCESS) return st;
        }

        // Terminal stage has no control variables
        s[N].Quu.zero();
        s[N].Qux.zero();
        s[N].qu.zero();

        return Status::SUCCESS;
    }

    // ═════════════════════════════════════════════════════════════════════
    //  Evaluate trial point for nonlinear KKT refinement (Shamanskii chord)
    // ═════════════════════════════════════════════════════════════════════

    bool evaluate_trial_point() {
        const int N = HORIZON;
        Stage* s = prob_->stages;

        for (int k = 0; k < N; ++k) {
            // Trial point: x_trial = x_k + dx[k], u_trial = u_k + du[k]
            Vec<NX> x_trial;
            Vec<NU> u_trial;
            for (int i = 0; i < NX; ++i)
                x_trial[i] = s[k].x[i] + riccati_ws_.dx[k][i];
            for (int i = 0; i < NU; ++i)
                u_trial[i] = s[k].u[i] + riccati_ws_.du[k][i];

            // NaN guard on trial point
            if (!is_finite(x_trial) || !is_finite(u_trial))
                return false;

            // Dynamics at trial point
            Vec<NX> fk;
            Status st = prob_->dynamics->discrete_step(x_trial, u_trial, prob_->dt, fk, k);
            if (st != Status::SUCCESS) return false;
            if (!is_finite(fk)) return false;

            // Dynamics defect: c_trial[k] = f(x_trial, u_trial) - (x_{k+1} + dx[k+1])
            for (int i = 0; i < NX; ++i)
                trial_stages_[k].c[i] = fk[i] - (s[k + 1].x[i] + riccati_ws_.dx[k + 1][i]);

            // Dynamics Jacobians at trial point (frozen in riccati_stages_, not updated)
            // We only need the defect c for the chord method

            // Cost gradient at trial point
            Vec<NX> qx_trial;
            Vec<NU> qu_trial;
            st = prob_->cost->stage_gradient(x_trial, u_trial, k, qx_trial, qu_trial);
            if (st != Status::SUCCESS) return false;
            trial_stages_[k].qx = qx_trial;
            trial_stages_[k].qu = qu_trial;

            // Constraints at trial point
            if (prob_->constraints) {
                Vec<NC> d_trial;
                st = prob_->constraints->evaluate(x_trial, u_trial, k, d_trial);
                if (st != Status::SUCCESS) return false;
                trial_stages_[k].d = d_trial;

                // Jacobians at trial point
                st = prob_->constraints->jacobian(x_trial, u_trial, k,
                                                   trial_stages_[k].Cx, trial_stages_[k].Cu);
                if (st != Status::SUCCESS) return false;
            }
        }

        // Terminal stage
        Vec<NX> xN_trial;
        for (int i = 0; i < NX; ++i)
            xN_trial[i] = s[N].x[i] + riccati_ws_.dx[N][i];
        if (!is_finite(xN_trial)) return false;

        Status st = prob_->cost->terminal_gradient(xN_trial, trial_stages_[N].qx);
        if (st != Status::SUCCESS) return false;

        if (prob_->constraints) {
            Vec<NC> dN_trial;
            st = prob_->constraints->evaluate_terminal(xN_trial, dN_trial);
            if (st != Status::SUCCESS) return false;
            trial_stages_[N].d = dN_trial;

            st = prob_->constraints->jacobian_terminal(xN_trial, trial_stages_[N].Cx);
            if (st != Status::SUCCESS) return false;
        }

        // Terminal has no u
        trial_stages_[N].qu.zero();

        return true;
    }

    // ═════════════════════════════════════════════════════════════════════
    //  Compute nonlinear KKT residual norm (cheap, no dynamics eval)
    // ═════════════════════════════════════════════════════════════════════

    // Check dynamics defect norm only (the true target of Shamanskii refinement).
    // Stationarity uses a DIFFERENT effective gradient (barrier-modified) that we
    // can't easily evaluate without updating dual/slack variables, so it's excluded.
    double dynamics_defect_norm() {
        const int N = HORIZON;
        Stage* s = prob_->stages;
        double max_def = 0.0;
        for (int k = 0; k < N; ++k) {
            double def = s[k].c.norm_inf();
            max_def = std::max(max_def, def);
        }
        return max_def;
    }

    // Trial-point dynamics defect norm (after evaluate_trial_point)
    double trial_dynamics_defect() {
        const int N = HORIZON;
        double max_def = 0.0;
        for (int k = 0; k < N; ++k)
            max_def = std::max(max_def, trial_stages_[k].c.norm_inf());
        return max_def;
    }

    // [REMOVED] compute_nonlinear_kkt_rhs() and refine_newton_direction()
    // Disabled per user instruction: the solving process operates exclusively
    // on scaled QP data from transform_qp(); no manual scaling code permitted.


    // =====================================================================
    //  Post-Riccati stationarity: compute stationarity using the Riccati
    //  corrector costates, which are consistent with the CURRENT model
    //  evaluation (same linearization point).  This avoids the mismatch
    //  in compute_kkt_residuals() where costates come from the PREVIOUS
    //  iteration's Riccati solve.
    //
    //  The Riccati costate p[k] satisfies the linearized stationarity.
    //  At convergence (Δz → 0), the nonlinear stationarity converges to
    //  the linearized stationarity, so this metric should go to ~0.
    // ═════════════════════════════════════════════════════════════════════

    // =====================================================================
    //  Post-Riccati stationarity: compute stationarity using the ORIGINAL
    //  Riccati costates (scaled space) BEFORE they are overwritten by
    //  costate recovery.  The Riccati solver's costates satisfy the
    //  linear KKT stationarity (both x and u) by construction, so the
    //  residual should be ≈ 0 (limited only by Riccati solve accuracy).
    //
    //  MUST be called BEFORE recover_dual_step(), which converts p[k]
    //  from scaled to physical space.
    // ═════════════════════════════════════════════════════════════════════

    void compute_post_riccati_stationarity() {
        const int N = HORIZON;
        Stage* s = prob_->stages;

        double post_stat = 0.0;

        for (int k = 0; k <= N; ++k) {
            // Build the MODIFIED gradient (same as build_kkt_rhs).
            // q̃ = qx + Cx^T·(centering - λ)  where centering = (σμ + λ(g+s))/s
            // All terms here are in SCALED space (after transform_qp).
            Vec<NX> lag_x = s[k].qx;
            Vec<NU> lag_u;
            if (k < N) lag_u = s[k].qu; else lag_u.zero();

            if (prob_->constraints) {
                for (int j = 0; j < NC; ++j) {
                    double sj = s[k].s[j];
                    if (sj < 1e-14) continue;
                    double lj = s[k].lambda[j];
                    double gj = s[k].d[j] + sj;  // g + s (constraint infeasibility)
                    double centering = (sigma_ * mu_ + lj * gj) / sj;
                    double w = centering - lj;  // net contribution per constraint
                    for (int i = 0; i < NX; ++i)
                        lag_x[i] += s[k].Cx(j, i) * w;
                    if (k < N)
                        for (int i = 0; i < NU; ++i)
                            lag_u[i] += s[k].Cu(j, i) * w;
                }
            }

            // Add costate terms using ORIGINAL Riccati costates (scaled space).
            // p[k] is in scaled space (before recover_dual_step).
            Vec<NX> cos_x; cos_x.zero();
            Vec<NU> cos_u; cos_u.zero();
            if (k < N) {
                for (int i = 0; i < NX; ++i)
                    for (int m = 0; m < NX; ++m)
                        cos_x[i] += s[k].A(m, i) * riccati_ws_.p[k+1][m];
                for (int i = 0; i < NU; ++i)
                    for (int m = 0; m < NX; ++m)
                        cos_u[i] += s[k].B(m, i) * riccati_ws_.p[k+1][m];
            }
            for (int i = 0; i < NX; ++i)
                cos_x[i] -= riccati_ws_.p[k][i];

            for (int i = 0; i < NX; ++i) lag_x[i] += cos_x[i];
            for (int i = 0; i < NU; ++i) lag_u[i] += cos_u[i];

            // ── Convert to PHYSICAL stationarity ──────────────────────
            // Transform: qx_scaled = (1/Lx) * qx_phys, so Lx = 1/inv_Lx.
            // Physical = scaled * Lx = scaled / inv_Lx  (component-wise).
            // Proof: physical stationarity = ∇_x L_phys
            //   = Lx · (∇_x̂ L_scaled) = Lx · lag_scaled = lag_scaled / inv_Lx.
            if (params_.enable_preconditioner) {
                for (int i = 0; i < NX; ++i)
                    lag_x[i] /= prec_.inv_Lx(k)[i];
                if (k < N)
                    for (int i = 0; i < NU; ++i)
                        lag_u[i] /= prec_.inv_Lu(k)[i];
            }

            double stat_x_full = lag_x.norm_inf();
            double stat_u_full = lag_u.norm_inf();
            double stat_abs = (stat_x_full > stat_u_full) ? stat_x_full : stat_u_full;

            // Scale-invariant denominator: use physical component magnitudes.
            // Recompute from original scaled vectors / inv_Lx = physical.
            double scale_x, scale_u;
            if (params_.enable_preconditioner) {
                double gx = 0.0, gu = 0.0;
                for (int i = 0; i < NX; ++i) {
                    gx = std::max(gx, std::fabs(s[k].qx[i] / prec_.inv_Lx(k)[i]));
                    gx = std::max(gx, std::fabs(cos_x[i] / prec_.inv_Lx(k)[i]));
                }
                scale_x = std::max({gx, 1.0});
                if (k < N) {
                    for (int i = 0; i < NU; ++i) {
                        gu = std::max(gu, std::fabs(s[k].qu[i] / prec_.inv_Lu(k)[i]));
                        gu = std::max(gu, std::fabs(cos_u[i] / prec_.inv_Lu(k)[i]));
                    }
                    scale_u = std::max({gu, 1.0});
                } else {
                    scale_u = 1.0;
                }
            } else {
                double grad_x_inf = s[k].qx.norm_inf();
                double grad_u_inf = (k < N) ? s[k].qu.norm_inf() : 0.0;
                double cos_x_inf = cos_x.norm_inf();
                double cos_u_inf = cos_u.norm_inf();
                scale_x = std::max({grad_x_inf, cos_x_inf, 1.0});
                scale_u = std::max({grad_u_inf, cos_u_inf, 1.0});
            }
            double scale = (scale_x > scale_u) ? scale_x : scale_u;
            double stat = stat_abs / scale;
            if (stat > post_stat) post_stat = stat;
        }

        // Override stat_inf_ with Riccati-consistent stationarity.
        stat_inf_ = post_stat;
    }

    // ═════════════════════════════════════════════════════════════════════
    //  Compute KKT residuals of the barrier subproblem
    // ═════════════════════════════════════════════════════════════════════

    void compute_kkt_residuals() {
        const int N = HORIZON;
        Stage* s = prob_->stages;

        primal_inf_ = 0.0;
        dyn_defect_ = 0.0;  // nonlinear dynamics defect only
        cons_viol_  = 0.0;  // constraint violation only
        stat_inf_   = 0.0;
        stat_worst_node_ = -1;
        compl_inf_  = 0.0;
        ineq_viol_  = 0.0;  // most-negative s or λ (0 if all ≥ 0)

        // Are Riccati costates available? (not on first iteration)
        bool costates_valid = (riccati_ws_.p[N].norm_inf() > 0.0 || has_costates_);

        // Save Riccati costates BEFORE recovery (they satisfy the scaled KKT
        // by construction; needed for compute_linear_kkt_residual).
        Vec<NX> riccati_p_save[N + 1];
        if (costates_valid) {
            for (int k = 0; k <= N; ++k)
                riccati_p_save[k] = riccati_ws_.p[k];
        }

        // ── Costate recovery: overwrite Riccati costates with KKT-consistent
        //    costates from the barrier-problem x-stationarity equation.
        //    SKIP when preconditioner is enabled: the recovery formula mixes
        //    physical gradients with scaled Hessians (from transform_qp),
        //    producing incorrect costates.  The saved Riccati costates
        //    (physical after recover_dual_step) are correct and are converted
        //    to scaled space inside compute_linear_kkt_residual().
        bool do_costate_recovery = costates_valid && !params_.enable_preconditioner;
        if (do_costate_recovery) {
            riccati_ws_.p[N] = s[N].qx;
            // Terminal: add Qxx·Δx + Cx^T·(λ + μ/s)
            for (int i = 0; i < NX; ++i)
                for (int j = 0; j < NX; ++j)
                    riccati_ws_.p[N][i] += s[N].Qxx(i,j) * riccati_ws_.dx[N][j];
            if (prob_->constraints) {
                for (int j = 0; j < NC; ++j) {
                    double lj = s[N].lambda[j];
                    double sj = s[N].s[j];
                    double dual_j = lj + ((sj > 1e-20) ? (mu_ / sj) : 0.0);
                    for (int i = 0; i < NX; ++i)
                        riccati_ws_.p[N][i] += s[N].Cx(j, i) * dual_j;
                }
            }
            for (int k = N - 1; k >= 0; --k) {
                riccati_ws_.p[k] = s[k].qx;
                // Add Qxx·Δx + Qux^T·Δu
                for (int i = 0; i < NX; ++i) {
                    for (int j = 0; j < NX; ++j)
                        riccati_ws_.p[k][i] += s[k].Qxx(i,j) * riccati_ws_.dx[k][j];
                    if (k < N)
                        for (int j = 0; j < NU; ++j)
                            riccati_ws_.p[k][i] += s[k].Qux(j,i) * riccati_ws_.du[k][j];
                }
                // Add Cx^T·(λ + μ/s)
                if (prob_->constraints) {
                    for (int j = 0; j < NC; ++j) {
                        double lj = s[k].lambda[j];
                        double sj = s[k].s[j];
                        double dual_j = lj + ((sj > 1e-20) ? (mu_ / sj) : 0.0);
                        for (int i = 0; i < NX; ++i)
                            riccati_ws_.p[k][i] += s[k].Cx(j, i) * dual_j;
                    }
                }
                // Add A^T·p[k+1]
                for (int i = 0; i < NX; ++i)
                    for (int m = 0; m < NX; ++m)
                        riccati_ws_.p[k][i] += s[k].A(m, i) * riccati_ws_.p[k + 1][m];
            }
        }

        // δp correction removed: the costate recovery now includes Cx^T·(μ/s),
        // making the barrier-free correction unnecessary.

        for (int k = 0; k <= N; ++k) {
            // ── 1. Primal feasibility ────────────────────────────
            // Dynamics defect: c_k = f(x_k,u_k) - x_{k+1}
            // Note: c was scaled by transform_qp (c *= Lx_{k+1}), so we must
            // unscale to get the physical dynamics defect.
            if (k < N) {
                double dc = 0.0;
                for (int i = 0; i < NX; ++i) {
                    double c_phys = s[k].c[i] * prec_.inv_Lx(k + 1)[i];
                    dc = std::max(dc, std::fabs(c_phys));
                }
                if (dc > dyn_defect_) dyn_defect_ = dc;
                if (dc > primal_inf_) primal_inf_ = dc;
            }
            // Constraint satisfaction: g(x,u) + s = 0
            // Use the ACTUAL nonlinear constraint, not the linearization
            if (prob_->constraints) {
                Vec<NC> g_val;
                if (k < N) {
                    prob_->constraints->evaluate(s[k].x, s[k].u, k, g_val);
                } else {
                    prob_->constraints->evaluate_terminal(s[k].x, g_val);
                }
                for (int j = 0; j < NC; ++j) {
                    double viol = std::fabs(g_val[j] + s[k].s[j]);
                    if (viol > cons_viol_) cons_viol_ = viol;
                    if (viol > primal_inf_) primal_inf_ = viol;
                }
            }

            // ── 2. Complementarity: |s_j·λ_j - μ| ───────────────
            if (prob_->constraints) {
                for (int j = 0; j < NC; ++j) {
                    double c = std::fabs(s[k].s[j] * s[k].lambda[j] - mu_);
                    if (c > compl_inf_) compl_inf_ = c;
                }
            }

            // ── 3. Inequality: s_j > 0, λ_j > 0 ──────────────────
            // Track worst violation (most negative): 0 means all satisfied
            for (int j = 0; j < NC; ++j) {
                if (s[k].s[j] < ineq_viol_)      ineq_viol_ = s[k].s[j];
                if (s[k].lambda[j] < ineq_viol_)  ineq_viol_ = s[k].lambda[j];
            }

            // ── 4. Stationarity: barrier Lagrangian gradient ─────
            // The barrier Lagrangian is:
            //   L_b = cost(x,u) + p^T·(f-x') + λ^T·(g+s) - μ·Σ ln(s)
            // At the barrier optimum, ∇_z L_b = 0 gives:
            //   ∇_x L_b = qx + Cx^T·(λ + μ/s) + A^T·p_{k+1} - p_k = 0
            //   ∇_u L_b = qu + Cu^T·(λ + μ/s) + B^T·p_{k+1}         = 0
            //
            // The costate recovery above defines p[k] using (λ + μ/s),
            // making x-stationarity = 0 by construction.  The u-stationarity
            // is the convergence measure.
            //
            // CRITICAL: we must use (λ + μ/s), not just λ, in the
            // stationarity computation.  Using only λ would leave a
            // (1-σ)·C^T·(μ/s) gap from the Mehrotra centering, creating
            // an artificial stationarity floor.
            //
            // SCALE INVARIANCE: qx, qu, Cx, Cu, A, B are all in SCALED
            // coordinates after transform_qp.  The costate p[k] is also
            // in scaled space (satisfying scaled x-stationarity).  We
            // unscale lag_x and lag_u before taking the norm to get a
            // scale-invariant convergence measure.
            {
                // Accumulate barrier Lagrangian gradient in SCALED space.
                // All terms (qx, Cx, A, B, costate p) are in scaled coords
                // after transform_qp.  We unscale at the end for a
                // scale-invariant convergence measure.
                Vec<NX> lag_x = s[k].qx;
                Vec<NU> lag_u;
                if (k < N) lag_u = s[k].qu; else lag_u.zero();

                // Track component magnitudes (in scaled space, for diagnostics)
                double grad_x_inf = lag_x.norm_inf();
                double grad_u_inf = lag_u.norm_inf();

                // Add inequality constraint dual: C^T·λ
                Vec<NX> ct_lam_x; ct_lam_x.zero();
                Vec<NU> ct_lam_u; ct_lam_u.zero();
                if (prob_->constraints) {
                    for (int j = 0; j < NC; ++j) {
                        double lj = s[k].lambda[j];
                        for (int i = 0; i < NX; ++i)
                            ct_lam_x[i] += s[k].Cx(j, i) * lj;
                        if (k < N)
                            for (int i = 0; i < NU; ++i)
                                ct_lam_u[i] += s[k].Cu(j, i) * lj;
                    }
                }
                double clam_x_inf = ct_lam_x.norm_inf();
                double clam_u_inf = ct_lam_u.norm_inf();
                for (int i = 0; i < NX; ++i) lag_x[i] += ct_lam_x[i];
                for (int i = 0; i < NU; ++i) lag_u[i] += ct_lam_u[i];

                // Add dynamics costate terms: A^T·p̂_{k+1} - p̂_k
                // Convert physical costate to scaled: p̂ = inv_Lx · p_phys
                Vec<NX> cos_x; cos_x.zero();
                Vec<NU> cos_u; cos_u.zero();
                if (costates_valid) {
                    // Pre-compute scaled costate for k+1
                    Vec<NX> p_next_s;
                    if (k < N) {
                        for (int m = 0; m < NX; ++m) {
                            p_next_s[m] = riccati_ws_.p[k+1][m];
                            if (params_.enable_preconditioner)
                                p_next_s[m] *= prec_.inv_Lx(k+1)[m];
                        }
                        for (int i = 0; i < NX; ++i)
                            for (int m = 0; m < NX; ++m)
                                cos_x[i] += s[k].A(m, i) * p_next_s[m];
                        for (int i = 0; i < NU; ++i)
                            for (int m = 0; m < NX; ++m)
                                cos_u[i] += s[k].B(m, i) * p_next_s[m];
                    }
                    Vec<NX> p_k_s;
                    for (int i = 0; i < NX; ++i) {
                        p_k_s[i] = riccati_ws_.p[k][i];
                        if (params_.enable_preconditioner)
                            p_k_s[i] *= prec_.inv_Lx(k)[i];
                    }
                    for (int i = 0; i < NX; ++i)
                        cos_x[i] -= p_k_s[i];
                }
                double cos_x_inf = cos_x.norm_inf();
                double cos_u_inf = cos_u.norm_inf();
                for (int i = 0; i < NX; ++i) lag_x[i] += cos_x[i];
                for (int i = 0; i < NU; ++i) lag_u[i] += cos_u[i];

                // ── Unscale all vectors component-wise ────────────
                // lag_scaled[i] = inv_L[i] · lag_phys[i]
                // → lag_phys[i] = lag_scaled[i] / inv_L[i]
                if (params_.enable_preconditioner) {
                    for (int i = 0; i < NX; ++i) {
                        double ilx = prec_.inv_Lx(k)[i];
                        lag_x[i]    /= ilx;
                        ct_lam_x[i] /= ilx;
                        cos_x[i]    /= ilx;
                    }
                    if (k < N) {
                        for (int i = 0; i < NU; ++i) {
                            double ilu = prec_.inv_Lu(k)[i];
                            lag_u[i]    /= ilu;
                            ct_lam_u[i] /= ilu;
                            cos_u[i]    /= ilu;
                        }
                    }
                }

                double stat_x_full = lag_x.norm_inf();
                double stat_u_full = lag_u.norm_inf();
                double stat_abs = (stat_x_full > stat_u_full) ? stat_x_full : stat_u_full;

                // Physical component magnitudes (for denominator)
                double grad_x_phys = grad_x_inf;
                double grad_u_phys = grad_u_inf;
                double clam_x_phys = ct_lam_x.norm_inf();
                double clam_u_phys = ct_lam_u.norm_inf();
                double cos_x_phys  = cos_x.norm_inf();
                double cos_u_phys  = cos_u.norm_inf();
                if (params_.enable_preconditioner) {
                    // Unscale grad vectors (we only have their inf-norms,
                    // so recompute from original scaled vectors).
                    // qx_scaled[i] = inv_Lx[i] * qx_phys[i]
                    // We need ||qx_phys||_inf = max_i |qx_scaled[i] / inv_Lx[i]|
                    double gx = 0.0, gu = 0.0;
                    for (int i = 0; i < NX; ++i)
                        gx = std::max(gx, std::fabs(s[k].qx[i] / prec_.inv_Lx(k)[i]));
                    grad_x_phys = gx;
                    if (k < N) {
                        for (int i = 0; i < NU; ++i)
                            gu = std::max(gu, std::fabs(s[k].qu[i] / prec_.inv_Lu(k)[i]));
                        grad_u_phys = gu;
                    }
                }
                double scale_x = std::max({grad_x_phys, clam_x_phys, cos_x_phys, 1.0});
                double scale_u = std::max({grad_u_phys, clam_u_phys, cos_u_phys, 1.0});
                double scale = (scale_x > scale_u) ? scale_x : scale_u;
                double stat = stat_abs / scale;
                if (stat > stat_inf_) {
                    stat_inf_ = stat;
                    stat_breakdown_[0] = grad_x_inf;
                    stat_breakdown_[1] = grad_u_inf;
                    stat_breakdown_[2] = clam_x_inf;
                    stat_breakdown_[3] = clam_u_inf;
                    stat_breakdown_[4] = cos_x_inf;
                    stat_breakdown_[5] = cos_u_inf;
                    stat_worst_node_ = k;
                }
            }
        }

        // Restore Riccati costates if we overwrote them
        if (do_costate_recovery) {
            for (int k = 0; k <= N; ++k)
                riccati_ws_.p[k] = riccati_p_save[k];
        }
    }

    bool kkt_converged() const {
        bool primal_ok  = (primal_inf_ <= params_.tol_primal);
        bool compl_ok   = (compl_inf_  <= params_.tol_compl);
        bool ineq_ok    = (ineq_viol_  >= -params_.tol_ineq);
        bool stat_ok    = (stat_inf_   <= params_.tol_stat);
        bool converged  = primal_ok && compl_ok && ineq_ok && stat_ok;
        return converged;
    }

    // ═════════════════════════════════════════════════════════════════════
    //  Compute linear KKT residual: ||K·Δz - b|| for the current Newton
    //  step, verifying that the Riccati recursion solved the linearized
    //  KKT system accurately.
    //
    //  The linearized KKT equations (after barrier slack elimination) are:
    //
    //    (D)  Δx_{k+1} - A_k·Δx_k - B_k·Δu_k - c_k = 0
    //    (Sx) q̃^x_k + H̄^xx_k·Δx_k + (H̄^ux_k)^T·Δu_k + ν_k - A_k^T·ν_{k+1} = 0
    //    (Su) q̃^u_k + H̄^ux_k·Δx_k + H̄^uu_k·Δu_k + B_k^T·ν_{k+1} = 0
    //    (Tx) q̃^x_N + H̄^xx_N·Δx_N + ν_N = 0
    //
    //  where ν_k = p_k + P_k·Δx_k (Riccati costate),
    //        q̃^x = qx + C^T·(μ/s - λ),
    //        q̃^u = qu + C^T·(μ/s - λ),
    //        H̄   = H + C^T·diag(λ/s)·C.
    //
    //  Also checks the complementarity equation:
    //    (C)  s_j·Δλ_j + λ_j·Δs_j - σμ = 0
    //
    //  Returns residuals in both absolute and relative form.
    // ═════════════════════════════════════════════════════════════════════

    KKTLinearResiduals compute_linear_kkt_residual() const {
        KKTLinearResiduals res;
        res.max_dyn_res        = 0.0;
        res.max_feas_res       = 0.0;
        res.max_stat_x_res     = 0.0;
        res.max_stat_u_res     = 0.0;
        res.max_stat_term_res  = 0.0;
        res.max_comp_res       = 0.0;
        res.max_riccati_x_res  = 0.0;
        res.max_riccati_u_res  = 0.0;
        res.rhs_dyn_norm       = 0.0;
        res.rhs_feas_norm      = 0.0;
        res.rhs_stat_x_norm    = 0.0;
        res.rhs_stat_u_norm    = 0.0;
        res.rhs_stat_term_norm = 0.0;
        res.rhs_comp_norm      = 0.0;
        res.worst_stage        = -1;
        res.worst_eq_type      = -1;

        const int N = HORIZON;
        const Stage* s = prob_->stages;
        const auto& ws = riccati_ws_;
        const auto& rs_stages = riccati_stages_;
        const double smu = sigma_ * mu_;

        // ── Convert workspace to scaled space if needed ────────────
        // After recover_primal_step/recover_dual_step, ws.dx/du/p are physical
        // but ws.P and riccati_stages_ are scaled.  Convert to scaled space
        // for consistent residual computation.
        // Convention: dx̂ = Lx·dx → dx̂[i] = dx[i]/inv_Lx[i]
        //             p̂ = inv_Lx·p → p̂[i] = p[i]*inv_Lx[i]
        Vec<NX> dx_s[N + 1], p_s[N + 1];
        Vec<NU> du_s[N];
        for (int k = 0; k <= N; ++k) {
            for (int i = 0; i < NX; ++i) {
                if (params_.enable_preconditioner) {
                    dx_s[k][i] = ws.dx[k][i] / prec_.inv_Lx(k)[i];  // = Lx·dx_phys
                    p_s[k][i]  = ws.p[k][i] * prec_.inv_Lx(k)[i];   // = inv_Lx·p_phys
                } else {
                    dx_s[k][i] = ws.dx[k][i];
                    p_s[k][i]  = ws.p[k][i];
                }
            }
            if (k < N) {
                for (int i = 0; i < NU; ++i) {
                    if (params_.enable_preconditioner)
                        du_s[k][i] = ws.du[k][i] / prec_.inv_Lu(k)[i];
                    else
                        du_s[k][i] = ws.du[k][i];
                }
            }
        }

        // ── Collect RHS norms first ───────────────────────────────
        for (int k = 0; k < N; ++k) {
            double cn = s[k].c.norm_inf();
            if (cn > res.rhs_dyn_norm) res.rhs_dyn_norm = cn;

            // Compute q̃^x = qx + C^T·((σμ + λ(g+s))/s - λ)
            Vec<NX> qx_tilde = s[k].qx;
            Vec<NU> qu_tilde = s[k].qu;
            if (prob_->constraints) {
                for (int j = 0; j < NC; ++j) {
                    double sj = s[k].s[j];
                    if (sj > 1e-14) {
                        double lj = s[k].lambda[j];
                        double gj = s[k].d[j] + sj;
                        double centering = (sigma_*mu_ + lj*gj) / sj;
                        double w = centering - lj;
                        for (int i = 0; i < NX; ++i) qx_tilde[i] += s[k].Cx(j,i) * w;
                        for (int i = 0; i < NU; ++i) qu_tilde[i] += s[k].Cu(j,i) * w;
                    }
                }
            }
            double qxn = qx_tilde.norm_inf();
            double qun = qu_tilde.norm_inf();
            if (qxn > res.rhs_stat_x_norm) res.rhs_stat_x_norm = qxn;
            if (qun > res.rhs_stat_u_norm) res.rhs_stat_u_norm = qun;
        }
        // Terminal
        {
            Vec<NX> qxN_tilde = s[N].qx;
            if (prob_->constraints) {
                for (int j = 0; j < NC; ++j) {
                    double sj = s[N].s[j];
                    if (sj > 1e-14) {
                        double lj = s[N].lambda[j];
                        double gj = s[N].d[j] + sj;
                        double centering = (sigma_*mu_ + lj*gj) / sj;
                        double w = centering - lj;
                        for (int i = 0; i < NX; ++i) qxN_tilde[i] += s[N].Cx(j,i) * w;
                    }
                }
            }
            double qxNn = qxN_tilde.norm_inf();
            if (qxNn > res.rhs_stat_term_norm) res.rhs_stat_term_norm = qxNn;
        }
        res.rhs_comp_norm = std::fabs(smu);

        // ── Compute residuals ─────────────────────────────────────
        for (int k = 0; k <= N; ++k) {
            const auto& dx_k  = dx_s[k];
            const auto& dx_k1 = (k < N) ? dx_s[k + 1] : dx_s[k]; // terminal: dx_k1 unused
            const auto& du_k  = (k < N) ? du_s[k] : du_s[0];     // terminal: no du

            const auto& rs = rs_stages[k];
            const Stage& stg = s[k];

            // ── (D) Dynamics residual (k < N only) ────────────────
            // In scaled space: r̂_dyn = Lx_{k+1} · r_phys  (since c, A, B are scaled by Lx_{k+1})
            // Unscale: r_phys[i] = r_scaled[i] * inv_Lx_{k+1}[i]
            if (k < N) {
                for (int i = 0; i < NX; ++i) {
                    double res_i = rs.c[i];  // c_k = f(x,u) - x_{k+1}
                    for (int j = 0; j < NX; ++j) res_i += rs.A(i,j) * dx_k[j];
                    for (int j = 0; j < NU; ++j) res_i += rs.B(i,j) * du_k[j];
                    res_i -= dx_k1[i];  // Δx_{k+1}
                    if (params_.enable_preconditioner)
                        res_i *= prec_.inv_Lx(k + 1)[i];  // unscale to physical
                    double ar = std::fabs(res_i);
                    if (ar > res.max_dyn_res) { res.max_dyn_res = ar; res.worst_stage = k; res.worst_eq_type = 0; }
                }
            }

            // ── (F) Constraint feasibility residual ───────────────
            // Δs + C·Δz + g + s = 0
            if (prob_->constraints) {
                for (int j = 0; j < NC; ++j) {
                    double cj_dz = 0.0;
                    for (int i = 0; i < NX; ++i)
                        cj_dz += stg.Cx(j,i) * dx_k[i];
                    if (k < N)
                        for (int i = 0; i < NU; ++i)
                            cj_dz += stg.Cu(j,i) * du_k[i];
                    double res_j = ds_[k][j] + cj_dz + stg.d[j] + stg.s[j];
                    double ar = std::fabs(res_j);
                    if (ar > res.max_feas_res) { res.max_feas_res = ar; res.worst_stage = k; res.worst_eq_type = 7; }
                    double rhs_j = std::fabs(stg.d[j] + stg.s[j]);
                    if (rhs_j > res.rhs_feas_norm) res.rhs_feas_norm = rhs_j;
                }
            }

            // ── Riccati costate ν̂_k = p̂_k + P̂_k·Δx̂_k (scaled space) ──
            Vec<NX> nu_k;
            {
                const auto& P_k = ws.P[k];  // P is always in scaled space
                for (int i = 0; i < NX; ++i) {
                    nu_k[i] = p_s[k][i];  // scaled costate
                    for (int j = 0; j < NX; ++j)
                        nu_k[i] += P_k(i,j) * dx_k[j];  // dx_k is scaled (dx_s)
                }
            }

            // Next-stage Riccati costate (for k < N)
            Vec<NX> nu_next;
            if (k < N) {
                const auto& P_next = ws.P[k + 1];
                for (int i = 0; i < NX; ++i) {
                    nu_next[i] = p_s[k + 1][i];
                    for (int j = 0; j < NX; ++j)
                        nu_next[i] += P_next(i,j) * dx_k1[j];
                }
            }

            // ── Compute q̃ (gradient with barrier centering) ──────
            Vec<NX> qx_tilde = stg.qx;
            Vec<NU> qu_tilde = stg.qu;
            if (prob_->constraints) {
                for (int j = 0; j < NC; ++j) {
                    double sj = stg.s[j];
                    if (sj > 1e-14) {
                        double lj = stg.lambda[j];
                        double gj = stg.d[j] + sj;
                        double centering = (sigma_*mu_ + lj*gj) / sj;
                        double w = centering - lj;
                        for (int i = 0; i < NX; ++i) qx_tilde[i] += stg.Cx(j,i) * w;
                        if (k < N)
                            for (int i = 0; i < NU; ++i) qu_tilde[i] += stg.Cu(j,i) * w;
                    }
                }
            }

            if (k < N) {
                // ── (S_x) Full KKT x-stationarity (Bellman equation) ──
                // Use MODIFIED gradient q̃x = qx + Cx^T·((σμ+λ(g+s))/s − λ)
                // which is what the Riccati recursion actually uses.
                for (int i = 0; i < NX; ++i) {
                    double res_i = qx_tilde[i];
                    for (int j = 0; j < NX; ++j) res_i += rs.Qxx(i,j) * dx_k[j];
                    for (int j = 0; j < NU; ++j) res_i += rs.Qux(j,i) * du_k[j];  // (H̄^ux)^T
                    res_i -= nu_k[i];  // -ν_k
                    for (int j = 0; j < NX; ++j) res_i += rs.A(j,i) * nu_next[j];  // +A^T·ν_{k+1}
                    // res_raw = inv_Lx · r_phys (scaled matrices × physical steps)
                    // r_phys = res_raw / inv_Lx
                    if (params_.enable_preconditioner)
                        res_i /= prec_.inv_Lx(k)[i];  // unscale: multiply by Lx_k
                    double ar = std::fabs(res_i);
                    if (ar > res.max_stat_x_res) { res.max_stat_x_res = ar; res.worst_stage = k; res.worst_eq_type = 1; }
                }

                // ── (S_u) u-stationarity ──────────────────────────
                // Use MODIFIED gradient q̃u (same as Riccati recursion).
                for (int i = 0; i < NU; ++i) {
                    double res_i = qu_tilde[i];
                    for (int j = 0; j < NU; ++j) res_i += rs.Quu(i,j) * du_k[j];
                    for (int j = 0; j < NX; ++j) res_i += rs.Qux(i,j) * dx_k[j];
                    for (int j = 0; j < NX; ++j) res_i += rs.B(j,i) * nu_next[j];  // B^T·ν_{k+1}
                    if (params_.enable_preconditioner)
                        res_i /= prec_.inv_Lu(k)[i];  // unscale: multiply by Lu_k
                    double ar = std::fabs(res_i);
                    if (ar > res.max_stat_u_res) { res.max_stat_u_res = ar; res.worst_stage = k; res.worst_eq_type = 2; }
                }

                // ── Pure Riccati x-stationarity (using modified gradient) ──
                // q̃x + H̄^xx·Δx + (H̄^ux)^T·Δu + A^T·ν_{k+1} - ν_k = 0
                // This is zero by construction (Riccati recursion solves this).
                for (int i = 0; i < NX; ++i) {
                    double res_i = qx_tilde[i];
                    for (int j = 0; j < NX; ++j) res_i += rs.Qxx(i,j) * dx_k[j];
                    for (int j = 0; j < NU; ++j) res_i += rs.Qux(j,i) * du_k[j];
                    for (int j = 0; j < NX; ++j) res_i += rs.A(j,i) * nu_next[j];  // +A^T·ν_{k+1}
                    res_i -= nu_k[i];  // -ν_k
                    if (params_.enable_preconditioner)
                        res_i /= prec_.inv_Lx(k)[i];
                    double ar = std::fabs(res_i);
                    if (ar > res.max_riccati_x_res) { res.max_riccati_x_res = ar; res.worst_stage = k; res.worst_eq_type = 5; }
                }

                // ── Pure Riccati u-stationarity (using modified gradient) ──
                // q̃u + H̄^uu·Δu + H̄^ux·Δx + B^T·ν_{k+1} = 0
                // This is zero by construction (Riccati recursion solves this).
                for (int i = 0; i < NU; ++i) {
                    double res_i = qu_tilde[i];
                    for (int j = 0; j < NU; ++j) res_i += rs.Quu(i,j) * du_k[j];
                    for (int j = 0; j < NX; ++j) res_i += rs.Qux(i,j) * dx_k[j];
                    for (int j = 0; j < NX; ++j) res_i += rs.B(j,i) * nu_next[j];
                    if (params_.enable_preconditioner)
                        res_i /= prec_.inv_Lu(k)[i];
                    double ar = std::fabs(res_i);
                    if (ar > res.max_riccati_u_res) { res.max_riccati_u_res = ar; res.worst_stage = k; res.worst_eq_type = 6; }
                }
            } else {
                // ── (T) Terminal x-stationarity ───────────────────
                // Use MODIFIED gradient q̃x (same as Riccati recursion).
                for (int i = 0; i < NX; ++i) {
                    double res_i = qx_tilde[i];
                    for (int j = 0; j < NX; ++j) res_i += rs.Qxx(i,j) * dx_k[j];
                    res_i -= nu_k[i];  // -ν_N
                    if (params_.enable_preconditioner)
                        res_i /= prec_.inv_Lx(N)[i];
                    double ar = std::fabs(res_i);
                    if (ar > res.max_stat_term_res) { res.max_stat_term_res = ar; res.worst_stage = k; res.worst_eq_type = 3; }
                }
            }

            // ── (C) Complementarity residual ──────────────────────
            // Mehrotra corrector: s·Δλ + λ·Δs = σμ - sλ - cross
            //   residual = s·Δλ + λ·Δs + sλ + cross - σμ
            // Predictor (σ=0, cross=0): s·Δλ + λ·Δs = -sλ
            //   residual = s·Δλ + λ·Δs + sλ
            if (prob_->constraints) {
                for (int j = 0; j < NC; ++j) {
                    double cross = (sigma_ > 0.0 && cross_term_accepted_)
                                 ? ds_aff_[k][j] * dlambda_aff_[k][j] : 0.0;
                    double sl = stg.s[j] * stg.lambda[j];
                    double res_c = stg.s[j] * dlambda_[k][j]
                                 + stg.lambda[j] * ds_[k][j]
                                 + sl
                                 + cross
                                 - smu;
                    double ar = std::fabs(res_c);
                    if (ar > res.max_comp_res) { res.max_comp_res = ar; res.worst_stage = k; res.worst_eq_type = 4; }
                }
            }
        }

        // ── Compute relative residuals ────────────────────────────
        res.rel_dyn_res        = res.max_dyn_res        / std::max(res.rhs_dyn_norm, 1e-14);
        res.rel_feas_res       = res.max_feas_res       / std::max(res.rhs_feas_norm, 1e-14);
        res.rel_stat_x_res     = res.max_stat_x_res     / std::max(res.rhs_stat_x_norm, 1e-14);
        res.rel_stat_u_res     = res.max_stat_u_res     / std::max(res.rhs_stat_u_norm, 1e-14);
        res.rel_stat_term_res  = res.max_stat_term_res  / std::max(res.rhs_stat_term_norm, 1e-14);
        res.rel_comp_res       = res.max_comp_res       / std::max(res.rhs_comp_norm, 1e-14);

        // Aggregate
        res.max_abs_res = res.max_dyn_res;
        if (res.max_feas_res      > res.max_abs_res) res.max_abs_res = res.max_feas_res;
        if (res.max_stat_x_res    > res.max_abs_res) res.max_abs_res = res.max_stat_x_res;
        if (res.max_stat_u_res    > res.max_abs_res) res.max_abs_res = res.max_stat_u_res;
        if (res.max_stat_term_res > res.max_abs_res) res.max_abs_res = res.max_stat_term_res;
        if (res.max_comp_res      > res.max_abs_res) res.max_abs_res = res.max_comp_res;

        res.max_rel_res = res.rel_dyn_res;
        if (res.rel_feas_res      > res.max_rel_res) res.max_rel_res = res.rel_feas_res;
        if (res.rel_stat_x_res    > res.max_rel_res) res.max_rel_res = res.rel_stat_x_res;
        if (res.rel_stat_u_res    > res.max_rel_res) res.max_rel_res = res.rel_stat_u_res;
        if (res.rel_stat_term_res > res.max_rel_res) res.max_rel_res = res.rel_stat_term_res;
        if (res.rel_comp_res      > res.max_rel_res) res.max_rel_res = res.rel_comp_res;

        return res;
    }

    // ═══════════════════════════════════════════════════════════════════
    //  Build LHS (Hessian + barrier Hessian) — identical for both steps
    // ═══════════════════════════════════════════════════════════════════

    void build_kkt_lhs() {
        const int N = HORIZON;
        Stage* s = prob_->stages;

        for (int k = 0; k <= N; ++k) {
            // ── Start with cost Hessian (already scaled by transform_qp) ──
            qxx_work_[k] = s[k].Qxx;
            quu_work_[k] = s[k].Quu;
            qux_work_[k] = s[k].Qux;

            if (prob_->constraints) {
                // ── Barrier Hessian: H_bar = C^T·diag(λ/s)·C ────────
                // Uses SCALED Cx/Cu (from transform_qp) so the barrier
                // Hessian is consistent with the scaled cost Hessian.
                for (int j = 0; j < NC; ++j) {
                    double sj = s[k].s[j];
                    double lj = s[k].lambda[j];
                    if (sj < 1e-14) continue;
                    double ratio = lj / sj;

                    for (int r = 0; r < NX; ++r) {
                        double cjr = s[k].Cx(j, r);
                        if (std::fabs(cjr) < 1e-30) continue;
                        for (int c = 0; c < NX; ++c) {
                            double cjc = s[k].Cx(j, c);
                            qxx_work_[k](r, c) += ratio * cjr * cjc;
                        }
                        if (k < N)
                            for (int c = 0; c < NU; ++c) {
                                double cujc = s[k].Cu(j, c);
                                qux_work_[k](c, r) += ratio * cujc * cjr;
                            }
                    }
                    if (k < N)
                        for (int r = 0; r < NU; ++r) {
                            double cur = s[k].Cu(j, r);
                            for (int c = 0; c < NU; ++c) {
                                double cuc = s[k].Cu(j, c);
                                quu_work_[k](r, c) += ratio * cur * cuc;
                            }
                        }
                }
            }

            // Copy LHS (Hessian + dynamics) into Riccati stages
            riccati_stages_[k].Qxx = qxx_work_[k];
            riccati_stages_[k].Quu = quu_work_[k];
            riccati_stages_[k].Qux = qux_work_[k];
            riccati_stages_[k].A   = s[k].A;
            riccati_stages_[k].B   = s[k].B;
            riccati_stages_[k].c   = s[k].c;
        }
    }

    // ═══════════════════════════════════════════════════════════════════
    //  Build RHS (gradient + barrier correction) — differs per step
    //  Predictor: σ=0 (affine), Corrector: σ from adaptive scheduling
    // ═══════════════════════════════════════════════════════════════════

    void build_kkt_rhs() {
        const int N = HORIZON;
        Stage* s = prob_->stages;

        for (int k = 0; k <= N; ++k) {
            // ── Gradient = cost gradient ───────────────────────────
            qx_work_[k] = s[k].qx;
            qu_work_[k] = s[k].qu;

            if (prob_->constraints) {
                // Add -C^T·λ (Lagrangian gradient with correct sign for
                // the reduced KKT stationarity equation).
                for (int j = 0; j < NC; ++j) {
                    double lj = s[k].lambda[j];
                    for (int i = 0; i < NX; ++i)
                        qx_work_[k][i] -= s[k].Cx(j, i) * lj;
                    if (k < N)
                        for (int i = 0; i < NU; ++i)
                            qu_work_[k][i] -= s[k].Cu(j, i) * lj;
                }

                // ── Barrier centering: +C^T·(σμ + λ(g+s))/s ────────
                // Mehrotra corrector RHS: s·Δλ + λ·Δs = σμ - sλ - cross.
                // Solving for Δλ: (σμ - sλ - cross - λ·Δs) / s.
                // Substituting Δs = -(g+s) - C·Δz:
                //   Δλ = (σμ - sλ - cross + λ(g+s) + λ·C·Δz) / s
                // The λ·C·Δz/s term merges into the Hessian (barrier Hessian
                // H̄ = H + C^T·(λ/s)·C).  The remaining gradient contribution
                // per constraint is (σμ - sλ + λ(g+s))/s = (σμ + λg)/s.
                // Combined with the -C^T·λ above, the net gradient is:
                //   ∇ℓ + C^T·[-λ + (σμ + λ(g+s))/s] = ∇ℓ + C^T·(σμ + λg)/s
                for (int j = 0; j < NC; ++j) {
                    double sj = s[k].s[j];
                    if (sj < 1e-14) continue;
                    double lj = s[k].lambda[j];
                    double gj = s[k].d[j] + sj;  // constraint infeasibility: g+s
                    double centering = (sigma_ * mu_ + lj * gj) / sj;

                    for (int i = 0; i < NX; ++i)
                        qx_work_[k][i] += s[k].Cx(j, i) * centering;
                    if (k < N)
                        for (int i = 0; i < NU; ++i)
                            qu_work_[k][i] += s[k].Cu(j, i) * centering;
                }
            }

            // Transfer to Riccati stages
            riccati_stages_[k].qx  = qx_work_[k];
            riccati_stages_[k].qu  = qu_work_[k];
        }
    }

    // ═════════════════════════════════════════════════════════════════════
    //  Solve the reduced KKT via Riccati recursion.
    //
    //  In Mehrotra predictor-corrector, the LHS (Hessian + dynamics)
    //  is identical for predictor and corrector — only the RHS changes.
    //  So we factor the LHS once, then do cheap RHS + forward twice.
    // ═════════════════════════════════════════════════════════════════════

    Status solve_kkt_lhs() {
        // Adaptive regularization: keep minimal to preserve Newton descent.
        // reg perturbs the Schur complement S → S + reg·I, which corrupts
        // the descent property: error ≈ reg · ||C^Tλ|| / λ_min(S).
        // For high cost/control ratios (e.g. 1000), even reg=1e-5 can
        // overwhelm the true descent term.
        // LDLT factorization stability only needs reg ≈ ε·κ(S).
        reg_used_ = 0.0;  // Reset each call — measure actual reg used
        double reg_base = 1e-12;
        if (cond_estimate_ > 1e6) {
            reg_base = 1e-8;
        } else if (cond_estimate_ > 1e5) {
            reg_base = 1e-10;
        } else if (cond_estimate_ > 1e4) {
            reg_base = 1e-11;
        }
        return Ricc::backward_lhs(riccati_stages_, riccati_ws_, reg_base, reg_used_);
    }

    Status solve_kkt_rhs_and_forward() {
        Status st = Ricc::backward_rhs(riccati_stages_, riccati_ws_);
        if (st != Status::SUCCESS) return st;

        Vec<NX> dx0;
        for (int i = 0; i < NX; ++i)
            dx0[i] = prob_->x0[i] - prob_->stages[0].x[i];

        // Scale dx0 for Riccati forward pass (dx̂0 = Lx·dx0)
        if (params_.enable_preconditioner) {
            prec_.scale_dx0(dx0);
        }

        return Ricc::forward(riccati_stages_, riccati_ws_, dx0);
    }

    // Legacy combined solve (kept for backward compat with tests)
    Status solve_kkt_via_riccati() {
        Status st = Ricc::backward(riccati_stages_, riccati_ws_, 1e-12, reg_used_);
        if (st != Status::SUCCESS) return st;

        Vec<NX> dx0;
        for (int i = 0; i < NX; ++i)
            dx0[i] = prob_->x0[i] - prob_->stages[0].x[i];

        return Ricc::forward(riccati_stages_, riccati_ws_, dx0);
    }

    // ═════════════════════════════════════════════════════════════════
    //  Adaptive centering: σ ∈ [σ_min, σ_max] based on convergence.
    //  When primal_inf or stationarity is high → σ → σ_max (slow μ reduce)
    //  When both are small                     → σ → σ_min (fast μ reduce)
    // ═════════════════════════════════════════════════════════════════

    double compute_adaptive_sigma() {
        constexpr double sigma_min = 0.3;
        constexpr double sigma_max = 0.8;

        // Use complementarity gap and primal infeasibility for sigma adaptation.
        // Note: stat_inf_ is NOT used here because it is computed in scaled
        // coordinates (after transform_qp) and is not scale-invariant.
        // compl_inf_ = |s·λ − μ| is scale-invariant (both s and λ are physical).
        double worst = std::max(primal_inf_, compl_inf_);

        // Map log10(worst) to [0, 1]:
        //   worst = 1.0   → log10 = 0   → t = 0/4 = 0.0   (not converged)
        //   worst = 1e-4  → log10 = -4  → t = 4/4 = 1.0   (well converged)
        double log_worst = std::log10(std::max(worst, 1e-16));
        double t = std::clamp(-log_worst / 4.0, 0.0, 1.0);

        // progress=0 (bad convergence) → σ = σ_max
        // progress=1 (good convergence) → σ = σ_min
        double sigma = sigma_max - (sigma_max - sigma_min) * t;

        // FTB-bottleneck override: when the affine predictor is also
        // blocked by FTB (aff_alpha_p_ < 0.1), the subproblem is
        // genuinely hard. Keep the standard adaptive σ (which reflects
        // KKT quality) rather than forcing a reduction.
        // The barrier schedule handles μ reduction speed separately.

        return sigma;
    }

    // ═════════════════════════════════════════════════════════════════════
    //  Recover Δs, Δλ from primal step (for the corrector direction)
    // ═════════════════════════════════════════════════════════════════════

    void recover_inequality_steps(double sigma, bool apply_cross = true) {
        const int N = HORIZON;
        Stage* s = prob_->stages;

        // ── Compute corrector steps for Δs, Δλ ──────────────────
        for (int k = 0; k <= N; ++k) {
            for (int j = 0; j < NC; ++j) {
                double gj = s[k].d[j];
                double cj_dz = 0.0;
                for (int i = 0; i < NX; ++i)
                    cj_dz += s[k].Cx(j, i) * riccati_ws_.dx[k][i];
                if (k < N)
                    for (int i = 0; i < NU; ++i)
                        cj_dz += s[k].Cu(j, i) * riccati_ws_.du[k][i];

                ds_[k][j] = -(gj + s[k].s[j]) - cj_dz;

                if (s[k].s[j] > 1e-14) {
                    // Mehrotra cross-term (corrector only, sigma > 0).
                    // Second-order correction: ds_aff·dλ_aff.
                    // Gated globally by the caller: when apply_cross is false
                    // the cross-term has been rejected because it pushed the
                    // predicted average complementarity above 2*mu.
                    double cross = (sigma > 0.0 && apply_cross)
                                 ? ds_aff_[k][j] * dlambda_aff_[k][j] : 0.0;
                    // dlambda from Mehrotra complementarity linearization:
                    //   s·dλ + λ·ds = σμ - sλ - cross
                    //   dλ = (σμ - sλ - cross - λ·ds) / s
                    // The -sλ term is the Newton linearization residual
                    // (target σμ minus current value sλ, minus cross correction).
                    // For predictor (σ=0, cross=0): dλ = (-sλ - λ·ds)/s = -λ - λ·ds/s
                    // For corrector (σ>0): dλ = (σμ - sλ - cross - λ·ds) / s
                    double sl_term = s[k].s[j] * s[k].lambda[j];
                    dlambda_[k][j] = (sigma * mu_ - sl_term - cross
                                      - s[k].lambda[j] * ds_[k][j]) / s[k].s[j];
                } else {
                    dlambda_[k][j] = 0.0;
                }
            }
        }
    }

    // ═════════════════════════════════════════════════════════════════════
    //  Fraction-to-boundary step size (for the corrector direction)
    // ═════════════════════════════════════════════════════════════════════

    double fraction_to_boundary() {
        const int N = HORIZON;
        double alpha = 1.0;

        for (int k = 0; k <= N; ++k) {
            Stage& s = prob_->stages[k];
            for (int j = 0; j < NC; ++j) {
                if (ds_[k][j] < -1e-16) {
                    double a = -params_.tau * s.s[j] / ds_[k][j];
                    if (a < alpha) alpha = a;
                }
                if (dlambda_[k][j] < -1e-16) {
                    double a = -params_.tau * s.lambda[j] / dlambda_[k][j];
                    if (a < alpha) alpha = a;
                }
            }
        }
        return alpha;
    }

    // ═════════════════════════════════════════════════════════════════════
    //  Apply primal-dual step: α for (z, s), αλ for λ
    // ═════════════════════════════════════════════════════════════════════

    void apply_primal_dual_step(double alpha, double alpha_lam) {
        const int N = HORIZON;
        Stage* s = prob_->stages;

        for (int k = 0; k <= N; ++k) {
            for (int i = 0; i < NX; ++i)
                s[k].x[i] += alpha * riccati_ws_.dx[k][i];
            if (k < N)
                for (int i = 0; i < NU; ++i)
                    s[k].u[i] += alpha * riccati_ws_.du[k][i];
            for (int j = 0; j < NC; ++j) {
                s[k].s[j]      += alpha      * ds_[k][j];
                s[k].lambda[j] += alpha_lam  * dlambda_[k][j];
            }
        }
    }

    // ═════════════════════════════════════════════════════════════════════
    //  Compute FTB step-size limits (used by main loop + SOC)
    // ═════════════════════════════════════════════════════════════════════

    // ═════════════════════════════════════════════════════════════════════
    //  Objective value (pure cost, no barrier/penalty)
    // ═════════════════════════════════════════════════════════════════════

    double compute_objective() {
        const int N = HORIZON;
        Stage* s = prob_->stages;
        double obj = 0.0;
        for (int k = 0; k < N; ++k)
            obj += prob_->cost->stage_cost(s[k].x, s[k].u, k);
        obj += prob_->cost->terminal_cost(s[N].x);
        return obj;
    }



    // ═════════════════════════════════════════════════════════════════════
    //  Directional derivative of cost along search direction
    //  Dphi = ∇cost · Δz  (used by filter switching condition)
    // ═════════════════════════════════════════════════════════════════════

    double cost_directional_derivative() const {
        double slope = 0.0;
        const int N = HORIZON;
        for (int k = 0; k <= N; ++k) {
            for (int i = 0; i < NX; ++i)
                slope += prob_->stages[k].qx[i] * riccati_ws_.dx[k][i];
            if (k < N)
                for (int i = 0; i < NU; ++i)
                    slope += prob_->stages[k].qu[i] * riccati_ws_.du[k][i];
        }
        return slope;
    }

    // ═══════════════════════════════════════════════════════════════════
    //  Diagnostic: log collapse-analysis metrics when α_ftb shrinks
    // ═══════════════════════════════════════════════════════════════════

    void diagnose_ftb_collapse(double alpha_ftb) {
        const int N = HORIZON;
        Stage* s = prob_->stages;

        printf("        ── ftb collapse @ α=%.2e ──\n", alpha_ftb);

        // ── 1. Which bound limits? Primal (s) vs dual (λ) ──────
        double tau_p = 1.0, tau_d = 1.0;
        int hit_s_k = -1, hit_s_j = -1, hit_l_k = -1, hit_l_j = -1;
        for (int k = 0; k <= N; ++k) {
            for (int j = 0; j < NC; ++j) {
                if (ds_[k][j] < -1e-16) {
                    double a = -params_.tau * s[k].s[j] / ds_[k][j];
                    if (a < tau_p) { tau_p = a; hit_s_k = k; hit_s_j = j; }
                }
                if (dlambda_[k][j] < -1e-16) {
                    double a = -params_.tau * s[k].lambda[j] / dlambda_[k][j];
                    if (a < tau_d) { tau_d = a; hit_l_k = k; hit_l_j = j; }
                }
            }
        }
        printf("        τ_p=%.2e  τ_d=%.2e  bottleneck=%s\n",
               tau_p, tau_d, (tau_p < tau_d) ? "PRIMAL" : "DUAL");
        if (tau_p < tau_d)
            printf("          s[%d,%d]=%.1e  Δs=%.1e  ratio=%5.1f\n",
                   hit_s_k, hit_s_j, s[hit_s_k].s[hit_s_j],
                   ds_[hit_s_k][hit_s_j],
                   s[hit_s_k].s[hit_s_j] > 1e-14 ? -ds_[hit_s_k][hit_s_j]/s[hit_s_k].s[hit_s_j] : 0.0);
        else
            printf("          λ[%d,%d]=%.1e  Δλ=%.1e  ratio=%5.1f\n",
                   hit_l_k, hit_l_j, s[hit_l_k].lambda[hit_l_j],
                   dlambda_[hit_l_k][hit_l_j],
                   s[hit_l_k].lambda[hit_l_j] > 1e-14 ? -dlambda_[hit_l_k][hit_l_j]/s[hit_l_k].lambda[hit_l_j] : 0.0);

        // ── 2. Proximity: closest variables to bounds ───────────
        double min_s = 1e100, min_lam = 1e100;
        for (int k = 0; k <= N; ++k) {
            for (int j = 0; j < NC; ++j) {
                if (s[k].s[j]      < min_s)   min_s   = s[k].s[j];
                if (s[k].lambda[j] < min_lam) min_lam = s[k].lambda[j];
            }
        }
        printf("        near-bound:  min_s=%.1e  min_λ=%.1e\n", min_s, min_lam);

        // ── 3. Step-to-distance ratio ───────────────────────────
        double max_ratio_s = 0.0, max_ratio_lam = 0.0;
        for (int k = 0; k <= N; ++k) {
            for (int j = 0; j < NC; ++j) {
                if (s[k].s[j] > 1e-14 && ds_[k][j] < 0) {
                    double r = -ds_[k][j] / s[k].s[j];
                    if (r > max_ratio_s) max_ratio_s = r;
                }
                if (s[k].lambda[j] > 1e-14 && dlambda_[k][j] < 0) {
                    double r = -dlambda_[k][j] / s[k].lambda[j];
                    if (r > max_ratio_lam) max_ratio_lam = r;
                }
            }
        }
        printf("        step/slack-ratio:  max(Δs/s)=%.1e  max(Δλ/λ)=%.1e\n",
               max_ratio_s, max_ratio_lam);

        // ── 4. Complementarity health ───────────────────────────
        double avg_mu = 0.0, max_mu = 0.0, min_mu_pair = 1e100;
        int count = 0;
        for (int k = 0; k <= N; ++k) {
            for (int j = 0; j < NC; ++j) {
                double c = s[k].s[j] * s[k].lambda[j];
                if (s[k].s[j] > 1e-14 && s[k].lambda[j] > 1e-14) {
                    avg_mu += c;
                    if (c > max_mu) max_mu = c;
                    if (c < min_mu_pair) min_mu_pair = c;
                    count++;
                }
            }
        }
        if (count > 0) avg_mu /= count;
        printf("        complementarity:  avg(sλ)=%.1e  max=%.1e  min=%.1e  spread=%.1e\n",
               avg_mu, max_mu, min_mu_pair, (min_mu_pair > 1e-14 ? max_mu/min_mu_pair : -1.0));

        // ── 5. Dual multiplier magnitude ───────────────────────
        double max_lam = 0.0;
        for (int k = 0; k <= N; ++k)
            for (int j = 0; j < NC; ++j)
                if (s[k].lambda[j] > max_lam) max_lam = s[k].lambda[j];
        printf("        dual:  ‖λ‖∞=%.1e  μ/‖λ‖∞=%.1e\n", max_lam, mu_ / (max_lam + 1e-14));

        // ── 6. Constraint violation ─────────────────────────────
        double max_viol = 0.0; int max_viol_j = -1;
        for (int k = 0; k <= N; ++k) {
            for (int j = 0; j < NC; ++j) {
                double viol = std::fabs(s[k].d[j] + s[k].s[j]);
                if (viol > max_viol) { max_viol = viol; max_viol_j = j; }
            }
        }
        printf("        constraint viol:  max=%.1e (j=%d)\n", max_viol, max_viol_j);

        // ── 7. Primal/dual ratio ────────────────────────────────
        printf("        primal/dual ratio:  primal=%.1e  dual=%.1e  ratio=%.1e\n",
               primal_inf_, stat_inf_,
               (stat_inf_ > 1e-14 ? primal_inf_ / stat_inf_ : -1.0));

        fflush(stdout);
    }

    // ═════════════════════════════════════════════════════════════════════
    //  Collect all monitoring statistics for this iteration
    // ═════════════════════════════════════════════════════════════════════

    // ═════════════════════════════════════════════════════════════════════
    //  Per-iteration diagnostics (FTB computed separately via compute_ftb_limits)
    // ═════════════════════════════════════════════════════════════════════

    void log_iteration(int iter, double sigma, double alpha_p, double alpha_d) {
        const int N = HORIZON;
        const Stage* s = prob_->stages;

        // ── KKT conditioning snapshot ────────────────────────────
        double max_lam = 0.0, min_slack = 1e100, max_slack = 0.0, min_lam = 1e100;
        double max_Hdiag = 0.0, min_Hdiag_nz = 1e100;
        double max_ratio = 0.0, min_ratio = 1e100;  // λ/s range
        double max_cost_diag = 0.0, min_cost_diag = 1e100;  // pure cost Hessian diag
        double max_bar_diag = 0.0, min_bar_diag_nz = 1e100;  // barrier Hessian diag contrib
        for (int k = 0; k <= N; ++k) {
            for (int j = 0; j < NC; ++j) {
                if (s[k].s[j] < min_slack) min_slack = s[k].s[j];
                if (s[k].s[j] > max_slack) max_slack = s[k].s[j];
                if (s[k].lambda[j] < min_lam) min_lam = s[k].lambda[j];
                if (s[k].lambda[j] > max_lam) max_lam = s[k].lambda[j];
                double r = s[k].lambda[j] / std::max(s[k].s[j], 1e-14);
                if (r > max_ratio) max_ratio = r;
                if (r < min_ratio) min_ratio = r;
            }
            for (int i = 0; i < NX; ++i) {
                double d = riccati_stages_[k].Qxx(i,i);
                if (d > max_Hdiag) max_Hdiag = d;
                if (d > 1e-14 && d < min_Hdiag_nz) min_Hdiag_nz = d;
                double cd = s[k].Qxx(i,i);
                if (cd > max_cost_diag) max_cost_diag = cd;
                if (cd > 1e-14 && cd < min_cost_diag) min_cost_diag = cd;
                double bd = d - cd;
                if (bd > max_bar_diag) max_bar_diag = bd;
                if (bd > 1e-14 && bd < min_bar_diag_nz) min_bar_diag_nz = bd;
            }
            if (k < N) {
                for (int i = 0; i < NU; ++i) {
                    double d = riccati_stages_[k].Quu(i,i);
                    if (d > max_Hdiag) max_Hdiag = d;
                    if (d > 1e-14 && d < min_Hdiag_nz) min_Hdiag_nz = d;
                    double cd = s[k].Quu(i,i);
                    if (cd > max_cost_diag) max_cost_diag = cd;
                    if (cd > 1e-14 && cd < min_cost_diag) min_cost_diag = cd;
                    double bd = d - cd;
                    if (bd > max_bar_diag) max_bar_diag = bd;
                    if (bd > 1e-14 && bd < min_bar_diag_nz) min_bar_diag_nz = bd;
                }
            }
        }
        if (min_Hdiag_nz > 1e99) min_Hdiag_nz = 1e-14;
        if (min_cost_diag > 1e99) min_cost_diag = 1e-14;
        if (min_bar_diag_nz > 1e99) min_bar_diag_nz = 1e-14;
        double condH = max_Hdiag / min_Hdiag_nz;
        double cost_cond = max_cost_diag / min_cost_diag;
        double bar_cond = (max_bar_diag > 1e-14) ? (max_bar_diag / min_bar_diag_nz) : 1.0;

        // Store condition estimate for adaptive regularization
        cond_estimate_ = condH;

        if (params_.verbosity < 1) return;

        // ── Dump stats line ────────────────────────────────────────
            double cur_cost = compute_objective();
            printf("[iter %3d] mu=%.2e sig=%.3f | "
                   "prim=%.2e(dyn=%.1e cons=%.1e) stat=%.2e compl=%.2e ineq=%.2e | "
                   "ap=%.3f ad=%.3f | cond=%.1e | reg=%.1e | "
                   "cost=%.4e | "
                   "linKKT=%.2e(%s)\n",
                   iter, mu_, sigma,
                   primal_inf_, dyn_defect_, cons_viol_, stat_inf_, compl_inf_, ineq_viol_,
                   alpha_p, alpha_d,
                   condH, reg_used_,
                   cur_cost,
                   linear_kkt_res_.max_rel_res, linear_kkt_res_.quality_label());
            if (params_.verbosity >= 2) {
                printf("  [diag] s=[%.2e,%.2e] lam=[%.2e,%.2e] "
                       "lam/s=[%.2e,%.2e]\n",
                       min_slack, max_slack,
                       min_lam, max_lam,
                       min_ratio, max_ratio);
                printf("  [cond] total=%.1e  cost=%.1e(%.1e/%.1e)  "
                       "barrier=%.1e(%.1e/%.1e)\n",
                       condH,
                       cost_cond, max_cost_diag, min_cost_diag,
                       bar_cond, max_bar_diag, min_bar_diag_nz);

                // ── Constraint violation breakdown ────────────────
                if (iter < 10 && prob_->constraints) {
                    double max_g_s = 0.0;
                    int worst_k_cons = -1, worst_j_cons = -1;
                    for (int kk = 0; kk <= N; ++kk) {
                        Vec<NC> g_val;
                        if (kk < N)
                            prob_->constraints->evaluate(s[kk].x, s[kk].u, kk, g_val);
                        else
                            prob_->constraints->evaluate_terminal(s[kk].x, g_val);
                        for (int jj = 0; jj < NC; ++jj) {
                            double gs = std::fabs(g_val[jj] + s[kk].s[jj]);
                            if (gs > max_g_s) { max_g_s = gs; worst_k_cons = kk; worst_j_cons = jj; }
                        }
                    }
                    if (worst_k_cons >= 0) {
                        Vec<NC> g_val;
                        if (worst_k_cons < N)
                            prob_->constraints->evaluate(s[worst_k_cons].x, s[worst_k_cons].u, worst_k_cons, g_val);
                        else
                            prob_->constraints->evaluate_terminal(s[worst_k_cons].x, g_val);
                        printf("  [cons] worst |g+s|=%.2e at k=%d j=%d | g=%.3e s=%.3e ds=%.3e\n",
                               max_g_s, worst_k_cons, worst_j_cons,
                               g_val[worst_j_cons], s[worst_k_cons].s[worst_j_cons],
                               ds_[worst_k_cons][worst_j_cons]);
                    }
                }

                // ── Average complementarity: current / predictor / corrector
                {
                    double mu_cur = 0.0, mu_pred = 0.0, mu_corr = 0.0;
                    int cnt = 0;
                    double alpha_aff_p = 1.0, alpha_aff_d = 1.0;
                    for (int kk = 0; kk <= N; ++kk) {
                        for (int jj = 0; jj < NC; ++jj) {
                            if (ds_aff_[kk][jj] < -1e-16) {
                                double a = -params_.tau * s[kk].s[jj] / ds_aff_[kk][jj];
                                if (a < alpha_aff_p) alpha_aff_p = a;
                            }
                            if (dlambda_aff_[kk][jj] < -1e-16) {
                                double a = -params_.tau * s[kk].lambda[jj] / dlambda_aff_[kk][jj];
                                if (a < alpha_aff_d) alpha_aff_d = a;
                            }
                        }
                    }
                    for (int kk = 0; kk <= N; ++kk) {
                        for (int jj = 0; jj < NC; ++jj) {
                            double sv = s[kk].s[jj];
                            double lv = s[kk].lambda[jj];
                            mu_cur += sv * lv;
                            // predictor at alpha_aff
                            double sp = sv + alpha_aff_p * ds_aff_[kk][jj];
                            double lp = lv + alpha_aff_d * dlambda_aff_[kk][jj];
                            mu_pred += std::max(sp, 0.0) * std::max(lp, 0.0);
                            // corrector at alpha_p, alpha_d
                            double sc = sv + alpha_p * ds_[kk][jj];
                            double lc = lv + alpha_d * dlambda_[kk][jj];
                            mu_corr += std::max(sc, 0.0) * std::max(lc, 0.0);
                            cnt++;
                        }
                    }
                    if (cnt > 0) {
                        mu_cur /= cnt; mu_pred /= cnt; mu_corr /= cnt;
                    }
                    printf("  [mu] cur=%.3e pred=%.3e corr=%.3e"
                           " (a_aff_p=%.3f a_aff_d=%.3f)\n",
                           mu_cur, mu_pred, mu_corr,
                           alpha_aff_p, alpha_aff_d);
                }

                // ── Cross-term ds_aff * dlambda_aff diagnostic ──────
                {
                    double max_abs_cross = 0.0, sum_abs_cross = 0.0;
                    double sum_sl = 0.0, sum_cross = 0.0;
                    int worst_kc = -1, worst_jc = -1;
                    double worst_cross_val = 0.0;
                    int total = 0;
                    for (int kk = 0; kk <= N; ++kk) {
                        for (int jj = 0; jj < NC; ++jj) {
                            double cross = ds_aff_[kk][jj] * dlambda_aff_[kk][jj];
                            double sl = s[kk].s[jj] * s[kk].lambda[jj];
                            sum_abs_cross += std::fabs(cross);
                            sum_cross += cross;
                            sum_sl += sl;
                            total++;
                            if (std::fabs(cross) > max_abs_cross) {
                                max_abs_cross = std::fabs(cross);
                                worst_kc = kk; worst_jc = jj;
                                worst_cross_val = cross;
                            }
                        }
                    }
                    double avg_abs = (total > 0) ? sum_abs_cross / total : 0.0;
                    printf("  [xterm] ds_aff*dL_aff: worst=%.3e (k=%d j=%d)"
                           " avg_abs=%.3e sum=%.3e sum_s*L=%.3e"
                           " ratio=%.2f\n",
                           worst_cross_val, worst_kc, worst_jc,
                           avg_abs, sum_cross, sum_sl,
                           (sum_sl > 0.0) ? sum_cross / sum_sl : 0.0);
                }

                // ── Slack distribution histogram (log-scale buckets) ──
                {
                    // Buckets: [0,0.01) [0.01,0.1) [0.1,1) [1,10) [10,100) [100,1e4) [1e4,inf)
                    constexpr int NB = 7;
                    const double edges[NB + 1] = {0.0, 0.01, 0.1, 1.0, 10.0, 100.0, 1e4, 1e100};
                    int hist[NB] = {};
                    double s_min = 1e100, s_max = 0.0;
                    for (int kk = 0; kk <= HORIZON; ++kk) {
                        for (int jj = 0; jj < NC; ++jj) {
                            double sv = s[kk].s[jj];
                            if (sv < s_min) s_min = sv;
                            if (sv > s_max) s_max = sv;
                            for (int b = 0; b < NB; ++b) {
                                if (sv >= edges[b] && sv < edges[b + 1]) {
                                    hist[b]++;
                                    break;
                                }
                            }
                        }
                    }
                    printf("  [sdist] min=%.2e max=%.2e | "
                           "<.01:%d  .01-.1:%d  .1-1:%d  1-10:%d"
                           "  10-100:%d  100-1e4:%d  >1e4:%d\n",
                           s_min, s_max,
                           hist[0], hist[1], hist[2], hist[3],
                           hist[4], hist[5], hist[6]);
                }

                // ── Step composition: feedforward vs feedback ────────
                {
                    double max_d = 0.0, max_du = 0.0, max_ds = 0.0;
                    double max_Kdx = 0.0, max_p = 0.0;
                    int worst_kd = -1, worst_kds = -1;
                    for (int kk = 0; kk <= N; ++kk) {
                        for (int jj = 0; jj < NC; ++jj) {
                            double ad = std::fabs(ds_[kk][jj]);
                            if (ad > max_ds) { max_ds = ad; worst_kds = kk; }
                        }
                        if (kk < N) {
                            for (int i = 0; i < NU; ++i) {
                                double ad = std::fabs(riccati_ws_.d[kk][i]);
                                if (ad > max_d) { max_d = ad; worst_kd = kk; }
                                double adu = std::fabs(riccati_ws_.du[kk][i]);
                                if (adu > max_du) max_du = adu;
                                double kdx = 0.0;
                                for (int j = 0; j < NX; ++j)
                                    kdx += std::fabs(riccati_ws_.K[kk](i,j) * riccati_ws_.dx[kk][j]);
                                if (kdx > max_Kdx) max_Kdx = kdx;
                            }
                        }
                        for (int i = 0; i < NX; ++i) {
                            double ap = std::fabs(riccati_ws_.p[kk][i]);
                            if (ap > max_p) max_p = ap;
                        }
                    }
                    printf("  [step] |d|=%.2e(wk=%d) |Kdx|=%.2e |du|=%.2e"
                           " |ds|=%.2e(wk=%d) |p|=%.2e"
                           " d/Kdx=%.1f\n",
                           max_d, worst_kd, max_Kdx, max_du,
                           max_ds, worst_kds, max_p,
                           (max_Kdx > 1e-14) ? max_d / max_Kdx : 0.0);

                    // Per-stage breakdown: |x|, |dx|, s_min, ds_min, ds_max
                    if (iter <= 2 || iter % 10 == 0) {
                        printf("  [per-stage] k : |x|_inf     |dx|_inf    s_min       ds_min      ds_max\n");
                        for (int kk = 0; kk <= N; ++kk) {
                            double mx = 0.0;
                            for (int i = 0; i < NX; ++i) {
                                double v = std::fabs(s[kk].x[i]);
                                if (v > mx) mx = v;
                            }
                            double mdx = 0.0;
                            for (int i = 0; i < NX; ++i) {
                                double v = std::fabs(riccati_ws_.dx[kk][i]);
                                if (v > mdx) mdx = v;
                            }
                            double smin = 1e100, dsmin = 0.0, dsmax = 0.0;
                            for (int jj = 0; jj < NC; ++jj) {
                                if (s[kk].s[jj] < smin) smin = s[kk].s[jj];
                                double dsv = ds_[kk][jj];
                                if (dsv < dsmin) dsmin = dsv;
                                if (dsv > dsmax) dsmax = dsv;
                            }
                            printf("  [per-stage] %2d: %.3e  %.3e  %.3e  %+.3e %+.3e\n",
                                   kk, mx, mdx, smin, dsmin, dsmax);
                        }
                    }
                }
            }
            fflush(stdout);
    }

    // ═════════════════════════════════════════════════════════════════════
    //  Re-project (s, λ) onto the central path.
    //
    //  mu_frac controls the target complementarity level:
    //    mu_frac = 1.0       → full re-projection: s·λ = μ  (stall recovery)
    //    mu_frac = m_safe    → gentle safeguard: s·λ ≥ m_safe·μ
    //
    //  For each (s_j, λ_j):
    //    floor_s = max(−d_j, mu_frac·μ)    ensures feasibility & positivity
    //    s_j     ← max(s_j, floor_s)       only pushes up if below floor
    //    λ_j     ← mu_frac·μ / s_j         sets complementarity
    // ═════════════════════════════════════════════════════════════════════

    void sz_complement(double mu_frac = 1.0) {
        const int N = HORIZON;
        Stage* s = prob_->stages;
        const double threshold = mu_frac * mu_;
        // Linear floor (not sqrt) — sqrt of small μ gives large floor that clamps slacks
        // e.g. μ=0.08 → sqrt=0.28 (too large) vs linear=0.008 (reasonable)
        const double floor_s   = threshold;

        for (int k = 0; k <= N; ++k) {
            for (int j = 0; j < NC; ++j) {
                s[k].s[j]      = std::max({-s[k].d[j], s[k].s[j], floor_s});
                s[k].lambda[j] = threshold / s[k].s[j];
            }
        }
    }

    // ═════════════════════════════════════════════════════════════════════
    //  Second-Order Correction (SOC) helpers
    // ═════════════════════════════════════════════════════════════════════

    /// Compute θ = total constraint violation (ℓ₁) at current base iterate.
    /// Uses stages_scaled_ flag to handle both scaled and physical c.
    double compute_theta() const {
        const int N = HORIZON;
        Stage* s = prob_->stages;
        double theta_dyn = 0.0;
        double theta_ineq = 0.0;
        if (stages_scaled_) {
            // c was scaled by transform_qp (c *= Lx_{k+1}), so unscale for physical theta.
            for (int k = 0; k < N; ++k)
                for (int i = 0; i < NX; ++i)
                    theta_dyn += std::fabs(s[k].c[i] * prec_.inv_Lx(k + 1)[i]);
        } else {
            // c is physical (from evaluate_model without transform_qp)
            for (int k = 0; k < N; ++k)
                for (int i = 0; i < NX; ++i)
                    theta_dyn += std::fabs(s[k].c[i]);
        }
        if (prob_->constraints) {
            for (int k = 0; k <= N; ++k)
                for (int j = 0; j < NC; ++j)
                    theta_ineq += std::fabs(s[k].d[j] + s[k].s[j]);
        }
        return theta_dyn + theta_ineq;
    }



    // (fallback: sz_complement used when line search exhausts)

    // Fraction-to-boundary limits (primal ap, dual ad) — used by SOC
    void compute_ftb_limits(double& ap, double& ad) const {
        ap = 1.0; ad = 1.0;
        const Stage* s = prob_->stages;
        for (int k = 0; k <= HORIZON; ++k) {
            for (int j = 0; j < NC; ++j) {
                if (ds_[k][j] < -1e-16) {
                    double a = -params_.tau * s[k].s[j] / ds_[k][j];
                    if (a < ap) ap = a;
                }
                if (dlambda_[k][j] < -1e-16) {
                    double a = -params_.tau * s[k].lambda[j] / dlambda_[k][j];
                    if (a < ad) ad = a;
                }
            }
        }
    }

    // ═════════════════════════════════════════════════════════════════════
    //  Trial-point evaluator for filter line search
    //  Wraps solver internals for the FilterLineSearch interface.
    // ═════════════════════════════════════════════════════════════════════

    class IPMTrialEvaluator
        : public TrialPointEvaluator<NX, NU, HORIZON> {
    public:
        using Solver = PaperIPMSolver<NX, NU, NC, HORIZON>;

        void bind(Solver* s) { solver_ = s; }

        bool evaluate(double alpha, double& out_theta, double& out_phi) override {
            // READ-ONLY: compute trial (θ, φ) without modifying solver state.
            const auto* sv = solver_;
            const int N = HORIZON;
            const auto* s = sv->prob_->stages;

            out_theta = 0.0;
            out_phi = 0.0;

            // Dynamics defects at trial point (using temporary values)
            for (int k = 0; k < N; ++k) {
                Vec<NX> xk_t   = s[k].x;
                Vec<NU> uk_t   = s[k].u;
                Vec<NX> xkp1_t = s[k+1].x;
                for (int i = 0; i < NX; ++i)   xk_t[i]   += alpha * sv->riccati_ws_.dx[k][i];
                for (int i = 0; i < NU; ++i)   uk_t[i]   += alpha * sv->riccati_ws_.du[k][i];
                for (int i = 0; i < NX; ++i)   xkp1_t[i] += alpha * sv->riccati_ws_.dx[k+1][i];

                out_phi += sv->prob_->cost->stage_cost(xk_t, uk_t, k);
                Vec<NX> fk;
                sv->prob_->dynamics->discrete_step(xk_t, uk_t, sv->prob_->dt, fk, k);
                for (int i = 0; i < NX; ++i) {
                    double def = std::fabs(fk[i] - xkp1_t[i]);
                    out_theta += def;  // Use SUM to match compute_theta()
                }
            }
            {
                Vec<NX> xN_t = s[N].x;
                for (int i = 0; i < NX; ++i) xN_t[i] += alpha * sv->riccati_ws_.dx[N][i];
                out_phi += sv->prob_->cost->terminal_cost(xN_t);
            }

            // Constraints at trial point
            if (sv->prob_->constraints) {
                for (int k = 0; k <= N; ++k) {
                    Vec<NC> d_t;
                    if (k < N) {
                        Vec<NX> xk_t = s[k].x;  Vec<NU> uk_t = s[k].u;
                        for (int i = 0; i < NX; ++i) xk_t[i] += alpha * sv->riccati_ws_.dx[k][i];
                        for (int i = 0; i < NU; ++i) uk_t[i] += alpha * sv->riccati_ws_.du[k][i];
                        sv->prob_->constraints->evaluate(xk_t, uk_t, k, d_t);
                    } else {
                        Vec<NX> xN_t = s[k].x;
                        for (int i = 0; i < NX; ++i) xN_t[i] += alpha * sv->riccati_ws_.dx[k][i];
                        sv->prob_->constraints->evaluate_terminal(xN_t, d_t);
                    }
                    for (int j = 0; j < NC; ++j)
                        out_theta += std::fabs(d_t[j] + s[k].s[j] + alpha * sv->ds_[k][j]);
                }
            } else {
                for (int k = 0; k <= N; ++k)
                    for (int j = 0; j < NC; ++j)
                        out_theta += std::fabs(s[k].s[j] + alpha * sv->ds_[k][j]);
            }

            // Barrier-adjusted objective
            double barrier_term = 0.0;
            const double mu = sv->mu_;
            for (int k = 0; k <= N; ++k)
                for (int j = 0; j < NC; ++j) {
                    double s_t = s[k].s[j] + alpha * sv->ds_[k][j];
                    if (s_t > 1e-20)
                        barrier_term += std::log(s_t);
                }
            out_phi -= mu * barrier_term;

            return std::isfinite(out_theta) && std::isfinite(out_phi);
        }

        double current_theta() const override {
            return solver_->compute_theta();
        }

        double current_phi() const override {
            double phi = solver_->compute_objective();
            // Barrier-adjusted objective
            const double mu = solver_->mu_;
            const int N = HORIZON;
            for (int k = 0; k <= N; ++k)
                for (int j = 0; j < NC; ++j)
                    if (solver_->prob_->stages[k].s[j] > 1e-20)
                        phi -= mu * std::log(solver_->prob_->stages[k].s[j]);
            return phi;
        }

        double compute_Dphi() const override {
            // Compute −dz^T·H·dz from the Riccati Hessian.
            // The Riccati step uses the Lagrangian Hessian (λ/s weights),
            // so −dz^T·H·dz is guaranteed ≤ 0 for convex problems.
            const auto* sv = solver_;
            const int N = HORIZON;
            double dzHdz = 0.0;
            for (int k = 0; k <= N; ++k) {
                for (int r = 0; r < NX; ++r)
                    for (int c = 0; c < NX; ++c)
                        dzHdz += sv->riccati_ws_.dx[k][r]
                                 * sv->riccati_stages_[k].Qxx(r, c)
                                 * sv->riccati_ws_.dx[k][c];
                if (k < N) {
                    for (int r = 0; r < NU; ++r)
                        for (int c = 0; c < NU; ++c)
                            dzHdz += sv->riccati_ws_.du[k][r]
                                     * sv->riccati_stages_[k].Quu(r, c)
                                     * sv->riccati_ws_.du[k][c];
                    for (int r = 0; r < NU; ++r)
                        for (int c = 0; c < NX; ++c)
                            dzHdz += 2.0 * sv->riccati_ws_.du[k][r]
                                     * sv->riccati_stages_[k].Qux(r, c)
                                     * sv->riccati_ws_.dx[k][c];
                }
            }
            double Dphi = -dzHdz;
            if (!std::isfinite(Dphi) || Dphi >= 0.0) Dphi = -1e-6;
            return Dphi;
        }

        bool compute_soc(double alpha, double& out_theta, double& out_phi) override {
            // SOC modifies the search direction (dx, du, ds, dlambda) to correct
            // for nonlinear dynamics defects.  If SOC fails, original direction
            // is restored so backtracking can continue with the original step.
            auto* sv = solver_;
            const int N = HORIZON;
            auto* s = sv->prob_->stages;

            // ── Save original search direction ──────────────────────
            for (int k = 0; k <= N; ++k) {
                sv->soc_save_dx_[k]   = sv->riccati_ws_.dx[k];
                sv->soc_save_ds_[k]   = sv->ds_[k];
                sv->soc_save_dlam_[k] = sv->dlambda_[k];
                if (k < N) sv->soc_save_du_[k] = sv->riccati_ws_.du[k];
            }

            // ── Compute trial dynamics defect at (base + α·dir) ─────
            for (int k = 0; k < N; ++k) {
                Vec<NX> xk_t   = s[k].x;
                Vec<NU> uk_t   = s[k].u;
                Vec<NX> xkp1_t = s[k+1].x;
                for (int i = 0; i < NX; ++i) xk_t[i]   += alpha * sv->soc_save_dx_[k][i];
                for (int i = 0; i < NU; ++i) uk_t[i]   += alpha * sv->soc_save_du_[k][i];
                for (int i = 0; i < NX; ++i) xkp1_t[i] += alpha * sv->soc_save_dx_[k+1][i];

                Vec<NX> fk;
                sv->prob_->dynamics->discrete_step(xk_t, uk_t, sv->prob_->dt, fk, k);
                for (int i = 0; i < NX; ++i)
                    sv->soc_save_c_[k][i] = fk[i] - xkp1_t[i];
            }

            // Replace dynamics defect with trial defect (scaled for Riccati)
            for (int k = 0; k < N; ++k) {
                sv->riccati_stages_[k].c = sv->soc_save_c_[k];
                if (sv->params_.enable_preconditioner) {
                    const auto& lx_next = sv->prec_.Lx(k + 1);
                    for (int i = 0; i < NX; ++i)
                        sv->riccati_stages_[k].c[i] *= lx_next[i];
                }
            }

            Status st = Ricc::backward_rhs(sv->riccati_stages_, sv->riccati_ws_);
            if (st != Status::SUCCESS) {
                // Restore original direction
                for (int k = 0; k <= N; ++k) {
                    sv->riccati_ws_.dx[k] = sv->soc_save_dx_[k];
                    sv->ds_[k] = sv->soc_save_ds_[k];  sv->dlambda_[k] = sv->soc_save_dlam_[k];
                    if (k < N) sv->riccati_ws_.du[k] = sv->soc_save_du_[k];
                }
                return false;
            }

            Vec<NX> dx0_soc;
            for (int i = 0; i < NX; ++i)
                dx0_soc[i] = sv->prob_->x0[i] - sv->prob_->stages[0].x[i];
            if (sv->params_.enable_preconditioner) {
                sv->prec_.scale_dx0(dx0_soc);
            }
            st = Ricc::forward(sv->riccati_stages_, sv->riccati_ws_, dx0_soc);
            if (st != Status::SUCCESS) {
                for (int k = 0; k <= N; ++k) {
                    sv->riccati_ws_.dx[k] = sv->soc_save_dx_[k];
                    sv->ds_[k] = sv->soc_save_ds_[k];  sv->dlambda_[k] = sv->soc_save_dlam_[k];
                    if (k < N) sv->riccati_ws_.du[k] = sv->soc_save_du_[k];
                }
                return false;
            }

            // Recover physical SOC step from scaled solution.
            // ds/dλ computed BEFORE primal recovery (scaled Cx·dx̂ = physical ds).
            sv->recover_inequality_steps(sv->sigma_);
            if (sv->params_.enable_preconditioner) {
                sv->prec_.recover_primal_step(sv->riccati_ws_);
                sv->prec_.recover_dual_step(sv->riccati_ws_);
            }
            double ap_soc, ad_soc_unused;
            sv->compute_ftb_limits(ap_soc, ad_soc_unused);
            double alpha_soc = std::min(alpha, ap_soc);

            // Evaluate SOC trial point (read-only)
            evaluate(alpha_soc, out_theta, out_phi);

            // If SOC didn't improve, restore original direction for backtracking
            // (If accepted, the modified direction is used in apply_primal_dual_step)
            if (out_theta >= sv->compute_theta() * 1.5) {
                // SOC didn't help enough — restore
                for (int k = 0; k <= N; ++k) {
                    sv->riccati_ws_.dx[k] = sv->soc_save_dx_[k];
                    sv->ds_[k] = sv->soc_save_ds_[k];  sv->dlambda_[k] = sv->soc_save_dlam_[k];
                    if (k < N) sv->riccati_ws_.du[k] = sv->soc_save_du_[k];
                }
            }
            return true;
        }

    private:
        Solver* solver_ = nullptr;
    };

    // ═════════════════════════════════════════════════════════════════════
    //  Data members
    // ═════════════════════════════════════════════════════════════════════

    Prob*        prob_ = nullptr;
    PaperIPMParams params_;
    double       mu_ = 1.0;
    BarrierUpdateStrategy barrier_strategy_;

    // Workspace for KKT building
    Mat<NX, NX>  qxx_work_[HORIZON + 1];
    Mat<NU, NU>  quu_work_[HORIZON + 1];
    Mat<NU, NX>  qux_work_[HORIZON + 1];
    Vec<NX>      qx_work_[HORIZON + 1];
    Vec<NU>      qu_work_[HORIZON + 1];

    // Preconditioner
    Prec prec_;

    // Riccati stages and workspace
    StageData<NX, NU, NC> riccati_stages_[HORIZON + 1];
    WS            riccati_ws_;
    double        reg_used_ = 1e-6;

    // Affine (predictor) steps for centering correction
    Vec<NC> ds_aff_[HORIZON + 1];
    Vec<NC> dlambda_aff_[HORIZON + 1];

    // Current (corrector) steps
    Vec<NC> ds_[HORIZON + 1];
    Vec<NC> dlambda_[HORIZON + 1];

    // KKT residuals
    double primal_inf_      = 0.0;  // dynamics defect + constraint satisfaction
    double dyn_defect_      = 0.0;  // max |f(x,u) - x_{k+1}| (nonlinear dynamics defect)
    double cons_viol_       = 0.0;  // max |g(x,u) + s| (constraint violation)
    double stat_inf_        = 0.0;  // stationarity: ‖∇L(z,λ,ν)‖∞  (NOT dual feasibility!)
    double stat_breakdown_[6] = {};  // [grad_x, grad_u, Cx^T·λ, Cu^T·λ, costate_x, costate_u]
    int    stat_worst_node_  = -1;  // node k where stationarity is worst
    double compl_inf_       = 0.0;  // complementarity: |s_j·λ_j − μ|
    double ineq_viol_       = 0.0;  // inequality: most-negative s_j or λ_j (≥0 = OK)
    bool   has_costates_    = false; // Riccati costates available (false until first solve)
    bool   stages_scaled_   = false; // true after transform_qp, false after evaluate_model

    // Adaptive regularization: condition estimate from previous iteration
    double cond_estimate_   = 1.0;

    // Linear KKT solution quality (computed each iteration)
    KKTLinearResiduals linear_kkt_res_;

    double sigma_            = 0.0;     // Mehrotra centering parameter (current iteration)
    bool   cross_term_accepted_ = true;  // cross-term gate result (fed to barrier FSM)
    double alpha_lambda_     = 1.0;     // FTB step for inequality duals λ (independent of line search)

    // Affine predictor FTB limits (for FTB-bottleneck detection)
    double aff_alpha_p_      = 1.0;     // affine primal FTB step
    double aff_alpha_d_      = 1.0;     // affine dual FTB step
    double last_alpha_p_     = 1.0;     // previous corrector primal FTB step

    // Filter line search
    FilterLineSearch<NX, NU, HORIZON> filter_ls_;
    IPMTrialEvaluator evaluator_;

    // SOC save buffers (persistent to avoid stack overflow on embedded targets)
    Vec<NX> soc_save_dx_[HORIZON + 1];
    Vec<NU> soc_save_du_[HORIZON + 1];
    Vec<NC> soc_save_ds_[HORIZON + 1];
    Vec<NC> soc_save_dlam_[HORIZON + 1];
    Vec<NX> soc_save_c_[HORIZON];

    // Nonlinear KKT iterative refinement workspace (Shamanskii chord method)
    StageData<NX, NU, NC> trial_stages_[HORIZON + 1];  // trial point evaluation
    Vec<NX> temp_corr_dx_[HORIZON + 1];                 // correction buffer (x)
    Vec<NU> temp_corr_du_[HORIZON];                     // correction buffer (u)

    // Debug: scaled corrector step saved BEFORE recovery (for invariance testing)
    Vec<NX> debug_scaled_dx_[HORIZON + 1];
    Vec<NU> debug_scaled_du_[HORIZON];
    Vec<NX> debug_scaled_p_[HORIZON + 1];
    // Debug: physical corrector step saved AFTER recovery
    Vec<NX> debug_phys_dx_[HORIZON + 1];
    Vec<NU> debug_phys_du_[HORIZON];
    Vec<NX> debug_phys_p_[HORIZON + 1];
    // Debug: pristine Riccati stages right after KKT build (before SOC/LS)
    StageData<NX, NU, NC> debug_pristine_stages_[HORIZON + 1];
    // Debug: sigma and mu at corrector step
    double debug_sigma_ = 0.0;
    double debug_mu_ = 0.0;
    double debug_primal_inf_ = 0.0;
    double debug_compl_inf_ = 0.0;
    // Debug: Riccati internals (P[N], S_fact[0], d[0], K[0])
    SymMat<NX> debug_P_term_;
    SymMat<NU> debug_S_fact0_;
    Vec<NU> debug_d0_;
    Mat<NU, NX> debug_K0_;

public:
    const Vec<NX>* debug_scaled_dx() const { return debug_scaled_dx_; }
    const Vec<NU>* debug_scaled_du() const { return debug_scaled_du_; }
    const Vec<NX>* debug_scaled_p()  const { return debug_scaled_p_; }
    const Vec<NX>* debug_phys_dx() const { return debug_phys_dx_; }
    const Vec<NU>* debug_phys_du() const { return debug_phys_du_; }
    const Vec<NX>* debug_phys_p()  const { return debug_phys_p_; }
    const Stage* debug_riccati_stages() const { return riccati_stages_; }
    const Stage* debug_pristine_stages() const { return debug_pristine_stages_; }
    double debug_sigma() const { return debug_sigma_; }
    double debug_mu() const { return debug_mu_; }
    double debug_primal_inf() const { return debug_primal_inf_; }
    double debug_compl_inf() const { return debug_compl_inf_; }
    const SymMat<NX>& debug_P_term() const { return debug_P_term_; }
    const SymMat<NU>& debug_S_fact0() const { return debug_S_fact0_; }
    const Vec<NU>& debug_d0() const { return debug_d0_; }
    const Mat<NU, NX>& debug_K0() const { return debug_K0_; }
    // Preconditioner scaling factors (for invariance testing)
    const Vec<NX>* debug_prec_Lx() const { return prec_.debug_Lx(); }
    const Vec<NU>* debug_prec_Lu() const { return prec_.debug_Lu(); }
    const Vec<NX>* debug_prec_inv_Lx() const { return prec_.debug_inv_Lx(); }
    const Vec<NU>* debug_prec_inv_Lu() const { return prec_.debug_inv_Lu(); }
    bool debug_precond_enabled() const { return params_.enable_preconditioner; }
};

} // namespace nmpc
