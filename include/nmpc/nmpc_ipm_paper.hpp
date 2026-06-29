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

    // === Globalization: Slack & Barrier Policy ===
    double s_min_init      = 0.01;    // minimum initial slack (Phase 1)
    double delta_slack     = 0.01;    // margin above constraint violation (Phase 1)
    double epsilon_g       = 0.01;    // feasibility threshold: Phase A→B transition (Phase 2)
    double c_floor         = 0.1;     // adaptive slack floor coefficient (Phase 3)
    double c_restoration   = 0.2;     // restoration slack enlargement coefficient (Phase 6)

    // === Nonlinear KKT Iterative Refinement (Shamanskii chord) ===
    bool   enable_refinement  = false;   // enable direction refinement before line search
    int    max_refine_iters   = 5;       // max chord passes
    double refine_tol         = 1e-6;    // nonlinear residual tolerance
    double refine_diverge_fac = 1.5;     // divergence guard: stop if ||r|| > fac * prev

    // Output
    int    verbosity     = 0;

    // Debug: freeze μ after N iterations (-1 = disabled)
    int    freeze_mu_after = -1;

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
    double max_riccati_u_scaled; // Same as max_riccati_u_res but in SCALED space (before inv_Lu unscaling)

    // Diagnostic: reconstruction error
    double max_qu_reconstruction_err; // max |qu_tilde - riccati_stages_.qu|
    int worst_qu_stage;               // stage with worst qu reconstruction error
    double max_forward_dyn_err;       // max |A·dx + B·du + c - dx_{k+1}| (forward pass consistency)

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
//  Per-iteration diagnostic snapshot for IPM convergence analysis
//  Designed to compare ContactIPM vs IPOPT behavior on hanging chain.
// ─────────────────────────────────────────────────────────────────────────────
struct IterDiag {
    // (A) Primal residual decomposition
    double r_dyn_inf = 0;       // ||x_{k+1} - f(x_k,u_k)||_inf  (nonlinear dyn defect)
    double r_dyn_2 = 0;         // ||x_{k+1} - f(x_k,u_k)||_2    (sum of squares)
    double r_ineq_inf = 0;      // max_i g_i(x,u)  (max constraint violation)
    double r_ineq_active_2 = 0; // ||g_i for active set||_2
    double r_ratio = 0;         // r_dyn_inf / (r_ineq_inf + 1e-12)
    int    n_active = 0;        // number of active inequality constraints

    // (B) Dual / stationarity breakdown
    double grad_L_x = 0;        // full stationarity norm (already stat_inf_)
    double grad_cost_x = 0;     // ||qx||_inf  (cost gradient)
    double grad_cost_u = 0;     // ||qu||_inf
    double dyn_dual_x = 0;      // ||A^T lambda_dyn||_inf  (costate contribution)
    double dyn_dual_u = 0;      // ||B^T lambda_dyn||_inf
    double ineq_dual_x = 0;     // ||Cx^T lambda_ineq||_inf
    double ineq_dual_u = 0;     // ||Cu^T lambda_ineq||_inf
    double stat_dom_ratio = 0;  // dyn_term / ineq_term

    // (C) Barrier coupling
    double mu = 0;
    double compl_min = 0;       // min |s_j * lambda_j|
    double compl_max = 0;       // max |s_j * lambda_j|
    double barrier_obj = 0;     // mu * sum log(-g_i) ≈ -mu * sum log(s_j)
    double slack_min = 0;       // min(s_j)
    double slack_max = 0;       // max(s_j)

    // (D) KKT system diagnostics
    double norm_H_F = 0;        // ||H_cost + H_barrier||_F (max over stages)
    double norm_Adyn_F = 0;     // ||A_dyn||_F (max over stages)
    double norm_Aineq_F = 0;    // ||C||_F (max over stages)
    double dx_inf = 0;          // ||dx||_inf
    double du_inf = 0;          // ||du||_inf
    double dp_inf = 0;          // ||dp||_inf  (costate step)
    double ds_inf = 0;          // ||ds||_inf  (slack step)
    double dlam_inf = 0;        // ||dlambda||_inf

    // (E) Step coupling: directional sensitivities
    double s_dyn_inf = 0;       // ||A*dx + B*du + c||_inf  (linearized dyn residual after step)
    double s_ineq_inf = 0;      // ||Cx*dx + Cu*du + g + s||_inf  (linearized ineq after step)
    int    sign_corr = 0;       // +1 if dx improves both, -1 if competing, 0 neutral

    // (F) Line search
    double alpha_p = 0;
    double alpha_d = 0;
    double theta_dyn = 0;       // dynamics contribution to theta
    double theta_ineq = 0;      // inequality contribution to theta
    bool   ls_rejected = false;
    int    ls_iters = 0;

    // (G) Scaling diagnostics
    double Lx_max = 0;          // max diagonal scaling for x
    double Lx_min = 0;          // min diagonal scaling for x
    double Lu_max = 0;          // max diagonal scaling for u
    double Lu_min = 0;          // min diagonal scaling for u
    double scale_ratio = 0;     // max(Lx,Lu) / min(Lx,Lu)

    // IPOPT-style compact line
    void print_compact(int iter) const {
        printf("%4d | %.2e | %.2e | %.2e | %.2e | %.3f | %.2e | %s\n",
               iter, mu, r_dyn_inf, r_ineq_inf, grad_L_x, alpha_p, compl_max,
               ls_rejected ? "REJ" : "acc");
    }

    // CSV header
    static void print_csv_header(FILE* f) {
        fprintf(f, "iter,mu,r_dyn_inf,r_dyn_2,r_ineq_inf,r_ineq_act2,r_ratio,n_active,"
                   "grad_Lx,grad_cost_x,grad_cost_u,dyn_dual_x,dyn_dual_u,ineq_dual_x,ineq_dual_u,stat_dom,"
                   "compl_min,compl_max,barrier_obj,slack_min,slack_max,"
                   "norm_H,norm_Adyn,norm_Aineq,dx_inf,du_inf,dp_inf,ds_inf,dlam_inf,"
                   "s_dyn_inf,s_ineq_inf,sign_corr,"
                   "alpha_p,alpha_d,theta_dyn,theta_ineq,ls_rej,ls_iters,"
                   "Lx_max,Lx_min,Lu_max,Lu_min,scale_ratio\n");
    }

    // CSV row
    void print_csv_row(FILE* f, int iter) const {
        fprintf(f, "%d,%.6e,%.6e,%.6e,%.6e,%.6e,%.6e,%d,"
                    "%.6e,%.6e,%.6e,%.6e,%.6e,%.6e,%.6e,%.6e,"
                    "%.6e,%.6e,%.6e,%.6e,%.6e,"
                    "%.6e,%.6e,%.6e,%.6e,%.6e,%.6e,%.6e,%.6e,"
                    "%.6e,%.6e,%d,"
                    "%.6e,%.6e,%.6e,%.6e,%d,%d,"
                    "%.6e,%.6e,%.6e,%.6e,%.6e\n",
                iter, mu, r_dyn_inf, r_dyn_2, r_ineq_inf, r_ineq_active_2, r_ratio, n_active,
                grad_L_x, grad_cost_x, grad_cost_u, dyn_dual_x, dyn_dual_u, ineq_dual_x, ineq_dual_u, stat_dom_ratio,
                compl_min, compl_max, barrier_obj, slack_min, slack_max,
                norm_H_F, norm_Adyn_F, norm_Aineq_F, dx_inf, du_inf, dp_inf, ds_inf, dlam_inf,
                s_dyn_inf, s_ineq_inf, sign_corr,
                alpha_p, alpha_d, theta_dyn, theta_ineq, ls_rejected ? 1 : 0, ls_iters,
                Lx_max, Lx_min, Lu_max, Lu_min, scale_ratio);
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

        mu_ = params_.mu_init;

        initialize_from_problem();
        evaluate_model();          // refresh after init modified stages

        // Compute preconditioner scaling ONCE per MPC solve (outside Newton loop)
        if (params_.enable_preconditioner) {
            prec_.compute(prob_->stages);
            // Note: transform_qp() is called inside the loop before compute_kkt_residuals
            // so that convergence checks use scaled residuals.
        }

        // mu_ already set by adaptive μ₀ above (or params_.mu_init if no constraints)
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
            bup.epsilon_g   = params_.epsilon_g;
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
        int    ls_fail_count = 0;  // consecutive filter-exhaustion count
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
                for (int k = 0; k <= HORIZON; ++k) {
                    // Slack/lambda FTB: only when constraints exist
                    if (prob_->constraints) {
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
                    }
                    // Bound FTB for affine predictor
                    if (prob_->n_bound_u > 0 && k < HORIZON) {
                        for (int i = 0; i < NU; ++i) {
                            double du_i = riccati_ws_.du[k][i];
                            double dL = prob_->stages[k].u[i] - prob_->u_lb[i];
                            double dU = prob_->u_ub[i] - prob_->stages[k].u[i];
                            if (dL > 1e-14 && du_i < -1e-16) {
                                double a = -params_.tau * dL / du_i;
                                if (a < ap_aff) ap_aff = a;
                            }
                            if (dU > 1e-14 && du_i > 1e-16) {
                                double a = params_.tau * dU / du_i;
                                if (a < ap_aff) ap_aff = a;
                            }
                        }
                    }
                    if (prob_->n_bound_x > 0) {
                        for (int i = 0; i < NX; ++i) {
                            double dx_i = riccati_ws_.dx[k][i];
                            double dL = prob_->stages[k].x[i] - prob_->x_lb[i];
                            double dU = prob_->x_ub[i] - prob_->stages[k].x[i];
                            if (dL > 1e-14 && dx_i < -1e-16) {
                                double a = -params_.tau * dL / dx_i;
                                if (a < ap_aff) ap_aff = a;
                            }
                            if (dU > 1e-14 && dx_i > 1e-16) {
                                double a = params_.tau * dU / dx_i;
                                if (a < ap_aff) ap_aff = a;
                            }
                        }
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
            // ── Constraint-normal correction (Fix C) ──
            // When Newton du worsens a violated constraint (Cu·du > 0),
            // replace the constraint-normal component with a repair step.
            // Applied in scaled space, BEFORE ds/dλ computation.
            // apply_constraint_normal_correction();  // DISABLED: makes prim worse (0.58→24.8)

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

            // ── Stationarity gradient decomposition at k=0 ────────
            // Decompose qu_tilde[0] into cost + constraint-barrier + dynamics
            // to understand WHY Cu·du is small for the bottleneck constraint.
            if (params_.verbosity >= 2) {
                const int kk = 0;
                const auto& stg = prob_->stages[kk];
                const auto& rs = riccati_stages_[kk];
                const auto& dx_k = riccati_ws_.dx[kk];
                const auto& du_k = riccati_ws_.du[kk];
                const auto& dx_k1 = riccati_ws_.dx[kk + 1];
                Vec<NX> nu_next;
                for (int i = 0; i < NX; ++i) {
                    nu_next[i] = riccati_ws_.p[kk + 1][i];
                    for (int j = 0; j < NX; ++j)
                        nu_next[i] += riccati_ws_.P[kk + 1](i, j) * dx_k1[j];
                }
                // qu_tilde = cost_grad + Cu^T*net
                // Net constraint-barrier contribution per constraint:
                //   Cu^T[j] * net_j  where net_j = (smu + lj*(gj+sj))/sj - lj
                const double smu = sigma_ * mu_;
                Vec<NU> cu_net_total;  cu_net_total.zero();
                double cu_net_j4 = 0.0;
                for (int j = 0; j < NC; ++j) {
                    double sj = stg.s[j];
                    if (sj < 1e-14) continue;
                    double lj = stg.lambda[j];
                    double gj_sj = stg.d[j] + sj;  // g+s
                    double net_j = (smu + lj * gj_sj) / sj - lj;
                    double cu_net_j = 0.0;
                    for (int i = 0; i < NU; ++i) {
                        double contrib = stg.Cu(j, i) * net_j;
                        cu_net_total[i] += contrib;
                        cu_net_j += contrib * contrib;
                    }
                    cu_net_j = std::sqrt(cu_net_j);
                    if (j == 4) {
                        cu_net_j4 = cu_net_j;
                    }
                }
                double cost_grad_norm = 0.0;
                for (int i = 0; i < NU; ++i)
                    cost_grad_norm += stg.qu[i] * stg.qu[i];
                cost_grad_norm = std::sqrt(cost_grad_norm);

                double cu_net_total_norm = 0.0;
                for (int i = 0; i < NU; ++i)
                    cu_net_total_norm += cu_net_total[i] * cu_net_total[i];
                cu_net_total_norm = std::sqrt(cu_net_total_norm);

                // B^T*nu (dynamics costate coupling)
                Vec<NU> bt_nu;
                for (int i = 0; i < NU; ++i) {
                    bt_nu[i] = 0.0;
                    for (int j = 0; j < NX; ++j)
                        bt_nu[i] += rs.B(j, i) * nu_next[j];
                }
                double bt_nu_norm = 0.0;
                for (int i = 0; i < NU; ++i)
                    bt_nu_norm += bt_nu[i] * bt_nu[i];
                bt_nu_norm = std::sqrt(bt_nu_norm);

                // Cu*du for bottleneck constraint j=4
                double Cu_du_j4 = 0.0;
                for (int i = 0; i < NU; ++i)
                    Cu_du_j4 += stg.Cu(4, i) * du_k[i];

                // du from Riccati formula: du = -K*dx - d
                double du_Kdx_norm = 0.0, du_d_norm = 0.0;
                Vec<NU> Kdx;
                for (int i = 0; i < NU; ++i) {
                    Kdx[i] = 0.0;
                    for (int j = 0; j < NX; ++j)
                        Kdx[i] += riccati_ws_.K[kk](i, j) * dx_k[j];
                    du_Kdx_norm += Kdx[i] * Kdx[i];
                    du_d_norm += riccati_ws_.d[kk][i] * riccati_ws_.d[kk][i];
                }
                du_Kdx_norm = std::sqrt(du_Kdx_norm);
                du_d_norm = std::sqrt(du_d_norm);

                // ── Schur complement S decomposition ────────
                // S = Quu + Cu^T*(λ/s)*Cu + B^T*P[1]*B
                // Compare magnitudes to see if barrier stiffness dominates.
                // Quu: cost Hessian (WITHOUT barrier, from prob_->stages)
                double Quu_F = 0.0;
                for (int i = 0; i < NU; ++i)
                    for (int jj = 0; jj < NU; ++jj)
                        Quu_F += stg.Quu(i, jj) * stg.Quu(i, jj);
                Quu_F = std::sqrt(Quu_F);

                // Cu^T*(λ/s)*Cu: barrier Hessian contribution
                // Compute full NU×NU matrix and its norm
                double barr_H[NU][NU] = {};
                double barr_F = 0.0;
                for (int j = 0; j < NC; ++j) {
                    double sj = stg.s[j];
                    if (sj < 1e-14) continue;
                    double ratio = stg.lambda[j] / sj;
                    for (int r = 0; r < NU; ++r)
                        for (int c = 0; c < NU; ++c)
                            barr_H[r][c] += ratio * stg.Cu(j, r) * stg.Cu(j, c);
                }
                for (int i = 0; i < NU; ++i)
                    for (int jj = 0; jj < NU; ++jj)
                        barr_F += barr_H[i][jj] * barr_H[i][jj];
                barr_F = std::sqrt(barr_F);

                // B^T*P[1]*B: dynamics cost-to-go contribution
                // BtPB[i][j] = sum_mn B[m][i]*P[1][m][n]*B[n][j]
                const auto& P1 = riccati_ws_.P[kk + 1];
                double BtPB_F = 0.0;
                for (int i = 0; i < NU; ++i)
                    for (int jj = 0; jj < NU; ++jj) {
                        double val = 0.0;
                        for (int m = 0; m < NX; ++m)
                            for (int n = 0; n < NX; ++n)
                                val += rs.B(m, i) * P1(m, n) * rs.B(n, jj);
                        BtPB_F += val * val;
                    }
                BtPB_F = std::sqrt(BtPB_F);

                // |Cu·du| / |du|: normal projection of step onto constraint j=4
                double du_norm = 0.0;
                for (int i = 0; i < NU; ++i)
                    du_norm += du_k[i] * du_k[i];
                du_norm = std::sqrt(du_norm);
                double Cu_du_abs = std::fabs(Cu_du_j4);
                double normal_ratio = (du_norm > 1e-30) ? Cu_du_abs / du_norm : 0.0;

                // S diagonal for the bottleneck constraint direction
                double S_diag_max = 0.0;
                for (int i = 0; i < NU; ++i) {
                    double d = stg.Quu(i, i) + barr_H[i][i];
                    // Add BtPB diagonal
                    for (int m = 0; m < NX; ++m)
                        for (int n = 0; n < NX; ++n)
                            d += rs.B(m, i) * P1(m, n) * rs.B(n, i);
                    S_diag_max = std::max(S_diag_max, std::fabs(d));
                }

                printf("  [S-dec] k=0: |Quu|=%.3e |Cu^T*L/S*Cu|=%.3e |B^T*P*B|=%.3e"
                       " S_diag_max=%.3e\n",
                       Quu_F, barr_F, BtPB_F, S_diag_max);
                printf("          |du|=%.3e |Cu*du|=%.3e ratio=%.4f"
                       " lambda[4]/s[4]=%.2e\n",
                       du_norm, Cu_du_abs, normal_ratio,
                       stg.lambda[4] / std::max(stg.s[4], 1e-30));

                // ── Directional diagnostics for bottleneck row j=4 ────
                // c = Cu[4,:]: single constraint row
                // n = c/|c|: unit normal to constraint j=4
                // cos_j4 = (c·du)/(|c|·|du|): directional cosine for this row only
                // k_n = n^T S n: effective stiffness along constraint normal
                // k_t = (tr(S) - k_n)/(NU-1): avg tangent stiffness
                // ||S^{-1}c||: compliance along constraint direction
                {
                    // c = Cu[4,:] in scaled space
                    double c_vec[NU];
                    double c_norm_sq = 0.0;
                    for (int i = 0; i < NU; ++i) {
                        c_vec[i] = stg.Cu(4, i);
                        c_norm_sq += c_vec[i] * c_vec[i];
                    }
                    double c_norm = std::sqrt(c_norm_sq);

                    // cos_j4 = (c·du) / (|c|·|du|)
                    double c_dot_du = Cu_du_j4;  // already computed above
                    double cos_j4 = (c_norm > 1e-30 && du_norm > 1e-30)
                        ? c_dot_du / (c_norm * du_norm) : 0.0;

                    // S = Quu + barr_H + BtPB  (full NU×NU Schur complement)
                    double S_mat[NU][NU] = {};
                    for (int i = 0; i < NU; ++i)
                        for (int jj = 0; jj < NU; ++jj) {
                            S_mat[i][jj] = stg.Quu(i, jj) + barr_H[i][jj];
                            // Add BtPB[i][jj]
                            for (int m = 0; m < NX; ++m)
                                for (int n = 0; n < NX; ++n)
                                    S_mat[i][jj] += rs.B(m, i) * P1(m, n) * rs.B(n, jj);
                        }

                    // tr(S)
                    double tr_S = 0.0;
                    for (int i = 0; i < NU; ++i)
                        tr_S += S_mat[i][i];

                    // k_n = c^T S c / (c^T c)  [directional stiffness along c]
                    double Sc[NU];
                    for (int i = 0; i < NU; ++i) {
                        Sc[i] = 0.0;
                        for (int jj = 0; jj < NU; ++jj)
                            Sc[i] += S_mat[i][jj] * c_vec[jj];
                    }
                    double ctSc = 0.0;
                    for (int i = 0; i < NU; ++i)
                        ctSc += c_vec[i] * Sc[i];
                    double k_n = (c_norm_sq > 1e-30) ? ctSc / c_norm_sq : 0.0;

                    // k_t_avg = (tr(S) - k_n) / (NU - 1)
                    double k_t_avg = (NU > 1) ? (tr_S - k_n) / (NU - 1) : 0.0;

                    // ||S^{-1} c||: solve S·x = c using stored factorization
                    Vec<NU> Sinv_c;
                    for (int i = 0; i < NU; ++i)
                        Sinv_c[i] = c_vec[i];
                    riccati_ws_.S_fact[kk].ldlt_solve(Sinv_c);
                    double Sinv_c_norm = Sinv_c.norm2();

                    // Also: r = qu_tilde + B^T·nu_next, cos_phi for j=4
                    double r_vec[NU];
                    double r_norm = 0.0;
                    for (int i = 0; i < NU; ++i) {
                        double Bt_nu_i = 0.0;
                        for (int m = 0; m < NX; ++m)
                            Bt_nu_i += rs.B(m, i) * nu_next[m];
                        r_vec[i] = rs.qu[i] + Bt_nu_i;
                        r_norm += r_vec[i] * r_vec[i];
                    }
                    r_norm = std::sqrt(r_norm);
                    double c_dot_r = 0.0;
                    for (int i = 0; i < NU; ++i)
                        c_dot_r += c_vec[i] * r_vec[i];
                    double cos_phi_j4 = (c_norm > 1e-30 && r_norm > 1e-30)
                        ? c_dot_r / (c_norm * r_norm) : 0.0;

                    printf("  [dir-j4] cos_j4=%+.4f cos_phi_j4=%+.4f"
                           " |c|=%.4e |du|=%.3e c·du=%+.4e\n",
                           cos_j4, cos_phi_j4, c_norm, du_norm, c_dot_du);
                    printf("          k_n=%.3e k_t_avg=%.3e k_n/k_t=%.2f"
                           " |S^{-1}c|=%.3e tr(S)=%.3e\n",
                           k_n, k_t_avg,
                           (k_t_avg > 1e-30) ? k_n / k_t_avg : 0.0,
                           Sinv_c_norm, tr_S);

                    // ── RHS normal/tangent decomposition ────
                    // η[4] = (σμ + λ(g+s))/s - λ
                    double s4 = stg.s[4];
                    double l4 = stg.lambda[4];
                    double g4_s4 = stg.d[4] + s4;
                    double eta4 = (s4 > 1e-30) ? (smu + l4 * g4_s4) / s4 - l4 : 0.0;

                    // Normal projection of total RHS: r_n = (c·r)/|c|
                    double r_n_scalar = (c_norm > 1e-30) ? c_dot_r / c_norm : 0.0;
                    double r_n_mag = std::fabs(r_n_scalar);
                    double r_t_mag = std::sqrt(std::max(r_norm * r_norm - r_n_mag * r_n_mag, 0.0));

                    // |Cu^T·η| / |B^T·ν|: is barrier force dominated by dynamics?
                    double ratio_cb = (bt_nu_norm > 1e-30)
                        ? cu_net_total_norm / bt_nu_norm : 0.0;

                    printf("  [rhs-dec] eta4=%+.3e |c|*eta4=%.3e"
                           " |Cu^T*eta|=%.3e |B^T*nu|=%.3e ratio=%.3f\n",
                           eta4, c_norm * std::fabs(eta4),
                           cu_net_total_norm, bt_nu_norm, ratio_cb);
                    printf("            r_n=%+.3e |r_n|=%.3e |r_t|=%.3e"
                           " |r_n|/|r_t|=%.4f lam4=%.3e s4=%.3e\n",
                           r_n_scalar, r_n_mag, r_t_mag,
                           (r_t_mag > 1e-30) ? r_n_mag / r_t_mag : 0.0,
                           l4, s4);

                    // ── Full 4×4 modal analysis of S ────
                    // Eigendecompose S = V diag(λ) V^T via Jacobi iteration
                    // For each mode: λ_i, α_i = v_i^T·r, β_i = α_i/λ_i,
                    //   γ_i = |n^T·v_i| (normal alignment)

                    // Unit normal to constraint j=4
                    double n_hat[NU];
                    for (int i = 0; i < NU; ++i)
                        n_hat[i] = c_vec[i] / c_norm;

                    // Jacobi eigendecomposition of symmetric S_mat
                    double A_jac[NU][NU];
                    double V_jac[NU][NU];
                    for (int i = 0; i < NU; ++i)
                        for (int jj = 0; jj < NU; ++jj) {
                            A_jac[i][jj] = S_mat[i][jj];
                            V_jac[i][jj] = (i == jj) ? 1.0 : 0.0;
                        }
                    for (int sweep = 0; sweep < 50; ++sweep) {
                        double off = 0.0;
                        for (int i = 0; i < NU; ++i)
                            for (int jj = i+1; jj < NU; ++jj)
                                off += A_jac[i][jj] * A_jac[i][jj];
                        if (off < 1e-30) break;
                        for (int pp = 0; pp < NU-1; ++pp)
                            for (int qq = pp+1; qq < NU; ++qq) {
                                double Appq = A_jac[pp][qq];
                                if (std::fabs(Appq) < 1e-15) continue;
                                double diff = A_jac[qq][qq] - A_jac[pp][pp];
                                double tau = diff / (2.0 * Appq);
                                double t_val = ((tau >= 0) ? 1.0 : -1.0)
                                    / (std::fabs(tau) + std::sqrt(1.0 + tau*tau));
                                double cs = 1.0 / std::sqrt(1.0 + t_val*t_val);
                                double sn = t_val * cs;
                                double app = A_jac[pp][pp];
                                double aqq = A_jac[qq][qq];
                                A_jac[pp][pp] = cs*cs*app - 2.0*sn*cs*Appq + sn*sn*aqq;
                                A_jac[qq][qq] = sn*sn*app + 2.0*sn*cs*Appq + cs*cs*aqq;
                                A_jac[pp][qq] = 0.0;
                                A_jac[qq][pp] = 0.0;
                                for (int rr = 0; rr < NU; ++rr) {
                                    if (rr == pp || rr == qq) continue;
                                    double arp = A_jac[rr][pp];
                                    double arq = A_jac[rr][qq];
                                    A_jac[rr][pp] = cs*arp - sn*arq;
                                    A_jac[pp][rr] = A_jac[rr][pp];
                                    A_jac[rr][qq] = sn*arp + cs*arq;
                                    A_jac[qq][rr] = A_jac[rr][qq];
                                }
                                for (int rr = 0; rr < NU; ++rr) {
                                    double vrp = V_jac[rr][pp];
                                    double vrq = V_jac[rr][qq];
                                    V_jac[rr][pp] = cs*vrp - sn*vrq;
                                    V_jac[rr][qq] = sn*vrp + cs*vrq;
                                }
                            }
                    }
                    // Extract eigenvalues and sort ascending
                    double eval[NU];
                    int sort_idx[NU];
                    for (int i = 0; i < NU; ++i) {
                        eval[i] = A_jac[i][i];
                        sort_idx[i] = i;
                    }
                    for (int i = 0; i < NU-1; ++i)
                        for (int jj = i+1; jj < NU; ++jj)
                            if (eval[sort_idx[i]] > eval[sort_idx[jj]])
                                std::swap(sort_idx[i], sort_idx[jj]);
                    double eval_s[NU], evec[NU][NU];
                    for (int i = 0; i < NU; ++i) {
                        eval_s[i] = eval[sort_idx[i]];
                        for (int r = 0; r < NU; ++r)
                            evec[i][r] = V_jac[r][sort_idx[i]];
                    }

                    // Modal table: α_i = v_i^T·r, β_i = α_i/λ_i, γ_i = |n^T·v_i|
                    double alpha[4], beta[4], gamma[4];
                    double d_mode[4][NU];  // d_i = -β_i·v_i
                    for (int ei = 0; ei < 4; ++ei) {
                        alpha[ei] = 0.0;
                        for (int d = 0; d < NU; ++d)
                            alpha[ei] += evec[ei][d] * r_vec[d];
                        beta[ei] = (std::fabs(eval_s[ei]) > 1e-30)
                            ? alpha[ei] / eval_s[ei] : 0.0;
                        gamma[ei] = 0.0;
                        for (int d = 0; d < NU; ++d)
                            gamma[ei] += n_hat[d] * evec[ei][d];
                        gamma[ei] = std::fabs(gamma[ei]);
                        for (int d = 0; d < NU; ++d)
                            d_mode[ei][d] = -beta[ei] * evec[ei][d];
                    }

                    // Normal/tangent decomposition of d and r
                    double d_n = 0.0, d_t_vec[NU] = {};
                    double r_n_v = 0.0, r_t_vec[NU] = {};
                    for (int d = 0; d < NU; ++d) {
                        double dn_comp = 0.0;
                        for (int j = 0; j < NU; ++j) dn_comp += n_hat[j] * du_k[j];
                        d_n = dn_comp;
                        d_t_vec[d] = du_k[d] - d_n * n_hat[d];
                        double rn_comp = 0.0;
                        for (int j = 0; j < NU; ++j) rn_comp += n_hat[j] * r_vec[j];
                        r_n_v = rn_comp;
                        r_t_vec[d] = r_vec[d] - r_n_v * n_hat[d];
                    }
                    double d_n_mag = std::fabs(d_n);
                    double d_t_norm = 0.0;
                    for (int d = 0; d < NU; ++d) d_t_norm += d_t_vec[d]*d_t_vec[d];
                    d_t_norm = std::sqrt(d_t_norm);
                    double r_n_v_mag = std::fabs(r_n_v);
                    double r_t_norm2 = 0.0;
                    for (int d = 0; d < NU; ++d) r_t_norm2 += r_t_vec[d]*r_t_vec[d];
                    r_t_norm2 = std::sqrt(r_t_norm2);

                    // Stage 1-4: Modal table with λ, α, β, γ, eigvec
                    printf("  [modal ] i | %10s %10s %10s %6s | %s\n",
                           "lambda", "alpha", "beta", "gamma", "eigvec");
                    for (int ei = 0; ei < 4; ++ei) {
                        printf("          %d | %10.3e %10.3e %10.3e %6.4f |"
                               " [%+.3f %+.3f %+.3f %+.3f]\n",
                               ei, eval_s[ei], alpha[ei], beta[ei], gamma[ei],
                               evec[ei][0], evec[ei][1], evec[ei][2], evec[ei][3]);
                    }
                    // Stage 5: Normal/tangent energy
                    printf("  [step  ] |d_n|=%.3e |d_T|=%.3e"
                           " |r_n|=%.3e |r_T|=%.3e\n",
                           d_n_mag, d_t_norm, r_n_v_mag, r_t_norm2);
                    // Stage 6: per-mode step contributions
                    printf("  [mode-d] |d_i|=[");
                    for (int ei = 0; ei < 4; ++ei) {
                        double di_norm = 0.0;
                        for (int d = 0; d < NU; ++d)
                            di_norm += d_mode[ei][d]*d_mode[ei][d];
                        di_norm = std::sqrt(di_norm);
                        printf("%.3e%s", di_norm, ei<3?" ":"");
                    }
                    printf("]\n");
                    // Stage 7: dominant mode identity
                    int dom_idx = 0;
                    double max_abs_beta = 0.0;
                    for (int ei = 0; ei < 4; ++ei)
                        if (std::fabs(beta[ei]) > max_abs_beta) {
                            max_abs_beta = std::fabs(beta[ei]);
                            dom_idx = ei;
                        }
                    printf("  [dominant] mode %d: lambda=%.3e gamma=%.4f"
                           " alpha=%+.3e beta=%+.3e\n",
                           dom_idx, eval_s[dom_idx], gamma[dom_idx],
                           alpha[dom_idx], beta[dom_idx]);

                    // Modal forcing decomposition:
                    // α_i = v_i^T·q_u + v_i^T·Cu^T·η + v_i^T·B^T·ν
                    {
                        // 1. q_u vector (cost gradient)
                        double qu_vec[NU];
                        for (int i = 0; i < NU; ++i)
                            qu_vec[i] = rs.qu[i];

                        // 2. Compute η vector and Cu^T·η (barrier force)
                        double eta_full[NC];
                        for (int j = 0; j < NC; ++j) {
                            double sj = stg.s[j];
                            double lj = stg.lambda[j];
                            double gjsj = stg.d[j] + sj;
                            eta_full[j] = (sj > 1e-30)
                                ? (smu + lj * gjsj) / sj - lj : 0.0;
                        }
                        double cut_eta[NU] = {};
                        for (int i = 0; i < NU; ++i)
                            for (int j = 0; j < NC; ++j)
                                cut_eta[i] += stg.Cu(j, i) * eta_full[j];

                        // 3. B^T·ν vector (dynamics coupling)
                        // ν = p[kk+1] + P[kk+1]·dx[kk+1]
                        const auto& Pfwd = riccati_ws_.P[kk + 1];
                        const auto& dxfwd = riccati_ws_.dx[kk + 1];
                        double nu_vec[NX];
                        for (int m = 0; m < NX; ++m) {
                            nu_vec[m] = riccati_ws_.p[kk+1][m];
                            for (int n = 0; n < NX; ++n)
                                nu_vec[m] += Pfwd(m, n) * dxfwd[n];
                        }
                        double bt_nu_vec[NU] = {};
                        for (int i = 0; i < NU; ++i)
                            for (int m = 0; m < NX; ++m)
                                bt_nu_vec[i] += riccati_ws_.BtP(i, m) * nu_vec[m];

                        // Project each onto eigenvectors
                        double a_cost[4], a_barr[4], a_dyn[4];
                        for (int ei = 0; ei < 4; ++ei) {
                            a_cost[ei] = 0; a_barr[ei] = 0; a_dyn[ei] = 0;
                            for (int d = 0; d < NU; ++d) {
                                a_cost[ei] += evec[ei][d] * qu_vec[d];
                                a_barr[ei] += evec[ei][d] * cut_eta[d];
                                a_dyn[ei]  += evec[ei][d] * bt_nu_vec[d];
                            }
                        }
                        printf("  [modal-f]         cost       barr       dyn"
                               "        sum    |alpha\n");
                        for (int ei = 0; ei < 4; ++ei) {
                            double sum3 = a_cost[ei] + a_barr[ei] + a_dyn[ei];
                            printf("            %d:  %+.3e  %+.3e  %+.3e"
                                   "  = %+.3e | %+.3e\n",
                                   ei, a_cost[ei], a_barr[ei], a_dyn[ei],
                                   sum3, alpha[ei]);
                        }
                        // Identify dominant source for mode 0
                        double abs_c = std::fabs(a_cost[0]);
                        double abs_b = std::fabs(a_barr[0]);
                        double abs_d = std::fabs(a_dyn[0]);
                        const char* src = (abs_d >= abs_c && abs_d >= abs_b) ? "DYN"
                                        : (abs_c >= abs_b) ? "COST" : "BARRIER";
                        printf("  [source] mode 0 excitation: %s"
                               " (|cost|=%.3e |barr|=%.3e |dyn|=%.3e)\n",
                               src, abs_c, abs_b, abs_d);

                        // ── Controllability-constraint projection ──
                        // Find N(C) ∩ N(B) at stage kk=0
                        // 1. Build Householder basis for null(c^T)
                        //    c = Cu[4,:], construct Q s.t. Q[:,1:3] span null(c^T)
                        {
                            double c4[NU];
                            double c4_norm_sq = 0.0;
                            for (int i = 0; i < NU; ++i) {
                                c4[i] = stg.Cu(4, i);
                                c4_norm_sq += c4[i] * c4[i];
                            }
                            double c4_norm = std::sqrt(c4_norm_sq);
                            if (c4_norm > 1e-12) {
                                // Householder vector v = c4 ± |c4|*e0
                                double v_hh[NU];
                                for (int i = 0; i < NU; ++i) v_hh[i] = c4[i];
                                double sign_c = (c4[0] >= 0) ? 1.0 : -1.0;
                                v_hh[0] += sign_c * c4_norm;
                                double v_sq = 0.0;
                                for (int i = 0; i < NU; ++i) v_sq += v_hh[i]*v_hh[i];
                                // Q = I - 2 v v^T / v^T v
                                double Q_mat[NU][NU];
                                for (int i = 0; i < NU; ++i)
                                    for (int jj = 0; jj < NU; ++jj) {
                                        Q_mat[i][jj] = (i==jj ? 1.0 : 0.0)
                                            - 2.0*v_hh[i]*v_hh[jj]/v_sq;
                                    }
                                // Z = Q[:,1:3] is NU×3 basis for null(c^T)
                                // 2. Project B onto null(C): B_Z = B·Z (NX×3)
                                double B_Z[NX][3];
                                for (int m = 0; m < NX; ++m)
                                    for (int j = 0; j < 3; ++j) {
                                        B_Z[m][j] = 0.0;
                                        for (int i = 0; i < NU; ++i)
                                            B_Z[m][j] += rs.B(m, i) * Q_mat[i][j+1];
                                    }
                                // 3. G = B_Z^T B_Z (3×3)
                                double G[3][3];
                                for (int i = 0; i < 3; ++i)
                                    for (int jj = 0; jj < 3; ++jj) {
                                        G[i][jj] = 0.0;
                                        for (int m = 0; m < NX; ++m)
                                            G[i][jj] += B_Z[m][i] * B_Z[m][jj];
                                    }
                                // 4. Jacobi eigendecomposition of 3×3 G
                                double GJ[3][3], VJ[3][3];
                                for (int i = 0; i < 3; ++i)
                                    for (int jj = 0; jj < 3; ++jj) {
                                        GJ[i][jj] = G[i][jj];
                                        VJ[i][jj] = (i==jj) ? 1.0 : 0.0;
                                    }
                                for (int sweep = 0; sweep < 50; ++sweep) {
                                    double off = 0.0;
                                    for (int i = 0; i < 3; ++i)
                                        for (int jj = i+1; jj < 3; ++jj)
                                            off += GJ[i][jj]*GJ[i][jj];
                                    if (off < 1e-30) break;
                                    for (int pp = 0; pp < 2; ++pp)
                                        for (int qq = pp+1; qq < 3; ++qq) {
                                            double Appq = GJ[pp][qq];
                                            if (std::fabs(Appq) < 1e-15) continue;
                                            double diff = GJ[qq][qq]-GJ[pp][pp];
                                            double tau = diff/(2.0*Appq);
                                            double tv = ((tau>=0)?1.0:-1.0)
                                                /(std::fabs(tau)+std::sqrt(1.0+tau*tau));
                                            double cs = 1.0/std::sqrt(1.0+tv*tv);
                                            double sn = tv*cs;
                                            for (int i = 0; i < 3; ++i) {
                                                double aip = GJ[i][pp], aiq = GJ[i][qq];
                                                GJ[i][pp] = cs*aip + sn*aiq;
                                                GJ[i][qq] = -sn*aip + cs*aiq;
                                            }
                                            for (int j = 0; j < 3; ++j) {
                                                double apj = GJ[pp][j], aqj = GJ[qq][j];
                                                GJ[pp][j] = cs*apj + sn*aqj;
                                                GJ[qq][j] = -sn*apj + cs*aqj;
                                            }
                                            for (int i = 0; i < 3; ++i) {
                                                double vip = VJ[i][pp], viq = VJ[i][qq];
                                                VJ[i][pp] = cs*vip + sn*viq;
                                                VJ[i][qq] = -sn*vip + cs*viq;
                                            }
                                        }
                                }
                                // Sort eigenvalues ascending
                                int si[3] = {0,1,2};
                                for (int i = 0; i < 2; ++i)
                                    for (int j = i+1; j < 3; ++j)
                                        if (GJ[si[i]][si[i]] > GJ[si[j]][si[j]])
                                            std::swap(si[i], si[j]);
                                double g_eval[3], g_evec[3][3];
                                for (int i = 0; i < 3; ++i) {
                                    g_eval[i] = GJ[si[i]][si[i]];
                                    for (int d = 0; d < 3; ++d)
                                        g_evec[i][d] = VJ[d][si[i]];
                                }
                                // 5. Dead direction in R^4
                                double d_dead[NU] = {};
                                for (int i = 0; i < NU; ++i)
                                    for (int d = 0; d < 3; ++d)
                                        d_dead[i] += Q_mat[i][d+1] * g_evec[0][d];
                                // 6. Projection of q_u onto dead direction
                                double q_dot_dead = 0.0;
                                double qu_norm_sq = 0.0;
                                double dead_norm_sq = 0.0;
                                for (int i = 0; i < NU; ++i) {
                                    q_dot_dead += rs.qu[i] * d_dead[i];
                                    qu_norm_sq += rs.qu[i]*rs.qu[i];
                                    dead_norm_sq += d_dead[i]*d_dead[i];
                                }
                                double qu_norm = std::sqrt(qu_norm_sq);
                                double dead_norm = std::sqrt(dead_norm_sq);
                                double cos_dead = (qu_norm>1e-30 && dead_norm>1e-30)
                                    ? q_dot_dead / (qu_norm * dead_norm) : 0.0;
                                // v0 alignment with dead direction
                                double v0_dot_dead = 0.0;
                                for (int i = 0; i < NU; ++i)
                                    v0_dot_dead += evec[0][i] * d_dead[i];
                                double cos_v0_dead = (dead_norm > 1e-30)
                                    ? v0_dot_dead / dead_norm : 0.0;
                                printf("  [null-CB] g_eval=[%.3e %.3e %.3e]"
                                       " rank(B|_nullC)=%d\n",
                                       g_eval[0], g_eval[1], g_eval[2],
                                       (g_eval[0] < 1e-6) ? 2 : 3);
                                printf("            d_dead=[%+.4f %+.4f %+.4f %+.4f]"
                                       " |d|=%.3e\n",
                                       d_dead[0], d_dead[1], d_dead[2], d_dead[3],
                                       dead_norm);
                                printf("            cos(q_u, dead)=%+.4f"
                                       " cos(v0, dead)=%+.4f\n",
                                       cos_dead, cos_v0_dead);
                                printf("            |q_u|=%.3e"
                                       " |q_u^T d_dead|/|d_dead|=%.3e"
                                       " (%.1f%% of |q_u|)\n",
                                       qu_norm,
                                       std::fabs(q_dot_dead)/dead_norm,
                                       100.0*std::fabs(cos_dead));
                                // Cross-check: B·d_dead and Cu[4,:]·d_dead
                                double Bd[NX], Bd_norm_sq = 0.0;
                                for (int m = 0; m < NX; ++m) {
                                    Bd[m] = 0.0;
                                    for (int i = 0; i < NU; ++i)
                                        Bd[m] += rs.B(m,i)*d_dead[i];
                                    Bd_norm_sq += Bd[m]*Bd[m];
                                }
                                printf("    [dd-chk] B*d_dead: [");
                                for (int m = 0; m < NX; ++m)
                                    printf("%.3e%s", Bd[m], m<NX-1?" ":"");
                                double cu4_dd = 0.0;
                                for (int i = 0; i < NU; ++i)
                                    cu4_dd += c4[i]*d_dead[i];
                                printf("]\n    [dd-chk] Cu4*d_dead=%.3e"
                                       " |B*d_dead|=%.3e\n",
                                       cu4_dd, std::sqrt(Bd_norm_sq));
                            }
                        }

                        // ── Per-mode S-decomposition ──
                        // λ_i = v_i^T S v_i = v_i^T Q_uu v_i
                        //                     + |W^{1/2} Cu v_i|^2
                        //                     + |P^{1/2} B v_i|^2
                        {
                            const auto& P1 = riccati_ws_.P[kk + 1];
                            printf("  [S-decomp] mode  Q_uu   Cu^TWCu  B^TPB"
                                   "  =sum   lambda  |Bv_i|\n");
                            for (int ei = 0; ei < 4; ++ei) {
                                double q_term = 0.0, cu_term = 0.0, bt_term = 0.0;
                                for (int i = 0; i < NU; ++i)
                                    for (int jj = 0; jj < NU; ++jj)
                                        q_term += evec[ei][i] * stg.Quu(i,jj)
                                                * evec[ei][jj];
                                for (int j = 0; j < NC; ++j) {
                                    double cv = 0.0;
                                    for (int i = 0; i < NU; ++i)
                                        cv += stg.Cu(j, i) * evec[ei][i];
                                    double wj = (stg.s[j] > 1e-30)
                                        ? stg.lambda[j] / stg.s[j] : 0.0;
                                    cu_term += wj * cv * cv;
                                }
                                double Bv[NX] = {};
                                for (int m = 0; m < NX; ++m)
                                    for (int i = 0; i < NU; ++i)
                                        Bv[m] += rs.B(m, i) * evec[ei][i];
                                for (int m = 0; m < NX; ++m)
                                    for (int n = 0; n < NX; ++n)
                                        bt_term += Bv[m] * P1(m,n) * Bv[n];
                                double Bv_norm = 0.0;
                                for (int m = 0; m < NX; ++m)
                                    Bv_norm += Bv[m]*Bv[m];
                                Bv_norm = std::sqrt(Bv_norm);
                                double sum3 = q_term + cu_term + bt_term;
                                printf("            %d:  %8.4f  %8.4f  %8.4f"
                                       "  %8.4f  %8.4f  %.6e\n",
                                       ei, q_term, cu_term, bt_term,
                                       sum3, eval_s[ei], Bv_norm);
                                // Cross-check for mode 0: Cu[j,:]·v0 and Bv
                                if (ei == 0) {
                                    printf("    [v0-chk] Cu[j]*v0: [");
                                    for (int j = 0; j < NC; ++j) {
                                        double cv = 0.0;
                                        for (int i = 0; i < NU; ++i)
                                            cv += stg.Cu(j,i)*evec[0][i];
                                        printf("%.3e%s", cv, j<NC-1?" ":"");
                                    }
                                    printf("]\n    [v0-chk] B*v0: [");
                                    for (int m = 0; m < NX; ++m) {
                                        double bv = 0.0;
                                        for (int i = 0; i < NU; ++i)
                                            bv += rs.B(m,i)*evec[0][i];
                                        printf("%.3e%s", bv, m<NX-1?" ":"");
                                    }
                                    printf("]\n    [v0-chk] v0=[");
                                    for (int i = 0; i < NU; ++i)
                                        printf("%+.6f%s", evec[0][i], i<NU-1?" ":"");
                                    printf("]\n");
                                }
                            }
                        }
                    }

                    // Energy fraction per mode: E_i = λ_i · β_i²
                    {
                        double E[4], E_total = 0.0;
                        for (int ei = 0; ei < 4; ++ei) {
                            E[ei] = eval_s[ei] * beta[ei] * beta[ei];
                            E_total += E[ei];
                        }
                        printf("  [energy]  E_i=lambda*beta^2:  [");
                        for (int ei = 0; ei < 4; ++ei)
                            printf("%.3e%s", E[ei], ei<3?" ":"");
                        printf("]  total=%.3e\n", E_total);
                        printf("            fraction:           [");
                        for (int ei = 0; ei < 4; ++ei)
                            printf("%.1f%%%s",
                                   (E_total > 1e-30)
                                       ? 100.0*E[ei]/E_total : 0.0,
                                   ei<3?" ":"");
                        printf("]\n");
                        // Also: step-norm fraction |β_i v_i|² / |du|²
                        double du_sq = 0.0;
                        for (int d = 0; d < NU; ++d)
                            du_sq += du_k[d]*du_k[d];
                        printf("  [step-nm] |beta_i*v_i|^2/|du|^2: [");
                        for (int ei = 0; ei < 4; ++ei) {
                            double bi_sq = beta[ei]*beta[ei];
                            printf("%.1f%%%s",
                                   (du_sq > 1e-30)
                                       ? 100.0*bi_sq/du_sq : 0.0,
                                   ei<3?" ":"");
                        }
                        printf("]\n");
                    }
                }

                printf("  [grad-dec] k=0: |cost_g|=%.3e |Cu^T*net|=%.3e |B^T*nu|=%.3e"
                       " | Cu^T*net_j4=%.3e Cu*du_j4=%+.3e\n",
                       cost_grad_norm, cu_net_total_norm, bt_nu_norm,
                       cu_net_j4, Cu_du_j4);
                printf("           du: |-K*dx|=%.3e |-d|=%.3e"
                       " qu_tilde=%.3e g0j4=%.3e s0j4=%.3e lam0j4=%.3e\n",
                       du_Kdx_norm, du_d_norm,
                       rs.qu.norm_inf(),
                       stg.d[4], stg.s[4], stg.lambda[4]);
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
                printf("  [kkt_res] dyn=%.2e feas=%.2e ricc_u=%.2e(%.2e) ricc_x=%.2e qu_err=%.2e fwd_err=%.2e"
                       " stat_x=%.2e stat_u=%.2e"
                       " term=%.2e comp=%.2e reg=%.2e schur=%.2e rdir=%.2e(%.2e)"
                       " | rel=%.2e(%s)\n",
                       linear_kkt_res_.max_dyn_res, linear_kkt_res_.max_feas_res,
                       linear_kkt_res_.max_riccati_u_res, linear_kkt_res_.max_riccati_u_scaled,
                       linear_kkt_res_.max_riccati_x_res,
                       linear_kkt_res_.max_qu_reconstruction_err,
                       linear_kkt_res_.max_forward_dyn_err,
                       linear_kkt_res_.max_stat_x_res, linear_kkt_res_.max_stat_u_res,
                       linear_kkt_res_.max_stat_term_res, linear_kkt_res_.max_comp_res,
                       reg_used_,
                       Ricc::schur_residual,
                       Ricc::riccati_direct_stationarity,
                       Ricc::riccati_direct_stationarity_corr,
                       linear_kkt_res_.max_rel_res, linear_kkt_res_.quality_label());

                // ── P·dx magnitude diagnostic ─────────────────────────
                // The Riccati costate ν = p + P·dx. When P grows large (barrier
                // Hessian dominance), the P·dx term can dominate and amplify
                // any forward-pass error in dx.
                {
                    double max_p = 0.0, max_Pdx = 0.0, max_P = 0.0, max_dx = 0.0;
                    int worst_Pdx_k = -1;
                    for (int kk = 0; kk <= HORIZON; ++kk) {
                        for (int i = 0; i < NX; ++i) {
                            double pv = std::fabs(riccati_ws_.p[kk][i]);
                            if (pv > max_p) max_p = pv;
                            double dxv = std::fabs(riccati_ws_.dx[kk][i]);
                            if (dxv > max_dx) max_dx = dxv;
                        }
                        double Pdx_inf = 0.0;
                        for (int i = 0; i < NX; ++i) {
                            double Pdx_i = 0.0;
                            for (int j = 0; j < NX; ++j)
                                Pdx_i += riccati_ws_.P[kk](i,j) * riccati_ws_.dx[kk][j];
                            if (std::fabs(Pdx_i) > Pdx_inf) Pdx_inf = std::fabs(Pdx_i);
                        }
                        if (Pdx_inf > max_Pdx) { max_Pdx = Pdx_inf; worst_Pdx_k = kk; }
                        for (int i = 0; i < NX; ++i)
                            for (int j = 0; j < NX; ++j) {
                                double Pv = std::fabs(riccati_ws_.P[kk](i,j));
                                if (Pv > max_P) max_P = Pv;
                            }
                    }
                    printf("  [Pdx-diag] |p|=%.2e |P|=%.2e |dx|=%.2e |P·dx|=%.2e (worst k=%d)"
                           " |P·dx|/|p|=%.1f\n",
                           max_p, max_P, max_dx, max_Pdx, worst_Pdx_k,
                           max_p > 1e-14 ? max_Pdx / max_p : 0.0);
                }

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
            // Track consecutive small FTB steps for constraint correction gating
            if (alpha_p < 0.10) ++low_ftb_count_; else low_ftb_count_ = 0;
            log_iteration(iter, sigma_, alpha_p, alpha_d);
            alpha_lambda_ = alpha_d;  // FTB step for λ (independent of line search)

            // ── Diagnostic snapshot (before line search) ──
            last_diag_ = compute_iter_diagnostics();
            last_diag_.alpha_p = alpha_p;
            last_diag_.alpha_d = alpha_d;

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

                // ── FTB primal variable breakdown at bottleneck stage ──
                // Show ALL constraints at bkw: g, s, ds, Cu*du, -(g+s), alpha_j
                // IMPORTANT: C is scaled, so must use scaled dz for C*dz
                if (bkw >= 0) {
                    const auto& stg_bk = prob_->stages[bkw];
                    printf("  [ftb-primal] stage k=%d: |du_phys|=[", bkw);
                    for (int i = 0; i < NU; ++i)
                        printf("%.2e%s", riccati_ws_.du[bkw][i], i<NU-1?" ":"");
                    printf("] |dx_phys|=[");
                    for (int i = 0; i < NX; ++i)
                        printf("%.2e%s", riccati_ws_.dx[bkw][i], i<NX-1?" ":"");
                    printf("]\n");
                    printf("  [ftb-primal] j | %8s %8s %10s %10s %10s %8s\n",
                           "g", "s", "C*dz", "-(g+s)", "ds", "alpha_j");
                    for (int jj = 0; jj < NC; ++jj) {
                        double gj = stg_bk.d[jj];
                        double sj = stg_bk.s[jj];
                        double C_dz_j = 0.0;
                        for (int i = 0; i < NX; ++i)
                            C_dz_j += stg_bk.Cx(jj, i) * debug_scaled_dx_[bkw][i];
                        for (int i = 0; i < NU; ++i)
                            C_dz_j += stg_bk.Cu(jj, i) * debug_scaled_du_[bkw][i];
                        double neg_gs = -(gj + sj);
                        double ds_j = neg_gs - C_dz_j;
                        double alpha_j = (ds_j < -1e-16 && sj > 1e-14)
                            ? -params_.tau * sj / ds_j : 1.0;
                        printf("            %d | %+.2e %+.2e %+.4e %+.4e %+.4e %8.4f%s\n",
                               jj, gj, sj, C_dz_j, neg_gs, ds_j, alpha_j,
                               (jj == bjw) ? " <-- BOTTLENECK" : "");
                    }
                }

                // ── Self-consistent normal compliance diagnostic ──
                // At the FTB bottleneck stage (bkw) and constraint (bjw):
                // Build S from first principles (undamped, unregularized):
                //   S_aug = Q_uu + B^T P B + Cu^T W Cu
                // Compute full Newton RHS:
                //   r_full = qu + B^T nu + Cu^T eta
                // Verify 5 identities:
                //   1) S construction (by definition)
                //   2) S*du ≈ -r_full (Newton equation)
                //   3) lambda_min <= n^T S n <= lambda_max (Rayleigh)
                //   4) n^T du = -n^T S^{-1} r_full (prediction)
                //   5) c_n = sum (v_i^T n)^2 / lambda_i (modal reconstruction)
                if (bkw >= 0 && bkw < HORIZON) {
                    const auto& stg_b = prob_->stages[bkw];
                    const auto& P_next = riccati_ws_.P[bkw + 1];

                    // ── Build S_aug = Q_uu + B^T P B + Cu^T W Cu ──
                    double S_aug[NU][NU] = {};
                    // Q_uu
                    for (int i = 0; i < NU; ++i)
                        for (int j = 0; j < NU; ++j)
                            S_aug[i][j] = stg_b.Quu(i, j);
                    // B^T P B
                    for (int r = 0; r < NU; ++r)
                        for (int c = 0; c < NU; ++c) {
                            double val = 0.0;
                            for (int m = 0; m < NX; ++m)
                                for (int nn = 0; nn < NX; ++nn)
                                    val += stg_b.B(m, r) * P_next(m, nn) * stg_b.B(nn, c);
                            S_aug[r][c] += val;
                        }
                    // Cu^T W Cu (barrier Hessian)
                    for (int j = 0; j < NC; ++j) {
                        double sj = stg_b.s[j];
                        if (sj < 1e-14) continue;
                        double wj = stg_b.lambda[j] / sj;
                        for (int r = 0; r < NU; ++r)
                            for (int c = 0; c < NU; ++c)
                                S_aug[r][c] += wj * stg_b.Cu(j, r) * stg_b.Cu(j, c);
                    }

                    // ── Unit normal n = Cu[bjw,:]^T / |Cu[bjw,:]| ──
                    double n_vec[NU];
                    double c_norm_sq = 0.0;
                    for (int i = 0; i < NU; ++i) {
                        n_vec[i] = stg_b.Cu(bjw, i);
                        c_norm_sq += n_vec[i] * n_vec[i];
                    }
                    double c_norm = std::sqrt(c_norm_sq);
                    if (c_norm > 1e-14)
                        for (int i = 0; i < NU; ++i)
                            n_vec[i] /= c_norm;

                    // ── Eigendecompose S_aug via Jacobi ──
                    double A_jac[NU][NU], V_jac[NU][NU];
                    for (int i = 0; i < NU; ++i)
                        for (int j = 0; j < NU; ++j) {
                            A_jac[i][j] = S_aug[i][j];
                            V_jac[i][j] = (i == j) ? 1.0 : 0.0;
                        }
                    for (int sweep = 0; sweep < 50; ++sweep) {
                        double off = 0.0;
                        for (int i = 0; i < NU; ++i)
                            for (int j = i+1; j < NU; ++j)
                                off += A_jac[i][j] * A_jac[i][j];
                        if (off < 1e-30) break;
                        for (int pp = 0; pp < NU-1; ++pp)
                            for (int qq = pp+1; qq < NU; ++qq) {
                                double apq = A_jac[pp][qq];
                                if (std::fabs(apq) < 1e-15) continue;
                                double diff = A_jac[qq][qq] - A_jac[pp][pp];
                                double tau = diff / (2.0 * apq);
                                double t_val = ((tau >= 0) ? 1.0 : -1.0)
                                    / (std::fabs(tau) + std::sqrt(1.0 + tau*tau));
                                double cs = 1.0 / std::sqrt(1.0 + t_val*t_val);
                                double sn = t_val * cs;
                                double app = A_jac[pp][pp], aqq = A_jac[qq][qq];
                                A_jac[pp][pp] = cs*cs*app - 2.0*sn*cs*apq + sn*sn*aqq;
                                A_jac[qq][qq] = sn*sn*app + 2.0*sn*cs*apq + cs*cs*aqq;
                                A_jac[pp][qq] = 0.0; A_jac[qq][pp] = 0.0;
                                for (int rr = 0; rr < NU; ++rr) {
                                    if (rr == pp || rr == qq) continue;
                                    double arp = A_jac[rr][pp], arq = A_jac[rr][qq];
                                    A_jac[rr][pp] = cs*arp - sn*arq;
                                    A_jac[pp][rr] = A_jac[rr][pp];
                                    A_jac[rr][qq] = sn*arp + cs*arq;
                                    A_jac[qq][rr] = A_jac[rr][qq];
                                }
                                for (int rr = 0; rr < NU; ++rr) {
                                    double vrp = V_jac[rr][pp], vrq = V_jac[rr][qq];
                                    V_jac[rr][pp] = cs*vrp - sn*vrq;
                                    V_jac[rr][qq] = sn*vrp + cs*vrq;
                                }
                            }
                    }
                    double eval_s[NU];
                    for (int i = 0; i < NU; ++i)
                        eval_s[i] = A_jac[i][i];
                    double lam_min = eval_s[0], lam_max = eval_s[0];
                    for (int i = 1; i < NU; ++i) {
                        lam_min = std::min(lam_min, eval_s[i]);
                        lam_max = std::max(lam_max, eval_s[i]);
                    }

                    // ── Identity 3: Rayleigh quotient bound ──
                    // k_n = n^T S_aug n
                    double Sn[NU];
                    for (int i = 0; i < NU; ++i) {
                        Sn[i] = 0.0;
                        for (int j = 0; j < NU; ++j)
                            Sn[i] += S_aug[i][j] * n_vec[j];
                    }
                    double k_n = 0.0;
                    for (int i = 0; i < NU; ++i)
                        k_n += n_vec[i] * Sn[i];

                    // ── Stiffness decomposition ──
                    double k_Q = 0.0;
                    for (int i = 0; i < NU; ++i)
                        for (int j = 0; j < NU; ++j)
                            k_Q += n_vec[i] * stg_b.Quu(i, j) * n_vec[j];
                    double Bn_vec[NX] = {};
                    for (int m = 0; m < NX; ++m)
                        for (int i = 0; i < NU; ++i)
                            Bn_vec[m] += stg_b.B(m, i) * n_vec[i];
                    double k_P = 0.0;
                    for (int m = 0; m < NX; ++m)
                        for (int nn = 0; nn < NX; ++nn)
                            k_P += Bn_vec[m] * P_next(m, nn) * Bn_vec[nn];
                    double k_B = k_n - k_Q - k_P;  // by construction

                    // ── Full Newton RHS ──
                    // The Riccati stages already include barrier gradient:
                    //   riccati_qu = qu_cost + Cu^T * (centering - lambda)
                    // So the full RHS for the reduced KKT is:
                    //   r_full = riccati_qu + B^T * nu
                    // where nu = p_{k+1} + P_{k+1} * c  (dynamics costate + defect)
                    // IMPORTANT: use SCALED costates (before recover_dual_step)
                    // to match the scaled-space S_aug
                    double nu_next[NX];
                    for (int m = 0; m < NX; ++m) {
                        nu_next[m] = debug_scaled_p_[bkw + 1][m];
                        for (int nn = 0; nn < NX; ++nn)
                            nu_next[m] += P_next(m, nn) * stg_b.c[nn];
                    }
                    // r_full = riccati_qu + B^T nu
                    // (riccati_qu already includes barrier Cu^T*eta)
                    double r_full[NU];
                    for (int i = 0; i < NU; ++i) {
                        double Bt_nu_i = 0.0;
                        for (int m = 0; m < NX; ++m)
                            Bt_nu_i += stg_b.B(m, i) * nu_next[m];
                        r_full[i] = riccati_stages_[bkw].qu[i] + Bt_nu_i;
                    }

                    // ── Identity 4: n^T du vs -n^T S^{-1} r_full ──
                    // IMPORTANT: S_aug is in SCALED space (from transform_qp),
                    // so we must use the SCALED du (before recover_primal_step).
                    const auto& du_scaled = debug_scaled_du_[bkw];
                    const auto& dx_scaled = debug_scaled_dx_[bkw];
                    // Solve S * x = r_full via eigendecomposition
                    // S^{-1} r = V diag(1/lambda) V^T r
                    double Vt_r[NU];
                    for (int i = 0; i < NU; ++i) {
                        Vt_r[i] = 0.0;
                        for (int j = 0; j < NU; ++j)
                            Vt_r[i] += V_jac[j][i] * r_full[j];
                    }
                    double Sinv_r[NU];
                    for (int i = 0; i < NU; ++i) {
                        Sinv_r[i] = 0.0;
                        for (int j = 0; j < NU; ++j) {
                            double inv_lam = (std::fabs(eval_s[j]) > 1e-30)
                                ? 1.0 / eval_s[j] : 0.0;
                            Sinv_r[i] += V_jac[i][j] * inv_lam * Vt_r[j];
                        }
                    }
                    double n_dot_Sinv_r = 0.0;
                    for (int i = 0; i < NU; ++i)
                        n_dot_Sinv_r += n_vec[i] * Sinv_r[i];
                    double d_n_pred_full = -n_dot_Sinv_r;

                    // Actual normal displacement (SCALED du)
                    double d_n_actual = 0.0;
                    for (int i = 0; i < NU; ++i)
                        d_n_actual += n_vec[i] * du_scaled[i];

                    // ── Identity 2: ||S*du + r_full|| (Newton residual) ──
                    double Sdu[NU];
                    for (int i = 0; i < NU; ++i) {
                        Sdu[i] = 0.0;
                        for (int j = 0; j < NU; ++j)
                            Sdu[i] += S_aug[i][j] * du_scaled[j];
                    }
                    double newton_res = 0.0;
                    for (int i = 0; i < NU; ++i) {
                        double res_i = Sdu[i] + r_full[i];
                        newton_res = std::max(newton_res, std::fabs(res_i));
                    }

                    // ── Identity 5: Modal reconstruction of compliance ──
                    // c_n_modal = sum (v_i^T n)^2 / lambda_i
                    double c_n_modal = 0.0;
                    for (int i = 0; i < NU; ++i) {
                        double vt_n = 0.0;
                        for (int j = 0; j < NU; ++j)
                            vt_n += V_jac[j][i] * n_vec[j];
                        if (std::fabs(eval_s[i]) > 1e-30)
                            c_n_modal += (vt_n * vt_n) / eval_s[i];
                    }
                    // Direct compliance from S_aug
                    double Sinv_n_vec[NU];
                    for (int i = 0; i < NU; ++i) {
                        Sinv_n_vec[i] = 0.0;
                        for (int j = 0; j < NU; ++j) {
                            double inv_lam = (std::fabs(eval_s[j]) > 1e-30)
                                ? 1.0 / eval_s[j] : 0.0;
                            double vt_n = 0.0;
                            for (int kk = 0; kk < NU; ++kk)
                                vt_n += V_jac[kk][j] * n_vec[kk];
                            Sinv_n_vec[i] += V_jac[i][j] * inv_lam * vt_n;
                        }
                    }
                    double c_n_direct = 0.0;
                    for (int i = 0; i < NU; ++i)
                        c_n_direct += n_vec[i] * Sinv_n_vec[i];

                    // Compute |r_full| and n^T r_full for output
                    double r_full_norm = 0.0;
                    for (int i = 0; i < NU; ++i)
                        r_full_norm += r_full[i] * r_full[i];
                    r_full_norm = std::sqrt(r_full_norm);
                    double n_dot_r_full = 0.0;
                    for (int i = 0; i < NU; ++i)
                        n_dot_r_full += n_vec[i] * r_full[i];

                    // ── Print results ──
                    printf("  [comply ] k=%d j=%d: k_Q=%.3e k_P=%.3e k_B=%.3e k_n=%.3e\n",
                           bkw, bjw, k_Q, k_P, k_B, k_n);
                    printf("            evals=[");
                    for (int i = 0; i < NU; ++i)
                        printf("%.3e%s", eval_s[i], i<NU-1?" ":"");
                    printf("]\n");
                    printf("            Id3: k_n=%.3e in [%.3e,%.3e]? %s\n",
                           k_n, lam_min, lam_max,
                           (k_n >= lam_min - 1e-6 && k_n <= lam_max + 1e-6) ? "YES" : "FAIL");
                    printf("            Id5: c_n_direct=%.4e c_n_modal=%.4e match=%s\n",
                           c_n_direct, c_n_modal,
                           (std::fabs(c_n_direct - c_n_modal) < 1e-6 * (std::fabs(c_n_direct) + 1e-30))
                               ? "YES" : "FAIL");
                    printf("            Id4: n^Tdu=%+.4e  -n^TS^{-1}r_full=%+.4e  match=%s\n",
                           d_n_actual, d_n_pred_full,
                           (std::fabs(d_n_actual - d_n_pred_full) < 0.01 * (std::fabs(d_n_actual) + std::fabs(d_n_pred_full) + 1e-30))
                               ? "YES" : "FAIL");
                    printf("            Id2: ||S*du + r_full||_inf=%.3e\n", newton_res);
                    printf("            s=%.2e lam=%.2e lam/s=%.2e |r_full|=%.3e n^Tr_full=%+.3e\n",
                           stg_b.s[bjw], stg_b.lambda[bjw],
                           stg_b.lambda[bjw] / std::max(stg_b.s[bjw], 1e-30),
                           r_full_norm, n_dot_r_full);

                    // ══════════════════════════════════════════════════════
                    //  SYSTEMATIC ELIMINATION OF ALL OTHER CAUSES
                    // ══════════════════════════════════════════════════════

                    // ── Check A: ds consistency ──
                    // ds = -(g+s) - C*dz.  C is scaled, so must use SCALED dz.
                    // C_scaled * dz_scaled = (C*inv_L) * (L*dz_phys) = C*dz_phys (invariant)
                    // But C_scaled * dz_phys != C_scaled * dz_scaled!
                    const auto& du_sc = debug_scaled_du_[bkw];
                    const auto& dx_sc = debug_scaled_dx_[bkw];
                    double Cz_dz_sc = 0.0;
                    for (int i = 0; i < NX; ++i)
                        Cz_dz_sc += stg_b.Cx(bjw, i) * dx_sc[i];
                    for (int i = 0; i < NU; ++i)
                        Cz_dz_sc += stg_b.Cu(bjw, i) * du_sc[i];
                    double ds_formula_sc = -(stg_b.d[bjw] + stg_b.s[bjw]) - Cz_dz_sc;
                    // Also compute with physical dz (should differ if scaling is active)
                    double Cz_dz_ph = 0.0;
                    for (int i = 0; i < NX; ++i)
                        Cz_dz_ph += stg_b.Cx(bjw, i) * riccati_ws_.dx[bkw][i];
                    for (int i = 0; i < NU; ++i)
                        Cz_dz_ph += stg_b.Cu(bjw, i) * riccati_ws_.du[bkw][i];
                    double ds_formula_ph = -(stg_b.d[bjw] + stg_b.s[bjw]) - Cz_dz_ph;
                    double ds_solver = ds_[bkw][bjw];
                    printf("  [verify] A: ds_solver=%+.6e  ds(scaled_dz)=%+.6e  ds(phys_dz)=%+.6e\n",
                           ds_solver, ds_formula_sc, ds_formula_ph);
                    printf("           |err_scaled|=%.2e %s  |err_phys|=%.2e %s\n",
                           std::fabs(ds_solver - ds_formula_sc),
                           (std::fabs(ds_solver - ds_formula_sc) < 1e-8 * (std::fabs(ds_solver) + 1e-16))
                               ? "PASS" : "FAIL",
                           std::fabs(ds_solver - ds_formula_ph),
                           (std::fabs(ds_solver - ds_formula_ph) < 1e-8 * (std::fabs(ds_solver) + 1e-16))
                               ? "PASS" : "FAIL");
                    // Use the correct one for subsequent checks
                    double Cz_dz = Cz_dz_sc;  // the correct one

                    // ── Check B: Newton step would fix equality residual ──
                    // r_c = g + s  (current equality residual)
                    // After full Newton step (alpha=1): r_c_new = r_c + ds = -C*dz
                    // (because ds = -(g+s) - C*dz, so r_c + ds = -C*dz)
                    double rc = stg_b.d[bjw] + stg_b.s[bjw];  // g + s
                    double rc_full = -Cz_dz;  // what r_c would be after alpha=1
                    double rc_alpha = rc + alpha_p * ds_solver;  // what r_c is after actual alpha
                    printf("  [verify] B: r_c=g+s=%+.4e  r_c(alpha=1)=%+.4e  r_c(alpha=%.4f)=%+.4e\n",
                           rc, rc_full, alpha_p, rc_alpha);
                    printf("           r_c reduction at alpha=1: %.1f%%  at actual alpha: %.2f%%\n",
                           (1.0 - std::fabs(rc_full) / (std::fabs(rc) + 1e-30)) * 100.0,
                           (1.0 - std::fabs(rc_alpha) / (std::fabs(rc) + 1e-30)) * 100.0);

                    // ── Check C: Slack-violation ratio ──
                    // If s << r_c, the slack is collapsing faster than feasibility improves
                    double sv_ratio = stg_b.s[bjw] / (std::fabs(rc) + 1e-30);
                    printf("  [verify] C: s/r_c = %.2e / %.2e = %.4e  %s\n",
                           stg_b.s[bjw], rc, sv_ratio,
                           (sv_ratio < 0.01) ? "DESYNC (s << r_c)" :
                           (sv_ratio > 100)  ? "DESYNC (r_c << s)" : "BALANCED");

                    // ── Check D: Barrier parameter health ──
                    double smu = sigma_ * mu_;
                    double implied_floor = 0.1 * mu_;  // tau * mu is the FTB target
                    printf("  [verify] D: mu=%.3e  sigma=%.3e  sigma*mu=%.3e  tau*mu=%.3e  s/(tau*mu)=%.2f\n",
                           mu_, sigma_, smu, params_.tau * mu_,
                           stg_b.s[bjw] / (params_.tau * mu_ + 1e-30));

                    // ── Check E: FTB alpha decomposition ──
                    // alpha = -tau*s/ds.  What if s were larger?
                    // If s were = r_c (balanced): alpha_bal = -tau*r_c/ds
                    double ds_neg = -ds_solver;  // positive quantity (ds < 0)
                    double alpha_bal = (ds_neg > 1e-16) ? params_.tau * std::fabs(rc) / ds_neg : 1.0;
                    double alpha_curr = (ds_neg > 1e-16) ? params_.tau * stg_b.s[bjw] / ds_neg : 1.0;
                    printf("  [verify] E: alpha_curr=%.4f (from s=%.2e)  alpha_bal=%.4f (if s=|r_c|)  ratio=%.1fx\n",
                           alpha_curr, stg_b.s[bjw], alpha_bal,
                           alpha_bal / (alpha_curr + 1e-30));
                    printf("           ds breakdown: -(g+s)=%+.4e  C*dz=%+.4e  ds=%+.4e\n",
                           -rc, Cz_dz, ds_solver);
                    printf("           |C*dz|/|g+s| = %.4f  (Newton vs centering)\n",
                           std::fabs(Cz_dz) / (std::fabs(rc) + 1e-30));
                }
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
                // ── Slack lifecycle: before step ──────────────────
                double g_old_j4 = 0.0;
                double Cu_du_pred_j4 = 0.0;
                if (params_.verbosity >= 1) {
                    auto& stg = prob_->stages[0];
                    g_old_j4 = stg.d[4];
                    // Cu·du for bottleneck constraint
                    // stg.Cu is SCALED (from transform_qp), du is PHYSICAL (recovered)
                    // Cu_scaled·du_phys = Cu_phys·inv_Lu·du_phys (not invariant!)
                    // Compute physical Cu·du properly:
                    for (int i = 0; i < NU; ++i) {
                        double Cu_phys_i = stg.Cu(4, i);
                        if (params_.enable_preconditioner)
                            Cu_phys_i /= prec_.inv_Lu(0)[i];  // unscale
                        Cu_du_pred_j4 += Cu_phys_i * riccati_ws_.du[0][i];
                    }
                    printf("  [lc0] pre-step  k0j4: g=%.4e s=%.4e g+s=%+.4e Cu·du=%+.4e\n",
                           stg.d[4], stg.s[4], stg.d[4]+stg.s[4], Cu_du_pred_j4);
                    printf("       u=[%.6f %.6f %.6f %.6f]\n",
                           stg.u[0], stg.u[1], stg.u[2], stg.u[3]);
                }
                apply_primal_dual_step(alpha, alpha_lambda_);

                // ── Slack lifecycle: after step (before eval) ──────
                if (params_.verbosity >= 1) {
                    auto& stg = prob_->stages[0];
                    printf("  [lc1] post-step k0j4: g=%.4e s=%.4e g+s=%+.4e (a=%.3f ds=%+.3e)\n",
                           stg.d[4], stg.s[4], stg.d[4]+stg.s[4],
                           alpha, ds_[0][4]);
                    printf("       u=[%.6f %.6f %.6f %.6f]\n",
                           stg.u[0], stg.u[1], stg.u[2], stg.u[3]);
                }

                // Re-evaluate model at new point for accurate theta reporting
                st = evaluate_model();
                if (st != Status::SUCCESS) return st;
                model_evaluated = true;
                stages_scaled_ = false;

                // ── Slack lifecycle: after evaluate_model ──────────
                if (params_.verbosity >= 1) {
                    auto& stg = prob_->stages[0];
                    double g_new_j4 = stg.d[4];
                    double dg_actual = g_new_j4 - g_old_j4;
                    double dg_pred = alpha * Cu_du_pred_j4;
                    double ratio = (std::fabs(dg_pred) > 1e-14)
                                   ? dg_actual / dg_pred : 0.0;
                    printf("  [lc2] post-eval k0j4: g=%.4e s=%.4e g+s=%+.4e"
                           " | pred=%.4e act=%.4e ratio=%.3f\n",
                           stg.d[4], stg.s[4], stg.d[4]+stg.s[4],
                           dg_pred, dg_actual, ratio);
                }

                if (params_.verbosity >= 1) {
                    double effective_alam = std::min(alpha_lambda_, alpha);
                    printf("  [step: a=%.4f alam=%.4f ls=%d soc=%s cost=%.4e theta=%.4e]\n",
                           alpha, effective_alam, ls_iters,
                           ls_result.soc_used ? "yes" : "no",
                           compute_objective(), compute_theta());
                }
                // ── Diagnostic: update with LS results ──
                last_diag_.ls_iters   = ls_result.ls_iters;
                last_diag_.ls_rejected = false;
                last_diag_.alpha_p    = alpha;
                // theta_dyn/theta_ineq already in pre-LS snapshot
            } else {
                alpha = ls_result.alpha;
                ++ls_fail_count;
                last_diag_.ls_rejected = true;
                last_diag_.alpha_p = alpha;
                if (params_.verbosity >= 1)
                    printf("  [LS fail: a=%.3e alam=%.2e ls=%d fail_count=%d] step too tiny\n",
                           alpha, alpha_lambda_, ls_result.ls_iters, ls_fail_count);
                // Try filter reset on first few failures
                if (ls_fail_count <= 3) {
                    filter_ls_.reset_filter();
                    if (params_.verbosity >= 1)
                        printf("  [filter reset: trying again after filter exhaustion]\n");
                    // Force a small step to escape the filter trap
                    double forced_alpha = std::max(alpha, 1e-6);
                    apply_primal_dual_step(forced_alpha, std::min(forced_alpha, alpha_lambda_));
                    model_evaluated = false;
                    continue;
                }
                ls_failed = true;
                break;
            }

            // Complementarity safeguard
            if (barrier_strategy_.m_safe() > 0.0) {
                if (params_.verbosity >= 3)
                    printf("  [safeguard: m_safe=%.3e]\n", barrier_strategy_.m_safe());
                sz_complement(barrier_strategy_.m_safe());
                // ── Slack lifecycle: after sz_complement ──────────
                if (params_.verbosity >= 1) {
                    auto& stg = prob_->stages[0];
                    printf("  [lc3] post-szC  k0j4: g=%.4e s=%.4e g+s=%+.4e mu=%.2e\n",
                           stg.d[4], stg.s[4], stg.d[4]+stg.s[4], mu_);
                }
            }

            // Barrier update: reduce μ if subproblem solved, else hold
            // Passes alpha_p and max_g_pos for graduated FTB reduction
            // and two-phase barrier schedule (Phase 4/5/6).
            {
                double E_mu = std::max(primal_inf_, compl_inf_);
                // Freeze μ for debugging: skip update after N iters
                bool frozen = (params_.freeze_mu_after >= 0 && iter >= params_.freeze_mu_after);
                bool mu_changed = false;
                if (!frozen) {
                    mu_changed = barrier_strategy_.update(
                        mu_, primal_inf_, compl_inf_, sigma_, stat_inf_,
                        cross_term_accepted_, alpha_p, max_g_pos_);
                }

                // ── Feasibility restoration: enlarge slacks (once) ───
                // When restoration triggers, artificially lift slacks once
                // to break the s/r_c desynchronization.  The flag prevents
                // repeated enlargement (which would create a positive feedback
                // loop: bigger slacks → weaker barrier → larger max_g+).
                if (barrier_strategy_.should_enlarge_slacks()
                    && params_.c_restoration * max_g_pos_ > mu_) {
                    double rest_threshold = params_.c_restoration * max_g_pos_;
                    Stage* stgs = prob_->stages;
                    for (int kk = 0; kk <= HORIZON; ++kk) {
                        for (int jj = 0; jj < NC; ++jj) {
                            double new_floor = std::max(rest_threshold, stgs[kk].s[jj]);
                            if (new_floor > stgs[kk].s[jj]) {
                                stgs[kk].s[jj] = new_floor;
                                stgs[kk].lambda[jj] = mu_ / stgs[kk].s[jj];
                            }
                        }
                    }
                    if (params_.verbosity >= 1)
                        printf("  [RESTORE: slacks enlarged to c_rest*max_g+=%.3e]\n",
                               rest_threshold);
                    barrier_strategy_.mark_slacks_enlarged();
                }

                if (params_.verbosity >= 1) {
                    const char* phase_str = (barrier_strategy_.phase()
                        == BarrierUpdateStrategy::Phase::INFEASIBILITY)
                        ? "A" : "B";
                    printf("  [barrier: E_mu=%.2e k*mu=%.2e %s mu=%.2e%s"
                           " phase=%s max_g+=%.3e ap=%.4f]\n",
                           E_mu, barrier_strategy_.kappa_eps() * mu_,
                           frozen ? "FROZEN" : (mu_changed ? "REDUCE" : "HOLD "),
                           mu_, frozen ? " (frozen)" : "",
                           phase_str, max_g_pos_, alpha_p);
                }
                if (mu_changed) {
                    filter_ls_.reset_filter();
                }
            }

            // ── End-of-iteration diagnostic emission ──
            if (params_.verbosity >= 1) {
                last_diag_.print_compact(iter);
            }
            if (diag_csv_) {
                last_diag_.print_csv_row(diag_csv_, iter);
                fflush(diag_csv_);
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
                // Stagnation hint: if complementarity is OK but stat+primal stuck
                if (c_ok && !s_ok && !p_ok
                    && final_status == Status::STAGNATION) {
                    printf("  * HINT: complementarity satisfied but stationarity/primal stalled.\n");
                    printf("    Cost=%.2f is well-converged. The KKT system may be at the\n",
                           out_stats.cost);
                    printf("    conditioning limit of the Hessian approximation.\n");
                    printf("    Try relaxing tol_stat/tol_primal or using analytic Hessians.\n");
                }
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

            if (NU > 0) {
                printf("First u* =       [");
                for (int i = 0; i < NU; ++i) {
                    printf("%.3f%s", prob_->stages[0].u[i], (i < NU - 1) ? ", " : "");
                }
                printf("]\n");
            }
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
                    double gj = s[k].d[j];
                    // Adaptive slack initialization (Phase 1):
                    //   violated (g > 0): s = max(mu, s_min) — small slack
                    //     so that |g+s| ≈ g (not 2g). The barrier will push
                    //     both x and s toward feasibility together.
                    //   feasible (g <= 0): s = max(-g + delta, s_min)
                    //     slack proportional to distance from boundary.
                    double s_init;
                    if (gj > 0.0) {
                        // Violated: small slack so g+s stays ≈ g, not 2g.
                        s_init = std::max(mu_, params_.s_min_init);
                    } else {
                        // Feasible: slack = distance from boundary + margin.
                        s_init = std::max(-gj + params_.delta_slack,
                                          params_.s_min_init);
                    }
                    s[k].s[j] = s_init;
                    s[k].lambda[j] = mu_ / s[k].s[j];
                }
            } else {
                for (int j = 0; j < NC; ++j) {
                    s[k].s[j]      = 1.0;
                    s[k].lambda[j] = 1.0;
                }
            }
            // Initialize bound multipliers: z = mu / d
            if (prob_->n_bound_u > 0 && k < N) {
                for (int i = 0; i < NU; ++i) {
                    double dL = s[k].u[i] - prob_->u_lb[i];
                    double dU = prob_->u_ub[i] - s[k].u[i];
                    s[k].z_L_u[i] = (dL > 1e-14) ? mu_ / dL : 0.0;
                    s[k].z_U_u[i] = (dU > 1e-14) ? mu_ / dU : 0.0;
                }
            }
            if (prob_->n_bound_x > 0) {
                for (int i = 0; i < NX; ++i) {
                    double dL = s[k].x[i] - prob_->x_lb[i];
                    double dU = prob_->x_ub[i] - s[k].x[i];
                    s[k].z_L_x[i] = (dL > 1e-14) ? mu_ / dL : 0.0;
                    s[k].z_U_x[i] = (dU > 1e-14) ? mu_ / dU : 0.0;
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

            // Add bound barrier gradient: -mu/dL + mu/dU
            // (physical, same coordinate system as qx/qu before unscaling)
            double bound_grad_x_inf = 0.0, bound_grad_u_inf = 0.0;
            if (prob_->n_bound_x > 0) {
                for (int i = 0; i < NX; ++i) {
                    double dL = s[k].x[i] - prob_->x_lb[i];
                    double dU = prob_->x_ub[i] - s[k].x[i];
                    double bg = 0.0;
                    if (dL > 1e-14) bg -= mu_ / dL;
                    if (dU > 1e-14) bg += mu_ / dU;
                    lag_x[i] += bg;
                    bound_grad_x_inf = std::max(bound_grad_x_inf, std::fabs(bg));
                }
            }
            if (prob_->n_bound_u > 0 && k < N) {
                for (int i = 0; i < NU; ++i) {
                    double dL = s[k].u[i] - prob_->u_lb[i];
                    double dU = prob_->u_ub[i] - s[k].u[i];
                    double bg = 0.0;
                    if (dL > 1e-14) bg -= mu_ / dL;
                    if (dU > 1e-14) bg += mu_ / dU;
                    lag_u[i] += bg;
                    bound_grad_u_inf = std::max(bound_grad_u_inf, std::fabs(bg));
                }
            }

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
                scale_x = std::max({gx, bound_grad_x_inf, 1.0});
                if (k < N) {
                    for (int i = 0; i < NU; ++i) {
                        gu = std::max(gu, std::fabs(s[k].qu[i] / prec_.inv_Lu(k)[i]));
                        gu = std::max(gu, std::fabs(cos_u[i] / prec_.inv_Lu(k)[i]));
                    }
                    scale_u = std::max({gu, bound_grad_u_inf, 1.0});
                } else {
                    scale_u = 1.0;
                }
            } else {
                double grad_x_inf = s[k].qx.norm_inf();
                double grad_u_inf = (k < N) ? s[k].qu.norm_inf() : 0.0;
                double cos_x_inf = cos_x.norm_inf();
                double cos_u_inf = cos_u.norm_inf();
                scale_x = std::max({grad_x_inf, cos_x_inf, bound_grad_x_inf, 1.0});
                scale_u = std::max({grad_u_inf, cos_u_inf, bound_grad_u_inf, 1.0});
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
        max_g_pos_  = 0.0;  // max positive constraint violation
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

        int worst_cons_k = -1, worst_cons_j = -1;
        for (int k = 0; k <= N; ++k) {
            // ── 1. Primal feasibility ────────────────────────
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
                    if (viol > cons_viol_) {
                        cons_viol_ = viol;
                        worst_cons_k = k;
                        worst_cons_j = j;
                    }
                    if (viol > primal_inf_) primal_inf_ = viol;
                    // Track max positive constraint violation (Phase 2)
                    if (g_val[j] > max_g_pos_) max_g_pos_ = g_val[j];
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

                // Add bound barrier gradient: -mu/dL + mu/dU
                double bound_grad_x_inf = 0.0, bound_grad_u_inf = 0.0;
                if (prob_->n_bound_x > 0) {
                    for (int i = 0; i < NX; ++i) {
                        double dL = s[k].x[i] - prob_->x_lb[i];
                        double dU = prob_->x_ub[i] - s[k].x[i];
                        double bg = 0.0;
                        if (dL > 1e-14) bg -= mu_ / dL;
                        if (dU > 1e-14) bg += mu_ / dU;
                        lag_x[i] += bg;
                        bound_grad_x_inf = std::max(bound_grad_x_inf, std::fabs(bg));
                    }
                }
                if (prob_->n_bound_u > 0 && k < N) {
                    for (int i = 0; i < NU; ++i) {
                        double dL = s[k].u[i] - prob_->u_lb[i];
                        double dU = prob_->u_ub[i] - s[k].u[i];
                        double bg = 0.0;
                        if (dL > 1e-14) bg -= mu_ / dL;
                        if (dU > 1e-14) bg += mu_ / dU;
                        lag_u[i] += bg;
                        bound_grad_u_inf = std::max(bound_grad_u_inf, std::fabs(bg));
                    }
                }

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
                double scale_x = std::max({grad_x_phys, clam_x_phys, cos_x_phys, bound_grad_x_inf, 1.0});
                double scale_u = std::max({grad_u_phys, clam_u_phys, cos_u_phys, bound_grad_u_inf, 1.0});
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

        // Log worst constraint location (diagnostic)
        if (params_.verbosity >= 2 && worst_cons_k >= 0 && worst_cons_k < N) {
            Vec<NC> g_worst;
            prob_->constraints->evaluate(s[worst_cons_k].x, s[worst_cons_k].u,
                                         worst_cons_k, g_worst);
            // Compute Cx·dx and Cu·du at worst constraint
            double cxdx = 0.0, cudu = 0.0;
            for (int i = 0; i < NX; ++i)
                cxdx += s[worst_cons_k].Cx(worst_cons_j, i) * riccati_ws_.dx[worst_cons_k][i];
            for (int i = 0; i < NU; ++i)
                cudu += s[worst_cons_k].Cu(worst_cons_j, i) * riccati_ws_.du[worst_cons_k][i];
            printf("  [worst-cons] k=%d j=%d g=%.4e s=%.4e g+s=%.4e lam=%.4e mu/s=%.4e ds=%.4e Cxdx=%.4e Cudu=%.4e\n",
                   worst_cons_k, worst_cons_j,
                   g_worst[worst_cons_j], s[worst_cons_k].s[worst_cons_j],
                   g_worst[worst_cons_j] + s[worst_cons_k].s[worst_cons_j],
                   s[worst_cons_k].lambda[worst_cons_j],
                   mu_ / (s[worst_cons_k].s[worst_cons_j] + 1e-14),
                   ds_[worst_cons_k][worst_cons_j],
                   cxdx, cudu);
            // Find max/min λ/s constraint
            double max_ls = 0.0, min_ls = 1e100;
            int max_k = -1, max_j = -1, min_k = -1, min_j = -1;
            for (int kk = 0; kk <= N; ++kk)
                for (int jj = 0; jj < NC; ++jj) {
                    double ls = s[kk].lambda[jj] / (s[kk].s[jj] + 1e-20);
                    if (ls > max_ls) { max_ls = ls; max_k = kk; max_j = jj; }
                    if (ls < min_ls) { min_ls = ls; min_k = kk; min_j = jj; }
                }
            printf("  [bar-cond] max_l/s=%.2e at k=%d j=%d (g=%.3e s=%.3e lam=%.3e) min_l/s=%.2e at k=%d j=%d\n",
                   max_ls, max_k, max_j, s[max_k].d[max_j], s[max_k].s[max_j], s[max_k].lambda[max_j],
                   min_ls, min_k, min_j);
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
        res.max_riccati_u_scaled = 0.0;
        res.max_qu_reconstruction_err = 0.0;
        res.worst_qu_stage = -1;
        res.max_forward_dyn_err = 0.0;
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

            // ── Direct qu comparison: qu_tilde vs riccati_stages_.qu ──
            if (k < N) {
                double max_qu_diff = 0.0;
                for (int i = 0; i < NU; ++i) {
                    double diff = std::fabs(qu_tilde[i] - rs_stages[k].qu[i]);
                    if (diff > max_qu_diff) max_qu_diff = diff;
                }
                if (max_qu_diff > res.max_qu_reconstruction_err) {
                    res.max_qu_reconstruction_err = max_qu_diff;
                    res.worst_qu_stage = k;
                }

                // ── Forward pass dynamics residual check ──────────────
                // Verify dx_{k+1} = A·dx + B·du + c (should be exact from forward pass)
                double max_dyn_fwd_err = 0.0;
                for (int i = 0; i < NX; ++i) {
                    double ax_bu_c = 0.0;
                    for (int j = 0; j < NX; ++j) ax_bu_c += rs.A(i,j) * dx_k[j];
                    for (int j = 0; j < NU; ++j) ax_bu_c += rs.B(i,j) * du_k[j];
                    ax_bu_c += rs.c[i];
                    double err = std::fabs(ax_bu_c - dx_k1[i]);
                    if (err > max_dyn_fwd_err) max_dyn_fwd_err = err;
                }
                if (max_dyn_fwd_err > res.max_forward_dyn_err) {
                    res.max_forward_dyn_err = max_dyn_fwd_err;
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
                    // Track SCALED residual (before unscaling) for Schur comparison
                    double ar_scaled = std::fabs(res_i);
                    if (ar_scaled > res.max_riccati_u_scaled) {
                        res.max_riccati_u_scaled = ar_scaled;
                    }
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

            // ── Variable bounds barrier: μ/d² on diagonal ────────────────
            // This is the key structural fix: bounds like |u| ≤ 1 are handled
            // as log-barrier on variables directly (matching IPOPT's x_L/x_U),
            // NOT as explicit constraints with slacks. The barrier curvature
            // μ/d² is DIAGONAL and goes directly onto Quu/Qxx — no coupling
            // with the constraint Jacobian, eliminating slack-primal
            // desynchronization that causes dual variable blowup.
            if (prob_->n_bound_u > 0 && k < N) {
                for (int i = 0; i < NU; ++i) {
                    double dL = s[k].u[i] - prob_->u_lb[i];
                    double dU = prob_->u_ub[i] - s[k].u[i];
                    if (dL > 1e-14) quu_work_[k](i, i) += mu_ / (dL * dL);
                    if (dU > 1e-14) quu_work_[k](i, i) += mu_ / (dU * dU);
                }
            }
            if (prob_->n_bound_x > 0) {
                for (int i = 0; i < NX; ++i) {
                    double dL = s[k].x[i] - prob_->x_lb[i];
                    double dU = prob_->x_ub[i] - s[k].x[i];
                    if (dL > 1e-14) qxx_work_[k](i, i) += mu_ / (dL * dL);
                    if (dU > 1e-14) qxx_work_[k](i, i) += mu_ / (dU * dU);
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
    //  Constraint-normal correction (Fix C)
    //  Replaces the Newton du component along violated constraint normals
    //  with a constraint-repair step when Cu·du > 0 (Newton worsens constraint).
    //  Applied in SCALED space before recover_inequality_steps.
    // ═════════════════════════════════════════════════════════════════════

    void apply_constraint_normal_correction() {
        if (!prob_->constraints) return;
        // Only fire after 5+ consecutive iterations with FTB α_p < 0.10
        if (low_ftb_count_ < 5) return;
        const int N = HORIZON;
        Stage* s = prob_->stages;

        // Find the FTB bottleneck: stage/constraint with smallest s/|ds|
        // This is the constraint that actually limits the step size.
        double worst_ratio = 1e100;
        int bkw = -1, bjw = -1;
        for (int k = 0; k <= N; ++k) {
            for (int j = 0; j < NC; ++j) {
                double sv = s[k].s[j];
                if (sv < 1e-14) continue;
                // Estimate ds = -(g+s) - Cx·dx - Cu·du
                double g = s[k].d[j];
                if (g >= -1e-10) continue;  // not violated
                double Cx_dx = 0.0;
                for (int i = 0; i < NX; ++i)
                    Cx_dx += s[k].Cx(j, i) * riccati_ws_.dx[k][i];
                double Cu_du = 0.0;
                if (k < N)
                    for (int i = 0; i < NU; ++i)
                        Cu_du += s[k].Cu(j, i) * riccati_ws_.du[k][i];
                double ds = -(g + sv) - Cx_dx - Cu_du;
                double ratio = (ds < -1e-14) ? sv / (-ds) : 1e100;
                if (ratio < worst_ratio) {
                    worst_ratio = ratio;
                    bkw = k;
                    bjw = j;
                }
            }
        }
        if (bkw < 0) return;  // no bottleneck found
        // If bottleneck is at terminal stage (no control), correct at k=N-1
        if (bkw >= N) bkw = N - 1;

        // At the bottleneck: Cu·du > 0 → Newton worsens constraint.
        // Augment the costate p[bkw+1] with a constraint-repair force
        // so the next Riccati solve naturally produces a better du.
        // Force: F = -Cu^T · η_repair / B^T (mapped through dynamics)
        // Simpler: directly correct du at the bottleneck stage.
        double Cu_du = 0.0;
        for (int i = 0; i < NU; ++i)
            Cu_du += s[bkw].Cu(bjw, i) * riccati_ws_.du[bkw][i];
        if (Cu_du <= 0) return;  // Newton already OK at bottleneck

        double g = s[bkw].d[bjw];
        double sv = s[bkw].s[bjw];
        double Cu_sq = 0.0;
        for (int i = 0; i < NU; ++i)
            Cu_sq += s[bkw].Cu(bjw, i) * s[bkw].Cu(bjw, i);
        if (Cu_sq < 1e-30) return;

        // Conservative correction: neutralize the worsening component
        // Target: new Cu·du = -(g+s) · 0.5  (reduce half the violation)
        double target_Cu_du = (g + sv) * 0.5;  // negative
        double alpha_n = (Cu_du - target_Cu_du) / Cu_sq;

        // Cap correction to fraction of |du| to avoid destroying Newton direction
        double du_norm = 0.0;
        for (int i = 0; i < NU; ++i)
            du_norm += riccati_ws_.du[bkw][i] * riccati_ws_.du[bkw][i];
        du_norm = std::sqrt(du_norm);
        double max_alpha = 0.5 * du_norm / std::sqrt(Cu_sq);
        alpha_n = std::min(alpha_n, max_alpha);
        if (alpha_n <= 0) return;

        for (int i = 0; i < NU; ++i)
            riccati_ws_.du[bkw][i] -= alpha_n * s[bkw].Cu(bjw, i);

        double corr_mag = alpha_n * std::sqrt(Cu_sq);
        double new_Cu_du = 0.0;
        for (int i = 0; i < NU; ++i)
            new_Cu_du += s[bkw].Cu(bjw, i) * riccati_ws_.du[bkw][i];

        if (params_.verbosity >= 2) {
            printf("  [corr] bottleneck k=%d j=%d g=%+.2e s=%.2e |"
                   " Cu·du: %+.2e→%+.2e |Δdu|=%.2e α_p_prev=%.3f\n",
                   bkw, bjw, g, sv, Cu_du, new_Cu_du, corr_mag, last_alpha_p_);
        }
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
            // Slack/lambda FTB: only when constraints exist
            if (prob_->constraints) {
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
            // Bound FTB: limit primal step so u/x stay within bounds
            if (prob_->n_bound_u > 0 && k < N) {
                for (int i = 0; i < NU; ++i) {
                    double du_i = riccati_ws_.du[k][i];
                    double dL = s.u[i] - prob_->u_lb[i];
                    double dU = prob_->u_ub[i] - s.u[i];
                    if (dL > 1e-14 && du_i < -1e-16) {
                        double a = -params_.tau * dL / du_i;
                        if (a < alpha) alpha = a;
                    }
                    if (dU > 1e-14 && du_i > 1e-16) {
                        double a = params_.tau * dU / du_i;
                        if (a < alpha) alpha = a;
                    }
                }
            }
            if (prob_->n_bound_x > 0) {
                for (int i = 0; i < NX; ++i) {
                    double dx_i = riccati_ws_.dx[k][i];
                    double dL = s.x[i] - prob_->x_lb[i];
                    double dU = prob_->x_ub[i] - s.x[i];
                    if (dL > 1e-14 && dx_i < -1e-16) {
                        double a = -params_.tau * dL / dx_i;
                        if (a < alpha) alpha = a;
                    }
                    if (dU > 1e-14 && dx_i > 1e-16) {
                        double a = params_.tau * dU / dx_i;
                        if (a < alpha) alpha = a;
                    }
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
            // Update bound multipliers: z = mu / d (always consistent)
            if (prob_->n_bound_u > 0 && k < N) {
                for (int i = 0; i < NU; ++i) {
                    double dL = s[k].u[i] - prob_->u_lb[i];
                    double dU = prob_->u_ub[i] - s[k].u[i];
                    s[k].z_L_u[i] = (dL > 1e-14) ? mu_ / dL : 0.0;
                    s[k].z_U_u[i] = (dU > 1e-14) ? mu_ / dU : 0.0;
                }
            }
            if (prob_->n_bound_x > 0) {
                for (int i = 0; i < NX; ++i) {
                    double dL = s[k].x[i] - prob_->x_lb[i];
                    double dU = prob_->x_ub[i] - s[k].x[i];
                    s[k].z_L_x[i] = (dL > 1e-14) ? mu_ / dL : 0.0;
                    s[k].z_U_x[i] = (dU > 1e-14) ? mu_ / dU : 0.0;
                }
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
        double max_sl_over_mu = 0.0, min_sl_over_mu = 1e100;  // centrality: sλ/μ
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
                // Centrality: sλ/μ (should be ≈ 1 on central path)
                double sl_mu = s[k].s[j] * s[k].lambda[j] / std::max(mu_, 1e-14);
                if (sl_mu > max_sl_over_mu) max_sl_over_mu = sl_mu;
                if (sl_mu < min_sl_over_mu) min_sl_over_mu = sl_mu;
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
            const char* phase_log = (barrier_strategy_.phase()
                == BarrierUpdateStrategy::Phase::INFEASIBILITY) ? "A" : "B";
            const char* rest_log = barrier_strategy_.restoration_active()
                ? " RESTORE" : "";
            printf("[iter %3d] mu=%.2e sig=%.3f | "
                   "prim=%.2e(dyn=%.1e cons=%.1e) stat=%.2e compl=%.2e ineq=%.2e | "
                   "ap=%.3f ad=%.3f | phase=%s max_g+=%.3e | "
                   "cond=%.1e reg=%.1e cost=%.4e | "
                   "linKKT=%.2e(%s)%s\n",
                   iter, mu_, sigma,
                   primal_inf_, dyn_defect_, cons_viol_, stat_inf_, compl_inf_, ineq_viol_,
                   alpha_p, alpha_d,
                   phase_log, max_g_pos_,
                   condH, reg_used_, cur_cost,
                   linear_kkt_res_.max_rel_res, linear_kkt_res_.quality_label(),
                   rest_log);

            // ── Compact diagnostic line (verbosity≥1) ────────────
            {
                int ftb_kp = -1, ftb_jp = -1;
                double ftb_ap = 1.0;
                for (int kk = 0; kk <= N; ++kk)
                    for (int jj = 0; jj < NC; ++jj)
                        if (ds_[kk][jj] < -1e-16) {
                            double a = -params_.tau * s[kk].s[jj] / ds_[kk][jj];
                            if (a < ftb_ap) { ftb_ap = a; ftb_kp = kk; ftb_jp = jj; }
                        }
                double bp_s  = (ftb_kp >= 0) ? s[ftb_kp].s[ftb_jp]  : -1;
                double bp_ds = (ftb_kp >= 0) ? ds_[ftb_kp][ftb_jp]   : 0;
                double bp_lam = (ftb_kp >= 0) ? s[ftb_kp].lambda[ftb_jp] : -1;
                printf("  [d] s=[%.2e,%.2e] lam=[%.2e,%.2e] sl/mu=[%.2f,%.2f]"
                       " | ftb: k=%d j=%d s=%.2e ds=%+.2e lam=%.2e\n",
                       min_slack, max_slack, min_lam, max_lam,
                       min_sl_over_mu, max_sl_over_mu,
                       ftb_kp, ftb_jp, bp_s, bp_ds, bp_lam);

                // ── Residual ratio and bottleneck details ────────────
                {
                    double rp = cons_viol_;   // ||g+s||_inf (primal feasibility)
                    double rc = compl_inf_;   // ||sλ - σμ||_inf (complementarity)
                    double ratio = (rc > 1e-30) ? rp / rc : 1e30;
                    double bp_ratio = (bp_s > 0 && bp_ds != 0) ? bp_ds / bp_s : 0;
                    double bp_s_pred = (ftb_kp >= 0) ? bp_s + bp_ds : 0; // predicted s after α=1
                    printf("  [res] rp=%.2e rc=%.2e rp/rc=%.2f"
                           " | bneck: ds/s=%+.2f s(a=1)=%.2e s(a=%.3f)=%.2e\n",
                           rp, rc, ratio,
                           bp_ratio, bp_s_pred,
                           alpha_p, (ftb_kp >= 0) ? bp_s + alpha_p * bp_ds : 0.0);
                }

                // ── ds decomposition at bottleneck constraint ─────────
                // IMPORTANT: Cx/Cu in stages are SCALED (Cx·inv_Lx).
                // Must use SCALED dx/du (from debug_scaled_dx_/du_) to match.
                if (ftb_kp >= 0 && prob_->constraints) {
                    int bk = ftb_kp, bj = ftb_jp;
                    double g_val = s[bk].d[bj];  // g(x,u) stored in d[]
                    double s_val = s[bk].s[bj];
                    double rp_val = g_val + s_val;       // g+s
                    double Cx_dx = 0.0, Cu_du = 0.0;
                    // Use scaled dx/du (consistent with scaled Cx/Cu)
                    for (int i = 0; i < NX; ++i)
                        Cx_dx += s[bk].Cx(bj, i) * debug_scaled_dx_[bk][i];
                    if (bk < N)
                        for (int i = 0; i < NU; ++i)
                            Cu_du += s[bk].Cu(bj, i) * debug_scaled_du_[bk][i];
                    double ds_actual = ds_[bk][bj];
                    double ds_predict = -rp_val - Cx_dx - Cu_du;
                    printf("  [dec] k=%d j=%d | g+s=%+.3e Cx·dx=%+.3e Cu·du=%+.3e"
                           " | ds=%+.3e ds_eq=%+.3e err=%.1e\n",
                           bk, bj, rp_val, Cx_dx, Cu_du,
                           ds_actual, ds_predict,
                           std::fabs(ds_actual - ds_predict));

                    // Also print physical ||du|| at bottleneck
                    double du_inf = 0.0;
                    if (bk < N)
                        for (int i = 0; i < NU; ++i)
                            du_inf = std::max(du_inf, std::fabs(riccati_ws_.du[bk][i]));
                    printf("  [reg] reg=%.1e |du_phys|=%.2e reg*|du|=%.2e"
                           " | ricc_u=%.2e (ricc_u > 10*reg*|du|? %s)\n",
                           reg_used_, du_inf, reg_used_ * du_inf,
                           linear_kkt_res_.max_stat_u_res,
                           (linear_kkt_res_.max_stat_u_res > 10 * reg_used_ * du_inf) ? "YES" : "no");

                    // ── Per-stage η/λ/s table for bottleneck constraint ──
                    // For constraint bj, print g, s, λ, μ/s, η=λ+μ/s, Cu·du at every stage.
                    // Uses SCALED Cu/dx/du (consistent with scaled stages).
                    {
                        printf("  [eta-table] j=%d  k   g          s          lambda     mu/s       eta        Cu_du\n", bj);
                        for (int kk = 0; kk <= N; ++kk) {
                            double g_kk  = s[kk].d[bj];
                            double s_kk  = s[kk].s[bj];
                            double lam_kk = s[kk].lambda[bj];
                            double mu_over_s = (s_kk > 1e-30) ? mu_ / s_kk : 0.0;
                            double eta_kk = lam_kk + mu_over_s;
                            // Cu·du: scaled Cu * scaled du (consistent)
                            double Cu_du_kk = 0.0;
                            if (kk < N) {
                                for (int i = 0; i < NU; ++i)
                                    Cu_du_kk += s[kk].Cu(bj, i) * debug_scaled_du_[kk][i];
                            }
                            printf("  [eta-table]     %2d  %+.3e  %.3e  %.3e  %.3e  %.3e  %+.3e\n",
                                   kk, g_kk, s_kk, lam_kk, mu_over_s, eta_kk, Cu_du_kk);
                        }
                    }

                    // ── Constraint-normal projection diagnostic ──
                    // At the bottleneck stage, compare:
                    //   n^T · Cu^T · η   (constraint force along normal)
                    //   n^T · B^T · ν     (dynamics force along normal)
                    // where n = Cu(bj,:)^T / |Cu(bj,:)| (unit normal in u-space).
                    // This tells whether the dynamics costate genuinely overwhelms
                    // the constraint along the constraint gradient direction, or
                    // whether it's mostly tangent (misleading large norm).
                    {
                        // Physical Cu row for constraint bj at bottleneck stage
                        double Cu_phys[NU];
                        double Cu_norm = 0.0;
                        for (int i = 0; i < NU; ++i) {
                            Cu_phys[i] = s[bk].Cu(bj, i);
                            if (params_.enable_preconditioner)
                                Cu_phys[i] /= prec_.inv_Lu(bk)[i];
                            Cu_norm += Cu_phys[i] * Cu_phys[i];
                        }
                        Cu_norm = std::sqrt(Cu_norm);

                        if (Cu_norm > 1e-30) {
                            // Unit normal: n = Cu^T / |Cu|
                            double n_hat[NU];
                            for (int i = 0; i < NU; ++i)
                                n_hat[i] = Cu_phys[i] / Cu_norm;

                            // η_j at bottleneck (effective dual multiplier)
                            double s_bk = s[bk].s[bj];
                            double lam_bk = s[bk].lambda[bj];
                            double eta_bk = lam_bk + ((s_bk > 1e-30) ? mu_ / s_bk : 0.0);
                            // n^T · Cu^T · η_j = |Cu| · η_j
                            double n_Cut_eta = Cu_norm * eta_bk;

                            // Full Cu^T · η (all constraints at bottleneck stage)
                            double CuT_eta_full[NU] = {};
                            for (int jj = 0; jj < NC; ++jj) {
                                double sj = s[bk].s[jj];
                                double lj = s[bk].lambda[jj];
                                double etaj = lj + ((sj > 1e-30) ? mu_ / sj : 0.0);
                                for (int i = 0; i < NU; ++i) {
                                    double Cu_ij = s[bk].Cu(jj, i);
                                    if (params_.enable_preconditioner)
                                        Cu_ij /= prec_.inv_Lu(bk)[i];
                                    CuT_eta_full[i] += Cu_ij * etaj;
                                }
                            }
                            double n_dot_CuT_eta = 0.0;
                            for (int i = 0; i < NU; ++i)
                                n_dot_CuT_eta += n_hat[i] * CuT_eta_full[i];

                            // B^T · ν at bottleneck stage
                            // ν = p[kk+1] + P[kk+1] · dx[kk+1] (physical)
                            double nu_k[NX];
                            if (bk < N) {
                                for (int i = 0; i < NX; ++i) {
                                    nu_k[i] = riccati_ws_.p[bk + 1][i];
                                    for (int jj = 0; jj < NX; ++jj)
                                        nu_k[i] += riccati_ws_.P[bk + 1](i, jj) * riccati_ws_.dx[bk + 1][jj];
                                }
                            } else {
                                for (int i = 0; i < NX; ++i)
                                    nu_k[i] = riccati_ws_.p[bk][i];
                            }
                            // B^T · ν
                            double Bt_nu[NU] = {};
                            if (bk < N) {
                                for (int i = 0; i < NU; ++i)
                                    for (int m = 0; m < NX; ++m)
                                        Bt_nu[i] += s[bk].B(m, i) * nu_k[m];
                            }
                            double n_dot_Bt_nu = 0.0;
                            double Bt_nu_norm = 0.0;
                            for (int i = 0; i < NU; ++i) {
                                n_dot_Bt_nu += n_hat[i] * Bt_nu[i];
                                Bt_nu_norm += Bt_nu[i] * Bt_nu[i];
                            }
                            Bt_nu_norm = std::sqrt(Bt_nu_norm);

                            // Tangent component of B^T ν
                            double Bt_nu_tangent = std::sqrt(std::max(0.0, Bt_nu_norm*Bt_nu_norm - n_dot_Bt_nu*n_dot_Bt_nu));

                            printf("  [normal-proj] k=%d j=%d |Cu|=%.3e"
                                   " | n·CuTη=%+.3e  n·Btν=%+.3e"
                                   " | Btν_norm=%.3e tangent=%.3e"
                                   " | ratio=%.2f\n",
                                   bk, bj, Cu_norm,
                                   n_dot_CuT_eta, n_dot_Bt_nu,
                                   Bt_nu_norm, Bt_nu_tangent,
                                   (std::fabs(n_dot_CuT_eta) > 1e-30)
                                       ? n_dot_Bt_nu / n_dot_CuT_eta : 0.0);

                            // ── Dynamics residual norm per stage ──
                            // ‖c_k‖ is the amplifier in P·c → costate.
                            // Large ‖c_k‖ at early stages ⇒ Fix B (feasibility restoration).
                            {
                                double max_ck = 0.0;
                                int worst_ck = -1;
                                for (int kk = 0; kk < N; ++kk) {
                                    double cn = 0.0;
                                    for (int i = 0; i < NX; ++i) cn += s[kk].c[i] * s[kk].c[i];
                                    cn = std::sqrt(cn);
                                    if (cn > max_ck) { max_ck = cn; worst_ck = kk; }
                                }
                                printf("  [c-norm] max‖c_k‖=%.3e at k=%d | c[0]=(",
                                       max_ck, worst_ck);
                                for (int i = 0; i < NX; ++i)
                                    printf("%s%.2e", i ? "," : "", s[0].c[i]);
                                printf(") ‖c[0]‖=%.3e\n",
                                       [&](){ double n=0; for(int i=0;i<NX;++i) n+=s[0].c[i]*s[0].c[i]; return std::sqrt(n); }());
                                // Also print ‖P_1·c_0‖ to verify it matches the -280 gap
                                if (N > 0) {
                                    double Pc0[NX] = {};
                                    for (int i = 0; i < NX; ++i)
                                        for (int m = 0; m < NX; ++m)
                                            Pc0[i] += riccati_ws_.P[1](i, m) * s[0].c[m];
                                    double BtPc[NU] = {};
                                    for (int i = 0; i < NU; ++i)
                                        for (int m = 0; m < NX; ++m)
                                            BtPc[i] += s[0].B(m, i) * Pc0[m];
                                    double n_BtPc = 0.0;
                                    for (int i = 0; i < NU; ++i) n_BtPc += n_hat[i] * BtPc[i];
                                    printf("  [c-norm] n·B^T·P_1·c_0 = %+.3e (compare to diff_qx)\n", n_BtPc);
                                }
                            }

                            // ── B^Tν stage-by-stage decomposition ──────────
                            // Two complementary decompositions of n^T·B_0^T·ν_1:
                            //
                            // (A) Full costate: n^T·B_0^T·ν_1 = Σ_k w_k·qx_k
                            //     where qx_k = ∂l/∂x + Cx^T·η_eff (cost+constraint)
                            //     w_1 = B_0·n, w_k = A_2^T·...·A_k^T·B_0·n
                            //
                            // (B) Constraint-force only: Σ_k w_k·(Cx_k^T·η_eff_k)
                            //     isolates how future-stage constraints drive k=0.
                            //
                            // Flow: output at k BEFORE flowing via A_{k+1}^T
                            //   k=1: w=B_0·n (identity flow)
                            //   k=2: w=A_2^T·B_0·n
                            //   k=3: w=A_2^T·A_3^T·B_0·n
                            if (bk == 0 && bk < N) {
                                printf("  [Ctv-dec] Stage decomp of n·B_0^T·ν_1\n");
                                double bn[NX] = {};
                                for (int m = 0; m < NX; ++m)
                                    for (int ii = 0; ii < NU; ++ii)
                                        bn[m] += s[0].B(m, ii) * n_hat[ii];

                                double w_fl[NX];
                                for (int m = 0; m < NX; ++m) w_fl[m] = bn[m];
                                double total_qx = 0.0, total_cx = 0.0;
                                double smu_loc = sigma_ * mu_;

                                for (int kk = 1; kk <= N; ++kk) {
                                    // η_eff_j = λ_j + (σμ + λ_j·(g_j+s_j)) / s_j
                                    // (matches qx construction in build_kkt_rhs)
                                    double Cxt_eta[NX] = {};
                                    for (int jj = 0; jj < NC; ++jj) {
                                        double sj = s[kk].s[jj];
                                        if (sj < 1e-14) continue;
                                        double lj = s[kk].lambda[jj];
                                        double gj = s[kk].d[jj] + sj;
                                        double eta = lj + (smu_loc + lj * gj) / sj;
                                        for (int i = 0; i < NX; ++i)
                                            Cxt_eta[i] += s[kk].Cx(jj, i) * eta;
                                    }

                                    // Full qx contribution
                                    double qdot = 0.0;
                                    for (int i = 0; i < NX; ++i)
                                        qdot += w_fl[i] * s[kk].qx[i];
                                    // Constraint-force contribution
                                    double cdot = 0.0;
                                    for (int i = 0; i < NX; ++i)
                                        cdot += w_fl[i] * Cxt_eta[i];

                                    double wnorm = 0.0;
                                    for (int i = 0; i < NX; ++i) wnorm += w_fl[i]*w_fl[i];
                                    wnorm = std::sqrt(wnorm);

                                    if (std::fabs(qdot) > 1e-6 || std::fabs(cdot) > 1e-6 || kk <= 5 || kk >= N-1)
                                        printf("  [Ctv-dec]   k=%2d  qx=%+.3e  CxTη=%+.3e  (cost=%+.3e) |w|=%.2e\n",
                                               kk, qdot, cdot, qdot - cdot, wnorm);

                                    total_qx += qdot;
                                    total_cx += cdot;

                                    // Flow w via A_{k+1}^T for next stage
                                    if (kk < N) {
                                        double w_new[NX] = {};
                                        for (int i = 0; i < NX; ++i)
                                            for (int jj = 0; jj < NX; ++jj)
                                                w_new[i] += s[kk+1].A(jj,i) * w_fl[jj];
                                        for (int i = 0; i < NX; ++i) w_fl[i] = w_new[i];
                                    }
                                }
                                printf("  [Ctv-dec]   SUM_qx=%+.3e  SUM_CxTη=%+.3e  SUM_cost=%+.3e\n",
                                       total_qx, total_cx, total_qx - total_cx);
                                printf("  [Ctv-dec]   direct n·Btν=%+.3e  (diff_qx=%.3e = P·c+Qup·d terms)\n",
                                       n_dot_Bt_nu, n_dot_Bt_nu - total_qx);

                                // Verify: n·B^T·p_1 from Riccati directly
                                {
                                    double p1_n = 0.0;
                                    for (int i = 0; i < NX; ++i) {
                                        double bni = 0.0;
                                        for (int m = 0; m < NU; ++m) bni += s[0].B(i, m) * n_hat[m];
                                        p1_n += bni * riccati_ws_.p[1][i];
                                    }
                                    printf("  [Ctv-dec]   n·B^T·p_1(direct)=%+.3e  SUM_qx=%+.3e  diff=%+.3e\n",
                                           p1_n, total_qx, p1_n - total_qx);
                                }

                                // ── qu decomposition at bottleneck stage ──
                                // From Riccati u-stationarity:
                                //   qu + Cu^T·η_eff + B^T·ν₁ = 0
                                //   qu = qu_cost + Qux·dx + Quu·du  (Hessian×step)
                                // Show: qu_cost·n, (Cu^Tη)·n, (B^Tν)·n
                                {
                                    // qu_cost = ∂l/∂u (pure cost gradient)
                                    double qu_cost_dot_n = 0.0;
                                    for (int i = 0; i < NU; ++i)
                                        qu_cost_dot_n += s[0].qu[i] * n_hat[i];
                                    // Cu^T·η_eff at k=0
                                    double CuT_eta_dot_n = 0.0;
                                    for (int jj = 0; jj < NC; ++jj) {
                                        double sj = s[0].s[jj];
                                        if (sj < 1e-14) continue;
                                        double lj = s[0].lambda[jj];
                                        double gj = s[0].d[jj] + sj;
                                        double eta = lj + (smu_loc + lj * gj) / sj;
                                        for (int i = 0; i < NU; ++i) {
                                            double Cu_ij = s[0].Cu(jj, i);
                                            CuT_eta_dot_n += n_hat[i] * Cu_ij * eta;
                                        }
                                    }
                                    printf("  [qu-dec] k=0: qu_cost·n=%+.3e  CuTη·n=%+.3e  Btν·n=%+.3e  |SUM·n=%+.3e|\n",
                                           qu_cost_dot_n, CuT_eta_dot_n, n_dot_Bt_nu,
                                           qu_cost_dot_n + CuT_eta_dot_n + n_dot_Bt_nu);
                                    // Also show: qu_cost, Hessian×dx/du, full qu
                                    double Hess_dx_du[NU] = {};
                                    for (int i = 0; i < NU; ++i) {
                                        for (int m = 0; m < NX; ++m)
                                            Hess_dx_du[i] += s[0].Qux(i, m) * debug_scaled_dx_[0][m];
                                        for (int m = 0; m < NU; ++m)
                                            Hess_dx_du[i] += s[0].Quu(i, m) * debug_scaled_du_[0][m];
                                    }
                                    double Hess_n = 0.0;
                                    for (int i = 0; i < NU; ++i) Hess_n += Hess_dx_du[i] * n_hat[i];
                                    printf("  [qu-dec]   qu_cost·n=%+.3e  H·dz·n=%+.3e  qu_total·n=%+.3e  (= -Btν·n? %.3e)\n",
                                           qu_cost_dot_n, Hess_n, qu_cost_dot_n + Hess_n,
                                           qu_cost_dot_n + Hess_n + n_dot_Bt_nu);
                                }
                            }
                        }
                    }
                }

                // ── Sorted α limits (top-5 smallest) ───────────────
                {
                    struct AlphaEntry { double alpha; int k, j; };
                    AlphaEntry entries[(HORIZON + 1) * NC];
                    int n_entries = 0;
                    for (int kk = 0; kk <= N; ++kk)
                        for (int jj = 0; jj < NC; ++jj)
                            if (ds_[kk][jj] < -1e-16) {
                                entries[n_entries].alpha = -params_.tau * s[kk].s[jj] / ds_[kk][jj];
                                entries[n_entries].k = kk;
                                entries[n_entries].j = jj;
                                ++n_entries;
                            }
                    // Simple insertion sort (n_entries is small)
                    for (int a = 1; a < n_entries; ++a)
                        for (int b = a; b > 0 && entries[b].alpha < entries[b-1].alpha; --b)
                            std::swap(entries[b], entries[b-1]);
                    // Print top-5
                    printf("  [alpha] top-5 limiting constraints:");
                    int top = std::min(n_entries, 5);
                    for (int a = 0; a < top; ++a)
                        printf(" k%d_j%d:%.4f", entries[a].k, entries[a].j, entries[a].alpha);
                    if (n_entries > 5)
                        printf(" ...(%d total)", n_entries);
                    printf("\n");
                }
            }

            if (params_.verbosity >= 2) {
                printf("  [diag] s=[%.2e,%.2e] lam=[%.2e,%.2e] "
                       "lam/s=[%.2e,%.2e] sl/mu=[%.1f,%.1f]\n",
                       min_slack, max_slack,
                       min_lam, max_lam,
                       min_ratio, max_ratio,
                       min_sl_over_mu, max_sl_over_mu);
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
    //  Adaptive slack floor (Phase 3):
    //    floor_s = max(mu_frac·μ, c_floor·max_g_pos_)
    //  Ensures slacks never collapse below constraint violation level.
    // ═════════════════════════════════════════════════════════════════════

    void sz_complement(double mu_frac = 1.0) {
        const int N = HORIZON;
        Stage* s = prob_->stages;
        const double threshold = mu_frac * mu_;
        // Adaptive floor (Phase 3): only for feasible constraints (g <= 0).
        // For violated constraints (g > 0), slack must be allowed to stay
        // small so the barrier can push x toward feasibility.  Applying
        // a large floor to violated constraints would double |g+s| and
        // make infeasibility worse.
        const double adaptive_floor = params_.c_floor * max_g_pos_;
        const double base_floor = std::max(threshold, adaptive_floor);

        for (int k = 0; k <= N; ++k) {
            for (int j = 0; j < NC; ++j) {
                double gj = s[k].d[j];
                // For violated constraints: floor tracks violation magnitude
                // (c_floor * gj) so FTB doesn't strangle the step.
                // For feasible constraints: adaptive floor prevents collapse.
                double floor_s = (gj > 0.0)
                    ? std::max(threshold, params_.c_floor * gj)
                    : base_floor;
                s[k].s[j]      = std::max({-gj, s[k].s[j], floor_s});
                s[k].lambda[j] = threshold / s[k].s[j];
            }
        }
    }

    // ═════════════════════════════════════════════════════════════════════
    //  Per-iteration diagnostic computation (Sections A–G)
    //  Called once per IPM iteration after corrector step is available.
    //  Does NOT modify solver state — read-only instrumentation.
    // ═════════════════════════════════════════════════════════════════════

    IterDiag compute_iter_diagnostics() const {
        IterDiag d;
        const int N = HORIZON;
        const Stage* s = prob_->stages;

        // ── (A) Primal residual decomposition ──
        // Nonlinear dynamics defect (physical space)
        double dyn_2_sum = 0.0;
        for (int k = 0; k < N; ++k) {
            double dc_max = 0.0;
            for (int i = 0; i < NX; ++i) {
                double c_phys = stages_scaled_ ? s[k].c[i] * prec_.inv_Lx(k+1)[i] : s[k].c[i];
                dc_max = std::max(dc_max, std::fabs(c_phys));
                dyn_2_sum += c_phys * c_phys;
            }
            d.r_dyn_inf = std::max(d.r_dyn_inf, dc_max);
        }
        d.r_dyn_2 = std::sqrt(dyn_2_sum);

        // Nonlinear inequality violation
        double ineq_active_sum = 0.0;
        double ineq_thresh = 1e-6;  // active set threshold
        if (prob_->constraints) {
            for (int k = 0; k <= N; ++k) {
                Vec<NC> g_val;
                if (k < N) prob_->constraints->evaluate(s[k].x, s[k].u, k, g_val);
                else prob_->constraints->evaluate_terminal(s[k].x, g_val);
                for (int j = 0; j < NC; ++j) {
                    double viol = g_val[j];  // g(x,u) <= 0 means violation when g > 0
                    d.r_ineq_inf = std::max(d.r_ineq_inf, viol);
                    if (std::fabs(viol) > ineq_thresh) {
                        ineq_active_sum += viol * viol;
                        d.n_active++;
                    }
                }
            }
        }
        d.r_ineq_active_2 = std::sqrt(ineq_active_sum);
        d.r_ratio = d.r_dyn_inf / (std::fabs(d.r_ineq_inf) + 1e-12);

        // ── (B) Dual / stationarity breakdown ──
        d.grad_L_x = stat_inf_;
        // Use stat_breakdown_ which has 6 components:
        // [0]=grad_x, [1]=grad_u, [2]=Cx^T*lam, [3]=Cu^T*lam, [4]=costate_x, [5]=costate_u
        d.grad_cost_x = stat_breakdown_[0];
        d.grad_cost_u = stat_breakdown_[1];
        d.ineq_dual_x = stat_breakdown_[2];
        d.ineq_dual_u = stat_breakdown_[3];
        d.dyn_dual_x  = stat_breakdown_[4];
        d.dyn_dual_u  = stat_breakdown_[5];
        double dyn_term  = d.dyn_dual_x + d.dyn_dual_u;
        double ineq_term = d.ineq_dual_x + d.ineq_dual_u;
        d.stat_dom_ratio = dyn_term / (ineq_term + 1e-14);

        // ── (C) Barrier coupling ──
        d.mu = mu_;
        d.compl_min = 1e100; d.compl_max = 0.0;
        d.slack_min = 1e100; d.slack_max = 0.0;
        d.barrier_obj = 0.0;
        for (int k = 0; k <= N; ++k) {
            for (int j = 0; j < NC; ++j) {
                double sj = s[k].s[j];
                double lj = s[k].lambda[j];
                double sl = sj * lj;
                d.compl_min = std::min(d.compl_min, sl);
                d.compl_max = std::max(d.compl_max, sl);
                d.slack_min = std::min(d.slack_min, sj);
                d.slack_max = std::max(d.slack_max, sj);
                if (sj > 1e-20) d.barrier_obj -= mu_ * std::log(sj);
            }
        }

        // ── (D) KKT system block norms + step norms ──
        for (int k = 0; k <= N; ++k) {
            // Hessian norm (Qxx + barrier Hessian contribution)
            double H_F = 0.0;
            for (int i = 0; i < NX; ++i)
                for (int j = 0; j < NX; ++j)
                    H_F += s[k].Qxx(i,j) * s[k].Qxx(i,j);
            if (prob_->constraints) {
                for (int jj = 0; jj < NC; ++jj) {
                    double w = s[k].lambda[jj] / std::max(s[k].s[jj], 1e-20);
                    for (int i = 0; i < NX; ++i)
                        for (int j = 0; j < NX; ++j)
                            H_F += w * w * s[k].Cx(jj,i) * s[k].Cx(jj,j)
                                   * s[k].Cx(jj,i) * s[k].Cx(jj,j);  // approximate
                }
            }
            d.norm_H_F = std::max(d.norm_H_F, std::sqrt(H_F));

            // Dynamics Jacobian norm
            if (k < N) {
                double A_F = 0.0;
                for (int i = 0; i < NX; ++i)
                    for (int j = 0; j < NX; ++j)
                        A_F += s[k].A(i,j) * s[k].A(i,j);
                d.norm_Adyn_F = std::max(d.norm_Adyn_F, std::sqrt(A_F));
            }

            // Constraint Jacobian norm
            if (prob_->constraints) {
                double C_F = 0.0;
                for (int jj = 0; jj < NC; ++jj) {
                    for (int i = 0; i < NX; ++i) C_F += s[k].Cx(jj,i) * s[k].Cx(jj,i);
                    if (k < N)
                        for (int i = 0; i < NU; ++i) C_F += s[k].Cu(jj,i) * s[k].Cu(jj,i);
                }
                d.norm_Aineq_F = std::max(d.norm_Aineq_F, std::sqrt(C_F));
            }

            // Step norms
            for (int i = 0; i < NX; ++i) {
                d.dx_inf = std::max(d.dx_inf, std::fabs(riccati_ws_.dx[k][i]));
                d.dp_inf = std::max(d.dp_inf, std::fabs(riccati_ws_.p[k][i]));
            }
            if (k < N)
                for (int i = 0; i < NU; ++i)
                    d.du_inf = std::max(d.du_inf, std::fabs(riccati_ws_.du[k][i]));
        }
        // Slack and dual steps
        for (int k = 0; k <= N; ++k) {
            for (int j = 0; j < NC; ++j) {
                d.ds_inf   = std::max(d.ds_inf, std::fabs(ds_[k][j]));
                d.dlam_inf = std::max(d.dlam_inf, std::fabs(dlambda_[k][j]));
            }
        }

        // ── (E) Step coupling: linearized residuals after step ──
        // s_dyn = ||A*dx + B*du + c - dx_{k+1}||  (should be ~linear KKT dyn residual)
        // s_ineq = ||Cx*dx + Cu*du + g + s + ds||  (should be ~linear KKT feas residual)
        for (int k = 0; k < N; ++k) {
            for (int i = 0; i < NX; ++i) {
                double rd = s[k].c[i];  // c_k (scaled if stages_scaled_)
                for (int j = 0; j < NX; ++j) rd += s[k].A(i,j) * riccati_ws_.dx[k][j];
                for (int j = 0; j < NU; ++j) rd += s[k].B(i,j) * riccati_ws_.du[k][j];
                rd -= riccati_ws_.dx[k+1][i];
                if (params_.enable_preconditioner) rd *= prec_.inv_Lx(k+1)[i];
                d.s_dyn_inf = std::max(d.s_dyn_inf, std::fabs(rd));
            }
        }
        if (prob_->constraints) {
            for (int k = 0; k <= N; ++k) {
                for (int j = 0; j < NC; ++j) {
                    double ri = s[k].d[j] + s[k].s[j] + ds_[k][j];
                    for (int i = 0; i < NX; ++i) ri += s[k].Cx(j,i) * riccati_ws_.dx[k][i];
                    if (k < N)
                        for (int i = 0; i < NU; ++i) ri += s[k].Cu(j,i) * riccati_ws_.du[k][i];
                    d.s_ineq_inf = std::max(d.s_ineq_inf, std::fabs(ri));
                }
            }
        }
        // Sign correlation: does step improve both or compete?
        // Compare predicted reduction direction for dyn vs ineq
        d.sign_corr = 0;
        if (d.s_dyn_inf < linear_kkt_res_.max_dyn_res * 0.99 &&
            d.s_ineq_inf < linear_kkt_res_.max_feas_res * 0.99)
            d.sign_corr = +1;  // both improve
        else if (d.s_dyn_inf > linear_kkt_res_.max_dyn_res * 1.01 &&
                 d.s_ineq_inf > linear_kkt_res_.max_feas_res * 1.01)
            d.sign_corr = -1;  // both worsen (competing)

        // ── (F) Line search — filled externally after LS ──
        d.alpha_p = 0; d.alpha_d = 0;  // placeholders
        d.theta_dyn = 0; d.theta_ineq = 0;
        {
            // Compute theta decomposition at current point
            for (int k = 0; k < N; ++k)
                for (int i = 0; i < NX; ++i) {
                    double c_phys = stages_scaled_ ? s[k].c[i] * prec_.inv_Lx(k+1)[i] : s[k].c[i];
                    d.theta_dyn += std::fabs(c_phys);
                }
            if (prob_->constraints) {
                for (int k = 0; k <= N; ++k)
                    for (int j = 0; j < NC; ++j)
                        d.theta_ineq += std::fabs(s[k].d[j] + s[k].s[j]);
            }
        }

        // ── (G) Scaling diagnostics ──
        if (params_.enable_preconditioner) {
            d.Lx_min = 1e100; d.Lx_max = 0.0;
            d.Lu_min = 1e100; d.Lu_max = 0.0;
            for (int k = 0; k <= N; ++k) {
                for (int i = 0; i < NX; ++i) {
                    double Lx = 1.0 / std::max(prec_.inv_Lx(k)[i], 1e-30);
                    d.Lx_min = std::min(d.Lx_min, Lx);
                    d.Lx_max = std::max(d.Lx_max, Lx);
                }
                if (k < N)
                    for (int i = 0; i < NU; ++i) {
                        double Lu = 1.0 / std::max(prec_.inv_Lu(k)[i], 1e-30);
                        d.Lu_min = std::min(d.Lu_min, Lu);
                        d.Lu_max = std::max(d.Lu_max, Lu);
                    }
            }
            double global_max = std::max(d.Lx_max, d.Lu_max);
            double global_min = std::min(d.Lx_min, d.Lu_min);
            d.scale_ratio = global_max / (global_min + 1e-30);
        }

        return d;
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
            // Slack/lambda FTB: only when constraints exist
            if (prob_->constraints) {
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
            // Bound FTB: limit primal step so u/x stay within bounds
            if (prob_->n_bound_u > 0 && k < HORIZON) {
                for (int i = 0; i < NU; ++i) {
                    double du_i = riccati_ws_.du[k][i];
                    double dL = s[k].u[i] - prob_->u_lb[i];
                    double dU = prob_->u_ub[i] - s[k].u[i];
                    if (dL > 1e-14 && du_i < -1e-16) {
                        double a = -params_.tau * dL / du_i;
                        if (a < ap) ap = a;
                    }
                    if (dU > 1e-14 && du_i > 1e-16) {
                        double a = params_.tau * dU / du_i;
                        if (a < ap) ap = a;
                    }
                }
            }
            if (prob_->n_bound_x > 0) {
                for (int i = 0; i < NX; ++i) {
                    double dx_i = riccati_ws_.dx[k][i];
                    double dL = s[k].x[i] - prob_->x_lb[i];
                    double dU = prob_->x_ub[i] - s[k].x[i];
                    if (dL > 1e-14 && dx_i < -1e-16) {
                        double a = -params_.tau * dL / dx_i;
                        if (a < ap) ap = a;
                    }
                    if (dU > 1e-14 && dx_i > 1e-16) {
                        double a = params_.tau * dU / dx_i;
                        if (a < ap) ap = a;
                    }
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
            }
            // When constraints = nullptr (variable bounds only), no slack contribution to theta.
            // This matches compute_theta() which also omits slack terms in this case.

            // Barrier-adjusted objective (only for explicit constraints)
            if (sv->prob_->constraints) {
                double barrier_term = 0.0;
                const double mu = sv->mu_;
                for (int k = 0; k <= N; ++k)
                    for (int j = 0; j < NC; ++j) {
                        double s_t = s[k].s[j] + alpha * sv->ds_[k][j];
                        if (s_t > 1e-20)
                            barrier_term += std::log(s_t);
                    }
                out_phi -= mu * barrier_term;
            }

            return std::isfinite(out_theta) && std::isfinite(out_phi);
        }

        double current_theta() const override {
            return solver_->compute_theta();
        }

        double current_phi() const override {
            double phi = solver_->compute_objective();
            // Barrier-adjusted objective (only for explicit constraints)
            if (solver_->prob_->constraints) {
                const double mu = solver_->mu_;
                const int N = HORIZON;
                for (int k = 0; k <= N; ++k)
                    for (int j = 0; j < NC; ++j)
                        if (solver_->prob_->stages[k].s[j] > 1e-20)
                            phi -= mu * std::log(solver_->prob_->stages[k].s[j]);
            }
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

            // Save theta0 BEFORE any modifications (compute_theta uses stages.d)
            const double theta0 = sv->compute_theta();

            // ── Save original search direction and constraint values ──
            for (int k = 0; k <= N; ++k) {
                sv->soc_save_dx_[k]   = sv->riccati_ws_.dx[k];
                sv->soc_save_ds_[k]   = sv->ds_[k];
                sv->soc_save_dlam_[k] = sv->dlambda_[k];
                sv->soc_save_d_[k]    = s[k].d;  // save original constraint values
                if (k < N) {
                    sv->soc_save_du_[k] = sv->riccati_ws_.du[k];
                    sv->soc_save_c_orig_[k] = s[k].c;  // save original dynamics defects
                }
            }

            // ── Compute trial dynamics defect and constraints at (base + α·dir) ──
            for (int k = 0; k < N; ++k) {
                Vec<NX> xk_t   = s[k].x;
                Vec<NU> uk_t   = s[k].u;
                Vec<NX> xkp1_t = s[k+1].x;
                for (int i = 0; i < NX; ++i) xk_t[i]   += alpha * sv->soc_save_dx_[k][i];
                for (int i = 0; i < NU; ++i) uk_t[i]   += alpha * sv->soc_save_du_[k][i];
                for (int i = 0; i < NX; ++i) xkp1_t[i] += alpha * sv->soc_save_dx_[k+1][i];

                // Dynamics defect at trial point
                Vec<NX> fk;
                sv->prob_->dynamics->discrete_step(xk_t, uk_t, sv->prob_->dt, fk, k);
                for (int i = 0; i < NX; ++i)
                    sv->soc_save_c_[k][i] = fk[i] - xkp1_t[i];

                // Path constraints at trial point (corrects nonlinear constraint defect)
                if (sv->prob_->constraints) {
                    Vec<NC> g_trial;
                    sv->prob_->constraints->evaluate(xk_t, uk_t, k, g_trial);
                    for (int j = 0; j < NC; ++j)
                        s[k].d[j] = g_trial[j];  // replace with trial constraint values
                }
            }

            // Terminal constraints at trial point
            if (sv->prob_->constraints) {
                Vec<NX> xN_t = s[N].x;
                for (int i = 0; i < NX; ++i)
                    xN_t[i] += alpha * sv->soc_save_dx_[N][i];
                Vec<NC> g_term_trial;
                sv->prob_->constraints->evaluate_terminal(xN_t, g_term_trial);
                for (int j = 0; j < NC; ++j)
                    s[N].d[j] = g_term_trial[j];
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
                // Restore original direction, constraint values, and dynamics defects
                for (int k = 0; k <= N; ++k) {
                    sv->riccati_ws_.dx[k] = sv->soc_save_dx_[k];
                    sv->ds_[k] = sv->soc_save_ds_[k];  sv->dlambda_[k] = sv->soc_save_dlam_[k];
                    s[k].d = sv->soc_save_d_[k];
                    if (k < N) {
                        sv->riccati_ws_.du[k] = sv->soc_save_du_[k];
                        s[k].c = sv->soc_save_c_orig_[k];
                    }
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
                    s[k].d = sv->soc_save_d_[k];
                    if (k < N) {
                        sv->riccati_ws_.du[k] = sv->soc_save_du_[k];
                        s[k].c = sv->soc_save_c_orig_[k];
                    }
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

            // If SOC didn't improve, restore original direction and constraints for backtracking
            // (If accepted, the modified direction is used in apply_primal_dual_step)
            if (out_theta >= theta0 * 1.5) {
                // SOC didn't help enough — restore
                for (int k = 0; k <= N; ++k) {
                    sv->riccati_ws_.dx[k] = sv->soc_save_dx_[k];
                    sv->ds_[k] = sv->soc_save_ds_[k];  sv->dlambda_[k] = sv->soc_save_dlam_[k];
                    s[k].d = sv->soc_save_d_[k];
                    if (k < N) {
                        sv->riccati_ws_.du[k] = sv->soc_save_du_[k];
                        s[k].c = sv->soc_save_c_orig_[k];
                    }
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
    double max_g_pos_       = 0.0;  // max positive constraint violation: max(g+, 0)
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
    double alpha_lambda_     = 1.0;     // coupled dual step: min(alpha_d, alpha_p)

    // Affine predictor FTB limits (for FTB-bottleneck detection)
    double aff_alpha_p_      = 1.0;     // affine primal FTB step
    double aff_alpha_d_      = 1.0;     // affine dual FTB step
    double last_alpha_p_     = 1.0;     // previous corrector primal FTB step
    int    low_ftb_count_    = 0;       // consecutive iters with α_p < 0.05

    // Filter line search
    FilterLineSearch<NX, NU, HORIZON> filter_ls_;
    IPMTrialEvaluator evaluator_;

    // SOC save buffers (persistent to avoid stack overflow on embedded targets)
    Vec<NX> soc_save_dx_[HORIZON + 1];
    Vec<NU> soc_save_du_[HORIZON + 1];
    Vec<NC> soc_save_ds_[HORIZON + 1];
    Vec<NC> soc_save_dlam_[HORIZON + 1];
    Vec<NX> soc_save_c_[HORIZON];
    Vec<NC> soc_save_d_[HORIZON + 1];  // save original constraint values for SOC
    Vec<NX> soc_save_c_orig_[HORIZON]; // save original dynamics defects for SOC

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

    // ── Diagnostic instrumentation ──
    IterDiag last_diag_;
    FILE*    diag_csv_ = nullptr;

public:
    // Enable CSV diagnostic output to file
    void enable_diag_csv(const char* filename) {
        diag_csv_ = fopen(filename, "w");
        if (diag_csv_) IterDiag::print_csv_header(diag_csv_);
    }
    void disable_diag_csv() {
        if (diag_csv_) { fclose(diag_csv_); diag_csv_ = nullptr; }
    }
    const IterDiag& last_diagnostic() const { return last_diag_; }
};

} // namespace nmpc
