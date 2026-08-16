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
 *   of a convex NLP, using a single primal-dual Newton step per iteration
 *   with an FSM-scheduled adaptive centering parameter σ, and Riccati
 *   recursion to exploit the multistage KKT structure.
 *
 * Key ideas:
 *   1. Primal-dual Newton with adaptive centering: one KKT solve per
 *      iteration; σ is scheduled by the barrier FSM (progress-based)
 *      rather than derived from a Mehrotra affine predictor.
 *   2. Block elimination of inequality slacks/duals from the KKT
 *   3. Riccati recursion (equivalent to block LDL^T) on the reduced
 *      multistage-banded KKT system → O(N·(nx+nu)³) per iteration
 */

#include "nmpc_core.hpp"
#include "nmpc_problem.hpp"
#include "nmpc_riccati.hpp"
#include "nmpc_barrier_manager.hpp"
#include "nmpc_filter_ls.hpp"
#include <algorithm>
#include <chrono>
#include "nmpc_preconditioner.hpp"

namespace nmpc {

// ─────────────────────────────────────────────────────────────────────────────
//  Paper-aligned IPM parameters
// ─────────────────────────────────────────────────────────────────────────────

struct PaperIPMParams {
    // Barrier
    double mu_init       = 1.0;
    double mu_min        = 5e-4;    // barrier parameter floor

    // Centering parameter scheduling (FSM-driven σ)
    double centering     = 0.1;     // minimum σ ∈ (0,1)

    // Single-loop convergence — per-condition tolerances
    int    max_iters     = 100;     // total Newton iterations
    double tol_primal    = 1e-6;    // primal infeasibility tolerance
    double tol_compl     = 1e-6;    // complementarity tolerance (|s·λ − μ|)
    double tol_ineq      = 1e-8;    // inequality tolerance (allow small negative s/λ)
    double tol_stat      = 0.5;     // stationarity tolerance (relative ‖∇L‖∞/scale)
    double mu_conv_threshold = 1e-3; // μ must be ≤ this for convergence (false-convergence guard)

    // MPCC interior relaxation (Leyffer-Lopez-Calva-Nocedal, Eq. 3.13).
    // ConstraintModel marks pairs of nonpositive rows whose negations are
    // complementary. For every pair the solver appends the explicit equality
    //
    //   g_a*g_b - theta(mu) + s_c = 0,  s_c > 0,
    //
    // and applies the existing log barrier to all three slacks s_a, s_b, s_c.
    // Thus this is Eq. 3.13's elastic interior formulation, not a penalty-only
    // relaxation. theta(mu) is proportional to the barrier parameter.
    double mpcc_relaxation_scale = 1.0;
    double tol_mpcc              = 1e-6;
    double mpcc_recovery_mu      = 0.1;
    int    mpcc_recovery_max_iters = 500;

    // Fraction-to-boundary
    double tau           = 0.999;    // tighter: 0.99 (was 0.995)

    // ── Globalization (line-search) ──
    int    max_ls_iters   = 20;
    double armijo_c       = 1e-4;
    // Also test a defect-preserving nonlinear dynamics retraction.
    // The ordinary trial remains candidate 0 with identical tests.
    bool   enable_nonlinear_rollout = true;

    // === Globalization: Second-Order Correction ===
    int    soc_max         = 4;       // max SOC attempts per iteration
    double kappa_soc       = 0.99;    // SOC stall threshold: abort if theta_soc > kappa_soc * theta_prev

    // === Complementarity Safeguard (HPIPM-style) ===
    double m_safe          = 0.5;    // enforce lambda_j * s_j >= m_safe * mu


    // === Barrier Update (σ-modulation) ===
    double kappa_eps       = 10.0;    // E_mu <= kappa_eps * mu → barrier solved
    double sigma_exp_easy   = 1.5;    // easy subproblems: σ^1.5 (more aggressive)
    double sigma_exp_normal = 1.0;    // standard subproblem: σ^1.0
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
    double bound_s_min     = 1e-12;   // numerical floor for bound-distance calculations

    // Eliminate simple-bound multipliers as z=mu/d until the primal Newton
    // direction becomes bound-limited, then retain independent multipliers.
    bool   primal_dual_bounds = true;
    double bound_pd_activation_fraction = 0.5; // dimensionless primal FTB threshold
    // === Nonlinear KKT Iterative Refinement (Shamanskii chord) ===
    bool   enable_refinement  = false;   // enable direction refinement before line search
    int    max_refine_iters   = 5;       // max chord passes
    double refine_tol         = 1e-6;    // nonlinear residual tolerance
    double refine_diverge_fac = 1.5;     // divergence guard: stop if ||r|| > fac * prev

    // Output
    int    verbosity     = 0;
    // Capture optional debug snapshots and per-iteration diagnostic records.
    // Disabling this does not skip residuals used by solver decisions; debug
    // getters retain their last captured values until diagnostics are enabled.
    bool   enable_runtime_diagnostics = true;

    // Wall-clock watchdog. A non-positive value preserves the legacy
    // unbounded behavior. On expiry, solve returns TIME_LIMIT and restores the
    // trajectory that was supplied at solve entry.
    double time_limit_ms = 0.0;

    // Debug: freeze μ after N iterations (-1 = disabled)
    int    freeze_mu_after = -1;

    // Preconditioning
    bool   enable_preconditioner = false;  // diagonal Jacobi preconditioner
    bool   bound_aware_preconditioner = false; // refresh scaling with bound barrier

    // === Exact (full-Lagrangian) Hessian ===
    // When true, adds second-order curvature of the dynamics (contracted
    // with the costate p_{k+1}) and inequality constraints (contracted with
    // the multiplier λ) to the Newton Hessian, giving exact-Newton steps
    // instead of Gauss-Newton. Default true uses full-Lagrangian curvature
    // (exact Newton); the indefinite-Hessian guard (reg_max / inertia_min_pivot)
    // falls back to Gauss-Newton when the curvature is indefinite.
    // Adaptive Gauss-Newton -> exact-curvature transition. The switch requires
    // local feasibility, stalled stationarity, and repeated disagreement between
    // accepted nonlinear reductions and the GN prediction.
    // Set false together with exact_hessian=false to remain in Gauss-Newton.
    bool   adaptive_exact_hessian     = true;
    int    gn_exact_min_iters        = 5;
    int    gn_exact_stall_window     = 3;
    int    gn_exact_poor_model_limit = 2;
    double gn_exact_feas_tol         = 1e-2;
    double gn_exact_min_progress     = 0.15;
    double gn_exact_min_model_ratio  = 0.25;

    bool   exact_hessian        = true;
    double exact_hessian_fd_eps = 1e-4;   // step for central 2nd finite-diff

    // === Riccati regularization and indefinite-Hessian guard ===
    // The exact Lagrangian Hessian is indefinite away from the solution;
    // the Riccati factorization must detect this and abort to the
    // Gauss-Newton fallback (caller handles this when backward_lhs
    // returns KKT_SINGULAR).
    //   riccati_relative_regularization: dimensionless diagonal shift applied
    //                     to Riccati value Hessians and stage Schur matrices.
    //   reg_max         : cap on the dimensionless per-stage relative shift;
    //                     exceeding it signals a hopeless Schur complement.
    //   inertia_min_pivot: require min(D_pivot)/max(D_pivot) ≥ this after
    //                     LDLT. Catches the "small positive pivot" case
    //                     where LDLT succeeds but κ(S)≈1e20 silently
    //                     poisons the next Riccati recursion (P_k blowup).
    //                     Default 0 disables the check; reg_max alone is
    //                     sufficient to abort the pathologically indefinite
    //                     case. Set to e.g. 1e-12 to additionally reject
    //                     mild conditioning issues proactively.
    double riccati_relative_regularization = 1e-12;
    double reg_max              = 1e12;
    double inertia_min_pivot    = 0.0;
};

namespace paper_ipm_detail {

inline double initial_cold_slack(
    double constraint_value, double mu, const PaperIPMParams& params) {
    const double floor = std::max(mu, params.s_min_init);
    return constraint_value > 0.0
        ? floor
        : std::max(-constraint_value + params.delta_slack, floor);
}

inline double initial_warm_slack(
    double constraint_value, double mu, const PaperIPMParams& params) {
    const double interior_floor = std::max(
        params.bound_s_min, std::min(mu, params.s_min_init));
    return std::max(
        -constraint_value, std::max(std::sqrt(mu), interior_floor));
}

}  // namespace paper_ipm_detail
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
//  Generic residual and scaling diagnostics for convergence analysis.
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
    ~PaperIPMSolver() { disable_diag_csv(); }

    using Prob  = NMPCProblem<NX, NU, NC, HORIZON>;
    using Stage = StageData<NX, NU, NC>;
    using WS    = RiccatiWorkspace<NX, NU, HORIZON>;
    using Ricc  = RiccatiSolver<NX, NU, NC, HORIZON>;
    using Prec  = HessianPreconditioner<NX, NU, HORIZON>;
    using Clock = std::chrono::steady_clock;

    struct PhaseTimer {
        explicit PhaseTimer(double* accumulator)
            : accumulator_(accumulator), start_(Clock::now()) {}
        ~PhaseTimer() {
            if (accumulator_)
                *accumulator_ += std::chrono::duration<double, std::milli>(
                    Clock::now() - start_).count();
        }
        double* accumulator_;
        Clock::time_point start_;
    };

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
        double sigma;        // centering parameter
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
        for (int j = 0; j < NU; ++j) {
            double btp = 0.0;
            for (int m = 0; m < NX; ++m) btp += s0.B(m, j) * p1[m];
            double btpc = 0.0;
            for (int m = 0; m < NX; ++m)
                for (int n = 0; n < NX; ++n)
                    btpc += s0.B(m, j) * P1(m, n) * s0.c[n];
            r.rhs_d[j] = s0.qu[j] + btp + btpc;
        }
        r.reg_used = reg_used_;
        r.sigma = sigma_;
        r.mu = mu_;
        return r;
    }

    Status init(const PaperIPMParams& params = PaperIPMParams{}) {
        params_ = params;
        mu_     = params_.mu_init;
        warm_start_ready_ = false;

        // Auto-derive stationarity tolerance from mu_min if not set
        if (params_.tol_stat < 0.0) {
            params_.tol_stat = 100.0 * params_.mu_min;
            if (params_.verbosity >= 1)
                printf("  [auto] tol_stat = %.1e  (100 * mu_min)\n", params_.tol_stat);
        }

        return Status::SUCCESS;
    }

    // Update only the watchdog budget; unlike init(), this preserves all warm
    // primal-dual/barrier state.
    void set_time_limit_ms(double time_limit_ms) {
        params_.time_limit_ms = time_limit_ms;
    }

    // ── Main solve (true single-loop IPM) ─────────────────────────

    Status solve(Prob& problem, SolverStats& out_stats) {
        return solve_impl(problem, out_stats, false);
    }

    Status solve_warm(Prob& problem, SolverStats& out_stats) {
        if (!warm_start_ready_) return Status::NOT_INITIALIZED;
        return solve_impl(problem, out_stats, true);
    }

    // A horizon shift invalidates the dynamic-programming costates. They are
    // recomputed by the first Riccati pass of the next warm solve.
    void invalidate_shifted_costates() {
        has_costates_ = false;
        for (int k = 0; k <= HORIZON; ++k)
            riccati_ws_.p[k].zero();
    }

private:
    Status solve_impl(Prob& problem, SolverStats& out_stats, bool warm_start) {
        out_stats.reset();
        active_stats_ = &out_stats;
        solve_start_ = Clock::now();
        deadline_enabled_ = params_.time_limit_ms > 0.0;
        if (deadline_enabled_) {
            deadline_ = solve_start_ + std::chrono::duration_cast<Clock::duration>(
                std::chrono::duration<double, std::milli>(params_.time_limit_ms));
        }
        timed_out_ = false;
        current_iteration_ = 0;
        entry_mu_ = mu_;
        entry_bound_pd_mode_ = bound_pd_mode_;
        entry_has_costates_ = has_costates_;
        entry_exact_hessian_ = params_.exact_hessian;
        entry_warm_start_ready_ = warm_start_ready_;
        current_solve_warm_ = warm_start;
        const bool capture_diagnostics = params_.enable_runtime_diagnostics
            || params_.verbosity > 0 || diag_csv_ != nullptr;
        entry_sigma_ = sigma_;
        entry_alpha_lambda_ = alpha_lambda_;
        entry_last_alpha_p_ = last_alpha_p_;
        entry_low_ftb_count_ = low_ftb_count_;
        for (int k = 0; k <= HORIZON; ++k)
            entry_costates_[k] = riccati_ws_.p[k];

        prob_ = &problem;
        for (int k = 0; k <= HORIZON; ++k)
            entry_stages_[k] = problem.stages[k];

        Status st = problem.validate();
        if (st != Status::SUCCESS) return finish_early(st);

        st = cache_constraint_metadata();
        if (st != Status::SUCCESS) return finish_early(st);

        if (!warm_start) {
            mu_ = params_.mu_init;
            bound_pd_mode_ = false;
            initialize_from_problem();
            has_costates_ = false;
            for (int k = 0; k <= HORIZON; ++k)
                riccati_ws_.p[k].zero();
        } else {
            // Preserve the current barrier and valid shifted primal-dual data.
            // Only newly appended or otherwise invalid entries are initialized.
            prepare_warm_start_from_problem();
        }
        st = evaluate_model();      // refresh after initialization/shift
        if (st != Status::SUCCESS) return finish_early(st);
        if (deadline_reached()) return finish_time_limit();

        // Compute preconditioner scaling ONCE per MPC solve (outside Newton loop)
        if (params_.enable_preconditioner) {
            if (params_.bound_aware_preconditioner)
                prec_.compute_bound_aware(prob_->stages, prob_->x_lb, prob_->x_ub,
                                          prob_->u_lb, prob_->u_ub, mu_,
                                          params_.bound_s_min);
            else
                prec_.compute(prob_->stages);
            // Note: transform_qp() is called inside the loop before compute_kkt_residuals
            // so that convergence checks use scaled residuals.
        }

        // mu_ already set by adaptive μ₀ above (or params_.mu_init if no constraints)
        // Initialize barrier strategy
        {
            BarrierUpdateParams bup;
            bup.mu_init     = params_.mu_init;
            bup.mu_min      = effective_mu_min();
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
            // theta is an L1 sum while tol_primal is an infinity-norm
            // tolerance. Scale the "sufficiently feasible" threshold by the
            // number of scalar primal equations so the two tests agree.
            int n_primal_rows = NX * HORIZON;
            for (int k = 0; k <= HORIZON; ++k)
                n_primal_rows += active_constraints(k);
            fp.theta_min = has_complementarity()
                ? std::max(1e-8, params_.tol_primal)
                : std::max(1e-4, params_.tol_primal * n_primal_rows);
            fp.soc_max    = params_.exact_hessian ? std::min(params_.soc_max, 1) : 0;
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
        bool   ls_failed = false;  // line search failure flag
        int    ls_fail_count = 0;  // consecutive filter-exhaustion count
        bool   model_evaluated = true;  // evaluate_model() already done before loop
        // Stagnation detection: require all KKT blocks to stop improving.
        double stat_best = 1e100;
        double primal_best = 1e100;
        double compl_best = 1e100;
        int    stagnation_count = 0;
        int    final_status_at_break = -1;  // -1=unknown, 0=stagnation, 1=converged
        constexpr int STAGNATION_LIMIT = 30;
        // State for adaptive GN -> exact-curvature switching.
        double gn_window_start_stat = 0.0;
        double gn_last_progress = 1.0;
        double gn_last_model_ratio = 1.0;
        int    gn_window_samples = 0;
        int    gn_poor_model_count = 0;
        bool   gn_switch_pending = false;


        const bool requested_exact_hessian = params_.exact_hessian;
        for (iter = 0; iter < params_.max_iters; ++iter) {
            current_iteration_ = iter;
            if (deadline_reached()) break;
            // Apply an adaptive transition requested by the preceding GN
            // residual/model-agreement window. Force a physical model refresh
            // because an accepted iterate may already have been evaluated GN.
            if (params_.adaptive_exact_hessian
                && gn_switch_pending && !params_.exact_hessian) {
                params_.exact_hessian = true;
                model_evaluated = false;
                if (params_.verbosity >= 1)
                    printf("  [hessian: adaptive GN-to-exact switch at iter=%d"
                           " progress=%.1f%% rho=%.3f]\n", iter,
                           100.0 * gn_last_progress, gn_last_model_ratio);
            }

            // 1. Evaluate model at current iterate (skip if already evaluated)
            if (!model_evaluated) {
                st = evaluate_model();
                if (st != Status::SUCCESS) return finish_early(st);
                if (deadline_reached()) break;
            }
            model_evaluated = false;
            stages_scaled_ = false;

            // 2. Apply FIXED preconditioner scaling to derivatives (before KKT build)
            // After this, prob_->stages derivatives are in SCALED space.
            if (params_.enable_preconditioner) {
                if (params_.bound_aware_preconditioner)
                    prec_.compute_bound_aware(prob_->stages, prob_->x_lb, prob_->x_ub,
                                              prob_->u_lb, prob_->u_ub, mu_,
                                              params_.bound_s_min);
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
                st = evaluate_model();
                if (st != Status::SUCCESS) return finish_early(st);
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

            // 4. Convergence check: KKT satisfied.
            //    No at_minimum gate: with the 2-block convergence test
            //    (linKKT + complementarity), kkt_converged() is reliable
            //    at any μ level.  Forcing μ→μ_min wastes iterations when
            //    the solution is already converged at larger μ.
            if (kkt_converged()) {
                final_status_at_break = 1;
                break;
            }

            // 3b. Stagnation detection is now handled post-Riccati (after
            //     compute_post_riccati_stationarity) where costates are
            //     consistent with the current model evaluation.
            if (params_.verbosity >= 3) {
                bool primal_ok  = (primal_inf_ <= params_.tol_primal);
                bool compl_ok   = (compl_inf_  <= params_.tol_compl);
                bool ineq_ok    = (
                    ineq_viol_ >= -params_.tol_ineq &&
                    max_g_pos_ <= params_.tol_ineq);
                bool stat_ok    = (stat_inf_ <= params_.tol_stat);
                bool mpcc_ok    = (!has_complementarity() ||
                                   mpcc_inf_ <= params_.tol_mpcc);
                bool mu_ok      = (mu_ <= params_.mu_conv_threshold);
                printf("  [conv: primal_ok=%d(%.2e<=%.2e) compl_ok=%d(%.2e<=%.2e) ineq_ok=%d(slack %.2e>=-%.2e, raw %.2e<=%.2e) stat_ok=%d(%.2e<=%.2e) mpcc_ok=%d(%.2e<=%.2e) mu_ok=%d(mu=%.2e<=%.2e)]\n",
                       primal_ok, primal_inf_, params_.tol_primal,
                       compl_ok, compl_inf_, params_.tol_compl,
                       ineq_ok, ineq_viol_, params_.tol_ineq,
                       max_g_pos_, params_.tol_ineq,
                       stat_ok, stat_inf_, params_.tol_stat,
                       mpcc_ok, mpcc_inf_, params_.tol_mpcc,
                       mu_ok, mu_, params_.mu_conv_threshold);
            }

            // 4. ── Build LHS once, factorize ─────────────────────────────
            {
                const auto phase_start = Clock::now();
                build_kkt_lhs();
                ++out_stats.kkt_assemblies;
                add_phase_ms(out_stats.kkt_assembly_time_ms, phase_start);
            }
            if (deadline_reached()) break;
            {
                const auto phase_start = Clock::now();
                ++out_stats.riccati_factorizations;
                st = solve_kkt_lhs();
                add_phase_ms(out_stats.riccati_time_ms, phase_start);
            }
            out_stats.regularization = reg_used_;
            out_stats.max_regularization = std::max(
                out_stats.max_regularization, reg_used_);
            if (out_stats.riccati_factorizations == 1) {
                out_stats.first_regularization = reg_used_;
            }
            if (st != Status::SUCCESS && params_.exact_hessian && has_costates_) {
                // Exact Hessian can be indefinite → Riccati factorization may
                // fail.  Fall back to Gauss-Newton for THIS iteration:
                // re-evaluate without curvature, re-scale, and refactor.
                if (params_.verbosity >= 2)
                    printf("  [exact-hess] factorization failed → Gauss-Newton fallback\n");
                bool saved = params_.exact_hessian;
                params_.exact_hessian = false;
                st = evaluate_model();
                params_.exact_hessian = saved;
                if (st != Status::SUCCESS) return finish_early(st);
                stages_scaled_ = false;
                if (params_.enable_preconditioner) {
                    if (params_.bound_aware_preconditioner)
                        prec_.compute_bound_aware(prob_->stages, prob_->x_lb, prob_->x_ub,
                                                  prob_->u_lb, prob_->u_ub, mu_,
                                                  params_.bound_s_min);
                    prec_.transform_qp(prob_->stages);
                    stages_scaled_ = true;
                }
                {
                    const auto phase_start = Clock::now();
                    build_kkt_lhs();
                    ++out_stats.kkt_assemblies;
                    add_phase_ms(out_stats.kkt_assembly_time_ms, phase_start);
                }
                {
                    const auto phase_start = Clock::now();
                    ++out_stats.riccati_factorizations;
                    st = solve_kkt_lhs();
                    add_phase_ms(out_stats.riccati_time_ms, phase_start);
                }
                out_stats.regularization = reg_used_;
                out_stats.max_regularization = std::max(
                    out_stats.max_regularization, reg_used_);
            }
            if (st != Status::SUCCESS) return finish_early(st);
            if (deadline_reached()) break;

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

            // ── Adaptive σ scheduling (FSM-driven, no affine predictor) ────
            sigma_ = compute_adaptive_sigma();

            // ── Single primal-dual Newton solve with centering σ·μ ────
            // Save sigma/mu for invariance debugging
            debug_sigma_ = sigma_;
            debug_mu_ = mu_;
            debug_primal_inf_ = primal_inf_;
            debug_compl_inf_ = compl_inf_;

            {
                const auto phase_start = Clock::now();
                build_kkt_rhs();
                add_phase_ms(out_stats.kkt_assembly_time_ms, phase_start);
            }
            if (deadline_reached()) break;

            if (capture_diagnostics) {
                // Save PRISTINE Riccati stages right after KKT build (before SOC/LS)
                for (int kk = 0; kk <= HORIZON; ++kk)
                    debug_pristine_stages_[kk] = riccati_stages_[kk];

                // Save Riccati internals for invariance debugging (before forward pass)
                debug_P_term_ = riccati_ws_.P[HORIZON];
                debug_S_fact0_ = riccati_ws_.S_fact[0];
            }

            {
                const auto phase_start = Clock::now();
                ++out_stats.riccati_rhs_solves;
                st = solve_kkt_rhs_and_forward();
                add_phase_ms(out_stats.riccati_time_ms, phase_start);
            }
            if (st != Status::SUCCESS) return finish_early(st);
            if (deadline_reached()) break;

            if (capture_diagnostics) {
                // Re-save Riccati internals AFTER forward pass
                for (int i = 0; i < NU; ++i) {
                    debug_d0_[i] = riccati_ws_.d[0][i];
                    for (int j = 0; j < NX; ++j)
                        debug_K0_(i, j) = riccati_ws_.K[0](i, j);
                }
            }

            // ── Nonlinear KKT iterative refinement (DISABLED) ────
            // if (params_.enable_refinement) {
            //     refine_newton_direction();
            // }

            // Compute ds/dλ BEFORE recovering primal step to physical space.
            // ── Constraint-normal correction (Fix C) ──
            // When Newton du worsens a violated constraint (Cu·du > 0),
            // replace the constraint-normal component with a repair step.
            // Applied in scaled space, BEFORE ds/dλ computation.
            // apply_constraint_normal_correction();  // DISABLED: makes prim worse (0.58→24.8)

            recover_inequality_steps(sigma_);
            if (params_.primal_dual_bounds)
                recover_bound_multiplier_steps(sigma_, true);

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

            if (capture_diagnostics) {
                // Save Newton step BEFORE recovery (scaled space for precond)
                for (int kk = 0; kk <= HORIZON; ++kk) {
                    debug_scaled_dx_[kk] = riccati_ws_.dx[kk];
                    debug_scaled_p_[kk]  = riccati_ws_.p[kk];
                    if (kk < HORIZON)
                        debug_scaled_du_[kk] = riccati_ws_.du[kk];
                }
            }

            // Recover physical step from scaled Riccati solution
            if (params_.enable_preconditioner) {
                prec_.recover_primal_step(riccati_ws_);
                prec_.recover_dual_step(riccati_ws_);
            }

            if (capture_diagnostics) {
                // Save Newton step AFTER recovery (physical space for both)
                for (int kk = 0; kk <= HORIZON; ++kk) {
                    debug_phys_dx_[kk] = riccati_ws_.dx[kk];
                    debug_phys_p_[kk]  = riccati_ws_.p[kk];
                    if (kk < HORIZON)
                        debug_phys_du_[kk] = riccati_ws_.du[kk];
                }
            }
            has_costates_ = true;  // costates now valid for stationarity check

            // ── Linear KKT residual check ────
            // ws.dx/du/p are now in PHYSICAL space (after recovery).
            // The function converts them back to scaled space internally
            // via dx_s = dx / inv_Lx = dx · Lx (round-trip: scaled→physical→scaled).
            // This is redundant but correct, and works whether preconditioner is on or off.
            linear_kkt_res_ = compute_linear_kkt_residual();

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
            if (kkt_converged()) {
                final_status_at_break = 1;
                break;
            }

            // ── Post-Riccati stagnation detection ──────────────
            // Switch curvature when accepted nonlinear steps repeatedly disagree
            // with GN and stationarity stalls. At the barrier floor, two locally
            // feasible samples suffice; otherwise globalization rejection plus a
            // stalled full window is the trigger.
            if (params_.adaptive_exact_hessian && !params_.exact_hessian) {
                if (gn_window_samples == 0)
                    gn_window_start_stat = std::max(stat_inf_, 1e-16);
                ++gn_window_samples;

                const double local_feas_tol = std::max(
                    params_.gn_exact_feas_tol, 20.0 * params_.tol_primal);
                if (iter + 1 >= params_.gn_exact_min_iters
                    && gn_window_samples >= 2
                    && barrier_strategy_.at_minimum(mu_)
                    && primal_inf_ <= local_feas_tol
                    && gn_poor_model_count >= params_.gn_exact_poor_model_limit)
                    gn_switch_pending = true;

                if (gn_window_samples >= params_.gn_exact_stall_window) {
                    gn_last_progress =
                        (gn_window_start_stat - stat_inf_) / gn_window_start_stat;
                    const bool residual_stalled =
                        gn_last_progress < params_.gn_exact_min_progress;
                    const bool curvature_ready = stat_inf_ >= 1.0
                        || barrier_strategy_.at_minimum(mu_);
                    const bool globalization_stalled =
                        gn_poor_model_count >= params_.gn_exact_poor_model_limit;
                    if (iter + 1 >= params_.gn_exact_min_iters
                        && primal_inf_ <= local_feas_tol
                        && residual_stalled && curvature_ready
                        && globalization_stalled)
                        gn_switch_pending = true;
                    gn_window_samples = 0;
                    gn_window_start_stat = stat_inf_;
                }
            }

            const int stag_limit = STAGNATION_LIMIT;
            bool kkt_progress = false;
            if (stat_inf_ < stat_best * 0.999) {
                stat_best = stat_inf_; kkt_progress = true;
            }
            if (primal_inf_ < primal_best * 0.999) {
                primal_best = primal_inf_; kkt_progress = true;
            }
            if (compl_inf_ < compl_best * 0.999) {
                compl_best = compl_inf_; kkt_progress = true;
            }
            if (kkt_progress) {
                stagnation_count = 0;
            } else {
                ++stagnation_count;
                if (stagnation_count >= stag_limit && barrier_strategy_.at_minimum(mu_)
                    && primal_inf_ <= params_.tol_primal
                    && compl_inf_ <= params_.tol_compl && !kkt_converged()) {
                    final_status_at_break = 0;
                    if (params_.verbosity >= 1)
                        printf("  [stagnation: stat=%.3e unchanged for %d iters - terminating\n",
                               stat_inf_, STAGNATION_LIMIT);
                    break;
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
                    double s_val = stg.s[worst_j];
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

            // ── KKT residual log ───
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

            if (capture_diagnostics) {
                // ── Diagnostic snapshot (before line search) ──
                last_diag_ = compute_iter_diagnostics();
                last_diag_.alpha_p = alpha_p;
                last_diag_.alpha_d = alpha_d;
            }

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
                            for (int j = 0; j < active_constraints(k); ++j)
                                theta_ineq_0 += std::fabs(s[k].d[j] + s[k].s[j]);
                    } else {
                        for (int k = 0; k <= HORIZON; ++k)
                            for (int j = 0; j < active_constraints(k); ++j)
                                theta_ineq_0 += std::fabs(s[k].s[j]);
                    }
                }
                printf("  [filter: theta_0=%.4e phi0=%.4e Dphi=%.3e"
                       " | th_dyn=%.4e th_ineq=%.4e | ap=%.4f ad=%.4f]\n",
                       evaluator_.current_theta(), evaluator_.current_phi(),
                       evaluator_.compute_Dphi(),
                       theta_dyn_0, theta_ineq_0, alpha_p, alpha_d);
            }

            const auto line_search_start = Clock::now();
            const bool warm_near_full_step_trial = current_solve_warm_
                && alpha_p >= 0.99;
            if (warm_near_full_step_trial)
                ++out_stats.warm_near_full_step_trials;
            LSResult ls_result = filter_ls_.search(evaluator_, alpha_p);
            add_phase_ms(out_stats.line_search_time_ms, line_search_start);
            if (warm_near_full_step_trial
                && ls_result.status == LSStatus::ACCEPTED
                && ls_result.ls_iters == 1
                && ls_result.candidate == 0)
                ++out_stats.warm_near_full_step_accepts;

            if (ls_result.status == LSStatus::TIME_LIMIT) {
                timed_out_ = true;
                break;
            }

            if (ls_result.status == LSStatus::ACCEPTED) {
                ls_fail_count = 0;
                alpha = ls_result.alpha;
                ls_iters = ls_result.ls_iters;
                if (ls_result.soc_used) out_stats.soc_steps++;

                if (params_.verbosity >= 2) {
                    printf("  [ls-accept] a=%.4e theta_trial=%.4e phi_trial=%.4e"
                           " candidate=%d soc=%d ls_iters=%d\n",
                           alpha, ls_result.theta_trial, ls_result.phi_trial,
                           ls_result.candidate,
                           ls_result.soc_used ? 1 : 0, ls_result.ls_iters);
                }

                // Actual versus predicted barrier-objective reduction. Small
                // accepted fractions are also evidence that the GN model is not
                // predictive over the available FTB step.
                if (!params_.exact_hessian) {
                    const double predicted = -alpha * ls_result.Dphi;
                    gn_last_model_ratio = predicted > 1e-14
                        ? (ls_result.phi0 - ls_result.phi_trial) / predicted
                        : 1.0;
                    const double accepted_fraction =
                        alpha / std::max(alpha_p, 1e-16);
                    const bool poor_model =
                        (predicted > 1e-14
                         && gn_last_model_ratio < params_.gn_exact_min_model_ratio)
                        || accepted_fraction < 0.25;
                    if (poor_model)
                        gn_poor_model_count = std::min(
                            gn_poor_model_count + 1,
                            params_.gn_exact_poor_model_limit);
                }

                // Apply the accepted step — the ONLY place variables are updated
                if (deadline_reached()) break;
                apply_primal_dual_step(alpha, alpha_lambda_, ls_result.candidate);
                if (deadline_reached()) break;

                // Re-evaluate model at new point for accurate theta reporting
                st = evaluate_model();
                if (st != Status::SUCCESS) return finish_early(st);
                model_evaluated = true;
                stages_scaled_ = false;
                recenter_feasible_slacks();
                refresh_barrier_metrics();

                if (params_.verbosity >= 1) {
                    double effective_alam = std::min(alpha_lambda_, alpha);
                    printf("  [step: a=%.4f alam=%.4f ls=%d soc=%s cost=%.4e theta=%.4e]\n",
                           alpha, effective_alam, ls_iters,
                           ls_result.soc_used ? "yes" : "no",
                           compute_objective(), compute_theta());
                }
                if (capture_diagnostics) {
                    // ── Diagnostic: update with LS results ──
                    last_diag_.ls_iters   = ls_result.ls_iters;
                    last_diag_.ls_rejected = false;
                    last_diag_.alpha_p    = alpha;
                }
                // theta_dyn/theta_ineq already in pre-LS snapshot
            } else {
                alpha = ls_result.alpha;
                ++ls_fail_count;
                if (capture_diagnostics) {
                    last_diag_.ls_rejected = true;
                    last_diag_.alpha_p = alpha;
                }
                if (params_.verbosity >= 1)
                    printf("  [LS fail: a=%.3e alam=%.2e ls=%d fail_count=%d] step too tiny\n",
                           alpha, alpha_lambda_, ls_result.ls_iters, ls_fail_count);
                // Try filter reset on first few failures
                if (ls_fail_count <= 3) {
                    filter_ls_.reset_filter();
                    if (params_.verbosity >= 1)
                        printf("  [filter reset: trying again after filter exhaustion]\n");
                    // Retry the same iterate against an empty filter.  Do not
                    // mutate it with an arbitrary micro-step: the Newton
                    // direction will be rebuilt on the next iteration.
                    model_evaluated = true;
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
                // Exact constraint curvature is multiplier-weighted. The
                // safeguard may change lambda after evaluate_model(), so the
                // cached Hessian must not be reused by the next iteration.
                if (params_.exact_hessian)
                    model_evaluated = false;
            }
            // Barrier update: reduce μ if subproblem solved, else hold
            // Passes accepted alpha and max_g_pos for graduated step reduction
            // and two-phase barrier schedule (Phase 4/5/6).
            {
                // Freeze μ for debugging: skip update after N iters
                double E_mu = std::max(primal_inf_, compl_inf_);
                bool frozen = (params_.freeze_mu_after >= 0 && iter >= params_.freeze_mu_after);
                bool mu_changed = false;
                if (!frozen) {
                    // Pass linear KKT residual for barrier stat gate.
                    // The nonlinear stat_inf_ has a structural floor from
                    // barrier gradients, preventing μ reduction.  The linear
                    // KKT residual is self-consistent and allows aggressive
                    // μ reduction when the Riccati solve is accurate.
                    double lin_kkt = linear_kkt_res_.max_rel_res;
                    mu_changed = barrier_strategy_.update(
                        mu_, primal_inf_, compl_inf_, sigma_, lin_kkt,
                        true,  /* cross_term_accepted: Mehrotra P-C removed, always healthy */
                        alpha_p, max_g_pos_, alpha);
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
                        for (int jj = 0; jj < active_constraints(kk); ++jj) {
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
                    // The appended product row contains theta(mu), so its
                    // value and derivatives belong to the new barrier
                    // subproblem and must be refreshed before the next step.
                    model_evaluated = false;
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

        if (timed_out_ || deadline_reached())
            return finish_time_limit();

        const auto finalization_start = Clock::now();
        // ── Solve summary ─────────────────────────────────────────
        // Re-evaluate for accurate final primal/complementarity stats.
        // Save the riccati-based stationarity (the convergence metric)
        // before re-evaluation overwrites it.
        double saved_stat = stat_inf_;
        st = evaluate_model();
        if (st != Status::SUCCESS) return finish_early(st);
        if (deadline_reached()) return finish_time_limit();
        // Restore scaled stages so compute_linear_kkt_residual is consistent
        // with the scaled Riccati workspace (dx, du, P, p).
        if (params_.enable_preconditioner) {
            prec_.transform_qp(prob_->stages);
        }
        compute_kkt_residuals();
        if (deadline_reached()) return finish_time_limit();
        stat_inf_ = saved_stat;  // restore Riccati-consistent stationarity

        out_stats.inner_iterations = iter;
        out_stats.barrier_param    = mu_;
        out_stats.primal_infeas    = primal_inf_;
        out_stats.dual_infeas      = stat_inf_;
        out_stats.complementarity  = compl_inf_;
        out_stats.mpcc_complementarity = mpcc_inf_;
        out_stats.mpcc_relaxation  = mpcc_relaxation();
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
            if (has_complementarity())
                printf("MPCC product:    %.3e  (tol=%.1e, theta=%.1e)  %s\n",
                       mpcc_inf_, params_.tol_mpcc, mpcc_relaxation(),
                       mpcc_inf_ <= params_.tol_mpcc ? "OK" : "FAIL");
            printf("Ineq slack/raw:  %.3e / %.3e  (tol=%.1e)  %s\n",
                   ineq_viol_, max_g_pos_, params_.tol_ineq,
                   ineq_viol_ >= -params_.tol_ineq &&
                       max_g_pos_ <= params_.tol_ineq ? "OK" : "FAIL");
            printf("SOC steps:       %d\n", out_stats.soc_steps);
            printf("Regularization:  %.1e\n", reg_used_);
            printf("Cost:            %.4f\n", compute_objective());

            // Convergence diagnosis
            bool p_ok = primal_inf_ <= params_.tol_primal;
            bool s_ok = stat_inf_   <= params_.tol_stat;
            bool c_ok = compl_inf_  <= params_.tol_compl;
            bool i_ok = ineq_viol_ >= -params_.tol_ineq &&
                        max_g_pos_ <= params_.tol_ineq;
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
                if (!i_ok)
                    printf("  * INEQUALITY violated: slack=%.3e raw=%.3e\n",
                           ineq_viol_, max_g_pos_);
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

        params_.exact_hessian = requested_exact_hessian;
        if (!warm_start)
            warm_start_ready_ = (final_status == Status::SUCCESS);
        add_phase_ms(out_stats.finalization_time_ms, finalization_start);
        out_stats.solve_time_ms = elapsed_ms(solve_start_);
        if (warm_start && final_status != Status::SUCCESS)
            restore_solve_entry();
        active_stats_ = nullptr;
        return final_status;
    }

    static double elapsed_ms(const Clock::time_point& start) {
        return std::chrono::duration<double, std::milli>(Clock::now() - start).count();
    }

    static void add_phase_ms(double& accumulator,
                             const Clock::time_point& start) {
        accumulator += elapsed_ms(start);
    }

    bool deadline_reached() {
        if (active_stats_) ++active_stats_->deadline_checks;
        if (!deadline_enabled_) return false;
        if (Clock::now() < deadline_) return false;
        timed_out_ = true;
        return true;
    }

    Status finish_early(Status status) {
        if (status == Status::TIME_LIMIT) return finish_time_limit();
        if (active_stats_)
            active_stats_->solve_time_ms = elapsed_ms(solve_start_);
        if (current_solve_warm_ && status != Status::SUCCESS)
            restore_solve_entry();
        else
            params_.exact_hessian = entry_exact_hessian_;
        active_stats_ = nullptr;
        return status;
    }

    void restore_solve_entry() {
        if (prob_) {
            for (int k = 0; k <= HORIZON; ++k)
                prob_->stages[k] = entry_stages_[k];
        }
        mu_ = entry_mu_;
        bound_pd_mode_ = entry_bound_pd_mode_;
        has_costates_ = entry_has_costates_;
        params_.exact_hessian = entry_exact_hessian_;
        warm_start_ready_ = entry_warm_start_ready_;
        sigma_ = entry_sigma_;
        alpha_lambda_ = entry_alpha_lambda_;
        last_alpha_p_ = entry_last_alpha_p_;
        low_ftb_count_ = entry_low_ftb_count_;
        for (int k = 0; k <= HORIZON; ++k)
            riccati_ws_.p[k] = entry_costates_[k];
    }

    Status finish_time_limit() {
        timed_out_ = true;
        restore_solve_entry();
        if (active_stats_) {
            active_stats_->time_limit_hit = 1;
            active_stats_->inner_iterations = current_iteration_;
            active_stats_->barrier_param = mu_;
            active_stats_->solve_time_ms = elapsed_ms(solve_start_);
        }
        active_stats_ = nullptr;
        return Status::TIME_LIMIT;
    }
    // ═════════════════════════════════════════════════════════════════════
    //  Initialize barrier variables from problem data
    // ═════════════════════════════════════════════════════════════════════

    int base_constraints(int k) const {
        return constraint_metadata_cached_ ? base_constraints_cache_[k] : 0;
    }

    int complementarity_pairs(int k) const {
        return constraint_metadata_cached_ ? complementarity_pairs_cache_[k] : 0;
    }

    int active_constraints(int k) const {
        return constraint_metadata_cached_ ? active_constraints_cache_[k] : 0;
    }

    bool has_complementarity() const {
        return constraint_metadata_cached_ && has_complementarity_cache_;
    }

    double mpcc_relaxation() const {
        return params_.mpcc_relaxation_scale * mu_;
    }

    double effective_mu_min() const {
        if (!has_complementarity()) return params_.mu_min;
        // At a feasible elastic row, 0 <= a*b < theta(mu). Keep the final
        // relaxation one order below the requested physical MPCC tolerance.
        const double mpcc_floor =
            0.1 * params_.tol_mpcc / params_.mpcc_relaxation_scale;
        return std::min(params_.mu_min, mpcc_floor);
    }

    Status cache_constraint_metadata() {
        constraint_metadata_cached_ = false;
        has_complementarity_cache_ = false;
        for (int k = 0; k <= HORIZON; ++k) {
            base_constraints_cache_[k] = 0;
            complementarity_pairs_cache_[k] = 0;
            active_constraints_cache_[k] = 0;
        }
        if (!prob_ || !prob_->constraints) {
            constraint_metadata_cached_ = true;
            return Status::SUCCESS;
        }
        if (params_.mpcc_relaxation_scale < 0.0 || params_.tol_mpcc < 0.0)
            return Status::BAD_ARGUMENT;
        for (int k = 0; k <= HORIZON; ++k) {
            const int base = std::clamp(
                prob_->constraints->num_constraints(k), 0, NC);
            const int pairs = std::max(
                0, prob_->constraints->num_complementarity_pairs(k));
            if (base + pairs > NC) return Status::BAD_ARGUMENT;
            if (pairs > 0 && (params_.mpcc_relaxation_scale <= 0.0 ||
                              params_.tol_mpcc <= 0.0))
                return Status::BAD_ARGUMENT;
            for (int p = 0; p < pairs; ++p) {
                int first = -1, second = -1;
                if (!prob_->constraints->complementarity_pair(
                        k, p, first, second))
                    return Status::BAD_ARGUMENT;
                if (first < 0 || first >= base || second < 0 ||
                    second >= base || first == second)
                    return Status::BAD_ARGUMENT;
                complementarity_first_cache_[k][p] = first;
                complementarity_second_cache_[k][p] = second;
            }
            base_constraints_cache_[k] = base;
            complementarity_pairs_cache_[k] = pairs;
            active_constraints_cache_[k] = base + pairs;
            has_complementarity_cache_ = has_complementarity_cache_ || pairs > 0;
        }
        constraint_metadata_cached_ = true;
        return Status::SUCCESS;
    }

    Status append_complementarity_values(int k, Vec<NC>& g) const {
        const int base = base_constraints(k);
        const int pairs = complementarity_pairs(k);
        const double theta = mpcc_relaxation();
        for (int p = 0; p < pairs; ++p) {
            const int first = complementarity_first_cache_[k][p];
            const int second = complementarity_second_cache_[k][p];
            g[base + p] = g[first] * g[second] - theta;
        }
        return Status::SUCCESS;
    }

    Status evaluate_constraints(const Vec<NX>& x, const Vec<NU>& u,
                                int k, Vec<NC>& g) const {
        Status st = prob_->constraints->evaluate(x, u, k, g);
        if (st != Status::SUCCESS) return st;
        return append_complementarity_values(k, g);
    }

    Status evaluate_terminal_constraints(const Vec<NX>& x,
                                         Vec<NC>& g) const {
        Status st = prob_->constraints->evaluate_terminal(x, g);
        if (st != Status::SUCCESS) return st;
        return append_complementarity_values(HORIZON, g);
    }

    Status jacobian_constraints(const Vec<NX>& x, const Vec<NU>& u, int k,
                                const Vec<NC>& g,
                                Mat<NC, NX>& Cx, Mat<NC, NU>& Cu) const {
        Status st = prob_->constraints->jacobian(x, u, k, Cx, Cu);
        if (st != Status::SUCCESS) return st;
        const int base = base_constraints(k);
        const int pairs = complementarity_pairs(k);
        for (int p = 0; p < pairs; ++p) {
            const int first = complementarity_first_cache_[k][p];
            const int second = complementarity_second_cache_[k][p];
            const int row = base + p;
            for (int i = 0; i < NX; ++i)
                Cx(row, i) = g[second] * Cx(first, i) +
                             g[first] * Cx(second, i);
            for (int i = 0; i < NU; ++i)
                Cu(row, i) = g[second] * Cu(first, i) +
                             g[first] * Cu(second, i);
        }
        return Status::SUCCESS;
    }

    Status evaluate_with_jacobian_constraints(
        const Vec<NX>& x, const Vec<NU>& u, int k, Vec<NC>& g,
        Mat<NC, NX>& Cx, Mat<NC, NU>& Cu) const {
        Status st = prob_->constraints->evaluate_with_jacobian(
            x, u, k, g, Cx, Cu);
        if (st != Status::SUCCESS) return st;
        st = append_complementarity_values(k, g);
        if (st != Status::SUCCESS) return st;
        const int base = base_constraints(k);
        const int pairs = complementarity_pairs(k);
        for (int p = 0; p < pairs; ++p) {
            const int first = complementarity_first_cache_[k][p];
            const int second = complementarity_second_cache_[k][p];
            const int row = base + p;
            for (int i = 0; i < NX; ++i)
                Cx(row, i) = g[second] * Cx(first, i) +
                             g[first] * Cx(second, i);
            for (int i = 0; i < NU; ++i)
                Cu(row, i) = g[second] * Cu(first, i) +
                             g[first] * Cu(second, i);
        }
        return Status::SUCCESS;
    }

    Status jacobian_terminal_constraints(const Vec<NX>& x, const Vec<NC>& g,
                                         Mat<NC, NX>& Cx) const {
        Status st = prob_->constraints->jacobian_terminal(x, Cx);
        if (st != Status::SUCCESS) return st;
        const int base = base_constraints(HORIZON);
        const int pairs = complementarity_pairs(HORIZON);
        for (int p = 0; p < pairs; ++p) {
            const int first = complementarity_first_cache_[HORIZON][p];
            const int second = complementarity_second_cache_[HORIZON][p];
            const int row = base + p;
            for (int i = 0; i < NX; ++i)
                Cx(row, i) = g[second] * Cx(first, i) +
                             g[first] * Cx(second, i);
        }
        return Status::SUCCESS;
    }

    Status evaluate_terminal_with_jacobian_constraints(
        const Vec<NX>& x, Vec<NC>& g, Mat<NC, NX>& Cx) const {
        Status st = prob_->constraints->evaluate_terminal_with_jacobian(
            x, g, Cx);
        if (st != Status::SUCCESS) return st;
        st = append_complementarity_values(HORIZON, g);
        if (st != Status::SUCCESS) return st;
        const int base = base_constraints(HORIZON);
        const int pairs = complementarity_pairs(HORIZON);
        for (int p = 0; p < pairs; ++p) {
            const int first = complementarity_first_cache_[HORIZON][p];
            const int second = complementarity_second_cache_[HORIZON][p];
            const int row = base + p;
            for (int i = 0; i < NX; ++i)
                Cx(row, i) = g[second] * Cx(first, i) +
                             g[first] * Cx(second, i);
        }
        return Status::SUCCESS;
    }

    double compute_mpcc_complementarity() const {
        if (!has_complementarity()) return 0.0;
        double residual = 0.0;
        for (int k = 0; k <= HORIZON; ++k) {
            const int pairs = complementarity_pairs(k);
            for (int p = 0; p < pairs; ++p) {
                const int first = complementarity_first_cache_[k][p];
                const int second = complementarity_second_cache_[k][p];
                residual = std::max(
                    residual,
                    std::fabs(prob_->stages[k].d[first] *
                              prob_->stages[k].d[second]));
            }
        }
        return residual;
    }

    // For a strictly feasible one-sided row g(z) < 0, the slack equality can
    // be eliminated exactly after accepting the nonlinear trial point. This
    // prevents first-order slack updates from leaving an O(alpha^2) g+s
    // residual when the filter repeatedly accepts short steps.
    void recenter_feasible_slacks() {
        if (!prob_->constraints) return;
        Stage* s = prob_->stages;
        for (int k = 0; k <= HORIZON; ++k) {
            for (int j = 0; j < active_constraints(k); ++j) {
                double gj = s[k].d[j];
                if (gj < -1e-10) {
                    double sj = std::max(-gj, 1e-12);
                    s[k].s[j] = sj;
                    s[k].lambda[j] = mu_ / sj;
                }
            }
        }
    }

    void initialize_from_problem() {
        const int N = HORIZON;
        Stage* s = prob_->stages;

        for (int k = 0; k <= N; ++k) {
            // The log-barrier formulation requires a strictly interior
            // mutable initial guess. This changes only the guess, not the
            // physical bounds. x[0] is fixed by x0 and is left untouched.
            if (k > 0 && prob_->n_bound_x > 0) {
                for (int i = 0; i < NX; ++i) {
                    if (prob_->x_lb[i] > -1e19)
                        s[k].x[i] = std::max(s[k].x[i],
                                             prob_->x_lb[i] + params_.bound_s_min);
                    if (prob_->x_ub[i] < 1e19)
                        s[k].x[i] = std::min(s[k].x[i],
                                             prob_->x_ub[i] - params_.bound_s_min);
                }
            }
            if (k < N && prob_->n_bound_u > 0) {
                for (int i = 0; i < NU; ++i) {
                    if (prob_->u_lb[i] > -1e19)
                        s[k].u[i] = std::max(s[k].u[i],
                                             prob_->u_lb[i] + params_.bound_s_min);
                    if (prob_->u_ub[i] < 1e19)
                        s[k].u[i] = std::min(s[k].u[i],
                                             prob_->u_ub[i] - params_.bound_s_min);
                }
            }
            if (prob_->constraints) {
                if (k < N) {
                    evaluate_constraints(s[k].x, s[k].u, k, s[k].d);
                } else {
                    evaluate_terminal_constraints(s[k].x, s[k].d);
                }
                for (int j = 0; j < active_constraints(k); ++j) {
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
                        s_init = paper_ipm_detail::initial_cold_slack(
                            gj, mu_, params_);
                    } else {
                        // Feasible: slack = distance from boundary + margin.
                        s_init = paper_ipm_detail::initial_cold_slack(
                            gj, mu_, params_);
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
                    double dL = std::max(s[k].u[i] - prob_->u_lb[i], 0.0) + params_.bound_s_min;
                    double dU = std::max(prob_->u_ub[i] - s[k].u[i], 0.0) + params_.bound_s_min;
                    s[k].z_L_u[i] = mu_ / dL;
                    s[k].z_U_u[i] = mu_ / dU;
                }
            }
            if (prob_->n_bound_x > 0) {
                for (int i = 0; i < NX; ++i) {
                    double dL = std::max(s[k].x[i] - prob_->x_lb[i], 0.0) + params_.bound_s_min;
                    double dU = std::max(prob_->x_ub[i] - s[k].x[i], 0.0) + params_.bound_s_min;
                    s[k].z_L_x[i] = mu_ / dL;
                    s[k].z_U_x[i] = mu_ / dU;
                }
            }
        }
    }

    void prepare_warm_start_from_problem() {
        const int N = HORIZON;
        Stage* s = prob_->stages;

        for (int k = 0; k <= N; ++k) {
            // Keep the shifted primal guess strictly inside finite bounds.
            // x[0] remains the measured fixed initial state.
            if (k > 0 && prob_->n_bound_x > 0) {
                for (int i = 0; i < NX; ++i) {
                    if (prob_->x_lb[i] > -1e19)
                        s[k].x[i] = std::max(
                            s[k].x[i], prob_->x_lb[i] + params_.bound_s_min);
                    if (prob_->x_ub[i] < 1e19)
                        s[k].x[i] = std::min(
                            s[k].x[i], prob_->x_ub[i] - params_.bound_s_min);
                }
            }
            if (k < N && prob_->n_bound_u > 0) {
                for (int i = 0; i < NU; ++i) {
                    if (prob_->u_lb[i] > -1e19)
                        s[k].u[i] = std::max(
                            s[k].u[i], prob_->u_lb[i] + params_.bound_s_min);
                    if (prob_->u_ub[i] < 1e19)
                        s[k].u[i] = std::min(
                            s[k].u[i], prob_->u_ub[i] - params_.bound_s_min);
                }
            }

            if (prob_->constraints) {
                if (k < N)
                    evaluate_constraints(s[k].x, s[k].u, k, s[k].d);
                else
                    evaluate_terminal_constraints(s[k].x, s[k].d);

                const int active = active_constraints(k);
                for (int j = 0; j < active; ++j) {
                    const bool slack_valid = std::isfinite(s[k].s[j])
                        && s[k].s[j] > params_.bound_s_min;
                    const bool dual_valid = std::isfinite(s[k].lambda[j])
                        && s[k].lambda[j] > params_.bound_s_min;
                    if (!slack_valid) {
                        const double gj = s[k].d[j];
                        // Avoid adding the cold-start margin to every newly
                        // appended row. For near-active rows, sqrt(mu) keeps
                        // the new slack/dual pair centered; safely inactive
                        // rows can satisfy g + s = 0 immediately.
                        s[k].s[j] =
                            paper_ipm_detail::initial_warm_slack(
                                gj, mu_, params_);
                    }
                    if (!dual_valid)
                        s[k].lambda[j] = mu_ / s[k].s[j];
                }
            }

            if (prob_->n_bound_u > 0 && k < N) {
                for (int i = 0; i < NU; ++i) {
                    const double dL = std::max(
                        s[k].u[i] - prob_->u_lb[i], 0.0)
                        + params_.bound_s_min;
                    const double dU = std::max(
                        prob_->u_ub[i] - s[k].u[i], 0.0)
                        + params_.bound_s_min;
                    if (!(std::isfinite(s[k].z_L_u[i])
                          && s[k].z_L_u[i] > params_.bound_s_min))
                        s[k].z_L_u[i] = mu_ / dL;
                    if (!(std::isfinite(s[k].z_U_u[i])
                          && s[k].z_U_u[i] > params_.bound_s_min))
                        s[k].z_U_u[i] = mu_ / dU;
                }
            }
            if (prob_->n_bound_x > 0) {
                for (int i = 0; i < NX; ++i) {
                    const double dL = std::max(
                        s[k].x[i] - prob_->x_lb[i], 0.0)
                        + params_.bound_s_min;
                    const double dU = std::max(
                        prob_->x_ub[i] - s[k].x[i], 0.0)
                        + params_.bound_s_min;
                    if (!(std::isfinite(s[k].z_L_x[i])
                          && s[k].z_L_x[i] > params_.bound_s_min))
                        s[k].z_L_x[i] = mu_ / dL;
                    if (!(std::isfinite(s[k].z_U_x[i])
                          && s[k].z_U_x[i] > params_.bound_s_min))
                        s[k].z_U_x[i] = mu_ / dU;
                }
            }
        }
    }

    // ═════════════════════════════════════════════════════════════════════
    //  Exact-Newton curvature (full Lagrangian Hessian)
    //
    //  Adds the second-order tensor terms that Gauss-Newton drops:
    //    Qxx += Σ_m (p_{k+1})_m ∂²f_m/∂x²  +  Σ_j λ_j ∂²g_j/∂x²
    //    Quu += (same, ∂²/∂u²)
    //    Qux += (same, ∂²/∂u∂x)
    //  The contracted (adjoint) Hessian of the scalar
    //    φ(z) = p_{k+1}ᵀ f(x,u) + λᵀ g(x,u),   z=(x,u)
    //  is formed by CENTRAL second finite differences of the existing
    //  discrete_step / evaluate.  Signs are '+' because stationarity uses
    //  L = ℓ + p_{k+1}ᵀ f + λᵀ g.  Called in PHYSICAL space (before
    //  transform_qp), so the augmented Hessian is scaled consistently.
    // ═════════════════════════════════════════════════════════════════════

    // Dispatch: use the analytic adjoint Hessian when every active model
    // provides it, otherwise fall back to finite differences.
    void add_exact_hessian(int k) {
        const bool has_dyn = (k < HORIZON);
        const bool dyn_ok = (!has_dyn) ||
                            prob_->dynamics->provides_adjoint_hessian();
        const bool con_ok = (prob_->constraints == nullptr) ||
                            prob_->constraints->provides_adjoint_hessian();
        if (dyn_ok && con_ok) add_exact_hessian_analytic(k);
        else                  add_exact_hessian_fd(k);
    }

    // Analytic exact-Newton curvature: ask the models for the contracted
    // Hessians  Σ_m p_m ∇²f_m  and  Σ_j λ_j ∇²g_j  and accumulate them onto
    // the cost Hessian blocks (Qxx, Qux, Quu) in O(1) model calls per stage.
    void add_exact_hessian_analytic(int k) {
        if (active_stats_) ++active_stats_->exact_hessian_analytic_calls;
        const int N = HORIZON;
        Stage& sk = prob_->stages[k];
        const bool has_ctrl = (k < N);
        const bool has_dyn  = (k < N);
        if (has_dyn)
            prob_->dynamics->adjoint_hessian(sk.x, sk.u, prob_->dt, k,
                                             riccati_ws_.p[k + 1],
                                             sk.Qxx, sk.Qux, sk.Quu);
        if (prob_->constraints) {
            Vec<NC> lambda_effective = sk.lambda;
            const int base = base_constraints(k);
            const int pairs = complementarity_pairs(k);
            for (int p = 0; p < pairs; ++p) {
                const int first = complementarity_first_cache_[k][p];
                const int second = complementarity_second_cache_[k][p];
                const double xi = sk.lambda[base + p];
                lambda_effective[first] += xi * sk.d[second];
                lambda_effective[second] += xi * sk.d[first];

                for (int i = 0; i < NX; ++i)
                    for (int j = 0; j < NX; ++j)
                        sk.Qxx(i, j) += xi * (
                            sk.Cx(first, i) * sk.Cx(second, j) +
                            sk.Cx(second, i) * sk.Cx(first, j));
                if (has_ctrl) {
                    for (int i = 0; i < NU; ++i) {
                        for (int j = 0; j < NU; ++j)
                            sk.Quu(i, j) += xi * (
                                sk.Cu(first, i) * sk.Cu(second, j) +
                                sk.Cu(second, i) * sk.Cu(first, j));
                        for (int j = 0; j < NX; ++j)
                            sk.Qux(i, j) += xi * (
                                sk.Cu(first, i) * sk.Cx(second, j) +
                                sk.Cu(second, i) * sk.Cx(first, j));
                    }
                }
            }
            if (has_ctrl)
                prob_->constraints->adjoint_hessian(
                    sk.x, sk.u, k, lambda_effective,
                    sk.Qxx, sk.Qux, sk.Quu);
            else
                prob_->constraints->adjoint_hessian_terminal(
                    sk.x, lambda_effective, sk.Qxx);
        }
    }

    // Finite-difference exact-Newton curvature (fallback used when the
    // models do not provide analytic adjoint Hessians).
    void add_exact_hessian_fd(int k) {
        if (active_stats_) ++active_stats_->exact_hessian_fd_calls;
        const int N = HORIZON;
        Stage& sk = prob_->stages[k];
        const bool has_ctrl = (k < N);
        const bool has_dyn  = (k < N);
        const int  NZ = NX + (has_ctrl ? NU : 0);
        const double eps = params_.exact_hessian_fd_eps;
        const double inv_e2 = 1.0 / (eps * eps);

        // φ(z + dz), dz a length-NZ perturbation in z-space (x then u).
        auto phi = [&](const double* dz) -> double {
            Vec<NX> xp; Vec<NU> up = sk.u;
            for (int i = 0; i < NX; ++i) xp[i] = sk.x[i] + dz[i];
            if (has_ctrl)
                for (int i = 0; i < NU; ++i) up[i] = sk.u[i] + dz[NX + i];
            double val = 0.0;
            if (has_dyn) {
                Vec<NX> fk;
                prob_->dynamics->discrete_step(xp, up, prob_->dt, fk, k);
                for (int m = 0; m < NX; ++m)
                    val += riccati_ws_.p[k + 1][m] * fk[m];
            }
            if (prob_->constraints) {
                Vec<NC> g;
                if (has_ctrl) evaluate_constraints(xp, up, k, g);
                else          evaluate_terminal_constraints(xp, g);
                for (int j = 0; j < active_constraints(k); ++j)
                    val += sk.lambda[j] * g[j];
            }
            return val;
        };

        // Dispatch a scalar Hessian entry h at z-indices (i,j) into blocks.
        auto add_block = [&](int i, int j, double h) {
            bool iu = (i >= NX), ju = (j >= NX);
            if (!iu && !ju) {
                sk.Qxx(i, j) += h;
                if (i != j) sk.Qxx(j, i) += h;
            } else if (iu && ju) {
                sk.Quu(i - NX, j - NX) += h;
                if (i != j) sk.Quu(j - NX, i - NX) += h;
            } else {
                // mixed x/u → Qux(u_idx, x_idx) (i<j ⇒ i is x, j is u)
                int xi = iu ? j : i;
                int ui = iu ? (i - NX) : (j - NX);
                sk.Qux(ui, xi) += h;
            }
        };

        double dz[NX + NU];
        for (int q = 0; q < NX + NU; ++q) dz[q] = 0.0;
        const double phi0 = phi(dz);

        // Diagonal: h_ii = (φ(+e_i) - 2φ(0) + φ(-e_i)) / eps²
        for (int i = 0; i < NZ; ++i) {
            dz[i] = eps;  const double pp = phi(dz);
            dz[i] = -eps; const double pm = phi(dz);
            dz[i] = 0.0;
            add_block(i, i, (pp - 2.0 * phi0 + pm) * inv_e2);
        }

        // Off-diagonal: h_ij = (φ(++)-φ(+-)-φ(-+)+φ(--)) / (4 eps²)
        for (int i = 0; i < NZ; ++i) {
            for (int j = i + 1; j < NZ; ++j) {
                dz[i] = eps;  dz[j] = eps;  const double pp = phi(dz);
                dz[i] = eps;  dz[j] = -eps; const double pm = phi(dz);
                dz[i] = -eps; dz[j] = eps;  const double mp = phi(dz);
                dz[i] = -eps; dz[j] = -eps; const double mm = phi(dz);
                dz[i] = 0.0;  dz[j] = 0.0;
                add_block(i, j, (pp - pm - mp + mm) * (0.25 * inv_e2));
            }
        }
    }

    // ═════════════════════════════════════════════════════════════════════
    //  Evaluate nonlinear model at current iterate
    // ═════════════════════════════════════════════════════════════════════

    Status evaluate_model() {
        if (active_stats_) ++active_stats_->model_evaluations;
        PhaseTimer timer(active_stats_ ? &active_stats_->model_eval_time_ms
                                       : nullptr);
        const int N = HORIZON;
        Stage* s = prob_->stages;

        for (int k = 0; k < N; ++k) {
            if (deadline_reached()) return Status::TIME_LIMIT;
            // Dynamics defect: c_k = f(x_k, u_k) - x_{k+1}
            Vec<NX> fk;
            const auto dynamics_eval_start = Clock::now();
            Status st = prob_->dynamics->discrete_step(s[k].x, s[k].u,
                                                        prob_->dt, fk, k);
            if (active_stats_)
                add_phase_ms(active_stats_->dynamics_eval_time_ms,
                             dynamics_eval_start);
            if (st != Status::SUCCESS) return st;
            for (int i = 0; i < NX; ++i)
                s[k].c[i] = fk[i] - s[k + 1].x[i];

            // Dynamics Jacobians
            const auto dynamics_jacobian_start = Clock::now();
            st = prob_->dynamics->linearize(s[k].x, s[k].u, prob_->dt,
                                            s[k].A, s[k].B, k);
            if (active_stats_)
                add_phase_ms(active_stats_->dynamics_jacobian_time_ms,
                             dynamics_jacobian_start);
            if (st != Status::SUCCESS) return st;

            // Cost
            const auto cost_derivative_start = Clock::now();
            s[k].cost = prob_->cost->stage_cost(s[k].x, s[k].u, k);
            st = prob_->cost->stage_gradient(s[k].x, s[k].u, k,
                                              s[k].qx, s[k].qu);
            if (st != Status::SUCCESS) return st;
            
            st = prob_->cost->stage_hessian(s[k].x, s[k].u, k,
                                             s[k].Qxx, s[k].Quu, s[k].Qux);
            if (active_stats_)
                add_phase_ms(active_stats_->cost_derivative_time_ms,
                             cost_derivative_start);
            if (st != Status::SUCCESS) return st;

            // Constraints
            if (prob_->constraints) {
                const auto constraint_start = Clock::now();
                st = evaluate_with_jacobian_constraints(
                    s[k].x, s[k].u, k, s[k].d, s[k].Cx, s[k].Cu);
                if (active_stats_)
                    add_phase_ms(
                        active_stats_->constraint_value_jacobian_time_ms,
                        constraint_start);
                if (st != Status::SUCCESS) return st;
            }

            // Exact-Newton curvature: add dynamics (⊗ costate p_{k+1}) and
            // constraint (⊗ multiplier λ) second-order terms onto the cost
            // Hessian.  Skipped on iter 0 (no costates → Gauss-Newton warm-up).
            if (params_.exact_hessian && has_costates_)
                add_exact_hessian(k);

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
            const auto constraint_start = Clock::now();
            st = evaluate_terminal_with_jacobian_constraints(
                s[N].x, s[N].d, s[N].Cx);
            if (active_stats_)
                add_phase_ms(
                    active_stats_->constraint_value_jacobian_time_ms,
                    constraint_start);
            if (st != Status::SUCCESS) return st;
        }

        // Terminal stage has no control variables
        s[N].Quu.zero();
        s[N].Qux.zero();
        s[N].qu.zero();

        // Exact-Newton curvature at terminal (constraint ⊗ λ only; no dynamics).
        if (params_.exact_hessian && has_costates_)
            add_exact_hessian(N);

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
                st = evaluate_constraints(x_trial, u_trial, k, d_trial);
                if (st != Status::SUCCESS) return false;
                trial_stages_[k].d = d_trial;

                // Jacobians at trial point
                st = jacobian_constraints(
                    x_trial, u_trial, k, d_trial,
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
            st = evaluate_terminal_constraints(xN_trial, dN_trial);
            if (st != Status::SUCCESS) return false;
            trial_stages_[N].d = dN_trial;

            st = jacobian_terminal_constraints(
                xN_trial, dN_trial, trial_stages_[N].Cx);
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
    //  costates, which are consistent with the CURRENT model
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

        // The Riccati recursion stores the affine value-gradient term p[k].
        // After the forward pass, the costate associated with the solved QP is
        //
        //     nu[k] = p[k] + P[k] * dx[k],
        //
        // not p[k] alone.  Using only p[k] leaves the P*dx contribution in the
        // reported nonlinear stationarity residual.  That residual can stay
        // O(1) even when the linear KKT system is solved to machine precision,
        // causing false stagnation at an otherwise converged primal solution.
        Vec<NX> nu[N + 1];
        for (int k = 0; k <= N; ++k) {
            nu[k] = riccati_ws_.p[k];
            for (int i = 0; i < NX; ++i)
                for (int j = 0; j < NX; ++j)
                    nu[k][i] += riccati_ws_.P[k](i, j) * riccati_ws_.dx[k][j];
        }

        double post_stat = 0.0;
        stat_abs_inf_ = 0.0;
        stat_scale_max_ = 1.0;

        // ── Worst-component diagnostic ──
        double worst_rel = 0.0;
        int    worst_k = -1;
        bool   worst_is_u = false;
        int    worst_i = -1;
        double worst_q = 0.0, worst_bar = 0.0, worst_ctw = 0.0, worst_cos = 0.0;
        double worst_scale = 1.0;
        // Global barrier metrics
        double global_min_dist = 1e30, global_max_bar_grad = 0.0, global_max_bar_hess = 0.0;
        int    global_min_dist_k = -1, global_min_dist_u = -1;  // stage, component

        for (int k = 0; k <= N; ++k) {
            // Build the MODIFIED gradient (same as build_kkt_rhs).
            // q̃ = qx + Cx^T·(centering - λ)  where centering = (σμ + λ(g+s))/s
            // All terms here are in SCALED space (after transform_qp).
            Vec<NX> lag_x = s[k].qx;
            Vec<NU> lag_u;
            if (k < N) lag_u = s[k].qu; else lag_u.zero();

            // Compute bound barrier gradient (part of full KKT stationarity equation).
            // bound_grad = -μ/dL + μ/dU  (physical space)
            // This is the bound dual contribution, separate from Riccati costate.
            Vec<NX> bound_grad_x; bound_grad_x.zero();
            Vec<NU> bound_grad_u; bound_grad_u.zero();
            double bound_grad_x_inf = 0.0, bound_grad_u_inf = 0.0;
            if (prob_->n_bound_x > 0) {
                for (int i = 0; i < NX; ++i) {
                    double dL = std::max(s[k].x[i] - prob_->x_lb[i], 0.0) + params_.bound_s_min;
                    double dU = std::max(prob_->x_ub[i] - s[k].x[i], 0.0) + params_.bound_s_min;
                    double bg;
                    if (use_primal_dual_bound()) {
                        bg = -(s[k].z_L_x[i] + dz_L_x_[k][i]);
                        bg += s[k].z_U_x[i] + dz_U_x_[k][i];
                    } else {
                        bg = -mu_ / dL + mu_ / dU;
                    }
                    if (params_.enable_preconditioner)
                        bg *= prec_.inv_Lx(k)[i];
                    bound_grad_x[i] = bg;
                    bound_grad_x_inf = std::max(bound_grad_x_inf, std::fabs(bg));
                }
            }
            if (prob_->n_bound_u > 0 && k < N) {
                for (int i = 0; i < NU; ++i) {
                    double dL = std::max(s[k].u[i] - prob_->u_lb[i], 0.0) + params_.bound_s_min;
                    double dU = std::max(prob_->u_ub[i] - s[k].u[i], 0.0) + params_.bound_s_min;
                    double bg;
                    if (use_primal_dual_bound()) {
                        bg = -(s[k].z_L_u[i] + dz_L_u_[k][i]);
                        bg += s[k].z_U_u[i] + dz_U_u_[k][i];
                    } else {
                        bg = -mu_ / dL + mu_ / dU;
                    }
                    if (params_.enable_preconditioner)
                        bg *= prec_.inv_Lu(k)[i];
                    bound_grad_u[i] = bg;
                    bound_grad_u_inf = std::max(bound_grad_u_inf, std::fabs(bg));
                }
            }

            // Compute Cx^T·λ for stationarity residual (full KKT equation uses λ)
            // and Cx^T·λ for scale denominator (same term)
            Vec<NX> ct_lam_x; ct_lam_x.zero();
            Vec<NU> ct_lam_u; ct_lam_u.zero();
            if (prob_->constraints) {
                for (int j = 0; j < active_constraints(k); ++j) {
                    double sj = s[k].s[j];
                    if (sj < 1e-14) continue;
                    // nu above is the costate of the solved Newton QP (the
                    // predicted new dynamics multiplier), so it must be paired
                    // with the predicted new inequality multiplier as well.
                    // Mixing nu_new with lambda_old creates a false residual
                    // of C^T * dlambda at active constraints.
                    double lj = s[k].lambda[j] + dlambda_[k][j];
                    for (int i = 0; i < NX; ++i)
                        ct_lam_x[i] += s[k].Cx(j, i) * lj;
                    if (k < N)
                        for (int i = 0; i < NU; ++i)
                            ct_lam_u[i] += s[k].Cu(j, i) * lj;
                }
            }
            // Add Cx^T·λ to stationarity residual (full KKT equation)
            for (int i = 0; i < NX; ++i) lag_x[i] += ct_lam_x[i];
            for (int i = 0; i < NU; ++i) lag_u[i] += ct_lam_u[i];

            // Add costate terms using Riccati costates (scaled space).
            Vec<NX> cos_x; cos_x.zero();
            Vec<NU> cos_u; cos_u.zero();
            if (k < N) {
                for (int i = 0; i < NX; ++i)
                    for (int m = 0; m < NX; ++m)
                        cos_x[i] += s[k].A(m, i) * nu[k+1][m];
                for (int i = 0; i < NU; ++i)
                    for (int m = 0; m < NX; ++m)
                        cos_u[i] += s[k].B(m, i) * nu[k+1][m];
            }
            for (int i = 0; i < NX; ++i)
                cos_x[i] -= nu[k][i];

            for (int i = 0; i < NX; ++i) lag_x[i] += cos_x[i];
            for (int i = 0; i < NU; ++i) lag_u[i] += cos_u[i];

            // ── Add bound barrier gradient to stationarity residual ──
            // The full KKT stationarity equation includes the barrier gradient:
            //   ∇_x L = qx + Cx^T·λ + (A^T·p_{k+1} - p_k) + ∇_x b = 0
            //   ∇_u L = qu + Cu^T·λ + B^T·p_{k+1} + ∇_u b = 0
            // where ∇b = -μ/dL + μ/dU is the bound barrier gradient.
            // Including it makes the stationarity check consistent with the
            // full KKT conditions, allowing convergence when the barrier
            // gradient offsets the costate on strongly coupled trajectories.
            for (int i = 0; i < NX; ++i) lag_x[i] += bound_grad_x[i];
            for (int i = 0; i < NU; ++i) lag_u[i] += bound_grad_u[i];

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

            // Scale denominator: terms from full KKT stationarity equation.
            // Full KKT includes cost gradient, constraint dual, costate, AND barrier gradient:
            //   ∇_x L = qx + Cx^T·λ + (A^T·p_{k+1} - p_k) + ∇_x b = 0
            //   ∇_u L = qu + Cu^T·λ + B^T·p_{k+1} + ∇_u b = 0
            // Including the barrier gradient makes the scale consistent with the
            // full KKT conditions, allowing convergence when the barrier dominates.
            // Each term converted to physical space via /inv_L.
            double scale_x, scale_u;
            if (params_.enable_preconditioner) {
                double gx = 0.0, gu = 0.0;
                for (int i = 0; i < NX; ++i) {
                    double qx_phys = std::fabs(s[k].qx[i] / prec_.inv_Lx(k)[i]);
                    double ctl_phys = std::fabs(ct_lam_x[i] / prec_.inv_Lx(k)[i]);
                    double cos_phys = std::fabs(cos_x[i] / prec_.inv_Lx(k)[i]);
                    double bar_phys = std::fabs(bound_grad_x[i] / prec_.inv_Lx(k)[i]);
                    gx = std::max({gx, qx_phys, ctl_phys, cos_phys, bar_phys});
                }
                scale_x = std::max({gx, 1.0});
                if (k < N) {
                    for (int i = 0; i < NU; ++i) {
                        double qu_phys = std::fabs(s[k].qu[i] / prec_.inv_Lu(k)[i]);
                        double ctl_phys = std::fabs(ct_lam_u[i] / prec_.inv_Lu(k)[i]);
                        double cos_phys = std::fabs(cos_u[i] / prec_.inv_Lu(k)[i]);
                        double bar_phys = std::fabs(bound_grad_u[i] / prec_.inv_Lu(k)[i]);
                        gu = std::max({gu, qu_phys, ctl_phys, cos_phys, bar_phys});
                    }
                    scale_u = std::max({gu, 1.0});
                } else {
                    scale_u = 1.0;
                }
            } else {
                double cos_x_inf = cos_x.norm_inf();
                double cos_u_inf = cos_u.norm_inf();
                double ctl_x_inf = ct_lam_x.norm_inf();
                double ctl_u_inf = ct_lam_u.norm_inf();
                double grad_x_inf = s[k].qx.norm_inf();
                double grad_u_inf = (k < N) ? s[k].qu.norm_inf() : 0.0;
                double bar_x_inf = bound_grad_x.norm_inf();
                double bar_u_inf = (k < N) ? bound_grad_u.norm_inf() : 0.0;
                scale_x = std::max({grad_x_inf, ctl_x_inf, cos_x_inf, bar_x_inf, 1.0});
                scale_u = std::max({grad_u_inf, ctl_u_inf, cos_u_inf, bar_u_inf, 1.0});
            }
            double scale = (scale_x > scale_u) ? scale_x : scale_u;
            double stat = stat_abs / scale;
            if (stat > post_stat) post_stat = stat;
            if (stat_abs > stat_abs_inf_) stat_abs_inf_ = stat_abs;
            if (scale > stat_scale_max_) stat_scale_max_ = scale;

            // ── Track worst component for diagnostic decomposition ──
            // Save physical-space four-term breakdown per component
            auto phys = [&](double v, int i, bool is_u) -> double {
                if (!params_.enable_preconditioner) return v;
                return v / (is_u ? prec_.inv_Lu(k)[i] : prec_.inv_Lx(k)[i]);
            };
            for (int i = 0; i < NX; ++i) {
                double rel = std::fabs(lag_x[i]) / scale;
                if (rel > worst_rel) {
                    worst_rel = rel;
                    worst_k = k; worst_is_u = false; worst_i = i;
                    worst_q     = phys(s[k].qx[i], i, false);
                    worst_bar   = phys(bound_grad_x[i], i, false);
                    worst_ctw   = phys(ct_lam_x[i], i, false);
                    worst_cos   = phys(cos_x[i], i, false);
                    worst_scale = scale;
                }
            }
            if (k < N) {
                for (int i = 0; i < NU; ++i) {
                    double rel = std::fabs(lag_u[i]) / scale;
                    if (rel > worst_rel) {
                        worst_rel = rel;
                        worst_k = k; worst_is_u = true; worst_i = i;
                        worst_q     = phys(s[k].qu[i], i, true);
                        worst_bar   = phys(bound_grad_u[i], i, true);
                        worst_ctw   = phys(ct_lam_u[i], i, true);
                        worst_cos   = phys(cos_u[i], i, true);
                        worst_scale = scale;
                    }
                }
            }
        }

        // ── Note: barrier gradient not in Newton RHS ──
        // Bound constraints handled via Hessian curvature + FTB step limits.

        // ── Compute global barrier metrics ──
        for (int kk = 0; kk <= N; ++kk) {
            if (prob_->n_bound_u > 0 && kk < N) {
                for (int i = 0; i < NU; ++i) {
                    double dL = std::max(s[kk].u[i] - prob_->u_lb[i], 0.0) + params_.bound_s_min;
                    double dU = std::max(prob_->u_ub[i] - s[kk].u[i], 0.0) + params_.bound_s_min;
                    if (dL < global_min_dist) { global_min_dist = dL; global_min_dist_k = kk; global_min_dist_u = i; }
                    if (dU < global_min_dist) { global_min_dist = dU; global_min_dist_k = kk; global_min_dist_u = i; }
                    double bg = std::fabs(-mu_ / dL + mu_ / dU);
                    if (bg > global_max_bar_grad) global_max_bar_grad = bg;
                    double bh = mu_ / (dL * dL) + mu_ / (dU * dU);
                    if (bh > global_max_bar_hess) global_max_bar_hess = bh;
                }
            }
            if (prob_->n_bound_x > 0) {
                for (int i = 0; i < NX; ++i) {
                    double dL = std::max(s[kk].x[i] - prob_->x_lb[i], 0.0) + params_.bound_s_min;
                    double dU = std::max(prob_->x_ub[i] - s[kk].x[i], 0.0) + params_.bound_s_min;
                    if (dL < global_min_dist) { global_min_dist = dL; global_min_dist_k = kk; global_min_dist_u = i + NU; }
                    if (dU < global_min_dist) { global_min_dist = dU; global_min_dist_k = kk; global_min_dist_u = i + NU; }
                }
            }
        }

        // ── Print worst-component diagnostic ──
        if (params_.verbosity >= 2 && worst_k >= 0) {
            printf("  [stat-decomp] worst: k=%d %s[%d]  rel=%.4f  scale=%.3e\n",
                   worst_k, worst_is_u ? "u" : "x", worst_i,
                   worst_rel, worst_scale);
            printf("    q     = %+.6e  (cost gradient)\n", worst_q);
            printf("    C^T*λ = %+.6e  (constraint dual)\n", worst_ctw);
            printf("    cos   = %+.6e  (costate A^Tp - p)\n", worst_cos);
            printf("    bar   = %+.6e  (bound barrier gradient)\n", worst_bar);
            printf("    sum   = %+.6e  (should = ∇L with barrier)\n",
                   worst_q + worst_ctw + worst_cos + worst_bar);
            if (global_min_dist < 1e30) {
                printf("  [bar-metrics] min_dist=%.3e at k=%d comp=%d\n",
                       global_min_dist, global_min_dist_k, global_min_dist_u);
                printf("    max|bar_grad|=%.3e  max(bar_hess_diag)=%.3e\n",
                       global_max_bar_grad, global_max_bar_hess);
            }
        }

        // Override stat_inf_ with Riccati-consistent stationarity.
        stat_inf_ = post_stat;
    }

    // ═════════════════════════════════════════════════════════════════════
    //  Compute KKT residuals of the barrier subproblem
    // ═════════════════════════════════════════════════════════════════════

    // Refresh the residual blocks consumed by the barrier manager after an
    // accepted step. evaluate_model() must have left c and d in physical space.
    void refresh_barrier_metrics() {
        const int N = HORIZON;
        Stage* s = prob_->stages;

        primal_inf_ = 0.0;
        dyn_defect_ = 0.0;
        cons_viol_ = 0.0;
        max_g_pos_ = 0.0;
        compl_inf_ = 0.0;
        ineq_viol_ = 0.0;

        for (int k = 0; k <= N; ++k) {
            if (k < N) {
                for (int i = 0; i < NX; ++i)
                    dyn_defect_ = std::max(dyn_defect_, std::fabs(s[k].c[i]));
                primal_inf_ = std::max(primal_inf_, dyn_defect_);
            }

            if (prob_->constraints) {
                for (int j = 0; j < active_constraints(k); ++j) {
                    double viol = std::fabs(s[k].d[j] + s[k].s[j]);
                    cons_viol_ = std::max(cons_viol_, viol);
                    primal_inf_ = std::max(primal_inf_, viol);
                    max_g_pos_ = std::max(max_g_pos_, s[k].d[j]);
                    compl_inf_ = std::max(
                        compl_inf_, std::fabs(s[k].s[j] * s[k].lambda[j] - mu_));
                    ineq_viol_ = std::min(
                        ineq_viol_, std::min(s[k].s[j], s[k].lambda[j]));
                }
            }

            if (prob_->n_bound_u > 0 && k < N) {
                for (int i = 0; i < NU; ++i) {
                    double raw_dL = s[k].u[i] - prob_->u_lb[i];
                    double raw_dU = prob_->u_ub[i] - s[k].u[i];
                    double dL = std::max(raw_dL, 0.0) + params_.bound_s_min;
                    double dU = std::max(raw_dU, 0.0) + params_.bound_s_min;
                    primal_inf_ = std::max(primal_inf_, std::max(-raw_dL, -raw_dU));
                    ineq_viol_ = std::min(ineq_viol_, std::min(raw_dL, raw_dU));
                    compl_inf_ = std::max(
                        compl_inf_, std::fabs(dL * s[k].z_L_u[i] - mu_));
                    compl_inf_ = std::max(
                        compl_inf_, std::fabs(dU * s[k].z_U_u[i] - mu_));
                    ineq_viol_ = std::min(
                        ineq_viol_, std::min(s[k].z_L_u[i], s[k].z_U_u[i]));
                }
            }

            if (prob_->n_bound_x > 0) {
                for (int i = 0; i < NX; ++i) {
                    double raw_dL = s[k].x[i] - prob_->x_lb[i];
                    double raw_dU = prob_->x_ub[i] - s[k].x[i];
                    double dL = std::max(raw_dL, 0.0) + params_.bound_s_min;
                    double dU = std::max(raw_dU, 0.0) + params_.bound_s_min;
                    primal_inf_ = std::max(primal_inf_, std::max(-raw_dL, -raw_dU));
                    ineq_viol_ = std::min(ineq_viol_, std::min(raw_dL, raw_dU));
                    compl_inf_ = std::max(
                        compl_inf_, std::fabs(dL * s[k].z_L_x[i] - mu_));
                    compl_inf_ = std::max(
                        compl_inf_, std::fabs(dU * s[k].z_U_x[i] - mu_));
                    ineq_viol_ = std::min(
                        ineq_viol_, std::min(s[k].z_L_x[i], s[k].z_U_x[i]));
                }
            }
        }
    }

    void compute_kkt_residuals() {
        PhaseTimer timer(active_stats_ ? &active_stats_->residual_eval_time_ms
                                       : nullptr);
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
                for (int j = 0; j < active_constraints(N); ++j) {
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
                    for (int j = 0; j < active_constraints(k); ++j) {
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
                    evaluate_constraints(s[k].x, s[k].u, k, g_val);
                } else {
                    evaluate_terminal_constraints(s[k].x, g_val);
                }
                for (int j = 0; j < active_constraints(k); ++j) {
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
                for (int j = 0; j < active_constraints(k); ++j) {
                    double c = std::fabs(s[k].s[j] * s[k].lambda[j] - mu_);
                    if (c > compl_inf_) compl_inf_ = c;
                }
            }

            // ── 3. Inequality: s_j > 0, λ_j > 0 ──────────────────
            if (prob_->n_bound_u > 0 && k < N) {
                for (int i = 0; i < NU; ++i) {
                    double raw_dL = s[k].u[i] - prob_->u_lb[i];
                    double raw_dU = prob_->u_ub[i] - s[k].u[i];
                    double dL = std::max(raw_dL, 0.0) + params_.bound_s_min;
                    double dU = std::max(raw_dU, 0.0) + params_.bound_s_min;
                    primal_inf_ = std::max(primal_inf_, std::max(-raw_dL, -raw_dU));
                    ineq_viol_ = std::min(ineq_viol_, std::min(raw_dL, raw_dU));
                    compl_inf_ = std::max(compl_inf_, std::fabs(dL * s[k].z_L_u[i] - mu_));
                    compl_inf_ = std::max(compl_inf_, std::fabs(dU * s[k].z_U_u[i] - mu_));
                }
            }
            if (prob_->n_bound_x > 0) {
                for (int i = 0; i < NX; ++i) {
                    double raw_dL = s[k].x[i] - prob_->x_lb[i];
                    double raw_dU = prob_->x_ub[i] - s[k].x[i];
                    double dL = std::max(raw_dL, 0.0) + params_.bound_s_min;
                    double dU = std::max(raw_dU, 0.0) + params_.bound_s_min;
                    primal_inf_ = std::max(primal_inf_, std::max(-raw_dL, -raw_dU));
                    ineq_viol_ = std::min(ineq_viol_, std::min(raw_dL, raw_dU));
                    compl_inf_ = std::max(compl_inf_, std::fabs(dL * s[k].z_L_x[i] - mu_));
                    compl_inf_ = std::max(compl_inf_, std::fabs(dU * s[k].z_U_x[i] - mu_));
                }
            }
            // Track worst violation (most negative): 0 means all satisfied
            for (int j = 0; j < active_constraints(k); ++j) {
                if (s[k].s[j] < ineq_viol_)      ineq_viol_ = s[k].s[j];
                if (s[k].lambda[j] < ineq_viol_)  ineq_viol_ = s[k].lambda[j];
            }

            // ── 4. Stationarity: barrier Lagrangian gradient ─────
            if (prob_->n_bound_u > 0 && k < N)
                for (int i = 0; i < NU; ++i) {
                    if (s[k].z_L_u[i] < ineq_viol_) ineq_viol_ = s[k].z_L_u[i];
                    if (s[k].z_U_u[i] < ineq_viol_) ineq_viol_ = s[k].z_U_u[i];
                }
            if (prob_->n_bound_x > 0)
                for (int i = 0; i < NX; ++i) {
                    if (s[k].z_L_x[i] < ineq_viol_) ineq_viol_ = s[k].z_L_x[i];
                    if (s[k].z_U_x[i] < ineq_viol_) ineq_viol_ = s[k].z_U_x[i];
                }
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
            // (1-σ)·C^T·(μ/s) gap from the primal-dual centering, creating
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

                if (prob_->n_bound_x > 0)
                    for (int i = 0; i < NX; ++i) {
                        double scale = params_.enable_preconditioner ? prec_.inv_Lx(k)[i] : 1.0;
                        lag_x[i] += scale * (-s[k].z_L_x[i] + s[k].z_U_x[i]);
                    }
                if (prob_->n_bound_u > 0 && k < N)
                    for (int i = 0; i < NU; ++i) {
                        double scale = params_.enable_preconditioner ? prec_.inv_Lu(k)[i] : 1.0;
                        lag_u[i] += scale * (-s[k].z_L_u[i] + s[k].z_U_u[i]);
                    }
                // Track component magnitudes (in scaled space, for diagnostics)
                double grad_x_inf = lag_x.norm_inf();
                double grad_u_inf = lag_u.norm_inf();

                // Compute bound barrier gradient for DIAGNOSTICS only.
                // NOT added to lag: Riccati stationarity = q + B^T·p (no barrier).
                double bound_grad_x_inf = 0.0, bound_grad_u_inf = 0.0;
                if (prob_->n_bound_x > 0) {
                    for (int i = 0; i < NX; ++i) {
                        double dL = std::max(s[k].x[i] - prob_->x_lb[i], 0.0) + params_.bound_s_min;
                        double dU = std::max(prob_->x_ub[i] - s[k].x[i], 0.0) + params_.bound_s_min;
                        bound_grad_x_inf = std::max(bound_grad_x_inf, std::fabs(-s[k].z_L_x[i] + s[k].z_U_x[i]));
                    }
                }
                if (prob_->n_bound_u > 0 && k < N) {
                    for (int i = 0; i < NU; ++i) {
                        double dL = std::max(s[k].u[i] - prob_->u_lb[i], 0.0) + params_.bound_s_min;
                        double dU = std::max(prob_->u_ub[i] - s[k].u[i], 0.0) + params_.bound_s_min;
                        bound_grad_u_inf = std::max(bound_grad_u_inf, std::fabs(-s[k].z_L_u[i] + s[k].z_U_u[i]));
                    }
                }

                // Add inequality constraint dual: C^T·λ
                Vec<NX> ct_lam_x; ct_lam_x.zero();
                Vec<NU> ct_lam_u; ct_lam_u.zero();
                if (prob_->constraints) {
                    for (int j = 0; j < active_constraints(k); ++j) {
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
                // EXCLUDE costate (cos_x, cos_u) from scale: at KKT the costate
                // balances the other terms, so including it makes stat → 1.0.
                double grad_x_phys = grad_x_inf;
                double grad_u_phys = grad_u_inf;
                double clam_x_phys = ct_lam_x.norm_inf();
                double clam_u_phys = ct_lam_u.norm_inf();
                if (params_.enable_preconditioner) {
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
                double scale_x = std::max({grad_x_phys, clam_x_phys, 1.0});
                double scale_u = std::max({grad_u_phys, clam_u_phys, 1.0});
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

        // Physical MPCC residual is separate from the barrier pair s*lambda.
        // For a marked pair, g_a*g_b equals the product of the two
        // nonnegative physical quantities because both rows use the same sign.
        mpcc_inf_ = compute_mpcc_complementarity();

        // Log worst constraint location (diagnostic)
        if (params_.verbosity >= 2 && worst_cons_k >= 0 && worst_cons_k < N) {
            Vec<NC> g_worst;
            evaluate_constraints(s[worst_cons_k].x, s[worst_cons_k].u,
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
                for (int jj = 0; jj < active_constraints(kk); ++jj) {
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
        bool ineq_ok    = (ineq_viol_ >= -params_.tol_ineq &&
                           max_g_pos_ <= params_.tol_ineq);
        bool stat_ok    = (stat_inf_ <= params_.tol_stat);
        bool mpcc_ok    = (!has_complementarity() ||
                           mpcc_inf_ <= params_.tol_mpcc);
        // μ threshold — prevents false convergence at large μ.
        // However, if all KKT residuals are small, the solution is genuinely
        // converged regardless of μ. The μ threshold is only needed when
        // the residuals are borderline.
        bool mu_ok      = (mu_ <= params_.mu_conv_threshold);
        bool kkt_satisfied = primal_ok && compl_ok && ineq_ok && stat_ok && mpcc_ok;
        // Declare convergence if all KKT criteria are satisfied, OR if
        // μ is small enough and KKT is nearly satisfied.
        bool converged  = kkt_satisfied || (mu_ok && primal_ok && compl_ok && ineq_ok && mpcc_ok && (stat_inf_ <= 10.0 * params_.tol_stat));
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
    //        q̃^x = qx + C^T·(σμ + λ(g+s))/s  (= qx + C^T·λ + C^T·w),
    //        q̃^u = qu + C^T·(σμ + λ(g+s))/s,
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

            // Compute q̃^x = qx + C^T·λ + C^T·w = qx + C^T·(σμ + λ(g+s))/s
            Vec<NX> qx_tilde = s[k].qx;
            Vec<NU> qu_tilde = s[k].qu;
            if (prob_->constraints) {
                for (int j = 0; j < active_constraints(k); ++j) {
                    double sj = s[k].s[j];
                    if (sj > 1e-14) {
                        double lj = s[k].lambda[j];
                        double gj = s[k].d[j] + sj;
                        double centering = (sigma_*mu_ + lj*gj) / sj;
                        // q̃ += C^T·(λ + w) = C^T·centering
                        for (int i = 0; i < NX; ++i) qx_tilde[i] += s[k].Cx(j,i) * centering;
                        for (int i = 0; i < NU; ++i) qu_tilde[i] += s[k].Cu(j,i) * centering;
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
                for (int j = 0; j < active_constraints(N); ++j) {
                    double sj = s[N].s[j];
                    if (sj > 1e-14) {
                        double lj = s[N].lambda[j];
                        double gj = s[N].d[j] + sj;
                        double centering = (sigma_*mu_ + lj*gj) / sj;
                        // q̃ += C^T·(λ + w) = C^T·centering
                        for (int i = 0; i < NX; ++i) qxN_tilde[i] += s[N].Cx(j,i) * centering;
                    }
                }
            }
            double qxNn = qxN_tilde.norm_inf();
            if (qxNn > res.rhs_stat_term_norm) res.rhs_stat_term_norm = qxNn;
        }
        // ── Collect complementarity RHS norms ──
        // The full Newton complementarity equation is:
        //   s·Δλ + λ·Δs = σμ - sλ
        // RHS norm should include |sλ| (not just σμ), otherwise tiny σμ
        // makes relative residual blow up even when absolute residual is small.
        res.rhs_comp_norm = std::fabs(smu);
        for (int k = 0; k <= N; ++k) {
            if (prob_->constraints) {
                for (int j = 0; j < active_constraints(k); ++j) {
                    double sl = std::fabs(s[k].s[j] * s[k].lambda[j]);
                    double rhs_j = std::max(sl, std::fabs(smu));
                    if (rhs_j > res.rhs_comp_norm) res.rhs_comp_norm = rhs_j;
                }
            }
            if (prob_->n_bound_u > 0 && k < N) {
                for (int i = 0; i < NU; ++i) {
                    double dL = std::max(s[k].u[i] - prob_->u_lb[i], 0.0)
                              + params_.bound_s_min;
                    double dU = std::max(prob_->u_ub[i] - s[k].u[i], 0.0)
                              + params_.bound_s_min;
                    res.rhs_comp_norm = std::max(res.rhs_comp_norm,
                        std::max(std::fabs(dL * s[k].z_L_u[i]),
                                 std::fabs(dU * s[k].z_U_u[i])));
                }
            }
            if (prob_->n_bound_x > 0) {
                for (int i = 0; i < NX; ++i) {
                    double dL = std::max(s[k].x[i] - prob_->x_lb[i], 0.0)
                              + params_.bound_s_min;
                    double dU = std::max(prob_->x_ub[i] - s[k].x[i], 0.0)
                              + params_.bound_s_min;
                    res.rhs_comp_norm = std::max(res.rhs_comp_norm,
                        std::max(std::fabs(dL * s[k].z_L_x[i]),
                                 std::fabs(dU * s[k].z_U_x[i])));
                }
            }
        }

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
                for (int j = 0; j < active_constraints(k); ++j) {
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
            // q̃ = qx + C^T·λ + C^T·w = qx + C^T·(σμ + λ(g+s))/s
            Vec<NX> qx_tilde = stg.qx;
            Vec<NU> qu_tilde = stg.qu;
            if (prob_->n_bound_u > 0 && k < N) {
                for (int i = 0; i < NU; ++i) {
                    double dL = std::max(stg.u[i] - prob_->u_lb[i], 0.0)
                              + params_.bound_s_min;
                    double dU = std::max(prob_->u_ub[i] - stg.u[i], 0.0)
                              + params_.bound_s_min;
                    double scale = params_.enable_preconditioner
                                 ? prec_.inv_Lu(k)[i] : 1.0;
                    double rcL = dL * stg.z_L_u[i] - sigma_ * mu_;
                    double rcU = dU * stg.z_U_u[i] - sigma_ * mu_;
                    qu_tilde[i] += scale * (-stg.z_L_u[i] + stg.z_U_u[i]
                        + rcL / dL - rcU / dU);
                }
            }
            if (prob_->n_bound_x > 0) {
                for (int i = 0; i < NX; ++i) {
                    double dL = std::max(stg.x[i] - prob_->x_lb[i], 0.0)
                              + params_.bound_s_min;
                    double dU = std::max(prob_->x_ub[i] - stg.x[i], 0.0)
                              + params_.bound_s_min;
                    double scale = params_.enable_preconditioner
                                 ? prec_.inv_Lx(k)[i] : 1.0;
                    double rcL = dL * stg.z_L_x[i] - sigma_ * mu_;
                    double rcU = dU * stg.z_U_x[i] - sigma_ * mu_;
                    qx_tilde[i] += scale * (-stg.z_L_x[i] + stg.z_U_x[i]
                        + rcL / dL - rcU / dU);
                }
            }
            if (prob_->constraints) {
                for (int j = 0; j < active_constraints(k); ++j) {
                    double sj = stg.s[j];
                    if (sj > 1e-14) {
                        double lj = stg.lambda[j];
                        double gj = stg.d[j] + sj;
                        double centering = (sigma_*mu_ + lj*gj) / sj;
                        // q̃ += C^T·(λ + w) = C^T·centering
                        for (int i = 0; i < NX; ++i) qx_tilde[i] += stg.Cx(j,i) * centering;
                        if (k < N)
                            for (int i = 0; i < NU; ++i) qu_tilde[i] += stg.Cu(j,i) * centering;
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
                // Use MODIFIED gradient q̃x = qx + Cx^T·(σμ+λ(g+s))/s
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
            // Primal-dual Newton: s·Δλ + λ·Δs = σμ - sλ
            //   residual = s·Δλ + λ·Δs + sλ - σμ
            if (prob_->constraints) {
                for (int j = 0; j < active_constraints(k); ++j) {
                    double sl = stg.s[j] * stg.lambda[j];
                    double res_c = stg.s[j] * dlambda_[k][j]
                                 + stg.lambda[j] * ds_[k][j]
                                 + sl
                                 - smu;
                    double ar = std::fabs(res_c);
                    if (ar > res.max_comp_res) { res.max_comp_res = ar; res.worst_stage = k; res.worst_eq_type = 4; }
                }
            }
            if (prob_->n_bound_u > 0 && k < N) {
                for (int i = 0; i < NU; ++i) {
                    double dL = std::max(stg.u[i] - prob_->u_lb[i], 0.0)
                              + params_.bound_s_min;
                    double dU = std::max(prob_->u_ub[i] - stg.u[i], 0.0)
                              + params_.bound_s_min;
                    double rL = dL * dz_L_u_[k][i] + stg.z_L_u[i] * ws.du[k][i]
                              + dL * stg.z_L_u[i] - smu;
                    double rU = dU * dz_U_u_[k][i] - stg.z_U_u[i] * ws.du[k][i]
                              + dU * stg.z_U_u[i] - smu;
                    double ar = std::max(std::fabs(rL), std::fabs(rU));
                    if (ar > res.max_comp_res) {
                        res.max_comp_res = ar;
                        res.worst_stage = k;
                        res.worst_eq_type = 4;
                    }
                }
            }
            if (prob_->n_bound_x > 0) {
                for (int i = 0; i < NX; ++i) {
                    double dL = std::max(stg.x[i] - prob_->x_lb[i], 0.0)
                              + params_.bound_s_min;
                    double dU = std::max(prob_->x_ub[i] - stg.x[i], 0.0)
                              + params_.bound_s_min;
                    double rL = dL * dz_L_x_[k][i] + stg.z_L_x[i] * ws.dx[k][i]
                              + dL * stg.z_L_x[i] - smu;
                    double rU = dU * dz_U_x_[k][i] - stg.z_U_x[i] * ws.dx[k][i]
                              + dU * stg.z_U_x[i] - smu;
                    double ar = std::max(std::fabs(rL), std::fabs(rU));
                    if (ar > res.max_comp_res) {
                        res.max_comp_res = ar;
                        res.worst_stage = k;
                        res.worst_eq_type = 4;
                    }
                }
            }
        }

        // ── Compute relative residuals ────────────────────────────
        // Denominator clamped at 1.0: when RHS is tiny, absolute residual
        // is the meaningful metric.  Prevents false "POOR" labels from
        // vanishing RHS norms.
        res.rel_dyn_res        = res.max_dyn_res        / std::max(res.rhs_dyn_norm, 1.0);
        res.rel_feas_res       = res.max_feas_res       / std::max(res.rhs_feas_norm, 1.0);
        res.rel_stat_x_res     = res.max_stat_x_res     / std::max(res.rhs_stat_x_norm, 1.0);
        res.rel_stat_u_res     = res.max_stat_u_res     / std::max(res.rhs_stat_u_norm, 1.0);
        res.rel_stat_term_res  = res.max_stat_term_res  / std::max(res.rhs_stat_term_norm, 1.0);
        res.rel_comp_res       = res.max_comp_res       / std::max(res.rhs_comp_norm, 1.0);

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
            // ── Start with the stage Hessian (already scaled by transform_qp).
            //    With exact_hessian this is the full Lagrangian Hessian
            //    (cost + dynamics·p + constraints·λ); otherwise Gauss-Newton
            //    (cost only).  The barrier Hessian is added below. ──
            qxx_work_[k] = s[k].Qxx;
            quu_work_[k] = s[k].Quu;
            qux_work_[k] = s[k].Qux;

            if (prob_->constraints) {
                // ── Barrier Hessian: H_bar = C^T·diag(λ/s)·C ────────
                // Uses SCALED Cx/Cu (from transform_qp) so the barrier
                // Hessian is consistent with the scaled cost Hessian.
                for (int j = 0; j < active_constraints(k); ++j) {
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
                    double dL = std::max(s[k].u[i] - prob_->u_lb[i], 0.0) + params_.bound_s_min;
                    double dU = std::max(prob_->u_ub[i] - s[k].u[i], 0.0) + params_.bound_s_min;
                    double scale = params_.enable_preconditioner ? prec_.inv_Lu(k)[i] : 1.0;
                    quu_work_[k](i, i) += scale * scale
                        * (s[k].z_L_u[i] / dL + s[k].z_U_u[i] / dU);
                }
            }
            if (prob_->n_bound_x > 0) {
                for (int i = 0; i < NX; ++i) {
                    double dL = std::max(s[k].x[i] - prob_->x_lb[i], 0.0) + params_.bound_s_min;
                    double dU = std::max(prob_->x_ub[i] - s[k].x[i], 0.0) + params_.bound_s_min;
                    double scale = params_.enable_preconditioner ? prec_.inv_Lx(k)[i] : 1.0;
                    qxx_work_[k](i, i) += scale * scale
                        * (s[k].z_L_x[i] / dL + s[k].z_U_x[i] / dU);
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
    //  Build RHS (gradient + barrier correction) using the FSM-scheduled σ.
    // ═══════════════════════════════════════════════════════════════════

    void build_kkt_rhs() {
        const int N = HORIZON;
        Stage* s = prob_->stages;

        for (int k = 0; k <= N; ++k) {
            // ── Gradient = cost gradient ───────────────────────────
            qx_work_[k] = s[k].qx;
            qu_work_[k] = s[k].qu;

            if (prob_->constraints) {
                // ── Lagrangian gradient: +C^T·λ ────────────────────
                // The Newton linearization of x-stationarity is:
                //   (qx + Cx^T·λ + A^T·p - p) + H̄·Δz + Cx^T·Δλ + A^T·Δp - Δp = 0
                // After eliminating Δλ = w + (λ/s)·C·Δz, the constant part is:
                //   q̃_x = qx + Cx^T·λ + Cx^T·w = qx + Cx^T·(λ + w)
                // where λ + w = (σμ + λ(g+s))/s.
                for (int j = 0; j < active_constraints(k); ++j) {
                    double lj = s[k].lambda[j];
                    for (int i = 0; i < NX; ++i)
                        qx_work_[k][i] += s[k].Cx(j, i) * lj;
                    if (k < N)
                        for (int i = 0; i < NU; ++i)
                            qu_work_[k][i] += s[k].Cu(j, i) * lj;
                }

                // ── Newton-effective dual: +C^T·w ──────────────────
                // From Schur complement elimination of (Δs, Δλ):
                //   Δλ = w + (λ/s)·C·Δz
                //   w = (σμ + λ(g+s))/s − λ
                // The λ/s·C·Δz part goes to LHS as barrier Hessian.
                // The constant part w goes to RHS as centering force.
                // Combined with +C^T·λ above: q̃ = qx + C^T·(λ + w)
                //                              = qx + C^T·(σμ + λ(g+s))/s
                for (int j = 0; j < active_constraints(k); ++j) {
                    double sj = s[k].s[j];
                    if (sj < 1e-14) continue;
                    double lj = s[k].lambda[j];
                    double gj = s[k].d[j] + sj;  // g+s at linearization
                    double w = (sigma_ * mu_ + lj * gj) / sj - lj;

                    for (int i = 0; i < NX; ++i)
                        qx_work_[k][i] += s[k].Cx(j, i) * w;
                    if (k < N)
                        for (int i = 0; i < NU; ++i)
                            qu_work_[k][i] += s[k].Cu(j, i) * w;
                }
            }

            // Variable-bound log-barrier gradient.  The corresponding
            // curvature is added in build_kkt_lhs(); omitting this first
            // derivative solves a different Newton system and leaves a large
            // stationarity floor at active bounds.
            if (prob_->n_bound_u > 0 && k < N) {
                for (int i = 0; i < NU; ++i) {
                    double dL = std::max(s[k].u[i] - prob_->u_lb[i], 0.0)
                              + params_.bound_s_min;
                    double dU = std::max(prob_->u_ub[i] - s[k].u[i], 0.0)
                              + params_.bound_s_min;
                    double scale = params_.enable_preconditioner
                                 ? prec_.inv_Lu(k)[i] : 1.0;
                    double rcL = dL * s[k].z_L_u[i] - sigma_ * mu_;
                    double rcU = dU * s[k].z_U_u[i] - sigma_ * mu_;
                    double bound_rhs = -s[k].z_L_u[i] + s[k].z_U_u[i]
                                     + rcL / dL - rcU / dU;
                    qu_work_[k][i] += scale * bound_rhs;
                }
            }
            if (prob_->n_bound_x > 0) {
                for (int i = 0; i < NX; ++i) {
                    double dL = std::max(s[k].x[i] - prob_->x_lb[i], 0.0)
                              + params_.bound_s_min;
                    double dU = std::max(prob_->x_ub[i] - s[k].x[i], 0.0)
                              + params_.bound_s_min;
                    double scale = params_.enable_preconditioner
                                 ? prec_.inv_Lx(k)[i] : 1.0;
                    double rcL = dL * s[k].z_L_x[i] - sigma_ * mu_;
                    double rcU = dU * s[k].z_U_x[i] - sigma_ * mu_;
                    double bound_rhs = -s[k].z_L_x[i] + s[k].z_U_x[i]
                                     + rcL / dL - rcU / dU;
                    qx_work_[k][i] += scale * bound_rhs;
                }
            }

            // Transfer to Riccati stages
            riccati_stages_[k].qx  = qx_work_[k];
            riccati_stages_[k].qu  = qu_work_[k];
        }
    }

    // ═════════════════════════════════════════════════════════════════════
    //  Solve the reduced KKT via Riccati recursion.
    // ═════════════════════════════════════════════════════════════════════

    Status solve_kkt_lhs() {
        // Apply configured dimensionless damping relative to the Riccati value
        // Hessian and stage Schur diagonals; retain the existing retry policy.
        reg_used_ = 0.0;  // Reset each call — measure actual reg used
        return Ricc::backward_lhs(
            riccati_stages_, riccati_ws_,
            params_.riccati_relative_regularization, reg_used_, params_.reg_max,
            params_.inertia_min_pivot);
    }

    Status solve_kkt_rhs_and_forward() {
        const bool riccati_diagnostics = params_.verbosity >= 2;
        Status st = Ricc::backward_rhs(
            riccati_stages_, riccati_ws_, riccati_diagnostics);
        if (st != Status::SUCCESS) return st;

        Vec<NX> dx0;
        for (int i = 0; i < NX; ++i)
            dx0[i] = prob_->x0[i] - prob_->stages[0].x[i];

        // Scale dx0 for Riccati forward pass (dx̂0 = Lx·dx0)
        if (params_.enable_preconditioner) {
            prec_.scale_dx0(dx0);
        }

        return Ricc::forward(
            riccati_stages_, riccati_ws_, dx0, riccati_diagnostics);
    }

    // Legacy combined solve (kept for backward compat with tests)
    Status solve_kkt_via_riccati() {
        const bool riccati_diagnostics = params_.verbosity >= 2;
        Status st = Ricc::backward(
            riccati_stages_, riccati_ws_,
            params_.riccati_relative_regularization, reg_used_,
            riccati_diagnostics);
        if (st != Status::SUCCESS) return st;

        Vec<NX> dx0;
        for (int i = 0; i < NX; ++i)
            dx0[i] = prob_->x0[i] - prob_->stages[0].x[i];

        return Ricc::forward(
            riccati_stages_, riccati_ws_, dx0, riccati_diagnostics);
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
            for (int j = 0; j < active_constraints(k); ++j) {
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
    //  Recover Δs, Δλ from primal step (primal-dual Newton direction)
    // ═════════════════════════════════════════════════════════════════════

    void recover_inequality_steps(double sigma) {
        const int N = HORIZON;
        Stage* s = prob_->stages;

        // ── Compute Δs, Δλ from primal-dual complementarity linearization ──
        for (int k = 0; k <= N; ++k) {
            // Rows beyond num_constraints(k) are storage padding, not KKT
            // variables. Clear their directions to avoid stale values.
            for (int j = active_constraints(k); j < NC; ++j)
                ds_[k][j] = dlambda_[k][j] = 0.0;

            for (int j = 0; j < active_constraints(k); ++j) {
                double gj = s[k].d[j];
                double cj_dz = 0.0;
                for (int i = 0; i < NX; ++i)
                    cj_dz += s[k].Cx(j, i) * riccati_ws_.dx[k][i];
                if (k < N)
                    for (int i = 0; i < NU; ++i)
                        cj_dz += s[k].Cu(j, i) * riccati_ws_.du[k][i];

                ds_[k][j] = -(gj + s[k].s[j]) - cj_dz;

                if (s[k].s[j] > 1e-14) {
                    // Primal-dual complementarity linearization:
                    //   s·dλ + λ·ds = σμ - sλ
                    //   dλ = (σμ - sλ - λ·ds) / s
                    double sl_term = s[k].s[j] * s[k].lambda[j];
                    dlambda_[k][j] = (sigma * mu_ - sl_term
                                      - s[k].lambda[j] * ds_[k][j]) / s[k].s[j];
                } else {
                    dlambda_[k][j] = 0.0;
                }
            }
        }
    }

    // ═════════════════════════════════════════════════════════════════════
    // Fraction-to-boundary step size (for the primal-dual Newton direction)
    // ═════════════════════════════════════════════════════════════════════

    void recover_bound_multiplier_steps(double sigma, bool primal_step_scaled) {
        const double target = sigma * mu_;
        const double bsm = params_.bound_s_min;
        Stage* s = prob_->stages;
        for (int k = 0; k <= HORIZON; ++k) {
            if (prob_->n_bound_u > 0 && k < HORIZON) {
                for (int i = 0; i < NU; ++i) {
                    double du = riccati_ws_.du[k][i];
                    if (primal_step_scaled && params_.enable_preconditioner)
                        du *= prec_.inv_Lu(k)[i];
                    double dL = std::max(s[k].u[i] - prob_->u_lb[i], 0.0) + bsm;
                    double dU = std::max(prob_->u_ub[i] - s[k].u[i], 0.0) + bsm;
                    dz_L_u_[k][i] = (target - dL * s[k].z_L_u[i]
                                      - s[k].z_L_u[i] * du) / dL;
                    dz_U_u_[k][i] = (target - dU * s[k].z_U_u[i]
                                      + s[k].z_U_u[i] * du) / dU;
                }
            }
            if (prob_->n_bound_x > 0) {
                for (int i = 0; i < NX; ++i) {
                    double dx = riccati_ws_.dx[k][i];
                    if (primal_step_scaled && params_.enable_preconditioner)
                        dx *= prec_.inv_Lx(k)[i];
                    double dL = std::max(s[k].x[i] - prob_->x_lb[i], 0.0) + bsm;
                    double dU = std::max(prob_->x_ub[i] - s[k].x[i], 0.0) + bsm;
                    dz_L_x_[k][i] = (target - dL * s[k].z_L_x[i]
                                      - s[k].z_L_x[i] * dx) / dL;
                    dz_U_x_[k][i] = (target - dU * s[k].z_U_x[i]
                                      + s[k].z_U_x[i] * dx) / dU;
                }
            }
        }
    }

    bool use_primal_dual_bound() const {
        return params_.primal_dual_bounds && bound_pd_mode_;
    }

    double fraction_to_boundary() {
        const int N = HORIZON;
        double alpha = 1.0;

        for (int k = 0; k <= N; ++k) {
            Stage& s = prob_->stages[k];
            // Slack/lambda FTB: only when constraints exist
            if (prob_->constraints) {
                for (int j = 0; j < active_constraints(k); ++j) {
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

    void apply_primal_dual_step(double alpha, double alpha_lam,
                                int trial_candidate = 0) {
        const int N = HORIZON;
        Stage* s = prob_->stages;
        const double bsm = params_.bound_s_min;
        const double alpha_bound_dual = std::min(alpha, alpha_lam);
        for (int k = 0; k <= N; ++k) {
            if (trial_candidate == 1) {
                s[k].x = rollout_x_[k];
                if (k < N) s[k].u = rollout_u_[k];
            } else {
                for (int i = 0; i < NX; ++i)
                    s[k].x[i] += alpha * riccati_ws_.dx[k][i];
                if (k < N)
                    for (int i = 0; i < NU; ++i)
                        s[k].u[i] += alpha * riccati_ws_.du[k][i];
            }
            for (int j = 0; j < active_constraints(k); ++j) {
                if (trial_candidate == 1)
                    s[k].s[j] = rollout_s_[k][j];
                else
                    s[k].s[j] += alpha * ds_[k][j];
                s[k].lambda[j] += alpha_bound_dual * dlambda_[k][j];
            }
            // Update bound multipliers: z = mu / d (with floor safeguard)
            if (prob_->n_bound_u > 0 && k < N) {
                for (int i = 0; i < NU; ++i) {
                    double dL = std::max(s[k].u[i] - prob_->u_lb[i], 0.0) + bsm;
                    double dU = std::max(prob_->u_ub[i] - s[k].u[i], 0.0) + bsm;
                    if (use_primal_dual_bound()) {
                        s[k].z_L_u[i] += alpha_bound_dual * dz_L_u_[k][i];
                        s[k].z_L_u[i] = std::max(mu_ / (1e10 * dL),
                            std::min(s[k].z_L_u[i], 1e10 * mu_ / dL));
                        s[k].z_U_u[i] += alpha_bound_dual * dz_U_u_[k][i];
                        s[k].z_U_u[i] = std::max(mu_ / (1e10 * dU),
                            std::min(s[k].z_U_u[i], 1e10 * mu_ / dU));
                    } else {
                        s[k].z_L_u[i] = mu_ / dL;
                        s[k].z_U_u[i] = mu_ / dU;
                    }
                }
            }
            if (prob_->n_bound_x > 0) {
                for (int i = 0; i < NX; ++i) {
                    double dL = std::max(s[k].x[i] - prob_->x_lb[i], 0.0) + bsm;
                    double dU = std::max(prob_->x_ub[i] - s[k].x[i], 0.0) + bsm;
                    if (use_primal_dual_bound()) {
                        s[k].z_L_x[i] += alpha_bound_dual * dz_L_x_[k][i];
                        s[k].z_L_x[i] = std::max(mu_ / (1e10 * dL),
                            std::min(s[k].z_L_x[i], 1e10 * mu_ / dL));
                        s[k].z_U_x[i] += alpha_bound_dual * dz_U_x_[k][i];
                        s[k].z_U_x[i] = std::max(mu_ / (1e10 * dU),
                            std::min(s[k].z_U_x[i], 1e10 * mu_ / dU));
                    } else {
                        s[k].z_L_x[i] = mu_ / dL;
                        s[k].z_U_x[i] = mu_ / dU;
                    }
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

        if (params_.verbosity < 1) return;

        // ── Dump stats line ────────────────────────────────────────
            double cur_cost = compute_objective();
            const char* phase_log = (barrier_strategy_.phase()
                == BarrierUpdateStrategy::Phase::INFEASIBILITY) ? "A" : "B";
            const char* rest_log = barrier_strategy_.restoration_active()
                ? " RESTORE" : "";
            printf("[iter %3d] mu=%.2e sig=%.3f | "
                   "prim=%.2e(dyn=%.1e cons=%.1e) stat=%.2e(abs=%.2e/sc=%.2e) compl=%.2e ineq=%.2e | "
                   "ap=%.3f ad=%.3f | phase=%s max_g+=%.3e | "
                   "cond=%.1e reg=%.1e cost=%.4e | "
                   "linKKT=%.2e(%s)%s\n",
                   iter, mu_, sigma,
                   primal_inf_, dyn_defect_, cons_viol_, stat_inf_, stat_abs_inf_, stat_scale_max_, compl_inf_, ineq_viol_,
                   alpha_p, alpha_d,
                   phase_log, max_g_pos_,
                   condH, reg_used_, cur_cost,
                   linear_kkt_res_.max_rel_res, linear_kkt_res_.quality_label(),
                   rest_log);

            // ── Slack/ds/lambda diagnostic ─────────────────────────
            if (NC > 0) {
                // Find bottleneck slack (most negative ds/s ratio)
                int bn_k = -1, bn_j = -1;
                double bn_ratio = 1e30;
                for (int kk = 0; kk <= N; ++kk)
                    for (int jj = 0; jj < NC; ++jj)
                        if (ds_[kk][jj] < -1e-16) {
                            double r = s[kk].s[jj] / (-ds_[kk][jj]);
                            if (r < bn_ratio) { bn_ratio = r; bn_k = kk; bn_j = jj; }
                        }
                if (bn_k >= 0) {
                    printf("       s_bn=[%.2e] ds_bn=[%+.2e] lam_bn=[%.2e] mu=%.2e | ",
                           s[bn_k].s[bn_j], ds_[bn_k][bn_j], s[bn_k].lambda[bn_j], mu_);
                } else {
                    printf("       s_bn=[n/a] ds_bn=[n/a] lam_bn=[n/a] mu=%.2e | ", mu_);
                }
                // Print first few slacks and lambdas
                printf("s=[");
                for (int jj = 0; jj < NC && jj < 3; ++jj)
                    printf("%.2e%s", s[0].s[jj], jj < NC-1 ? "," : "");
                printf("] lam=[");
                for (int jj = 0; jj < NC && jj < 3; ++jj)
                    printf("%.2e%s", s[0].lambda[jj], jj < NC-1 ? "," : "");
                printf("] ds=[");
                for (int jj = 0; jj < NC && jj < 3; ++jj)
                    printf("%+.2e%s", ds_[0][jj], jj < NC-1 ? "," : "");
                printf("]\n");
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
                            evaluate_constraints(s[kk].x, s[kk].u, kk, g_val);
                        else
                            evaluate_terminal_constraints(s[kk].x, g_val);
                        for (int jj = 0; jj < NC; ++jj) {
                            double gs = std::fabs(g_val[jj] + s[kk].s[jj]);
                            if (gs > max_g_s) { max_g_s = gs; worst_k_cons = kk; worst_j_cons = jj; }
                        }
                    }
                    if (worst_k_cons >= 0) {
                        Vec<NC> g_val;
                        if (worst_k_cons < N)
                            evaluate_constraints(s[worst_k_cons].x, s[worst_k_cons].u, worst_k_cons, g_val);
                        else
                            evaluate_terminal_constraints(s[worst_k_cons].x, g_val);
                        printf("  [cons] worst |g+s|=%.2e at k=%d j=%d | g=%.3e s=%.3e ds=%.3e\n",
                               max_g_s, worst_k_cons, worst_j_cons,
                               g_val[worst_j_cons], s[worst_k_cons].s[worst_j_cons],
                               ds_[worst_k_cons][worst_j_cons]);
                    }
                }

                // ── Average complementarity: current / after-step ────────
                {
                    double mu_cur = 0.0, mu_corr = 0.0;
                    int cnt = 0;
                    for (int kk = 0; kk <= N; ++kk) {
                        for (int jj = 0; jj < NC; ++jj) {
                            double sv = s[kk].s[jj];
                            double lv = s[kk].lambda[jj];
                            mu_cur += sv * lv;
                            // after-step at alpha_p, alpha_d
                            double sc = sv + alpha_p * ds_[kk][jj];
                            double lc = lv + alpha_d * dlambda_[kk][jj];
                            mu_corr += std::max(sc, 0.0) * std::max(lc, 0.0);
                            cnt++;
                        }
                    }
                    if (cnt > 0) {
                        mu_cur /= cnt; mu_corr /= cnt;
                    }
                    printf("  [mu] cur=%.3e after_step=%.3e\n",
                           mu_cur, mu_corr);
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
                    if (iter <= 2 || iter % 10 == 0 || params_.verbosity >= 3) {
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
    //  Conditional complementarity safeguard.
    //
    //  For each (k, j), if s·λ < mu_frac·μ (complementarity collapsed below
    //  the safeguard level), reset s to a numerically safe value and sync λ
    //  so that s·λ = mu_frac·μ.
    //
    //  When the Newton step maintains complementarity (s·λ ≥ threshold),
    //  both s and λ are LEFT UNTOUCHED. This preserves primal-dual
    //  consistency of the Newton iterate — the safeguard fires only when
    //  the step failed to maintain complementarity.
    //
    //  mu_frac controls the target complementarity level:
    //    mu_frac = 1.0       → full re-projection: s·λ = μ  (stall recovery)
    //    mu_frac = m_safe    → gentle safeguard: s·λ ≥ m_safe·μ
    // ═════════════════════════════════════════════════════════════════════

    void sz_complement(double mu_frac = 1.0) {
        const int N = HORIZON;
        Stage* s = prob_->stages;
        const double threshold = mu_frac * mu_;

        for (int k = 0; k <= N; ++k) {
            for (int j = 0; j < active_constraints(k); ++j) {
                double sj = s[k].s[j];
                double lj = s[k].lambda[j];
                // Trigger: complementarity collapsed below safeguard level.
                // When the Newton step maintains s·λ ≥ threshold, leave
                // (s, λ) untouched to preserve primal-dual consistency.
                if (sj * lj < threshold) {
                    // Keep primal s (Newton iterate); floor to prevent
                    // barrier Hessian singularity; sync dual λ.
                    double s_new = std::max(sj, params_.bound_s_min);
                    s[k].s[j]      = s_new;
                    s[k].lambda[j] = threshold / s_new;
                }
            }
        }
    }

    // ═════════════════════════════════════════════════════════════════════
    //  Per-iteration diagnostic computation (Sections A–G)
    //  Called once per IPM iteration after the Newton step is available.
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
                if (k < N) evaluate_constraints(s[k].x, s[k].u, k, g_val);
                else evaluate_terminal_constraints(s[k].x, g_val);
                for (int j = 0; j < active_constraints(k); ++j) {
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
            for (int j = 0; j < active_constraints(k); ++j) {
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
                for (int jj = 0; jj < active_constraints(k); ++jj) {
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
                for (int jj = 0; jj < active_constraints(k); ++jj) {
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
            for (int j = 0; j < active_constraints(k); ++j) {
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
                for (int j = 0; j < active_constraints(k); ++j) {
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
                    for (int j = 0; j < active_constraints(k); ++j)
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
                for (int j = 0; j < active_constraints(k); ++j)
                    theta_ineq += std::fabs(s[k].d[j] + s[k].s[j]);
        }
        return theta_dyn + theta_ineq;
    }



    // (fallback: sz_complement used when line search exhausts)

    // Fraction-to-boundary limits (primal ap, dual ad) — used by SOC
    void compute_ftb_limits(double& ap, double& ad) {
        ap = 1.0;
        ad = 1.0;
        double bound_ap = 1.0;
        const Stage* s = prob_->stages;

        for (int k = 0; k <= HORIZON; ++k) {
            if (prob_->constraints) {
                for (int j = 0; j < active_constraints(k); ++j) {
                    if (ds_[k][j] < -1e-16)
                        ap = std::min(ap,
                            -params_.tau * s[k].s[j] / ds_[k][j]);
                    if (dlambda_[k][j] < -1e-16)
                        ad = std::min(ad,
                            -params_.tau * s[k].lambda[j] / dlambda_[k][j]);
                }
            }

            if (prob_->n_bound_u > 0 && k < HORIZON) {
                for (int i = 0; i < NU; ++i) {
                    const double du = riccati_ws_.du[k][i];
                    const double dL = s[k].u[i] - prob_->u_lb[i];
                    const double dU = prob_->u_ub[i] - s[k].u[i];
                    if (dL > 1e-14 && du < -1e-16)
                        bound_ap = std::min(bound_ap,
                            -params_.tau * dL / du);
                    if (dU > 1e-14 && du > 1e-16)
                        bound_ap = std::min(bound_ap,
                            params_.tau * dU / du);
                }
            }
            if (prob_->n_bound_x > 0) {
                for (int i = 0; i < NX; ++i) {
                    const double dx = riccati_ws_.dx[k][i];
                    const double dL = s[k].x[i] - prob_->x_lb[i];
                    const double dU = prob_->x_ub[i] - s[k].x[i];
                    if (dL > 1e-14 && dx < -1e-16)
                        bound_ap = std::min(bound_ap,
                            -params_.tau * dL / dx);
                    if (dU > 1e-14 && dx > 1e-16)
                        bound_ap = std::min(bound_ap,
                            params_.tau * dU / dx);
                }
            }
        }
        ap = std::min(ap, bound_ap);

        // Simple bounds are exactly eliminable as z=mu/d while their Newton
        // step remains safely interior. Activate independent multipliers only
        // when the primal FTB ratio predicts meaningful bound saturation.
        if (params_.primal_dual_bounds && !bound_pd_mode_
            && bound_ap < params_.bound_pd_activation_fraction) {
            bound_pd_mode_ = true;
            if (params_.verbosity >= 2)
                printf("  [bounds: activate primal-dual, ap_bound=%.3e]\n",
                       bound_ap);
        }

        if (!use_primal_dual_bound()) return;
        for (int k = 0; k <= HORIZON; ++k) {
            if (prob_->n_bound_u > 0 && k < HORIZON) {
                for (int i = 0; i < NU; ++i) {
                    if (dz_L_u_[k][i] < -1e-16)
                        ad = std::min(ad, -params_.tau * s[k].z_L_u[i]
                                          / dz_L_u_[k][i]);
                    if (dz_U_u_[k][i] < -1e-16)
                        ad = std::min(ad, -params_.tau * s[k].z_U_u[i]
                                          / dz_U_u_[k][i]);
                }
            }
            if (prob_->n_bound_x > 0) {
                for (int i = 0; i < NX; ++i) {
                    if (dz_L_x_[k][i] < -1e-16)
                        ad = std::min(ad, -params_.tau * s[k].z_L_x[i]
                                          / dz_L_x_[k][i]);
                    if (dz_U_x_[k][i] < -1e-16)
                        ad = std::min(ad, -params_.tau * s[k].z_U_x[i]
                                          / dz_U_x_[k][i]);
                }
            }
        }
    }
    // ═════════════════════════════════════════════════════════════════════

    //  Trial-point evaluator for filter line search
    class IPMTrialEvaluator
        : public TrialPointEvaluator<NX, NU, HORIZON> {
    public:
        using Solver = PaperIPMSolver<NX, NU, NC, HORIZON>;

        void bind(Solver* s) { solver_ = s; }

        bool time_limit_reached() const override {
            return solver_ && solver_->deadline_reached();
        }

        bool evaluate(double alpha, double& out_theta, double& out_phi) override {
            if (time_limit_reached()) return false;
            if (solver_->active_stats_) ++solver_->active_stats_->line_search_evals;
            // READ-ONLY: compute trial (θ, φ) without modifying solver state.
            const auto* sv = solver_;
            const int N = HORIZON;
            const auto* s = sv->prob_->stages;

            out_theta = 0.0;
            out_phi = 0.0;

            // Dynamics defects at trial point (using temporary values)
            for (int k = 0; k < N; ++k) {
                if (time_limit_reached()) return false;
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
                    if (time_limit_reached()) return false;
                    Vec<NC> d_t;
                    if (k < N) {
                        Vec<NX> xk_t = s[k].x;  Vec<NU> uk_t = s[k].u;
                        for (int i = 0; i < NX; ++i) xk_t[i] += alpha * sv->riccati_ws_.dx[k][i];
                        for (int i = 0; i < NU; ++i) uk_t[i] += alpha * sv->riccati_ws_.du[k][i];
                        sv->evaluate_constraints(xk_t, uk_t, k, d_t);
                    } else {
                        Vec<NX> xN_t = s[k].x;
                        for (int i = 0; i < NX; ++i) xN_t[i] += alpha * sv->riccati_ws_.dx[k][i];
                        sv->evaluate_terminal_constraints(xN_t, d_t);
                    }
                    for (int j = 0; j < sv->active_constraints(k); ++j)
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
                    for (int j = 0; j < sv->active_constraints(k); ++j) {
                        double s_t = s[k].s[j] + alpha * sv->ds_[k][j];
                        if (s_t > 1e-20)
                            barrier_term += std::log(s_t);
                    }
                out_phi -= mu * barrier_term;
            }

            // Variable-bound barrier terms must be part of the same merit
            // objective whose gradient and Hessian define the Newton step.
            const double mu = sv->mu_;
            const double bsm = sv->params_.bound_s_min;
            if (sv->prob_->n_bound_u > 0) {
                for (int k = 0; k < N; ++k)
                    for (int i = 0; i < NU; ++i) {
                        double u_t = s[k].u[i] + alpha * sv->riccati_ws_.du[k][i];
                        double dL = std::max(u_t - sv->prob_->u_lb[i], 0.0) + bsm;
                        double dU = std::max(sv->prob_->u_ub[i] - u_t, 0.0) + bsm;
                        out_phi -= mu * (std::log(dL) + std::log(dU));
                    }
            }
            if (sv->prob_->n_bound_x > 0) {
                for (int k = 0; k <= N; ++k)
                    for (int i = 0; i < NX; ++i) {
                        double x_t = s[k].x[i] + alpha * sv->riccati_ws_.dx[k][i];
                        double dL = std::max(x_t - sv->prob_->x_lb[i], 0.0) + bsm;
                        double dU = std::max(sv->prob_->x_ub[i] - x_t, 0.0) + bsm;
                        out_phi -= mu * (std::log(dL) + std::log(dU));
                    }
            }

            return std::isfinite(out_theta) && std::isfinite(out_phi);
        }

        int trial_candidate_count() const override {
            return solver_->params_.enable_nonlinear_rollout ? 2 : 1;
        }

        bool evaluate_candidate(int candidate, double alpha,
                                double& out_theta, double& out_phi) override {
            if (candidate == 0) return evaluate(alpha, out_theta, out_phi);
            if (candidate == 1) return evaluate_rollout(alpha, out_theta, out_phi);
            return false;
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
                    for (int j = 0; j < solver_->active_constraints(k); ++j)
                        if (solver_->prob_->stages[k].s[j] > 1e-20)
                            phi -= mu * std::log(solver_->prob_->stages[k].s[j]);
            }
            const double mu = solver_->mu_;
            const double bsm = solver_->params_.bound_s_min;
            if (solver_->prob_->n_bound_u > 0) {
                for (int k = 0; k < HORIZON; ++k)
                    for (int i = 0; i < NU; ++i) {
                        double u = solver_->prob_->stages[k].u[i];
                        double dL = std::max(u - solver_->prob_->u_lb[i], 0.0) + bsm;
                        double dU = std::max(solver_->prob_->u_ub[i] - u, 0.0) + bsm;
                        phi -= mu * (std::log(dL) + std::log(dU));
                    }
            }
            if (solver_->prob_->n_bound_x > 0) {
                for (int k = 0; k <= HORIZON; ++k)
                    for (int i = 0; i < NX; ++i) {
                        double x = solver_->prob_->stages[k].x[i];
                        double dL = std::max(x - solver_->prob_->x_lb[i], 0.0) + bsm;
                        double dU = std::max(solver_->prob_->x_ub[i] - x, 0.0) + bsm;
                        phi -= mu * (std::log(dL) + std::log(dU));
                    }
            }
            return phi;
        }

        double compute_Dphi() const override {
            // Compute −dz^T·H·dz from the Riccati Hessian.
            // The Riccati step uses the Lagrangian Hessian (λ/s weights),
            // so −dz^T·H·dz is guaranteed ≤ 0 for convex problems.
            const auto* sv = solver_;
            const int N = HORIZON;
            const auto* s = sv->prob_->stages;
            const double mu = sv->mu_;
            const double bsm = sv->params_.bound_s_min;
            double Dphi = 0.0;
            for (int k = 0; k <= N; ++k) {
                for (int i = 0; i < NX; ++i)
                    Dphi += s[k].qx[i] * sv->riccati_ws_.dx[k][i];
                if (k < N)
                    for (int i = 0; i < NU; ++i)
                        Dphi += s[k].qu[i] * sv->riccati_ws_.du[k][i];

                if (sv->prob_->constraints)
                    for (int j = 0; j < sv->active_constraints(k); ++j)
                        Dphi -= mu * sv->ds_[k][j]
                                 / std::max(s[k].s[j], 1e-20);

                if (sv->prob_->n_bound_x > 0)
                    for (int i = 0; i < NX; ++i) {
                        const double dL =
                            std::max(s[k].x[i] - sv->prob_->x_lb[i], 0.0) + bsm;
                        const double dU =
                            std::max(sv->prob_->x_ub[i] - s[k].x[i], 0.0) + bsm;
                        Dphi += (-mu / dL + mu / dU)
                                 * sv->riccati_ws_.dx[k][i];
                    }

                if (k < N && sv->prob_->n_bound_u > 0)
                    for (int i = 0; i < NU; ++i) {
                        const double dL =
                            std::max(s[k].u[i] - sv->prob_->u_lb[i], 0.0) + bsm;
                        const double dU =
                            std::max(sv->prob_->u_ub[i] - s[k].u[i], 0.0) + bsm;
                        Dphi += (-mu / dL + mu / dU)
                                 * sv->riccati_ws_.du[k][i];
                    }
            }

            // Non-descent directions go through the filter feasibility branch;
            // do not fabricate a negative derivative for the Armijo test.
            return std::isfinite(Dphi) ? Dphi : 0.0;
        }

        bool compute_soc(double alpha, double& out_theta, double& out_phi) override {
            if (time_limit_reached()) return false;
            // SOC modifies the search direction (dx, du, ds, dlambda) to correct
            // for nonlinear dynamics defects.  If SOC fails, original direction
            // is restored so backtracking can continue with the original step.
            auto* sv = solver_;
            const int N = HORIZON;
            auto* s = sv->prob_->stages;

            // Compare the correction against the uncorrected trial, not the
            // base iterate.  At a feasible base theta0 is zero, which would
            // otherwise restore every nonzero SOC direction unconditionally.
            double theta_before, phi_before;
            if (!evaluate(alpha, theta_before, phi_before)) return false;

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
                if (time_limit_reached()) {
                    reject_soc();
                    return false;
                }
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
                    sv->evaluate_constraints(xk_t, uk_t, k, g_trial);
                    for (int j = 0; j < sv->active_constraints(k); ++j)
                        s[k].d[j] = g_trial[j];  // replace with trial constraint values
                }
            }

            // Terminal constraints at trial point
            if (sv->prob_->constraints) {
                Vec<NX> xN_t = s[N].x;
                for (int i = 0; i < NX; ++i)
                    xN_t[i] += alpha * sv->soc_save_dx_[N][i];
                Vec<NC> g_term_trial;
                sv->evaluate_terminal_constraints(xN_t, g_term_trial);
                for (int j = 0; j < sv->active_constraints(N); ++j)
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

            const bool riccati_diagnostics = sv->params_.verbosity >= 2;
            Status st = Ricc::backward_rhs(
                sv->riccati_stages_, sv->riccati_ws_, riccati_diagnostics);
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
                if (sv->use_primal_dual_bound())
                    sv->recover_bound_multiplier_steps(sv->sigma_, false);
                return false;
            }

            Vec<NX> dx0_soc;
            for (int i = 0; i < NX; ++i)
                dx0_soc[i] = sv->prob_->x0[i] - sv->prob_->stages[0].x[i];
            if (sv->params_.enable_preconditioner) {
                sv->prec_.scale_dx0(dx0_soc);
            }
            st = Ricc::forward(sv->riccati_stages_, sv->riccati_ws_,
                               dx0_soc, riccati_diagnostics);
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
                if (sv->use_primal_dual_bound())
                    sv->recover_bound_multiplier_steps(sv->sigma_, false);
                return false;
            }

            if (time_limit_reached()) {
                reject_soc();
                return false;
            }

            // Recover physical SOC step from scaled solution.
            // ds/dλ computed BEFORE primal recovery (scaled Cx·dx̂ = physical ds).
            sv->recover_inequality_steps(sv->sigma_);
            if (sv->params_.primal_dual_bounds)
                sv->recover_bound_multiplier_steps(sv->sigma_, true);
            if (sv->params_.enable_preconditioner) {
                sv->prec_.recover_primal_step(sv->riccati_ws_);
                sv->prec_.recover_dual_step(sv->riccati_ws_);
            }
            double ap_soc, ad_soc_unused;
            sv->compute_ftb_limits(ap_soc, ad_soc_unused);
            double alpha_soc = std::min(alpha, ap_soc);

            // Evaluate SOC trial point (read-only)
            if (!evaluate(alpha_soc, out_theta, out_phi)) {
                reject_soc();
                return false;
            }

            // If SOC did not improve the rejected trial, restore the original
            // direction.  If accepted, apply_primal_dual_step uses this SOC direction.
            if (out_theta >= theta_before) {
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
                if (sv->use_primal_dual_bound())
                    sv->recover_bound_multiplier_steps(sv->sigma_, false);
            }
            return true;
        }


        void reject_soc() override {
            auto* sv = solver_;
            auto* s = sv->prob_->stages;
            for (int k = 0; k <= HORIZON; ++k) {
                sv->riccati_ws_.dx[k] = sv->soc_save_dx_[k];
                sv->ds_[k] = sv->soc_save_ds_[k];
                sv->dlambda_[k] = sv->soc_save_dlam_[k];
                s[k].d = sv->soc_save_d_[k];
                if (k < HORIZON) {
                    sv->riccati_ws_.du[k] = sv->soc_save_du_[k];
                    s[k].c = sv->soc_save_c_orig_[k];
                }
            }
            if (sv->use_primal_dual_bound())
                sv->recover_bound_multiplier_steps(sv->sigma_, false);
        }
    private:
        bool evaluate_rollout(double alpha, double& out_theta, double& out_phi) {
            if (time_limit_reached()) return false;
            if (solver_->active_stats_) ++solver_->active_stats_->line_search_evals;
            auto* sv = solver_;
            const int N = HORIZON;
            const auto* s = sv->prob_->stages;


            // Defect-preserving nonlinear retraction:
            // x^r_{k+1} = f(x^r_k,u+αΔu)
            //             + (1-α)(x_{k+1}-f(x_k,u_k)).
            // It equals the current trajectory at α=0, becomes a full rollout
            // at α=1, and has the same tangent as the Newton direction at zero.
            sv->rollout_x_[0] = s[0].x;
            for (int i = 0; i < NX; ++i)
                sv->rollout_x_[0][i] += alpha * sv->riccati_ws_.dx[0][i];

            out_theta = 0.0;
            out_phi = 0.0;
            for (int k = 0; k < N; ++k) {
                if (time_limit_reached()) return false;
                sv->rollout_u_[k] = s[k].u;
                for (int i = 0; i < NU; ++i)
                    sv->rollout_u_[k][i] += alpha * sv->riccati_ws_.du[k][i];

                Vec<NX> f_base, f_rollout;
                sv->prob_->dynamics->discrete_step(
                    s[k].x, s[k].u, sv->prob_->dt, f_base, k);
                sv->prob_->dynamics->discrete_step(
                    sv->rollout_x_[k], sv->rollout_u_[k], sv->prob_->dt,
                    f_rollout, k);
                for (int i = 0; i < NX; ++i) {
                    const double old_defect = s[k+1].x[i] - f_base[i];
                    sv->rollout_x_[k+1][i] = f_rollout[i]
                        + (1.0 - alpha) * old_defect;
                    out_theta += std::fabs(
                        f_rollout[i] - sv->rollout_x_[k+1][i]);
                }
                out_phi += sv->prob_->cost->stage_cost(
                    sv->rollout_x_[k], sv->rollout_u_[k], k);
            }
            out_phi += sv->prob_->cost->terminal_cost(sv->rollout_x_[N]);

            // The nonlinear retraction is not covered by the linear FTB ratio.
            // Reject it if it leaves the strict interior of any finite bound.
            if (sv->prob_->n_bound_x > 0) {
                for (int k = 0; k <= N; ++k)
                    for (int i = 0; i < NX; ++i) {
                        const double x = sv->rollout_x_[k][i];
                        if ((sv->prob_->x_lb[i] > -1e19
                             && !(x > sv->prob_->x_lb[i]))
                            || (sv->prob_->x_ub[i] < 1e19
                                && !(x < sv->prob_->x_ub[i])))
                            return false;
                    }
            }
            if (sv->prob_->n_bound_u > 0) {
                for (int k = 0; k < N; ++k)
                    for (int i = 0; i < NU; ++i) {
                        const double u = sv->rollout_u_[k][i];
                        if ((sv->prob_->u_lb[i] > -1e19
                             && !(u > sv->prob_->u_lb[i]))
                            || (sv->prob_->u_ub[i] < 1e19
                                && !(u < sv->prob_->u_ub[i])))
                            return false;
                    }
            }

            if (sv->prob_->constraints) {
                double barrier_term = 0.0;
                for (int k = 0; k <= N; ++k) {
                    if (time_limit_reached()) return false;
                    Vec<NC> d_t;
                    if (k < N)
                        sv->evaluate_constraints(
                            sv->rollout_x_[k], sv->rollout_u_[k], k, d_t);
                    else
                        sv->evaluate_terminal_constraints(
                            sv->rollout_x_[k], d_t);

                    for (int j = 0; j < sv->active_constraints(k); ++j) {
                        const double old_residual = s[k].d[j] + s[k].s[j];
                        const double residual_preserving_slack =
                            (1.0 - alpha) * old_residual - d_t[j];
                        if (residual_preserving_slack > 0.0) {
                            sv->rollout_s_[k][j] =
                                residual_preserving_slack;
                        } else {
                            // A newly activated contact row can be infeasible
                            // while the iterate remains strictly interior.
                            // Keep the FTB-positive linear slack and expose the
                            // elastic residual to the filter instead of
                            // rejecting the dynamics-feasible retraction.
                            sv->rollout_s_[k][j] = std::max(
                                s[k].s[j] + alpha * sv->ds_[k][j],
                                sv->params_.s_min_init);
                        }
                        out_theta += std::fabs(
                            d_t[j] + sv->rollout_s_[k][j]);
                        barrier_term += std::log(sv->rollout_s_[k][j]);
                    }
                }
                out_phi -= sv->mu_ * barrier_term;
            }

            const double mu = sv->mu_;
            const double bsm = sv->params_.bound_s_min;
            if (sv->prob_->n_bound_u > 0) {
                for (int k = 0; k < N; ++k)
                    for (int i = 0; i < NU; ++i) {
                        const double u = sv->rollout_u_[k][i];
                        const double dL = std::max(
                            u - sv->prob_->u_lb[i], 0.0) + bsm;
                        const double dU = std::max(
                            sv->prob_->u_ub[i] - u, 0.0) + bsm;

                        out_phi -= mu * (std::log(dL) + std::log(dU));
                    }
            }
            if (sv->prob_->n_bound_x > 0) {
                for (int k = 0; k <= N; ++k)
                    for (int i = 0; i < NX; ++i) {
                        const double x = sv->rollout_x_[k][i];
                        const double dL = std::max(
                            x - sv->prob_->x_lb[i], 0.0) + bsm;
                        const double dU = std::max(
                            sv->prob_->x_ub[i] - x, 0.0) + bsm;

                        out_phi -= mu * (std::log(dL) + std::log(dU));
                    }
            }

            return std::isfinite(out_theta) && std::isfinite(out_phi);
        }

        Solver* solver_ = nullptr;
    };

    // ═════════════════════════════════════════════════════════════════════
    //  Data members
    // ═════════════════════════════════════════════════════════════════════

    Prob*        prob_ = nullptr;
    PaperIPMParams params_;
    double       mu_ = 1.0;
    bool         bound_pd_mode_ = false;
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

    // Persistent nonlinear-retraction candidate (avoids horizon-sized stack use).
    Vec<NX> rollout_x_[HORIZON + 1];
    Vec<NU> rollout_u_[HORIZON];
    Vec<NC> rollout_s_[HORIZON + 1];

    // Primal-dual Newton step (Δs, Δλ for inequality slacks/multipliers)
    Vec<NC> ds_[HORIZON + 1];
    Vec<NC> dlambda_[HORIZON + 1];
    Vec<NU> dz_L_u_[HORIZON], dz_U_u_[HORIZON];
    Vec<NX> dz_L_x_[HORIZON + 1], dz_U_x_[HORIZON + 1];

    // KKT residuals
    double primal_inf_      = 0.0;  // dynamics defect + constraint satisfaction
    double dyn_defect_      = 0.0;  // max |f(x,u) - x_{k+1}| (nonlinear dynamics defect)
    double cons_viol_       = 0.0;  // max |g(x,u) + s| (constraint violation)
    double max_g_pos_       = 0.0;  // max positive constraint violation: max(g+, 0)
    double stat_inf_        = 0.0;  // stationarity: ‖∇L(z,λ,ν)‖∞ / scale  (relative, NOT dual feasibility!)
    double stat_abs_inf_    = 0.0;  // stationarity numerator: max ‖∇L‖∞ across stages
    double stat_scale_max_  = 1.0;  // stationarity denominator: max scale across stages
    double stat_breakdown_[6] = {};  // [grad_x, grad_u, Cx^T·λ, Cu^T·λ, costate_x, costate_u]
    int    stat_worst_node_  = -1;  // node k where stationarity is worst
    double compl_inf_       = 0.0;  // barrier complementarity: |s_j*lambda_j - mu|
    double mpcc_inf_        = 0.0;  // physical complementarity: max |a_i*b_i|
    double ineq_viol_       = 0.0;  // inequality: most-negative s_j or λ_j (≥0 = OK)
    bool   has_costates_    = false; // Riccati costates available (false until first solve)
    bool   warm_start_ready_ = false; // a successful solve supplied reusable barrier data
    bool   stages_scaled_   = false; // true after transform_qp, false after evaluate_model

    SolverStats* active_stats_ = nullptr;
    Clock::time_point solve_start_{};
    Clock::time_point deadline_{};
    bool deadline_enabled_ = false;
    bool timed_out_ = false;
    int current_iteration_ = 0;
    Stage entry_stages_[HORIZON + 1];
    Vec<NX> entry_costates_[HORIZON + 1];
    double entry_mu_ = 0.0;
    bool entry_bound_pd_mode_ = false;
    bool entry_has_costates_ = false;
    bool entry_exact_hessian_ = false;
    bool entry_warm_start_ready_ = false;
    bool current_solve_warm_ = false;
    double entry_sigma_ = 0.0;
    double entry_alpha_lambda_ = 1.0;
    double entry_last_alpha_p_ = 1.0;
    int entry_low_ftb_count_ = 0;
    bool constraint_metadata_cached_ = false;
    bool has_complementarity_cache_ = false;
    int base_constraints_cache_[HORIZON + 1] = {};
    int complementarity_pairs_cache_[HORIZON + 1] = {};
    int active_constraints_cache_[HORIZON + 1] = {};
    int complementarity_first_cache_[HORIZON + 1][NC] = {};
    int complementarity_second_cache_[HORIZON + 1][NC] = {};

    // Linear KKT solution quality (computed each iteration)
    KKTLinearResiduals linear_kkt_res_;

    double sigma_            = 0.0;     // primal-dual centering parameter (current iteration, FSM-scheduled)
    double alpha_lambda_     = 1.0;     // coupled dual step: min(alpha_d, alpha_p)

    double last_alpha_p_     = 1.0;     // previous primal FTB step
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

    // Debug: scaled Newton step saved BEFORE recovery (for invariance testing)
    Vec<NX> debug_scaled_dx_[HORIZON + 1];
    Vec<NU> debug_scaled_du_[HORIZON];
    Vec<NX> debug_scaled_p_[HORIZON + 1];
    // Debug: physical Newton step saved AFTER recovery
    Vec<NX> debug_phys_dx_[HORIZON + 1];
    Vec<NU> debug_phys_du_[HORIZON];
    Vec<NX> debug_phys_p_[HORIZON + 1];
    // Debug: pristine Riccati stages right after KKT build (before SOC/LS)
    StageData<NX, NU, NC> debug_pristine_stages_[HORIZON + 1];
    // Debug: sigma and mu at Newton step
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
    // These accessors expose the last captured debug snapshot. When runtime
    // diagnostics are disabled they intentionally do not describe the latest
    // solve; convergence and SolverStats remain current.
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
