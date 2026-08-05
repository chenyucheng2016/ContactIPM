#pragma once
/**
 * @file    contact_ipm.hpp
 * @brief   Public ContactIPM solver interface.
 *
 * This is the public facade for the structured primal-dual interior-point
 * implementation used for contact-rich nonlinear optimal control.
 *
 * Usage:
 *   1. Implement DynamicsModel, CostModel, ConstraintModel
 *   2. Create NMPCProblem<NX, NU, NC, HORIZON>
 *   3. Configure and call solve() or solve_mpcc_with_recovery()
 *   4. Extract solution from problem.stages[]
 */

#include <algorithm>
#include <cmath>
#include <memory>
#include <string>

#include "nmpc_core.hpp"
#include "nmpc_problem.hpp"
#include "nmpc_ipm_paper.hpp"

namespace nmpc {

using ContactIPMParams = PaperIPMParams;

template <int NX, int NU, int NC, int HORIZON>
class ContactIPM {
public:
    using Problem = NMPCProblem<NX, NU, NC, HORIZON>;

    ContactIPM()
        : ipm_(std::make_unique<
              PaperIPMSolver<NX, NU, NC, HORIZON>>()) {}

    Status configure(const ContactIPMParams& params = ContactIPMParams{}) {
        configured_params_ = params;
        return ipm_->init(configured_params_);
    }

    Status solve(Problem& problem) {
        Status init_status = ipm_->init(configured_params_);
        if (init_status != Status::SUCCESS) return init_status;
        SolverStats stats;
        Status st = ipm_->solve(problem, stats);
        last_stats_ = stats;
        return st;
    }

    // A bounded, problem-independent MPCC recovery portfolio. It first tries
    // the requested formulation, then two geometric barrier-continuation
    // stages. If those fail, it restarts the original primal guess at a
    // conservative barrier with Gauss-Newton curvature before one final
    // tightening solve. All attempted iterations are reported.
    Status solve_mpcc_with_recovery(Problem& problem) {
        Vec<NX> initial_x[HORIZON + 1];
        Vec<NU> initial_u[HORIZON + 1];
        for (int k = 0; k <= HORIZON; ++k) {
            initial_x[k] = problem.stages[k].x;
            initial_u[k] = problem.stages[k].u;
        }

        int total_iterations = 0;
        Status status = solve_with(configured_params_, problem, total_iterations);
        if (status == Status::SUCCESS || !has_complementarity(problem))
            return finish_recovery(status, total_iterations);

        const double tightening_mu = std::min(
            configured_params_.mu_init,
            configured_params_.mu_conv_threshold);
        ContactIPMParams continuation = configured_params_;
        continuation.mu_init = std::sqrt(
            configured_params_.mu_init * tightening_mu);
        continuation.s_min_init = continuation.mu_init;
        continuation.max_iters = std::max(
            continuation.max_iters,
            configured_params_.mpcc_recovery_max_iters);
        status = solve_with(continuation, problem, total_iterations);
        if (status == Status::SUCCESS)
            return finish_recovery(status, total_iterations);

        ContactIPMParams exact_tightening = continuation;
        exact_tightening.mu_init = tightening_mu;
        exact_tightening.s_min_init = tightening_mu;
        status = solve_with(exact_tightening, problem, total_iterations);
        if (status == Status::SUCCESS)
            return finish_recovery(status, total_iterations);

        for (int k = 0; k <= HORIZON; ++k) {
            problem.stages[k].x = initial_x[k];
            problem.stages[k].u = initial_u[k];
        }
        ContactIPMParams restoration = configured_params_;
        restoration.mu_init = std::max(
            configured_params_.mpcc_recovery_mu,
            configured_params_.mu_conv_threshold);
        restoration.s_min_init = restoration.mu_init;
        restoration.exact_hessian = false;
        restoration.max_iters = std::max(
            restoration.max_iters,
            configured_params_.mpcc_recovery_max_iters);
        status = solve_with(restoration, problem, total_iterations);
        if (status == Status::SUCCESS)
            return finish_recovery(status, total_iterations);

        ContactIPMParams tightening = continuation;
        tightening.mu_init = tightening_mu;
        tightening.s_min_init = tightening_mu;
        tightening.exact_hessian = false;
        status = solve_with(tightening, problem, total_iterations);
        return finish_recovery(status, total_iterations);
    }

    const SolverStats& last_stats() const { return last_stats_; }

    // Expose step snapshot for regression testing
    auto get_step_snapshot() const { return ipm_->get_step_snapshot(); }

    // Expose Riccati diagnostics for invariance debugging
    auto get_riccati_diag() const { return ipm_->get_riccati_diag(); }

    // Expose scaled corrector step (before recovery) for invariance testing
    auto debug_scaled_dx() const { return ipm_->debug_scaled_dx(); }
    auto debug_scaled_du() const { return ipm_->debug_scaled_du(); }
    auto debug_scaled_p()  const { return ipm_->debug_scaled_p(); }
    // Expose physical corrector step (after recovery)
    auto debug_phys_dx() const { return ipm_->debug_phys_dx(); }
    auto debug_phys_du() const { return ipm_->debug_phys_du(); }
    auto debug_phys_p()  const { return ipm_->debug_phys_p(); }
    auto debug_riccati_stages() const { return ipm_->debug_riccati_stages(); }
    auto debug_pristine_stages() const { return ipm_->debug_pristine_stages(); }
    double debug_sigma() const { return ipm_->debug_sigma(); }
    double debug_mu() const { return ipm_->debug_mu(); }
    double debug_primal_inf() const { return ipm_->debug_primal_inf(); }
    double debug_compl_inf() const { return ipm_->debug_compl_inf(); }
    const auto& debug_P_term() const { return ipm_->debug_P_term(); }
    const auto& debug_S_fact0() const { return ipm_->debug_S_fact0(); }
    const auto& debug_d0() const { return ipm_->debug_d0(); }
    const auto& debug_K0() const { return ipm_->debug_K0(); }
    auto debug_prec_Lx() const { return ipm_->debug_prec_Lx(); }
    auto debug_prec_Lu() const { return ipm_->debug_prec_Lu(); }
    auto debug_prec_inv_Lx() const { return ipm_->debug_prec_inv_Lx(); }
    auto debug_prec_inv_Lu() const { return ipm_->debug_prec_inv_Lu(); }
    bool debug_precond_enabled() const { return ipm_->debug_precond_enabled(); }

    // ── Diagnostic instrumentation pass-through ──
    void enable_diag_csv(const char* filename) {
        diag_csv_path_ = filename == nullptr ? "" : filename;
        if (!diag_csv_path_.empty())
            ipm_->enable_diag_csv(diag_csv_path_.c_str());
    }
    void disable_diag_csv() {
        diag_csv_path_.clear();
        ipm_->disable_diag_csv();
    }
    const IterDiag& last_diagnostic() const { return ipm_->last_diagnostic(); }

    void get_first_control(const Problem& prob, Vec<NU>& u0) const {
        u0 = prob.stages[0].u;
    }

    void shift_for_warmstart(Problem& prob, const Vec<NX>& x1_actual) {
        for (int k = 0; k < HORIZON; ++k) {
            prob.stages[k].x = prob.stages[k + 1].x;
            prob.stages[k].u = prob.stages[k + 1].u;
        }
        prob.stages[HORIZON].x = prob.stages[HORIZON - 1].x;
        prob.stages[0].x = x1_actual;
    }

private:
    bool has_complementarity(const Problem& problem) const {
        if (problem.constraints == nullptr) return false;
        for (int k = 0; k <= HORIZON; ++k) {
            if (problem.constraints->num_complementarity_pairs(k) > 0)
                return true;
        }
        return false;
    }

    Status solve_with(const ContactIPMParams& params, Problem& problem,
                      int& total_iterations) {
        ipm_ = std::make_unique<
            PaperIPMSolver<NX, NU, NC, HORIZON>>();
        if (!diag_csv_path_.empty())
            ipm_->enable_diag_csv(diag_csv_path_.c_str());
        Status status = ipm_->init(params);
        if (status != Status::SUCCESS) return status;
        SolverStats stats;
        status = ipm_->solve(problem, stats);
        total_iterations += stats.inner_iterations;
        last_stats_ = stats;
        return status;
    }

    Status finish_recovery(Status status, int total_iterations) {
        last_stats_.inner_iterations = total_iterations;
        return status;
    }

    std::unique_ptr<PaperIPMSolver<NX, NU, NC, HORIZON>> ipm_;
    SolverStats last_stats_;
    ContactIPMParams configured_params_;
    std::string diag_csv_path_;
};

} // namespace nmpc
