#pragma once
/**
 * @file    nmpc_solver_paper.hpp
 * @brief   Top-level NMPC solver using the paper-aligned IPM.
 *
 * This is a thin wrapper around PaperIPMSolver that provides the same
 * interface as NMPCSolver (in nmpc_solver.hpp), using the paper-aligned
 * Mehrotra predictor-corrector IPM for convex nonlinear MPC.
 *
 * Usage (identical to NMPCSolver):
 *   1. Implement DynamicsModel, CostModel, ConstraintModel
 *   2. Create NMPCProblem<NX, NU, NC, HORIZON>
 *   3. Call solve() → Status + SolverStats
 *   4. Extract solution from problem.stages[]
 */

#include "nmpc_core.hpp"
#include "nmpc_problem.hpp"
#include "nmpc_ipm_paper.hpp"

namespace nmpc {

template <int NX, int NU, int NC, int HORIZON>
class NMPCSolverPaper {
public:
    using Problem = NMPCProblem<NX, NU, NC, HORIZON>;

    NMPCSolverPaper() = default;

    Status configure(const PaperIPMParams& params = PaperIPMParams{}) {
        return ipm_.init(params);
    }

    Status solve(Problem& problem) {
        SolverStats stats;
        Status st = ipm_.solve(problem, stats);
        last_stats_ = stats;
        return st;
    }

    const SolverStats& last_stats() const { return last_stats_; }

    // Expose step snapshot for regression testing
    auto get_step_snapshot() const { return ipm_.get_step_snapshot(); }

    // Expose Riccati diagnostics for invariance debugging
    auto get_riccati_diag() const { return ipm_.get_riccati_diag(); }

    // Expose scaled corrector step (before recovery) for invariance testing
    auto debug_scaled_dx() const { return ipm_.debug_scaled_dx(); }
    auto debug_scaled_du() const { return ipm_.debug_scaled_du(); }
    auto debug_scaled_p()  const { return ipm_.debug_scaled_p(); }
    // Expose physical corrector step (after recovery)
    auto debug_phys_dx() const { return ipm_.debug_phys_dx(); }
    auto debug_phys_du() const { return ipm_.debug_phys_du(); }
    auto debug_phys_p()  const { return ipm_.debug_phys_p(); }
    auto debug_riccati_stages() const { return ipm_.debug_riccati_stages(); }
    auto debug_pristine_stages() const { return ipm_.debug_pristine_stages(); }
    double debug_sigma() const { return ipm_.debug_sigma(); }
    double debug_mu() const { return ipm_.debug_mu(); }
    double debug_primal_inf() const { return ipm_.debug_primal_inf(); }
    double debug_compl_inf() const { return ipm_.debug_compl_inf(); }
    const auto& debug_P_term() const { return ipm_.debug_P_term(); }
    const auto& debug_S_fact0() const { return ipm_.debug_S_fact0(); }
    const auto& debug_d0() const { return ipm_.debug_d0(); }
    const auto& debug_K0() const { return ipm_.debug_K0(); }
    auto debug_prec_Lx() const { return ipm_.debug_prec_Lx(); }
    auto debug_prec_Lu() const { return ipm_.debug_prec_Lu(); }
    auto debug_prec_inv_Lx() const { return ipm_.debug_prec_inv_Lx(); }
    auto debug_prec_inv_Lu() const { return ipm_.debug_prec_inv_Lu(); }
    bool debug_precond_enabled() const { return ipm_.debug_precond_enabled(); }

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
    PaperIPMSolver<NX, NU, NC, HORIZON> ipm_;
    SolverStats last_stats_;
};

} // namespace nmpc
