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
