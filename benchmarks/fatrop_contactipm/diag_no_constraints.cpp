/**
 * @file    diag_no_constraints.cpp
 * @brief   Test: solve quadcopter_mpc with ALL constraints deactivated.
 *          If the solver converges → barrier Hessian was the bottleneck.
 *          If still fails → cost/dynamics conditioning is the bottleneck.
 */
#include "fatrop_benchmark_common.hpp"
#include "quadcopter_mpc.c"

constexpr int NX = 6, NU = 4, NC = 7, N = 25;
constexpr double DT = 0.08;

// All constraints return -1e10 (deeply inactive → negligible barrier Hessian)
struct InactiveConstraints : nmpc::ConstraintModel<NX, NU, NC> {
    nmpc::Status evaluate(const nmpc::Vec<NX>&, const nmpc::Vec<NU>&, int,
                          nmpc::Vec<NC>& g) override {
        for (int j = 0; j < NC; ++j) g[j] = -1e10;
        return nmpc::Status::SUCCESS;
    }
    nmpc::Status evaluate_terminal(const nmpc::Vec<NX>&, nmpc::Vec<NC>& g) override {
        for (int j = 0; j < NC; ++j) g[j] = -1e10;
        return nmpc::Status::SUCCESS;
    }
    nmpc::Status jacobian(const nmpc::Vec<NX>&, const nmpc::Vec<NU>&, int,
                          nmpc::Mat<NC, NX>& Cx, nmpc::Mat<NC, NU>& Cu) override {
        Cx.zero(); Cu.zero();
        return nmpc::Status::SUCCESS;
    }
    nmpc::Status jacobian_terminal(const nmpc::Vec<NX>&,
                                   nmpc::Mat<NC, NX>& Cx) override {
        Cx.zero();
        return nmpc::Status::SUCCESS;
    }
};

int main() {
    printf("=== QuadCopterMPC: NO CONSTRAINTS (barrier-free) ===\n");

    nmpc::CodegenDynamics<NX, NU> dyn;
    dyn.set_functions(f_expl, f_jac_x, f_jac_u);

    // Use ORIGINAL cost (with the 1e7 condition number)
    nmpc::CodegenCost<NX, NU> cost;
    cost.l_stage_fn = l_stage;
    cost.l_stage_grad_x_fn = l_stage_grad_x;
    cost.l_stage_grad_u_fn = l_stage_grad_u;
    cost.l_stage_hess_xx_fn = l_stage_hess_xx;
    cost.l_stage_hess_uu_fn = l_stage_hess_uu;
    cost.l_stage_hess_ux_fn = l_stage_hess_ux;
    cost.l_term_fn = l_term;
    cost.l_term_grad_fn = l_term_grad;
    cost.l_term_hess_fn = l_term_hess;

    InactiveConstraints cons;

    nmpc::NMPCProblem<NX, NU, NC, N> prob;
    prob.dynamics = &dyn;
    prob.cost = &cost;
    prob.constraints = &cons;
    prob.dt = DT;

    prob.x0[0] = 0.; prob.x0[1] = 0.; prob.x0[2] = 2.5;
    prob.x0[3] = 1.; prob.x0[4] = 1.; prob.x0[5] = 1.;
    nmpc::Vec<NU> u0;
    u0[0] = 9.81; u0[1] = M_PI/10; u0[2] = M_PI/10; u0[3] = M_PI/10;
    prob.stages[0].x = prob.x0;
    prob.stages[0].u = u0;
    for (int k = 0; k < N; ++k) {
        prob.stages[k].u = u0;
        dyn.discrete_step(prob.stages[k].x, prob.stages[k].u, DT, prob.stages[k+1].x);
    }
    prob.stages[N].u = u0;

    auto params = fatrop_bench::default_params();
    params.verbosity = 2;
    params.max_iters = 200;

    auto solver = std::make_unique<nmpc::NMPCSolverPaper<NX, NU, NC, N>>();
    solver->configure(params);
    auto st = solver->solve(prob);

    printf("\n=== RESULT ===\n");
    printf("Status: %s\n", nmpc::status_string(st));
    auto& s = solver->last_stats();
    printf("Iterations: %d\n", s.inner_iterations);
    printf("Primal inf: %.3e\n", s.primal_infeas);
    printf("Stationarity: %.3e\n", s.dual_infeas);
    printf("Complementarity: %.3e\n", s.complementarity);
    printf("Cost: %.6f\n", s.cost);
    return 0;
}
