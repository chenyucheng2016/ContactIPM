/**
 * @file    diag_verbose.cpp
 * @brief   Run quadcopter benchmark with verbose output to diagnose failure.
 */
#include "fatrop_benchmark_common.hpp"
#include "quadcopter_mpc.c"

constexpr int NX = 6, NU = 4, NC = 7, N = 25;
constexpr double DT = 0.08;

int main() {
    printf("=== QuadCopterMPC Verbose Diagnostic ===\n");

    nmpc::CodegenDynamics<NX, NU> dyn;
    dyn.set_functions(f_expl, f_jac_x, f_jac_u);

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

    nmpc::CodegenConstraints<NX, NU, NC> cons;
    cons.g_path_fn = g_path;
    cons.g_path_jac_x_fn = g_path_jac_x;
    cons.g_path_jac_u_fn = g_path_jac_u;
    cons.g_term_fn = g_term;
    cons.g_term_jac_fn = g_term_jac;

    nmpc::NMPCProblem<NX, NU, NC, N> prob;
    prob.dynamics = &dyn;
    prob.cost = &cost;
    prob.constraints = &cons;
    prob.dt = DT;

    // x0 = [px=0, py=0, pz=2.5, vx=1, vy=1, vz=1]
    prob.x0[0] = 0.; prob.x0[1] = 0.; prob.x0[2] = 2.5;
    prob.x0[3] = 1.; prob.x0[4] = 1.; prob.x0[5] = 1.;
    nmpc::Vec<NU> u0;
    u0[0] = 9.81; u0[1] = M_PI/10; u0[2] = M_PI/10; u0[3] = M_PI/10;
    // Forward-simulate initial guess
    prob.stages[0].x = prob.x0;
    prob.stages[0].u = u0;
    for (int k = 0; k < N; ++k) {
        prob.stages[k].u = u0;
        dyn.discrete_step(prob.stages[k].x, prob.stages[k].u, DT, prob.stages[k+1].x);
    }
    prob.stages[N].u = u0;

    auto params = fatrop_bench::default_params();
    params.verbosity = 2;  // full diagnostics
    params.max_iters = 200;
    params.enable_refinement = false;
    params.enable_preconditioner = true;

    auto solver = std::make_unique<nmpc::NMPCSolverPaper<NX, NU, NC, N>>();
    solver->configure(params);
    auto st = solver->solve(prob);

    printf("\n=== Final trajectory (stages 0-3) ===\n");
    for (int k = 0; k <= 3; ++k) {
        auto& s = prob.stages[k];
        printf("k=%d: x=[%.4f %.4f %.4f %.4f %.4f %.4f]\n       u=[%.4f %.4f %.4f %.4f]\n       s=[", k,
               s.x[0], s.x[1], s.x[2], s.x[3], s.x[4], s.x[5],
               s.u[0], s.u[1], s.u[2], s.u[3]);
        for (int j = 0; j < NC; ++j) printf("%.2e%s", s.s[j], j<NC-1?" ":"");
        printf("]\n       lam=[");
        for (int j = 0; j < NC; ++j) printf("%.2e%s", s.lambda[j], j<NC-1?" ":"");
        printf("]\n       g=[");
        nmpc::Vec<NC> gk;
        if (k < N) cons.evaluate(s.x, s.u, k, gk); else cons.evaluate_terminal(s.x, gk);
        for (int j = 0; j < NC; ++j) printf("%.3e%s", gk[j], j<NC-1?" ":"");
        printf("]\n");
    }

    printf("\n=== Final stats ===\n");
    printf("Status: %s\n", nmpc::status_string(st));
    return 0;
}
