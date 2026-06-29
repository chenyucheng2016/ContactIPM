/**
 * @file    diag_cart_pendulum_mpc.cpp
 * @brief   Diagnostic run of cart_pendulum_mpc with verbose output.
 */
#include "codegen_driver.hpp"
#include "nmpc/nmpc_solver_paper.hpp"
#include <cstdio>
#include <memory>

#ifndef M_PI
#define M_PI 3.14159265358979323846
#endif

// Include generated C code
#include "cart_pendulum_mpc.c"

constexpr int NX = 4, NU = 1, NC = 6, N = 25;
constexpr double DT = 0.1;

int main() {
    printf("=== CartPendulumMPC Diagnostic (verbose) ===\n");

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

    // x0 = [pos=0, theta=pi, vel=0, omega=1]
    prob.x0[0] = 0.0;
    prob.x0[1] = M_PI;
    prob.x0[2] = 0.0;
    prob.x0[3] = 1.0;
    for (int k = 0; k <= N; ++k) {
        prob.stages[k].x = prob.x0;
        prob.stages[k].u.zero();
    }

    // Verbose solver params
    nmpc::PaperIPMParams pp;
    pp.mu_init = 0.2;
    pp.max_iters = 100;
    pp.mu_min = 5e-5;
    pp.tol_primal = 3e-4;
    pp.tol_compl = 1e-4;
    pp.tol_ineq = 1e-4;
    pp.tol_stat = 0.02;
    pp.enable_preconditioner = true;
    pp.verbosity = 2;  // VERBOSE

    auto solver = std::make_unique<nmpc::NMPCSolverPaper<NX, NU, NC, N>>();
    solver->configure(pp);
    nmpc::Status st = solver->solve(prob);

    printf("\n=== DIAGNOSTIC COMPLETE ===\n");
    printf("Status: %s\n", nmpc::status_string(st));

    return 0;
}
