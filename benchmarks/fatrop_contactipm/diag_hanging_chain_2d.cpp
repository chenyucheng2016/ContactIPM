/**
 * @file    diag_hanging_chain_2d.cpp
 * @brief   Diagnostic run of hanging_chain_2d with structured IPM instrumentation.
 *   |u|<=1 handled as VARIABLE BOUNDS (not path constraints).
 *
 * Produces:
 *   - IPOPT-style compact line per iteration (to stdout)
 *   - CSV file (diag_hanging_chain_2d.csv) for plotting
 *   - Final diagnostic summary
 */
#include "codegen_driver.hpp"
#include "nmpc/nmpc_solver_paper.hpp"
#include <cstdio>
#include <memory>

#include "hanging_chain_2d.c"

constexpr int NX = 26, NU = 2, NC = 4, N = 25;
constexpr double DT = 4.0 / 25;
constexpr int NO_MASSES = 6;
constexpr int DIM = 2;

int main() {
    printf("=== HangingChain2D Diagnostic (structured IPM instrumentation) ===\n");

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
    // NO path constraints: |u|<=1 as variable bounds instead
    prob.constraints = nullptr;
    prob.dt = DT;

    // Variable bounds: |u| <= 1
    prob.init_bounds_free();
    prob.u_lb[0] = -1.0; prob.u_ub[0] = 1.0;
    prob.u_lb[1] = -1.0; prob.u_ub[1] = 1.0;
    prob.n_bound_u = 2;

    // Initial state: linear interpolation from ground [0,0] to endpoint [1,0]
    nmpc::Vec<NX> x0;
    x0.zero();
    for (int i = 0; i < NO_MASSES + 1; ++i) {
        double frac = (double)(i + 1) / (NO_MASSES + 1);
        x0[i * DIM] = frac;      // x-component
        x0[i * DIM + 1] = 0.0;   // y-component
    }

    prob.x0 = x0;
    for (int k = 0; k <= N; ++k) {
        prob.stages[k].x = x0;
        if (k < N) prob.stages[k].u.zero();
    }

    // Solver params — moderate verbosity, diagnostics come from IterDiag
    nmpc::PaperIPMParams pp;
    pp.mu_init = 0.1;
    pp.max_iters = 200;
    pp.mu_min = 5e-5;
    pp.tol_primal = 1e-3;
    pp.tol_compl = 1e-3;
    pp.tol_ineq = 1e-3;
    pp.tol_stat = 0.05;
    pp.enable_preconditioner = true;
    pp.verbosity = 1;  // compact iteration log via IterDiag

    auto solver = std::make_unique<nmpc::NMPCSolverPaper<NX, NU, NC, N>>();
    solver->configure(pp);

    // Enable CSV diagnostic output
    solver->enable_diag_csv("diag_hanging_chain_2d.csv");

    // Print IPOPT-style comparison header
    printf("\n");
    printf(" iter |    mu      |  r_dyn     |  r_ineq    |  stat      |  alpha   |  compl     | LS\n");
    printf("------+------------+------------+------------+------------+----------+------------+-----\n");

    nmpc::Status st = solver->solve(prob);

    solver->disable_diag_csv();

    // Final diagnostic summary
    const auto& d = solver->last_diagnostic();
    printf("\n=== DIAGNOSTIC COMPLETE ===\n");
    printf("Status: %s\n", nmpc::status_string(st));
    printf("\nFinal iteration diagnostics:\n");
    printf("  (A) r_dyn_inf=%.3e  r_ineq_inf=%.3e  ratio(dyn/ineq)=%.3e  n_active=%d\n",
           d.r_dyn_inf, d.r_ineq_inf, d.r_ratio, d.n_active);
    printf("  (B) stat=%.3e  grad_x=%.3e  grad_u=%.3e  dyn_dual=%.3e  ineq_dual=%.3e  dom_ratio=%.3e\n",
           d.grad_L_x, d.grad_cost_x, d.grad_cost_u, d.dyn_dual_x + d.dyn_dual_u,
           d.ineq_dual_x + d.ineq_dual_u, d.stat_dom_ratio);
    printf("  (C) mu=%.3e  compl=[%.3e,%.3e]  slack=[%.3e,%.3e]  barrier_obj=%.3e\n",
           d.mu, d.compl_min, d.compl_max, d.slack_min, d.slack_max, d.barrier_obj);
    printf("  (D) ||H||=%.3e  ||A_dyn||=%.3e  ||A_ineq||=%.3e  ||dx||=%.3e  ||du||=%.3e\n",
           d.norm_H_F, d.norm_Adyn_F, d.norm_Aineq_F, d.dx_inf, d.du_inf);
    printf("  (E) s_dyn=%.3e  s_ineq=%.3e  sign_corr=%d\n",
           d.s_dyn_inf, d.s_ineq_inf, d.sign_corr);
    printf("  (F) alpha=%.4f  theta_dyn=%.3e  theta_ineq=%.3e  ls_rej=%d\n",
           d.alpha_p, d.theta_dyn, d.theta_ineq, d.ls_rejected ? 1 : 0);
    printf("  (G) Lx=[%.3e,%.3e]  Lu=[%.3e,%.3e]  scale_ratio=%.3e\n",
           d.Lx_min, d.Lx_max, d.Lu_min, d.Lu_max, d.scale_ratio);
    printf("\nCSV written to: diag_hanging_chain_2d.csv\n");

    return 0;
}
