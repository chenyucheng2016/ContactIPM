/**
 * @file    bench_quadcopter_mpc.cpp
 * @brief   Fatrop Benchmark: QuadCopterMPC (ContactIPM)
 *   NX=6, NU=4, NC=7, N=25, dt=0.08
 */
#include "fatrop_benchmark_common.hpp"
#include "quadcopter_mpc.c"

constexpr int NX = 6, NU = 4, NC = 7, N = 25;
constexpr double DT = 0.08;

int main() {
    printf("=== QuadCopterMPC (ContactIPM) ===\n");

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
    // Forward-simulate initial guess for dynamic feasibility
    prob.stages[0].x = prob.x0;
    prob.stages[0].u = u0;
    for (int k = 0; k < N; ++k) {
        prob.stages[k].u = u0;
        dyn.discrete_step(prob.stages[k].x, prob.stages[k].u, DT, prob.stages[k+1].x);
    }
    prob.stages[N].u = u0;

    fatrop_bench::run_benchmark(prob, fatrop_bench::default_params(),
                                "quadcopter_mpc");

    // ── Debug: single verbose solve ───────
    {
        nmpc::NMPCSolverPaper<NX, NU, NC, N> dbg_solver;
        nmpc::PaperIPMParams pp = fatrop_bench::default_params();
        pp.max_iters = 2000;
        pp.verbosity = 2;
        dbg_solver.configure(pp);
        prob.x0[0] = 0.; prob.x0[1] = 0.; prob.x0[2] = 2.5;
        prob.x0[3] = 1.; prob.x0[4] = 1.; prob.x0[5] = 1.;
        prob.stages[0].x = prob.x0;
        prob.stages[0].u = u0;
        for (int k = 0; k < N; ++k) {
            prob.stages[k].u = u0;
            dyn.discrete_step(prob.stages[k].x, prob.stages[k].u, DT, prob.stages[k+1].x);
        }
        prob.stages[N].u = u0;
        printf("\n=== DEBUG SOLVE ===\n");
        nmpc::Status dbg_st = dbg_solver.solve(prob);
        printf("Status: %s\n", nmpc::status_string(dbg_st));
        printf("Iterations: %d\n", dbg_solver.last_stats().inner_iterations);
    }

    return 0;
}
