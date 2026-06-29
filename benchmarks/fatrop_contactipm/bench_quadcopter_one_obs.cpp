/**
 * @file    bench_quadcopter_one_obs.cpp
 * @brief   Fatrop Benchmark: QuadCopterOneObs (ContactIPM, free-time + obstacle)
 *   NX=7(+T), NU=5(+slack), NC=8, N=50
 */
#include "fatrop_benchmark_common.hpp"
#include "quadcopter_one_obs.c"

constexpr int NX = 7, NU = 5, NC = 8, N = 50;
constexpr double T_INIT = 5.0;
constexpr double DT = T_INIT / N;

int main() {
    printf("=== QuadCopterOneObs (ContactIPM, free-time) ===\n");

    nmpc::CodegenDynamics<NX, NU> dyn;
    dyn.set_functions(f_expl, f_jac_x, f_jac_u);
    dyn.free_time_idx = 6;

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

    // x0 = [0,0,7.5, 0,0,0, T=5]
    nmpc::Vec<NX> x0;
    x0.zero();
    x0[2] = 7.5;
    x0[6] = T_INIT;

    nmpc::Vec<NU> u0;
    u0.zero();
    u0[0] = 9.81;  // hover thrust

    prob.x0 = x0;
    // Linear interpolation from start [0,0,7.5] to target [10,10,7.5]
    // Add bell-shaped velocity to break hover-equilibrium degeneracy
    double pf[3] = {10.0, 10.0, 7.5};
    double v_init[3] = {0.5, 0.5, 0.0};
    for (int k = 0; k <= N; ++k) {
        double alpha = (double)k / N;
        nmpc::Vec<NX> xk;
        xk[0] = alpha * pf[0];
        xk[1] = alpha * pf[1];
        xk[2] = x0[2];
        double bell = 4.0 * alpha * (1.0 - alpha);
        xk[3] = bell * v_init[0];
        xk[4] = bell * v_init[1];
        xk[5] = bell * v_init[2];
        xk[6] = T_INIT;
        prob.stages[k].x = xk;
        prob.stages[k].u = u0;
    }

    fatrop_bench::run_benchmark(prob, fatrop_bench::default_params(),
                                "quadcopter_one_obs", 3);
    return 0;
}
