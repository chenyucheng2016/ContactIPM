/**
 * @file    bench_cart_pendulum_time.cpp
 * @brief   Fatrop Benchmark: CartPendulumTime (ContactIPM, free-time)
 *   NX=5(+T), NU=2(+dT), NC=6, N=100, dt=T/N (free-time)
 */
#include "fatrop_benchmark_common.hpp"
#include "cart_pendulum_time.c"

constexpr int NX = 5, NU = 2, NC = 6, N = 100;
constexpr double T_INIT = 2.0;
constexpr double DT = T_INIT / N;

int main() {
    printf("=== CartPendulumTime (ContactIPM, free-time) ===\n");

    nmpc::CodegenDynamics<NX, NU> dyn;
    dyn.set_functions(f_expl, f_jac_x, f_jac_u);
    dyn.free_time_idx = 4;  // T is x[4]

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

    // x0 = [pos=0, theta=0, vel=0, omega=0, T=T_INIT]
    nmpc::Vec<NX> x0;
    x0.zero();
    x0[4] = T_INIT;

    prob.x0 = x0;
    // Linear interpolation: theta from 0 (down) to pi (up)
    for (int k = 0; k <= N; ++k) {
        double alpha = (double)k / N;
        nmpc::Vec<NX> xk;
        xk.zero();
        xk[1] = alpha * M_PI;  // theta: interpolate down to up
        xk[4] = T_INIT;         // T stays constant
        prob.stages[k].x = xk;
        prob.stages[k].u.zero();
    }

    fatrop_bench::run_benchmark(prob, fatrop_bench::default_params(),
                                "cart_pendulum_time", 3);
    return 0;
}
