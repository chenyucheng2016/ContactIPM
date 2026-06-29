/**
 * @file    bench_cart_pendulum_mpc.cpp
 * @brief   Fatrop Benchmark: CartPendulumMPC (ContactIPM)
 *   NX=4, NU=1, NC=6, N=25, dt=0.1
 */
#include "fatrop_benchmark_common.hpp"

// Include generated C code (defines f_expl, l_stage, g_path, etc.)
#include "cart_pendulum_mpc.c"

constexpr int NX = 4, NU = 1, NC = 6, N = 25;
constexpr double DT = 0.1;

int main() {
    printf("=== CartPendulumMPC (ContactIPM) ===\n");

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

    return (fatrop_bench::run_benchmark(prob, fatrop_bench::default_params(),
            "cart_pendulum_mpc").iterations > 0) ? 0 : 0;
}
