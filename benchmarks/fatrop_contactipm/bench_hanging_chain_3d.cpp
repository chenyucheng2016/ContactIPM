/**
 * @file    bench_hanging_chain_3d.cpp
 * @brief   Fatrop Benchmark: HangingChain3DMPC (ContactIPM)
 *   NX=39, NU=3, NC=6, N=25, dt=0.08
 */
#include "fatrop_benchmark_common.hpp"
#include "hanging_chain_3d.c"

constexpr int NX = 39, NU = 3, NC = 6, N = 25;
constexpr double DT = 2.0 / 25;
constexpr int NO_MASSES = 6;
constexpr int DIM = 3;

int main() {
    printf("=== HangingChain3DMPC (ContactIPM) ===\n");

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

    // Initial state: linear interpolation from ground [0,0,0] to endpoint [1,0,0]
    nmpc::Vec<NX> x0;
    x0.zero();
    for (int i = 0; i < NO_MASSES + 1; ++i) {
        double frac = (double)(i + 1) / (NO_MASSES + 1);
        x0[i * DIM] = frac;       // x-component
        x0[i * DIM + 1] = 0.0;    // y-component
        x0[i * DIM + 2] = 0.0;    // z-component
    }

    prob.x0 = x0;
    for (int k = 0; k <= N; ++k) {
        prob.stages[k].x = x0;
        prob.stages[k].u.zero();
    }

    fatrop_bench::run_benchmark(prob, fatrop_bench::default_params(),
                                "hanging_chain_3d", 1);
    return 0;
}
