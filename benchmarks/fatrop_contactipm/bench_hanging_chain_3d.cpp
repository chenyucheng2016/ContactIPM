/**
 * @file    bench_hanging_chain_3d.cpp
 * @brief   Fatrop Benchmark: HangingChain3DMPC (ContactIPM) COLD START
 *   NX=39, NU=3, NC=6, N=40, dt=2.0/40
 *   |u|<=1 handled as VARIABLE BOUNDS (not path constraints)
 */
#include "fatrop_benchmark_common.hpp"
#include "hanging_chain_3d.c"

constexpr int NX = 39, NU = 3, NC = 6, N = 40;
constexpr double DT = 2.0 / 40;
constexpr int NO_MASSES = 6;
constexpr int DIM = 3;

int main() {
    printf("=== HangingChain3DMPC (ContactIPM, COLD START) ===\n");

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
    prob.u_lb[2] = -1.0; prob.u_ub[2] = 1.0;
    prob.n_bound_u = 3;

    // Initial state
    nmpc::Vec<NX> x0;
    x0.zero();
    for (int i = 0; i < NO_MASSES + 1; ++i) {
        double frac = (double)(i + 1) / (NO_MASSES + 1);
        x0[i * DIM] = frac;
        x0[i * DIM + 1] = 0.0;
        x0[i * DIM + 2] = 0.0;
    }

    prob.x0 = x0;
    // Cold start: forward-simulate from x0 with u=0
    prob.stages[0].x = x0;
    for (int k = 0; k < N; ++k) {
        prob.stages[k].u.zero();
        dyn.discrete_step(prob.stages[k].x, prob.stages[k].u, DT, prob.stages[k+1].x);
    }
    prob.stages[N].u.zero();

    fatrop_bench::run_benchmark(prob, fatrop_bench::default_params(),
                                "hanging_chain_3d", 1);
    return 0;
}
