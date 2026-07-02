/**
 * @file    bench_hanging_chain_2d.cpp
 * @brief   Fatrop Benchmark: HangingChain2DMPC (ContactIPM) COLD START
 *   NX=26, NU=2, NC=4, N=40, dt=2.0/40
 *   |u|<=1 handled as VARIABLE BOUNDS (not path constraints)
 */
#include "fatrop_benchmark_common.hpp"
#include "hanging_chain_2d.c"

constexpr int NX = 26, NU = 2, NC = 4, N = 40;
constexpr double DT = 2.0 / 40;
constexpr int NO_MASSES = 6;
constexpr int DIM = 2;

int main() {
    printf("=== HangingChain2DMPC (ContactIPM, COLD START) ===\n");

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

    // Initial state
    nmpc::Vec<NX> x0;
    x0.zero();
    for (int i = 0; i < NO_MASSES + 1; ++i) {
        double frac = (double)(i + 1) / (NO_MASSES + 1);
        x0[i * DIM] = frac;
        x0[i * DIM + 1] = 0.0;
    }

    prob.x0 = x0;
    // Cold start: forward-simulate from x0 with u=0 (matches IPOPT initialization)
    prob.stages[0].x = x0;
    for (int k = 0; k < N; ++k) {
        prob.stages[k].u.zero();
        dyn.discrete_step(prob.stages[k].x, prob.stages[k].u, DT, prob.stages[k+1].x);
    }
    prob.stages[N].u.zero();

    auto pp = fatrop_bench::default_params();
    pp.s_min_init  = 1e-6;
    pp.delta_slack = 1e-6;
    auto result = fatrop_bench::run_benchmark(prob, pp, "hanging_chain_2d", 3);

    // Save trajectory to CSV for comparison
    if (std::string(result.status) == "Success") {
        FILE* fp = fopen("contactipm_hc_trajectory.csv", "w");
        if (fp) {
            fprintf(fp, "k");
            for (int i = 0; i < NX; i++) fprintf(fp, ",x%d", i);
            for (int i = 0; i < NU; i++) fprintf(fp, ",u%d", i);
            fprintf(fp, "\n");
            for (int k = 0; k <= N; k++) {
                fprintf(fp, "%d", k);
                for (int i = 0; i < NX; i++) fprintf(fp, ",%.15e", prob.stages[k].x.data[i]);
                if (k < N)
                    for (int i = 0; i < NU; i++) fprintf(fp, ",%.15e", prob.stages[k].u.data[i]);
                else
                    for (int i = 0; i < NU; i++) fprintf(fp, ",0.0");
                fprintf(fp, "\n");
            }
            fclose(fp);
            printf("Trajectory saved to contactipm_hc_trajectory.csv\n");
        }
    }

    return 0;
}
