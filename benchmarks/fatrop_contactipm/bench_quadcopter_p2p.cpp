/**
 * @file    bench_quadcopter_p2p.cpp
 * @brief   Fatrop Benchmark: QuadCopterP2P (ContactIPM, free-time)
 *   NX=7(+T), NU=5(+dT), NC=7, N=25
 */
#include "fatrop_benchmark_common.hpp"
#include "quadcopter_p2p.c"

constexpr int NX = 7, NU = 5, NC = 7, N = 25;
constexpr double T_INIT = 3.0;
constexpr double DT = T_INIT / N;

int main() {
    printf("=== QuadCopterP2P (ContactIPM, free-time) ===\n"); fflush(stdout);

    nmpc::CodegenDynamics<NX, NU> dyn;
    dyn.set_functions(f_expl, f_jac_x, f_jac_u);
    dyn.free_time_idx = 6;  // T is x[6]

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

    // x0 = [px=0, py=0, pz=2.5, vx=0, vy=0, vz=0, T=3.0]
    nmpc::Vec<NX> x0;
    x0.zero();
    x0[2] = 2.5;
    x0[6] = T_INIT;

    nmpc::Vec<NU> u0;
    u0.zero();
    u0[0] = 9.81;  // thrust = hover

    prob.x0 = x0;
    // Linear interpolation from start to target position
    // Add small initial velocity to break hover-equilibrium degeneracy
    double pf[3] = {0.01, 5.0, 2.5};  // target position
    double v_init[3] = {0.01, 0.5, 0.0};  // small velocity toward target
    for (int k = 0; k <= N; ++k) {
        double alpha = (double)k / N;
        nmpc::Vec<NX> xk;
        xk.zero();
        xk[0] = alpha * pf[0];      // px: 0 -> 0.01
        xk[1] = alpha * pf[1];      // py: 0 -> 5.0
        xk[2] = x0[2];               // pz: stays at 2.5
        // Bell-shaped velocity profile (zero at endpoints, max in middle)
        double bell = 4.0 * alpha * (1.0 - alpha);
        xk[3] = bell * v_init[0];   // vx
        xk[4] = bell * v_init[1];   // vy
        xk[5] = bell * v_init[2];   // vz
        xk[6] = T_INIT;              // T stays constant
        prob.stages[k].x = xk;
        prob.stages[k].u = u0;
    }

    fatrop_bench::run_benchmark(prob, fatrop_bench::default_params(),
                                "quadcopter_p2p", 3);
    return 0;
}
