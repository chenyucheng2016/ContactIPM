/**
 * @file    bench_truck_trailer_time.cpp
 * @brief   Fatrop Benchmark: TruckTrailerTime (ContactIPM, free-time)
 *   NX=6(+T), NU=3(+dT), NC=10, N=50
 */
#include "fatrop_benchmark_common.hpp"
#include "truck_trailer_time.c"

constexpr int NX = 6, NU = 3, NC = 10, N = 50;
constexpr double T_INIT = 20.0;
constexpr double DT = T_INIT / N;

int main() {
    printf("=== TruckTrailerTime (ContactIPM, free-time) ===\n");

    nmpc::CodegenDynamics<NX, NU> dyn;
    dyn.set_functions(f_expl, f_jac_x, f_jac_u);
    dyn.free_time_idx = 5;  // T is x[5]

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

    // x0 = [theta2=0, x2=0, y2=0, theta1=0, theta0=0.1, T=20]
    nmpc::Vec<NX> x0;
    x0.zero();
    x0[4] = 0.1;   // theta0 initial
    x0[5] = T_INIT;

    nmpc::Vec<NU> u0;
    u0.zero();
    u0[1] = -0.2;  // initial velocity

    prob.x0 = x0;
    // Interpolate from start to target: x2=0->0, y2=0->-2, theta2=0->pi/2
    double tf_y = -2.0, tf_angle = M_PI / 2.0;
    for (int k = 0; k <= N; ++k) {
        double alpha = (double)k / N;
        nmpc::Vec<NX> xk;
        xk.zero();
        xk[0] = alpha * tf_angle;              // theta2: 0 -> pi/2
        xk[1] = 0.0;                             // x2: stays at 0
        xk[2] = alpha * tf_y;                    // y2: 0 -> -2
        xk[3] = alpha * tf_angle * 0.5;         // theta1: partial turn
        xk[4] = (1.0 - alpha) * 0.1 + alpha * tf_angle * 0.3;  // theta0: interpolate
        xk[5] = T_INIT;
        prob.stages[k].x = xk;
        prob.stages[k].u = u0;
    }

    fatrop_bench::run_benchmark(prob, fatrop_bench::default_params(),
                                "truck_trailer_time", 3);
    return 0;
}
