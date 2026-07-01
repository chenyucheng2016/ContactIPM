/**
 * @file    bench_cart_pendulum_mpc.cpp
 * @brief   Fatrop Benchmark: CartPendulumMPC (ContactIPM)
 *   NX=4, NU=1, NC=0, N=25, dt=0.1
 *   Bounds: |u|<=5, |x[0]|<=1, |x[2]|<=2 as VARIABLE BOUNDS
 */
#include "fatrop_benchmark_common.hpp"

// Include generated C code (defines f_expl, l_stage, etc.)
#include "cart_pendulum_mpc.c"

constexpr int NX = 4, NU = 1, NC = 1, N = 25;  // NC=1 placeholder (unused)
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

    nmpc::NMPCProblem<NX, NU, NC, N> prob;
    prob.dynamics = &dyn;
    prob.cost = &cost;
    // NO path constraints: bounds handled as variable bounds
    prob.constraints = nullptr;
    prob.dt = DT;

    // Variable bounds: |u|<=5, |x[0]|<=1, |x[2]|<=2
    prob.init_bounds_free();
    prob.u_lb[0] = -5.0; prob.u_ub[0] = 5.0;
    prob.n_bound_u = 1;
    prob.x_lb[0] = -1.0; prob.x_ub[0] = 1.0;   // position
    prob.x_lb[2] = -2.0; prob.x_ub[2] = 2.0;   // velocity
    prob.n_bound_x = 2;

    // x0 = [pos=0, theta=pi, vel=0, omega=1]
    prob.x0[0] = 0.0;
    prob.x0[1] = M_PI;
    prob.x0[2] = 0.0;
    prob.x0[3] = 1.0;

    // Forward simulate with u=0 (same as IPOPT initialization)
    prob.stages[0].x = prob.x0;
    for (int k = 0; k < N; ++k) {
        prob.stages[k].u.zero();
        // RK4 integration
        double xk[NX], uk[NU] = {0.0};
        for (int i = 0; i < NX; i++) xk[i] = prob.stages[k].x.data[i];
        double k1[NX], k2[NX], k3[NX], k4[NX], xtmp[NX];
        const double* arg[2]; double* res[1];
        arg[0] = xk; arg[1] = uk; res[0] = k1;
        f_expl(arg, res, nullptr, nullptr, 0);
        for (int i = 0; i < NX; i++) xtmp[i] = xk[i] + 0.5*DT*k1[i];
        arg[0] = xtmp; res[0] = k2;
        f_expl(arg, res, nullptr, nullptr, 0);
        for (int i = 0; i < NX; i++) xtmp[i] = xk[i] + 0.5*DT*k2[i];
        arg[0] = xtmp; res[0] = k3;
        f_expl(arg, res, nullptr, nullptr, 0);
        for (int i = 0; i < NX; i++) xtmp[i] = xk[i] + DT*k3[i];
        arg[0] = xtmp; res[0] = k4;
        f_expl(arg, res, nullptr, nullptr, 0);
        for (int i = 0; i < NX; i++)
            prob.stages[k+1].x.data[i] = xk[i] + (DT/6.0)*(k1[i] + 2.0*k2[i] + 2.0*k3[i] + k4[i]);
    }

    auto result = fatrop_bench::run_benchmark(prob, fatrop_bench::default_params(),
            "cart_pendulum_mpc");

    // Save trajectory to CSV for comparison (regardless of status)
    {
        FILE* fp = fopen("contactipm_cp_trajectory.csv", "w");
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
            printf("Trajectory saved to contactipm_cp_trajectory.csv\n");
        }
    }

    return 0;
}
