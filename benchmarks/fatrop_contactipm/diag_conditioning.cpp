/**
 * @file    diag_conditioning.cpp
 * @brief   Test: does the solver converge with better-conditioned cost?
 *          Replace 1000*psi_dot^2 with 1*psi_dot^2 to fix conditioning.
 */
#include "fatrop_benchmark_common.hpp"
#include "quadcopter_mpc.c"

constexpr int NX = 6, NU = 4, NC = 7, N = 25;
constexpr double DT = 0.08;

// Wrapped cost that aggressively improves Hessian conditioning
// Original Huu diag: [0.0002, 20.0002, 20.0002, 2020.0002] → cond = 1e7
// Fixed Huu diag:    [1.0,    20.0,     20.0,     20.0]       → cond = 20
// Fix: add (1.0 - 0.0002) to Huu[0,0], subtract 2000 from Huu[3,3]
struct WellConditionedCost : nmpc::CostModel<NX, NU> {
    double stage_cost(const nmpc::Vec<NX>& x, const nmpc::Vec<NU>& u, int) override {
        double v[1]; nmpc::casadi_call(l_stage, x.data, u.data, v, 1);
        // Subtract 999*u[3]^2 from psi_dot penalty
        // Add 0.5*(1-0.0002)*u[0]^2 to thrust penalty
        return v[0] - 999.0 * u[3] * u[3] + 0.5 * (1.0 - 0.0002) * u[0] * u[0];
    }
    double terminal_cost(const nmpc::Vec<NX>& x) override {
        double v[1]; nmpc::casadi_call_1in(l_term, x.data, v, 1);
        return v[0];
    }
    nmpc::Status stage_gradient(const nmpc::Vec<NX>& x, const nmpc::Vec<NU>& u, int,
                                nmpc::Vec<NX>& qx, nmpc::Vec<NU>& qu) override {
        nmpc::casadi_call(l_stage_grad_x, x.data, u.data, qx.data, NX);
        nmpc::casadi_call(l_stage_grad_u, x.data, u.data, qu.data, NU);
        qu[3] -= 2.0 * 999.0 * u[3];
        qu[0] += (1.0 - 0.0002) * u[0];
        return nmpc::Status::SUCCESS;
    }
    nmpc::Status stage_hessian(const nmpc::Vec<NX>& x, const nmpc::Vec<NU>& u, int,
                               nmpc::Mat<NX,NX>& Qxx, nmpc::Mat<NU,NU>& Quu,
                               nmpc::Mat<NU,NX>& Qux) override {
        nmpc::casadi_call(l_stage_hess_xx, x.data, u.data, Qxx.data, NX*NX);
        nmpc::casadi_call(l_stage_hess_uu, x.data, u.data, Quu.data, NU*NU);
        nmpc::casadi_call(l_stage_hess_ux, x.data, u.data, Qux.data, NU*NX);
        Quu(3,3) -= 2.0 * 999.0;   // psi_dot: 2020→2
        Quu(0,0) += (1.0 - 0.0002); // thrust: 0.0002→1.0
        return nmpc::Status::SUCCESS;
    }
    nmpc::Status terminal_gradient(const nmpc::Vec<NX>& x, nmpc::Vec<NX>& qx) override {
        nmpc::casadi_call_1in(l_term_grad, x.data, qx.data, NX);
        return nmpc::Status::SUCCESS;
    }
    nmpc::Status terminal_hessian(const nmpc::Vec<NX>& x, nmpc::Mat<NX,NX>& Qxx) override {
        nmpc::casadi_call_1in(l_term_hess, x.data, Qxx.data, NX*NX);
        return nmpc::Status::SUCCESS;
    }
};

int main() {
    printf("=== QuadCopterMPC with Well-Conditioned Cost ===\n");

    nmpc::CodegenDynamics<NX, NU> dyn;
    dyn.set_functions(f_expl, f_jac_x, f_jac_u);

    WellConditionedCost cost;

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

    prob.x0[0] = 0.; prob.x0[1] = 0.; prob.x0[2] = 2.5;
    prob.x0[3] = 1.; prob.x0[4] = 1.; prob.x0[5] = 1.;
    nmpc::Vec<NU> u0;
    u0[0] = 9.81; u0[1] = M_PI/10; u0[2] = M_PI/10; u0[3] = M_PI/10;
    // Forward-simulate
    prob.stages[0].x = prob.x0;
    prob.stages[0].u = u0;
    for (int k = 0; k < N; ++k) {
        prob.stages[k].u = u0;
        dyn.discrete_step(prob.stages[k].x, prob.stages[k].u, DT, prob.stages[k+1].x);
    }
    prob.stages[N].u = u0;

    auto params = fatrop_bench::default_params();
    params.verbosity = 2;
    params.max_iters = 200;

    auto solver = std::make_unique<nmpc::NMPCSolverPaper<NX, NU, NC, N>>();
    solver->configure(params);
    auto st = solver->solve(prob);

    printf("\n=== RESULT ===\n");
    printf("Status: %s\n", nmpc::status_string(st));
    auto& s = solver->last_stats();
    printf("Iterations: %d\n", s.inner_iterations);
    printf("Primal inf: %.3e\n", s.primal_infeas);
    printf("Stationarity: %.3e\n", s.dual_infeas);
    printf("Complementarity: %.3e\n", s.complementarity);
    printf("Cost: %.6f\n", s.cost);
    return 0;
}
