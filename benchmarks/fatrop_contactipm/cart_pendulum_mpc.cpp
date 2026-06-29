/**
 * @file    cart_pendulum_mpc.cpp
 * @brief   Fatrop benchmark: CartPendulumMPC using ContactIPM with casadi-generated code.
 *
 * Problem: 4-state cart-pole swing-up (MPC formulation)
 *   States:  [pos, theta, vel, omega]
 *   Control: [Fex]
 *   N=25, T=2.5s, dt=0.1s
 *
 * Generated C functions from casadi:
 *   f_expl, f_jac_x, f_jac_u, l_stage, l_stage_grad_x, l_stage_grad_u,
 *   l_stage_hess_xx, l_stage_hess_uu, l_stage_hess_ux, l_term, l_term_grad,
 *   l_term_hess, g_path, g_path_jac_x, g_path_jac_u, g_term, g_term_jac
 */

#include <cstdio>
#include <cmath>
#include <chrono>

#ifndef M_PI
#define M_PI 3.14159265358979323846
#endif

#include "codegen_driver.hpp"
#include "nmpc/nmpc_solver_paper.hpp"

// ── Declare generated C functions ────────────────────────────────────────
extern "C" {
    int f_expl(const double** arg, double** res, long long* iw, double* w, int mem);
    int f_jac_x(const double** arg, double** res, long long* iw, double* w, int mem);
    int f_jac_u(const double** arg, double** res, long long* iw, double* w, int mem);

    int l_stage(const double** arg, double** res, long long* iw, double* w, int mem);
    int l_stage_grad_x(const double** arg, double** res, long long* iw, double* w, int mem);
    int l_stage_grad_u(const double** arg, double** res, long long* iw, double* w, int mem);
    int l_stage_hess_xx(const double** arg, double** res, long long* iw, double* w, int mem);
    int l_stage_hess_uu(const double** arg, double** res, long long* iw, double* w, int mem);
    int l_stage_hess_ux(const double** arg, double** res, long long* iw, double* w, int mem);

    int l_term(const double** arg, double** res, long long* iw, double* w, int mem);
    int l_term_grad(const double** arg, double** res, long long* iw, double* w, int mem);
    int l_term_hess(const double** arg, double** res, long long* iw, double* w, int mem);

    int g_path(const double** arg, double** res, long long* iw, double* w, int mem);
    int g_path_jac_x(const double** arg, double** res, long long* iw, double* w, int mem);
    int g_path_jac_u(const double** arg, double** res, long long* iw, double* w, int mem);

    int g_term(const double** arg, double** res, long long* iw, double* w, int mem);
    int g_term_jac(const double** arg, double** res, long long* iw, double* w, int mem);
}

// ── Problem dimensions ───────────────────────────────────────────────────
constexpr int NX = 4;   // [pos, theta, vel, omega]
constexpr int NU = 1;   // [Fex]
constexpr int NC = 6;   // force(2) + pos(2) + vel(2)
constexpr int N  = 25;  // horizon
constexpr double DT = 0.1;

using VecX = nmpc::Vec<NX>;
using VecU = nmpc::Vec<NU>;
using VecC = nmpc::Vec<NC>;

int main() {
    printf("═══════════════════════════════════════════════════════\n");
    printf("  Fatrop Benchmark: CartPendulumMPC (ContactIPM)\n");
    printf("═══════════════════════════════════════════════════════\n\n");

    // ── Set up codegen models ────────────────────────────────────────────
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

    // ── Create problem ───────────────────────────────────────────────────
    using Problem = nmpc::NMPCProblem<NX, NU, NC, N>;
    Problem prob;
    prob.dynamics = &dyn;
    prob.cost = &cost;
    prob.constraints = &cons;
    prob.dt = DT;

    // Initial state: [pos=0, theta=pi, vel=0, omega=1.0]
    VecX x0;
    x0[0] = 0.0;
    x0[1] = M_PI;
    x0[2] = 0.0;
    x0[3] = 1.0;
    prob.x0 = x0;

    // Initialize all stages with x0 and zero control
    for (int k = 0; k <= N; ++k) {
        prob.stages[k].x = x0;
        prob.stages[k].u.zero();
    }

    // ── Configure solver ─────────────────────────────────────────────────
    nmpc::NMPCSolverPaper<NX, NU, NC, N> solver;
    nmpc::PaperIPMParams pp;
    pp.mu_init = 0.2;
    pp.max_iters = 500;
    pp.mu_min = 1e-4;
    pp.tol_primal = 1e-4;
    pp.tol_compl = 1e-4;
    pp.tol_ineq = 1e-4;
    pp.tol_stat = 0.02;
    pp.enable_preconditioner = true;  // diagonal Jacobi preconditioner
    pp.verbosity = 2;
    solver.configure(pp);

    // ── Warm timing loop ─────────────────────────────────────────────────
    nmpc::StageData<NX, NU, NC> guess[N + 1];
    VecX x0_saved = prob.x0;
    for (int k = 0; k <= N; ++k) guess[k] = prob.stages[k];

    constexpr int NTIMINGS = 5;
    double times_ms[NTIMINGS];
    double min_ms = 1e12;
    nmpc::Status st = nmpc::Status::SUCCESS;
    for (int ii = 0; ii < NTIMINGS; ++ii) {
        prob.x0 = x0_saved;
        for (int k = 0; k <= N; ++k) prob.stages[k] = guess[k];
        nmpc::PaperIPMParams quiet = pp;
        quiet.verbosity = 0;
        solver.configure(quiet);
        auto t_start = std::chrono::high_resolution_clock::now();
        st = solver.solve(prob);
        auto t_end = std::chrono::high_resolution_clock::now();
        times_ms[ii] = std::chrono::duration<double, std::milli>(t_end - t_start).count();
        if (times_ms[ii] < min_ms) min_ms = times_ms[ii];
    }
    // Sort times
    for (int i = 0; i < NTIMINGS; ++i)
        for (int j = i + 1; j < NTIMINGS; ++j)
            if (times_ms[j] < times_ms[i]) {
                double tmp = times_ms[i]; times_ms[i] = times_ms[j]; times_ms[j] = tmp;
            }
    double median_ms = times_ms[NTIMINGS / 2];

    const auto& s = solver.last_stats();

    printf("\n=== SOLVE COMPLETE ===\n");
    printf("Status: %s\n", nmpc::status_string(st));
    printf("Iterations: %d\n", s.inner_iterations);
    printf("Final mu: %.3e\n", s.barrier_param);
    printf("Primal inf: %.3e\n", s.primal_infeas);
    printf("Stationarity: %.3e\n", s.dual_infeas);
    printf("Complementarity: %.3e\n", s.complementarity);
    printf("Cost: %.4f\n", s.cost);
    printf("First u* = [%.3f]\n", prob.stages[0].u[0]);

    printf("\n=== BENCHMARK SUMMARY ===\n");
    printf("Solver:          ContactIPM\n");
    printf("Problem:         cart_pendulum_mpc\n");
    printf("Status:          %s\n", nmpc::status_string(st));
    printf("Iterations:      %d\n", s.inner_iterations);
    printf("Solve time:      %.3f ms\n", min_ms);
    printf("Median time:     %.3f ms\n", median_ms);
    printf("Primal inf:      %.3e\n", s.primal_infeas);
    printf("Stationarity:    %.3e\n", s.dual_infeas);
    printf("Complementarity: %.3e\n", s.complementarity);
    printf("Cost:            %.4f\n", s.cost);

    return 0;
}
