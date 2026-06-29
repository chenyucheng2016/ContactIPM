#pragma once
/**
 * @file    fatrop_benchmark_common.hpp
 * @brief   Common benchmark runner for fatrop ContactIPM benchmarks.
 *
 * Provides a timing loop and standardized output format.
 */

#include "codegen_driver.hpp"
#include "nmpc/nmpc_solver_paper.hpp"
#include <cstdio>
#include <chrono>
#include <memory>

#ifndef M_PI
#define M_PI 3.14159265358979323846
#endif

namespace fatrop_bench {

template <int NX, int NU, int NC, int N>
struct BenchResult {
    const char* status;
    int iterations;
    double solve_time_ms;
    double median_time_ms;
    double primal_inf;
    double stationarity;
    double complementarity;
    double cost;
    double final_mu;
};

template <int NX, int NU, int NC, int N>
BenchResult<NX, NU, NC, N> run_benchmark(
    nmpc::NMPCProblem<NX, NU, NC, N>& prob,
    nmpc::PaperIPMParams pp,
    const char* problem_name,
    int ntimings = 5)
{
    auto solver = std::make_unique<nmpc::NMPCSolverPaper<NX, NU, NC, N>>();
    solver->configure(pp);

    // Save initial guess (heap-allocated to avoid stack overflow for large NX)
    auto guess = std::make_unique<nmpc::StageData<NX, NU, NC>[]>(N + 1);
    nmpc::Vec<NX> x0_saved = prob.x0;
    for (int k = 0; k <= N; ++k) guess[k] = prob.stages[k];

    // Diagnostic verbose run (prints per-iteration stats)
    {
        prob.x0 = x0_saved;
        for (int k = 0; k <= N; ++k) prob.stages[k] = guess[k];
        nmpc::PaperIPMParams verbose_pp = pp;
        verbose_pp.verbosity = 1;
        solver->configure(verbose_pp);
        solver->solve(prob);
        printf("\n--- Timing runs (verbosity=0) ---\n");
    }

    // Timing loop
    double times_ms[16];
    double min_ms = 1e12;
    nmpc::Status st = nmpc::Status::SUCCESS;
    int nt = (ntimings > 16) ? 16 : ntimings;

    for (int ii = 0; ii < nt; ++ii) {
        prob.x0 = x0_saved;
        for (int k = 0; k <= N; ++k) prob.stages[k] = guess[k];
        nmpc::PaperIPMParams quiet = pp;
        quiet.verbosity = 0;
        solver->configure(quiet);
        auto t0 = std::chrono::high_resolution_clock::now();
        st = solver->solve(prob);
        auto t1 = std::chrono::high_resolution_clock::now();
        times_ms[ii] = std::chrono::duration<double, std::milli>(t1 - t0).count();
        if (times_ms[ii] < min_ms) min_ms = times_ms[ii];
    }

    // Sort times
    for (int i = 0; i < nt; ++i)
        for (int j = i + 1; j < nt; ++j)
            if (times_ms[j] < times_ms[i]) {
                double tmp = times_ms[i]; times_ms[i] = times_ms[j]; times_ms[j] = tmp;
            }
    double median_ms = times_ms[nt / 2];

    const auto& s = solver->last_stats();
    BenchResult<NX, NU, NC, N> r;
    r.status = nmpc::status_string(st);
    r.iterations = s.inner_iterations;
    r.solve_time_ms = min_ms;
    r.median_time_ms = median_ms;
    r.primal_inf = s.primal_infeas;
    r.stationarity = s.dual_infeas;
    r.complementarity = s.complementarity;
    r.cost = s.cost;
    r.final_mu = s.barrier_param;

    // Print summary
    printf("\n=== BENCHMARK SUMMARY ===\n");
    printf("Solver:          ContactIPM\n");
    printf("Problem:         %s\n", problem_name);
    printf("Status:          %s\n", r.status);
    printf("Iterations:      %d\n", r.iterations);
    printf("Solve time:      %.3f ms (min)\n", r.solve_time_ms);
    printf("Median time:     %.3f ms\n", r.median_time_ms);
    printf("Primal inf:      %.3e\n", r.primal_inf);
    printf("Stationarity:    %.3e\n", r.stationarity);
    printf("Complementarity: %.3e\n", r.complementarity);
    printf("Cost:            %.6f\n", r.cost);
    printf("Final mu:        %.3e\n", r.final_mu);

    return r;
}

// Default solver params for benchmarks
inline nmpc::PaperIPMParams default_params() {
    nmpc::PaperIPMParams pp;
    pp.mu_init = 0.1;
    pp.max_iters = 500;
    pp.mu_min = 5e-5;
    pp.tol_primal = 3e-4;   // primal feasibility tolerance
    pp.tol_compl = 1e-4;
    pp.tol_ineq = 1e-4;
    pp.tol_stat = 0.02;
    pp.enable_preconditioner = true;  // diagonal Jacobi preconditioner
    pp.verbosity = 2;
    return pp;
}

} // namespace fatrop_bench
