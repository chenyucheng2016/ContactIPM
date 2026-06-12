/**
 * @file    trajectory_dump.hpp
 * @brief   Header-only utility to dump NMPC trajectories to stdout and JSON.
 *
 * Provides two template functions:
 *   print_trajectory_table()  – human-readable table to stdout
 *   dump_trajectory_json()    – structured JSON file for comparison scripts
 */
#pragma once

#include <cstdio>
#include <cstring>
#include <string>
#include "nmpc/nmpc_problem.hpp"

namespace nmpc {

/**
 * Print a human-readable trajectory table to stdout.
 * Columns: k | x[0..NX] | u[0..NU] | cost_k | g[0..NC]
 */
template <int NX, int NU, int NC, int N>
void print_trajectory_table(const NMPCProblem<NX, NU, NC, N>& prob) {
    // Header
    printf("\n─── Trajectory Table ───\n");
    printf("  k |");
    for (int i = 0; i < NX; ++i) printf(" x[%d]  ", i);
    printf("|");
    for (int i = 0; i < NU; ++i) printf(" u[%d]  ", i);
    printf("| cost_k  |");
    for (int i = 0; i < NC; ++i) printf(" g[%d]  ", i);
    printf("\n");

    // Separator
    printf("----+");
    for (int i = 0; i < NX; ++i) printf("-------");
    printf("+");
    for (int i = 0; i < NU; ++i) printf("-------");
    printf("+---------+");
    for (int i = 0; i < NC; ++i) printf("-------");
    printf("\n");

    Vec<NC> g_vec;
    for (int k = 0; k <= N; ++k) {
        printf("%3d |", k);
        for (int i = 0; i < NX; ++i) printf(" %6.3f", prob.stages[k].x[i]);
        printf(" |");
        if (k < N) {
            for (int i = 0; i < NU; ++i) printf(" %6.3f", prob.stages[k].u[i]);
        } else {
            for (int i = 0; i < NU; ++i) printf("   -   ");
        }
        printf(" | %7.4f |", prob.stages[k].cost);

        // Evaluate constraints
        if (k < N && prob.constraints) {
            prob.constraints->evaluate(prob.stages[k].x, prob.stages[k].u, k, g_vec);
        } else if (k == N && prob.constraints) {
            prob.constraints->evaluate_terminal(prob.stages[k].x, g_vec);
        }
        for (int i = 0; i < NC; ++i) {
            if (g_vec[i] < -1e8) printf("   -   ");
            else printf(" %6.3f", g_vec[i]);
        }
        printf("\n");
    }

    // Summary
    double total = 0.0;
    for (int k = 0; k < N; ++k) total += prob.stages[k].cost;
    double terminal = prob.stages[N].cost;
    printf("───\n");
    printf("Running cost sum: %.6f\n", total);
    printf("Terminal cost:    %.6f\n", terminal);
    printf("Total (solver):   %.6f\n", total + terminal);
}

/**
 * Dump trajectory to a JSON file.
 * Format mirrors the Python acados dump for consistent comparison.
 */
template <int NX, int NU, int NC, int N>
void dump_trajectory_json(const NMPCProblem<NX, NU, NC, N>& prob,
                          const char* filepath,
                          const char* problem_name,
                          int iterations,
                          double cost_total,
                          double dt) {
    FILE* fp = fopen(filepath, "w");
    if (!fp) {
        fprintf(stderr, "WARNING: could not open %s for writing\n", filepath);
        return;
    }

    fprintf(fp, "{\n");
    fprintf(fp, "  \"solver\": \"ContactIPM\",\n");
    fprintf(fp, "  \"problem\": \"%s\",\n", problem_name);
    fprintf(fp, "  \"NX\": %d, \"NU\": %d, \"NC\": %d, \"N\": %d,\n", NX, NU, NC, N);
    fprintf(fp, "  \"dt\": %.6f,\n", dt);
    fprintf(fp, "  \"iterations\": %d,\n", iterations);
    fprintf(fp, "  \"cost_total\": %.8f,\n", cost_total);

    // Compute running + terminal cost
    double running_sum = 0.0;
    for (int k = 0; k < N; ++k) running_sum += prob.stages[k].cost;
    double terminal_cost = prob.stages[N].cost;
    fprintf(fp, "  \"cost_running\": %.8f,\n", running_sum);
    fprintf(fp, "  \"cost_terminal\": %.8f,\n", terminal_cost);

    fprintf(fp, "  \"stages\": [\n");

    Vec<NC> g_vec;
    for (int k = 0; k <= N; ++k) {
        fprintf(fp, "    {\"k\": %d", k);

        // x
        fprintf(fp, ", \"x\": [");
        for (int i = 0; i < NX; ++i)
            fprintf(fp, "%.10f%s", prob.stages[k].x[i], (i < NX-1) ? ", " : "");
        fprintf(fp, "]");

        // u
        fprintf(fp, ", \"u\": [");
        if (k < N) {
            for (int i = 0; i < NU; ++i)
                fprintf(fp, "%.10f%s", prob.stages[k].u[i], (i < NU-1) ? ", " : "");
        }
        fprintf(fp, "]");

        // s (slacks)
        fprintf(fp, ", \"s\": [");
        for (int i = 0; i < NC; ++i)
            fprintf(fp, "%.10f%s", prob.stages[k].s[i], (i < NC-1) ? ", " : "");
        fprintf(fp, "]");

        // lambda (dual multipliers)
        fprintf(fp, ", \"lambda\": [");
        for (int i = 0; i < NC; ++i)
            fprintf(fp, "%.10f%s", prob.stages[k].lambda[i], (i < NC-1) ? ", " : "");
        fprintf(fp, "]");

        // cost
        fprintf(fp, ", \"cost\": %.10f", prob.stages[k].cost);

        // g (constraint values)
        if (k < N && prob.constraints) {
            prob.constraints->evaluate(prob.stages[k].x, prob.stages[k].u, k, g_vec);
        } else if (k == N && prob.constraints) {
            prob.constraints->evaluate_terminal(prob.stages[k].x, g_vec);
        }
        fprintf(fp, ", \"g\": [");
        for (int i = 0; i < NC; ++i)
            fprintf(fp, "%.10f%s", g_vec[i], (i < NC-1) ? ", " : "");
        fprintf(fp, "]");

        fprintf(fp, "}%s\n", (k < N) ? "," : "");
    }

    fprintf(fp, "  ]\n");
    fprintf(fp, "}\n");
    fclose(fp);
    printf("Trajectory JSON saved to: %s\n", filepath);
}

} // namespace nmpc
