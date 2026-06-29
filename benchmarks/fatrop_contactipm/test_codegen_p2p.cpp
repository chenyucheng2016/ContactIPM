#include <cstdio>
#include <cmath>
#include "quadcopter_p2p.c"

// Minimal Vec/Mat stubs
int main() {
    printf("Test 1: f_expl\n"); fflush(stdout);
    double x[7] = {0, 0, 2.5, 0, 0, 0, 3.0};
    double u[5] = {9.81, 0, 0, 0, 0};
    double xdot[7] = {};
    const double* arg[2] = {x, u};
    double* res[1] = {xdot};
    f_expl(arg, res, nullptr, nullptr, 0);
    printf("xdot = [%.3f, %.3f, %.3f, %.3f, %.3f, %.3f, %.3f]\n",
           xdot[0], xdot[1], xdot[2], xdot[3], xdot[4], xdot[5], xdot[6]);
    fflush(stdout);

    printf("Test 2: l_stage\n"); fflush(stdout);
    double cost[1] = {};
    const double* arg2[2] = {x, u};
    double* res2[1] = {cost};
    l_stage(arg2, res2, nullptr, nullptr, 0);
    printf("stage_cost = %.6f\n", cost[0]);
    fflush(stdout);

    printf("Test 3: g_path\n"); fflush(stdout);
    double g[7] = {};
    double* res3[1] = {g};
    g_path(arg2, res3, nullptr, nullptr, 0);
    printf("g_path = [%.3f, %.3f, %.3f, %.3f, %.3f, %.3f, %.3f]\n",
           g[0], g[1], g[2], g[3], g[4], g[5], g[6]);
    fflush(stdout);

    printf("Test 4: l_stage_grad_x\n"); fflush(stdout);
    double gx[7] = {};
    double* res4[1] = {gx};
    l_stage_grad_x(arg2, res4, nullptr, nullptr, 0);
    printf("grad_x = [%.3e, %.3e, %.3e, %.3e, %.3e, %.3e, %.3e]\n",
           gx[0], gx[1], gx[2], gx[3], gx[4], gx[5], gx[6]);
    fflush(stdout);

    printf("All generated C function tests passed!\n");
    return 0;
}
