/**
 * @file    verify_codegen.cpp
 * @brief   Verify CasADi codegen gradients, Hessians, and constraint Jacobians
 *          against finite differences.
 */
#include <cstdio>
#include <cmath>
#include <cstring>
#include "codegen_driver.hpp"
#include "quadcopter_mpc.c"

using namespace nmpc;

constexpr int NX = 6, NU = 4, NC = 7;

int main() {
    printf("=== CasADi Codegen Verification ===\n\n");

    // Test point
    double x[NX] = {0.5, -0.3, 2.8, 0.8, 0.6, 0.2};
    double u[NU] = {10.5, 0.2, 0.15, 0.1};

    printf("Test point:\n  x = [%.3f, %.3f, %.3f, %.3f, %.3f, %.3f]\n", x[0],x[1],x[2],x[3],x[4],x[5]);
    printf("  u = [%.3f, %.3f, %.3f, %.3f]\n\n", u[0],u[1],u[2],u[3]);

    const double eps = 1e-6;

    // === 1. Cost gradient: l_stage_grad_{x,u} vs FD of l_stage ===
    printf("--- Cost gradient w.r.t. x ---\n");
    double gx_casadi[NX], gu_casadi[NU];
    casadi_call(l_stage_grad_x, x, u, gx_casadi);
    casadi_call(l_stage_grad_u, x, u, gu_casadi);

    // FD gradient w.r.t. x
    double gx_fd[NX];
    for (int j = 0; j < NX; ++j) {
        double xp[NX], xm[NX];
        memcpy(xp, x, NX*sizeof(double)); memcpy(xm, x, NX*sizeof(double));
        xp[j] += eps; xm[j] -= eps;
        double fp[1], fm[1];
        casadi_call(l_stage, xp, u, fp);
        casadi_call(l_stage, xm, u, fm);
        gx_fd[j] = (fp[0] - fm[0]) / (2.0*eps);
    }
    double max_diff_gx = 0;
    for (int i = 0; i < NX; ++i) {
        double d = std::abs(gx_casadi[i] - gx_fd[i]);
        if (d > max_diff_gx) max_diff_gx = d;
        printf("  gx[%d]: CasADi=%12.6e  FD=%12.6e  diff=%8.2e\n", i, gx_casadi[i], gx_fd[i], d);
    }
    printf("  Max diff: %.2e\n\n", max_diff_gx);

    // FD gradient w.r.t. u
    double gu_fd[NU];
    for (int j = 0; j < NU; ++j) {
        double up[NU], um[NU];
        memcpy(up, u, NU*sizeof(double)); memcpy(um, u, NU*sizeof(double));
        up[j] += eps; um[j] -= eps;
        double fp[1], fm[1];
        casadi_call(l_stage, x, up, fp);
        casadi_call(l_stage, x, um, fm);
        gu_fd[j] = (fp[0] - fm[0]) / (2.0*eps);
    }
    printf("--- Cost gradient w.r.t. u ---\n");
    double max_diff_gu = 0;
    for (int i = 0; i < NU; ++i) {
        double d = std::abs(gu_casadi[i] - gu_fd[i]);
        if (d > max_diff_gu) max_diff_gu = d;
        printf("  gu[%d]: CasADi=%12.6e  FD=%12.6e  diff=%8.2e\n", i, gu_casadi[i], gu_fd[i], d);
    }
    printf("  Max diff: %.2e\n\n", max_diff_gu);

    // === 2. Cost Hessian: l_stage_hess_xx vs FD of l_stage_grad_x ===
    printf("--- Cost Hessian xx ---\n");
    double Hxx_casadi[NX*NX], Huu_casadi[NU*NU], Hux_casadi[NU*NX];
    casadi_call(l_stage_hess_xx, x, u, Hxx_casadi);
    casadi_call(l_stage_hess_uu, x, u, Huu_casadi);
    casadi_call(l_stage_hess_ux, x, u, Hux_casadi);

    // FD Hessian xx: d(grad_x)/dx
    double Hxx_fd[NX*NX];
    for (int j = 0; j < NX; ++j) {
        double xp[NX], xm[NX], gx_p[NX], gx_m[NX];
        memcpy(xp, x, NX*sizeof(double)); memcpy(xm, x, NX*sizeof(double));
        xp[j] += eps; xm[j] -= eps;
        casadi_call(l_stage_grad_x, xp, u, gx_p);
        casadi_call(l_stage_grad_x, xm, u, gx_m);
        for (int i = 0; i < NX; ++i)
            Hxx_fd[i + j*NX] = (gx_p[i] - gx_m[i]) / (2.0*eps);
    }
    double max_diff_Hxx = 0;
    for (int j = 0; j < NX; ++j)
        for (int i = 0; i < NX; ++i) {
            double d = std::abs(Hxx_casadi[i + j*NX] - Hxx_fd[i + j*NX]);
            if (d > max_diff_Hxx) max_diff_Hxx = d;
        }
    printf("  Hxx max diff (CasADi vs FD): %.2e\n", max_diff_Hxx);
    for (int i = 0; i < NX; ++i) {
        printf("  Hxx[%d,:]: CasADi=[", i);
        for (int j = 0; j < NX; ++j) printf("%8.4f ", Hxx_casadi[i + j*NX]);
        printf("]  FD=[");
        for (int j = 0; j < NX; ++j) printf("%8.4f ", Hxx_fd[i + j*NX]);
        printf("]\n");
    }
    printf("\n");

    // FD Hessian uu: d(grad_u)/du
    double Huu_fd[NU*NU];
    for (int j = 0; j < NU; ++j) {
        double up[NU], um[NU], gu_p[NU], gu_m[NU];
        memcpy(up, u, NU*sizeof(double)); memcpy(um, u, NU*sizeof(double));
        up[j] += eps; um[j] -= eps;
        casadi_call(l_stage_grad_u, x, up, gu_p);
        casadi_call(l_stage_grad_u, x, um, gu_m);
        for (int i = 0; i < NU; ++i)
            Huu_fd[i + j*NU] = (gu_p[i] - gu_m[i]) / (2.0*eps);
    }
    double max_diff_Huu = 0;
    for (int j = 0; j < NU; ++j)
        for (int i = 0; i < NU; ++i) {
            double d = std::abs(Huu_casadi[i + j*NU] - Huu_fd[i + j*NU]);
            if (d > max_diff_Huu) max_diff_Huu = d;
        }
    printf("  Huu max diff (CasADi vs FD): %.2e\n", max_diff_Huu);
    for (int i = 0; i < NU; ++i) {
        printf("  Huu[%d,:]: CasADi=[", i);
        for (int j = 0; j < NU; ++j) printf("%8.4f ", Huu_casadi[i + j*NU]);
        printf("]  FD=[");
        for (int j = 0; j < NU; ++j) printf("%8.4f ", Huu_fd[i + j*NU]);
        printf("]\n");
    }
    printf("\n");

    // FD Hessian ux: d(grad_u)/dx
    double Hux_fd[NU*NX];
    for (int j = 0; j < NX; ++j) {
        double xp[NX], xm[NX], gu_p[NU], gu_m[NU];
        memcpy(xp, x, NX*sizeof(double)); memcpy(xm, x, NX*sizeof(double));
        xp[j] += eps; xm[j] -= eps;
        casadi_call(l_stage_grad_u, xp, u, gu_p);
        casadi_call(l_stage_grad_u, xm, u, gu_m);
        for (int i = 0; i < NU; ++i)
            Hux_fd[i + j*NU] = (gu_p[i] - gu_m[i]) / (2.0*eps);
    }
    double max_diff_Hux = 0;
    for (int j = 0; j < NX; ++j)
        for (int i = 0; i < NU; ++i) {
            double d = std::abs(Hux_casadi[i + j*NU] - Hux_fd[i + j*NU]);
            if (d > max_diff_Hux) max_diff_Hux = d;
        }
    printf("  Hux max diff (CasADi vs FD): %.2e\n\n", max_diff_Hux);

    // === 3. Constraint Jacobians ===
    printf("--- Constraint Jacobian w.r.t. x ---\n");
    double Cx_casadi[NC*NX], Cu_casadi[NC*NU];
    casadi_call(g_path_jac_x, x, u, Cx_casadi);
    casadi_call(g_path_jac_u, x, u, Cu_casadi);

    // FD constraint Jac x
    double Cx_fd[NC*NX];
    for (int j = 0; j < NX; ++j) {
        double xp[NX], xm[NX], g_p[NC], g_m[NC];
        memcpy(xp, x, NX*sizeof(double)); memcpy(xm, x, NX*sizeof(double));
        xp[j] += eps; xm[j] -= eps;
        casadi_call(g_path, xp, u, g_p);
        casadi_call(g_path, xm, u, g_m);
        for (int i = 0; i < NC; ++i)
            Cx_fd[i + j*NC] = (g_p[i] - g_m[i]) / (2.0*eps);
    }
    double max_diff_Cx = 0;
    for (int j = 0; j < NX; ++j)
        for (int i = 0; i < NC; ++i) {
            double d = std::abs(Cx_casadi[i + j*NC] - Cx_fd[i + j*NC]);
            if (d > max_diff_Cx) max_diff_Cx = d;
        }
    printf("  Cx max diff (CasADi vs FD): %.2e\n", max_diff_Cx);
    for (int i = 0; i < NC; ++i) {
        printf("  Cx[%d,:]: CasADi=[", i);
        for (int j = 0; j < NX; ++j) printf("%10.4e ", Cx_casadi[i + j*NC]);
        printf("]\n  Cx[%d,:]: FD    =[", i);
        for (int j = 0; j < NX; ++j) printf("%10.4e ", Cx_fd[i + j*NC]);
        printf("]\n");
    }
    printf("\n");

    // FD constraint Jac u
    printf("--- Constraint Jacobian w.r.t. u ---\n");
    double Cu_fd[NC*NU];
    for (int j = 0; j < NU; ++j) {
        double up[NU], um[NU], g_p[NC], g_m[NC];
        memcpy(up, u, NU*sizeof(double)); memcpy(um, u, NU*sizeof(double));
        up[j] += eps; um[j] -= eps;
        casadi_call(g_path, x, up, g_p);
        casadi_call(g_path, x, um, g_m);
        for (int i = 0; i < NC; ++i)
            Cu_fd[i + j*NC] = (g_p[i] - g_m[i]) / (2.0*eps);
    }
    double max_diff_Cu = 0;
    for (int j = 0; j < NU; ++j)
        for (int i = 0; i < NC; ++i) {
            double d = std::abs(Cu_casadi[i + j*NC] - Cu_fd[i + j*NC]);
            if (d > max_diff_Cu) max_diff_Cu = d;
        }
    printf("  Cu max diff (CasADi vs FD): %.2e\n", max_diff_Cu);
    for (int i = 0; i < NC; ++i) {
        printf("  Cu[%d,:]: CasADi=[", i);
        for (int j = 0; j < NU; ++j) printf("%10.4e ", Cu_casadi[i + j*NC]);
        printf("]\n  Cu[%d,:]: FD    =[", i);
        for (int j = 0; j < NU; ++j) printf("%10.4e ", Cu_fd[i + j*NC]);
        printf("]\n");
    }
    printf("\n");

    // === 4. Terminal cost gradient and Hessian ===
    printf("--- Terminal cost gradient ---\n");
    double pt_casadi[NX];
    casadi_call_1in(l_term_grad, x, pt_casadi);
    double pt_fd[NX];
    for (int j = 0; j < NX; ++j) {
        double xp[NX], xm[NX];
        memcpy(xp, x, NX*sizeof(double)); memcpy(xm, x, NX*sizeof(double));
        xp[j] += eps; xm[j] -= eps;
        double fp[1], fm[1];
        casadi_call_1in(l_term, xp, fp);
        casadi_call_1in(l_term, xm, fm);
        pt_fd[j] = (fp[0] - fm[0]) / (2.0*eps);
    }
    double max_diff_pt = 0;
    for (int i = 0; i < NX; ++i) {
        double d = std::abs(pt_casadi[i] - pt_fd[i]);
        if (d > max_diff_pt) max_diff_pt = d;
        printf("  pt[%d]: CasADi=%12.6e  FD=%12.6e  diff=%8.2e\n", i, pt_casadi[i], pt_fd[i], d);
    }
    printf("  Max diff: %.2e\n\n", max_diff_pt);

    // Terminal Hessian
    printf("--- Terminal Hessian ---\n");
    double Pt_casadi[NX*NX];
    casadi_call_1in(l_term_hess, x, Pt_casadi);
    double Pt_fd[NX*NX];
    for (int j = 0; j < NX; ++j) {
        double xp[NX], xm[NX], gp[NX], gm[NX];
        memcpy(xp, x, NX*sizeof(double)); memcpy(xm, x, NX*sizeof(double));
        xp[j] += eps; xm[j] -= eps;
        casadi_call_1in(l_term_grad, xp, gp);
        casadi_call_1in(l_term_grad, xm, gm);
        for (int i = 0; i < NX; ++i)
            Pt_fd[i + j*NX] = (gp[i] - gm[i]) / (2.0*eps);
    }
    double max_diff_Pt = 0;
    for (int j = 0; j < NX; ++j)
        for (int i = 0; i < NX; ++i) {
            double d = std::abs(Pt_casadi[i + j*NX] - Pt_fd[i + j*NX]);
            if (d > max_diff_Pt) max_diff_Pt = d;
        }
    printf("  Pt max diff: %.2e\n\n", max_diff_Pt);

    // === Summary ===
    printf("=== SUMMARY ===\n");
    printf("  grad_x  max diff: %.2e  %s\n", max_diff_gx, max_diff_gx < 1e-4 ? "OK" : "FAIL");
    printf("  grad_u  max diff: %.2e  %s\n", max_diff_gu, max_diff_gu < 1e-4 ? "OK" : "FAIL");
    printf("  Hxx     max diff: %.2e  %s\n", max_diff_Hxx, max_diff_Hxx < 1e-4 ? "OK" : "FAIL");
    printf("  Huu     max diff: %.2e  %s\n", max_diff_Huu, max_diff_Huu < 1e-4 ? "OK" : "FAIL");
    printf("  Hux     max diff: %.2e  %s\n", max_diff_Hux, max_diff_Hux < 1e-4 ? "OK" : "FAIL");
    printf("  Cx      max diff: %.2e  %s\n", max_diff_Cx, max_diff_Cx < 1e-4 ? "OK" : "FAIL");
    printf("  Cu      max diff: %.2e  %s\n", max_diff_Cu, max_diff_Cu < 1e-4 ? "OK" : "FAIL");
    printf("  t_grad  max diff: %.2e  %s\n", max_diff_pt, max_diff_pt < 1e-4 ? "OK" : "FAIL");
    printf("  t_hess  max diff: %.2e  %s\n", max_diff_Pt, max_diff_Pt < 1e-4 ? "OK" : "FAIL");

    return 0;
}
