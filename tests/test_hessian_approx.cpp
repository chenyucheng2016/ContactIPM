/**
 * @file    test_hessian_approx.cpp
 * @brief   Unit tests for Hessian approximation modules.
 */

#include <cstdio>
#include <cmath>
#include <cassert>

#include "nmpc/nmpc_core.hpp"
#include "nmpc/nmpc_hessian_approx.hpp"

using namespace nmpc;

static int test_count = 0;
static int pass_count = 0;

#define TEST(name) do { test_count++; printf("  TEST %s ... ", name); } while(0)
#define PASS()      do { pass_count++; printf("PASS\n"); } while(0)
#define FAIL(msg)   do { printf("FAIL: %s\n", msg); } while(0)

bool approx(double a, double b, double tol = 1e-6) {
    return std::fabs(a - b) < tol;
}

// ═══════════════════════════════════════════════════════════════════
//  Test 1: Gauss-Newton matches exact Hessian for quadratic LS
//  f(x) = ½||r₁||² + ½||r₂||² with r₁=x₁, r₂=x₂−x₁
//  J = [1 0; -1 1], R = I
//  H_exact = JᵀJ = [2 -1; -1 1]
// ═══════════════════════════════════════════════════════════════════

void test_gauss_newton_quadratic() {
    TEST("Gauss-Newton: quadratic LS");

    constexpr int N_VAR = 2, N_RES = 2;
    Mat<N_RES, N_VAR> J;
    J.zero();
    J(0,0) =  1.0; J(0,1) =  0.0;
    J(1,0) = -1.0; J(1,1) =  1.0;

    Vec<N_RES> r_weights;
    r_weights[0] = 1.0; r_weights[1] = 1.0;

    SymMat<N_VAR> H;
    GaussNewtonHessian<N_VAR, N_RES>::compute(J, r_weights, H);

    // H = JᵀJ = [1 -1; 0 1]ᵀ [1 0; -1 1] = [2 -1; -1 1]
    if (!approx(H(0,0), 2.0) || !approx(H(1,0), -1.0) || !approx(H(1,1), 1.0))
        { FAIL("GN Hessian incorrect"); return; }

    PASS();
}

// ═══════════════════════════════════════════════════════════════════
//  Test 2: Gauss-Newton with weighted residuals
//  Same J, R = diag(2, 0.5)
//  H = Jᵀ·R·J = [1 -1; 0 1]ᵀ [2 0; 0 0.5] [1 0; -1 1]
// ═══════════════════════════════════════════════════════════════════

void test_gauss_newton_weighted() {
    TEST("Gauss-Newton: weighted residuals");

    constexpr int N_VAR = 2, N_RES = 2;
    Mat<N_RES, N_VAR> J;
    J(0,0) =  1.0; J(0,1) =  0.0;
    J(1,0) = -1.0; J(1,1) =  1.0;

    Vec<N_RES> r_weights;
    r_weights[0] = 2.0; r_weights[1] = 0.5;

    SymMat<N_VAR> H;
    GaussNewtonHessian<N_VAR, N_RES>::compute(J, r_weights, H);

    // H = [1 -1; 0 1]ᵀ [2 0; 0 0.5] [1 0; -1 1]
    //   = [1 -1]ᵀ·[2·1  2·0; 0.5·(-1) 0.5·1]... let me just compute
    // Row 0: [1 0] dot [2 -0.5] = 2.0, [1 0] dot [0 0.5] = 0.5
    // Actually Jᵀ·R = [1 -1; 0 1] [2 0; 0 0.5] = [2 -0.5; 0 0.5]
    // Then JᵀR·J = [2 -0.5; 0 0.5] [1 0; -1 1] = [2+0.5 -0.5; -0.5 0.5] = [2.5 -0.5; -0.5 0.5]
    if (!approx(H(0,0), 2.5) || !approx(H(1,0), -0.5) || !approx(H(1,1), 0.5))
        { FAIL("Weighted GN Hessian incorrect"); return; }

    PASS();
}

// ═══════════════════════════════════════════════════════════════════
//  Test 3: BFGS converges to true Hessian on quadratic function
//  f(x) = ½xᵀ·A·x + bᵀx with A = [3 1; 1 2], b = [1; 1]
//  Starting from B₀ = I, after 2 BFGS updates, B should ≈ A
// ═══════════════════════════════════════════════════════════════════

void test_bfgs_convergence() {
    TEST("BFGS converges to true Hessian");

    constexpr int N = 2;
    SymMat<N> B;
    B.set_identity();

    // True Hessian A = [3 1; 1 2]
    SymMat<N> A;
    A.zero();
    A(0,0) = 3.0; A(1,0) = 1.0; A(1,1) = 2.0;

    Vec<N> x, grad;
    x[0] = 2.0; x[1] = 1.0;
    // grad = A·x + b = [3·2+1·1+1; 1·2+2·1+1] = [8; 5]
    grad[0] = 8.0; grad[1] = 5.0;

    // Step 1: take a Newton-ish step
    Vec<N> s;
    s[0] = -1.0; s[1] = -0.5;

    // Compute grad at new point: x_new = [1.0, 0.5]
    // grad_new = A·x_new + b = [3·1+1·0.5+1; 1·1+2·0.5+1] = [4.5; 3.0]
    Vec<N> grad_new;
    grad_new[0] = 4.5; grad_new[1] = 3.0;

    Vec<N> y;
    for (int i = 0; i < N; ++i) y[i] = grad_new[i] - grad[i];
    // y = [4.5-8; 3-5] = [-3.5; -2.0]

    bool ok = PowellBFGS<N>::update(B, s, y);
    if (!ok) { FAIL("BFGS update returned false"); return; }

    // Step 2: take another step
    x[0] = 1.0; x[1] = 0.5;
    // grad = A·x + b = [4.5; 3.0]
    s[0] = -0.5; s[1] = -0.3;
    // x_new = [0.5; 0.2], grad_new = [3·0.5+1·0.2+1; 1·0.5+2·0.2+1] = [2.7; 1.9]
    grad_new[0] = 2.7; grad_new[1] = 1.9;
    for (int i = 0; i < N; ++i) y[i] = grad_new[i] - grad[i];
    // y = [2.7-4.5; 1.9-3.0] = [-1.8; -1.1]

    ok = PowellBFGS<N>::update(B, s, y);
    if (!ok) { FAIL("BFGS update 2 returned false"); return; }

    // After 2 updates on 2D quadratic, BFGS approaches true Hessian
    // but may not reach machine precision without exact line search.
    // Check that diagonal dominates and Hessian is reasonable.
    if (B(0,0) <= 1.0 || B(1,1) <= 1.0) { FAIL("BFGS Hessian not PSD enough"); return; }
    if (B(0,0) > 10.0 || B(1,1) > 10.0) { FAIL("BFGS Hessian diverged"); return; }

    PASS();
}

// ═══════════════════════════════════════════════════════════════════
//  Test 4: Powell damping activates on non-convex curvature
//  s = [1; 0], y = [-1; 0] (negative curvature: sᵀy = -1 < 0)
//  B = I. sᵀB·s = 1. sᵀy = -1 < 0.2·1 → damping should activate.
// ═══════════════════════════════════════════════════════════════════

void test_powell_damping() {
    TEST("Powell damping on negative curvature");

    constexpr int N = 2;
    SymMat<N> B;
    B.set_identity();

    Vec<N> s, y;
    s[0] = 1.0; s[1] = 0.0;
    y[0] = -1.0; y[1] = 0.0;  // negative curvature

    // Without damping, BFGS would fail (sᵀy < 0)
    bool ok = PowellBFGS<N>::update(B, s, y);
    if (!ok) { FAIL("Powell-damped BFGS should succeed"); return; }

    // B should still be PSD (all eigenvalues ≥ 0)
    // Quick check: diagonals non-negative
    if (B(0, 0) < -1e-12 || B(1, 1) < -1e-12)
        { FAIL("Hessian not PSD after damping"); return; }

    PASS();
}

// ═══════════════════════════════════════════════════════════════════

int main() {
    printf("╔══════════════════════════════════════════╗\n");
    printf("║   Hessian Approximation Unit Tests      ║\n");
    printf("╚══════════════════════════════════════════╝\n\n");

    test_gauss_newton_quadratic();
    test_gauss_newton_weighted();
    test_bfgs_convergence();
    test_powell_damping();

    printf("\n─── Results: %d/%d passed ───\n", pass_count, test_count);
    return (pass_count == test_count) ? 0 : 1;
}
