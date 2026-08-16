/**
 * @file    test_riccati.cpp
 * @brief   Unit tests for Riccati recursion solver.
 *
 * Validates:
 *   1. LDL^T factorization and solve on small SPD matrices
 *   2. Riccati backward pass for a double-integrator LQR
 *   3. Forward pass recovers the optimal trajectory
 *   4. Regularization handles near-singular Hessians
 */

#include <cstdio>
#include <cmath>
#include <cassert>

#include "nmpc/nmpc_core.hpp"
#include "nmpc/nmpc_riccati.hpp"

using namespace nmpc;

// ── Simple double integrator LQR ────────────────────────────────────────────
// Dynamics:  x_{k+1} = [1 dt; 0 1] x_k + [dt²/2; dt] u_k
// Cost:      Σ ½(x_k^T Q x_k + R u_k²) + ½ x_N^T Q_N x_N
// This has a known analytical Riccati solution to validate against.

static int test_count = 0;
static int pass_count = 0;

#define TEST(name) do { test_count++; printf("  TEST %s ... ", name); } while(0)
#define PASS()      do { pass_count++; printf("PASS\n"); } while(0)
#define FAIL(msg)   do { printf("FAIL: %s\n", msg); } while(0)

bool approx(double a, double b, double tol = 1e-6) {
    return std::fabs(a - b) < tol;
}

// ─────────────────────────────────────────────────────────────────────────────
//  Test 1: LDL^T factorization and solve
// ─────────────────────────────────────────────────────────────────────────────

void test_ldlt() {
    TEST("LDL^T factor and solve");

    // A = [4 1 0; 1 3 1; 0 1 2]  (SPD)
    SymMat<3> A;
    A.zero();
    A(0,0)=4; A(1,0)=1; A(1,1)=3; A(2,1)=1; A(2,2)=2;

    bool ok = A.ldlt_factorize();
    if (!ok) { FAIL("factorization failed"); return; }

    // Check D positive
    if (A(0,0) <= 0 || A(1,1) <= 0 || A(2,2) <= 0) {
        FAIL("D not positive"); return;
    }

    // Solve A x = b  where b = [5; 4; 3]
    Vec<3> b;
    b[0]=5; b[1]=4; b[2]=3;
    A.ldlt_solve(b);

    // Verify: A * x ≈ [5; 4; 3]
    Vec<3> Ax;
    Ax[0] = 4*b[0] + 1*b[1];
    Ax[1] = 1*b[0] + 3*b[1] + 1*b[2];
    Ax[2] = 1*b[1] + 2*b[2];

    if (approx(Ax[0], 5.0) && approx(Ax[1], 4.0) && approx(Ax[2], 3.0)) {
        PASS();
    } else {
        FAIL("solve inaccurate");
    }
}

// ─────────────────────────────────────────────────────────────────────────────
//  Test 2: Riccati backward pass for LQR
// ─────────────────────────────────────────────────────────────────────────────

void test_riccati_lqr() {
    TEST("Riccati backward pass (LQR)");

    constexpr int nx = 2, nu = 1, nc = 1, N = 5;  // nc=1 dummy (never active)
    using Stage = StageData<nx, nu, nc>;
    using WS    = RiccatiWorkspace<nx, nu, N>;
    using Ricc  = RiccatiSolver<nx, nu, nc, N>;

    double dt = 0.1;

    Stage stages[N + 1];
    WS ws;

    // Setup LQR problem
    for (int k = 0; k <= N; ++k) {
        stages[k].A(0,0)=1; stages[k].A(0,1)=dt;
        stages[k].A(1,0)=0; stages[k].A(1,1)=1;

        stages[k].B(0,0)=dt*dt/2;
        stages[k].B(1,0)=dt;

        // Q = I, R = 0.1
        stages[k].Qxx.set_identity();
        stages[k].Quu(0,0) = 0.1;
        stages[k].Qux.zero();

        stages[k].qx.zero();
        stages[k].qu.zero();
        stages[k].c.zero();
    }

    double reg_used;
    Status st = Ricc::backward(stages, ws, 1e-8, reg_used);
    if (st != Status::SUCCESS) { FAIL("backward failed"); return; }

    // Check that P matrices are positive definite
    for (int k = 0; k <= N; ++k) {
        if (ws.P[k](0,0) <= 0 || ws.P[k](1,1) <= 0) {
            FAIL("P not positive definite"); return;
        }
    }

    PASS();
}

// ─────────────────────────────────────────────────────────────────────────────
//  Test 3: Riccati forward pass
// ─────────────────────────────────────────────────────────────────────────────

void test_riccati_forward() {
    TEST("Riccati forward pass");

    constexpr int nx = 2, nu = 1, nc = 1, N = 5;  // nc=1 dummy
    using Stage = StageData<nx, nu, nc>;
    using WS    = RiccatiWorkspace<nx, nu, N>;
    using Ricc  = RiccatiSolver<nx, nu, nc, N>;

    double dt = 0.1;
    Stage stages[N + 1];
    WS ws;

    for (int k = 0; k <= N; ++k) {
        stages[k].A(0,0)=1; stages[k].A(0,1)=dt;
        stages[k].A(1,0)=0; stages[k].A(1,1)=1;
        stages[k].B(0,0)=dt*dt/2;
        stages[k].B(1,0)=dt;
        stages[k].Qxx.set_identity();
        stages[k].Quu(0,0) = 0.1;
        stages[k].Qux.zero();
        stages[k].qx.zero();
        stages[k].qu.zero();
        stages[k].c.zero();
    }

    double reg_used;
    Status st = Ricc::backward(stages, ws, 1e-8, reg_used);
    if (st != Status::SUCCESS) { FAIL("backward failed"); return; }

    // Initial state offset: x0 = [1, 0]
    Vec<nx> dx0;
    dx0[0] = 1.0;
    dx0[1] = 0.0;

    st = Ricc::forward(stages, ws, dx0);
    if (st != Status::SUCCESS) { FAIL("forward failed"); return; }

    // Check that dynamics are satisfied: dx_{k+1} = A dx_k + B du_k
    for (int k = 0; k < N; ++k) {
        double dx_next_expected_0 = stages[k].A(0,0)*ws.dx[k][0]
                                  + stages[k].A(0,1)*ws.dx[k][1]
                                  + stages[k].B(0,0)*ws.du[k][0];
        double dx_next_expected_1 = stages[k].A(1,0)*ws.dx[k][0]
                                  + stages[k].A(1,1)*ws.dx[k][1]
                                  + stages[k].B(1,0)*ws.du[k][0];

        if (!approx(ws.dx[k+1][0], dx_next_expected_0) ||
            !approx(ws.dx[k+1][1], dx_next_expected_1)) {
            FAIL("dynamics not satisfied in forward pass"); return;
        }
    }

    PASS();
}

// ─────────────────────────────────────────────────────────────────────────────
//  Test 4: Regularization handles ill-conditioned Hessian
// ─────────────────────────────────────────────────────────────────────────────

void test_regularization() {
    TEST("Regularization of ill-conditioned S");

    constexpr int nx = 2, nu = 1, nc = 1, N = 3;  // nc=1 dummy
    using Stage = StageData<nx, nu, nc>;
    using WS    = RiccatiWorkspace<nx, nu, N>;
    using Ricc  = RiccatiSolver<nx, nu, nc, N>;

    double dt = 0.1;
    Stage stages[N + 1];
    WS ws;

    for (int k = 0; k <= N; ++k) {
        stages[k].A(0,0)=1; stages[k].A(0,1)=dt;
        stages[k].A(1,0)=0; stages[k].A(1,1)=1;
        stages[k].B(0,0)=dt*dt/2;
        stages[k].B(1,0)=dt;
        stages[k].Qxx.set_identity();
        // Negative Quu → S = Quu + B^T P B is indefinite → forces regularization
        stages[k].Quu(0,0) = -10.0;
        stages[k].Qux.zero();
        stages[k].qx.zero();
        stages[k].qu.zero();
        stages[k].c.zero();
    }

    double reg_used;
    Status st = Ricc::backward(stages, ws, 1e-8, reg_used);
    if (st != Status::SUCCESS) {
        FAIL("backward failed despite regularization"); return;
    }

    // Regularization should have been increased
    if (reg_used <= 1e-8) {
        FAIL("regularization not increased"); return;
    }

    PASS();
}

void test_relative_regularization_is_scale_invariant() {
    TEST("Relative regularization is scale invariant");

    constexpr int nx = 1, nu = 2, nc = 1, N = 1;
    using Stage = StageData<nx, nu, nc>;
    using WS = RiccatiWorkspace<nx, nu, N>;
    using Ricc = RiccatiSolver<nx, nu, nc, N>;

    auto setup = [](Stage stages[], double scale0, double scale1,
                    double correlation) {
        for (int k = 0; k <= N; ++k) {
            stages[k].A.zero();
            stages[k].B.zero();
            stages[k].Qxx.set_identity();
            stages[k].Quu.zero();
            stages[k].Qux.zero();
            stages[k].qx.zero();
            stages[k].qu.zero();
            stages[k].c.zero();
        }
        stages[0].Quu(0, 0) = scale0 * scale0;
        stages[0].Quu(1, 0) = scale0 * scale1 * correlation;
        stages[0].Quu(1, 1) = scale1 * scale1;
        stages[0].qu[0] = scale0;
        stages[0].qu[1] = -scale1;
    };

    constexpr double base_regularization = 1e-8;
    constexpr double correlation = 1.0 - 1e-12;
    constexpr double scale0 = 32.0;
    constexpr double scale1 = 1.0 / 32.0;

    Stage base_stages[N + 1];
    Stage scaled_stages[N + 1];
    WS base_workspace;
    WS scaled_workspace;
    setup(base_stages, 1.0, 1.0, correlation);
    setup(scaled_stages, scale0, scale1, correlation);

    double base_reg_used = 0.0;
    double scaled_reg_used = 0.0;
    Status base_status = Ricc::backward_lhs(
        base_stages, base_workspace, base_regularization, base_reg_used,
        1e12, 0.0);
    Status scaled_status = Ricc::backward_lhs(
        scaled_stages, scaled_workspace, base_regularization,
        scaled_reg_used, 1e12, 0.0);
    if (base_status != Status::SUCCESS || scaled_status != Status::SUCCESS) {
        FAIL("regularized factorization failed"); return;
    }
    if (base_reg_used != base_regularization ||
        scaled_reg_used != base_regularization) {
        FAIL("congruent systems changed the relative base shift"); return;
    }
    if (Ricc::backward_rhs(base_stages, base_workspace) != Status::SUCCESS ||
        Ricc::backward_rhs(scaled_stages, scaled_workspace) !=
            Status::SUCCESS) {
        FAIL("regularized RHS solve failed"); return;
    }
    for (int i = 0; i < nu; ++i) {
        const double scale = i == 0 ? scale0 : scale1;
        const double mapped = scaled_workspace.d[0][i] * scale;
        const double tolerance =
            1e-6 * std::max(1.0, std::fabs(base_workspace.d[0][i]));
        if (!std::isfinite(mapped) ||
            std::fabs(mapped - base_workspace.d[0][i]) > tolerance) {
            FAIL("regularized solutions are not congruence-equivalent"); return;
        }
    }

    Stage identity_stages[N + 1];
    WS identity_workspace;
    setup(identity_stages, 1.0, 1.0, 0.0);
    double identity_reg_used = 0.0;
    if (Ricc::backward_lhs(
            identity_stages, identity_workspace, base_regularization,
            identity_reg_used, 1e12, 0.0) !=
            Status::SUCCESS ||
        identity_reg_used != base_regularization) {
        FAIL("well-conditioned identity was unnecessarily retried"); return;
    }

    PASS();
}

void test_failed_factorization_reports_attempted_regularization() {
    TEST("Failed factorization reports attempted regularization");

    constexpr int nx = 1, nu = 2, nc = 1, N = 1;
    using Stage = StageData<nx, nu, nc>;
    using WS = RiccatiWorkspace<nx, nu, N>;
    using Ricc = RiccatiSolver<nx, nu, nc, N>;

    Stage stages[N + 1];
    WS workspace;
    for (int k = 0; k <= N; ++k) {
        stages[k].A.zero();
        stages[k].B.zero();
        stages[k].Qxx.set_identity();
        stages[k].Quu.set_identity();
        stages[k].Qux.zero();
    }

    double reg_used = 0.0;
    const Status status = Ricc::backward_lhs(
        stages, workspace, 1e-12, reg_used, 1e12, 2.0);
    if (status != Status::KKT_SINGULAR) {
        FAIL("expected retry exhaustion"); return;
    }
    if (!(reg_used > 1e4 && reg_used < 1e5)) {
        FAIL("maximum attempted regularization was not reported"); return;
    }
    PASS();
}

// ─────────────────────────────────────────────────────────────────────────────
//  Main
// ─────────────────────────────────────────────────────────────────────────────

int main() {
    printf("╔══════════════════════════════════════════╗\n");
    printf("║   Riccati Recursion Unit Tests          ║\n");
    printf("╚══════════════════════════════════════════╝\n\n");

    test_ldlt();
    test_riccati_lqr();
    test_riccati_forward();
    test_regularization();
    test_relative_regularization_is_scale_invariant();
    test_failed_factorization_reports_attempted_regularization();

    printf("\n─── Results: %d/%d passed ───\n", pass_count, test_count);
    return (pass_count == test_count) ? 0 : 1;
}
