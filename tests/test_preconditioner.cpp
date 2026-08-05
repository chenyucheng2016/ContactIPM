/**
 * @file    test_preconditioner.cpp
 * @brief   Unit tests for HessianPreconditioner:
 *            1. Cholesky factorization correctness
 *            2. Triangular inverse correctness
 *            3. Congruence transform: L^{-1}·H·L^{-T} ≈ I
 *            4. Condition number reduction
 *            5. Diagonal Hessian (Jacobi equivalence)
 *            6. Non-diagonal Hessian
 *            7. Scale/unscale round-trip identity
 *            8. Sandwich product correctness
 *            9. Condition estimate
 */

#include <cstdio>
#include <cmath>
#include <cassert>

#include "nmpc/nmpc_core.hpp"
#include "nmpc/nmpc_problem.hpp"
#include "nmpc/nmpc_preconditioner.hpp"

using namespace nmpc;

static int test_count = 0;
static int pass_count = 0;

#define TEST(name) do { test_count++; printf("  TEST %s ... ", name); } while(0)
#define PASS()      do { pass_count++; printf("PASS\n"); } while(0)
#define FAIL(msg)   do { printf("FAIL: %s\n", msg); } while(0)

constexpr int NX = 2;
constexpr int NU = 1;
constexpr int NC = 1;
constexpr int HORIZON = 3;

using Stage = StageData<NX, NU, NC>;
using Prec  = HessianPreconditioner<NX, NU, HORIZON>;

// ═══════════════════════════════════════════════════════════════════════════
//  Helpers
// ═══════════════════════════════════════════════════════════════════════════

void fill_stages_diagonal(Stage stages[], double a, double b, double c) {
    for (int k = 0; k <= HORIZON; ++k) {
        stages[k].Qxx.zero();
        stages[k].Qxx(0, 0) = a;
        stages[k].Qxx(1, 1) = b;
        if (k < HORIZON) {
            stages[k].Quu.zero();
            stages[k].Quu(0, 0) = c;
            stages[k].Qux.zero();
            stages[k].A.set_identity();
            stages[k].B.zero(); stages[k].B(0, 0) = 0.1;
            stages[k].c.zero();
            stages[k].qx.zero(); stages[k].qu.zero();
        }
    }
    stages[HORIZON].qx.zero();
}

template <int R, int C>
bool mat_approx(const Mat<R, C>& A, const Mat<R, C>& B, double tol = 1e-10) {
    for (int r = 0; r < R; ++r)
        for (int c = 0; c < C; ++c)
            if (std::fabs(A(r, c) - B(r, c)) > tol) return false;
    return true;
}

// ═══════════════════════════════════════════════════════════════════════════
//  Test 1: Cholesky — L · L^T = M
// ═══════════════════════════════════════════════════════════════════════════

void test_cholesky_correctness() {
    TEST("Cholesky: L * L^T = M");

    Stage stages[HORIZON + 1];
    fill_stages_diagonal(stages, 100.0, 0.01, 1.0);

    Prec prec;
    prec.compute(stages);

    Mat<NX, NX> L = prec.Lx(0);
    Mat<NX, NX> LLT;
    LLT.zero();
    for (int r = 0; r < NX; ++r)
        for (int c = 0; c < NX; ++c)
            for (int m = 0; m < NX; ++m)
                LLT(r, c) += L(r, m) * L(c, m);

    bool ok = (std::fabs(LLT(0, 0) - 100.0) < 1e-10 &&
               std::fabs(LLT(1, 1) - 0.01) < 1e-10 &&
               std::fabs(LLT(0, 1)) < 1e-10);

    if (!ok) {
        printf("L*L^T = [%.6e, %.6e; %.6e, %.6e] ",
               LLT(0,0), LLT(0,1), LLT(1,0), LLT(1,1));
        FAIL("L*L^T != M"); return;
    }

    if (std::fabs(prec.Lu(0)(0, 0) - 1.0) > 1e-10) {
        FAIL("Lu != 1.0"); return;
    }
    PASS();
}

// ═══════════════════════════════════════════════════════════════════════════
//  Test 2: Triangular inverse — L · L^{-1} = I
// ═══════════════════════════════════════════════════════════════════════════

void test_triangular_inverse() {
    TEST("Triangular inverse: L * L^{-1} = I");

    Stage stages[HORIZON + 1];
    fill_stages_diagonal(stages, 100.0, 0.01, 4.0);

    Prec prec;
    prec.compute(stages);

    Mat<NX, NX> product;
    product.zero();
    for (int r = 0; r < NX; ++r)
        for (int c = 0; c < NX; ++c)
            for (int m = 0; m < NX; ++m)
                product(r, c) += prec.Lx(0)(r, m) * prec.inv_Lx(0)(m, c);

    Mat<NX, NX> eye;
    eye.set_identity();
    if (!mat_approx(product, eye, 1e-10)) {
        FAIL("L * inv_L != I"); return;
    }

    double prod_u = prec.Lu(0)(0, 0) * prec.inv_Lu(0)(0, 0);
    if (std::fabs(prod_u - 1.0) > 1e-10) {
        FAIL("Lu * inv_Lu != 1"); return;
    }
    PASS();
}

// ═══════════════════════════════════════════════════════════════════════════
//  Test 3: Congruence — L^{-1}·H·L^{-T} ≈ I
// ═══════════════════════════════════════════════════════════════════════════

void test_congruence_identity() {
    TEST("Congruence: L^{-1}*H*L^{-T} = I");

    Stage stages[HORIZON + 1];
    fill_stages_diagonal(stages, 200.0, 0.01, 0.001);

    Prec prec;
    prec.compute(stages);

    // Apply congruence to a copy of Qxx
    Mat<NX, NX> scaled_Qxx = stages[0].Qxx;
    prec.sandwich_symmetric(scaled_Qxx, prec.inv_Lx(0));

    Mat<NX, NX> eye;
    eye.set_identity();
    if (!mat_approx(scaled_Qxx, eye, 1e-8)) {
        printf("scaled Qxx = [%.6e,%.6e; %.6e,%.6e] ",
               scaled_Qxx(0,0), scaled_Qxx(0,1),
               scaled_Qxx(1,0), scaled_Qxx(1,1));
        FAIL("congruence != I"); return;
    }

    // Quu congruence
    Mat<NU, NU> scaled_Quu = stages[0].Quu;
    prec.sandwich_symmetric(scaled_Quu, prec.inv_Lu(0));
    if (std::fabs(scaled_Quu(0, 0) - 1.0) > 1e-8) {
        FAIL("Quu congruence != 1"); return;
    }
    PASS();
}

// helper for condition computation
double prec_condition(const Stage stages[]) {
    double mx = 0.0, mn = 1e100;
    for (int k = 0; k <= HORIZON; ++k) {
        for (int i = 0; i < NX; ++i) {
            double d = stages[k].Qxx(i, i);
            if (d > mx) mx = d;
            if (d > 1e-14 && d < mn) mn = d;
        }
        if (k < HORIZON) {
            for (int i = 0; i < NU; ++i) {
                double d = stages[k].Quu(i, i);
                if (d > mx) mx = d;
                if (d > 1e-14 && d < mn) mn = d;
            }
        }
    }
    return mx / std::max(mn, 1e-14);
}

// ═══════════════════════════════════════════════════════════════════════════
//  Test 4: Condition number reduction
// ═══════════════════════════════════════════════════════════════════════════

void test_condition_reduction() {
    TEST("Condition number reduction");

    Stage stages[HORIZON + 1];
    fill_stages_diagonal(stages, 10000.0, 0.001, 0.0001);

    double cond_orig = prec_condition(stages);

    Prec prec;
    prec.compute(stages);

    // After congruence, each block is ~I, so condition ≈ 1
    Mat<NX, NX> Q_scaled = stages[0].Qxx;
    prec.sandwich_symmetric(Q_scaled, prec.inv_Lx(0));
    double max_d = std::max(std::fabs(Q_scaled(0,0)), std::fabs(Q_scaled(1,1)));
    double min_d = std::min(std::fabs(Q_scaled(0,0)), std::fabs(Q_scaled(1,1)));
    double cond_scaled = max_d / std::max(min_d, 1e-14);

    printf("(orig=%.1e -> scaled=%.1e) ", cond_orig, cond_scaled);

    if (cond_orig < 1e6) { FAIL("original cond too small"); return; }
    if (cond_scaled > 2.0) { FAIL("scaled cond not ~1"); return; }
    PASS();
}

// ═══════════════════════════════════════════════════════════════════════════
//  Test 5: Diagonal Hessian — Cholesky = Jacobi
// ═══════════════════════════════════════════════════════════════════════════

void test_diagonal_jacobi() {
    TEST("Diagonal Hessian: Cholesky = Jacobi");

    Stage stages[HORIZON + 1];
    fill_stages_diagonal(stages, 25.0, 9.0, 4.0);

    Prec prec;
    prec.compute(stages);

    bool lx_ok = (std::fabs(prec.Lx(0)(0, 0) - 5.0) < 1e-10 &&
                   std::fabs(prec.Lx(0)(1, 1) - 3.0) < 1e-10);
    bool lu_ok = (std::fabs(prec.Lu(0)(0, 0) - 2.0) < 1e-10);
    bool inv_ok = (std::fabs(prec.inv_Lx(0)(0, 0) - 0.2) < 1e-10 &&
                    std::fabs(prec.inv_Lx(0)(1, 1) - 1.0/3.0) < 1e-10);

    if (!lx_ok) { FAIL("Lx != diag(5,3)"); return; }
    if (!lu_ok) { FAIL("Lu != 2"); return; }
    if (!inv_ok) { FAIL("inv_Lx wrong"); return; }
    PASS();
}

// ═══════════════════════════════════════════════════════════════════════════
//  Test 6: Non-diagonal Hessian
// ═══════════════════════════════════════════════════════════════════════════

void test_nondiagonal_hessian() {
    TEST("Non-diagonal Hessian");

    Stage stages[HORIZON + 1];
    for (int k = 0; k <= HORIZON; ++k) {
        stages[k].Qxx.zero();
        stages[k].Qxx(0, 0) = 4.0; stages[k].Qxx(0, 1) = 2.0;
        stages[k].Qxx(1, 0) = 2.0; stages[k].Qxx(1, 1) = 3.0;
        if (k < HORIZON) {
            stages[k].Quu.zero(); stages[k].Quu(0, 0) = 1.0;
            stages[k].Qux.zero();
            stages[k].A.set_identity();
            stages[k].B.zero(); stages[k].B(0, 0) = 0.1;
            stages[k].c.zero();
            stages[k].qx.zero(); stages[k].qu.zero();
        }
    }
    stages[HORIZON].qx.zero();

    Prec prec;
    prec.compute(stages);

    // Verify L·L^T = Qxx
    Mat<NX, NX> L = prec.Lx(0);
    Mat<NX, NX> LLT;
    LLT.zero();
    for (int r = 0; r < NX; ++r)
        for (int c = 0; c < NX; ++c)
            for (int m = 0; m < NX; ++m)
                LLT(r, c) += L(r, m) * L(c, m);

    Mat<NX, NX> Qxx_orig;
    Qxx_orig.zero();
    Qxx_orig(0, 0) = 4.0; Qxx_orig(0, 1) = 2.0;
    Qxx_orig(1, 0) = 2.0; Qxx_orig(1, 1) = 3.0;

    if (!mat_approx(LLT, Qxx_orig, 1e-10)) {
        FAIL("L*L^T != Qxx"); return;
    }

    // Congruence should give ~I
    Mat<NX, NX> scaled = Qxx_orig;
    prec.sandwich_symmetric(scaled, prec.inv_Lx(0));
    Mat<NX, NX> eye;
    eye.set_identity();
    if (!mat_approx(scaled, eye, 1e-8)) {
        printf("scaled = [%.6e,%.6e; %.6e,%.6e] ",
               scaled(0,0), scaled(0,1), scaled(1,0), scaled(1,1));
        FAIL("congruence != I"); return;
    }
    PASS();
}

// ═══════════════════════════════════════════════════════════════════════════
//  Test 7: Scale/unscale round-trip identity
//  scale then unscale should recover the original vector.
// ═══════════════════════════════════════════════════════════════════════════

void test_scale_unscale_roundtrip() {
    TEST("Scale/unscale round-trip");

    Stage stages[HORIZON + 1];
    fill_stages_diagonal(stages, 100.0, 0.01, 4.0);

    Prec prec;
    prec.compute(stages);

    // Test x scaling: scale_x then unscale_x should give identity
    Vec<NX> v_orig; v_orig[0] = 3.7; v_orig[1] = -2.1;
    Vec<NX> v = v_orig;
    prec.scale_x(0, v);      // v ← Lx^{-1} · v
    prec.unscale_x(0, v);    // v ← Lx · v
    if (std::fabs(v[0] - v_orig[0]) > 1e-12 ||
        std::fabs(v[1] - v_orig[1]) > 1e-12) {
        FAIL("x scale/unscale not identity"); return;
    }

    // Test u scaling
    Vec<NU> u_orig; u_orig[0] = -5.3;
    Vec<NU> u = u_orig;
    prec.scale_u(0, u);
    prec.unscale_u(0, u);
    if (std::fabs(u[0] - u_orig[0]) > 1e-12) {
        FAIL("u scale/unscale not identity"); return;
    }

    // Test invT scaling: scale_x_invT then unscale_x_T
    v = v_orig;
    prec.scale_x_invT(0, v);   // v ← Lx^{-T} · v
    prec.unscale_x_T(0, v);    // v ← Lx^T · v
    if (std::fabs(v[0] - v_orig[0]) > 1e-12 ||
        std::fabs(v[1] - v_orig[1]) > 1e-12) {
        FAIL("invT scale/unscale not identity"); return;
    }

    PASS();
}

// ═══════════════════════════════════════════════════════════════════════════
//  Test 8: Sandwich product correctness
//  Verify sandwich_symmetric computes T·M·T^T correctly.
// ═══════════════════════════════════════════════════════════════════════════

void test_sandwich_product() {
    TEST("Sandwich product correctness");

    Prec prec;  // don't need compute, just use the method

    // M = [4, 2; 2, 3], T = [2, 0; 0, 3]
    Mat<NX, NX> M;
    M(0, 0) = 4.0; M(0, 1) = 2.0;
    M(1, 0) = 2.0; M(1, 1) = 3.0;

    Mat<NX, NX> T;
    T.zero();
    T(0, 0) = 2.0; T(1, 1) = 3.0;

    // Expected: T·M·T^T = [2,0;0,3]·[4,2;2,3]·[2,0;0,3]
    //   = [8,4;6,9]·[2,0;0,3] = [16,12;12,27]
    prec.sandwich_symmetric(M, T);

    if (std::fabs(M(0, 0) - 16.0) > 1e-10 ||
        std::fabs(M(0, 1) - 12.0) > 1e-10 ||
        std::fabs(M(1, 0) - 12.0) > 1e-10 ||
        std::fabs(M(1, 1) - 27.0) > 1e-10) {
        printf("result = [%.1f,%.1f; %.1f,%.1f], expected [16,12;12,27] ",
               M(0,0), M(0,1), M(1,0), M(1,1));
        FAIL("sandwich wrong"); return;
    }
    PASS();
}

// ═══════════════════════════════════════════════════════════════════════════
//  Test 9: Condition estimate
// ═══════════════════════════════════════════════════════════════════════════

void test_condition_estimate() {
    TEST("Condition estimate");

    Stage stages[HORIZON + 1];
    fill_stages_diagonal(stages, 100.0, 0.01, 1.0);

    Prec prec;
    prec.compute(stages);

    double cond = prec.condition_estimate(stages);
    // max diag = 100, min diag = 0.01, cond = 10000
    if (std::fabs(cond - 10000.0) > 1.0) {
        printf("cond = %.2e, expected 1e4 ", cond);
        FAIL("condition estimate wrong"); return;
    }
    PASS();
}

// ═══════════════════════════════════════════════════════════════════════════

int main() {
    printf("=== Hessian Preconditioner Unit Tests ===\n\n");

    test_cholesky_correctness();
    test_triangular_inverse();
    test_congruence_identity();
    test_condition_reduction();
    test_diagonal_jacobi();
    test_nondiagonal_hessian();
    test_scale_unscale_roundtrip();
    test_sandwich_product();
    test_condition_estimate();

    printf("\n--- Results: %d/%d passed ---\n", pass_count, test_count);
    return (pass_count == test_count) ? 0 : 1;
}
