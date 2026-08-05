/**
 * @file    test_kkt_diag.cpp
 * @brief   Unit tests for KKT diagnostics module.
 */

#include <cstdio>
#include <cmath>
#include <cassert>

#include "nmpc/nmpc_core.hpp"
#include "nmpc/nmpc_kkt_diag.hpp"

using namespace nmpc;

static int test_count = 0;
static int pass_count = 0;

#define TEST(name) do { test_count++; printf("  TEST %s ... ", name); } while(0)
#define PASS()      do { pass_count++; printf("PASS\n"); } while(0)
#define FAIL(msg)   do { printf("FAIL: %s\n", msg); } while(0)

// ═══════════════════════════════════════════════════════════════════
//  Test 1: Condition estimate on well-conditioned SPD matrix
// ═══════════════════════════════════════════════════════════════════

void test_cond_estimate_well() {
    TEST("Cond estimate: well-conditioned SPD");

    // A = [4 1; 1 3] — condition number ≈ 1.3 (eigenvalues ~4.6, ~2.4)
    SymMat<2> A;
    A.zero();
    A(0,0) = 4.0; A(1,0) = 1.0; A(1,1) = 3.0;

    double cond = cond_estimate_diag<2>(A);
    // Diag ratio: 4/3 ≈ 1.33. True cond ≈ 1.67. Diag estimate is within 2x.
    if (cond < 1.0 || cond > 3.0) { FAIL("cond estimate wrong"); return; }

    PASS();
}

// ═══════════════════════════════════════════════════════════════════
//  Test 2: Condition estimate on ill-conditioned matrix
// ═══════════════════════════════════════════════════════════════════

void test_cond_estimate_ill() {
    TEST("Cond estimate: ill-conditioned");

    SymMat<2> A;
    A.zero();
    A(0,0) = 1e8; A(1,0) = 0.0; A(1,1) = 1e-2;

    double cond = cond_estimate_diag<2>(A);
    if (cond < 1e9 || cond > 1e11) { FAIL("cond estimate wrong for ill"); return; }

    PASS();
}

// ═══════════════════════════════════════════════════════════════════
//  Test 3: Inertia of an SPD matrix (after LDL^T)
// ═══════════════════════════════════════════════════════════════════

void test_inertia_spd() {
    TEST("Inertia: SPD matrix (all positive pivots)");

    SymMat<3> A;
    A.zero();
    A(0,0)=4; A(1,0)=1; A(1,1)=3; A(2,1)=1; A(2,2)=2;

    bool ok = A.ldlt_factorize();
    if (!ok) { FAIL("LDL^T factorization failed"); return; }

    KKTDiag d = diagnose_kkt<3>(A);
    if (!d.is_psd()) { FAIL("should be PSD"); return; }
    if (!d.is_full_rank()) { FAIL("should be full rank"); return; }
    if (d.pos_pivots != 3) { FAIL("all pivots should be positive"); return; }

    PASS();
}

// ═══════════════════════════════════════════════════════════════════
//  Test 4: Inertia of indefinite matrix
// ═══════════════════════════════════════════════════════════════════

void test_inertia_indefinite() {
    TEST("Inertia: indefinite matrix");

    SymMat<2> A;
    A.zero();
    A(0,0) = 1.0; A(1,0) = 0.0; A(1,1) = -2.0;

    // LDL^T on indefinite: should return false (negative pivot detected)
    bool ok = A.ldlt_factorize();
    if (ok) { FAIL("LDL^T should fail on indefinite"); return; }

    PASS();
}

// ═══════════════════════════════════════════════════════════════════
//  Test 5: Rank estimate of Jacobian
// ═══════════════════════════════════════════════════════════════════

void test_rank_estimate() {
    TEST("Rank estimate of Jacobian");

    // Full rank: J = [1 2; 3 4] → rank 2
    Mat<2, 2> J;
    J(0,0)=1; J(0,1)=2; J(1,0)=3; J(1,1)=4;

    int r = estimate_rank<2,2>(J, 1e-6);
    if (r != 2) { FAIL("should be rank 2"); return; }

    // Rank deficient: J = [1 0; 0 0] → column norms: col0=1, col1=0 → rank 1
    Mat<2, 2> J2;
    J2(0,0)=1; J2(0,1)=0; J2(1,0)=0; J2(1,1)=0;
    r = estimate_rank<2,2>(J2, 1e-6);
    if (r != 1) { FAIL("should be rank 1"); return; }

    PASS();
}

// ═══════════════════════════════════════════════════════════════════

int main() {
    printf("╔══════════════════════════════════════════╗\n");
    printf("║   KKT Diagnostics Unit Tests            ║\n");
    printf("╚══════════════════════════════════════════╝\n\n");

    test_cond_estimate_well();
    test_cond_estimate_ill();
    test_inertia_spd();
    test_inertia_indefinite();
    test_rank_estimate();

    printf("\n─── Results: %d/%d passed ───\n", pass_count, test_count);
    return (pass_count == test_count) ? 0 : 1;
}
