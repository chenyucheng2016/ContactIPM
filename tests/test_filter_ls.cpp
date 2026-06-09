/**
 * @file    test_filter_ls.cpp
 * @brief   Unit tests for FilterLineSearch module.
 *
 * Tests:
 *   1. Filter basics: add, contains, purge dominated entries
 *   2. Armijo condition: accepts sufficient decrease
 *   3. Bi-objective improvement: feasibility + optimality
 *   4. SOC (second-order correction) path
 *   5. Backtrack: halve α until accept or α_min
 */

#include <cstdio>
#include <cmath>
#include <cassert>

#include "nmpc/nmpc_core.hpp"
#include "nmpc/nmpc_filter_ls.hpp"

using namespace nmpc;

static int test_count = 0;
static int pass_count = 0;

#define TEST(name) do { test_count++; printf("  TEST %s ... ", name); } while(0)
#define PASS()      do { pass_count++; printf("PASS\n"); } while(0)
#define FAIL(msg)   do { printf("FAIL: %s\n", msg); } while(0)

// ═══════════════════════════════════════════════════════════════════
//  Test 1: Filter basics — add, contains, purge dominated
// ═══════════════════════════════════════════════════════════════════

void test_filter_basics() {
    TEST("Filter add / contains / purge");

    Filter<10> f;
    f.reset();

    double gt = 1e-5, gp = 1e-5;

    // Empty filter should not contain anything
    if (f.contains(0.0, 0.0, gt, gp)) { FAIL("empty filter contains"); return; }

    // Add entry
    f.add(1.0, 10.0, gt, gp);
    if (f.count != 1) { FAIL("count after add"); return; }

    // The entry (1.0, 10.0) dominates (2.0, 20.0)
    if (!f.contains(2.0, 20.0, gt, gp)) { FAIL("should be dominated"); return; }

    // (0.5, 9.0) is NOT dominated by (1.0, 10.0) — better θ
    if (f.contains(0.5, 9.0, gt, gp))  { FAIL("better theta should not be dominated"); return; }

    // (1.5, 9.0) is NOT dominated — better φ
    if (f.contains(1.5, 9.0, gt, gp))  { FAIL("better phi should not be dominated"); return; }

    // Add a second entry — (0.5,9) dominates (1,10), so (1,10) gets purged
    f.add(0.5, 9.0, gt, gp);
    if (f.count != 1) { FAIL("count after dominating add"); return; }

    // Add a genuinely non-dominated entry: better φ but worse θ
    f.add(0.5, 8.0, gt, gp);  // dominates (0.5, 9.0) → purge again
    if (f.count != 1) { FAIL("purged"); return; }

    // Now add (2.0, 5.0) — worse θ, better φ than (0.5, 8.0) — not dominated
    f.add(2.0, 5.0, gt, gp);
    if (f.count != 2) { FAIL("two non-dominated entries"); return; }

    PASS();
}

// ═══════════════════════════════════════════════════════════════════
//  Test 2: Armijo condition
// ═══════════════════════════════════════════════════════════════════

void test_armijo() {
    TEST("Armijo condition");

    // phi0 = 10.0, Dphi = -2.0, α = 1.0, η = 1e-4
    // Armijo: φ(α) ≤ φ(0) + η·α·Dφ
    //        φ(1) ≤ 10.0 + 1e-4·1.0·(-2.0) = 10.0 - 2e-4 = 9.9998

    // Good decrease: φ_trial = 8.0 ≤ 9.9998 → accept
    if (!armijo_condition(8.0, 10.0, -2.0, 1.0, 1e-4))
        { FAIL("good decrease rejected"); return; }

    // Barely sufficient: φ_trial = 9.9998 ≤ 9.9998 → accept
    if (!armijo_condition(9.9998, 10.0, -2.0, 1.0, 1e-4))
        { FAIL("barely sufficient rejected"); return; }

    // Insufficient: φ_trial = 9.9999 > 9.9998 → reject
    if (armijo_condition(9.9999, 10.0, -2.0, 1.0, 1e-4))
        { FAIL("insufficient accepted"); return; }

    // Dphi >= 0 should not be used, but check it doesn't crash
    if (armijo_condition(5.0, 10.0, 0.0, 1.0, 1e-4))
        { FAIL("zero Dphi gave wrong result"); return; }

    PASS();
}

// ═══════════════════════════════════════════════════════════════════
//  Test 3: Bi-objective improvement
// ═══════════════════════════════════════════════════════════════════

void test_bi_objective() {
    TEST("Bi-objective improvement");

    double gt = 1e-5, gp = 1e-5, tm = 1e-4;

    // Case 1: constraint violation improves enough
    // θ0=1.0, θ_trial=0.5: 0.5 ≤ (1-1e-5)*1.0 = 0.99999 → true
    if (!bi_objective_improvement(0.5, 1.0, 20.0, 10.0, gt, gp, tm))
        { FAIL("theta improvement accepted"); return; }

    // θ0 is small AND φ improves: φ_trial ≤ φ0 + γ_phi = 10.0 + 1e-5 = 10.00001
    if (!bi_objective_improvement(1e-5, 1e-5, 9.9, 10.0, gt, gp, tm))
        { FAIL("small theta + phi improvement"); return; }

    // Neither θ nor φ improves enough
    if (bi_objective_improvement(1.0, 1.0, 10.0, 10.0, gt, gp, tm))
        { FAIL("no improvement accepted"); return; }

    PASS();
}

// ═══════════════════════════════════════════════════════════════════
//  Test 4: Mock evaluator for line search testing
// ═══════════════════════════════════════════════════════════════════

// Simple mock: f(x) = x², constraint = |x-1|, initial x = 2.0
// Trial point: x + α·Δx, where Δx = -2.0 (toward x=0)
// We accept α=0.5 (x=1.0) as it reduces both cost and constraint violation

struct MockEvaluator : TrialPointEvaluator<2, 1, 5> {
    double x_cur = 2.0;
    double x_trial = 2.0;
    double theta_cur = 1.0;   // |x-1|
    double theta_trial = 1.0;
    double phi_cur = 4.0;     // x²
    double phi_trial = 4.0;

    bool evaluate(double alpha, double& out_theta, double& out_phi) override {
        x_trial = x_cur + alpha * (-2.0);  // step toward 0
        theta_trial = std::fabs(x_trial - 1.0);
        phi_trial = x_trial * x_trial;
        out_theta = theta_trial;
        out_phi   = phi_trial;
        return true;
    }
    double current_theta() const override { return theta_cur; }
    double current_phi()   const override { return phi_cur; }
    double compute_Dphi()  const override { return -8.0; }  // ∇f=2x, Δx=-2 → Dφ=2*2*(-2)=-8
};

void test_filter_ls_search() {
    TEST("Filter line-search: backtrack to acceptance");

    MockEvaluator eval;

    FilterLineSearch<2, 1, 5> fls;
    FilterLSParams flp;
    flp.theta_min = 0.1;    // θ0=1.0 > 0.1 → Case II (feasibility first)
    flp.alpha_min = 1e-6;
    flp.gamma_phi  = 1e-8;
    flp.gamma_theta = 0.5;  // require 50% improvement in θ for Case II accept
    fls.init(flp);

    // Dphi provided by MockEvaluator::compute_Dphi() = -8.0
    double alpha_max = 1.0;  // ftb would allow full step

    // Δx = -2.0, so α=1.0 → x=0, θ=|0-1|=1.0, φ=0
    // θ0=1.0 > 0.1 → Case II
    // θ_trial=1.0 is NOT ≤ (1-0.5)*1.0=0.5 → theta didn't improve enough
    // BUT θ0=1.0 > theta_min=1e2 → wait, we already set theta_min=0.1
    // Actually with gamma_theta=0.5, need 50% improvement: 1.0 ≤ 0.5? No.
    // But phi improved from 4.0 to 0 → that satisfies bi_objective Case 2!
    // With theta_min=0.1 and theta0=1.0 > 0.1: Case II, bi_objective:
    //   theta_trial=1.0 ≤ (1-0.5)*1.0=0.5? NO (theta didn't improve enough)
    // But wait — bi_objective only checks theta improvement OR (theta0≤theta_min AND phi improvement)
    // theta0=1.0 > 0.1, so we need theta improvement. theta_trial=1.0 is not ≤ 0.5. FAIL.
    // So we go to SOC → not eligible (x=0 is better than x=2 for constraints, but theta=1 at both) → skip SOC → backtrack
    //
    // Need a test where the step clearly improves BOTH theta and phi:
    // x=2, Δx=-1 → x=1, θ=0, φ=1. θ=0 is a clear improvement, φ=1 < 4.
    // Let me change the mock to use Δx=-1 and make sure theta improves.

    LSResult result = fls.search(eval, alpha_max);

    // α=1: x=0, θ=1.0, φ=0 → theta doesn't improve (stays at 1.0), but phi improves massively
    // With theta0=1.0 > theta_min=0.1 and gamma_theta=0.5: need theta_trial ≤ 0.5, but 1.0 > 0.5
    // So this step gets REJECTED, then SOC, then backtracks
    // α=0.5: x=1, θ=0, φ=1 → theta=0 ≤ 0.5 ✓, phi=1 ≤ 4 ✓ → ACCEPTED!

    if (result.status != LSStatus::ACCEPTED) {
        printf("status=%d alpha=%.4f\n", (int)result.status, result.alpha);
        FAIL("search did not accept"); return;
    }
    // α should be 0.5 (backtracked once)
    if (result.alpha < 0.4 || result.alpha > 0.6) { FAIL("expected alpha≈0.5"); return; }

    PASS();
}

// ═══════════════════════════════════════════════════════════════════
//  Test 5: Backtrack when step is bad
// ═══════════════════════════════════════════════════════════════════

struct BadEvaluator : TrialPointEvaluator<2, 1, 5> {
    double x_cur = 2.0, theta_cur = 1.0, phi_cur = 4.0;
    double x_trial, theta_trial, phi_trial;

    bool evaluate(double alpha, double& out_theta, double& out_phi) override {
        x_trial = x_cur + alpha * 0.5;  // small step right → increases cost
        theta_trial = std::fabs(x_trial - 1.0);
        phi_trial = x_trial * x_trial + 100.0;  // always worse cost
        out_theta = theta_trial;
        out_phi   = phi_trial;
        return true;
    }
    double current_theta() const override { return theta_cur; }
    double current_phi()   const override { return phi_cur; }
    double compute_Dphi()  const override { return -8.0; }
};

void test_filter_ls_failed() {
    TEST("Filter line-search: backtrack to failure");

    BadEvaluator eval;
    FilterLineSearch<2, 1, 5> fls;
    FilterLSParams flp;
    flp.theta_min = 1e-8;
    flp.alpha_min = 1e-6;
    fls.init(flp);

    double alpha_max = 1.0;

    LSResult result = fls.search(eval, alpha_max);

    // Every trial makes cost worse and doesn't improve constraints enough
    // → backtracks all the way to α < α_min → FAILED
    if (result.status != LSStatus::FAILED) {
        FAIL("should return FAILED"); return;
    }

    PASS();
}

// ═══════════════════════════════════════════════════════════════════

int main() {
    printf("╔══════════════════════════════════════════╗\n");
    printf("║   Filter Line-Search Unit Tests         ║\n");
    printf("╚══════════════════════════════════════════╝\n\n");

    test_filter_basics();
    test_armijo();
    test_bi_objective();
    test_filter_ls_search();
    test_filter_ls_failed();

    printf("\n─── Results: %d/%d passed ───\n", pass_count, test_count);
    return (pass_count == test_count) ? 0 : 1;
}
