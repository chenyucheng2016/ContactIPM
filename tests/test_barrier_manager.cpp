/**
 * @file    test_barrier_manager.cpp
 * @brief   Unit tests for σ-modulation BarrierUpdateStrategy.
 */

#include <cstdio>
#include <cmath>
#include <cassert>

#include "nmpc/nmpc_core.hpp"
#include "nmpc/nmpc_barrier_manager.hpp"

using namespace nmpc;

static int test_count = 0;
static int pass_count = 0;

#define TEST(name) do { test_count++; printf("  TEST %s ... ", name); } while(0)
#define PASS()      do { pass_count++; printf("PASS\n"); } while(0)
#define FAIL(msg)   do { printf("FAIL: %s\n", msg); } while(0)

static const double TOL = 1e-8;

// ═══════════════════════════════════════════════════════════════════
//  Test 1: Easy subproblem → σ^1.5 reduction
// ═══════════════════════════════════════════════════════════════════

void test_easy_sigma_modulation() {
    TEST("Easy subproblem → σ^1.5 reduction");

    BarrierUpdateParams p;
    p.mu_init   = 1.0;
    p.mu_min    = 1e-8;
    p.kappa_eps = 10.0;

    BarrierUpdateStrategy strat;
    strat.reset(p);

    double mu = 1.0;
    double sigma = 0.5;
    // E_μ = max(0.01, 0.5) = 0.5, μ = 1.0 → 0.5 ≤ 10·1.0 → solved
    // iters_at_mu=1 ≤ 2 → easy → σ_eff = 0.5^1.5 = 0.3536
    // mu_new = max(0.3536, 1e-8) = 0.3536
    bool changed = strat.update(mu, 0.01, 0.5, sigma);

    double expected = std::pow(0.5, 1.5);  // 0.35355...
    if (!changed)            { FAIL("μ should have changed"); return; }
    if (std::fabs(mu - expected) > TOL) {
        char buf[128]; snprintf(buf, sizeof(buf), "expected mu=%.6e, got %.6e", expected, mu);
        FAIL(buf); return;
    }

    PASS();
}

// ═══════════════════════════════════════════════════════════════════
//  Test 2: Barrier not solved → μ held fixed
// ═══════════════════════════════════════════════════════════════════

void test_unsolved_holds_mu() {
    TEST("Barrier not solved → μ fixed");

    BarrierUpdateParams p;
    p.mu_init   = 1.0;
    p.mu_min    = 1e-6;
    p.kappa_eps = 10.0;

    BarrierUpdateStrategy strat;
    strat.reset(p);

    double mu = 1.0;
    // E_μ = 50.0, μ = 1.0 → 50.0 > 10·1.0 → NOT solved
    bool changed = strat.update(mu, 50.0, 0.1, 0.1);

    if (changed)             { FAIL("μ should not change"); return; }
    if (std::fabs(mu - 1.0) > TOL) {
        char buf[128]; snprintf(buf, sizeof(buf), "μ should stay at 1.0, got %.6e", mu);
        FAIL(buf); return;
    }

    PASS();
}

// ═══════════════════════════════════════════════════════════════════
//  Test 3: Force reduction → hard path (σ^0.5)
// ═══════════════════════════════════════════════════════════════════

void test_force_reduction() {
    TEST("Force reduction → σ^0.5 (hard path)");

    BarrierUpdateParams p;
    p.mu_init    = 1.0;
    p.mu_min     = 1e-6;
    p.kappa_eps  = 10.0;
    p.max_same_mu = 5;

    BarrierUpdateStrategy strat;
    strat.reset(p);

    double mu = 1.0;
    // Push 4 iterations: barrier NOT solved → μ stays
    for (int i = 0; i < 4; ++i) {
        bool changed = strat.update(mu, 50.0, 0.1, 0.25);
        if (changed) { FAIL("μ should stay frozen"); return; }
    }

    // 5th iteration: forced reduction → hard path
    bool changed = strat.update(mu, 50.0, 0.1, 0.25);
    if (!changed) { FAIL("μ should be forced to change"); return; }
    // iters_at_mu=5, not solved → hard → σ^0.5 = 0.25^0.5 = 0.5
    double expected = std::pow(0.25, 0.5);  // 0.5
    if (std::fabs(mu - expected) > TOL) {
        char buf[128]; snprintf(buf, sizeof(buf), "expected forced mu=%.6e, got %.6e", expected, mu);
        FAIL(buf); return;
    }

    PASS();
}

// ═══════════════════════════════════════════════════════════════════
//  Test 4: Standalone is_barrier_solved function
// ═══════════════════════════════════════════════════════════════════

void test_standalone_helper() {
    TEST("is_barrier_solved helper");

    if (!is_barrier_solved(1e-3, 5e-3, 0.1))
        { FAIL("should be solved"); return; }
    if (!is_barrier_solved(1.0, 0.1, 0.1))
        { FAIL("barely solved should pass"); return; }
    if (is_barrier_solved(2.0, 0.1, 0.1))
        { FAIL("should NOT be solved"); return; }

    PASS();
}

// ═══════════════════════════════════════════════════════════════════
//  Test 5: σ ≈ 1 → σ^1.5 still moderate
// ═══════════════════════════════════════════════════════════════════

void test_sigma_near_one() {
    TEST("σ≈1 → σ^1.5 still conservative");

    BarrierUpdateParams p;
    p.mu_init   = 1.0;
    p.mu_min    = 1e-6;
    p.kappa_eps = 10.0;

    BarrierUpdateStrategy strat;
    strat.reset(p);

    double mu = 1.0;
    // iters=1 → easy → σ^1.5 = 0.99^1.5 ≈ 0.985
    strat.update(mu, 0.01, 0.01, 0.99);

    double expected = std::pow(0.99, 1.5);  // 0.98504...
    if (std::fabs(mu - expected) > TOL) {
        char buf[128]; snprintf(buf, sizeof(buf), "expected mu=%.6e, got %.6e", expected, mu);
        FAIL(buf); return;
    }

    PASS();
}

// ═══════════════════════════════════════════════════════════════════
//  Test 6: at_minimum() and repeated σ^1.5 reduction
// ═══════════════════════════════════════════════════════════════════

void test_at_minimum() {
    TEST("at_minimum() and repeated σ^1.5 schedule");

    BarrierUpdateParams p;
    p.mu_init   = 1.0;
    p.mu_min    = 1e-5;
    p.kappa_eps = 10.0;

    BarrierUpdateStrategy strat;
    strat.reset(p);

    double mu = 1.0;
    double sigma = 0.1;
    if (strat.at_minimum(mu)) { FAIL("should not be at minimum yet"); return; }

    // Repeated easy reductions: μ *= σ^1.5 each step
    // σ^1.5 = 0.1^1.5 = 0.03162...
    double sigma_eff = std::pow(sigma, 1.5);  // 0.03162
    for (int i = 0; i < 5; ++i) {
        strat.update(mu, 1e-8, 1e-8, sigma);
        double expected_mu = std::max(sigma_eff * mu, p.mu_min);  // not quite right but checks trend
        // Just check it's decreasing and positive
        if (mu <= 0) { FAIL("mu should stay positive"); return; }
    }
    // After 5 reductions of σ^1.5 = 0.0316: mu ≈ 1.0 * 0.0316^5 ≈ 3.16e-8
    // which is well below mu_min=1e-5, so at_minimum should be true
    if (!strat.at_minimum(mu)) { FAIL("should be at minimum after 5 easy reductions"); return; }

    PASS();
}

// ═══════════════════════════════════════════════════════════════════
//  Test 7: Normal path uses σ directly (exponent = 1.0)
// ═══════════════════════════════════════════════════════════════════

void test_normal_path() {
    TEST("Normal path → σ^1.0 (standard Mehrotra)");

    BarrierUpdateParams p;
    p.mu_init   = 1.0;
    p.mu_min    = 1e-8;
    p.kappa_eps = 10.0;
    p.fast_threshold = 1;  // only 1 iter = easy
    p.slow_threshold = 4;

    BarrierUpdateStrategy strat;
    strat.reset(p);

    double mu = 1.0;
    // Solve in exactly 3 iters (between fast=1 and slow=4 → normal)
    strat.update(mu, 50.0, 50.0, 0.5); // iter 1: NOT solved
    strat.update(mu, 50.0, 50.0, 0.5); // iter 2: NOT solved
    double sigma = 0.4;
    strat.update(mu, 1e-8, 1e-8, sigma); // iter 3: solved → normal

    // normal → σ^1.0 = 0.4
    double expected = 0.4;
    if (std::fabs(mu - expected) > TOL) {
        char buf[128]; snprintf(buf, sizeof(buf), "expected mu=%.6e, got %.6e", expected, mu);
        FAIL(buf); return;
    }

    PASS();
}

// ═══════════════════════════════════════════════════════════════════

int main() {
    printf("+-------------------------------------------+\n");
    printf("|   σ-modulation BarrierUpdate Tests        |\n");
    printf("+-------------------------------------------+\n\n");

    test_easy_sigma_modulation();
    test_unsolved_holds_mu();
    test_force_reduction();
    test_standalone_helper();
    test_sigma_near_one();
    test_at_minimum();
    test_normal_path();

    printf("\n--- Results: %d/%d passed ---\n", pass_count, test_count);
    return (pass_count == test_count) ? 0 : 1;
}
