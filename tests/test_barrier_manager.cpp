/**
 * @file    test_barrier_manager.cpp
 * @brief   Unit tests for BarrierUpdateStrategy module.
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

// ═══════════════════════════════════════════════════════════════════
//  Test 1: Barrier solved → μ reduced via Mehrotra/LOQO rule
// ═══════════════════════════════════════════════════════════════════

void test_solved_reduces_mu() {
    TEST("Barrier solved → μ reduced");

    BarrierUpdateParams p;
    p.mu_init   = 1.0;
    p.mu_min    = 1e-6;
    p.kappa_eps = 10.0;
    p.kappa_mu  = 0.2;

    BarrierUpdateStrategy strat;
    strat.reset(p);

    double mu = 1.0;
    // E_μ = max(0.01, 0.5) = 0.5, μ = 1.0 → 0.5 ≤ 10·1.0 → solved
    // Adaptive: σ·μ = 0.1·1.0 = 0.1
    // Monotone: κ_mu·μ = 0.2·1.0 = 0.2
    // mu_new = max(0.1, 0.2) = 0.2
    bool changed = strat.update(mu, 0.01, 0.5, 0.1);

    if (!changed)            { FAIL("μ should have changed"); return; }
    if (std::fabs(mu - 0.2) > 1e-10) {
        char buf[128]; snprintf(buf, sizeof(buf), "expected mu=0.2, got %.6e", mu);
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
    if (std::fabs(mu - 1.0) > 1e-10) {
        char buf[128]; snprintf(buf, sizeof(buf), "μ should stay at 1.0, got %.6e", mu);
        FAIL(buf); return;
    }

    PASS();
}

// ═══════════════════════════════════════════════════════════════════
//  Test 3: Force reduction after max_same_mu iterations
// ═══════════════════════════════════════════════════════════════════

void test_force_reduction() {
    TEST("Force reduction after max iterations at same μ");

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
        bool changed = strat.update(mu, 50.0, 0.1, 0.1);
        if (changed) { FAIL("μ should stay frozen"); return; }
    }

    // 5th iteration: forced reduction
    bool changed = strat.update(mu, 50.0, 0.1, 0.1);
    if (!changed) { FAIL("μ should be forced to change"); return; }
    // Monotone: κ_mu·μ = 0.2·1.0 = 0.2
    // Adaptive: σ·μ = 0.1·1.0 = 0.1
    // max(0.2, 0.1) = 0.2
    if (std::fabs(mu - 0.2) > 1e-10) {
        char buf[128]; snprintf(buf, sizeof(buf), "expected forced mu=0.2, got %.6e", mu);
        FAIL(buf); return;
    }

    PASS();
}

// ═══════════════════════════════════════════════════════════════════
//  Test 4: Standalone is_barrier_solved function
// ═══════════════════════════════════════════════════════════════════

void test_standalone_helper() {
    TEST("is_barrier_solved helper");

    // E_μ = max(1e-3, 5e-3) = 5e-3, μ = 0.1 → 5e-3 ≤ 10·0.1 = 1.0 → YES
    if (!is_barrier_solved(1e-3, 5e-3, 0.1))
        { FAIL("should be solved"); return; }

    // E_μ = max(1.0, 0.1) = 1.0, μ = 0.1 → 1.0 ≤ 1.0 → YES (barely)
    if (!is_barrier_solved(1.0, 0.1, 0.1))
        { FAIL("barely solved should pass"); return; }

    // E_μ = max(2.0, 0.1) = 2.0, μ = 0.1 → 2.0 > 1.0 → NO
    if (is_barrier_solved(2.0, 0.1, 0.1))
        { FAIL("should NOT be solved"); return; }

    PASS();
}

// ═══════════════════════════════════════════════════════════════════
//  Test 5: σ ≈ 1 → conservative μ reduction
// ═══════════════════════════════════════════════════════════════════

void test_sigma_near_one() {
    TEST("σ≈1 → conservative μ reduction");

    BarrierUpdateParams p;
    p.mu_init   = 1.0;
    p.mu_min    = 1e-6;
    p.kappa_eps = 10.0;
    p.kappa_mu  = 0.2;

    BarrierUpdateStrategy strat;
    strat.reset(p);

    double mu = 1.0;
    // σ = 0.99 → adaptive = max(0.99, 1e-6) = 0.99
    // monotone = max(0.2, 1e-6) = 0.2
    // mu_new = max(0.99, 0.2) = 0.99
    strat.update(mu, 0.01, 0.01, 0.99);

    if (std::fabs(mu - 0.99) > 1e-10) {
        char buf[128]; snprintf(buf, sizeof(buf), "expected mu=0.99, got %.6e", mu);
        FAIL(buf); return;
    }

    PASS();
}

// ═══════════════════════════════════════════════════════════════════
//  Test 6: at_minimum() and μ_min floor
// ═══════════════════════════════════════════════════════════════════

void test_at_minimum() {
    TEST("at_minimum() and μ_min floor");

    BarrierUpdateParams p;
    p.mu_init   = 1e-4;
    p.mu_min    = 1e-5;
    p.kappa_eps = 10.0;
    p.kappa_mu  = 0.2;

    BarrierUpdateStrategy strat;
    strat.reset(p);

    double mu = 1e-4;
    if (strat.at_minimum(mu)) { FAIL("should not be at minimum yet"); return; }

    // Solve → reduce: max(σ·μ, κ_μ·μ, μ_min)
    // σ = 0.05 → adaptive = max(0.05·1e-4, 1e-5) = 1e-5
    // monotone = max(0.2·1e-4, 1e-5) = 2e-5
    // result = max(1e-5, 2e-5) = 2e-5
    strat.update(mu, 1e-8, 1e-8, 0.05);

    if (strat.at_minimum(mu)) { FAIL("mu=2e-5 should not be at minimum"); return; }

    // One more reduction:
    // adaptive = max(0.05·2e-5, 1e-5) = 1e-5
    // monotone = max(0.2·2e-5, 1e-5) = 1e-5
    // result = max(1e-5, 1e-5) = 1e-5
    strat.update(mu, 1e-8, 1e-8, 0.05);

    if (!strat.at_minimum(mu)) { FAIL("mu=1e-5 should be at minimum"); return; }

    PASS();
}

// ═══════════════════════════════════════════════════════════════════

int main() {
    printf("+-------------------------------------------+\n");
    printf("|   BarrierUpdateStrategy Unit Tests        |\n");
    printf("+-------------------------------------------+\n\n");

    test_solved_reduces_mu();
    test_unsolved_holds_mu();
    test_force_reduction();
    test_standalone_helper();
    test_sigma_near_one();
    test_at_minimum();

    printf("\n--- Results: %d/%d passed ---\n", pass_count, test_count);
    return (pass_count == test_count) ? 0 : 1;
}
