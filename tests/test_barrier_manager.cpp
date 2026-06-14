/**
 * @file    test_barrier_manager.cpp
 * @brief   Unit tests for sigma-modulation BarrierUpdateStrategy.
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

void test_easy_sigma_modulation() {
    TEST("Easy subproblem -> sigma^1.5 reduction");
    BarrierUpdateParams p; p.mu_init=1.0; p.mu_min=1e-8; p.kappa_eps=10.0;
    BarrierUpdateStrategy strat; strat.reset(p);
    double mu = 1.0; double sigma = 0.5;
    bool changed = strat.update(mu, 0.01, 0.5, sigma);
    double expected = std::pow(0.5, 1.5);
    if (!changed) { FAIL("mu should have changed"); return; }
    if (std::fabs(mu - expected) > TOL) { FAIL("expected mu wrong"); return; }
    PASS();
}

void test_unsolved_holds_mu() {
    TEST("Barrier not solved -> mu fixed");
    BarrierUpdateParams p; p.mu_init=1.0; p.mu_min=1e-6; p.kappa_eps=10.0;
    BarrierUpdateStrategy strat; strat.reset(p);
    double mu = 1.0;
    bool changed = strat.update(mu, 50.0, 0.1, 0.1);
    if (changed) { FAIL("mu should not change"); return; }
    if (std::fabs(mu - 1.0) > TOL) { FAIL("mu should stay at 1.0"); return; }
    PASS();
}

void test_force_reduction() {
    TEST("Force reduction -> sigma^0.5 (hard path)");
    BarrierUpdateParams p; p.mu_init=1.0; p.mu_min=1e-6; p.kappa_eps=10.0; p.max_same_mu=5;
    BarrierUpdateStrategy strat; strat.reset(p);
    double mu = 1.0;
    for (int i = 0; i < 4; ++i) { bool c = strat.update(mu, 50.0, 0.1, 0.25); if (c) { FAIL("mu should stay frozen"); return; } }
    bool changed = strat.update(mu, 50.0, 0.1, 0.25);
    if (!changed) { FAIL("mu should be forced to change"); return; }
    double expected = std::pow(0.25, 0.5);
    if (std::fabs(mu - expected) > TOL) { FAIL("forced mu wrong"); return; }
    PASS();
}

void test_standalone_helper() {
    TEST("is_barrier_solved helper");
    if (!is_barrier_solved(1e-3, 5e-3, 0.1)) { FAIL("should be solved"); return; }
    if (!is_barrier_solved(1.0, 0.1, 0.1)) { FAIL("barely solved should pass"); return; }
    if (is_barrier_solved(2.0, 0.1, 0.1)) { FAIL("should NOT be solved"); return; }
    PASS();
}

void test_sigma_near_one() {
    TEST("sigma~1 -> sigma^1.5 still conservative");
    BarrierUpdateParams p; p.mu_init=1.0; p.mu_min=1e-6; p.kappa_eps=10.0;
    BarrierUpdateStrategy strat; strat.reset(p);
    double mu = 1.0;
    strat.update(mu, 0.01, 0.01, 0.99);
    double expected = std::pow(0.99, 1.5);
    if (std::fabs(mu - expected) > TOL) { FAIL("expected mu wrong"); return; }
    PASS();
}

void test_at_minimum() {
    TEST("at_minimum() and repeated reduction schedule");
    BarrierUpdateParams p; p.mu_init=1.0; p.mu_min=1e-5; p.kappa_eps=10.0;
    BarrierUpdateStrategy strat; strat.reset(p);
    double mu = 1.0; double sigma = 0.1;
    if (strat.at_minimum(mu)) { FAIL("should not be at minimum yet"); return; }
    // Repeated easy reductions: mu *= sigma^1.5 each step.
    // sigma^1.5 = 0.1^1.5 = 0.03162, so after 5 steps
    // mu = 1.0 * 0.03162^5 = 3.16e-8, well below mu_min=1e-5.
    for (int i = 0; i < 5; ++i) {
        strat.update(mu, 1e-8, 1e-8, sigma);
        if (mu <= 0) { FAIL("mu should stay positive"); return; }
    }
    if (!strat.at_minimum(mu)) { FAIL("should be at minimum after 5 easy reductions"); return; }
    PASS();
}

void test_normal_path() {
    TEST("Normal path -> sigma^1.0 (standard Mehrotra)");
    BarrierUpdateParams p; p.mu_init=1.0; p.mu_min=1e-8; p.kappa_eps=10.0; p.fast_threshold=1; p.slow_threshold=4;
    BarrierUpdateStrategy strat; strat.reset(p);
    double mu = 1.0;
    strat.update(mu, 50.0, 50.0, 0.5);
    strat.update(mu, 50.0, 50.0, 0.5);
    double sigma = 0.4;
    strat.update(mu, 1e-8, 1e-8, sigma);
    double expected = 0.4;
    if (std::fabs(mu - expected) > TOL) { FAIL("expected mu wrong"); return; }
    PASS();
}

int main() {
    printf("+-------------------------------------------+\n");
    printf("|   sigma-modulation BarrierUpdate Tests    |\n");
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
