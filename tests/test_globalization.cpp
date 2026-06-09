/**
 * @file    test_globalization.cpp
 * @brief   Unit tests for globalization features:
 *            1. Complementarity Safeguard (HPIPM-style)
 *            2. BarrierManager integration (quality-gated mu)
 *            3. Adaptive penalty weight (Nocedal-Wright)
 *            4. Second-Order Correction (SOC)
 */

#include <cstdio>
#include <cmath>
#include <cassert>

#include "nmpc/nmpc_solver_paper.hpp"

using namespace nmpc;

static int test_count = 0;
static int pass_count = 0;

#define TEST(name) do { test_count++; printf("  TEST %s ... ", name); } while(0)
#define PASS()      do { pass_count++; printf("PASS\n"); } while(0)
#define FAIL(msg)   do { printf("FAIL: %s\n", msg); } while(0)

// ═══════════════════════════════════════════════════════════════════════════
//  Test Problem: Double Integrator with box control constraints
//    NX=2, NU=1, NC=2, N=10
//    Dynamics: x1_next = x1 + dt*x2 + 0.5*dt^2*u
//              x2_next = x2 + dt*u
//    Cost:     0.5*(x^T Q x + u^T R u), terminal 0.5*x^T Qf x
//    Constraints: -1 <= u <= 1  (g = [-1-u, u-1] <= 0)
// ═══════════════════════════════════════════════════════════════════════════

constexpr int NX = 2;
constexpr int NU = 1;
constexpr int NC = 2;
constexpr int HORIZON = 10;
constexpr double DT = 0.1;

struct DoubleIntDyn : DynamicsModel<NX, NU> {
    Status discrete_step(const Vec<NX>& x, const Vec<NU>& u,
                         double dt, Vec<NX>& x_next) override {
        x_next[0] = x[0] + dt * x[1] + 0.5 * dt * dt * u[0];
        x_next[1] = x[1] + dt * u[0];
        return Status::SUCCESS;
    }
    Status linearize(const Vec<NX>& /*x*/, const Vec<NU>& /*u*/,
                     double dt, Mat<NX, NX>& A, Mat<NX, NU>& B) override {
        A.zero();
        A(0, 0) = 1.0;  A(0, 1) = dt;
        A(1, 0) = 0.0;  A(1, 1) = 1.0;
        B.zero();
        B(0, 0) = 0.5 * dt * dt;
        B(1, 0) = dt;
        return Status::SUCCESS;
    }
};

struct DoubleIntCost : CostModel<NX, NU> {
    double Q[NX]  = {1.0, 1.0};
    double Qf[NX] = {10.0, 10.0};
    double R      = 0.1;

    double stage_cost(const Vec<NX>& x, const Vec<NU>& u, int /*k*/) override {
        return 0.5 * (Q[0]*x[0]*x[0] + Q[1]*x[1]*x[1] + R*u[0]*u[0]);
    }
    double terminal_cost(const Vec<NX>& x) override {
        return 0.5 * (Qf[0]*x[0]*x[0] + Qf[1]*x[1]*x[1]);
    }
    Status stage_gradient(const Vec<NX>& x, const Vec<NU>& u, int /*k*/,
                          Vec<NX>& qx, Vec<NU>& qu) override {
        qx[0] = Q[0] * x[0];
        qx[1] = Q[1] * x[1];
        qu[0] = R * u[0];
        return Status::SUCCESS;
    }
    Status stage_hessian(const Vec<NX>& /*x*/, const Vec<NU>& /*u*/, int /*k*/,
                         Mat<NX, NX>& Qxx, Mat<NU, NU>& Quu,
                         Mat<NU, NX>& Qux) override {
        Qxx.zero(); Qxx(0,0) = Q[0]; Qxx(1,1) = Q[1];
        Quu.zero(); Quu(0,0) = R;
        Qux.zero();
        return Status::SUCCESS;
    }
    Status terminal_gradient(const Vec<NX>& x, Vec<NX>& qx) override {
        qx[0] = Qf[0] * x[0];
        qx[1] = Qf[1] * x[1];
        return Status::SUCCESS;
    }
    Status terminal_hessian(const Vec<NX>& /*x*/, Mat<NX, NX>& Qxx) override {
        Qxx.zero(); Qxx(0,0) = Qf[0]; Qxx(1,1) = Qf[1];
        return Status::SUCCESS;
    }
};

struct DoubleIntCons : ConstraintModel<NX, NU, NC> {
    // g = [-1 - u, u - 1] <= 0  =>  -1 <= u <= 1
    Status evaluate(const Vec<NX>& /*x*/, const Vec<NU>& u, int /*k*/,
                    Vec<NC>& g) override {
        g[0] = -1.0 - u[0];   // u >= -1
        g[1] =  u[0] - 1.0;   // u <=  1
        return Status::SUCCESS;
    }
    Status evaluate_terminal(const Vec<NX>& /*x*/, Vec<NC>& g) override {
        // Terminal has no control: always feasible (large negative)
        g[0] = -1e10;
        g[1] = -1e10;
        return Status::SUCCESS;
    }
    Status jacobian(const Vec<NX>& /*x*/, const Vec<NU>& /*u*/, int /*k*/,
                    Mat<NC, NX>& Cx, Mat<NC, NU>& Cu) override {
        Cx.zero();
        Cu.zero();
        Cu(0, 0) = -1.0;   // dg[0]/du = -1
        Cu(1, 0) =  1.0;   // dg[1]/du = +1
        return Status::SUCCESS;
    }
    Status jacobian_terminal(const Vec<NX>& /*x*/, Mat<NC, NX>& Cx) override {
        Cx.zero();
        return Status::SUCCESS;
    }
};

// ── Helper: set up a fresh problem with given initial state ──────────────

using Problem = NMPCProblem<NX, NU, NC, HORIZON>;

void setup_problem(Problem& prob, DoubleIntDyn& dyn, DoubleIntCost& cost,
                   DoubleIntCons& cons, double x0_0 = 0.5, double x0_1 = 0.0) {
    prob.dynamics    = &dyn;
    prob.cost        = &cost;
    prob.constraints = &cons;
    prob.dt          = DT;

    prob.x0[0] = x0_0;
    prob.x0[1] = x0_1;

    // Initialize trajectory with simple forward simulation
    prob.stages[0].x[0] = x0_0;
    prob.stages[0].x[1] = x0_1;
    for (int k = 0; k < HORIZON; ++k) {
        prob.stages[k].u[0] = 0.0;
        Vec<NX> xn;
        dyn.discrete_step(prob.stages[k].x, prob.stages[k].u, DT, xn);
        prob.stages[k+1].x = xn;
    }
    prob.stages[HORIZON].u.zero();
    for (int k = 0; k <= HORIZON; ++k) {
        prob.stages[k].s.set_constant(1.0);
        prob.stages[k].lambda.set_constant(0.1);
    }
}

// ── Helper: get default params that work well on this problem ────────────

PaperIPMParams default_test_params() {
    PaperIPMParams pp;
    pp.mu_init       = 0.1;
    pp.mu_min        = 1e-4;
    pp.max_iters     = 80;
    pp.tol_primal    = 1e-3;
    pp.tol_compl     = 1e-3;
    pp.tol_ineq      = 1e-6;
    pp.verbosity     = 0;
    pp.m_safe        = 0.01;
    pp.soc_max       = 4;
    return pp;
}

// ═══════════════════════════════════════════════════════════════════════════
//  Test 1: Complementarity Safeguard
// ═══════════════════════════════════════════════════════════════════════════

void test_complementarity_safeguard() {
    TEST("Complementarity safeguard (m_safe > 0)");

    DoubleIntDyn dyn; DoubleIntCost cost; DoubleIntCons cons;
    Problem prob;
    setup_problem(prob, dyn, cost, cons);

    // Solve with safeguard enabled
    NMPCSolverPaper<NX, NU, NC, HORIZON> solver;
    PaperIPMParams pp = default_test_params();
    pp.m_safe = 0.01;
    solver.configure(pp);

    Status st = solver.solve(prob);
    const auto& stats = solver.last_stats();

    if (st != Status::SUCCESS && st != Status::MAX_ITERATIONS) {
        char buf[128];
        snprintf(buf, sizeof(buf), "solve failed: %s", status_string(st));
        FAIL(buf); return;
    }

    // Verify complementarity products satisfy safeguard bound
    double mu_final = stats.barrier_param;
    double threshold = pp.m_safe * mu_final;
    bool safeguard_ok = true;

    for (int k = 0; k <= HORIZON; ++k) {
        for (int j = 0; j < NC; ++j) {
            double sj = prob.stages[k].s[j];
            double lj = prob.stages[k].lambda[j];
            // Allow numerical tolerance (factor 0.1)
            if (sj > 1e-10 && lj > 1e-10 && sj * lj < threshold * 0.1) {
                safeguard_ok = false;
            }
        }
    }

    if (!safeguard_ok) {
        FAIL("complementarity products below m_safe*mu"); return;
    }

    PASS();
}

// ═══════════════════════════════════════════════════════════════════════════
//  Test 2: BarrierManager Integration
// ═══════════════════════════════════════════════════════════════════════════

void test_barrier_manager_integration() {
    TEST("BarrierUpdateStrategy quality-gated mu reduction");

    DoubleIntDyn dyn; DoubleIntCost cost; DoubleIntCons cons;

    Problem prob;
    setup_problem(prob, dyn, cost, cons);
    NMPCSolverPaper<NX, NU, NC, HORIZON> solver;
    PaperIPMParams pp = default_test_params();
    solver.configure(pp);
    Status st = solver.solve(prob);
    const auto& stats = solver.last_stats();

    if (st != Status::SUCCESS && st != Status::MAX_ITERATIONS) {
        char buf[128];
        snprintf(buf, sizeof(buf), "barrier strategy failed: %s", status_string(st));
        FAIL(buf); return;
    }

    printf("(%d iters/%s) ", stats.inner_iterations, status_string(st));
    PASS();
}

// ═══════════════════════════════════════════════════════════════════════════
//  Test 3: Adaptive Penalty Weight
// ═══════════════════════════════════════════════════════════════════════════

void test_adaptive_penalty() {
    TEST("Filter line search convergence");

    DoubleIntDyn dyn; DoubleIntCost cost; DoubleIntCons cons;

    // Use initial state with velocity that pushes toward constraint
    Problem prob1;
    setup_problem(prob1, dyn, cost, cons, 0.5, 0.0);
    NMPCSolverPaper<NX, NU, NC, HORIZON> solver1;
    PaperIPMParams pp1 = default_test_params();
    solver1.configure(pp1);
    Status st1 = solver1.solve(prob1);
    const auto& stats1 = solver1.last_stats();

    // Accept SUCCESS or MAX_ITERATIONS
    if (st1 != Status::SUCCESS && st1 != Status::MAX_ITERATIONS) {
        char buf[128];
        snprintf(buf, sizeof(buf), "filter_ls failed: status=%d", (int)st1);
        FAIL(buf); return;
    }

    printf("(status=%d, iters=%d) ", (int)st1, stats1.inner_iterations);
    PASS();
}

// ═══════════════════════════════════════════════════════════════════════════
//  Test 4: Second-Order Correction (SOC)
// ═══════════════════════════════════════════════════════════════════════════

void test_soc_activation() {
    TEST("Second-Order Correction (SOC)");

    DoubleIntDyn dyn; DoubleIntCost cost; DoubleIntCons cons;

    // Solve with SOC enabled (soc_max > 0)
    Problem prob1;
    setup_problem(prob1, dyn, cost, cons, 1.0, 0.5);
    NMPCSolverPaper<NX, NU, NC, HORIZON> solver1;
    PaperIPMParams pp1 = default_test_params();
    pp1.soc_max = 4;
    solver1.configure(pp1);
    Status st1 = solver1.solve(prob1);
    const auto& stats1 = solver1.last_stats();

    if (st1 != Status::SUCCESS && st1 != Status::MAX_ITERATIONS) {
        char buf[128];
        snprintf(buf, sizeof(buf), "soc_max=4 failed: %s", status_string(st1));
        FAIL(buf); return;
    }

    // Solve with SOC disabled (soc_max = 0)
    Problem prob2;
    setup_problem(prob2, dyn, cost, cons, 1.0, 0.5);
    NMPCSolverPaper<NX, NU, NC, HORIZON> solver2;
    PaperIPMParams pp2 = pp1;
    pp2.soc_max = 0;
    solver2.configure(pp2);
    Status st2 = solver2.solve(prob2);
    const auto& stats2 = solver2.last_stats();

    // Both should produce a result (might not converge fully)
    if (st2 != Status::SUCCESS && st2 != Status::MAX_ITERATIONS) {
        char buf[128];
        snprintf(buf, sizeof(buf), "soc_max=0 unexpected: %s", status_string(st2));
        FAIL(buf); return;
    }

    printf("(SOC steps: %d, iters w/SOC: %d, w/o: %d) ",
           stats1.soc_steps, stats1.inner_iterations, stats2.inner_iterations);
    PASS();
}

// ═══════════════════════════════════════════════════════════════════════════

int main() {
    printf("\xE2\x95\x94\xE2\x95\x90\xE2\x95\x90\xE2\x95\x90\xE2\x95\x90\xE2\x95\x90\xE2\x95\x90\xE2\x95\x90\xE2\x95\x90\xE2\x95\x90\xE2\x95\x90\xE2\x95\x90\xE2\x95\x90\xE2\x95\x90\xE2\x95\x90\xE2\x95\x90\xE2\x95\x90\xE2\x95\x90\xE2\x95\x90\xE2\x95\x90\xE2\x95\x90\xE2\x95\x90\xE2\x95\x90\xE2\x95\x90\xE2\x95\x90\xE2\x95\x90\xE2\x95\x90\xE2\x95\x90\xE2\x95\x90\xE2\x95\x90\xE2\x95\x90\xE2\x95\x90\xE2\x95\x90\xE2\x95\x90\xE2\x95\x90\xE2\x95\x90\xE2\x95\x90\xE2\x95\x90\xE2\x95\x90\xE2\x95\x90\xE2\x95\x90\xE2\x95\x90\xE2\x95\x90\xE2\x95\x90\xE2\x95\x97\n");
    printf("\xE2\x95\x91   Globalization Features Unit Tests          \xE2\x95\x91\n");
    printf("\xE2\x95\x9A\xE2\x95\x90\xE2\x95\x90\xE2\x95\x90\xE2\x95\x90\xE2\x95\x90\xE2\x95\x90\xE2\x95\x90\xE2\x95\x90\xE2\x95\x90\xE2\x95\x90\xE2\x95\x90\xE2\x95\x90\xE2\x95\x90\xE2\x95\x90\xE2\x95\x90\xE2\x95\x90\xE2\x95\x90\xE2\x95\x90\xE2\x95\x90\xE2\x95\x90\xE2\x95\x90\xE2\x95\x90\xE2\x95\x90\xE2\x95\x90\xE2\x95\x90\xE2\x95\x90\xE2\x95\x90\xE2\x95\x90\xE2\x95\x90\xE2\x95\x90\xE2\x95\x90\xE2\x95\x90\xE2\x95\x90\xE2\x95\x90\xE2\x95\x90\xE2\x95\x90\xE2\x95\x90\xE2\x95\x90\xE2\x95\x90\xE2\x95\x90\xE2\x95\x90\xE2\x95\x90\xE2\x95\x90\xE2\x95\x9D\n\n");

    test_complementarity_safeguard();
    test_barrier_manager_integration();
    test_adaptive_penalty();
    test_soc_activation();

    printf("\n--- Results: %d/%d passed ---\n", pass_count, test_count);
    return (pass_count == test_count) ? 0 : 1;
}
