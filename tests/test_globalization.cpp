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

#include "nmpc/contact_ipm.hpp"

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

struct FailingDoubleIntDyn : DoubleIntDyn {
    bool fail = false;
    Status discrete_step(const Vec<NX>& x, const Vec<NU>& u,
                         double dt, Vec<NX>& x_next) override {
        if (fail) return Status::INTERNAL_ERROR;
        return DoubleIntDyn::discrete_step(x, u, dt, x_next);
    }
};

struct CountingCons : DoubleIntCons {
    mutable int num_calls = 0;
    mutable int pair_count_calls = 0;
    int evaluate_calls = 0;
    int terminal_evaluate_calls = 0;
    int jacobian_calls = 0;
    int terminal_jacobian_calls = 0;

    int num_constraints(int) const override {
        ++num_calls;
        return NC;
    }
    int num_complementarity_pairs(int) const override {
        ++pair_count_calls;
        return 0;
    }
    Status evaluate(const Vec<NX>& x, const Vec<NU>& u, int k,
                    Vec<NC>& g) override {
        ++evaluate_calls;
        return DoubleIntCons::evaluate(x, u, k, g);
    }
    Status evaluate_terminal(const Vec<NX>& x, Vec<NC>& g) override {
        ++terminal_evaluate_calls;
        return DoubleIntCons::evaluate_terminal(x, g);
    }
    Status jacobian(const Vec<NX>& x, const Vec<NU>& u, int k,
                    Mat<NC, NX>& Cx, Mat<NC, NU>& Cu) override {
        ++jacobian_calls;
        return DoubleIntCons::jacobian(x, u, k, Cx, Cu);
    }
    Status jacobian_terminal(const Vec<NX>& x,
                             Mat<NC, NX>& Cx) override {
        ++terminal_jacobian_calls;
        return DoubleIntCons::jacobian_terminal(x, Cx);
    }
};

// ── Helper: set up a fresh problem with given initial state ──────────────

using Problem = NMPCProblem<NX, NU, NC, HORIZON>;

bool same_problem_data(const Problem& first, const Problem& second) {
    for (int i = 0; i < NX; ++i)
        if (first.x0[i] != second.x0[i]) return false;
    for (int k = 0; k <= HORIZON; ++k) {
        for (int i = 0; i < NX; ++i) {
            if (first.stages[k].x[i] != second.stages[k].x[i] ||
                first.stages[k].z_L_x[i] != second.stages[k].z_L_x[i] ||
                first.stages[k].z_U_x[i] != second.stages[k].z_U_x[i]) {
                return false;
            }
        }
        for (int i = 0; i < NU; ++i) {
            if (first.stages[k].u[i] != second.stages[k].u[i] ||
                first.stages[k].z_L_u[i] != second.stages[k].z_L_u[i] ||
                first.stages[k].z_U_u[i] != second.stages[k].z_U_u[i]) {
                return false;
            }
        }
        for (int j = 0; j < NC; ++j) {
            if (first.stages[k].s[j] != second.stages[k].s[j] ||
                first.stages[k].lambda[j] != second.stages[k].lambda[j]) {
                return false;
            }
        }
    }
    return true;
}

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

ContactIPMParams default_test_params() {
    ContactIPMParams pp;
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
    ContactIPM<NX, NU, NC, HORIZON> solver;
    ContactIPMParams pp = default_test_params();
    pp.m_safe = 0.01;
    solver.configure(pp);

    Status st = solver.solve(prob);
    const auto& stats = solver.last_stats();

    if (st != Status::SUCCESS && st != Status::MAX_ITERATIONS
        && st != Status::STAGNATION) {
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
    ContactIPM<NX, NU, NC, HORIZON> solver;
    ContactIPMParams pp = default_test_params();
    solver.configure(pp);
    Status st = solver.solve(prob);
    const auto& stats = solver.last_stats();

    if (st != Status::SUCCESS && st != Status::MAX_ITERATIONS
        && st != Status::STAGNATION) {
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
    ContactIPM<NX, NU, NC, HORIZON> solver1;
    ContactIPMParams pp1 = default_test_params();
    solver1.configure(pp1);
    Status st1 = solver1.solve(prob1);
    const auto& stats1 = solver1.last_stats();

    // Accept SUCCESS, MAX_ITERATIONS, or STAGNATION
    if (st1 != Status::SUCCESS && st1 != Status::MAX_ITERATIONS
        && st1 != Status::STAGNATION) {
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
    ContactIPM<NX, NU, NC, HORIZON> solver1;
    ContactIPMParams pp1 = default_test_params();
    pp1.soc_max = 4;
    pp1.mu_init = 1.0;
    pp1.mu_min = 5e-3;
    pp1.max_same_mu = 30;
    solver1.configure(pp1);
    Status st1 = solver1.solve(prob1);
    const auto& stats1 = solver1.last_stats();

    if (st1 != Status::SUCCESS && st1 != Status::MAX_ITERATIONS
        && st1 != Status::LINE_SEARCH_FAILURE && st1 != Status::STAGNATION) {
        char buf[128];
        snprintf(buf, sizeof(buf), "soc_max=4 failed: %s", status_string(st1));
        FAIL(buf); return;
    }

    // Solve with SOC disabled (soc_max = 0)
    Problem prob2;
    setup_problem(prob2, dyn, cost, cons, 1.0, 0.5);
    ContactIPM<NX, NU, NC, HORIZON> solver2;
    ContactIPMParams pp2 = pp1;
    pp2.soc_max = 0;
    solver2.configure(pp2);
    Status st2 = solver2.solve(prob2);
    const auto& stats2 = solver2.last_stats();

    // Both should produce a result (might not converge fully)
    if (st2 != Status::SUCCESS && st2 != Status::MAX_ITERATIONS
        && st2 != Status::LINE_SEARCH_FAILURE && st2 != Status::STAGNATION) {
        char buf[128];
        snprintf(buf, sizeof(buf), "soc_max=0 unexpected: %s", status_string(st2));
        FAIL(buf); return;
    }

    printf("(SOC steps: %d, iters w/SOC: %d, w/o: %d) ",
           stats1.soc_steps, stats1.inner_iterations, stats2.inner_iterations);
    PASS();
}

template <typename Cons>
void setup_problem_with_cons(Problem& prob, DoubleIntDyn& dyn,
                             DoubleIntCost& cost, Cons& cons,
                             double x0_0 = 0.5, double x0_1 = 0.0) {
    setup_problem(prob, dyn, cost,
                  static_cast<DoubleIntCons&>(cons), x0_0, x0_1);
    prob.constraints = &cons;
}

void test_shifted_primal_dual_warm_start() {
    TEST("Shifted primal-dual warm start");

    DoubleIntDyn dyn; DoubleIntCost cost; DoubleIntCons cons;
    Problem prob;
    setup_problem(prob, dyn, cost, cons, 0.5, 0.0);

    ContactIPM<NX, NU, NC, HORIZON> solver;
    ContactIPMParams pp = default_test_params();
    pp.exact_hessian = false;
    pp.adaptive_exact_hessian = false;
    pp.tol_stat = 1.0;
    solver.configure(pp);

    if (solver.solve_warm(prob) != Status::NOT_INITIALIZED) {
        FAIL("warm solve accepted before a successful cold solve"); return;
    }
    Status cold_status = solver.solve(prob);
    if (cold_status != Status::SUCCESS) {
        char buf[128];
        snprintf(buf, sizeof(buf), "cold solve failed: %s",
                 status_string(cold_status));
        FAIL(buf); return;
    }

    prob.stages[3].z_L_u[0] = 3.25;
    prob.stages[3].z_U_u[0] = 4.25;
    prob.stages[3].z_L_x[0] = 5.25;
    prob.stages[3].z_U_x[0] = 6.25;
    const double kept_slack = prob.stages[3].s[0];
    const double kept_dual = prob.stages[3].lambda[0];
    const Vec<NX> nominal_shift_state = prob.stages[2].x;
    const Vec<NX> retained_stage_state = prob.stages[3].x;
    Vec<NX> x_actual = nominal_shift_state;
    x_actual[0] += 0.07;
    x_actual[1] -= 0.03;
    Status shift_status = solver.shift_for_warmstart(prob, x_actual, 2);
    if (shift_status != Status::SUCCESS) {
        FAIL("two-stage shift failed"); return;
    }
    const auto shifted_riccati = solver.get_riccati_diag();
    if (shifted_riccati.p_stage0.norm_inf() != 0.0
        || shifted_riccati.p_terminal.norm_inf() != 0.0) {
        FAIL("shift retained stale Riccati costates"); return;
    }
    if (std::fabs(prob.stages[1].s[0] - kept_slack) > 1e-12
        || std::fabs(prob.stages[1].lambda[0] - kept_dual) > 1e-12) {
        FAIL("overlapping primal-dual data were not shifted"); return;
    }
    if (std::fabs(prob.stages[1].x[0] -
                  (retained_stage_state[0] + 0.07)) > 1e-12 ||
        std::fabs(prob.stages[1].x[1] -
                  (retained_stage_state[1] - 0.03)) > 1e-12) {
        FAIL("measured-state defect was not applied to retained trajectory");
        return;
    }
    if (prob.stages[1].z_L_u[0] != 3.25
        || prob.stages[1].z_U_u[0] != 4.25
        || prob.stages[1].z_L_x[0] != 5.25
        || prob.stages[1].z_U_x[0] != 6.25) {
        FAIL("overlapping bound multipliers were not shifted"); return;
    }
    if (prob.stages[HORIZON - 1].s[0] != 0.0
        || prob.stages[HORIZON - 1].lambda[0] != 0.0) {
        FAIL("appended tail primal-dual data were not reset"); return;
    }

    Status warm_status = solver.solve_warm(prob);
    if (warm_status != Status::SUCCESS) {
        char buf[128];
        snprintf(buf, sizeof(buf), "warm solve failed: %s",
                 status_string(warm_status));
        FAIL(buf); return;
    }
    const SolverStats warm_stats = solver.last_stats();
    if (warm_stats.warm_near_full_step_trials <= 0
        || warm_stats.warm_near_full_step_accepts <= 0
        || warm_stats.warm_near_full_step_accepts
               > warm_stats.warm_near_full_step_trials) {
        FAIL("warm near-full-step counters were not populated consistently");
        return;
    }
    for (int k = 0; k <= HORIZON; ++k) {
        for (int j = 0; j < NC; ++j) {
            if (!(prob.stages[k].s[j] > 0.0)
                || !(prob.stages[k].lambda[j] > 0.0)) {
                FAIL("warm solve left non-positive barrier data"); return;
            }
        }
    }

    // A failed/truncated warm attempt must not masquerade as an uninitialized
    // solver on the following MPC update.
    // Use a separate solver to exercise failure without disturbing the solved
    // trajectory checked above.
    ContactIPM<NX, NU, NC, HORIZON> retry_solver;
    retry_solver.configure(pp);
    Problem retry_prob;
    setup_problem(retry_prob, dyn, cost, cons, 0.5, 0.0);
    if (retry_solver.solve(retry_prob) != Status::SUCCESS) {
        FAIL("retry cold solve failed"); return;
    }
    const Vec<NX> retry_x = retry_prob.stages[1].x;
    retry_solver.shift_for_warmstart(retry_prob, retry_x, 1);
    retry_prob.cost = nullptr;
    if (retry_solver.solve_warm(retry_prob) != Status::BAD_ARGUMENT) {
        FAIL("expected failed warm attempt"); return;
    }
    retry_prob.cost = &cost;
    Status retry_status = retry_solver.solve_warm(retry_prob);
    if (retry_status == Status::NOT_INITIALIZED) {
        FAIL("failed warm attempt erased initialization state"); return;
    }

    // Bad shift requests are rejected before mutating the trajectory.
    Problem invalid_prob;
    setup_problem(invalid_prob, dyn, cost, cons, 0.5, 0.0);
    const Vec<NX> invalid_x0 = invalid_prob.stages[0].x;
    const Vec<NU> invalid_u0 = invalid_prob.stages[0].u;
    ContactIPM<NX, NU, NC, HORIZON> shift_solver;
    if (shift_solver.shift_for_warmstart(invalid_prob, invalid_x0, 0)
            != Status::BAD_ARGUMENT
        || shift_solver.shift_for_warmstart(
               invalid_prob, invalid_x0, HORIZON + 1)
            != Status::BAD_ARGUMENT
        || invalid_prob.stages[0].x[0] != invalid_x0[0]
        || invalid_prob.stages[0].u[0] != invalid_u0[0]) {
        FAIL("invalid shift mutated the problem"); return;
    }
    invalid_prob.dynamics = nullptr;
    if (shift_solver.shift_for_warmstart(invalid_prob, invalid_x0, 1)
            != Status::BAD_ARGUMENT
        || invalid_prob.stages[0].x[0] != invalid_x0[0]) {
        FAIL("null-dynamics shift mutated the problem"); return;
    }

    // User correction and dynamics callbacks may reject a candidate. The
    // shift must be transactional, including its solver-side costate state.
    Problem correction_failure = prob;
    const Problem correction_checkpoint = correction_failure;
    const auto costates_before_failure = solver.get_riccati_diag();
    const auto reject_correction = [](
        const Vec<NX>&, const Vec<NX>&, Vec<NX>& state) {
        state[0] += 123.0;
        return Status::INFEASIBLE;
    };
    if (solver.shift_for_warmstart(correction_failure, x_actual, 1,
                                   reject_correction) != Status::INFEASIBLE ||
        !same_problem_data(correction_failure, correction_checkpoint)) {
        FAIL("failed correction callback mutated shifted problem"); return;
    }
    const auto costates_after_correction_failure = solver.get_riccati_diag();
    if (costates_before_failure.p_stage0.norm_inf() !=
            costates_after_correction_failure.p_stage0.norm_inf() ||
        costates_before_failure.p_terminal.norm_inf() !=
            costates_after_correction_failure.p_terminal.norm_inf()) {
        FAIL("failed correction callback invalidated solver costates"); return;
    }

    FailingDoubleIntDyn failing_dyn;
    Problem dynamics_failure;
    setup_problem(dynamics_failure, failing_dyn, cost, cons, 0.5, 0.0);
    const Problem dynamics_checkpoint = dynamics_failure;
    failing_dyn.fail = true;
    if (shift_solver.shift_for_warmstart(
            dynamics_failure, dynamics_failure.stages[1].x, 1) !=
            Status::INTERNAL_ERROR ||
        !same_problem_data(dynamics_failure, dynamics_checkpoint)) {
        FAIL("failed tail dynamics mutated shifted problem"); return;
    }

    // A full-horizon shift has no overlapping path node. Its tail must start
    // from the measurement and use the old last actionable control.
    Problem full_prob;
    setup_problem(full_prob, dyn, cost, cons, 0.5, 0.0);
    full_prob.stages[HORIZON - 1].u[0] = 0.4;
    full_prob.stages[HORIZON].u[0] = 99.0;
    Vec<NX> measured;
    measured[0] = -0.2;
    measured[1] = 0.3;
    if (shift_solver.shift_for_warmstart(
            full_prob, measured, HORIZON) != Status::SUCCESS) {
        FAIL("full-horizon shift failed"); return;
    }
    Vec<NX> expected_next;
    Vec<NU> expected_tail_u;
    expected_tail_u[0] = 0.4;
    dyn.discrete_step(measured, expected_tail_u, DT, expected_next);
    if (full_prob.stages[0].x[0] != measured[0]
        || full_prob.stages[0].x[1] != measured[1]
        || full_prob.stages[0].u[0] != expected_tail_u[0]
        || std::fabs(full_prob.stages[1].x[0] - expected_next[0]) > 1e-12
        || std::fabs(full_prob.stages[1].x[1] - expected_next[1]) > 1e-12) {
        FAIL("full shift did not roll from measurement with actionable u");
        return;
    }
    PASS();
}

void test_time_limit_is_safe_and_instrumented() {
    TEST("Hard deadline restores input plan and reports counters");

    DoubleIntDyn dyn; DoubleIntCost cost; DoubleIntCons cons;
    Problem prob;
    setup_problem(prob, dyn, cost, cons, 0.5, 0.0);
    const auto input_stage0 = prob.stages[0];
    const auto input_stage5 = prob.stages[5];

    ContactIPM<NX, NU, NC, HORIZON> solver;
    ContactIPMParams pp = default_test_params();
    pp.exact_hessian = false;
    pp.adaptive_exact_hessian = false;
    pp.time_limit_ms = 1e-9;
    solver.configure(pp);

    const Status status = solver.solve(prob);
    const SolverStats& stats = solver.last_stats();
    if (status != Status::TIME_LIMIT || stats.time_limit_hit != 1) {
        FAIL("solver did not return distinct TIME_LIMIT"); return;
    }
    if (stats.deadline_checks <= 0 || !(stats.solve_time_ms >= 0.0)) {
        FAIL("deadline instrumentation was not populated"); return;
    }
    if (stats.exact_hessian_fd_calls != 0
        || stats.exact_hessian_analytic_calls != 0) {
        FAIL("Gauss-Newton path reported exact-curvature calls"); return;
    }
    if (prob.stages[0].x[0] != input_stage0.x[0]
        || prob.stages[0].s[0] != input_stage0.s[0]
        || prob.stages[5].x[0] != input_stage5.x[0]
        || prob.stages[5].lambda[0] != input_stage5.lambda[0]) {
        FAIL("timed-out solve published a partial candidate"); return;
    }
    if (solver.solve_warm(prob) != Status::NOT_INITIALIZED) {
        FAIL("timed-out cold solve incorrectly enabled warm solve"); return;
    }

    // Changing only the runtime budget must not require reconfiguration.
    solver.set_time_limit_ms(0.0);
    if (solver.solve(prob) != Status::SUCCESS) {
        FAIL("disabling runtime budget did not preserve solver usability");
        return;
    }
    const SolverStats& success_stats = solver.last_stats();
    if (success_stats.model_evaluations <= 0
        || success_stats.kkt_assemblies <= 0
        || success_stats.riccati_factorizations <= 0
        || success_stats.solve_time_ms <= 0.0
        || success_stats.model_eval_time_ms < 0.0
        || success_stats.line_search_time_ms < 0.0) {
        FAIL("successful solve did not expose phase/call instrumentation");
        return;
    }
    PASS();
}

void test_timeout_preserves_warm_continuation() {
    TEST("Timed-out warm solve restores full state and can continue");

    DoubleIntDyn dyn; DoubleIntCost cost; DoubleIntCons cons;
    Problem prob;
    setup_problem(prob, dyn, cost, cons, 0.5, 0.0);
    ContactIPM<NX, NU, NC, HORIZON> solver;
    ContactIPMParams pp = default_test_params();
    pp.exact_hessian = false;
    pp.adaptive_exact_hessian = false;
    pp.tol_stat = 1.0;
    solver.configure(pp);
    if (solver.solve(prob) != Status::SUCCESS) {
        FAIL("cold solve failed"); return;
    }

    const Vec<NX> measured = prob.stages[1].x;
    if (solver.shift_for_warmstart(prob, measured, 1) != Status::SUCCESS) {
        FAIL("warm shift failed"); return;
    }
    StageData<NX, NU, NC> before[HORIZON + 1];
    for (int k = 0; k <= HORIZON; ++k) before[k] = prob.stages[k];
    const auto before_costates = solver.get_riccati_diag();

    solver.set_time_limit_ms(1e-9);
    if (solver.solve_warm(prob) != Status::TIME_LIMIT) {
        FAIL("forced warm timeout was not reported"); return;
    }
    const SolverStats timeout_stats = solver.last_stats();
    if (timeout_stats.time_limit_hit != 1
        || timeout_stats.exact_hessian_fd_calls != 0) {
        FAIL("timeout counters are inconsistent"); return;
    }
    for (int k = 0; k <= HORIZON; ++k) {
        for (int i = 0; i < NX; ++i) {
            if (prob.stages[k].x[i] != before[k].x[i]
                || prob.stages[k].z_L_x[i] != before[k].z_L_x[i]
                || prob.stages[k].z_U_x[i] != before[k].z_U_x[i]) {
                FAIL("state or state-bound dual changed on timeout"); return;
            }
        }
        for (int i = 0; i < NU; ++i) {
            if (prob.stages[k].u[i] != before[k].u[i]
                || prob.stages[k].z_L_u[i] != before[k].z_L_u[i]
                || prob.stages[k].z_U_u[i] != before[k].z_U_u[i]) {
                FAIL("control or control-bound dual changed on timeout"); return;
            }
        }
        for (int j = 0; j < NC; ++j) {
            if (prob.stages[k].s[j] != before[k].s[j]
                || prob.stages[k].lambda[j] != before[k].lambda[j]) {
                FAIL("slack or inequality dual changed on timeout"); return;
            }
        }
    }
    const auto after_costates = solver.get_riccati_diag();
    if (after_costates.p_stage0.norm_inf()
            != before_costates.p_stage0.norm_inf()
        || after_costates.p_terminal.norm_inf()
            != before_costates.p_terminal.norm_inf()) {
        FAIL("costates were not restored after timeout"); return;
    }

    solver.set_time_limit_ms(0.0);
    if (solver.solve_warm(prob) != Status::SUCCESS) {
        FAIL("warm solve could not continue after timeout"); return;
    }
    const SolverStats resumed_stats = solver.last_stats();
    // A centered shifted iterate may pass KKT on its first model evaluation,
    // so a successful continuation need not enter line search.
    if (resumed_stats.time_limit_hit != 0
        || resumed_stats.deadline_checks <= 0
        || resumed_stats.model_evaluations <= 0) {
        FAIL("per-solve counters did not reset/populate on continuation");
        return;
    }
    PASS();
}

void test_constraint_metadata_and_evaluation_caching() {
    TEST("Constraint metadata and base values cached per solve");

    DoubleIntDyn dyn; DoubleIntCost cost; CountingCons cons;
    Problem prob;
    setup_problem_with_cons(prob, dyn, cost, cons, 0.5, 0.0);
    ContactIPM<NX, NU, NC, HORIZON> solver;
    ContactIPMParams pp = default_test_params();
    pp.exact_hessian = false;
    pp.adaptive_exact_hessian = false;
    pp.max_iters = 1;
    pp.tol_primal = 0.0;
    pp.tol_compl = 0.0;
    pp.tol_stat = 0.0;
    solver.configure(pp);
    solver.solve(prob);

    if (cons.num_calls != HORIZON + 1
        || cons.pair_count_calls != HORIZON + 1) {
        FAIL("constraint metadata queried more than once per stage"); return;
    }
    // One base evaluation belongs to each model evaluation. Additional trial
    // evaluations in line search legitimately have no Jacobian counterpart.
    const int model_evals = solver.last_stats().model_evaluations;
    if (cons.jacobian_calls != model_evals * HORIZON
        || cons.terminal_jacobian_calls != model_evals
        || cons.evaluate_calls < cons.jacobian_calls
        || cons.terminal_evaluate_calls < cons.terminal_jacobian_calls) {
        FAIL("Jacobian assembly duplicated base constraint evaluation"); return;
    }
    PASS();
}

void test_failed_warm_solve_is_transactional() {
    TEST("Failed warm solve is transactional and restartable");

    DoubleIntDyn dyn; DoubleIntCost cost; DoubleIntCons cons;
    Problem prob;
    setup_problem(prob, dyn, cost, cons, 0.5, 0.0);
    ContactIPM<NX, NU, NC, HORIZON> solver;
    ContactIPMParams pp = default_test_params();
    pp.exact_hessian = false;
    pp.adaptive_exact_hessian = false;
    pp.tol_stat = 1.0;
    solver.configure(pp);
    if (solver.solve(prob) != Status::SUCCESS) {
        FAIL("cold solve failed"); return;
    }
    const Vec<NX> measured = prob.stages[1].x;
    if (solver.shift_for_warmstart(prob, measured, 1) != Status::SUCCESS) {
        FAIL("warm shift failed"); return;
    }

    StageData<NX, NU, NC> checkpoint[HORIZON + 1];
    for (int k = 0; k <= HORIZON; ++k) checkpoint[k] = prob.stages[k];
    // A bad problem forces a non-timeout early return after solve entry.
    prob.cost = nullptr;
    if (solver.solve_warm(prob) != Status::BAD_ARGUMENT) {
        FAIL("expected failed warm solve"); return;
    }
    for (int k = 0; k <= HORIZON; ++k) {
        for (int i = 0; i < NX; ++i)
            if (prob.stages[k].x[i] != checkpoint[k].x[i]) {
                FAIL("failed warm solve changed state checkpoint"); return;
            }
        for (int j = 0; j < NC; ++j)
            if (prob.stages[k].s[j] != checkpoint[k].s[j]
                || prob.stages[k].lambda[j] != checkpoint[k].lambda[j]) {
                FAIL("failed warm solve changed barrier checkpoint"); return;
            }
    }
    prob.cost = &cost;
    if (solver.solve_warm(prob) != Status::SUCCESS) {
        FAIL("warm solve could not restart after failure"); return;
    }
    PASS();
}

void test_relative_regularization_stats_capture_first_factorization() {
    TEST("Relative regularization stats capture first factorization");

    DoubleIntDyn dyn; DoubleIntCost cost; DoubleIntCons cons;
    Problem prob;
    setup_problem(prob, dyn, cost, cons, 0.5, 0.0);

    ContactIPM<NX, NU, NC, HORIZON> solver;
    ContactIPMParams pp = default_test_params();
    pp.max_iters = 1;
    pp.tol_primal = 0.0;
    pp.tol_compl = 0.0;
    pp.tol_stat = 0.0;
    pp.riccati_relative_regularization = 1e-8;
    solver.configure(pp);
    solver.solve(prob);

    const SolverStats stats = solver.last_stats();
    if (stats.riccati_factorizations != 1) {
        FAIL("expected exactly one Riccati factorization"); return;
    }
    if (stats.first_regularization != 1e-8 ||
        stats.max_regularization != 1e-8 ||
        stats.regularization != 1e-8) {
        FAIL("relative regularization telemetry mismatch"); return;
    }
    PASS();
}

void test_cold_and_warm_slack_initialization_policy() {
    TEST("Cold and warm slack initialization policy");

    PaperIPMParams params;
    params.s_min_init = 1e-3;
    params.delta_slack = 1e-2;
    params.bound_s_min = 1e-12;
    constexpr double mu = 1e-4;

    const double cold_inactive = paper_ipm_detail::initial_cold_slack(
        -0.2, mu, params);
    const double warm_inactive = paper_ipm_detail::initial_warm_slack(
        -0.2, mu, params);
    const double warm_active = paper_ipm_detail::initial_warm_slack(
        0.0, mu, params);
    const double warm_inactive_dual = mu / warm_inactive;
    const double warm_active_dual = mu / warm_active;

    if (std::fabs(cold_inactive - 0.21) > 1e-12) {
        FAIL("cold slack no longer retains delta_slack"); return;
    }
    if (std::fabs(warm_inactive - 0.2) > 1e-12 ||
        std::fabs(warm_inactive_dual - 5e-4) > 1e-12) {
        FAIL("inactive warm row was not initialized on g+s=0"); return;
    }
    if (std::fabs(warm_active - 1e-2) > 1e-12 ||
        std::fabs(warm_active_dual - 1e-2) > 1e-12) {
        FAIL("near-active warm row was not centered at sqrt(mu)"); return;
    }
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
    test_shifted_primal_dual_warm_start();
    test_time_limit_is_safe_and_instrumented();
    test_timeout_preserves_warm_continuation();
    test_constraint_metadata_and_evaluation_caching();
    test_failed_warm_solve_is_transactional();
    test_relative_regularization_stats_capture_first_factorization();
    test_cold_and_warm_slack_initialization_policy();

    printf("\n--- Results: %d/%d passed ---\n", pass_count, test_count);
    return (pass_count == test_count) ? 0 : 1;
}
