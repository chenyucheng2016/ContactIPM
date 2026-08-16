#include <algorithm>
#include <cmath>
#include <cstdio>

#include "examples/quadruped_cito/quadruped_cito_model.hpp"
#include "nmpc/contact_ipm.hpp"

namespace {

using namespace quadruped_cito;

constexpr int kHorizon = 2;

void configure_tracking_cost(QuadraticTrackingCost<kHorizon>& cost) {
    cost.state_weights.set_constant(1e-3);
    cost.terminal_weights.set_constant(1e-2);
    cost.control_weights.set_constant(1e-5);
    for (int axis = 0; axis < 3; ++axis) {
        cost.state_weights[StateIndex::base_position(axis)] = 10.0;
        cost.terminal_weights[StateIndex::base_position(axis)] = 100.0;
        cost.state_weights[StateIndex::linear_velocity(axis)] = 1.0;
        cost.terminal_weights[StateIndex::linear_velocity(axis)] = 10.0;
        cost.state_weights[StateIndex::angular_velocity(axis)] = 1.0;
        cost.terminal_weights[StateIndex::angular_velocity(axis)] = 10.0;
    }
    for (int element = 0; element < 4; ++element) {
        cost.state_weights[StateIndex::quaternion(element)] = 5.0;
        cost.terminal_weights[StateIndex::quaternion(element)] = 50.0;
    }
    for (int foot = 0; foot < kNumFeet; ++foot) {
        for (int axis = 0; axis < 3; ++axis) {
            cost.control_weights[ControlIndex::foot_velocity(foot, axis)] =
                1e-2;
        }
        cost.control_weights[ControlIndex::motion_slack(foot)] = 1e-2;
    }
}

double independent_dynamics_defect(const Problem<kHorizon>& problem,
                                   SRBDDynamics& dynamics) {
    double maximum = 0.0;
    for (int stage = 0; stage < kHorizon; ++stage) {
        Vec<kStateDim> predicted;
        dynamics.discrete_step(problem.stages[stage].x,
                               problem.stages[stage].u, problem.dt,
                               predicted);
        for (int state = 0; state < kStateDim; ++state) {
            maximum = std::max(
                maximum,
                std::fabs(predicted[state] -
                          problem.stages[stage + 1].x[state]));
        }
    }
    return maximum;
}

void independent_contact_audit(
    const Problem<kHorizon>& problem,
    ContactConstraints<kHorizon>& constraints, double& inequality_violation,
    double& complementarity_product) {
    inequality_violation = 0.0;
    complementarity_product = 0.0;
    for (int stage = 0; stage < kHorizon; ++stage) {
        Vec<kConstraintCapacity> rows;
        constraints.evaluate(problem.stages[stage].x,
                             problem.stages[stage].u, stage, rows);
        for (int row = 0; row < kUserConstraintRows; ++row)
            inequality_violation = std::max(inequality_violation, rows[row]);
        for (int pair = 0; pair < kComplementarityPairs; ++pair) {
            int first = -1;
            int second = -1;
            constraints.complementarity_pair(stage, pair, first, second);
            complementarity_product = std::max(
                complementarity_product,
                std::fabs(rows[first] * rows[second]));
        }
    }
}

}  // namespace

int main() {
    RobotParameters parameters;
    PlaneTerrain terrain;
    SRBDDynamics dynamics(parameters);
    ContactConstraints<kHorizon> constraints(terrain, parameters);
    QuadraticTrackingCost<kHorizon> cost;
    Problem<kHorizon> problem;
    initialize_standing_problem(problem, dynamics, cost, constraints, 0.03);
    configure_tracking_cost(cost);

    nmpc::ContactIPMParams solver_parameters;
    solver_parameters.mu_init = 1e-3;
    solver_parameters.mu_min = 1e-5;
    solver_parameters.mu_conv_threshold = 1e-5;
    solver_parameters.max_same_mu = 20;
    solver_parameters.max_iters = 150;
    solver_parameters.tol_primal = 1e-5;
    solver_parameters.tol_compl = 1e-5;
    solver_parameters.tol_ineq = 1e-7;
    solver_parameters.tol_stat = 0.1;
    solver_parameters.tol_mpcc = 1e-4;
    solver_parameters.exact_hessian = true;
    solver_parameters.enable_preconditioner = true;
    solver_parameters.verbosity = 0;

    nmpc::ContactIPM<kStateDim, kControlDim, kConstraintCapacity, kHorizon>
        solver;
    if (solver.configure(solver_parameters) != nmpc::Status::SUCCESS) {
        std::printf("quadruped solver configuration failed\n");
        return 1;
    }
    const nmpc::Status status = solver.solve(problem);
    if (status != nmpc::Status::SUCCESS) {
        const auto& stats = solver.last_stats();
        std::printf(
            "quadruped standing solve failed: %s, iterations=%d, "
            "primal=%.3e, stationarity=%.3e, mpcc=%.3e\n",
            nmpc::status_string(status), stats.inner_iterations,
            stats.primal_infeas, stats.dual_infeas,
            stats.mpcc_complementarity);
        return 1;
    }

    const double dynamics_defect =
        independent_dynamics_defect(problem, dynamics);
    double inequality_violation = 0.0;
    double complementarity_product = 0.0;
    independent_contact_audit(problem, constraints, inequality_violation,
                              complementarity_product);
    if (dynamics_defect > solver_parameters.tol_primal ||
        inequality_violation > solver_parameters.tol_ineq ||
        complementarity_product > solver_parameters.tol_mpcc) {
        std::printf(
            "quadruped standing audit failed: dynamics=%.3e, "
            "inequality=%.3e, mpcc=%.3e\n",
            dynamics_defect, inequality_violation,
            complementarity_product);
        return 1;
    }

    const auto& stats = solver.last_stats();
    std::printf(
        "Quadruped CITO standing solve passed: iterations=%d, "
        "dynamics=%.3e, inequality=%.3e, mpcc=%.3e\n",
        stats.inner_iterations, dynamics_defect, inequality_violation,
        complementarity_product);
    return 0;
}
