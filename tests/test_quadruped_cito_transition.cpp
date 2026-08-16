#include <algorithm>
#include <chrono>
#include <cmath>
#include <cstdio>
#include <memory>
#include <string>

#include "examples/quadruped_cito/quadruped_cito_go1.hpp"
#include "examples/quadruped_cito/quadruped_cito_model.hpp"
#include "examples/quadruped_cito/quadruped_cito_plan.hpp"
#include "examples/quadruped_cito/quadruped_cito_plan_io.hpp"
#include "examples/quadruped_cito/quadruped_cito_terrain.hpp"
#include "nmpc/contact_ipm.hpp"

namespace {

using namespace quadruped_cito;

#ifndef QUADRUPED_CITO_TEST_HORIZON
#define QUADRUPED_CITO_TEST_HORIZON 20
#endif

#ifndef QUADRUPED_CITO_TEST_TIMESTEP
#define QUADRUPED_CITO_TEST_TIMESTEP 0.08
#endif

#ifndef QUADRUPED_CITO_TEST_TERRAIN_AMPLITUDE
#define QUADRUPED_CITO_TEST_TERRAIN_AMPLITUDE 0.025
#endif

#ifndef QUADRUPED_CITO_TEST_TERRAIN_SLOPE_X
#define QUADRUPED_CITO_TEST_TERRAIN_SLOPE_X 0.0
#endif

#ifndef QUADRUPED_CITO_TEST_TERRAIN_SLOPE_Y
#define QUADRUPED_CITO_TEST_TERRAIN_SLOPE_Y 0.0
#endif

constexpr int kHorizon = QUADRUPED_CITO_TEST_HORIZON;
constexpr double kTimeStep = QUADRUPED_CITO_TEST_TIMESTEP;
constexpr int kMovingFoot = 0;
constexpr double kRequestedStep = 0.08;
#ifdef QUADRUPED_CITO_TEST_USE_RECOVERY
constexpr const char* kSolveMode = "bounded_recovery";
#else
constexpr const char* kSolveMode = "direct";
#endif

#ifdef QUADRUPED_CITO_TEST_GO1
using PhysicalTerrain = SharedTerrain;
using TestTerrain = Go1FootCenterTerrain<PhysicalTerrain>;
#elif defined(QUADRUPED_CITO_TEST_SMOOTH_STEP)
using TestTerrain = SmoothStepTerrain;
#else
using TestTerrain = SinusoidalHeightTerrain;
#endif

void configure_cost(QuadraticTrackingCost<kHorizon>& cost) {
    cost.state_weights.set_constant(1e-3);
    cost.terminal_weights.set_constant(1.0);
    cost.control_weights.set_constant(1e-5);
    for (int axis = 0; axis < 3; ++axis) {
        cost.state_weights[StateIndex::base_position(axis)] = 20.0;
        cost.terminal_weights[StateIndex::base_position(axis)] = 200.0;
        cost.state_weights[StateIndex::linear_velocity(axis)] = 2.0;
        cost.terminal_weights[StateIndex::linear_velocity(axis)] = 20.0;
        cost.state_weights[StateIndex::angular_velocity(axis)] = 2.0;
        cost.terminal_weights[StateIndex::angular_velocity(axis)] = 20.0;
    }
    for (int element = 0; element < 4; ++element) {
        cost.state_weights[StateIndex::quaternion(element)] = 10.0;
        cost.terminal_weights[StateIndex::quaternion(element)] = 100.0;
    }
    for (int foot = 0; foot < kNumFeet; ++foot) {
        for (int axis = 0; axis < 3; ++axis) {
            cost.control_weights[ControlIndex::foot_velocity(foot, axis)] =
                0.02;
        }
        cost.control_weights[ControlIndex::motion_slack(foot)] = 0.02;
    }
    for (int axis = 0; axis < 3; ++axis) {
        cost.terminal_weights[
            StateIndex::foot_position(kMovingFoot, axis)] = 1000.0;
    }
    cost.references[kHorizon][StateIndex::foot_position(kMovingFoot, 0)] +=
        kRequestedStep;
}

template <typename Terrain>
void initialize_feasible_transition_guess(Problem<kHorizon>& problem,
                                          const RobotParameters& parameters,
                                          const Terrain& terrain) {
    double foot_path[kHorizon + 1][3];
    const double initial_x = problem.x0[
        StateIndex::foot_position(kMovingFoot, 0)];
    const double foot_y = problem.x0[
        StateIndex::foot_position(kMovingFoot, 1)];
    for (int stage = 0; stage <= kHorizon; ++stage) {
        const double raw_phase =
            (stage - 2.0) / static_cast<double>(kHorizon - 4);
        const double phase = std::max(0.0, std::min(1.0, raw_phase));
        const double smooth_phase =
            phase * phase * (3.0 - 2.0 * phase);
        Vec<3> surface_position;
        surface_position[0] = initial_x + kRequestedStep * smooth_phase;
        surface_position[1] = foot_y;
        surface_position[2] =
            terrain.height(surface_position[0], surface_position[1]);
        TerrainSample surface_sample;
        terrain.sample(surface_position, surface_sample);
        const double clearance =
            0.05 * std::sin(3.14159265358979323846 * phase);
        for (int axis = 0; axis < 3; ++axis) {
            foot_path[stage][axis] = surface_position[axis] +
                clearance * surface_sample.normal[axis];
        }
    }
    const Vec<kStateDim> standing = problem.x0;
    for (int stage = 0; stage <= kHorizon; ++stage) {
        problem.stages[stage].x = standing;
        for (int axis = 0; axis < 3; ++axis) {
            problem.stages[stage].x[
                StateIndex::foot_position(kMovingFoot, axis)] =
                foot_path[stage][axis];
        }
        problem.stages[stage].u.zero();
    }

    const double total_force = parameters.mass * parameters.gravity;
    const double base_x = problem.x0[StateIndex::base_position(0)];
    const double base_y = problem.x0[StateIndex::base_position(1)];
    const double front_x = 0.5 *
        (problem.x0[StateIndex::foot_position(0, 0)] +
         problem.x0[StateIndex::foot_position(1, 0)]);
    const double rear_x = 0.5 *
        (problem.x0[StateIndex::foot_position(2, 0)] +
         problem.x0[StateIndex::foot_position(3, 0)]);
    const double left_y = 0.5 *
        (problem.x0[StateIndex::foot_position(0, 1)] +
         problem.x0[StateIndex::foot_position(2, 1)]);
    const double right_y = 0.5 *
        (problem.x0[StateIndex::foot_position(1, 1)] +
         problem.x0[StateIndex::foot_position(3, 1)]);
    const double span_x = front_x - rear_x;
    const double span_y = left_y - right_y;
    const double positive_x_weight = (base_x - rear_x) / span_x;
    const double positive_y_weight = (base_y - right_y) / span_y;
    const double four_support_weights[kNumFeet] = {
        positive_x_weight * positive_y_weight,
        positive_x_weight * (1.0 - positive_y_weight),
        (1.0 - positive_x_weight) * positive_y_weight,
        (1.0 - positive_x_weight) * (1.0 - positive_y_weight)};
    const double three_support_weights[kNumFeet] = {
        0.0, positive_x_weight, positive_y_weight,
        1.0 - positive_x_weight - positive_y_weight};
    const double landing_foot_weight = 0.10;
    const double landing_foot_x = foot_path[kHorizon][0];
    const double landing_support1_weight = positive_x_weight -
        (landing_foot_x - rear_x) * landing_foot_weight / span_x;
    const double landing_support2_weight =
        positive_y_weight - landing_foot_weight;
    const double landing_weights[kNumFeet] = {
        landing_foot_weight, landing_support1_weight,
        landing_support2_weight,
        1.0 - landing_foot_weight - landing_support1_weight -
            landing_support2_weight};
    for (int stage = 0; stage < kHorizon; ++stage) {
        const double* support_weights = stage < 2
            ? four_support_weights
            : (stage >= kHorizon - 2
                   ? landing_weights
                   : three_support_weights);
        for (int foot = 0; foot < kNumFeet; ++foot) {
            problem.stages[stage].u[
                ControlIndex::contact_force(foot, 2)] =
                total_force * support_weights[foot];
        }
        double speed_sq = 0.0;
        for (int axis = 0; axis < 3; ++axis) {
            const double velocity =
                (foot_path[stage + 1][axis] - foot_path[stage][axis]) /
                kTimeStep;
            problem.stages[stage].u[
                ControlIndex::foot_velocity(kMovingFoot, axis)] = velocity;
            speed_sq += velocity * velocity;
        }
        problem.stages[stage].u[
            ControlIndex::motion_slack(kMovingFoot)] =
            std::sqrt(speed_sq);
    }
}

double dynamics_defect(const Problem<kHorizon>& problem,
                       SRBDDynamics& dynamics, int* worst_stage = nullptr,
                       int* worst_state = nullptr) {
    double maximum = 0.0;
    for (int stage = 0; stage < kHorizon; ++stage) {
        Vec<kStateDim> predicted;
        dynamics.discrete_step(problem.stages[stage].x,
                               problem.stages[stage].u, problem.dt,
                               predicted);
        for (int state = 0; state < kStateDim; ++state) {
            const double defect = std::fabs(
                predicted[state] - problem.stages[stage + 1].x[state]);
            if (defect > maximum) {
                maximum = defect;
                if (worst_stage) *worst_stage = stage;
                if (worst_state) *worst_state = state;
            }
        }
    }
    return maximum;
}

}  // namespace

int main(int argc, char** argv) {
    const bool write_plan = argc == 3;
    if ((argc != 1 && !write_plan) ||
        (write_plan && std::string(argv[1]) != "--write-plan")) {
        std::printf("usage: test_quadruped_cito_transition "
                    "[--write-plan <path>]\n");
        return 2;
    }
    RobotParameters parameters;
#ifdef QUADRUPED_CITO_TEST_GO1
    parameters = go1_robot_parameters();
#ifdef QUADRUPED_CITO_TEST_GO1_SLOPE
    PhysicalTerrain physical_terrain = register_go1_terrain(
        SharedTerrain::slope(0.10));
#elif defined(QUADRUPED_CITO_TEST_GO1_SMOOTH)
    PhysicalTerrain physical_terrain = register_go1_terrain(
        SharedTerrain::sinusoidal());
#else
    PhysicalTerrain physical_terrain = register_go1_terrain(
        SharedTerrain::flat());
#endif
    TestTerrain terrain(physical_terrain);
#else
    TestTerrain terrain;
#ifdef QUADRUPED_CITO_TEST_SMOOTH_STEP
    terrain.step_height = 0.04;
    terrain.step_center_x = 0.34;
    terrain.sharpness = 30.0;
#else
    terrain.amplitude = QUADRUPED_CITO_TEST_TERRAIN_AMPLITUDE;
    terrain.slope_x = QUADRUPED_CITO_TEST_TERRAIN_SLOPE_X;
    terrain.slope_y = QUADRUPED_CITO_TEST_TERRAIN_SLOPE_Y;
    terrain.wave_number_x = 4.0;
    terrain.wave_number_y = 3.0;
#endif
#endif
    SRBDDynamics dynamics(parameters);
    ContactConstraints<kHorizon, TestTerrain> constraints(
        terrain, parameters);
    QuadraticTrackingCost<kHorizon> cost;
    auto problem_storage = std::make_unique<Problem<kHorizon>>();
    Problem<kHorizon>& problem = *problem_storage;
    initialize_standing_problem(problem, dynamics, cost, constraints,
                                kTimeStep);
    Vec<kStateDim> terrain_standing = problem.x0;
#ifdef QUADRUPED_CITO_TEST_GO1
    terrain_standing = go1_home_state(terrain);
#else
    terrain_standing[StateIndex::base_position(0)] = -0.02;
    terrain_standing[StateIndex::base_position(1)] = -0.01;
    for (int foot = 0; foot < kNumFeet; ++foot) {
        const double x = terrain_standing[
            StateIndex::foot_position(foot, 0)];
        const double y = terrain_standing[
            StateIndex::foot_position(foot, 1)];
        terrain_standing[StateIndex::foot_position(foot, 2)] =
            terrain.height(x, y);
    }
#endif
    problem.x0 = terrain_standing;
    cost.set_reference_all(terrain_standing);
    for (int stage = 0; stage <= kHorizon; ++stage)
        problem.stages[stage].x = terrain_standing;
    configure_cost(cost);
    cost.references[kHorizon][
        StateIndex::foot_position(kMovingFoot, 2)] =
        terrain.height(
            terrain_standing[
                StateIndex::foot_position(kMovingFoot, 0)] +
                kRequestedStep,
            terrain_standing[
                StateIndex::foot_position(kMovingFoot, 1)]);
    initialize_feasible_transition_guess(problem, parameters, terrain);

    double initial_inequality_violation = 0.0;
    double initial_complementarity = 0.0;
    int initial_worst_stage = -1;
    int initial_worst_row = -1;
    for (int stage = 0; stage < kHorizon; ++stage) {
        Vec<kConstraintCapacity> rows;
        constraints.evaluate(problem.stages[stage].x,
                             problem.stages[stage].u, stage, rows);
        for (int row = 0; row < kUserConstraintRows; ++row) {
            if (rows[row] > initial_inequality_violation) {
                initial_inequality_violation = rows[row];
                initial_worst_stage = stage;
                initial_worst_row = row;
            }
        }
        for (int pair = 0; pair < kComplementarityPairs; ++pair) {
            int first = -1;
            int second = -1;
            constraints.complementarity_pair(stage, pair, first, second);
            initial_complementarity = std::max(
                initial_complementarity,
                std::fabs(rows[first] * rows[second]));
        }
    }
    int initial_worst_dynamics_stage = -1;
    int initial_worst_dynamics_state = -1;
    const double initial_dynamics_defect = dynamics_defect(
        problem, dynamics, &initial_worst_dynamics_stage,
        &initial_worst_dynamics_state);
    if (initial_dynamics_defect > 1e-12 ||
        initial_inequality_violation > 1e-12 ||
        initial_complementarity > 1e-12) {
        std::printf(
            "transition seed is not feasible: dynamics=%.3e "
            "(stage=%d, state=%d), "
            "inequality=%.3e (stage=%d, row=%d), mpcc=%.3e\n",
            initial_dynamics_defect, initial_worst_dynamics_stage,
            initial_worst_dynamics_state, initial_inequality_violation,
            initial_worst_stage, initial_worst_row,
            initial_complementarity);
        return 1;
    }

    nmpc::ContactIPMParams solver_parameters;
    solver_parameters.mu_init = 0.1;
    solver_parameters.mu_min = 1e-5;
    solver_parameters.mu_conv_threshold = 1e-5;
    solver_parameters.max_same_mu = 100;
    solver_parameters.max_iters = 500;
    solver_parameters.mpcc_recovery_max_iters = 500;
    solver_parameters.s_min_init = 0.1;
    solver_parameters.tol_primal = 2e-5;
    solver_parameters.tol_compl = 2e-5;
    solver_parameters.tol_ineq = 1e-6;
    solver_parameters.tol_stat = 0.2;
    solver_parameters.tol_mpcc = 1e-4;
    solver_parameters.exact_hessian = false;
    solver_parameters.enable_preconditioner = true;
    solver_parameters.verbosity = 0;

    nmpc::ContactIPM<kStateDim, kControlDim, kConstraintCapacity, kHorizon>
        solver;
    solver.configure(solver_parameters);
    const auto solve_start = std::chrono::steady_clock::now();
#ifdef QUADRUPED_CITO_TEST_USE_RECOVERY
    const nmpc::Status status = solver.solve_mpcc_with_recovery(problem);
#else
    const nmpc::Status status = solver.solve(problem);
#endif
    const auto solve_end = std::chrono::steady_clock::now();
    const double solve_milliseconds =
        std::chrono::duration<double, std::milli>(solve_end - solve_start)
            .count();
    const auto& stats = solver.last_stats();
    if (status != nmpc::Status::SUCCESS) {
        double returned_inequality = 0.0;
        double returned_mpcc = 0.0;
        int returned_inequality_stage = -1;
        int returned_inequality_row = -1;
        int returned_mpcc_stage = -1;
        int returned_mpcc_pair = -1;
        for (int stage = 0; stage < kHorizon; ++stage) {
            Vec<kConstraintCapacity> rows;
            constraints.evaluate(problem.stages[stage].x,
                                 problem.stages[stage].u, stage, rows);
            for (int row = 0; row < kUserConstraintRows; ++row) {
                if (rows[row] > returned_inequality) {
                    returned_inequality = rows[row];
                    returned_inequality_stage = stage;
                    returned_inequality_row = row;
                }
            }
            for (int pair = 0; pair < kComplementarityPairs; ++pair) {
                int first = -1;
                int second = -1;
                constraints.complementarity_pair(
                    stage, pair, first, second);
                const double product =
                    std::fabs(rows[first] * rows[second]);
                if (product > returned_mpcc) {
                    returned_mpcc = product;
                    returned_mpcc_stage = stage;
                    returned_mpcc_pair = pair;
                }
            }
        }
        int returned_dynamics_stage = -1;
        int returned_dynamics_state = -1;
        const double returned_dynamics = dynamics_defect(
            problem, dynamics, &returned_dynamics_stage,
            &returned_dynamics_state);
        std::printf(
            "quadruped transition solve failed: horizon=%d, mode=%s, %s, "
            "iterations=%d, solve_ms=%.3f, "
            "primal=%.3e, stationarity=%.3e, mpcc=%.3e; "
            "audit dynamics=%.3e (stage=%d,state=%d), "
            "inequality=%.3e (stage=%d,row=%d), "
            "mpcc=%.3e (stage=%d,pair=%d)\n",
            kHorizon, kSolveMode, nmpc::status_string(status),
            stats.inner_iterations, solve_milliseconds,
            stats.primal_infeas, stats.dual_infeas,
            stats.mpcc_complementarity, returned_dynamics,
            returned_dynamics_stage, returned_dynamics_state,
            returned_inequality, returned_inequality_stage,
            returned_inequality_row, returned_mpcc,
            returned_mpcc_stage, returned_mpcc_pair);
        return 1;
    }

    ContactPlan<kHorizon> plan;
    ContactClassification classification;
    if (extract_contact_plan(problem, terrain, classification, plan) !=
        nmpc::Status::SUCCESS) {
        std::printf("quadruped transition plan extraction failed\n");
        return 1;
    }

    double maximum_inequality_violation = 0.0;
    double maximum_complementarity = 0.0;
    double maximum_gap = 0.0;
    double maximum_speed_when_unloaded = 0.0;
    int moving_unloaded_stages = 0;
    for (int stage = 0; stage < kHorizon; ++stage) {
        Vec<kConstraintCapacity> rows;
        constraints.evaluate(problem.stages[stage].x,
                             problem.stages[stage].u, stage, rows);
        for (int row = 0; row < kUserConstraintRows; ++row) {
            maximum_inequality_violation =
                std::max(maximum_inequality_violation, rows[row]);
        }
        for (int pair = 0; pair < kComplementarityPairs; ++pair) {
            int first = -1;
            int second = -1;
            constraints.complementarity_pair(stage, pair, first, second);
            maximum_complementarity = std::max(
                maximum_complementarity,
                std::fabs(rows[first] * rows[second]));
        }
        const Vec<3> foot_position = state_vector3(
            problem.stages[stage].x,
            StateIndex::foot_position(kMovingFoot, 0));
        const Vec<3> force = control_vector3(
            problem.stages[stage].u,
            ControlIndex::contact_force(kMovingFoot, 0));
        const Vec<3> foot_velocity = control_vector3(
            problem.stages[stage].u,
            ControlIndex::foot_velocity(kMovingFoot, 0));
        TerrainSample terrain_sample;
        terrain.sample(foot_position, terrain_sample);
        const double gap = terrain_sample.gap;
        const double normal_force = dot3(terrain_sample.normal, force);
        const double speed = std::sqrt(foot_velocity.norm2_sq());
        maximum_gap = std::max(maximum_gap, gap);
        if (!plan.stages[stage].feet[kMovingFoot].planned_contact &&
            normal_force < classification.minimum_contact_force &&
            speed > 0.01) {
            ++moving_unloaded_stages;
            maximum_speed_when_unloaded =
                std::max(maximum_speed_when_unloaded, speed);
        }
    }

    const double initial_x = problem.x0[
        StateIndex::foot_position(kMovingFoot, 0)];
    const double final_x = problem.stages[kHorizon].x[
        StateIndex::foot_position(kMovingFoot, 0)];
    const double displacement = final_x - initial_x;
    const double defect = dynamics_defect(problem, dynamics);
    if (defect > solver_parameters.tol_primal ||
        maximum_inequality_violation > solver_parameters.tol_ineq ||
        maximum_complementarity > solver_parameters.tol_mpcc ||
        displacement < 0.06 || maximum_gap < 0.02 ||
        moving_unloaded_stages == 0) {
        std::printf(
            "quadruped transition audit failed: displacement=%.3f, "
            "gap=%.3f, unloaded_moving=%d, speed=%.3f, dynamics=%.3e, "
            "inequality=%.3e, mpcc=%.3e\n",
            displacement, maximum_gap, moving_unloaded_stages,
            maximum_speed_when_unloaded, defect,
            maximum_inequality_violation, maximum_complementarity);
        return 1;
    }
    if (write_plan && write_contact_plan(plan, argv[2]) !=
                          nmpc::Status::SUCCESS) {
        std::printf("failed to write quadruped transition plan to %s\n",
                    argv[2]);
        return 1;
    }

    std::printf(
        "Quadruped CITO transition passed: horizon=%d, duration=%.2f s, "
        "mode=%s, problem_kib=%.1f, "
        "displacement=%.3f m, "
        "clearance=%.3f m, unloaded_moving_stages=%d, iterations=%d, "
        "solve_ms=%.3f, dynamics=%.3e, inequality=%.3e, mpcc=%.3e\n",
        kHorizon, kHorizon * kTimeStep, kSolveMode,
        sizeof(Problem<kHorizon>) / 1024.0, displacement, maximum_gap,
        moving_unloaded_stages,
        stats.inner_iterations, solve_milliseconds, defect,
        maximum_inequality_violation, maximum_complementarity);
    return 0;
}
