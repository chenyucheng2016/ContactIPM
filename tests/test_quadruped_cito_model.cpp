#include <algorithm>
#include <cmath>
#include <cstdio>

#include "examples/quadruped_cito/quadruped_cito_model.hpp"
#include "examples/quadruped_cito/quadruped_cito_terrain.hpp"

namespace {

using namespace quadruped_cito;

constexpr int kHorizon = 3;

double max_state_difference(const Vec<kStateDim>& first,
                            const Vec<kStateDim>& second) {
    double error = 0.0;
    for (int i = 0; i < kStateDim; ++i)
        error = std::max(error, std::fabs(first[i] - second[i]));
    return error;
}

Vec<kStateDim> derivative_test_state() {
    Vec<kStateDim> state = standing_state(0.38);
    state[StateIndex::base_position(0)] = 0.12;
    state[StateIndex::base_position(1)] = -0.07;
    state[StateIndex::quaternion(0)] = 0.97;
    state[StateIndex::quaternion(1)] = 0.12;
    state[StateIndex::quaternion(2)] = -0.06;
    state[StateIndex::quaternion(3)] = 0.08;
    state[StateIndex::linear_velocity(0)] = 0.22;
    state[StateIndex::linear_velocity(1)] = -0.13;
    state[StateIndex::linear_velocity(2)] = 0.04;
    state[StateIndex::angular_velocity(0)] = 0.31;
    state[StateIndex::angular_velocity(1)] = -0.23;
    state[StateIndex::angular_velocity(2)] = 0.17;
    for (int foot = 0; foot < kNumFeet; ++foot) {
        state[StateIndex::foot_position(foot, 0)] += 0.01 * (foot + 1);
        state[StateIndex::foot_position(foot, 1)] -= 0.005 * foot;
        state[StateIndex::foot_position(foot, 2)] = 0.015 * foot;
    }
    return state;
}

Vec<kControlDim> derivative_test_control() {
    Vec<kControlDim> control;
    control.zero();
    for (int foot = 0; foot < kNumFeet; ++foot) {
        control[ControlIndex::contact_force(foot, 0)] = 3.0 - 0.4 * foot;
        control[ControlIndex::contact_force(foot, 1)] = -1.0 + 0.3 * foot;
        control[ControlIndex::contact_force(foot, 2)] = 31.0 + 2.0 * foot;
        control[ControlIndex::foot_velocity(foot, 0)] = 0.06 * (foot + 1);
        control[ControlIndex::foot_velocity(foot, 1)] = -0.03 * (foot + 1);
        control[ControlIndex::foot_velocity(foot, 2)] = 0.02 * foot;
        control[ControlIndex::motion_slack(foot)] = 0.5 + 0.1 * foot;
    }
    return control;
}

double relative_derivative_error(double finite_difference,
                                 double analytic) {
    return std::fabs(finite_difference - analytic) /
        (1.0 + std::max(std::fabs(finite_difference), std::fabs(analytic)));
}

int check_shared_terrain_derivatives(const SharedTerrain& terrain,
                                     const char* name,
                                     const Vec<3>& position) {
    TerrainSample center;
    if (!terrain.sample(position, center)) {
        std::printf("%s terrain sample failed\n", name);
        return 1;
    }
    const double expected_gap = position[2] -
        terrain.height(position[0], position[1]);
    if (std::fabs(center.gap - expected_gap) > 1e-13) {
        std::printf("%s terrain height/sample mismatch %.3e\n", name,
                    std::fabs(center.gap - expected_gap));
        return 1;
    }

    Vec<3> direction;
    direction[0] = 0.37;
    direction[1] = -0.29;
    direction[2] = 0.41;
    constexpr double epsilon = 1e-6;
    Vec<3> plus_position = position;
    Vec<3> minus_position = position;
    for (int axis = 0; axis < 3; ++axis) {
        plus_position[axis] += epsilon * direction[axis];
        minus_position[axis] -= epsilon * direction[axis];
    }
    TerrainSample plus;
    TerrainSample minus;
    if (!terrain.sample(plus_position, plus) ||
        !terrain.sample(minus_position, minus)) {
        std::printf("%s perturbed terrain sample failed\n", name);
        return 1;
    }

    double analytic_gap = 0.0;
    for (int axis = 0; axis < 3; ++axis)
        analytic_gap += center.gap_gradient[axis] * direction[axis];
    double maximum_error = relative_derivative_error(
        (plus.gap - minus.gap) / (2.0 * epsilon), analytic_gap);
    for (int vector_axis = 0; vector_axis < 3; ++vector_axis) {
        double analytic_normal = 0.0;
        double analytic_tangent1 = 0.0;
        double analytic_tangent2 = 0.0;
        for (int coordinate = 0; coordinate < 3; ++coordinate) {
            analytic_normal += center.normal_gradient(
                vector_axis, coordinate) * direction[coordinate];
            analytic_tangent1 += center.tangent1_gradient(
                vector_axis, coordinate) * direction[coordinate];
            analytic_tangent2 += center.tangent2_gradient(
                vector_axis, coordinate) * direction[coordinate];
        }
        maximum_error = std::max(maximum_error, relative_derivative_error(
            (plus.normal[vector_axis] - minus.normal[vector_axis]) /
                (2.0 * epsilon),
            analytic_normal));
        maximum_error = std::max(maximum_error, relative_derivative_error(
            (plus.tangent1[vector_axis] - minus.tangent1[vector_axis]) /
                (2.0 * epsilon),
            analytic_tangent1));
        maximum_error = std::max(maximum_error, relative_derivative_error(
            (plus.tangent2[vector_axis] - minus.tangent2[vector_axis]) /
                (2.0 * epsilon),
            analytic_tangent2));
    }
    if (maximum_error > 2e-8) {
        std::printf("%s terrain directional derivative error %.3e\n", name,
                    maximum_error);
        return 1;
    }
    return 0;
}

int check_shared_contact_derivatives(const SharedTerrain& terrain,
                                     const char* name) {
    RobotParameters parameters;
    parameters.minimum_pair_normal_force = 5.0;
    ContactConstraints<kHorizon, SharedTerrain> constraints(terrain,
                                                            parameters);
    Vec<kStateDim> state = derivative_test_state();
    Vec<kControlDim> control = derivative_test_control();
    for (int foot = 0; foot < kNumFeet; ++foot) {
        const double x = -0.09 + 0.11 * foot;
        const double y = (foot % 2 == 0 ? 0.13 : -0.12) + 0.01 * foot;
        state[StateIndex::foot_position(foot, 0)] = x;
        state[StateIndex::foot_position(foot, 1)] = y;
        state[StateIndex::foot_position(foot, 2)] =
            terrain.height(x, y) + 0.012 * (foot + 1);
    }

    Mat<kConstraintCapacity, kStateDim> Cx;
    Mat<kConstraintCapacity, kControlDim> Cu;
    if (constraints.jacobian(state, control, 0, Cx, Cu) !=
        nmpc::Status::SUCCESS) {
        std::printf("%s contact Jacobian failed\n", name);
        return 1;
    }
    Vec<kStateDim> state_direction;
    Vec<kControlDim> control_direction;
    for (int index = 0; index < kStateDim; ++index) {
        state_direction[index] = (index % 2 == 0 ? 0.17 : -0.11) +
            0.003 * index;
    }
    for (int index = 0; index < kControlDim; ++index) {
        control_direction[index] = (index % 3 == 0 ? -0.23 : 0.14) -
            0.002 * index;
    }

    constexpr double epsilon = 1e-6;
    double maximum_error = 0.0;
    int worst_row = -1;
    int worst_direction = -1;
    for (int direction_kind = 0; direction_kind < 2; ++direction_kind) {
        Vec<kStateDim> state_plus = state;
        Vec<kStateDim> state_minus = state;
        Vec<kControlDim> control_plus = control;
        Vec<kControlDim> control_minus = control;
        if (direction_kind == 0) {
            for (int index = 0; index < kStateDim; ++index) {
                state_plus[index] += epsilon * state_direction[index];
                state_minus[index] -= epsilon * state_direction[index];
            }
        } else {
            for (int index = 0; index < kControlDim; ++index) {
                control_plus[index] += epsilon * control_direction[index];
                control_minus[index] -= epsilon * control_direction[index];
            }
        }
        Vec<kConstraintCapacity> rows_plus;
        Vec<kConstraintCapacity> rows_minus;
        if (constraints.evaluate(state_plus, control_plus, 0, rows_plus) !=
                nmpc::Status::SUCCESS ||
            constraints.evaluate(state_minus, control_minus, 0, rows_minus) !=
                nmpc::Status::SUCCESS) {
            std::printf("%s perturbed contact evaluation failed\n", name);
            return 1;
        }
        for (int row = 0; row < constraints.num_constraints(0); ++row) {
            double analytic = 0.0;
            if (direction_kind == 0) {
                for (int column = 0; column < kStateDim; ++column)
                    analytic += Cx(row, column) * state_direction[column];
            } else {
                for (int column = 0; column < kControlDim; ++column)
                    analytic += Cu(row, column) * control_direction[column];
            }
            const double finite_difference =
                (rows_plus[row] - rows_minus[row]) / (2.0 * epsilon);
            const double error = relative_derivative_error(
                finite_difference, analytic);
            if (error > maximum_error) {
                maximum_error = error;
                worst_row = row;
                worst_direction = direction_kind;
            }
        }
    }
    if (maximum_error > 2e-7) {
        std::printf("%s contact directional derivative error %.3e "
                    "(row=%d, direction=%s)\n",
                    name, maximum_error, worst_row,
                    worst_direction == 0 ? "state" : "control");
        return 1;
    }
    return 0;
}

int test_shared_terrain_and_contact_derivatives() {
    SharedTerrain terrains[5] = {
        SharedTerrain::flat(),
        SharedTerrain::slope(0.11, -0.07),
        SharedTerrain::sinusoidal(0.04, 4.2, 3.1),
        SharedTerrain::smooth_step(0.04, 0.14, 18.0),
        SharedTerrain::random_smooth(17, 0.04)};
    const char* names[5] = {
        "flat", "slope", "sinusoidal", "smooth_step", "random_smooth"};
    for (int terrain_index = 0; terrain_index < 5; ++terrain_index) {
        terrains[terrain_index].offset = 0.013;
        if (terrain_index >= 2) {
            terrains[terrain_index].slope_x += 0.025;
            terrains[terrain_index].slope_y -= 0.018;
        }
        Vec<3> position;
        position[0] = terrain_index == 3 ? 0.153 : 0.17;
        position[1] = -0.11;
        position[2] = terrains[terrain_index].height(
            position[0], position[1]) + 0.037;
        if (check_shared_terrain_derivatives(
                terrains[terrain_index], names[terrain_index], position) != 0 ||
            check_shared_contact_derivatives(
                terrains[terrain_index], names[terrain_index]) != 0) {
            return 1;
        }
    }
    return 0;
}

struct CountingTerrain {
    static constexpr bool kAffineFrame = false;
    SharedTerrain terrain = SharedTerrain::random_smooth(7, 0.03);
    mutable int samples = 0;

    bool sample(const Vec<3>& position, TerrainSample& result) const {
        ++samples;
        return terrain.sample(position, result);
    }
};

int test_contact_terrain_sample_reuse() {
    RobotParameters parameters;
    parameters.minimum_pair_normal_force = 5.0;
    CountingTerrain terrain;
    ContactConstraints<kHorizon, CountingTerrain> constraints(terrain,
                                                              parameters);
    const Vec<kStateDim> state = derivative_test_state();
    const Vec<kControlDim> control = derivative_test_control();
    Vec<kConstraintCapacity> rows;
    if (constraints.evaluate(state, control, 0, rows) !=
            nmpc::Status::SUCCESS ||
        terrain.samples != kNumFeet) {
        std::printf("constraint evaluation terrain samples %d (expected %d)\n",
                    terrain.samples, kNumFeet);
        return 1;
    }
    terrain.samples = 0;
    Mat<kConstraintCapacity, kStateDim> Cx;
    Mat<kConstraintCapacity, kControlDim> Cu;
    if (constraints.jacobian(state, control, 0, Cx, Cu) !=
            nmpc::Status::SUCCESS ||
        terrain.samples != kNumFeet) {
        std::printf("constraint Jacobian terrain samples %d (expected %d)\n",
                    terrain.samples, kNumFeet);
        return 1;
    }
    terrain.samples = 0;
    if (constraints.evaluate_with_jacobian(
            state, control, 0, rows, Cx, Cu) != nmpc::Status::SUCCESS ||
        terrain.samples != kNumFeet) {
        std::printf("fused constraint terrain samples %d (expected %d)\n",
                    terrain.samples, kNumFeet);
        return 1;
    }
    return 0;
}

int test_standing_equilibrium() {
    RobotParameters parameters;
    SRBDDynamics dynamics(parameters);
    PlaneTerrain terrain;
    ContactConstraints<kHorizon> constraints(terrain, parameters);
    const Vec<kStateDim> state = standing_state();
    const Vec<kControlDim> control = standing_control(parameters);

    Vec<kStateDim> next;
    if (dynamics.discrete_step(state, control, 0.02, next) !=
        nmpc::Status::SUCCESS) {
        std::printf("standing dynamics returned failure\n");
        return 1;
    }
    const double dynamics_error = max_state_difference(state, next);
    if (dynamics_error > 1e-12) {
        std::printf("standing dynamics defect %.3e\n", dynamics_error);
        return 1;
    }

    Vec<kConstraintCapacity> rows;
    constraints.evaluate(state, control, 0, rows);
    double maximum_violation = 0.0;
    for (int row = 0; row < kUserConstraintRows; ++row)
        maximum_violation = std::max(maximum_violation, rows[row]);
    if (maximum_violation > 1e-12) {
        std::printf("standing constraint violation %.3e\n",
                    maximum_violation);
        return 1;
    }

    double maximum_product = 0.0;
    for (int pair = 0; pair < kComplementarityPairs; ++pair) {
        int first = -1;
        int second = -1;
        if (!constraints.complementarity_pair(0, pair, first, second))
            return 1;
        maximum_product = std::max(
            maximum_product, std::fabs(rows[first] * rows[second]));
    }
    if (maximum_product > 1e-12) {
        std::printf("standing complementarity product %.3e\n",
                    maximum_product);
        return 1;
    }
    if (constraints.num_constraints(0) != kContactConstraintRows ||
        constraints.num_constraints(0) +
                constraints.num_complementarity_pairs(0) >
            kConstraintCapacity) {
        std::printf("contact constraint capacity mismatch\n");
        return 1;
    }
    return 0;
}

int test_dynamics_jacobian() {
    RobotParameters parameters;
    SRBDDynamics dynamics(parameters);
    const Vec<kStateDim> state = derivative_test_state();
    const Vec<kControlDim> control = derivative_test_control();
    constexpr double dt = 0.027;
    constexpr double epsilon = 1e-6;

    Mat<kStateDim, kStateDim> A;
    Mat<kStateDim, kControlDim> B;
    if (dynamics.linearize(state, control, dt, A, B) !=
        nmpc::Status::SUCCESS) {
        std::printf("dynamics linearization returned failure\n");
        return 1;
    }

    double maximum_error = 0.0;
    for (int column = 0; column < kStateDim; ++column) {
        Vec<kStateDim> plus = state;
        Vec<kStateDim> minus = state;
        Vec<kStateDim> f_plus;
        Vec<kStateDim> f_minus;
        plus[column] += epsilon;
        minus[column] -= epsilon;
        dynamics.discrete_step(plus, control, dt, f_plus);
        dynamics.discrete_step(minus, control, dt, f_minus);
        for (int row = 0; row < kStateDim; ++row) {
            const double finite_difference =
                (f_plus[row] - f_minus[row]) / (2.0 * epsilon);
            maximum_error = std::max(
                maximum_error, std::fabs(finite_difference - A(row, column)));
        }
    }
    for (int column = 0; column < kControlDim; ++column) {
        Vec<kControlDim> plus = control;
        Vec<kControlDim> minus = control;
        Vec<kStateDim> f_plus;
        Vec<kStateDim> f_minus;
        plus[column] += epsilon;
        minus[column] -= epsilon;
        dynamics.discrete_step(state, plus, dt, f_plus);
        dynamics.discrete_step(state, minus, dt, f_minus);
        for (int row = 0; row < kStateDim; ++row) {
            const double finite_difference =
                (f_plus[row] - f_minus[row]) / (2.0 * epsilon);
            maximum_error = std::max(
                maximum_error, std::fabs(finite_difference - B(row, column)));
        }
    }
    if (maximum_error > 2e-7) {
        std::printf("dynamics Jacobian maximum error %.3e\n", maximum_error);
        return 1;
    }
    return 0;
}

int test_constraint_jacobian_and_pairs() {
    RobotParameters parameters;
    parameters.minimum_pair_normal_force = 5.0;
    PlaneTerrain terrain;
    if (!terrain.set_plane(0.12, -0.18, 1.0, 0.025)) return 1;
    ContactConstraints<kHorizon> constraints(terrain, parameters);
    if (constraints.num_constraints(0) != kUserConstraintRows) {
        std::printf("support-pair constraints were not enabled\n");
        return 1;
    }
    const Vec<kStateDim> state = derivative_test_state();
    const Vec<kControlDim> control = derivative_test_control();
    constexpr double epsilon = 1e-6;

    Vec<kConstraintCapacity> rows;
    if (constraints.evaluate(state, control, 0, rows) !=
        nmpc::Status::SUCCESS) {
        return 1;
    }
    TerrainSample terrain_sample;
    const Vec<3> first_position = state_vector3(
        state, StateIndex::foot_position(0, 0));
    if (!terrain.sample(first_position, terrain_sample)) return 1;
    const double expected_pair_row =
        (parameters.minimum_pair_normal_force -
         dot3(terrain_sample.normal,
              control_vector3(control, ControlIndex::contact_force(0, 0))) -
         dot3(terrain_sample.normal,
              control_vector3(control, ControlIndex::contact_force(1, 0)))) /
        parameters.force_scale;
    if (std::fabs(rows[ContactRow::support_pair(0)] - expected_pair_row) >
        1e-12) {
        std::printf("support-pair constraint mismatch\n");
        return 1;
    }

    Mat<kConstraintCapacity, kStateDim> Cx;
    Mat<kConstraintCapacity, kControlDim> Cu;
    constraints.jacobian(state, control, 0, Cx, Cu);
    double maximum_error = 0.0;
    for (int column = 0; column < kStateDim; ++column) {
        Vec<kStateDim> plus = state;
        Vec<kStateDim> minus = state;
        Vec<kConstraintCapacity> g_plus;
        Vec<kConstraintCapacity> g_minus;
        plus[column] += epsilon;
        minus[column] -= epsilon;
        constraints.evaluate(plus, control, 0, g_plus);
        constraints.evaluate(minus, control, 0, g_minus);
        for (int row = 0; row < kUserConstraintRows; ++row) {
            const double finite_difference =
                (g_plus[row] - g_minus[row]) / (2.0 * epsilon);
            maximum_error = std::max(
                maximum_error,
                std::fabs(finite_difference - Cx(row, column)));
        }
    }
    for (int column = 0; column < kControlDim; ++column) {
        Vec<kControlDim> plus = control;
        Vec<kControlDim> minus = control;
        Vec<kConstraintCapacity> g_plus;
        Vec<kConstraintCapacity> g_minus;
        plus[column] += epsilon;
        minus[column] -= epsilon;
        constraints.evaluate(state, plus, 0, g_plus);
        constraints.evaluate(state, minus, 0, g_minus);
        for (int row = 0; row < kUserConstraintRows; ++row) {
            const double finite_difference =
                (g_plus[row] - g_minus[row]) / (2.0 * epsilon);
            maximum_error = std::max(
                maximum_error,
                std::fabs(finite_difference - Cu(row, column)));
        }
    }
    if (maximum_error > 1e-7) {
        std::printf("constraint Jacobian maximum error %.3e\n",
                    maximum_error);
        return 1;
    }

    for (int foot = 0; foot < kNumFeet; ++foot) {
        int first = -1;
        int second = -1;
        if (!constraints.complementarity_pair(
                0, 2 * foot, first, second) ||
            first != ContactRow::index(foot, ContactRow::GAP) ||
            second != ContactRow::index(foot, ContactRow::NORMAL_FORCE)) {
            std::printf("gap-force pair mismatch at foot %d\n", foot);
            return 1;
        }
        if (!constraints.complementarity_pair(
                0, 2 * foot + 1, first, second) ||
            first != ContactRow::index(foot, ContactRow::NORMAL_FORCE) ||
            second != ContactRow::index(foot, ContactRow::MOTION_SLACK)) {
            std::printf("force-speed pair mismatch at foot %d\n", foot);
            return 1;
        }
    }
    return 0;
}

int test_designated_foot_hard_support_schedule() {
    SinusoidalHeightTerrain terrain;
    terrain.offset = 0.01;
    terrain.slope_x = 0.03;
    terrain.slope_y = -0.02;
    terrain.amplitude = 0.025;
    terrain.wave_number_x = 2.4;
    terrain.wave_number_y = 3.1;
    const int schedule[kHorizon] = {2, 1, 0};
    const int invalid_schedule[kHorizon] = {2, kNumFeet, 0};

    RobotParameters parameters;
    ContactConstraints<kHorizon, SinusoidalHeightTerrain> constraints(
        terrain, parameters);
    if (constraints.num_constraints(0) != kContactConstraintRows ||
        constraints.set_designated_foot_schedule(nullptr, kHorizon) !=
            nmpc::Status::BAD_ARGUMENT ||
        constraints.set_designated_foot_schedule(schedule, kHorizon - 1) !=
            nmpc::Status::BAD_ARGUMENT ||
        constraints.set_designated_foot_schedule(
            invalid_schedule, kHorizon) != nmpc::Status::BAD_ARGUMENT ||
        constraints.num_constraints(0) != kContactConstraintRows) {
        std::printf("invalid hard-support schedule was accepted\n");
        return 1;
    }
    RobotParameters pair_parameters = parameters;
    pair_parameters.minimum_pair_normal_force = 5.0;
    ContactConstraints<kHorizon, SinusoidalHeightTerrain> pair_constraints(
        terrain, pair_parameters);
    if (pair_constraints.set_designated_foot_schedule(
            schedule, kHorizon) != nmpc::Status::BAD_ARGUMENT) {
        std::printf("hard-support and support-pair modes were combined\n");
        return 1;
    }
    if (constraints.set_designated_foot_schedule(schedule, kHorizon) !=
            nmpc::Status::SUCCESS ||
        constraints.num_constraints(0) != kUserConstraintRows) {
        std::printf("valid hard-support schedule was rejected\n");
        return 1;
    }

    Vec<kStateDim> state = derivative_test_state();
    Vec<kControlDim> control = derivative_test_control();
    constexpr double support_force = 12.0;
    constexpr double support_slack = 5e-5;
    for (int foot = 0; foot < kNumFeet; ++foot) {
        const Vec<3> position = state_vector3(
            state, StateIndex::foot_position(foot, 0));
        TerrainSample sample;
        if (!terrain.sample(position, sample)) return 1;
        for (int axis = 0; axis < 3; ++axis) {
            control[ControlIndex::contact_force(foot, axis)] =
                support_force * sample.normal[axis];
            control[ControlIndex::foot_velocity(foot, axis)] = 0.0;
        }
        control[ControlIndex::motion_slack(foot)] = support_slack;
    }
    const int designated_foot = schedule[0];
    for (int axis = 0; axis < 3; ++axis) {
        control[ControlIndex::contact_force(designated_foot, axis)] = 0.0;
        control[ControlIndex::foot_velocity(designated_foot, axis)] =
            0.1 * (axis + 1);
    }
    control[ControlIndex::motion_slack(designated_foot)] = 0.5;

    Vec<kConstraintCapacity> rows;
    if (constraints.evaluate(state, control, 0, rows) !=
        nmpc::Status::SUCCESS) {
        return 1;
    }
    for (int support_slot = 0; support_slot < kNumFeet - 1;
         ++support_slot) {
        if (rows[ContactRow::hard_support_force(support_slot)] > 0.0 ||
            rows[ContactRow::hard_support_motion(support_slot)] > 0.0) {
            std::printf("feasible hard-support row was violated\n");
            return 1;
        }
    }
    if (std::fabs(rows[ContactRow::index(
            designated_foot, ContactRow::NORMAL_FORCE)]) > 1e-12 ||
        rows[ContactRow::index(
            designated_foot, ContactRow::MOTION_SLACK)] >= 0.0) {
        std::printf("designated foot was not left contact-implicit\n");
        return 1;
    }
    constraints.evaluate(state, control, 1, rows);
    if (!(rows[ContactRow::hard_support_force(1)] > 0.0) ||
        !(rows[ContactRow::hard_support_motion(1)] > 0.0)) {
        std::printf("per-stage designated foot schedule was not applied\n");
        return 1;
    }

    const int low_force_foot = 1;
    TerrainSample low_force_sample;
    if (!terrain.sample(
            state_vector3(state,
                StateIndex::foot_position(low_force_foot, 0)),
            low_force_sample)) {
        return 1;
    }
    for (int axis = 0; axis < 3; ++axis) {
        control[ControlIndex::contact_force(low_force_foot, axis)] =
            9.0 * low_force_sample.normal[axis];
    }
    constraints.evaluate(state, control, 0, rows);
    if (!(rows[ContactRow::hard_support_force(1)] > 0.0)) {
        std::printf("low support force was not detected\n");
        return 1;
    }
    for (int axis = 0; axis < 3; ++axis) {
        control[ControlIndex::contact_force(low_force_foot, axis)] =
            support_force * low_force_sample.normal[axis];
    }
    control[ControlIndex::motion_slack(3)] = 2e-4;
    constraints.evaluate(state, control, 0, rows);
    if (!(rows[ContactRow::hard_support_motion(2)] > 0.0)) {
        std::printf("support motion was not detected\n");
        return 1;
    }

    control = derivative_test_control();
    Mat<kConstraintCapacity, kStateDim> Cx;
    Mat<kConstraintCapacity, kControlDim> Cu;
    if (constraints.jacobian(state, control, 0, Cx, Cu) !=
        nmpc::Status::SUCCESS) {
        return 1;
    }
    constexpr double epsilon = 1e-6;
    double maximum_error = 0.0;
    for (int column = 0; column < kStateDim; ++column) {
        Vec<kStateDim> plus = state;
        Vec<kStateDim> minus = state;
        Vec<kConstraintCapacity> g_plus;
        Vec<kConstraintCapacity> g_minus;
        plus[column] += epsilon;
        minus[column] -= epsilon;
        constraints.evaluate(plus, control, 0, g_plus);
        constraints.evaluate(minus, control, 0, g_minus);
        for (int row = kContactConstraintRows;
             row < kUserConstraintRows; ++row) {
            const double finite_difference =
                (g_plus[row] - g_minus[row]) / (2.0 * epsilon);
            maximum_error = std::max(
                maximum_error,
                std::fabs(finite_difference - Cx(row, column)));
        }
    }
    for (int column = 0; column < kControlDim; ++column) {
        Vec<kControlDim> plus = control;
        Vec<kControlDim> minus = control;
        Vec<kConstraintCapacity> g_plus;
        Vec<kConstraintCapacity> g_minus;
        plus[column] += epsilon;
        minus[column] -= epsilon;
        constraints.evaluate(state, plus, 0, g_plus);
        constraints.evaluate(state, minus, 0, g_minus);
        for (int row = kContactConstraintRows;
             row < kUserConstraintRows; ++row) {
            const double finite_difference =
                (g_plus[row] - g_minus[row]) / (2.0 * epsilon);
            maximum_error = std::max(
                maximum_error,
                std::fabs(finite_difference - Cu(row, column)));
        }
    }
    if (maximum_error > 5e-7) {
        std::printf("hard-support Jacobian maximum error %.3e\n",
                    maximum_error);
        return 1;
    }
    return 0;
}

int test_constraint_adjoint_hessian() {
    RobotParameters parameters;
    PlaneTerrain terrain;
    ContactConstraints<kHorizon> constraints(terrain, parameters);
    Vec<kConstraintCapacity> multipliers;
    multipliers.zero();
    for (int foot = 0; foot < kNumFeet; ++foot) {
        multipliers[ContactRow::index(foot, ContactRow::LEG_REACH)] =
            0.7 + 0.2 * foot;
    }
    Mat<kStateDim, kStateDim> Hxx;
    Mat<kControlDim, kStateDim> Hux;
    Mat<kControlDim, kControlDim> Huu;
    Hxx.zero();
    Hux.zero();
    Huu.zero();
    constraints.adjoint_hessian(derivative_test_state(),
                                derivative_test_control(), 0, multipliers,
                                Hxx, Hux, Huu);
    for (int foot = 0; foot < kNumFeet; ++foot) {
        const double reach =
            multipliers[ContactRow::index(foot, ContactRow::LEG_REACH)];
        for (int axis = 0; axis < 3; ++axis) {
            const int base = StateIndex::base_position(axis);
            const int foot_state = StateIndex::foot_position(foot, axis);
            const double expected_reach =
                2.0 * reach /
                (parameters.max_leg_reach * parameters.max_leg_reach);
            if (std::fabs(Hxx(base, foot_state) + expected_reach) > 1e-12 ||
                std::fabs(Hxx(foot_state, foot_state) - expected_reach) >
                    1e-12) {
                std::printf("constraint Hessian mismatch at foot %d\n", foot);
                return 1;
            }
        }
    }
    return 0;
}

int test_smooth_height_terrain_jacobian() {
    RobotParameters parameters;
    parameters.minimum_pair_normal_force = 5.0;
    SinusoidalHeightTerrain terrain;
    terrain.offset = 0.015;
    terrain.slope_x = 0.04;
    terrain.slope_y = -0.03;
    terrain.amplitude = 0.055;
    terrain.wave_number_x = 3.2;
    terrain.wave_number_y = 2.7;
    ContactConstraints<kHorizon, SinusoidalHeightTerrain> constraints(
        terrain, parameters);
    Vec<kStateDim> state = derivative_test_state();
    Vec<kControlDim> control = derivative_test_control();
    constexpr double epsilon = 1e-6;

    Mat<kConstraintCapacity, kStateDim> Cx;
    Mat<kConstraintCapacity, kControlDim> Cu;
    if (constraints.jacobian(state, control, 0, Cx, Cu) !=
        nmpc::Status::SUCCESS) {
        std::printf("smooth-terrain Jacobian returned failure\n");
        return 1;
    }

    double maximum_error = 0.0;
    for (int foot = 0; foot < kNumFeet; ++foot) {
        for (int axis = 0; axis < 3; ++axis) {
            const int column = StateIndex::foot_position(foot, axis);
            Vec<kStateDim> plus = state;
            Vec<kStateDim> minus = state;
            Vec<kConstraintCapacity> g_plus;
            Vec<kConstraintCapacity> g_minus;
            plus[column] += epsilon;
            minus[column] -= epsilon;
            constraints.evaluate(plus, control, 0, g_plus);
            constraints.evaluate(minus, control, 0, g_minus);
            for (int offset = 0;
                 offset <= ContactRow::CLEARANCE_NEG_T2;
                 ++offset) {
                const int row = kRowsPerFoot * foot + offset;
                const double finite_difference =
                    (g_plus[row] - g_minus[row]) / (2.0 * epsilon);
                maximum_error = std::max(
                    maximum_error,
                    std::fabs(finite_difference - Cx(row, column)));
            }
            for (int row = kContactConstraintRows;
                 row < kUserConstraintRows; ++row) {
                const double finite_difference =
                    (g_plus[row] - g_minus[row]) / (2.0 * epsilon);
                maximum_error = std::max(
                    maximum_error,
                    std::fabs(finite_difference - Cx(row, column)));
            }
        }
    }
    if (maximum_error > 5e-7) {
        std::printf("smooth-terrain Jacobian maximum error %.3e\n",
                    maximum_error);
        return 1;
    }
    if (constraints.provides_adjoint_hessian()) {
        std::printf("nonlinear terrain incorrectly advertises exact Hessian\n");
        return 1;
    }
    return 0;
}

int test_smooth_step_terrain_jacobian() {
    RobotParameters parameters;
    SmoothStepTerrain terrain;
    terrain.step_height = 0.04;
    terrain.step_center_x = 0.34;
    terrain.sharpness = 30.0;
    ContactConstraints<kHorizon, SmoothStepTerrain> constraints(
        terrain, parameters);
    Vec<kStateDim> state = derivative_test_state();
    const Vec<kControlDim> control = derivative_test_control();
    for (int foot = 0; foot < kNumFeet; ++foot) {
        const double x = 0.31 + 0.02 * foot;
        state[StateIndex::foot_position(foot, 0)] = x;
        state[StateIndex::foot_position(foot, 2)] =
            terrain.height(x, state[StateIndex::foot_position(foot, 1)]) +
            0.03;
    }

    Mat<kConstraintCapacity, kStateDim> Cx;
    Mat<kConstraintCapacity, kControlDim> Cu;
    if (constraints.jacobian(state, control, 0, Cx, Cu) !=
        nmpc::Status::SUCCESS) {
        std::printf("smooth-step Jacobian returned failure\n");
        return 1;
    }
    constexpr double epsilon = 1e-6;
    double maximum_error = 0.0;
    for (int foot = 0; foot < kNumFeet; ++foot) {
        const int column = StateIndex::foot_position(foot, 0);
        Vec<kStateDim> plus = state;
        Vec<kStateDim> minus = state;
        Vec<kConstraintCapacity> g_plus;
        Vec<kConstraintCapacity> g_minus;
        plus[column] += epsilon;
        minus[column] -= epsilon;
        constraints.evaluate(plus, control, 0, g_plus);
        constraints.evaluate(minus, control, 0, g_minus);
        for (int offset = 0; offset <= ContactRow::CLEARANCE_NEG_T2;
             ++offset) {
            const int row = kRowsPerFoot * foot + offset;
            const double finite_difference =
                (g_plus[row] - g_minus[row]) / (2.0 * epsilon);
            maximum_error = std::max(
                maximum_error,
                std::fabs(finite_difference - Cx(row, column)));
        }
    }
    if (maximum_error > 2e-6) {
        std::printf("smooth-step Jacobian maximum error %.3e\n",
                    maximum_error);
        return 1;
    }
    return 0;
}

int test_cost_derivatives_and_problem_setup() {
    RobotParameters parameters;
    SRBDDynamics dynamics(parameters);
    PlaneTerrain terrain;
    ContactConstraints<kHorizon> constraints(terrain, parameters);
    QuadraticTrackingCost<kHorizon> cost;
    Problem<kHorizon> problem;
    initialize_standing_problem(problem, dynamics, cost, constraints, 0.03);

    for (int i = 0; i < kStateDim; ++i) {
        cost.state_weights[i] = 0.2 + 0.03 * i;
        cost.terminal_weights[i] = 1.0 + 0.05 * i;
    }
    for (int i = 0; i < kControlDim; ++i)
        cost.control_weights[i] = 0.01 + 0.002 * i;

    Vec<kStateDim> state = derivative_test_state();
    Vec<kControlDim> control = derivative_test_control();
    Vec<kStateDim> qx;
    Vec<kControlDim> qu;
    cost.stage_gradient(state, control, 1, qx, qu);
    constexpr double epsilon = 1e-6;
    double maximum_error = 0.0;
    for (int i = 0; i < kStateDim; ++i) {
        Vec<kStateDim> plus = state;
        Vec<kStateDim> minus = state;
        plus[i] += epsilon;
        minus[i] -= epsilon;
        const double finite_difference =
            (cost.stage_cost(plus, control, 1) -
             cost.stage_cost(minus, control, 1)) /
            (2.0 * epsilon);
        maximum_error =
            std::max(maximum_error, std::fabs(finite_difference - qx[i]));
    }
    for (int i = 0; i < kControlDim; ++i) {
        Vec<kControlDim> plus = control;
        Vec<kControlDim> minus = control;
        plus[i] += epsilon;
        minus[i] -= epsilon;
        const double finite_difference =
            (cost.stage_cost(state, plus, 1) -
             cost.stage_cost(state, minus, 1)) /
            (2.0 * epsilon);
        maximum_error =
            std::max(maximum_error, std::fabs(finite_difference - qu[i]));
    }
    if (maximum_error > 2e-6) {
        std::printf("cost gradient maximum error %.3e\n", maximum_error);
        return 1;
    }
    if (problem.validate() != nmpc::Status::SUCCESS ||
        problem.n_bound_x != 4 || problem.n_bound_u != kControlDim) {
        std::printf("standing problem setup failed\n");
        return 1;
    }
    return 0;
}

}  // namespace

int main() {
    int failures = 0;
    failures += test_standing_equilibrium();
    failures += test_dynamics_jacobian();
    failures += test_constraint_jacobian_and_pairs();
    failures += test_designated_foot_hard_support_schedule();
    failures += test_constraint_adjoint_hessian();
    failures += test_smooth_height_terrain_jacobian();
    failures += test_smooth_step_terrain_jacobian();
    failures += test_shared_terrain_and_contact_derivatives();
    failures += test_contact_terrain_sample_reuse();
    failures += test_cost_derivatives_and_problem_setup();
    if (failures == 0) {
        std::printf("Quadruped CITO model tests passed.\n");
        return 0;
    }
    std::printf("Quadruped CITO model tests failed: %d\n", failures);
    return 1;
}
