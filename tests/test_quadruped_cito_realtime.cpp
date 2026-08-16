#include <algorithm>
#include <chrono>
#include <cmath>
#include <cstdio>
#include <limits>
#include <memory>

#include "examples/quadruped_cito/quadruped_cito_realtime_planner.hpp"

namespace {

using namespace quadruped_cito;

bool same_problem(const QuadrupedCITORealtimePlanner::ProblemType& first,
                  const QuadrupedCITORealtimePlanner::ProblemType& second) {
    for (int stage = 0; stage <= kRealtimeHorizon; ++stage) {
        for (int state = 0; state < kStateDim; ++state) {
            if (first.stages[stage].x[state] != second.stages[stage].x[state])
                return false;
        }
        for (int control = 0; control < kControlDim; ++control) {
            if (first.stages[stage].u[control] !=
                second.stages[stage].u[control]) {
                return false;
            }
        }
        for (int row = 0; row < kConstraintCapacity; ++row) {
            if (first.stages[stage].s[row] != second.stages[stage].s[row] ||
                first.stages[stage].lambda[row] !=
                    second.stages[stage].lambda[row]) {
                return false;
            }
        }
    }
    return true;
}

bool same_plan(const ContactPlan<kRealtimeHorizon>& first,
               const ContactPlan<kRealtimeHorizon>& second) {
    for (int stage = 0; stage < kRealtimeHorizon; ++stage) {
        if (first.stages[stage].time != second.stages[stage].time ||
            first.stages[stage].duration != second.stages[stage].duration) {
            return false;
        }
        for (int foot = 0; foot < kNumFeet; ++foot) {
            if (first.stages[stage].feet[foot].planned_contact !=
                    second.stages[stage].feet[foot].planned_contact ||
                first.stages[stage].feet[foot].normal_force !=
                    second.stages[stage].feet[foot].normal_force) {
                return false;
            }
            for (int axis = 0; axis < 3; ++axis) {
                if (first.stages[stage].feet[foot].position_world[axis] !=
                        second.stages[stage].feet[foot].position_world[axis] ||
                    first.stages[stage].feet[foot].force_world[axis] !=
                        second.stages[stage].feet[foot].force_world[axis]) {
                    return false;
                }
            }
        }
    }
    return first.terminal.time == second.terminal.time;
}

bool nearly_equal(double first, double second, double tolerance = 1e-12) {
    return std::fabs(first - second) <= tolerance;
}

bool tail_dynamics_defect(
    const QuadrupedCITORealtimePlanner::ProblemType& problem,
    SRBDDynamics& dynamics, int first_stage, double& maximum_defect) {
    maximum_defect = 0.0;
    for (int stage = first_stage; stage < kRealtimeHorizon; ++stage) {
        Vec<kStateDim> predicted;
        if (dynamics.discrete_step(problem.stages[stage].x,
                                   problem.stages[stage].u, problem.dt,
                                   predicted) != nmpc::Status::SUCCESS) {
            return false;
        }
        for (int state = 0; state < kStateDim; ++state) {
            maximum_defect = std::max(
                maximum_defect,
                std::fabs(predicted[state] -
                          problem.stages[stage + 1].x[state]));
        }
    }
    return true;
}

bool test_measured_state_correction() {
    const SharedTerrain terrain = SharedTerrain::flat();
    const Vec<kStateDim> nominal = go1_home_state(terrain);

    {
        Vec<kStateDim> actual = nominal;
        Vec<kStateDim> corrected = nominal;
        actual[StateIndex::base_position(0)] += 0.04;
        if (realtime_detail::apply_measured_state_defect(
                actual, nominal, corrected) != nmpc::Status::SUCCESS ||
            !nearly_equal(corrected[StateIndex::base_position(0)],
                          actual[StateIndex::base_position(0)]) ||
            !nearly_equal(corrected[StateIndex::foot_position(0, 0)],
                          nominal[StateIndex::foot_position(0, 0)])) {
            std::printf("base-state correction was not independent\n");
            return false;
        }
    }

    {
        Vec<kStateDim> actual = nominal;
        Vec<kStateDim> corrected = nominal;
        actual[StateIndex::foot_position(2, 1)] -= 0.025;
        if (realtime_detail::apply_measured_state_defect(
                actual, nominal, corrected) != nmpc::Status::SUCCESS ||
            !nearly_equal(corrected[StateIndex::foot_position(2, 1)],
                          actual[StateIndex::foot_position(2, 1)]) ||
            !nearly_equal(corrected[StateIndex::base_position(1)],
                          nominal[StateIndex::base_position(1)])) {
            std::printf("foot-state correction was not independent\n");
            return false;
        }
    }

    {
        Vec<kStateDim> actual = nominal;
        Vec<kStateDim> corrected = nominal;
        constexpr double half_angle = 0.04;
        actual[StateIndex::quaternion(0)] = std::cos(half_angle);
        actual[StateIndex::quaternion(3)] = std::sin(half_angle);
        if (realtime_detail::apply_measured_state_defect(
                actual, nominal, corrected) != nmpc::Status::SUCCESS) {
            std::printf("attitude correction failed\n");
            return false;
        }
        double norm_sq = 0.0;
        for (int element = 0; element < 4; ++element) {
            const double value = corrected[StateIndex::quaternion(element)];
            norm_sq += value * value;
            if (!nearly_equal(value,
                              actual[StateIndex::quaternion(element)])) {
                std::printf("attitude defect was not propagated\n");
                return false;
            }
        }
        if (!nearly_equal(norm_sq, 1.0)) {
            std::printf("corrected attitude was not normalized\n");
            return false;
        }
    }
    return true;
}

bool test_initial_force_projection(const SharedTerrain& physical_terrain) {
    realtime_detail::PlannerTerrain terrain(physical_terrain);
    const RobotParameters robot = go1_robot_parameters();
    SRBDDynamics dynamics(robot);
    QuadraticTrackingCost<kRealtimeHorizon> cost;
    ContactConstraints<kRealtimeHorizon, realtime_detail::PlannerTerrain>
        constraints(terrain, robot);
    auto problem =
        std::make_unique<QuadrupedCITORealtimePlanner::ProblemType>();
    initialize_standing_problem(*problem, dynamics, cost, constraints,
                                kRealtimeTimeStep);
    const Vec<kStateDim> standing = go1_home_state(terrain);
    problem->x0 = standing;
    for (int stage = 0; stage <= kRealtimeHorizon; ++stage)
        problem->stages[stage].x = standing;
    double original_normal_force[kRealtimeHorizon][kNumFeet];
    double original_tangent1_force[kRealtimeHorizon][kNumFeet];
    double original_tangent2_force[kRealtimeHorizon][kNumFeet];
    for (int stage = 0; stage < kRealtimeHorizon; ++stage) {
        for (int foot = 0; foot < kNumFeet; ++foot) {
            const Vec<3> position = state_vector3(
                problem->stages[stage].x,
                StateIndex::foot_position(foot, 0));
            TerrainSample sample;
            if (!terrain.sample(position, sample)) return false;
            const Vec<3> force = control_vector3(
                problem->stages[stage].u,
                ControlIndex::contact_force(foot, 0));
            original_normal_force[stage][foot] = dot3(sample.normal, force);
            original_tangent1_force[stage][foot] =
                dot3(sample.tangent1, force);
            original_tangent2_force[stage][foot] =
                dot3(sample.tangent2, force);
        }
    }
    if (realtime_detail::blend_initial_contact_forces_toward_terrain_normals(
            *problem, terrain) != nmpc::Status::SUCCESS) {
        return false;
    }

    for (int stage = 0; stage < kRealtimeHorizon; ++stage) {
        for (int foot = 0; foot < kNumFeet; ++foot) {
            const Vec<3> position = state_vector3(
                problem->stages[stage].x,
                StateIndex::foot_position(foot, 0));
            TerrainSample sample;
            if (!terrain.sample(position, sample)) return false;
            const Vec<3> force = control_vector3(
                problem->stages[stage].u,
                ControlIndex::contact_force(foot, 0));
            if (!nearly_equal(
                    dot3(sample.normal, force),
                    original_normal_force[stage][foot]) ||
                !nearly_equal(
                    dot3(sample.tangent1, force),
                    0.75 * original_tangent1_force[stage][foot]) ||
                !nearly_equal(
                    dot3(sample.tangent2, force),
                    0.75 * original_tangent2_force[stage][foot])) {
                std::printf("force blend changed components at %d/%d\n",
                            stage, foot);
                return false;
            }
        }
    }

    if (realtime_detail::initialize_transition_guess(
            *problem, robot, terrain) != nmpc::Status::SUCCESS) {
        return false;
    }

    return true;
}

bool test_cold_retry_status_filter() {
    if (!realtime_detail::cold_start_retryable(
            nmpc::Status::MAX_ITERATIONS) ||
        !realtime_detail::cold_start_retryable(nmpc::Status::STAGNATION) ||
        !realtime_detail::cold_start_retryable(
            nmpc::Status::LINE_SEARCH_FAILURE) ||
        !realtime_detail::cold_start_retryable(
            nmpc::Status::KKT_SINGULAR)) {
        std::printf("cold convergence status was not retryable\n");
        return false;
    }
    if (realtime_detail::cold_start_retryable(nmpc::Status::SUCCESS) ||
        realtime_detail::cold_start_retryable(nmpc::Status::TIME_LIMIT) ||
        realtime_detail::cold_start_retryable(nmpc::Status::BAD_ARGUMENT) ||
        realtime_detail::cold_start_retryable(nmpc::Status::NAN_DETECTED) ||
        realtime_detail::cold_start_retryable(
            nmpc::Status::NOT_INITIALIZED) ||
        realtime_detail::cold_start_retryable(
            nmpc::Status::INTERNAL_ERROR)) {
        std::printf("cold non-convergence status was retryable\n");
        return false;
    }
    return true;
}

bool test_random_cold_alternate_guess() {
    const SharedTerrain physical_terrain =
        SharedTerrain::random_smooth(1002, 0.02);
    auto planner = std::make_unique<QuadrupedCITORealtimePlanner>(
        physical_terrain);
    const Vec<kStateDim> standing = planner->nominal_state();
    if (planner->initialize(standing) != nmpc::Status::SUCCESS)
        return false;
    const RealtimePlannerResult cold = planner->cold_solve();
    if (!cold.published || !cold.full_kkt || !cold.task_pass ||
        cold.status != nmpc::Status::SUCCESS || cold.solver_ms <= 0.0 ||
        cold.stats.exact_hessian_analytic_calls != 0 ||
        cold.stats.exact_hessian_fd_calls != 0) {
        std::printf(
            "random cold alternate guess failed: status=%s fallback=%d "
            "published=%d iterations=%d solver_ms=%.3f\n",
            nmpc::status_string(cold.status),
            cold.cold_fallback_used ? 1 : 0, cold.published ? 1 : 0,
            cold.stats.inner_iterations, cold.solver_ms);
        return false;
    }

    realtime_detail::PlannerTerrain terrain(physical_terrain);
    const RobotParameters robot = go1_robot_parameters();
    SRBDDynamics dynamics(robot);
    QuadraticTrackingCost<kRealtimeHorizon> cost;
    ContactConstraints<kRealtimeHorizon, realtime_detail::PlannerTerrain>
        constraints(terrain, robot);
    auto problem =
        std::make_unique<QuadrupedCITORealtimePlanner::ProblemType>();
    initialize_standing_problem(*problem, dynamics, cost, constraints,
                                kRealtimeTimeStep);
    problem->x0 = standing;
    cost.set_reference_all(standing);
    for (int stage = 0; stage <= kRealtimeHorizon; ++stage)
        problem->stages[stage].x = standing;
    realtime_detail::configure_cost(cost);
    const double initial_foot_x = standing[
        StateIndex::foot_position(kRealtimeMovingFoot, 0)];
    const double target_x = initial_foot_x + kRealtimeRequestedStep;
    const double target_y = standing[
        StateIndex::foot_position(kRealtimeMovingFoot, 1)];
    cost.references[kRealtimeHorizon][StateIndex::foot_position(
        kRealtimeMovingFoot, 0)] = target_x;
    cost.references[kRealtimeHorizon][StateIndex::foot_position(
        kRealtimeMovingFoot, 2)] = terrain.height(target_x, target_y);
    if (realtime_detail::initialize_transition_guess(
            *problem, robot, terrain,
            realtime_detail::kFallbackTerrainNormalForceBlend) !=
        nmpc::Status::SUCCESS) {
        return false;
    }
    using Solver = nmpc::ContactIPM<
        kStateDim, kControlDim, kConstraintCapacity, kRealtimeHorizon>;
    Solver alternate_solver;
    alternate_solver.configure(realtime_detail::solver_parameters());
    const nmpc::Status alternate_status = alternate_solver.solve(*problem);
    const nmpc::SolverStats& alternate_stats =
        alternate_solver.last_stats();
    const RealtimePlannerAudit alternate_audit =
        realtime_detail::audit_solution(
            *problem, dynamics, constraints, terrain, cost, initial_foot_x);
    if (!realtime_detail::full_kkt(
            alternate_status, alternate_stats, alternate_audit,
            realtime_detail::solver_parameters()) ||
        alternate_audit.task_displacement < 0.06 ||
        alternate_audit.clearance < kRealtimeMinimumClearance) {
        std::printf("random cold alternate policy failed: %s\n",
                    nmpc::status_string(alternate_status));
        return false;
    }
    return true;
}

bool test_random_warm_tail_initialization() {
    auto planner = std::make_unique<QuadrupedCITORealtimePlanner>(
        SharedTerrain::random_smooth(1002, 0.04));
    const Vec<kStateDim> standing = planner->nominal_state();
    if (planner->initialize(standing) != nmpc::Status::SUCCESS)
        return false;
    const RealtimePlannerResult cold = planner->cold_solve();
    if (!cold.published || cold.cold_fallback_used) {
        std::printf("random warm-tail cold setup failed: %s fallback=%d\n",
                    nmpc::status_string(cold.status),
                    cold.cold_fallback_used ? 1 : 0);
        return false;
    }

    auto active_problem =
        std::make_unique<QuadrupedCITORealtimePlanner::ProblemType>(
            planner->problem());
    int active_plan_stage = 0;
    for (int update = 0; update <= 4; ++update) {
        active_plan_stage = std::min(
            kRealtimeHorizon,
            active_plan_stage + planner->shift_steps());
        Vec<kStateDim> measured =
            active_problem->stages[active_plan_stage].x;
        measured[StateIndex::base_position(0)] +=
            update % 2 == 0 ? 0.0005 : -0.0005;
        const RealtimePlannerResult warm = planner->warm_update(measured);
        if (!warm.published || warm.status != nmpc::Status::SUCCESS ||
            !warm.full_kkt || !warm.task_pass ||
            warm.stats.inner_iterations > 10 ||
            warm.stats.exact_hessian_analytic_calls != 0 ||
            warm.stats.exact_hessian_fd_calls != 0) {
            std::printf(
                "random warm-tail update %d failed: status=%s "
                "iterations=%d line_search=%d\n",
                update, nmpc::status_string(warm.status),
                warm.stats.inner_iterations, warm.stats.line_search_evals);
            return false;
        }
        *active_problem = planner->problem();
        active_plan_stage = 0;
    }
    return true;
}

bool test_sloped_warm_solve(double slope_x, double slope_y) {
    SharedTerrain physical_terrain = SharedTerrain::slope(slope_x, slope_y);
    auto planner = std::make_unique<QuadrupedCITORealtimePlanner>(
        physical_terrain);
    const Vec<kStateDim> standing = planner->nominal_state();
    if (planner->initialize(standing) != nmpc::Status::SUCCESS)
        return false;
    const RealtimePlannerResult cold = planner->cold_solve();
    if (!cold.published) return false;
    Vec<kStateDim> measured =
        planner->problem().stages[planner->shift_steps()].x;
    measured[StateIndex::base_position(0)] += 0.0005;
    const RealtimePlannerResult warm = planner->warm_update(measured);
    if (!warm.published || warm.status != nmpc::Status::SUCCESS ||
        !warm.full_kkt || !warm.task_pass ||
        warm.stats.exact_hessian_analytic_calls != 0 ||
        warm.stats.exact_hessian_fd_calls != 0) {
        std::printf("sloped warm solve failed (%+.2f, %+.2f): %s\n",
                    slope_x, slope_y, nmpc::status_string(warm.status));
        return false;
    }
    return true;
}

bool test_sustained_task_sequence() {
    const SharedTerrain physical_terrain = SharedTerrain::flat();
    realtime_detail::PlannerTerrain terrain(physical_terrain);
    const Vec<kStateDim> standing = go1_home_state(terrain);
    realtime_detail::RealtimeContactTaskSequence sequence;
    sequence.initialize(standing);

    for (int id = 0; id < 12; ++id) {
        const RealtimeContactTask task = sequence.task(id);
        const int expected_foot = kRealtimeSustainedFootOrder[id % kNumFeet];
        const double expected_origin = standing[
            StateIndex::foot_position(expected_foot, 0)] +
            (id / kNumFeet) * kRealtimeRequestedStep;
        if (task.id != static_cast<std::uint64_t>(id) ||
            task.moving_foot != expected_foot ||
            task.start_knot != kRealtimeSwingFirstStage +
                                   id * kRealtimeSustainedTaskSpacing ||
            task.touchdown_knot != task.start_knot +
                                       kRealtimeSustainedSwingStages ||
            !nearly_equal(task.origin_x, expected_origin) ||
            !nearly_equal(task.target_x,
                          expected_origin + kRealtimeRequestedStep)) {
            std::printf("sustained task %d was not deterministic\n", id);
            return false;
        }
        if (id > 0 &&
            !sequence.advance(kRealtimeSustainedTaskSpacing)) {
            return false;
        }
        if (sequence.active_task().id !=
                static_cast<std::uint64_t>(id) ||
            sequence.active_task().moving_foot != expected_foot) {
            std::printf("sustained task %d did not become active\n", id);
            return false;
        }
    }
    if (sequence.completed_task_count(
            sequence.task(11).touchdown_knot) != 12) {
        std::printf("twelve sustained touchdowns were not counted\n");
        return false;
    }

    realtime_detail::RealtimeContactTaskSequence capped(2);
    capped.initialize(standing);
    const RealtimeContactTask final_task = capped.task(1);
    const int final_hold_knot = final_task.touchdown_knot +
        3 * kRealtimeSustainedTaskSpacing;
    if (!capped.advance(final_hold_knot) ||
        capped.active_task().id != 1 ||
        capped.completed_task_count(final_hold_knot) != 2 ||
        capped.moving_foot_at(final_hold_knot) != -1) {
        std::printf("capped sustained sequence did not enter final hold\n");
        return false;
    }
    Vec<kStateDim> final_reference;
    Vec<kStateDim> later_reference;
    capped.reference_state(final_hold_knot, terrain, final_reference);
    capped.reference_state(
        final_hold_knot + 10 * kRealtimeSustainedTaskSpacing,
        terrain, later_reference);
    for (int state = 0; state < kStateDim; ++state) {
        if (!nearly_equal(final_reference[state], later_reference[state])) {
            std::printf("capped sustained final reference did not hold\n");
            return false;
        }
    }

    realtime_detail::RealtimeContactTaskSequence insertion;
    insertion.initialize(standing);
    const RobotParameters robot = go1_robot_parameters();
    SRBDDynamics dynamics(robot);
    QuadraticTrackingCost<kRealtimeHorizon> cost;
    ContactConstraints<kRealtimeHorizon, realtime_detail::PlannerTerrain>
        constraints(terrain, robot);
    auto problem =
        std::make_unique<QuadrupedCITORealtimePlanner::ProblemType>();
    initialize_standing_problem(*problem, dynamics, cost, constraints,
                                kRealtimeTimeStep);
    problem->x0 = standing;
    for (int stage = 0; stage <= kRealtimeHorizon; ++stage)
        problem->stages[stage].x = standing;
    if (realtime_detail::seed_sustained_range(
            *problem, robot, terrain, insertion, 0) !=
        nmpc::Status::SUCCESS) {
        return false;
    }
    const int first_foot = kRealtimeSustainedFootOrder[0];
    const int second_foot = kRealtimeSustainedFootOrder[1];
    if (problem->stages[kRealtimeSwingFirstStage + 10]
            .x[StateIndex::foot_position(first_foot, 2)] <=
            standing[StateIndex::foot_position(first_foot, 2)] +
                kRealtimeMinimumClearance ||
        problem->stages[kRealtimeSwingFirstStage +
                        kRealtimeSustainedTaskSpacing + 10]
            .x[StateIndex::foot_position(second_foot, 2)] <=
            standing[StateIndex::foot_position(second_foot, 2)] +
                kRealtimeMinimumClearance) {
        std::printf("beginning-of-horizon sustained swings were not seeded\n");
        return false;
    }

    constexpr int kInsertionAdvance = 16;
    if (!insertion.advance(kInsertionAdvance) ||
        realtime_detail::seed_sustained_range(
            *problem, robot, terrain, insertion,
            kRealtimeHorizon - kInsertionAdvance) !=
            nmpc::Status::SUCCESS) {
        return false;
    }
    const int third_foot = kRealtimeSustainedFootOrder[2];
    const int first_appended_swing_stage =
        kRealtimeSwingFirstStage + 2 * kRealtimeSustainedTaskSpacing -
        insertion.absolute_knot();
    if (first_appended_swing_stage <
            kRealtimeHorizon - kInsertionAdvance ||
        problem->stages[first_appended_swing_stage + 1]
            .x[StateIndex::foot_position(third_foot, 2)] <=
            standing[StateIndex::foot_position(third_foot, 2)]) {
        std::printf("end-of-horizon sustained swing was not appended\n");
        return false;
    }
    return true;
}

bool test_sustained_warm_tail_rollout_seed() {
    const SharedTerrain physical_terrain = SharedTerrain::flat();
    realtime_detail::PlannerTerrain terrain(physical_terrain);
    const RobotParameters robot = go1_robot_parameters();
    SRBDDynamics dynamics(robot);
    QuadraticTrackingCost<kRealtimeHorizon> cost;
    ContactConstraints<kRealtimeHorizon, realtime_detail::PlannerTerrain>
        constraints(terrain, robot);
    auto problem =
        std::make_unique<QuadrupedCITORealtimePlanner::ProblemType>();
    initialize_standing_problem(*problem, dynamics, cost, constraints,
                                kRealtimeTimeStep);
    const Vec<kStateDim> standing = go1_home_state(terrain);
    problem->x0 = standing;
    for (int stage = 0; stage <= kRealtimeHorizon; ++stage)
        problem->stages[stage].x = standing;

    realtime_detail::RealtimeContactTaskSequence sequence;
    sequence.initialize(standing);
    if (realtime_detail::seed_sustained_range(
            *problem, robot, terrain, sequence, 0) !=
        nmpc::Status::SUCCESS) {
        return false;
    }

    using Solver = nmpc::ContactIPM<
        kStateDim, kControlDim, kConstraintCapacity, kRealtimeHorizon>;
    Solver solver;
    constexpr int kShiftSteps = 16;
    Vec<kStateDim> measured = problem->stages[kShiftSteps].x;
    measured[StateIndex::base_position(0)] += 0.004;
    measured[StateIndex::foot_position(0, 0)] += 0.003;
    if (solver.shift_for_warmstart(
            *problem, measured, kShiftSteps,
            realtime_detail::apply_measured_state_defect) !=
            nmpc::Status::SUCCESS ||
        !sequence.advance(kShiftSteps)) {
        return false;
    }

    constexpr int kFirstAppendedStage = kRealtimeHorizon - kShiftSteps;
    const Vec<kStateDim> boundary =
        problem->stages[kFirstAppendedStage].x;
    double defect_before = 0.0;
    if (!tail_dynamics_defect(*problem, dynamics, kFirstAppendedStage,
                              defect_before)) {
        return false;
    }
    if (realtime_detail::seed_sustained_range(
            *problem, robot, terrain, sequence, kFirstAppendedStage) !=
        nmpc::Status::SUCCESS) {
        return false;
    }

    for (int state = 0; state < kStateDim; ++state) {
        if (problem->stages[kFirstAppendedStage].x[state] !=
            boundary[state]) {
            std::printf("sustained warm seed replaced the retained boundary\n");
            return false;
        }
    }
    double defect_after = 0.0;
    if (!tail_dynamics_defect(*problem, dynamics, kFirstAppendedStage,
                              defect_after) ||
        defect_after > defect_before + 1e-12 || defect_after > 1e-10) {
        std::printf(
            "sustained warm tail dynamics worsened: before=%.3e after=%.3e\n",
            defect_before, defect_after);
        return false;
    }

    bool seeded_motion = false;
    for (int stage = kFirstAppendedStage;
         stage < kRealtimeHorizon; ++stage) {
        double vertical_force = 0.0;
        for (int foot = 0; foot < kNumFeet; ++foot) {
            vertical_force += problem->stages[stage].u[
                ControlIndex::contact_force(foot, 2)];
            for (int axis = 0; axis < 3; ++axis) {
                seeded_motion = seeded_motion || std::fabs(
                    problem->stages[stage].u[
                        ControlIndex::foot_velocity(foot, axis)]) > 1e-12;
            }
        }
        if (!nearly_equal(vertical_force, robot.mass * robot.gravity, 1e-9)) {
            std::printf("sustained warm tail did not seed support controls\n");
            return false;
        }
    }
    if (!seeded_motion) {
        std::printf("sustained warm tail did not seed foot motion controls\n");
        return false;
    }

    Vec<kStateDim> collapsed = standing;
    for (int foot = 0; foot < kNumFeet; ++foot) {
        collapsed[StateIndex::foot_position(foot, 0)] = 0.0;
        collapsed[StateIndex::foot_position(foot, 1)] = 0.0;
    }
    double support[kNumFeet];
    if (!realtime_detail::sustained_support_weights(
            collapsed, 2, support) ||
        !nearly_equal(support[0], 1.0 / 3.0) ||
        !nearly_equal(support[1], 1.0 / 3.0) ||
        !nearly_equal(support[2], 0.0) ||
        !nearly_equal(support[3], 1.0 / 3.0)) {
        std::printf("degenerate support did not use allowed contacts\n");
        return false;
    }
    return true;
}

bool test_random_smooth_measured_state_seed() {
    const SharedTerrain physical_terrain = register_go1_terrain(
        SharedTerrain::random_smooth(1000, 0.02));
    realtime_detail::PlannerTerrain terrain(physical_terrain);
    const Go1SRBDCalibration calibration = go1_srbd_calibration();
    Vec<kStateDim> measured = go1_home_state(terrain);
    for (int axis = 0; axis < 3; ++axis) {
        measured[StateIndex::base_position(axis)] =
            calibration.com_position_world[axis];
    }
    measured[StateIndex::foot_position(0, 2)] =
        calibration.foot_positions_world[0][2];
    measured[StateIndex::foot_position(3, 2)] =
        calibration.foot_positions_world[3][2];

    RealtimePlannerConfig configuration;
    configuration.sustained_contact_tasks = true;
    auto planner = std::make_unique<QuadrupedCITORealtimePlanner>(
        physical_terrain, configuration);
    const nmpc::Status status = planner->initialize(measured);
    if (status != nmpc::Status::SUCCESS) {
        std::printf("random-smooth measured-state seed failed: %s\n",
                    nmpc::status_string(status));
        return false;
    }

    const RobotParameters robot = go1_robot_parameters();
    const auto& problem = planner->problem();
    realtime_detail::RealtimeContactTaskSequence sequence;
    sequence.initialize(measured);
    for (int stage = 0; stage <= kRealtimeHorizon; ++stage) {
        for (int state = 0; state < kStateDim; ++state) {
            if (!realtime_detail::finite_value(
                    problem.stages[stage].x[state])) {
                std::printf(
                    "random-smooth seed state was non-finite at stage %d\n",
                    stage);
                return false;
            }
        }
        if (stage == kRealtimeHorizon) continue;
        for (int control = 0; control < kControlDim; ++control) {
            if (!realtime_detail::finite_value(
                    problem.stages[stage].u[control])) {
                std::printf(
                    "random-smooth seed control was non-finite at stage %d\n",
                    stage);
                return false;
            }
        }

        const int excluded_foot = sequence.moving_foot_at(stage);
        double vertical_force_sum = 0.0;
        for (int foot = 0; foot < kNumFeet; ++foot) {
            const Vec<3> force = control_vector3(
                problem.stages[stage].u,
                ControlIndex::contact_force(foot, 0));
            vertical_force_sum += force[2];
            if (force[0] != 0.0 || force[1] != 0.0 ||
                (foot == excluded_foot && force[2] != 0.0)) {
                std::printf(
                    "random-smooth seed violated vertical/excluded support "
                    "at stage %d foot %d\n", stage, foot);
                return false;
            }

            const Vec<3> position = state_vector3(
                problem.stages[stage].x,
                StateIndex::foot_position(foot, 0));
            TerrainSample sample;
            if (!terrain.sample(position, sample)) return false;
            const double normal_force = dot3(sample.normal, force);
            const double tangent_force1 = dot3(sample.tangent1, force);
            const double tangent_force2 = dot3(sample.tangent2, force);
            const double friction_limit = robot.friction * normal_force;
            const double tolerance = 1e-12 *
                std::max(1.0, std::fabs(force[2]));
            if (normal_force < -tolerance ||
                std::fabs(tangent_force1) > friction_limit + tolerance ||
                std::fabs(tangent_force2) > friction_limit + tolerance) {
                std::printf(
                    "random-smooth seed left friction cone at stage %d "
                    "foot %d\n", stage, foot);
                return false;
            }
        }
        if (!nearly_equal(vertical_force_sum,
                          robot.mass * robot.gravity, 1e-9)) {
            std::printf(
                "random-smooth seed did not balance gravity at stage %d\n",
                stage);
            return false;
        }
    }

    SRBDDynamics dynamics(robot);
    double maximum_defect = 0.0;
    if (!tail_dynamics_defect(problem, dynamics, 0, maximum_defect) ||
        maximum_defect > 1e-12) {
        std::printf(
            "random-smooth seed dynamics defect was %.3e\n",
            maximum_defect);
        return false;
    }
    return true;
}

bool test_sustained_checkpoint_audit() {
    const SharedTerrain physical_terrain = SharedTerrain::flat();
    realtime_detail::PlannerTerrain terrain(physical_terrain);
    const Vec<kStateDim> standing = go1_home_state(terrain);
    realtime_detail::RealtimeContactTaskSequence sequence;
    sequence.initialize(standing);
    if (!sequence.advance(8 * 4)) return false;

    const RobotParameters robot = go1_robot_parameters();
    SRBDDynamics dynamics(robot);
    QuadraticTrackingCost<kRealtimeHorizon> cost;
    ContactConstraints<kRealtimeHorizon, realtime_detail::PlannerTerrain>
        constraints(terrain, robot);
    auto problem =
        std::make_unique<QuadrupedCITORealtimePlanner::ProblemType>();
    initialize_standing_problem(*problem, dynamics, cost, constraints,
                                kRealtimeTimeStep);
    problem->x0 = standing;
    for (int stage = 0; stage <= kRealtimeHorizon; ++stage)
        problem->stages[stage].x = standing;
    cost.set_reference_all(standing);
    realtime_detail::configure_cost(cost, true);
    realtime_detail::configure_sustained_references(cost, sequence, terrain);
    if (realtime_detail::seed_sustained_range(
            *problem, robot, terrain, sequence, 0) !=
        nmpc::Status::SUCCESS) {
        return false;
    }

    const RealtimeContactTask task = sequence.active_task();
    const int checkpoint_stage =
        (static_cast<int>(task.id) + 1) *
            kRealtimeSustainedTaskSpacing - 1 - sequence.absolute_knot();
    cost.references[kRealtimeHorizon][
        StateIndex::foot_position(task.moving_foot, 0)] += 0.25;
    for (int axis = 0; axis < 3; ++axis) {
        problem->stages[kRealtimeHorizon - 1].u[
            ControlIndex::contact_force(task.moving_foot, axis)] = 0.0;
    }
    const RealtimePlannerAudit checkpoint_audit =
        realtime_detail::audit_solution(
            *problem, dynamics, constraints, terrain, cost,
            task.origin_x, task.moving_foot, checkpoint_stage);
    const Vec<3> horizon_position = state_vector3(
        problem->stages[kRealtimeHorizon].x,
        StateIndex::foot_position(task.moving_foot, 0));
    TerrainSample horizon_sample;
    if (!terrain.sample(horizon_position, horizon_sample)) return false;
    const Vec<3> horizon_force = control_vector3(
        problem->stages[kRealtimeHorizon - 1].u,
        ControlIndex::contact_force(task.moving_foot, 0));
    if (dot3(horizon_sample.normal, horizon_force) >= 5.0 ||
        checkpoint_audit.terminal_normal_force < 5.0 ||
        checkpoint_audit.terminal_foot_error > 1e-12) {
        std::printf(
            "sustained audit did not use the task-settled checkpoint: "
            "horizon_force=%.3f checkpoint_force=%.3f error=%.3e\n",
            dot3(horizon_sample.normal, horizon_force),
            checkpoint_audit.terminal_normal_force,
            checkpoint_audit.terminal_foot_error);
        return false;
    }
    return true;
}

bool test_sustained_failed_candidate_rebase() {
    RealtimePlannerConfig configuration;
    configuration.sustained_contact_tasks = true;
    auto planner = std::make_unique<QuadrupedCITORealtimePlanner>(
        SharedTerrain::flat(), configuration);
    const Vec<kStateDim> standing = planner->nominal_state();
    if (planner->initialize(standing) != nmpc::Status::SUCCESS)
        return false;
    const RealtimePlannerResult cold = planner->cold_solve();
    if (!cold.published) return false;

    const auto converged =
        std::make_unique<QuadrupedCITORealtimePlanner::ProblemType>(
            planner->problem());
    const int shift = planner->shift_steps();
    Vec<kStateDim> disturbed = converged->stages[shift].x;
    disturbed[StateIndex::base_position(0)] += 0.02;
    disturbed[StateIndex::foot_position(0, 0)] += 0.01;
    const auto expired = QuadrupedCITORealtimePlanner::Clock::now() -
        std::chrono::milliseconds(1);
    const RealtimePlannerResult first_miss = planner->warm_update(
        disturbed, expired);
    if (first_miss.status != nmpc::Status::TIME_LIMIT ||
        !first_miss.shift_consumed || first_miss.published ||
        !same_problem(planner->problem(), *converged)) {
        std::printf("failed-candidate rebase setup did not time out\n");
        return false;
    }

    const int accumulated_shift = 2 * shift;
    Vec<kStateDim> measured = converged->stages[accumulated_shift].x;
    measured[StateIndex::base_position(0)] += 0.02;
    measured[StateIndex::foot_position(0, 0)] += 0.01;
    const RealtimePlannerResult second_miss = planner->warm_update(
        measured, expired);
    if (second_miss.status != nmpc::Status::TIME_LIMIT ||
        !second_miss.shift_consumed || second_miss.published ||
        first_miss.audit.dynamics < 1e-2 ||
        !realtime_detail::finite_value(second_miss.audit.dynamics) ||
        !realtime_detail::finite_value(second_miss.audit.inequality) ||
        !realtime_detail::finite_value(second_miss.audit.mpcc) ||
        second_miss.audit.dynamics > 1e-10 ||
        planner->absolute_knot() != accumulated_shift ||
        !same_problem(planner->problem(), *converged)) {
        std::printf(
            "rebased candidate cascaded: status=%s first=%.3e "
            "second=%.3e knot=%d\n",
            nmpc::status_string(second_miss.status),
            first_miss.audit.dynamics, second_miss.audit.dynamics,
            planner->absolute_knot());
        return false;
    }
    return true;
}

bool test_publication_headroom_validation() {
    for (double invalid : {-1.0,
                           std::numeric_limits<double>::quiet_NaN()}) {
        RealtimePlannerConfig configuration;
        configuration.publication_headroom_ms = invalid;
        auto planner = std::make_unique<QuadrupedCITORealtimePlanner>(
            SharedTerrain::flat(), configuration);
        if (planner->initialize(planner->nominal_state()) !=
            nmpc::Status::BAD_ARGUMENT) {
            std::printf("invalid publication headroom was accepted\n");
            return false;
        }
    }
    return true;
}

bool test_sustained_timeout_recovery() {
    RealtimePlannerConfig configuration;
    configuration.sustained_contact_tasks = true;
    auto planner = std::make_unique<QuadrupedCITORealtimePlanner>(
        SharedTerrain::flat(), configuration);
    const Vec<kStateDim> standing = planner->nominal_state();
    if (planner->initialize(standing) != nmpc::Status::SUCCESS)
        return false;
    const RealtimePlannerResult cold = planner->cold_solve();
    if (!cold.published || cold.task_id != 0 ||
        cold.moving_foot != kRealtimeSustainedFootOrder[0] ||
        planner->last_valid_task() == nullptr ||
        planner->last_valid_task()->id != 0) {
        std::printf("sustained cold plan failed: %s published=%d\n",
                    nmpc::status_string(cold.status), cold.published ? 1 : 0);
        return false;
    }

    const RealtimeContactTask published_task = *planner->last_valid_task();
    for (int update = 0; update < 7; ++update) {
        const Vec<kStateDim> measured =
            planner->problem().stages[planner->shift_steps()].x;
        const auto deadline = QuadrupedCITORealtimePlanner::Clock::now() +
            std::chrono::milliseconds(200);
        const RealtimePlannerResult warm = planner->warm_update(
            measured, deadline);
        if (!warm.published || warm.task_id != published_task.id) {
            std::printf("sustained pre-boundary update %d failed: %s\n",
                        update, nmpc::status_string(warm.status));
            return false;
        }
    }
    const int elapsed_periods = 1;
    const int shift = planner->shift_steps();
    const auto converged_problem =
        std::make_unique<QuadrupedCITORealtimePlanner::ProblemType>(
            planner->problem());
    const Vec<kStateDim> timeout_measurement =
        converged_problem->stages[shift].x;
    const auto expired = QuadrupedCITORealtimePlanner::Clock::now() -
        std::chrono::milliseconds(1);
    const RealtimePlannerResult timeout = planner->warm_update(
        timeout_measurement, expired, elapsed_periods);
    if (timeout.published || !timeout.shift_consumed ||
        !timeout.deadline_miss || timeout.moving_foot != -1 ||
        planner->absolute_knot() != 8 * planner->shift_steps() ||
        planner->active_task().id != 1 ||
        planner->last_valid_task() == nullptr ||
        planner->last_valid_task()->id != published_task.id ||
        !same_problem(planner->problem(), *converged_problem)) {
        std::printf("sustained timeout leaked candidate task metadata\n");
        return false;
    }

    const Vec<kStateDim> recovery_measurement =
        converged_problem->stages[2 * planner->shift_steps()].x;
    const RealtimePlannerResult recovery = planner->warm_update(
        recovery_measurement);
    if (!recovery.published || !recovery.shift_consumed ||
        recovery.task_id != planner->active_task().id ||
        recovery.moving_foot != planner->active_task().moving_foot ||
        !recovery.task_transition_witness ||
        planner->absolute_knot() != 9 * planner->shift_steps() ||
        planner->last_valid_task() == nullptr ||
        planner->last_valid_task()->id != recovery.task_id) {
        std::printf("sustained timeout recovery failed: %s\n",
                    nmpc::status_string(recovery.status));
        return false;
    }
    return true;
}

bool test_sustained_model_matched_smoke() {
    RealtimePlannerConfig configuration;
    configuration.sustained_contact_tasks = true;
    auto planner = std::make_unique<QuadrupedCITORealtimePlanner>(
        SharedTerrain::flat(), configuration);
    const Vec<kStateDim> standing = planner->nominal_state();
    if (planner->initialize(standing) != nmpc::Status::SUCCESS)
        return false;
    const RealtimePlannerResult cold = planner->cold_solve();
    if (!cold.published) return false;

    constexpr int kUpdates = 100;
    double latencies[kUpdates];
    int publications = 0;
    int fallbacks = 0;
    int task_transitions = 0;
    std::uint64_t previous_task = cold.task_id;
    for (int update = 0; update < kUpdates; ++update) {
        const Vec<kStateDim> measured =
            planner->problem().stages[planner->shift_steps()].x;
        const auto deadline = QuadrupedCITORealtimePlanner::Clock::now() +
            std::chrono::milliseconds(200);
        const RealtimePlannerResult warm = planner->warm_update(
            measured, deadline);
        latencies[update] = warm.end_to_end_ms;
        if (warm.end_to_end_ms > 180.0) {
            std::printf(
                "sustained slow update=%d task=%llu latency=%.3f "
                "iters=%d line_search=%d shift=%.3f model=%.3f "
                "kkt=%.3f riccati=%.3f ls=%.3f residual=%.3f "
                "finalize=%.3f audit=%.3f extract=%.3f\n",
                update,
                static_cast<unsigned long long>(planner->active_task().id),
                warm.end_to_end_ms, warm.stats.inner_iterations,
                warm.stats.line_search_evals, warm.shift_ms,
                warm.stats.model_eval_time_ms,
                warm.stats.kkt_assembly_time_ms,
                warm.stats.riccati_time_ms,
                warm.stats.line_search_time_ms,
                warm.stats.residual_eval_time_ms,
                warm.stats.finalization_time_ms,
                warm.audit_ms, warm.plan_extract_ms);
        }
        publications += warm.published ? 1 : 0;
        fallbacks += warm.fallback ? 1 : 0;
        if (!warm.published || !warm.full_kkt || !warm.task_pass ||
            warm.stats.exact_hessian_analytic_calls != 0 ||
            warm.stats.exact_hessian_fd_calls != 0 ||
            planner->last_valid_task() == nullptr ||
            planner->last_valid_task()->id != warm.task_id) {
            std::printf(
                "sustained model-matched update %d failed: task=%llu "
                "status=%s published=%d fallback=%d full_kkt=%d "
                "task_pass=%d clearance=%.6f unloaded=%d disp=%.6f "
                "term_err=%.6f dyn=%.3e ineq=%.3e audit_mpcc=%.3e "
                "gap=%.3e speed=%.3e force=%.3f primal=%.3e dual=%.3e "
                "compl=%.3e solver_mpcc=%.3e mu=%.3e latency=%.3f\n",
                update, static_cast<unsigned long long>(
                            planner->active_task().id),
                nmpc::status_string(warm.status), warm.published ? 1 : 0,
                warm.fallback ? 1 : 0, warm.full_kkt ? 1 : 0,
                warm.task_pass ? 1 : 0, warm.audit.clearance,
                warm.audit.moving_unloaded_stages,
                warm.audit.task_displacement,
                warm.audit.terminal_foot_error,
                warm.audit.dynamics, warm.audit.inequality,
                warm.audit.mpcc, warm.audit.terminal_gap,
                warm.audit.terminal_speed,
                warm.audit.terminal_normal_force,
                warm.stats.primal_infeas, warm.stats.dual_infeas,
                warm.stats.complementarity,
                warm.stats.mpcc_complementarity,
                warm.stats.barrier_param,
                warm.end_to_end_ms);
            return false;
        }
        if (warm.task_id != previous_task) {
            ++task_transitions;
            if (!warm.task_transition_witness) {
                std::printf("task %llu reused an old transition witness\n",
                            static_cast<unsigned long long>(warm.task_id));
                return false;
            }
            previous_task = warm.task_id;
        }
    }
    std::sort(latencies, latencies + kUpdates);
    const double p99 = latencies[98];
    const double maximum = latencies[kUpdates - 1];
    if (publications != kUpdates || fallbacks != 0 || p99 > 180.0 ||
        task_transitions < 12 || planner->active_task().id < 13) {
        std::printf(
            "sustained smoke incomplete: publications=%d fallbacks=%d "
            "transitions=%d final_task=%llu p99=%.3f ms\n",
            publications, fallbacks, task_transitions,
            static_cast<unsigned long long>(planner->active_task().id),
            p99);
        return false;
    }
    std::printf(
        "Sustained model-matched smoke: publications=%d fallbacks=%d "
        "transitions=%d final_task=%llu p99=%.3f ms max=%.3f ms\n",
        publications, fallbacks, task_transitions,
        static_cast<unsigned long long>(planner->active_task().id),
        p99, maximum);
    return true;
}

}  // namespace

int main() {
    if (!test_measured_state_correction()) return 1;
    if (!test_sustained_task_sequence()) return 1;
    if (!test_sustained_warm_tail_rollout_seed()) return 1;
    if (!test_random_smooth_measured_state_seed()) return 1;
    if (!test_sustained_checkpoint_audit()) return 1;
    if (!test_publication_headroom_validation()) return 1;
    if (!test_sustained_failed_candidate_rebase()) return 1;
    if (!test_sustained_timeout_recovery()) return 1;
    if (!test_sustained_model_matched_smoke()) return 1;
    if (!test_cold_retry_status_filter()) return 1;
    if (!test_initial_force_projection(SharedTerrain::flat()) ||
        !test_initial_force_projection(SharedTerrain::slope(0.10)) ||
        !test_initial_force_projection(SharedTerrain::sinusoidal(0.02))) {
        return 1;
    }
    if (!test_sloped_warm_solve(0.10, 0.0) ||
        !test_sloped_warm_solve(-0.10, 0.0) ||
        !test_sloped_warm_solve(0.0, 0.10)) {
        return 1;
    }
    if (!test_random_cold_alternate_guess() ||
        !test_random_warm_tail_initialization()) {
        return 1;
    }

    auto planner = std::make_unique<QuadrupedCITORealtimePlanner>(
        SharedTerrain::flat());
    const Vec<kStateDim> standing = planner->nominal_state();
    if (planner->initialize(standing) != nmpc::Status::SUCCESS) {
        std::printf("realtime planner initialization failed\n");
        return 1;
    }
    const RealtimePlannerResult cold = planner->cold_solve();
    if (!cold.published || cold.cold_fallback_used ||
        planner->last_valid_plan() == nullptr) {
        std::printf("realtime cold plan was not publishable\n");
        return 1;
    }

    auto checkpoint =
        std::make_unique<QuadrupedCITORealtimePlanner::ProblemType>(
            planner->problem());
    const ContactPlan<kRealtimeHorizon> published =
        *planner->last_valid_plan();
    const int base_shift = planner->shift_steps();
    if (planner->next_shift_steps() != base_shift) return 1;

    const RealtimePlannerResult invalid_elapsed = planner->warm_update(
        checkpoint->stages[base_shift].x,
        QuadrupedCITORealtimePlanner::Clock::time_point::max(), 0);
    if (invalid_elapsed.status != nmpc::Status::BAD_ARGUMENT ||
        invalid_elapsed.shift_consumed ||
        !same_problem(planner->problem(), *checkpoint)) {
        std::printf("invalid elapsed period consumed a planner shift\n");
        return 1;
    }

    const auto expired = QuadrupedCITORealtimePlanner::Clock::now() -
        std::chrono::milliseconds(1);
    constexpr int kElapsedPeriodsAfterInjectedDrop = 2;
    const int accumulated_shift =
        kElapsedPeriodsAfterInjectedDrop * base_shift;
    const Vec<kStateDim> timeout_measurement =
        checkpoint->stages[accumulated_shift].x;
    const RealtimePlannerResult timeout = planner->warm_update(
        timeout_measurement, expired,
        kElapsedPeriodsAfterInjectedDrop);
    if (timeout.published || !timeout.fallback || !timeout.deadline_miss ||
        !timeout.shift_consumed ||
        timeout.status != nmpc::Status::TIME_LIMIT) {
        std::printf("deadline miss did not use audited fallback\n");
        return 1;
    }
    if (!same_problem(planner->problem(), *checkpoint)) {
        std::printf("deadline miss retained a failed private warm state\n");
        return 1;
    }
    if (planner->last_valid_plan() == nullptr ||
        !same_plan(*planner->last_valid_plan(), published)) {
        std::printf("deadline miss changed published execution plan\n");
        return 1;
    }
    if (planner->next_shift_steps() != base_shift) {
        std::printf("deadline miss compounded the next horizon shift\n");
        return 1;
    }

    const Vec<kStateDim> recovery_measurement =
        checkpoint->stages[accumulated_shift + base_shift].x;
    const RealtimePlannerResult recovery = planner->warm_update(
        recovery_measurement);
    if (!recovery.published || recovery.fallback ||
        !recovery.shift_consumed ||
        recovery.status != nmpc::Status::SUCCESS ||
        planner->last_valid_plan() == nullptr) {
        std::printf("private warm state did not recover to publication: %s\n",
                    nmpc::status_string(recovery.status));
        return 1;
    }

    std::printf("Realtime planner recovery invariants passed.\n");
    return 0;
}
