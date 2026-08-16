#pragma once

#include <algorithm>
#include <chrono>
#include <cmath>
#include <cstdint>
#include <memory>

#include "examples/quadruped_cito/quadruped_cito_go1.hpp"
#include "examples/quadruped_cito/quadruped_cito_model.hpp"
#include "examples/quadruped_cito/quadruped_cito_plan.hpp"
#include "nmpc/contact_ipm.hpp"

namespace quadruped_cito {

constexpr int kFourStepHorizon = 100;
constexpr double kFourStepTimeStep = 0.06;

enum class FourStepSeed {
    FORWARD,
    REVERSE,
    LEFT_FIRST,
    RIGHT_FIRST
};

constexpr int kFourStepSeedCount = 4;

inline const char* four_step_seed_name(FourStepSeed seed) {
    switch (seed) {
        case FourStepSeed::FORWARD: return "forward";
        case FourStepSeed::REVERSE: return "reverse";
        case FourStepSeed::LEFT_FIRST: return "left_first";
        case FourStepSeed::RIGHT_FIRST: return "right_first";
    }
    return "unknown";
}

struct FourStepPlannerReport {
    nmpc::Status status = nmpc::Status::NOT_INITIALIZED;
    nmpc::Status solver_status = nmpc::Status::NOT_INITIALIZED;
    FourStepSeed seed = FourStepSeed::FORWARD;
    bool audit_passed = false;
    bool schedule_audit_passed = false;
    int iterations = 0;
    double solve_ms = 0.0;
    double objective = 1e300;
    double seed_dynamics = 0.0;
    double dynamics = 0.0;
    double inequality = 0.0;
    double mpcc = 0.0;
    double base_displacement = 0.0;
    double foot_displacement[kNumFeet] = {};
    double foot_clearance[kNumFeet] = {};
    double moving_normal_force[kNumFeet] = {};
    double terminal_normal_force[kNumFeet];
    double terminal_gap[kNumFeet] = {};
    double terminal_speed[kNumFeet] = {};
    int contact_events[kNumFeet] = {};
    int first_motion_stage[kNumFeet];
    int seed_liftoff_stage[kNumFeet];
    int seed_touchdown_stage[kNumFeet];
    int optimized_liftoff_stage[kNumFeet];
    int optimized_touchdown_stage[kNumFeet];
    int schedule_timing_change[kNumFeet] = {};
    int seed_contact_masks[kFourStepHorizon] = {};
    int optimized_contact_masks[kFourStepHorizon] = {};
    std::uint64_t seed_schedule_fingerprint = 0;
    std::uint64_t optimized_schedule_fingerprint = 0;
    int seed_worst_support_stage = -1;
    int seed_worst_excluded_foot = -1;
    double seed_minimum_support_weight = 0.0;
    int seed_support_projections = 0;

    FourStepPlannerReport() {
        for (int foot = 0; foot < kNumFeet; ++foot) {
            terminal_normal_force[foot] = 1e300;
            first_motion_stage[foot] = -1;
            seed_liftoff_stage[foot] = -1;
            seed_touchdown_stage[foot] = -1;
            optimized_liftoff_stage[foot] = -1;
            optimized_touchdown_stage[foot] = -1;
        }
    }
};

struct FourStepMultiStartReport {
    FourStepPlannerReport candidates[kFourStepSeedCount];
    int successful_candidates = 0;
    int selected_candidate = -1;
    double total_solve_ms = 0.0;
};

namespace four_step_detail {

constexpr int kTerminalSupportStages = 5;
constexpr int kSwingFeet[kNumFeet] = {2, 3, 0, 1};
constexpr double kRequestedStep = 0.08;
constexpr double kSeedSwingClearance = 0.05;
constexpr int kFirstSwingStage = 3;
constexpr int kSwingSpacing = 24;
constexpr int kThirdSwingAdvance = 2;
constexpr int kFinalSwingAdvance = 3;
constexpr int kSwingStages = 20;
constexpr int kSeedSwingOrders[kFourStepSeedCount][kNumFeet] = {
    {0, 1, 2, 3},
    {3, 2, 1, 0},
    {0, 2, 1, 3},
    {1, 3, 0, 2}};
using PlannerTerrain = Go1FootCenterTerrain<SharedTerrain>;

inline int seed_index(FourStepSeed seed) {
    return static_cast<int>(seed);
}

inline void configure_cost(
    QuadraticTrackingCost<kFourStepHorizon>& cost) {
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
                0.002;
            cost.terminal_weights[
                StateIndex::foot_position(foot, axis)] = 1000.0;
        }
        cost.control_weights[ControlIndex::motion_slack(foot)] = 0.002;
        cost.references[kFourStepHorizon][
            StateIndex::foot_position(foot, 0)] += kRequestedStep;
    }
}

inline bool project_support_weights(double weights[kNumFeet],
                                    int stage, int excluded_foot,
                                    FourStepPlannerReport& report) {
    double sum = 0.0;
    double minimum = weights[0];
    for (int foot = 0; foot < kNumFeet; ++foot) {
        minimum = std::min(minimum, weights[foot]);
        weights[foot] = std::max(0.0, weights[foot]);
        sum += weights[foot];
    }
    if (minimum < report.seed_minimum_support_weight) {
        report.seed_minimum_support_weight = minimum;
        report.seed_worst_support_stage = stage;
        report.seed_worst_excluded_foot = excluded_foot;
    }
    if (minimum < 0.0) ++report.seed_support_projections;
    if (!(sum > 1e-12)) return false;
    for (int foot = 0; foot < kNumFeet; ++foot) weights[foot] /= sum;
    return true;
}

inline bool three_support_weights(const Vec<kStateDim>& state,
                                  int excluded_foot,
                                  double weights[kNumFeet],
                                  int stage,
                                  FourStepPlannerReport& report) {
    int active[3];
    int count = 0;
    for (int foot = 0; foot < kNumFeet; ++foot) {
        weights[foot] = 0.0;
        if (foot != excluded_foot) active[count++] = foot;
    }
    const double xa = state[StateIndex::foot_position(active[0], 0)];
    const double ya = state[StateIndex::foot_position(active[0], 1)];
    const double xb = state[StateIndex::foot_position(active[1], 0)];
    const double yb = state[StateIndex::foot_position(active[1], 1)];
    const double xc = state[StateIndex::foot_position(active[2], 0)];
    const double yc = state[StateIndex::foot_position(active[2], 1)];
    const double support_x = state[StateIndex::base_position(0)];
    const double support_y = state[StateIndex::base_position(1)];
    const double denominator =
        (yb - yc) * (xa - xc) + (xc - xb) * (ya - yc);
    if (std::fabs(denominator) < 1e-12) return false;
    weights[active[0]] =
        ((yb - yc) * (support_x - xc) +
         (xc - xb) * (support_y - yc)) / denominator;
    weights[active[1]] =
        ((yc - ya) * (support_x - xc) +
         (xa - xc) * (support_y - yc)) / denominator;
    weights[active[2]] = 1.0 - weights[active[0]] - weights[active[1]];
    return project_support_weights(weights, stage, excluded_foot, report);
}

inline bool four_support_weights(const Vec<kStateDim>& state,
                                 double weights[kNumFeet],
                                 int stage,
                                 FourStepPlannerReport& report) {
    const double front_x = 0.5 *
        (state[StateIndex::foot_position(0, 0)] +
         state[StateIndex::foot_position(1, 0)]);
    const double rear_x = 0.5 *
        (state[StateIndex::foot_position(2, 0)] +
         state[StateIndex::foot_position(3, 0)]);
    const double left_y = 0.5 *
        (state[StateIndex::foot_position(0, 1)] +
         state[StateIndex::foot_position(2, 1)]);
    const double right_y = 0.5 *
        (state[StateIndex::foot_position(1, 1)] +
         state[StateIndex::foot_position(3, 1)]);
    if (std::fabs(front_x - rear_x) < 1e-12 ||
        std::fabs(left_y - right_y) < 1e-12) {
        return false;
    }
    const double positive_x =
        (state[StateIndex::base_position(0)] - rear_x) /
        (front_x - rear_x);
    const double positive_y =
        (state[StateIndex::base_position(1)] - right_y) /
        (left_y - right_y);
    weights[0] = positive_x * positive_y;
    weights[1] = positive_x * (1.0 - positive_y);
    weights[2] = (1.0 - positive_x) * positive_y;
    weights[3] = (1.0 - positive_x) * (1.0 - positive_y);
    return project_support_weights(weights, stage, -1, report);
}

inline double smooth_phase(int stage, int first_stage, int last_stage) {
    const double raw = static_cast<double>(stage - first_stage) /
                       static_cast<double>(last_stage - first_stage);
    const double phase = std::max(0.0, std::min(1.0, raw));
    return phase * phase * (3.0 - 2.0 * phase);
}

inline bool initialize_guess(Problem<kFourStepHorizon>& problem,
                             const RobotParameters& parameters,
                             const PlannerTerrain& terrain,
                             FourStepSeed seed,
                             bool advance_late_swings,
                             FourStepPlannerReport& report) {
    int order[kNumFeet];
    int starts[kNumFeet];
    int ends[kNumFeet];
    for (int sequence = 0; sequence < kNumFeet; ++sequence) {
        order[sequence] =
            kSeedSwingOrders[seed_index(seed)][sequence];
        const int swing_advance = advance_late_swings
            ? (sequence == kNumFeet - 2
                   ? kThirdSwingAdvance
                   : (sequence == kNumFeet - 1 ? kFinalSwingAdvance : 0))
            : 0;
        starts[order[sequence]] =
            kFirstSwingStage + kSwingSpacing * sequence - swing_advance;
        ends[order[sequence]] = starts[order[sequence]] + kSwingStages;
    }

    double x_path[kNumFeet][kFourStepHorizon + 1];
    double z_path[kNumFeet][kFourStepHorizon + 1];
    for (int swing = 0; swing < kNumFeet; ++swing) {
        const int foot = kSwingFeet[swing];
        const double initial_x =
            problem.x0[StateIndex::foot_position(foot, 0)];
        const double foot_y =
            problem.x0[StateIndex::foot_position(foot, 1)];
        for (int stage = 0; stage <= kFourStepHorizon; ++stage) {
            const double raw = static_cast<double>(stage - starts[swing]) /
                               static_cast<double>(ends[swing] -
                                                   starts[swing]);
            const double phase = std::max(0.0, std::min(1.0, raw));
            x_path[swing][stage] = initial_x +
                kRequestedStep * smooth_phase(
                    stage, starts[swing], ends[swing]);
            z_path[swing][stage] =
                terrain.height(x_path[swing][stage], foot_y) +
                kSeedSwingClearance *
                    std::sin(3.14159265358979323846 * phase);
        }
    }

    for (int stage = 0; stage <= kFourStepHorizon; ++stage) {
        problem.stages[stage].x = problem.x0;
        problem.stages[stage].u.zero();
        for (int swing = 0; swing < kNumFeet; ++swing) {
            const int foot = kSwingFeet[swing];
            problem.stages[stage].x[
                StateIndex::foot_position(foot, 0)] = x_path[swing][stage];
            problem.stages[stage].x[
                StateIndex::foot_position(foot, 2)] = z_path[swing][stage];
        }
    }

    for (int stage = 0; stage < kFourStepHorizon; ++stage) {
        int moving_foot = -1;
        for (int swing = 0; swing < kNumFeet; ++swing) {
            const int foot = kSwingFeet[swing];
            double speed_sq = 0.0;
            const int moving_axes[2] = {0, 2};
            for (int axis : moving_axes) {
                const double velocity =
                    (axis == 0
                         ? x_path[swing][stage + 1] - x_path[swing][stage]
                         : z_path[swing][stage + 1] - z_path[swing][stage]) /
                    kFourStepTimeStep;
                problem.stages[stage].u[
                    ControlIndex::foot_velocity(foot, axis)] = velocity;
                speed_sq += velocity * velocity;
            }
            const double speed = std::sqrt(speed_sq);
            problem.stages[stage].u[
                ControlIndex::motion_slack(foot)] = speed;
            if (speed > 1e-12) moving_foot = foot;
        }

        double weights[kNumFeet];
        const int projections_before = report.seed_support_projections;
        if (moving_foot >= 0) {
            if (!three_support_weights(problem.stages[stage].x,
                                       moving_foot, weights, stage, report)) {
                report.seed_worst_support_stage = stage;
                report.seed_worst_excluded_foot = moving_foot;
                report.seed_minimum_support_weight = *std::min_element(
                    weights, weights + kNumFeet);
                return false;
            }
        } else {
            int next_unloaded = -1;
            for (int sequence = 0; sequence < kNumFeet; ++sequence) {
                const int swing = order[sequence];
                if (stage < starts[swing]) {
                    next_unloaded = kSwingFeet[swing];
                    break;
                }
            }
            if (next_unloaded < 0 || stage < starts[order[0]]) {
                if (!four_support_weights(problem.stages[stage].x,
                                          weights, stage, report)) {
                    report.seed_worst_support_stage = stage;
                    report.seed_worst_excluded_foot = -1;
                    report.seed_minimum_support_weight = *std::min_element(
                        weights, weights + kNumFeet);
                    return false;
                }
            } else if (!three_support_weights(problem.stages[stage].x,
                                              next_unloaded, weights,
                                              stage, report)) {
                report.seed_worst_support_stage = stage;
                report.seed_worst_excluded_foot = next_unloaded;
                report.seed_minimum_support_weight = *std::min_element(
                    weights, weights + kNumFeet);
                return false;
            }
        }
        if (report.seed_support_projections > projections_before) {
            for (int axis = 0; axis < 2; ++axis) {
                double projected_position = 0.0;
                for (int foot = 0; foot < kNumFeet; ++foot) {
                    projected_position += weights[foot] *
                        problem.stages[stage].x[
                            StateIndex::foot_position(foot, axis)];
                }
                problem.stages[stage].x[
                    StateIndex::base_position(axis)] = projected_position;
            }
        }
        const double total_force = parameters.mass * parameters.gravity;
        for (int foot = 0; foot < kNumFeet; ++foot) {
            problem.stages[stage].u[
                ControlIndex::contact_force(foot, 2)] =
                total_force * weights[foot];
        }
    }
    return true;
}

inline double dynamics_defect(const Problem<kFourStepHorizon>& problem,
                              SRBDDynamics& dynamics) {
    double maximum = 0.0;
    for (int stage = 0; stage < kFourStepHorizon; ++stage) {
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

inline double trajectory_objective(
    Problem<kFourStepHorizon>& problem,
    QuadraticTrackingCost<kFourStepHorizon>& cost) {
    double objective = 0.0;
    for (int stage = 0; stage < kFourStepHorizon; ++stage) {
        objective += cost.stage_cost(problem.stages[stage].x,
                                     problem.stages[stage].u, stage);
    }
    return objective +
        cost.terminal_cost(problem.stages[kFourStepHorizon].x);
}

inline void record_contact_schedule(
    const Problem<kFourStepHorizon>& problem, const PlannerTerrain& terrain,
    int contact_masks[kFourStepHorizon], int liftoff_stage[kNumFeet],
    int touchdown_stage[kNumFeet], std::uint64_t& fingerprint) {
    bool previous_contact[kNumFeet] = {true, true, true, true};
    fingerprint = 1469598103934665603ULL;
    for (int foot = 0; foot < kNumFeet; ++foot) {
        liftoff_stage[foot] = -1;
        touchdown_stage[foot] = -1;
    }
    for (int stage = 0; stage < kFourStepHorizon; ++stage) {
        int mask = 0;
        for (int foot = 0; foot < kNumFeet; ++foot) {
            const Vec<3> position = state_vector3(
                problem.stages[stage].x,
                StateIndex::foot_position(foot, 0));
            const Vec<3> force = control_vector3(
                problem.stages[stage].u,
                ControlIndex::contact_force(foot, 0));
            TerrainSample sample;
            terrain.sample(position, sample);
            const bool contact = sample.gap <= 2e-3 &&
                dot3(sample.normal, force) >= 5.0;
            if (contact) mask |= 1 << foot;
            if (previous_contact[foot] && !contact &&
                liftoff_stage[foot] < 0) {
                liftoff_stage[foot] = stage;
            }
            if (!previous_contact[foot] && contact &&
                touchdown_stage[foot] < 0) {
                touchdown_stage[foot] = stage;
            }
            previous_contact[foot] = contact;
        }
        contact_masks[stage] = mask;
        fingerprint ^= static_cast<std::uint64_t>(mask + 17 * stage);
        fingerprint *= 1099511628211ULL;
    }
}

inline void audit_solution(
    const Problem<kFourStepHorizon>& problem, SRBDDynamics& dynamics,
    ContactConstraints<kFourStepHorizon, PlannerTerrain>& constraints,
    const PlannerTerrain& terrain, FourStepPlannerReport& report) {
    report.dynamics = dynamics_defect(problem, dynamics);
    record_contact_schedule(
        problem, terrain, report.optimized_contact_masks,
        report.optimized_liftoff_stage, report.optimized_touchdown_stage,
        report.optimized_schedule_fingerprint);
    report.base_displacement =
        problem.stages[kFourStepHorizon].x[StateIndex::base_position(0)] -
        problem.x0[StateIndex::base_position(0)];
    bool previous_contact[kNumFeet] = {true, true, true, true};
    for (int foot = 0; foot < kNumFeet; ++foot) {
        report.foot_displacement[foot] =
            problem.stages[kFourStepHorizon].x[
                StateIndex::foot_position(foot, 0)] -
            problem.x0[StateIndex::foot_position(foot, 0)];
    }
    for (int stage = 0; stage < kFourStepHorizon; ++stage) {
        Vec<kConstraintCapacity> rows;
        constraints.evaluate(problem.stages[stage].x,
                             problem.stages[stage].u, stage, rows);
        for (int row = 0; row < kUserConstraintRows; ++row) {
            report.inequality = std::max(report.inequality, rows[row]);
        }
        for (int pair = 0; pair < kComplementarityPairs; ++pair) {
            int first = -1;
            int second = -1;
            constraints.complementarity_pair(stage, pair, first, second);
            report.mpcc = std::max(
                report.mpcc, std::fabs(rows[first] * rows[second]));
        }
        for (int foot = 0; foot < kNumFeet; ++foot) {
            const Vec<3> position = state_vector3(
                problem.stages[stage].x,
                StateIndex::foot_position(foot, 0));
            const Vec<3> velocity = control_vector3(
                problem.stages[stage].u,
                ControlIndex::foot_velocity(foot, 0));
            const Vec<3> force = control_vector3(
                problem.stages[stage].u,
                ControlIndex::contact_force(foot, 0));
            TerrainSample sample;
            terrain.sample(position, sample);
            const bool contact =
                (report.optimized_contact_masks[stage] & (1 << foot)) != 0;
            if (previous_contact[foot] && !contact)
                ++report.contact_events[foot];
            previous_contact[foot] = contact;
            report.foot_clearance[foot] = std::max(
                report.foot_clearance[foot], sample.gap);
            if (velocity.norm2() > 0.01) {
                report.moving_normal_force[foot] = std::max(
                    report.moving_normal_force[foot],
                    dot3(sample.normal, force));
                if (report.first_motion_stage[foot] < 0)
                    report.first_motion_stage[foot] = stage;
            }
            if (stage >= kFourStepHorizon - kTerminalSupportStages) {
                report.terminal_normal_force[foot] = std::min(
                    report.terminal_normal_force[foot],
                    dot3(sample.normal, force));
                report.terminal_gap[foot] = std::max(
                    report.terminal_gap[foot], sample.gap);
                report.terminal_speed[foot] = std::max(
                    report.terminal_speed[foot], velocity.norm2());
            }
        }
    }
    for (int foot = 0; foot < kNumFeet; ++foot) {
        if (report.seed_liftoff_stage[foot] >= 0 &&
            report.optimized_liftoff_stage[foot] >= 0) {
            report.schedule_timing_change[foot] += std::abs(
                report.optimized_liftoff_stage[foot] -
                report.seed_liftoff_stage[foot]);
        }
        if (report.seed_touchdown_stage[foot] >= 0 &&
            report.optimized_touchdown_stage[foot] >= 0) {
            report.schedule_timing_change[foot] += std::abs(
                report.optimized_touchdown_stage[foot] -
                report.seed_touchdown_stage[foot]);
        }
    }
}

inline bool audit_passes(const FourStepPlannerReport& report,
                         const nmpc::ContactIPMParams& parameters) {
    if (report.dynamics > parameters.tol_primal ||
        report.inequality > parameters.tol_ineq ||
        report.mpcc > parameters.tol_mpcc ||
        report.base_displacement < 0.05) {
        return false;
    }
    for (int foot = 0; foot < kNumFeet; ++foot) {
        if (report.foot_displacement[foot] < 0.06 ||
            report.foot_clearance[foot] < 0.02 ||
            report.moving_normal_force[foot] >= 5.0 ||
            report.terminal_normal_force[foot] < 5.0 ||
            report.terminal_gap[foot] > 1e-4 ||
            report.terminal_speed[foot] > 0.01 ||
            report.contact_events[foot] != 1) {
            return false;
        }
    }
    return true;
}

inline bool schedule_audit_passes(const FourStepPlannerReport& report) {
    for (int foot = 0; foot < kNumFeet; ++foot) {
        if (report.contact_events[foot] != 1) return false;
    }
    return true;
}

inline int schedule_deformation(const FourStepPlannerReport& report) {
    int total = 0;
    for (int foot = 0; foot < kNumFeet; ++foot)
        total += report.schedule_timing_change[foot];
    return total;
}

inline nmpc::ContactIPMParams solver_parameters() {
    nmpc::ContactIPMParams parameters;
    parameters.mu_init = 0.1;
    parameters.mu_min = 1e-5;
    parameters.mu_conv_threshold = 1e-5;
    parameters.max_same_mu = 100;
    parameters.max_iters = 700;
    parameters.mpcc_recovery_max_iters = 700;
    parameters.s_min_init = 0.1;
    parameters.tol_primal = 2e-5;
    parameters.tol_compl = 2e-5;
    parameters.tol_ineq = 1e-6;
    parameters.tol_stat = 0.2;
    parameters.tol_mpcc = 1e-4;
    parameters.exact_hessian = false;
    parameters.enable_preconditioner = true;
    parameters.verbosity = 0;
    return parameters;
}

}  // namespace four_step_detail

inline nmpc::Status plan_go1_four_step(
    const Vec<kStateDim>& initial_state, const SharedTerrain& physical_terrain,
    FourStepSeed seed, ContactPlan<kFourStepHorizon>& plan,
    FourStepPlannerReport* output_report = nullptr) {
    FourStepPlannerReport local_report;
    FourStepPlannerReport& report = output_report ? *output_report
                                                   : local_report;
    report = FourStepPlannerReport();
    report.seed = seed;
    RobotParameters parameters = go1_robot_parameters();
    parameters.minimum_pair_normal_force = 20.0;
    const four_step_detail::PlannerTerrain terrain(physical_terrain);
    SRBDDynamics dynamics(parameters);
    ContactConstraints<kFourStepHorizon, four_step_detail::PlannerTerrain>
        constraints(terrain, parameters);
    QuadraticTrackingCost<kFourStepHorizon> cost;
    auto problem_storage =
        std::make_unique<Problem<kFourStepHorizon>>();
    Problem<kFourStepHorizon>& problem = *problem_storage;
    initialize_standing_problem(problem, dynamics, cost, constraints,
                                kFourStepTimeStep);
    problem.x0 = initial_state;
    cost.set_reference_all(initial_state);
    for (int stage = 0; stage <= kFourStepHorizon; ++stage)
        problem.stages[stage].x = initial_state;
    four_step_detail::configure_cost(cost);
    cost.references[kFourStepHorizon][StateIndex::base_position(0)] += 0.08;
    for (int foot = 0; foot < kNumFeet; ++foot) {
        cost.references[kFourStepHorizon][
            StateIndex::foot_position(foot, 2)] =
            terrain.height(
                initial_state[StateIndex::foot_position(foot, 0)] + 0.08,
                initial_state[StateIndex::foot_position(foot, 1)]);
    }
    if (!four_step_detail::initialize_guess(
            problem, parameters, terrain, seed,
            physical_terrain.kind == TerrainKind::SINUSOIDAL ||
                physical_terrain.kind == TerrainKind::RANDOM_SMOOTH,
            report)) {
        report.status = nmpc::Status::INFEASIBLE;
        return report.status;
    }
    report.seed_dynamics = four_step_detail::dynamics_defect(problem,
                                                              dynamics);
    four_step_detail::record_contact_schedule(
        problem, terrain, report.seed_contact_masks,
        report.seed_liftoff_stage, report.seed_touchdown_stage,
        report.seed_schedule_fingerprint);

    const nmpc::ContactIPMParams solver_parameters =
        four_step_detail::solver_parameters();
    nmpc::ContactIPM<kStateDim, kControlDim, kConstraintCapacity,
                     kFourStepHorizon> solver;
    solver.configure(solver_parameters);
    const auto start = std::chrono::steady_clock::now();
    report.solver_status = solver.solve(problem);
    report.status = report.solver_status;
    const auto end = std::chrono::steady_clock::now();
    report.solve_ms =
        std::chrono::duration<double, std::milli>(end - start).count();
    report.iterations = solver.last_stats().inner_iterations;
    report.objective = four_step_detail::trajectory_objective(problem, cost);
    four_step_detail::audit_solution(problem, dynamics, constraints, terrain,
                                     report);
    report.schedule_audit_passed =
        four_step_detail::schedule_audit_passes(report);
    report.audit_passed =
        four_step_detail::audit_passes(report, solver_parameters);
    if (report.solver_status != nmpc::Status::SUCCESS)
        return report.status;
    if (!report.audit_passed) {
        report.status = nmpc::Status::INFEASIBLE;
        return report.status;
    }
    ContactClassification classification;
    if (extract_contact_plan(problem, terrain, classification, plan) !=
        Status::SUCCESS) {
        report.status = nmpc::Status::INTERNAL_ERROR;
    }
    return report.status;
}

inline nmpc::Status plan_go1_four_step(
    const Vec<kStateDim>& initial_state, const SharedTerrain& physical_terrain,
    bool reverse_order, ContactPlan<kFourStepHorizon>& plan,
    FourStepPlannerReport* output_report = nullptr) {
    return plan_go1_four_step(
        initial_state, physical_terrain,
        reverse_order ? FourStepSeed::REVERSE : FourStepSeed::FORWARD,
        plan, output_report);
}

inline nmpc::Status plan_go1_four_step_multistart(
    const Vec<kStateDim>& initial_state, const SharedTerrain& physical_terrain,
    ContactPlan<kFourStepHorizon>& plan,
    FourStepPlannerReport* selected_report = nullptr,
    FourStepMultiStartReport* output_report = nullptr) {
    FourStepMultiStartReport local_report;
    FourStepMultiStartReport& report = output_report ? *output_report
                                                      : local_report;
    report = FourStepMultiStartReport();
    nmpc::Status selected_status = nmpc::Status::INFEASIBLE;
    int best_schedule_deformation = 2 * kNumFeet * kFourStepHorizon;
    double best_objective = 1e300;
    for (int candidate = 0; candidate < kFourStepSeedCount; ++candidate) {
        ContactPlan<kFourStepHorizon> candidate_plan;
        FourStepPlannerReport& candidate_report = report.candidates[candidate];
        const nmpc::Status status = plan_go1_four_step(
            initial_state, physical_terrain,
            static_cast<FourStepSeed>(candidate), candidate_plan,
            &candidate_report);
        report.total_solve_ms += candidate_report.solve_ms;
        if (status != nmpc::Status::SUCCESS) continue;
        ++report.successful_candidates;
        const int candidate_schedule_deformation =
            four_step_detail::schedule_deformation(candidate_report);
        if (candidate_schedule_deformation > best_schedule_deformation ||
            (candidate_schedule_deformation == best_schedule_deformation &&
             candidate_report.objective >= best_objective)) {
            continue;
        }
        best_schedule_deformation = candidate_schedule_deformation;
        best_objective = candidate_report.objective;
        report.selected_candidate = candidate;
        selected_status = status;
        plan = candidate_plan;
        if (selected_report) *selected_report = candidate_report;
    }
    return selected_status;
}

}  // namespace quadruped_cito
