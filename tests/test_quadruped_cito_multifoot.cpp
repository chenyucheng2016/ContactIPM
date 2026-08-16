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
#include "nmpc/contact_ipm.hpp"

namespace {

using namespace quadruped_cito;

#if QUADRUPED_CITO_EIGHT_STEP
constexpr int kHorizon = 220;
constexpr int kTerminalSupportStages = 5;
constexpr int kMovingFootCount = 4;
constexpr int kMovingFeet[kMovingFootCount] = {2, 3, 0, 1};
constexpr int kSwingCount = 8;
constexpr int kSwingFeet[kSwingCount] = {2, 3, 0, 1, 2, 3, 0, 1};
constexpr double kMinimumBaseDisplacement = 0.10;
constexpr double kSeedSwingClearance = 0.06;
constexpr double kFootMotionWeight = 0.002;
constexpr double kRequestedStep = 0.08;
constexpr int kFirstSwingStage = 10;
constexpr int kSwingSpacing = 26;
constexpr int kSwingStages = 18;
#elif QUADRUPED_CITO_FOUR_STEP
constexpr int kHorizon = 100;
constexpr int kTerminalSupportStages = 5;
constexpr int kMovingFootCount = 4;
constexpr int kMovingFeet[kMovingFootCount] = {2, 3, 0, 1};
constexpr int kSwingCount = 4;
constexpr int kSwingFeet[kSwingCount] = {2, 3, 0, 1};
constexpr double kMinimumBaseDisplacement = 0.05;
constexpr double kSeedSwingClearance = 0.05;
constexpr double kFootMotionWeight = 0.002;
constexpr double kRequestedStep = 0.08;
constexpr int kFirstSwingStage = 3;
constexpr int kSwingSpacing = 24;
constexpr int kSwingStages = 20;
#elif QUADRUPED_CITO_LONG_HORIZON
constexpr int kHorizon = 100;
constexpr int kTerminalSupportStages = 10;
constexpr int kMovingFootCount = 2;
constexpr int kMovingFeet[kMovingFootCount] = {2, 3};
constexpr int kSwingCount = 2;
constexpr int kSwingFeet[kSwingCount] = {2, 3};
constexpr double kMinimumBaseDisplacement = -1.0;
constexpr double kSeedSwingClearance = 0.05;
constexpr double kFootMotionWeight = 0.02;
constexpr double kRequestedStep = 0.08;
constexpr int kFirstSwingStage = 3;
constexpr int kSwingSpacing = 24;
constexpr int kSwingStages = 20;
#else
constexpr int kHorizon = 50;
constexpr int kTerminalSupportStages = 3;
constexpr int kMovingFootCount = 2;
constexpr int kMovingFeet[kMovingFootCount] = {2, 3};
constexpr int kSwingCount = 2;
constexpr int kSwingFeet[kSwingCount] = {2, 3};
constexpr double kMinimumBaseDisplacement = -1.0;
constexpr double kSeedSwingClearance = 0.05;
constexpr double kFootMotionWeight = 0.02;
constexpr double kRequestedStep = 0.08;
constexpr int kFirstSwingStage = 3;
constexpr int kSwingSpacing = 24;
constexpr int kSwingStages = 20;
#endif
constexpr double kTimeStep = 0.06;
using PhysicalTerrain = SharedTerrain;
using PlannerTerrain = Go1FootCenterTerrain<PhysicalTerrain>;

struct AuditResult {
    double dynamics = 0.0;
    double inequality = 0.0;
    double contact_inequality = 0.0;
    double support_inequality = 0.0;
    double mpcc = 0.0;
    double base_displacement = 0.0;
    double displacement[kMovingFootCount] = {};
    double clearance[kMovingFootCount] = {};
    double maximum_normal_force_while_moving[kMovingFootCount] = {};
    double minimum_terminal_normal_force[kMovingFootCount];
    double maximum_terminal_gap[kMovingFootCount] = {};
    double maximum_terminal_speed[kMovingFootCount] = {};
    int moving_stages[kMovingFootCount] = {};
    int contact_events[kMovingFootCount] = {};
    int first_motion_stage[kMovingFootCount];
    int worst_contact_stage = -1;
    int worst_contact_row = -1;
    int worst_inequality_stage = -1;
    int worst_inequality_row = -1;

    AuditResult() {
        for (int moving = 0; moving < kMovingFootCount; ++moving) {
            minimum_terminal_normal_force[moving] = 1e300;
            first_motion_stage[moving] = -1;
        }
    }
};

double requested_displacement(int foot) {
    int swings = 0;
    for (int swing = 0; swing < kSwingCount; ++swing)
        if (kSwingFeet[swing] == foot) ++swings;
    return swings * kRequestedStep;
}

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
                kFootMotionWeight;
        }
        cost.control_weights[ControlIndex::motion_slack(foot)] =
            kFootMotionWeight;
    }
    for (int moving = 0; moving < kMovingFootCount; ++moving) {
        for (int axis = 0; axis < 3; ++axis) {
            cost.terminal_weights[
                StateIndex::foot_position(kMovingFeet[moving], axis)] =
                1000.0;
        }
        const int foot = kMovingFeet[moving];
        cost.references[kHorizon][StateIndex::foot_position(foot, 0)] +=
            requested_displacement(foot);
    }
}

bool three_support_weights_at(const Vec<kStateDim>& state, int excluded_foot,
                              double support_x, double support_y,
                              double weights[kNumFeet]) {
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
    for (int foot : active)
        if (weights[foot] < -1e-12) return false;
    return true;
}

bool three_support_weights(const Vec<kStateDim>& state, int excluded_foot,
                           double weights[kNumFeet]) {
    return three_support_weights_at(
        state, excluded_foot,
        state[StateIndex::base_position(0)],
        state[StateIndex::base_position(1)], weights);
}

bool support_weights_at(const Vec<kStateDim>& state, double support_x,
                        double support_y, double weights[kNumFeet]) {
    for (int excluded = 0; excluded < kNumFeet; ++excluded) {
        if (three_support_weights_at(state, excluded, support_x, support_y,
                                     weights)) {
            return true;
        }
    }
    return false;
}

double support_triangle_determinant(const Vec<kStateDim>& state, int first,
                                    int second, int third) {
    const double x0 = state[StateIndex::foot_position(first, 0)];
    const double y0 = state[StateIndex::foot_position(first, 1)];
    const double x1 = state[StateIndex::foot_position(second, 0)];
    const double y1 = state[StateIndex::foot_position(second, 1)];
    const double x2 = state[StateIndex::foot_position(third, 0)];
    const double y2 = state[StateIndex::foot_position(third, 1)];
    return (x1 - x0) * (y2 - y0) - (x2 - x0) * (y1 - y0);
}

bool balanced_support_weights_at(const Vec<kStateDim>& state,
                                 double support_x, double support_y,
                                 double minimum_weight,
                                 double weights[kNumFeet]) {
    if (!support_weights_at(state, support_x, support_y, weights))
        return false;
    double nullspace[kNumFeet];
    for (int omitted = 0; omitted < kNumFeet; ++omitted) {
        int active[3];
        int count = 0;
        for (int foot = 0; foot < kNumFeet; ++foot)
            if (foot != omitted) active[count++] = foot;
        nullspace[omitted] = (omitted % 2 == 0 ? 1.0 : -1.0) *
            support_triangle_determinant(
                state, active[0], active[1], active[2]);
    }
    double lower = -1e300;
    double upper = 1e300;
    for (int foot = 0; foot < kNumFeet; ++foot) {
        if (nullspace[foot] > 1e-12) {
            lower = std::max(lower,
                             (minimum_weight - weights[foot]) /
                                 nullspace[foot]);
        } else if (nullspace[foot] < -1e-12) {
            upper = std::min(upper,
                             (minimum_weight - weights[foot]) /
                                 nullspace[foot]);
        } else if (weights[foot] < minimum_weight) {
            return false;
        }
    }
    if (lower > upper) return false;
    const double adjustment = 0.5 * (lower + upper);
    for (int foot = 0; foot < kNumFeet; ++foot)
        weights[foot] += adjustment * nullspace[foot];
    return true;
}

void four_support_weights(const Vec<kStateDim>& state,
                           double weights[kNumFeet]) {
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
}

double smooth_phase(int stage, int first_stage, int last_stage) {
    const double raw = static_cast<double>(stage - first_stage) /
                       static_cast<double>(last_stage - first_stage);
    const double phase = std::max(0.0, std::min(1.0, raw));
    return phase * phase * (3.0 - 2.0 * phase);
}

bool initialize_guess(Problem<kHorizon>& problem,
                       const RobotParameters& parameters,
                       const PlannerTerrain& terrain,
                       bool reverse_order) {
    int order[kSwingCount];
    int starts[kSwingCount];
    int ends[kSwingCount];
    for (int sequence = 0; sequence < kSwingCount; ++sequence) {
        order[sequence] = reverse_order
            ? kSwingCount - 1 - sequence
            : sequence;
        starts[order[sequence]] =
            kFirstSwingStage + kSwingSpacing * sequence;
        ends[order[sequence]] = starts[order[sequence]] + kSwingStages;
    }
    double x_path[kMovingFootCount][kHorizon + 1];
    double z_path[kMovingFootCount][kHorizon + 1];
    for (int moving = 0; moving < kMovingFootCount; ++moving) {
        const int foot = kMovingFeet[moving];
        const double initial_x =
            problem.x0[StateIndex::foot_position(foot, 0)];
        const double foot_y =
            problem.x0[StateIndex::foot_position(foot, 1)];
        for (int stage = 0; stage <= kHorizon; ++stage) {
            double progress = 0.0;
            double swing_height = 0.0;
            for (int swing = 0; swing < kSwingCount; ++swing) {
                if (kSwingFeet[swing] != foot) continue;
                const double raw =
                    static_cast<double>(stage - starts[swing]) /
                    static_cast<double>(ends[swing] - starts[swing]);
                const double phase = std::max(0.0, std::min(1.0, raw));
                progress += smooth_phase(stage, starts[swing], ends[swing]);
                swing_height = std::max(
                    swing_height,
                    std::sin(3.14159265358979323846 * phase));
            }
            x_path[moving][stage] =
                initial_x + kRequestedStep * progress;
            z_path[moving][stage] =
                terrain.height(x_path[moving][stage], foot_y) +
                kSeedSwingClearance * swing_height;
        }
    }

    const Vec<kStateDim> standing = problem.x0;
    for (int stage = 0; stage <= kHorizon; ++stage) {
        problem.stages[stage].x = standing;
        problem.stages[stage].u.zero();
        for (int moving = 0; moving < kMovingFootCount; ++moving) {
            const int foot = kMovingFeet[moving];
            problem.stages[stage].x[
                StateIndex::foot_position(foot, 0)] = x_path[moving][stage];
            problem.stages[stage].x[
                StateIndex::foot_position(foot, 2)] = z_path[moving][stage];
        }
    }

#if QUADRUPED_CITO_EIGHT_STEP
    double base_x_path[kHorizon + 1];
    double base_y_path[kHorizon + 1];
    const double initial_base_x = standing[StateIndex::base_position(0)];
    const double initial_base_y = standing[StateIndex::base_position(1)];
    for (int stage = 0; stage <= kHorizon; ++stage) {
        base_x_path[stage] = initial_base_x;
        base_y_path[stage] = initial_base_y;
    }
    double previous_x = initial_base_x;
    double previous_y = initial_base_y;
    int previous_stage = 1;
    for (int sequence = 0; sequence < kSwingCount; ++sequence) {
        const int swing = order[sequence];
        const int excluded_foot = kSwingFeet[swing];
        double target_x = 0.0;
        const double target_y = initial_base_y;
        for (int moving = 0; moving < kMovingFootCount; ++moving) {
            const int foot = kMovingFeet[moving];
            if (foot == excluded_foot) continue;
            target_x += x_path[moving][starts[swing]] / 3.0;
        }
        for (int stage = previous_stage; stage <= starts[swing]; ++stage) {
            const double phase = smooth_phase(
                stage, previous_stage, starts[swing]);
            base_x_path[stage] =
                (1.0 - phase) * previous_x + phase * target_x;
            base_y_path[stage] =
                (1.0 - phase) * previous_y + phase * target_y;
        }
        for (int stage = starts[swing]; stage <= ends[swing]; ++stage) {
            base_x_path[stage] = target_x;
            base_y_path[stage] = target_y;
        }
        previous_stage = ends[swing];
        previous_x = target_x;
        previous_y = target_y;
    }
    const double terminal_base_x = initial_base_x + 2.0 * kRequestedStep;
    for (int stage = previous_stage; stage <= kHorizon; ++stage) {
        const double phase = smooth_phase(stage, previous_stage, kHorizon);
        base_x_path[stage] =
            (1.0 - phase) * previous_x + phase * terminal_base_x;
        base_y_path[stage] = (1.0 - phase) * previous_y +
                             phase * initial_base_y;
    }
    for (int stage = 0; stage <= kHorizon; ++stage) {
        problem.stages[stage].x[StateIndex::base_position(0)] =
            base_x_path[stage];
        problem.stages[stage].x[StateIndex::base_position(1)] =
            base_y_path[stage];
        for (int axis = 0; axis < 2; ++axis) {
            problem.stages[stage].x[StateIndex::linear_velocity(axis)] =
                stage < kHorizon
                    ? ((axis == 0 ? base_x_path[stage + 1]
                                  : base_y_path[stage + 1]) -
                       (axis == 0 ? base_x_path[stage]
                                  : base_y_path[stage])) /
                          kTimeStep
                    : 0.0;
        }
    }
#endif

    for (int stage = 0; stage < kHorizon; ++stage) {
        int moving_foot = -1;
        for (int moving = 0; moving < kMovingFootCount; ++moving) {
            const int foot = kMovingFeet[moving];
            double speed_sq = 0.0;
            const int moving_axes[2] = {0, 2};
            for (int axis : moving_axes) {
                const double next = axis == 0
                    ? x_path[moving][stage + 1]
                    : z_path[moving][stage + 1];
                const double current = axis == 0
                    ? x_path[moving][stage]
                    : z_path[moving][stage];
                const double velocity = (next - current) / kTimeStep;
                problem.stages[stage].u[
                    ControlIndex::foot_velocity(foot, axis)] = velocity;
                speed_sq += velocity * velocity;
            }
            const double speed = std::sqrt(speed_sq);
            problem.stages[stage].u[ControlIndex::motion_slack(foot)] = speed;
            if (speed > 1e-12) moving_foot = foot;
        }

        double weights[kNumFeet];
#if QUADRUPED_CITO_EIGHT_STEP
        Vec<3> net_force;
        for (int axis = 0; axis < 3; ++axis) {
            const double velocity = problem.stages[stage].x[
                StateIndex::linear_velocity(axis)];
            const double next_velocity = problem.stages[stage + 1].x[
                StateIndex::linear_velocity(axis)];
            net_force[axis] = parameters.mass *
                (next_velocity - velocity) / kTimeStep;
        }
        net_force[2] += parameters.mass * parameters.gravity;
        const double base_x = problem.stages[stage].x[
            StateIndex::base_position(0)];
        const double base_y = problem.stages[stage].x[
            StateIndex::base_position(1)];
        const double lever_height = problem.stages[stage].x[
            StateIndex::base_position(2)] - terrain.height(base_x, base_y);
        const double support_x = base_x -
            lever_height * net_force[0] / net_force[2];
        const double support_y = base_y -
            lever_height * net_force[1] / net_force[2];
        const bool weights_ok = moving_foot >= 0
            ? three_support_weights_at(problem.stages[stage].x, moving_foot,
                                       support_x, support_y, weights)
            : balanced_support_weights_at(
                  problem.stages[stage].x, support_x, support_y,
                  0.081, weights);
        if (!weights_ok) {
            std::printf(
                "support-transfer seed failed: stage=%d excluded_foot=%d "
                "base=(%.6f,%.6f) support=(%.6f,%.6f)\n",
                stage, moving_foot, base_x, base_y, support_x, support_y);
            return false;
        }
#else
        const double total_force = parameters.mass * parameters.gravity;
        if (moving_foot >= 0) {
            if (!three_support_weights(problem.stages[stage].x, moving_foot,
                                       weights)) {
                std::printf(
                    "three-support seed failed: stage=%d excluded_foot=%d "
                    "base=(%.6f,%.6f)\n",
                    stage, moving_foot,
                    problem.stages[stage].x[StateIndex::base_position(0)],
                    problem.stages[stage].x[StateIndex::base_position(1)]);
                return false;
            }
        } else {
            int next_unloaded = -1;
            for (int sequence = 0; sequence < kSwingCount; ++sequence) {
                const int swing = order[sequence];
                if (stage < starts[swing]) {
                    next_unloaded = kSwingFeet[swing];
                    break;
                }
            }
            if (next_unloaded < 0 || stage < starts[order[0]]) {
                four_support_weights(problem.stages[stage].x, weights);
            } else if (!three_support_weights(problem.stages[stage].x,
                                              next_unloaded, weights)) {
                std::printf(
                    "three-support seed failed: stage=%d excluded_foot=%d "
                    "base=(%.6f,%.6f)\n",
                    stage, next_unloaded,
                    problem.stages[stage].x[StateIndex::base_position(0)],
                    problem.stages[stage].x[StateIndex::base_position(1)]);
                return false;
            }
        }
#endif
        for (int foot = 0; foot < kNumFeet; ++foot) {
            for (int axis = 0; axis < 3; ++axis) {
                problem.stages[stage].u[
                    ControlIndex::contact_force(foot, axis)] =
#if QUADRUPED_CITO_EIGHT_STEP
                    net_force[axis] * weights[foot];
#else
                    axis == 2 ? total_force * weights[foot] : 0.0;
#endif
            }
        }
    }
    return true;
}

double dynamics_defect(const Problem<kHorizon>& problem,
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

void audit_solution(const Problem<kHorizon>& problem, SRBDDynamics& dynamics,
                    ContactConstraints<kHorizon, PlannerTerrain>& constraints,
                    const PlannerTerrain& terrain,
                    AuditResult& audit) {
    audit.dynamics = dynamics_defect(problem, dynamics);
    audit.base_displacement =
        problem.stages[kHorizon].x[StateIndex::base_position(0)] -
        problem.x0[StateIndex::base_position(0)];
    for (int moving = 0; moving < kMovingFootCount; ++moving) {
        const int foot = kMovingFeet[moving];
        audit.displacement[moving] =
            problem.stages[kHorizon].x[
                StateIndex::foot_position(foot, 0)] -
            problem.x0[StateIndex::foot_position(foot, 0)];
    }
    bool previous_contact[kMovingFootCount];
    for (int moving = 0; moving < kMovingFootCount; ++moving)
        previous_contact[moving] = true;
    for (int stage = 0; stage < kHorizon; ++stage) {
        Vec<kConstraintCapacity> rows;
        constraints.evaluate(problem.stages[stage].x,
                             problem.stages[stage].u, stage, rows);
        for (int row = 0; row < kUserConstraintRows; ++row) {
            if (row < kContactConstraintRows) {
                if (rows[row] > audit.contact_inequality) {
                    audit.contact_inequality = rows[row];
                    audit.worst_contact_stage = stage;
                    audit.worst_contact_row = row;
                }
            } else {
                audit.support_inequality = std::max(
                    audit.support_inequality, rows[row]);
            }
            if (rows[row] > audit.inequality) {
                audit.inequality = rows[row];
                audit.worst_inequality_stage = stage;
                audit.worst_inequality_row = row;
            }
        }
        for (int pair = 0; pair < kComplementarityPairs; ++pair) {
            int first = -1;
            int second = -1;
            constraints.complementarity_pair(stage, pair, first, second);
            audit.mpcc = std::max(
                audit.mpcc, std::fabs(rows[first] * rows[second]));
        }
        for (int moving = 0; moving < kMovingFootCount; ++moving) {
            const int foot = kMovingFeet[moving];
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
            const bool current_contact = sample.gap <= 2e-3 &&
                dot3(sample.normal, force) >= 5.0;
            if (previous_contact[moving] && !current_contact)
                ++audit.contact_events[moving];
            previous_contact[moving] = current_contact;
            audit.clearance[moving] =
                std::max(audit.clearance[moving], sample.gap);
            if (velocity.norm2() > 0.01) {
                ++audit.moving_stages[moving];
                audit.maximum_normal_force_while_moving[moving] = std::max(
                    audit.maximum_normal_force_while_moving[moving],
                    dot3(sample.normal, force));
                if (audit.first_motion_stage[moving] < 0)
                    audit.first_motion_stage[moving] = stage;
            }
            if (stage >= kHorizon - kTerminalSupportStages) {
                audit.minimum_terminal_normal_force[moving] = std::min(
                    audit.minimum_terminal_normal_force[moving],
                    dot3(sample.normal, force));
                audit.maximum_terminal_gap[moving] = std::max(
                    audit.maximum_terminal_gap[moving], sample.gap);
                audit.maximum_terminal_speed[moving] = std::max(
                    audit.maximum_terminal_speed[moving], velocity.norm2());
            }
        }
    }
}

bool audit_passes(const AuditResult& audit,
                  const nmpc::ContactIPMParams& parameters) {
    if (audit.dynamics > parameters.tol_primal ||
        audit.inequality > parameters.tol_ineq ||
        audit.mpcc > parameters.tol_mpcc) {
        return false;
    }
    if (audit.base_displacement < kMinimumBaseDisplacement) return false;
    for (int moving = 0; moving < kMovingFootCount; ++moving) {
        if (audit.displacement[moving] < 0.06 ||
            audit.clearance[moving] < 0.02 ||
            audit.moving_stages[moving] == 0 ||
            audit.maximum_normal_force_while_moving[moving] >= 5.0 ||
            audit.minimum_terminal_normal_force[moving] < 5.0 ||
            audit.maximum_terminal_gap[moving] > 1e-4 ||
            audit.maximum_terminal_speed[moving] > 0.01) {
            return false;
        }
#if QUADRUPED_CITO_EIGHT_STEP
        if (audit.contact_events[moving] < 2 ||
            audit.clearance[moving] > 0.10) {
            return false;
        }
#endif
    }
    return true;
}

bool run_case(bool reverse_order, const PhysicalTerrain& physical_terrain,
              AuditResult& audit, const char* plan_path,
              const char* fixed_plan_path = nullptr) {
    RobotParameters parameters = go1_robot_parameters();
#if QUADRUPED_CITO_FOUR_STEP || QUADRUPED_CITO_EIGHT_STEP
    parameters.minimum_pair_normal_force = 20.0;
#endif
    const PlannerTerrain terrain(physical_terrain);
    SRBDDynamics dynamics(parameters);
    ContactConstraints<kHorizon, PlannerTerrain> constraints(
        terrain, parameters);
    QuadraticTrackingCost<kHorizon> cost;
    auto problem_storage = std::make_unique<Problem<kHorizon>>();
    Problem<kHorizon>& problem = *problem_storage;
    initialize_standing_problem(problem, dynamics, cost, constraints,
                                kTimeStep);
    Vec<kStateDim> standing = go1_home_state(terrain);
#if QUADRUPED_CITO_FOUR_STEP || QUADRUPED_CITO_EIGHT_STEP
    standing[StateIndex::base_position(0)] = 0.04;
    standing[StateIndex::base_position(1)] = 0.0;
#else
    standing[StateIndex::base_position(0)] = 0.10;
#endif
    problem.x0 = standing;
    cost.set_reference_all(standing);
    for (int stage = 0; stage <= kHorizon; ++stage)
        problem.stages[stage].x = standing;
    configure_cost(cost);
#if QUADRUPED_CITO_EIGHT_STEP
    cost.references[kHorizon][StateIndex::base_position(0)] +=
        2.0 * kRequestedStep;
#elif QUADRUPED_CITO_FOUR_STEP
    cost.references[kHorizon][StateIndex::base_position(0)] +=
        kRequestedStep;
#endif
    for (int moving = 0; moving < kMovingFootCount; ++moving) {
        const int foot = kMovingFeet[moving];
        cost.references[kHorizon][StateIndex::foot_position(foot, 2)] =
            terrain.height(
                standing[StateIndex::foot_position(foot, 0)] +
                    requested_displacement(foot),
                standing[StateIndex::foot_position(foot, 1)]);
    }
    if (!initialize_guess(problem, parameters, terrain, reverse_order)) {
        std::printf("multifoot seed construction failed: reverse=%d\n",
                    reverse_order ? 1 : 0);
        return false;
    }

    AuditResult seed_audit;
    audit_solution(problem, dynamics, constraints, terrain, seed_audit);
    if (seed_audit.dynamics > 1e-10 ||
        seed_audit.contact_inequality > 1e-10 ||
        seed_audit.mpcc > 1e-10) {
        std::printf(
            "multifoot seed infeasible: reverse=%d, dynamics=%.3e, "
            "contact_inequality=%.3e (stage=%d,row=%d), "
            "support_inequality=%.3e "
            "(stage=%d,row=%d), mpcc=%.3e\n",
            reverse_order ? 1 : 0, seed_audit.dynamics,
            seed_audit.contact_inequality, seed_audit.worst_contact_stage,
            seed_audit.worst_contact_row, seed_audit.support_inequality,
            seed_audit.worst_inequality_stage,
            seed_audit.worst_inequality_row, seed_audit.mpcc);
        return false;
    }
    if (fixed_plan_path) {
        ContactPlan<kHorizon> fixed_plan;
        ContactClassification classification;
        if (extract_contact_plan(problem, terrain, classification,
                                 fixed_plan) != Status::SUCCESS ||
            write_contact_plan(fixed_plan, fixed_plan_path) !=
                Status::SUCCESS) {
            std::printf("failed to export fixed-schedule plan: %s\n",
                        fixed_plan_path);
            return false;
        }
    }

    nmpc::ContactIPMParams solver_parameters;
    solver_parameters.mu_init = 0.1;
    solver_parameters.mu_min = 1e-5;
    solver_parameters.mu_conv_threshold = 1e-5;
    solver_parameters.max_same_mu = 100;
    solver_parameters.max_iters = 700;
    solver_parameters.mpcc_recovery_max_iters = 700;
    solver_parameters.s_min_init = 0.1;
#if QUADRUPED_CITO_EIGHT_STEP
    solver_parameters.tol_primal = 1e-4;
#else
    solver_parameters.tol_primal = 2e-5;
#endif
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
    const auto start = std::chrono::steady_clock::now();
    const nmpc::Status status = solver.solve(problem);
    const auto end = std::chrono::steady_clock::now();
    const double solve_ms =
        std::chrono::duration<double, std::milli>(end - start).count();
    audit_solution(problem, dynamics, constraints, terrain, audit);
    if (status != nmpc::Status::SUCCESS) {
        const auto& stats = solver.last_stats();
        std::printf(
            "multifoot solver stopped: reverse=%d, status=%s, iterations=%d, "
            "primal=%.3e, stationarity=%.3e, mpcc=%.3e, "
            "audit_dynamics=%.3e, audit_inequality=%.3e, "
            "audit_mpcc=%.3e\n",
            reverse_order ? 1 : 0, nmpc::status_string(status),
            stats.inner_iterations, stats.primal_infeas, stats.dual_infeas,
            stats.mpcc_complementarity, audit.dynamics, audit.inequality,
            audit.mpcc);
        return false;
    }
    if (!audit_passes(audit, solver_parameters)) {
        std::printf(
            "multifoot audit failed: reverse=%d, dynamics=%.3e, "
            "inequality=%.3e, mpcc=%.3e, base_displacement=%.3f\n",
            reverse_order ? 1 : 0, audit.dynamics, audit.inequality,
            audit.mpcc, audit.base_displacement);
        for (int moving = 0; moving < kMovingFootCount; ++moving) {
            std::printf(
                "  foot=%d displacement=%.3f clearance=%.3f moving=%d "
                "contact_events=%d "
                "moving_force=%.3f terminal_force=%.3f "
                "terminal_gap=%.3e terminal_speed=%.3e\n",
                kMovingFeet[moving], audit.displacement[moving],
                audit.clearance[moving], audit.moving_stages[moving],
                audit.contact_events[moving],
                audit.maximum_normal_force_while_moving[moving],
                audit.minimum_terminal_normal_force[moving],
                audit.maximum_terminal_gap[moving],
                audit.maximum_terminal_speed[moving]);
        }
        return false;
    }
    if (plan_path) {
        ContactPlan<kHorizon> plan;
        ContactClassification classification;
        if (extract_contact_plan(problem, terrain, classification, plan) !=
                Status::SUCCESS ||
            write_contact_plan(plan, plan_path) != Status::SUCCESS) {
            std::printf("failed to export multifoot plan: %s\n", plan_path);
            return false;
        }
    }
    std::printf(
        "Quadruped CITO multifoot passed: terrain=%s, reverse=%d, "
        "iterations=%d, solve_ms=%.3f, base_displacement=%.3f, "
        "dynamics=%.3e, inequality=%.3e, mpcc=%.3e\n",
        terrain_name(physical_terrain.kind), reverse_order ? 1 : 0,
        solver.last_stats().inner_iterations, solve_ms,
        audit.base_displacement, audit.dynamics, audit.inequality,
        audit.mpcc);
    for (int moving = 0; moving < kMovingFootCount; ++moving) {
        std::printf(
            "  foot=%d first_motion=%d contact_events=%d "
            "terminal_force=%.3f\n",
            kMovingFeet[moving], audit.first_motion_stage[moving],
            audit.contact_events[moving],
            audit.minimum_terminal_normal_force[moving]);
    }
    return true;
}

}  // namespace

int main(int argc, char** argv) {
#if QUADRUPED_CITO_LONG_HORIZON
    PhysicalTerrain terrain = register_go1_terrain(SharedTerrain::flat());
    const char* plan_path = nullptr;
    const char* fixed_plan_path = nullptr;
    bool reverse_order = false;
    for (int argument = 1; argument < argc; ++argument) {
        const std::string option = argv[argument];
        if (option == "--write-plan" && argument + 1 < argc && !plan_path) {
            plan_path = argv[++argument];
        } else if (option == "--write-fixed-plan" &&
                   argument + 1 < argc && !fixed_plan_path) {
            fixed_plan_path = argv[++argument];
        } else if (option == "--reverse" && !reverse_order) {
            reverse_order = true;
        } else if (option == "--terrain" && argument + 1 < argc) {
            const std::string name = argv[++argument];
            if (name == "flat") {
                terrain = register_go1_terrain(SharedTerrain::flat());
            } else if (name == "smooth") {
                terrain = register_go1_terrain(SharedTerrain::sinusoidal());
            } else if (name == "slope") {
                terrain = register_go1_terrain(SharedTerrain::slope(0.10));
            } else {
                std::printf("unknown terrain: %s\n", name.c_str());
                return 2;
            }
        } else {
            std::printf("usage: test_quadruped_cito_long_horizon "
                        "[--reverse] "
                        "[--terrain flat|smooth|slope] "
                        "[--write-plan <path>] "
                        "[--write-fixed-plan <path>]\n");
            return 2;
        }
    }
    if (argc > 8) {
        std::printf("usage: test_quadruped_cito_long_horizon "
                    "[--reverse] "
                    "[--terrain flat|smooth|slope] "
                    "[--write-plan <path>] "
                    "[--write-fixed-plan <path>]\n");
        return 2;
    }
    AuditResult audit;
    return run_case(reverse_order, terrain, audit, plan_path,
                    fixed_plan_path) ? 0 : 1;
#else
    const bool write_plans = argc == 5 &&
        std::string(argv[1]) == "--write-forward" &&
        std::string(argv[3]) == "--write-reverse";
    if (argc != 1 && !write_plans) {
        std::printf("usage: test_quadruped_cito_multifoot "
                    "[--write-forward <path> --write-reverse <path>]\n");
        return 2;
    }
    const PhysicalTerrain terrain = register_go1_terrain(SharedTerrain::flat());
    AuditResult forward;
    AuditResult reverse;
    if (!run_case(false, terrain, forward, write_plans ? argv[2] : nullptr) ||
        !run_case(true, terrain, reverse,
                  write_plans ? argv[4] : nullptr)) {
        return 1;
    }
    if (!(forward.first_motion_stage[0] < forward.first_motion_stage[1]) ||
        !(reverse.first_motion_stage[1] < reverse.first_motion_stage[0])) {
        std::printf(
            "multifoot warm-start sensitivity was not preserved: "
            "forward=(%d,%d), reverse=(%d,%d)\n",
            forward.first_motion_stage[0], forward.first_motion_stage[1],
            reverse.first_motion_stage[0], reverse.first_motion_stage[1]);
        return 1;
    }
    return 0;
#endif
}
