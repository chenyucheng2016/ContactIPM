#pragma once

#include <algorithm>
#include <chrono>
#include <cmath>
#include <cstdint>
#include <cstring>
#include <limits>
#include <memory>

#include "examples/quadruped_cito/quadruped_cito_go1.hpp"
#include "examples/quadruped_cito/quadruped_cito_plan.hpp"
#include "nmpc/contact_ipm.hpp"

namespace quadruped_cito {

constexpr int kRealtimeHorizon = 50;
constexpr double kRealtimeTimeStep = 0.05;
constexpr int kRealtimeMovingFoot = 0;
constexpr double kRealtimeRequestedStep = 0.08;
constexpr double kRealtimeSwingClearance = 0.055;
constexpr double kRealtimeMinimumClearance = 0.02;
constexpr double kRealtimePublicationDualTolerance = 0.2;
constexpr double kRealtimeRiccatiRelativeRegularization = 1e-8;
constexpr int kRealtimeSwingFirstStage = 3;
constexpr int kRealtimeSwingLastStage = 39;
constexpr int kRealtimeSustainedTaskSpacing = 30;
constexpr int kRealtimeSustainedSwingStages = 20;
constexpr int kRealtimeSustainedFootOrder[kNumFeet] = {2, 3, 0, 1};
constexpr double kRealtimeHardSupportMinimumNormalForce = 10.0;

static_assert(kStateDim == 25,
              "real-time planner must use the full SRBD state");
static_assert(kControlDim == 28,
              "real-time planner must use all contact controls");
static_assert(kConstraintCapacity == 86,
              "real-time planner must use all contact constraints");

struct RealtimePlannerConfig {
    int rate_hz = 5;
    bool sustained_contact_tasks = false;
    double publication_headroom_ms = 0.0;
    // Zero preserves the historical unbounded deterministic task sequence.
    int maximum_contact_tasks = 0;
};

struct RealtimeContactTask {
    std::uint64_t id = 0;
    int moving_foot = kRealtimeMovingFoot;
    int start_knot = kRealtimeSwingFirstStage;
    int touchdown_knot = kRealtimeSwingLastStage;
    double origin_x = 0.0;
    double target_x = 0.0;
};

struct RealtimePlannerAudit {
    double dynamics = 0.0;
    double inequality = 0.0;
    double mpcc = 0.0;
    // In sustained mode these terminal_* fields describe the current task's
    // settled checkpoint, not a later horizon endpoint that may contain the
    // next foot's swing.
    double terminal_foot_error = 0.0;
    double terminal_gap = 0.0;
    double terminal_speed = 0.0;
    double terminal_normal_force = 0.0;
    double clearance = 0.0;
    double task_displacement = 0.0;
    int moving_unloaded_stages = 0;
};

struct RealtimePlannerResult {
    nmpc::Status status = nmpc::Status::NOT_INITIALIZED;
    nmpc::SolverStats stats;
    RealtimePlannerAudit audit;
    double shift_ms = 0.0;
    double solver_ms = 0.0;
    double audit_ms = 0.0;
    double plan_extract_ms = 0.0;
    double end_to_end_ms = 0.0;
    std::uint64_t solver_terrain_samples = 0;
    std::uint64_t audit_terrain_samples = 0;
    std::uint64_t plan_extract_terrain_samples = 0;
    double solver_terrain_ms = 0.0;
    double audit_terrain_ms = 0.0;
    double plan_extract_terrain_ms = 0.0;
    bool full_kkt = false;
    bool task_pass = false;
    bool deadline_miss = false;
    bool published = false;
    bool fallback = false;
    bool cold_fallback_used = false;
    bool shift_consumed = false;
    std::uint64_t task_id = 0;
    int moving_foot = -1;
    bool task_transition_witness = false;
};

namespace realtime_detail {

inline bool finite_value(double value) {
    static_assert(sizeof(double) == sizeof(std::uint64_t),
                  "finite-value check requires a 64-bit double");
    static_assert(std::numeric_limits<double>::is_iec559,
                  "finite-value check requires IEEE-754 doubles");
    std::uint64_t bits = 0;
    std::memcpy(&bits, &value, sizeof(bits));
    return (bits & UINT64_C(0x7ff0000000000000)) !=
           UINT64_C(0x7ff0000000000000);
}

class PlannerTerrain {
public:
    static constexpr bool kAffineFrame = SharedTerrain::kAffineFrame;

    explicit PlannerTerrain(const SharedTerrain& terrain)
        : terrain_(terrain) {}

    bool sample(const Vec<3>& position, TerrainSample& result) const {
        const auto start = Clock::now();
        ++sample_calls_;
        const bool valid = terrain_.sample(position, result);
        sample_time_ms_ += std::chrono::duration<double, std::milli>(
            Clock::now() - start).count();
        return valid;
    }

    double height(double x, double y) const {
        Vec<3> position;
        position[0] = x;
        position[1] = y;
        position[2] = 0.0;
        TerrainSample sample_result;
        if (!sample(position, sample_result) ||
            std::fabs(sample_result.gap_gradient[2]) <= 1e-12) {
            return 0.0;
        }
        return -sample_result.gap / sample_result.gap_gradient[2];
    }

    void reset_sample_calls() const {
        sample_calls_ = 0;
        sample_time_ms_ = 0.0;
    }
    std::uint64_t sample_calls() const { return sample_calls_; }
    double sample_time_ms() const { return sample_time_ms_; }

private:
    Go1FootCenterTerrain<SharedTerrain> terrain_;
    using Clock = std::chrono::steady_clock;
    mutable std::uint64_t sample_calls_ = 0;
    mutable double sample_time_ms_ = 0.0;
};

class RealtimeContactTaskSequence {
public:
    explicit RealtimeContactTaskSequence(int maximum_contact_tasks = 0)
        : maximum_contact_tasks_(maximum_contact_tasks) {}

    void initialize(const Vec<kStateDim>& initial_state) {
        initial_state_ = initial_state;
        absolute_knot_ = 0;
        initialized_ = true;
    }

    bool advance(int knots) {
        if (!initialized_ || knots <= 0) return false;
        absolute_knot_ += knots;
        return true;
    }

    int absolute_knot() const { return absolute_knot_; }
    int maximum_contact_tasks() const { return maximum_contact_tasks_; }

    RealtimeContactTask active_task() const {
        std::uint64_t id = static_cast<std::uint64_t>(
            absolute_knot_ / kRealtimeSustainedTaskSpacing);
        if (maximum_contact_tasks_ > 0) {
            id = std::min(
                id, static_cast<std::uint64_t>(maximum_contact_tasks_ - 1));
        }
        return task(id);
    }

    RealtimeContactTask task(std::uint64_t id) const {
        RealtimeContactTask result;
        result.id = id;
        result.moving_foot = kRealtimeSustainedFootOrder[
            static_cast<int>(id % kNumFeet)];
        result.start_knot = kRealtimeSwingFirstStage +
            static_cast<int>(id) * kRealtimeSustainedTaskSpacing;
        result.touchdown_knot = result.start_knot +
            kRealtimeSustainedSwingStages;
        const int cycle = static_cast<int>(id / kNumFeet);
        result.origin_x = initial_state_[StateIndex::foot_position(
            result.moving_foot, 0)] + cycle * kRealtimeRequestedStep;
        result.target_x = result.origin_x + kRealtimeRequestedStep;
        return result;
    }

    int moving_foot_at(int absolute_knot) const {
        if (absolute_knot < kRealtimeSwingFirstStage) return -1;
        const int id = (absolute_knot - kRealtimeSwingFirstStage) /
            kRealtimeSustainedTaskSpacing;
        if (maximum_contact_tasks_ > 0 && id >= maximum_contact_tasks_)
            return -1;
        const RealtimeContactTask candidate = task(
            static_cast<std::uint64_t>(id));
        return absolute_knot < candidate.touchdown_knot
            ? candidate.moving_foot
            : -1;
    }

    int designated_support_foot_at(int absolute_knot) const {
        const int moving_foot = moving_foot_at(absolute_knot);
        if (moving_foot >= 0) return moving_foot;
        int task_index = absolute_knot / kRealtimeSustainedTaskSpacing;
        if (maximum_contact_tasks_ > 0) {
            task_index = std::min(task_index, maximum_contact_tasks_ - 1);
        }
        return task(static_cast<std::uint64_t>(task_index)).moving_foot;
    }

    int completed_task_count(int absolute_knot) const {
        const int first_touchdown = kRealtimeSwingFirstStage +
            kRealtimeSustainedSwingStages;
        if (absolute_knot < first_touchdown) return 0;
        const int completed = 1 + (absolute_knot - first_touchdown) /
            kRealtimeSustainedTaskSpacing;
        return maximum_contact_tasks_ > 0
            ? std::min(completed, maximum_contact_tasks_)
            : completed;
    }

    void foot_position(int absolute_knot, int foot,
                       const PlannerTerrain& terrain,
                       Vec<3>& position) const {
        position = state_vector3(
            initial_state_, StateIndex::foot_position(foot, 0));
        const int latest_id = absolute_knot < kRealtimeSwingFirstStage
            ? -1
            : (absolute_knot - kRealtimeSwingFirstStage) /
                  kRealtimeSustainedTaskSpacing;
        const int final_id = maximum_contact_tasks_ > 0
            ? std::min(latest_id, maximum_contact_tasks_ - 1)
            : latest_id;
        for (int id = 0; id <= final_id; ++id) {
            const RealtimeContactTask candidate = task(
                static_cast<std::uint64_t>(id));
            if (candidate.moving_foot != foot) continue;
            double phase = 1.0;
            if (absolute_knot < candidate.touchdown_knot) {
                phase = static_cast<double>(absolute_knot -
                                             candidate.start_knot) /
                    kRealtimeSustainedSwingStages;
                phase = std::max(0.0, std::min(1.0, phase));
            }
            const double smooth_phase =
                phase * phase * (3.0 - 2.0 * phase);
            position[0] = candidate.origin_x +
                kRealtimeRequestedStep * smooth_phase;
            position[1] = initial_state_[StateIndex::foot_position(foot, 1)];
            position[2] = terrain.height(position[0], position[1]);
            if (phase < 1.0) {
                Vec<3> surface = position;
                TerrainSample sample;
                if (terrain.sample(surface, sample)) {
                    const double clearance = kRealtimeSwingClearance *
                        std::sin(3.14159265358979323846 * phase);
                    for (int axis = 0; axis < 3; ++axis)
                        position[axis] += clearance * sample.normal[axis];
                }
            }
        }
    }

    void reference_state(int absolute_knot,
                         const PlannerTerrain& terrain,
                         Vec<kStateDim>& reference) const {
        reference = initial_state_;
        double initial_mean_height = 0.0;
        double reference_mean_height = 0.0;
        for (int foot = 0; foot < kNumFeet; ++foot) {
            initial_mean_height += initial_state_[
                StateIndex::foot_position(foot, 2)] / kNumFeet;
            Vec<3> position;
            foot_position(absolute_knot, foot, terrain, position);
            reference_mean_height += position[2] / kNumFeet;
            for (int axis = 0; axis < 3; ++axis) {
                reference[StateIndex::foot_position(foot, axis)] =
                    position[axis];
            }
        }
        reference[StateIndex::base_position(0)] +=
            completed_task_count(absolute_knot) *
            kRealtimeRequestedStep / kNumFeet;
        reference[StateIndex::base_position(2)] +=
            reference_mean_height - initial_mean_height;
    }

private:
    Vec<kStateDim> initial_state_;
    int absolute_knot_ = 0;
    int maximum_contact_tasks_ = 0;
    bool initialized_ = false;
};

using RealtimeProblem = Problem<kRealtimeHorizon>;
inline constexpr double kInitialTerrainNormalForceBlend = 0.25;
inline constexpr double kFallbackTerrainNormalForceBlend = 0.50;

inline bool cold_start_retryable(nmpc::Status status) {
    return status == nmpc::Status::MAX_ITERATIONS ||
           status == nmpc::Status::STAGNATION ||
           status == nmpc::Status::LINE_SEARCH_FAILURE ||
           status == nmpc::Status::KKT_SINGULAR;
}

inline nmpc::Status blend_initial_contact_forces_toward_terrain_normals(
    RealtimeProblem& problem, const PlannerTerrain& terrain,
    double blend = kInitialTerrainNormalForceBlend) {
    if (blend < 0.0 || blend > 1.0) return nmpc::Status::BAD_ARGUMENT;
    for (int stage = 0; stage < kRealtimeHorizon; ++stage) {
        for (int foot = 0; foot < kNumFeet; ++foot) {
            const Vec<3> position = state_vector3(
                problem.stages[stage].x,
                StateIndex::foot_position(foot, 0));
            TerrainSample sample;
            if (!terrain.sample(position, sample))
                return nmpc::Status::BAD_ARGUMENT;
            const Vec<3> force = control_vector3(
                problem.stages[stage].u,
                ControlIndex::contact_force(foot, 0));
            const double normal_force = dot3(sample.normal, force);
            for (int axis = 0; axis < 3; ++axis) {
                const double projected = normal_force * sample.normal[axis];
                problem.stages[stage].u[
                    ControlIndex::contact_force(foot, axis)] =
                    force[axis] + blend *
                        (projected - force[axis]);
            }
        }
    }
    return nmpc::Status::SUCCESS;
}

inline nmpc::ContactIPMParams solver_parameters() {
    nmpc::ContactIPMParams parameters;
    parameters.mu_init = 0.1;
    parameters.mu_min = 1e-5;
    parameters.mu_conv_threshold = 1e-5;
    parameters.max_same_mu = 100;
    parameters.max_iters = 200;
    parameters.s_min_init = 0.1;
    parameters.tol_primal = 2e-5;
    parameters.tol_compl = 2e-5;
    parameters.tol_ineq = 1e-6;
    // The solver's mu-floor acceptance permits 10 * tol_stat. Keep the
    // internal tolerance one decade below the independent publication gate.
    parameters.tol_stat = 0.02;
    parameters.tol_mpcc = 1e-4;
    parameters.exact_hessian = false;
    parameters.adaptive_exact_hessian = false;
    parameters.enable_nonlinear_rollout = false;
    parameters.enable_preconditioner = true;
    parameters.riccati_relative_regularization =
        kRealtimeRiccatiRelativeRegularization;
    parameters.inertia_min_pivot = 0.0;
    parameters.verbosity = 0;
    parameters.enable_runtime_diagnostics = false;
    return parameters;
}

inline void configure_cost(QuadraticTrackingCost<kRealtimeHorizon>& cost,
                           bool sustained_contact_tasks = false) {
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
                0.005;
            if (sustained_contact_tasks) {
                cost.state_weights[
                    StateIndex::foot_position(foot, axis)] = 1.0;
            }
        }
        cost.control_weights[ControlIndex::motion_slack(foot)] = 0.005;
    }
    for (int foot = 0; foot < kNumFeet; ++foot) {
        if (!sustained_contact_tasks && foot != kRealtimeMovingFoot) continue;
        for (int axis = 0; axis < 3; ++axis) {
            cost.terminal_weights[
                StateIndex::foot_position(foot, axis)] = 1000.0;
        }
    }
}

inline nmpc::Status initialize_transition_guess(
    RealtimeProblem& problem, const RobotParameters& parameters,
    const PlannerTerrain& terrain,
    double force_blend = kInitialTerrainNormalForceBlend) {
    double foot_path[kRealtimeHorizon + 1][3];
    const double initial_x = problem.x0[
        StateIndex::foot_position(kRealtimeMovingFoot, 0)];
    const double foot_y = problem.x0[
        StateIndex::foot_position(kRealtimeMovingFoot, 1)];
    for (int stage = 0; stage <= kRealtimeHorizon; ++stage) {
        const double raw_phase =
            static_cast<double>(stage - kRealtimeSwingFirstStage) /
            static_cast<double>(kRealtimeSwingLastStage -
                                kRealtimeSwingFirstStage);
        const double phase = std::max(0.0, std::min(1.0, raw_phase));
        const double smooth_phase = phase * phase * (3.0 - 2.0 * phase);
        Vec<3> surface;
        surface[0] = initial_x + kRealtimeRequestedStep * smooth_phase;
        surface[1] = foot_y;
        surface[2] = terrain.height(surface[0], surface[1]);
        TerrainSample sample;
        terrain.sample(surface, sample);
        const double clearance = kRealtimeSwingClearance *
            std::sin(3.14159265358979323846 * phase);
        for (int axis = 0; axis < 3; ++axis) {
            foot_path[stage][axis] =
                surface[axis] + clearance * sample.normal[axis];
        }
    }

    const Vec<kStateDim> standing = problem.x0;
    for (int stage = 0; stage <= kRealtimeHorizon; ++stage) {
        problem.stages[stage].x = standing;
        problem.stages[stage].u.zero();
        for (int axis = 0; axis < 3; ++axis) {
            problem.stages[stage].x[
                StateIndex::foot_position(kRealtimeMovingFoot, axis)] =
                foot_path[stage][axis];
        }
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
    const double positive_x = (base_x - rear_x) / (front_x - rear_x);
    const double positive_y = (base_y - right_y) / (left_y - right_y);
    const double four_support[kNumFeet] = {
        positive_x * positive_y,
        positive_x * (1.0 - positive_y),
        (1.0 - positive_x) * positive_y,
        (1.0 - positive_x) * (1.0 - positive_y)};
    const double three_support[kNumFeet] = {
        0.0, positive_x, positive_y, 1.0 - positive_x - positive_y};
    const double landing_foot_weight = 0.10;
    const double landing_support1 = positive_x -
        (foot_path[kRealtimeHorizon][0] - rear_x) * landing_foot_weight /
            (front_x - rear_x);
    const double landing_support2 = positive_y - landing_foot_weight;
    const double landing[kNumFeet] = {
        landing_foot_weight, landing_support1, landing_support2,
        1.0 - landing_foot_weight - landing_support1 - landing_support2};

    for (int stage = 0; stage < kRealtimeHorizon; ++stage) {
        const double* support = stage < kRealtimeSwingFirstStage
            ? four_support
            : (stage >= kRealtimeSwingLastStage ? landing : three_support);
        for (int foot = 0; foot < kNumFeet; ++foot) {
            problem.stages[stage].u[
                ControlIndex::contact_force(foot, 2)] =
                total_force * support[foot];
        }
        double speed_sq = 0.0;
        for (int axis = 0; axis < 3; ++axis) {
            const double velocity =
                (foot_path[stage + 1][axis] - foot_path[stage][axis]) /
                kRealtimeTimeStep;
            problem.stages[stage].u[
                ControlIndex::foot_velocity(kRealtimeMovingFoot, axis)] =
                velocity;
            speed_sq += velocity * velocity;
        }
        problem.stages[stage].u[
            ControlIndex::motion_slack(kRealtimeMovingFoot)] =
            std::sqrt(speed_sq);
    }
    return blend_initial_contact_forces_toward_terrain_normals(
        problem, terrain, force_blend);
}

inline bool normalize_support_weights(double weights[kNumFeet]) {
    double sum = 0.0;
    for (int foot = 0; foot < kNumFeet; ++foot) {
        if (!finite_value(weights[foot])) return false;
        weights[foot] = std::max(0.0, weights[foot]);
        sum += weights[foot];
    }
    if (sum <= 1e-12) return false;
    for (int foot = 0; foot < kNumFeet; ++foot) weights[foot] /= sum;
    return true;
}

inline bool equal_allowed_support_weights(int excluded_foot,
                                          double weights[kNumFeet]) {
    if (excluded_foot < -1 || excluded_foot >= kNumFeet) return false;
    const int allowed_count = kNumFeet - (excluded_foot >= 0 ? 1 : 0);
    if (allowed_count <= 0) return false;
    for (int foot = 0; foot < kNumFeet; ++foot) {
        weights[foot] = foot == excluded_foot
            ? 0.0
            : 1.0 / static_cast<double>(allowed_count);
    }
    return true;
}

inline bool sustained_support_weights(const Vec<kStateDim>& state,
                                      int excluded_foot,
                                      double weights[kNumFeet]) {
    if (excluded_foot < -1 || excluded_foot >= kNumFeet) return false;
    if (excluded_foot < 0) {
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
        if (std::fabs(front_x - rear_x) <= 1e-12 ||
            std::fabs(left_y - right_y) <= 1e-12) {
            return equal_allowed_support_weights(excluded_foot, weights);
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
        return normalize_support_weights(weights) ||
               equal_allowed_support_weights(excluded_foot, weights);
    }

    int active[3];
    int active_count = 0;
    for (int foot = 0; foot < kNumFeet; ++foot) {
        weights[foot] = 0.0;
        if (foot != excluded_foot) active[active_count++] = foot;
    }
    const double xa = state[StateIndex::foot_position(active[0], 0)];
    const double ya = state[StateIndex::foot_position(active[0], 1)];
    const double xb = state[StateIndex::foot_position(active[1], 0)];
    const double yb = state[StateIndex::foot_position(active[1], 1)];
    const double xc = state[StateIndex::foot_position(active[2], 0)];
    const double yc = state[StateIndex::foot_position(active[2], 1)];
    const double denominator =
        (yb - yc) * (xa - xc) + (xc - xb) * (ya - yc);
    if (std::fabs(denominator) <= 1e-12)
        return equal_allowed_support_weights(excluded_foot, weights);
    const double base_x = state[StateIndex::base_position(0)];
    const double base_y = state[StateIndex::base_position(1)];
    weights[active[0]] =
        ((yb - yc) * (base_x - xc) +
         (xc - xb) * (base_y - yc)) / denominator;
    weights[active[1]] =
        ((yc - ya) * (base_x - xc) +
         (xa - xc) * (base_y - yc)) / denominator;
    weights[active[2]] = 1.0 - weights[active[0]] - weights[active[1]];
    return normalize_support_weights(weights) ||
           equal_allowed_support_weights(excluded_foot, weights);
}

inline void configure_sustained_references(
    QuadraticTrackingCost<kRealtimeHorizon>& cost,
    const RealtimeContactTaskSequence& sequence,
    const PlannerTerrain& terrain) {
    for (int stage = 0; stage < kRealtimeHorizon; ++stage) {
        sequence.reference_state(sequence.absolute_knot() + stage,
                                 terrain, cost.references[stage]);
    }
    const int terminal_knot = sequence.absolute_knot() + kRealtimeHorizon;
    sequence.reference_state(terminal_knot, terrain,
                             cost.references[kRealtimeHorizon]);
}

inline nmpc::Status seed_sustained_range(
    RealtimeProblem& problem, const RobotParameters& parameters,
    const PlannerTerrain& terrain,
    const RealtimeContactTaskSequence& sequence, int first_stage) {
    if (first_stage < 0 || first_stage > kRealtimeHorizon ||
        problem.dynamics == nullptr)
        return nmpc::Status::BAD_ARGUMENT;
    const double total_force = parameters.mass * parameters.gravity;
    if (!finite_value(total_force) || !(total_force > 0.0) ||
        !finite_value(parameters.friction) || parameters.friction < 0.0) {
        return nmpc::Status::BAD_ARGUMENT;
    }
    for (int stage = first_stage; stage < kRealtimeHorizon; ++stage) {
        problem.stages[stage].u.zero();
        for (int foot = 0; foot < kNumFeet; ++foot) {
            Vec<3> position;
            Vec<3> next_position;
            sequence.foot_position(sequence.absolute_knot() + stage,
                                   foot, terrain, position);
            sequence.foot_position(sequence.absolute_knot() + stage + 1,
                                   foot, terrain, next_position);
            double speed_sq = 0.0;
            for (int axis = 0; axis < 3; ++axis) {
                const double velocity =
                    (next_position[axis] - position[axis]) / problem.dt;
                problem.stages[stage].u[
                    ControlIndex::foot_velocity(foot, axis)] = velocity;
                speed_sq += velocity * velocity;
            }
            problem.stages[stage].u[ControlIndex::motion_slack(foot)] =
                std::sqrt(speed_sq);
        }

        double support[kNumFeet];
        const int absolute_knot = sequence.absolute_knot() + stage;
        const int moving_foot = sequence.moving_foot_at(absolute_knot);
        if (!sustained_support_weights(problem.stages[stage].x,
                                       moving_foot, support)) {
            return nmpc::Status::BAD_ARGUMENT;
        }
        for (int foot = 0; foot < kNumFeet; ++foot) {
            const Vec<3> position = state_vector3(
                problem.stages[stage].x,
                StateIndex::foot_position(foot, 0));
            TerrainSample sample;
            if (!terrain.sample(position, sample))
                return nmpc::Status::BAD_ARGUMENT;
            const double vertical_force = total_force * support[foot];
            Vec<3> force;
            force.zero();
            force[2] = vertical_force;
            const double terrain_normal_force = dot3(sample.normal, force);
            const double tangent_force1 = dot3(sample.tangent1, force);
            const double tangent_force2 = dot3(sample.tangent2, force);
            const double friction_limit =
                parameters.friction * terrain_normal_force;
            if (!finite_value(terrain_normal_force) ||
                !finite_value(tangent_force1) ||
                !finite_value(tangent_force2) ||
                !finite_value(friction_limit) ||
                terrain_normal_force < 0.0 ||
                std::fabs(tangent_force1) > friction_limit ||
                std::fabs(tangent_force2) > friction_limit) {
                return nmpc::Status::BAD_ARGUMENT;
            }
            for (int axis = 0; axis < 3; ++axis) {
                problem.stages[stage].u[
                    ControlIndex::contact_force(foot, axis)] = force[axis];
            }
        }

        Vec<kStateDim> next_state;
        const nmpc::Status rollout_status = problem.dynamics->discrete_step(
            problem.stages[stage].x, problem.stages[stage].u, problem.dt,
            next_state, stage);
        if (rollout_status != nmpc::Status::SUCCESS) return rollout_status;
        problem.stages[stage + 1].x = next_state;
    }
    return nmpc::Status::SUCCESS;
}

inline void shift_references(QuadraticTrackingCost<kRealtimeHorizon>& cost,
                             int shift_steps) {
    const Vec<kStateDim> terminal = cost.references[kRealtimeHorizon];
    for (int stage = 0; stage <= kRealtimeHorizon - shift_steps; ++stage)
        cost.references[stage] = cost.references[stage + shift_steps];
    for (int stage = kRealtimeHorizon - shift_steps + 1;
         stage <= kRealtimeHorizon; ++stage) {
        cost.references[stage] = terminal;
    }
}

inline nmpc::Status apply_measured_state_defect(
    const Vec<kStateDim>& actual, const Vec<kStateDim>& nominal,
    Vec<kStateDim>& state) {
    for (int index = 0; index < kStateDim; ++index) {
        if (index < StateIndex::quaternion(0) ||
            index > StateIndex::quaternion(3)) {
            state[index] += actual[index] - nominal[index];
        }
    }

    Vec<4> actual_quaternion;
    Vec<4> nominal_quaternion;
    Vec<4> state_quaternion_value;
    if (!normalize_quaternion(state_quaternion(actual), actual_quaternion) ||
        !normalize_quaternion(state_quaternion(nominal), nominal_quaternion) ||
        !normalize_quaternion(state_quaternion(state), state_quaternion_value)) {
        return nmpc::Status::BAD_ARGUMENT;
    }

    double actual_nominal_dot = 0.0;
    for (int element = 0; element < 4; ++element)
        actual_nominal_dot += actual_quaternion[element] *
                              nominal_quaternion[element];
    if (actual_nominal_dot < 0.0) {
        for (int element = 0; element < 4; ++element)
            actual_quaternion[element] = -actual_quaternion[element];
    }

    const auto multiply = [](const Vec<4>& first, const Vec<4>& second) {
        Vec<4> product;
        product[0] = first[0] * second[0] - first[1] * second[1] -
                     first[2] * second[2] - first[3] * second[3];
        product[1] = first[0] * second[1] + first[1] * second[0] +
                     first[2] * second[3] - first[3] * second[2];
        product[2] = first[0] * second[2] - first[1] * second[3] +
                     first[2] * second[0] + first[3] * second[1];
        product[3] = first[0] * second[3] + first[1] * second[2] -
                     first[2] * second[1] + first[3] * second[0];
        return product;
    };

    Vec<4> nominal_conjugate = nominal_quaternion;
    for (int element = 1; element < 4; ++element)
        nominal_conjugate[element] = -nominal_conjugate[element];
    const Vec<4> orientation_defect = multiply(
        actual_quaternion, nominal_conjugate);
    const Vec<4> corrected_raw = multiply(
        orientation_defect, state_quaternion_value);
    Vec<4> corrected;
    if (!normalize_quaternion(corrected_raw, corrected))
        return nmpc::Status::BAD_ARGUMENT;
    for (int element = 0; element < 4; ++element)
        state[StateIndex::quaternion(element)] = corrected[element];
    return nmpc::Status::SUCCESS;
}

inline RealtimePlannerAudit audit_solution(
    const RealtimeProblem& problem, SRBDDynamics& dynamics,
    ContactConstraints<kRealtimeHorizon, PlannerTerrain>& constraints,
    const PlannerTerrain& terrain,
    const QuadraticTrackingCost<kRealtimeHorizon>& cost,
    double initial_moving_foot_x,
    int moving_foot = kRealtimeMovingFoot,
    int task_checkpoint_stage = kRealtimeHorizon) {
    RealtimePlannerAudit audit;
    task_checkpoint_stage = std::max(
        0, std::min(kRealtimeHorizon, task_checkpoint_stage));
    for (int stage = 0; stage < kRealtimeHorizon; ++stage) {
        Vec<kStateDim> predicted;
        dynamics.discrete_step(problem.stages[stage].x,
                               problem.stages[stage].u, problem.dt,
                               predicted);
        for (int state = 0; state < kStateDim; ++state) {
            audit.dynamics = std::max(
                audit.dynamics,
                std::fabs(predicted[state] -
                          problem.stages[stage + 1].x[state]));
        }

        Vec<kConstraintCapacity> rows;
        constraints.evaluate(problem.stages[stage].x,
                             problem.stages[stage].u, stage, rows);
        for (int row = 0; row < kUserConstraintRows; ++row)
            audit.inequality = std::max(audit.inequality, rows[row]);
        for (int pair = 0; pair < kComplementarityPairs; ++pair) {
            int first = -1;
            int second = -1;
            constraints.complementarity_pair(stage, pair, first, second);
            audit.mpcc = std::max(
                audit.mpcc, std::fabs(rows[first] * rows[second]));
        }

        const Vec<3> position = state_vector3(
            problem.stages[stage].x,
            StateIndex::foot_position(moving_foot, 0));
        const Vec<3> force = control_vector3(
            problem.stages[stage].u,
            ControlIndex::contact_force(moving_foot, 0));
        const Vec<3> velocity = control_vector3(
            problem.stages[stage].u,
            ControlIndex::foot_velocity(moving_foot, 0));
        TerrainSample sample;
        terrain.sample(position, sample);
        audit.clearance = std::max(audit.clearance, sample.gap);
        if (sample.gap > 2e-3 && dot3(sample.normal, force) < 5.0 &&
            velocity.norm2() > 0.01) {
            ++audit.moving_unloaded_stages;
        }
    }
    for (int axis = 0; axis < 3; ++axis) {
        const int state =
            StateIndex::foot_position(moving_foot, axis);
        audit.terminal_foot_error = std::max(
            audit.terminal_foot_error,
            std::fabs(problem.stages[task_checkpoint_stage].x[state] -
                      cost.references[task_checkpoint_stage][state]));
    }
    const Vec<3> terminal_position = state_vector3(
        problem.stages[task_checkpoint_stage].x,
        StateIndex::foot_position(moving_foot, 0));
    TerrainSample terminal_sample;
    terrain.sample(terminal_position, terminal_sample);
    audit.terminal_gap = terminal_sample.gap;
    const int checkpoint_control_stage = std::min(
        task_checkpoint_stage, kRealtimeHorizon - 1);
    const Vec<3> terminal_force = control_vector3(
        problem.stages[checkpoint_control_stage].u,
        ControlIndex::contact_force(moving_foot, 0));
    const Vec<3> terminal_velocity = control_vector3(
        problem.stages[checkpoint_control_stage].u,
        ControlIndex::foot_velocity(moving_foot, 0));
    audit.terminal_normal_force =
        dot3(terminal_sample.normal, terminal_force);
    audit.terminal_speed = std::sqrt(terminal_velocity.norm2_sq());
    audit.task_displacement = terminal_position[0] - initial_moving_foot_x;
    return audit;
}

inline bool full_kkt(const nmpc::Status status,
                     const nmpc::SolverStats& stats,
                     const RealtimePlannerAudit& audit,
                     const nmpc::ContactIPMParams& parameters) {
    return status == nmpc::Status::SUCCESS &&
           stats.primal_infeas <= parameters.tol_primal &&
           stats.dual_infeas <= kRealtimePublicationDualTolerance &&
           stats.complementarity <= parameters.tol_compl &&
           stats.mpcc_complementarity <= parameters.tol_mpcc &&
           stats.barrier_param <= parameters.mu_conv_threshold &&
           audit.dynamics <= parameters.tol_primal &&
           audit.inequality <= parameters.tol_ineq &&
           audit.mpcc <= parameters.tol_mpcc &&
           audit.terminal_foot_error <= 0.02 &&
           std::fabs(audit.terminal_gap) <= 1e-4 &&
           audit.terminal_speed <= 0.01 &&
           audit.terminal_normal_force >= 5.0;
}

}  // namespace realtime_detail

class QuadrupedCITORealtimePlanner {
public:
    using Clock = std::chrono::steady_clock;
    using ProblemType = realtime_detail::RealtimeProblem;

    explicit QuadrupedCITORealtimePlanner(
        SharedTerrain physical_terrain,
        RealtimePlannerConfig configuration = RealtimePlannerConfig{})
        : physical_terrain_(physical_terrain),
          planner_terrain_(physical_terrain_),
          configuration_(configuration),
          robot_(go1_robot_parameters()),
          dynamics_(robot_),
          constraints_(planner_terrain_, robot_),
          problem_(std::make_unique<ProblemType>()),
          cold_seed_problem_(std::make_unique<ProblemType>()),
          converged_problem_(std::make_unique<ProblemType>()),
          converged_cost_(std::make_unique<
              QuadraticTrackingCost<kRealtimeHorizon>>()),
          solver_(std::make_unique<Solver>()),
          task_sequence_(configuration.maximum_contact_tasks),
          converged_task_sequence_(configuration.maximum_contact_tasks) {
        solver_parameters_ = realtime_detail::solver_parameters();
        solver_->configure(solver_parameters_);
    }

    nmpc::Status initialize(const Vec<kStateDim>& measured_state) {
        if (configuration_.rate_hz <= 0 ||
            !realtime_detail::finite_value(
                configuration_.publication_headroom_ms) ||
            configuration_.publication_headroom_ms < 0.0 ||
            configuration_.maximum_contact_tasks < 0) {
            return nmpc::Status::BAD_ARGUMENT;
        }
        const double requested_shift =
            1.0 / (configuration_.rate_hz * kRealtimeTimeStep);
        shift_steps_ = static_cast<int>(std::lround(requested_shift));
        if (shift_steps_ <= 0 || shift_steps_ > kRealtimeHorizon ||
            std::fabs(requested_shift - shift_steps_) > 1e-9) {
            return nmpc::Status::BAD_ARGUMENT;
        }

        initialize_standing_problem(*problem_, dynamics_, cost_, constraints_,
                                    kRealtimeTimeStep);
        problem_->x0 = measured_state;
        cost_.set_reference_all(measured_state);
        for (int stage = 0; stage <= kRealtimeHorizon; ++stage)
            problem_->stages[stage].x = measured_state;
        realtime_detail::configure_cost(
            cost_, configuration_.sustained_contact_tasks);
        nmpc::Status guess_status = nmpc::Status::SUCCESS;
        if (configuration_.sustained_contact_tasks) {
            task_sequence_.initialize(measured_state);
            const nmpc::Status support_status =
                configure_sustained_support_schedule();
            if (support_status != nmpc::Status::SUCCESS)
                return support_status;
            realtime_detail::configure_sustained_references(
                cost_, task_sequence_, planner_terrain_);
            guess_status = realtime_detail::seed_sustained_range(
                *problem_, robot_, planner_terrain_, task_sequence_, 0);
        } else {
            const double target_x = measured_state[
                StateIndex::foot_position(kRealtimeMovingFoot, 0)] +
                kRealtimeRequestedStep;
            const double target_y = measured_state[
                StateIndex::foot_position(kRealtimeMovingFoot, 1)];
            cost_.references[kRealtimeHorizon][
                StateIndex::foot_position(kRealtimeMovingFoot, 0)] = target_x;
            cost_.references[kRealtimeHorizon][
                StateIndex::foot_position(kRealtimeMovingFoot, 2)] =
                planner_terrain_.height(target_x, target_y);
            guess_status = realtime_detail::initialize_transition_guess(
                *problem_, robot_, planner_terrain_, 0.0);
        }
        if (guess_status != nmpc::Status::SUCCESS) return guess_status;
        *cold_seed_problem_ = *problem_;
        if (!configuration_.sustained_contact_tasks) {
            guess_status = realtime_detail::
                blend_initial_contact_forces_toward_terrain_normals(
                    *problem_, planner_terrain_,
                    realtime_detail::kInitialTerrainNormalForceBlend);
        }
        if (guess_status != nmpc::Status::SUCCESS) return guess_status;
        const RealtimeContactTask task = current_task();
        initial_moving_foot_x_ = task.origin_x;
        initialized_ = true;
        has_valid_plan_ = false;
        has_converged_warm_start_ = false;
        task_witness_valid_ = false;
        task_witness_id_ = task.id;
        pending_shift_steps_ = 0;
        next_shift_steps_ = shift_steps_;
        return nmpc::Status::SUCCESS;
    }

    Vec<kStateDim> nominal_state() const {
        return go1_home_state(planner_terrain_);
    }

    RealtimePlannerResult cold_solve(
        Clock::time_point deadline = Clock::time_point::max()) {
        if (!initialized_) return RealtimePlannerResult{};
        const Clock::time_point start = Clock::now();
        const Clock::time_point solver_start = Clock::now();
        configure_time_limit(deadline);
        planner_terrain_.reset_sample_calls();
        nmpc::Status status = solver_->solve(*problem_);
        const nmpc::SolverStats primary_stats = solver_->last_stats();
        bool cold_fallback_used = false;
        const bool deadline_available =
            deadline == Clock::time_point::max() || Clock::now() < deadline;
        if (realtime_detail::cold_start_retryable(status) &&
            deadline_available) {
            cold_fallback_used = true;
            *problem_ = *cold_seed_problem_;
            if (!configuration_.sustained_contact_tasks) {
                status = realtime_detail::
                    blend_initial_contact_forces_toward_terrain_normals(
                        *problem_, planner_terrain_,
                        realtime_detail::kFallbackTerrainNormalForceBlend);
            } else {
                status = nmpc::Status::SUCCESS;
            }
            solver_ = std::make_unique<Solver>();
            solver_->configure(solver_parameters_);
            if (status == nmpc::Status::SUCCESS) {
                if (deadline != Clock::time_point::max() &&
                    Clock::now() >= deadline) {
                    status = nmpc::Status::TIME_LIMIT;
                } else {
                    configure_time_limit(deadline);
                    status = solver_->solve(*problem_);
                }
            }
        }
        const Clock::time_point solver_end = Clock::now();
        const std::uint64_t solver_terrain_samples =
            planner_terrain_.sample_calls();
        const double solver_terrain_ms = planner_terrain_.sample_time_ms();
        const int candidate_index = 1 - valid_plan_index_;
        task_buffers_[candidate_index] = current_task();
        bool candidate_ready = false;
        bool candidate_has_transition_witness = false;
        RealtimePlannerResult result = finish_candidate(
            status, start, solver_start, solver_end, deadline,
            solver_terrain_samples, solver_terrain_ms,
            plan_buffers_[candidate_index], candidate_ready,
            candidate_has_transition_witness);
        if (cold_fallback_used) {
            accumulate_solver_work(result.stats, primary_stats);
            result.cold_fallback_used = true;
        }
        finalize_publication(result, candidate_index, candidate_ready,
                             candidate_has_transition_witness, start,
                             deadline);
        return result;
    }

    RealtimePlannerResult warm_update(
        const Vec<kStateDim>& measured_state,
        Clock::time_point deadline = Clock::time_point::max(),
        int elapsed_periods = 1) {
        RealtimePlannerResult result;
        if (!initialized_ || !has_converged_warm_start_) return result;
        const Clock::time_point start = Clock::now();
        if (next_shift_steps_ <= 0 ||
            next_shift_steps_ > kRealtimeHorizon) {
            result.status = nmpc::Status::BAD_ARGUMENT;
            result.fallback = has_valid_plan_;
            return result;
        }
        if (elapsed_periods <= 0) {
            result.status = nmpc::Status::BAD_ARGUMENT;
            result.fallback = has_valid_plan_;
            return result;
        }
        const long long requested_shift_steps =
            static_cast<long long>(pending_shift_steps_) +
            static_cast<long long>(elapsed_periods) * shift_steps_;
        const bool pending_recovery =
            configuration_.sustained_contact_tasks &&
            pending_shift_steps_ > 0;
        const bool deadline_already_expired =
            deadline != Clock::time_point::max() && Clock::now() >= deadline;
        const bool recovery_seed = pending_recovery &&
            (requested_shift_steps > kRealtimeHorizon ||
             deadline_already_expired);
        if (requested_shift_steps <= 0 ||
            requested_shift_steps > std::numeric_limits<int>::max() ||
            (!recovery_seed && requested_shift_steps > kRealtimeHorizon)) {
            result.status = nmpc::Status::BAD_ARGUMENT;
            result.fallback = has_valid_plan_;
            return result;
        }
        const int absolute_shift_steps =
            static_cast<int>(requested_shift_steps);
        const int trajectory_shift_steps = recovery_seed
            ? std::min(absolute_shift_steps, kRealtimeHorizon)
            : absolute_shift_steps;
        restore_converged_warm_start();
        const Clock::time_point shift_start = Clock::now();
        const nmpc::Status shift_status = solver_->shift_for_warmstart(
            *problem_, measured_state, trajectory_shift_steps,
            realtime_detail::apply_measured_state_defect);
        const Clock::time_point shift_end = Clock::now();
        result.shift_ms = elapsed_ms(shift_start, shift_end);
        if (shift_status != nmpc::Status::SUCCESS) {
            restore_converged_warm_start();
            result.status = shift_status;
            result.end_to_end_ms = elapsed_ms(start, Clock::now());
            result.deadline_miss = Clock::now() > deadline;
            result.fallback = has_valid_plan_;
            next_shift_steps_ = shift_steps_;
            return result;
        }
        if (configuration_.sustained_contact_tasks) {
            if (!task_sequence_.advance(absolute_shift_steps)) {
                restore_converged_warm_start();
                result.status = nmpc::Status::BAD_ARGUMENT;
                result.end_to_end_ms = elapsed_ms(start, Clock::now());
                result.fallback = has_valid_plan_;
                next_shift_steps_ = shift_steps_;
                return result;
            }
            const nmpc::Status support_status =
                configure_sustained_support_schedule();
            if (support_status != nmpc::Status::SUCCESS) {
                restore_converged_warm_start();
                result.status = support_status;
                result.end_to_end_ms = elapsed_ms(start, Clock::now());
                result.fallback = has_valid_plan_;
                next_shift_steps_ = shift_steps_;
                return result;
            }
            realtime_detail::configure_sustained_references(
                cost_, task_sequence_, planner_terrain_);
            const RealtimeContactTask task = current_task();
            if (recovery_seed) {
                for (int stage = 0; stage <= kRealtimeHorizon; ++stage)
                    problem_->stages[stage] = {};
                problem_->x0 = measured_state;
                problem_->stages[0].x = measured_state;
            }
            const nmpc::Status seed_status =
                realtime_detail::seed_sustained_range(
                    *problem_, robot_, planner_terrain_, task_sequence_,
                    recovery_seed
                        ? 0
                        : kRealtimeHorizon - trajectory_shift_steps);
            if (seed_status != nmpc::Status::SUCCESS) {
                restore_converged_warm_start();
                result.status = seed_status;
                result.end_to_end_ms = elapsed_ms(start, Clock::now());
                result.fallback = has_valid_plan_;
                next_shift_steps_ = shift_steps_;
                return result;
            }
            if (task.id != task_witness_id_) {
                task_witness_id_ = task.id;
                task_witness_valid_ = false;
            }
            initial_moving_foot_x_ = task.origin_x;
        } else {
            realtime_detail::shift_references(cost_, trajectory_shift_steps);
        }

        const Clock::time_point solver_start = Clock::now();
        // Optional runner-level headroom covers audit, extraction, and handoff.
        configure_time_limit(deadline,
                             configuration_.publication_headroom_ms);
        planner_terrain_.reset_sample_calls();
        const nmpc::Status status = solver_->solve_warm(*problem_);
        const Clock::time_point solver_end = Clock::now();
        const std::uint64_t solver_terrain_samples =
            planner_terrain_.sample_calls();
        const double solver_terrain_ms = planner_terrain_.sample_time_ms();
        const int candidate_index = 1 - valid_plan_index_;
        task_buffers_[candidate_index] = current_task();
        bool candidate_ready = false;
        bool candidate_has_transition_witness = false;
        result = finish_candidate(status, start, solver_start, solver_end,
                                  deadline, solver_terrain_samples,
                                  solver_terrain_ms,
                                  plan_buffers_[candidate_index],
                                  candidate_ready,
                                  candidate_has_transition_witness);
        result.shift_consumed = true;
        // Account for every consumed period, but only a published candidate
        // becomes the basis of a later warm start. The next request rebases
        // the last converged trajectory by this complete accumulated shift.
        pending_shift_steps_ = absolute_shift_steps;
        next_shift_steps_ = shift_steps_;
        result.shift_ms = elapsed_ms(shift_start, shift_end);
        finalize_publication(result, candidate_index, candidate_ready,
                             candidate_has_transition_witness, start,
                             deadline);
        if (!result.published) discard_failed_problem();
        return result;
    }

    int shift_steps() const { return shift_steps_; }
    int next_shift_steps() const { return next_shift_steps_; }
    const SharedTerrain& physical_terrain() const { return physical_terrain_; }
    const realtime_detail::PlannerTerrain& planner_terrain() const {
        return planner_terrain_;
    }
    const ProblemType& problem() const { return *problem_; }
    const ContactPlan<kRealtimeHorizon>* last_valid_plan() const {
        return has_valid_plan_ ? &plan_buffers_[valid_plan_index_] : nullptr;
    }
    RealtimeContactTask active_task() const { return current_task(); }
    const RealtimeContactTask* last_valid_task() const {
        return has_valid_plan_ ? &task_buffers_[valid_plan_index_] : nullptr;
    }
    int absolute_knot() const {
        return configuration_.sustained_contact_tasks
            ? task_sequence_.absolute_knot()
            : 0;
    }
    bool has_valid_plan() const { return has_valid_plan_; }

private:
    using Solver = nmpc::ContactIPM<
        kStateDim, kControlDim, kConstraintCapacity, kRealtimeHorizon>;

    RealtimeContactTask current_task() const {
        if (configuration_.sustained_contact_tasks)
            return task_sequence_.active_task();
        RealtimeContactTask task;
        task.origin_x = initialized_
            ? initial_moving_foot_x_
            : problem_->x0[StateIndex::foot_position(
                  kRealtimeMovingFoot, 0)];
        task.target_x = task.origin_x + kRealtimeRequestedStep;
        return task;
    }

    nmpc::Status configure_sustained_support_schedule() {
        if (!configuration_.sustained_contact_tasks)
            return nmpc::Status::SUCCESS;
        int designated_feet[kRealtimeHorizon];
        for (int stage = 0; stage < kRealtimeHorizon; ++stage) {
            const int absolute_knot = task_sequence_.absolute_knot() + stage;
            // The current hard-support model has three support slots. In an
            // all-support interval, leave only the current/next task foot
            // unconstrained; the executed audit remains the final authority.
            designated_feet[stage] =
                task_sequence_.designated_support_foot_at(absolute_knot);
        }
        return constraints_.set_designated_foot_schedule(
            designated_feet, kRealtimeHorizon,
            kRealtimeHardSupportMinimumNormalForce);
    }

    static double elapsed_ms(Clock::time_point start,
                             Clock::time_point end) {
        return std::chrono::duration<double, std::milli>(end - start).count();
    }

    static void accumulate_solver_work(
        nmpc::SolverStats& combined,
        const nmpc::SolverStats& earlier) {
        combined.outer_iterations += earlier.outer_iterations;
        combined.inner_iterations += earlier.inner_iterations;
        combined.riccati_failures += earlier.riccati_failures;
        combined.line_search_evals += earlier.line_search_evals;
        combined.warm_near_full_step_trials +=
            earlier.warm_near_full_step_trials;
        combined.warm_near_full_step_accepts +=
            earlier.warm_near_full_step_accepts;
        combined.model_evaluations += earlier.model_evaluations;
        combined.kkt_assemblies += earlier.kkt_assemblies;
        combined.riccati_factorizations +=
            earlier.riccati_factorizations;
        combined.riccati_rhs_solves += earlier.riccati_rhs_solves;
        combined.exact_hessian_analytic_calls +=
            earlier.exact_hessian_analytic_calls;
        combined.exact_hessian_fd_calls += earlier.exact_hessian_fd_calls;
        combined.deadline_checks += earlier.deadline_checks;
        combined.time_limit_hit += earlier.time_limit_hit;
        combined.watchdog_resets += earlier.watchdog_resets;
        combined.soc_steps += earlier.soc_steps;
        combined.max_regularization = std::max(
            combined.max_regularization, earlier.max_regularization);
        if (earlier.riccati_factorizations > 0)
            combined.first_regularization = earlier.first_regularization;
        combined.solve_time_ms += earlier.solve_time_ms;
        combined.model_eval_time_ms += earlier.model_eval_time_ms;
        combined.dynamics_eval_time_ms += earlier.dynamics_eval_time_ms;
        combined.dynamics_jacobian_time_ms +=
            earlier.dynamics_jacobian_time_ms;
        combined.cost_derivative_time_ms +=
            earlier.cost_derivative_time_ms;
        combined.constraint_eval_time_ms +=
            earlier.constraint_eval_time_ms;
        combined.constraint_jacobian_time_ms +=
            earlier.constraint_jacobian_time_ms;
        combined.constraint_value_jacobian_time_ms +=
            earlier.constraint_value_jacobian_time_ms;
        combined.residual_eval_time_ms += earlier.residual_eval_time_ms;
        combined.kkt_assembly_time_ms += earlier.kkt_assembly_time_ms;
        combined.riccati_time_ms += earlier.riccati_time_ms;
        combined.line_search_time_ms += earlier.line_search_time_ms;
        combined.finalization_time_ms += earlier.finalization_time_ms;
    }

    void configure_time_limit(Clock::time_point deadline,
                              double headroom_ms = 0.0) {
        double remaining_ms = 0.0;
        if (deadline != Clock::time_point::max()) {
            remaining_ms = std::max(
                0.001,
                std::chrono::duration<double, std::milli>(
                    deadline - Clock::now()).count() - headroom_ms);
        }
        solver_->set_time_limit_ms(remaining_ms);
    }

    void restore_converged_warm_start() {
        *problem_ = *converged_problem_;
        cost_ = *converged_cost_;
        task_sequence_ = converged_task_sequence_;
        if (configuration_.sustained_contact_tasks)
            (void)configure_sustained_support_schedule();
    }

    void save_converged_warm_start() {
        *converged_problem_ = *problem_;
        *converged_cost_ = cost_;
        converged_task_sequence_ = task_sequence_;
        has_converged_warm_start_ = true;
        pending_shift_steps_ = 0;
    }

    void discard_failed_problem() {
        *problem_ = *converged_problem_;
        cost_ = *converged_cost_;
    }

    RealtimePlannerResult finish_candidate(
        nmpc::Status status, Clock::time_point start,
        Clock::time_point solver_start, Clock::time_point solver_end,
        Clock::time_point deadline,
        std::uint64_t solver_terrain_samples,
        double solver_terrain_ms,
        ContactPlan<kRealtimeHorizon>& candidate,
        bool& candidate_ready,
        bool& candidate_has_transition_witness) {
        RealtimePlannerResult result;
        result.status = status;
        result.stats = solver_->last_stats();
        result.solver_ms = elapsed_ms(solver_start, solver_end);
        result.solver_terrain_samples = solver_terrain_samples;
        result.solver_terrain_ms = solver_terrain_ms;

        const Clock::time_point audit_start = Clock::now();
        planner_terrain_.reset_sample_calls();
        const RealtimeContactTask task = current_task();
        int task_checkpoint_stage = kRealtimeHorizon;
        if (configuration_.sustained_contact_tasks) {
            const int checkpoint_knot =
                (static_cast<int>(task.id) + 1) *
                    kRealtimeSustainedTaskSpacing - 1;
            task_checkpoint_stage = checkpoint_knot -
                task_sequence_.absolute_knot();
        }
        result.audit = realtime_detail::audit_solution(
            *problem_, dynamics_, constraints_, planner_terrain_, cost_,
            task.origin_x, task.moving_foot, task_checkpoint_stage);
        result.full_kkt = realtime_detail::full_kkt(
            status, result.stats, result.audit, solver_parameters_);
        const bool has_transition_witness =
            result.audit.clearance >= 0.02 &&
            result.audit.moving_unloaded_stages > 0;
        result.task_pass =
            result.audit.task_displacement >= 0.06 &&
            ((task_witness_id_ == task.id && task_witness_valid_) ||
             has_transition_witness);
        const Clock::time_point audit_end = Clock::now();
        result.audit_terrain_samples = planner_terrain_.sample_calls();
        result.audit_terrain_ms = planner_terrain_.sample_time_ms();
        result.audit_ms = elapsed_ms(audit_start, audit_end);

        result.deadline_miss = Clock::now() > deadline;
        if (result.full_kkt && result.task_pass && !result.deadline_miss) {
            ContactClassification classification;
            const Clock::time_point extract_start = Clock::now();
            planner_terrain_.reset_sample_calls();
            const nmpc::Status extract_status = extract_contact_plan(
                *problem_, planner_terrain_, classification, candidate);
            const Clock::time_point extract_end = Clock::now();
            result.plan_extract_terrain_samples =
                planner_terrain_.sample_calls();
            result.plan_extract_terrain_ms =
                planner_terrain_.sample_time_ms();
            result.plan_extract_ms = elapsed_ms(extract_start, extract_end);
            result.deadline_miss = extract_end > deadline;
            if (extract_status == nmpc::Status::SUCCESS &&
                !result.deadline_miss) {
                candidate_ready = true;
                candidate_has_transition_witness =
                    has_transition_witness;
            }
        }
        result.end_to_end_ms = elapsed_ms(start, Clock::now());
        return result;
    }

    void finalize_publication(
        RealtimePlannerResult& result, int candidate_index,
        bool candidate_ready, bool candidate_has_transition_witness,
        Clock::time_point start, Clock::time_point deadline) {
        const Clock::time_point before_commit = Clock::now();
        result.deadline_miss = result.deadline_miss ||
                               before_commit > deadline;
        Clock::time_point final_time = before_commit;
        if (candidate_ready && !result.deadline_miss) {
            const int previous_index = valid_plan_index_;
            const bool previously_valid = has_valid_plan_;
            valid_plan_index_ = candidate_index;
            has_valid_plan_ = true;
            final_time = Clock::now();
            result.deadline_miss = final_time > deadline;
            if (result.deadline_miss) {
                valid_plan_index_ = previous_index;
                has_valid_plan_ = previously_valid;
            } else {
                const RealtimeContactTask& task = task_buffers_[candidate_index];
                if (task_witness_id_ != task.id) {
                    task_witness_id_ = task.id;
                    task_witness_valid_ = false;
                }
                task_witness_valid_ = task_witness_valid_ ||
                                      candidate_has_transition_witness;
                result.task_id = task.id;
                result.moving_foot = task.moving_foot;
                result.task_transition_witness =
                    candidate_has_transition_witness;
                result.published = true;
                save_converged_warm_start();
            }
        }
        result.fallback = has_valid_plan_ && !result.published;
        result.end_to_end_ms = elapsed_ms(start, final_time);
    }

    SharedTerrain physical_terrain_;
    realtime_detail::PlannerTerrain planner_terrain_;
    RealtimePlannerConfig configuration_;
    RobotParameters robot_;
    SRBDDynamics dynamics_;
    ContactConstraints<kRealtimeHorizon, realtime_detail::PlannerTerrain>
        constraints_;
    QuadraticTrackingCost<kRealtimeHorizon> cost_;
    std::unique_ptr<ProblemType> problem_;
    std::unique_ptr<ProblemType> cold_seed_problem_;
    std::unique_ptr<ProblemType> converged_problem_;
    std::unique_ptr<QuadraticTrackingCost<kRealtimeHorizon>> converged_cost_;
    std::unique_ptr<Solver> solver_;
    nmpc::ContactIPMParams solver_parameters_;
    ContactPlan<kRealtimeHorizon> plan_buffers_[2];
    RealtimeContactTask task_buffers_[2];
    realtime_detail::RealtimeContactTaskSequence task_sequence_;
    realtime_detail::RealtimeContactTaskSequence converged_task_sequence_;
    double initial_moving_foot_x_ = 0.0;
    int shift_steps_ = 0;
    int next_shift_steps_ = 0;
    int pending_shift_steps_ = 0;
    int valid_plan_index_ = 0;
    bool initialized_ = false;
    bool has_valid_plan_ = false;
    bool has_converged_warm_start_ = false;
    bool task_witness_valid_ = false;
    std::uint64_t task_witness_id_ = 0;
};

}  // namespace quadruped_cito
