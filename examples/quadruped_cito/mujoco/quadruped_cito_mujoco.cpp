#include <algorithm>
#include <array>
#include <cerrno>
#include <chrono>
#include <cmath>
#include <condition_variable>
#include <cstdio>
#include <cstdint>
#include <cstdlib>
#include <cstring>
#include <fstream>
#include <limits>
#include <mutex>
#include <string>
#include <thread>
#include <vector>

#if defined(_WIN32)
#define NOMINMAX
#include <windows.h>
#elif defined(__linux__)
#include <pthread.h>
#include <sched.h>
#include <time.h>
#endif

#include <mujoco/mujoco.h>

#include "examples/quadruped_cito/mujoco/mujoco_go1_adapter.hpp"
#include "examples/quadruped_cito/mujoco/mujoco_replay_snapshot.hpp"
#include "examples/quadruped_cito/mujoco/quadruped_cito_realtime_execution.hpp"
#include "examples/quadruped_cito/quadruped_cito_contact_feedback.hpp"
#include "examples/quadruped_cito/quadruped_cito_four_step_planner.hpp"
#include "examples/quadruped_cito/quadruped_cito_go1.hpp"
#include "examples/quadruped_cito/quadruped_cito_plan_io.hpp"
#include "examples/quadruped_cito/quadruped_cito_realtime_planner.hpp"
#include "examples/quadruped_cito/quadruped_cito_replay_metrics.hpp"
#include "examples/quadruped_cito/quadruped_cito_swing_reference.hpp"

namespace {

using namespace quadruped_cito;

bool parse_terrain(const std::string& name, unsigned int seed,
                   double amplitude, double slope_x, double slope_y,
                   double step_height, bool slope_x_set, bool slope_y_set,
                   SharedTerrain& terrain) {
    if (name == "flat") {
        terrain = register_go1_terrain(SharedTerrain::flat());
    } else if (name == "smooth") {
        terrain = register_go1_terrain(SharedTerrain::sinusoidal());
    } else if (name == "sinusoidal") {
        terrain = SharedTerrain::sinusoidal(amplitude);
        terrain.slope_x = slope_x_set ? slope_x : 0.0;
        terrain.slope_y = slope_y_set ? slope_y : 0.0;
        terrain = register_go1_terrain(terrain);
    } else if (name == "random_smooth" || name == "random-smooth") {
        terrain = register_go1_terrain(
            SharedTerrain::random_smooth(seed, amplitude));
        if (slope_x_set) terrain.slope_x = slope_x;
        if (slope_y_set) terrain.slope_y = slope_y;
    } else if (name == "slope" || name == "longitudinal-slope" ||
               name == "longitudinal_slope") {
        terrain = register_go1_terrain(
            SharedTerrain::slope(slope_x_set ? slope_x : 0.10,
                                 slope_y_set ? slope_y : 0.0));
    } else if (name == "cross-slope" || name == "cross_slope") {
        terrain = register_go1_terrain(
            SharedTerrain::slope(slope_x_set ? slope_x : 0.0,
                                 slope_y_set ? slope_y : 0.10));
    } else if (name == "smooth_step" || name == "smooth-step") {
        terrain = SharedTerrain::smooth_step(step_height);
        terrain.slope_x = slope_x_set ? slope_x : 0.0;
        terrain.slope_y = slope_y_set ? slope_y : 0.0;
        terrain = register_go1_terrain(terrain);
    } else {
        return false;
    }
    return true;
}

enum class ForceReferenceMode { CITO, UNIFORM };

// Derived contact forces can drift by a few millinewtons after restoring the
// same integration state in a fresh MuJoCo process; contact masks remain exact.
constexpr double kSnapshotContactForceTolerance = 5e-3;

enum class ReplanSeedPolicy {
    ALTERNATING,
    FORWARD,
    REVERSE,
    LEFT_FIRST,
    RIGHT_FIRST,
    CYCLING,
    MULTISTART
};

const char* replan_seed_policy_name(ReplanSeedPolicy policy) {
    switch (policy) {
        case ReplanSeedPolicy::ALTERNATING: return "alternating";
        case ReplanSeedPolicy::FORWARD: return "forward";
        case ReplanSeedPolicy::REVERSE: return "reverse";
        case ReplanSeedPolicy::LEFT_FIRST: return "left_first";
        case ReplanSeedPolicy::RIGHT_FIRST: return "right_first";
        case ReplanSeedPolicy::CYCLING: return "cycling";
        case ReplanSeedPolicy::MULTISTART: return "multistart";
    }
    return "unknown";
}

bool parse_replan_seed_policy(const std::string& name,
                              ReplanSeedPolicy& policy) {
    if (name == "alternating") {
        policy = ReplanSeedPolicy::ALTERNATING;
    } else if (name == "forward") {
        policy = ReplanSeedPolicy::FORWARD;
    } else if (name == "reverse") {
        policy = ReplanSeedPolicy::REVERSE;
    } else if (name == "left_first") {
        policy = ReplanSeedPolicy::LEFT_FIRST;
    } else if (name == "right_first") {
        policy = ReplanSeedPolicy::RIGHT_FIRST;
    } else if (name == "cycling") {
        policy = ReplanSeedPolicy::CYCLING;
    } else if (name == "multistart") {
        policy = ReplanSeedPolicy::MULTISTART;
    } else {
        return false;
    }
    return true;
}

FourStepSeed segment_seed(ReplanSeedPolicy policy, int segment) {
    if (policy == ReplanSeedPolicy::REVERSE) return FourStepSeed::REVERSE;
    if (policy == ReplanSeedPolicy::LEFT_FIRST)
        return FourStepSeed::LEFT_FIRST;
    if (policy == ReplanSeedPolicy::RIGHT_FIRST)
        return FourStepSeed::RIGHT_FIRST;
    if (policy == ReplanSeedPolicy::CYCLING)
        return static_cast<FourStepSeed>(segment % kFourStepSeedCount);
    if (policy == ReplanSeedPolicy::ALTERNATING && segment % 2 != 0)
        return FourStepSeed::REVERSE;
    return FourStepSeed::FORWARD;
}

struct ReplayOptions {
    double mass_scale = 1.0;
    double friction_scale = 1.0;
    Vec<3> push_force_world;
    double push_start = 0.0;
    double push_duration = 0.0;
    ForceReferenceMode force_reference = ForceReferenceMode::CITO;
    bool require_timing = true;
    bool early_contact_feedback = true;
    bool confirmed_stabilization = true;
    bool late_touchdown_search = false;
    bool trace_execution = false;
    double stabilization_duration = 2.0;
    double maximum_stabilization_duration = 3.0;
    std::string snapshot_prefix;

    ReplayOptions() { push_force_world.zero(); }
};

const char* force_reference_name(ForceReferenceMode mode) {
    return mode == ForceReferenceMode::CITO ? "cito" : "uniform";
}

constexpr int kContactExecutionPolicyVersion = 2;

ConvexWBCParameters go1_wbc_parameters() {
    ConvexWBCParameters parameters;
    parameters.force_tracking_weight = 0.01;
    parameters.friction = 0.6;
    parameters.maximum_normal_force = 100.0;
    parameters.maximum_iterations = 200;
    parameters.convergence_tolerance = 1e-10;
    for (int axis = 0; axis < 3; ++axis) {
        parameters.wrench_weights[axis] = 1.0;
        parameters.wrench_weights[3 + axis] = 10.0;
        parameters.swing_position_kp[axis] = 800.0;
        parameters.swing_velocity_kd[axis] = 30.0;
        parameters.base_position_kp[axis] = 500.0;
        parameters.base_velocity_kd[axis] = 45.0;
    }
    parameters.base_position_kp[2] = 300.0;
    parameters.base_velocity_kd[2] = 30.0;
    parameters.swing_position_kp[2] = 2400.0;
    parameters.swing_velocity_kd[2] = 60.0;
    return parameters;
}

ContactExecutionFeedbackParameters go1_contact_feedback_parameters(
    const ConvexWBCParameters& wbc_parameters) {
    ContactExecutionFeedbackParameters parameters;
    parameters.contact_blend_force = wbc_parameters.contact_blend_force;
    return parameters;
}

MujocoReplaySnapshotConfiguration snapshot_configuration(
    const SharedTerrain& terrain, unsigned int terrain_seed,
    double terrain_amplitude, const ReplayOptions& options) {
    MujocoReplaySnapshotConfiguration configuration;
    configuration.terrain = terrain;
    configuration.terrain_seed = terrain_seed;
    configuration.terrain_amplitude_argument = terrain_amplitude;
    configuration.mass_scale = options.mass_scale;
    configuration.friction_scale = options.friction_scale;
    configuration.push_force_world = options.push_force_world;
    configuration.push_start = options.push_start;
    configuration.push_duration = options.push_duration;
    configuration.force_reference =
        options.force_reference == ForceReferenceMode::CITO ? 0 : 1;
    configuration.require_timing = options.require_timing;
    configuration.early_contact_feedback = options.early_contact_feedback;
    configuration.confirmed_stabilization = options.confirmed_stabilization;
    configuration.late_touchdown_search = options.late_touchdown_search;
    configuration.trace_execution = options.trace_execution;
    configuration.stabilization_duration = options.stabilization_duration;
    configuration.maximum_stabilization_duration =
        options.maximum_stabilization_duration;
    configuration.contact_execution_policy_version =
        kContactExecutionPolicyVersion;
    configuration.synchronized_contact_cache = true;
    configuration.history_aware_recontact = true;
    configuration.measured_load_torque_handoff = true;
    configuration.wbc_parameters = go1_wbc_parameters();
    configuration.contact_feedback_parameters =
        go1_contact_feedback_parameters(configuration.wbc_parameters);
    return configuration;
}

ReplayOptions replay_options_from_snapshot(
    const MujocoReplaySnapshotConfiguration& configuration,
    bool trace_execution_override) {
    ReplayOptions options;
    options.mass_scale = configuration.mass_scale;
    options.friction_scale = configuration.friction_scale;
    options.push_force_world = configuration.push_force_world;
    options.push_start = configuration.push_start;
    options.push_duration = configuration.push_duration;
    options.force_reference = configuration.force_reference == 0
        ? ForceReferenceMode::CITO
        : ForceReferenceMode::UNIFORM;
    options.require_timing = configuration.require_timing;
    options.early_contact_feedback = configuration.early_contact_feedback;
    options.confirmed_stabilization = configuration.confirmed_stabilization;
    options.late_touchdown_search = configuration.late_touchdown_search;
    options.trace_execution =
        configuration.trace_execution || trace_execution_override;
    options.stabilization_duration = configuration.stabilization_duration;
    options.maximum_stabilization_duration =
        configuration.maximum_stabilization_duration;
    return options;
}

std::string snapshot_stem(const std::string& prefix, int segment) {
    char suffix[32] = {};
    std::snprintf(suffix, sizeof(suffix), "_segment_%03d", segment);
    return prefix + suffix;
}

std::string paired_plan_path(const std::string& snapshot_path) {
    constexpr const char* extension = ".snapshot";
    const std::size_t extension_length = std::strlen(extension);
    if (snapshot_path.size() <= extension_length ||
        snapshot_path.compare(snapshot_path.size() - extension_length,
                              extension_length, extension) != 0) {
        return {};
    }
    return snapshot_path.substr(0, snapshot_path.size() - extension_length) +
           ".plan";
}

bool file_exists(const std::string& path) {
    FILE* file = std::fopen(path.c_str(), "r");
    if (!file) return false;
    std::fclose(file);
    return true;
}

void initialize_standing_reference(const mjModel* model,
                                   const MujocoGo1Adapter& adapter,
                                   const WholeBodyState& state,
                                   ContactPlanSample& reference) {
    reference.time = adapter.time();
    reference.stage_index = 0;
    reference.stage_phase = 0.0;
    reference.base = state.base;
    reference.base.linear_velocity_world.zero();
    reference.base.angular_velocity_body.zero();
    const double normal_force = adapter.total_mass() *
                                std::fabs(model->opt.gravity[2]) /
                                kNumFeet;
    for (int foot = 0; foot < kNumFeet; ++foot) {
        reference.feet[foot].position_world =
            state.foot_positions_world[foot];
        reference.feet[foot].velocity_world.zero();
        reference.feet[foot].force_world.zero();
        reference.feet[foot].force_world[2] = normal_force;
        reference.feet[foot].terrain_normal_world.zero();
        reference.feet[foot].terrain_normal_world[2] = 1.0;
        reference.feet[foot].terrain_gap = 0.0;
        reference.feet[foot].normal_force = normal_force;
        reference.feet[foot].planned_contact = true;
    }
}

int run_standing_replay(const mjModel* model, mjData* data,
                        const MujocoGo1Adapter& adapter, double duration) {
    if (!(duration > 0.0) || !(model->opt.timestep > 0.0)) return 2;
    WholeBodyState state;
    if (adapter.read_whole_body_state(state) != Status::SUCCESS) return 1;
    ContactPlanSample reference;
    initialize_standing_reference(model, adapter, state, reference);

    const ConvexWBCParameters parameters = go1_wbc_parameters();
    ConvexWholeBodyController controller(parameters);
    ConvexWBCCommand command;
    MujocoActuatorReport reports[kNumFeet][kGo1JointsPerLeg];

    const int ticks = static_cast<int>(
        std::ceil(duration / model->opt.timestep));
    double squared_position_error = 0.0;
    double maximum_orientation_error = 0.0;
    double maximum_foot_slip = 0.0;
    int saturated_joint_ticks = 0;
    for (int tick = 0; tick < ticks; ++tick) {
        mj_step1(model, data);
        if (adapter.read_whole_body_state(state) != Status::SUCCESS ||
            controller.compute(reference, state, command) != Status::SUCCESS ||
            adapter.apply_command(command, reports) != Status::SUCCESS) {
            std::printf("standing replay controller failed at tick %d\n", tick);
            return 1;
        }
        mj_step2(model, data);
        if (adapter.update_applied_torques(reports) != Status::SUCCESS)
            return 1;

        double position_error_sq = 0.0;
        for (int axis = 0; axis < 3; ++axis) {
            const double error = state.base.position_world[axis] -
                                 reference.base.position_world[axis];
            position_error_sq += error * error;
        }
        squared_position_error += position_error_sq;
        maximum_orientation_error = std::max(
            maximum_orientation_error,
            orientation_error_world(
                reference.base.orientation_body_to_world,
                state.base.orientation_body_to_world).norm2());
        for (int foot = 0; foot < kNumFeet; ++foot) {
            const double dx = state.foot_positions_world[foot][0] -
                              reference.feet[foot].position_world[0];
            const double dy = state.foot_positions_world[foot][1] -
                              reference.feet[foot].position_world[1];
            maximum_foot_slip = std::max(
                maximum_foot_slip, std::sqrt(dx * dx + dy * dy));
            for (int joint = 0; joint < kGo1JointsPerLeg; ++joint) {
                if (reports[foot][joint].saturated)
                    ++saturated_joint_ticks;
            }
        }
    }

    if (adapter.read_whole_body_state(state) != Status::SUCCESS) return 1;
    const double position_rms =
        std::sqrt(squared_position_error / std::max(ticks, 1));
    const double saturation_fraction =
        static_cast<double>(saturated_joint_ticks) /
        std::max(ticks * kNumFeet * kGo1JointsPerLeg, 1);
    std::printf(
        "standing replay: duration=%.3f position_rms=%.6f "
        "max_orientation_error=%.6f max_foot_slip=%.6f "
        "saturation_fraction=%.6f final_height=%.6f\n",
        duration, position_rms, maximum_orientation_error, maximum_foot_slip,
        saturation_fraction, state.base.position_world[2]);
    if (position_rms > 0.02 || maximum_orientation_error > 0.35 ||
        maximum_foot_slip > 0.02 ||
        state.base.position_world[2] <
            reference.base.position_world[2] - 0.05) {
        std::printf("standing replay acceptance gate failed\n");
        return 1;
    }
    return 0;
}

constexpr int kStabilizationConfirmationTicks = 25;

int stabilize_for_replanning(const mjModel* model, mjData* data,
                              const MujocoGo1Adapter& adapter,
                              const SharedTerrain& terrain,
                              double minimum_duration,
                              double maximum_duration,
                              bool require_confirmation) {
    if (!(minimum_duration > 0.0) ||
        maximum_duration < minimum_duration ||
        !(model->opt.timestep > 0.0)) {
        return 2;
    }

    WholeBodyState state;
    if (adapter.read_whole_body_state(state) != Status::SUCCESS) return 1;
    ContactPlanSample reference;
    initialize_standing_reference(model, adapter, state, reference);
    reference.base.orientation_body_to_world.zero();
    reference.base.orientation_body_to_world[0] = 1.0;

    double mean_foot_x = 0.0;
    double mean_foot_y = 0.0;
    double mean_contact_height = 0.0;
    const Go1FootCenterTerrain<SharedTerrain> planner_terrain(terrain);
    for (int foot = 0; foot < kNumFeet; ++foot) {
        const Vec<3>& position = state.foot_positions_world[foot];
        mean_foot_x += position[0] / kNumFeet;
        mean_foot_y += position[1] / kNumFeet;
        mean_contact_height +=
            planner_terrain.height(position[0], position[1]) / kNumFeet;
        TerrainSample terrain_sample;
        if (!terrain.sample(position, terrain_sample)) return 1;
        reference.feet[foot].terrain_normal_world = terrain_sample.normal;
        reference.feet[foot].normal_force = dot3(
            terrain_sample.normal, reference.feet[foot].force_world);
    }
    reference.base.position_world[0] = mean_foot_x + 0.04;
    reference.base.position_world[1] = mean_foot_y;
    reference.base.position_world[2] = mean_contact_height +
        Go1SRBDCalibration{}.com_height_above_foot_centers;

    const ConvexWBCParameters parameters = go1_wbc_parameters();
    ConvexWholeBodyController controller(parameters);
    ConvexWBCCommand command;
    MujocoActuatorReport reports[kNumFeet][kGo1JointsPerLeg];
    const int minimum_ticks = static_cast<int>(
        std::ceil(minimum_duration / model->opt.timestep));
    const int maximum_ticks = static_cast<int>(
        std::ceil(maximum_duration / model->opt.timestep));
    double maximum_foot_slip = 0.0;
    int contact_count = 0;
    double position_error = 1e300;
    double linear_speed = 1e300;
    double angular_speed = 1e300;
    double orientation_error = 1e300;
    bool accepted = false;
    int executed_ticks = 0;
    ConsecutiveGateConfirmation gate_confirmation(
        kStabilizationConfirmationTicks);
    for (int tick = 0; tick < maximum_ticks; ++tick) {
        mj_step1(model, data);
        if (adapter.read_whole_body_state(state) != Status::SUCCESS ||
            controller.compute(reference, state, command) != Status::SUCCESS ||
            adapter.apply_command(command, reports) != Status::SUCCESS) {
            std::printf("replanning stabilization failed at tick %d\n", tick);
            return 1;
        }
        mj_step2(model, data);
        if (adapter.update_applied_torques(reports) != Status::SUCCESS)
            return 1;
        for (int foot = 0; foot < kNumFeet; ++foot) {
            const double dx = state.foot_positions_world[foot][0] -
                              reference.feet[foot].position_world[0];
            const double dy = state.foot_positions_world[foot][1] -
                              reference.feet[foot].position_world[1];
            maximum_foot_slip = std::max(
                maximum_foot_slip, std::sqrt(dx * dx + dy * dy));
        }
        executed_ticks = tick + 1;
        if (executed_ticks < minimum_ticks) continue;

        MujocoFootContact contacts[kNumFeet];
        if (adapter.read_whole_body_state(state) != Status::SUCCESS ||
            adapter.read_foot_contacts(contacts) != Status::SUCCESS) {
            return 1;
        }
        contact_count = 0;
        for (int foot = 0; foot < kNumFeet; ++foot) {
            if (contacts[foot].in_contact) ++contact_count;
        }
        double squared_position_error = 0.0;
        for (int axis = 0; axis < 3; ++axis) {
            const double error = state.base.position_world[axis] -
                                 reference.base.position_world[axis];
            squared_position_error += error * error;
        }
        position_error = std::sqrt(squared_position_error);
        linear_speed = state.base.linear_velocity_world.norm2();
        angular_speed = state.base.angular_velocity_body.norm2();
        Vec<4> identity_orientation;
        identity_orientation.zero();
        identity_orientation[0] = 1.0;
        orientation_error = orientation_error_world(
            identity_orientation,
            state.base.orientation_body_to_world).norm2();
        const bool gate_passes = contact_count == kNumFeet &&
            position_error <= 0.035 && linear_speed <= 0.035 &&
            angular_speed <= 0.06 && orientation_error <= 0.10 &&
            maximum_foot_slip <= 0.02;
        if (!require_confirmation) {
            accepted = true;
            break;
        }
        accepted = gate_confirmation.update(gate_passes);
        if (accepted) break;
    }
    const double duration = executed_ticks * model->opt.timestep;
    std::printf(
        "replanning stabilization: duration=%.3f contacts=%d "
        "position_error=%.6f orientation_error=%.6f "
        "linear_speed=%.6f angular_speed=%.6f "
        "max_foot_slip=%.6f target=(%.6f, %.6f, %.6f) accepted=%d\n",
        duration, contact_count, position_error, orientation_error,
        linear_speed, angular_speed, maximum_foot_slip,
        reference.base.position_world[0],
        reference.base.position_world[1], reference.base.position_world[2],
        accepted ? 1 : 0);
    return accepted ? 0 : 1;
}

constexpr double kTouchdownDetectionForce = 0.5;
constexpr double kContactConfirmationForce = 1.0;
constexpr double kMinimumSwingClearance = 0.02;
constexpr int kContactLossGraceTicks = 5;
constexpr double kTouchdownTraceHalfWindow = 0.3;

struct ReplaySegmentReport {
    int moving_feet = 0;
    int swing_events = 0;
    bool task_success = false;
    bool timing_success = false;
};

struct PlannedTouchdown {
    int foot = -1;
    int event = -1;
    double time = 0.0;
};

template <int Horizon>
std::vector<PlannedTouchdown> planned_touchdowns(
    const ContactPlan<Horizon>& plan) {
    std::vector<PlannedTouchdown> touchdowns;
    int event_count[kNumFeet] = {};
    for (int stage = 1; stage < Horizon; ++stage) {
        for (int foot = 0; foot < kNumFeet; ++foot) {
            if (!plan.stages[stage - 1].feet[foot].planned_contact &&
                plan.stages[stage].feet[foot].planned_contact) {
                PlannedTouchdown touchdown;
                touchdown.foot = foot;
                touchdown.event = event_count[foot]++;
                touchdown.time = plan.stages[stage].time;
                touchdowns.push_back(touchdown);
            }
        }
    }
    return touchdowns;
}

template <typename Sample>
int plan_sample_contact_mask(const Sample& sample) {
    int contact_mask = 0;
    for (int foot = 0; foot < kNumFeet; ++foot) {
        if (sample.feet[foot].planned_contact)
            contact_mask |= 1 << foot;
    }
    return contact_mask;
}

template <typename Sample>
void print_plan_transition(const Sample& sample, int segment_index,
                           int stage, int contact_mask) {
    Vec<4> identity_orientation;
    identity_orientation.zero();
    identity_orientation[0] = 1.0;
    std::printf(
        "  plan transition: segment=%d stage=%d time=%.3f mask=0x%x "
        "base=(%.4f,%.4f,%.4f) attitude_error=%.4f "
        "normal_forces=(%.2f,%.2f,%.2f,%.2f)\n",
        segment_index, stage, sample.time, contact_mask,
        sample.base.position_world[0], sample.base.position_world[1],
        sample.base.position_world[2],
        orientation_error_world(
            identity_orientation,
            sample.base.orientation_body_to_world).norm2(),
        sample.feet[0].normal_force, sample.feet[1].normal_force,
        sample.feet[2].normal_force, sample.feet[3].normal_force);
}

template <int Horizon>
void print_plan_transitions(const ContactPlan<Horizon>& plan,
                            int segment_index) {
    int previous_mask = -1;
    for (int stage = 0; stage < Horizon; ++stage) {
        const int contact_mask = plan_sample_contact_mask(plan.stages[stage]);
        if (stage == 0 || contact_mask != previous_mask)
            print_plan_transition(plan.stages[stage], segment_index, stage,
                                  contact_mask);
        previous_mask = contact_mask;
    }
}

Vec<kStateDim> planner_state_from_whole_body(
    const WholeBodyState& state, const SharedTerrain& terrain,
    double* maximum_foot_height_correction = nullptr,
    const MujocoFootContact* contacts = nullptr) {
    Vec<kStateDim> planner_state;
    planner_state.zero();
    double maximum_correction = 0.0;
    const Go1FootCenterTerrain<SharedTerrain> planner_terrain(terrain);
    for (int axis = 0; axis < 3; ++axis) {
        planner_state[StateIndex::base_position(axis)] =
            state.base.position_world[axis];
        planner_state[StateIndex::linear_velocity(axis)] =
            state.base.linear_velocity_world[axis];
        planner_state[StateIndex::angular_velocity(axis)] =
            state.base.angular_velocity_body[axis];
        for (int foot = 0; foot < kNumFeet; ++foot) {
            planner_state[StateIndex::foot_position(foot, axis)] =
                state.foot_positions_world[foot][axis];
        }
    }
    for (int foot = 0; foot < kNumFeet; ++foot) {
        const bool confirmed_contact = !contacts ||
            (contacts[foot].in_contact &&
             contacts[foot].normal_force >= kContactConfirmationForce);
        if (!confirmed_contact) continue;
        const double rigid_height = planner_terrain.height(
            state.foot_positions_world[foot][0],
            state.foot_positions_world[foot][1]);
        maximum_correction = std::max(
            maximum_correction,
            std::fabs(rigid_height - state.foot_positions_world[foot][2]));
        planner_state[StateIndex::foot_position(foot, 2)] = rigid_height;
    }
    for (int element = 0; element < 4; ++element) {
        planner_state[StateIndex::quaternion(element)] =
            state.base.orientation_body_to_world[element];
    }
    if (maximum_foot_height_correction)
        *maximum_foot_height_correction = maximum_correction;
    return planner_state;
}

template <int Horizon>
int execute_contact_plan_segment(const mjModel* model, mjData* data,
                                 const MujocoGo1Adapter& adapter,
                                 const SharedTerrain& terrain,
                                 const ReplayOptions& options,
                                 const ContactPlan<Horizon>& plan,
                                 bool initialize_from_plan,
                                 int segment_index,
                                 ReplaySegmentReport* output_report,
                                 const ConvexWBCParameters* wbc_parameters =
                                     nullptr,
                                 const ContactExecutionFeedbackParameters*
                                     feedback_parameters_override = nullptr) {
    if (initialize_from_plan) {
        MujocoInitializationReport initialization;
        if (adapter.initialize_state_from_plan(
                plan.stages[0], &initialization) != Status::SUCCESS) {
            std::printf(
                "failed to initialize Go1 from native ContactIPM plan\n");
            return 1;
        }
        std::printf(
            "native plan initialization: iterations=%d base_error=%.3e "
            "foot_error=%.3e\n",
            initialization.iterations, initialization.maximum_base_error,
            initialization.maximum_foot_error);
    }
    WholeBodyState state;
    if (adapter.read_whole_body_state(state) != Status::SUCCESS) return 1;
    if (!initialize_from_plan) {
        double maximum_base_error = 0.0;
        double maximum_foot_error = 0.0;
        for (int axis = 0; axis < 3; ++axis) {
            maximum_base_error = std::max(
                maximum_base_error,
                std::fabs(state.base.position_world[axis] -
                          plan.stages[0].base.position_world[axis]));
            for (int foot = 0; foot < kNumFeet; ++foot) {
                maximum_foot_error = std::max(
                    maximum_foot_error,
                    std::fabs(state.foot_positions_world[foot][axis] -
                              plan.stages[0].feet[foot].position_world[axis]));
            }
        }
        std::printf(
            "continuous plan handoff: segment=%d base_error=%.3e "
            "foot_error=%.3e\n",
            segment_index, maximum_base_error, maximum_foot_error);
    }

    const ConvexWBCParameters active_wbc_parameters = wbc_parameters
        ? *wbc_parameters
        : go1_wbc_parameters();
    ConvexWholeBodyController controller(active_wbc_parameters);
    ContactExecutionFeedbackParameters feedback_parameters =
        feedback_parameters_override
        ? *feedback_parameters_override
        : go1_contact_feedback_parameters(active_wbc_parameters);
    ConvexWBCCommand command;
    MujocoActuatorReport reports[kNumFeet][kGo1JointsPerLeg];
    MujocoFootContact contacts[kNumFeet];
    Vec<3> contact_sample_foot_positions[kNumFeet];
    const double start_time = data->time;
    double contact_sample_plan_time = plan.stages[0].time;
    const double duration = plan.terminal.time - plan.stages[0].time;
    if (!(duration > 0.0) || !(model->opt.timestep > 0.0)) return 1;
    std::printf(
        "ContactIPM execution policy: version=%d synchronized_contacts=1 "
        "history_aware_recontact=1 measured_load_torque_handoff=1\n",
        kContactExecutionPolicyVersion);
    const int ticks = static_cast<int>(
        std::ceil(duration / model->opt.timestep));
    const std::vector<PlannedTouchdown> touchdown_windows =
        planned_touchdowns(plan);

    FootReplayMetrics foot_metrics[kNumFeet];
    bool has_planned_motion[kNumFeet] = {};
    if (adapter.read_foot_contacts(contacts) != Status::SUCCESS) return 1;
    for (int foot = 0; foot < kNumFeet; ++foot) {
        contact_sample_foot_positions[foot] =
            state.foot_positions_world[foot];
        for (int stage = 0; stage < Horizon; ++stage) {
            double displacement_sq = 0.0;
            for (int axis = 0; axis < 3; ++axis) {
                const double displacement =
                    plan.stages[stage].feet[foot].position_world[axis] -
                    plan.stages[0].feet[foot].position_world[axis];
                displacement_sq += displacement * displacement;
            }
            if (displacement_sq > 0.02 * 0.02)
                has_planned_motion[foot] = true;
        }
        foot_metrics[foot].initialize(
            plan.stages[0].feet[foot].planned_contact,
            contacts[foot].in_contact &&
                contacts[foot].normal_force >= kTouchdownDetectionForce);
    }
    StanceContactConfirmation contact_confirmation[kNumFeet];
    ContactTorqueBlendHandoff contact_handoff[kNumFeet];
    bool previous_contact_geometry[kNumFeet];
    int late_touchdown_search_ticks[kNumFeet] = {};
    EarlyContactFeedback early_contact_feedback[kNumFeet];
    bool early_contact_support[kNumFeet] = {};
    int early_contact_activations = 0;
    int late_touchdown_search_total_ticks = 0;
    double squared_base_position_error = 0.0;
    int saturated_joint_ticks = 0;
    bool divergence_reported = false;
    const Go1SRBDCalibration calibration;
    for (int foot = 0; foot < kNumFeet; ++foot) {
        contact_confirmation[foot].initialize(
            plan.stages[0].feet[foot].planned_contact,
            contacts[foot].in_contact &&
                contacts[foot].normal_force >= kContactConfirmationForce);
        previous_contact_geometry[foot] = contacts[foot].in_contact;
    }
    const int push_body = mj_name2id(model, mjOBJ_BODY, "trunk");
    if (push_body < 0) return 1;
    struct TouchdownControlTrace {
        bool active = false;
        int foot = -1;
        int event = -1;
        double plan_time = 0.0;
        double relative_time = 0.0;
        int plan_mask = 0;
        int support_mask = 0;
        int measured_mask = 0;
        Vec<3> foot_position_error;
        Vec<3> foot_velocity_error;
        Vec<3> base_position_error;
        Vec<3> base_velocity_error;
        Vec<3> orientation_error;
        Vec<3> angular_velocity_error;
        double jacobian_condition = 0.0;
    } control_traces[kNumFeet];
    for (int tick = 0; tick < ticks; ++tick) {
        mj_step1(model, data);
        const double plan_time = data->time - start_time + plan.stages[0].time;
        const double elapsed = data->time - start_time;
        const bool push_active = elapsed >= options.push_start &&
            elapsed < options.push_start + options.push_duration;
        for (int axis = 0; axis < 3; ++axis) {
            data->xfrc_applied[6 * push_body + axis] =
                push_active ? options.push_force_world[axis] : 0.0;
        }
        ContactPlanSample plan_reference;
        if (sample_contact_plan(plan, plan_time, plan_reference) !=
                PlanSampleStatus::ACTIVE ||
            adapter.read_whole_body_state(state) != Status::SUCCESS) {
            std::printf("ContactIPM replay controller failed at tick %d\n", tick);
            return 1;
        }
        shape_swing_references(plan, terrain, plan_reference);
        ContactPlanSample control_reference = plan_reference;
        bool newly_confirmed_contact[kNumFeet] = {};

        for (int foot = 0; foot < kNumFeet; ++foot) {
            FootReplayMetrics& metrics = foot_metrics[foot];
            const bool previously_confirmed =
                contact_confirmation[foot].confirmed();
            const bool current_planned_contact =
                plan_reference.feet[foot].planned_contact;
            const bool current_measured_contact =
                contacts[foot].in_contact &&
                contacts[foot].normal_force >= kTouchdownDetectionForce;
            const bool current_load_bearing_contact =
                contacts[foot].in_contact &&
                contacts[foot].normal_force >= kContactConfirmationForce;
            const bool contact_confirmed = contact_confirmation[foot].update(
                current_planned_contact, contacts[foot].in_contact,
                current_load_bearing_contact, kContactLossGraceTicks);
            if (!current_planned_contact) {
                late_touchdown_search_ticks[foot] = 0;
            }
            newly_confirmed_contact[foot] =
                !previously_confirmed && contact_confirmed;
            if (options.trace_execution &&
                (previously_confirmed != contact_confirmed ||
                 previous_contact_geometry[foot] != contacts[foot].in_contact)) {
                std::printf(
                    "contact state trace: segment=%d foot=%d time=%.6f "
                    "planned=%d geometry=%d force=%.6f confirmed=%d "
                    "established=%d\n",
                    segment_index, foot, plan_time,
                    current_planned_contact ? 1 : 0,
                    contacts[foot].in_contact ? 1 : 0,
                    contacts[foot].normal_force,
                    contact_confirmed ? 1 : 0,
                    contact_confirmation[foot].established() ? 1 : 0);
            }
            previous_contact_geometry[foot] = contacts[foot].in_contact;
            const bool scheduled_touchdown = has_planned_motion[foot] &&
                !metrics.previous_planned_contact && current_planned_contact;
            const bool measured_touchdown = has_planned_motion[foot] &&
                metrics.active_event >= 0 &&
                !metrics.previous_measured_contact &&
                current_measured_contact &&
                metrics.events[metrics.active_event].measured_touchdown_time <
                    0.0 &&
                metrics.events[metrics.active_event].maximum_clearance >=
                    kMinimumSwingClearance;
            const Vec<3>& position = state.foot_positions_world[foot];
            const double clearance = position[2] -
                terrain.height(position[0], position[1]) -
                calibration.foot_center_contact_offset;
            if (has_planned_motion[foot]) {
                metrics.observe(plan_time, contact_sample_plan_time,
                                current_planned_contact,
                                current_measured_contact,
                                plan_reference.feet[foot].position_world,
                                contact_sample_foot_positions[foot],
                                clearance, kMinimumSwingClearance);
            }
            if (scheduled_touchdown) {
                double position_error_sq = 0.0;
                for (int axis = 0; axis < 3; ++axis) {
                    const double error = position[axis] -
                        plan_reference.feet[foot].position_world[axis];
                    position_error_sq += error * error;
                }
                std::printf(
                    "  foot %d scheduled touchdown: time=%.6f gap=%.6f "
                    "position_error=%.6f contact=%d normal_force=%.6f\n",
                    foot, plan_time, clearance,
                    std::sqrt(position_error_sq),
                    contacts[foot].in_contact ? 1 : 0,
                    contacts[foot].normal_force);
            }
            if (measured_touchdown) {
                std::printf(
                    "  foot %d measured touchdown: time=%.6f "
                    "observed_time=%.6f sample_age=%.6f "
                    "normal_force=%.6f\n",
                    foot, contact_sample_plan_time, plan_time,
                    plan_time - contact_sample_plan_time,
                    contacts[foot].normal_force);
            }

            const bool swing_clearance_reached =
                metrics.active_event >= 0 &&
                metrics.events[metrics.active_event].maximum_clearance >=
                    kMinimumSwingClearance;
            const bool previous_early_support =
                early_contact_support[foot];
            early_contact_support[foot] = options.early_contact_feedback &&
                early_contact_feedback[foot].update(
                    current_planned_contact, current_load_bearing_contact,
                    swing_clearance_reached, feedback_parameters);
            if (early_contact_support[foot]) {
                apply_early_contact_support(
                    position, feedback_parameters,
                    control_reference.feet[foot]);
            }
            if (!previous_early_support && early_contact_support[foot]) {
                ++early_contact_activations;
                std::printf(
                    "contact feedback: segment=%d foot=%d time=%.6f "
                    "event=early_touchdown_support normal_force=%.6f\n",
                    segment_index, foot, plan_time,
                    contacts[foot].normal_force);
            } else if (previous_early_support &&
                       !early_contact_support[foot] &&
                       !current_planned_contact) {
                std::printf(
                    "contact feedback: segment=%d foot=%d time=%.6f "
                    "event=early_touchdown_release\n",
                    segment_index, foot, plan_time);
            }
            if (current_planned_contact && !contact_confirmed) {
                ++late_touchdown_search_ticks[foot];
                if (options.late_touchdown_search) {
                    ++late_touchdown_search_total_ticks;
                    apply_late_touchdown_search(
                        late_touchdown_search_ticks[foot],
                        model->opt.timestep, feedback_parameters,
                        control_reference.feet[foot]);
                } else {
                    control_reference.feet[foot].planned_contact = false;
                    control_reference.feet[foot].normal_force = 0.0;
                    control_reference.feet[foot].force_world.zero();
                }
            } else if (current_planned_contact) {
                late_touchdown_search_ticks[foot] = 0;
            }
        }
        if (options.force_reference == ForceReferenceMode::UNIFORM) {
            int active_feet = 0;
            for (int foot = 0; foot < kNumFeet; ++foot)
                if (control_reference.feet[foot].planned_contact) ++active_feet;
            if (active_feet == 0) return 1;
            const RobotParameters parameters = go1_robot_parameters();
            const double vertical_force =
                parameters.mass * parameters.gravity / active_feet;
            for (int foot = 0; foot < kNumFeet; ++foot) {
                control_reference.feet[foot].force_world.zero();
                control_reference.feet[foot].normal_force = 0.0;
                if (control_reference.feet[foot].planned_contact) {
                    control_reference.feet[foot].force_world[2] =
                        vertical_force;
                    control_reference.feet[foot].normal_force = dot3(
                        control_reference.feet[foot].terrain_normal_world,
                        control_reference.feet[foot].force_world);
                }
            }
        }
        for (int foot = 0; foot < kNumFeet; ++foot) {
            contact_handoff[foot].update(
                plan_reference.feet[foot].planned_contact,
                newly_confirmed_contact[foot],
                contacts[foot].normal_force,
                control_reference.feet[foot]);
        }
        if (controller.compute(control_reference, state, command) !=
                Status::SUCCESS ||
            adapter.apply_command(command, reports) != Status::SUCCESS) {
            std::printf("ContactIPM replay controller failed at tick %d\n", tick);
            return 1;
        }
        double base_error_sq = 0.0;
        for (int axis = 0; axis < 3; ++axis) {
            const double error = state.base.position_world[axis] -
                                 plan_reference.base.position_world[axis];
            base_error_sq += error * error;
        }
        if (options.trace_execution) {
            for (TouchdownControlTrace& trace : control_traces)
                trace.active = false;
            int plan_mask = 0;
            int support_mask = 0;
            int measured_mask = 0;
            for (int foot = 0; foot < kNumFeet; ++foot) {
                if (plan_reference.feet[foot].planned_contact)
                    plan_mask |= 1 << foot;
                if (control_reference.feet[foot].planned_contact)
                    support_mask |= 1 << foot;
                if (contacts[foot].in_contact)
                    measured_mask |= 1 << foot;
            }
            const Go1FootCenterTerrain<SharedTerrain> planner_terrain(terrain);
            for (const PlannedTouchdown& touchdown : touchdown_windows) {
                const double relative_time = plan_time - touchdown.time;
                if (std::fabs(relative_time) > kTouchdownTraceHalfWindow)
                    continue;
                const int foot = touchdown.foot;
                const Vec<3>& measured = state.foot_positions_world[foot];
                const Vec<3>& desired =
                    plan_reference.feet[foot].position_world;
                const double dx = measured[0] - desired[0];
                const double dy = measured[1] - desired[1];
                const double dz = measured[2] - desired[2];
                std::printf(
                    "touchdown trace: segment=%d tick=%d foot=%d event=%d "
                    "time=%.6f relative_time=%.6f "
                    "plan_mask=0x%x support_mask=0x%x measured_mask=0x%x "
                    "plan_contact=%d support_contact=%d measured_contact=%d "
                    "desired_gap=%.6f measured_gap=%.6f "
                    "desired_vz=%.6f measured_vz=%.6f "
                    "normal_force=%.6f contact_blend=%.6f "
                    "foot_position_error=%.6f wrench_residual=%.6f\n",
                    segment_index, tick, foot, touchdown.event,
                    plan_time, relative_time, plan_mask, support_mask,
                    measured_mask,
                    plan_reference.feet[foot].planned_contact ? 1 : 0,
                    control_reference.feet[foot].planned_contact ? 1 : 0,
                    contacts[foot].in_contact ? 1 : 0,
                    desired[2] - planner_terrain.height(desired[0], desired[1]),
                    measured[2] - planner_terrain.height(
                        measured[0], measured[1]),
                    plan_reference.feet[foot].velocity_world[2],
                    state.foot_velocities_world[foot][2],
                    contacts[foot].normal_force,
                    command.contact_blend[foot],
                    std::sqrt(dx * dx + dy * dy + dz * dz),
                    command.wrench_residual);
                TouchdownControlTrace& trace = control_traces[foot];
                trace.active = true;
                trace.foot = foot;
                trace.event = touchdown.event;
                trace.plan_time = plan_time;
                trace.relative_time = relative_time;
                trace.plan_mask = plan_mask;
                trace.support_mask = support_mask;
                trace.measured_mask = measured_mask;
                for (int axis = 0; axis < 3; ++axis) {
                    trace.foot_position_error[axis] =
                        state.foot_positions_world[foot][axis] -
                        plan_reference.feet[foot].position_world[axis];
                    trace.foot_velocity_error[axis] =
                        state.foot_velocities_world[foot][axis] -
                        plan_reference.feet[foot].velocity_world[axis];
                    trace.base_position_error[axis] =
                        state.base.position_world[axis] -
                        plan_reference.base.position_world[axis];
                    trace.base_velocity_error[axis] =
                        state.base.linear_velocity_world[axis] -
                        plan_reference.base.linear_velocity_world[axis];
                    trace.angular_velocity_error[axis] =
                        state.base.angular_velocity_body[axis] -
                        plan_reference.base.angular_velocity_body[axis];
                }
                trace.orientation_error = orientation_error_world(
                    state.base.orientation_body_to_world,
                    plan_reference.base.orientation_body_to_world);
                trace.jacobian_condition = frobenius_condition_number_3x3(
                    state.foot_jacobians_body[foot]);
            }
        }
        if (!divergence_reported &&
            std::fabs(state.base.position_world[1] -
                      plan_reference.base.position_world[1]) > 0.03) {
            int planned_mask = 0;
            int measured_mask = 0;
            for (int foot = 0; foot < kNumFeet; ++foot) {
                if (control_reference.feet[foot].planned_contact)
                    planned_mask |= 1 << foot;
                if (contacts[foot].in_contact)
                    measured_mask |= 1 << foot;
            }
            std::printf(
                "execution divergence: segment=%d time=%.3f "
                "planned_mask=0x%x measured_mask=0x%x "
                "base=(%.4f,%.4f,%.4f) reference=(%.4f,%.4f,%.4f) "
                "attitude_error=%.4f wrench_residual=%.4f\n",
                segment_index, plan_time, planned_mask, measured_mask,
                state.base.position_world[0], state.base.position_world[1],
                state.base.position_world[2],
                plan_reference.base.position_world[0],
                plan_reference.base.position_world[1],
                plan_reference.base.position_world[2],
                orientation_error_world(
                    plan_reference.base.orientation_body_to_world,
                    state.base.orientation_body_to_world).norm2(),
                command.wrench_residual);
            divergence_reported = true;
        }
        squared_base_position_error += base_error_sq;
        mj_step2(model, data);
        // mj_contactForce reads efc_force, which is solved in mj_step2.  Cache
        // the synchronized result before the next mj_step1 rebuilds contacts.
        if (adapter.read_foot_contacts(contacts) != Status::SUCCESS ||
            adapter.update_applied_torques(reports) != Status::SUCCESS)
            return 1;
        for (int foot = 0; foot < kNumFeet; ++foot) {
            contact_sample_foot_positions[foot] =
                state.foot_positions_world[foot];
        }
        contact_sample_plan_time = plan_time;
        if (options.trace_execution) {
            for (const TouchdownControlTrace& trace : control_traces) {
                if (!trace.active) continue;
                const int foot = trace.foot;
                int saturated_mask = 0;
                for (int joint = 0; joint < kGo1JointsPerLeg; ++joint) {
                    if (reports[foot][joint].saturated)
                        saturated_mask |= 1 << joint;
                }
                std::printf(
                    "touchdown control trace: segment=%d tick=%d foot=%d "
                    "event=%d time=%.6f relative_time=%.6f "
                    "plan_mask=0x%x support_mask=0x%x measured_mask=0x%x "
                    "foot_position_error=(%.6f,%.6f,%.6f) "
                    "foot_velocity_error=(%.6f,%.6f,%.6f) "
                    "joint_position=(%.6f,%.6f,%.6f) "
                    "joint_velocity=(%.6f,%.6f,%.6f) "
                    "requested_torque=(%.6f,%.6f,%.6f) "
                    "predicted_torque=(%.6f,%.6f,%.6f) "
                    "applied_torque=(%.6f,%.6f,%.6f) "
                    "torque_headroom=(%.6f,%.6f,%.6f) "
                    "saturated_mask=0x%x swing_force=(%.6f,%.6f,%.6f) "
                    "jacobian_condition=%.6f "
                    "base_position_error=(%.6f,%.6f,%.6f) "
                    "base_velocity_error=(%.6f,%.6f,%.6f) "
                    "orientation_error_world=(%.6f,%.6f,%.6f) "
                    "angular_velocity_error_body=(%.6f,%.6f,%.6f) "
                    "wrench_residual=(%.6f,%.6f,%.6f,%.6f,%.6f,%.6f)\n",
                    segment_index, tick, foot, trace.event, trace.plan_time,
                    trace.relative_time, trace.plan_mask, trace.support_mask,
                    trace.measured_mask,
                    trace.foot_position_error[0],
                    trace.foot_position_error[1],
                    trace.foot_position_error[2],
                    trace.foot_velocity_error[0],
                    trace.foot_velocity_error[1],
                    trace.foot_velocity_error[2],
                    state.joint_positions[foot][0],
                    state.joint_positions[foot][1],
                    state.joint_positions[foot][2],
                    state.joint_velocities[foot][0],
                    state.joint_velocities[foot][1],
                    state.joint_velocities[foot][2],
                    reports[foot][0].requested_joint_torque,
                    reports[foot][1].requested_joint_torque,
                    reports[foot][2].requested_joint_torque,
                    reports[foot][0].predicted_joint_torque,
                    reports[foot][1].predicted_joint_torque,
                    reports[foot][2].predicted_joint_torque,
                    reports[foot][0].applied_joint_torque,
                    reports[foot][1].applied_joint_torque,
                    reports[foot][2].applied_joint_torque,
                    reports[foot][0].requested_torque_headroom,
                    reports[foot][1].requested_torque_headroom,
                    reports[foot][2].requested_torque_headroom,
                    saturated_mask,
                    command.swing_feedback_forces_world[foot][0],
                    command.swing_feedback_forces_world[foot][1],
                    command.swing_feedback_forces_world[foot][2],
                    trace.jacobian_condition,
                    trace.base_position_error[0],
                    trace.base_position_error[1],
                    trace.base_position_error[2],
                    trace.base_velocity_error[0],
                    trace.base_velocity_error[1],
                    trace.base_velocity_error[2],
                    trace.orientation_error[0],
                    trace.orientation_error[1],
                    trace.orientation_error[2],
                    trace.angular_velocity_error[0],
                    trace.angular_velocity_error[1],
                    trace.angular_velocity_error[2],
                    command.wrench_residual_components[0],
                    command.wrench_residual_components[1],
                    command.wrench_residual_components[2],
                    command.wrench_residual_components[3],
                    command.wrench_residual_components[4],
                    command.wrench_residual_components[5]);
            }
        }
        for (int foot = 0; foot < kNumFeet; ++foot) {
            for (int joint = 0; joint < kGo1JointsPerLeg; ++joint) {
                if (reports[foot][joint].saturated)
                    ++saturated_joint_ticks;
            }
        }
    }

    if (adapter.read_whole_body_state(state) != Status::SUCCESS) return 1;
    double maximum_plan_step = 0.0;
    for (int stage = 0; stage < Horizon; ++stage) {
        maximum_plan_step = std::max(maximum_plan_step,
                                     plan.stages[stage].duration);
    }
    const int touchdown_tolerance_nodes = segment_index == 0 ? 1 : 2;
    const double touchdown_tolerance = std::max(
        0.05, touchdown_tolerance_nodes * maximum_plan_step +
                  model->opt.timestep);
    const double base_position_rms = std::sqrt(
        squared_base_position_error / std::max(ticks, 1));
    const double saturation_fraction =
        static_cast<double>(saturated_joint_ticks) /
        std::max(ticks * kNumFeet * kGo1JointsPerLeg, 1);
    std::printf(
        "ContactIPM replay: segment=%d horizon=%d duration=%.3f "
        "force_reference=%s "
        "mass_scale=%.3f friction_scale=%.3f push=(%.3f,%.3f,%.3f) "
        "push_window=(%.3f,%.3f) "
        "touchdown_tolerance=%.6f touchdown_tolerance_nodes=%d "
        "base_position_rms=%.6f "
        "saturation_fraction=%.6f early_contact_activations=%d "
        "late_touchdown_search_ticks=%d "
        "final_base=(%.6f, %.6f, %.6f)\n",
        segment_index, Horizon, duration,
        force_reference_name(options.force_reference),
        options.mass_scale, options.friction_scale,
        options.push_force_world[0], options.push_force_world[1],
        options.push_force_world[2], options.push_start,
        options.push_duration, touchdown_tolerance,
        touchdown_tolerance_nodes, base_position_rms,
        saturation_fraction, early_contact_activations,
        late_touchdown_search_total_ticks, state.base.position_world[0],
        state.base.position_world[1], state.base.position_world[2]);
    int moving_feet = 0;
    int swing_events = 0;
    bool feet_task_accepted = true;
    bool feet_timing_accepted = true;
    for (int foot = 0; foot < kNumFeet; ++foot) {
        const FootReplayMetrics& metrics = foot_metrics[foot];
        if (metrics.events.empty()) continue;
        ++moving_feet;
        bool foot_unloaded = true;
        bool foot_task_accepted = true;
        bool foot_timing_accepted = true;
        double minimum_clearance = 1e300;
        double maximum_landing_error = 0.0;
        double maximum_touchdown_error = 0.0;
        double maximum_slip = 0.0;
        for (std::size_t event_index = 0;
             event_index < metrics.events.size(); ++event_index) {
            const SwingReplayEvent& event = metrics.events[event_index];
            const double landing_error = event.landing_error();
            const double touchdown_error = event.touchdown_error();
            const bool task_accepted = event.measured_unload() &&
                event.maximum_clearance >= kMinimumSwingClearance &&
                landing_error <= 0.03 &&
                event.maximum_post_touchdown_slip <= 0.02;
            const bool timing_accepted =
                touchdown_error <= touchdown_tolerance + 1e-9;
            const bool accepted = task_accepted && timing_accepted;
            foot_unloaded = foot_unloaded && event.measured_unload();
            foot_task_accepted = foot_task_accepted && task_accepted;
            foot_timing_accepted = foot_timing_accepted && timing_accepted;
            minimum_clearance = std::min(minimum_clearance,
                                         event.maximum_clearance);
            maximum_landing_error = std::max(maximum_landing_error,
                                              landing_error);
            maximum_touchdown_error = std::max(maximum_touchdown_error,
                                                touchdown_error);
            maximum_slip = std::max(maximum_slip,
                                    event.maximum_post_touchdown_slip);
            std::printf(
                "  event segment=%d foot=%d index=%zu: unload=%d clearance=%.6f "
                "landing_error=%.6f planned_liftoff=%.6f "
                "measured_liftoff=%.6f planned_touchdown=%.6f "
                "measured_touchdown=%.6f touchdown_error=%.6f "
                "post_touchdown_slip=%.6f accepted=%d task_accepted=%d "
                "timing_accepted=%d signed_touchdown_error=%.6f\n",
                segment_index, foot, event_index,
                event.measured_unload() ? 1 : 0,
                event.maximum_clearance, landing_error,
                event.planned_liftoff_time, event.measured_liftoff_time,
                event.planned_touchdown_time, event.measured_touchdown_time,
                touchdown_error, event.maximum_post_touchdown_slip,
                accepted ? 1 : 0, task_accepted ? 1 : 0,
                timing_accepted ? 1 : 0, event.signed_touchdown_error());
        }
        swing_events += static_cast<int>(metrics.events.size());
        feet_task_accepted = feet_task_accepted && foot_task_accepted;
        feet_timing_accepted = feet_timing_accepted && foot_timing_accepted;
        const SwingReplayEvent& last_event = metrics.events.back();
        std::printf(
            "  foot %d: events=%zu unload=%d clearance=%.6f "
            "landing_error=%.6f "
            "planned_touchdown=%.6f measured_touchdown=%.6f "
            "touchdown_error=%.6f post_touchdown_slip=%.6f accepted=%d "
            "task_accepted=%d timing_accepted=%d\n",
            foot, metrics.events.size(), foot_unloaded ? 1 : 0,
            minimum_clearance, maximum_landing_error,
            last_event.planned_touchdown_time,
            last_event.measured_touchdown_time, maximum_touchdown_error,
            maximum_slip,
            foot_task_accepted && foot_timing_accepted ? 1 : 0,
            foot_task_accepted ? 1 : 0, foot_timing_accepted ? 1 : 0);
    }
    const bool base_accepted = state.base.position_world[2] >=
        plan.terminal.base.position_world[2] - 0.05;
    const bool task_success = moving_feet > 0 && feet_task_accepted &&
        base_accepted;
    const bool timing_success = moving_feet > 0 && feet_timing_accepted;
    std::printf(
        "ContactIPM replay gates: segment=%d moving_feet=%d swing_events=%d "
        "task_success=%d timing_success=%d\n",
        segment_index, moving_feet, swing_events, task_success ? 1 : 0,
        timing_success ? 1 : 0);
    if (output_report) {
        output_report->moving_feet = moving_feet;
        output_report->swing_events = swing_events;
        output_report->task_success = task_success;
        output_report->timing_success = timing_success;
    }
    if (!task_success || (options.require_timing && !timing_success)) {
        std::printf("ContactIPM replay acceptance gate failed\n");
        return 1;
    }
    return 0;
}

template <int Horizon>
int run_contact_plan_replay(const mjModel* model, mjData* data,
                            const MujocoGo1Adapter& adapter,
                            const SharedTerrain& terrain,
                            const ReplayOptions& options,
                            const char* plan_path) {
    ContactPlan<Horizon> plan;
    if (read_contact_plan(plan_path, plan) != Status::SUCCESS) {
        std::printf("failed to read %d-stage ContactIPM plan: %s\n",
                    Horizon, plan_path);
        return 1;
    }
    return execute_contact_plan_segment(
        model, data, adapter, terrain, options, plan, true, 0, nullptr);
}

int run_snapshot_replay(const mjModel* model, mjData* data,
                         const MujocoGo1Adapter& adapter,
                         const MujocoReplaySnapshot& snapshot,
                         const ReplayOptions& options,
                         const char* plan_path,
                         bool base_z_kp_override_set,
                         double base_z_kp_override) {
    ContactPlan<kFourStepHorizon> plan;
    if (read_contact_plan(plan_path, plan) != Status::SUCCESS) {
        std::printf("failed to read snapshot ContactIPM plan: %s\n",
                    plan_path);
        return 1;
    }
    const std::uint64_t plan_fingerprint = contact_plan_fingerprint(plan);
    if (plan_fingerprint != snapshot.paired_plan_fingerprint) {
        std::printf(
            "snapshot plan fingerprint mismatch: expected=%llu actual=%llu\n",
            static_cast<unsigned long long>(
                snapshot.paired_plan_fingerprint),
            static_cast<unsigned long long>(plan_fingerprint));
        return 1;
    }
    const MujocoReplaySnapshotConfiguration& stored_configuration =
        snapshot.configuration;
    const bool legacy_policy =
        stored_configuration.contact_execution_policy_version == 0;
    if (!legacy_policy &&
        (stored_configuration.contact_execution_policy_version !=
             kContactExecutionPolicyVersion ||
         !stored_configuration.synchronized_contact_cache ||
         !stored_configuration.history_aware_recontact ||
         !stored_configuration.measured_load_torque_handoff)) {
        std::printf(
            "snapshot execution policy mismatch: stored_version=%d "
            "effective_version=%d\n",
            stored_configuration.contact_execution_policy_version,
            kContactExecutionPolicyVersion);
        return 1;
    }
    MujocoReplaySnapshotRestoreReport restore_report;
    if (restore_mujoco_replay_snapshot(
            snapshot, model, data, adapter, &restore_report,
            kSnapshotContactForceTolerance) !=
        Status::SUCCESS) {
        std::printf(
            "failed to restore MuJoCo boundary snapshot: "
            "contact_mask_matches=%d max_contact_force_error=%.3e\n",
            restore_report.contact_mask_matches ? 1 : 0,
            restore_report.maximum_contact_force_error);
        return 1;
    }
    std::printf(
        "restored MuJoCo boundary snapshot: segment=%d sim_time=%.6f "
        "contacts_match=%d max_contact_force_error=%.3e\n",
        snapshot.segment, data->time,
        restore_report.contact_mask_matches ? 1 : 0,
        restore_report.maximum_contact_force_error);
    ConvexWBCParameters wbc_parameters =
        snapshot.configuration.wbc_parameters;
    const double stored_base_z_kp = wbc_parameters.base_position_kp[2];
    if (base_z_kp_override_set)
        wbc_parameters.base_position_kp[2] = base_z_kp_override;
    std::printf(
        "snapshot WBC parameters: stored_base_z_kp=%.6f "
        "effective_base_z_kp=%.6f override=%d\n",
        stored_base_z_kp, wbc_parameters.base_position_kp[2],
        base_z_kp_override_set ? 1 : 0);
    std::printf(
        "snapshot execution policy: stored_version=%d effective_version=%d "
        "legacy=%d\n",
        snapshot.configuration.contact_execution_policy_version,
        kContactExecutionPolicyVersion, legacy_policy ? 1 : 0);
    if (legacy_policy) {
        std::printf(
            "snapshot execution policy warning: legacy snapshot has no "
            "controller policy identity; replay uses effective version=%d\n",
            kContactExecutionPolicyVersion);
    }
    return execute_contact_plan_segment(
        model, data, adapter, snapshot.configuration.terrain, options, plan,
        false, snapshot.segment, nullptr,
        &wbc_parameters,
        &snapshot.configuration.contact_feedback_parameters);
}

int run_replanned_course(const mjModel* model, mjData* data,
                         const MujocoGo1Adapter& adapter,
                         const SharedTerrain& terrain,
                         const ReplayOptions& options,
                         int replan_count,
                         ReplanSeedPolicy seed_policy,
                         unsigned int terrain_seed,
                         double terrain_amplitude) {
    Vec<3> initial_base_position;
    int total_swing_events = 0;
    int task_successful_segments = 0;
    int timing_successful_segments = 0;
    for (int segment = 0; segment < replan_count; ++segment) {
        if (segment > 0 && stabilize_for_replanning(
                model, data, adapter, terrain,
                options.stabilization_duration,
                options.maximum_stabilization_duration,
                options.confirmed_stabilization) != 0) {
            return 1;
        }
        WholeBodyState state;
        MujocoFootContact contacts[kNumFeet];
        if (adapter.read_whole_body_state(state) != Status::SUCCESS ||
            adapter.read_foot_contacts(contacts) != Status::SUCCESS) {
            return 1;
        }
        int contact_count = 0;
        for (int foot = 0; foot < kNumFeet; ++foot) {
            if (contacts[foot].in_contact) ++contact_count;
        }
        std::printf(
            "replanning boundary: segment=%d sim_time=%.3f contacts=%d "
            "linear_speed=%.6f angular_speed=%.6f "
            "base=(%.6f, %.6f, %.6f)\n",
            segment, data->time, contact_count,
            state.base.linear_velocity_world.norm2(),
            state.base.angular_velocity_body.norm2(),
            state.base.position_world[0], state.base.position_world[1],
            state.base.position_world[2]);
        for (int foot = 0; foot < kNumFeet; ++foot) {
            std::printf(
                "  replanning foot: segment=%d foot=%d "
                "position=(%.6f, %.6f, %.6f)\n",
                segment, foot, state.foot_positions_world[foot][0],
                state.foot_positions_world[foot][1],
                state.foot_positions_world[foot][2]);
        }

        ContactPlan<kFourStepHorizon> plan;
        FourStepPlannerReport planner_report;
        Vec<kStateDim> planner_state;
        if (segment == 0) {
            const Go1FootCenterTerrain<SharedTerrain> planner_terrain(terrain);
            planner_state = go1_home_state(planner_terrain);
            planner_state[StateIndex::base_position(0)] = 0.04;
            planner_state[StateIndex::base_position(1)] = 0.0;
            for (int axis = 0; axis < 3; ++axis) {
                initial_base_position[axis] =
                    planner_state[StateIndex::base_position(axis)];
            }
        } else {
            double maximum_foot_height_correction = 0.0;
            planner_state = planner_state_from_whole_body(
                state, terrain, &maximum_foot_height_correction);
            std::printf(
                "replanning state abstraction: segment=%d "
                "max_foot_height_correction=%.6f\n",
                segment, maximum_foot_height_correction);
        }
        FourStepMultiStartReport multistart_report;
        const FourStepSeed seed = segment_seed(seed_policy, segment);
        const bool use_multistart =
            seed_policy == ReplanSeedPolicy::MULTISTART;
        const nmpc::Status planner_status = use_multistart
            ? plan_go1_four_step_multistart(
                  planner_state, terrain, plan, &planner_report,
                  &multistart_report)
            : plan_go1_four_step(
                  planner_state, terrain, seed, plan, &planner_report);
        if (use_multistart) {
            for (int candidate = 0; candidate < kFourStepSeedCount;
                 ++candidate) {
                const FourStepPlannerReport& candidate_report =
                    multistart_report.candidates[candidate];
                std::printf(
                    "  replanning candidate: segment=%d seed=%s status=%s "
                    "solver_status=%s audit_passed=%d schedule_passed=%d "
                    "objective=%.6f iterations=%d solve_ms=%.3f "
                    "fingerprint=%llu events=(%d,%d,%d,%d) "
                    "schedule_deformation=%d\n",
                    segment, four_step_seed_name(candidate_report.seed),
                    nmpc::status_string(candidate_report.status),
                    nmpc::status_string(candidate_report.solver_status),
                    candidate_report.audit_passed ? 1 : 0,
                    candidate_report.schedule_audit_passed ? 1 : 0,
                    candidate_report.objective, candidate_report.iterations,
                    candidate_report.solve_ms,
                    static_cast<unsigned long long>(
                        candidate_report.optimized_schedule_fingerprint),
                    candidate_report.contact_events[0],
                    candidate_report.contact_events[1],
                    candidate_report.contact_events[2],
                    candidate_report.contact_events[3],
                    four_step_detail::schedule_deformation(candidate_report));
            }
        }
        std::printf(
            "replanning solve: segment=%d seed_policy=%s seed=%s status=%s "
            "solver_status=%s audit_passed=%d schedule_passed=%d iterations=%d "
            "solve_ms=%.3f objective=%.6f seed_dynamics=%.3e dynamics=%.3e "
            "inequality=%.3e mpcc=%.3e base_displacement=%.3f "
            "seed_fingerprint=%llu optimized_fingerprint=%llu "
            "seed_worst_stage=%d seed_excluded_foot=%d "
            "seed_min_weight=%.6f seed_support_projections=%d "
            "schedule_deformation=%d\n",
            segment, replan_seed_policy_name(seed_policy),
            four_step_seed_name(planner_report.seed),
            nmpc::status_string(planner_status),
            nmpc::status_string(planner_report.solver_status),
            planner_report.audit_passed ? 1 : 0,
            planner_report.schedule_audit_passed ? 1 : 0,
            planner_report.iterations, planner_report.solve_ms,
            planner_report.objective,
            planner_report.seed_dynamics, planner_report.dynamics,
            planner_report.inequality, planner_report.mpcc,
            planner_report.base_displacement,
            static_cast<unsigned long long>(
                planner_report.seed_schedule_fingerprint),
            static_cast<unsigned long long>(
                planner_report.optimized_schedule_fingerprint),
            planner_report.seed_worst_support_stage,
            planner_report.seed_worst_excluded_foot,
            planner_report.seed_minimum_support_weight,
            planner_report.seed_support_projections,
            four_step_detail::schedule_deformation(planner_report));
        for (int foot = 0; foot < kNumFeet; ++foot) {
            std::printf(
                "  replanning audit: segment=%d foot=%d displacement=%.6f "
                "clearance=%.6f moving_force=%.6f terminal_force=%.6f "
                "terminal_gap=%.6e terminal_speed=%.6f events=%d "
                "seed_schedule=(%d,%d) optimized_schedule=(%d,%d) "
                "timing_change=%d\n",
                segment, foot, planner_report.foot_displacement[foot],
                planner_report.foot_clearance[foot],
                planner_report.moving_normal_force[foot],
                planner_report.terminal_normal_force[foot],
                planner_report.terminal_gap[foot],
                planner_report.terminal_speed[foot],
                planner_report.contact_events[foot],
                planner_report.seed_liftoff_stage[foot],
                planner_report.seed_touchdown_stage[foot],
                planner_report.optimized_liftoff_stage[foot],
                planner_report.optimized_touchdown_stage[foot],
                planner_report.schedule_timing_change[foot]);
        }
        if (planner_status != nmpc::Status::SUCCESS) return 1;
        print_plan_transitions(plan, segment);

        bool initialize_segment_from_plan = segment == 0;
        if (!options.snapshot_prefix.empty() && segment == 0) {
            MujocoInitializationReport initialization;
            if (adapter.initialize_state_from_plan(
                    plan.stages[0], &initialization) != Status::SUCCESS ||
                adapter.read_whole_body_state(state) != Status::SUCCESS ||
                adapter.read_foot_contacts(contacts) != Status::SUCCESS) {
                std::printf(
                    "failed to prepare segment 0 boundary snapshot\n");
                return 1;
            }
            std::printf(
                "native plan initialization: iterations=%d base_error=%.3e "
                "foot_error=%.3e\n",
                initialization.iterations, initialization.maximum_base_error,
                initialization.maximum_foot_error);
            initialize_segment_from_plan = false;
        }
        if (!options.snapshot_prefix.empty()) {
            const std::string stem =
                snapshot_stem(options.snapshot_prefix, segment);
            const std::string plan_path = stem + ".plan";
            const std::string snapshot_path = stem + ".snapshot";
            const std::string temporary_plan_path = plan_path + ".tmp";
            MujocoReplaySnapshot snapshot;
            const MujocoReplaySnapshotConfiguration configuration =
                snapshot_configuration(
                    terrain, terrain_seed, terrain_amplitude, options);
            const std::uint64_t plan_fingerprint =
                contact_plan_fingerprint(plan);
            if (file_exists(snapshot_path) ||
                capture_mujoco_replay_snapshot(
                    model, data, segment, configuration, plan_fingerprint,
                    contacts, snapshot) != Status::SUCCESS) {
                std::remove(temporary_plan_path.c_str());
                std::printf(
                    "failed to write replanning boundary snapshot: "
                    "segment=%d prefix=%s\n",
                    segment, options.snapshot_prefix.c_str());
                return 1;
            }
            bool published_plan = false;
            if (file_exists(plan_path)) {
                ContactPlan<kFourStepHorizon> existing_plan;
                if (read_contact_plan(plan_path.c_str(), existing_plan) !=
                        Status::SUCCESS ||
                    contact_plan_fingerprint(existing_plan) !=
                        plan_fingerprint) {
                    std::printf(
                        "existing boundary plan does not match segment %d: "
                        "%s\n",
                        segment, plan_path.c_str());
                    return 1;
                }
            } else if (write_contact_plan(
                           plan, temporary_plan_path.c_str()) !=
                           Status::SUCCESS ||
                       std::rename(
                           temporary_plan_path.c_str(), plan_path.c_str()) !=
                           0) {
                std::remove(temporary_plan_path.c_str());
                std::printf(
                    "failed to publish replanning boundary plan: "
                    "segment=%d path=%s\n",
                    segment, plan_path.c_str());
                return 1;
            } else {
                published_plan = true;
            }
            if (write_mujoco_replay_snapshot(
                    snapshot, snapshot_path.c_str()) != Status::SUCCESS) {
                if (published_plan) std::remove(plan_path.c_str());
                std::printf(
                    "failed to publish replanning boundary snapshot: "
                    "segment=%d path=%s\n",
                    segment, snapshot_path.c_str());
                return 1;
            }
            std::printf(
                "wrote replanning boundary snapshot: segment=%d "
                "snapshot=%s plan=%s\n",
                segment, snapshot_path.c_str(), plan_path.c_str());
        }

        ReplaySegmentReport segment_report;
        if (execute_contact_plan_segment(
                model, data, adapter, terrain, options, plan,
                initialize_segment_from_plan, segment, &segment_report) != 0) {
            return 1;
        }
        total_swing_events += segment_report.swing_events;
        if (segment_report.task_success) ++task_successful_segments;
        if (segment_report.timing_success) ++timing_successful_segments;
    }

    WholeBodyState final_state;
    if (adapter.read_whole_body_state(final_state) != Status::SUCCESS)
        return 1;
    const double base_displacement =
        final_state.base.position_world[0] - initial_base_position[0];
    const int expected_swing_events = kNumFeet * replan_count;
    const double minimum_base_displacement = 0.04 * replan_count;
    const bool accepted = total_swing_events == expected_swing_events &&
        task_successful_segments == replan_count &&
        (!options.require_timing ||
         timing_successful_segments == replan_count) &&
        base_displacement >= minimum_base_displacement &&
        final_state.base.position_world[2] >= initial_base_position[2] - 0.05;
    std::printf(
        "replanned course gates: segments=%d seed_policy=%s "
        "swing_events=%d expected_swing_events=%d "
        "base_displacement=%.6f minimum_base_displacement=%.6f "
        "final_base=(%.6f, %.6f, %.6f) "
        "task_successful_segments=%d timing_successful_segments=%d "
        "timing_required=%d "
        "accepted=%d\n",
        replan_count, replan_seed_policy_name(seed_policy),
        total_swing_events, expected_swing_events, base_displacement,
        minimum_base_displacement,
        final_state.base.position_world[0],
        final_state.base.position_world[1],
        final_state.base.position_world[2], task_successful_segments,
        timing_successful_segments, options.require_timing ? 1 : 0,
        accepted ? 1 : 0);
    return accepted ? 0 : 1;
}

double percentile_ms(std::vector<double> samples, double percentile) {
    if (samples.empty()) return 0.0;
    std::sort(samples.begin(), samples.end());
    const double rank = percentile * static_cast<double>(samples.size() - 1);
    const std::size_t lower = static_cast<std::size_t>(std::floor(rank));
    const std::size_t upper = static_cast<std::size_t>(std::ceil(rank));
    const double fraction = rank - static_cast<double>(lower);
    return samples[lower] * (1.0 - fraction) + samples[upper] * fraction;
}

const char* realtime_platform_name() {
#if defined(_WIN32)
    return "windows";
#elif defined(__linux__)
    return "linux";
#else
    return "unsupported";
#endif
}

bool realtime_environment_is_wsl() {
#if defined(__linux__)
    return std::getenv("WSL_INTEROP") != nullptr ||
        std::getenv("WSL_DISTRO_NAME") != nullptr;
#else
    return false;
#endif
}

double current_thread_cpu_time_ms() {
#if defined(_WIN32)
    FILETIME creation_time;
    FILETIME exit_time;
    FILETIME kernel_time;
    FILETIME user_time;
    if (!GetThreadTimes(
            GetCurrentThread(), &creation_time, &exit_time,
            &kernel_time, &user_time)) {
        return std::numeric_limits<double>::quiet_NaN();
    }
    ULARGE_INTEGER kernel_ticks;
    kernel_ticks.LowPart = kernel_time.dwLowDateTime;
    kernel_ticks.HighPart = kernel_time.dwHighDateTime;
    ULARGE_INTEGER user_ticks;
    user_ticks.LowPart = user_time.dwLowDateTime;
    user_ticks.HighPart = user_time.dwHighDateTime;
    return 1e-4 * static_cast<double>(
        kernel_ticks.QuadPart + user_ticks.QuadPart);
#elif defined(__linux__)
    timespec stamp{};
    if (clock_gettime(CLOCK_THREAD_CPUTIME_ID, &stamp) != 0)
        return std::numeric_limits<double>::quiet_NaN();
    return 1000.0 * static_cast<double>(stamp.tv_sec) +
        1e-6 * static_cast<double>(stamp.tv_nsec);
#else
    return std::numeric_limits<double>::quiet_NaN();
#endif
}

struct ExecutionTimingStamp {
    std::chrono::steady_clock::time_point wall;
    double thread_cpu_ms = std::numeric_limits<double>::quiet_NaN();
};

ExecutionTimingStamp capture_execution_timing_stamp() {
    return {std::chrono::steady_clock::now(), current_thread_cpu_time_ms()};
}

void finish_execution_phase(
    realtime_execution_detail::ExecutionPhase phase,
    ExecutionTimingStamp& started,
    realtime_execution_detail::ExecutionPhaseTiming& timing) {
    const ExecutionTimingStamp finished = capture_execution_timing_stamp();
    auto& sample = timing[static_cast<std::size_t>(phase)];
    sample.wall_ms += std::chrono::duration<double, std::milli>(
        finished.wall - started.wall).count();
    if (std::isfinite(sample.thread_cpu_ms) &&
        std::isfinite(started.thread_cpu_ms) &&
        std::isfinite(finished.thread_cpu_ms)) {
        sample.thread_cpu_ms += std::max(
            0.0, finished.thread_cpu_ms - started.thread_cpu_ms);
    } else {
        sample.thread_cpu_ms = std::numeric_limits<double>::quiet_NaN();
    }
    started = finished;
}

std::vector<int> current_allowed_cpus() {
    std::vector<int> cpus;
#if defined(_WIN32)
    DWORD_PTR process_mask = 0;
    DWORD_PTR system_mask = 0;
    if (GetProcessAffinityMask(
            GetCurrentProcess(), &process_mask, &system_mask)) {
        for (int cpu = 0; cpu < static_cast<int>(8 * sizeof(DWORD_PTR)); ++cpu)
            if ((process_mask & (static_cast<DWORD_PTR>(1) << cpu)) != 0)
                cpus.push_back(cpu);
    }
#elif defined(__linux__)
    cpu_set_t mask;
    CPU_ZERO(&mask);
    if (pthread_getaffinity_np(
            pthread_self(), sizeof(mask), &mask) == 0) {
        for (int cpu = 0; cpu < CPU_SETSIZE; ++cpu)
            if (CPU_ISSET(cpu, &mask)) cpus.push_back(cpu);
    }
#endif
    return cpus;
}

bool read_cpu_topology(
    int logical_cpu,
    realtime_execution_detail::CpuTopologyEntry& topology) {
#if defined(__linux__)
    const std::string base = "/sys/devices/system/cpu/cpu" +
        std::to_string(logical_cpu) + "/topology/";
    std::ifstream package_stream(base + "physical_package_id");
    std::ifstream core_stream(base + "core_id");
    int package_id = -1;
    int core_id = -1;
    if (!(package_stream >> package_id) || !(core_stream >> core_id))
        return false;
    topology.logical_cpu = logical_cpu;
    topology.package_id = package_id;
    topology.core_id = core_id;
    return true;
#else
    (void)logical_cpu;
    (void)topology;
    return false;
#endif
}

struct ThreadSchedulingResult {
    bool affinity_applied = false;
    bool priority_applied = false;
    int affinity_error = 0;
    int priority_error = 0;
};

ThreadSchedulingResult configure_current_thread_scheduling(
    const char* role, int cpu, bool high_priority) {
    ThreadSchedulingResult result;
#if defined(_WIN32)
    if (cpu >= 0 && cpu < static_cast<int>(8 * sizeof(DWORD_PTR))) {
        const DWORD_PTR mask = static_cast<DWORD_PTR>(1) << cpu;
        if (SetThreadAffinityMask(GetCurrentThread(), mask) != 0)
            result.affinity_applied = true;
        else
            result.affinity_error = static_cast<int>(GetLastError());
    }
    const int requested_priority = high_priority
        ? THREAD_PRIORITY_HIGHEST
        : THREAD_PRIORITY_ABOVE_NORMAL;
    if (SetThreadPriority(GetCurrentThread(), requested_priority))
        result.priority_applied = true;
    else
        result.priority_error = static_cast<int>(GetLastError());
#elif defined(__linux__)
    if (cpu >= 0 && cpu < CPU_SETSIZE) {
        cpu_set_t mask;
        CPU_ZERO(&mask);
        CPU_SET(cpu, &mask);
        result.affinity_error = pthread_setaffinity_np(
            pthread_self(), sizeof(mask), &mask);
        result.affinity_applied = result.affinity_error == 0;
    }
    sched_param parameters{};
    const int maximum_priority = sched_get_priority_max(SCHED_FIFO);
    const int minimum_priority = sched_get_priority_min(SCHED_FIFO);
    if (maximum_priority >= minimum_priority) {
        const double fraction = high_priority ? 0.80 : 0.60;
        parameters.sched_priority = minimum_priority + static_cast<int>(
            fraction * (maximum_priority - minimum_priority));
        result.priority_error = pthread_setschedparam(
            pthread_self(), SCHED_FIFO, &parameters);
        result.priority_applied = result.priority_error == 0;
    } else {
        result.priority_error = errno != 0 ? errno : EINVAL;
    }
#else
    (void)cpu;
    (void)high_priority;
    result.affinity_error = ENOTSUP;
    result.priority_error = ENOTSUP;
#endif
    std::printf(
        "real-time thread scheduling: role=%s platform=%s cpu=%d "
        "affinity_applied=%d affinity_error=%d priority_policy=%s "
        "priority_applied=%d priority_error=%d affinity_fallback=%s "
        "priority_fallback=%s\n",
        role, realtime_platform_name(), cpu,
        result.affinity_applied ? 1 : 0, result.affinity_error,
#if defined(_WIN32)
        high_priority ? "highest" : "above_normal",
#elif defined(__linux__)
        "SCHED_FIFO",
#else
        "unsupported",
#endif
        result.priority_applied ? 1 : 0, result.priority_error,
        result.affinity_applied ? "none" : "not_pinned",
        result.priority_applied ? "none" : "inherited");
    return result;
}

struct RealtimePublicationDecision {
    using Clock = QuadrupedCITORealtimePlanner::Clock;
    RealtimePlannerResult result;
    ContactPlan<kRealtimeHorizon> plan;
    RealtimeContactTask task;
    Clock::time_point requested_at = Clock::time_point::max();
    Clock::time_point deadline = Clock::time_point::max();
    double request_to_result_ms = 0.0;
    double request_sim_time = 0.0;
    double request_height_correction = 0.0;
    int request_elapsed_periods = 1;
    int sequence = 0;
    bool independently_valid = false;
    bool task_metadata_valid = false;
    bool stale = false;
    bool accepted = false;
};

struct RealtimeTaskLedgerEntry {
    RealtimeContactTask task;
};

bool valid_sustained_task_definition(const RealtimeContactTask& task) {
    const std::uint64_t maximum_id = static_cast<std::uint64_t>(
        (std::numeric_limits<int>::max() - kRealtimeSwingFirstStage) /
        kRealtimeSustainedTaskSpacing);
    if (task.id > maximum_id) return false;
    const int expected_start = kRealtimeSwingFirstStage +
        static_cast<int>(task.id) * kRealtimeSustainedTaskSpacing;
    return task.moving_foot == kRealtimeSustainedFootOrder[
               static_cast<int>(task.id % kNumFeet)] &&
        task.start_knot == expected_start &&
        task.touchdown_knot == expected_start +
            kRealtimeSustainedSwingStages &&
        std::fabs(task.target_x - task.origin_x -
                  kRealtimeRequestedStep) <= 1e-9;
}

RealtimePublicationDecision attempt_realtime_plan_update(
    QuadrupedCITORealtimePlanner& planner,
    const Vec<kStateDim>& measured_state, double request_sim_time,
    double request_height_correction, int elapsed_periods, int sequence,
    QuadrupedCITORealtimePlanner::Clock::time_point requested_at,
    QuadrupedCITORealtimePlanner::Clock::time_point deadline) {
    using Clock = QuadrupedCITORealtimePlanner::Clock;
    RealtimePublicationDecision decision;
    decision.requested_at = requested_at;
    decision.deadline = deadline;
    decision.request_sim_time = request_sim_time;
    decision.request_height_correction = request_height_correction;
    decision.request_elapsed_periods = elapsed_periods;
    decision.sequence = sequence;
    decision.result = planner.warm_update(
        measured_state, deadline, elapsed_periods);
    decision.independently_valid =
        decision.result.full_kkt && decision.result.task_pass;
    const RealtimeContactTask* valid_task = planner.last_valid_task();
    decision.task_metadata_valid = valid_task &&
        valid_task->id == decision.result.task_id &&
        valid_task->moving_foot == decision.result.moving_foot;
    const bool candidate_acceptable = decision.result.published &&
        decision.independently_valid && planner.last_valid_plan() &&
        decision.task_metadata_valid;
    decision.stale = decision.result.deadline_miss || Clock::now() > deadline;
    if (candidate_acceptable && !decision.stale) {
        decision.plan = *planner.last_valid_plan();
        decision.task = *valid_task;
    }
    const Clock::time_point completed_at = Clock::now();
    decision.request_to_result_ms =
        std::chrono::duration<double, std::milli>(
            completed_at - requested_at).count();
    decision.stale = decision.stale || completed_at > deadline;
    decision.accepted = candidate_acceptable && !decision.stale;
    return decision;
}

class RealtimePlannerWorker {
public:
    RealtimePlannerWorker(QuadrupedCITORealtimePlanner& planner,
                          double deadline_ms, int planner_cpu)
        : planner_(planner), deadline_ms_(deadline_ms),
          planner_cpu_(planner_cpu),
          thread_(&RealtimePlannerWorker::run, this) {}

    ~RealtimePlannerWorker() { stop(); }

    bool enqueue(const Vec<kStateDim>& measured_state,
                 double request_sim_time,
                 double request_height_correction, int elapsed_periods,
                 int sequence) {
        const auto requested_at = QuadrupedCITORealtimePlanner::Clock::now();
        std::unique_lock<std::mutex> lock(mutex_, std::try_to_lock);
        if (!lock.owns_lock()) return false;
        if (stop_requested_ || request_pending_ || worker_busy_ ||
            result_pending_)
            return false;
        measured_state_ = measured_state;
        request_sim_time_ = request_sim_time;
        request_height_correction_ = request_height_correction;
        request_elapsed_periods_ = elapsed_periods;
        request_sequence_ = sequence;
        request_started_at_ = requested_at;
        request_deadline_ = requested_at +
            std::chrono::duration_cast<
                QuadrupedCITORealtimePlanner::Clock::duration>(
                    std::chrono::duration<double, std::milli>(deadline_ms_));
        request_pending_ = true;
        lock.unlock();
        condition_.notify_one();
        return true;
    }

    bool take_result(RealtimePublicationDecision& decision) {
        std::unique_lock<std::mutex> lock(mutex_, std::try_to_lock);
        if (!lock.owns_lock()) return false;
        if (!result_pending_) return false;
        decision = result_;
        result_pending_ = false;
        return true;
    }

    void scheduling_status(bool& reported,
                           ThreadSchedulingResult& scheduling) {
        std::lock_guard<std::mutex> lock(mutex_);
        reported = planner_scheduling_reported_;
        scheduling = planner_scheduling_;
    }

    void stop() {
        {
            std::lock_guard<std::mutex> lock(mutex_);
            if (stop_requested_) return;
            stop_requested_ = true;
            request_pending_ = false;
        }
        condition_.notify_one();
        if (thread_.joinable()) thread_.join();
    }

private:
    void run() {
        const ThreadSchedulingResult scheduling =
            configure_current_thread_scheduling(
                "planner", planner_cpu_, false);
        {
            std::lock_guard<std::mutex> lock(mutex_);
            planner_scheduling_ = scheduling;
            planner_scheduling_reported_ = true;
        }
        for (;;) {
            Vec<kStateDim> measured_state;
            double request_sim_time = 0.0;
            double request_height_correction = 0.0;
            int elapsed_periods = 1;
            int sequence = 0;
            QuadrupedCITORealtimePlanner::Clock::time_point requested_at;
            QuadrupedCITORealtimePlanner::Clock::time_point deadline;
            {
                std::unique_lock<std::mutex> lock(mutex_);
                condition_.wait(lock, [this] {
                    return stop_requested_ || request_pending_;
                });
                if (stop_requested_) return;
                measured_state = measured_state_;
                request_sim_time = request_sim_time_;
                request_height_correction = request_height_correction_;
                elapsed_periods = request_elapsed_periods_;
                sequence = request_sequence_;
                requested_at = request_started_at_;
                deadline = request_deadline_;
                request_pending_ = false;
                worker_busy_ = true;
            }
            RealtimePublicationDecision decision =
                attempt_realtime_plan_update(
                    planner_, measured_state, request_sim_time,
                    request_height_correction, elapsed_periods, sequence,
                    requested_at, deadline);
            result_ = decision;
            {
                std::lock_guard<std::mutex> lock(mutex_);
                result_pending_ = true;
                worker_busy_ = false;
            }
        }
    }

    QuadrupedCITORealtimePlanner& planner_;
    double deadline_ms_ = 0.0;
    int planner_cpu_ = -1;
    std::mutex mutex_;
    std::condition_variable condition_;
    std::thread thread_;
    Vec<kStateDim> measured_state_;
    RealtimePublicationDecision result_;
    double request_sim_time_ = 0.0;
    double request_height_correction_ = 0.0;
    int request_elapsed_periods_ = 1;
    int request_sequence_ = 0;
    QuadrupedCITORealtimePlanner::Clock::time_point request_started_at_;
    QuadrupedCITORealtimePlanner::Clock::time_point request_deadline_;
    bool request_pending_ = false;
    bool result_pending_ = false;
    bool worker_busy_ = false;
    bool stop_requested_ = false;
    ThreadSchedulingResult planner_scheduling_;
    bool planner_scheduling_reported_ = false;
};

int run_realtime_replanning(const mjModel* model, mjData* data,
                            const MujocoGo1Adapter& adapter,
                            const SharedTerrain& terrain,
                            const ReplayOptions& options,
                            double requested_duration,
                            bool disable_planner_updates,
                            bool sustained_mode) {
    constexpr double kPlannerPeriod = 0.2;
    constexpr double kPlannerDeadlineMs = 200.0;
    constexpr double kPlannerP99GateMs = 180.0;
    constexpr double kWbcPeriod = 0.002;

    if (!(requested_duration > 0.0) ||
        std::fabs(model->opt.timestep - kWbcPeriod) > 1e-12) {
        std::printf(
            "real-time replanning requires positive duration and a 0.002 s "
            "MuJoCo timestep (actual=%.9f)\n",
            model->opt.timestep);
        return 1;
    }

    WholeBodyState state;
    MujocoFootContact contacts[kNumFeet];
    if (adapter.read_whole_body_state(state) != Status::SUCCESS ||
        adapter.read_foot_contacts(contacts) != Status::SUCCESS) {
        return 1;
    }
    double maximum_foot_height_correction = 0.0;
    const Vec<kStateDim> initial_planner_state = planner_state_from_whole_body(
        state, terrain, &maximum_foot_height_correction, contacts);
    double initial_foot_x =
        state.foot_positions_world[kRealtimeMovingFoot][0];
    double initial_base_x = state.base.position_world[0];

    RealtimePlannerConfig planner_configuration;
    planner_configuration.rate_hz = 5;
    planner_configuration.sustained_contact_tasks = sustained_mode;
    QuadrupedCITORealtimePlanner planner(terrain, planner_configuration);
    if (planner.initialize(initial_planner_state) != Status::SUCCESS) {
        std::printf("failed to initialize N=50 real-time planner\n");
        return 1;
    }

    std::printf(
        "real-time MuJoCo mode: scheduling=background_worker "
        "wall_concurrent=1 planner_updates_enabled=%d sustained_mode=%d "
        "planner_rate_hz=5 planner_period=0.200 planner_deadline_ms=200.000 "
        "planner_horizon_s=2.500 planner_knots=50 planner_dt_s=0.050 "
        "wbc_period=0.002 wbc_rate_hz=500 "
        "max_foot_height_correction=%.6f\n",
        disable_planner_updates ? 0 : 1, sustained_mode ? 1 : 0,
        maximum_foot_height_correction);
    const RealtimePlannerResult cold = planner.cold_solve();
    std::uint64_t exact_hessian_analytic_calls =
        cold.stats.exact_hessian_analytic_calls;
    std::uint64_t exact_hessian_fd_calls = cold.stats.exact_hessian_fd_calls;
    std::printf(
        "real-time cold solve (excluded): status=%s full_kkt=%d task_pass=%d "
        "published=%d solver_ms=%.3f audit_ms=%.3f extract_ms=%.3f "
        "end_to_end_ms=%.3f\n",
        nmpc::status_string(cold.status), cold.full_kkt ? 1 : 0,
        cold.task_pass ? 1 : 0, cold.published ? 1 : 0, cold.solver_ms,
        cold.audit_ms, cold.plan_extract_ms, cold.end_to_end_ms);
    const ContactPlan<kRealtimeHorizon>* cold_plan =
        planner.last_valid_plan();
    const RealtimeContactTask* cold_task = planner.last_valid_task();
    if (!cold.published || !cold.full_kkt || !cold.task_pass || !cold_plan ||
        !cold_task || cold_task->id != cold.task_id ||
        cold_task->moving_foot != cold.moving_foot ||
        (sustained_mode && !valid_sustained_task_definition(*cold_task))) {
        std::printf(
            "real-time cold plan rejected by independent publication gate\n");
        return 1;
    }
    ContactPlan<kRealtimeHorizon> plan_slots[2];
    RealtimeContactTask task_slots[2];
    plan_slots[0] = *cold_plan;
    task_slots[0] = *cold_task;
    ContactPlan<kRealtimeHorizon>* active_plan = &plan_slots[0];
    ContactPlan<kRealtimeHorizon>* staging_plan = &plan_slots[1];
    RealtimeContactTask* active_task = &task_slots[0];
    RealtimeContactTask* staging_task = &task_slots[1];
    std::vector<RealtimeTaskLedgerEntry> task_ledger;
    task_ledger.reserve(16);
    if (sustained_mode) task_ledger.push_back({*cold_task});
    MujocoInitializationReport initialization;
    if (adapter.initialize_state_from_plan(
            active_plan->stages[0], &initialization) != Status::SUCCESS ||
        adapter.read_whole_body_state(state) != Status::SUCCESS ||
        adapter.read_foot_contacts(contacts) != Status::SUCCESS) {
        std::printf(
            "failed to initialize Go1 from audited real-time cold plan\n");
        return 1;
    }
    initial_foot_x = state.foot_positions_world[kRealtimeMovingFoot][0];
    initial_base_x = state.base.position_world[0];
    std::printf(
        "real-time cold-plan initialization: iterations=%d "
        "base_error=%.3e foot_error=%.3e\n",
        initialization.iterations, initialization.maximum_base_error,
        initialization.maximum_foot_error);
    double active_plan_start_time = data->time;

    ContactPlanSample initial_reference;
    if (sample_contact_plan(*active_plan, 0.0, initial_reference) !=
            PlanSampleStatus::ACTIVE) {
        return 1;
    }
    shape_swing_references(*active_plan, terrain, initial_reference);

    const ConvexWBCParameters wbc_parameters = go1_wbc_parameters();
    ConvexWholeBodyController controller(wbc_parameters);
    const ContactExecutionFeedbackParameters feedback_parameters =
        go1_contact_feedback_parameters(wbc_parameters);
    ConvexWBCCommand command;
    MujocoActuatorReport reports[kNumFeet][kGo1JointsPerLeg];
    Vec<3> contact_sample_foot_positions[kNumFeet];
    StanceContactConfirmation contact_confirmation[kNumFeet];
    ContactTorqueBlendHandoff contact_handoff[kNumFeet];
    EarlyContactFeedback early_contact_feedback[kNumFeet];
    bool early_contact_support[kNumFeet] = {};
    int late_touchdown_search_ticks[kNumFeet] = {};
    for (int foot = 0; foot < kNumFeet; ++foot) {
        contact_sample_foot_positions[foot] = state.foot_positions_world[foot];
        const bool measured_contact = contacts[foot].in_contact &&
            contacts[foot].normal_force >= kContactConfirmationForce;
        contact_confirmation[foot].initialize(
            initial_reference.feet[foot].planned_contact, measured_contact);
    }
    FootReplayMetrics foot_metrics[kNumFeet];
    for (int foot = 0; foot < kNumFeet; ++foot) {
        foot_metrics[foot].initialize(
            initial_reference.feet[foot].planned_contact,
            contacts[foot].in_contact &&
                contacts[foot].normal_force >= kTouchdownDetectionForce);
    }

    const int push_body = mj_name2id(model, mjOBJ_BODY, "trunk");
    if (push_body < 0) return 1;
    const realtime_execution_detail::CpuAssignment cpu_assignment =
        realtime_execution_detail::choose_cpu_assignment(
            current_allowed_cpus());
    realtime_execution_detail::CpuTopologyEntry wbc_topology;
    realtime_execution_detail::CpuTopologyEntry planner_topology;
    const bool topology_available =
        read_cpu_topology(cpu_assignment.wbc_cpu, wbc_topology) &&
        read_cpu_topology(cpu_assignment.planner_cpu, planner_topology);
    const bool separate_physical_cores = topology_available &&
        realtime_execution_detail::are_separate_physical_cores(
            wbc_topology, planner_topology);
    std::printf(
        "real-time logical CPU assignment: platform=%s planned_wbc_cpu=%d "
        "planned_planner_cpu=%d separate_logical_cpus=%d "
        "assignment_available=%d topology_available=%d "
        "separate_physical_cores=%d\n",
        realtime_platform_name(), cpu_assignment.wbc_cpu,
        cpu_assignment.planner_cpu,
        cpu_assignment.separate_logical_cpus ? 1 : 0,
        cpu_assignment.wbc_cpu >= 0 ? 1 : 0,
        topology_available ? 1 : 0, separate_physical_cores ? 1 : 0);
    const ThreadSchedulingResult wbc_scheduling =
        configure_current_thread_scheduling(
            "wbc", cpu_assignment.wbc_cpu, true);
    const double start_time = data->time;
    double contact_sample_time = 0.0;
    double next_replan_time = start_time + kPlannerPeriod;
    const int ticks = static_cast<int>(
        std::ceil(requested_duration / model->opt.timestep));
    RealtimePlannerWorker planner_worker(
        planner, kPlannerDeadlineMs, cpu_assignment.planner_cpu);
    int replan_opportunities = 0;
    int warm_attempts = 0;
    int worker_busy_skips = 0;
    realtime_execution_detail::PlannerRequestPeriodTracker
        planner_request_periods;
    int maximum_request_elapsed_periods = 1;
    int usable_updates = 0;
    int accepted_publications = 0;
    int fallback_updates = 0;
    int deadline_misses = 0;
    int invalid_candidate_rejections = 0;
    int stale_candidate_rejections = 0;
    int task_metadata_rejections = 0;
    int fresh_task_transition_witnesses = 0;
    int invalid_publications = 0;
    int stale_publications = 0;
    int consecutive_fallbacks = 0;
    int maximum_consecutive_fallbacks = 0;
    int expired_plan_ticks = 0;
    int saturated_joint_ticks = 0;
    int wbc_ticks = 0;
    int wall_wbc_deadline_misses = 0;
    double maximum_task_progress = 0.0;
    double squared_base_position_error = 0.0;
    std::vector<double> warm_end_to_end_ms;
    std::vector<double> wbc_compute_ms;
    std::vector<double> complete_tick_work_ms;
    std::vector<double> scheduled_tick_response_ms;
    std::vector<double> wbc_scheduling_lateness_ms;
    std::array<std::vector<double>,
               realtime_execution_detail::kExecutionPhaseCount>
        phase_wall_ms, phase_thread_cpu_ms;
    std::array<int, realtime_execution_detail::kDeadlineMissCauseCount>
        complete_tick_miss_counts{}, scheduled_response_miss_counts{};
    warm_end_to_end_ms.reserve(ticks / 100 + 16);
    wbc_compute_ms.reserve(ticks);
    complete_tick_work_ms.reserve(ticks);
    scheduled_tick_response_ms.reserve(ticks);
    wbc_scheduling_lateness_ms.reserve(ticks);
    for (std::size_t phase = 0;
         phase < realtime_execution_detail::kExecutionPhaseCount; ++phase) {
        phase_wall_ms[phase].reserve(ticks);
        phase_thread_cpu_ms[phase].reserve(ticks);
    }
    const auto wall_loop_start = std::chrono::steady_clock::now();
    auto next_wall_tick = wall_loop_start;
    using realtime_execution_detail::ExecutionPhase;

    for (int tick = 0; tick < ticks; ++tick) {
        const auto scheduled_tick_start = next_wall_tick;
        ExecutionTimingStamp phase_started =
            capture_execution_timing_stamp();
        const auto tick_started = phase_started.wall;
        realtime_execution_detail::ExecutionPhaseTiming tick_timing;
        for (auto& phase : tick_timing) phase.thread_cpu_ms = 0.0;
        wbc_scheduling_lateness_ms.push_back(std::max(
            0.0, std::chrono::duration<double, std::milli>(
                tick_started - scheduled_tick_start).count()));
        next_wall_tick += std::chrono::duration_cast<
            std::chrono::steady_clock::duration>(
                std::chrono::duration<double>(kWbcPeriod));
        finish_execution_phase(
            ExecutionPhase::HOUSEKEEPING, phase_started, tick_timing);
        mj_step1(model, data);
        finish_execution_phase(
            ExecutionPhase::MJ_STEP1, phase_started, tick_timing);
        if (adapter.read_whole_body_state(state) != Status::SUCCESS) return 1;
        const double elapsed = data->time - start_time;

        RealtimePublicationDecision decision;
        if (planner_worker.take_result(decision)) {
            const RealtimePlannerResult& update = decision.result;
            exact_hessian_analytic_calls +=
                update.stats.exact_hessian_analytic_calls;
            exact_hessian_fd_calls += update.stats.exact_hessian_fd_calls;
            planner_request_periods.record_shift_result(
                update.shift_consumed);
            const bool independently_valid = decision.independently_valid;
            const bool changes_task = decision.accepted &&
                decision.task.id != active_task->id;
            const bool task_definition_valid = !sustained_mode ||
                valid_sustained_task_definition(decision.task);
            const bool task_metadata_valid =
                realtime_execution_detail::task_publication_metadata_gate(
                    active_task->id, decision.task.id,
                    decision.task_metadata_valid && task_definition_valid,
                    update.task_transition_witness);
            bool accept_publication = false;
            realtime_execution_detail::PublicationTiming publication_timing;
            if (decision.accepted && task_metadata_valid) {
                *staging_plan = decision.plan;
                *staging_task = decision.task;
                std::swap(active_plan, staging_plan);
                std::swap(active_task, staging_task);
                publication_timing =
                    realtime_execution_detail::evaluate_publication_timing(
                        decision.requested_at, decision.deadline,
                        std::chrono::steady_clock::now(), decision.stale);
                if (publication_timing.stale) {
                    std::swap(active_plan, staging_plan);
                    std::swap(active_task, staging_task);
                } else {
                    accept_publication = true;
                }
            } else {
                publication_timing =
                    realtime_execution_detail::evaluate_publication_timing(
                        decision.requested_at, decision.deadline,
                        std::chrono::steady_clock::now(), decision.stale);
            }
            const bool publication_is_stale = publication_timing.stale;
            warm_end_to_end_ms.push_back(
                publication_timing.request_to_handoff_ms);
            if (independently_valid) ++usable_updates;
            if (publication_is_stale) ++deadline_misses;
            if (!independently_valid) ++invalid_candidate_rejections;
            if (publication_is_stale) ++stale_candidate_rejections;
            if (update.published && !task_metadata_valid)
                ++task_metadata_rejections;
            if (update.published && !independently_valid)
                ++invalid_publications;
            if (update.published && publication_is_stale)
                ++stale_publications;

            if (accept_publication) {
                active_plan_start_time = decision.request_sim_time;
                ++accepted_publications;
                consecutive_fallbacks = 0;
                if (changes_task) {
                    task_ledger.push_back({*active_task});
                    if (update.task_transition_witness)
                        ++fresh_task_transition_witnesses;
                }
                const int planned_contact_mask =
                    plan_sample_contact_mask(active_plan->stages[0]);
                for (int foot = 0; foot < kNumFeet; ++foot) {
                    const bool planned_contact =
                        (planned_contact_mask & (1 << foot)) != 0;
                    const bool load_bearing_contact =
                        contacts[foot].in_contact &&
                        contacts[foot].normal_force >=
                            kContactConfirmationForce;
                    contact_confirmation[foot].initialize(
                        planned_contact, load_bearing_contact);
                    late_touchdown_search_ticks[foot] = 0;
                }
            } else {
                ++fallback_updates;
                ++consecutive_fallbacks;
                maximum_consecutive_fallbacks = std::max(
                    maximum_consecutive_fallbacks, consecutive_fallbacks);
            }
            if (options.trace_execution) {
                std::printf(
                    "real-time planner result: update=%d "
                    "request_sim_time=%.6f apply_sim_time=%.6f status=%s "
                    "elapsed_periods=%d "
                    "full_kkt=%d task_pass=%d deadline_miss=%d "
                    "planner_published=%d accepted_publication=%d "
                    "task_id=%llu moving_foot=%d "
                    "task_transition_witness=%d task_metadata_valid=%d "
                    "fallback=%d shift_ms=%.3f solver_ms=%.3f "
                    "audit_ms=%.3f extract_ms=%.3f "
                    "planner_internal_ms=%.3f request_to_result_ms=%.3f "
                    "request_to_handoff_ms=%.3f outer_iterations=%d "
                    "inner_iterations=%d line_search_evals=%d "
                    "primal=%.3e dual=%.3e complementarity=%.3e "
                    "stats_mpcc=%.3e barrier=%.3e dynamics=%.3e "
                    "inequality=%.3e mpcc=%.3e task_displacement=%.6f "
                    "clearance=%.6f terminal_foot_error=%.3e "
                    "terminal_gap=%.3e terminal_speed=%.3e "
                    "terminal_normal_force=%.3e moving_unloaded_stages=%d "
                    "height_correction=%.6f\n",
                    decision.sequence, decision.request_sim_time, data->time,
                    nmpc::status_string(update.status),
                    decision.request_elapsed_periods,
                    update.full_kkt ? 1 : 0, update.task_pass ? 1 : 0,
                    publication_is_stale ? 1 : 0,
                    update.published ? 1 : 0,
                    accept_publication ? 1 : 0,
                    static_cast<unsigned long long>(update.task_id),
                    update.moving_foot,
                    update.task_transition_witness ? 1 : 0,
                    task_metadata_valid ? 1 : 0,
                    accept_publication ? 0 : 1, update.shift_ms,
                    update.solver_ms, update.audit_ms,
                    update.plan_extract_ms, update.end_to_end_ms,
                    decision.request_to_result_ms,
                    publication_timing.request_to_handoff_ms,
                    update.stats.outer_iterations,
                    update.stats.inner_iterations,
                    update.stats.line_search_evals,
                    update.stats.primal_infeas, update.stats.dual_infeas,
                    update.stats.complementarity,
                    update.stats.mpcc_complementarity,
                    update.stats.barrier_param,
                    update.audit.dynamics, update.audit.inequality,
                    update.audit.mpcc, update.audit.task_displacement,
                    update.audit.clearance, update.audit.terminal_foot_error,
                    update.audit.terminal_gap, update.audit.terminal_speed,
                    update.audit.terminal_normal_force,
                    update.audit.moving_unloaded_stages,
                    decision.request_height_correction);
            }
        }

        if (!disable_planner_updates &&
            data->time + 0.5 * model->opt.timestep >= next_replan_time) {
            ++replan_opportunities;
            double height_correction = 0.0;
            const Vec<kStateDim> measured_state =
                planner_state_from_whole_body(
                    state, terrain, &height_correction, contacts);
            const int request_elapsed_periods =
                planner_request_periods.elapsed_periods;
            const bool enqueued = planner_worker.enqueue(
                    measured_state, data->time, height_correction,
                    request_elapsed_periods, replan_opportunities);
            planner_request_periods.record_enqueue_result(enqueued);
            if (enqueued) {
                ++warm_attempts;
                maximum_request_elapsed_periods = std::max(
                    maximum_request_elapsed_periods,
                    request_elapsed_periods);
                if (options.trace_execution) {
                    std::printf(
                        "real-time planner request: opportunity=%d "
                        "sim_time=%.6f height_correction=%.6f "
                        "elapsed_periods=%d enqueued=1\n",
                        replan_opportunities, data->time, height_correction,
                        request_elapsed_periods);
                }
            } else {
                ++worker_busy_skips;
                ++fallback_updates;
                ++consecutive_fallbacks;
                maximum_consecutive_fallbacks = std::max(
                    maximum_consecutive_fallbacks, consecutive_fallbacks);
                if (options.trace_execution) {
                    std::printf(
                        "real-time planner request: opportunity=%d "
                        "sim_time=%.6f height_correction=%.6f enqueued=0 "
                        "elapsed_periods=%d reason=worker_unavailable\n",
                        replan_opportunities, data->time, height_correction,
                        request_elapsed_periods);
                }
            }
            do {
                next_replan_time += kPlannerPeriod;
            } while (next_replan_time <= data->time +
                     0.5 * model->opt.timestep);
        }
        const bool push_active = elapsed >= options.push_start &&
            elapsed < options.push_start + options.push_duration;
        for (int axis = 0; axis < 3; ++axis) {
            data->xfrc_applied[6 * push_body + axis] =
                push_active ? options.push_force_world[axis] : 0.0;
        }

        double plan_time = data->time - active_plan_start_time;
        if (plan_time >= active_plan->terminal.time) {
            ++expired_plan_ticks;
            plan_time = std::max(
                active_plan->stages[0].time,
                active_plan->terminal.time - 0.5 * model->opt.timestep);
        }
        ContactPlanSample plan_reference;
        if (sample_contact_plan(*active_plan, plan_time, plan_reference) !=
                PlanSampleStatus::ACTIVE) {
            std::printf("active real-time plan sampling failed at tick %d\n",
                        tick);
            return 1;
        }
        shape_swing_references(*active_plan, terrain, plan_reference);
        ContactPlanSample control_reference = plan_reference;
        bool newly_confirmed_contact[kNumFeet] = {};
        const Go1SRBDCalibration calibration;
        for (int foot = 0; foot < kNumFeet; ++foot) {
            const bool previously_confirmed =
                contact_confirmation[foot].confirmed();
            const bool planned_contact =
                plan_reference.feet[foot].planned_contact;
            const bool measured_contact = contacts[foot].in_contact &&
                contacts[foot].normal_force >= kTouchdownDetectionForce;
            const bool load_bearing_contact = contacts[foot].in_contact &&
                contacts[foot].normal_force >= kContactConfirmationForce;
            if (!planned_contact) late_touchdown_search_ticks[foot] = 0;
            const bool confirmed = contact_confirmation[foot].update(
                planned_contact, contacts[foot].in_contact,
                load_bearing_contact, kContactLossGraceTicks);
            newly_confirmed_contact[foot] =
                !previously_confirmed && confirmed;

            const Vec<3>& position = state.foot_positions_world[foot];
            const double clearance = position[2] -
                terrain.height(position[0], position[1]) -
                calibration.foot_center_contact_offset;
            foot_metrics[foot].observe(
                elapsed, contact_sample_time, planned_contact,
                measured_contact, plan_reference.feet[foot].position_world,
                contact_sample_foot_positions[foot], clearance,
                kMinimumSwingClearance);
            const bool swing_clearance_reached =
                (sustained_mode || foot == kRealtimeMovingFoot) &&
                foot_metrics[foot].active_event >= 0 &&
                foot_metrics[foot].events[
                    foot_metrics[foot].active_event].maximum_clearance >=
                    kMinimumSwingClearance;
            early_contact_support[foot] = options.early_contact_feedback &&
                early_contact_feedback[foot].update(
                    planned_contact, load_bearing_contact,
                    swing_clearance_reached, feedback_parameters);
            if (early_contact_support[foot]) {
                apply_early_contact_support(
                    position, feedback_parameters,
                    control_reference.feet[foot]);
            }
            if (planned_contact && !confirmed) {
                ++late_touchdown_search_ticks[foot];
                if (options.late_touchdown_search) {
                    apply_late_touchdown_search(
                        late_touchdown_search_ticks[foot],
                        model->opt.timestep, feedback_parameters,
                        control_reference.feet[foot]);
                } else {
                    control_reference.feet[foot].planned_contact = false;
                    control_reference.feet[foot].normal_force = 0.0;
                    control_reference.feet[foot].force_world.zero();
                }
            } else if (planned_contact) {
                late_touchdown_search_ticks[foot] = 0;
            }
        }
        if (options.force_reference == ForceReferenceMode::UNIFORM) {
            int active_feet = 0;
            for (int foot = 0; foot < kNumFeet; ++foot)
                if (control_reference.feet[foot].planned_contact) ++active_feet;
            if (active_feet == 0) return 1;
            const double vertical_force = go1_robot_parameters().mass *
                go1_robot_parameters().gravity / active_feet;
            for (int foot = 0; foot < kNumFeet; ++foot) {
                control_reference.feet[foot].force_world.zero();
                control_reference.feet[foot].normal_force = 0.0;
                if (control_reference.feet[foot].planned_contact) {
                    control_reference.feet[foot].force_world[2] =
                        vertical_force;
                    control_reference.feet[foot].normal_force = dot3(
                        control_reference.feet[foot].terrain_normal_world,
                        control_reference.feet[foot].force_world);
                }
            }
        }
        for (int foot = 0; foot < kNumFeet; ++foot) {
            contact_handoff[foot].update(
                plan_reference.feet[foot].planned_contact,
                newly_confirmed_contact[foot], contacts[foot].normal_force,
                control_reference.feet[foot]);
        }
        finish_execution_phase(
            ExecutionPhase::STATE_REFERENCE_CONTACT,
            phase_started, tick_timing);
        const auto wbc_compute_start = std::chrono::steady_clock::now();
        if (controller.compute(control_reference, state, command) !=
                Status::SUCCESS ||
            adapter.apply_command(command, reports) != Status::SUCCESS) {
            std::printf("real-time WBC failed at tick %d\n", tick);
            return 1;
        }
        wbc_compute_ms.push_back(std::chrono::duration<double, std::milli>(
            std::chrono::steady_clock::now() - wbc_compute_start).count());
        finish_execution_phase(
            ExecutionPhase::WBC_APPLY, phase_started, tick_timing);
        double base_error_sq = 0.0;
        for (int axis = 0; axis < 3; ++axis) {
            const double error = state.base.position_world[axis] -
                plan_reference.base.position_world[axis];
            base_error_sq += error * error;
        }
        squared_base_position_error += base_error_sq;

        finish_execution_phase(
            ExecutionPhase::HOUSEKEEPING, phase_started, tick_timing);
        mj_step2(model, data);
        finish_execution_phase(
            ExecutionPhase::MJ_STEP2, phase_started, tick_timing);
        if (adapter.read_foot_contacts(contacts) != Status::SUCCESS ||
            adapter.update_applied_torques(reports) != Status::SUCCESS) {
            return 1;
        }
        // mj_step2's applied contact forces belong to the pre-integration
        // state read after mj_step1. Preserve that source-time pairing here;
        // the next mj_step1 refreshes the whole-body state.
        finish_execution_phase(
            ExecutionPhase::STATE_REFERENCE_CONTACT,
            phase_started, tick_timing);
        for (int foot = 0; foot < kNumFeet; ++foot) {
            contact_sample_foot_positions[foot] =
                state.foot_positions_world[foot];
            for (int joint = 0; joint < kGo1JointsPerLeg; ++joint) {
                if (reports[foot][joint].saturated)
                    ++saturated_joint_ticks;
            }
        }
        contact_sample_time = elapsed;
        maximum_task_progress = std::max(
            maximum_task_progress,
            state.foot_positions_world[kRealtimeMovingFoot][0] -
                initial_foot_x);
        ++wbc_ticks;
        finish_execution_phase(
            ExecutionPhase::HOUSEKEEPING, phase_started, tick_timing);
        const auto tick_finished = phase_started.wall;
        const double complete_tick_wall_ms =
            std::chrono::duration<double, std::milli>(
                tick_finished - tick_started).count();
        const double scheduled_response_wall_ms =
            std::chrono::duration<double, std::milli>(
                tick_finished - scheduled_tick_start).count();
        complete_tick_work_ms.push_back(complete_tick_wall_ms);
        scheduled_tick_response_ms.push_back(scheduled_response_wall_ms);
        for (std::size_t phase = 0;
             phase < realtime_execution_detail::kExecutionPhaseCount;
             ++phase) {
            phase_wall_ms[phase].push_back(tick_timing[phase].wall_ms);
            if (realtime_execution_detail::has_thread_cpu_time(
                    tick_timing[phase])) {
                phase_thread_cpu_ms[phase].push_back(
                    tick_timing[phase].thread_cpu_ms);
            }
        }
        const double deadline_ms = 1000.0 * kWbcPeriod;
        const auto complete_miss =
            realtime_execution_detail::classify_deadline_miss(
                tick_timing, complete_tick_wall_ms,
                complete_tick_wall_ms, deadline_ms);
        const auto scheduled_miss =
            realtime_execution_detail::classify_deadline_miss(
                tick_timing, complete_tick_wall_ms,
                scheduled_response_wall_ms, deadline_ms);
        ++complete_tick_miss_counts[static_cast<std::size_t>(complete_miss)];
        ++scheduled_response_miss_counts[
            static_cast<std::size_t>(scheduled_miss)];
        if (tick_finished > next_wall_tick) {
            ++wall_wbc_deadline_misses;
        } else {
            std::this_thread::sleep_until(next_wall_tick);
        }
    }

    const auto wall_loop_end = std::chrono::steady_clock::now();
    const double wall_loop_seconds = std::chrono::duration<double>(
        wall_loop_end - wall_loop_start).count();
    const double actual_wall_tick_rate = wall_loop_seconds > 0.0
        ? static_cast<double>(wbc_ticks) / wall_loop_seconds
        : 0.0;
    planner_worker.stop();
    RealtimePublicationDecision final_decision;
    if (planner_worker.take_result(final_decision)) {
        const RealtimePlannerResult& update = final_decision.result;
        exact_hessian_analytic_calls +=
            update.stats.exact_hessian_analytic_calls;
        exact_hessian_fd_calls += update.stats.exact_hessian_fd_calls;
        planner_request_periods.record_shift_result(update.shift_consumed);
        const auto final_timing =
            realtime_execution_detail::evaluate_publication_timing(
                final_decision.requested_at, final_decision.deadline,
                std::chrono::steady_clock::now(), final_decision.stale);
        warm_end_to_end_ms.push_back(final_timing.request_to_handoff_ms);
        if (final_decision.independently_valid) ++usable_updates;
        if (final_timing.stale) ++deadline_misses;
        if (!final_decision.independently_valid)
            ++invalid_candidate_rejections;
        if (final_timing.stale) ++stale_candidate_rejections;
        if (update.published && !final_decision.task_metadata_valid)
            ++task_metadata_rejections;
        if (update.published && !final_decision.independently_valid)
            ++invalid_publications;
        if (update.published && final_timing.stale)
            ++stale_publications;
    }
    bool planner_scheduling_reported = false;
    ThreadSchedulingResult planner_scheduling;
    planner_worker.scheduling_status(
        planner_scheduling_reported, planner_scheduling);

    if (adapter.read_whole_body_state(state) != Status::SUCCESS) return 1;
    maximum_task_progress = std::max(
        maximum_task_progress,
        state.foot_positions_world[kRealtimeMovingFoot][0] - initial_foot_x);
    const double route_progress =
        state.base.position_world[0] - initial_base_x;
    const double simulation_duration = data->time - start_time;
    bool missing_touchdown = false;
    double minimum_clearance = 1e300;
    double maximum_landing_error = 0.0;
    double maximum_slip = 0.0;
    int swing_events = 0;
    int contact_tasks_commanded = 0;
    int contact_tasks_completed = 0;
    int completed_gait_cycles = 0;
    int pending_contact_tasks = 0;
    int right_censored_contact_tasks = 0;
    int unassigned_contact_events = 0;
    int task_ledger_mismatches = 0;
    int completed_tasks_per_foot[kNumFeet] = {};
    constexpr double kTaskScheduleTolerance =
        kPlannerPeriod + kWbcPeriod;
    if (sustained_mode) {
        std::size_t next_event_index[kNumFeet] = {};
        for (std::size_t ledger_index = 0;
             ledger_index < task_ledger.size(); ++ledger_index) {
            const RealtimeContactTask& task = task_ledger[ledger_index].task;
            const int foot = task.moving_foot;
            const double expected_liftoff =
                task.start_knot * kRealtimeTimeStep;
            const double expected_touchdown =
                task.touchdown_knot * kRealtimeTimeStep;
            const bool due = expected_touchdown <=
                simulation_duration + 0.5 * kWbcPeriod;
            const bool right_censored = !due;
            if (right_censored) {
                ++right_censored_contact_tasks;
                if (ledger_index + 1 != task_ledger.size())
                    ++task_ledger_mismatches;
            }

            const FootReplayMetrics& metrics = foot_metrics[foot];
            const bool event_bound =
                next_event_index[foot] < metrics.events.size();
            const std::size_t event_index = next_event_index[foot];
            const SwingReplayEvent* event = event_bound
                ? &metrics.events[next_event_index[foot]++] : nullptr;
            const double planned_liftoff = event
                ? event->planned_liftoff_time : -1.0;
            const double measured_liftoff = event
                ? event->measured_liftoff_time : -1.0;
            const double planned_touchdown = event
                ? event->planned_touchdown_time : -1.0;
            const double measured_touchdown = event
                ? event->measured_touchdown_time : -1.0;
            const bool liftoff_matches = event &&
                std::fabs(planned_liftoff - expected_liftoff) <=
                    kTaskScheduleTolerance;
            const bool touchdown_matches = event &&
                planned_touchdown >= 0.0 &&
                std::fabs(planned_touchdown - expected_touchdown) <=
                    kTaskScheduleTolerance;
            const bool schedule_matches = due
                ? liftoff_matches && touchdown_matches
                : (!event && expected_liftoff > simulation_duration) ||
                  (liftoff_matches &&
                   (planned_touchdown < 0.0 || touchdown_matches));
            if (!schedule_matches) ++task_ledger_mismatches;
            const bool completed = due && event && schedule_matches &&
                event->measured_unload() && measured_touchdown >= 0.0;
            if (due) {
                ++contact_tasks_commanded;
                if (completed) {
                    ++contact_tasks_completed;
                    ++completed_tasks_per_foot[foot];
                    minimum_clearance = std::min(
                        minimum_clearance, event->maximum_clearance);
                    maximum_landing_error = std::max(
                        maximum_landing_error, event->landing_error());
                    maximum_slip = std::max(
                        maximum_slip, event->maximum_post_touchdown_slip);
                } else {
                    ++pending_contact_tasks;
                }
            }
            std::printf(
                "real-time task ledger event: task_id=%llu "
                "task_sequence=%llu moving_foot=%d target_x=%.6f "
                "expected_liftoff=%.6f expected_touchdown=%.6f "
                "event_bound=%d event_index=%zu planned_liftoff=%.6f "
                "measured_liftoff=%.6f planned_touchdown=%.6f "
                "measured_touchdown=%.6f clearance=%.6f "
                "landing_error=%.6f slip=%.6f due=%d completed=%d "
                "right_censored=%d missing=%d schedule_match=%d\n",
                static_cast<unsigned long long>(task.id),
                static_cast<unsigned long long>(task.id), foot,
                task.target_x, expected_liftoff, expected_touchdown,
                event_bound ? 1 : 0, event_index, planned_liftoff,
                measured_liftoff, planned_touchdown, measured_touchdown,
                event ? event->maximum_clearance : 0.0,
                completed ? event->landing_error() : 0.0,
                event ? event->maximum_post_touchdown_slip : 0.0,
                due ? 1 : 0, completed ? 1 : 0,
                right_censored ? 1 : 0,
                due && !completed ? 1 : 0,
                schedule_matches ? 1 : 0);
        }
        for (int foot = 0; foot < kNumFeet; ++foot) {
            const FootReplayMetrics& metrics = foot_metrics[foot];
            for (std::size_t event_index = next_event_index[foot];
                 event_index < metrics.events.size(); ++event_index) {
                const SwingReplayEvent& event = metrics.events[event_index];
                ++unassigned_contact_events;
                std::printf(
                    "real-time unassigned execution event: foot=%d "
                    "event_index=%zu planned_liftoff=%.6f "
                    "planned_touchdown=%.6f measured_touchdown=%.6f\n",
                    foot, event_index, event.planned_liftoff_time,
                    event.planned_touchdown_time,
                    event.measured_touchdown_time);
            }
        }
        swing_events = static_cast<int>(task_ledger.size());
        missing_touchdown = pending_contact_tasks > 0;
        completed_gait_cycles = completed_tasks_per_foot[0];
        for (int foot = 1; foot < kNumFeet; ++foot) {
            completed_gait_cycles = std::min(
                completed_gait_cycles, completed_tasks_per_foot[foot]);
        }
    } else {
        const FootReplayMetrics& metrics =
            foot_metrics[kRealtimeMovingFoot];
        swing_events = static_cast<int>(metrics.events.size());
        missing_touchdown = metrics.events.empty();
        for (std::size_t event_index = 0;
             event_index < metrics.events.size(); ++event_index) {
            const SwingReplayEvent& event = metrics.events[event_index];
            const bool event_missing = !event.measured_unload() ||
                event.planned_touchdown_time < 0.0 ||
                event.measured_touchdown_time < 0.0;
            missing_touchdown = missing_touchdown || event_missing;
            minimum_clearance = std::min(
                minimum_clearance, event.maximum_clearance);
            maximum_landing_error = std::max(
                maximum_landing_error, event.landing_error());
            maximum_slip = std::max(
                maximum_slip, event.maximum_post_touchdown_slip);
            std::printf(
                "real-time execution event: index=%zu unload=%d "
                "planned_touchdown=%.6f measured_touchdown=%.6f "
                "clearance=%.6f landing_error=%.6f slip=%.6f missing=%d\n",
                event_index, event.measured_unload() ? 1 : 0,
                event.planned_touchdown_time, event.measured_touchdown_time,
                event.maximum_clearance, event.landing_error(),
                event.maximum_post_touchdown_slip, event_missing ? 1 : 0);
        }
        contact_tasks_commanded = 1;
        contact_tasks_completed = !missing_touchdown &&
            maximum_task_progress >= 0.06 ? 1 : 0;
    }
    if (contact_tasks_completed == 0) minimum_clearance = 0.0;
    const int accepted_task_transitions = sustained_mode
        ? std::max(0, static_cast<int>(task_ledger.size()) - 1)
        : 0;
    const bool task_ledger_integrity_gate = !sustained_mode ||
        (pending_contact_tasks == 0 &&
         right_censored_contact_tasks <= 1 &&
         unassigned_contact_events == 0 && task_ledger_mismatches == 0);
    const bool sustained_contact_execution = sustained_mode &&
        task_ledger_integrity_gate &&
        realtime_execution_detail::sustained_contact_qualification_gate(
            simulation_duration, contact_tasks_commanded,
            contact_tasks_completed, completed_gait_cycles,
            route_progress);
    const double usable_fraction = replan_opportunities > 0
        ? static_cast<double>(usable_updates) / replan_opportunities
        : 0.0;
    const double accepted_publication_fraction = replan_opportunities > 0
        ? static_cast<double>(accepted_publications) / replan_opportunities
        : 0.0;
    const double p99_ms = percentile_ms(warm_end_to_end_ms, 0.99);
    const realtime_execution_detail::TimingSummary wbc_compute =
        realtime_execution_detail::summarize_timing_ms(
            wbc_compute_ms, 1000.0 * kWbcPeriod);
    const realtime_execution_detail::TimingSummary complete_tick_work =
        realtime_execution_detail::summarize_timing_ms(
            complete_tick_work_ms, 1000.0 * kWbcPeriod);
    const realtime_execution_detail::TimingSummary scheduled_tick_response =
        realtime_execution_detail::summarize_timing_ms(
            scheduled_tick_response_ms, 1000.0 * kWbcPeriod);
    const realtime_execution_detail::TimingSummary scheduling_lateness =
        realtime_execution_detail::summarize_timing_ms(
            wbc_scheduling_lateness_ms, 1000.0 * kWbcPeriod);
    const char* phase_names[] = {
        "mj_step1", "state_reference_contact", "wbc_apply", "mj_step2",
        "housekeeping"};
    for (std::size_t phase = 0;
         phase < realtime_execution_detail::kExecutionPhaseCount; ++phase) {
        const auto wall = realtime_execution_detail::summarize_timing_ms(
            phase_wall_ms[phase], 1000.0 * kWbcPeriod);
        const auto cpu = realtime_execution_detail::summarize_timing_ms(
            phase_thread_cpu_ms[phase], 1000.0 * kWbcPeriod);
        std::printf(
            "real-time execution phase timing: phase=%s samples=%d "
            "wall_p50_ms=%.6f wall_p90_ms=%.6f wall_p99_ms=%.6f "
            "wall_max_ms=%.6f thread_cpu_samples=%d "
            "thread_cpu_p50_ms=%.6f thread_cpu_p90_ms=%.6f "
            "thread_cpu_p99_ms=%.6f thread_cpu_max_ms=%.6f\n",
            phase_names[phase], wall.samples, wall.p50_ms, wall.p90_ms,
            wall.p99_ms, wall.maximum_ms, cpu.samples, cpu.p50_ms,
            cpu.p90_ms, cpu.p99_ms, cpu.maximum_ms);
    }
    const std::array<int, realtime_execution_detail::kDeadlineMissCauseCount>*
        miss_counts[] = {
            &complete_tick_miss_counts, &scheduled_response_miss_counts};
    const char* miss_scopes[] = {
        "complete_tick_work", "scheduled_tick_response"};
    for (int scope = 0; scope < 2; ++scope) {
        const auto& counts = *miss_counts[scope];
        std::printf(
            "real-time execution miss attribution: scope=%s misses=%d "
            "mujoco_physics=%d wbc_or_control_work=%d "
            "scheduling_or_blocking=%d thread_cpu_time_unavailable=%d\n",
            miss_scopes[scope], wbc_ticks - counts[0], counts[1], counts[2],
            counts[3], counts[4]);
    }
    constexpr double kMinimumRealtimeDeadlineFraction = 0.999;
    const bool wbc_compute_gate =
        realtime_execution_detail::meets_deadline_fraction(
            wbc_compute, kMinimumRealtimeDeadlineFraction);
    const bool complete_tick_work_gate =
        realtime_execution_detail::meets_deadline_fraction(
            complete_tick_work, kMinimumRealtimeDeadlineFraction);
    const bool scheduled_tick_response_gate =
        realtime_execution_detail::meets_deadline_fraction(
            scheduled_tick_response, kMinimumRealtimeDeadlineFraction);
    constexpr double kWallRateTargetHz = 500.0;
    constexpr double kWallRateRelativeTolerance = 0.01;
    const bool wall_rate_gate =
        realtime_execution_detail::meets_rate_target(
            actual_wall_tick_rate, kWallRateTargetHz,
            kWallRateRelativeTolerance);
    const bool publication_rate_gate =
        accepted_publication_fraction + 1e-15 >= 0.99;
    const bool logical_thread_isolation_applied =
        cpu_assignment.separate_logical_cpus &&
        wbc_scheduling.affinity_applied && planner_scheduling_reported &&
        planner_scheduling.affinity_applied;
#if defined(__linux__)
    const bool realtime_priorities_applied =
        wbc_scheduling.priority_applied && planner_scheduling_reported &&
        planner_scheduling.priority_applied;
#else
    const bool realtime_priorities_applied = false;
#endif
    const bool physical_core_isolation_verified =
        logical_thread_isolation_applied && separate_physical_cores;
    const bool environment_limited = realtime_environment_is_wsl();
    const bool paper_grade_scheduling_gate =
        realtime_execution_detail::paper_grade_scheduling_gate(
            physical_core_isolation_verified,
            realtime_priorities_applied, environment_limited);
    const double base_position_rms = std::sqrt(
        squared_base_position_error / std::max(wbc_ticks, 1));
    const bool contact_execution_quality_gate = sustained_mode
        ? task_ledger_integrity_gate && contact_tasks_completed > 0 &&
          !missing_touchdown &&
          minimum_clearance >= 0.02 && maximum_landing_error <= 0.03 &&
          maximum_slip <= 0.02
        : maximum_task_progress >= 0.06 && !missing_touchdown &&
          minimum_clearance >= 0.02 && maximum_landing_error <= 0.03 &&
          maximum_slip <= 0.02;
    const bool execution_gates_passed =
        warm_attempts > 0 && usable_fraction >= 0.99 &&
        publication_rate_gate &&
        p99_ms <= kPlannerP99GateMs &&
        maximum_consecutive_fallbacks <= 1 && expired_plan_ticks == 0 &&
        contact_execution_quality_gate && saturated_joint_ticks == 0 &&
        invalid_publications == 0 && stale_publications == 0 &&
        task_metadata_rejections == 0 &&
        fresh_task_transition_witnesses == accepted_task_transitions &&
        exact_hessian_analytic_calls == 0 && exact_hessian_fd_calls == 0 &&
        wbc_compute_gate && complete_tick_work_gate &&
        scheduled_tick_response_gate && wall_rate_gate;
    const bool integration_accepted =
        execution_gates_passed && paper_grade_scheduling_gate;
    const bool accepted = integration_accepted && sustained_contact_execution;
    const char* paper_grade_verdict = environment_limited
        ? "environment_limited_wsl"
        : !execution_gates_passed
            ? "execution_gates_failed"
            : !paper_grade_scheduling_gate
                ? "scheduling_not_applied"
                : !sustained_contact_execution
                    ? sustained_mode
                        ? "sustained_contact_gate_failed"
                        : "single_task_integration_only"
                    : "paper_grade_pass";
    std::printf(
        "real-time execution gates: scheduling=background_worker "
        "wall_concurrent=1 planner_updates_enabled=%d sustained_mode=%d "
        "planner_horizon_s=2.500 planner_knots=50 planner_dt_s=0.050 "
        "planner_rate_hz=5 wbc_rate_hz=500 duration_s=%.6f "
        "duration=%.6f "
        "mujoco_version=%s "
        "wall_duration=%.6f wbc_ticks=%d "
        "logical_wbc_rate_hz=500 actual_wall_tick_rate_hz=%.3f "
        "wall_wbc_deadline_misses=%d "
        "wbc_compute_samples=%d wbc_compute_within_2ms=%d "
        "wbc_compute_within_2ms_fraction=%.6f "
        "wbc_compute_p50_ms=%.6f wbc_compute_p90_ms=%.6f "
        "wbc_compute_p99_ms=%.6f wbc_compute_max_ms=%.6f "
        "wbc_compute_gate=%d "
        "complete_tick_work_samples=%d complete_tick_work_within_2ms=%d "
        "complete_tick_work_within_2ms_fraction=%.6f "
        "complete_tick_work_p50_ms=%.6f "
        "complete_tick_work_p90_ms=%.6f "
        "complete_tick_work_p99_ms=%.6f "
        "complete_tick_work_max_ms=%.6f complete_tick_work_gate=%d "
        "scheduled_tick_response_samples=%d "
        "scheduled_tick_response_within_2ms=%d "
        "scheduled_tick_response_within_2ms_fraction=%.6f "
        "scheduled_tick_response_p50_ms=%.6f "
        "scheduled_tick_response_p90_ms=%.6f "
        "scheduled_tick_response_p99_ms=%.6f "
        "scheduled_tick_response_max_ms=%.6f "
        "scheduled_tick_response_gate=%d "
        "wall_rate_target_hz=500 wall_rate_tolerance_fraction=0.010000 "
        "wall_rate_gate=%d separate_logical_cpus=%d "
        "wbc_affinity_applied=%d wbc_priority_applied=%d "
        "planner_scheduling_reported=%d planner_affinity_applied=%d "
        "planner_priority_applied=%d logical_thread_isolation_applied=%d "
        "physical_core_isolation_verified=%d "
        "realtime_priorities_applied=%d environment=%s "
        "environment_limited=%d paper_grade_scheduling_gate=%d "
        "scheduling_lateness_p50_ms=%.6f "
        "scheduling_lateness_p90_ms=%.6f "
        "scheduling_lateness_p99_ms=%.6f "
        "scheduling_lateness_max_ms=%.6f replan_opportunities=%d "
        "worker_busy_skips=%d maximum_request_elapsed_periods=%d "
        "warm_attempts=%d usable_updates=%d "
        "usable_fraction=%.6f accepted_publications=%d "
        "accepted_publication_fraction=%.6f publication_rate_gate=%d "
        "fallbacks=%d "
        "max_consecutive_fallbacks=%d deadline_misses=%d "
        "invalid_candidate_rejections=%d stale_candidate_rejections=%d "
        "invalid_publications=%d stale_publications=%d "
        "task_metadata_rejections=%d accepted_task_transitions=%d "
        "fresh_task_transition_witnesses=%d active_task_id=%llu "
        "active_moving_foot=%d "
        "exact_hessian_analytic_calls=%llu exact_hessian_fd_calls=%llu "
        "zero_hessian_gate=%d "
        "expired_plan_ticks=%d "
        "planner_latency_scope=request_to_wbc_handoff "
        "planner_p99_ms=%.3f "
        "task_progress=%.6f route_progress_m=%.6f swing_events=%d "
        "pending_contact_tasks=%d right_censored_contact_tasks=%d "
        "unassigned_contact_events=%d task_ledger_mismatches=%d "
        "task_ledger_integrity_gate=%d missing_touchdown=%d "
        "contact_tasks_commanded=%d contact_tasks_completed=%d "
        "completed_gait_cycles=%d duration_gate=%d "
        "sustained_contact_execution=%d execution_evidence_scope=%s "
        "minimum_clearance=%.6f maximum_landing_error=%.6f "
        "maximum_slip=%.6f saturated_joint_ticks=%d "
        "base_position_rms=%.6f contact_execution_quality_gate=%d "
        "execution_gates_passed=%d "
        "single_task_integration_accepted=%d "
        "sustained_integration_accepted=%d "
        "paper_grade_verdict=%s accepted=%d\n",
        disable_planner_updates ? 0 : 1, sustained_mode ? 1 : 0,
        simulation_duration, requested_duration, mj_versionString(),
        wall_loop_seconds, wbc_ticks,
        actual_wall_tick_rate, wall_wbc_deadline_misses,
        wbc_compute.samples, wbc_compute.within_deadline,
        wbc_compute.within_deadline_fraction, wbc_compute.p50_ms,
        wbc_compute.p90_ms, wbc_compute.p99_ms, wbc_compute.maximum_ms,
        wbc_compute_gate ? 1 : 0, complete_tick_work.samples,
        complete_tick_work.within_deadline,
        complete_tick_work.within_deadline_fraction,
        complete_tick_work.p50_ms, complete_tick_work.p90_ms,
        complete_tick_work.p99_ms, complete_tick_work.maximum_ms,
        complete_tick_work_gate ? 1 : 0, scheduled_tick_response.samples,
        scheduled_tick_response.within_deadline,
        scheduled_tick_response.within_deadline_fraction,
        scheduled_tick_response.p50_ms, scheduled_tick_response.p90_ms,
        scheduled_tick_response.p99_ms, scheduled_tick_response.maximum_ms,
        scheduled_tick_response_gate ? 1 : 0, wall_rate_gate ? 1 : 0,
        cpu_assignment.separate_logical_cpus ? 1 : 0,
        wbc_scheduling.affinity_applied ? 1 : 0,
        wbc_scheduling.priority_applied ? 1 : 0,
        planner_scheduling_reported ? 1 : 0,
        planner_scheduling.affinity_applied ? 1 : 0,
        planner_scheduling.priority_applied ? 1 : 0,
        logical_thread_isolation_applied ? 1 : 0,
        physical_core_isolation_verified ? 1 : 0,
        realtime_priorities_applied ? 1 : 0,
        environment_limited ? "wsl" : realtime_platform_name(),
        environment_limited ? 1 : 0,
        paper_grade_scheduling_gate ? 1 : 0,
        scheduling_lateness.p50_ms,
        scheduling_lateness.p90_ms, scheduling_lateness.p99_ms,
        scheduling_lateness.maximum_ms,
        replan_opportunities, worker_busy_skips,
        maximum_request_elapsed_periods, warm_attempts, usable_updates,
        usable_fraction, accepted_publications,
        accepted_publication_fraction, publication_rate_gate ? 1 : 0,
        fallback_updates,
        maximum_consecutive_fallbacks, deadline_misses,
        invalid_candidate_rejections, stale_candidate_rejections,
        invalid_publications, stale_publications,
        task_metadata_rejections, accepted_task_transitions,
        fresh_task_transition_witnesses,
        static_cast<unsigned long long>(active_task->id),
        active_task->moving_foot,
        static_cast<unsigned long long>(exact_hessian_analytic_calls),
        static_cast<unsigned long long>(exact_hessian_fd_calls),
        exact_hessian_analytic_calls == 0 && exact_hessian_fd_calls == 0 ? 1 : 0,
        expired_plan_ticks, p99_ms,
        maximum_task_progress, route_progress, swing_events,
        pending_contact_tasks, right_censored_contact_tasks,
        unassigned_contact_events, task_ledger_mismatches,
        task_ledger_integrity_gate ? 1 : 0, missing_touchdown ? 1 : 0,
        contact_tasks_commanded, contact_tasks_completed,
        completed_gait_cycles, simulation_duration >= 20.0 ? 1 : 0,
        sustained_contact_execution ? 1 : 0,
        sustained_mode ? "sustained_multi_contact" : "single_contact_task",
        minimum_clearance,
        maximum_landing_error, maximum_slip, saturated_joint_ticks,
        base_position_rms, contact_execution_quality_gate ? 1 : 0,
        execution_gates_passed ? 1 : 0,
        !sustained_mode && integration_accepted ? 1 : 0,
        sustained_mode && accepted ? 1 : 0, paper_grade_verdict,
        accepted ? 1 : 0);
    return accepted ? 0 : 1;
}

}  // namespace

int main(int argc, char** argv) {
    std::setvbuf(stdout, nullptr, _IOLBF, 0);
    if (argc < 2) {
        std::printf(
            "usage: quadruped_cito_mujoco <go1-scene.xml> "
            "[--terrain flat|slope|cross-slope|sinusoidal|smooth_step|"
            "random_smooth] "
            "[--terrain-seed <seed>] [--terrain-amplitude <meters>] "
            "[--slope-x <grade>] [--slope-y <grade>] "
            "[--step-height <meters>] "
            "[--stand <seconds> | --replay-plan <plan.txt> | "
            "--replay-snapshot <snapshot.txt> | "
            "--replan-twice | --replan-count <count> | "
            "--realtime-replan-duration <seconds>] "
            "[--realtime-disable-planner] [--realtime-sustained] "
            "[--seed-policy alternating|forward|reverse|left_first|"
            "right_first|cycling|multistart] "
            "[--mass-scale <scale>] [--friction-scale <scale>] "
            "[--push <fx> <fy> <fz> <start> <duration>] "
            "[--force-reference cito|uniform] [--allow-timing-miss] "
            "[--disable-early-contact-feedback] "
            "[--fixed-stabilization-handoff] "
            "[--enable-late-touchdown-search] "
            "[--stabilization-duration <minimum-seconds>] "
            "[--maximum-stabilization-duration <seconds>] "
            "[--write-boundary-snapshots <path-prefix>] "
            "[--snapshot-base-z-kp <gain>] "
            "[--trace-execution]\n");
        return 2;
    }
    std::string terrain_argument = "flat";
    unsigned int terrain_seed = 0;
    double terrain_amplitude = 0.02;
    double slope_x = 0.0;
    double slope_y = 0.0;
    double step_height = 0.03;
    bool slope_x_set = false;
    bool slope_y_set = false;
    std::string plan_path;
    std::string replay_snapshot_path;
    bool run_standing = false;
    bool run_plan = false;
    bool run_snapshot = false;
    bool run_replanning = false;
    bool run_realtime_mode = false;
    bool realtime_disable_planner = false;
    bool realtime_sustained = false;
    int replan_count = 2;
    double realtime_replan_duration = 0.0;
    ReplanSeedPolicy replan_seed_policy = ReplanSeedPolicy::ALTERNATING;
    double standing_duration = 0.0;
    double snapshot_base_z_kp = 0.0;
    bool snapshot_base_z_kp_set = false;
    ReplayOptions replay_options;
    bool snapshot_incompatible_option = false;
    for (int argument = 2; argument < argc; ++argument) {
        const std::string option = argv[argument];
        if (option == "--terrain" && argument + 1 < argc) {
            snapshot_incompatible_option = true;
            terrain_argument = argv[++argument];
        } else if (option == "--terrain-seed" && argument + 1 < argc) {
            snapshot_incompatible_option = true;
            terrain_seed = static_cast<unsigned int>(
                std::strtoul(argv[++argument], nullptr, 10));
        } else if (option == "--terrain-amplitude" &&
                   argument + 1 < argc) {
            snapshot_incompatible_option = true;
            terrain_amplitude = std::atof(argv[++argument]);
        } else if (option == "--slope-x" && argument + 1 < argc) {
            snapshot_incompatible_option = true;
            slope_x = std::atof(argv[++argument]);
            slope_x_set = true;
        } else if (option == "--slope-y" && argument + 1 < argc) {
            snapshot_incompatible_option = true;
            slope_y = std::atof(argv[++argument]);
            slope_y_set = true;
        } else if (option == "--step-height" && argument + 1 < argc) {
            snapshot_incompatible_option = true;
            step_height = std::atof(argv[++argument]);
        } else if (option == "--stand" && argument + 1 < argc &&
                   !run_standing && !run_plan && !run_snapshot &&
                   !run_replanning && !run_realtime_mode) {
            run_standing = true;
            standing_duration = std::atof(argv[++argument]);
        } else if (option == "--replay-plan" && argument + 1 < argc &&
                   !run_standing && !run_plan && !run_snapshot &&
                   !run_replanning && !run_realtime_mode) {
            run_plan = true;
            plan_path = argv[++argument];
        } else if (option == "--replay-snapshot" &&
                   argument + 1 < argc && !run_standing && !run_plan &&
                   !run_snapshot && !run_replanning &&
                   !run_realtime_mode) {
            run_snapshot = true;
            replay_snapshot_path = argv[++argument];
        } else if (option == "--replan-twice" &&
                   !run_standing && !run_plan && !run_snapshot &&
                   !run_replanning && !run_realtime_mode) {
            run_replanning = true;
            replan_count = 2;
        } else if (option == "--replan-count" && argument + 1 < argc &&
                   !run_standing && !run_plan && !run_snapshot &&
                   !run_replanning && !run_realtime_mode) {
            run_replanning = true;
            replan_count = std::atoi(argv[++argument]);
        } else if (option == "--realtime-replan-duration" &&
                   argument + 1 < argc && !run_standing && !run_plan &&
                   !run_snapshot && !run_replanning &&
                   !run_realtime_mode) {
            run_realtime_mode = true;
            realtime_replan_duration = std::atof(argv[++argument]);
        } else if (option == "--realtime-disable-planner") {
            realtime_disable_planner = true;
        } else if (option == "--realtime-sustained") {
            realtime_sustained = true;
        } else if (option == "--seed-policy" && argument + 1 < argc) {
            snapshot_incompatible_option = true;
            if (!parse_replan_seed_policy(
                    argv[++argument], replan_seed_policy)) {
                std::printf("invalid replanning seed policy\n");
                return 2;
            }
        } else if (option == "--mass-scale" && argument + 1 < argc) {
            snapshot_incompatible_option = true;
            replay_options.mass_scale = std::atof(argv[++argument]);
        } else if (option == "--friction-scale" && argument + 1 < argc) {
            snapshot_incompatible_option = true;
            replay_options.friction_scale = std::atof(argv[++argument]);
        } else if (option == "--push" && argument + 5 < argc) {
            snapshot_incompatible_option = true;
            for (int axis = 0; axis < 3; ++axis)
                replay_options.push_force_world[axis] =
                    std::atof(argv[++argument]);
            replay_options.push_start = std::atof(argv[++argument]);
            replay_options.push_duration = std::atof(argv[++argument]);
        } else if (option == "--force-reference" && argument + 1 < argc) {
            snapshot_incompatible_option = true;
            const std::string mode = argv[++argument];
            if (mode == "cito") {
                replay_options.force_reference = ForceReferenceMode::CITO;
            } else if (mode == "uniform") {
                replay_options.force_reference = ForceReferenceMode::UNIFORM;
            } else {
                std::printf("invalid force-reference mode\n");
                return 2;
            }
        } else if (option == "--allow-timing-miss") {
            snapshot_incompatible_option = true;
            replay_options.require_timing = false;
        } else if (option == "--disable-early-contact-feedback") {
            snapshot_incompatible_option = true;
            replay_options.early_contact_feedback = false;
        } else if (option == "--fixed-stabilization-handoff") {
            snapshot_incompatible_option = true;
            replay_options.confirmed_stabilization = false;
        } else if (option == "--enable-late-touchdown-search") {
            snapshot_incompatible_option = true;
            replay_options.late_touchdown_search = true;
        } else if (option == "--stabilization-duration" &&
                   argument + 1 < argc) {
            snapshot_incompatible_option = true;
            replay_options.stabilization_duration =
                std::atof(argv[++argument]);
        } else if (option == "--maximum-stabilization-duration" &&
                   argument + 1 < argc) {
            snapshot_incompatible_option = true;
            replay_options.maximum_stabilization_duration =
                std::atof(argv[++argument]);
        } else if (option == "--trace-execution") {
            replay_options.trace_execution = true;
        } else if (option == "--snapshot-base-z-kp" &&
                   argument + 1 < argc) {
            snapshot_base_z_kp = std::atof(argv[++argument]);
            snapshot_base_z_kp_set = true;
        } else if (option == "--write-boundary-snapshots" &&
                   argument + 1 < argc) {
            replay_options.snapshot_prefix = argv[++argument];
        } else {
            std::printf("invalid replay arguments\n");
            return 2;
        }
    }
    MujocoReplaySnapshot replay_snapshot;
    if (run_snapshot) {
        if (snapshot_incompatible_option) {
            std::printf(
                "snapshot replay uses stored terrain/controller options; "
                "only --trace-execution and --snapshot-base-z-kp may "
                "override them\n");
            return 2;
        }
        if (read_mujoco_replay_snapshot(
                replay_snapshot_path.c_str(), replay_snapshot) !=
            nmpc::Status::SUCCESS) {
            std::printf("failed to read MuJoCo boundary snapshot: %s\n",
                        replay_snapshot_path.c_str());
            return 2;
        }
        terrain_seed = replay_snapshot.configuration.terrain_seed;
        terrain_amplitude =
            replay_snapshot.configuration.terrain_amplitude_argument;
        replay_options = replay_options_from_snapshot(
            replay_snapshot.configuration, replay_options.trace_execution);
        terrain_argument = terrain_name(replay_snapshot.configuration.terrain.kind);
    }
    quadruped_cito::SharedTerrain terrain;
    if (run_snapshot) {
        terrain = replay_snapshot.configuration.terrain;
    } else if (!parse_terrain(
                   terrain_argument, terrain_seed, terrain_amplitude,
                   slope_x, slope_y, step_height, slope_x_set, slope_y_set,
                   terrain)) {
        std::printf("invalid terrain or replay arguments\n");
        return 2;
    }
    if (
         !(terrain_amplitude > 0.0) ||
        !std::isfinite(slope_x) || !std::isfinite(slope_y) ||
        !(step_height > 0.0) ||
        (run_standing && !(standing_duration > 0.0)) ||
        (run_replanning && (replan_count < 1 || replan_count > 32)) ||
        (run_realtime_mode && !(realtime_replan_duration > 0.0)) ||
        (realtime_disable_planner && !run_realtime_mode) ||
        (realtime_sustained && !run_realtime_mode) ||
        (!run_replanning &&
         replan_seed_policy != ReplanSeedPolicy::ALTERNATING) ||
        (!replay_options.snapshot_prefix.empty() && !run_replanning) ||
        (snapshot_base_z_kp_set &&
         (!run_snapshot || !(snapshot_base_z_kp > 0.0) ||
          mju_isBad(snapshot_base_z_kp))) ||
        !(replay_options.mass_scale > 0.0) ||
        !(replay_options.friction_scale > 0.0) ||
        replay_options.push_start < 0.0 ||
        replay_options.push_duration < 0.0 ||
        !(replay_options.stabilization_duration > 0.0) ||
        replay_options.maximum_stabilization_duration <
            replay_options.stabilization_duration) {
        std::printf("invalid terrain or replay arguments\n");
        return 2;
    }

    char error[1024] = {};
    mjModel* model = mj_loadXML(argv[1], nullptr, error, sizeof(error));
    if (!model) {
        std::printf("failed to load Go1 model: %s\n", error);
        return 1;
    }
    quadruped_cito::MujocoTerrainReport terrain_report;
    if (quadruped_cito::configure_shared_terrain(
            model, terrain, &terrain_report) != nmpc::Status::SUCCESS) {
        std::printf("failed to configure shared %s terrain\n",
                    terrain_argument.c_str());
        mj_deleteModel(model);
        return 1;
    }
    const int terrain_geom = mj_name2id(model, mjOBJ_GEOM, "cito_terrain");
    if (terrain_geom < 0) {
        std::printf("MuJoCo terrain geom is missing\n");
        mj_deleteModel(model);
        return 1;
    }
    for (int friction = 0; friction < 3; ++friction) {
        model->geom_friction[3 * terrain_geom + friction] *=
            replay_options.friction_scale;
    }
    std::printf(
        "MuJoCo terrain: kind=%s seed=%u amplitude=%.6f grid=%dx%d "
        "height=[%.6f, %.6f] "
        "grid_error=%.3e\n",
        terrain_argument.c_str(), terrain_seed, terrain_amplitude,
        terrain_report.rows,
        terrain_report.columns, terrain_report.minimum_height,
        terrain_report.maximum_height, terrain_report.maximum_grid_error);
    if (quadruped_cito::configure_go1_torque_actuators(model) !=
        nmpc::Status::SUCCESS) {
        std::printf("failed to configure Go1 torque actuators\n");
        mj_deleteModel(model);
        return 1;
    }
    mjData* data = mj_makeData(model);
    if (!data) {
        std::printf("failed to allocate MuJoCo data\n");
        mj_deleteModel(model);
        return 1;
    }
    if (replay_options.mass_scale != 1.0) {
        for (int body = 1; body < model->nbody; ++body) {
            model->body_mass[body] *= replay_options.mass_scale;
            for (int axis = 0; axis < 3; ++axis)
                model->body_inertia[3 * body + axis] *=
                    replay_options.mass_scale;
        }
        mj_setConst(model, data);
    }

    quadruped_cito::MujocoGo1Adapter adapter(model, data);
    if (adapter.initialize() != nmpc::Status::SUCCESS) {
        std::printf("Go1 model does not satisfy the adapter naming contract\n");
        mj_deleteData(data);
        mj_deleteModel(model);
        return 1;
    }

    const int home_key = mj_name2id(model, mjOBJ_KEY, "home");
    if (home_key >= 0) mj_resetDataKeyframe(model, data, home_key);
    for (int actuator = 0; actuator < model->nu; ++actuator)
        data->ctrl[actuator] = 0.0;
    mj_forward(model, data);
    quadruped_cito::WholeBodyState state;
    quadruped_cito::MujocoFootContact contacts[
        quadruped_cito::kNumFeet];
    if (adapter.read_whole_body_state(state) != nmpc::Status::SUCCESS ||
        adapter.read_foot_contacts(contacts) != nmpc::Status::SUCCESS) {
        std::printf("failed to read initial Go1 state\n");
        mj_deleteData(data);
        mj_deleteModel(model);
        return 1;
    }

    std::printf(
        "MuJoCo Go1 ready: nq=%lld nv=%lld nu=%lld mass=%.6f kg "
        "base=(%.6f, %.6f, %.6f)\n",
        static_cast<long long>(model->nq),
        static_cast<long long>(model->nv),
        static_cast<long long>(model->nu), adapter.total_mass(),
        state.base.position_world[0], state.base.position_world[1],
        state.base.position_world[2]);
    for (int foot = 0; foot < quadruped_cito::kNumFeet; ++foot) {
        std::printf(
            "  foot %d: position=(%.6f, %.6f, %.6f) contact=%d "
            "normal_force=%.6f\n",
            foot, state.foot_positions_world[foot][0],
            state.foot_positions_world[foot][1],
            state.foot_positions_world[foot][2],
            contacts[foot].in_contact ? 1 : 0,
            contacts[foot].normal_force);
    }

    int replay_status = 0;
    if (run_standing) {
        replay_status = run_standing_replay(
            model, data, adapter, standing_duration);
    } else if (run_plan) {
        const int horizon = read_contact_plan_horizon(plan_path.c_str());
        if (horizon == 20) {
            replay_status = run_contact_plan_replay<20>(
                model, data, adapter, terrain, replay_options,
                plan_path.c_str());
        } else if (horizon == 50) {
            replay_status = run_contact_plan_replay<50>(
                model, data, adapter, terrain, replay_options,
                plan_path.c_str());
        } else if (horizon == 100) {
            replay_status = run_contact_plan_replay<100>(
                model, data, adapter, terrain, replay_options,
                plan_path.c_str());
        } else if (horizon == 220) {
            replay_status = run_contact_plan_replay<220>(
                model, data, adapter, terrain, replay_options,
                plan_path.c_str());
        } else {
            std::printf("unsupported ContactIPM plan horizon: %d\n", horizon);
            replay_status = 1;
        }
    } else if (run_snapshot) {
        const std::string snapshot_plan_path =
            paired_plan_path(replay_snapshot_path);
        if (snapshot_plan_path.empty()) {
            std::printf(
                "snapshot path must end in .snapshot to locate its plan\n");
            replay_status = 1;
        } else {
            replay_status = run_snapshot_replay(
                model, data, adapter, replay_snapshot, replay_options,
                snapshot_plan_path.c_str(), snapshot_base_z_kp_set,
                snapshot_base_z_kp);
        }
    } else if (run_replanning) {
        replay_status = run_replanned_course(
            model, data, adapter, terrain, replay_options, replan_count,
            replan_seed_policy, terrain_seed, terrain_amplitude);
    } else if (run_realtime_mode) {
        replay_status = run_realtime_replanning(
            model, data, adapter, terrain, replay_options,
            realtime_replan_duration, realtime_disable_planner,
            realtime_sustained);
    }

    mj_deleteData(data);
    mj_deleteModel(model);
    return replay_status;
}
