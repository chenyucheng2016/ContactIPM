#pragma once

#include <cmath>
#include <cstdint>
#include <cstdio>
#include <fstream>
#include <iomanip>
#include <limits>
#include <string>
#include <vector>

#include <mujoco/mujoco.h>

#include "examples/quadruped_cito/mujoco/mujoco_go1_adapter.hpp"
#include "examples/quadruped_cito/quadruped_cito_contact_feedback.hpp"

namespace quadruped_cito {

struct MujocoReplaySnapshotConfiguration {
    SharedTerrain terrain;
    unsigned int terrain_seed = 0;
    double terrain_amplitude_argument = 0.02;
    double mass_scale = 1.0;
    double friction_scale = 1.0;
    Vec<3> push_force_world;
    double push_start = 0.0;
    double push_duration = 0.0;
    int force_reference = 0;
    bool require_timing = true;
    bool early_contact_feedback = true;
    bool confirmed_stabilization = true;
    bool late_touchdown_search = false;
    bool trace_execution = false;
    double stabilization_duration = 2.0;
    double maximum_stabilization_duration = 3.0;
    int contact_execution_policy_version = 0;
    bool synchronized_contact_cache = false;
    bool history_aware_recontact = false;
    bool measured_load_torque_handoff = false;
    ConvexWBCParameters wbc_parameters;
    ContactExecutionFeedbackParameters contact_feedback_parameters;

    MujocoReplaySnapshotConfiguration() { push_force_world.zero(); }
};

struct MujocoReplaySnapshot {
    int segment = -1;
    std::uint64_t model_configuration_fingerprint = 0;
    std::uint64_t paired_plan_fingerprint = 0;
    int mujoco_version = 0;
    int state_signature = mjSTATE_INTEGRATION;
    int state_size = 0;
    long long nq = 0;
    long long nv = 0;
    long long na = 0;
    long long nu = 0;
    long long nbody = 0;
    long long njnt = 0;
    long long ngeom = 0;
    long long nmocap = 0;
    long long nuserdata = 0;
    long long npluginstate = 0;
    long long nhistory = 0;
    double timestep = 0.0;
    MujocoReplaySnapshotConfiguration configuration;
    MujocoFootContact contacts[kNumFeet];
    std::vector<mjtNum> state;
};

struct MujocoReplaySnapshotRestoreReport {
    bool contact_mask_matches = false;
    double maximum_contact_force_error = 0.0;
};

namespace replay_snapshot_detail {

inline bool finite(double value) { return std::isfinite(value); }

template <int Size>
inline bool finite_vector(const Vec<Size>& value) {
    for (int index = 0; index < Size; ++index) {
        if (!finite(value[index])) return false;
    }
    return true;
}

template <int Size>
inline bool nonnegative_vector(const Vec<Size>& value) {
    for (int index = 0; index < Size; ++index) {
        if (!(value[index] >= 0.0) || !finite(value[index])) return false;
    }
    return true;
}

template <int Size>
inline bool positive_vector(const Vec<Size>& value) {
    for (int index = 0; index < Size; ++index) {
        if (!(value[index] > 0.0) || !finite(value[index])) return false;
    }
    return true;
}

inline bool valid_terrain(const SharedTerrain& terrain) {
    const int kind = static_cast<int>(terrain.kind);
    return kind >= static_cast<int>(TerrainKind::FLAT) &&
           kind <= static_cast<int>(TerrainKind::RANDOM_SMOOTH) &&
           finite(terrain.offset) && finite(terrain.slope_x) &&
           finite(terrain.slope_y) && finite(terrain.amplitude) &&
           finite(terrain.wave_number_x) && finite(terrain.wave_number_y) &&
           finite(terrain.phase_x) && finite(terrain.phase_y) &&
           finite(terrain.secondary_amplitude) &&
           finite(terrain.secondary_wave_number_x) &&
           finite(terrain.secondary_wave_number_y) &&
           finite(terrain.secondary_phase_x) &&
           finite(terrain.secondary_phase_y) && finite(terrain.step_height) &&
           finite(terrain.step_center_x) && finite(terrain.step_sharpness);
}

inline bool valid_configuration(
    const MujocoReplaySnapshotConfiguration& configuration) {
    const ConvexWBCParameters& wbc = configuration.wbc_parameters;
    const ContactExecutionFeedbackParameters& feedback =
        configuration.contact_feedback_parameters;
    return valid_terrain(configuration.terrain) &&
           configuration.terrain_amplitude_argument > 0.0 &&
           finite(configuration.terrain_amplitude_argument) &&
           configuration.mass_scale > 0.0 && finite(configuration.mass_scale) &&
           configuration.friction_scale > 0.0 &&
           finite(configuration.friction_scale) &&
           finite_vector(configuration.push_force_world) &&
           configuration.push_start >= 0.0 &&
           finite(configuration.push_start) &&
           configuration.push_duration >= 0.0 &&
           finite(configuration.push_duration) &&
           (configuration.force_reference == 0 ||
            configuration.force_reference == 1) &&
           configuration.stabilization_duration > 0.0 &&
           finite(configuration.stabilization_duration) &&
           configuration.maximum_stabilization_duration >=
               configuration.stabilization_duration &&
           finite(configuration.maximum_stabilization_duration) &&
           configuration.contact_execution_policy_version >= 0 &&
           nonnegative_vector(wbc.base_position_kp) &&
           nonnegative_vector(wbc.base_velocity_kd) &&
           nonnegative_vector(wbc.base_orientation_kp) &&
           nonnegative_vector(wbc.base_angular_velocity_kd) &&
           nonnegative_vector(wbc.swing_position_kp) &&
           nonnegative_vector(wbc.swing_velocity_kd) &&
           positive_vector(wbc.wrench_weights) &&
           wbc.force_tracking_weight > 0.0 &&
           finite(wbc.force_tracking_weight) && wbc.friction >= 0.0 &&
           finite(wbc.friction) && wbc.maximum_normal_force > 0.0 &&
           finite(wbc.maximum_normal_force) &&
           wbc.contact_blend_force > 0.0 &&
           finite(wbc.contact_blend_force) && wbc.maximum_iterations > 0 &&
           wbc.convergence_tolerance >= 0.0 &&
           finite(wbc.convergence_tolerance) &&
           feedback.early_contact_confirmation_ticks > 0 &&
           feedback.early_contact_loss_grace_ticks > 0 &&
           feedback.contact_blend_force > 0.0 &&
           finite(feedback.contact_blend_force) &&
           feedback.late_touchdown_search_speed > 0.0 &&
           finite(feedback.late_touchdown_search_speed) &&
           feedback.maximum_late_touchdown_search >= 0.0 &&
           finite(feedback.maximum_late_touchdown_search);
}

inline bool valid_contact(const MujocoFootContact& contact) {
    return finite(contact.normal_force) && finite_vector(contact.force_world);
}

inline bool valid_snapshot(const MujocoReplaySnapshot& snapshot) {
    if (snapshot.segment < 0 || snapshot.model_configuration_fingerprint == 0 ||
        snapshot.paired_plan_fingerprint == 0 ||
        snapshot.mujoco_version <= 0 ||
        snapshot.state_signature != mjSTATE_INTEGRATION ||
        snapshot.state_size <= 0 || snapshot.nq <= 0 || snapshot.nv <= 0 ||
        snapshot.nu < 0 || snapshot.na < 0 || snapshot.nbody <= 0 ||
        snapshot.njnt < 0 || snapshot.ngeom < 0 || snapshot.nmocap < 0 ||
        snapshot.nuserdata < 0 || snapshot.npluginstate < 0 ||
        snapshot.nhistory < 0 || !(snapshot.timestep > 0.0) ||
        !valid_configuration(snapshot.configuration) ||
        snapshot.state.size() != static_cast<std::size_t>(snapshot.state_size)) {
        return false;
    }
    for (int foot = 0; foot < kNumFeet; ++foot) {
        if (!valid_contact(snapshot.contacts[foot])) return false;
    }
    for (mjtNum value : snapshot.state) {
        if (!finite(static_cast<double>(value))) return false;
    }
    return true;
}

inline bool read_tag(std::istream& input, const char* expected) {
    std::string tag;
    return static_cast<bool>(input >> tag) && tag == expected;
}

inline bool read_bool(std::istream& input, bool& value) {
    int raw = -1;
    if (!(input >> raw) || (raw != 0 && raw != 1)) return false;
    value = raw != 0;
    return true;
}

template <int Size>
inline void write_vector(std::ostream& output, const Vec<Size>& value) {
    for (int index = 0; index < Size; ++index) output << ' ' << value[index];
}

template <int Size>
inline bool read_vector(std::istream& input, Vec<Size>& value) {
    for (int index = 0; index < Size; ++index) {
        if (!(input >> value[index])) return false;
    }
    return true;
}

inline bool write_snapshot_stream(std::ostream& output,
                                  const MujocoReplaySnapshot& snapshot) {
    const SharedTerrain& terrain = snapshot.configuration.terrain;
    const MujocoReplaySnapshotConfiguration& config = snapshot.configuration;
    const ConvexWBCParameters& wbc = config.wbc_parameters;
    const ContactExecutionFeedbackParameters& feedback =
        config.contact_feedback_parameters;

    output << std::setprecision(17);
    output << "quadruped_cito_mujoco_snapshot_v2\n";
    output << "segment " << snapshot.segment << '\n';
    output << "fingerprints " << snapshot.model_configuration_fingerprint
           << ' ' << snapshot.paired_plan_fingerprint << '\n';
    output << "mujoco " << snapshot.mujoco_version << ' '
           << snapshot.state_signature << ' ' << snapshot.state_size << '\n';
    output << "model " << snapshot.nq << ' ' << snapshot.nv << ' '
           << snapshot.na << ' ' << snapshot.nu << ' ' << snapshot.nbody << ' '
           << snapshot.njnt << ' ' << snapshot.ngeom << ' ' << snapshot.nmocap
           << ' ' << snapshot.nuserdata << ' ' << snapshot.npluginstate << ' '
           << snapshot.nhistory << ' ' << snapshot.timestep << '\n';
    output << "terrain " << static_cast<int>(terrain.kind) << ' '
           << terrain.offset << ' ' << terrain.slope_x << ' ' << terrain.slope_y
           << ' ' << terrain.amplitude << ' ' << terrain.wave_number_x << ' '
           << terrain.wave_number_y << ' ' << terrain.phase_x << ' '
           << terrain.phase_y << ' ' << terrain.secondary_amplitude << ' '
           << terrain.secondary_wave_number_x << ' '
           << terrain.secondary_wave_number_y << ' '
           << terrain.secondary_phase_x << ' ' << terrain.secondary_phase_y
           << ' ' << terrain.step_height << ' ' << terrain.step_center_x << ' '
           << terrain.step_sharpness << '\n';
    output << "terrain_arguments " << config.terrain_seed << ' '
           << config.terrain_amplitude_argument << '\n';
    output << "physics " << config.mass_scale << ' ' << config.friction_scale
           << '\n';
    output << "push";
    write_vector(output, config.push_force_world);
    output << ' ' << config.push_start << ' ' << config.push_duration << '\n';
    output << "execution " << config.force_reference << ' '
           << (config.require_timing ? 1 : 0) << ' '
           << (config.early_contact_feedback ? 1 : 0) << ' '
           << (config.confirmed_stabilization ? 1 : 0) << ' '
           << (config.late_touchdown_search ? 1 : 0) << ' '
           << (config.trace_execution ? 1 : 0) << ' '
           << config.stabilization_duration << ' '
           << config.maximum_stabilization_duration << '\n';
    output << "controller_policy "
           << config.contact_execution_policy_version << ' '
           << (config.synchronized_contact_cache ? 1 : 0) << ' '
           << (config.history_aware_recontact ? 1 : 0) << ' '
           << (config.measured_load_torque_handoff ? 1 : 0) << '\n';
    output << "wbc_base_position_kp";
    write_vector(output, wbc.base_position_kp);
    output << "\nwbc_base_velocity_kd";
    write_vector(output, wbc.base_velocity_kd);
    output << "\nwbc_base_orientation_kp";
    write_vector(output, wbc.base_orientation_kp);
    output << "\nwbc_base_angular_velocity_kd";
    write_vector(output, wbc.base_angular_velocity_kd);
    output << "\nwbc_swing_position_kp";
    write_vector(output, wbc.swing_position_kp);
    output << "\nwbc_swing_velocity_kd";
    write_vector(output, wbc.swing_velocity_kd);
    output << "\nwbc_wrench_weights";
    write_vector(output, wbc.wrench_weights);
    output << "\nwbc_scalars " << wbc.force_tracking_weight << ' '
           << wbc.friction << ' ' << wbc.maximum_normal_force << ' '
           << wbc.contact_blend_force << ' ' << wbc.maximum_iterations << ' '
           << wbc.convergence_tolerance << '\n';
    output << "contact_feedback " << feedback.early_contact_confirmation_ticks
           << ' ' << feedback.early_contact_loss_grace_ticks << ' '
           << feedback.contact_blend_force << ' '
           << feedback.late_touchdown_search_speed << ' '
           << feedback.maximum_late_touchdown_search << '\n';
    for (int foot = 0; foot < kNumFeet; ++foot) {
        output << "contact " << foot << ' '
               << (snapshot.contacts[foot].in_contact ? 1 : 0) << ' '
               << snapshot.contacts[foot].normal_force;
        write_vector(output, snapshot.contacts[foot].force_world);
        output << '\n';
    }
    output << "state";
    for (mjtNum value : snapshot.state) output << ' ' << value;
    output << "\nend\n";
    return static_cast<bool>(output);
}

inline bool read_snapshot_stream(std::istream& input,
                                 MujocoReplaySnapshot& snapshot) {
    MujocoReplaySnapshot result;
    int terrain_kind = -1;
    SharedTerrain& terrain = result.configuration.terrain;
    MujocoReplaySnapshotConfiguration& config = result.configuration;
    ConvexWBCParameters& wbc = config.wbc_parameters;
    ContactExecutionFeedbackParameters& feedback =
        config.contact_feedback_parameters;
    std::string format;
    if (!(input >> format)) return false;
    const bool version_two = format == "quadruped_cito_mujoco_snapshot_v2";
    if (!version_two && format != "quadruped_cito_mujoco_snapshot_v1")
        return false;

    if (!read_tag(input, "segment") || !(input >> result.segment) ||
        !read_tag(input, "fingerprints") ||
        !(input >> result.model_configuration_fingerprint >>
          result.paired_plan_fingerprint) ||
        !read_tag(input, "mujoco") ||
        !(input >> result.mujoco_version >> result.state_signature >>
          result.state_size) ||
        !read_tag(input, "model") ||
        !(input >> result.nq >> result.nv >> result.na >> result.nu >>
          result.nbody >> result.njnt >> result.ngeom >> result.nmocap >>
          result.nuserdata >> result.npluginstate >> result.nhistory >>
          result.timestep) ||
        !read_tag(input, "terrain") ||
        !(input >> terrain_kind >> terrain.offset >> terrain.slope_x >>
          terrain.slope_y >> terrain.amplitude >> terrain.wave_number_x >>
          terrain.wave_number_y >> terrain.phase_x >> terrain.phase_y >>
          terrain.secondary_amplitude >> terrain.secondary_wave_number_x >>
          terrain.secondary_wave_number_y >> terrain.secondary_phase_x >>
          terrain.secondary_phase_y >> terrain.step_height >>
          terrain.step_center_x >> terrain.step_sharpness) ||
        !read_tag(input, "terrain_arguments") ||
        !(input >> config.terrain_seed >> config.terrain_amplitude_argument) ||
        !read_tag(input, "physics") ||
        !(input >> config.mass_scale >> config.friction_scale) ||
        !read_tag(input, "push") ||
        !read_vector(input, config.push_force_world) ||
        !(input >> config.push_start >> config.push_duration) ||
        !read_tag(input, "execution") || !(input >> config.force_reference) ||
        !read_bool(input, config.require_timing) ||
        !read_bool(input, config.early_contact_feedback) ||
        !read_bool(input, config.confirmed_stabilization) ||
        !read_bool(input, config.late_touchdown_search) ||
        !read_bool(input, config.trace_execution) ||
        !(input >> config.stabilization_duration >>
          config.maximum_stabilization_duration)) {
        return false;
    }
    if (version_two &&
        (!read_tag(input, "controller_policy") ||
         !(input >> config.contact_execution_policy_version) ||
         !read_bool(input, config.synchronized_contact_cache) ||
         !read_bool(input, config.history_aware_recontact) ||
         !read_bool(input, config.measured_load_torque_handoff))) {
        return false;
    }
    if (!read_tag(input, "wbc_base_position_kp") ||
        !read_vector(input, wbc.base_position_kp) ||
        !read_tag(input, "wbc_base_velocity_kd") ||
        !read_vector(input, wbc.base_velocity_kd) ||
        !read_tag(input, "wbc_base_orientation_kp") ||
        !read_vector(input, wbc.base_orientation_kp) ||
        !read_tag(input, "wbc_base_angular_velocity_kd") ||
        !read_vector(input, wbc.base_angular_velocity_kd) ||
        !read_tag(input, "wbc_swing_position_kp") ||
        !read_vector(input, wbc.swing_position_kp) ||
        !read_tag(input, "wbc_swing_velocity_kd") ||
        !read_vector(input, wbc.swing_velocity_kd) ||
        !read_tag(input, "wbc_wrench_weights") ||
        !read_vector(input, wbc.wrench_weights) ||
        !read_tag(input, "wbc_scalars") ||
        !(input >> wbc.force_tracking_weight >> wbc.friction >>
          wbc.maximum_normal_force >> wbc.contact_blend_force >>
          wbc.maximum_iterations >> wbc.convergence_tolerance) ||
        !read_tag(input, "contact_feedback") ||
        !(input >> feedback.early_contact_confirmation_ticks >>
          feedback.early_contact_loss_grace_ticks >>
          feedback.contact_blend_force >>
          feedback.late_touchdown_search_speed >>
          feedback.maximum_late_touchdown_search)) {
        return false;
    }
    if (terrain_kind < static_cast<int>(TerrainKind::FLAT) ||
        terrain_kind > static_cast<int>(TerrainKind::RANDOM_SMOOTH)) {
        return false;
    }
    terrain.kind = static_cast<TerrainKind>(terrain_kind);
    for (int expected_foot = 0; expected_foot < kNumFeet; ++expected_foot) {
        int foot = -1;
        if (!read_tag(input, "contact") || !(input >> foot) ||
            foot != expected_foot ||
            !read_bool(input, result.contacts[foot].in_contact) ||
            !(input >> result.contacts[foot].normal_force) ||
            !read_vector(input, result.contacts[foot].force_world)) {
            return false;
        }
    }
    if (result.state_size <= 0 || !read_tag(input, "state")) return false;
    result.state.resize(static_cast<std::size_t>(result.state_size));
    for (mjtNum& value : result.state) {
        if (!(input >> value)) return false;
    }
    if (!read_tag(input, "end")) return false;
    std::string trailing;
    if (input >> trailing) return false;
    if (!valid_snapshot(result)) return false;
    snapshot = result;
    return true;
}

}  // namespace replay_snapshot_detail

inline std::uint64_t mujoco_model_configuration_fingerprint(
    const mjModel* model) {
    if (!model) return 0;
    const mjtSize model_size = mj_sizeModel(model);
    if (model_size <= 0 ||
        model_size > static_cast<mjtSize>(std::numeric_limits<int>::max())) {
        return 0;
    }
    std::vector<unsigned char> buffer(static_cast<std::size_t>(model_size));
    mj_saveModel(model, nullptr, buffer.data(), static_cast<int>(model_size));
    std::uint64_t fingerprint = 14695981039346656037ULL;
    for (unsigned char byte : buffer) {
        fingerprint ^= static_cast<std::uint64_t>(byte);
        fingerprint *= 1099511628211ULL;
    }
    return fingerprint == 0 ? 1 : fingerprint;
}

inline Status capture_mujoco_replay_snapshot(
    const mjModel* model, const mjData* data, int segment,
    const MujocoReplaySnapshotConfiguration& configuration,
    std::uint64_t paired_plan_fingerprint,
    const MujocoFootContact contacts[kNumFeet],
    MujocoReplaySnapshot& snapshot) {
    if (!model || !data || !contacts || segment < 0 ||
        paired_plan_fingerprint == 0 ||
        !replay_snapshot_detail::valid_configuration(configuration)) {
        return Status::BAD_ARGUMENT;
    }
    MujocoReplaySnapshot result;
    result.segment = segment;
    result.model_configuration_fingerprint =
        mujoco_model_configuration_fingerprint(model);
    result.paired_plan_fingerprint = paired_plan_fingerprint;
    result.mujoco_version = mj_version();
    result.state_signature = mjSTATE_INTEGRATION;
    result.state_size = mj_stateSize(model, result.state_signature);
    result.nq = model->nq;
    result.nv = model->nv;
    result.na = model->na;
    result.nu = model->nu;
    result.nbody = model->nbody;
    result.njnt = model->njnt;
    result.ngeom = model->ngeom;
    result.nmocap = model->nmocap;
    result.nuserdata = model->nuserdata;
    result.npluginstate = model->npluginstate;
    result.nhistory = model->nhistory;
    result.timestep = model->opt.timestep;
    result.configuration = configuration;
    for (int foot = 0; foot < kNumFeet; ++foot) result.contacts[foot] = contacts[foot];
    if (result.state_size <= 0) return Status::INTERNAL_ERROR;
    result.state.resize(static_cast<std::size_t>(result.state_size));
    mj_getState(model, data, result.state.data(), result.state_signature);
    if (!replay_snapshot_detail::valid_snapshot(result))
        return Status::BAD_ARGUMENT;
    snapshot = result;
    return Status::SUCCESS;
}

inline Status write_mujoco_replay_snapshot(
    const MujocoReplaySnapshot& snapshot, const char* path) {
    if (!path || path[0] == '\0' ||
        !replay_snapshot_detail::valid_snapshot(snapshot)) {
        return Status::BAD_ARGUMENT;
    }
    const std::string destination(path);
    {
        std::ifstream existing(destination);
        if (existing.good()) return Status::BAD_ARGUMENT;
    }
    const std::string temporary = destination + ".tmp";
    {
        std::ofstream output(temporary, std::ios::out | std::ios::trunc);
        if (!output ||
            !replay_snapshot_detail::write_snapshot_stream(output, snapshot)) {
            output.close();
            std::remove(temporary.c_str());
            return Status::BAD_ARGUMENT;
        }
        output.close();
        if (!output) {
            std::remove(temporary.c_str());
            return Status::BAD_ARGUMENT;
        }
    }
    if (std::rename(temporary.c_str(), destination.c_str()) != 0) {
        std::remove(temporary.c_str());
        return Status::BAD_ARGUMENT;
    }
    return Status::SUCCESS;
}

inline Status read_mujoco_replay_snapshot(const char* path,
                                          MujocoReplaySnapshot& snapshot) {
    if (!path || path[0] == '\0') return Status::BAD_ARGUMENT;
    std::ifstream input(path);
    if (!input) return Status::BAD_ARGUMENT;
    return replay_snapshot_detail::read_snapshot_stream(input, snapshot)
        ? Status::SUCCESS
        : Status::BAD_ARGUMENT;
}

inline Status validate_mujoco_replay_snapshot_model(
    const MujocoReplaySnapshot& snapshot, const mjModel* model) {
    if (!model || !replay_snapshot_detail::valid_snapshot(snapshot))
        return Status::BAD_ARGUMENT;
    const bool compatible = snapshot.mujoco_version == mj_version() &&
        snapshot.model_configuration_fingerprint ==
            mujoco_model_configuration_fingerprint(model) &&
        snapshot.state_signature == mjSTATE_INTEGRATION &&
        snapshot.state_size == mj_stateSize(model, mjSTATE_INTEGRATION) &&
        snapshot.nq == model->nq && snapshot.nv == model->nv &&
        snapshot.na == model->na && snapshot.nu == model->nu &&
        snapshot.nbody == model->nbody && snapshot.njnt == model->njnt &&
        snapshot.ngeom == model->ngeom && snapshot.nmocap == model->nmocap &&
        snapshot.nuserdata == model->nuserdata &&
        snapshot.npluginstate == model->npluginstate &&
        snapshot.nhistory == model->nhistory &&
        snapshot.timestep == model->opt.timestep;
    return compatible ? Status::SUCCESS : Status::BAD_ARGUMENT;
}

inline Status restore_mujoco_replay_snapshot(
    const MujocoReplaySnapshot& snapshot, const mjModel* model, mjData* data,
    const MujocoGo1Adapter& adapter,
    MujocoReplaySnapshotRestoreReport* report = nullptr,
    double contact_force_tolerance = 1e-6) {
    if (!data || !adapter.initialized() || !(contact_force_tolerance >= 0.0) ||
        validate_mujoco_replay_snapshot_model(snapshot, model) !=
            Status::SUCCESS) {
        return Status::BAD_ARGUMENT;
    }
    if (report) *report = MujocoReplaySnapshotRestoreReport{};
    const int state_size = mj_stateSize(model, mjSTATE_INTEGRATION);
    std::vector<mjtNum> previous_state(static_cast<std::size_t>(state_size));
    mj_getState(model, data, previous_state.data(), mjSTATE_INTEGRATION);
    const auto rollback = [&]() {
        mj_setState(model, data, previous_state.data(), mjSTATE_INTEGRATION);
        mj_forward(model, data);
    };
    mj_setState(model, data, snapshot.state.data(), snapshot.state_signature);
    mj_forward(model, data);

    MujocoFootContact restored[kNumFeet];
    if (adapter.read_foot_contacts(restored) != Status::SUCCESS) {
        rollback();
        return Status::INTERNAL_ERROR;
    }
    bool mask_matches = true;
    double maximum_force_error = 0.0;
    for (int foot = 0; foot < kNumFeet; ++foot) {
        mask_matches = mask_matches &&
            restored[foot].in_contact == snapshot.contacts[foot].in_contact;
        maximum_force_error = std::max(
            maximum_force_error,
            std::fabs(restored[foot].normal_force -
                      snapshot.contacts[foot].normal_force));
        for (int axis = 0; axis < 3; ++axis) {
            maximum_force_error = std::max(
                maximum_force_error,
                std::fabs(restored[foot].force_world[axis] -
                          snapshot.contacts[foot].force_world[axis]));
        }
    }
    if (report) {
        report->contact_mask_matches = mask_matches;
        report->maximum_contact_force_error = maximum_force_error;
    }
    if (!mask_matches || maximum_force_error > contact_force_tolerance) {
        rollback();
        return Status::INTERNAL_ERROR;
    }
    return Status::SUCCESS;
}

}  // namespace quadruped_cito
