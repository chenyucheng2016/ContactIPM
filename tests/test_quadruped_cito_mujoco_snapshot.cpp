#include <cmath>
#include <cstdio>
#include <sstream>
#include <string>
#include <vector>

#include <mujoco/mujoco.h>

#include "examples/quadruped_cito/mujoco/mujoco_replay_snapshot.hpp"

namespace {

using namespace quadruped_cito;

bool equal_contact(const MujocoFootContact& first,
                   const MujocoFootContact& second) {
    if (first.in_contact != second.in_contact ||
        first.normal_force != second.normal_force) {
        return false;
    }
    for (int axis = 0; axis < 3; ++axis) {
        if (first.force_world[axis] != second.force_world[axis]) return false;
    }
    return true;
}

bool equal_configuration(const MujocoReplaySnapshotConfiguration& first,
                         const MujocoReplaySnapshotConfiguration& second) {
    const SharedTerrain& a = first.terrain;
    const SharedTerrain& b = second.terrain;
    if (a.kind != b.kind || a.offset != b.offset ||
        a.slope_x != b.slope_x || a.slope_y != b.slope_y ||
        a.amplitude != b.amplitude ||
        a.wave_number_x != b.wave_number_x ||
        a.wave_number_y != b.wave_number_y || a.phase_x != b.phase_x ||
        a.phase_y != b.phase_y ||
        a.secondary_amplitude != b.secondary_amplitude ||
        a.secondary_wave_number_x != b.secondary_wave_number_x ||
        a.secondary_wave_number_y != b.secondary_wave_number_y ||
        a.secondary_phase_x != b.secondary_phase_x ||
        a.secondary_phase_y != b.secondary_phase_y ||
        a.step_height != b.step_height ||
        a.step_center_x != b.step_center_x ||
        a.step_sharpness != b.step_sharpness ||
        first.terrain_seed != second.terrain_seed ||
        first.terrain_amplitude_argument !=
            second.terrain_amplitude_argument ||
        first.mass_scale != second.mass_scale ||
        first.friction_scale != second.friction_scale ||
        first.push_start != second.push_start ||
        first.push_duration != second.push_duration ||
        first.force_reference != second.force_reference ||
        first.require_timing != second.require_timing ||
        first.early_contact_feedback != second.early_contact_feedback ||
        first.confirmed_stabilization != second.confirmed_stabilization ||
        first.late_touchdown_search != second.late_touchdown_search ||
        first.trace_execution != second.trace_execution ||
        first.stabilization_duration != second.stabilization_duration ||
        first.maximum_stabilization_duration !=
            second.maximum_stabilization_duration ||
        first.contact_execution_policy_version !=
            second.contact_execution_policy_version ||
        first.synchronized_contact_cache !=
            second.synchronized_contact_cache ||
        first.history_aware_recontact != second.history_aware_recontact ||
        first.measured_load_torque_handoff !=
            second.measured_load_torque_handoff) {
        return false;
    }
    for (int axis = 0; axis < 3; ++axis) {
        if (first.push_force_world[axis] != second.push_force_world[axis] ||
            first.wbc_parameters.base_position_kp[axis] !=
                second.wbc_parameters.base_position_kp[axis] ||
            first.wbc_parameters.base_velocity_kd[axis] !=
                second.wbc_parameters.base_velocity_kd[axis] ||
            first.wbc_parameters.base_orientation_kp[axis] !=
                second.wbc_parameters.base_orientation_kp[axis] ||
            first.wbc_parameters.base_angular_velocity_kd[axis] !=
                second.wbc_parameters.base_angular_velocity_kd[axis] ||
            first.wbc_parameters.swing_position_kp[axis] !=
                second.wbc_parameters.swing_position_kp[axis] ||
            first.wbc_parameters.swing_velocity_kd[axis] !=
                second.wbc_parameters.swing_velocity_kd[axis]) {
            return false;
        }
    }
    for (int row = 0; row < 6; ++row) {
        if (first.wbc_parameters.wrench_weights[row] !=
            second.wbc_parameters.wrench_weights[row]) {
            return false;
        }
    }
    const ConvexWBCParameters& wa = first.wbc_parameters;
    const ConvexWBCParameters& wb = second.wbc_parameters;
    const ContactExecutionFeedbackParameters& fa =
        first.contact_feedback_parameters;
    const ContactExecutionFeedbackParameters& fb =
        second.contact_feedback_parameters;
    return wa.force_tracking_weight == wb.force_tracking_weight &&
        wa.friction == wb.friction &&
        wa.maximum_normal_force == wb.maximum_normal_force &&
        wa.contact_blend_force == wb.contact_blend_force &&
        wa.maximum_iterations == wb.maximum_iterations &&
        wa.convergence_tolerance == wb.convergence_tolerance &&
        fa.early_contact_confirmation_ticks ==
            fb.early_contact_confirmation_ticks &&
        fa.early_contact_loss_grace_ticks ==
            fb.early_contact_loss_grace_ticks &&
        fa.contact_blend_force == fb.contact_blend_force &&
        fa.late_touchdown_search_speed == fb.late_touchdown_search_speed &&
        fa.maximum_late_touchdown_search ==
            fb.maximum_late_touchdown_search;
}

bool equal_snapshot(const MujocoReplaySnapshot& first,
                    const MujocoReplaySnapshot& second) {
    if (first.segment != second.segment ||
        first.model_configuration_fingerprint !=
            second.model_configuration_fingerprint ||
        first.paired_plan_fingerprint != second.paired_plan_fingerprint ||
        first.mujoco_version != second.mujoco_version ||
        first.state_signature != second.state_signature ||
        first.state_size != second.state_size || first.nq != second.nq ||
        first.nv != second.nv || first.na != second.na ||
        first.nu != second.nu || first.nbody != second.nbody ||
        first.njnt != second.njnt || first.ngeom != second.ngeom ||
        first.nmocap != second.nmocap ||
        first.nuserdata != second.nuserdata ||
        first.npluginstate != second.npluginstate ||
        first.nhistory != second.nhistory ||
        first.timestep != second.timestep ||
        !equal_configuration(first.configuration, second.configuration) ||
        first.state != second.state) {
        return false;
    }
    for (int foot = 0; foot < kNumFeet; ++foot) {
        if (!equal_contact(first.contacts[foot], second.contacts[foot]))
            return false;
    }
    return true;
}

MujocoReplaySnapshotConfiguration test_configuration() {
    MujocoReplaySnapshotConfiguration configuration;
    configuration.terrain = SharedTerrain::random_smooth(17, 0.023);
    configuration.terrain_seed = 17;
    configuration.terrain_amplitude_argument = 0.023;
    configuration.mass_scale = 1.07;
    configuration.friction_scale = 0.85;
    configuration.push_force_world[0] = 2.0;
    configuration.push_force_world[1] = -3.0;
    configuration.push_force_world[2] = 1.0;
    configuration.push_start = 1.25;
    configuration.push_duration = 0.14;
    configuration.force_reference = 1;
    configuration.require_timing = false;
    configuration.early_contact_feedback = false;
    configuration.confirmed_stabilization = false;
    configuration.late_touchdown_search = true;
    configuration.trace_execution = true;
    configuration.stabilization_duration = 1.75;
    configuration.maximum_stabilization_duration = 3.25;
    configuration.contact_execution_policy_version = 7;
    configuration.synchronized_contact_cache = true;
    configuration.history_aware_recontact = true;
    configuration.measured_load_torque_handoff = true;
    for (int axis = 0; axis < 3; ++axis) {
        configuration.wbc_parameters.base_position_kp[axis] = 101.0 + axis;
        configuration.wbc_parameters.base_velocity_kd[axis] = 21.0 + axis;
        configuration.wbc_parameters.base_orientation_kp[axis] = 61.0 + axis;
        configuration.wbc_parameters.base_angular_velocity_kd[axis] =
            9.0 + axis;
        configuration.wbc_parameters.swing_position_kp[axis] = 151.0 + axis;
        configuration.wbc_parameters.swing_velocity_kd[axis] = 11.0 + axis;
    }
    for (int row = 0; row < 6; ++row)
        configuration.wbc_parameters.wrench_weights[row] = 2.0 + row;
    configuration.wbc_parameters.force_tracking_weight = 0.015;
    configuration.wbc_parameters.friction = 0.57;
    configuration.wbc_parameters.maximum_normal_force = 91.0;
    configuration.wbc_parameters.contact_blend_force = 19.0;
    configuration.wbc_parameters.maximum_iterations = 137;
    configuration.wbc_parameters.convergence_tolerance = 3e-11;
    configuration.contact_feedback_parameters.
        early_contact_confirmation_ticks = 4;
    configuration.contact_feedback_parameters.
        early_contact_loss_grace_ticks = 6;
    configuration.contact_feedback_parameters.contact_blend_force = 18.0;
    configuration.contact_feedback_parameters.late_touchdown_search_speed =
        0.04;
    configuration.contact_feedback_parameters.maximum_late_touchdown_search =
        0.015;
    return configuration;
}

bool legacy_snapshot_round_trip(const MujocoReplaySnapshot& snapshot) {
    std::ostringstream output;
    if (!replay_snapshot_detail::write_snapshot_stream(output, snapshot))
        return false;
    std::string text = output.str();
    const std::string version_two = "quadruped_cito_mujoco_snapshot_v2";
    const std::string version_one = "quadruped_cito_mujoco_snapshot_v1";
    if (text.compare(0, version_two.size(), version_two) != 0) return false;
    text.replace(0, version_two.size(), version_one);

    const std::size_t policy_begin = text.find("controller_policy ");
    const std::size_t policy_end = text.find('\n', policy_begin);
    if (policy_begin == std::string::npos || policy_end == std::string::npos)
        return false;
    text.erase(policy_begin, policy_end - policy_begin + 1);

    MujocoReplaySnapshot loaded;
    std::istringstream input(text);
    if (!replay_snapshot_detail::read_snapshot_stream(input, loaded))
        return false;
    MujocoReplaySnapshot expected = snapshot;
    expected.configuration.contact_execution_policy_version = 0;
    expected.configuration.synchronized_contact_cache = false;
    expected.configuration.history_aware_recontact = false;
    expected.configuration.measured_load_torque_handoff = false;
    return equal_snapshot(expected, loaded);
}

}  // namespace

int main(int argc, char** argv) {
    if (argc != 3) {
        std::printf("usage: test_quadruped_cito_mujoco_snapshot "
                    "<go1-scene.xml> <snapshot-path>\n");
        return 2;
    }
    char error[1024] = {};
    mjModel* model = mj_loadXML(argv[1], nullptr, error, sizeof(error));
    if (!model) {
        std::printf("failed to load Go1 model: %s\n", error);
        return 1;
    }
    if (configure_go1_torque_actuators(model) != Status::SUCCESS) {
        mj_deleteModel(model);
        return 1;
    }
    mjData* data = mj_makeData(model);
    if (!data) {
        mj_deleteModel(model);
        return 1;
    }
    MujocoGo1Adapter adapter(model, data);
    if (adapter.initialize() != Status::SUCCESS) {
        mj_deleteData(data);
        mj_deleteModel(model);
        return 1;
    }
    const int home = mj_name2id(model, mjOBJ_KEY, "home");
    if (home >= 0) mj_resetDataKeyframe(model, data, home);
    mj_forward(model, data);
    for (int tick = 0; tick < 25; ++tick) mj_step(model, data);
    mj_forward(model, data);

    WholeBodyState original_body;
    MujocoFootContact original_contacts[kNumFeet];
    if (adapter.read_whole_body_state(original_body) != Status::SUCCESS ||
        adapter.read_foot_contacts(original_contacts) != Status::SUCCESS) {
        mj_deleteData(data);
        mj_deleteModel(model);
        return 1;
    }
    MujocoReplaySnapshot original;
    constexpr std::uint64_t kPlanFingerprint = 0x123456789abcdef0ULL;
    MujocoReplaySnapshotConfiguration invalid_configuration =
        test_configuration();
    invalid_configuration.force_reference = 2;
    MujocoReplaySnapshot rejected;
    if (capture_mujoco_replay_snapshot(
            model, data, 3, invalid_configuration, kPlanFingerprint,
            original_contacts, rejected) != Status::BAD_ARGUMENT) {
        std::printf("invalid force reference was accepted\n");
        mj_deleteData(data);
        mj_deleteModel(model);
        return 1;
    }
    if (capture_mujoco_replay_snapshot(
            model, data, 3, test_configuration(), kPlanFingerprint,
            original_contacts, original) != Status::SUCCESS ||
        write_mujoco_replay_snapshot(original, argv[2]) != Status::SUCCESS) {
        std::printf("failed to capture or write snapshot\n");
        mj_deleteData(data);
        mj_deleteModel(model);
        return 1;
    }
    if (write_mujoco_replay_snapshot(original, argv[2]) !=
        Status::BAD_ARGUMENT) {
        std::printf("snapshot writer overwrote an existing snapshot\n");
        std::remove(argv[2]);
        mj_deleteData(data);
        mj_deleteModel(model);
        return 1;
    }
    MujocoReplaySnapshot loaded;
    if (read_mujoco_replay_snapshot(argv[2], loaded) != Status::SUCCESS ||
        !equal_snapshot(original, loaded) ||
        validate_mujoco_replay_snapshot_model(loaded, model) !=
            Status::SUCCESS) {
        std::printf("snapshot text round trip failed\n");
        std::remove(argv[2]);
        mj_deleteData(data);
        mj_deleteModel(model);
        return 1;
    }
    if (!legacy_snapshot_round_trip(original)) {
        std::printf("legacy snapshot compatibility failed\n");
        std::remove(argv[2]);
        mj_deleteData(data);
        mj_deleteModel(model);
        return 1;
    }

    MujocoReplaySnapshot wrong_model = loaded;
    ++wrong_model.model_configuration_fingerprint;
    if (validate_mujoco_replay_snapshot_model(wrong_model, model) !=
        Status::BAD_ARGUMENT) {
        std::printf("model fingerprint mismatch was accepted\n");
        std::remove(argv[2]);
        mj_deleteData(data);
        mj_deleteModel(model);
        return 1;
    }

    std::vector<mjtNum> rollback_expected(original.state.size());
    mj_getState(model, data, rollback_expected.data(), mjSTATE_INTEGRATION);
    MujocoReplaySnapshot wrong_contacts = loaded;
    wrong_contacts.contacts[0].in_contact =
        !wrong_contacts.contacts[0].in_contact;
    MujocoReplaySnapshotRestoreReport rejected_restore_report;
    if (restore_mujoco_replay_snapshot(
            wrong_contacts, model, data, adapter, &rejected_restore_report) !=
            Status::INTERNAL_ERROR) {
        std::printf("contact mismatch was accepted\n");
        std::remove(argv[2]);
        mj_deleteData(data);
        mj_deleteModel(model);
        return 1;
    }
    std::vector<mjtNum> rollback_actual(original.state.size());
    mj_getState(model, data, rollback_actual.data(), mjSTATE_INTEGRATION);
    if (rollback_actual != rollback_expected) {
        std::printf("failed restore did not roll back integration state\n");
        std::remove(argv[2]);
        mj_deleteData(data);
        mj_deleteModel(model);
        return 1;
    }

    data->qpos[0] += 0.1;
    data->qvel[0] += 0.2;
    data->time += 1.0;
    mj_forward(model, data);
    MujocoReplaySnapshotRestoreReport restore_report;
    if (restore_mujoco_replay_snapshot(
            loaded, model, data, adapter, &restore_report) !=
            Status::SUCCESS ||
        !restore_report.contact_mask_matches ||
        restore_report.maximum_contact_force_error > 1e-8) {
        std::printf("snapshot restoration failed: contact_error=%.3e\n",
                    restore_report.maximum_contact_force_error);
        std::remove(argv[2]);
        mj_deleteData(data);
        mj_deleteModel(model);
        return 1;
    }
    std::vector<mjtNum> restored(original.state.size());
    mj_getState(model, data, restored.data(), mjSTATE_INTEGRATION);
    if (restored != original.state) {
        std::printf("restored integration state differs from captured state\n");
        std::remove(argv[2]);
        mj_deleteData(data);
        mj_deleteModel(model);
        return 1;
    }
    WholeBodyState restored_body;
    if (adapter.read_whole_body_state(restored_body) != Status::SUCCESS) {
        std::remove(argv[2]);
        mj_deleteData(data);
        mj_deleteModel(model);
        return 1;
    }
    for (int axis = 0; axis < 3; ++axis) {
        if (restored_body.base.position_world[axis] !=
            original_body.base.position_world[axis]) {
            std::printf("restored base state differs from captured state\n");
            std::remove(argv[2]);
            mj_deleteData(data);
            mj_deleteModel(model);
            return 1;
        }
    }
    std::printf("MuJoCo replay snapshot round trip: state_size=%d "
                "contact_error=%.3e\n",
                original.state_size,
                restore_report.maximum_contact_force_error);
    std::remove(argv[2]);
    mj_deleteData(data);
    mj_deleteModel(model);
    return 0;
}
