#include "examples/quadruped_cito/mujoco/mujoco_go1_adapter.hpp"

#include <algorithm>
#include <cmath>
#include <limits>
#include <vector>

namespace quadruped_cito {
namespace {

constexpr const char* kTrunkBodyName = "trunk";
constexpr const char* kFootNames[kNumFeet] = {"FL", "FR", "RL", "RR"};
constexpr const char* kJointNames[kNumFeet][kGo1JointsPerLeg] = {
    {"FL_hip_joint", "FL_thigh_joint", "FL_calf_joint"},
    {"FR_hip_joint", "FR_thigh_joint", "FR_calf_joint"},
    {"RL_hip_joint", "RL_thigh_joint", "RL_calf_joint"},
    {"RR_hip_joint", "RR_thigh_joint", "RR_calf_joint"}};

bool finite3(const Vec<3>& vector) {
    for (int axis = 0; axis < 3; ++axis) {
        if (!std::isfinite(vector[axis])) return false;
    }
    return true;
}

bool solve_damped_3x3(double matrix[3][3], const double rhs[3],
                      double solution[3]) {
    double augmented[3][4] = {};
    for (int row = 0; row < 3; ++row) {
        for (int column = 0; column < 3; ++column)
            augmented[row][column] = matrix[row][column];
        augmented[row][3] = rhs[row];
    }
    for (int pivot = 0; pivot < 3; ++pivot) {
        int best = pivot;
        for (int row = pivot + 1; row < 3; ++row) {
            if (std::fabs(augmented[row][pivot]) >
                std::fabs(augmented[best][pivot])) {
                best = row;
            }
        }
        if (std::fabs(augmented[best][pivot]) <= 1e-12) return false;
        for (int column = pivot; column < 4; ++column)
            std::swap(augmented[pivot][column], augmented[best][column]);
        const double divisor = augmented[pivot][pivot];
        for (int column = pivot; column < 4; ++column)
            augmented[pivot][column] /= divisor;
        for (int row = 0; row < 3; ++row) {
            if (row == pivot) continue;
            const double factor = augmented[row][pivot];
            for (int column = pivot; column < 4; ++column)
                augmented[row][column] -= factor * augmented[pivot][column];
        }
    }
    for (int row = 0; row < 3; ++row) solution[row] = augmented[row][3];
    return true;
}

}  // namespace

Status configure_go1_torque_actuators(mjModel* model) {
    if (!model) return Status::BAD_ARGUMENT;
    for (int foot = 0; foot < kNumFeet; ++foot) {
        for (int joint = 0; joint < kGo1JointsPerLeg; ++joint) {
            const int joint_id =
                mj_name2id(model, mjOBJ_JOINT, kJointNames[foot][joint]);
            if (joint_id < 0) return Status::BAD_ARGUMENT;
            int matched_actuator = -1;
            for (int actuator = 0; actuator < model->nu; ++actuator) {
                if (model->actuator_trntype[actuator] == mjTRN_JOINT &&
                    model->actuator_trnid[2 * actuator] == joint_id) {
                    if (matched_actuator >= 0) return Status::BAD_ARGUMENT;
                    matched_actuator = actuator;
                }
            }
            if (matched_actuator < 0 ||
                !model->actuator_forcelimited[matched_actuator] ||
                !(model->actuator_forcerange[2 * matched_actuator] <
                  model->actuator_forcerange[2 * matched_actuator + 1])) {
                return Status::BAD_ARGUMENT;
            }

            model->actuator_dyntype[matched_actuator] = mjDYN_NONE;
            model->actuator_gaintype[matched_actuator] = mjGAIN_FIXED;
            model->actuator_biastype[matched_actuator] = mjBIAS_NONE;
            for (int parameter = 0; parameter < mjNDYN; ++parameter)
                model->actuator_dynprm[mjNDYN * matched_actuator + parameter] =
                    0.0;
            for (int parameter = 0; parameter < mjNGAIN; ++parameter)
                model->actuator_gainprm[
                    mjNGAIN * matched_actuator + parameter] = 0.0;
            for (int parameter = 0; parameter < mjNBIAS; ++parameter)
                model->actuator_biasprm[
                    mjNBIAS * matched_actuator + parameter] = 0.0;
            model->actuator_gainprm[mjNGAIN * matched_actuator] = 1.0;
            model->actuator_ctrllimited[matched_actuator] = 0;
        }
    }
    return Status::SUCCESS;
}

Status configure_shared_terrain(mjModel* model,
                                const SharedTerrain& terrain,
                                MujocoTerrainReport* report) {
    if (!model) return Status::BAD_ARGUMENT;
    const int hfield =
        mj_name2id(model, mjOBJ_HFIELD, "cito_heightfield");
    const int geom = mj_name2id(model, mjOBJ_GEOM, "cito_terrain");
    if (hfield < 0 || geom < 0 ||
        model->geom_type[geom] != mjGEOM_HFIELD ||
        model->geom_dataid[geom] != hfield) {
        return Status::BAD_ARGUMENT;
    }
    const int rows = model->hfield_nrow[hfield];
    const int columns = model->hfield_ncol[hfield];
    if (rows < 2 || columns < 2) return Status::BAD_ARGUMENT;
    const int address = model->hfield_adr[hfield];
    const double radius_x = model->hfield_size[4 * hfield];
    const double radius_y = model->hfield_size[4 * hfield + 1];
    if (!(radius_x > 0.0) || !(radius_y > 0.0))
        return Status::BAD_ARGUMENT;

    double minimum_height = std::numeric_limits<double>::infinity();
    double maximum_height = -std::numeric_limits<double>::infinity();
    for (int row = 0; row < rows; ++row) {
        const double y = -radius_y + 2.0 * radius_y * row /
                                       static_cast<double>(rows - 1);
        for (int column = 0; column < columns; ++column) {
            const double x = -radius_x + 2.0 * radius_x * column /
                                           static_cast<double>(columns - 1);
            const double height = terrain.height(x, y);
            minimum_height = std::min(minimum_height, height);
            maximum_height = std::max(maximum_height, height);
        }
    }
    const double height_base = model->geom_pos[3 * geom + 2];
    const double height_scale = model->hfield_size[4 * hfield + 2];
    if (!(height_scale > 0.0) || minimum_height < height_base ||
        maximum_height > height_base + height_scale) {
        return Status::BAD_ARGUMENT;
    }
    double maximum_grid_error = 0.0;
    for (int row = 0; row < rows; ++row) {
        const double y = -radius_y + 2.0 * radius_y * row /
                                       static_cast<double>(rows - 1);
        for (int column = 0; column < columns; ++column) {
            const double x = -radius_x + 2.0 * radius_x * column /
                                           static_cast<double>(columns - 1);
            const double expected = terrain.height(x, y);
            const float normalized = static_cast<float>(
                (expected - height_base) / height_scale);
            model->hfield_data[address + row * columns + column] = normalized;
            const double reconstructed = height_base +
                                         height_scale * normalized;
            maximum_grid_error = std::max(
                maximum_grid_error, std::fabs(reconstructed - expected));
        }
    }
    model->geom_friction[3 * geom] = 0.8;
    model->geom_friction[3 * geom + 1] = 0.02;
    model->geom_friction[3 * geom + 2] = 0.01;

    if (report) {
        report->rows = rows;
        report->columns = columns;
        report->minimum_height = minimum_height;
        report->maximum_height = maximum_height;
        report->maximum_grid_error = maximum_grid_error;
    }
    return maximum_grid_error <= 1e-7 ? Status::SUCCESS
                                      : Status::INTERNAL_ERROR;
}

Status MujocoGo1Adapter::initialize() {
    initialized_ = false;
    if (!model_ || !data_) return Status::BAD_ARGUMENT;

    trunk_body_id_ = mj_name2id(model_, mjOBJ_BODY, kTrunkBodyName);
    if (trunk_body_id_ < 0) return Status::BAD_ARGUMENT;
    for (int joint = 0; joint < model_->njnt; ++joint) {
        if (model_->jnt_bodyid[joint] == trunk_body_id_ &&
            model_->jnt_type[joint] == mjJNT_FREE) {
            free_joint_qpos_address_ = model_->jnt_qposadr[joint];
            break;
        }
    }
    if (free_joint_qpos_address_ < 0) return Status::BAD_ARGUMENT;

    for (int foot = 0; foot < kNumFeet; ++foot) {
        foot_site_ids_[foot] =
            mj_name2id(model_, mjOBJ_SITE, kFootNames[foot]);
        foot_geom_ids_[foot] =
            mj_name2id(model_, mjOBJ_GEOM, kFootNames[foot]);
        if (foot_site_ids_[foot] < 0 || foot_geom_ids_[foot] < 0)
            return Status::BAD_ARGUMENT;

        for (int joint = 0; joint < kGo1JointsPerLeg; ++joint) {
            const int joint_id =
                mj_name2id(model_, mjOBJ_JOINT, kJointNames[foot][joint]);
            if (joint_id < 0 || model_->jnt_type[joint_id] != mjJNT_HINGE)
                return Status::BAD_ARGUMENT;
            joint_ids_[foot][joint] = joint_id;
            joint_qpos_addresses_[foot][joint] = model_->jnt_qposadr[joint_id];
            joint_dof_addresses_[foot][joint] = model_->jnt_dofadr[joint_id];

            int matched_actuator = -1;
            for (int actuator = 0; actuator < model_->nu; ++actuator) {
                if (model_->actuator_trntype[actuator] == mjTRN_JOINT &&
                    model_->actuator_trnid[2 * actuator] == joint_id) {
                    if (matched_actuator >= 0) return Status::BAD_ARGUMENT;
                    matched_actuator = actuator;
                }
            }
            if (matched_actuator < 0) return Status::BAD_ARGUMENT;
            actuator_ids_[foot][joint] = matched_actuator;
        }
    }

    initialized_ = true;
    return Status::SUCCESS;
}

Status MujocoGo1Adapter::read_whole_body_state(WholeBodyState& state) const {
    if (!initialized_ || !model_ || !data_) return Status::BAD_ARGUMENT;

    mj_subtreeVel(model_, data_);
    for (int axis = 0; axis < 3; ++axis) {
        state.base.position_world[axis] =
            data_->subtree_com[3 * trunk_body_id_ + axis];
    }
    for (int element = 0; element < 4; ++element) {
        state.base.orientation_body_to_world[element] =
            data_->xquat[4 * trunk_body_id_ + element];
    }

    mjtNum trunk_velocity_world[6] = {};
    mj_objectVelocity(model_, data_, mjOBJ_BODY, trunk_body_id_,
                      trunk_velocity_world, 0);
    for (int axis = 0; axis < 3; ++axis) {
        state.base.linear_velocity_world[axis] =
            data_->subtree_linvel[3 * trunk_body_id_ + axis];
    }

    Mat<3, 3> rotation;
    unit_rotation_matrix(state.base.orientation_body_to_world, rotation);
    Vec<3> angular_velocity_world;
    for (int axis = 0; axis < 3; ++axis)
        angular_velocity_world[axis] = trunk_velocity_world[axis];
    rotation.mul_vec_transpose(angular_velocity_world,
                               state.base.angular_velocity_body);

    std::vector<mjtNum> jacobian(3 * model_->nv, 0.0);
    for (int foot = 0; foot < kNumFeet; ++foot) {
        for (int axis = 0; axis < 3; ++axis) {
            state.foot_positions_world[foot][axis] =
                data_->site_xpos[3 * foot_site_ids_[foot] + axis];
        }
        mjtNum foot_velocity_world[6] = {};
        mj_objectVelocity(model_, data_, mjOBJ_SITE, foot_site_ids_[foot],
                          foot_velocity_world, 0);
        for (int axis = 0; axis < 3; ++axis) {
            state.foot_velocities_world[foot][axis] =
                foot_velocity_world[3 + axis];
        }

        std::fill(jacobian.begin(), jacobian.end(), 0.0);
        mj_jacSite(model_, data_, jacobian.data(), nullptr,
                   foot_site_ids_[foot]);
        for (int joint = 0; joint < kGo1JointsPerLeg; ++joint) {
            Vec<3> column_world;
            const int dof = joint_dof_addresses_[foot][joint];
            for (int axis = 0; axis < 3; ++axis)
                column_world[axis] = jacobian[axis * model_->nv + dof];
            Vec<3> column_body;
            rotation.mul_vec_transpose(column_world, column_body);
            for (int axis = 0; axis < 3; ++axis)
                state.foot_jacobians_body[foot](axis, joint) =
                    column_body[axis];
        }
        for (int joint = 0; joint < kGo1JointsPerLeg; ++joint) {
            state.joint_positions[foot][joint] =
                data_->qpos[joint_qpos_addresses_[foot][joint]];
            state.joint_velocities[foot][joint] =
                data_->qvel[joint_dof_addresses_[foot][joint]];
            state.joint_bias_torques[foot][joint] =
                data_->qfrc_bias[joint_dof_addresses_[foot][joint]];
        }
        state.swing_feedforward_joint_torques[foot].zero();

        if (!finite3(state.foot_positions_world[foot]) ||
            !finite3(state.foot_velocities_world[foot]) ||
            !finite3(state.joint_positions[foot]) ||
            !finite3(state.joint_velocities[foot])) {
            return Status::NAN_DETECTED;
        }
    }
    return finite3(state.base.position_world) &&
                   finite3(state.base.linear_velocity_world) &&
                   finite3(state.base.angular_velocity_body)
        ? Status::SUCCESS
        : Status::NAN_DETECTED;
}

Status MujocoGo1Adapter::initialize_state_from_plan(
    const ContactPlanStage& initial_stage,
    MujocoInitializationReport* report) const {
    if (!initialized_ || !model_ || !data_) return Status::BAD_ARGUMENT;
    for (int velocity = 0; velocity < model_->nv; ++velocity)
        data_->qvel[velocity] = 0.0;
    for (int actuator = 0; actuator < model_->nu; ++actuator)
        data_->ctrl[actuator] = 0.0;
    for (int element = 0; element < 4; ++element) {
        data_->qpos[free_joint_qpos_address_ + 3 + element] =
            initial_stage.base.orientation_body_to_world[element];
    }

    constexpr int kMaximumIterations = 80;
    constexpr double kTolerance = 2e-7;
    constexpr double kDamping = 1e-6;
    std::vector<mjtNum> jacobian(3 * model_->nv, 0.0);
    int iterations = 0;
    for (; iterations < kMaximumIterations; ++iterations) {
        mj_forward(model_, data_);
        double base_error[3] = {};
        double maximum_base_error = 0.0;
        for (int axis = 0; axis < 3; ++axis) {
            base_error[axis] = initial_stage.base.position_world[axis] -
                data_->subtree_com[3 * trunk_body_id_ + axis];
            data_->qpos[free_joint_qpos_address_ + axis] += base_error[axis];
            maximum_base_error = std::max(maximum_base_error,
                                           std::fabs(base_error[axis]));
        }
        mj_forward(model_, data_);

        double maximum_foot_error = 0.0;
        for (int foot = 0; foot < kNumFeet; ++foot) {
            double error[3] = {};
            for (int axis = 0; axis < 3; ++axis) {
                error[axis] = initial_stage.feet[foot].position_world[axis] -
                    data_->site_xpos[3 * foot_site_ids_[foot] + axis];
                maximum_foot_error = std::max(maximum_foot_error,
                                               std::fabs(error[axis]));
            }
            std::fill(jacobian.begin(), jacobian.end(), 0.0);
            mj_jacSite(model_, data_, jacobian.data(), nullptr,
                       foot_site_ids_[foot]);
            double normal_matrix[3][3] = {};
            double normal_rhs[3] = {};
            for (int row = 0; row < 3; ++row) {
                for (int column = 0; column < 3; ++column) {
                    const int column_dof =
                        joint_dof_addresses_[foot][column];
                    normal_rhs[column] +=
                        jacobian[row * model_->nv + column_dof] * error[row];
                    for (int other = 0; other < 3; ++other) {
                        const int other_dof =
                            joint_dof_addresses_[foot][other];
                        normal_matrix[column][other] +=
                            jacobian[row * model_->nv + column_dof] *
                            jacobian[row * model_->nv + other_dof];
                    }
                }
            }
            for (int diagonal = 0; diagonal < 3; ++diagonal)
                normal_matrix[diagonal][diagonal] += kDamping;
            double increment[3] = {};
            if (!solve_damped_3x3(normal_matrix, normal_rhs, increment))
                return Status::INTERNAL_ERROR;
            for (int joint = 0; joint < kGo1JointsPerLeg; ++joint) {
                const int joint_id = joint_ids_[foot][joint];
                const int qpos = joint_qpos_addresses_[foot][joint];
                data_->qpos[qpos] += std::max(
                    -0.15, std::min(0.15, increment[joint]));
                if (model_->jnt_limited[joint_id]) {
                    data_->qpos[qpos] = std::max(
                        model_->jnt_range[2 * joint_id],
                        std::min(model_->jnt_range[2 * joint_id + 1],
                                 data_->qpos[qpos]));
                }
            }
        }
        if (maximum_base_error <= kTolerance &&
            maximum_foot_error <= kTolerance) {
            break;
        }
    }

    mj_forward(model_, data_);
    WholeBodyState state;
    if (read_whole_body_state(state) != Status::SUCCESS)
        return Status::NAN_DETECTED;
    double maximum_base_error = 0.0;
    double maximum_foot_error = 0.0;
    for (int axis = 0; axis < 3; ++axis) {
        maximum_base_error = std::max(
            maximum_base_error,
            std::fabs(state.base.position_world[axis] -
                      initial_stage.base.position_world[axis]));
    }
    for (int foot = 0; foot < kNumFeet; ++foot) {
        for (int axis = 0; axis < 3; ++axis) {
            maximum_foot_error = std::max(
                maximum_foot_error,
                std::fabs(state.foot_positions_world[foot][axis] -
                          initial_stage.feet[foot].position_world[axis]));
        }
    }
    if (report) {
        report->iterations = iterations;
        report->maximum_base_error = maximum_base_error;
        report->maximum_foot_error = maximum_foot_error;
    }
    return maximum_base_error <= 2e-6 && maximum_foot_error <= 2e-6
        ? Status::SUCCESS
        : Status::MAX_ITERATIONS;
}

Status MujocoGo1Adapter::read_foot_contacts(
    MujocoFootContact contacts[kNumFeet]) const {
    if (!initialized_ || !model_ || !data_ || !contacts)
        return Status::BAD_ARGUMENT;
    for (int foot = 0; foot < kNumFeet; ++foot) {
        contacts[foot].in_contact = false;
        contacts[foot].normal_force = 0.0;
        contacts[foot].force_world.zero();
    }

    for (int contact_id = 0; contact_id < data_->ncon; ++contact_id) {
        const mjContact& contact = data_->contact[contact_id];
        for (int foot = 0; foot < kNumFeet; ++foot) {
            const bool foot_is_geom1 = contact.geom1 == foot_geom_ids_[foot];
            const bool foot_is_geom2 = contact.geom2 == foot_geom_ids_[foot];
            if (!foot_is_geom1 && !foot_is_geom2) continue;
            const int other_geom = foot_is_geom1 ? contact.geom2 : contact.geom1;
            if (other_geom < 0 || model_->geom_bodyid[other_geom] != 0)
                continue;

            mjtNum contact_wrench[6] = {};
            mj_contactForce(model_, data_, contact_id, contact_wrench);
            const double sign = foot_is_geom2 ? 1.0 : -1.0;
            for (int axis = 0; axis < 3; ++axis) {
                double force = 0.0;
                for (int contact_axis = 0; contact_axis < 3;
                     ++contact_axis) {
                    force += contact.frame[3 * contact_axis + axis] *
                             contact_wrench[contact_axis];
                }
                contacts[foot].force_world[axis] += sign * force;
            }
            contacts[foot].normal_force += std::fabs(contact_wrench[0]);
            contacts[foot].in_contact = true;
        }
    }
    return Status::SUCCESS;
}

Status MujocoGo1Adapter::apply_command(
    const ConvexWBCCommand& command,
    MujocoActuatorReport reports[kNumFeet][kGo1JointsPerLeg]) const {
    if (!initialized_ || !model_ || !data_ || !reports)
        return Status::BAD_ARGUMENT;
    for (int foot = 0; foot < kNumFeet; ++foot) {
        for (int joint = 0; joint < kGo1JointsPerLeg; ++joint) {
            const double requested = command.joint_torques[foot][joint];
            const int actuator = actuator_ids_[foot][joint];
            const double gear = model_->actuator_gear[6 * actuator];
            if (!std::isfinite(requested) || !std::isfinite(gear) ||
                std::fabs(gear) <= 1e-12) {
                return Status::BAD_ARGUMENT;
            }
            const double requested_control = requested / gear;
            double limited_control = requested_control;
            double minimum_control = -std::numeric_limits<double>::infinity();
            double maximum_control = std::numeric_limits<double>::infinity();
            if (model_->actuator_ctrllimited[actuator]) {
                minimum_control = std::max(
                    minimum_control,
                    static_cast<double>(
                        model_->actuator_ctrlrange[2 * actuator]));
                maximum_control = std::min(
                    maximum_control,
                    static_cast<double>(
                        model_->actuator_ctrlrange[2 * actuator + 1]));
                limited_control = std::max(
                    model_->actuator_ctrlrange[2 * actuator],
                    std::min(model_->actuator_ctrlrange[2 * actuator + 1],
                             requested_control));
            }
            if (model_->actuator_forcelimited[actuator]) {
                minimum_control = std::max(
                    minimum_control,
                    static_cast<double>(
                        model_->actuator_forcerange[2 * actuator]));
                maximum_control = std::min(
                    maximum_control,
                    static_cast<double>(
                        model_->actuator_forcerange[2 * actuator + 1]));
                limited_control = std::max(
                    model_->actuator_forcerange[2 * actuator],
                    std::min(model_->actuator_forcerange[2 * actuator + 1],
                             limited_control));
            }
            data_->ctrl[actuator] = requested_control;
            reports[foot][joint].requested_joint_torque = requested;
            reports[foot][joint].predicted_joint_torque =
                limited_control * gear;
            reports[foot][joint].applied_joint_torque = 0.0;
            const double first_limit = minimum_control * gear;
            const double second_limit = maximum_control * gear;
            const double minimum_torque = std::min(first_limit, second_limit);
            const double maximum_torque = std::max(first_limit, second_limit);
            reports[foot][joint].requested_torque_headroom = std::min(
                requested - minimum_torque, maximum_torque - requested);
            reports[foot][joint].saturated =
                std::fabs(limited_control - requested_control) > 1e-12;
        }
    }
    return Status::SUCCESS;
}

Status MujocoGo1Adapter::update_applied_torques(
    MujocoActuatorReport reports[kNumFeet][kGo1JointsPerLeg]) const {
    if (!initialized_ || !model_ || !data_ || !reports)
        return Status::BAD_ARGUMENT;
    for (int foot = 0; foot < kNumFeet; ++foot) {
        for (int joint = 0; joint < kGo1JointsPerLeg; ++joint) {
            const int dof = joint_dof_addresses_[foot][joint];
            reports[foot][joint].applied_joint_torque =
                data_->qfrc_actuator[dof];
            reports[foot][joint].saturated =
                reports[foot][joint].saturated ||
                std::fabs(reports[foot][joint].applied_joint_torque -
                          reports[foot][joint].requested_joint_torque) > 1e-8;
        }
    }
    return Status::SUCCESS;
}

double MujocoGo1Adapter::total_mass() const {
    return model_ ? mj_getTotalmass(model_) : 0.0;
}

bool MujocoGo1Adapter::valid_leg_joint(int foot, int joint) const {
    return initialized_ && foot >= 0 && foot < kNumFeet && joint >= 0 &&
           joint < kGo1JointsPerLeg;
}

int MujocoGo1Adapter::joint_qpos_address(int foot, int joint) const {
    return valid_leg_joint(foot, joint)
        ? joint_qpos_addresses_[foot][joint]
        : -1;
}

int MujocoGo1Adapter::joint_dof_address(int foot, int joint) const {
    return valid_leg_joint(foot, joint)
        ? joint_dof_addresses_[foot][joint]
        : -1;
}

int MujocoGo1Adapter::actuator_id(int foot, int joint) const {
    return valid_leg_joint(foot, joint) ? actuator_ids_[foot][joint] : -1;
}

int MujocoGo1Adapter::foot_site_id(int foot) const {
    return initialized_ && foot >= 0 && foot < kNumFeet
        ? foot_site_ids_[foot]
        : -1;
}

}  // namespace quadruped_cito
