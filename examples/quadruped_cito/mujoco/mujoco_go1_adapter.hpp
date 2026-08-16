#pragma once

#include <mujoco/mujoco.h>

#include "examples/quadruped_cito/quadruped_cito_terrain.hpp"
#include "examples/quadruped_cito/quadruped_cito_wbc.hpp"

namespace quadruped_cito {

constexpr int kGo1JointsPerLeg = 3;

struct MujocoFootContact {
    bool in_contact = false;
    double normal_force = 0.0;
    Vec<3> force_world;
};

struct MujocoActuatorReport {
    double requested_joint_torque = 0.0;
    double predicted_joint_torque = 0.0;
    double applied_joint_torque = 0.0;
    double requested_torque_headroom = 0.0;
    bool saturated = false;
};

struct MujocoTerrainReport {
    int rows = 0;
    int columns = 0;
    double minimum_height = 0.0;
    double maximum_height = 0.0;
    double maximum_grid_error = 0.0;
};

struct MujocoInitializationReport {
    int iterations = 0;
    double maximum_base_error = 0.0;
    double maximum_foot_error = 0.0;
};

Status configure_go1_torque_actuators(mjModel* model);
Status configure_shared_terrain(mjModel* model,
                                const SharedTerrain& terrain,
                                MujocoTerrainReport* report = nullptr);

class MujocoGo1Adapter {
public:
    MujocoGo1Adapter(const mjModel* model, mjData* data)
        : model_(model), data_(data) {}

    Status initialize();
    bool initialized() const { return initialized_; }

    Status read_whole_body_state(WholeBodyState& state) const;
    Status initialize_state_from_plan(
        const ContactPlanStage& initial_stage,
        MujocoInitializationReport* report = nullptr) const;
    Status read_foot_contacts(MujocoFootContact contacts[kNumFeet]) const;
    Status apply_command(
        const ConvexWBCCommand& command,
        MujocoActuatorReport reports[kNumFeet][kGo1JointsPerLeg]) const;
    Status update_applied_torques(
        MujocoActuatorReport reports[kNumFeet][kGo1JointsPerLeg]) const;

    double time() const { return data_ ? data_->time : 0.0; }
    double total_mass() const;
    int joint_qpos_address(int foot, int joint) const;
    int joint_dof_address(int foot, int joint) const;
    int actuator_id(int foot, int joint) const;
    int foot_site_id(int foot) const;

private:
    bool valid_leg_joint(int foot, int joint) const;

    const mjModel* model_ = nullptr;
    mjData* data_ = nullptr;
    bool initialized_ = false;
    int trunk_body_id_ = -1;
    int free_joint_qpos_address_ = -1;
    int foot_site_ids_[kNumFeet] = {-1, -1, -1, -1};
    int foot_geom_ids_[kNumFeet] = {-1, -1, -1, -1};
    int joint_ids_[kNumFeet][kGo1JointsPerLeg] = {};
    int joint_qpos_addresses_[kNumFeet][kGo1JointsPerLeg] = {};
    int joint_dof_addresses_[kNumFeet][kGo1JointsPerLeg] = {};
    int actuator_ids_[kNumFeet][kGo1JointsPerLeg] = {};
};

}  // namespace quadruped_cito
