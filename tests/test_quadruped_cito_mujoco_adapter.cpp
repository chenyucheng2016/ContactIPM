#include <algorithm>
#include <cmath>
#include <cstdint>
#include <cstdio>

#include <mujoco/mujoco.h>

#include "examples/quadruped_cito/quadruped_cito_go1.hpp"
#include "examples/quadruped_cito/mujoco/mujoco_go1_adapter.hpp"

namespace {

using namespace quadruped_cito;

void reset_home(const mjModel* model, mjData* data) {
    const int home_key = mj_name2id(model, mjOBJ_KEY, "home");
    if (home_key >= 0)
        mj_resetDataKeyframe(model, data, home_key);
    else
        mj_resetData(model, data);
    for (int actuator = 0; actuator < model->nu; ++actuator)
        data->ctrl[actuator] = 0.0;
}

double random_signed(std::uint32_t& state) {
    state = 1664525u * state + 1013904223u;
    const double unit = static_cast<double>(state) /
                        static_cast<double>(UINT32_MAX);
    return 2.0 * unit - 1.0;
}

bool contact_involves_geom(const mjData* data, int geom_id,
                           int* constraint_address = nullptr) {
    for (int contact_id = 0; contact_id < data->ncon; ++contact_id) {
        const mjContact& contact = data->contact[contact_id];
        if (contact.geom1 != geom_id && contact.geom2 != geom_id) continue;
        if (constraint_address) *constraint_address = contact.efc_address;
        return true;
    }
    return false;
}

double contact_normal_force(const mjModel* model, const mjData* data,
                            int geom_id) {
    double normal_force = 0.0;
    for (int contact_id = 0; contact_id < data->ncon; ++contact_id) {
        const mjContact& contact = data->contact[contact_id];
        if (contact.geom1 != geom_id && contact.geom2 != geom_id) continue;
        mjtNum wrench[6] = {};
        mj_contactForce(model, data, contact_id, wrench);
        normal_force += std::fabs(wrench[0]);
    }
    return normal_force;
}

int test_split_step_contact_force_synchronization() {
    static constexpr char kModelXml[] = R"(
<mujoco model="split-step contact phase">
  <option timestep="0.002" gravity="0 0 -9.81"
          solver="Newton" iterations="100" tolerance="1e-12"/>
  <default><geom condim="1"/></default>
  <worldbody>
    <geom name="floor" type="plane" size="1 1 0.1"/>
    <body pos="-0.1 0 0.045">
      <joint name="heavy_z" type="slide" axis="0 0 1"/>
      <geom name="heavy_foot" type="sphere" size="0.05" mass="10"/>
    </body>
    <body pos="0.1 0 0.049">
      <joint name="light_z" type="slide" axis="0 0 1"/>
      <geom name="light_foot" type="sphere" size="0.05" mass="0.1"/>
    </body>
  </worldbody>
</mujoco>)";

    char error[1024] = {};
    mjSpec* spec = mj_parseXMLString(kModelXml, nullptr, error, sizeof(error));
    if (!spec) {
        std::printf("failed to parse split-step contact model: %s\n", error);
        return 1;
    }
    mjModel* model = mj_compile(spec, nullptr);
    if (!model) {
        std::printf("failed to compile split-step contact model: %s\n",
                    mjs_getError(spec));
        mj_deleteSpec(spec);
        return 1;
    }
    mj_deleteSpec(spec);
    mjData* data = mj_makeData(model);
    if (!data) {
        mj_deleteModel(model);
        return 1;
    }

    const int heavy_joint = mj_name2id(model, mjOBJ_JOINT, "heavy_z");
    const int light_joint = mj_name2id(model, mjOBJ_JOINT, "light_z");
    const int heavy_geom = mj_name2id(model, mjOBJ_GEOM, "heavy_foot");
    const int light_geom = mj_name2id(model, mjOBJ_GEOM, "light_foot");
    if (heavy_joint < 0 || light_joint < 0 || heavy_geom < 0 ||
        light_geom < 0) {
        mj_deleteData(data);
        mj_deleteModel(model);
        return 1;
    }
    const int heavy_qpos = model->jnt_qposadr[heavy_joint];
    const int light_qpos = model->jnt_qposadr[light_joint];

    // Establish a solved, deeply loaded contact for the heavy sphere while
    // keeping the light sphere clear of the floor.
    data->qpos[heavy_qpos] = 0.0;
    data->qpos[light_qpos] = 0.151;
    mj_forward(model, data);
    mj_step1(model, data);
    mj_step2(model, data);
    int heavy_address = -1;
    const double heavy_post_force =
        contact_normal_force(model, data, heavy_geom);
    const bool heavy_post_contact =
        contact_involves_geom(data, heavy_geom, &heavy_address);
    const bool light_post_contact = contact_involves_geom(data, light_geom);

    // Switch the current collision topology without solving constraints: the
    // light sphere now has a shallow contact and the heavy sphere is clear.
    // mj_step1 rebuilds contacts but does not recompute efc_force.
    data->qpos[heavy_qpos] = 0.155;
    data->qpos[light_qpos] = 0.0;
    mju_zero(data->qvel, model->nv);
    mj_step1(model, data);
    int light_address = -1;
    const bool heavy_pre_contact = contact_involves_geom(data, heavy_geom);
    const bool light_pre_contact =
        contact_involves_geom(data, light_geom, &light_address);
    const double light_pre_force =
        contact_normal_force(model, data, light_geom);

    // The synchronized result is available only after the constraint solve in
    // mj_step2. It must reflect the light sphere's much smaller mass/depth.
    mj_step2(model, data);
    const double light_post_force =
        contact_normal_force(model, data, light_geom);
    const bool light_solved_contact = contact_involves_geom(data, light_geom);

    std::printf(
        "split-step contact phase: heavy_post=%.9f light_pre=%.9f "
        "light_post=%.9f addresses=(%d,%d)\n",
        heavy_post_force, light_pre_force, light_post_force,
        heavy_address, light_address);
    const double stale_error = std::fabs(light_pre_force - heavy_post_force);
    const double stale_tolerance =
        std::max(1e-10, 1e-12 * heavy_post_force);
    const bool passed =
        heavy_post_contact && !light_post_contact &&
        !heavy_pre_contact && light_pre_contact && light_solved_contact &&
        heavy_address >= 0 && light_address == heavy_address &&
        heavy_post_force > 100.0 && stale_error <= stale_tolerance &&
        light_post_force > 0.0 && light_post_force < 0.1 * light_pre_force;

    mj_deleteData(data);
    mj_deleteModel(model);
    if (!passed) {
        std::printf(
            "split-step contact forces were not stale before and synchronized "
            "after mj_step2 as expected\n");
        return 1;
    }
    return 0;
}

int test_shared_terrains(mjModel* model, mjData* data) {
    const SharedTerrain terrains[] = {
        SharedTerrain::flat(), SharedTerrain::sinusoidal(),
        SharedTerrain::slope(0.12), SharedTerrain::smooth_step(),
        SharedTerrain::random_smooth(7, 0.02)};
    for (const SharedTerrain& terrain : terrains) {
        MujocoTerrainReport report;
        if (configure_shared_terrain(model, terrain, &report) !=
            Status::SUCCESS) {
            std::printf("failed to configure shared %s terrain\n",
                        terrain_name(terrain.kind));
            return 1;
        }
        std::printf(
            "shared terrain %s: grid=%dx%d height=[%.6f, %.6f] "
            "max_grid_error=%.3e\n",
            terrain_name(terrain.kind), report.rows, report.columns,
            report.minimum_height, report.maximum_height,
            report.maximum_grid_error);
        if (report.rows < 129 || report.columns < 129 ||
            report.maximum_grid_error > 1e-7) {
            return 1;
        }
        mj_forward(model, data);
        const int terrain_geom =
            mj_name2id(model, mjOBJ_GEOM, "cito_terrain");
        const mjtByte terrain_group[mjNGROUP] = {1, 0, 0, 0, 0, 0};
        const double sample_points[][2] = {
            {0.1881, 0.12675}, {0.1881, -0.12675},
            {-0.1881, 0.12675}, {-0.1881, -0.12675}};
        double maximum_ray_error = 0.0;
        for (const auto& point : sample_points) {
            const mjtNum origin[3] = {point[0], point[1], 1.0};
            const mjtNum direction[3] = {0.0, 0.0, -1.0};
            int hit_geom = -1;
            mjtNum normal[3] = {};
            const double distance = mj_ray(
                model, data, origin, direction, terrain_group, 1, -1,
                &hit_geom, normal);
            if (!(distance >= 0.0) || hit_geom != terrain_geom) return 1;
            const double ray_height = origin[2] - distance;
            const double analytic_height =
                terrain.height(point[0], point[1]);
            std::printf(
                "    ray (%.5f, %.5f): mujoco=%.6f analytic=%.6f\n",
                point[0], point[1], ray_height, analytic_height);
            maximum_ray_error = std::max(
                maximum_ray_error,
                std::fabs(ray_height - analytic_height));
        }
        std::printf("  maximum MuJoCo ray-height error: %.3e\n",
                    maximum_ray_error);
        if (maximum_ray_error > 5e-4) return 1;
    }
    return configure_shared_terrain(model, SharedTerrain::flat()) ==
                   Status::SUCCESS
        ? 0
        : 1;
}

int test_plan_initialization(mjModel* model, mjData* data,
                             const MujocoGo1Adapter& adapter) {
    const SharedTerrain physical_terrains[] = {
        register_go1_terrain(SharedTerrain::flat()),
        register_go1_terrain(SharedTerrain::sinusoidal()),
        register_go1_terrain(SharedTerrain::smooth_step()),
        register_go1_terrain(SharedTerrain::slope(0.10)),
        register_go1_terrain(SharedTerrain::random_smooth(7, 0.02))};
    for (const SharedTerrain& physical_terrain : physical_terrains) {
        if (configure_shared_terrain(model, physical_terrain) !=
            Status::SUCCESS) {
            return 1;
        }
        reset_home(model, data);
        mj_forward(model, data);
        const Go1FootCenterTerrain<SharedTerrain> planner_terrain(
            physical_terrain);
        const Vec<kStateDim> initial = go1_home_state(planner_terrain);
        ContactPlanStage stage;
        stage.base = base_plan_sample(initial);
        for (int foot = 0; foot < kNumFeet; ++foot) {
            stage.feet[foot].position_world = state_vector3(
                initial, StateIndex::foot_position(foot, 0));
        }
        MujocoInitializationReport report;
        if (adapter.initialize_state_from_plan(stage, &report) !=
            Status::SUCCESS) {
            std::printf("Go1 %s plan initialization failed\n",
                        terrain_name(physical_terrain.kind));
            return 1;
        }
        std::printf(
            "Go1 %s plan initialization: iterations=%d base_error=%.3e "
            "foot_error=%.3e\n",
            terrain_name(physical_terrain.kind), report.iterations,
            report.maximum_base_error, report.maximum_foot_error);
    }
    configure_shared_terrain(
        model, register_go1_terrain(SharedTerrain::flat()));
    reset_home(model, data);
    mj_forward(model, data);
    return 0;
}

int audit_srbd_calibration(const mjModel* model, mjData* data,
                           const MujocoGo1Adapter& adapter) {
    reset_home(model, data);
    mj_forward(model, data);
    const int trunk = mj_name2id(model, mjOBJ_BODY, "trunk");
    if (trunk < 0) return 1;

    WholeBodyState state;
    if (adapter.read_whole_body_state(state) != Status::SUCCESS) return 1;
    const Go1SRBDCalibration calibration = go1_srbd_calibration();
    const mjtNum* composite_inertia = data->crb + 10 * trunk;
    std::printf(
        "Go1 SRBD calibration: mass=%.9f "
        "trunk=(%.9f, %.9f, %.9f) com=(%.9f, %.9f, %.9f) "
        "crb=(%.9f, %.9f, %.9f, %.9f, %.9f, %.9f, %.9f, %.9f, %.9f, %.9f)\n",
        mj_getTotalmass(model), data->xpos[3 * trunk],
        data->xpos[3 * trunk + 1], data->xpos[3 * trunk + 2],
        data->subtree_com[3 * trunk], data->subtree_com[3 * trunk + 1],
        data->subtree_com[3 * trunk + 2], composite_inertia[0],
        composite_inertia[1], composite_inertia[2], composite_inertia[3],
        composite_inertia[4], composite_inertia[5], composite_inertia[6],
        composite_inertia[7], composite_inertia[8], composite_inertia[9]);
    for (int foot = 0; foot < kNumFeet; ++foot) {
        std::printf(
            "  home foot %d=(%.9f, %.9f, %.9f), relative_com=(%.9f, %.9f, %.9f)\n",
            foot, state.foot_positions_world[foot][0],
            state.foot_positions_world[foot][1],
            state.foot_positions_world[foot][2],
            state.foot_positions_world[foot][0] - data->subtree_com[3 * trunk],
            state.foot_positions_world[foot][1] - data->subtree_com[3 * trunk + 1],
            state.foot_positions_world[foot][2] - data->subtree_com[3 * trunk + 2]);
    }

    double maximum_error = std::fabs(mj_getTotalmass(model) - calibration.mass);
    for (int axis = 0; axis < 3; ++axis) {
        maximum_error = std::max(
            maximum_error,
            std::fabs(state.base.position_world[axis] -
                      calibration.com_position_world[axis]));
    }
    for (int element = 0; element < 6; ++element) {
        maximum_error = std::max(
            maximum_error,
            std::fabs(composite_inertia[element] -
                      calibration.composite_inertia_upper[element]));
    }
    for (int foot = 0; foot < kNumFeet; ++foot) {
        for (int axis = 0; axis < 3; ++axis) {
            maximum_error = std::max(
                maximum_error,
                std::fabs(state.foot_positions_world[foot][axis] -
                          calibration.foot_positions_world[foot][axis]));
        }
        const char* foot_names[kNumFeet] = {"FL", "FR", "RL", "RR"};
        const int foot_geom =
            mj_name2id(model, mjOBJ_GEOM, foot_names[foot]);
        if (foot_geom < 0) return 1;
        maximum_error = std::max(
            maximum_error,
            std::fabs(model->geom_size[3 * foot_geom] -
                      calibration.foot_center_contact_offset));
    }
    const double maximum_product = std::max(
        std::fabs(composite_inertia[3]),
        std::max(std::fabs(composite_inertia[4]),
                 std::fabs(composite_inertia[5])));
    const double product_ratio = maximum_product /
        std::max(composite_inertia[0],
                 std::max(composite_inertia[1], composite_inertia[2]));
    std::printf(
        "Go1 calibration maximum error: %.3e, diagonal-SRBD omitted-product ratio: %.6f\n",
        maximum_error, product_ratio);
    if (maximum_error > 1e-8 || product_ratio > 0.05) {
        std::printf("Go1 SRBD calibration acceptance gate failed\n");
        return 1;
    }
    return 0;
}

int test_jacobians(const mjModel* model, mjData* data,
                   const MujocoGo1Adapter& adapter) {
    constexpr int kConfigurations = 100;
    constexpr double kPerturbation = 1e-7;
    std::uint32_t random_state = 20270810u;
    double maximum_error = 0.0;

    for (int configuration = 0; configuration < kConfigurations;
         ++configuration) {
        reset_home(model, data);
        for (int foot = 0; foot < kNumFeet; ++foot) {
            for (int joint = 0; joint < kGo1JointsPerLeg; ++joint) {
                const int qpos = adapter.joint_qpos_address(foot, joint);
                data->qpos[qpos] += 0.15 * random_signed(random_state);
            }
        }
        mj_forward(model, data);

        WholeBodyState state;
        if (adapter.read_whole_body_state(state) != Status::SUCCESS) {
            std::printf("adapter state read failed in Jacobian test\n");
            return 1;
        }
        Mat<3, 3> rotation;
        unit_rotation_matrix(state.base.orientation_body_to_world, rotation);

        for (int foot = 0; foot < kNumFeet; ++foot) {
            for (int joint = 0; joint < kGo1JointsPerLeg; ++joint) {
                const int qpos = adapter.joint_qpos_address(foot, joint);
                const double original = data->qpos[qpos];
                Vec<3> plus;
                Vec<3> minus;
                data->qpos[qpos] = original + kPerturbation;
                mj_forward(model, data);
                for (int axis = 0; axis < 3; ++axis) {
                    plus[axis] = data->site_xpos[
                        3 * adapter.foot_site_id(foot) + axis];
                }
                data->qpos[qpos] = original - kPerturbation;
                mj_forward(model, data);
                for (int axis = 0; axis < 3; ++axis) {
                    minus[axis] = data->site_xpos[
                        3 * adapter.foot_site_id(foot) + axis];
                }
                data->qpos[qpos] = original;

                Vec<3> analytic_body;
                for (int axis = 0; axis < 3; ++axis) {
                    analytic_body[axis] =
                        state.foot_jacobians_body[foot](axis, joint);
                }
                Vec<3> analytic_world;
                rotation.mul_vec(analytic_body, analytic_world);
                for (int axis = 0; axis < 3; ++axis) {
                    const double finite_difference =
                        (plus[axis] - minus[axis]) /
                        (2.0 * kPerturbation);
                    maximum_error = std::max(
                        maximum_error,
                        std::fabs(finite_difference - analytic_world[axis]));
                }
            }
        }
    }

    std::printf("Go1 maximum foot-Jacobian error: %.3e\n", maximum_error);
    if (maximum_error > 1e-5) {
        std::printf("Go1 foot Jacobian failed finite differences\n");
        return 1;
    }
    return 0;
}

int test_velocity_mapping(const mjModel* model, mjData* data,
                          const MujocoGo1Adapter& adapter) {
    reset_home(model, data);
    for (int dof = 0; dof < model->nv; ++dof) data->qvel[dof] = 0.0;
    for (int foot = 0; foot < kNumFeet; ++foot) {
        for (int joint = 0; joint < kGo1JointsPerLeg; ++joint) {
            data->qvel[adapter.joint_dof_address(foot, joint)] =
                0.1 * (1 + 3 * foot + joint);
        }
    }
    mj_forward(model, data);

    WholeBodyState state;
    if (adapter.read_whole_body_state(state) != Status::SUCCESS) return 1;
    Mat<3, 3> rotation;
    unit_rotation_matrix(state.base.orientation_body_to_world, rotation);
    double maximum_error = 0.0;
    for (int foot = 0; foot < kNumFeet; ++foot) {
        Vec<3> velocity_body;
        velocity_body.zero();
        for (int axis = 0; axis < 3; ++axis) {
            for (int joint = 0; joint < kGo1JointsPerLeg; ++joint) {
                velocity_body[axis] +=
                    state.foot_jacobians_body[foot](axis, joint) *
                    data->qvel[adapter.joint_dof_address(foot, joint)];
            }
        }
        Vec<3> velocity_world;
        rotation.mul_vec(velocity_body, velocity_world);
        for (int axis = 0; axis < 3; ++axis) {
            maximum_error = std::max(
                maximum_error,
                std::fabs(velocity_world[axis] -
                          state.foot_velocities_world[foot][axis]));
        }
        for (int joint = 0; joint < kGo1JointsPerLeg; ++joint) {
            maximum_error = std::max(
                maximum_error,
                std::fabs(state.joint_positions[foot][joint] -
                          data->qpos[adapter.joint_qpos_address(foot, joint)]));
            maximum_error = std::max(
                maximum_error,
                std::fabs(state.joint_velocities[foot][joint] -
                          data->qvel[adapter.joint_dof_address(foot, joint)]));
        }
    }
    std::printf("Go1 maximum foot-velocity mapping error: %.3e\n",
                maximum_error);
    return maximum_error <= 1e-10 ? 0 : 1;
}

int test_torque_mapping(const mjModel* model, mjData* data,
                        const MujocoGo1Adapter& adapter) {
    reset_home(model, data);
    mj_forward(model, data);
    ConvexWBCCommand command;
    MujocoActuatorReport reports[kNumFeet][kGo1JointsPerLeg];
    for (int foot = 0; foot < kNumFeet; ++foot) {
        for (int joint = 0; joint < kGo1JointsPerLeg; ++joint) {
            command.joint_torques[foot][joint] =
                0.25 * (1 + 3 * foot + joint);
        }
    }
    if (adapter.apply_command(command, reports) != Status::SUCCESS) return 1;
    mj_forward(model, data);
    if (adapter.update_applied_torques(reports) != Status::SUCCESS) return 1;

    double maximum_error = 0.0;
    double minimum_headroom = 1e300;
    for (int foot = 0; foot < kNumFeet; ++foot) {
        for (int joint = 0; joint < kGo1JointsPerLeg; ++joint) {
            maximum_error = std::max(
                maximum_error,
                std::fabs(reports[foot][joint].applied_joint_torque -
                          command.joint_torques[foot][joint]));
            minimum_headroom = std::min(
                minimum_headroom,
                reports[foot][joint].requested_torque_headroom);
        }
    }
    std::printf("Go1 maximum joint-torque mapping error: %.3e\n",
                maximum_error);
    if (maximum_error > 1e-10 || !(minimum_headroom > 0.0)) return 1;

    for (int foot = 0; foot < kNumFeet; ++foot) {
        for (int joint = 0; joint < kGo1JointsPerLeg; ++joint)
            command.joint_torques[foot][joint] = 1e6;
    }
    if (adapter.apply_command(command, reports) != Status::SUCCESS) return 1;
    for (int foot = 0; foot < kNumFeet; ++foot) {
        for (int joint = 0; joint < kGo1JointsPerLeg; ++joint) {
            if (!(reports[foot][joint].requested_torque_headroom < 0.0) ||
                !reports[foot][joint].saturated ||
                !(std::fabs(reports[foot][joint].predicted_joint_torque) <
                  std::fabs(reports[foot][joint].requested_joint_torque))) {
                return 1;
            }
        }
    }
    return 0;
}

int test_contact_mapping(const mjModel* model, mjData* data,
                         const MujocoGo1Adapter& adapter) {
    reset_home(model, data);
    int free_qpos = -1;
    for (int joint = 0; joint < model->njnt; ++joint) {
        if (model->jnt_type[joint] == mjJNT_FREE) {
            free_qpos = model->jnt_qposadr[joint];
            break;
        }
    }
    if (free_qpos < 0) {
        std::printf("Go1 model has no floating-base joint\n");
        return 1;
    }

    MujocoFootContact contacts[kNumFeet];
    bool found_contact = false;
    for (int step = 0; step <= 100 && !found_contact; ++step) {
        reset_home(model, data);
        data->qpos[free_qpos + 2] -= 1e-3 * step;
        mj_forward(model, data);
        if (adapter.read_foot_contacts(contacts) != Status::SUCCESS) return 1;
        for (int foot = 0; foot < kNumFeet; ++foot)
            found_contact = found_contact || contacts[foot].in_contact;
    }
    if (!found_contact) {
        std::printf("Go1 foot contact was not detected\n");
        return 1;
    }

    double normal_force = 0.0;
    double vertical_force = 0.0;
    for (int foot = 0; foot < kNumFeet; ++foot) {
        normal_force += contacts[foot].normal_force;
        vertical_force += contacts[foot].force_world[2];
    }
    std::printf(
        "Go1 contact mapping: normal_force=%.6f vertical_force=%.6f\n",
        normal_force, vertical_force);
    if (!(normal_force > 0.0) || !(vertical_force > 0.0)) {
        std::printf("Go1 contact-force direction is inconsistent\n");
        return 1;
    }
    return 0;
}

}  // namespace

int main(int argc, char** argv) {
    if (argc != 2) {
        std::printf("usage: test_quadruped_cito_mujoco_adapter "
                    "<go1-scene.xml>\n");
        return 2;
    }
    char error[1024] = {};
    mjModel* model = mj_loadXML(argv[1], nullptr, error, sizeof(error));
    if (!model) {
        std::printf("failed to load Go1 model: %s\n", error);
        return 1;
    }
    if (configure_go1_torque_actuators(model) != Status::SUCCESS) {
        std::printf("failed to configure Go1 torque actuators\n");
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
        std::printf("Go1 adapter initialization failed\n");
        mj_deleteData(data);
        mj_deleteModel(model);
        return 1;
    }

    int failures = 0;
    failures += test_split_step_contact_force_synchronization();
    failures += test_shared_terrains(model, data);
    failures += test_plan_initialization(model, data, adapter);
    failures += audit_srbd_calibration(model, data, adapter);
    failures += test_jacobians(model, data, adapter);
    failures += test_velocity_mapping(model, data, adapter);
    failures += test_torque_mapping(model, data, adapter);
    failures += test_contact_mapping(model, data, adapter);
    mj_deleteData(data);
    mj_deleteModel(model);

    if (failures == 0) {
        std::printf("MuJoCo Go1 adapter acceptance tests passed.\n");
        return 0;
    }
    std::printf("MuJoCo Go1 adapter acceptance tests failed: %d\n",
                failures);
    return 1;
}
