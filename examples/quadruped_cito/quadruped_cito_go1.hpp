#pragma once

#include "examples/quadruped_cito/quadruped_cito_terrain.hpp"

namespace quadruped_cito {

// Nominal Unitree Go1 values extracted from the MuJoCo Menagerie home
// keyframe.  The planner uses the articulated robot's subtree center of mass
// as its SRBD position.  The diagonal SRBD inertia intentionally omits the
// measured products of inertia, which remain available here for validation.
struct Go1SRBDCalibration {
    double mass = 12.743448;
    double composite_inertia_upper[6] = {
        0.126838993, 0.394032031, 0.422881102,
        -0.000437751, -0.014326178, -0.000091401};
    double com_position_world[3] = {
        -0.002112950, 0.000876778, 0.251008293};
    double foot_positions_world[kNumFeet][3] = {
        {0.188100000, 0.126750000, 0.005194154},
        {0.188100000, -0.126750000, 0.005194154},
        {-0.188100000, 0.126750000, 0.005194154},
        {-0.188100000, -0.126750000, 0.005194154}};
    // Menagerie foot collision-sphere radius.  The lower 5.194 mm home foot
    // center above the plane is a compliant equilibrium with penetration; it
    // is not the geometric zero-clearance surface used by CITO.
    double foot_center_contact_offset = 0.023;
    double nominal_ground_height = -0.017805846;
    double com_height_above_foot_centers = 0.245814139;
};

inline Go1SRBDCalibration go1_srbd_calibration() {
    return Go1SRBDCalibration{};
}

inline RobotParameters go1_robot_parameters() {
    const Go1SRBDCalibration calibration = go1_srbd_calibration();
    RobotParameters parameters;
    parameters.mass = calibration.mass;
    for (int axis = 0; axis < 3; ++axis)
        parameters.inertia[axis] = calibration.composite_inertia_upper[axis];
    parameters.friction = 0.8;
    parameters.max_leg_reach = 0.50;
    parameters.max_contact_force = 250.0;
    return parameters;
}

inline SharedTerrain register_go1_terrain(SharedTerrain terrain) {
    terrain.offset += go1_srbd_calibration().nominal_ground_height;
    return terrain;
}

// CITO plans the foot-site center while MuJoCo collides a compliant sphere.
// This adapter keeps both systems on the same physical terrain descriptor and
// shifts only the planner's zero-clearance surface by the collision radius.
template <typename PhysicalTerrain>
class Go1FootCenterTerrain {
public:
    static constexpr bool kAffineFrame = PhysicalTerrain::kAffineFrame;

    explicit Go1FootCenterTerrain(
        const PhysicalTerrain& physical_terrain,
        double contact_offset =
            Go1SRBDCalibration{}.foot_center_contact_offset)
        : physical_terrain_(physical_terrain), contact_offset_(contact_offset) {}

    bool sample(const Vec<3>& position, TerrainSample& result) const {
        if (!physical_terrain_.sample(position, result)) return false;
        result.gap -= contact_offset_;
        return true;
    }

    double height(double x, double y) const {
        Vec<3> position;
        position[0] = x;
        position[1] = y;
        position[2] = 0.0;
        TerrainSample terrain_sample;
        if (!sample(position, terrain_sample) ||
            std::fabs(terrain_sample.gap_gradient[2]) <= 1e-12) {
            return 0.0;
        }
        return -terrain_sample.gap / terrain_sample.gap_gradient[2];
    }

private:
    const PhysicalTerrain& physical_terrain_;
    double contact_offset_;
};

template <typename PlannerTerrain>
inline Vec<kStateDim> go1_home_state(const PlannerTerrain& terrain) {
    const Go1SRBDCalibration calibration = go1_srbd_calibration();
    Vec<kStateDim> state;
    state.zero();
    state[StateIndex::quaternion(0)] = 1.0;

    double mean_foot_height = 0.0;
    for (int foot = 0; foot < kNumFeet; ++foot) {
        Vec<3> position;
        position[0] = calibration.foot_positions_world[foot][0];
        position[1] = calibration.foot_positions_world[foot][1];
        position[2] = 0.0;
        TerrainSample sample;
        if (terrain.sample(position, sample) &&
            std::fabs(sample.gap_gradient[2]) > 1e-12) {
            position[2] -= sample.gap / sample.gap_gradient[2];
        } else {
            position[2] = calibration.foot_positions_world[foot][2];
        }
        mean_foot_height += position[2] / static_cast<double>(kNumFeet);
        for (int axis = 0; axis < 3; ++axis)
            state[StateIndex::foot_position(foot, axis)] = position[axis];
    }
    state[StateIndex::base_position(0)] = calibration.com_position_world[0];
    state[StateIndex::base_position(1)] = calibration.com_position_world[1];
    state[StateIndex::base_position(2)] =
        mean_foot_height + calibration.com_height_above_foot_centers;
    return state;
}

}  // namespace quadruped_cito
