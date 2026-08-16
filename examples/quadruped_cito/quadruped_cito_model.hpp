#pragma once

#include <algorithm>
#include <cmath>

#include "nmpc/nmpc_problem.hpp"

namespace quadruped_cito {

using nmpc::ConstraintModel;
using nmpc::CostModel;
using nmpc::DynamicsModel;
using nmpc::Mat;
using nmpc::NMPCProblem;
using nmpc::Status;
using nmpc::Vec;

inline constexpr int kNumFeet = 4;
inline constexpr int kStateDim = 25;
inline constexpr int kControlDim = 28;
inline constexpr int kRowsPerFoot = 18;
inline constexpr int kPairsPerFoot = 2;
inline constexpr int kContactConstraintRows = kNumFeet * kRowsPerFoot;
inline constexpr int kSupportPairRows = kNumFeet * (kNumFeet - 1) / 2;
inline constexpr int kUserConstraintRows =
    kContactConstraintRows + kSupportPairRows;
inline constexpr int kComplementarityPairs = kNumFeet * kPairsPerFoot;
inline constexpr int kConstraintCapacity =
    kUserConstraintRows + kComplementarityPairs;

struct StateIndex {
    static constexpr int base_position(int axis) { return axis; }
    static constexpr int quaternion(int element) { return 3 + element; }
    static constexpr int linear_velocity(int axis) { return 7 + axis; }
    static constexpr int angular_velocity(int axis) { return 10 + axis; }
    static constexpr int foot_position(int foot, int axis) {
        return 13 + 3 * foot + axis;
    }
};

struct ControlIndex {
    static constexpr int contact_force(int foot, int axis) {
        return 3 * foot + axis;
    }
    static constexpr int foot_velocity(int foot, int axis) {
        return 12 + 3 * foot + axis;
    }
    static constexpr int motion_slack(int foot) { return 24 + foot; }
};

struct ContactRow {
    enum Offset {
        GAP = 0,
        NORMAL_FORCE = 1,
        FRICTION_POS_T1 = 2,
        FRICTION_NEG_T1 = 3,
        FRICTION_POS_T2 = 4,
        FRICTION_NEG_T2 = 5,
        MOTION_SLACK = 6,
        CLEARANCE_POS_T1 = 7,
        CLEARANCE_NEG_T1 = 8,
        CLEARANCE_POS_T2 = 9,
        CLEARANCE_NEG_T2 = 10,
        LEG_REACH = 11,
        VELOCITY_POS_X = 12,
        VELOCITY_NEG_X = 13,
        VELOCITY_POS_Y = 14,
        VELOCITY_NEG_Y = 15,
        VELOCITY_POS_Z = 16,
        VELOCITY_NEG_Z = 17
    };

    static constexpr int index(int foot, Offset offset) {
        return kRowsPerFoot * foot + static_cast<int>(offset);
    }

    static constexpr int velocity_envelope(int foot, int axis,
                                           bool positive) {
        return kRowsPerFoot * foot + 12 + 2 * axis +
               (positive ? 0 : 1);
    }

    static constexpr int clearance_envelope(int foot, int tangent,
                                            bool positive) {
        return kRowsPerFoot * foot + 7 + 2 * tangent +
               (positive ? 0 : 1);
    }

    static constexpr int support_pair(int pair) {
        return kContactConstraintRows + pair;
    }

    static constexpr int hard_support_force(int support_slot) {
        return kContactConstraintRows + 2 * support_slot;
    }

    static constexpr int hard_support_motion(int support_slot) {
        return kContactConstraintRows + 2 * support_slot + 1;
    }
};

struct RobotParameters {
    double mass = 15.0;
    double gravity = 9.81;
    double inertia[3] = {0.25, 0.80, 0.90};
    double friction = 0.60;
    double max_leg_reach = 0.55;
    double max_contact_force = 400.0;
    double max_foot_speed = 3.0;
    double gap_scale = 0.05;
    double force_scale = 40.0;
    double speed_scale = 1.0;
    double clearance_rate = 2.3;
    double clearance_tolerance = 7e-3;
    double minimum_pair_normal_force = 0.0;
};

inline double dot3(const Vec<3>& a, const Vec<3>& b) {
    return a[0] * b[0] + a[1] * b[1] + a[2] * b[2];
}

inline Vec<3> cross3(const Vec<3>& a, const Vec<3>& b) {
    Vec<3> result;
    result[0] = a[1] * b[2] - a[2] * b[1];
    result[1] = a[2] * b[0] - a[0] * b[2];
    result[2] = a[0] * b[1] - a[1] * b[0];
    return result;
}

inline void skew3(const Vec<3>& value, Mat<3, 3>& skew) {
    skew.zero();
    skew(0, 1) = -value[2];
    skew(0, 2) = value[1];
    skew(1, 0) = value[2];
    skew(1, 2) = -value[0];
    skew(2, 0) = -value[1];
    skew(2, 1) = value[0];
}

inline bool normalize_quaternion(const Vec<4>& raw, Vec<4>& unit,
                                 Mat<4, 4>* jacobian = nullptr) {
    const double norm = raw.norm2();
    if (!(norm > 1e-12) || !std::isfinite(norm)) return false;
    for (int i = 0; i < 4; ++i) unit[i] = raw[i] / norm;
    if (jacobian != nullptr) {
        for (int row = 0; row < 4; ++row) {
            for (int col = 0; col < 4; ++col) {
                const double identity = row == col ? 1.0 : 0.0;
                (*jacobian)(row, col) =
                    (identity - unit[row] * unit[col]) / norm;
            }
        }
    }
    return true;
}

inline void unit_rotation_matrix(const Vec<4>& q, Mat<3, 3>& rotation) {
    const double w = q[0];
    const double x = q[1];
    const double y = q[2];
    const double z = q[3];
    rotation(0, 0) = 1.0 - 2.0 * (y * y + z * z);
    rotation(0, 1) = 2.0 * (x * y - w * z);
    rotation(0, 2) = 2.0 * (x * z + w * y);
    rotation(1, 0) = 2.0 * (x * y + w * z);
    rotation(1, 1) = 1.0 - 2.0 * (x * x + z * z);
    rotation(1, 2) = 2.0 * (y * z - w * x);
    rotation(2, 0) = 2.0 * (x * z - w * y);
    rotation(2, 1) = 2.0 * (y * z + w * x);
    rotation(2, 2) = 1.0 - 2.0 * (x * x + y * y);
}

inline bool rotation_matrix_and_derivatives(const Vec<4>& raw,
                                            Mat<3, 3>& rotation,
                                            Mat<3, 3> derivatives[4]) {
    Vec<4> q;
    Mat<4, 4> normalization_jacobian;
    if (!normalize_quaternion(raw, q, &normalization_jacobian)) return false;
    unit_rotation_matrix(q, rotation);

    Mat<3, 3> unit_derivatives[4];
    for (auto& derivative : unit_derivatives) derivative.zero();
    const double w = q[0];
    const double x = q[1];
    const double y = q[2];
    const double z = q[3];

    unit_derivatives[0](0, 1) = -2.0 * z;
    unit_derivatives[0](0, 2) = 2.0 * y;
    unit_derivatives[0](1, 0) = 2.0 * z;
    unit_derivatives[0](1, 2) = -2.0 * x;
    unit_derivatives[0](2, 0) = -2.0 * y;
    unit_derivatives[0](2, 1) = 2.0 * x;

    unit_derivatives[1](0, 1) = 2.0 * y;
    unit_derivatives[1](0, 2) = 2.0 * z;
    unit_derivatives[1](1, 0) = 2.0 * y;
    unit_derivatives[1](1, 1) = -4.0 * x;
    unit_derivatives[1](1, 2) = -2.0 * w;
    unit_derivatives[1](2, 0) = 2.0 * z;
    unit_derivatives[1](2, 1) = 2.0 * w;
    unit_derivatives[1](2, 2) = -4.0 * x;

    unit_derivatives[2](0, 0) = -4.0 * y;
    unit_derivatives[2](0, 1) = 2.0 * x;
    unit_derivatives[2](0, 2) = 2.0 * w;
    unit_derivatives[2](1, 0) = 2.0 * x;
    unit_derivatives[2](1, 2) = 2.0 * z;
    unit_derivatives[2](2, 0) = -2.0 * w;
    unit_derivatives[2](2, 1) = 2.0 * z;
    unit_derivatives[2](2, 2) = -4.0 * y;

    unit_derivatives[3](0, 0) = -4.0 * z;
    unit_derivatives[3](0, 1) = -2.0 * w;
    unit_derivatives[3](0, 2) = 2.0 * x;
    unit_derivatives[3](1, 0) = 2.0 * w;
    unit_derivatives[3](1, 1) = -4.0 * z;
    unit_derivatives[3](1, 2) = 2.0 * y;
    unit_derivatives[3](2, 0) = 2.0 * x;
    unit_derivatives[3](2, 1) = 2.0 * y;

    for (int raw_element = 0; raw_element < 4; ++raw_element) {
        derivatives[raw_element].zero();
        for (int unit_element = 0; unit_element < 4; ++unit_element) {
            const double chain =
                normalization_jacobian(unit_element, raw_element);
            for (int row = 0; row < 3; ++row) {
                for (int col = 0; col < 3; ++col) {
                    derivatives[raw_element](row, col) +=
                        unit_derivatives[unit_element](row, col) * chain;
                }
            }
        }
    }
    return true;
}

inline Vec<4> state_quaternion(const Vec<kStateDim>& state) {
    Vec<4> quaternion;
    for (int i = 0; i < 4; ++i)
        quaternion[i] = state[StateIndex::quaternion(i)];
    return quaternion;
}

inline Vec<3> state_vector3(const Vec<kStateDim>& state, int start) {
    Vec<3> value;
    for (int i = 0; i < 3; ++i) value[i] = state[start + i];
    return value;
}

inline Vec<3> control_vector3(const Vec<kControlDim>& control, int start) {
    Vec<3> value;
    for (int i = 0; i < 3; ++i) value[i] = control[start + i];
    return value;
}

struct TerrainSample {
    double gap = 0.0;
    Vec<3> gap_gradient;
    Vec<3> normal;
    Vec<3> tangent1;
    Vec<3> tangent2;
    Mat<3, 3> normal_gradient;
    Mat<3, 3> tangent1_gradient;
    Mat<3, 3> tangent2_gradient;
};

inline bool sample_height_field(const Vec<3>& position, double height,
                                double hx, double hy, double hxx,
                                double hxy, double hyy,
                                TerrainSample& result) {
    result.gap = position[2] - height;
    result.gap_gradient[0] = -hx;
    result.gap_gradient[1] = -hy;
    result.gap_gradient[2] = 1.0;

    Vec<3> raw_normal = result.gap_gradient;
    const double normal_length = raw_normal.norm2();
    if (!(normal_length > 1e-12) || !std::isfinite(normal_length))
        return false;
    for (int axis = 0; axis < 3; ++axis)
        result.normal[axis] = raw_normal[axis] / normal_length;

    Vec<3> raw_tangent1;
    raw_tangent1[0] = 1.0;
    raw_tangent1[1] = 0.0;
    raw_tangent1[2] = hx;
    const double tangent_length = raw_tangent1.norm2();
    for (int axis = 0; axis < 3; ++axis)
        result.tangent1[axis] = raw_tangent1[axis] / tangent_length;
    result.tangent2 = cross3(result.normal, result.tangent1);

    result.normal_gradient.zero();
    result.tangent1_gradient.zero();
    result.tangent2_gradient.zero();
    const double raw_normal_derivatives[3][2] = {
        {-hxx, -hxy}, {-hxy, -hyy}, {0.0, 0.0}};
    const double raw_tangent_derivatives[3][2] = {
        {0.0, 0.0}, {0.0, 0.0}, {hxx, hxy}};
    for (int coordinate = 0; coordinate < 2; ++coordinate) {
        Vec<3> normal_derivative;
        Vec<3> tangent_derivative;
        double normal_projection = 0.0;
        double tangent_projection = 0.0;
        for (int axis = 0; axis < 3; ++axis) {
            normal_projection += result.normal[axis] *
                raw_normal_derivatives[axis][coordinate];
            tangent_projection += result.tangent1[axis] *
                raw_tangent_derivatives[axis][coordinate];
        }
        for (int axis = 0; axis < 3; ++axis) {
            normal_derivative[axis] =
                (raw_normal_derivatives[axis][coordinate] -
                 result.normal[axis] * normal_projection) /
                normal_length;
            tangent_derivative[axis] =
                (raw_tangent_derivatives[axis][coordinate] -
                 result.tangent1[axis] * tangent_projection) /
                tangent_length;
            result.normal_gradient(axis, coordinate) =
                normal_derivative[axis];
            result.tangent1_gradient(axis, coordinate) =
                tangent_derivative[axis];
        }
        const Vec<3> tangent2_derivative =
            cross3(normal_derivative, result.tangent1);
        const Vec<3> frame_derivative =
            cross3(result.normal, tangent_derivative);
        for (int axis = 0; axis < 3; ++axis) {
            result.tangent2_gradient(axis, coordinate) =
                tangent2_derivative[axis] + frame_derivative[axis];
        }
    }
    return true;
}

struct PlaneTerrain {
    static constexpr bool kAffineFrame = true;

    Vec<3> normal;
    Vec<3> tangent1;
    Vec<3> tangent2;
    double offset = 0.0;

    PlaneTerrain() { set_plane(0.0, 0.0, 1.0, 0.0); }

    bool set_plane(double nx, double ny, double nz, double plane_offset) {
        Vec<3> requested;
        requested[0] = nx;
        requested[1] = ny;
        requested[2] = nz;
        const double length = requested.norm2();
        if (!(length > 1e-12) || !std::isfinite(length)) return false;
        for (int i = 0; i < 3; ++i) normal[i] = requested[i] / length;
        offset = plane_offset;

        Vec<3> reference;
        reference.zero();
        reference[std::fabs(normal[2]) < 0.9 ? 2 : 0] = 1.0;
        tangent1 = cross3(reference, normal);
        const double tangent_length = tangent1.norm2();
        for (int i = 0; i < 3; ++i) tangent1[i] /= tangent_length;
        tangent2 = cross3(normal, tangent1);
        return true;
    }

    double gap(const Vec<3>& position) const {
        return dot3(normal, position) - offset;
    }

    bool sample(const Vec<3>& position, TerrainSample& result) const {
        result.gap = gap(position);
        result.gap_gradient = normal;
        result.normal = normal;
        result.tangent1 = tangent1;
        result.tangent2 = tangent2;
        result.normal_gradient.zero();
        result.tangent1_gradient.zero();
        result.tangent2_gradient.zero();
        return true;
    }
};

struct SinusoidalHeightTerrain {
    static constexpr bool kAffineFrame = false;

    double offset = 0.0;
    double slope_x = 0.0;
    double slope_y = 0.0;
    double amplitude = 0.04;
    double wave_number_x = 4.0;
    double wave_number_y = 3.0;

    double height(double x, double y) const {
        return offset + slope_x * x + slope_y * y +
               amplitude * std::sin(wave_number_x * x) *
                   std::sin(wave_number_y * y);
    }

    bool sample(const Vec<3>& position, TerrainSample& result) const {
        const double x = position[0];
        const double y = position[1];
        const double sx = std::sin(wave_number_x * x);
        const double cx = std::cos(wave_number_x * x);
        const double sy = std::sin(wave_number_y * y);
        const double cy = std::cos(wave_number_y * y);
        const double hx = slope_x +
            amplitude * wave_number_x * cx * sy;
        const double hy = slope_y +
            amplitude * wave_number_y * sx * cy;
        const double hxx = -amplitude * wave_number_x * wave_number_x *
                           sx * sy;
        const double hxy = amplitude * wave_number_x * wave_number_y *
                           cx * cy;
        const double hyy = -amplitude * wave_number_y * wave_number_y *
                           sx * sy;
        const double terrain_height = offset + slope_x * x + slope_y * y +
                                      amplitude * sx * sy;

        return sample_height_field(position, terrain_height, hx, hy, hxx,
                                   hxy, hyy, result);
    }
};

struct SmoothStepTerrain {
    static constexpr bool kAffineFrame = false;

    double offset = 0.0;
    double step_height = 0.04;
    double step_center_x = 0.34;
    double sharpness = 30.0;

    double height(double x, double) const {
        const double transition = std::tanh(sharpness *
                                            (x - step_center_x));
        return offset + 0.5 * step_height * (1.0 + transition);
    }

    bool sample(const Vec<3>& position, TerrainSample& result) const {
        const double transition = std::tanh(
            sharpness * (position[0] - step_center_x));
        const double sech_sq = 1.0 - transition * transition;
        const double hx = 0.5 * step_height * sharpness * sech_sq;
        const double hxx = -step_height * sharpness * sharpness *
                           sech_sq * transition;
        return sample_height_field(position,
                                   offset + 0.5 * step_height *
                                       (1.0 + transition),
                                   hx, 0.0, hxx, 0.0, 0.0, result);
    }
};

class SRBDDynamics final : public DynamicsModel<kStateDim, kControlDim> {
public:
    explicit SRBDDynamics(const RobotParameters& parameters = {})
        : parameters_(parameters) {}

    const RobotParameters& parameters() const { return parameters_; }

    Status discrete_step(const Vec<kStateDim>& state,
                         const Vec<kControlDim>& control, double dt,
                         Vec<kStateDim>& next) override {
        if (!(dt > 0.0) || !(parameters_.mass > 0.0))
            return Status::BAD_ARGUMENT;
        for (double inertia : parameters_.inertia)
            if (!(inertia > 0.0)) return Status::BAD_ARGUMENT;

        const Vec<4> raw_quaternion = state_quaternion(state);
        Vec<4> quaternion;
        if (!normalize_quaternion(raw_quaternion, quaternion))
            return Status::BAD_ARGUMENT;
        Mat<3, 3> rotation;
        unit_rotation_matrix(quaternion, rotation);

        const Vec<3> base_position =
            state_vector3(state, StateIndex::base_position(0));
        const Vec<3> linear_velocity =
            state_vector3(state, StateIndex::linear_velocity(0));
        const Vec<3> angular_velocity =
            state_vector3(state, StateIndex::angular_velocity(0));

        Vec<3> force_sum;
        Vec<3> torque_world;
        force_sum.zero();
        torque_world.zero();
        for (int foot = 0; foot < kNumFeet; ++foot) {
            const Vec<3> force = control_vector3(
                control, ControlIndex::contact_force(foot, 0));
            const Vec<3> foot_position = state_vector3(
                state, StateIndex::foot_position(foot, 0));
            Vec<3> lever;
            for (int axis = 0; axis < 3; ++axis) {
                force_sum[axis] += force[axis];
                lever[axis] = foot_position[axis] - base_position[axis];
            }
            const Vec<3> torque = cross3(lever, force);
            for (int axis = 0; axis < 3; ++axis)
                torque_world[axis] += torque[axis];
        }

        Vec<3> torque_body;
        rotation.mul_vec_transpose(torque_world, torque_body);
        Vec<3> inertia_omega;
        for (int axis = 0; axis < 3; ++axis)
            inertia_omega[axis] = parameters_.inertia[axis] *
                                  angular_velocity[axis];
        const Vec<3> gyroscopic = cross3(angular_velocity, inertia_omega);

        for (int axis = 0; axis < 3; ++axis) {
            next[StateIndex::base_position(axis)] =
                base_position[axis] + dt * linear_velocity[axis];
            const double gravity = axis == 2 ? parameters_.gravity : 0.0;
            next[StateIndex::linear_velocity(axis)] =
                linear_velocity[axis] +
                dt * (force_sum[axis] / parameters_.mass - gravity);
            next[StateIndex::angular_velocity(axis)] =
                angular_velocity[axis] +
                dt * (torque_body[axis] - gyroscopic[axis]) /
                    parameters_.inertia[axis];
        }

        Vec<4> quaternion_euler;
        const double wx = angular_velocity[0];
        const double wy = angular_velocity[1];
        const double wz = angular_velocity[2];
        quaternion_euler[0] = quaternion[0] - 0.5 * dt *
            (quaternion[1] * wx + quaternion[2] * wy + quaternion[3] * wz);
        quaternion_euler[1] = quaternion[1] + 0.5 * dt *
            (quaternion[0] * wx + quaternion[2] * wz - quaternion[3] * wy);
        quaternion_euler[2] = quaternion[2] + 0.5 * dt *
            (quaternion[0] * wy + quaternion[3] * wx - quaternion[1] * wz);
        quaternion_euler[3] = quaternion[3] + 0.5 * dt *
            (quaternion[0] * wz + quaternion[1] * wy - quaternion[2] * wx);
        Vec<4> quaternion_next;
        if (!normalize_quaternion(quaternion_euler, quaternion_next))
            return Status::BAD_ARGUMENT;
        for (int element = 0; element < 4; ++element)
            next[StateIndex::quaternion(element)] = quaternion_next[element];

        for (int foot = 0; foot < kNumFeet; ++foot) {
            for (int axis = 0; axis < 3; ++axis) {
                next[StateIndex::foot_position(foot, axis)] =
                    state[StateIndex::foot_position(foot, axis)] + dt *
                    control[ControlIndex::foot_velocity(foot, axis)];
            }
        }
        return Status::SUCCESS;
    }

    Status linearize(const Vec<kStateDim>& state,
                     const Vec<kControlDim>& control, double dt,
                     Mat<kStateDim, kStateDim>& A,
                     Mat<kStateDim, kControlDim>& B) override {
        if (!(dt > 0.0) || !(parameters_.mass > 0.0))
            return Status::BAD_ARGUMENT;
        for (double inertia : parameters_.inertia)
            if (!(inertia > 0.0)) return Status::BAD_ARGUMENT;
        A.zero();
        B.zero();

        for (int axis = 0; axis < 3; ++axis) {
            A(StateIndex::base_position(axis),
              StateIndex::base_position(axis)) = 1.0;
            A(StateIndex::base_position(axis),
              StateIndex::linear_velocity(axis)) = dt;
            A(StateIndex::linear_velocity(axis),
              StateIndex::linear_velocity(axis)) = 1.0;
            A(StateIndex::angular_velocity(axis),
              StateIndex::angular_velocity(axis)) = 1.0;
            for (int foot = 0; foot < kNumFeet; ++foot) {
                B(StateIndex::linear_velocity(axis),
                  ControlIndex::contact_force(foot, axis)) =
                    dt / parameters_.mass;
                A(StateIndex::foot_position(foot, axis),
                  StateIndex::foot_position(foot, axis)) = 1.0;
                B(StateIndex::foot_position(foot, axis),
                  ControlIndex::foot_velocity(foot, axis)) = dt;
            }
        }

        const Vec<4> raw_quaternion = state_quaternion(state);
        Vec<4> quaternion;
        Mat<4, 4> current_normalization_jacobian;
        if (!normalize_quaternion(raw_quaternion, quaternion,
                                  &current_normalization_jacobian))
            return Status::BAD_ARGUMENT;

        const Vec<3> angular_velocity =
            state_vector3(state, StateIndex::angular_velocity(0));
        Mat<4, 4> quaternion_step_wrt_unit;
        quaternion_step_wrt_unit.set_identity();
        const double wx = angular_velocity[0];
        const double wy = angular_velocity[1];
        const double wz = angular_velocity[2];
        const double half_dt = 0.5 * dt;
        const double omega_matrix[4][4] = {
            {0.0, -wx, -wy, -wz},
            {wx, 0.0, wz, -wy},
            {wy, -wz, 0.0, wx},
            {wz, wy, -wx, 0.0}};
        for (int row = 0; row < 4; ++row)
            for (int col = 0; col < 4; ++col)
                quaternion_step_wrt_unit(row, col) +=
                    half_dt * omega_matrix[row][col];

        Vec<4> quaternion_euler;
        quaternion_step_wrt_unit.mul_vec(quaternion, quaternion_euler);
        Vec<4> quaternion_next;
        Mat<4, 4> next_normalization_jacobian;
        if (!normalize_quaternion(quaternion_euler, quaternion_next,
                                  &next_normalization_jacobian))
            return Status::BAD_ARGUMENT;

        Mat<4, 4> step_wrt_raw;
        step_wrt_raw.zero();
        for (int row = 0; row < 4; ++row) {
            for (int col = 0; col < 4; ++col) {
                for (int middle1 = 0; middle1 < 4; ++middle1) {
                    for (int middle2 = 0; middle2 < 4; ++middle2) {
                        step_wrt_raw(row, col) +=
                            next_normalization_jacobian(row, middle1) *
                            quaternion_step_wrt_unit(middle1, middle2) *
                            current_normalization_jacobian(middle2, col);
                    }
                }
                A(StateIndex::quaternion(row),
                  StateIndex::quaternion(col)) = step_wrt_raw(row, col);
            }
        }

        const double quaternion_rate_wrt_omega[4][3] = {
            {-quaternion[1], -quaternion[2], -quaternion[3]},
            {quaternion[0], -quaternion[3], quaternion[2]},
            {quaternion[3], quaternion[0], -quaternion[1]},
            {-quaternion[2], quaternion[1], quaternion[0]}};
        for (int row = 0; row < 4; ++row) {
            for (int axis = 0; axis < 3; ++axis) {
                double value = 0.0;
                for (int middle = 0; middle < 4; ++middle) {
                    value += next_normalization_jacobian(row, middle) *
                             half_dt *
                             quaternion_rate_wrt_omega[middle][axis];
                }
                A(StateIndex::quaternion(row),
                  StateIndex::angular_velocity(axis)) = value;
            }
        }

        Mat<3, 3> rotation;
        Mat<3, 3> rotation_derivatives[4];
        if (!rotation_matrix_and_derivatives(raw_quaternion, rotation,
                                             rotation_derivatives))
            return Status::BAD_ARGUMENT;
        const Vec<3> base_position =
            state_vector3(state, StateIndex::base_position(0));
        Vec<3> torque_world;
        torque_world.zero();

        for (int foot = 0; foot < kNumFeet; ++foot) {
            const Vec<3> force = control_vector3(
                control, ControlIndex::contact_force(foot, 0));
            const Vec<3> foot_position = state_vector3(
                state, StateIndex::foot_position(foot, 0));
            Vec<3> lever;
            for (int axis = 0; axis < 3; ++axis)
                lever[axis] = foot_position[axis] - base_position[axis];
            const Vec<3> torque = cross3(lever, force);
            for (int axis = 0; axis < 3; ++axis)
                torque_world[axis] += torque[axis];

            Mat<3, 3> force_skew;
            skew3(force, force_skew);
            Mat<3, 3> lever_skew;
            skew3(lever, lever_skew);
            for (int body_axis = 0; body_axis < 3; ++body_axis) {
                const double scale = dt / parameters_.inertia[body_axis];
                for (int axis = 0; axis < 3; ++axis) {
                    double base_value = 0.0;
                    double foot_value = 0.0;
                    double force_value = 0.0;
                    for (int world_axis = 0; world_axis < 3; ++world_axis) {
                        base_value += rotation(world_axis, body_axis) *
                                      force_skew(world_axis, axis);
                        foot_value -= rotation(world_axis, body_axis) *
                                      force_skew(world_axis, axis);
                        force_value += rotation(world_axis, body_axis) *
                                       lever_skew(world_axis, axis);
                    }
                    A(StateIndex::angular_velocity(body_axis),
                      StateIndex::base_position(axis)) += scale * base_value;
                    A(StateIndex::angular_velocity(body_axis),
                      StateIndex::foot_position(foot, axis)) +=
                        scale * foot_value;
                    B(StateIndex::angular_velocity(body_axis),
                      ControlIndex::contact_force(foot, axis)) +=
                        scale * force_value;
                }
            }
        }

        for (int quaternion_element = 0; quaternion_element < 4;
             ++quaternion_element) {
            for (int body_axis = 0; body_axis < 3; ++body_axis) {
                double derivative = 0.0;
                for (int world_axis = 0; world_axis < 3; ++world_axis) {
                    derivative +=
                        rotation_derivatives[quaternion_element](world_axis,
                                                                 body_axis) *
                        torque_world[world_axis];
                }
                A(StateIndex::angular_velocity(body_axis),
                  StateIndex::quaternion(quaternion_element)) +=
                    dt * derivative / parameters_.inertia[body_axis];
            }
        }

        const double ix = parameters_.inertia[0];
        const double iy = parameters_.inertia[1];
        const double iz = parameters_.inertia[2];
        const double gyro_jacobian[3][3] = {
            {0.0, (iz - iy) * angular_velocity[2],
             (iz - iy) * angular_velocity[1]},
            {(ix - iz) * angular_velocity[2], 0.0,
             (ix - iz) * angular_velocity[0]},
            {(iy - ix) * angular_velocity[1],
             (iy - ix) * angular_velocity[0], 0.0}};
        for (int row = 0; row < 3; ++row) {
            for (int col = 0; col < 3; ++col) {
                A(StateIndex::angular_velocity(row),
                  StateIndex::angular_velocity(col)) -=
                    dt * gyro_jacobian[row][col] /
                    parameters_.inertia[row];
            }
        }
        return Status::SUCCESS;
    }

private:
    RobotParameters parameters_;
};

template <int Horizon>
class QuadraticTrackingCost final
    : public CostModel<kStateDim, kControlDim> {
public:
    QuadraticTrackingCost() {
        state_weights.zero();
        terminal_weights.zero();
        control_weights.zero();
    }

    void set_reference_all(const Vec<kStateDim>& reference) {
        for (int stage = 0; stage <= Horizon; ++stage)
            references[stage] = reference;
    }

    double stage_cost(const Vec<kStateDim>& state,
                      const Vec<kControlDim>& control, int stage) override {
        const int reference_stage = std::max(0, std::min(stage, Horizon));
        double value = 0.0;
        for (int i = 0; i < kStateDim; ++i) {
            const double error = state[i] - references[reference_stage][i];
            value += 0.5 * state_weights[i] * error * error;
        }
        for (int i = 0; i < kControlDim; ++i)
            value += 0.5 * control_weights[i] * control[i] * control[i];
        return value;
    }

    double terminal_cost(const Vec<kStateDim>& state) override {
        double value = 0.0;
        for (int i = 0; i < kStateDim; ++i) {
            const double error = state[i] - references[Horizon][i];
            value += 0.5 * terminal_weights[i] * error * error;
        }
        return value;
    }

    Status stage_gradient(const Vec<kStateDim>& state,
                          const Vec<kControlDim>& control, int stage,
                          Vec<kStateDim>& state_gradient,
                          Vec<kControlDim>& control_gradient) override {
        const int reference_stage = std::max(0, std::min(stage, Horizon));
        for (int i = 0; i < kStateDim; ++i) {
            state_gradient[i] = state_weights[i] *
                (state[i] - references[reference_stage][i]);
        }
        for (int i = 0; i < kControlDim; ++i)
            control_gradient[i] = control_weights[i] * control[i];
        return Status::SUCCESS;
    }

    Status stage_hessian(const Vec<kStateDim>&,
                         const Vec<kControlDim>&, int,
                         Mat<kStateDim, kStateDim>& Qxx,
                         Mat<kControlDim, kControlDim>& Quu,
                         Mat<kControlDim, kStateDim>& Qux) override {
        Qxx.zero();
        Quu.zero();
        Qux.zero();
        for (int i = 0; i < kStateDim; ++i) Qxx(i, i) = state_weights[i];
        for (int i = 0; i < kControlDim; ++i)
            Quu(i, i) = control_weights[i];
        return Status::SUCCESS;
    }

    Status terminal_gradient(const Vec<kStateDim>& state,
                             Vec<kStateDim>& gradient) override {
        for (int i = 0; i < kStateDim; ++i) {
            gradient[i] = terminal_weights[i] *
                (state[i] - references[Horizon][i]);
        }
        return Status::SUCCESS;
    }

    Status terminal_hessian(const Vec<kStateDim>&,
                            Mat<kStateDim, kStateDim>& Qxx) override {
        Qxx.zero();
        for (int i = 0; i < kStateDim; ++i)
            Qxx(i, i) = terminal_weights[i];
        return Status::SUCCESS;
    }

    Vec<kStateDim> references[Horizon + 1];
    Vec<kStateDim> state_weights;
    Vec<kStateDim> terminal_weights;
    Vec<kControlDim> control_weights;
};

template <int Horizon, typename Terrain = PlaneTerrain>
class ContactConstraints final
    : public ConstraintModel<kStateDim, kControlDim, kConstraintCapacity> {
public:
    ContactConstraints(const Terrain& terrain,
                       const RobotParameters& parameters = {})
        : terrain_(terrain), parameters_(parameters) {}

    Status set_designated_foot_schedule(
        const int* designated_feet, int stage_count,
        double minimum_normal_force = 10.0,
        double motion_slack_tolerance = 1e-4) {
        if (designated_feet == nullptr || stage_count != Horizon ||
            parameters_.minimum_pair_normal_force > 0.0 ||
            !std::isfinite(minimum_normal_force) ||
            !(minimum_normal_force > 0.0) ||
            !std::isfinite(motion_slack_tolerance) ||
            motion_slack_tolerance < 0.0) {
            return Status::BAD_ARGUMENT;
        }
        for (int stage = 0; stage < Horizon; ++stage) {
            if (designated_feet[stage] < 0 ||
                designated_feet[stage] >= kNumFeet) {
                return Status::BAD_ARGUMENT;
            }
        }
        for (int stage = 0; stage < Horizon; ++stage)
            designated_feet_[stage] = designated_feet[stage];
        hard_support_minimum_normal_force_ = minimum_normal_force;
        hard_support_motion_slack_tolerance_ = motion_slack_tolerance;
        hard_support_schedule_enabled_ = true;
        return Status::SUCCESS;
    }

    int num_constraints(int stage) const override {
        if (stage >= Horizon) return 0;
        return kContactConstraintRows +
            (hard_support_schedule_enabled_ ||
             parameters_.minimum_pair_normal_force > 0.0
                 ? kSupportPairRows
                 : 0);
    }

    int num_complementarity_pairs(int stage) const override {
        return stage < Horizon ? kComplementarityPairs : 0;
    }

    bool complementarity_pair(int stage, int pair, int& first,
                              int& second) const override {
        if (stage >= Horizon || pair < 0 || pair >= kComplementarityPairs)
            return false;
        const int foot = pair / kPairsPerFoot;
        if (pair % kPairsPerFoot == 0) {
            first = ContactRow::index(foot, ContactRow::GAP);
            second = ContactRow::index(foot, ContactRow::NORMAL_FORCE);
        } else {
            first = ContactRow::index(foot, ContactRow::NORMAL_FORCE);
            second = ContactRow::index(foot, ContactRow::MOTION_SLACK);
        }
        return true;
    }

    Status evaluate(const Vec<kStateDim>& state,
                    const Vec<kControlDim>& control, int stage,
                    Vec<kConstraintCapacity>& rows) override {
        TerrainSample terrain_samples[kNumFeet];
        const Status status = sample_feet(state, terrain_samples);
        return status == Status::SUCCESS
            ? evaluate_from_samples(state, control, stage, terrain_samples,
                                    rows)
            : status;
    }

    Status jacobian(const Vec<kStateDim>& state,
                    const Vec<kControlDim>& control, int stage,
                    Mat<kConstraintCapacity, kStateDim>& Cx,
                    Mat<kConstraintCapacity, kControlDim>& Cu) override {
        TerrainSample terrain_samples[kNumFeet];
        const Status status = sample_feet(state, terrain_samples);
        return status == Status::SUCCESS
            ? jacobian_from_samples(state, control, stage, terrain_samples,
                                    Cx, Cu)
            : status;
    }

    Status evaluate_with_jacobian(
        const Vec<kStateDim>& state, const Vec<kControlDim>& control,
        int stage, Vec<kConstraintCapacity>& rows,
        Mat<kConstraintCapacity, kStateDim>& Cx,
        Mat<kConstraintCapacity, kControlDim>& Cu) override {
        TerrainSample terrain_samples[kNumFeet];
        Status status = sample_feet(state, terrain_samples);
        if (status != Status::SUCCESS) return status;
        status = evaluate_from_samples(state, control, stage,
                                       terrain_samples, rows);
        return status == Status::SUCCESS
            ? jacobian_from_samples(state, control, stage, terrain_samples,
                                    Cx, Cu)
            : status;
    }

private:
    Status sample_feet(const Vec<kStateDim>& state,
                       TerrainSample terrain_samples[kNumFeet]) const {
        for (int foot = 0; foot < kNumFeet; ++foot) {
            const Vec<3> position = state_vector3(
                state, StateIndex::foot_position(foot, 0));
            if (!terrain_.sample(position, terrain_samples[foot]))
                return Status::BAD_ARGUMENT;
        }
        return Status::SUCCESS;
    }

    Status evaluate_from_samples(
        const Vec<kStateDim>& state, const Vec<kControlDim>& control, int stage,
        const TerrainSample terrain_samples[kNumFeet],
        Vec<kConstraintCapacity>& rows) const {
        rows.zero();
        if (hard_support_schedule_enabled_ &&
            (stage < 0 || stage >= Horizon)) {
            return Status::BAD_ARGUMENT;
        }
        double normal_forces[kNumFeet];
        for (int foot = 0; foot < kNumFeet; ++foot) {
            const Vec<3> foot_position = state_vector3(
                state, StateIndex::foot_position(foot, 0));
            const Vec<3> base_position = state_vector3(
                state, StateIndex::base_position(0));
            const Vec<3> force = control_vector3(
                control, ControlIndex::contact_force(foot, 0));
            const Vec<3> foot_velocity = control_vector3(
                control, ControlIndex::foot_velocity(foot, 0));
            const double motion_slack =
                control[ControlIndex::motion_slack(foot)];
            Vec<3> lever;
            for (int axis = 0; axis < 3; ++axis)
                lever[axis] = foot_position[axis] - base_position[axis];

            const TerrainSample& terrain_sample = terrain_samples[foot];
            const double normal_force = dot3(terrain_sample.normal, force);
            normal_forces[foot] = normal_force;
            const double tangent_force1 =
                dot3(terrain_sample.tangent1, force);
            const double tangent_force2 =
                dot3(terrain_sample.tangent2, force);
            const double reach_scale_sq =
                parameters_.max_leg_reach * parameters_.max_leg_reach;
            rows[ContactRow::index(foot, ContactRow::GAP)] =
                -terrain_sample.gap / parameters_.gap_scale;
            rows[ContactRow::index(foot, ContactRow::NORMAL_FORCE)] =
                -normal_force / parameters_.force_scale;
            rows[ContactRow::index(foot, ContactRow::FRICTION_POS_T1)] =
                (tangent_force1 - parameters_.friction * normal_force) /
                parameters_.force_scale;
            rows[ContactRow::index(foot, ContactRow::FRICTION_NEG_T1)] =
                (-tangent_force1 - parameters_.friction * normal_force) /
                parameters_.force_scale;
            rows[ContactRow::index(foot, ContactRow::FRICTION_POS_T2)] =
                (tangent_force2 - parameters_.friction * normal_force) /
                parameters_.force_scale;
            rows[ContactRow::index(foot, ContactRow::FRICTION_NEG_T2)] =
                (-tangent_force2 - parameters_.friction * normal_force) /
                parameters_.force_scale;
            rows[ContactRow::index(foot, ContactRow::MOTION_SLACK)] =
                -motion_slack / parameters_.speed_scale;
            if (hard_support_schedule_enabled_ &&
                foot != designated_feet_[stage]) {
                const int support_slot = foot < designated_feet_[stage]
                    ? foot : foot - 1;
                rows[ContactRow::hard_support_force(support_slot)] =
                    (hard_support_minimum_normal_force_ - normal_force) /
                    parameters_.force_scale;
                rows[ContactRow::hard_support_motion(support_slot)] =
                    (motion_slack - hard_support_motion_slack_tolerance_) /
                    parameters_.speed_scale;
            }
            for (int axis = 0; axis < 3; ++axis) {
                rows[ContactRow::velocity_envelope(foot, axis, true)] =
                    (foot_velocity[axis] - motion_slack) /
                    parameters_.speed_scale;
                rows[ContactRow::velocity_envelope(foot, axis, false)] =
                    (-foot_velocity[axis] - motion_slack) /
                    parameters_.speed_scale;
            }
            const double tangent_velocity1 =
                dot3(terrain_sample.tangent1, foot_velocity);
            const double tangent_velocity2 =
                dot3(terrain_sample.tangent2, foot_velocity);
            const double clearance_limit = parameters_.clearance_rate *
                (terrain_sample.gap + parameters_.clearance_tolerance);
            rows[ContactRow::clearance_envelope(foot, 0, true)] =
                (tangent_velocity1 - clearance_limit) /
                parameters_.speed_scale;
            rows[ContactRow::clearance_envelope(foot, 0, false)] =
                (-tangent_velocity1 - clearance_limit) /
                parameters_.speed_scale;
            rows[ContactRow::clearance_envelope(foot, 1, true)] =
                (tangent_velocity2 - clearance_limit) /
                parameters_.speed_scale;
            rows[ContactRow::clearance_envelope(foot, 1, false)] =
                (-tangent_velocity2 - clearance_limit) /
                parameters_.speed_scale;
            rows[ContactRow::index(foot, ContactRow::LEG_REACH)] =
                (lever.norm2_sq() - reach_scale_sq) / reach_scale_sq;
        }
        if (hard_support_schedule_enabled_) return Status::SUCCESS;
        int pair = 0;
        for (int first = 0; first < kNumFeet; ++first) {
            for (int second = first + 1; second < kNumFeet;
                 ++second, ++pair) {
                rows[ContactRow::support_pair(pair)] =
                    (parameters_.minimum_pair_normal_force -
                     normal_forces[first] - normal_forces[second]) /
                    parameters_.force_scale;
            }
        }
        return Status::SUCCESS;
    }

    Status evaluate_terminal(const Vec<kStateDim>&,
                             Vec<kConstraintCapacity>& rows) override {
        rows.zero();
        return Status::SUCCESS;
    }

    Status jacobian_from_samples(
        const Vec<kStateDim>& state, const Vec<kControlDim>& control, int stage,
        const TerrainSample terrain_samples[kNumFeet],
        Mat<kConstraintCapacity, kStateDim>& Cx,
        Mat<kConstraintCapacity, kControlDim>& Cu) const {
        Cx.zero();
        Cu.zero();
        if (hard_support_schedule_enabled_ &&
            (stage < 0 || stage >= Horizon)) {
            return Status::BAD_ARGUMENT;
        }
        for (int foot = 0; foot < kNumFeet; ++foot) {
            const Vec<3> force = control_vector3(
                control, ControlIndex::contact_force(foot, 0));
            const TerrainSample& terrain_sample = terrain_samples[foot];
            const int gap_row = ContactRow::index(foot, ContactRow::GAP);
            const int normal_row =
                ContactRow::index(foot, ContactRow::NORMAL_FORCE);
            const int motion_row =
                ContactRow::index(foot, ContactRow::MOTION_SLACK);
            const int reach_row =
                ContactRow::index(foot, ContactRow::LEG_REACH);
            const Vec<3> foot_velocity = control_vector3(
                control, ControlIndex::foot_velocity(foot, 0));
            const int motion_control = ControlIndex::motion_slack(foot);
            const double reach_scale_sq =
                parameters_.max_leg_reach * parameters_.max_leg_reach;
            Cu(motion_row, motion_control) = -1.0 / parameters_.speed_scale;
            const int hard_support_slot = hard_support_schedule_enabled_ &&
                    foot != designated_feet_[stage]
                ? (foot < designated_feet_[stage] ? foot : foot - 1)
                : -1;
            if (hard_support_slot >= 0) {
                Cu(ContactRow::hard_support_motion(hard_support_slot),
                   motion_control) = 1.0 / parameters_.speed_scale;
            }
            for (int axis = 0; axis < 3; ++axis) {
                const int foot_state = StateIndex::foot_position(foot, axis);
                const int base_state = StateIndex::base_position(axis);
                const int force_control =
                    ControlIndex::contact_force(foot, axis);
                const int velocity_control =
                    ControlIndex::foot_velocity(foot, axis);
                const double lever = state[foot_state] - state[base_state];

                Cx(gap_row, foot_state) =
                    -terrain_sample.gap_gradient[axis] /
                    parameters_.gap_scale;
                double normal_position_derivative = 0.0;
                double tangent1_position_derivative = 0.0;
                double tangent2_position_derivative = 0.0;
                for (int vector_axis = 0; vector_axis < 3; ++vector_axis) {
                    normal_position_derivative +=
                        terrain_sample.normal_gradient(vector_axis, axis) *
                        force[vector_axis];
                    tangent1_position_derivative +=
                        terrain_sample.tangent1_gradient(vector_axis, axis) *
                        force[vector_axis];
                    tangent2_position_derivative +=
                        terrain_sample.tangent2_gradient(vector_axis, axis) *
                        force[vector_axis];
                }
                Cx(normal_row, foot_state) =
                    -normal_position_derivative / parameters_.force_scale;
                Cu(normal_row, force_control) =
                    -terrain_sample.normal[axis] / parameters_.force_scale;
                if (hard_support_slot >= 0) {
                    const int hard_force_row =
                        ContactRow::hard_support_force(hard_support_slot);
                    Cx(hard_force_row, foot_state) =
                        -normal_position_derivative /
                        parameters_.force_scale;
                    Cu(hard_force_row, force_control) =
                        -terrain_sample.normal[axis] /
                        parameters_.force_scale;
                }
                Cu(ContactRow::index(foot, ContactRow::FRICTION_POS_T1),
                   force_control) =
                    (terrain_sample.tangent1[axis] -
                     parameters_.friction * terrain_sample.normal[axis]) /
                    parameters_.force_scale;
                Cx(ContactRow::index(foot, ContactRow::FRICTION_POS_T1),
                   foot_state) =
                    (tangent1_position_derivative -
                     parameters_.friction * normal_position_derivative) /
                    parameters_.force_scale;
                Cu(ContactRow::index(foot, ContactRow::FRICTION_NEG_T1),
                   force_control) =
                    (-terrain_sample.tangent1[axis] -
                     parameters_.friction * terrain_sample.normal[axis]) /
                    parameters_.force_scale;
                Cx(ContactRow::index(foot, ContactRow::FRICTION_NEG_T1),
                   foot_state) =
                    (-tangent1_position_derivative -
                     parameters_.friction * normal_position_derivative) /
                    parameters_.force_scale;
                Cu(ContactRow::index(foot, ContactRow::FRICTION_POS_T2),
                   force_control) =
                    (terrain_sample.tangent2[axis] -
                     parameters_.friction * terrain_sample.normal[axis]) /
                    parameters_.force_scale;
                Cx(ContactRow::index(foot, ContactRow::FRICTION_POS_T2),
                   foot_state) =
                    (tangent2_position_derivative -
                     parameters_.friction * normal_position_derivative) /
                    parameters_.force_scale;
                Cu(ContactRow::index(foot, ContactRow::FRICTION_NEG_T2),
                   force_control) =
                    (-terrain_sample.tangent2[axis] -
                     parameters_.friction * terrain_sample.normal[axis]) /
                    parameters_.force_scale;
                Cx(ContactRow::index(foot, ContactRow::FRICTION_NEG_T2),
                   foot_state) =
                    (-tangent2_position_derivative -
                     parameters_.friction * normal_position_derivative) /
                    parameters_.force_scale;
                Cu(ContactRow::velocity_envelope(foot, axis, true),
                   velocity_control) = 1.0 / parameters_.speed_scale;
                Cu(ContactRow::velocity_envelope(foot, axis, true),
                   motion_control) = -1.0 / parameters_.speed_scale;
                Cu(ContactRow::velocity_envelope(foot, axis, false),
                   velocity_control) = -1.0 / parameters_.speed_scale;
                Cu(ContactRow::velocity_envelope(foot, axis, false),
                   motion_control) = -1.0 / parameters_.speed_scale;
                double tangent1_velocity_derivative = 0.0;
                double tangent2_velocity_derivative = 0.0;
                for (int vector_axis = 0; vector_axis < 3; ++vector_axis) {
                    tangent1_velocity_derivative +=
                        terrain_sample.tangent1_gradient(vector_axis, axis) *
                        foot_velocity[vector_axis];
                    tangent2_velocity_derivative +=
                        terrain_sample.tangent2_gradient(vector_axis, axis) *
                        foot_velocity[vector_axis];
                }
                const double clearance_position_derivatives[2] = {
                    tangent1_velocity_derivative,
                    tangent2_velocity_derivative};
                const double clearance_velocity_derivatives[2] = {
                    terrain_sample.tangent1[axis],
                    terrain_sample.tangent2[axis]};
                for (int tangent = 0; tangent < 2; ++tangent) {
                    const double gap_term = parameters_.clearance_rate *
                        terrain_sample.gap_gradient[axis];
                    const int positive_row = ContactRow::clearance_envelope(
                        foot, tangent, true);
                    const int negative_row = ContactRow::clearance_envelope(
                        foot, tangent, false);
                    Cx(positive_row, foot_state) =
                        (clearance_position_derivatives[tangent] - gap_term) /
                        parameters_.speed_scale;
                    Cx(negative_row, foot_state) =
                        (-clearance_position_derivatives[tangent] - gap_term) /
                        parameters_.speed_scale;
                    Cu(positive_row, velocity_control) =
                        clearance_velocity_derivatives[tangent] /
                        parameters_.speed_scale;
                    Cu(negative_row, velocity_control) =
                        -clearance_velocity_derivatives[tangent] /
                        parameters_.speed_scale;
                }
                Cx(reach_row, foot_state) = 2.0 * lever / reach_scale_sq;
                Cx(reach_row, base_state) = -2.0 * lever / reach_scale_sq;
            }
        }
        if (hard_support_schedule_enabled_) return Status::SUCCESS;
        int pair = 0;
        for (int first = 0; first < kNumFeet; ++first) {
            for (int second = first + 1; second < kNumFeet;
                 ++second, ++pair) {
                const int row = ContactRow::support_pair(pair);
                const int feet[2] = {first, second};
                for (int pair_foot : feet) {
                    const Vec<3> force = control_vector3(
                        control, ControlIndex::contact_force(pair_foot, 0));
                    const TerrainSample& terrain_sample =
                        terrain_samples[pair_foot];
                    for (int axis = 0; axis < 3; ++axis) {
                        double normal_position_derivative = 0.0;
                        for (int vector_axis = 0; vector_axis < 3;
                             ++vector_axis) {
                            normal_position_derivative +=
                                terrain_sample.normal_gradient(
                                    vector_axis, axis) * force[vector_axis];
                        }
                        Cx(row, StateIndex::foot_position(pair_foot, axis)) =
                            -normal_position_derivative /
                            parameters_.force_scale;
                        Cu(row, ControlIndex::contact_force(pair_foot, axis)) =
                            -terrain_sample.normal[axis] /
                            parameters_.force_scale;
                    }
                }
            }
        }
        return Status::SUCCESS;
    }

public:
    Status jacobian_terminal(const Vec<kStateDim>&,
                             Mat<kConstraintCapacity, kStateDim>& Cx) override {
        Cx.zero();
        return Status::SUCCESS;
    }

    bool provides_adjoint_hessian() const override {
        return Terrain::kAffineFrame;
    }

    Status adjoint_hessian(
        const Vec<kStateDim>&, const Vec<kControlDim>&, int stage,
        const Vec<kConstraintCapacity>& multipliers,
        Mat<kStateDim, kStateDim>& Hxx,
        Mat<kControlDim, kStateDim>&,
        Mat<kControlDim, kControlDim>&) override {
        if (stage >= Horizon) return Status::SUCCESS;
        for (int foot = 0; foot < kNumFeet; ++foot) {
            const double reach_multiplier =
                multipliers[ContactRow::index(foot, ContactRow::LEG_REACH)];
            for (int axis = 0; axis < 3; ++axis) {
                const int base = StateIndex::base_position(axis);
                const int foot_state = StateIndex::foot_position(foot, axis);
                const double reach_scale_sq =
                    parameters_.max_leg_reach * parameters_.max_leg_reach;
                Hxx(base, base) += 2.0 * reach_multiplier / reach_scale_sq;
                Hxx(foot_state, foot_state) +=
                    2.0 * reach_multiplier / reach_scale_sq;
                Hxx(base, foot_state) -=
                    2.0 * reach_multiplier / reach_scale_sq;
                Hxx(foot_state, base) -=
                    2.0 * reach_multiplier / reach_scale_sq;
            }
        }
        return Status::SUCCESS;
    }

private:
    const Terrain& terrain_;
    RobotParameters parameters_;
    int designated_feet_[Horizon] = {};
    double hard_support_minimum_normal_force_ = 10.0;
    double hard_support_motion_slack_tolerance_ = 1e-4;
    bool hard_support_schedule_enabled_ = false;
};

inline Vec<kStateDim> standing_state(double base_height = 0.32) {
    Vec<kStateDim> state;
    state.zero();
    state[StateIndex::base_position(2)] = base_height;
    state[StateIndex::quaternion(0)] = 1.0;
    const double foot_positions[kNumFeet][3] = {
        {0.30, 0.20, 0.0}, {0.30, -0.20, 0.0},
        {-0.30, 0.20, 0.0}, {-0.30, -0.20, 0.0}};
    for (int foot = 0; foot < kNumFeet; ++foot)
        for (int axis = 0; axis < 3; ++axis)
            state[StateIndex::foot_position(foot, axis)] =
                foot_positions[foot][axis];
    return state;
}

inline Vec<kControlDim> standing_control(const RobotParameters& parameters) {
    Vec<kControlDim> control;
    control.zero();
    const double normal_force = parameters.mass * parameters.gravity /
                                static_cast<double>(kNumFeet);
    for (int foot = 0; foot < kNumFeet; ++foot)
        control[ControlIndex::contact_force(foot, 2)] = normal_force;
    return control;
}

template <int Horizon>
using Problem = NMPCProblem<kStateDim, kControlDim, kConstraintCapacity,
                            Horizon>;

template <int Horizon, typename Terrain>
inline void initialize_standing_problem(
    Problem<Horizon>& problem, SRBDDynamics& dynamics,
    QuadraticTrackingCost<Horizon>& cost,
    ContactConstraints<Horizon, Terrain>& constraints, double dt,
    double base_height = 0.32) {
    problem.dynamics = &dynamics;
    problem.cost = &cost;
    problem.constraints = &constraints;
    problem.dt = dt;
    problem.init_bounds_free();

    const Vec<kStateDim> state = standing_state(base_height);
    const Vec<kControlDim> control = standing_control(dynamics.parameters());
    problem.x0 = state;
    cost.set_reference_all(state);
    for (int stage = 0; stage <= Horizon; ++stage) {
        problem.stages[stage].x = state;
        problem.stages[stage].u = control;
    }
    problem.stages[Horizon].u.zero();

    for (int element = 0; element < 4; ++element) {
        const int index = StateIndex::quaternion(element);
        problem.x_lb[index] = -1.1;
        problem.x_ub[index] = 1.1;
    }
    problem.n_bound_x = 4;
    for (int foot = 0; foot < kNumFeet; ++foot) {
        for (int axis = 0; axis < 3; ++axis) {
            const int force = ControlIndex::contact_force(foot, axis);
            problem.u_lb[force] = -dynamics.parameters().max_contact_force;
            problem.u_ub[force] = dynamics.parameters().max_contact_force;
            const int foot_velocity = ControlIndex::foot_velocity(foot, axis);
            problem.u_lb[foot_velocity] =
                -dynamics.parameters().max_foot_speed;
            problem.u_ub[foot_velocity] =
                dynamics.parameters().max_foot_speed;
        }
    }
    problem.n_bound_u = kControlDim;
}

}  // namespace quadruped_cito
