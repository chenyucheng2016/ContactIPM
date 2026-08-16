#pragma once

#include <algorithm>
#include <cmath>

#include "examples/quadruped_cito/quadruped_cito_plan.hpp"

namespace quadruped_cito {

struct WholeBodyState {
    BasePlanSample base;
    Vec<3> foot_positions_world[kNumFeet];
    Vec<3> foot_velocities_world[kNumFeet];
    Vec<3> joint_positions[kNumFeet];
    Vec<3> joint_velocities[kNumFeet];
    Mat<3, 3> foot_jacobians_body[kNumFeet];
    Vec<3> joint_bias_torques[kNumFeet];
    Vec<3> swing_feedforward_joint_torques[kNumFeet];
};

struct ConvexWBCParameters {
    Vec<3> base_position_kp;
    Vec<3> base_velocity_kd;
    Vec<3> base_orientation_kp;
    Vec<3> base_angular_velocity_kd;
    Vec<3> swing_position_kp;
    Vec<3> swing_velocity_kd;
    Vec<6> wrench_weights;
    double force_tracking_weight = 1.0;
    double friction = 0.6;
    double maximum_normal_force = 400.0;
    double contact_blend_force = 20.0;
    int maximum_iterations = 100;
    double convergence_tolerance = 1e-9;

    ConvexWBCParameters() {
        base_position_kp.set_constant(80.0);
        base_velocity_kd.set_constant(15.0);
        base_orientation_kp.set_constant(60.0);
        base_angular_velocity_kd.set_constant(8.0);
        swing_position_kp.set_constant(150.0);
        swing_velocity_kd.set_constant(8.0);
        for (int axis = 0; axis < 3; ++axis) {
            wrench_weights[axis] = 1.0;
            wrench_weights[3 + axis] = 2.0;
        }
    }
};

struct ConvexWBCCommand {
    Vec<3> contact_forces_world[kNumFeet];
    Vec<3> joint_torques[kNumFeet];
    Vec<3> swing_feedback_forces_world[kNumFeet];
    double contact_blend[kNumFeet] = {0.0, 0.0, 0.0, 0.0};
    Vec<6> wrench_residual_components;
    double wrench_residual = 0.0;
    int force_iterations = 0;
};

inline bool terrain_basis_from_normal(const Vec<3>& requested_normal,
                                      Vec<3>& normal, Vec<3>& tangent1,
                                      Vec<3>& tangent2) {
    const double length = requested_normal.norm2();
    if (!(length > 1e-12) || !std::isfinite(length)) return false;
    for (int axis = 0; axis < 3; ++axis)
        normal[axis] = requested_normal[axis] / length;
    Vec<3> reference;
    reference.zero();
    reference[std::fabs(normal[2]) < 0.9 ? 2 : 0] = 1.0;
    tangent1 = cross3(reference, normal);
    const double tangent_length = tangent1.norm2();
    if (!(tangent_length > 1e-12)) return false;
    for (int axis = 0; axis < 3; ++axis)
        tangent1[axis] /= tangent_length;
    tangent2 = cross3(normal, tangent1);
    return true;
}

inline void project_friction_cone(Vec<3>& force, const Vec<3>& normal,
                                  const Vec<3>& tangent1,
                                  const Vec<3>& tangent2, double friction,
                                  double maximum_normal_force) {
    const double normal_input = dot3(normal, force);
    const double tangent_input1 = dot3(tangent1, force);
    const double tangent_input2 = dot3(tangent2, force);
    const double tangent_radius = std::sqrt(
        tangent_input1 * tangent_input1 + tangent_input2 * tangent_input2);

    double candidates[4];
    int candidate_count = 0;
    candidates[candidate_count++] = std::max(
        0.0, std::min(maximum_normal_force, normal_input));
    const double boundary = friction > 0.0
        ? tangent_radius / friction
        : maximum_normal_force;
    candidates[candidate_count++] = std::max(
        0.0, std::min(maximum_normal_force, boundary));
    if (friction > 0.0) {
        const double cone_candidate =
            (normal_input + friction * tangent_radius) /
            (1.0 + friction * friction);
        candidates[candidate_count++] = std::max(
            0.0, std::min(std::min(maximum_normal_force, boundary),
                          cone_candidate));
    }
    candidates[candidate_count++] = maximum_normal_force;

    double best_normal = 0.0;
    double best_cost = 1e300;
    for (int candidate = 0; candidate < candidate_count; ++candidate) {
        const double projected_normal = candidates[candidate];
        const double maximum_tangent = friction * projected_normal;
        const double projected_radius =
            std::min(tangent_radius, maximum_tangent);
        const double normal_error = projected_normal - normal_input;
        const double tangent_error = projected_radius - tangent_radius;
        const double cost = normal_error * normal_error +
                            tangent_error * tangent_error;
        if (cost < best_cost) {
            best_cost = cost;
            best_normal = projected_normal;
        }
    }

    double projected_tangent1 = 0.0;
    double projected_tangent2 = 0.0;
    if (tangent_radius > 1e-12) {
        const double scale =
            std::min(1.0, friction * best_normal / tangent_radius);
        projected_tangent1 = scale * tangent_input1;
        projected_tangent2 = scale * tangent_input2;
    }
    for (int axis = 0; axis < 3; ++axis) {
        force[axis] = best_normal * normal[axis] +
                      projected_tangent1 * tangent1[axis] +
                      projected_tangent2 * tangent2[axis];
    }
}

inline Vec<3> orientation_error_world(const Vec<4>& desired_raw,
                                      const Vec<4>& current_raw) {
    Vec<4> desired;
    Vec<4> current;
    Vec<3> error;
    error.zero();
    if (!normalize_quaternion(desired_raw, desired) ||
        !normalize_quaternion(current_raw, current)) {
        return error;
    }
    const double w = desired[0] * current[0] +
                     desired[1] * current[1] +
                     desired[2] * current[2] +
                     desired[3] * current[3];
    error[0] = desired[1] * current[0] -
               desired[0] * current[1] -
               desired[2] * current[3] +
               desired[3] * current[2];
    error[1] = desired[2] * current[0] -
               desired[0] * current[2] -
               desired[3] * current[1] +
               desired[1] * current[3];
    error[2] = desired[3] * current[0] -
               desired[0] * current[3] -
               desired[1] * current[2] +
               desired[2] * current[1];
    const double sign = w < 0.0 ? -2.0 : 2.0;
    for (int axis = 0; axis < 3; ++axis) error[axis] *= sign;
    return error;
}

inline double frobenius_condition_number_3x3(const Mat<3, 3>& matrix) {
    double matrix_norm_sq = 0.0;
    for (int row = 0; row < 3; ++row) {
        for (int column = 0; column < 3; ++column)
            matrix_norm_sq += matrix(row, column) * matrix(row, column);
    }
    if (!std::isfinite(matrix_norm_sq) || !(matrix_norm_sq > 0.0))
        return 1e300;
    const double determinant =
        matrix(0, 0) * (matrix(1, 1) * matrix(2, 2) -
                        matrix(1, 2) * matrix(2, 1)) -
        matrix(0, 1) * (matrix(1, 0) * matrix(2, 2) -
                        matrix(1, 2) * matrix(2, 0)) +
        matrix(0, 2) * (matrix(1, 0) * matrix(2, 1) -
                        matrix(1, 1) * matrix(2, 0));
    const double determinant_scale =
        matrix_norm_sq * std::sqrt(matrix_norm_sq);
    if (!std::isfinite(determinant) ||
        std::fabs(determinant) <= 1e-12 * determinant_scale) {
        return 1e300;
    }

    double cofactor_norm_sq = 0.0;
    const double cofactors[3][3] = {
        {matrix(1, 1) * matrix(2, 2) - matrix(1, 2) * matrix(2, 1),
         matrix(1, 2) * matrix(2, 0) - matrix(1, 0) * matrix(2, 2),
         matrix(1, 0) * matrix(2, 1) - matrix(1, 1) * matrix(2, 0)},
        {matrix(0, 2) * matrix(2, 1) - matrix(0, 1) * matrix(2, 2),
         matrix(0, 0) * matrix(2, 2) - matrix(0, 2) * matrix(2, 0),
         matrix(0, 1) * matrix(2, 0) - matrix(0, 0) * matrix(2, 1)},
        {matrix(0, 1) * matrix(1, 2) - matrix(0, 2) * matrix(1, 1),
         matrix(0, 2) * matrix(1, 0) - matrix(0, 0) * matrix(1, 2),
         matrix(0, 0) * matrix(1, 1) - matrix(0, 1) * matrix(1, 0)}};
    for (int row = 0; row < 3; ++row) {
        for (int column = 0; column < 3; ++column) {
            cofactor_norm_sq +=
                cofactors[row][column] * cofactors[row][column];
        }
    }
    const double condition =
        std::sqrt(matrix_norm_sq * cofactor_norm_sq) /
        std::fabs(determinant);
    return std::isfinite(condition) ? condition : 1e300;
}

class ConvexWholeBodyController {
public:
    explicit ConvexWholeBodyController(
        const ConvexWBCParameters& parameters = {})
        : parameters_(parameters) {}

    const ConvexWBCParameters& parameters() const { return parameters_; }

    Status compute(const ContactPlanSample& reference,
                   const WholeBodyState& state,
                   ConvexWBCCommand& command) const {
        if (!(parameters_.force_tracking_weight > 0.0) ||
            !(parameters_.friction >= 0.0) ||
            !(parameters_.maximum_normal_force > 0.0) ||
            parameters_.maximum_iterations <= 0 ||
            !(parameters_.convergence_tolerance >= 0.0)) {
            return Status::BAD_ARGUMENT;
        }
        Vec<4> current_quaternion;
        if (!normalize_quaternion(state.base.orientation_body_to_world,
                                  current_quaternion)) {
            return Status::BAD_ARGUMENT;
        }
        Mat<3, 3> rotation;
        unit_rotation_matrix(current_quaternion, rotation);

        Mat<6, 12> wrench_map;
        wrench_map.zero();
        Vec<12> reference_forces;
        reference_forces.zero();
        Vec<6> desired_wrench;
        desired_wrench.zero();
        Vec<3> normals[kNumFeet];
        Vec<3> tangents1[kNumFeet];
        Vec<3> tangents2[kNumFeet];
        bool active[kNumFeet];
        for (int foot = 0; foot < kNumFeet; ++foot) {
            active[foot] = reference.feet[foot].planned_contact;
            if (!terrain_basis_from_normal(
                    reference.feet[foot].terrain_normal_world,
                    normals[foot], tangents1[foot], tangents2[foot])) {
                return Status::BAD_ARGUMENT;
            }
            Vec<3> lever;
            for (int axis = 0; axis < 3; ++axis) {
                lever[axis] = state.foot_positions_world[foot][axis] -
                              state.base.position_world[axis];
                wrench_map(axis, 3 * foot + axis) = 1.0;
                reference_forces[3 * foot + axis] = active[foot]
                    ? reference.feet[foot].force_world[axis]
                    : 0.0;
                desired_wrench[axis] += reference_forces[3 * foot + axis];
            }
            Mat<3, 3> lever_skew;
            skew3(lever, lever_skew);
            for (int row = 0; row < 3; ++row) {
                for (int col = 0; col < 3; ++col) {
                    wrench_map(3 + row, 3 * foot + col) =
                        lever_skew(row, col);
                }
            }
            Vec<3> planned_lever;
            for (int axis = 0; axis < 3; ++axis) {
                planned_lever[axis] =
                    reference.feet[foot].position_world[axis] -
                    reference.base.position_world[axis];
            }
            const Vec<3> planned_force = reference.feet[foot].force_world;
            const Vec<3> planned_torque =
                cross3(planned_lever, planned_force);
            for (int axis = 0; axis < 3; ++axis)
                desired_wrench[3 + axis] +=
                    active[foot] ? planned_torque[axis] : 0.0;
        }

        const Vec<3> orientation_error = orientation_error_world(
            reference.base.orientation_body_to_world,
            state.base.orientation_body_to_world);
        Vec<3> angular_velocity_error_body;
        for (int axis = 0; axis < 3; ++axis) {
            desired_wrench[axis] +=
                parameters_.base_position_kp[axis] *
                    (reference.base.position_world[axis] -
                     state.base.position_world[axis]) +
                parameters_.base_velocity_kd[axis] *
                    (reference.base.linear_velocity_world[axis] -
                     state.base.linear_velocity_world[axis]);
            angular_velocity_error_body[axis] =
                reference.base.angular_velocity_body[axis] -
                state.base.angular_velocity_body[axis];
        }
        Vec<3> angular_velocity_error_world;
        rotation.mul_vec(angular_velocity_error_body,
                         angular_velocity_error_world);
        for (int axis = 0; axis < 3; ++axis) {
            desired_wrench[3 + axis] +=
                parameters_.base_orientation_kp[axis] *
                    orientation_error[axis] +
                parameters_.base_angular_velocity_kd[axis] *
                    angular_velocity_error_world[axis];
        }

        Vec<12> forces = reference_forces;
        for (int foot = 0; foot < kNumFeet; ++foot) {
            Vec<3> force;
            for (int axis = 0; axis < 3; ++axis)
                force[axis] = forces[3 * foot + axis];
            if (active[foot]) {
                project_friction_cone(force, normals[foot], tangents1[foot],
                                      tangents2[foot], parameters_.friction,
                                      parameters_.maximum_normal_force);
            } else {
                force.zero();
            }
            for (int axis = 0; axis < 3; ++axis)
                forces[3 * foot + axis] = force[axis];
        }

        double lipschitz_bound = parameters_.force_tracking_weight;
        for (int row = 0; row < 6; ++row) {
            double row_norm_sq = 0.0;
            for (int col = 0; col < 12; ++col)
                row_norm_sq += wrench_map(row, col) *
                               wrench_map(row, col);
            lipschitz_bound += parameters_.wrench_weights[row] * row_norm_sq;
        }
        const double step = 1.0 / lipschitz_bound;
        command.force_iterations = parameters_.maximum_iterations;
        for (int iteration = 0; iteration < parameters_.maximum_iterations;
             ++iteration) {
            Vec<6> residual;
            for (int row = 0; row < 6; ++row) {
                residual[row] = -desired_wrench[row];
                for (int col = 0; col < 12; ++col)
                    residual[row] += wrench_map(row, col) * forces[col];
            }
            Vec<12> gradient;
            for (int col = 0; col < 12; ++col) {
                gradient[col] = parameters_.force_tracking_weight *
                    (forces[col] - reference_forces[col]);
                for (int row = 0; row < 6; ++row) {
                    gradient[col] += wrench_map(row, col) *
                        parameters_.wrench_weights[row] * residual[row];
                }
            }
            const Vec<12> previous_forces = forces;
            for (int col = 0; col < 12; ++col) {
                forces[col] -= step * gradient[col];
            }
            for (int foot = 0; foot < kNumFeet; ++foot) {
                Vec<3> force;
                for (int axis = 0; axis < 3; ++axis)
                    force[axis] = forces[3 * foot + axis];
                if (active[foot]) {
                    project_friction_cone(
                        force, normals[foot], tangents1[foot],
                        tangents2[foot], parameters_.friction,
                        parameters_.maximum_normal_force);
                } else {
                    force.zero();
                }
                for (int axis = 0; axis < 3; ++axis)
                    forces[3 * foot + axis] = force[axis];
            }
            double maximum_change = 0.0;
            for (int col = 0; col < 12; ++col) {
                maximum_change = std::max(
                    maximum_change,
                    std::fabs(forces[col] - previous_forces[col]));
            }
            if (maximum_change <= parameters_.convergence_tolerance) {
                command.force_iterations = iteration + 1;
                break;
            }
        }

        Vec<6> final_residual;
        command.wrench_residual = 0.0;
        for (int row = 0; row < 6; ++row) {
            final_residual[row] = -desired_wrench[row];
            for (int col = 0; col < 12; ++col)
                final_residual[row] += wrench_map(row, col) * forces[col];
            command.wrench_residual_components[row] = final_residual[row];
            command.wrench_residual = std::max(
                command.wrench_residual, std::fabs(final_residual[row]));
        }

        for (int foot = 0; foot < kNumFeet; ++foot) {
            for (int axis = 0; axis < 3; ++axis)
                command.contact_forces_world[foot][axis] =
                    forces[3 * foot + axis];
            const double blend_denominator =
                std::max(parameters_.contact_blend_force, 1e-12);
            command.contact_blend[foot] = active[foot]
                ? std::max(0.0, std::min(
                      1.0, reference.feet[foot].normal_force /
                               blend_denominator))
                : 0.0;

            Vec<3> stance_force_body;
            rotation.mul_vec_transpose(command.contact_forces_world[foot],
                                       stance_force_body);
            Vec<3> swing_force_world;
            for (int axis = 0; axis < 3; ++axis) {
                swing_force_world[axis] =
                    parameters_.swing_position_kp[axis] *
                        (reference.feet[foot].position_world[axis] -
                         state.foot_positions_world[foot][axis]) +
                    parameters_.swing_velocity_kd[axis] *
                        (reference.feet[foot].velocity_world[axis] -
                         state.foot_velocities_world[foot][axis]);
                command.swing_feedback_forces_world[foot][axis] =
                    swing_force_world[axis];
            }
            Vec<3> swing_force_body;
            rotation.mul_vec_transpose(swing_force_world, swing_force_body);
            Vec<3> stance_jacobian_torque;
            Vec<3> swing_feedback_torque;
            state.foot_jacobians_body[foot].mul_vec_transpose(
                stance_force_body, stance_jacobian_torque);
            state.foot_jacobians_body[foot].mul_vec_transpose(
                swing_force_body, swing_feedback_torque);
            for (int joint = 0; joint < 3; ++joint) {
                const double stance_torque =
                    state.joint_bias_torques[foot][joint] -
                    stance_jacobian_torque[joint];
                const double swing_torque =
                    state.joint_bias_torques[foot][joint] +
                    swing_feedback_torque[joint] +
                    state.swing_feedforward_joint_torques[foot][joint];
                command.joint_torques[foot][joint] =
                    command.contact_blend[foot] * stance_torque +
                    (1.0 - command.contact_blend[foot]) * swing_torque;
            }
        }
        return Status::SUCCESS;
    }

private:
    ConvexWBCParameters parameters_;
};

}  // namespace quadruped_cito
