#include <algorithm>
#include <cmath>
#include <cstdio>

#include "examples/quadruped_cito/quadruped_cito_contact_feedback.hpp"
#include "examples/quadruped_cito/quadruped_cito_plan_io.hpp"
#include "examples/quadruped_cito/quadruped_cito_replay_metrics.hpp"
#include "examples/quadruped_cito/quadruped_cito_swing_reference.hpp"
#include "examples/quadruped_cito/quadruped_cito_wbc.hpp"

namespace {

using namespace quadruped_cito;

void set_identity_quaternion(Vec<4>& quaternion) {
    quaternion.zero();
    quaternion[0] = 1.0;
}

void set_identity_matrix(Mat<3, 3>& matrix) {
    matrix.zero();
    for (int axis = 0; axis < 3; ++axis) matrix(axis, axis) = 1.0;
}

int test_plan_sampling() {
    ContactPlan<2> plan;
    for (int stage = 0; stage < 2; ++stage) {
        plan.stages[stage].time = 0.1 * stage;
        plan.stages[stage].duration = 0.1;
        plan.stages[stage].base.position_world.zero();
        plan.stages[stage].base.position_world[0] = stage;
        set_identity_quaternion(
            plan.stages[stage].base.orientation_body_to_world);
        plan.stages[stage].base.linear_velocity_world.zero();
        plan.stages[stage].base.angular_velocity_body.zero();
        for (int foot = 0; foot < kNumFeet; ++foot) {
            FootPlanSample& sample = plan.stages[stage].feet[foot];
            sample.position_world.zero();
            sample.position_world[0] = stage + 0.1 * foot;
            sample.velocity_world.zero();
            sample.velocity_world[0] = 0.2 + foot;
            sample.force_world.zero();
            sample.force_world[2] = 10.0 + foot;
            sample.terrain_normal_world.zero();
            sample.terrain_normal_world[2] = 1.0;
            sample.normal_force = sample.force_world[2];
            sample.planned_contact = true;
        }
    }
    plan.terminal.time = 0.2;
    plan.terminal.base = plan.stages[1].base;
    plan.terminal.base.position_world[0] = 2.0;
    for (int foot = 0; foot < kNumFeet; ++foot) {
        plan.terminal.foot_positions_world[foot] =
            plan.stages[1].feet[foot].position_world;
        plan.terminal.foot_positions_world[foot][0] = 2.0 + 0.1 * foot;
    }
    const char* fingerprint_path =
        "quadruped_cito_plan_fingerprint_test.plan";
    const std::uint64_t fingerprint = contact_plan_fingerprint(plan);
    ContactPlan<2> round_tripped_plan;
    if (write_contact_plan(plan, fingerprint_path) != Status::SUCCESS ||
        read_contact_plan(fingerprint_path, round_tripped_plan) !=
            Status::SUCCESS ||
        contact_plan_fingerprint(round_tripped_plan) != fingerprint) {
        std::remove(fingerprint_path);
        std::printf("contact-plan fingerprint round trip failed\n");
        return 1;
    }
    std::remove(fingerprint_path);
    round_tripped_plan.stages[0].time += 1e-3;
    if (contact_plan_fingerprint(round_tripped_plan) == fingerprint) {
        std::printf("contact-plan fingerprint missed a plan change\n");
        return 1;
    }

    ContactPlanSample sample;
    if (sample_contact_plan(plan, 0.05, sample) !=
            PlanSampleStatus::ACTIVE ||
        sample.stage_index != 0 ||
        std::fabs(sample.stage_phase - 0.5) > 1e-12 ||
        std::fabs(sample.base.position_world[0] - 0.5) > 1e-12 ||
        std::fabs(sample.feet[2].position_world[0] - 0.7) > 1e-12 ||
        std::fabs(sample.feet[2].force_world[2] - 12.0) > 1e-12) {
        std::printf("active contact-plan sampling failed\n");
        return 1;
    }
    if (std::fabs(sample.base.orientation_body_to_world.norm2() - 1.0) >
        1e-12) {
        std::printf("sampled quaternion is not normalized\n");
        return 1;
    }
    if (sample_contact_plan(plan, -0.1, sample) !=
            PlanSampleStatus::BEFORE_START ||
        sample.stage_index != 0 || sample.stage_phase != 0.0) {
        std::printf("pre-start contact-plan sampling failed\n");
        return 1;
    }
    if (sample_contact_plan(plan, 0.2, sample) !=
            PlanSampleStatus::EXPIRED ||
        sample.stage_index != 2 ||
        sample.feet[0].force_world.norm2() != 0.0 ||
        sample.feet[0].planned_contact) {
        std::printf("expired contact-plan sampling failed\n");
        return 1;
    }
    return 0;
}

int test_repeated_swing_metrics() {
    FootReplayMetrics metrics;
    Vec<3> planned_position;
    Vec<3> measured_position;
    planned_position.zero();
    measured_position.zero();
    metrics.initialize(true, true);

    metrics.observe(0.1, 0.09, false, true, planned_position,
                    measured_position, 0.0, 0.02);
    metrics.observe(0.11, 0.10, false, false, planned_position,
                    measured_position, 0.0, 0.02);
    metrics.observe(0.115, 0.105, false, true, planned_position,
                    measured_position, 0.001, 0.02);
    measured_position[2] = 0.03;
    metrics.observe(0.12, 0.11, false, false, planned_position,
                    measured_position, 0.03, 0.02);
    planned_position[0] = 0.08;
    measured_position[0] = 0.079;
    measured_position[2] = 0.0;
    metrics.observe(0.20, 0.19, true, true, planned_position,
                    measured_position, 0.0, 0.02);
    measured_position[0] = 0.081;
    metrics.observe(0.22, 0.21, true, true, planned_position,
                    measured_position, 0.0, 0.02);

    metrics.observe(0.30, 0.29, false, true, planned_position,
                    measured_position, 0.0, 0.02);
    measured_position[2] = 0.04;
    metrics.observe(0.32, 0.31, false, false, planned_position,
                    measured_position, 0.04, 0.02);
    planned_position[0] = 0.16;
    measured_position[0] = 0.158;
    measured_position[2] = 0.0;
    metrics.observe(0.40, 0.39, false, true, planned_position,
                    measured_position, 0.0, 0.02);
    metrics.observe(0.41, 0.40, true, true, planned_position,
                    measured_position, 0.0, 0.02);

    if (metrics.events.size() != 2) {
        std::printf("repeated swing events were not recorded independently\n");
        return 1;
    }
    const SwingReplayEvent& first = metrics.events[0];
    const SwingReplayEvent& second = metrics.events[1];
    if (!first.measured_unload() || !second.measured_unload() ||
        std::fabs(first.planned_liftoff_time - 0.1) > 1e-12 ||
        std::fabs(first.measured_liftoff_time - 0.10) > 1e-12 ||
        std::fabs(first.planned_touchdown_time - 0.20) > 1e-12 ||
        std::fabs(first.measured_touchdown_time - 0.19) > 1e-12 ||
        std::fabs(second.planned_liftoff_time - 0.30) > 1e-12 ||
        std::fabs(second.measured_liftoff_time - 0.31) > 1e-12 ||
        std::fabs(second.planned_touchdown_time - 0.41) > 1e-12 ||
        std::fabs(second.measured_touchdown_time - 0.39) > 1e-12 ||
        std::fabs(first.maximum_clearance - 0.03) > 1e-12 ||
        std::fabs(second.maximum_clearance - 0.04) > 1e-12 ||
        std::fabs(first.landing_error() - 0.001) > 1e-12 ||
        std::fabs(second.landing_error() - 0.002) > 1e-12 ||
        std::fabs(first.touchdown_error() - 0.01) > 1e-12 ||
        std::fabs(second.touchdown_error() - 0.02) > 1e-12 ||
        std::fabs(first.signed_touchdown_error() + 0.01) > 1e-12 ||
        std::fabs(second.signed_touchdown_error() + 0.02) > 1e-12 ||
        std::fabs(first.maximum_post_touchdown_slip - 0.002) > 1e-12) {
        std::printf("repeated swing event metrics are incorrect\n");
        return 1;
    }
    return 0;
}

int test_swing_reference_shaping() {
    ContactPlan<5> plan;
    for (int stage = 0; stage < 5; ++stage) {
        plan.stages[stage].time = 0.1 * stage;
        plan.stages[stage].duration = 0.1;
        plan.stages[stage].base.position_world.zero();
        set_identity_quaternion(
            plan.stages[stage].base.orientation_body_to_world);
        plan.stages[stage].base.linear_velocity_world.zero();
        plan.stages[stage].base.angular_velocity_body.zero();
        for (int foot = 0; foot < kNumFeet; ++foot) {
            FootPlanSample& foot_sample = plan.stages[stage].feet[foot];
            foot_sample.position_world.zero();
            foot_sample.velocity_world.set_constant(10.0);
            foot_sample.force_world.zero();
            foot_sample.terrain_normal_world.zero();
            foot_sample.terrain_normal_world[2] = 1.0;
            foot_sample.planned_contact = true;
        }
    }
    plan.stages[1].feet[0].planned_contact = false;
    plan.stages[2].feet[0].planned_contact = false;
    plan.stages[3].feet[0].planned_contact = false;
    plan.stages[2].feet[0].position_world[0] = 0.09;
    plan.stages[2].feet[0].position_world[2] = 0.09;
    plan.stages[3].feet[0].position_world[0] = 0.10;
    plan.stages[3].feet[0].position_world[2] = 0.01;
    plan.stages[4].feet[0].position_world[0] = 0.10;
    plan.terminal.time = 0.5;
    plan.terminal.base = plan.stages[4].base;
    for (int foot = 0; foot < kNumFeet; ++foot) {
        plan.terminal.foot_positions_world[foot] =
            plan.stages[4].feet[foot].position_world;
    }

    ContactPlanSample sample;
    if (sample_contact_plan(plan, 0.10, sample) !=
            PlanSampleStatus::ACTIVE) {
        return 1;
    }
    shape_swing_references(plan, sample);
    if (sample.feet[0].position_world.norm2() > 1e-12 ||
        sample.feet[0].velocity_world.norm2() > 1e-12) {
        std::printf("swing reference is discontinuous at liftoff\n");
        return 1;
    }

    sample_contact_plan(plan, 0.25, sample);
    shape_swing_references(plan, sample);
    if (std::fabs(sample.feet[0].position_world[0] - 0.05) > 1e-12 ||
        std::fabs(sample.feet[0].position_world[2] - 0.09) > 1e-12 ||
        std::fabs(sample.feet[0].velocity_world[0] - 0.5) > 1e-12 ||
        std::fabs(sample.feet[0].velocity_world[2]) > 1e-12 ||
        std::fabs(sample.feet[1].velocity_world[0] - 10.0) > 1e-12) {
        std::printf("swing reference apex or stance isolation failed\n");
        return 1;
    }

    sample_contact_plan(plan, 0.40, sample);
    const FootPlanSample touchdown = sample.feet[0];
    shape_swing_references(plan, sample);
    double touchdown_change = 0.0;
    for (int axis = 0; axis < 3; ++axis) {
        touchdown_change = std::max(
            touchdown_change,
            std::fabs(sample.feet[0].position_world[axis] -
                      touchdown.position_world[axis]));
        touchdown_change = std::max(
            touchdown_change,
            std::fabs(sample.feet[0].velocity_world[axis] -
                      touchdown.velocity_world[axis]));
    }
    if (!sample.feet[0].planned_contact ||
        touchdown_change > 1e-12) {
        std::printf("swing reference changed the touchdown stance sample\n");
        return 1;
    }
    return 0;
}

struct CrestTerrain {
    double height(double x, double) const {
        return 0.04 * std::sin(3.14159265358979323846 * x / 0.10);
    }

    bool sample(const Vec<3>& position, TerrainSample& sample) const {
        const double phase = 3.14159265358979323846 * position[0] / 0.10;
        const double scale = 3.14159265358979323846 / 0.10;
        return sample_height_field(
            position, height(position[0], position[1]),
            0.04 * scale * std::cos(phase), 0.0,
            -0.04 * scale * scale * std::sin(phase), 0.0, 0.0, sample);
    }
};

int test_terrain_relative_swing_reference() {
    ContactPlan<5> plan;
    for (int stage = 0; stage < 5; ++stage) {
        plan.stages[stage].time = 0.1 * stage;
        plan.stages[stage].duration = 0.1;
        for (int foot = 0; foot < kNumFeet; ++foot) {
            FootPlanSample& foot_sample = plan.stages[stage].feet[foot];
            foot_sample.position_world.zero();
            foot_sample.velocity_world.zero();
            foot_sample.planned_contact = true;
        }
    }
    plan.stages[1].feet[0].planned_contact = false;
    plan.stages[2].feet[0].planned_contact = false;
    plan.stages[3].feet[0].planned_contact = false;
    plan.stages[2].feet[0].position_world[0] = 0.09;
    plan.stages[2].feet[0].position_world[2] = 0.09;
    plan.stages[3].feet[0].position_world[0] = 0.10;
    plan.stages[4].feet[0].position_world[0] = 0.10;
    plan.terminal.time = 0.5;

    ContactPlanSample sample;
    if (sample_contact_plan(plan, 0.25, sample) != PlanSampleStatus::ACTIVE)
        return 1;
    const CrestTerrain terrain;
    shape_swing_references(plan, terrain, sample);
    const double planned_apex_clearance =
        plan.stages[2].feet[0].position_world[2] -
        terrain.height(plan.stages[2].feet[0].position_world[0], 0.0);
    const double shaped_clearance = sample.feet[0].position_world[2] -
        terrain.height(sample.feet[0].position_world[0], 0.0);
    if (std::fabs(sample.feet[0].position_world[0] - 0.05) > 1e-12 ||
        std::fabs(shaped_clearance - planned_apex_clearance) > 1e-12) {
        std::printf("terrain-relative swing clearance was not preserved\n");
        return 1;
    }
    return 0;
}

int test_contact_execution_feedback() {
    ContactExecutionFeedbackParameters parameters;
    EarlyContactFeedback feedback;
    if (feedback.update(false, true, false, parameters) ||
        feedback.update(false, true, true, parameters) ||
        feedback.update(false, false, true, parameters) ||
        feedback.update(false, true, true, parameters) ||
        feedback.update(false, true, true, parameters) ||
        !feedback.update(false, true, true, parameters)) {
        std::printf("early-contact persistence filter failed\n");
        return 1;
    }
    for (int tick = 0;
         tick + 1 < parameters.early_contact_loss_grace_ticks; ++tick) {
        if (!feedback.update(false, false, true, parameters)) {
            std::printf("early-contact loss grace failed\n");
            return 1;
        }
    }
    if (feedback.update(false, false, true, parameters) ||
        feedback.update(true, true, true, parameters)) {
        std::printf("early-contact feedback did not reset\n");
        return 1;
    }

    FootPlanSample reference;
    reference.position_world.zero();
    reference.velocity_world.set_constant(1.0);
    reference.force_world.set_constant(2.0);
    reference.normal_force = 2.0;
    reference.planned_contact = false;
    Vec<3> measured_position;
    measured_position.set_constant(0.1);
    apply_early_contact_support(measured_position, parameters, reference);
    double position_error = 0.0;
    for (int axis = 0; axis < 3; ++axis) {
        position_error = std::max(
            position_error,
            std::fabs(reference.position_world[axis] -
                      measured_position[axis]));
    }
    if (!reference.planned_contact ||
        position_error > 1e-12 ||
        reference.velocity_world.norm2() > 1e-12 ||
        reference.force_world.norm2() > 1e-12 ||
        reference.normal_force != parameters.contact_blend_force) {
        std::printf("early-contact support reference failed\n");
        return 1;
    }

    reference.position_world[2] = 0.02;
    reference.velocity_world.zero();
    reference.force_world.set_constant(2.0);
    reference.normal_force = 2.0;
    reference.planned_contact = true;
    apply_late_touchdown_search(1000, 0.002, parameters, reference);
    if (reference.planned_contact ||
        std::fabs(reference.position_world[2]) > 1e-12 ||
        std::fabs(reference.velocity_world[2] +
                  parameters.late_touchdown_search_speed) > 1e-12 ||
        reference.force_world.norm2() > 1e-12 ||
        reference.normal_force != 0.0) {
        std::printf("late-touchdown bounded search failed\n");
        return 1;
    }

    ConsecutiveGateConfirmation confirmation(3);
    if (confirmation.update(true) || confirmation.update(false) ||
        confirmation.update(true) || confirmation.update(true) ||
        !confirmation.update(true)) {
        std::printf("consecutive gate confirmation failed\n");
        return 1;
    }
    return 0;
}

int test_stance_contact_confirmation() {
    constexpr int kLossGraceTicks = 3;
    StanceContactConfirmation confirmation;

    // Segment-start planned support keeps the historical startup grace, but
    // geometry alone does not make it eligible for later reacquisition.
    confirmation.initialize(true, false);
    if (!confirmation.update(true, true, false, kLossGraceTicks) ||
        confirmation.established()) {
        std::printf("stance confirmation lost startup grace\n");
        return 1;
    }
    for (int tick = 0; tick < kLossGraceTicks; ++tick)
        confirmation.update(true, false, false, kLossGraceTicks);
    if (confirmation.update(true, true, false, kLossGraceTicks) ||
        confirmation.established()) {
        std::printf("never-established stance reacquired from geometry\n");
        return 1;
    }
    if (!confirmation.update(true, true, true, kLossGraceTicks) ||
        !confirmation.established()) {
        std::printf("load-bearing stance was not established\n");
        return 1;
    }

    // Low-force geometry is sufficient to preserve an established stance.
    for (int tick = 0; tick < 2 * kLossGraceTicks; ++tick) {
        if (!confirmation.update(true, true, false, kLossGraceTicks)) {
            std::printf("established geometry contact was dropped\n");
            return 1;
        }
    }

    // True geometry loss observes the full grace interval, then geometry
    // return immediately reacquires because this stance previously bore load.
    for (int tick = 0; tick + 1 < kLossGraceTicks; ++tick) {
        if (!confirmation.update(true, false, false, kLossGraceTicks)) {
            std::printf("stance geometry-loss grace was shortened\n");
            return 1;
        }
    }
    if (confirmation.update(true, false, false, kLossGraceTicks) ||
        !confirmation.established() ||
        !confirmation.update(true, true, false, kLossGraceTicks)) {
        std::printf("established stance failed geometry reacquisition\n");
        return 1;
    }

    // A real liftoff clears recovery history, so the next scheduled touchdown
    // still requires load instead of being accepted from geometry alone.
    if (confirmation.update(false, true, true, kLossGraceTicks) ||
        confirmation.established() ||
        confirmation.update(true, true, false, kLossGraceTicks)) {
        std::printf("stance liftoff did not clear recovery history\n");
        return 1;
    }
    if (!confirmation.update(true, true, true, kLossGraceTicks) ||
        !confirmation.established()) {
        std::printf("new scheduled touchdown was not load-confirmed\n");
        return 1;
    }
    return 0;
}

int test_contact_torque_blend_handoff() {
    ContactTorqueBlendHandoff handoff;
    FootPlanSample reference;
    reference.normal_force = 20.0;
    reference.force_world[0] = 2.0;
    reference.force_world[1] = -3.0;
    reference.force_world[2] = 20.0;
    reference.planned_contact = true;
    const Vec<3> original_force = reference.force_world;

    handoff.update(true, false, 1.0, reference);
    if (reference.normal_force != 20.0) {
        std::printf("inactive torque-blend handoff changed reference\n");
        return 1;
    }
    handoff.update(true, true, 1.0, reference);
    if (reference.normal_force != 1.0) {
        std::printf("torque-blend handoff did not arm at 1 N\n");
        return 1;
    }
    reference.normal_force = 20.0;
    handoff.update(true, false, 10.0, reference);
    if (reference.normal_force != 10.0) {
        std::printf("torque-blend handoff did not limit at 10 N\n");
        return 1;
    }
    reference.normal_force = 20.0;
    handoff.update(true, false, 20.0, reference);
    if (reference.normal_force != 20.0) {
        std::printf("torque-blend handoff changed disarm tick\n");
        return 1;
    }
    reference.normal_force = 20.0;
    handoff.update(true, false, 1.0, reference);
    if (reference.normal_force != 20.0) {
        std::printf("torque-blend handoff implicitly rearmed\n");
        return 1;
    }

    handoff.update(true, true, 1.0, reference);
    reference.normal_force = 20.0;
    handoff.update(false, false, 1.0, reference);
    reference.normal_force = 20.0;
    handoff.update(true, false, 1.0, reference);
    if (reference.normal_force != 20.0) {
        std::printf("planned liftoff did not reset torque-blend handoff\n");
        return 1;
    }
    for (int axis = 0; axis < 3; ++axis) {
        if (reference.force_world[axis] != original_force[axis]) {
            std::printf("torque-blend handoff changed force reference\n");
            return 1;
        }
    }
    if (!reference.planned_contact) {
        std::printf("torque-blend handoff changed contact bit\n");
        return 1;
    }
    return 0;
}

void initialize_wbc_case(ContactPlanSample& reference,
                         WholeBodyState& state, double mass) {
    reference.time = 0.0;
    reference.stage_index = 0;
    reference.stage_phase = 0.0;
    reference.base.position_world.zero();
    set_identity_quaternion(reference.base.orientation_body_to_world);
    reference.base.linear_velocity_world.zero();
    reference.base.angular_velocity_body.zero();
    state.base = reference.base;
    const double foot_positions[kNumFeet][3] = {
        {0.3, 0.2, 0.0}, {0.3, -0.2, 0.0},
        {-0.3, 0.2, 0.0}, {-0.3, -0.2, 0.0}};
    const double normal_force = mass * 9.81 / kNumFeet;
    for (int foot = 0; foot < kNumFeet; ++foot) {
        for (int axis = 0; axis < 3; ++axis) {
            state.foot_positions_world[foot][axis] =
                foot_positions[foot][axis];
            reference.feet[foot].position_world[axis] =
                foot_positions[foot][axis];
        }
        state.foot_velocities_world[foot].zero();
        set_identity_matrix(state.foot_jacobians_body[foot]);
        state.joint_bias_torques[foot].zero();
        state.swing_feedforward_joint_torques[foot].zero();
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

bool friction_feasible(const Vec<3>& force, double friction,
                       double maximum_normal_force) {
    const double normal = force[2];
    const double tangent = std::sqrt(force[0] * force[0] +
                                     force[1] * force[1]);
    return normal >= -1e-10 && normal <= maximum_normal_force + 1e-10 &&
           tangent <= friction * normal + 1e-8;
}

int test_convex_wbc() {
    constexpr double mass = 15.0;
    ContactPlanSample reference;
    WholeBodyState state;
    initialize_wbc_case(reference, state, mass);
    ConvexWBCParameters parameters;
    parameters.maximum_iterations = 300;
    parameters.convergence_tolerance = 1e-12;
    ConvexWholeBodyController controller(parameters);
    ConvexWBCCommand command;
    if (controller.compute(reference, state, command) !=
        nmpc::Status::SUCCESS) {
        std::printf("standing convex WBC returned failure\n");
        return 1;
    }
    Vec<3> force_sum;
    force_sum.zero();
    for (int foot = 0; foot < kNumFeet; ++foot) {
        for (int axis = 0; axis < 3; ++axis)
            force_sum[axis] += command.contact_forces_world[foot][axis];
        if (!friction_feasible(command.contact_forces_world[foot],
                               parameters.friction,
                               parameters.maximum_normal_force) ||
            std::fabs(command.joint_torques[foot][2] +
                      command.contact_forces_world[foot][2]) > 1e-9) {
            std::printf("standing WBC force or torque mapping failed\n");
            return 1;
        }
    }
    if (std::fabs(force_sum[2] - mass * 9.81) > 1e-8 ||
        std::fabs(force_sum[0]) > 1e-8 || std::fabs(force_sum[1]) > 1e-8) {
        std::printf("standing WBC wrench mismatch\n");
        return 1;
    }

    state.base.position_world[0] = -0.05;
    if (controller.compute(reference, state, command) !=
        nmpc::Status::SUCCESS) {
        return 1;
    }
    force_sum.zero();
    for (int foot = 0; foot < kNumFeet; ++foot) {
        for (int axis = 0; axis < 3; ++axis)
            force_sum[axis] += command.contact_forces_world[foot][axis];
        if (!friction_feasible(command.contact_forces_world[foot],
                               parameters.friction,
                               parameters.maximum_normal_force)) {
            std::printf("feedback WBC violated friction cone\n");
            return 1;
        }
    }
    if (!(force_sum[0] > 0.1)) {
        std::printf("WBC did not oppose base position error\n");
        return 1;
    }

    state.base.position_world[0] = 0.0;
    reference.feet[0].planned_contact = false;
    reference.feet[0].normal_force = 0.0;
    reference.feet[0].force_world.zero();
    reference.feet[0].position_world[0] += 0.02;
    if (controller.compute(reference, state, command) !=
        nmpc::Status::SUCCESS) {
        return 1;
    }
    if (command.contact_forces_world[0].norm2() > 1e-12 ||
        command.contact_blend[0] != 0.0 ||
        std::fabs(command.joint_torques[0][0] - 3.0) > 1e-8 ||
        std::fabs(command.swing_feedback_forces_world[0][0] - 3.0) > 1e-8 ||
        command.force_iterations >= parameters.maximum_iterations) {
        std::printf("swing-foot WBC mapping failed: force=%.3e, torque=%.3e\n",
                    command.contact_forces_world[0].norm2(),
                    command.joint_torques[0][0]);
        return 1;
    }
    double residual_maximum = 0.0;
    for (int row = 0; row < 6; ++row) {
        residual_maximum = std::max(
            residual_maximum,
            std::fabs(command.wrench_residual_components[row]));
    }
    if (std::fabs(residual_maximum - command.wrench_residual) > 1e-12)
        return 1;
    return 0;
}

int test_jacobian_condition_number() {
    Mat<3, 3> identity;
    identity.zero();
    for (int axis = 0; axis < 3; ++axis) identity(axis, axis) = 1.0;
    if (std::fabs(frobenius_condition_number_3x3(identity) - 3.0) > 1e-12)
        return 1;
    for (int axis = 0; axis < 3; ++axis) identity(axis, axis) = 1e-5;
    if (std::fabs(frobenius_condition_number_3x3(identity) - 3.0) > 1e-12)
        return 1;

    Mat<3, 3> diagonal;
    diagonal.zero();
    diagonal(0, 0) = 1.0;
    diagonal(1, 1) = 2.0;
    diagonal(2, 2) = 4.0;
    if (std::fabs(frobenius_condition_number_3x3(diagonal) - 5.25) >
        1e-12) {
        return 1;
    }
    diagonal(2, 2) = 0.0;
    return frobenius_condition_number_3x3(diagonal) >= 1e299 ? 0 : 1;
}

int test_friction_projection() {
    Vec<3> normal;
    normal.zero();
    normal[2] = 1.0;
    Vec<3> tangent1;
    Vec<3> tangent2;
    Vec<3> normalized;
    terrain_basis_from_normal(normal, normalized, tangent1, tangent2);
    Vec<3> force;
    force[0] = 100.0;
    force[1] = -30.0;
    force[2] = 10.0;
    project_friction_cone(force, normalized, tangent1, tangent2, 0.5, 80.0);
    if (!friction_feasible(force, 0.5, 80.0)) {
        std::printf("friction projection is infeasible\n");
        return 1;
    }
    return 0;
}

}  // namespace

int main() {
    int failures = 0;
    failures += test_plan_sampling();
    failures += test_repeated_swing_metrics();
    failures += test_swing_reference_shaping();
    failures += test_terrain_relative_swing_reference();
    failures += test_contact_execution_feedback();
    failures += test_stance_contact_confirmation();
    failures += test_contact_torque_blend_handoff();
    failures += test_friction_projection();
    failures += test_convex_wbc();
    failures += test_jacobian_condition_number();
    if (failures == 0) {
        std::printf("Quadruped CITO execution tests passed.\n");
        return 0;
    }
    std::printf("Quadruped CITO execution tests failed: %d\n", failures);
    return 1;
}
