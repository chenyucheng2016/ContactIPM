#pragma once

#include "examples/quadruped_cito/quadruped_cito_model.hpp"

namespace quadruped_cito {

struct ContactClassification {
    double maximum_contact_gap = 2e-3;
    double minimum_contact_force = 5.0;
};

struct BasePlanSample {
    Vec<3> position_world;
    Vec<4> orientation_body_to_world;
    Vec<3> linear_velocity_world;
    Vec<3> angular_velocity_body;
};

struct FootPlanSample {
    Vec<3> position_world;
    Vec<3> velocity_world;
    Vec<3> force_world;
    Vec<3> terrain_normal_world;
    double terrain_gap = 0.0;
    double normal_force = 0.0;
    bool planned_contact = false;
};

struct ContactPlanStage {
    double time = 0.0;
    double duration = 0.0;
    BasePlanSample base;
    FootPlanSample feet[kNumFeet];
};

struct ContactPlanTerminal {
    double time = 0.0;
    BasePlanSample base;
    Vec<3> foot_positions_world[kNumFeet];
};

template <int Horizon>
struct ContactPlan {
    ContactPlanStage stages[Horizon];
    ContactPlanTerminal terminal;
};

enum class PlanSampleStatus {
    ACTIVE,
    BEFORE_START,
    EXPIRED
};

struct ContactPlanSample {
    double time = 0.0;
    int stage_index = 0;
    double stage_phase = 0.0;
    BasePlanSample base;
    FootPlanSample feet[kNumFeet];
};

inline Vec<4> normalized_quaternion_lerp(const Vec<4>& first,
                                         const Vec<4>& second,
                                         double phase) {
    double dot = 0.0;
    for (int element = 0; element < 4; ++element)
        dot += first[element] * second[element];
    Vec<4> blended;
    for (int element = 0; element < 4; ++element) {
        const double aligned_second =
            dot < 0.0 ? -second[element] : second[element];
        blended[element] =
            (1.0 - phase) * first[element] + phase * aligned_second;
    }
    Vec<4> normalized;
    if (!normalize_quaternion(blended, normalized)) {
        normalized = first;
    }
    return normalized;
}

inline BasePlanSample interpolate_base_sample(const BasePlanSample& first,
                                              const BasePlanSample& second,
                                              double phase) {
    BasePlanSample result;
    for (int axis = 0; axis < 3; ++axis) {
        result.position_world[axis] =
            (1.0 - phase) * first.position_world[axis] +
            phase * second.position_world[axis];
        result.linear_velocity_world[axis] =
            (1.0 - phase) * first.linear_velocity_world[axis] +
            phase * second.linear_velocity_world[axis];
        result.angular_velocity_body[axis] =
            (1.0 - phase) * first.angular_velocity_body[axis] +
            phase * second.angular_velocity_body[axis];
    }
    result.orientation_body_to_world = normalized_quaternion_lerp(
        first.orientation_body_to_world,
        second.orientation_body_to_world, phase);
    return result;
}

template <int Horizon>
PlanSampleStatus sample_contact_plan(const ContactPlan<Horizon>& plan,
                                     double time,
                                     ContactPlanSample& sample) {
    static_assert(Horizon > 0, "contact plan horizon must be positive");
    if (time >= plan.terminal.time) {
        sample.time = time;
        sample.stage_index = Horizon;
        sample.stage_phase = 1.0;
        sample.base = plan.terminal.base;
        for (int foot = 0; foot < kNumFeet; ++foot) {
            sample.feet[foot] = plan.stages[Horizon - 1].feet[foot];
            sample.feet[foot].position_world =
                plan.terminal.foot_positions_world[foot];
            sample.feet[foot].velocity_world.zero();
            sample.feet[foot].force_world.zero();
            sample.feet[foot].normal_force = 0.0;
            sample.feet[foot].planned_contact = false;
        }
        return PlanSampleStatus::EXPIRED;
    }

    const bool before_start = time < plan.stages[0].time;
    const double active_time = before_start ? plan.stages[0].time : time;
    int stage = 0;
    while (stage + 1 < Horizon &&
           active_time >= plan.stages[stage + 1].time) {
        ++stage;
    }
    const ContactPlanStage& current = plan.stages[stage];
    const double phase = current.duration > 0.0
        ? std::max(0.0, std::min(
              1.0, (active_time - current.time) / current.duration))
        : 0.0;
    const BasePlanSample& next_base = stage + 1 < Horizon
        ? plan.stages[stage + 1].base
        : plan.terminal.base;
    sample.time = time;
    sample.stage_index = stage;
    sample.stage_phase = phase;
    sample.base = interpolate_base_sample(current.base, next_base, phase);
    for (int foot = 0; foot < kNumFeet; ++foot) {
        sample.feet[foot] = current.feet[foot];
        const Vec<3>& next_position = stage + 1 < Horizon
            ? plan.stages[stage + 1].feet[foot].position_world
            : plan.terminal.foot_positions_world[foot];
        for (int axis = 0; axis < 3; ++axis) {
            sample.feet[foot].position_world[axis] =
                (1.0 - phase) * current.feet[foot].position_world[axis] +
                phase * next_position[axis];
        }
    }
    return before_start ? PlanSampleStatus::BEFORE_START
                        : PlanSampleStatus::ACTIVE;
}

inline BasePlanSample base_plan_sample(const Vec<kStateDim>& state) {
    BasePlanSample sample;
    for (int axis = 0; axis < 3; ++axis) {
        sample.position_world[axis] =
            state[StateIndex::base_position(axis)];
        sample.linear_velocity_world[axis] =
            state[StateIndex::linear_velocity(axis)];
        sample.angular_velocity_body[axis] =
            state[StateIndex::angular_velocity(axis)];
    }
    for (int element = 0; element < 4; ++element) {
        sample.orientation_body_to_world[element] =
            state[StateIndex::quaternion(element)];
    }
    return sample;
}

template <int Horizon, typename Terrain>
Status extract_contact_plan(
    const Problem<Horizon>& problem, const Terrain& terrain,
    const ContactClassification& classification,
    ContactPlan<Horizon>& plan) {
    for (int stage = 0; stage < Horizon; ++stage) {
        ContactPlanStage& output = plan.stages[stage];
        output.time = stage * problem.dt;
        output.duration = problem.dt;
        output.base = base_plan_sample(problem.stages[stage].x);
        for (int foot = 0; foot < kNumFeet; ++foot) {
            FootPlanSample& foot_output = output.feet[foot];
            foot_output.position_world = state_vector3(
                problem.stages[stage].x,
                StateIndex::foot_position(foot, 0));
            foot_output.velocity_world = control_vector3(
                problem.stages[stage].u,
                ControlIndex::foot_velocity(foot, 0));
            foot_output.force_world = control_vector3(
                problem.stages[stage].u,
                ControlIndex::contact_force(foot, 0));

            TerrainSample terrain_sample;
            if (!terrain.sample(foot_output.position_world, terrain_sample))
                return Status::BAD_ARGUMENT;
            foot_output.terrain_normal_world = terrain_sample.normal;
            foot_output.terrain_gap = terrain_sample.gap;
            foot_output.normal_force =
                dot3(terrain_sample.normal, foot_output.force_world);
            foot_output.planned_contact =
                foot_output.terrain_gap <=
                    classification.maximum_contact_gap &&
                foot_output.normal_force >=
                    classification.minimum_contact_force;
        }
    }

    plan.terminal.time = Horizon * problem.dt;
    plan.terminal.base = base_plan_sample(problem.stages[Horizon].x);
    for (int foot = 0; foot < kNumFeet; ++foot) {
        plan.terminal.foot_positions_world[foot] = state_vector3(
            problem.stages[Horizon].x,
            StateIndex::foot_position(foot, 0));
    }
    return Status::SUCCESS;
}

}  // namespace quadruped_cito
