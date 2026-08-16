#pragma once

#include <algorithm>
#include <cmath>

#include "examples/quadruped_cito/quadruped_cito_plan.hpp"

namespace quadruped_cito {

inline double smooth_step(double phase) {
    return phase * phase * (3.0 - 2.0 * phase);
}

inline double smooth_step_derivative(double phase) {
    return 6.0 * phase * (1.0 - phase);
}

template <int Horizon, typename Terrain>
void shape_swing_references(
    const ContactPlan<Horizon>& plan, const Terrain& terrain,
    ContactPlanSample& sample) {
    if (sample.stage_index < 0 || sample.stage_index >= Horizon) return;
    for (int foot = 0; foot < kNumFeet; ++foot) {
        if (sample.feet[foot].planned_contact) continue;

        int liftoff_stage = sample.stage_index;
        while (liftoff_stage > 0 &&
               !plan.stages[liftoff_stage - 1].feet[foot].planned_contact) {
            --liftoff_stage;
        }
        int touchdown_stage = sample.stage_index + 1;
        while (touchdown_stage < Horizon &&
               !plan.stages[touchdown_stage].feet[foot].planned_contact) {
            ++touchdown_stage;
        }
        if (touchdown_stage >= Horizon) continue;

        const double liftoff_time = plan.stages[liftoff_stage].time;
        const double touchdown_time = plan.stages[touchdown_stage].time;
        const double duration = touchdown_time - liftoff_time;
        if (!(duration > 0.0)) continue;
        const double phase = std::max(
            0.0, std::min(1.0, (sample.time - liftoff_time) / duration));
        const Vec<3>& start =
            plan.stages[liftoff_stage].feet[foot].position_world;
        const Vec<3>& target =
            plan.stages[touchdown_stage].feet[foot].position_world;

        const double blend = smooth_step(phase);
        const double blend_rate = smooth_step_derivative(phase) / duration;
        for (int axis = 0; axis < 2; ++axis) {
            const double displacement = target[axis] - start[axis];
            sample.feet[foot].position_world[axis] =
                start[axis] + blend * displacement;
            sample.feet[foot].velocity_world[axis] =
                blend_rate * displacement;
        }

        const double start_clearance =
            start[2] - terrain.height(start[0], start[1]);
        const double target_clearance =
            target[2] - terrain.height(target[0], target[1]);
        double apex_clearance = std::max(start_clearance, target_clearance);
        for (int stage = liftoff_stage; stage < touchdown_stage; ++stage) {
            const Vec<3>& position =
                plan.stages[stage].feet[foot].position_world;
            apex_clearance = std::max(
                apex_clearance,
                position[2] - terrain.height(position[0], position[1]));
        }

        const bool ascending = phase <= 0.5;
        const double half_phase = ascending
            ? 2.0 * phase
            : 2.0 * phase - 1.0;
        const double clearance_start = ascending
            ? start_clearance : apex_clearance;
        const double clearance_target = ascending
            ? apex_clearance : target_clearance;
        const double clearance_displacement =
            clearance_target - clearance_start;
        const double clearance = clearance_start +
            smooth_step(half_phase) * clearance_displacement;
        const double clearance_rate =
            2.0 * smooth_step_derivative(half_phase) *
            clearance_displacement / duration;

        Vec<3> terrain_position = sample.feet[foot].position_world;
        terrain_position[2] = terrain.height(
            terrain_position[0], terrain_position[1]);
        TerrainSample terrain_sample;
        if (!terrain.sample(terrain_position, terrain_sample) ||
            std::fabs(terrain_sample.gap_gradient[2]) <= 1e-12) {
            continue;
        }
        const double height_rate =
            -terrain_sample.gap_gradient[0] /
                terrain_sample.gap_gradient[2] *
                sample.feet[foot].velocity_world[0] -
            terrain_sample.gap_gradient[1] /
                terrain_sample.gap_gradient[2] *
                sample.feet[foot].velocity_world[1];
        sample.feet[foot].position_world[2] =
            terrain_position[2] + clearance;
        sample.feet[foot].velocity_world[2] =
            height_rate + clearance_rate;
    }
}

struct FlatSwingTerrain {
    double height(double, double) const { return 0.0; }

    bool sample(const Vec<3>& position, TerrainSample& sample) const {
        sample.gap = position[2];
        sample.gap_gradient.zero();
        sample.gap_gradient[2] = 1.0;
        return true;
    }
};

template <int Horizon>
void shape_swing_references(const ContactPlan<Horizon>& plan,
                            ContactPlanSample& sample) {
    shape_swing_references(plan, FlatSwingTerrain{}, sample);
}

}  // namespace quadruped_cito
