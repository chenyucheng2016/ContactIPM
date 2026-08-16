#include <cstdio>
#include <memory>

#include "examples/quadruped_cito/quadruped_cito_realtime_planner.hpp"

namespace {

using namespace quadruped_cito;

int support_schedule_foot(
    const realtime_detail::RealtimeContactTaskSequence& sequence,
    int absolute_knot) {
    return sequence.designated_support_foot_at(absolute_knot);
}

bool classified_contact(const QuadrupedCITORealtimePlanner& planner,
                        int stage, int foot) {
    const Vec<3> position = state_vector3(
        planner.problem().stages[stage].x,
        StateIndex::foot_position(foot, 0));
    TerrainSample terrain_sample;
    if (!planner.planner_terrain().sample(position, terrain_sample))
        return false;
    const Vec<3> force = control_vector3(
        planner.problem().stages[stage].u,
        ControlIndex::contact_force(foot, 0));
    const ContactClassification classification;
    return terrain_sample.gap <= classification.maximum_contact_gap &&
        dot3(terrain_sample.normal, force) >=
            classification.minimum_contact_force;
}

}  // namespace

int main() {
    constexpr int kTaskCount = 24;
    RealtimePlannerConfig configuration;
    configuration.sustained_contact_tasks = true;
    configuration.maximum_contact_tasks = kTaskCount;
    const SharedTerrain terrain = register_go1_terrain(
        SharedTerrain::sinusoidal(0.02, 4.0, 3.0));
    auto planner = std::make_unique<QuadrupedCITORealtimePlanner>(
        terrain, configuration);
    const Vec<kStateDim> standing = planner->nominal_state();
    if (planner->initialize(standing) != nmpc::Status::SUCCESS) return 1;
    const RealtimePlannerResult cold = planner->cold_solve();
    if (!cold.published) {
        std::printf("sustained topology cold solve failed: %s\n",
                    nmpc::status_string(cold.status));
        return 1;
    }

    realtime_detail::RealtimeContactTaskSequence sequence(kTaskCount);
    sequence.initialize(standing);
    int designated_unloaded_stages = 0;
    for (int stage = 0; stage < kRealtimeHorizon; ++stage) {
        const int designated = support_schedule_foot(sequence, stage);
        for (int foot = 0; foot < kNumFeet; ++foot) {
            const bool contact = classified_contact(*planner, stage, foot);
            if (foot == designated) {
                designated_unloaded_stages += contact ? 0 : 1;
            } else if (!contact) {
                std::printf(
                    "hard-support topology failed: stage=%d designated=%d "
                    "unloaded_foot=%d\n",
                    stage, designated, foot);
                return 1;
            }
        }
    }
    if (designated_unloaded_stages == 0) {
        std::printf("cold plan contained no designated-foot transition\n");
        return 1;
    }
    std::printf(
        "sustained cold support topology passed: designated_unloaded=%d\n",
        designated_unloaded_stages);
    return 0;
}
