#pragma once

#include <cstdio>
#include <cstddef>
#include <cstdint>
#include <cstring>

#include "examples/quadruped_cito/quadruped_cito_plan.hpp"

namespace quadruped_cito {
namespace detail {

inline void fingerprint_bytes(std::uint64_t& fingerprint,
                              const void* data, std::size_t size) {
    const unsigned char* bytes = static_cast<const unsigned char*>(data);
    for (std::size_t index = 0; index < size; ++index) {
        fingerprint ^= bytes[index];
        fingerprint *= 1099511628211ULL;
    }
}

template <typename Value>
inline void fingerprint_value(std::uint64_t& fingerprint,
                              const Value& value) {
    fingerprint_bytes(fingerprint, &value, sizeof(value));
}

inline bool write_base_sample(FILE* file, const BasePlanSample& base) {
    return std::fprintf(
               file,
               "base %.17g %.17g %.17g %.17g %.17g %.17g %.17g "
               "%.17g %.17g %.17g %.17g %.17g %.17g\n",
               base.position_world[0], base.position_world[1],
               base.position_world[2], base.orientation_body_to_world[0],
               base.orientation_body_to_world[1],
               base.orientation_body_to_world[2],
               base.orientation_body_to_world[3],
               base.linear_velocity_world[0], base.linear_velocity_world[1],
               base.linear_velocity_world[2], base.angular_velocity_body[0],
               base.angular_velocity_body[1],
               base.angular_velocity_body[2]) > 0;
}

inline bool read_base_sample(FILE* file, BasePlanSample& base) {
    char tag[16] = {};
    return std::fscanf(
               file,
               "%15s %lf %lf %lf %lf %lf %lf %lf %lf %lf %lf %lf %lf %lf",
               tag, &base.position_world[0], &base.position_world[1],
               &base.position_world[2], &base.orientation_body_to_world[0],
               &base.orientation_body_to_world[1],
               &base.orientation_body_to_world[2],
               &base.orientation_body_to_world[3],
               &base.linear_velocity_world[0],
               &base.linear_velocity_world[1],
               &base.linear_velocity_world[2],
               &base.angular_velocity_body[0],
               &base.angular_velocity_body[1],
               &base.angular_velocity_body[2]) == 14 &&
           std::strcmp(tag, "base") == 0;
}

inline bool write_foot_sample(FILE* file, int foot,
                              const FootPlanSample& sample) {
    return std::fprintf(
               file,
               "foot %d %.17g %.17g %.17g %.17g %.17g %.17g %.17g %.17g "
               "%.17g %.17g %.17g %.17g %.17g %.17g %d\n",
               foot, sample.position_world[0], sample.position_world[1],
               sample.position_world[2], sample.velocity_world[0],
               sample.velocity_world[1], sample.velocity_world[2],
               sample.force_world[0], sample.force_world[1],
               sample.force_world[2], sample.terrain_normal_world[0],
               sample.terrain_normal_world[1],
               sample.terrain_normal_world[2], sample.terrain_gap,
               sample.normal_force, sample.planned_contact ? 1 : 0) > 0;
}

inline bool read_foot_sample(FILE* file, int expected_foot,
                             FootPlanSample& sample) {
    char tag[16] = {};
    int foot = -1;
    int planned_contact = 0;
    const int fields = std::fscanf(
        file,
        "%15s %d %lf %lf %lf %lf %lf %lf %lf %lf %lf %lf %lf %lf %lf "
        "%lf %d",
        tag, &foot, &sample.position_world[0], &sample.position_world[1],
        &sample.position_world[2], &sample.velocity_world[0],
        &sample.velocity_world[1], &sample.velocity_world[2],
        &sample.force_world[0], &sample.force_world[1],
        &sample.force_world[2], &sample.terrain_normal_world[0],
        &sample.terrain_normal_world[1], &sample.terrain_normal_world[2],
        &sample.terrain_gap, &sample.normal_force, &planned_contact);
    if (fields != 17 || std::strcmp(tag, "foot") != 0 ||
        foot != expected_foot || (planned_contact != 0 && planned_contact != 1)) {
        return false;
    }
    sample.planned_contact = planned_contact != 0;
    return true;
}

}  // namespace detail

template <int Horizon>
std::uint64_t contact_plan_fingerprint(const ContactPlan<Horizon>& plan) {
    std::uint64_t fingerprint = 14695981039346656037ULL;
    detail::fingerprint_value(fingerprint, Horizon);
    for (int stage = 0; stage < Horizon; ++stage) {
        const ContactPlanStage& sample = plan.stages[stage];
        detail::fingerprint_value(fingerprint, sample.time);
        detail::fingerprint_value(fingerprint, sample.duration);
        detail::fingerprint_bytes(
            fingerprint, sample.base.position_world.data, 3 * sizeof(double));
        detail::fingerprint_bytes(
            fingerprint, sample.base.orientation_body_to_world.data,
            4 * sizeof(double));
        detail::fingerprint_bytes(
            fingerprint, sample.base.linear_velocity_world.data,
            3 * sizeof(double));
        detail::fingerprint_bytes(
            fingerprint, sample.base.angular_velocity_body.data,
            3 * sizeof(double));
        for (int foot = 0; foot < kNumFeet; ++foot) {
            const FootPlanSample& foot_sample = sample.feet[foot];
            detail::fingerprint_bytes(
                fingerprint, foot_sample.position_world.data,
                3 * sizeof(double));
            detail::fingerprint_bytes(
                fingerprint, foot_sample.velocity_world.data,
                3 * sizeof(double));
            detail::fingerprint_bytes(
                fingerprint, foot_sample.force_world.data,
                3 * sizeof(double));
            detail::fingerprint_bytes(
                fingerprint, foot_sample.terrain_normal_world.data,
                3 * sizeof(double));
            detail::fingerprint_value(fingerprint, foot_sample.terrain_gap);
            detail::fingerprint_value(fingerprint, foot_sample.normal_force);
            const unsigned char contact =
                foot_sample.planned_contact ? 1U : 0U;
            detail::fingerprint_value(fingerprint, contact);
        }
    }
    detail::fingerprint_value(fingerprint, plan.terminal.time);
    detail::fingerprint_bytes(
        fingerprint, plan.terminal.base.position_world.data,
        3 * sizeof(double));
    detail::fingerprint_bytes(
        fingerprint, plan.terminal.base.orientation_body_to_world.data,
        4 * sizeof(double));
    detail::fingerprint_bytes(
        fingerprint, plan.terminal.base.linear_velocity_world.data,
        3 * sizeof(double));
    detail::fingerprint_bytes(
        fingerprint, plan.terminal.base.angular_velocity_body.data,
        3 * sizeof(double));
    for (int foot = 0; foot < kNumFeet; ++foot) {
        detail::fingerprint_bytes(
            fingerprint, plan.terminal.foot_positions_world[foot].data,
            3 * sizeof(double));
    }
    return fingerprint;
}

inline int read_contact_plan_horizon(const char* path) {
    if (!path || path[0] == '\0') return 0;
    FILE* file = std::fopen(path, "r");
    if (!file) return 0;
    char header[32] = {};
    int horizon = 0;
    const bool ok = std::fscanf(file, "%31s %d", header, &horizon) == 2 &&
                    std::strcmp(header, "quadruped_cito_plan_v1") == 0 &&
                    horizon > 0;
    std::fclose(file);
    return ok ? horizon : 0;
}

template <int Horizon>
Status write_contact_plan(const ContactPlan<Horizon>& plan, const char* path) {
    if (!path || path[0] == '\0') return Status::BAD_ARGUMENT;
    FILE* file = std::fopen(path, "w");
    if (!file) return Status::BAD_ARGUMENT;
    bool ok = std::fprintf(file, "quadruped_cito_plan_v1 %d\n", Horizon) > 0;
    for (int stage = 0; ok && stage < Horizon; ++stage) {
        ok = std::fprintf(file, "stage %d %.17g %.17g\n", stage,
                          plan.stages[stage].time,
                          plan.stages[stage].duration) > 0 &&
             detail::write_base_sample(file, plan.stages[stage].base);
        for (int foot = 0; ok && foot < kNumFeet; ++foot) {
            ok = detail::write_foot_sample(
                file, foot, plan.stages[stage].feet[foot]);
        }
    }
    ok = ok && std::fprintf(file, "terminal %.17g\n", plan.terminal.time) > 0 &&
         detail::write_base_sample(file, plan.terminal.base);
    for (int foot = 0; ok && foot < kNumFeet; ++foot) {
        const Vec<3>& position = plan.terminal.foot_positions_world[foot];
        ok = std::fprintf(file, "terminal_foot %d %.17g %.17g %.17g\n",
                          foot, position[0], position[1], position[2]) > 0;
    }
    ok = std::fclose(file) == 0 && ok;
    return ok ? Status::SUCCESS : Status::BAD_ARGUMENT;
}

template <int Horizon>
Status read_contact_plan(const char* path, ContactPlan<Horizon>& plan) {
    if (!path || path[0] == '\0') return Status::BAD_ARGUMENT;
    FILE* file = std::fopen(path, "r");
    if (!file) return Status::BAD_ARGUMENT;
    char header[32] = {};
    int horizon = 0;
    bool ok = std::fscanf(file, "%31s %d", header, &horizon) == 2 &&
              std::strcmp(header, "quadruped_cito_plan_v1") == 0 &&
              horizon == Horizon;
    for (int stage = 0; ok && stage < Horizon; ++stage) {
        char tag[16] = {};
        int index = -1;
        ok = std::fscanf(file, "%15s %d %lf %lf", tag, &index,
                         &plan.stages[stage].time,
                         &plan.stages[stage].duration) == 4 &&
             std::strcmp(tag, "stage") == 0 && index == stage &&
             detail::read_base_sample(file, plan.stages[stage].base);
        for (int foot = 0; ok && foot < kNumFeet; ++foot) {
            ok = detail::read_foot_sample(
                file, foot, plan.stages[stage].feet[foot]);
        }
    }
    if (ok) {
        char tag[16] = {};
        ok = std::fscanf(file, "%15s %lf", tag, &plan.terminal.time) == 2 &&
             std::strcmp(tag, "terminal") == 0 &&
             detail::read_base_sample(file, plan.terminal.base);
    }
    for (int foot = 0; ok && foot < kNumFeet; ++foot) {
        char tag[24] = {};
        int index = -1;
        Vec<3>& position = plan.terminal.foot_positions_world[foot];
        ok = std::fscanf(file, "%23s %d %lf %lf %lf", tag, &index,
                         &position[0], &position[1], &position[2]) == 5 &&
             std::strcmp(tag, "terminal_foot") == 0 && index == foot;
    }
    std::fclose(file);
    return ok ? Status::SUCCESS : Status::BAD_ARGUMENT;
}

}  // namespace quadruped_cito
