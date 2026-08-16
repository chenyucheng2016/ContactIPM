#pragma once

#include <algorithm>
#include <array>
#include <cmath>
#include <cstdint>
#include <limits>
#include <vector>

#include "examples/quadruped_cito/quadruped_cito_model.hpp"

namespace quadruped_cito {

struct RollingAuditThresholds {
    double maximum_contact_gap = 0.002;
    double minimum_contact_force = 5.0;
    double minimum_swing_clearance = 0.02;
    double maximum_swing_clearance = 0.10;
    double maximum_touchdown_gap = 0.002;
    double maximum_touchdown_speed = 0.01;
    double minimum_touchdown_force = 5.0;
    double requested_task_displacement = 0.08;
    double task_displacement_tolerance = 0.01;
    double maximum_target_error = 0.01;
    double maximum_stance_slip = 0.002;
    double maximum_abs_roll = 20.0 * 3.14159265358979323846 / 180.0;
    double maximum_abs_pitch = 20.0 * 3.14159265358979323846 / 180.0;
    double minimum_base_progress = 0.35;
    double maximum_base_lateral_deviation = 0.10;
    double minimum_foot_progress = 0.42;
    double expected_time_step = 0.05;
    double time_step_tolerance = 1e-9;
    int schedule_tolerance_knots = 4;
    int minimum_final_support_samples = 4;
    int required_task_count = 24;
    std::array<int, kNumFeet> expected_foot_order = {{2, 3, 0, 1}};
};

struct RollingContactTask {
    std::uint64_t id = 0;
    int ordinal = 0;
    int moving_foot = -1;
    int start_knot = -1;
    int touchdown_knot = -1;
    double origin_x = 0.0;
    double target_x = 0.0;
};

struct RollingExecutedSample {
    int absolute_knot = -1;
    double time = -1.0;
    Vec<kStateDim> state;
    Vec<kControlDim> control;
};

struct RollingFootContact {
    Vec<3> position_world;
    Vec<3> velocity_world;
    Vec<3> force_world;
    Vec<3> terrain_normal_world;
    double terrain_gap = 0.0;
    double normal_force = 0.0;
    double speed = 0.0;
    bool contact = false;
};

struct RollingContactFrame {
    int absolute_knot = -1;
    double time = -1.0;
    std::array<RollingFootContact, kNumFeet> feet;
};

struct RollingTaskAudit {
    RollingContactTask task;
    int liftoff_count = 0;
    int touchdown_count = 0;
    int schedule_violation_count = 0;
    int chatter_count = 0;
    double measured_liftoff_time = -1.0;
    double measured_touchdown_time = -1.0;
    double maximum_clearance = 0.0;
    double touchdown_gap = std::numeric_limits<double>::infinity();
    double touchdown_speed = std::numeric_limits<double>::infinity();
    double touchdown_normal_force =
        -std::numeric_limits<double>::infinity();
    double displacement = 0.0;
    double target_error = std::numeric_limits<double>::infinity();
    double maximum_post_touchdown_slip = 0.0;
    Vec<3> liftoff_position;
    Vec<3> touchdown_position;
    bool pass = false;
};

struct RollingAuditReport {
    bool ledger_valid = true;
    bool input_valid = true;
    bool all_tasks_completed = false;
    bool final_support = false;
    bool pass = false;
    int samples = 0;
    int last_absolute_knot = -1;
    int completed_tasks = 0;
    int unexpected_liftoffs = 0;
    int unexpected_touchdowns = 0;
    int non_designated_airborne_samples = 0;
    int schedule_violation_count = 0;
    int task_order_violations = 0;
    int contact_chatter_events = 0;
    int consecutive_final_support_samples = 0;
    double base_progress = 0.0;
    double maximum_base_lateral_deviation = 0.0;
    double maximum_abs_roll = 0.0;
    double maximum_abs_pitch = 0.0;
    std::array<double, kNumFeet> foot_progress = {{0.0, 0.0, 0.0, 0.0}};
    std::vector<RollingTaskAudit> tasks;
};

namespace rolling_audit_detail {

inline bool finite(double value) { return std::isfinite(value); }

template <int Size>
bool finite_vector(const Vec<Size>& value) {
    for (int index = 0; index < Size; ++index) {
        if (!finite(value[index])) return false;
    }
    return true;
}

inline bool valid_thresholds(const RollingAuditThresholds& thresholds) {
    if (!finite(thresholds.maximum_contact_gap) ||
        !finite(thresholds.minimum_contact_force) ||
        !finite(thresholds.minimum_swing_clearance) ||
        !finite(thresholds.maximum_swing_clearance) ||
        !finite(thresholds.maximum_touchdown_gap) ||
        !finite(thresholds.maximum_touchdown_speed) ||
        !finite(thresholds.minimum_touchdown_force) ||
        !finite(thresholds.requested_task_displacement) ||
        !finite(thresholds.task_displacement_tolerance) ||
        !finite(thresholds.maximum_target_error) ||
        !finite(thresholds.maximum_stance_slip) ||
        !finite(thresholds.maximum_abs_roll) ||
        !finite(thresholds.maximum_abs_pitch) ||
        !finite(thresholds.minimum_base_progress) ||
        !finite(thresholds.maximum_base_lateral_deviation) ||
        !finite(thresholds.minimum_foot_progress) ||
        !finite(thresholds.expected_time_step) ||
        !finite(thresholds.time_step_tolerance)) {
        return false;
    }
    if (thresholds.maximum_swing_clearance <
            thresholds.minimum_swing_clearance ||
        thresholds.maximum_touchdown_speed < 0.0 ||
        thresholds.task_displacement_tolerance < 0.0 ||
        thresholds.maximum_target_error < 0.0 ||
        thresholds.maximum_stance_slip < 0.0 ||
        thresholds.maximum_abs_roll <= 0.0 ||
        thresholds.maximum_abs_pitch <= 0.0 ||
        thresholds.maximum_base_lateral_deviation < 0.0 ||
        thresholds.expected_time_step <= 0.0 ||
        thresholds.time_step_tolerance < 0.0 ||
        thresholds.schedule_tolerance_knots < 0 ||
        thresholds.minimum_final_support_samples <= 0 ||
        thresholds.required_task_count <= 0) {
        return false;
    }
    for (int foot : thresholds.expected_foot_order) {
        if (foot < 0 || foot >= kNumFeet) return false;
    }
    for (int first = 0; first < kNumFeet; ++first) {
        for (int second = first + 1; second < kNumFeet; ++second) {
            if (thresholds.expected_foot_order[first] ==
                thresholds.expected_foot_order[second]) {
                return false;
            }
        }
    }
    return true;
}

inline void quaternion_roll_pitch(const Vec<kStateDim>& state,
                                  double& roll, double& pitch) {
    Vec<4> quaternion;
    if (!normalize_quaternion(state_quaternion(state), quaternion)) {
        roll = std::numeric_limits<double>::infinity();
        pitch = std::numeric_limits<double>::infinity();
        return;
    }
    const double w = quaternion[0];
    const double x = quaternion[1];
    const double y = quaternion[2];
    const double z = quaternion[3];
    roll = std::atan2(2.0 * (w * x + y * z),
                      1.0 - 2.0 * (x * x + y * y));
    const double pitch_argument = 2.0 * (w * y - z * x);
    pitch = std::asin(std::max(-1.0, std::min(1.0, pitch_argument)));
}

inline bool task_passes(const RollingTaskAudit& audit,
                        const RollingAuditThresholds& thresholds) {
    return audit.liftoff_count == 1 && audit.touchdown_count == 1 &&
        audit.schedule_violation_count == 0 && audit.chatter_count == 0 &&
        audit.measured_liftoff_time >= 0.0 &&
        audit.measured_touchdown_time > audit.measured_liftoff_time &&
        audit.maximum_clearance >= thresholds.minimum_swing_clearance &&
        audit.maximum_clearance <= thresholds.maximum_swing_clearance &&
        std::fabs(audit.touchdown_gap) <=
            thresholds.maximum_touchdown_gap &&
        audit.touchdown_speed <= thresholds.maximum_touchdown_speed &&
        audit.touchdown_normal_force >= thresholds.minimum_touchdown_force &&
        std::fabs(audit.displacement -
                  thresholds.requested_task_displacement) <=
            thresholds.task_displacement_tolerance &&
        audit.target_error <= thresholds.maximum_target_error &&
        audit.maximum_post_touchdown_slip <=
            thresholds.maximum_stance_slip;
}

}  // namespace rolling_audit_detail

template <typename Terrain>
bool classify_rolling_contacts(const Terrain& terrain,
                               const RollingExecutedSample& sample,
                               const RollingAuditThresholds& thresholds,
                               RollingContactFrame& frame) {
    if (!rolling_audit_detail::valid_thresholds(thresholds) ||
        !rolling_audit_detail::finite(sample.time) ||
        !rolling_audit_detail::finite_vector(sample.state) ||
        !rolling_audit_detail::finite_vector(sample.control)) {
        return false;
    }
    frame.absolute_knot = sample.absolute_knot;
    frame.time = sample.time;
    for (int foot = 0; foot < kNumFeet; ++foot) {
        RollingFootContact& output = frame.feet[foot];
        output.position_world = state_vector3(
            sample.state, StateIndex::foot_position(foot, 0));
        output.velocity_world = control_vector3(
            sample.control, ControlIndex::foot_velocity(foot, 0));
        output.force_world = control_vector3(
            sample.control, ControlIndex::contact_force(foot, 0));
        TerrainSample terrain_sample;
        if (!terrain.sample(output.position_world, terrain_sample) ||
            !rolling_audit_detail::finite(terrain_sample.gap) ||
            !rolling_audit_detail::finite_vector(terrain_sample.normal)) {
            return false;
        }
        output.terrain_normal_world = terrain_sample.normal;
        output.terrain_gap = terrain_sample.gap;
        output.normal_force = dot3(terrain_sample.normal, output.force_world);
        output.speed = output.velocity_world.norm2();
        if (!rolling_audit_detail::finite(output.normal_force) ||
            !rolling_audit_detail::finite(output.speed)) {
            return false;
        }
        output.contact =
            std::fabs(output.terrain_gap) <=
                thresholds.maximum_contact_gap &&
            output.normal_force >= thresholds.minimum_contact_force;
    }
    return true;
}

template <typename Terrain>
class RollingSRBDAudit {
public:
    RollingSRBDAudit(const Terrain& terrain,
                     const Vec<kStateDim>& initial_state,
                     const std::vector<RollingContactTask>& tasks,
                     const RollingAuditThresholds& thresholds = {})
        : terrain_(terrain), initial_state_(initial_state),
          thresholds_(thresholds) {
        report_.tasks.resize(tasks.size());
        for (std::size_t index = 0; index < tasks.size(); ++index)
            report_.tasks[index].task = tasks[index];
        previous_contact_.fill(true);
        active_task_by_foot_.fill(-1);
        stance_task_by_foot_.fill(-1);
        for (int foot = 0; foot < kNumFeet; ++foot) {
            last_contact_position_[foot] = state_vector3(
                initial_state_, StateIndex::foot_position(foot, 0));
        }
        report_.ledger_valid = validate_ledger();
        report_.input_valid =
            rolling_audit_detail::valid_thresholds(thresholds_) &&
            rolling_audit_detail::finite_vector(initial_state_);
    }

    bool observe(const RollingExecutedSample& sample) {
        if (!report_.ledger_valid || !report_.input_valid ||
            sample.absolute_knot < 0 || sample.time < 0.0 ||
            (report_.samples == 0 &&
             (sample.absolute_knot != 0 ||
              std::fabs(sample.time) > thresholds_.time_step_tolerance)) ||
            (report_.samples > 0 &&
             (sample.absolute_knot != report_.last_absolute_knot + 1 ||
              std::fabs((sample.time - last_time_) -
                        thresholds_.expected_time_step) >
                  thresholds_.time_step_tolerance))) {
            report_.input_valid = false;
            return false;
        }

        RollingContactFrame contact_frame;
        if (!classify_rolling_contacts(
                terrain_, sample, thresholds_, contact_frame)) {
            report_.input_valid = false;
            return false;
        }

        update_route_metrics(sample.state);
        process_transitions(contact_frame);
        update_airborne_and_slip_metrics(contact_frame);

        bool all_support = true;
        for (int foot = 0; foot < kNumFeet; ++foot) {
            previous_contact_[foot] = contact_frame.feet[foot].contact;
            if (contact_frame.feet[foot].contact) {
                last_contact_position_[foot] =
                    contact_frame.feet[foot].position_world;
            } else {
                all_support = false;
            }
        }
        if (next_task_index_ == static_cast<int>(report_.tasks.size()) &&
            all_support) {
            ++report_.consecutive_final_support_samples;
        } else {
            report_.consecutive_final_support_samples = 0;
        }

        last_contact_frame_ = contact_frame;
        last_time_ = sample.time;
        report_.last_absolute_knot = sample.absolute_knot;
        ++report_.samples;
        return true;
    }

    const RollingContactFrame& last_contact_frame() const {
        return last_contact_frame_;
    }

    RollingAuditReport finalize() const {
        RollingAuditReport result = report_;
        result.completed_tasks = 0;
        for (RollingTaskAudit& task : result.tasks) {
            task.pass = rolling_audit_detail::task_passes(task, thresholds_);
            if (task.liftoff_count == 1 && task.touchdown_count == 1)
                ++result.completed_tasks;
        }
        result.all_tasks_completed =
            result.completed_tasks == static_cast<int>(result.tasks.size()) &&
            static_cast<int>(result.tasks.size()) ==
                thresholds_.required_task_count;
        result.final_support = result.consecutive_final_support_samples >=
            thresholds_.minimum_final_support_samples;

        bool task_pass = true;
        for (const RollingTaskAudit& task : result.tasks)
            task_pass = task_pass && task.pass;
        bool foot_progress_pass = true;
        for (double progress : result.foot_progress) {
            foot_progress_pass = foot_progress_pass &&
                progress >= thresholds_.minimum_foot_progress;
        }
        result.pass = result.ledger_valid && result.input_valid &&
            result.all_tasks_completed && result.final_support && task_pass &&
            foot_progress_pass && result.unexpected_liftoffs == 0 &&
            result.unexpected_touchdowns == 0 &&
            result.non_designated_airborne_samples == 0 &&
            result.schedule_violation_count == 0 &&
            result.task_order_violations == 0 &&
            result.contact_chatter_events == 0 &&
            result.base_progress >= thresholds_.minimum_base_progress &&
            result.maximum_base_lateral_deviation <=
                thresholds_.maximum_base_lateral_deviation &&
            result.maximum_abs_roll <= thresholds_.maximum_abs_roll &&
            result.maximum_abs_pitch <= thresholds_.maximum_abs_pitch;
        return result;
    }

private:
    bool validate_ledger() const {
        if (static_cast<int>(report_.tasks.size()) !=
            thresholds_.required_task_count) {
            return false;
        }
        for (std::size_t index = 0; index < report_.tasks.size(); ++index) {
            const RollingContactTask& task = report_.tasks[index].task;
            if (task.id != index || task.ordinal != static_cast<int>(index) ||
                task.moving_foot != thresholds_.expected_foot_order[
                    index % thresholds_.expected_foot_order.size()] ||
                task.start_knot < 0 ||
                task.touchdown_knot <= task.start_knot ||
                !rolling_audit_detail::finite(task.origin_x) ||
                !rolling_audit_detail::finite(task.target_x) ||
                std::fabs((task.target_x - task.origin_x) -
                          thresholds_.requested_task_displacement) >
                    thresholds_.task_displacement_tolerance) {
                return false;
            }
            if (index > 0) {
                const RollingContactTask& previous =
                    report_.tasks[index - 1].task;
                if (task.start_knot - thresholds_.schedule_tolerance_knots <=
                    previous.touchdown_knot +
                        thresholds_.schedule_tolerance_knots) {
                    return false;
                }
            }
        }
        return true;
    }

    int owner_for_liftoff(int absolute_knot, int foot) const {
        for (std::size_t index = 0; index < report_.tasks.size(); ++index) {
            const RollingContactTask& task = report_.tasks[index].task;
            if (task.moving_foot != foot) continue;
            const int first = std::max(
                0, task.start_knot - thresholds_.schedule_tolerance_knots);
            const int last = task.touchdown_knot +
                thresholds_.schedule_tolerance_knots;
            if (absolute_knot >= first && absolute_knot <= last)
                return static_cast<int>(index);
        }
        return -1;
    }

    bool inside_task_window(int task_index, int absolute_knot) const {
        const RollingContactTask& task = report_.tasks[task_index].task;
        return absolute_knot >= std::max(
                   0, task.start_knot - thresholds_.schedule_tolerance_knots) &&
            absolute_knot <= task.touchdown_knot +
                thresholds_.schedule_tolerance_knots;
    }

    void record_liftoff(int foot, int task_index,
                        const RollingFootContact& contact, double time) {
        RollingTaskAudit& task = report_.tasks[task_index];
        ++task.liftoff_count;
        if (task.liftoff_count != 1 || active_task_by_foot_[foot] >= 0) {
            ++task.chatter_count;
            ++report_.contact_chatter_events;
            ++report_.unexpected_liftoffs;
            return;
        }
        if (task_index != next_task_index_) {
            ++task.schedule_violation_count;
            ++report_.schedule_violation_count;
            ++report_.task_order_violations;
        }
        task.measured_liftoff_time = time;
        task.liftoff_position = last_contact_position_[foot];
        task.maximum_clearance = std::max(
            task.maximum_clearance, contact.terrain_gap);
        active_task_by_foot_[foot] = task_index;
        stance_task_by_foot_[foot] = -1;
    }

    void record_touchdown(int foot, int task_index,
                          const RollingFootContact& contact, double time,
                          int absolute_knot) {
        RollingTaskAudit& task = report_.tasks[task_index];
        ++task.touchdown_count;
        if (task.touchdown_count != 1) {
            ++task.chatter_count;
            ++report_.contact_chatter_events;
            ++report_.unexpected_touchdowns;
            return;
        }
        if (!inside_task_window(task_index, absolute_knot)) {
            ++task.schedule_violation_count;
            ++report_.schedule_violation_count;
        }
        task.measured_touchdown_time = time;
        task.touchdown_position = contact.position_world;
        task.touchdown_gap = contact.terrain_gap;
        task.touchdown_speed = contact.speed;
        task.touchdown_normal_force = contact.normal_force;
        task.displacement = task.touchdown_position[0] -
            task.liftoff_position[0];
        task.target_error = std::fabs(
            task.touchdown_position[0] - task.task.target_x);
        report_.foot_progress[foot] += std::fabs(task.displacement);
        stance_task_by_foot_[foot] = task_index;
        active_task_by_foot_[foot] = -1;
        advance_completed_tasks();
    }

    void advance_completed_tasks() {
        while (next_task_index_ < static_cast<int>(report_.tasks.size())) {
            const RollingTaskAudit& task = report_.tasks[next_task_index_];
            if (task.liftoff_count != 1 || task.touchdown_count != 1) break;
            ++next_task_index_;
        }
    }

    void process_transitions(const RollingContactFrame& frame) {
        for (int foot = 0; foot < kNumFeet; ++foot) {
            const RollingFootContact& contact = frame.feet[foot];
            if (previous_contact_[foot] && !contact.contact) {
                const int owner = owner_for_liftoff(
                    frame.absolute_knot, foot);
                if (owner < 0) {
                    ++report_.unexpected_liftoffs;
                    ++report_.schedule_violation_count;
                } else {
                    record_liftoff(foot, owner, contact, frame.time);
                }
            } else if (!previous_contact_[foot] && contact.contact) {
                const int owner = active_task_by_foot_[foot];
                if (owner < 0) {
                    ++report_.unexpected_touchdowns;
                    ++report_.contact_chatter_events;
                } else {
                    record_touchdown(foot, owner, contact, frame.time,
                                     frame.absolute_knot);
                }
            }
        }
    }

    void update_airborne_and_slip_metrics(const RollingContactFrame& frame) {
        int active_swings = 0;
        for (int foot = 0; foot < kNumFeet; ++foot) {
            const RollingFootContact& contact = frame.feet[foot];
            const int active_task = active_task_by_foot_[foot];
            if (!contact.contact) {
                if (active_task < 0) {
                    ++report_.non_designated_airborne_samples;
                } else {
                    ++active_swings;
                    RollingTaskAudit& task = report_.tasks[active_task];
                    task.maximum_clearance = std::max(
                        task.maximum_clearance, contact.terrain_gap);
                    if (!inside_task_window(active_task,
                                            frame.absolute_knot)) {
                        ++task.schedule_violation_count;
                        ++report_.schedule_violation_count;
                    }
                }
            }
            const int stance_task = stance_task_by_foot_[foot];
            if (contact.contact && stance_task >= 0) {
                RollingTaskAudit& task = report_.tasks[stance_task];
                const double dx = contact.position_world[0] -
                    task.touchdown_position[0];
                const double dy = contact.position_world[1] -
                    task.touchdown_position[1];
                task.maximum_post_touchdown_slip = std::max(
                    task.maximum_post_touchdown_slip,
                    std::sqrt(dx * dx + dy * dy));
            }
        }
        if (active_swings > 1) {
            report_.non_designated_airborne_samples += active_swings - 1;
            ++report_.task_order_violations;
        }
    }

    void update_route_metrics(const Vec<kStateDim>& state) {
        report_.base_progress =
            state[StateIndex::base_position(0)] -
            initial_state_[StateIndex::base_position(0)];
        report_.maximum_base_lateral_deviation = std::max(
            report_.maximum_base_lateral_deviation,
            std::fabs(state[StateIndex::base_position(1)] -
                      initial_state_[StateIndex::base_position(1)]));
        double roll = 0.0;
        double pitch = 0.0;
        rolling_audit_detail::quaternion_roll_pitch(state, roll, pitch);
        report_.maximum_abs_roll = std::max(
            report_.maximum_abs_roll, std::fabs(roll));
        report_.maximum_abs_pitch = std::max(
            report_.maximum_abs_pitch, std::fabs(pitch));
    }

    Terrain terrain_;
    Vec<kStateDim> initial_state_;
    RollingAuditThresholds thresholds_;
    RollingAuditReport report_;
    RollingContactFrame last_contact_frame_;
    std::array<bool, kNumFeet> previous_contact_;
    std::array<int, kNumFeet> active_task_by_foot_;
    std::array<int, kNumFeet> stance_task_by_foot_;
    std::array<Vec<3>, kNumFeet> last_contact_position_;
    int next_task_index_ = 0;
    double last_time_ = -1.0;
};

}  // namespace quadruped_cito
