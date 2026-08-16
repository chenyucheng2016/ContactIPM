#pragma once

#include <algorithm>
#include <cmath>
#include <vector>

#include "examples/quadruped_cito/quadruped_cito_plan.hpp"

namespace quadruped_cito {

struct SwingReplayEvent {
    double planned_liftoff_time = -1.0;
    double measured_liftoff_time = -1.0;
    double planned_touchdown_time = -1.0;
    double measured_touchdown_time = -1.0;
    double maximum_clearance = 0.0;
    double maximum_post_touchdown_slip = 0.0;
    Vec<3> planned_touchdown_position;
    Vec<3> measured_touchdown_position;

    bool measured_unload() const { return measured_liftoff_time >= 0.0; }

    double landing_error() const {
        if (planned_touchdown_time < 0.0 || measured_touchdown_time < 0.0)
            return 1e300;
        double error_sq = 0.0;
        for (int axis = 0; axis < 3; ++axis) {
            const double error = measured_touchdown_position[axis] -
                                 planned_touchdown_position[axis];
            error_sq += error * error;
        }
        return std::sqrt(error_sq);
    }

    double touchdown_error() const {
        return planned_touchdown_time >= 0.0 &&
                       measured_touchdown_time >= 0.0
                   ? std::fabs(measured_touchdown_time -
                               planned_touchdown_time)
                   : 1e300;
    }

    double signed_touchdown_error() const {
        return planned_touchdown_time >= 0.0 &&
                       measured_touchdown_time >= 0.0
                   ? measured_touchdown_time - planned_touchdown_time
                   : 1e300;
    }
};

struct FootReplayMetrics {
    std::vector<SwingReplayEvent> events;
    bool previous_planned_contact = true;
    bool previous_measured_contact = true;
    int active_event = -1;
    int slip_event = -1;

    void initialize(bool planned_contact, bool measured_contact) {
        previous_planned_contact = planned_contact;
        previous_measured_contact = measured_contact;
    }

    void observe(double plan_time, double measured_sample_time,
                 bool planned_contact, bool measured_contact,
                 const Vec<3>& planned_position,
                 const Vec<3>& measured_sample_position, double clearance,
                 double minimum_touchdown_clearance = 0.0) {
        if (previous_planned_contact && !planned_contact) {
            events.emplace_back();
            active_event = static_cast<int>(events.size()) - 1;
            slip_event = -1;
            events.back().planned_liftoff_time = plan_time;
        }

        if (active_event >= 0) {
            SwingReplayEvent& event = events[active_event];
            if (!planned_contact) {
                event.maximum_clearance = std::max(
                    event.maximum_clearance, clearance);
            }
            if (previous_measured_contact && !measured_contact &&
                event.measured_liftoff_time < 0.0) {
                event.measured_liftoff_time = measured_sample_time;
            }
            if (!previous_measured_contact && measured_contact &&
                event.measured_touchdown_time < 0.0 &&
                event.maximum_clearance >= minimum_touchdown_clearance) {
                event.measured_touchdown_time = measured_sample_time;
                event.measured_touchdown_position = measured_sample_position;
                slip_event = active_event;
            }
            if (!previous_planned_contact && planned_contact) {
                event.planned_touchdown_time = plan_time;
                event.planned_touchdown_position = planned_position;
            }
        }

        if (slip_event >= 0 && planned_contact) {
            SwingReplayEvent& event = events[slip_event];
            const double dx = measured_sample_position[0] -
                              event.measured_touchdown_position[0];
            const double dy = measured_sample_position[1] -
                              event.measured_touchdown_position[1];
            event.maximum_post_touchdown_slip = std::max(
                event.maximum_post_touchdown_slip,
                std::sqrt(dx * dx + dy * dy));
        }

        previous_planned_contact = planned_contact;
        previous_measured_contact = measured_contact;
    }
};

}  // namespace quadruped_cito
