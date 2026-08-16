#pragma once

#include <algorithm>

#include "examples/quadruped_cito/quadruped_cito_plan.hpp"

namespace quadruped_cito {

struct ContactExecutionFeedbackParameters {
    int early_contact_confirmation_ticks = 3;
    int early_contact_loss_grace_ticks = 5;
    double contact_blend_force = 20.0;
    double late_touchdown_search_speed = 0.05;
    double maximum_late_touchdown_search = 0.02;
};

class EarlyContactFeedback {
public:
    bool update(bool planned_contact, bool measured_contact,
                bool swing_clearance_reached,
                const ContactExecutionFeedbackParameters& parameters) {
        if (planned_contact) {
            reset();
            return false;
        }
        if (measured_contact && swing_clearance_reached) {
            loss_ticks_ = 0;
            ++contact_ticks_;
            if (contact_ticks_ >= parameters.early_contact_confirmation_ticks)
                supporting_ = true;
        } else if (supporting_) {
            ++loss_ticks_;
            if (loss_ticks_ >= parameters.early_contact_loss_grace_ticks)
                reset();
        } else {
            contact_ticks_ = 0;
        }
        return supporting_;
    }

private:
    void reset() {
        contact_ticks_ = 0;
        loss_ticks_ = 0;
        supporting_ = false;
    }

    int contact_ticks_ = 0;
    int loss_ticks_ = 0;
    bool supporting_ = false;
};

class StanceContactConfirmation {
public:
    void initialize(bool planned_contact, bool load_bearing_contact) {
        confirmed_ = planned_contact;
        established_ = planned_contact && load_bearing_contact;
        geometry_loss_ticks_ = 0;
    }

    bool update(bool planned_contact, bool geometry_contact,
                bool load_bearing_contact, int loss_grace_ticks) {
        if (!planned_contact) {
            confirmed_ = false;
            established_ = false;
            geometry_loss_ticks_ = 0;
        } else if (load_bearing_contact) {
            confirmed_ = true;
            established_ = true;
            geometry_loss_ticks_ = 0;
        } else if (confirmed_) {
            if (geometry_contact) {
                geometry_loss_ticks_ = 0;
            } else if (++geometry_loss_ticks_ >= loss_grace_ticks) {
                confirmed_ = false;
            }
        } else if (established_ && geometry_contact) {
            confirmed_ = true;
            geometry_loss_ticks_ = 0;
        }
        return confirmed_;
    }

    bool established() const { return established_; }
    bool confirmed() const { return confirmed_; }

private:
    int geometry_loss_ticks_ = 0;
    bool confirmed_ = false;
    bool established_ = false;
};

class ContactTorqueBlendHandoff {
public:
    // After contact confirmation, retain the full planned GRF in the QP but
    // limit only the stance/swing torque blend until measured load reaches the
    // planned normal load.  If it never does, the handoff remains load-aware
    // for the rest of that continuous planned-contact phase.
    void update(bool planned_contact, bool newly_confirmed,
                double measured_normal_force, FootPlanSample& reference) {
        if (!planned_contact) {
            armed_ = false;
            return;
        }
        if (newly_confirmed) armed_ = true;
        if (!armed_) return;

        const double planned_normal_force = reference.normal_force;
        if (measured_normal_force >= planned_normal_force) {
            armed_ = false;
        } else {
            reference.normal_force = std::max(0.0, measured_normal_force);
        }
    }

private:
    bool armed_ = false;
};

class ConsecutiveGateConfirmation {
public:
    explicit ConsecutiveGateConfirmation(int required_ticks)
        : required_ticks_(required_ticks) {}

    bool update(bool condition) {
        consecutive_ticks_ = condition ? consecutive_ticks_ + 1 : 0;
        return required_ticks_ > 0 &&
            consecutive_ticks_ >= required_ticks_;
    }

private:
    int required_ticks_ = 0;
    int consecutive_ticks_ = 0;
};

inline void apply_early_contact_support(
    const Vec<3>& measured_position,
    const ContactExecutionFeedbackParameters& parameters,
    FootPlanSample& reference) {
    reference.position_world = measured_position;
    reference.velocity_world.zero();
    reference.force_world.zero();
    reference.normal_force = parameters.contact_blend_force;
    reference.terrain_gap = 0.0;
    reference.planned_contact = true;
}

inline void apply_late_touchdown_search(
    int search_ticks, double timestep,
    const ContactExecutionFeedbackParameters& parameters,
    FootPlanSample& reference) {
    const double search_depth = std::min(
        parameters.maximum_late_touchdown_search,
        parameters.late_touchdown_search_speed * timestep * search_ticks);
    reference.position_world[2] -= search_depth;
    reference.velocity_world[2] = -parameters.late_touchdown_search_speed;
    reference.force_world.zero();
    reference.normal_force = 0.0;
    reference.planned_contact = false;
}

}  // namespace quadruped_cito
