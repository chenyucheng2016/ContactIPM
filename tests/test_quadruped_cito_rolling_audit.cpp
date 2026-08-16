#include <cmath>
#include <cstdio>
#include <vector>

#include "examples/quadruped_cito/quadruped_cito_rolling_audit.hpp"

namespace {

using namespace quadruped_cito;

enum class Fault {
    NONE,
    NON_DESIGNATED_LIFTOFF,
    DESIGNATED_CHATTER,
    FAST_SHORT_TOUCHDOWN
};

Vec<kStateDim> standing_state() {
    Vec<kStateDim> state;
    state.zero();
    state[StateIndex::quaternion(0)] = 1.0;
    state[StateIndex::base_position(2)] = 0.25;
    const double x[kNumFeet] = {0.20, 0.20, -0.20, -0.20};
    const double y[kNumFeet] = {0.13, -0.13, 0.13, -0.13};
    for (int foot = 0; foot < kNumFeet; ++foot) {
        state[StateIndex::foot_position(foot, 0)] = x[foot];
        state[StateIndex::foot_position(foot, 1)] = y[foot];
    }
    return state;
}

std::vector<RollingContactTask> two_task_ledger(
    const Vec<kStateDim>& initial) {
    std::vector<RollingContactTask> tasks(2);
    tasks[0].id = 0;
    tasks[0].ordinal = 0;
    tasks[0].moving_foot = 2;
    tasks[0].start_knot = 2;
    tasks[0].touchdown_knot = 5;
    tasks[0].origin_x = initial[StateIndex::foot_position(2, 0)];
    tasks[0].target_x = tasks[0].origin_x + 0.08;
    tasks[1].id = 1;
    tasks[1].ordinal = 1;
    tasks[1].moving_foot = 3;
    tasks[1].start_knot = 8;
    tasks[1].touchdown_knot = 11;
    tasks[1].origin_x = initial[StateIndex::foot_position(3, 0)];
    tasks[1].target_x = tasks[1].origin_x + 0.08;
    return tasks;
}

RollingAuditThresholds two_task_thresholds() {
    RollingAuditThresholds thresholds;
    thresholds.required_task_count = 2;
    thresholds.minimum_foot_progress = 0.0;
    thresholds.minimum_base_progress = 0.35;
    thresholds.schedule_tolerance_knots = 0;
    return thresholds;
}

void set_swing_sample(Vec<kStateDim>& state, Vec<kControlDim>& control,
                      const RollingContactTask& task, int knot,
                      Fault fault) {
    const int foot = task.moving_foot;
    double x = task.origin_x;
    double z = 0.0;
    if (knot >= task.touchdown_knot) {
        x = task.target_x;
    } else if (knot >= task.start_knot) {
        const int swing_knot = knot - task.start_knot;
        if (swing_knot == 1) {
            x = task.origin_x + 0.04;
            z = 0.03;
        } else if (swing_knot >= 2) {
            x = task.target_x;
            z = 0.03;
        }
        control[ControlIndex::contact_force(foot, 2)] = 0.0;
    }
    if (fault == Fault::DESIGNATED_CHATTER && task.id == 0) {
        if (knot == 4) {
            x = task.target_x;
            z = 0.0;
            control[ControlIndex::contact_force(foot, 2)] = 50.0;
        } else if (knot == 5) {
            x = task.target_x;
            z = 0.03;
            control[ControlIndex::contact_force(foot, 2)] = 0.0;
        } else if (knot == 6) {
            x = task.target_x;
            z = 0.0;
            control[ControlIndex::contact_force(foot, 2)] = 50.0;
        }
    }
    if (fault == Fault::FAST_SHORT_TOUCHDOWN && task.id == 0 &&
        knot == task.touchdown_knot) {
        x = task.target_x - 0.02;
        control[ControlIndex::foot_velocity(foot, 0)] = 0.02;
    }
    state[StateIndex::foot_position(foot, 0)] = x;
    state[StateIndex::foot_position(foot, 2)] = z;
}

RollingAuditReport run_synthetic(Fault fault) {
    const PlaneTerrain terrain;
    const Vec<kStateDim> initial = standing_state();
    const std::vector<RollingContactTask> tasks = two_task_ledger(initial);
    RollingSRBDAudit<PlaneTerrain> audit(
        terrain, initial, tasks, two_task_thresholds());
    for (int knot = 0; knot <= 14; ++knot) {
        RollingExecutedSample sample;
        sample.absolute_knot = knot;
        sample.time = 0.05 * knot;
        sample.state = initial;
        sample.state[StateIndex::base_position(0)] = 0.40 * knot / 14.0;
        sample.control.zero();
        for (int foot = 0; foot < kNumFeet; ++foot)
            sample.control[ControlIndex::contact_force(foot, 2)] = 50.0;
        for (const RollingContactTask& task : tasks)
            set_swing_sample(sample.state, sample.control, task, knot, fault);
        if (fault == Fault::NON_DESIGNATED_LIFTOFF && knot == 3)
            sample.control[ControlIndex::contact_force(0, 2)] = 0.0;
        if (!audit.observe(sample)) {
            std::printf("synthetic observation %d was rejected\n", knot);
            return audit.finalize();
        }
    }
    return audit.finalize();
}

int test_contact_classification_requires_gap_and_force() {
    const PlaneTerrain terrain;
    RollingAuditThresholds thresholds;
    RollingExecutedSample sample;
    sample.absolute_knot = 0;
    sample.time = 0.0;
    sample.state = standing_state();
    sample.control.zero();
    for (int foot = 0; foot < kNumFeet; ++foot)
        sample.control[ControlIndex::contact_force(foot, 2)] = 50.0;
    sample.state[StateIndex::foot_position(0, 2)] = 0.003;
    sample.control[ControlIndex::contact_force(1, 2)] = 4.0;
    sample.state[StateIndex::foot_position(3, 2)] = -0.003;
    RollingContactFrame frame;
    if (!classify_rolling_contacts(terrain, sample, thresholds, frame) ||
        frame.feet[0].contact || frame.feet[1].contact ||
        !frame.feet[2].contact || frame.feet[3].contact) {
        std::printf("executed-contact AND classification failed\n");
        return 1;
    }
    return 0;
}

int test_nominal_multistep_audit() {
    const RollingAuditReport report = run_synthetic(Fault::NONE);
    if (!report.pass || report.completed_tasks != 2 ||
        !report.final_support || report.tasks.size() != 2 ||
        !report.tasks[0].pass || !report.tasks[1].pass ||
        std::fabs(report.tasks[0].displacement - 0.08) > 1e-12 ||
        std::fabs(report.tasks[1].maximum_clearance - 0.03) > 1e-12) {
        std::printf(
            "nominal rolling audit failed: pass=%d completed=%d support=%d "
            "unexpected=%d/%d airborne=%d schedule=%d order=%d chatter=%d\n",
            report.pass ? 1 : 0, report.completed_tasks,
            report.final_support ? 1 : 0, report.unexpected_liftoffs,
            report.unexpected_touchdowns,
            report.non_designated_airborne_samples,
            report.schedule_violation_count, report.task_order_violations,
            report.contact_chatter_events);
        return 1;
    }
    return 0;
}

int test_topology_failures_are_rejected() {
    const RollingAuditReport non_designated =
        run_synthetic(Fault::NON_DESIGNATED_LIFTOFF);
    if (non_designated.pass || non_designated.unexpected_liftoffs == 0 ||
        non_designated.non_designated_airborne_samples == 0) {
        std::printf("non-designated liftoff escaped rolling audit\n");
        return 1;
    }
    const RollingAuditReport chatter =
        run_synthetic(Fault::DESIGNATED_CHATTER);
    if (chatter.pass || chatter.contact_chatter_events == 0 ||
        chatter.tasks[0].liftoff_count < 2 ||
        chatter.tasks[0].chatter_count == 0) {
        std::printf("designated-foot chatter escaped rolling audit\n");
        return 1;
    }
    return 0;
}

int test_touchdown_quality_is_rejected() {
    const RollingAuditReport report =
        run_synthetic(Fault::FAST_SHORT_TOUCHDOWN);
    if (report.pass || report.tasks[0].pass ||
        report.tasks[0].touchdown_speed <= 0.01 ||
        std::fabs(report.tasks[0].displacement - 0.06) > 1e-12) {
        std::printf("bad touchdown escaped rolling audit\n");
        return 1;
    }
    return 0;
}

int test_ledger_and_sample_integrity() {
    const PlaneTerrain terrain;
    const Vec<kStateDim> initial = standing_state();
    std::vector<RollingContactTask> tasks = two_task_ledger(initial);
    tasks[1].moving_foot = 0;
    RollingSRBDAudit<PlaneTerrain> bad_ledger(
        terrain, initial, tasks, two_task_thresholds());
    if (bad_ledger.finalize().ledger_valid) {
        std::printf("invalid deterministic ledger was accepted\n");
        return 1;
    }

    tasks = two_task_ledger(initial);
    RollingSRBDAudit<PlaneTerrain> audit(
        terrain, initial, tasks, two_task_thresholds());
    RollingExecutedSample sample;
    sample.absolute_knot = 0;
    sample.time = 0.0;
    sample.state = initial;
    sample.control.zero();
    for (int foot = 0; foot < kNumFeet; ++foot)
        sample.control[ControlIndex::contact_force(foot, 2)] = 50.0;
    if (!audit.observe(sample) || audit.observe(sample) ||
        audit.finalize().input_valid) {
        std::printf("non-monotone executed sample sequence was accepted\n");
        return 1;
    }
    return 0;
}

}  // namespace

int main() {
    int failures = 0;
    failures += test_contact_classification_requires_gap_and_force();
    failures += test_nominal_multistep_audit();
    failures += test_topology_failures_are_rejected();
    failures += test_touchdown_quality_is_rejected();
    failures += test_ledger_and_sample_integrity();
    if (failures == 0)
        std::printf("quadruped rolling SRBD audit tests passed\n");
    return failures == 0 ? 0 : 1;
}
