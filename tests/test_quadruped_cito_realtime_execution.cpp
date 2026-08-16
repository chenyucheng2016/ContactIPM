#include <chrono>
#include <cmath>
#include <cstdio>
#include <vector>

#include "examples/quadruped_cito/mujoco/quadruped_cito_realtime_execution.hpp"

namespace {

using quadruped_cito::realtime_execution_detail::CpuAssignment;
using quadruped_cito::realtime_execution_detail::CpuTopologyEntry;
using quadruped_cito::realtime_execution_detail::DeadlineMissCause;
using quadruped_cito::realtime_execution_detail::ExecutionPhaseTiming;
using quadruped_cito::realtime_execution_detail::PhaseTimingSample;
using quadruped_cito::realtime_execution_detail::PlannerRequestPeriodTracker;
using quadruped_cito::realtime_execution_detail::PublicationTiming;
using quadruped_cito::realtime_execution_detail::TimingSummary;
using quadruped_cito::realtime_execution_detail::choose_cpu_assignment;
using quadruped_cito::realtime_execution_detail::are_separate_physical_cores;
using quadruped_cito::realtime_execution_detail::classify_deadline_miss;
using quadruped_cito::realtime_execution_detail::evaluate_publication_timing;
using quadruped_cito::realtime_execution_detail::meets_deadline_fraction;
using quadruped_cito::realtime_execution_detail::meets_rate_target;
using quadruped_cito::realtime_execution_detail::paper_grade_scheduling_gate;
using quadruped_cito::realtime_execution_detail::sustained_contact_execution_gate;
using quadruped_cito::realtime_execution_detail::
    sustained_contact_qualification_gate;
using quadruped_cito::realtime_execution_detail::summarize_timing_ms;
using quadruped_cito::realtime_execution_detail::
    task_publication_metadata_gate;

bool near(double first, double second) {
    return std::fabs(first - second) <= 1e-12;
}

PhaseTimingSample phase(double wall_ms, double thread_cpu_ms) {
    PhaseTimingSample sample;
    sample.wall_ms = wall_ms;
    sample.thread_cpu_ms = thread_cpu_ms;
    return sample;
}

}  // namespace

int main() {
    const CpuAssignment none = choose_cpu_assignment({});
    const CpuAssignment one = choose_cpu_assignment({7});
    const CpuAssignment two = choose_cpu_assignment({5, 2, 5});
    const CpuAssignment many = choose_cpu_assignment({7, 3, 9, 1, 5});
    if (none.wbc_cpu != -1 || none.planner_cpu != -1 ||
        none.separate_logical_cpus ||
        one.wbc_cpu != 7 || one.planner_cpu != 7 ||
        one.separate_logical_cpus ||
        two.wbc_cpu != 5 || two.planner_cpu != 2 ||
        !two.separate_logical_cpus || many.wbc_cpu != 9 ||
        many.planner_cpu != 1 || !many.separate_logical_cpus) {
        std::printf("CPU assignment invariant failed\n");
        return 1;
    }
    if (!are_separate_physical_cores(
            CpuTopologyEntry{0, 0, 0}, CpuTopologyEntry{2, 0, 1}) ||
        are_separate_physical_cores(
            CpuTopologyEntry{0, 0, 0}, CpuTopologyEntry{1, 0, 0}) ||
        are_separate_physical_cores(
            CpuTopologyEntry{}, CpuTopologyEntry{1, 0, 1})) {
        std::printf("physical CPU topology invariant failed\n");
        return 1;
    }

    const TimingSummary known = summarize_timing_ms(
        {0.5, 1.0, 1.5, 2.0, 2.5}, 2.0);
    if (known.samples != 5 || known.within_deadline != 4 ||
        !near(known.within_deadline_fraction, 0.8) ||
        !near(known.p50_ms, 1.5) || !near(known.p90_ms, 2.3) ||
        !near(known.p99_ms, 2.48) || !near(known.maximum_ms, 2.5)) {
        std::printf("timing summary invariant failed\n");
        return 1;
    }

    std::vector<double> one_miss(1000, 1.0);
    one_miss.back() = 2.1;
    const TimingSummary passing = summarize_timing_ms(one_miss, 2.0);
    one_miss[998] = 2.1;
    const TimingSummary failing = summarize_timing_ms(one_miss, 2.0);
    if (!meets_deadline_fraction(passing, 0.999) ||
        meets_deadline_fraction(failing, 0.999)) {
        std::printf("99.9%% timing gate invariant failed\n");
        return 1;
    }

    std::vector<double> fast_work(1000, 1.0);
    std::vector<double> late_response = fast_work;
    late_response[998] = 2.1;
    late_response[999] = 2.1;
    if (!meets_deadline_fraction(
            summarize_timing_ms(fast_work, 2.0), 0.999) ||
        meets_deadline_fraction(
            summarize_timing_ms(late_response, 2.0), 0.999)) {
        std::printf("scheduled-response late-wakeup gate invariant failed\n");
        return 1;
    }

    if (!meets_rate_target(495.0, 500.0, 0.01) ||
        !meets_rate_target(505.0, 500.0, 0.01) ||
        meets_rate_target(494.99, 500.0, 0.01) ||
        meets_rate_target(505.01, 500.0, 0.01) ||
        meets_rate_target(500.0, 0.0, 0.01)) {
        std::printf("wall-rate gate invariant failed\n");
        return 1;
    }

    const TimingSummary empty = summarize_timing_ms({}, 2.0);
    if (empty.samples != 0 || meets_deadline_fraction(empty, 0.999)) {
        std::printf("empty timing summary invariant failed\n");
        return 1;
    }

    using Clock = std::chrono::steady_clock;
    const Clock::time_point requested{};
    const auto deadline = requested + std::chrono::milliseconds(200);
    const PublicationTiming on_time_publication = evaluate_publication_timing(
        requested, deadline, requested + std::chrono::milliseconds(200),
        false);
    const PublicationTiming late_publication = evaluate_publication_timing(
        requested, deadline, requested + std::chrono::microseconds(200001),
        false);
    const PublicationTiming already_stale = evaluate_publication_timing(
        requested, deadline, requested + std::chrono::milliseconds(150),
        true);
    if (!near(on_time_publication.request_to_handoff_ms, 200.0) ||
        on_time_publication.stale ||
        !near(late_publication.request_to_handoff_ms, 200.001) ||
        !late_publication.stale || !already_stale.stale) {
        std::printf("publication timing invariant failed\n");
        return 1;
    }

    PlannerRequestPeriodTracker request_periods;
    if (request_periods.elapsed_periods != 1) {
        std::printf("initial planner period tracking failed\n");
        return 1;
    }
    request_periods.record_enqueue_result(true);
    request_periods.record_shift_result(false);
    if (request_periods.elapsed_periods != 2) {
        std::printf("unconsumed planner shift was not restored\n");
        return 1;
    }
    request_periods.record_enqueue_result(true);
    request_periods.record_enqueue_result(false);
    request_periods.record_shift_result(true);
    if (request_periods.elapsed_periods != 2) {
        std::printf("consumed shift lost a later dropped period\n");
        return 1;
    }
    request_periods.record_enqueue_result(true);
    request_periods.record_shift_result(true);
    if (request_periods.elapsed_periods != 1) {
        std::printf("consumed planner request did not reset period count\n");
        return 1;
    }

    if (!paper_grade_scheduling_gate(true, true, false) ||
        paper_grade_scheduling_gate(false, true, false) ||
        paper_grade_scheduling_gate(true, false, false) ||
        paper_grade_scheduling_gate(true, true, true)) {
        std::printf("paper-grade scheduling gate invariant failed\n");
        return 1;
    }
    if (sustained_contact_execution_gate(1, 1) ||
        sustained_contact_execution_gate(2, 1) ||
        !sustained_contact_execution_gate(2, 2)) {
        std::printf("sustained contact evidence gate failed\n");
        return 1;
    }
    if (!sustained_contact_qualification_gate(20.0, 12, 12, 3, 0.18) ||
        sustained_contact_qualification_gate(19.999, 12, 12, 3, 0.18) ||
        sustained_contact_qualification_gate(20.0, 11, 11, 3, 0.18) ||
        sustained_contact_qualification_gate(20.0, 13, 12, 3, 0.18) ||
        sustained_contact_qualification_gate(20.0, 12, 12, 2, 0.18) ||
        sustained_contact_qualification_gate(20.0, 12, 12, 3, 0.179)) {
        std::printf("sustained contact qualification gate failed\n");
        return 1;
    }
    if (!task_publication_metadata_gate(7, 7, true, false) ||
        !task_publication_metadata_gate(7, 8, true, true) ||
        task_publication_metadata_gate(7, 8, true, false) ||
        task_publication_metadata_gate(7, 6, true, true) ||
        task_publication_metadata_gate(7, 9, true, true) ||
        task_publication_metadata_gate(7, 7, false, true)) {
        std::printf("task publication metadata gate failed\n");
        return 1;
    }

    const ExecutionPhaseTiming on_time = {{
        phase(0.2, 0.2), phase(0.3, 0.3), phase(0.4, 0.4),
        phase(0.2, 0.2), phase(0.1, 0.1)}};
    const ExecutionPhaseTiming scheduling = {{
        phase(0.2, 0.1), phase(0.4, 0.2), phase(0.4, 0.2),
        phase(0.3, 0.1), phase(0.2, 0.1)}};
    const ExecutionPhaseTiming physics = {{
        phase(0.8, 0.7), phase(0.2, 0.2), phase(0.2, 0.2),
        phase(1.5, 1.4), phase(0.1, 0.1)}};
    const ExecutionPhaseTiming control = {{
        phase(0.2, 0.2), phase(0.2, 0.2), phase(2.2, 2.1),
        phase(0.2, 0.2), phase(0.1, 0.1)}};
    ExecutionPhaseTiming unavailable = physics;
    unavailable[3] = PhaseTimingSample{1.5};
    if (classify_deadline_miss(on_time, 1.2, 1.2, 2.0) !=
            DeadlineMissCause::NONE ||
        classify_deadline_miss(scheduling, 2.5, 2.5, 2.0) !=
            DeadlineMissCause::SCHEDULING_OR_BLOCKING ||
        classify_deadline_miss(physics, 2.8, 2.8, 2.0) !=
            DeadlineMissCause::MUJOCO_PHYSICS ||
        classify_deadline_miss(control, 2.9, 2.9, 2.0) !=
            DeadlineMissCause::WBC_OR_CONTROL_WORK ||
        classify_deadline_miss(on_time, 1.2, 2.1, 2.0) !=
            DeadlineMissCause::SCHEDULING_OR_BLOCKING ||
        classify_deadline_miss(unavailable, 2.8, 2.8, 2.0) !=
            DeadlineMissCause::THREAD_CPU_TIME_UNAVAILABLE) {
        std::printf("deadline attribution invariant failed\n");
        return 1;
    }
    std::printf("Realtime execution timing invariants passed.\n");
    return 0;
}
