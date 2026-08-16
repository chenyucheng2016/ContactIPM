#pragma once

#include <algorithm>
#include <array>
#include <chrono>
#include <cmath>
#include <cstdint>
#include <limits>
#include <vector>

namespace quadruped_cito {
namespace realtime_execution_detail {

struct CpuAssignment {
    int wbc_cpu = -1;
    int planner_cpu = -1;
    bool separate_logical_cpus = false;
};

struct CpuTopologyEntry {
    int logical_cpu = -1;
    int package_id = -1;
    int core_id = -1;
};

inline bool are_separate_physical_cores(
    const CpuTopologyEntry& first, const CpuTopologyEntry& second) {
    return first.logical_cpu >= 0 && second.logical_cpu >= 0 &&
        first.package_id >= 0 && second.package_id >= 0 &&
        first.core_id >= 0 && second.core_id >= 0 &&
        (first.package_id != second.package_id ||
         first.core_id != second.core_id);
}

inline CpuAssignment choose_cpu_assignment(std::vector<int> allowed_cpus) {
    std::sort(allowed_cpus.begin(), allowed_cpus.end());
    allowed_cpus.erase(
        std::unique(allowed_cpus.begin(), allowed_cpus.end()),
        allowed_cpus.end());
    CpuAssignment assignment;
    if (allowed_cpus.empty()) return assignment;
    assignment.wbc_cpu = allowed_cpus.back();
    assignment.planner_cpu = allowed_cpus.size() >= 2
        ? allowed_cpus.front()
        : allowed_cpus.back();
    assignment.separate_logical_cpus =
        assignment.wbc_cpu != assignment.planner_cpu;
    return assignment;
}

struct TimingSummary {
    int samples = 0;
    int within_deadline = 0;
    double within_deadline_fraction = 0.0;
    double p50_ms = 0.0;
    double p90_ms = 0.0;
    double p99_ms = 0.0;
    double maximum_ms = 0.0;
};

struct PhaseTimingSample {
    double wall_ms = 0.0;
    double thread_cpu_ms = std::numeric_limits<double>::quiet_NaN();
};

struct PublicationTiming {
    double request_to_handoff_ms = 0.0;
    bool stale = false;
};

struct PlannerRequestPeriodTracker {
    int elapsed_periods = 1;
    int enqueued_periods = 0;

    void record_enqueue_result(bool enqueued) {
        if (enqueued) {
            enqueued_periods = elapsed_periods;
            elapsed_periods = 1;
        } else if (elapsed_periods < std::numeric_limits<int>::max()) {
            ++elapsed_periods;
        }
    }

    void record_shift_result(bool shift_consumed) {
        if (!shift_consumed && enqueued_periods > 0) {
            const int available = std::numeric_limits<int>::max() -
                elapsed_periods;
            elapsed_periods += std::min(enqueued_periods, available);
        }
        enqueued_periods = 0;
    }
};

inline PublicationTiming evaluate_publication_timing(
    std::chrono::steady_clock::time_point requested_at,
    std::chrono::steady_clock::time_point deadline,
    std::chrono::steady_clock::time_point handed_off_at,
    bool already_stale) {
    PublicationTiming timing;
    timing.request_to_handoff_ms =
        std::chrono::duration<double, std::milli>(
            handed_off_at - requested_at).count();
    timing.stale = already_stale || handed_off_at > deadline;
    return timing;
}

inline bool paper_grade_scheduling_gate(
    bool physical_core_isolation_verified,
    bool realtime_priorities_applied,
    bool environment_limited) {
    return physical_core_isolation_verified &&
        realtime_priorities_applied && !environment_limited;
}

inline bool sustained_contact_execution_gate(
    int commanded_tasks, int completed_tasks) {
    return commanded_tasks >= 2 && completed_tasks >= commanded_tasks;
}

inline bool sustained_contact_qualification_gate(
    double duration_seconds, int commanded_tasks, int completed_tasks,
    int completed_gait_cycles, double route_progress_m) {
    return duration_seconds >= 20.0 && commanded_tasks >= 12 &&
        completed_tasks == commanded_tasks && completed_gait_cycles >= 3 &&
        route_progress_m >= 0.18;
}

inline bool task_publication_metadata_gate(
    std::uint64_t active_task_id, std::uint64_t candidate_task_id,
    bool paired_metadata_valid, bool candidate_transition_witness) {
    if (!paired_metadata_valid) return false;
    if (candidate_task_id == active_task_id) return true;
    return active_task_id < std::numeric_limits<std::uint64_t>::max() &&
        candidate_task_id == active_task_id + 1 &&
        candidate_transition_witness;
}

enum class ExecutionPhase {
    MJ_STEP1,
    STATE_REFERENCE_CONTACT,
    WBC_APPLY,
    MJ_STEP2,
    HOUSEKEEPING,
    COUNT
};

constexpr std::size_t kExecutionPhaseCount =
    static_cast<std::size_t>(ExecutionPhase::COUNT);
using ExecutionPhaseTiming =
    std::array<PhaseTimingSample, kExecutionPhaseCount>;

enum class DeadlineMissCause {
    NONE,
    MUJOCO_PHYSICS,
    WBC_OR_CONTROL_WORK,
    SCHEDULING_OR_BLOCKING,
    THREAD_CPU_TIME_UNAVAILABLE,
    COUNT
};

constexpr std::size_t kDeadlineMissCauseCount =
    static_cast<std::size_t>(DeadlineMissCause::COUNT);

inline bool has_thread_cpu_time(const PhaseTimingSample& sample) {
    return std::isfinite(sample.thread_cpu_ms) &&
        sample.thread_cpu_ms >= 0.0;
}

inline DeadlineMissCause classify_deadline_miss(
    const ExecutionPhaseTiming& phases, double complete_tick_wall_ms,
    double response_wall_ms, double deadline_ms) {
    if (response_wall_ms <= deadline_ms)
        return DeadlineMissCause::NONE;
    if (complete_tick_wall_ms <= deadline_ms)
        return DeadlineMissCause::SCHEDULING_OR_BLOCKING;

    double total_cpu_ms = 0.0;
    for (const PhaseTimingSample& phase : phases) {
        if (!has_thread_cpu_time(phase))
            return DeadlineMissCause::THREAD_CPU_TIME_UNAVAILABLE;
        total_cpu_ms += phase.thread_cpu_ms;
    }
    if (total_cpu_ms <= deadline_ms)
        return DeadlineMissCause::SCHEDULING_OR_BLOCKING;

    const auto index = [](ExecutionPhase phase) {
        return static_cast<std::size_t>(phase);
    };
    const double physics_cpu_ms =
        phases[index(ExecutionPhase::MJ_STEP1)].thread_cpu_ms +
        phases[index(ExecutionPhase::MJ_STEP2)].thread_cpu_ms;
    return physics_cpu_ms > total_cpu_ms - physics_cpu_ms
        ? DeadlineMissCause::MUJOCO_PHYSICS
        : DeadlineMissCause::WBC_OR_CONTROL_WORK;
}

inline double percentile_ms(const std::vector<double>& sorted_samples,
                            double percentile) {
    if (sorted_samples.empty()) return 0.0;
    const double rank = percentile *
        static_cast<double>(sorted_samples.size() - 1);
    const std::size_t lower = static_cast<std::size_t>(std::floor(rank));
    const std::size_t upper = static_cast<std::size_t>(std::ceil(rank));
    const double fraction = rank - static_cast<double>(lower);
    return sorted_samples[lower] * (1.0 - fraction) +
        sorted_samples[upper] * fraction;
}

inline TimingSummary summarize_timing_ms(std::vector<double> samples,
                                         double deadline_ms) {
    TimingSummary summary;
    summary.samples = static_cast<int>(samples.size());
    for (double sample : samples) {
        if (sample <= deadline_ms) ++summary.within_deadline;
    }
    if (summary.samples > 0) {
        summary.within_deadline_fraction =
            static_cast<double>(summary.within_deadline) / summary.samples;
    }
    std::sort(samples.begin(), samples.end());
    summary.p50_ms = percentile_ms(samples, 0.50);
    summary.p90_ms = percentile_ms(samples, 0.90);
    summary.p99_ms = percentile_ms(samples, 0.99);
    summary.maximum_ms = percentile_ms(samples, 1.00);
    return summary;
}

inline bool meets_deadline_fraction(const TimingSummary& summary,
                                    double minimum_fraction) {
    return summary.samples > 0 &&
        summary.within_deadline_fraction + 1e-15 >= minimum_fraction;
}

inline bool meets_rate_target(double actual_rate_hz, double target_rate_hz,
                              double relative_tolerance) {
    return target_rate_hz > 0.0 && relative_tolerance >= 0.0 &&
        actual_rate_hz >= target_rate_hz * (1.0 - relative_tolerance) &&
        actual_rate_hz <= target_rate_hz * (1.0 + relative_tolerance);
}

}  // namespace realtime_execution_detail
}  // namespace quadruped_cito
