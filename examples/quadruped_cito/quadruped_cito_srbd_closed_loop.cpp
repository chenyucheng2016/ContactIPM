#include <algorithm>
#include <chrono>
#include <cmath>
#include <cstdint>
#include <cstdio>
#include <cstdlib>
#include <filesystem>
#include <fstream>
#include <iomanip>
#include <limits>
#include <memory>
#include <sstream>
#include <string>
#include <utility>
#include <vector>

#include "examples/quadruped_cito/quadruped_cito_realtime_planner.hpp"
#include "examples/quadruped_cito/quadruped_cito_rolling_audit.hpp"

namespace {

using namespace quadruped_cito;

constexpr int kContactTaskCap = 24;
constexpr int kDefaultWarmUpdates = 200;
constexpr int kReplanRateHz = 5;

struct Options {
    std::string output_directory;
    int updates = kDefaultWarmUpdates;
    double terrain_amplitude = 0.02;
    double wave_number_x = 4.0;
    double wave_number_y = 3.0;
    // Zero means correctness mode with no wall-clock publication deadline.
    double deadline_ms = 0.0;
};

bool parse_int(const char* text, int& value) {
    char* end = nullptr;
    const long parsed = std::strtol(text, &end, 10);
    if (end == text || *end != '\0') return false;
    value = static_cast<int>(parsed);
    return static_cast<long>(value) == parsed;
}

bool parse_double(const char* text, double& value) {
    char* end = nullptr;
    const double parsed = std::strtod(text, &end);
    if (end == text || *end != '\0' || !std::isfinite(parsed)) return false;
    value = parsed;
    return true;
}

bool parse_options(int argc, char** argv, Options& options, bool& help) {
    help = false;
    for (int argument = 1; argument < argc; ++argument) {
        const std::string name(argv[argument]);
        if (name == "--help" || name == "-h") {
            help = true;
            return true;
        }
        if (argument + 1 >= argc) return false;
        const char* value = argv[++argument];
        if (name == "--output-dir") {
            options.output_directory = value;
        } else if (name == "--updates") {
            if (!parse_int(value, options.updates)) return false;
        } else if (name == "--terrain-amplitude") {
            if (!parse_double(value, options.terrain_amplitude)) return false;
        } else if (name == "--wave-number-x") {
            if (!parse_double(value, options.wave_number_x)) return false;
        } else if (name == "--wave-number-y") {
            if (!parse_double(value, options.wave_number_y)) return false;
        } else if (name == "--deadline-ms") {
            if (!parse_double(value, options.deadline_ms)) return false;
        } else {
            return false;
        }
    }
    return !options.output_directory.empty() && options.updates > 0 &&
           options.terrain_amplitude >= 0.0 &&
           options.wave_number_x > 0.0 && options.wave_number_y > 0.0 &&
           options.deadline_ms >= 0.0;
}

void print_usage() {
    std::printf(
        "usage: quadruped_cito_srbd_closed_loop --output-dir PATH "
        "[--updates N] [--terrain-amplitude M] [--wave-number-x KX] "
        "[--wave-number-y KY] [--deadline-ms MS]\n"
        "\n"
        "Defaults implement the frozen correctness experiment: N=50, "
        "dt=0.05 s, 5 Hz replanning, 24 contact tasks, 200 warm updates, "
        "and no wall-clock deadline. --updates is a development override; "
        "--deadline-ms 200 enables the later timed phase.\n");
}

struct ContactFields {
    double terrain_gap[kNumFeet]{};
    double normal_force[kNumFeet]{};
    double foot_speed[kNumFeet]{};
    double terrain_normal[kNumFeet][3]{};
    int contact[kNumFeet]{};
};

ContactFields contact_fields(
    const Vec<kStateDim>& state, const Vec<kControlDim>* control,
    const Go1FootCenterTerrain<SharedTerrain>& terrain) {
    ContactFields fields;
    for (int foot = 0; foot < kNumFeet; ++foot) {
        const Vec<3> position = state_vector3(
            state, StateIndex::foot_position(foot, 0));
        TerrainSample sample;
        if (!terrain.sample(position, sample)) {
            fields.terrain_gap[foot] =
                std::numeric_limits<double>::quiet_NaN();
            fields.normal_force[foot] =
                std::numeric_limits<double>::quiet_NaN();
            continue;
        }
        fields.terrain_gap[foot] = sample.gap;
        for (int axis = 0; axis < 3; ++axis)
            fields.terrain_normal[foot][axis] = sample.normal[axis];
        if (control == nullptr) continue;
        const Vec<3> force = control_vector3(
            *control, ControlIndex::contact_force(foot, 0));
        const Vec<3> velocity = control_vector3(
            *control, ControlIndex::foot_velocity(foot, 0));
        fields.normal_force[foot] = dot3(sample.normal, force);
        fields.foot_speed[foot] = velocity.norm2();
        fields.contact[foot] =
            sample.gap <= ContactClassification{}.maximum_contact_gap &&
            fields.normal_force[foot] >=
                ContactClassification{}.minimum_contact_force;
    }
    return fields;
}

struct ExecutedRecord {
    int knot = 0;
    double time_s = 0.0;
    int update_index = 0;
    int substep = 0;
    int source_publication_id = -1;
    int source_plan_stage = -1;
    std::uint64_t task_id = 0;
    int moving_foot = -1;
    int has_control = 0;
    nmpc::Status plant_status = nmpc::Status::NOT_INITIALIZED;
    Vec<kStateDim> state;
    Vec<kControlDim> control;
    ContactFields contact;
};

struct AcceptedStageRecord {
    int publication_id = -1;
    std::string source;
    int update_index = -1;
    double handoff_time_s = 0.0;
    std::uint64_t task_id = 0;
    int moving_foot = -1;
    int stage = 0;
    double relative_time_s = 0.0;
    double absolute_time_s = 0.0;
    int has_control = 0;
    Vec<kStateDim> state;
    Vec<kControlDim> control;
    ContactFields contact;
};

struct PlannerRecord {
    int update_index = 0;
    double request_time_s = 0.0;
    double handoff_time_s = 0.0;
    int executed_from_publication_id = -1;
    int executed_stage_begin = -1;
    int executed_stage_end = -1;
    int publication_id = -1;
    int active_publication_id_after = -1;
    std::uint64_t task_id = 0;
    int moving_foot = -1;
    RealtimePlannerResult result;
};

void append_plan_stages(
    const QuadrupedCITORealtimePlanner::ProblemType& problem,
    const Go1FootCenterTerrain<SharedTerrain>& terrain, int publication_id,
    const char* source, int update_index, double handoff_time_s,
    const RealtimeContactTask& task,
    std::vector<AcceptedStageRecord>& output) {
    for (int stage = 0; stage <= kRealtimeHorizon; ++stage) {
        AcceptedStageRecord record;
        record.publication_id = publication_id;
        record.source = source;
        record.update_index = update_index;
        record.handoff_time_s = handoff_time_s;
        record.task_id = task.id;
        record.moving_foot = task.moving_foot;
        record.stage = stage;
        record.relative_time_s = stage * problem.dt;
        record.absolute_time_s = handoff_time_s + record.relative_time_s;
        record.state = problem.stages[stage].x;
        record.control.zero();
        if (stage < kRealtimeHorizon) {
            record.has_control = 1;
            record.control = problem.stages[stage].u;
            record.contact = contact_fields(
                record.state, &record.control, terrain);
        } else {
            record.contact = contact_fields(record.state, nullptr, terrain);
        }
        output.push_back(record);
    }
}

std::string csv_text(const std::string& value) {
    if (value.find_first_of(",\"\n\r") == std::string::npos) return value;
    std::string quoted = "\"";
    for (char character : value) {
        if (character == '\"') quoted += '\"';
        quoted += character;
    }
    quoted += '\"';
    return quoted;
}

void write_state_header(std::ostream& output) {
    for (int index = 0; index < kStateDim; ++index)
        output << ",x" << index;
}

void write_control_header(std::ostream& output) {
    for (int index = 0; index < kControlDim; ++index)
        output << ",u" << index;
}

void write_contact_header(std::ostream& output) {
    for (int foot = 0; foot < kNumFeet; ++foot)
        output << ",terrain_gap_foot" << foot;
    for (int foot = 0; foot < kNumFeet; ++foot)
        output << ",normal_force_foot" << foot;
    for (int foot = 0; foot < kNumFeet; ++foot)
        output << ",foot_speed_foot" << foot;
    for (int foot = 0; foot < kNumFeet; ++foot) {
        for (int axis = 0; axis < 3; ++axis)
            output << ",terrain_normal_foot" << foot << '_' << axis;
    }
    for (int foot = 0; foot < kNumFeet; ++foot)
        output << ",contact_foot" << foot;
}

void write_state(std::ostream& output, const Vec<kStateDim>& state) {
    for (int index = 0; index < kStateDim; ++index)
        output << ',' << state[index];
}

void write_control(std::ostream& output, const Vec<kControlDim>& control,
                   bool has_control) {
    for (int index = 0; index < kControlDim; ++index) {
        output << ',';
        if (has_control) output << control[index];
    }
}

void write_contact(std::ostream& output, const ContactFields& contact) {
    for (double gap : contact.terrain_gap) output << ',' << gap;
    for (double force : contact.normal_force) output << ',' << force;
    for (double speed : contact.foot_speed) output << ',' << speed;
    for (const auto& normal : contact.terrain_normal) {
        for (double component : normal) output << ',' << component;
    }
    for (int active : contact.contact) output << ',' << active;
}

ContactFields contact_fields(const RollingContactFrame& frame) {
    ContactFields fields;
    for (int foot = 0; foot < kNumFeet; ++foot) {
        const RollingFootContact& input = frame.feet[foot];
        fields.terrain_gap[foot] = input.terrain_gap;
        fields.normal_force[foot] = input.normal_force;
        fields.foot_speed[foot] = input.speed;
        fields.contact[foot] = input.contact;
        for (int axis = 0; axis < 3; ++axis)
            fields.terrain_normal[foot][axis] =
                input.terrain_normal_world[axis];
    }
    return fields;
}

bool write_executed_trajectory(
    const std::filesystem::path& path,
    const std::vector<ExecutedRecord>& records) {
    std::ofstream output(path);
    if (!output.is_open()) return false;
    output << std::setprecision(17);
    output << "knot,time_s,update_index,substep,source_publication_id,"
              "source_plan_stage,task_id,moving_foot,has_control,plant_status";
    write_state_header(output);
    write_control_header(output);
    write_contact_header(output);
    output << '\n';
    for (const ExecutedRecord& record : records) {
        output << record.knot << ',' << record.time_s << ','
               << record.update_index << ',' << record.substep << ','
               << record.source_publication_id << ','
               << record.source_plan_stage << ',' << record.task_id << ','
               << record.moving_foot << ',' << record.has_control << ','
               << nmpc::status_string(record.plant_status);
        write_state(output, record.state);
        write_control(output, record.control, record.has_control != 0);
        write_contact(output, record.contact);
        output << '\n';
    }
    return output.good();
}

bool write_accepted_plan_stages(
    const std::filesystem::path& path,
    const std::vector<AcceptedStageRecord>& records) {
    std::ofstream output(path);
    if (!output.is_open()) return false;
    output << std::setprecision(17);
    output << "publication_id,source,update_index,handoff_time_s,task_id,"
              "moving_foot,stage,relative_time_s,absolute_time_s,has_control";
    write_state_header(output);
    write_control_header(output);
    write_contact_header(output);
    output << '\n';
    for (const AcceptedStageRecord& record : records) {
        output << record.publication_id << ',' << record.source << ','
               << record.update_index << ',' << record.handoff_time_s << ','
               << record.task_id << ',' << record.moving_foot << ','
               << record.stage << ',' << record.relative_time_s << ','
               << record.absolute_time_s << ',' << record.has_control;
        write_state(output, record.state);
        write_control(output, record.control, record.has_control != 0);
        write_contact(output, record.contact);
        output << '\n';
    }
    return output.good();
}

bool write_planner_updates(const std::filesystem::path& path,
                           const std::vector<PlannerRecord>& records) {
    std::ofstream output(path);
    if (!output.is_open()) return false;
    output << std::setprecision(17);
    output << "update_index,request_time_s,handoff_time_s,"
              "executed_from_publication_id,executed_stage_begin,"
              "executed_stage_end,status,full_kkt,task_pass,published,"
              "fallback,deadline_miss,shift_consumed,publication_id,"
              "active_publication_id_after,task_id,moving_foot,"
              "task_transition_witness,shift_ms,solver_ms,audit_ms,"
              "plan_extract_ms,end_to_end_ms,primal_infeas,dual_infeas,"
              "complementarity,mpcc_complementarity,barrier_param,"
              "audit_dynamics,audit_inequality,audit_mpcc,"
              "terminal_foot_error,terminal_gap,terminal_speed,"
              "terminal_normal_force,task_displacement,clearance,"
              "moving_unloaded_stages,inner_iterations,"
              "exact_hessian_analytic_calls,exact_hessian_fd_calls\n";
    for (const PlannerRecord& record : records) {
        const RealtimePlannerResult& result = record.result;
        output << record.update_index << ',' << record.request_time_s << ','
               << record.handoff_time_s << ','
               << record.executed_from_publication_id << ','
               << record.executed_stage_begin << ','
               << record.executed_stage_end << ','
               << nmpc::status_string(result.status) << ','
               << result.full_kkt << ',' << result.task_pass << ','
               << result.published << ',' << result.fallback << ','
               << result.deadline_miss << ',' << result.shift_consumed << ','
               << record.publication_id << ','
               << record.active_publication_id_after << ','
               << record.task_id << ',' << record.moving_foot << ','
               << result.task_transition_witness << ',' << result.shift_ms
               << ',' << result.solver_ms << ',' << result.audit_ms << ','
               << result.plan_extract_ms << ',' << result.end_to_end_ms << ','
               << result.stats.primal_infeas << ','
               << result.stats.dual_infeas << ','
               << result.stats.complementarity << ','
               << result.stats.mpcc_complementarity << ','
               << result.stats.barrier_param << ',' << result.audit.dynamics
               << ',' << result.audit.inequality << ',' << result.audit.mpcc
               << ',' << result.audit.terminal_foot_error << ','
               << result.audit.terminal_gap << ','
               << result.audit.terminal_speed << ','
               << result.audit.terminal_normal_force << ','
               << result.audit.task_displacement << ','
               << result.audit.clearance << ','
               << result.audit.moving_unloaded_stages << ','
               << result.stats.inner_iterations << ','
               << result.stats.exact_hessian_analytic_calls << ','
               << result.stats.exact_hessian_fd_calls << '\n';
    }
    return output.good();
}

bool write_key_values(
    const std::filesystem::path& path,
    const std::vector<std::pair<std::string, std::string>>& values) {
    std::ofstream output(path);
    if (!output.is_open()) return false;
    output << "key,value\n";
    for (const auto& value : values)
        output << csv_text(value.first) << ',' << csv_text(value.second) << '\n';
    return output.good();
}

bool write_task_ledger(const std::filesystem::path& path,
                       const RollingAuditReport& report) {
    std::ofstream output(path);
    if (!output.is_open()) return false;
    output << std::setprecision(17);
    output << "task_ordinal,task_id,moving_foot,start_knot,touchdown_knot,"
              "origin_x,target_x,measured_liftoff_time,"
              "measured_touchdown_time,liftoff_count,touchdown_count,"
              "maximum_clearance,touchdown_gap,touchdown_speed,"
              "touchdown_normal_force,displacement,target_error,"
              "maximum_post_touchdown_slip,schedule_violation_count,"
              "chatter_count,pass\n";
    for (const RollingTaskAudit& audit : report.tasks) {
        const RollingContactTask& task = audit.task;
        output << task.ordinal << ',' << task.id << ',' << task.moving_foot
               << ',' << task.start_knot << ',' << task.touchdown_knot << ','
               << task.origin_x << ',' << task.target_x << ','
               << audit.measured_liftoff_time << ','
               << audit.measured_touchdown_time << ',' << audit.liftoff_count
               << ',' << audit.touchdown_count << ','
               << audit.maximum_clearance << ',' << audit.touchdown_gap << ','
               << audit.touchdown_speed << ','
               << audit.touchdown_normal_force << ',' << audit.displacement
               << ',' << audit.target_error << ','
               << audit.maximum_post_touchdown_slip << ','
               << audit.schedule_violation_count << ',' << audit.chatter_count
               << ',' << audit.pass << '\n';
    }
    return output.good();
}

template <typename Value>
std::string as_string(Value value) {
    std::ostringstream output;
    output << std::setprecision(17) << value;
    return output.str();
}

}  // namespace

int main(int argc, char** argv) {
    Options options;
    bool help = false;
    if (!parse_options(argc, argv, options, help)) {
        print_usage();
        return 2;
    }
    if (help) {
        print_usage();
        return 0;
    }

    const SharedTerrain physical_terrain = register_go1_terrain(
        SharedTerrain::sinusoidal(
            options.terrain_amplitude, options.wave_number_x,
            options.wave_number_y));
    const Go1FootCenterTerrain<SharedTerrain> contact_terrain(
        physical_terrain);

    RealtimePlannerConfig configuration;
    configuration.rate_hz = kReplanRateHz;
    configuration.sustained_contact_tasks = true;
    configuration.maximum_contact_tasks = kContactTaskCap;
    QuadrupedCITORealtimePlanner planner(physical_terrain, configuration);

    const Vec<kStateDim> initial_state = planner.nominal_state();
    Vec<kStateDim> plant_state = initial_state;
    SRBDDynamics plant(go1_robot_parameters());
    realtime_detail::RealtimeContactTaskSequence execution_tasks(
        kContactTaskCap);
    execution_tasks.initialize(initial_state);
    std::vector<RollingContactTask> audit_tasks;
    audit_tasks.reserve(kContactTaskCap);
    for (int ordinal = 0; ordinal < kContactTaskCap; ++ordinal) {
        const RealtimeContactTask source = execution_tasks.task(
            static_cast<std::uint64_t>(ordinal));
        RollingContactTask task;
        task.id = source.id;
        task.ordinal = ordinal;
        task.moving_foot = source.moving_foot;
        task.start_knot = source.start_knot;
        task.touchdown_knot = source.touchdown_knot;
        task.origin_x = source.origin_x;
        task.target_x = source.target_x;
        audit_tasks.push_back(task);
    }
    RollingSRBDAudit<Go1FootCenterTerrain<SharedTerrain>> rolling_audit(
        contact_terrain, initial_state, audit_tasks);

    std::vector<ExecutedRecord> executed_records;
    std::vector<AcceptedStageRecord> accepted_stage_records;
    std::vector<PlannerRecord> planner_records;
    executed_records.reserve(
        static_cast<std::size_t>(options.updates * 4 + 1));
    accepted_stage_records.reserve(
        static_cast<std::size_t>((options.updates + 1) *
                                 (kRealtimeHorizon + 1)));
    planner_records.reserve(static_cast<std::size_t>(options.updates));

    bool run_completed = false;
    std::string failure_reason;
    int completed_updates = 0;
    int plant_knots = 0;
    int published_updates = 0;
    int fallback_updates = 0;
    int consecutive_fallbacks = 0;
    int maximum_consecutive_fallbacks = 0;
    int exact_hessian_calls = 0;
    int next_publication_id = 0;
    int active_publication_id = -1;
    int active_plan_stage = 0;
    RealtimePlannerResult cold;

    const nmpc::Status initialize_status = planner.initialize(initial_state);
    if (initialize_status != nmpc::Status::SUCCESS) {
        failure_reason = std::string("planner_initialize_") +
                         nmpc::status_string(initialize_status);
    } else {
        cold = planner.cold_solve();
        exact_hessian_calls += cold.stats.exact_hessian_analytic_calls +
                               cold.stats.exact_hessian_fd_calls;
        if (!cold.published) {
            failure_reason = std::string("cold_solve_") +
                             nmpc::status_string(cold.status);
        }
    }

    std::unique_ptr<QuadrupedCITORealtimePlanner::ProblemType> active_problem;
    if (failure_reason.empty()) {
        active_publication_id = next_publication_id++;
        active_problem = std::make_unique<
            QuadrupedCITORealtimePlanner::ProblemType>(planner.problem());
        const RealtimeContactTask* cold_task = planner.last_valid_task();
        const RealtimeContactTask task = cold_task != nullptr
            ? *cold_task
            : planner.active_task();
        append_plan_stages(
            *active_problem, contact_terrain, active_publication_id, "cold",
            -1, 0.0, task, accepted_stage_records);
    }

    for (int update = 0;
         failure_reason.empty() && update < options.updates; ++update) {
        const int executed_stage_begin = active_plan_stage;
        if (active_plan_stage + planner.shift_steps() > kRealtimeHorizon) {
            failure_reason = "active_plan_expired";
            break;
        }

        for (int substep = 0; substep < planner.shift_steps(); ++substep) {
            const int source_stage = active_plan_stage;
            const Vec<kControlDim> control =
                active_problem->stages[source_stage].u;
            Vec<kStateDim> next_state;
            const nmpc::Status plant_status = plant.discrete_step(
                plant_state, control, kRealtimeTimeStep, next_state);

            const RealtimeContactTask scheduled_task =
                execution_tasks.active_task();
            ExecutedRecord record;
            record.knot = plant_knots;
            record.time_s = plant_knots * kRealtimeTimeStep;
            record.update_index = update;
            record.substep = substep;
            record.source_publication_id = active_publication_id;
            record.source_plan_stage = source_stage;
            record.task_id = scheduled_task.id;
            record.moving_foot = execution_tasks.moving_foot_at(plant_knots);
            record.has_control = 1;
            record.plant_status = plant_status;
            record.state = plant_state;
            record.control = control;
            RollingExecutedSample audit_sample;
            audit_sample.absolute_knot = plant_knots;
            audit_sample.time = record.time_s;
            audit_sample.state = plant_state;
            audit_sample.control = control;
            if (!rolling_audit.observe(audit_sample)) {
                failure_reason = "rolling_audit_observe_failed";
            } else {
                record.contact = contact_fields(
                    rolling_audit.last_contact_frame());
            }
            executed_records.push_back(record);

            if (!failure_reason.empty()) break;
            if (plant_status != nmpc::Status::SUCCESS ||
                !nmpc::is_finite(next_state)) {
                failure_reason = std::string("plant_step_") +
                                 nmpc::status_string(plant_status);
                break;
            }
            plant_state = next_state;
            ++active_plan_stage;
            ++plant_knots;
            execution_tasks.advance(1);
        }
        if (!failure_reason.empty()) break;

        PlannerRecord planner_record;
        planner_record.update_index = update;
        planner_record.request_time_s = plant_knots * kRealtimeTimeStep;
        planner_record.handoff_time_s = planner_record.request_time_s;
        planner_record.executed_from_publication_id = active_publication_id;
        planner_record.executed_stage_begin = executed_stage_begin;
        planner_record.executed_stage_end = active_plan_stage - 1;

        QuadrupedCITORealtimePlanner::Clock::time_point deadline =
            QuadrupedCITORealtimePlanner::Clock::time_point::max();
        if (options.deadline_ms > 0.0) {
            deadline = QuadrupedCITORealtimePlanner::Clock::now() +
                std::chrono::duration_cast<
                    QuadrupedCITORealtimePlanner::Clock::duration>(
                    std::chrono::duration<double, std::milli>(
                        options.deadline_ms));
        }
        planner_record.result = planner.warm_update(plant_state, deadline);
        exact_hessian_calls +=
            planner_record.result.stats.exact_hessian_analytic_calls +
            planner_record.result.stats.exact_hessian_fd_calls;

        if (planner_record.result.published) {
            planner_record.publication_id = next_publication_id++;
            ++published_updates;
            const RealtimeContactTask* valid_task = planner.last_valid_task();
            const RealtimeContactTask task = valid_task != nullptr
                ? *valid_task
                : planner.active_task();
            append_plan_stages(
                planner.problem(), contact_terrain,
                planner_record.publication_id, "warm", update,
                planner_record.handoff_time_s, task,
                accepted_stage_records);
            *active_problem = planner.problem();
            active_plan_stage = 0;
            active_publication_id = planner_record.publication_id;
            consecutive_fallbacks = 0;
        } else {
            ++fallback_updates;
            ++consecutive_fallbacks;
            maximum_consecutive_fallbacks = std::max(
                maximum_consecutive_fallbacks, consecutive_fallbacks);
        }
        const RealtimeContactTask attempted_task = planner.active_task();
        planner_record.task_id = planner_record.result.published
            ? planner_record.result.task_id
            : attempted_task.id;
        planner_record.moving_foot = planner_record.result.published
            ? planner_record.result.moving_foot
            : attempted_task.moving_foot;
        planner_record.active_publication_id_after = active_publication_id;
        planner_records.push_back(planner_record);
        ++completed_updates;
    }

    ExecutedRecord terminal_record;
    terminal_record.knot = plant_knots;
    terminal_record.time_s = plant_knots * kRealtimeTimeStep;
    terminal_record.update_index = completed_updates;
    terminal_record.substep = 0;
    terminal_record.source_publication_id = active_publication_id;
    terminal_record.source_plan_stage = active_plan_stage;
    terminal_record.task_id = execution_tasks.active_task().id;
    terminal_record.moving_foot = execution_tasks.moving_foot_at(plant_knots);
    terminal_record.plant_status = failure_reason.empty()
        ? nmpc::Status::SUCCESS
        : nmpc::Status::NOT_INITIALIZED;
    terminal_record.state = plant_state;
    terminal_record.control.zero();
    terminal_record.contact = executed_records.empty()
        ? contact_fields(plant_state, nullptr, contact_terrain)
        : executed_records.back().contact;
    executed_records.push_back(terminal_record);

    run_completed = failure_reason.empty() &&
                    completed_updates == options.updates;
    if (!run_completed && failure_reason.empty())
        failure_reason = "incomplete_update_count";

    std::error_code directory_error;
    const std::filesystem::path output_directory(options.output_directory);
    std::filesystem::create_directories(output_directory, directory_error);
    if (directory_error) {
        std::fprintf(stderr, "failed to create output directory: %s\n",
                     directory_error.message().c_str());
        return 3;
    }

    const int completed_tasks =
        execution_tasks.completed_task_count(plant_knots);
    const RollingAuditReport audit_report = rolling_audit.finalize();
    const int final_touchdown_knot = kRealtimeSwingFirstStage +
        (kContactTaskCap - 1) * kRealtimeSustainedTaskSpacing +
        kRealtimeSustainedSwingStages;
    const double final_hold_s = std::max(
        0.0, (plant_knots - final_touchdown_knot) * kRealtimeTimeStep);
    const bool frozen_protocol = options.updates == kDefaultWarmUpdates;
    const int minimum_timed_publications = static_cast<int>(std::ceil(
        0.99 * static_cast<double>(options.updates)));
    const bool planner_acceptance = options.deadline_ms > 0.0
        ? published_updates >= minimum_timed_publications &&
              maximum_consecutive_fallbacks <= 1
        : published_updates == options.updates && fallback_updates == 0;
    const bool experiment_pass = frozen_protocol && run_completed &&
        completed_tasks == kContactTaskCap && final_hold_s > 0.0 &&
        planner_acceptance && exact_hessian_calls == 0 && audit_report.pass;

    std::vector<std::pair<std::string, std::string>> metadata = {
        {"schema_version", "1"},
        {"runner", "quadruped_cito_srbd_closed_loop"},
        {"execution_model", "independent_SRBDDynamics_object"},
        {"feedback_source", "integrated_plant_state"},
        {"episode_resets", "0"},
        {"cold_starts", "1"},
        {"terrain", "sinusoidal"},
        {"terrain_physical_offset_m", as_string(physical_terrain.offset)},
        {"terrain_amplitude_m", as_string(options.terrain_amplitude)},
        {"wave_number_x", as_string(options.wave_number_x)},
        {"wave_number_y", as_string(options.wave_number_y)},
        {"state_dimension", as_string(kStateDim)},
        {"control_dimension", as_string(kControlDim)},
        {"horizon_knots", as_string(kRealtimeHorizon)},
        {"dt_s", as_string(kRealtimeTimeStep)},
        {"horizon_s", as_string(kRealtimeHorizon * kRealtimeTimeStep)},
        {"replan_rate_hz", as_string(kReplanRateHz)},
        {"executed_knots_per_update", as_string(planner.shift_steps())},
        {"contact_task_cap", as_string(kContactTaskCap)},
        {"requested_warm_updates", as_string(options.updates)},
        {"frozen_paper_warm_updates", as_string(kDefaultWarmUpdates)},
        {"frozen_protocol",
         frozen_protocol ? "1" : "0"},
        {"development_updates_override",
         options.updates == kDefaultWarmUpdates ? "0" : "1"},
        {"deadline_mode",
         options.deadline_ms > 0.0 ? "wall_clock" : "unbounded"},
        {"deadline_ms", as_string(options.deadline_ms)},
        {"rolling_audit_required_tasks",
         as_string(RollingAuditThresholds{}.required_task_count)},
        {"telemetry_write_phase", "after_simulation"}};

    const std::string cold_status = initialize_status == nmpc::Status::SUCCESS
        ? nmpc::status_string(cold.status)
        : nmpc::status_string(initialize_status);
    std::vector<std::pair<std::string, std::string>> summary = {
        {"schema_version", "1"},
        {"run_completed", run_completed ? "1" : "0"},
        {"failure_reason", failure_reason},
        {"cold_status", cold_status},
        {"cold_published", cold.published ? "1" : "0"},
        {"cold_solver_ms", as_string(cold.solver_ms)},
        {"cold_end_to_end_ms", as_string(cold.end_to_end_ms)},
        {"requested_warm_updates", as_string(options.updates)},
        {"completed_warm_updates", as_string(completed_updates)},
        {"published_warm_updates", as_string(published_updates)},
        {"fallback_warm_updates", as_string(fallback_updates)},
        {"maximum_consecutive_fallbacks",
         as_string(maximum_consecutive_fallbacks)},
        {"accepted_plan_count", as_string(next_publication_id)},
        {"plant_knots", as_string(plant_knots)},
        {"simulated_time_s", as_string(plant_knots * kRealtimeTimeStep)},
        {"completed_contact_tasks", as_string(completed_tasks)},
        {"final_hold_s", as_string(final_hold_s)},
        {"exact_hessian_calls", as_string(exact_hessian_calls)},
        {"rolling_audit_pass", audit_report.pass ? "1" : "0"},
        {"planner_acceptance", planner_acceptance ? "1" : "0"},
        {"experiment_pass", experiment_pass ? "1" : "0"},
        {"rolling_audit_ledger_valid",
         audit_report.ledger_valid ? "1" : "0"},
        {"rolling_audit_input_valid",
         audit_report.input_valid ? "1" : "0"},
        {"rolling_audit_all_tasks_completed",
         audit_report.all_tasks_completed ? "1" : "0"},
        {"rolling_audit_final_support",
         audit_report.final_support ? "1" : "0"},
        {"rolling_audit_samples", as_string(audit_report.samples)},
        {"rolling_audit_completed_tasks",
         as_string(audit_report.completed_tasks)},
        {"rolling_audit_unexpected_liftoffs",
         as_string(audit_report.unexpected_liftoffs)},
        {"rolling_audit_unexpected_touchdowns",
         as_string(audit_report.unexpected_touchdowns)},
        {"rolling_audit_non_designated_airborne_samples",
         as_string(audit_report.non_designated_airborne_samples)},
        {"rolling_audit_schedule_violation_count",
         as_string(audit_report.schedule_violation_count)},
        {"rolling_audit_task_order_violations",
         as_string(audit_report.task_order_violations)},
        {"rolling_audit_contact_chatter_events",
         as_string(audit_report.contact_chatter_events)},
        {"rolling_audit_consecutive_final_support_samples",
         as_string(audit_report.consecutive_final_support_samples)},
        {"rolling_audit_base_progress",
         as_string(audit_report.base_progress)},
        {"rolling_audit_maximum_base_lateral_deviation",
         as_string(audit_report.maximum_base_lateral_deviation)},
        {"rolling_audit_maximum_abs_roll",
         as_string(audit_report.maximum_abs_roll)},
        {"rolling_audit_maximum_abs_pitch",
         as_string(audit_report.maximum_abs_pitch)},
        {"rolling_audit_foot0_progress",
         as_string(audit_report.foot_progress[0])},
        {"rolling_audit_foot1_progress",
         as_string(audit_report.foot_progress[1])},
        {"rolling_audit_foot2_progress",
         as_string(audit_report.foot_progress[2])},
        {"rolling_audit_foot3_progress",
         as_string(audit_report.foot_progress[3])},
        {"final_base_x",
         as_string(plant_state[StateIndex::base_position(0)])},
        {"final_base_y",
         as_string(plant_state[StateIndex::base_position(1)])},
        {"final_base_z",
         as_string(plant_state[StateIndex::base_position(2)])}};

    bool write_ok = true;
    write_ok = write_key_values(output_directory / "metadata.csv", metadata) &&
               write_ok;
    write_ok = write_key_values(output_directory / "summary.csv", summary) &&
               write_ok;
    write_ok = write_planner_updates(
                   output_directory / "planner_updates.csv", planner_records) &&
               write_ok;
    write_ok = write_executed_trajectory(
                   output_directory / "executed_trajectory.csv",
                   executed_records) &&
               write_ok;
    write_ok = write_accepted_plan_stages(
                   output_directory / "accepted_plan_stages.csv",
                   accepted_stage_records) &&
               write_ok;
    write_ok = write_task_ledger(
                   output_directory / "task_ledger.csv", audit_report) &&
               write_ok;
    if (!write_ok) {
        std::fprintf(stderr, "failed to write one or more telemetry files\n");
        return 3;
    }

    std::printf(
        "SRBD closed-loop run %s: warm_updates=%d/%d published=%d "
        "fallbacks=%d tasks=%d/%d audit=%s simulated_time=%.3f s "
        "output=%s\n",
        run_completed ? "completed" : "failed", completed_updates,
        options.updates, published_updates, fallback_updates, completed_tasks,
        kContactTaskCap, audit_report.pass ? "PASS" : "FAIL",
        plant_knots * kRealtimeTimeStep,
        options.output_directory.c_str());
    if (!run_completed)
        std::fprintf(stderr, "failure_reason=%s\n", failure_reason.c_str());
    return run_completed && (!frozen_protocol || experiment_pass) ? 0 : 1;
}
