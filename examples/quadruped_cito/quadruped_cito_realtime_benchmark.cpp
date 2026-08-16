#include <algorithm>
#include <chrono>
#include <cmath>
#include <cstdint>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <memory>
#include <string>
#include <vector>

#include "examples/quadruped_cito/quadruped_cito_realtime_planner.hpp"

namespace {

using namespace quadruped_cito;

#if defined(__clang__)
constexpr const char* kCompilerId = "clang";
constexpr int kCompilerMajor = __clang_major__;
constexpr int kCompilerMinor = __clang_minor__;
constexpr int kCompilerPatch = __clang_patchlevel__;
#elif defined(__GNUC__)
constexpr const char* kCompilerId = "gcc";
constexpr int kCompilerMajor = __GNUC__;
constexpr int kCompilerMinor = __GNUC_MINOR__;
constexpr int kCompilerPatch = __GNUC_PATCHLEVEL__;
#elif defined(_MSC_VER)
constexpr const char* kCompilerId = "msvc";
constexpr int kCompilerMajor = _MSC_VER / 100;
constexpr int kCompilerMinor = _MSC_VER % 100;
constexpr int kCompilerPatch = _MSC_FULL_VER % 100000;
#else
constexpr const char* kCompilerId = "unknown";
constexpr int kCompilerMajor = 0;
constexpr int kCompilerMinor = 0;
constexpr int kCompilerPatch = 0;
#endif

#if defined(__FAST_MATH__)
constexpr int kFastMathEnabled = 1;
#else
constexpr int kFastMathEnabled = 0;
#endif

#if defined(NDEBUG)
constexpr int kNdebugEnabled = 1;
#else
constexpr int kNdebugEnabled = 0;
#endif

struct Options {
    int rate_hz = 5;
    int updates = 100;
    int episode_updates = 10;
    std::string terrain = "sinusoidal";
    double amplitude = 0.025;
    unsigned int seed = 0;
    double slope_x = 0.0;
    double slope_y = 0.0;
    bool slope_x_set = false;
    bool slope_y_set = false;
    double wave_number_x = 4.0;
    double wave_number_y = 3.0;
    double step_height = 0.02;
    double step_center_x = 0.23;
    double step_sharpness = 18.0;
    double measurement_offset = 0.0005;
    std::string csv_path;
};

bool parse_int(const char* text, int& value) {
    char* end = nullptr;
    const long parsed = std::strtol(text, &end, 10);
    if (end == text || *end != '\0') return false;
    value = static_cast<int>(parsed);
    return static_cast<long>(value) == parsed;
}

bool parse_unsigned(const char* text, unsigned int& value) {
    char* end = nullptr;
    const unsigned long parsed = std::strtoul(text, &end, 10);
    if (end == text || *end != '\0') return false;
    value = static_cast<unsigned int>(parsed);
    return static_cast<unsigned long>(value) == parsed;
}

bool parse_double(const char* text, double& value) {
    char* end = nullptr;
    const double parsed = std::strtod(text, &end);
    if (end == text || *end != '\0' || !std::isfinite(parsed)) return false;
    value = parsed;
    return true;
}

bool option_value(int argc, char** argv, int& argument, const char*& value) {
    if (argument + 1 >= argc) return false;
    value = argv[++argument];
    return true;
}

bool parse_options(int argc, char** argv, Options& options) {
    for (int argument = 1; argument < argc; ++argument) {
        const std::string name(argv[argument]);
        const char* value = nullptr;
        if (name == "--help" || name == "-h") return false;
        if (!option_value(argc, argv, argument, value)) return false;
        if (name == "--rate") {
            if (!parse_int(value, options.rate_hz)) return false;
        } else if (name == "--updates") {
            if (!parse_int(value, options.updates)) return false;
        } else if (name == "--episode-updates") {
            if (!parse_int(value, options.episode_updates)) return false;
        } else if (name == "--terrain") {
            options.terrain = value;
        } else if (name == "--terrain-amplitude") {
            if (!parse_double(value, options.amplitude)) return false;
        } else if (name == "--terrain-seed") {
            if (!parse_unsigned(value, options.seed)) return false;
        } else if (name == "--slope-x") {
            if (!parse_double(value, options.slope_x)) return false;
            options.slope_x_set = true;
        } else if (name == "--slope-y") {
            if (!parse_double(value, options.slope_y)) return false;
            options.slope_y_set = true;
        } else if (name == "--wave-number-x") {
            if (!parse_double(value, options.wave_number_x)) return false;
        } else if (name == "--wave-number-y") {
            if (!parse_double(value, options.wave_number_y)) return false;
        } else if (name == "--step-height") {
            if (!parse_double(value, options.step_height)) return false;
        } else if (name == "--step-center-x") {
            if (!parse_double(value, options.step_center_x)) return false;
        } else if (name == "--step-sharpness") {
            if (!parse_double(value, options.step_sharpness)) return false;
        } else if (name == "--measurement-offset") {
            if (!parse_double(value, options.measurement_offset)) return false;
        } else if (name == "--csv") {
            options.csv_path = value;
        } else {
            return false;
        }
    }
    return (options.rate_hz == 5 || options.rate_hz == 10) &&
           options.updates > 0 && options.episode_updates > 0 &&
           options.episode_updates * (20 / options.rate_hz) <=
               kRealtimeHorizon &&
           options.amplitude >= 0.0 &&
           options.step_height >= 0.0 && options.step_sharpness > 0.0 &&
           options.wave_number_x >= 0.0 && options.wave_number_y >= 0.0 &&
           options.measurement_offset >= 0.0;
}

void print_usage() {
    std::printf(
        "usage: quadruped_cito_realtime_benchmark "
        "[--rate 5|10] [--updates N] [--episode-updates N] "
        "[--terrain flat|slope|cross-slope|sinusoidal|smooth_step|"
        "random_smooth] [--terrain-amplitude M] [--terrain-seed N] "
        "[--slope-x GRADE] [--slope-y GRADE] "
        "[--wave-number-x K] [--wave-number-y K] "
        "[--step-height M] [--step-center-x M] [--step-sharpness K] "
        "[--measurement-offset M] [--csv PATH]\n");
}

bool make_terrain(const Options& options, SharedTerrain& terrain,
                  std::string& terrain_label) {
    if (options.terrain == "flat") {
        terrain = SharedTerrain::flat();
        terrain_label = "flat";
    } else if (options.terrain == "slope" ||
               options.terrain == "longitudinal-slope" ||
               options.terrain == "longitudinal_slope") {
        const double slope_x = options.slope_x_set ? options.slope_x : 0.10;
        terrain = SharedTerrain::slope(slope_x, options.slope_y);
        terrain_label = "slope";
    } else if (options.terrain == "cross-slope" ||
               options.terrain == "cross_slope") {
        const double slope_y = options.slope_y_set ? options.slope_y : 0.10;
        terrain = SharedTerrain::slope(options.slope_x, slope_y);
        terrain_label = "cross_slope";
    } else if (options.terrain == "sinusoidal" ||
               options.terrain == "smooth") {
        terrain = SharedTerrain::sinusoidal(
            options.amplitude, options.wave_number_x, options.wave_number_y);
        terrain.slope_x = options.slope_x;
        terrain.slope_y = options.slope_y;
        terrain_label = "sinusoidal";
    } else if (options.terrain == "smooth_step" ||
               options.terrain == "smooth-step") {
        terrain = SharedTerrain::smooth_step(
            options.step_height, options.step_center_x,
            options.step_sharpness);
        terrain.slope_x = options.slope_x;
        terrain.slope_y = options.slope_y;
        terrain_label = "smooth_step";
    } else if (options.terrain == "random_smooth" ||
               options.terrain == "random-smooth") {
        terrain = SharedTerrain::random_smooth(options.seed,
                                                options.amplitude);
        terrain.slope_x = options.slope_x;
        terrain.slope_y = options.slope_y;
        terrain_label = "random_smooth";
    } else {
        return false;
    }
    terrain = register_go1_terrain(terrain);
    return true;
}

void hash_bytes(std::uint64_t& hash, const void* data, std::size_t size) {
    const auto* bytes = static_cast<const unsigned char*>(data);
    for (std::size_t index = 0; index < size; ++index) {
        hash ^= bytes[index];
        hash *= UINT64_C(1099511628211);
    }
}

std::uint64_t terrain_fingerprint(const SharedTerrain& terrain) {
    std::uint64_t hash = UINT64_C(1469598103934665603);
    const int kind = static_cast<int>(terrain.kind);
    hash_bytes(hash, &kind, sizeof(kind));
    const double values[] = {
        terrain.offset,
        terrain.slope_x,
        terrain.slope_y,
        terrain.amplitude,
        terrain.wave_number_x,
        terrain.wave_number_y,
        terrain.phase_x,
        terrain.phase_y,
        terrain.secondary_amplitude,
        terrain.secondary_wave_number_x,
        terrain.secondary_wave_number_y,
        terrain.secondary_phase_x,
        terrain.secondary_phase_y,
        terrain.step_height,
        terrain.step_center_x,
        terrain.step_sharpness};
    hash_bytes(hash, values, sizeof(values));
    return hash;
}

double percentile(std::vector<double> values, double fraction) {
    if (values.empty()) return 0.0;
    std::sort(values.begin(), values.end());
    const int index = std::max(
        0, static_cast<int>(std::ceil(fraction * values.size())) - 1);
    return values[std::min(index, static_cast<int>(values.size()) - 1)];
}

int contact_task_episode_updates(int shift_steps) {
    return (kRealtimeSwingLastStage + shift_steps - 1) / shift_steps;
}

const char* result_class(const RealtimePlannerResult& result) {
    if (result.published) return "FULL_KKT";
    if (result.deadline_miss) return "DEADLINE_FALLBACK";
    if (!result.full_kkt) {
        return result.status == nmpc::Status::SUCCESS
            ? "AUDIT_FAILED"
            : "NOT_CONVERGED";
    }
    if (!result.task_pass) return "TASK_FAILED";
    return "PLAN_EXTRACTION_FAILED";
}

void print_result(FILE* output, const char* record_type,
                  const std::string& terrain_label,
                  std::uint64_t fingerprint, unsigned int seed, int episode,
                  int update, int rate_hz, int shift_steps,
                  const RealtimePlannerResult& result,
                  double end_to_end_ms, int consecutive_fallbacks) {
    std::fprintf(
        output,
        "%s,%s,%016llx,%u,%d,%d,%d,%d,%s,%s,%d,%d,%d,%d,%d,%d,%d,%d,%d,%d,%d,%d,"
        "%.3e,%.3e,%.3e,"
        "%.3f,%.3f,%.3f,%.3f,%.3f,%.3f,%.3f,%.3f,%.3f,%.3f,%.3f,%.3f,"
        "%llu,%llu,%llu,%.3f,%.3f,%.3f,"
        "%.3f,%.3f,%.3f,%.3f,%.3f,%d,%d,%d,%d,%d,%d,%d,"
        "%.3e,%.3e,%.3e,%.3e,%.3e,%.3e,%.3e,%.3e,%.3e,%.3e,%.3e,%.3f,"
        "%.3f,%.3f,%d\n",
        record_type, terrain_label.c_str(),
        static_cast<unsigned long long>(fingerprint), seed, episode, update,
        rate_hz, shift_steps, nmpc::status_string(result.status),
        result_class(result),
        result.stats.inner_iterations, result.stats.line_search_evals,
        result.stats.warm_near_full_step_trials,
        result.stats.warm_near_full_step_accepts,
        result.stats.model_evaluations, result.stats.kkt_assemblies,
        result.stats.riccati_factorizations,
        result.stats.riccati_rhs_solves,
        result.stats.exact_hessian_analytic_calls,
        result.stats.exact_hessian_fd_calls,
        result.stats.deadline_checks, result.stats.time_limit_hit,
        result.stats.regularization,
        result.stats.max_regularization,
        result.stats.first_regularization,
        result.stats.model_eval_time_ms,
        result.stats.dynamics_eval_time_ms,
        result.stats.dynamics_jacobian_time_ms,
        result.stats.cost_derivative_time_ms,
        result.stats.constraint_eval_time_ms,
        result.stats.constraint_jacobian_time_ms,
        result.stats.constraint_value_jacobian_time_ms,
        result.stats.residual_eval_time_ms,
        result.stats.kkt_assembly_time_ms,
        result.stats.riccati_time_ms,
        result.stats.line_search_time_ms,
        result.stats.finalization_time_ms,
        static_cast<unsigned long long>(result.solver_terrain_samples),
        static_cast<unsigned long long>(result.audit_terrain_samples),
        static_cast<unsigned long long>(
            result.plan_extract_terrain_samples),
        result.solver_terrain_ms, result.audit_terrain_ms,
        result.plan_extract_terrain_ms,
        result.shift_ms, result.solver_ms, result.audit_ms,
        result.plan_extract_ms, end_to_end_ms,
        result.deadline_miss ? 1 : 0, result.full_kkt ? 1 : 0,
        result.task_pass ? 1 : 0, result.published ? 1 : 0,
        result.fallback ? 1 : 0, result.cold_fallback_used ? 1 : 0,
        consecutive_fallbacks,
        result.stats.primal_infeas, result.stats.dual_infeas,
        result.stats.complementarity, result.stats.mpcc_complementarity,
        result.stats.barrier_param, result.audit.dynamics,
        result.audit.inequality, result.audit.mpcc,
        result.audit.terminal_foot_error, result.audit.terminal_gap,
        result.audit.terminal_speed, result.audit.terminal_normal_force,
        result.audit.task_displacement, result.audit.clearance,
        result.audit.moving_unloaded_stages);
}

}  // namespace

int main(int argc, char** argv) {
    Options options;
    if (!parse_options(argc, argv, options)) {
        print_usage();
        return 2;
    }

    SharedTerrain terrain;
    std::string terrain_label;
    if (!make_terrain(options, terrain, terrain_label)) {
        print_usage();
        return 2;
    }
    FILE* output = stdout;
    if (!options.csv_path.empty()) {
        output = std::fopen(options.csv_path.c_str(), "w");
        if (output == nullptr) {
            std::fprintf(stderr, "failed to open CSV path: %s\n",
                         options.csv_path.c_str());
            return 2;
        }
    }

    const std::uint64_t fingerprint = terrain_fingerprint(terrain);
    const int shift_steps = static_cast<int>(std::lround(
        1.0 / (options.rate_hz * kRealtimeTimeStep)));
    const int required_contact_task_updates =
        contact_task_episode_updates(shift_steps);
    const double deadline_ms = 1000.0 / options.rate_hz;
    std::fprintf(
        output,
        "benchmark_config,terrain=%s,terrain_fingerprint=%016llx,"
        "terrain_seed=%u,terrain_kind=%s,offset=%.9g,slope_x=%.9g,"
        "slope_y=%.9g,amplitude=%.9g,wave_number_x=%.9g,"
        "wave_number_y=%.9g,phase_x=%.9g,phase_y=%.9g,"
        "secondary_amplitude=%.9g,secondary_wave_number_x=%.9g,"
        "secondary_wave_number_y=%.9g,secondary_phase_x=%.9g,"
        "secondary_phase_y=%.9g,step_height=%.9g,step_center_x=%.9g,"
        "step_sharpness=%.9g,nx=%d,nu=%d,nc=%d,"
        "N=%d,dt=%.3f,horizon_s=%.3f,rate_hz=%d,shift_steps=%d,"
        "deadline_ms=%.3f,updates=%d,episode_updates=%d,"
        "contact_task_episode_updates=%d,episode_reset=standing,"
        "measurement_offset=%.9g,"
        "contact_task_update_definition=warm_update_in_complete_active_to_"
        "settled_episode,"
        "riccati_reg_base=%.3e,recovery=off,"
        "exact_hessian=off,"
        "adaptive_exact_hessian=off,nonlinear_rollout=off,"
        "terrain_convention=go1_foot_center,compiler=%s,"
        "compiler_major=%d,compiler_minor=%d,compiler_patch=%d,"
        "fast_math=%d,ndebug=%d,cplusplus=%ld\n",
        terrain_label.c_str(),
        static_cast<unsigned long long>(fingerprint), options.seed,
        terrain_name(terrain.kind), terrain.offset, terrain.slope_x,
        terrain.slope_y, terrain.amplitude, terrain.wave_number_x,
        terrain.wave_number_y, terrain.phase_x, terrain.phase_y,
        terrain.secondary_amplitude, terrain.secondary_wave_number_x,
        terrain.secondary_wave_number_y, terrain.secondary_phase_x,
        terrain.secondary_phase_y, terrain.step_height,
        terrain.step_center_x, terrain.step_sharpness,
        kStateDim, kControlDim, kConstraintCapacity, kRealtimeHorizon,
        kRealtimeTimeStep, kRealtimeHorizon * kRealtimeTimeStep,
        options.rate_hz, shift_steps, deadline_ms, options.updates,
        options.episode_updates, required_contact_task_updates,
        options.measurement_offset,
        kRealtimeRiccatiRelativeRegularization, kCompilerId,
        kCompilerMajor, kCompilerMinor, kCompilerPatch,
        kFastMathEnabled, kNdebugEnabled, static_cast<long>(__cplusplus));
    std::fprintf(
        output,
        "state_assumption,predicted_knot_plus_alternating_base_x_offset_m="
        "%.6f\n",
        options.measurement_offset);
    std::fprintf(
        output,
        "record_type,terrain,terrain_fingerprint,terrain_seed,episode,update,"
        "rate_hz,shift_steps,status,class,iterations,line_search_evals,"
        "warm_near_full_step_trials,warm_near_full_step_accepts,"
        "model_evaluations,kkt_assemblies,riccati_factorizations,"
        "riccati_rhs_solves,exact_hessian_analytic_calls,"
        "exact_hessian_fd_calls,deadline_checks,time_limit_hit,"
        "regularization,max_regularization,first_regularization,"
        "model_eval_ms,dynamics_eval_ms,dynamics_jacobian_ms,"
        "cost_derivative_ms,constraint_eval_ms,constraint_jacobian_ms,"
        "constraint_value_jacobian_ms,residual_eval_ms,kkt_assembly_ms,"
        "riccati_ms,line_search_ms,"
        "finalization_ms,solver_terrain_samples,audit_terrain_samples,"
        "plan_extract_terrain_samples,solver_terrain_ms,audit_terrain_ms,"
        "plan_extract_terrain_ms,shift_ms,"
        "solver_ms,audit_ms,plan_extract_ms,end_to_end_ms,"
        "deadline_miss,full_kkt,task_pass,published,fallback,"
        "cold_fallback_used,consecutive_fallbacks,primal,stationarity,"
        "complementarity,"
        "solver_mpcc,barrier,dynamics,inequality,raw_mpcc,"
        "terminal_foot_error,terminal_gap,terminal_speed,"
        "terminal_normal_force,task_displacement,clearance,"
        "moving_unloaded_stages\n");

    std::vector<double> solver_times;
    std::vector<double> end_to_end_times;
    int full_kkt_updates = 0;
    int task_pass_updates = 0;
    int published_updates = 0;
    int fallback_updates = 0;
    int deadline_misses = 0;
    int invalid_publications = 0;
    int consecutive_fallbacks = 0;
    int maximum_consecutive_fallbacks = 0;
    int exact_hessian_calls = 0;
    int regularization_retry_updates = 0;
    double max_regularization = 0.0;
    double max_first_regularization = 0.0;
    int warm_near_full_step_trials = 0;
    int warm_near_full_step_accepts = 0;
    int completed_updates = 0;
    int contact_task_updates = 0;
    int completed_contact_tasks = 0;
    int episodes = 0;
    int cold_failures = 0;
    int cold_fallbacks = 0;
    double max_cold_solver_ms = 0.0;
    RealtimePlannerConfig planner_configuration;
    planner_configuration.rate_hz = options.rate_hz;

    while (completed_updates < options.updates) {
        const int episode = episodes++;
        auto planner = std::make_unique<QuadrupedCITORealtimePlanner>(
            terrain, planner_configuration);
        const Vec<kStateDim> standing = planner->nominal_state();
        const nmpc::Status initialize_status = planner->initialize(standing);
        if (initialize_status != nmpc::Status::SUCCESS) {
            ++cold_failures;
            std::fprintf(stderr, "planner initialization failed: %s\n",
                         nmpc::status_string(initialize_status));
            break;
        }
        const RealtimePlannerResult cold = planner->cold_solve();
        print_result(output, "cold", terrain_label, fingerprint, options.seed,
                     episode, -1, options.rate_hz, planner->shift_steps(), cold,
                     cold.end_to_end_ms, 0);
        exact_hessian_calls +=
            cold.stats.exact_hessian_analytic_calls +
            cold.stats.exact_hessian_fd_calls;
        cold_fallbacks += cold.cold_fallback_used ? 1 : 0;
        max_cold_solver_ms = std::max(max_cold_solver_ms, cold.solver_ms);
        if (!cold.published) {
            ++cold_failures;
            break;
        }
        auto active_problem =
            std::make_unique<QuadrupedCITORealtimePlanner::ProblemType>(
                planner->problem());
        int active_plan_stage = 0;
        consecutive_fallbacks = 0;
        const int episode_end = std::min(
            options.updates,
            completed_updates + options.episode_updates);
        int episode_warm_updates = 0;
        bool episode_has_active_transition = false;
        bool episode_has_settled_transition = false;

        while (completed_updates < episode_end) {
            const int update = completed_updates;
            const auto update_start =
                QuadrupedCITORealtimePlanner::Clock::now();
            const auto deadline = update_start +
                std::chrono::duration_cast<
                    QuadrupedCITORealtimePlanner::Clock::duration>(
                    std::chrono::duration<double, std::milli>(deadline_ms));
            active_plan_stage = std::min(
                kRealtimeHorizon,
                active_plan_stage + planner->shift_steps());
            const int applied_shift_steps = planner->next_shift_steps();
            Vec<kStateDim> actual_state =
                active_problem->stages[active_plan_stage].x;
            actual_state[StateIndex::base_position(0)] +=
                update % 2 == 0 ? options.measurement_offset
                                : -options.measurement_offset;
            RealtimePlannerResult result =
                planner->warm_update(actual_state, deadline);
            const auto update_end =
                QuadrupedCITORealtimePlanner::Clock::now();
            const double end_to_end_ms =
                std::chrono::duration<double, std::milli>(
                    update_end - update_start).count();
            result.deadline_miss = result.deadline_miss ||
                                   end_to_end_ms > deadline_ms;

            ++episode_warm_updates;
            if (episode_warm_updates <= required_contact_task_updates &&
                result.published &&
                result.task_pass &&
                result.audit.moving_unloaded_stages > 0) {
                episode_has_active_transition = true;
            }
            if (episode_warm_updates == required_contact_task_updates &&
                result.published &&
                result.task_pass &&
                result.audit.moving_unloaded_stages == 0) {
                episode_has_settled_transition = true;
            }
            ++completed_updates;
            full_kkt_updates += result.full_kkt ? 1 : 0;
            task_pass_updates += result.task_pass ? 1 : 0;
            published_updates += result.published ? 1 : 0;
            fallback_updates += result.fallback ? 1 : 0;
            deadline_misses += result.deadline_miss ? 1 : 0;
            exact_hessian_calls +=
                result.stats.exact_hessian_analytic_calls +
                result.stats.exact_hessian_fd_calls;
            regularization_retry_updates +=
                result.stats.max_regularization >
                        kRealtimeRiccatiRelativeRegularization * 1.0001
                    ? 1
                    : 0;
            max_regularization = std::max(
                max_regularization, result.stats.max_regularization);
            max_first_regularization = std::max(
                max_first_regularization,
                result.stats.first_regularization);
            warm_near_full_step_trials +=
                result.stats.warm_near_full_step_trials;
            warm_near_full_step_accepts +=
                result.stats.warm_near_full_step_accepts;
            invalid_publications +=
                result.published &&
                        (!result.full_kkt || !result.task_pass ||
                         result.deadline_miss)
                    ? 1
                    : 0;
            consecutive_fallbacks = result.fallback
                ? consecutive_fallbacks + 1
                : 0;
            maximum_consecutive_fallbacks = std::max(
                maximum_consecutive_fallbacks, consecutive_fallbacks);
            solver_times.push_back(result.solver_ms);
            end_to_end_times.push_back(end_to_end_ms);
            print_result(output, "update", terrain_label, fingerprint,
                         options.seed, episode, update, options.rate_hz,
                         applied_shift_steps, result, end_to_end_ms,
                         consecutive_fallbacks);
            if (result.published) {
                *active_problem = planner->problem();
                active_plan_stage = 0;
            }
        }
        if (episode_warm_updates >= required_contact_task_updates &&
            episode_has_active_transition &&
            episode_has_settled_transition) {
            ++completed_contact_tasks;
            contact_task_updates += required_contact_task_updates;
        }
    }

    const double p50_solver = percentile(solver_times, 0.50);
    const double p90_solver = percentile(solver_times, 0.90);
    const double p99_solver = percentile(solver_times, 0.99);
    const double max_solver = percentile(solver_times, 1.00);
    const double p50_e2e = percentile(end_to_end_times, 0.50);
    const double p90_e2e = percentile(end_to_end_times, 0.90);
    const double p99_e2e = percentile(end_to_end_times, 0.99);
    const double max_e2e = percentile(end_to_end_times, 1.00);
    const int minimum_usable =
        static_cast<int>(std::ceil(0.99 * options.updates));
    const int unqualified_updates = completed_updates - contact_task_updates;
    const bool accepted = completed_updates == options.updates &&
                          cold_failures == 0 &&
                          contact_task_updates == completed_updates &&
                          completed_contact_tasks > 0 &&
                          full_kkt_updates >= minimum_usable &&
                          task_pass_updates >= minimum_usable &&
                          published_updates >= minimum_usable &&
                          invalid_publications == 0 &&
                          exact_hessian_calls == 0 &&
                          maximum_consecutive_fallbacks <= 1 &&
                          p99_e2e <= 180.0;
    std::fprintf(
        output,
        "summary,terrain=%s,terrain_fingerprint=%016llx,terrain_seed=%u,"
        "verdict=%s,completed=%d,updates=%d,episodes=%d,cold_failures=%d,"
        "cold_fallbacks=%d,max_cold_solver_ms=%.3f,"
        "contact_task_episode_updates=%d,completed_contact_tasks=%d,"
        "contact_task_updates=%d,unqualified_updates=%d,"
        "full_kkt=%d,task_pass=%d,"
        "published=%d,fallbacks=%d,deadline_misses=%d,"
        "invalid_publications=%d,exact_hessian_calls=%d,"
        "regularization_retry_updates=%d,max_regularization=%.3e,"
        "max_first_regularization=%.3e,"
        "warm_near_full_step_trials=%d,warm_near_full_step_accepts=%d,"
        "max_consecutive_fallbacks=%d,"
        "minimum_usable=%d,solver_p50_ms=%.3f,solver_p90_ms=%.3f,"
        "solver_p99_ms=%.3f,solver_max_ms=%.3f,e2e_p50_ms=%.3f,"
        "e2e_p90_ms=%.3f,e2e_p99_ms=%.3f,e2e_max_ms=%.3f,"
        "acceptance=all_updates_belong_to_complete_active_to_settled_"
        "contact_task_episodes_and_"
        "usable_rate_ge_0.99_and_e2e_p99_le_180ms_and_no_"
        "invalid_publication_and_zero_exact_hessian_calls_and_no_"
        "consecutive_fallback\n",
        terrain_label.c_str(),
        static_cast<unsigned long long>(fingerprint), options.seed,
        accepted ? "PASS" : "FAIL", completed_updates, options.updates,
        episodes, cold_failures, cold_fallbacks, max_cold_solver_ms,
        required_contact_task_updates,
        completed_contact_tasks, contact_task_updates, unqualified_updates,
        full_kkt_updates,
        task_pass_updates, published_updates,
        fallback_updates, deadline_misses, invalid_publications,
        exact_hessian_calls, regularization_retry_updates, max_regularization,
        max_first_regularization, warm_near_full_step_trials,
        warm_near_full_step_accepts,
        maximum_consecutive_fallbacks, minimum_usable,
        p50_solver,
        p90_solver, p99_solver, max_solver, p50_e2e, p90_e2e, p99_e2e,
        max_e2e);

    if (output != stdout) std::fclose(output);
    return accepted ? 0 : 1;
}
