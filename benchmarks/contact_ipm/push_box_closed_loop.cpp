#include <algorithm>
#include <chrono>
#include <cmath>
#include <cstdlib>
#include <filesystem>
#include <fstream>
#include <iomanip>
#include <iostream>
#include <random>
#include <sstream>
#include <string>
#include <vector>

#include "push_box_model.hpp"

using namespace nmpc;
using namespace contact_benchmark::push_box;

namespace {

constexpr int PLANT_SUBSTEPS = 10;
constexpr int GOAL_HOLD_STEPS = 10;
constexpr double KICK_TIME = 1.5;
constexpr double GOAL_TRANSLATION_TOL = 0.10;
constexpr double GOAL_ANGLE_TOL = 0.10;
constexpr double PATH_TRACKING_SCALE = 1.0;
constexpr double EFFORT_SCALE = 1.0;

struct Scenario {
    std::string group;
    int group_index = 0;
    unsigned seed = 0;
    Vec<NX> initial_offset{};
    double mass_scale = 1.0;
    double friction_scale = 1.0;
    int kick_step = -1;
    Vec<NX> kick{};
    double position_noise_std = 0.0;
    double angle_noise_std = 0.0;
};

struct PlanAudit {
    double dynamics_defect = 0.0;
    double side_violation = 0.0;
    double physical_mpcc = 0.0;
};

struct RolloutResult {
    Scenario scenario;
    bool task_success = false;
    int steps = 0;
    int goal_hold_steps = 0;
    int primary_failures = 0;
    int cold_fallbacks = 0;
    int unrecovered_failures = 0;
    int deadline_misses_all = 0;
    int deadline_misses_warm = 0;
    int invalid_plans = 0;
    double final_translation_error = 0.0;
    double final_angular_error = 0.0;
    double integrated_tracking_error = 0.0;
    double control_effort = 0.0;
    double control_variation = 0.0;
    double max_plan_dynamics_defect = 0.0;
    double max_plan_side_violation = 0.0;
    double max_plan_physical_mpcc = 0.0;
    double max_applied_side_violation = 0.0;
    double max_applied_physical_mpcc = 0.0;
    double max_one_step_model_error = 0.0;
    Vec<NX> initial_state{};
    Vec<NX> final_state{};
    std::vector<double> solve_times;
    std::vector<int> iterations;
};

double quantile(std::vector<double> values, double probability) {
    if (values.empty()) return 0.0;
    std::sort(values.begin(), values.end());
    const double index = probability * static_cast<double>(values.size() - 1);
    const auto lower = static_cast<std::size_t>(std::floor(index));
    const auto upper = static_cast<std::size_t>(std::ceil(index));
    const double weight = index - static_cast<double>(lower);
    return values[lower] * (1.0 - weight) + values[upper] * weight;
}

void cold_start(Problem& problem, const Vec<NX>& initial_state) {
    problem.x0 = initial_state;
    for (int k = 0; k <= HORIZON; ++k) {
        problem.stages[k].x.zero();
        problem.stages[k].u.zero();
    }
    problem.stages[0].x = initial_state;
}

PlanAudit audit_plan(Problem& problem, PushBoxDynamics& dynamics,
                     PushBoxContacts& contacts) {
    PlanAudit audit;
    for (int k = 0; k < HORIZON; ++k) {
        Vec<NX> predicted;
        dynamics.discrete_step(problem.stages[k].x, problem.stages[k].u,
                               DT, predicted);
        for (int i = 0; i < NX; ++i)
            audit.dynamics_defect = std::max(
                audit.dynamics_defect,
                std::fabs(predicted[i] - problem.stages[k + 1].x[i]));

        Vec<NC> rows;
        contacts.evaluate(problem.stages[k].x, problem.stages[k].u, k, rows);
        for (int row = 0; row < 8; ++row)
            audit.side_violation = std::max(
                audit.side_violation, std::max(0.0, rows[row]));
        for (int pair = 0; pair < 10; ++pair)
            audit.physical_mpcc = std::max(
                audit.physical_mpcc,
                std::fabs(rows[PushBoxContacts::pairs[pair][0]] *
                          rows[PushBoxContacts::pairs[pair][1]]));
    }
    return audit;
}

void audit_control(const Vec<NX>& state, const Vec<NU>& control,
                   PushBoxContacts& contacts, double& side_violation,
                   double& physical_mpcc) {
    Vec<NC> rows;
    contacts.evaluate(state, control, 0, rows);
    for (int row = 0; row < 8; ++row)
        side_violation = std::max(
            side_violation, std::max(0.0, rows[row]));
    for (int pair = 0; pair < 10; ++pair)
        physical_mpcc = std::max(
            physical_mpcc,
            std::fabs(rows[PushBoxContacts::pairs[pair][0]] *
                      rows[PushBoxContacts::pairs[pair][1]]));
}

void add_scaled(const Vec<NX>& x, const Vec<NX>& direction, double scale,
                Vec<NX>& result) {
    for (int i = 0; i < NX; ++i)
        result[i] = x[i] + scale * direction[i];
}

void plant_step_rk4(const PushBoxDynamics& plant, const Vec<NX>& state,
                    const Vec<NU>& control, Vec<NX>& next) {
    next = state;
    const double step = DT / static_cast<double>(PLANT_SUBSTEPS);
    for (int substep = 0; substep < PLANT_SUBSTEPS; ++substep) {
        Vec<NX> k1, k2, k3, k4, work;
        plant.velocity(next, control, k1);
        add_scaled(next, k1, 0.5 * step, work);
        plant.velocity(work, control, k2);
        add_scaled(next, k2, 0.5 * step, work);
        plant.velocity(work, control, k3);
        add_scaled(next, k3, step, work);
        plant.velocity(work, control, k4);
        for (int i = 0; i < NX; ++i)
            next[i] += step *
                (k1[i] + 2.0 * k2[i] + 2.0 * k3[i] + k4[i]) / 6.0;
    }
}

Vec<NX> measured_state(const Vec<NX>& state, const Scenario& scenario,
                       std::mt19937& generator) {
    std::normal_distribution<double> position_noise(
        0.0, scenario.position_noise_std);
    std::normal_distribution<double> angle_noise(
        0.0, scenario.angle_noise_std);
    Vec<NX> measured = state;
    measured[0] += position_noise(generator);
    measured[1] += position_noise(generator);
    measured[2] += angle_noise(generator);
    return measured;
}

std::vector<Scenario> make_scenarios(unsigned seed, bool smoke) {
    std::vector<Scenario> scenarios;
    std::mt19937 generator(seed);
    std::uniform_real_distribution<double> position(-0.10, 0.10);
    std::uniform_real_distribution<double> angle(-0.10, 0.10);
    std::uniform_real_distribution<double> scale(0.80, 1.20);
    std::uniform_real_distribution<double> kick_position(-0.15, 0.15);
    std::uniform_real_distribution<double> kick_angle(-0.12, 0.12);

    auto append = [&](const std::string& group, int index) -> Scenario& {
        Scenario scenario;
        scenario.group = group;
        scenario.group_index = index;
        scenario.seed = seed + static_cast<unsigned>(scenarios.size());
        scenarios.push_back(scenario);
        return scenarios.back();
    };

    for (int i = 0; i < 5; ++i)
        append("nominal", i);
    if (smoke) {
        scenarios.resize(1);
        return scenarios;
    }

    for (int i = 0; i < 15; ++i) {
        Scenario& scenario = append("initial_pose", i);
        scenario.initial_offset[0] = position(generator);
        scenario.initial_offset[1] = position(generator);
        scenario.initial_offset[2] = angle(generator);
    }
    for (int i = 0; i < 10; ++i) {
        Scenario& scenario = append("model_mismatch", i);
        scenario.mass_scale = scale(generator);
        scenario.friction_scale = scale(generator);
    }
    for (int i = 0; i < 10; ++i) {
        Scenario& scenario = append("pose_kick", i);
        scenario.kick_step = static_cast<int>(KICK_TIME / DT);
        scenario.kick[0] = kick_position(generator);
        scenario.kick[1] = kick_position(generator);
        scenario.kick[2] = kick_angle(generator);
    }
    for (int i = 0; i < 10; ++i) {
        Scenario& scenario = append("combined", i);
        scenario.initial_offset[0] = 0.75 * position(generator);
        scenario.initial_offset[1] = 0.75 * position(generator);
        scenario.initial_offset[2] = 0.75 * angle(generator);
        scenario.mass_scale = 0.85 + 0.30 *
            std::generate_canonical<double, 20>(generator);
        scenario.friction_scale = 0.85 + 0.30 *
            std::generate_canonical<double, 20>(generator);
        scenario.kick_step = static_cast<int>(KICK_TIME / DT);
        scenario.kick[0] = 0.75 * kick_position(generator);
        scenario.kick[1] = 0.75 * kick_position(generator);
        scenario.kick[2] = 0.75 * kick_angle(generator);
        scenario.position_noise_std = 0.001;
        scenario.angle_noise_std = 0.001;
    }
    return scenarios;
}

void write_trajectory_header(std::ofstream& output) {
    output << "step,time,x,y,theta,"
              "measured_x,measured_y,measured_theta,"
              "contact_x,contact_y,"
              "lambda1,lambda2,lambda3,lambda4,solve_time_s,iterations,"
              "solver_success,physical_mpcc,side_violation\n";
}

RolloutResult run_rollout(const Scenario& scenario, int max_steps,
                          const std::filesystem::path& trajectory_directory,
                          int rollout_index) {
    PushBoxDynamics model;
    PushBoxDynamics plant;
    plant.mass = NOMINAL_MASS * scenario.mass_scale;
    plant.friction = NOMINAL_FRICTION * scenario.friction_scale;
    PushBoxCost cost;
    cost.path_tracking_scale = PATH_TRACKING_SCALE;
    cost.effort_scale = EFFORT_SCALE;
    PushBoxContacts contacts;
    Problem problem;
    Vec<NX> true_state = scenario.initial_offset;
    initialize_problem(problem, model, cost, contacts, true_state);

    ContactIPMParams params = solver_parameters();
    ContactIPM<NX, NU, NC, HORIZON> solver;
    solver.configure(params);
    std::mt19937 noise_generator(scenario.seed);

    RolloutResult result;
    result.scenario = scenario;
    result.initial_state = true_state;

    std::ofstream trajectory;
    if (!trajectory_directory.empty()) {
        std::filesystem::create_directories(trajectory_directory);
        std::ostringstream filename;
        filename << "rollout_" << std::setfill('0') << std::setw(2)
                 << rollout_index << '_' << scenario.group << ".csv";
        trajectory.open(trajectory_directory / filename.str());
        trajectory << std::setprecision(17);
        write_trajectory_header(trajectory);
    }

    Vec<NU> previous_control;
    previous_control.zero();
    bool have_previous_control = false;
    int goal_hold = 0;
    Vec<NX> measurement =
        measured_state(true_state, scenario, noise_generator);
    problem.x0 = measurement;
    problem.stages[0].x = measurement;

    for (int step = 0; step < max_steps; ++step) {
        const auto solve_start = std::chrono::steady_clock::now();
        Status status = solver.solve_mpcc_with_recovery(problem);
        if (status != Status::SUCCESS) {
            ++result.primary_failures;
            ++result.cold_fallbacks;
            cold_start(problem, measurement);
            status = solver.solve_mpcc_with_recovery(problem);
        }
        const auto solve_stop = std::chrono::steady_clock::now();
        const double solve_time =
            std::chrono::duration<double>(solve_stop - solve_start).count();
        result.solve_times.push_back(solve_time);
        result.iterations.push_back(solver.last_stats().inner_iterations);
        if (solve_time > DT) {
            ++result.deadline_misses_all;
            if (step > 0) ++result.deadline_misses_warm;
        }

        Vec<NU> control;
        if (status == Status::SUCCESS) {
            solver.get_first_control(problem, control);
            const PlanAudit audit = audit_plan(problem, model, contacts);
            result.max_plan_dynamics_defect = std::max(
                result.max_plan_dynamics_defect, audit.dynamics_defect);
            result.max_plan_side_violation = std::max(
                result.max_plan_side_violation, audit.side_violation);
            result.max_plan_physical_mpcc = std::max(
                result.max_plan_physical_mpcc, audit.physical_mpcc);
            if (audit.dynamics_defect > params.tol_primal ||
                audit.side_violation > params.tol_ineq ||
                audit.physical_mpcc > params.tol_mpcc) {
                ++result.invalid_plans;
            }
        } else {
            ++result.unrecovered_failures;
            control = previous_control;
        }

        double applied_side = 0.0;
        double applied_mpcc = 0.0;
        audit_control(true_state, control, contacts,
                      applied_side, applied_mpcc);
        result.max_applied_side_violation = std::max(
            result.max_applied_side_violation, applied_side);
        result.max_applied_physical_mpcc = std::max(
            result.max_applied_physical_mpcc, applied_mpcc);

        const double translation =
            translation_error(true_state, cost.target);
        const double angular = angular_error(true_state, cost.target);
        result.integrated_tracking_error += DT *
            (translation * translation + angular * angular);
        for (int i = 2; i < NU; ++i)
            result.control_effort += DT * control[i] * control[i];
        if (have_previous_control) {
            for (int i = 0; i < NU; ++i) {
                const double difference = control[i] - previous_control[i];
                result.control_variation += difference * difference;
            }
        }

        Vec<NX> nominal_next;
        model.discrete_step(true_state, control, DT, nominal_next);
        Vec<NX> plant_next;
        plant_step_rk4(plant, true_state, control, plant_next);
        double one_step_error = 0.0;
        for (int i = 0; i < NX; ++i) {
            const double difference = plant_next[i] - nominal_next[i];
            one_step_error += difference * difference;
        }
        result.max_one_step_model_error = std::max(
            result.max_one_step_model_error, std::sqrt(one_step_error));

        if (trajectory) {
            trajectory << step << ',' << step * DT << ','
                       << true_state[0] << ',' << true_state[1] << ','
                       << true_state[2] << ','
                       << measurement[0] << ',' << measurement[1] << ','
                       << measurement[2];
            for (int i = 0; i < NU; ++i)
                trajectory << ',' << control[i];
            trajectory << ',' << solve_time << ','
                       << solver.last_stats().inner_iterations << ','
                       << (status == Status::SUCCESS ? 1 : 0) << ','
                       << applied_mpcc << ',' << applied_side << '\n';
        }

        true_state = plant_next;
        if (step == scenario.kick_step) {
            for (int i = 0; i < NX; ++i)
                true_state[i] += scenario.kick[i];
        }

        const double next_translation =
            translation_error(true_state, cost.target);
        const double next_angular = angular_error(true_state, cost.target);
        const bool disturbance_complete =
            scenario.kick_step < 0 || step >= scenario.kick_step;
        if (disturbance_complete &&
            next_translation <= GOAL_TRANSLATION_TOL &&
            next_angular <= GOAL_ANGLE_TOL) {
            ++goal_hold;
        } else {
            goal_hold = 0;
        }

        ++result.steps;
        previous_control = control;
        have_previous_control = true;
        if (goal_hold >= GOAL_HOLD_STEPS) break;

        measurement =
            measured_state(true_state, scenario, noise_generator);
        if (status == Status::SUCCESS) {
            problem.x0 = measurement;
            solver.shift_for_warmstart(problem, measurement);
        } else {
            cold_start(problem, measurement);
        }
    }

    result.goal_hold_steps = goal_hold;
    result.final_state = true_state;
    result.final_translation_error =
        translation_error(true_state, cost.target);
    result.final_angular_error = angular_error(true_state, cost.target);
    result.task_success =
        result.goal_hold_steps >= GOAL_HOLD_STEPS &&
        result.final_translation_error <= GOAL_TRANSLATION_TOL &&
        result.final_angular_error <= GOAL_ANGLE_TOL &&
        result.unrecovered_failures == 0 &&
        result.invalid_plans == 0 &&
        result.max_applied_side_violation <= params.tol_ineq &&
        result.max_applied_physical_mpcc <= params.tol_mpcc;
    return result;
}

void write_vector(std::ostream& output, const Vec<NX>& vector) {
    output << '[';
    for (int i = 0; i < NX; ++i) {
        if (i) output << ',';
        output << vector[i];
    }
    output << ']';
}

template <typename T>
void write_array(std::ostream& output, const std::vector<T>& values) {
    output << '[';
    for (std::size_t i = 0; i < values.size(); ++i) {
        if (i) output << ',';
        output << values[i];
    }
    output << ']';
}

void write_results(const std::filesystem::path& output_path,
                   const std::vector<RolloutResult>& results,
                   unsigned seed, int max_steps) {
    if (!output_path.parent_path().empty())
        std::filesystem::create_directories(output_path.parent_path());
    std::ofstream output(output_path);
    const PushBoxCost reported_cost;
    output << std::setprecision(17);
    output << "{\n"
           << "  \"schema_version\": 1,\n"
           << "  \"benchmark\": \"push_box_closed_loop\",\n"
           << "  \"seed\": " << seed << ",\n"
           << "  \"configuration\": {"
           << "\"horizon\":" << HORIZON
           << ",\"control_dt\":" << DT
           << ",\"plant_substeps\":" << PLANT_SUBSTEPS
           << ",\"max_steps\":" << max_steps
           << ",\"goal_hold_steps\":" << GOAL_HOLD_STEPS
           << ",\"translation_tolerance\":" << GOAL_TRANSLATION_TOL
           << ",\"angular_tolerance\":" << GOAL_ANGLE_TOL
           << ",\"path_tracking_scale\":" << PATH_TRACKING_SCALE
           << ",\"effort_scale\":" << EFFORT_SCALE
           << ",\"solver_mpcc_tolerance\":1e-5"
           << ",\"target\":[" << reported_cost.target[0] << ','
           << reported_cost.target[1] << ',' << reported_cost.target[2] << ']'
           << ",\"prediction_integrator\":\"forward_euler\""
           << ",\"plant_integrator\":\"rk4\""
           << ",\"initialization\":\"zero_then_shifted_warm_start\""
           << ",\"disturbance_type\":\"pose_kick_quasistatic_model\"},\n"
           << "  \"rollouts\": [\n";
    for (std::size_t index = 0; index < results.size(); ++index) {
        const RolloutResult& result = results[index];
        output << "    {\"index\":" << index
               << ",\"group\":\"" << result.scenario.group << '"'
               << ",\"group_index\":" << result.scenario.group_index
               << ",\"seed\":" << result.scenario.seed
               << ",\"task_success\":"
               << (result.task_success ? "true" : "false")
               << ",\"steps\":" << result.steps
               << ",\"goal_hold_steps\":" << result.goal_hold_steps
               << ",\"primary_failures\":" << result.primary_failures
               << ",\"cold_fallbacks\":" << result.cold_fallbacks
               << ",\"unrecovered_failures\":"
               << result.unrecovered_failures
               << ",\"invalid_plans\":" << result.invalid_plans
               << ",\"deadline_misses_all\":"
               << result.deadline_misses_all
               << ",\"deadline_misses_warm\":"
               << result.deadline_misses_warm
               << ",\"mass_scale\":" << result.scenario.mass_scale
               << ",\"friction_scale\":"
               << result.scenario.friction_scale
               << ",\"kick_step\":" << result.scenario.kick_step
               << ",\"position_noise_std\":"
               << result.scenario.position_noise_std
               << ",\"angle_noise_std\":"
               << result.scenario.angle_noise_std
               << ",\"initial_offset\":";
        write_vector(output, result.scenario.initial_offset);
        output << ",\"kick\":";
        write_vector(output, result.scenario.kick);
        output << ",\"initial_state\":";
        write_vector(output, result.initial_state);
        output << ",\"final_state\":";
        write_vector(output, result.final_state);
        output << ",\"final_translation_error\":"
               << result.final_translation_error
               << ",\"final_angular_error\":"
               << result.final_angular_error
               << ",\"integrated_tracking_error\":"
               << result.integrated_tracking_error
               << ",\"control_effort\":" << result.control_effort
               << ",\"control_variation\":" << result.control_variation
               << ",\"max_plan_dynamics_defect\":"
               << result.max_plan_dynamics_defect
               << ",\"max_plan_side_violation\":"
               << result.max_plan_side_violation
               << ",\"max_plan_physical_mpcc\":"
               << result.max_plan_physical_mpcc
               << ",\"max_applied_side_violation\":"
               << result.max_applied_side_violation
               << ",\"max_applied_physical_mpcc\":"
               << result.max_applied_physical_mpcc
               << ",\"max_one_step_model_error\":"
               << result.max_one_step_model_error
               << ",\"solve_times_s\":";
        write_array(output, result.solve_times);
        output << ",\"iterations\":";
        write_array(output, result.iterations);
        output << '}' << (index + 1 == results.size() ? "\n" : ",\n");
    }
    output << "  ]\n}\n";
}

}  // namespace

int main(int argc, char** argv) {
    std::string mode = "smoke";
    std::filesystem::path output_path = "push_box_closed_loop.json";
    std::filesystem::path trajectory_directory;
    unsigned seed = 2027;
    int max_steps = 100;
    for (int index = 1; index < argc; ++index) {
        const std::string argument = argv[index];
        if (argument == "--mode" && index + 1 < argc)
            mode = argv[++index];
        else if (argument == "--output" && index + 1 < argc)
            output_path = argv[++index];
        else if (argument == "--trajectory-dir" && index + 1 < argc)
            trajectory_directory = argv[++index];
        else if (argument == "--seed" && index + 1 < argc)
            seed = static_cast<unsigned>(std::strtoul(argv[++index], nullptr, 10));
        else if (argument == "--max-steps" && index + 1 < argc)
            max_steps = std::atoi(argv[++index]);
        else {
            std::cerr << "usage: " << argv[0]
                      << " [--mode smoke|suite] [--output FILE]"
                         " [--trajectory-dir DIR] [--seed N]"
                         " [--max-steps N]\n";
            return 2;
        }
    }
    if ((mode != "smoke" && mode != "suite") || max_steps < 1)
        return 2;

    const std::vector<Scenario> scenarios =
        make_scenarios(seed, mode == "smoke");
    std::vector<RolloutResult> results;
    std::vector<double> all_solve_times;
    int successes = 0;
    for (std::size_t index = 0; index < scenarios.size(); ++index) {
        RolloutResult result = run_rollout(
            scenarios[index], max_steps, trajectory_directory,
            static_cast<int>(index));
        if (result.task_success) ++successes;
        all_solve_times.insert(all_solve_times.end(),
                               result.solve_times.begin(),
                               result.solve_times.end());
        std::cout << '[' << (index + 1) << '/' << scenarios.size() << "] "
                  << result.scenario.group << ' '
                  << (result.task_success ? "success" : "failure")
                  << " steps=" << result.steps
                  << " error=(" << result.final_translation_error << ", "
                  << result.final_angular_error << ")\n";
        results.push_back(std::move(result));
    }
    write_results(output_path, results, seed, max_steps);
    std::cout << std::setprecision(6)
              << "success=" << successes << '/' << results.size() << '\n'
              << "solve_time_median_s="
              << quantile(all_solve_times, 0.5) << '\n'
              << "solve_time_p90_s="
              << quantile(all_solve_times, 0.9) << '\n'
              << "solve_time_p99_s="
              << quantile(all_solve_times, 0.99) << '\n'
              << "output=" << output_path.string() << '\n';
    return successes == static_cast<int>(results.size()) ? 0 : 1;
}
