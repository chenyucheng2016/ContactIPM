#include <cstdio>
#include <string>

#include "examples/quadruped_cito/quadruped_cito_four_step_planner.hpp"
#include "examples/quadruped_cito/quadruped_cito_plan_io.hpp"

namespace {

using namespace quadruped_cito;

bool parse_terrain(const std::string& name, SharedTerrain& terrain) {
    if (name == "flat") {
        terrain = register_go1_terrain(SharedTerrain::flat());
    } else if (name == "smooth") {
        terrain = register_go1_terrain(SharedTerrain::sinusoidal());
    } else if (name == "random_smooth") {
        terrain = register_go1_terrain(SharedTerrain::random_smooth(7, 0.02));
    } else if (name == "slope") {
        terrain = register_go1_terrain(SharedTerrain::slope(0.10));
    } else {
        return false;
    }
    return true;
}

bool parse_seed(const std::string& name, FourStepSeed& seed) {
    if (name == "forward") {
        seed = FourStepSeed::FORWARD;
    } else if (name == "reverse") {
        seed = FourStepSeed::REVERSE;
    } else if (name == "left_first") {
        seed = FourStepSeed::LEFT_FIRST;
    } else if (name == "right_first") {
        seed = FourStepSeed::RIGHT_FIRST;
    } else {
        return false;
    }
    return true;
}

void print_usage() {
    std::printf(
        "usage: test_quadruped_cito_four_step [--reverse | "
        "--seed forward|reverse|left_first|right_first | --multistart] "
        "[--terrain flat|smooth|random_smooth|slope] "
        "[--write-plan <path>]\n");
}

}  // namespace

int main(int argc, char** argv) {
    using namespace quadruped_cito;
    SharedTerrain terrain = register_go1_terrain(SharedTerrain::flat());
    FourStepSeed seed = FourStepSeed::FORWARD;
    bool seed_set = false;
    bool multistart = false;
    const char* plan_path = nullptr;
    for (int argument = 1; argument < argc; ++argument) {
        const std::string option = argv[argument];
        if (option == "--reverse" && !seed_set && !multistart) {
            seed = FourStepSeed::REVERSE;
            seed_set = true;
        } else if (option == "--seed" && argument + 1 < argc &&
                   !seed_set && !multistart) {
            if (!parse_seed(argv[++argument], seed)) {
                print_usage();
                return 2;
            }
            seed_set = true;
        } else if (option == "--multistart" && !seed_set && !multistart) {
            multistart = true;
        } else if (option == "--terrain" && argument + 1 < argc) {
            if (!parse_terrain(argv[++argument], terrain)) {
                print_usage();
                return 2;
            }
        } else if (option == "--write-plan" && argument + 1 < argc &&
                   !plan_path) {
            plan_path = argv[++argument];
        } else {
            print_usage();
            return 2;
        }
    }

    const Go1FootCenterTerrain<SharedTerrain> planner_terrain(terrain);
    Vec<kStateDim> initial_state = go1_home_state(planner_terrain);
    initial_state[StateIndex::base_position(0)] = 0.04;
    initial_state[StateIndex::base_position(1)] = 0.0;
    ContactPlan<kFourStepHorizon> plan;
    FourStepPlannerReport report;
    FourStepMultiStartReport multistart_report;
    const nmpc::Status status = multistart
        ? plan_go1_four_step_multistart(
              initial_state, terrain, plan, &report, &multistart_report)
        : plan_go1_four_step(initial_state, terrain, seed, plan, &report);
    int distinct_candidate_schedules = 0;
    if (multistart) {
        for (int candidate = 0; candidate < kFourStepSeedCount; ++candidate) {
            const FourStepPlannerReport& candidate_report =
                multistart_report.candidates[candidate];
            std::printf(
                "  multistart candidate: seed=%s status=%s objective=%.6f "
                "solver_status=%s audit_passed=%d schedule_passed=%d "
                "iterations=%d solve_ms=%.3f fingerprint=%llu "
                "schedule_deformation=%d\n",
                four_step_seed_name(candidate_report.seed),
                nmpc::status_string(candidate_report.status),
                candidate_report.objective,
                nmpc::status_string(candidate_report.solver_status),
                candidate_report.audit_passed ? 1 : 0,
                candidate_report.schedule_audit_passed ? 1 : 0,
                candidate_report.iterations,
                candidate_report.solve_ms,
                static_cast<unsigned long long>(
                    candidate_report.optimized_schedule_fingerprint),
                four_step_detail::schedule_deformation(candidate_report));
            if (candidate_report.optimized_schedule_fingerprint == 0) return 1;
            bool first_occurrence = true;
            for (int previous = 0; previous < candidate; ++previous) {
                if (multistart_report.candidates[previous]
                        .optimized_schedule_fingerprint ==
                    candidate_report.optimized_schedule_fingerprint) {
                    first_occurrence = false;
                }
            }
            if (first_occurrence) ++distinct_candidate_schedules;
        }
        if (distinct_candidate_schedules < 2) return 1;
        const FourStepPlannerReport& selected =
            multistart_report.candidates[
                multistart_report.selected_candidate];
        const int selected_deformation =
            four_step_detail::schedule_deformation(selected);
        for (int candidate = 0; candidate < kFourStepSeedCount; ++candidate) {
            const FourStepPlannerReport& candidate_report =
                multistart_report.candidates[candidate];
            if (candidate_report.status != nmpc::Status::SUCCESS) continue;
            const int candidate_deformation =
                four_step_detail::schedule_deformation(candidate_report);
            if (candidate_deformation < selected_deformation ||
                (candidate_deformation == selected_deformation &&
                 candidate_report.objective < selected.objective)) {
                std::printf("multistart selected a dominated schedule\n");
                return 1;
            }
        }
    }
    std::printf(
        "callable four-step planner: terrain=%s seed=%s multistart=%d "
        "status=%s iterations=%d solve_ms=%.3f objective=%.6f "
        "seed_dynamics=%.3e "
        "dynamics=%.3e inequality=%.3e mpcc=%.3e "
        "base_displacement=%.3f seed_fingerprint=%llu "
        "optimized_fingerprint=%llu\n",
        terrain_name(terrain.kind), four_step_seed_name(report.seed),
        multistart ? 1 : 0,
        nmpc::status_string(status), report.iterations, report.solve_ms,
        report.objective, report.seed_dynamics, report.dynamics,
        report.inequality, report.mpcc, report.base_displacement,
        static_cast<unsigned long long>(report.seed_schedule_fingerprint),
        static_cast<unsigned long long>(
            report.optimized_schedule_fingerprint));
    if (status != nmpc::Status::SUCCESS) {
        for (int foot = 0; foot < kNumFeet; ++foot) {
            std::printf(
                "  rejected schedule: foot=%d events=%d "
                "optimized=(%d,%d)\n",
                foot, report.contact_events[foot],
                report.optimized_liftoff_stage[foot],
                report.optimized_touchdown_stage[foot]);
        }
        return 1;
    }
    if (report.seed_schedule_fingerprint == 0 ||
        report.optimized_schedule_fingerprint == 0 ||
        (multistart && (multistart_report.successful_candidates < 1 ||
                       multistart_report.selected_candidate < 0))) {
        return 1;
    }

    int discovered_order[kNumFeet] = {0, 1, 2, 3};
    for (int first = 0; first < kNumFeet; ++first) {
        if (report.first_motion_stage[first] < 0) return 1;
        for (int second = first + 1; second < kNumFeet; ++second) {
            if (report.first_motion_stage[discovered_order[second]] <
                report.first_motion_stage[discovered_order[first]]) {
                const int temporary = discovered_order[first];
                discovered_order[first] = discovered_order[second];
                discovered_order[second] = temporary;
            }
        }
    }
    for (int index = 1; index < kNumFeet; ++index) {
        if (report.first_motion_stage[discovered_order[index - 1]] ==
            report.first_motion_stage[discovered_order[index]]) {
            return 1;
        }
    }
    std::printf(
        "callable planner order: seed=%s discovered=%d,%d,%d,%d\n",
        four_step_seed_name(report.seed),
        discovered_order[0], discovered_order[1],
        discovered_order[2], discovered_order[3]);
    for (int foot = 0; foot < kNumFeet; ++foot) {
        std::printf(
            "  schedule audit: foot=%d seed=(%d,%d) optimized=(%d,%d) "
            "timing_change=%d\n",
            foot, report.seed_liftoff_stage[foot],
            report.seed_touchdown_stage[foot],
            report.optimized_liftoff_stage[foot],
            report.optimized_touchdown_stage[foot],
            report.schedule_timing_change[foot]);
        if (report.optimized_liftoff_stage[foot] < 0 ||
            report.optimized_touchdown_stage[foot] < 0) {
            return 1;
        }
    }
    if (plan_path && write_contact_plan(plan, plan_path) != Status::SUCCESS) {
        std::printf("failed to write four-step plan: %s\n", plan_path);
        return 1;
    }
    return 0;
}
