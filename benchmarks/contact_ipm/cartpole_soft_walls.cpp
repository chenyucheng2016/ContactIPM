#include <algorithm>
#include <chrono>
#include <cmath>
#include <cstdio>
#include <cstdlib>
#include <fstream>
#include <memory>
#include <string>
#include <vector>

#include "nmpc/contact_ipm.hpp"
#include "case_overrides.hpp"
#include "trajectory_output.hpp"

#ifndef NMPC_SOURCE_DIR
#define NMPC_SOURCE_DIR "."
#endif

using namespace nmpc;

namespace {

constexpr int NX = 4;
constexpr int NU = 3;
constexpr int NC = 6;       // 4 side rows + 2 solver-generated product rows
constexpr int HORIZON = 99; // CRISP uses 100 nodes
constexpr double DT = 0.02;
constexpr double MC = 1.0;
constexpr double MP = 0.1;
constexpr double LENGTH = 0.8;
constexpr double GRAVITY = 9.8;
constexpr double WALL_LEFT = 1.0;
constexpr double WALL_RIGHT = 1.0;
constexpr double STIFF_LEFT = 200.0;
constexpr double STIFF_RIGHT = 200.0;

using Problem = NMPCProblem<NX, NU, NC, HORIZON>;

struct CartpoleDynamics final : DynamicsModel<NX, NU> {
    Status discrete_step(const Vec<NX>& x, const Vec<NU>& u, double dt,
                         Vec<NX>& next) override {
        const double position = x[0];
        const double angle = x[1];
        const double velocity = x[2];
        const double angular_velocity = x[3];
        const double force = u[0];
        const double lambda_left = u[1];
        const double lambda_right = u[2];
        const double c = std::cos(angle);
        const double s = std::sin(angle);
        const double denominator = -MP * c * c + MC + MP;

        const double acceleration =
            (lambda_right - lambda_left + force + lambda_left * c * c -
             lambda_right * c * c - GRAVITY * MP * c * s +
             LENGTH * MP * angular_velocity * angular_velocity * s) /
            denominator;
        const double angular_acceleration =
            -(lambda_left * MC * c - lambda_right * MC * c + MP * force * c -
              GRAVITY * MP * MP * s - GRAVITY * MC * MP * s +
              LENGTH * MP * MP * angular_velocity * angular_velocity * c * s) /
            (LENGTH * MP * denominator);

        const double velocity_next = velocity + dt * acceleration;
        const double angular_velocity_next =
            angular_velocity + dt * angular_acceleration;
        next[0] = position + dt * velocity_next;
        next[1] = angle + dt * angular_velocity_next;
        next[2] = velocity_next;
        next[3] = angular_velocity_next;
        return Status::SUCCESS;
    }

    Status linearize(const Vec<NX>& x, const Vec<NU>& u, double dt,
                     Mat<NX, NX>& A, Mat<NX, NU>& B) override {
        // Central differences keep this benchmark independent of an AD package.
        constexpr double eps = 1e-6;
        for (int j = 0; j < NX; ++j) {
            Vec<NX> xp = x, xm = x, fp, fm;
            xp[j] += eps;
            xm[j] -= eps;
            discrete_step(xp, u, dt, fp);
            discrete_step(xm, u, dt, fm);
            for (int i = 0; i < NX; ++i)
                A(i, j) = (fp[i] - fm[i]) / (2.0 * eps);
        }
        for (int j = 0; j < NU; ++j) {
            Vec<NU> up = u, um = u;
            Vec<NX> fp, fm;
            up[j] += eps;
            um[j] -= eps;
            discrete_step(x, up, dt, fp);
            discrete_step(x, um, dt, fm);
            for (int i = 0; i < NX; ++i)
                B(i, j) = (fp[i] - fm[i]) / (2.0 * eps);
        }
        return Status::SUCCESS;
    }
};

struct CartpoleCost final : CostModel<NX, NU> {
    Vec<NX> target;

    CartpoleCost() { target.zero(); }

    double stage_cost(const Vec<NX>&, const Vec<NU>& u, int) override {
        return 0.001 * u[0] * u[0];
    }

    double terminal_cost(const Vec<NX>& x) override {
        double value = 0.0;
        for (int i = 0; i < NX; ++i) {
            const double error = x[i] - target[i];
            value += 100.0 * error * error;
        }
        return value;
    }

    Status stage_gradient(const Vec<NX>&, const Vec<NU>& u, int,
                          Vec<NX>& qx, Vec<NU>& qu) override {
        qx.zero();
        qu.zero();
        qu[0] = 0.002 * u[0];
        return Status::SUCCESS;
    }

    Status stage_hessian(const Vec<NX>&, const Vec<NU>&, int,
                         Mat<NX, NX>& Qxx, Mat<NU, NU>& Quu,
                         Mat<NU, NX>& Qux) override {
        Qxx.zero();
        Quu.zero();
        Qux.zero();
        Quu(0, 0) = 0.002;
        return Status::SUCCESS;
    }

    Status terminal_gradient(const Vec<NX>& x, Vec<NX>& qx) override {
        for (int i = 0; i < NX; ++i) qx[i] = 200.0 * (x[i] - target[i]);
        return Status::SUCCESS;
    }

    Status terminal_hessian(const Vec<NX>&, Mat<NX, NX>& Qxx) override {
        Qxx.zero();
        for (int i = 0; i < NX; ++i) Qxx(i, i) = 200.0;
        return Status::SUCCESS;
    }
};

struct CartpoleContacts final : ConstraintModel<NX, NU, NC> {
    int num_constraints(int k) const override { return k < HORIZON ? 4 : 0; }

    int num_complementarity_pairs(int k) const override {
        return k < HORIZON ? 2 : 0;
    }

    bool complementarity_pair(int k, int pair, int& first,
                              int& second) const override {
        if (k >= HORIZON || pair < 0 || pair > 1) return false;
        first = pair;
        second = pair + 2;
        return true;
    }

    Status evaluate(const Vec<NX>& x, const Vec<NU>& u, int,
                    Vec<NC>& rows) override {
        rows.zero();
        const double pole_x = LENGTH * std::sin(x[1]);
        const double gap_left =
            WALL_LEFT - x[0] - pole_x + u[1] / STIFF_LEFT;
        const double gap_right =
            WALL_RIGHT + x[0] + pole_x + u[2] / STIFF_RIGHT;
        rows[0] = -u[1];
        rows[1] = -u[2];
        rows[2] = -gap_left;
        rows[3] = -gap_right;
        return Status::SUCCESS;
    }

    Status evaluate_terminal(const Vec<NX>&, Vec<NC>& rows) override {
        rows.zero();
        return Status::SUCCESS;
    }

    Status jacobian(const Vec<NX>& x, const Vec<NU>&, int,
                    Mat<NC, NX>& Cx, Mat<NC, NU>& Cu) override {
        Cx.zero();
        Cu.zero();
        Cu(0, 1) = -1.0;
        Cu(1, 2) = -1.0;
        Cx(2, 0) = 1.0;
        Cx(2, 1) = LENGTH * std::cos(x[1]);
        Cu(2, 1) = -1.0 / STIFF_LEFT;
        Cx(3, 0) = -1.0;
        Cx(3, 1) = -LENGTH * std::cos(x[1]);
        Cu(3, 2) = -1.0 / STIFF_RIGHT;
        return Status::SUCCESS;
    }

    Status jacobian_terminal(const Vec<NX>&,
                             Mat<NC, NX>& Cx) override {
        Cx.zero();
        return Status::SUCCESS;
    }
};

bool load_crisp_guess(const std::string& filename, Problem& problem) {
    std::ifstream input(filename);
    if (!input) return false;
    std::vector<double> values;
    double value = 0.0;
    while (input >> value) values.push_back(value);
    constexpr int width = NX + NU;
    if (values.size() != static_cast<std::size_t>((HORIZON + 1) * width))
        return false;
    for (int k = 0; k <= HORIZON; ++k) {
        for (int i = 0; i < NX; ++i)
            problem.stages[k].x[i] = values[k * width + i];
        if (k < HORIZON)
            for (int i = 0; i < NU; ++i)
                problem.stages[k].u[i] = values[k * width + NX + i];
    }
    problem.stages[HORIZON].u.zero();
    problem.x0 = problem.stages[0].x;
    return true;
}

void initialize_passive_guess(Problem& problem, CartpoleDynamics& dynamics) {
    problem.x0.zero();
    problem.x0[1] = 3.14159265358979323846;
    problem.x0[3] = 2.8;
    problem.stages[0].x = problem.x0;
    for (int k = 0; k < HORIZON; ++k) {
        problem.stages[k].u.zero();
        dynamics.discrete_step(problem.stages[k].x, problem.stages[k].u,
                               DT, problem.stages[k + 1].x);
    }
    problem.stages[HORIZON].u.zero();
}

} // namespace

int main(int argc, char** argv) {
    auto dynamics = std::make_unique<CartpoleDynamics>();
    auto cost = std::make_unique<CartpoleCost>();
    auto contacts = std::make_unique<CartpoleContacts>();
    auto problem = std::make_unique<Problem>();
    problem->dynamics = dynamics.get();
    problem->cost = cost.get();
    problem->constraints = contacts.get();
    problem->dt = DT;
    problem->init_bounds_free();

    const std::string default_guess =
        std::string(NMPC_SOURCE_DIR) +
        "/benchmarks/CRISP/src/examples/pushbot/initial_guess_pushbot_example.txt";
    const std::string guess_file = argc > 1 ? argv[1] : default_guess;
    if (!load_crisp_guess(guess_file, *problem)) {
        std::printf("Could not load %s; using passive rollout.\n", guess_file.c_str());
        initialize_passive_guess(*problem, *dynamics);
    }

    contact_benchmark::apply_vector_override(
        "CONTACT_BENCHMARK_INITIAL_STATE", problem->x0);
    problem->stages[0].x = problem->x0;
    contact_benchmark::apply_vector_override(
        "CONTACT_BENCHMARK_TARGET_STATE", cost->target);

    ContactIPMParams params;
    params.mu_init = 0.1;
    params.mu_min = 1e-4; // internally tightened to satisfy tol_mpcc
    params.mu_conv_threshold = 1e-5;
    params.tol_primal = 1e-6;
    params.tol_compl = 1e-6;
    params.tol_ineq = 1e-8;
    params.tol_stat = 1e-3;
    params.tol_mpcc = 1e-5;
    params.mpcc_relaxation_scale = 1.0;
    params.max_same_mu = 8;
    params.max_iters = 300;
    params.enable_preconditioner = true;
    params.exact_hessian = false;
    params.verbosity = 0;
    if (const char* value = std::getenv("CONTACTIPM_CART_START_MU")) {
        params.mu_init = std::atof(value);
        params.s_min_init = params.mu_init;
    }
    if (const char* value = std::getenv("CONTACTIPM_CART_EXACT_HESSIAN"))
        params.exact_hessian = std::atoi(value) != 0;
    if (const char* value = std::getenv("CONTACTIPM_CART_PRECONDITIONER"))
        params.enable_preconditioner = std::atoi(value) != 0;
    if (const char* value = std::getenv("CONTACTIPM_EXACT_HESSIAN"))
        params.exact_hessian = std::atoi(value) != 0;
    if (const char* value = std::getenv("CONTACTIPM_PRECONDITIONER"))
        params.enable_preconditioner = std::atoi(value) != 0;
    if (const char* value = std::getenv("CONTACTIPM_MPCC_RECOVERY_MU"))
        params.mpcc_recovery_mu = std::atof(value);
    bool enable_recovery = true;
    if (const char* value = std::getenv("CONTACTIPM_MPCC_RECOVERY"))
        enable_recovery = std::atoi(value) != 0;

    auto solver = std::make_unique<ContactIPM<NX, NU, NC, HORIZON>>();
    Status status = solver->configure(params);
    if (status != Status::SUCCESS) {
        std::printf("Configuration failed: %s\n", status_string(status));
        return 2;
    }
    if (const char* path = std::getenv("CONTACTIPM_DIAG_CSV"))
        solver->enable_diag_csv(path);

    const auto start = std::chrono::steady_clock::now();
    status = enable_recovery
        ? solver->solve_mpcc_with_recovery(*problem)
        : solver->solve(*problem);
    const auto stop = std::chrono::steady_clock::now();
    const double solve_seconds =
        std::chrono::duration<double>(stop - start).count();
    const SolverStats& stats = solver->last_stats();
    const Vec<NX>& terminal = problem->stages[HORIZON].x;
    const double constraint_violation =
        std::max(stats.primal_infeas, stats.mpcc_complementarity);
    const bool task_success =
        status == Status::SUCCESS && constraint_violation < 1e-5 &&
        std::fabs(terminal[0] - cost->target[0]) < 0.1 &&
        std::fabs(terminal[1] - cost->target[1]) < 3.14159265358979323846 / 6.0 &&
        std::fabs(terminal[2] - cost->target[2]) < 0.5 &&
        std::fabs(terminal[3] - cost->target[3]) < 0.1 * 3.14159265358979323846;
    auto trajectory = contact_benchmark::trajectory_file(
        "contactipm", "cartpole_soft_walls");
    if (trajectory) {
        for (int k = 0; k <= HORIZON; ++k) {
            for (int i = 0; i < NX; ++i)
                trajectory << problem->stages[k].x[i] << ' ';
            for (int i = 0; i < NU; ++i)
                trajectory << problem->stages[k].u[i]
                           << (i + 1 == NU ? '\n' : ' ');
        }
    }

    std::printf("\n=== CARTPOLE WITH SOFT WALLS ===\n");
    std::printf("Status:              %s\n", status_string(status));
    std::printf("Task success:        %s\n", task_success ? "yes" : "no");
    std::printf("Solve time:          %.6f s\n", solve_seconds);
    std::printf("Iterations:          %d\n", stats.inner_iterations);
    std::printf("Objective:           %.9e\n", stats.cost);
    std::printf("Primal infeas:       %.3e\n", stats.primal_infeas);
    std::printf("Barrier residual:    %.3e\n", stats.complementarity);
    std::printf("Physical MPCC:       %.3e\n", stats.mpcc_complementarity);
    std::printf("Final theta(mu):     %.3e\n", stats.mpcc_relaxation);
    std::printf("Terminal state:      [%.6f, %.6f, %.6f, %.6f]\n",
                terminal[0], terminal[1], terminal[2], terminal[3]);
    return task_success ? 0 : 1;
}