#include <algorithm>
#include <chrono>
#include <cmath>
#include <cstdio>
#include <memory>

#include "nmpc/contact_ipm.hpp"
#include "case_overrides.hpp"
#include "trajectory_output.hpp"

using namespace nmpc;

namespace {

constexpr int NX = 4;       // payload/cart positions and velocities
constexpr int NU = 3;       // positive slip, friction force, active force
constexpr int NC = 9;       // 6 side rows + 3 generated product rows
constexpr int HORIZON = 199;
constexpr double DT = 0.02;
constexpr double MASS_PAYLOAD = 1.0;
constexpr double MASS_CART = 2.0;
constexpr double FRICTION = 0.2;
constexpr double GRAVITY = 9.81;
constexpr double HALF_CART_LENGTH = 1.0;

using Problem = NMPCProblem<NX, NU, NC, HORIZON>;

struct TransportDynamics final : DynamicsModel<NX, NU> {
    Status discrete_step(const Vec<NX>& x, const Vec<NU>& u, double dt,
                         Vec<NX>& next) override {
        const double friction_force = u[1];
        const double active_force = u[2];
        const double payload_velocity_next =
            x[2] + dt * friction_force / MASS_PAYLOAD;
        const double cart_velocity_next =
            x[3] + dt * (active_force - friction_force) / MASS_CART;
        next[0] = x[0] + dt * payload_velocity_next;
        next[1] = x[1] + dt * cart_velocity_next;
        next[2] = payload_velocity_next;
        next[3] = cart_velocity_next;
        return Status::SUCCESS;
    }

    Status linearize(const Vec<NX>&, const Vec<NU>&, double dt,
                     Mat<NX, NX>& A, Mat<NX, NU>& B) override {
        A.zero();
        B.zero();
        A(0, 0) = 1.0;
        A(0, 2) = dt;
        A(1, 1) = 1.0;
        A(1, 3) = dt;
        A(2, 2) = 1.0;
        A(3, 3) = 1.0;
        B(0, 1) = dt * dt / MASS_PAYLOAD;
        B(1, 1) = -dt * dt / MASS_CART;
        B(1, 2) = dt * dt / MASS_CART;
        B(2, 1) = dt / MASS_PAYLOAD;
        B(3, 1) = -dt / MASS_CART;
        B(3, 2) = dt / MASS_CART;
        return Status::SUCCESS;
    }
};

struct TransportCost final : CostModel<NX, NU> {
    Vec<NX> target;

    TransportCost() {
        target.zero();
        target[0] = -0.5;
        target[1] = 0.0;
        target[2] = -2.0;
        target[3] = -2.0;
    }

    double stage_cost(const Vec<NX>&, const Vec<NU>& u, int) override {
        return 1e-4 * (u[1] * u[1] + u[2] * u[2]);
    }

    double terminal_cost(const Vec<NX>& x) override {
        const double e0 = x[0] - target[0];
        const double e1 = x[1] - target[1];
        const double e2 = x[2] - target[2];
        const double e3 = x[3] - target[3];
        return 1e5 * (e0 * e0 + e1 * e1) + 10.0 * (e2 * e2 + e3 * e3);
    }

    Status stage_gradient(const Vec<NX>&, const Vec<NU>& u, int,
                          Vec<NX>& qx, Vec<NU>& qu) override {
        qx.zero();
        qu.zero();
        qu[1] = 2e-4 * u[1];
        qu[2] = 2e-4 * u[2];
        return Status::SUCCESS;
    }

    Status stage_hessian(const Vec<NX>&, const Vec<NU>&, int,
                         Mat<NX, NX>& Qxx, Mat<NU, NU>& Quu,
                         Mat<NU, NX>& Qux) override {
        Qxx.zero();
        Quu.zero();
        Qux.zero();
        Quu(1, 1) = 2e-4;
        Quu(2, 2) = 2e-4;
        return Status::SUCCESS;
    }

    Status terminal_gradient(const Vec<NX>& x, Vec<NX>& qx) override {
        qx.zero();
        qx[0] = 2e5 * (x[0] - target[0]);
        qx[1] = 2e5 * (x[1] - target[1]);
        qx[2] = 20.0 * (x[2] - target[2]);
        qx[3] = 20.0 * (x[3] - target[3]);
        return Status::SUCCESS;
    }

    Status terminal_hessian(const Vec<NX>&, Mat<NX, NX>& Qxx) override {
        Qxx.zero();
        Qxx(0, 0) = 2e5;
        Qxx(1, 1) = 2e5;
        Qxx(2, 2) = 20.0;
        Qxx(3, 3) = 20.0;
        return Status::SUCCESS;
    }
};

struct TransportContacts final : ConstraintModel<NX, NU, NC> {
    static constexpr int pairs[3][2] = {{0, 1}, {1, 2}, {0, 3}};

    int num_constraints(int k) const override { return k < HORIZON ? 6 : 2; }
    int num_complementarity_pairs(int k) const override {
        return k < HORIZON ? 3 : 0;
    }

    bool complementarity_pair(int, int pair, int& first,
                              int& second) const override {
        if (pair < 0 || pair >= 3) return false;
        first = pairs[pair][0];
        second = pairs[pair][1];
        return true;
    }

    Status evaluate(const Vec<NX>& x, const Vec<NU>& u, int,
                    Vec<NC>& rows) override {
        rows.zero();
        const double friction_limit =
            FRICTION * MASS_PAYLOAD * GRAVITY;
        const double negative_slip = u[0] - x[2] + x[3];
        rows[0] = -u[0];
        rows[1] = -negative_slip;
        rows[2] = u[1] - friction_limit;
        rows[3] = -u[1] - friction_limit;
        rows[4] = x[1] - x[0] - HALF_CART_LENGTH;
        rows[5] = x[0] - x[1] - HALF_CART_LENGTH;
        return Status::SUCCESS;
    }

    Status evaluate_terminal(const Vec<NX>& x, Vec<NC>& rows) override {
        rows.zero();
        rows[0] = x[1] - x[0] - HALF_CART_LENGTH;
        rows[1] = x[0] - x[1] - HALF_CART_LENGTH;
        return Status::SUCCESS;
    }

    Status jacobian(const Vec<NX>&, const Vec<NU>&, int,
                    Mat<NC, NX>& Cx, Mat<NC, NU>& Cu) override {
        Cx.zero();
        Cu.zero();
        Cu(0, 0) = -1.0;
        Cx(1, 2) = 1.0;
        Cx(1, 3) = -1.0;
        Cu(1, 0) = -1.0;
        Cu(2, 1) = 1.0;
        Cu(3, 1) = -1.0;
        Cx(4, 0) = -1.0;
        Cx(4, 1) = 1.0;
        Cx(5, 0) = 1.0;
        Cx(5, 1) = -1.0;
        return Status::SUCCESS;
    }

    Status jacobian_terminal(const Vec<NX>&,
                             Mat<NC, NX>& Cx) override {
        Cx.zero();
        Cx(0, 0) = -1.0;
        Cx(0, 1) = 1.0;
        Cx(1, 0) = 1.0;
        Cx(1, 1) = -1.0;
        return Status::SUCCESS;
    }
};

constexpr int TransportContacts::pairs[3][2];

} // namespace

int main() {
    auto dynamics = std::make_unique<TransportDynamics>();
    auto cost = std::make_unique<TransportCost>();
    auto contacts = std::make_unique<TransportContacts>();
    auto problem = std::make_unique<Problem>();
    problem->dynamics = dynamics.get();
    problem->cost = cost.get();
    problem->constraints = contacts.get();
    problem->dt = DT;
    problem->init_bounds_free();
    problem->x0.zero();
    problem->x0[0] = 3.5;
    problem->x0[1] = 3.0;
    problem->x0[2] = -4.0;
    problem->x0[3] = -4.0;
    // The frozen CRISP source case appends two zero algebraic contact variables
    // to the four physical states.  ContactIPM stores those variables in u.
    contact_benchmark::apply_vector_prefix_override(
        "CONTACT_BENCHMARK_INITIAL_STATE", problem->x0, 6);
    contact_benchmark::apply_vector_prefix_override(
        "CONTACT_BENCHMARK_TARGET_STATE", cost->target, 6);
    for (int k = 0; k <= HORIZON; ++k) {
        problem->stages[k].x.zero();
        problem->stages[k].u.zero();
    }
    problem->stages[0].x = problem->x0;

    ContactIPMParams params;
    params.mu_init = 0.1;
    params.mu_min = 1e-4;
    params.mu_conv_threshold = 1e-5;
    params.tol_primal = 1e-8;
    params.tol_compl = 1e-6;
    params.tol_ineq = 1e-8;
    params.tol_stat = 1e-3;
    params.tol_mpcc = 1e-5;
    params.max_same_mu = 20;
    params.max_iters = 300;
    params.enable_preconditioner = true;
    params.exact_hessian = false;
    params.verbosity = 0;
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
    if (status != Status::SUCCESS) return 2;
    const auto start = std::chrono::steady_clock::now();
    status = enable_recovery
        ? solver->solve_mpcc_with_recovery(*problem)
        : solver->solve(*problem);
    const auto stop = std::chrono::steady_clock::now();
    const double seconds = std::chrono::duration<double>(stop - start).count();
    const SolverStats& stats = solver->last_stats();
    const int total_iterations = stats.inner_iterations;
    const Vec<NX>& terminal = problem->stages[HORIZON].x;
    const double translation_error = std::sqrt(
        (terminal[0] - cost->target[0]) * (terminal[0] - cost->target[0]) +
        (terminal[1] - cost->target[1]) * (terminal[1] - cost->target[1]));
    const double velocity_error = std::sqrt(
        (terminal[2] - cost->target[2]) * (terminal[2] - cost->target[2]) +
        (terminal[3] - cost->target[3]) * (terminal[3] - cost->target[3]));
    const bool task_success =
        status == Status::SUCCESS && stats.primal_infeas < 1e-5 &&
        stats.mpcc_complementarity < 1e-5 && translation_error < 0.1 &&
        velocity_error < 0.5;
    auto trajectory =
        contact_benchmark::trajectory_file("contactipm", "transport");
    if (trajectory) {
        for (int k = 0; k <= HORIZON; ++k) {
            const auto& stage = problem->stages[k];
            for (int i = 0; i < NX; ++i) trajectory << stage.x[i] << ' ';
            if (k < HORIZON) {
                const double positive_slip = stage.u[0];
                const double negative_slip =
                    positive_slip - stage.x[2] + stage.x[3];
                trajectory << positive_slip << ' ' << negative_slip << ' '
                           << stage.u[1] << ' ' << stage.u[2] << '\n';
            } else {
                trajectory << "0 0 0 0\n";
            }
        }
    }

    std::printf("\n=== TRANSPORT ===\n");
    std::printf("Status:              %s\n", status_string(status));
    std::printf("Task success:        %s\n", task_success ? "yes" : "no");
    std::printf("Solve time:          %.6f s\n", seconds);
    std::printf("Iterations:          %d total (%d final stage)\n",
                total_iterations, stats.inner_iterations);
    std::printf("Objective:           %.9e\n", stats.cost);
    std::printf("Primal infeas:       %.3e\n", stats.primal_infeas);
    std::printf("Stationarity:        %.3e\n", stats.dual_infeas);
    std::printf("Barrier residual:    %.3e\n", stats.complementarity);
    std::printf("Physical MPCC:       %.3e\n", stats.mpcc_complementarity);
    std::printf("Translation error:   %.6f\n", translation_error);
    std::printf("Velocity error:      %.6f\n", velocity_error);
    std::printf("Terminal physical:   [%.6f, %.6f, %.6f, %.6f]\n",
                terminal[0], terminal[1], terminal[2], terminal[3]);
    return task_success ? 0 : 1;
}