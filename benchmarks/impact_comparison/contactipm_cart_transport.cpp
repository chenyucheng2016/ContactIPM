#include <algorithm>
#include <chrono>
#include <cmath>
#include <cstdio>
#include <cstdlib>
#include <memory>

#include "nmpc/contact_ipm.hpp"
#include "../contact_ipm/trajectory_output.hpp"

using namespace nmpc;

namespace {

constexpr int NX = 4;
constexpr int NU = 3;  // friction force, active force, retained positive slip
constexpr int NC = 9;  // 6 side rows + 3 generated product rows
constexpr int HORIZON = 300;
constexpr double DT = 0.02;
constexpr double MASS_PAYLOAD = 0.1;
constexpr double MASS_CART = 0.2;
constexpr double FRICTION = 0.2;
constexpr double GRAVITY = 9.81;
constexpr double HALF_CART_LENGTH = 1.0;

using Problem = NMPCProblem<NX, NU, NC, HORIZON>;

double negative_slip(const Vec<NX>& x, const Vec<NU>& u) {
    return u[2] - x[2] + x[3];
}

struct Dynamics final : DynamicsModel<NX, NU> {
    Status discrete_step(const Vec<NX>& x, const Vec<NU>& u, double dt,
                         Vec<NX>& next) override {
        next[0] = x[0] + dt * x[2];
        next[1] = x[1] + dt * x[3];
        next[2] = x[2] + dt * u[0] / MASS_PAYLOAD;
        next[3] = x[3] + dt * (u[1] - u[0]) / MASS_CART;
        return Status::SUCCESS;
    }

    Status linearize(const Vec<NX>&, const Vec<NU>&, double dt,
                     Mat<NX, NX>& A, Mat<NX, NU>& B) override {
        A.zero();
        B.zero();
        for (int i = 0; i < NX; ++i) A(i, i) = 1.0;
        A(0, 2) = dt;
        A(1, 3) = dt;
        B(2, 0) = dt / MASS_PAYLOAD;
        B(3, 0) = -dt / MASS_CART;
        B(3, 1) = dt / MASS_CART;
        return Status::SUCCESS;
    }
};

struct Cost final : CostModel<NX, NU> {
    Vec<NX> target;

    double stage_cost(const Vec<NX>& x, const Vec<NU>& u, int) override {
        const double w = negative_slip(x, u);
        return 1e-6 *
            (u[0] * u[0] + u[1] * u[1] + u[2] * u[2] + w * w);
    }

    double terminal_cost(const Vec<NX>& x) override {
        double value = 0.0;
        for (int i = 0; i < NX; ++i) {
            const double error = x[i] - target[i];
            value += 5000.0 * error * error;
        }
        return value;
    }

    Status stage_gradient(const Vec<NX>& x, const Vec<NU>& u, int,
                          Vec<NX>& qx, Vec<NU>& qu) override {
        qx.zero();
        qu.zero();
        const double w = negative_slip(x, u);
        qx[2] = -2e-6 * w;
        qx[3] = 2e-6 * w;
        qu[0] = 2e-6 * u[0];
        qu[1] = 2e-6 * u[1];
        qu[2] = 2e-6 * (u[2] + w);
        return Status::SUCCESS;
    }

    Status stage_hessian(const Vec<NX>&, const Vec<NU>&, int,
                         Mat<NX, NX>& Qxx, Mat<NU, NU>& Quu,
                         Mat<NU, NX>& Qux) override {
        Qxx.zero();
        Quu.zero();
        Qux.zero();
        Qxx(2, 2) = 2e-6;
        Qxx(3, 3) = 2e-6;
        Qxx(2, 3) = -2e-6;
        Qxx(3, 2) = -2e-6;
        Quu(0, 0) = 2e-6;
        Quu(1, 1) = 2e-6;
        Quu(2, 2) = 4e-6;
        Qux(2, 2) = -2e-6;
        Qux(2, 3) = 2e-6;
        return Status::SUCCESS;
    }

    Status terminal_gradient(const Vec<NX>& x, Vec<NX>& qx) override {
        for (int i = 0; i < NX; ++i)
            qx[i] = 10000.0 * (x[i] - target[i]);
        return Status::SUCCESS;
    }

    Status terminal_hessian(const Vec<NX>&, Mat<NX, NX>& Qxx) override {
        Qxx.zero();
        for (int i = 0; i < NX; ++i) Qxx(i, i) = 10000.0;
        return Status::SUCCESS;
    }
};

struct Contacts final : ConstraintModel<NX, NU, NC> {
    static constexpr int pairs[3][2] = {{0, 1}, {1, 2}, {0, 3}};

    int num_constraints(int k) const override { return k < HORIZON ? 6 : 0; }
    int num_complementarity_pairs(int k) const override {
        return k < HORIZON ? 3 : 0;
    }

    bool complementarity_pair(int k, int pair, int& first,
                              int& second) const override {
        if (k >= HORIZON || pair < 0 || pair >= 3) return false;
        first = pairs[pair][0];
        second = pairs[pair][1];
        return true;
    }

    Status evaluate(const Vec<NX>& x, const Vec<NU>& u, int,
                    Vec<NC>& rows) override {
        rows.zero();
        const double friction_limit =
            FRICTION * MASS_PAYLOAD * GRAVITY;
        rows[0] = -u[2];
        rows[1] = -negative_slip(x, u);
        rows[2] = u[0] - friction_limit;
        rows[3] = -u[0] - friction_limit;
        rows[4] = x[1] - x[0] - HALF_CART_LENGTH;
        rows[5] = x[0] - x[1] - HALF_CART_LENGTH;
        return Status::SUCCESS;
    }

    Status evaluate_terminal(const Vec<NX>&, Vec<NC>& rows) override {
        rows.zero();
        return Status::SUCCESS;
    }

    Status jacobian(const Vec<NX>&, const Vec<NU>&, int,
                    Mat<NC, NX>& Cx, Mat<NC, NU>& Cu) override {
        Cx.zero();
        Cu.zero();
        Cu(0, 2) = -1.0;
        Cx(1, 2) = 1.0;
        Cx(1, 3) = -1.0;
        Cu(1, 2) = -1.0;
        Cu(2, 0) = 1.0;
        Cu(3, 0) = -1.0;
        Cx(4, 0) = -1.0;
        Cx(4, 1) = 1.0;
        Cx(5, 0) = 1.0;
        Cx(5, 1) = -1.0;
        return Status::SUCCESS;
    }

    Status jacobian_terminal(const Vec<NX>&, Mat<NC, NX>& Cx) override {
        Cx.zero();
        return Status::SUCCESS;
    }
};

constexpr int Contacts::pairs[3][2];

bool parse_args(int argc, char** argv, Vec<NX>& x0, Vec<NX>& target) {
    if (argc != 9) {
        std::fprintf(stderr,
            "Usage: %s x0_x1 x0_x2 x0_v1 x0_v2 "
            "goal_x1 goal_x2 goal_v1 goal_v2\n",
            argv[0]);
        return false;
    }
    for (int i = 0; i < NX; ++i) {
        x0[i] = std::atof(argv[1 + i]);
        target[i] = std::atof(argv[5 + i]);
    }
    return true;
}

}  // namespace

int main(int argc, char** argv) {
    auto dynamics = std::make_unique<Dynamics>();
    auto cost = std::make_unique<Cost>();
    auto contacts = std::make_unique<Contacts>();
    auto problem = std::make_unique<Problem>();
    problem->dynamics = dynamics.get();
    problem->cost = cost.get();
    problem->constraints = contacts.get();
    problem->dt = DT;
    problem->init_bounds_free();
    if (!parse_args(argc, argv, problem->x0, cost->target)) return 2;

    double initialization_error = 0.0;
    double initial_control_max = 0.0;
    double initial_reconstructed_w_max = 0.0;
    for (int k = 0; k <= HORIZON; ++k) {
        problem->stages[k].x = problem->x0;
        problem->stages[k].u.zero();
        for (int i = 0; i < NX; ++i)
            initialization_error = std::max(
                initialization_error,
                std::fabs(problem->stages[k].x[i] - problem->x0[i]));
        for (int i = 0; i < NU; ++i)
            initial_control_max =
                std::max(initial_control_max, std::fabs(problem->stages[k].u[i]));
        initial_reconstructed_w_max = std::max(
            initial_reconstructed_w_max,
            std::fabs(negative_slip(
                problem->stages[k].x, problem->stages[k].u)));
    }

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
    params.max_iters = 400;
    params.enable_preconditioner = true;
    params.exact_hessian = false;
    params.verbosity = 0;

    auto solver = std::make_unique<ContactIPM<NX, NU, NC, HORIZON>>();
    Status status = solver->configure(params);
    if (status != Status::SUCCESS) return 2;

    const auto start = std::chrono::steady_clock::now();
    status = solver->solve_mpcc_with_recovery(*problem);
    const auto stop = std::chrono::steady_clock::now();
    const double seconds = std::chrono::duration<double>(stop - start).count();
    const SolverStats& stats = solver->last_stats();

    auto trajectory =
        contact_benchmark::trajectory_file("contactipm", "impact_cart_transport");
    if (trajectory) {
        for (int k = 0; k <= HORIZON; ++k) {
            const auto& stage = problem->stages[k];
            for (int i = 0; i < NX; ++i) {
                if (i) trajectory << ' ';
                trajectory << stage.x[i];
            }
            const double f = k < HORIZON ? stage.u[0] : 0.0;
            const double active = k < HORIZON ? stage.u[1] : 0.0;
            const double v = k < HORIZON ? stage.u[2] : 0.0;
            const double w =
                k < HORIZON ? negative_slip(stage.x, stage.u) : 0.0;
            trajectory << ' ' << f << ' ' << active
                       << ' ' << v << ' ' << w << '\n';
        }
    }

    const Vec<NX>& terminal = problem->stages[HORIZON].x;
    const double position_error = std::hypot(
        terminal[0] - cost->target[0], terminal[1] - cost->target[1]);
    const double velocity_error = std::hypot(
        terminal[2] - cost->target[2], terminal[3] - cost->target[3]);

    std::printf("\n=== CONTACTIPM ON IMPACT CART TRANSPORT ===\n");
    std::printf("Status:                    %s\n", status_string(status));
    std::printf("Solve time:                %.9f s\n", seconds);
    std::printf("Iterations:                %d\n", stats.inner_iterations);
    std::printf("Objective:                 %.12e\n", stats.cost);
    std::printf("Primal infeasibility:      %.3e\n", stats.primal_infeas);
    std::printf("Physical complementarity: %.3e\n",
                stats.mpcc_complementarity);
    std::printf("Position error:            %.9e\n", position_error);
    std::printf("Velocity error:            %.9e\n", velocity_error);
    std::printf("Initialization state error:%.3e\n", initialization_error);
    std::printf("Initialization control max:%.3e\n", initial_control_max);
    std::printf("Initial reconstructed w max:%.3e\n",
                initial_reconstructed_w_max);
    return status == Status::SUCCESS ? 0 : 1;
}
