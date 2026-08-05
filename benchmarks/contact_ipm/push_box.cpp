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

using namespace nmpc;

namespace {

constexpr int NX = 3;
constexpr int NU = 6;
constexpr int NC = 18;      // 8 side rows + 10 solver-generated product rows
#ifndef CONTACT_BENCHMARK_NODES
#define CONTACT_BENCHMARK_NODES 100
#endif
static_assert(CONTACT_BENCHMARK_NODES >= 2,
              "Push Box requires at least two nodes");
constexpr int HORIZON = CONTACT_BENCHMARK_NODES - 1;
#ifndef CONTACT_BENCHMARK_DT
#define CONTACT_BENCHMARK_DT 0.02
#endif
static_assert(CONTACT_BENCHMARK_DT > 0.0,
              "Push Box requires a positive time step");
constexpr double DT = CONTACT_BENCHMARK_DT;
constexpr double HALF_LENGTH = 0.5;
constexpr double HALF_WIDTH = 0.25;
constexpr double MASS = 1.0;
constexpr double FRICTION = 0.5;
constexpr double GRAVITY = 9.8;
constexpr double ROTATION_SCALE = 0.4;
constexpr double RADIUS =
    0.55901699437494742410; // sqrt(HALF_LENGTH^2 + HALF_WIDTH^2)
constexpr double PI = 3.14159265358979323846;

using Problem = NMPCProblem<NX, NU, NC, HORIZON>;

struct PushBoxDynamics final : DynamicsModel<NX, NU> {
    Status discrete_step(const Vec<NX>& x, const Vec<NU>& u, double dt,
                         Vec<NX>& next) override {
        const double theta = x[2];
        const double contact_x = u[0];
        const double contact_y = u[1];
        const double lambda1 = u[2];
        const double lambda2 = u[3];
        const double lambda3 = u[4];
        const double lambda4 = u[5];
        const double c = std::cos(theta);
        const double s = std::sin(theta);
        const double inverse_drag = 1.0 / (FRICTION * MASS * GRAVITY);
        const double vx = inverse_drag *
            (c * (lambda2 + lambda4) - s * (lambda1 + lambda3));
        const double vy = inverse_drag *
            (s * (lambda2 + lambda4) + c * (lambda1 + lambda3));
        const double omega =
            (-contact_y * (lambda2 + lambda4) +
             contact_x * (lambda1 + lambda3)) /
            (FRICTION * MASS * GRAVITY * ROTATION_SCALE * RADIUS);
        next[0] = x[0] + dt * vx;
        next[1] = x[1] + dt * vy;
        next[2] = x[2] + dt * omega;
        return Status::SUCCESS;
    }

    Status linearize(const Vec<NX>& x, const Vec<NU>& u, double dt,
                     Mat<NX, NX>& A, Mat<NX, NU>& B) override {
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

    bool provides_adjoint_hessian() const override { return true; }

    Status adjoint_hessian(const Vec<NX>& x, const Vec<NU>& u, double dt,
                           int, const Vec<NX>& p, Mat<NX, NX>& Hxx,
                           Mat<NU, NX>& Hux,
                           Mat<NU, NU>& Huu) override {
        const double force_x = u[2] + u[4];
        const double force_y = u[3] + u[5];
        const double c = std::cos(x[2]);
        const double s = std::sin(x[2]);
        const double translation = dt / (FRICTION * MASS * GRAVITY);
        const double rotation = dt /
            (FRICTION * MASS * GRAVITY * ROTATION_SCALE * RADIUS);

        Hxx(2, 2) += translation *
            (p[0] * (-c * force_y + s * force_x) +
             p[1] * (-s * force_y - c * force_x));
        for (int force : {2, 4}) {
            Hux(force, 2) += translation * (-p[0] * c - p[1] * s);
            Huu(force, 0) += p[2] * rotation;
            Huu(0, force) += p[2] * rotation;
        }
        for (int force : {3, 5}) {
            Hux(force, 2) += translation * (-p[0] * s + p[1] * c);
            Huu(force, 1) -= p[2] * rotation;
            Huu(1, force) -= p[2] * rotation;
        }
        return Status::SUCCESS;
    }
};

struct PushBoxCost final : CostModel<NX, NU> {
    Vec<NX> target;
    double tracking_scale = 1.0;
    double effort_scale = 1.0;

    PushBoxCost() {
        const double angle = 12.0 * 2.0 * PI / 18.0;
        target[0] = 3.0 * std::cos(angle);
        target[1] = 3.0 * std::sin(angle);
        target[2] = angle;
    }

    double stage_cost(const Vec<NX>&, const Vec<NU>& u, int) override {
        double value = 0.0;
        for (int i = 2; i < NU; ++i)
            value += 0.001 * effort_scale * u[i] * u[i];
        return value;
    }

    double terminal_cost(const Vec<NX>& x) override {
        double value = 0.0;
        for (int i = 0; i < NX; ++i) {
            const double error = x[i] - target[i];
            value += 100.0 * tracking_scale * error * error;
        }
        return value;
    }

    Status stage_gradient(const Vec<NX>&, const Vec<NU>& u, int,
                          Vec<NX>& qx, Vec<NU>& qu) override {
        qx.zero();
        qu.zero();
        for (int i = 2; i < NU; ++i)
            qu[i] = 0.002 * effort_scale * u[i];
        return Status::SUCCESS;
    }

    Status stage_hessian(const Vec<NX>&, const Vec<NU>&, int,
                         Mat<NX, NX>& Qxx, Mat<NU, NU>& Quu,
                         Mat<NU, NX>& Qux) override {
        Qxx.zero();
        Quu.zero();
        Qux.zero();
        for (int i = 2; i < NU; ++i)
            Quu(i, i) = 0.002 * effort_scale;
        return Status::SUCCESS;
    }

    Status terminal_gradient(const Vec<NX>& x, Vec<NX>& qx) override {
        for (int i = 0; i < NX; ++i)
            qx[i] = 200.0 * tracking_scale * (x[i] - target[i]);
        return Status::SUCCESS;
    }

    Status terminal_hessian(const Vec<NX>&, Mat<NX, NX>& Qxx) override {
        Qxx.zero();
        for (int i = 0; i < NX; ++i)
            Qxx(i, i) = 200.0 * tracking_scale;
        return Status::SUCCESS;
    }
};

struct PushBoxContacts final : ConstraintModel<NX, NU, NC> {
    bool provides_adjoint_hessian() const override { return true; }

    static constexpr int pairs[10][2] = {
        {0, 4}, {1, 5}, {2, 6}, {3, 7},
        {0, 1}, {0, 2}, {0, 3}, {1, 2}, {1, 3}, {2, 3}};

    int num_constraints(int k) const override { return k < HORIZON ? 8 : 0; }

    int num_complementarity_pairs(int k) const override {
        return k < HORIZON ? 10 : 0;
    }

    bool complementarity_pair(int k, int pair, int& first,
                              int& second) const override {
        if (k >= HORIZON || pair < 0 || pair >= 10) return false;
        first = pairs[pair][0];
        second = pairs[pair][1];
        return true;
    }

    Status evaluate(const Vec<NX>&, const Vec<NU>& u, int,
                    Vec<NC>& rows) override {
        rows.zero();
        const double contact_x = u[0];
        const double contact_y = u[1];
        rows[0] = -u[2];
        rows[1] = -u[3];
        rows[2] = u[4];
        rows[3] = u[5];
        rows[4] = -contact_y - HALF_WIDTH;
        rows[5] = -contact_x - HALF_LENGTH;
        rows[6] = contact_y - HALF_WIDTH;
        rows[7] = contact_x - HALF_LENGTH;
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
        Cu(1, 3) = -1.0;
        Cu(2, 4) = 1.0;
        Cu(3, 5) = 1.0;
        Cu(4, 1) = -1.0;
        Cu(5, 0) = -1.0;
        Cu(6, 1) = 1.0;
        Cu(7, 0) = 1.0;
        return Status::SUCCESS;
    }

    Status jacobian_terminal(const Vec<NX>&,
                             Mat<NC, NX>& Cx) override {
        Cx.zero();
        return Status::SUCCESS;
    }
};

constexpr int PushBoxContacts::pairs[10][2];

bool load_guess(const std::string& filename, Problem& problem) {
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
    problem.stages[0].x = problem.x0;
    return true;
}

} // namespace

int main(int argc, char** argv) {
    auto dynamics = std::make_unique<PushBoxDynamics>();
    auto cost = std::make_unique<PushBoxCost>();
    auto contacts = std::make_unique<PushBoxContacts>();
    auto problem = std::make_unique<Problem>();
    problem->dynamics = dynamics.get();
    problem->cost = cost.get();
    problem->constraints = contacts.get();
    problem->dt = DT;
    problem->init_bounds_free();
    problem->x0.zero();
    contact_benchmark::apply_vector_override(
        "CONTACT_BENCHMARK_INITIAL_STATE", problem->x0);
    contact_benchmark::apply_vector_override(
        "CONTACT_BENCHMARK_TARGET_STATE", cost->target);
    if (const char* value = std::getenv("CONTACT_BENCHMARK_TRACKING_SCALE"))
        cost->tracking_scale = std::atof(value);
    if (const char* value = std::getenv("CONTACT_BENCHMARK_EFFORT_SCALE"))
        cost->effort_scale = std::atof(value);
    if (!(cost->tracking_scale > 0.0) || !(cost->effort_scale > 0.0) ||
        !std::isfinite(cost->tracking_scale) ||
        !std::isfinite(cost->effort_scale)) {
        std::fprintf(stderr, "objective scales must be positive\n");
        return 2;
    }
    for (int k = 0; k <= HORIZON; ++k) {
        problem->stages[k].x.zero();
        problem->stages[k].u.zero();
    }
    problem->stages[0].x = problem->x0;
    if (argc > 1 && !load_guess(argv[1], *problem)) {
        std::printf("Could not load initial guess: %s\n", argv[1]);
        return 2;
    }

    ContactIPMParams params;
    params.mu_init = 1e-4;
    params.mu_min = 1e-4;
    params.mu_conv_threshold = 1e-5;
    params.tol_primal = 1e-6;
    params.tol_compl = 1e-6;
    params.tol_ineq = 1e-8;
    params.tol_stat = 1e-3;
    params.tol_mpcc = 1e-5;
    params.mpcc_relaxation_scale = 1.0;
    params.max_same_mu = 30;
    params.s_min_init = 1e-4;
    params.max_iters = 200;
    params.enable_preconditioner = true;
    params.exact_hessian = true;
    params.verbosity = 0;
    if (const char* value = std::getenv("CONTACTIPM_PUSH_START_MU")) {
        params.mu_init = std::atof(value);
        params.s_min_init = params.mu_init;
    }
    if (const char* value = std::getenv("CONTACTIPM_PUSH_EXACT_HESSIAN"))
        params.exact_hessian = std::atoi(value) != 0;
    if (const char* value = std::getenv("CONTACTIPM_PUSH_PRECONDITIONER"))
        params.enable_preconditioner = std::atoi(value) != 0;
    if (const char* value = std::getenv("CONTACTIPM_PUSH_RECOVERY_MU"))
        params.mpcc_recovery_mu = std::atof(value);
    bool enable_recovery = true;
    if (const char* value = std::getenv("CONTACTIPM_PUSH_RECOVERY"))
        enable_recovery = std::atoi(value) != 0;
    if (const char* value = std::getenv("CONTACTIPM_EXACT_HESSIAN"))
        params.exact_hessian = std::atoi(value) != 0;
    if (const char* value = std::getenv("CONTACTIPM_PRECONDITIONER"))
        params.enable_preconditioner = std::atoi(value) != 0;
    if (const char* value = std::getenv("CONTACTIPM_MPCC_RECOVERY_MU"))
        params.mpcc_recovery_mu = std::atof(value);
    if (const char* value = std::getenv("CONTACTIPM_MPCC_RECOVERY"))
        enable_recovery = std::atoi(value) != 0;

    auto solver = std::make_unique<ContactIPM<NX, NU, NC, HORIZON>>();
    Status status = solver->configure(params);
    if (status != Status::SUCCESS) return 2;
    if (const char* path = std::getenv("CONTACTIPM_DIAG_CSV"))
        solver->enable_diag_csv(path);
    const auto start = std::chrono::steady_clock::now();
    status = enable_recovery
        ? solver->solve_mpcc_with_recovery(*problem)
        : solver->solve(*problem);
    const auto stop = std::chrono::steady_clock::now();
    const double seconds = std::chrono::duration<double>(stop - start).count();
    const SolverStats& stats = solver->last_stats();

    const Vec<NX>& terminal = problem->stages[HORIZON].x;
    double max_dynamics_defect = 0.0;
    double max_side_violation = 0.0;
    double physical_mpcc = 0.0;
    double objective = cost->terminal_cost(terminal);
    for (int k = 0; k < HORIZON; ++k) {
        Vec<NX> predicted;
        dynamics->discrete_step(problem->stages[k].x, problem->stages[k].u,
                                DT, predicted);
        for (int i = 0; i < NX; ++i)
            max_dynamics_defect = std::max(
                max_dynamics_defect,
                std::fabs(predicted[i] - problem->stages[k + 1].x[i]));
        Vec<NC> rows;
        contacts->evaluate(problem->stages[k].x, problem->stages[k].u,
                           k, rows);
        for (int j = 0; j < 8; ++j)
            max_side_violation = std::max(max_side_violation, rows[j]);
        for (int p = 0; p < 10; ++p)
            physical_mpcc = std::max(
                physical_mpcc,
                std::fabs(rows[PushBoxContacts::pairs[p][0]] *
                          rows[PushBoxContacts::pairs[p][1]]));
        objective += cost->stage_cost(
            problem->stages[k].x, problem->stages[k].u, k);
    }
    double translation_error = 0.0;
    for (int i = 0; i < 2; ++i) {
        const double error = terminal[i] - cost->target[i];
        translation_error += error * error;
    }
    translation_error = std::sqrt(translation_error);
    const double angular_error = std::fabs(terminal[2] - cost->target[2]);
    const bool task_success =
        status == Status::SUCCESS &&
        max_dynamics_defect <= params.tol_primal &&
        max_side_violation <= params.tol_ineq &&
        physical_mpcc <= params.tol_mpcc &&
        translation_error < 0.1 && angular_error < PI / 6.0;
    auto trajectory =
        contact_benchmark::trajectory_file("contactipm", "push_box");
    if (trajectory) {
        for (int k = 0; k <= HORIZON; ++k) {
            for (int i = 0; i < NX; ++i)
                trajectory << problem->stages[k].x[i] << ' ';
            for (int i = 0; i < NU; ++i)
                trajectory << problem->stages[k].u[i]
                           << (i + 1 == NU ? '\n' : ' ');
        }
    }

    std::printf("\n=== PUSH BOX ===\n");
    std::printf("Optimizer status:    %s\n", status_string(status));
    std::printf("Task success:        %s\n", task_success ? "yes" : "no");
    std::printf("Solve time:          %.6f s\n", seconds);
    std::printf("Iterations:          %d total\n", stats.inner_iterations);
    std::printf("Objective:           %.9e\n", objective);
    std::printf("Dynamics defect:     %.3e\n", max_dynamics_defect);
    std::printf("Primal infeas:       %.3e\n", stats.primal_infeas);
    std::printf("Barrier residual:    %.3e\n", stats.complementarity);
    std::printf("Physical MPCC:       %.3e\n", physical_mpcc);
    std::printf("Translation error:   %.6f\n", translation_error);
    std::printf("Angular error:       %.6f\n", angular_error);
    std::printf("Terminal state:      [%.6f, %.6f, %.6f]\n",
                terminal[0], terminal[1], terminal[2]);
    std::printf("Target state:        [%.6f, %.6f, %.6f]\n",
                cost->target[0], cost->target[1], cost->target[2]);
    return task_success ? 0 : 1;
}