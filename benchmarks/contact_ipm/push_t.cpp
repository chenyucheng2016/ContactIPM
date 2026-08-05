#include <algorithm>
#include <chrono>
#include <cmath>
#include <cstdio>
#include <cstdlib>
#include <memory>

#include "nmpc/contact_ipm.hpp"
#include "trajectory_output.hpp"

using namespace nmpc;

namespace {

constexpr int NX = 3;       // planar pose
constexpr int NU = 17;      // contact point, seven split vars, eight forces
constexpr int NC = 77;      // 34 side rows + 43 generated product rows
#ifndef CONTACT_BENCHMARK_NODES
#define CONTACT_BENCHMARK_NODES 50
#endif
static_assert(CONTACT_BENCHMARK_NODES >= 2,
              "Push T requires at least two nodes");
constexpr int HORIZON = CONTACT_BENCHMARK_NODES - 1;
#ifndef CONTACT_BENCHMARK_DT
#define CONTACT_BENCHMARK_DT 0.05
#endif
static_assert(CONTACT_BENCHMARK_DT > 0.0,
              "Push T requires a positive time step");
constexpr double DT = CONTACT_BENCHMARK_DT;
constexpr double UNIT = 0.05;
constexpr double MASS = 1.0;
constexpr double FRICTION = 0.4;
constexpr double GRAVITY = 9.8;
constexpr double RADIUS = 2.8 * UNIT;
constexpr double ROTATION_SCALE = 0.4;
constexpr double DC = 2.6429;
constexpr double TARGET_X = 0.01;
constexpr double TARGET_Y = 0.01;
constexpr double TARGET_ANGLE = 0.01;
constexpr double PI = 3.14159265358979323846;

using Problem = NMPCProblem<NX, NU, NC, HORIZON>;

inline int v_index(int contact) { return 2 + contact; }
inline int lambda_index(int contact) { return 9 + contact; }

struct PushTDynamics final : DynamicsModel<NX, NU> {
    Status discrete_step(const Vec<NX>& x, const Vec<NU>& u, double dt,
                         Vec<NX>& next) override {
        double odd_sum = 0.0, even_sum = 0.0;
        for (int i = 0; i < 8; ++i) {
            if ((i & 1) == 0) odd_sum += u[lambda_index(i)];
            else              even_sum += u[lambda_index(i)];
        }
        const double ctheta = std::cos(x[2]);
        const double stheta = std::sin(x[2]);
        const double inverse_drag = 1.0 / (FRICTION * MASS * GRAVITY);
        const double vx = inverse_drag *
            (ctheta * even_sum - stheta * odd_sum);
        const double vy = inverse_drag *
            (stheta * even_sum + ctheta * odd_sum);
        const double omega =
            (-u[1] * even_sum + u[0] * odd_sum) /
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
        double odd_sum = 0.0, even_sum = 0.0;
        for (int i = 0; i < 8; ++i) {
            if ((i & 1) == 0) odd_sum += u[lambda_index(i)];
            else              even_sum += u[lambda_index(i)];
        }
        const double ctheta = std::cos(x[2]);
        const double stheta = std::sin(x[2]);
        const double translation =
            dt / (FRICTION * MASS * GRAVITY);
        const double rotation =
            dt / (FRICTION * MASS * GRAVITY *
                  ROTATION_SCALE * RADIUS);

        Hxx(2, 2) += translation * (
            p[0] * (-ctheta * even_sum + stheta * odd_sum) +
            p[1] * (-stheta * even_sum - ctheta * odd_sum));
        for (int i = 0; i < 8; ++i) {
            const int force = lambda_index(i);
            if ((i & 1) == 0) {
                Hux(force, 2) += translation *
                    (-p[0] * ctheta - p[1] * stheta);
                Huu(force, 0) += p[2] * rotation;
                Huu(0, force) += p[2] * rotation;
            } else {
                Hux(force, 2) += translation *
                    (-p[0] * stheta + p[1] * ctheta);
                Huu(force, 1) -= p[2] * rotation;
                Huu(1, force) -= p[2] * rotation;
            }
        }
        return Status::SUCCESS;
    }
};

struct PushTCost final : CostModel<NX, NU> {
    double tracking_scale = 1.0;
    double effort_scale = 1.0;

    double pose_cost(const Vec<NX>& x, double weight) const {
        const double dx = x[0] - TARGET_X;
        const double dy = x[1] - TARGET_Y;
        const double da = x[2] - TARGET_ANGLE;
        return tracking_scale * weight *
            (dx * dx + dy * dy + 2.0 - 2.0 * std::cos(da));
    }

    double stage_cost(const Vec<NX>& x, const Vec<NU>& u, int) override {
        double value = pose_cost(x, 1.0);
        for (int i = 0; i < 8; ++i)
            value += 0.01 * effort_scale *
                u[lambda_index(i)] * u[lambda_index(i)];
        return value;
    }

    double terminal_cost(const Vec<NX>& x) override {
        return pose_cost(x, 100.0);
    }

    Status stage_gradient(const Vec<NX>& x, const Vec<NU>& u, int,
                          Vec<NX>& qx, Vec<NU>& qu) override {
        qx.zero();
        qu.zero();
        qx[0] = 2.0 * tracking_scale * (x[0] - TARGET_X);
        qx[1] = 2.0 * tracking_scale * (x[1] - TARGET_Y);
        qx[2] = 2.0 * tracking_scale *
            std::sin(x[2] - TARGET_ANGLE);
        for (int i = 0; i < 8; ++i)
            qu[lambda_index(i)] =
                0.02 * effort_scale * u[lambda_index(i)];
        return Status::SUCCESS;
    }

    Status stage_hessian(const Vec<NX>&, const Vec<NU>&, int,
                         Mat<NX, NX>& Qxx, Mat<NU, NU>& Quu,
                         Mat<NU, NX>& Qux) override {
        Qxx.zero();
        Quu.zero();
        Qux.zero();
        Qxx(0, 0) = 2.0 * tracking_scale;
        Qxx(1, 1) = 2.0 * tracking_scale;
        Qxx(2, 2) = 2.0 * tracking_scale;
        for (int i = 0; i < 8; ++i)
            Quu(lambda_index(i), lambda_index(i)) =
                0.02 * effort_scale;
        return Status::SUCCESS;
    }

    Status terminal_gradient(const Vec<NX>& x, Vec<NX>& qx) override {
        qx.zero();
        qx[0] = 200.0 * tracking_scale * (x[0] - TARGET_X);
        qx[1] = 200.0 * tracking_scale * (x[1] - TARGET_Y);
        qx[2] = 200.0 * tracking_scale *
            std::sin(x[2] - TARGET_ANGLE);
        return Status::SUCCESS;
    }

    Status terminal_hessian(const Vec<NX>&,
                            Mat<NX, NX>& Qxx) override {
        Qxx.zero();
        Qxx(0, 0) = 200.0 * tracking_scale;
        Qxx(1, 1) = 200.0 * tracking_scale;
        Qxx(2, 2) = 200.0 * tracking_scale;
        return Status::SUCCESS;
    }
};

struct PushTContacts final : ConstraintModel<NX, NU, NC> {
    bool provides_adjoint_hessian() const override { return true; }

    int num_constraints(int k) const override { return k < HORIZON ? 34 : 0; }
    int num_complementarity_pairs(int k) const override {
        return k < HORIZON ? 43 : 0;
    }

    bool complementarity_pair(int k, int pair, int& first,
                              int& second) const override {
        if (k >= HORIZON || pair < 0 || pair >= 43) return false;
        if (pair < 7) {
            first = 2 * pair;
            second = 2 * pair + 1;
            return true;
        }
        if (pair < 15) {
            first = 18 + pair - 7;
            second = 26 + pair - 7;
            return true;
        }
        int index = pair - 15;
        for (int i = 0; i < 8; ++i) {
            const int count = 7 - i;
            if (index < count) {
                first = 18 + i;
                second = 18 + i + 1 + index;
                return true;
            }
            index -= count;
        }
        return false;
    }

    void split_values(const Vec<NU>& u, double v[7], double w[7]) const {
        const double h[7] = {
            u[0] - 2.0 * UNIT,
            u[1] - (4.0 - DC) * UNIT,
            u[1] - (3.0 - DC) * UNIT,
            u[0] - 0.5 * UNIT,
            u[1] + DC * UNIT,
            u[0] + 0.5 * UNIT,
            u[0] + 2.0 * UNIT};
        for (int i = 0; i < 7; ++i) {
            v[i] = u[v_index(i)];
            w[i] = v[i] - h[i];
        }
    }

    void gap_values(const Vec<NU>& u, double gap[8]) const {
        double v[7], w[7];
        split_values(u, v, w);
        gap[0] = (4.0 - DC) * UNIT - u[1];
        gap[1] = v[0] + w[0] + v[1] + w[1] + v[2] + w[2] - UNIT;
        gap[2] = v[0] + w[0] + v[2] + w[2] + v[3] + w[3] - 1.5 * UNIT;
        gap[3] = v[2] + w[2] + v[3] + w[3] + v[4] + w[4] - 3.0 * UNIT;
        gap[4] = v[3] + w[3] + v[4] + w[4] + v[5] + w[5] - UNIT;
        gap[5] = v[2] + w[2] + v[4] + w[4] + v[5] + w[5] - 3.0 * UNIT;
        gap[6] = v[2] + w[2] + v[5] + w[5] + v[6] + w[6] - 1.5 * UNIT;
        gap[7] = v[1] + w[1] + v[2] + w[2] + v[6] + w[6] - UNIT;
    }

    Status evaluate(const Vec<NX>&, const Vec<NU>& u, int,
                    Vec<NC>& rows) override {
        rows.zero();
        double v[7], w[7];
        split_values(u, v, w);
        for (int i = 0; i < 7; ++i) {
            rows[2 * i] = -v[i];
            rows[2 * i + 1] = -w[i];
        }
        rows[14] = -u[0] - 2.0 * UNIT;
        rows[15] = u[0] - 2.0 * UNIT;
        rows[16] = -u[1] - DC * UNIT;
        rows[17] = u[1] - (4.0 - DC) * UNIT;
        const double force_sign[8] = {-1.0, -1.0, 1.0, -1.0,
                                       1.0, 1.0, 1.0, 1.0};
        for (int i = 0; i < 8; ++i)
            rows[18 + i] = -force_sign[i] * u[lambda_index(i)];
        double gap[8];
        gap_values(u, gap);
        for (int i = 0; i < 8; ++i) rows[26 + i] = -gap[i];
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
        const int coordinate[7] = {0, 1, 1, 0, 1, 0, 0};
        for (int i = 0; i < 7; ++i) {
            Cu(2 * i, v_index(i)) = -1.0;
            Cu(2 * i + 1, v_index(i)) = -1.0;
            Cu(2 * i + 1, coordinate[i]) = 1.0;
        }
        Cu(14, 0) = -1.0;
        Cu(15, 0) = 1.0;
        Cu(16, 1) = -1.0;
        Cu(17, 1) = 1.0;
        const double force_sign[8] = {-1.0, -1.0, 1.0, -1.0,
                                       1.0, 1.0, 1.0, 1.0};
        for (int i = 0; i < 8; ++i)
            Cu(18 + i, lambda_index(i)) = -force_sign[i];
        Cu(26, 1) = 1.0;
        const int memberships[7][8] = {
            {0,1,1,0,0,0,0,0}, {0,1,0,0,0,0,0,1},
            {0,1,1,1,0,1,1,1}, {0,0,1,1,1,0,0,0},
            {0,0,0,1,1,1,0,0}, {0,0,0,0,1,1,1,0},
            {0,0,0,0,0,0,1,1}};
        for (int contact = 0; contact < 7; ++contact)
            for (int gap = 1; gap < 8; ++gap)
                if (memberships[contact][gap]) {
                    Cu(26 + gap, v_index(contact)) = -2.0;
                    Cu(26 + gap, coordinate[contact]) += 1.0;
                }
        return Status::SUCCESS;
    }

    Status jacobian_terminal(const Vec<NX>&,
                             Mat<NC, NX>& Cx) override {
        Cx.zero();
        return Status::SUCCESS;
    }
};

struct PushTMetrics {
    double dynamics = 0.0;
    double side_violation = 0.0;
    double physical_mpcc = 0.0;
    double position_error = 0.0;
    double angular_error = 0.0;
};

PushTMetrics evaluate_metrics(const Problem& problem,
                              PushTDynamics& dynamics,
                              PushTContacts& contacts) {
    PushTMetrics metrics;
    const Vec<NX>& terminal = problem.stages[HORIZON].x;
    metrics.position_error = std::hypot(
        terminal[0] - TARGET_X, terminal[1] - TARGET_Y);
    const double angle_delta = terminal[2] - TARGET_ANGLE;
    metrics.angular_error = std::fabs(
        std::atan2(std::sin(angle_delta), std::cos(angle_delta)));
    for (int k = 0; k < HORIZON; ++k) {
        Vec<NX> predicted;
        dynamics.discrete_step(problem.stages[k].x,
                               problem.stages[k].u, DT, predicted);
        for (int i = 0; i < NX; ++i)
            metrics.dynamics = std::max(
                metrics.dynamics,
                std::fabs(predicted[i] - problem.stages[k + 1].x[i]));
        Vec<NC> rows;
        contacts.evaluate(problem.stages[k].x,
                          problem.stages[k].u, k, rows);
        for (int j = 0; j < 34; ++j)
            metrics.side_violation =
                std::max(metrics.side_violation, rows[j]);
        for (int p = 0; p < 43; ++p) {
            int first = -1, second = -1;
            contacts.complementarity_pair(k, p, first, second);
            metrics.physical_mpcc = std::max(
                metrics.physical_mpcc,
                std::fabs(rows[first] * rows[second]));
        }
    }
    return metrics;
}

bool passes_task(const PushTMetrics& metrics,
                 const ContactIPMParams& params) {
    return metrics.dynamics <= params.tol_primal &&
           metrics.side_violation <= params.tol_ineq &&
           metrics.physical_mpcc <= params.tol_mpcc &&
           metrics.position_error < 0.1 &&
           metrics.angular_error < PI / 6.0;
}

} // namespace

int main(int argc, char** argv) {
    constexpr int NUM_SEGMENTS = 50;
    constexpr double RADIUS_MIN = 0.25;
    constexpr double RADIUS_MAX = 0.5;
    int segment_begin = 0;
    int segment_end = NUM_SEGMENTS;
    if (argc == 2) {
        const int requested = std::atoi(argv[1]);
        if (requested < 0 || requested >= NUM_SEGMENTS) return 2;
        segment_begin = requested;
        segment_end = requested + 1;
    } else if (argc != 1) {
        return 2;
    }
    auto dynamics = std::make_unique<PushTDynamics>();
    auto cost = std::make_unique<PushTCost>();
    auto contacts = std::make_unique<PushTContacts>();
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

    ContactIPMParams params;
    params.mu_init = 1.0;
    params.mu_min = 1e-4;
    params.mu_conv_threshold = 1e-5;
    params.tol_primal = 1e-5;
    params.tol_compl = 1e-5;
    params.tol_ineq = 1e-8;
    params.tol_stat = 1e-3;
    params.tol_mpcc = 1e-5;
    params.max_same_mu = 30;
    params.max_iters = 150;
    params.enable_preconditioner = true;
    params.exact_hessian = true;
    if (const char* value = std::getenv("CONTACTIPM_EXACT_HESSIAN"))
        params.exact_hessian = std::atoi(value) != 0;
    if (const char* value = std::getenv("CONTACTIPM_PRECONDITIONER"))
        params.enable_preconditioner = std::atoi(value) != 0;
    if (const char* value = std::getenv("CONTACTIPM_MPCC_RECOVERY_MU"))
        params.mpcc_recovery_mu = std::atof(value);
    bool enable_recovery = true;
    if (const char* value = std::getenv("CONTACTIPM_MPCC_RECOVERY"))
        enable_recovery = std::atoi(value) != 0;
    params.verbosity = 0;
    if (const char* value = std::getenv("CONTACTIPM_VERBOSITY"))
        params.verbosity = std::atoi(value);

    int successful_segments = 0;
    int total_iterations = 0;
    double worst_dynamics = 0.0;
    double worst_side_violation = 0.0;
    double worst_physical_mpcc = 0.0;
    double worst_position_error = 0.0;
    double worst_angular_error = 0.0;
    const auto suite_start = std::chrono::steady_clock::now();

    for (int segment = segment_begin; segment < segment_end; ++segment) {
        const double fraction =
            static_cast<double>(segment) / (NUM_SEGMENTS - 1);
        const double angle = 2.0 * PI * segment / NUM_SEGMENTS;
        const double radius =
            RADIUS_MIN + (RADIUS_MAX - RADIUS_MIN) * fraction;

        auto problem = std::make_unique<Problem>();
        problem->dynamics = dynamics.get();
        problem->cost = cost.get();
        problem->constraints = contacts.get();
        problem->dt = DT;
        problem->init_bounds_free();
        problem->x0.zero();
        problem->x0[0] = radius * std::cos(angle);
        problem->x0[1] = radius * std::sin(angle);
        problem->x0[2] = angle;
        for (int k = 0; k <= HORIZON; ++k) {
            problem->stages[k].x.zero();
            problem->stages[k].u.zero();
        }
        problem->stages[0].x = problem->x0;

        auto solver =
            std::make_unique<ContactIPM<NX, NU, NC, HORIZON>>();
        if (const char* path = std::getenv("CONTACTIPM_DIAG_CSV"))
            solver->enable_diag_csv(path);
        Status status = solver->configure(params);
        if (status != Status::SUCCESS) return 2;
        status = enable_recovery
            ? solver->solve_mpcc_with_recovery(*problem)
            : solver->solve(*problem);
        const int segment_iterations = solver->last_stats().inner_iterations;
        PushTMetrics metrics =
            evaluate_metrics(*problem, *dynamics, *contacts);
        const SolverStats& stats = solver->last_stats();
        total_iterations += segment_iterations;
        const bool segment_success =
            status == Status::SUCCESS && passes_task(metrics, params);
        auto trajectory =
            contact_benchmark::trajectory_file("contactipm", "push_t", segment);
        if (trajectory) {
            for (int k = 0; k <= HORIZON; ++k) {
                const auto& stage = problem->stages[k];
                const double cx = k < HORIZON ? stage.u[0] : 0.0;
                const double cy = k < HORIZON ? stage.u[1] : 0.0;
                const double h[7] = {
                    cx - 2.0 * UNIT,
                    cy - (4.0 - DC) * UNIT,
                    cy - (3.0 - DC) * UNIT,
                    cx - 0.5 * UNIT,
                    cy + DC * UNIT,
                    cx + 0.5 * UNIT,
                    cx + 2.0 * UNIT};
                trajectory << stage.x[0] << ' ' << stage.x[1] << ' '
                           << stage.x[2] << ' ' << cx << ' ' << cy;
                for (int i = 0; i < 7; ++i) {
                    const double v =
                        k < HORIZON ? stage.u[v_index(i)]
                                    : std::max(h[i], 0.0);
                    trajectory << ' ' << v << ' ' << v - h[i];
                }
                for (int i = 0; i < 8; ++i)
                    trajectory << ' '
                               << (k < HORIZON
                                       ? stage.u[lambda_index(i)] : 0.0);
                trajectory << ' ' << std::cos(stage.x[2]) << ' '
                           << std::sin(stage.x[2]) << '\n';
            }
        }
        if (segment_success) ++successful_segments;
        worst_dynamics = std::max(worst_dynamics, metrics.dynamics);
        worst_side_violation =
            std::max(worst_side_violation, metrics.side_violation);
        worst_physical_mpcc =
            std::max(worst_physical_mpcc, metrics.physical_mpcc);
        worst_position_error =
            std::max(worst_position_error, metrics.position_error);
        worst_angular_error =
            std::max(worst_angular_error, metrics.angular_error);
        std::printf(
            "segment %02d: status=%s task=%s primal=%.2e stat=%.2e "
            "mpcc=%.2e pos=%.4f angle=%.4f\n",
            segment, status_string(status),
            segment_success ? "yes" : "no", stats.primal_infeas,
            stats.dual_infeas,
            metrics.physical_mpcc, metrics.position_error,
            metrics.angular_error);
    }

    const auto suite_stop = std::chrono::steady_clock::now();
    const double suite_seconds =
        std::chrono::duration<double>(suite_stop - suite_start).count();
    const int attempted_segments = segment_end - segment_begin;
    std::printf("\n=== PUSH T SOURCE INITIAL CONDITIONS ===\n");
    std::printf("Successful segments: %d/%d\n",
                successful_segments, attempted_segments);
    std::printf("Total solve time:    %.6f s\n", suite_seconds);
    std::printf("Total iterations:    %d\n", total_iterations);
    std::printf("Worst dynamics:      %.3e\n", worst_dynamics);
    std::printf("Worst side violation:%.3e\n", worst_side_violation);
    std::printf("Worst physical MPCC: %.3e\n", worst_physical_mpcc);
    std::printf("Worst position error:%.6f\n", worst_position_error);
    std::printf("Worst angular error: %.6f\n", worst_angular_error);
    return successful_segments == attempted_segments ? 0 : 1;
}