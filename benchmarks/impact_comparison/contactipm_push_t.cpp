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

constexpr int NX = 3;
constexpr int NU = 17;  // cx, cy, eight forces, seven retained split variables
constexpr int NC = 77;  // 34 side rows + 43 generated product rows
constexpr int HORIZON = 50;
constexpr double DT = 0.05;
constexpr double MASS = 0.1;
constexpr double GRAVITY = 9.8;
constexpr double FRICTION = 0.4;
constexpr double ROTATION_SCALE = 0.4;
constexpr double UNIT = 0.05;
constexpr double CONTACT_RADIUS = 2.8 * UNIT;
constexpr double DC = 2.6429;

using Problem = NMPCProblem<NX, NU, NC, HORIZON>;

inline int lambda_index(int contact) { return 2 + contact; }
inline int v_index(int split) { return 10 + split; }

void split_values(const Vec<NU>& u, double v[7], double w[7]) {
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

void gap_values(const Vec<NU>& u, double gap[8]) {
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

struct Dynamics final : DynamicsModel<NX, NU> {
    Status discrete_step(const Vec<NX>& x, const Vec<NU>& u, double dt,
                         Vec<NX>& next) override {
        double force_x = 0.0;
        double force_y = 0.0;
        for (int i = 0; i < 8; ++i) {
            if ((i & 1) == 0) force_y += u[lambda_index(i)];
            else force_x += u[lambda_index(i)];
        }
        const double c = std::cos(x[2]);
        const double s = std::sin(x[2]);
        const double inverse_drag = 1.0 / (FRICTION * MASS * GRAVITY);
        next[0] = x[0] + dt * inverse_drag *
            (force_x * c - force_y * s);
        next[1] = x[1] + dt * inverse_drag *
            (force_x * s + force_y * c);
        next[2] = x[2] + dt * inverse_drag /
            (ROTATION_SCALE * CONTACT_RADIUS) *
            (-u[1] * force_x + u[0] * force_y);
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
        double force_x = 0.0;
        double force_y = 0.0;
        for (int i = 0; i < 8; ++i) {
            if ((i & 1) == 0) force_y += u[lambda_index(i)];
            else force_x += u[lambda_index(i)];
        }
        const double c = std::cos(x[2]);
        const double s = std::sin(x[2]);
        const double translation = dt / (FRICTION * MASS * GRAVITY);
        const double rotation = translation /
            (ROTATION_SCALE * CONTACT_RADIUS);

        Hxx(2, 2) += translation *
            (p[0] * (-c * force_x + s * force_y) +
             p[1] * (-s * force_x - c * force_y));
        for (int i = 0; i < 8; ++i) {
            const int force = lambda_index(i);
            if ((i & 1) == 0) {
                Hux(force, 2) += translation * (-p[0] * c - p[1] * s);
                Huu(force, 0) += p[2] * rotation;
                Huu(0, force) += p[2] * rotation;
            } else {
                Hux(force, 2) += translation * (-p[0] * s + p[1] * c);
                Huu(force, 1) -= p[2] * rotation;
                Huu(1, force) -= p[2] * rotation;
            }
        }
        return Status::SUCCESS;
    }
};

struct Cost final : CostModel<NX, NU> {
    Vec<NX> target;

    double stage_cost(const Vec<NX>&, const Vec<NU>& u, int) override {
        double value = 0.0;
        for (int i = 0; i < NU; ++i) value += 0.01 * u[i] * u[i];
        double v[7], w[7];
        split_values(u, v, w);
        for (double wi : w) value += 0.01 * wi * wi;
        return value;
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
        for (int i = 0; i < NU; ++i) qu[i] = 0.02 * u[i];
        double v[7], w[7];
        split_values(u, v, w);
        const int coordinate[7] = {0, 1, 1, 0, 1, 0, 0};
        for (int i = 0; i < 7; ++i) {
            qu[v_index(i)] += 0.02 * w[i];
            qu[coordinate[i]] -= 0.02 * w[i];
        }
        return Status::SUCCESS;
    }

    Status stage_hessian(const Vec<NX>&, const Vec<NU>&, int,
                         Mat<NX, NX>& Qxx, Mat<NU, NU>& Quu,
                         Mat<NU, NX>& Qux) override {
        Qxx.zero();
        Quu.zero();
        Qux.zero();
        for (int i = 0; i < NU; ++i) Quu(i, i) = 0.02;
        const int coordinate[7] = {0, 1, 1, 0, 1, 0, 0};
        for (int i = 0; i < 7; ++i) {
            const int c = coordinate[i];
            const int v = v_index(i);
            Quu(c, c) += 0.02;
            Quu(v, v) += 0.02;
            Quu(c, v) -= 0.02;
            Quu(v, c) -= 0.02;
        }
        return Status::SUCCESS;
    }

    Status terminal_gradient(const Vec<NX>& x, Vec<NX>& qx) override {
        for (int i = 0; i < NX; ++i)
            qx[i] = 200.0 * (x[i] - target[i]);
        return Status::SUCCESS;
    }

    Status terminal_hessian(const Vec<NX>&, Mat<NX, NX>& Qxx) override {
        Qxx.zero();
        for (int i = 0; i < NX; ++i) Qxx(i, i) = 200.0;
        return Status::SUCCESS;
    }
};

struct Contacts final : ConstraintModel<NX, NU, NC> {
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
        const double force_sign[8] =
            {-1.0, -1.0, 1.0, -1.0, 1.0, 1.0, 1.0, 1.0};
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
        const double force_sign[8] =
            {-1.0, -1.0, 1.0, -1.0, 1.0, 1.0, 1.0, 1.0};
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

    Status jacobian_terminal(const Vec<NX>&, Mat<NC, NX>& Cx) override {
        Cx.zero();
        return Status::SUCCESS;
    }
};

bool parse_args(int argc, char** argv, Vec<NX>& x0, Vec<NX>& target) {
    if (argc != 7) {
        std::fprintf(stderr,
            "Usage: %s x0_px x0_py x0_theta goal_px goal_py goal_theta\n",
            argv[0]);
        return false;
    }
    for (int i = 0; i < NX; ++i) {
        x0[i] = std::atof(argv[1 + i]);
        target[i] = std::atof(argv[4 + i]);
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
        const double alpha = static_cast<double>(k) / HORIZON;
        for (int i = 0; i < NX; ++i) {
            const double expected =
                (1.0 - alpha) * problem->x0[i] + alpha * cost->target[i];
            problem->stages[k].x[i] = expected;
            initialization_error = std::max(
                initialization_error,
                std::fabs(problem->stages[k].x[i] - expected));
        }
        problem->stages[k].u.zero();
        for (int i = 0; i < NU; ++i)
            initial_control_max =
                std::max(initial_control_max, std::fabs(problem->stages[k].u[i]));
        double v[7], w[7];
        split_values(problem->stages[k].u, v, w);
        for (double wi : w)
            initial_reconstructed_w_max =
                std::max(initial_reconstructed_w_max, std::fabs(wi));
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
    params.max_iters = 300;
    params.enable_preconditioner = true;
    params.exact_hessian = true;
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
        contact_benchmark::trajectory_file("contactipm", "impact_push_t");
    if (trajectory) {
        for (int k = 0; k <= HORIZON; ++k) {
            const auto& stage = problem->stages[k];
            trajectory << stage.x[0] << ' ' << stage.x[1] << ' ' << stage.x[2];
            trajectory << ' ' << (k < HORIZON ? stage.u[0] : 0.0);
            trajectory << ' ' << (k < HORIZON ? stage.u[1] : 0.0);
            for (int i = 0; i < 8; ++i)
                trajectory << ' '
                    << (k < HORIZON ? stage.u[lambda_index(i)] : 0.0);
            double v[7] = {}, w[7] = {};
            if (k < HORIZON) split_values(stage.u, v, w);
            for (double vi : v) trajectory << ' ' << vi;
            for (double wi : w) trajectory << ' ' << wi;
            trajectory << '\n';
        }
    }

    const Vec<NX>& terminal = problem->stages[HORIZON].x;
    const double translation_error = std::hypot(
        terminal[0] - cost->target[0], terminal[1] - cost->target[1]);
    const double angular_error = std::fabs(terminal[2] - cost->target[2]);

    std::printf("\n=== CONTACTIPM ON IMPACT PUSH T ===\n");
    std::printf("Status:                    %s\n", status_string(status));
    std::printf("Solve time:                %.9f s\n", seconds);
    std::printf("Iterations:                %d\n", stats.inner_iterations);
    std::printf("Objective:                 %.12e\n", stats.cost);
    std::printf("Primal infeasibility:      %.3e\n", stats.primal_infeas);
    std::printf("Physical complementarity: %.3e\n",
                stats.mpcc_complementarity);
    std::printf("Translation error:         %.9e\n", translation_error);
    std::printf("Angular error:             %.9e\n", angular_error);
    std::printf("Initialization state error:%.3e\n", initialization_error);
    std::printf("Initialization control max:%.3e\n", initial_control_max);
    std::printf("Initial reconstructed w max:%.3e\n",
                initial_reconstructed_w_max);
    return status == Status::SUCCESS ? 0 : 1;
}
