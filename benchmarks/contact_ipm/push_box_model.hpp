#pragma once

#include <algorithm>
#include <cmath>

#include "nmpc/contact_ipm.hpp"

namespace contact_benchmark::push_box {

using namespace nmpc;

inline constexpr int NX = 3;
inline constexpr int NU = 6;
inline constexpr int NC = 18;
#ifndef CONTACT_BENCHMARK_NODES
#define CONTACT_BENCHMARK_NODES 100
#endif
static_assert(CONTACT_BENCHMARK_NODES >= 2,
              "Push Box requires at least two nodes");
inline constexpr int HORIZON = CONTACT_BENCHMARK_NODES - 1;
#ifndef CONTACT_BENCHMARK_DT
#define CONTACT_BENCHMARK_DT 0.02
#endif
static_assert(CONTACT_BENCHMARK_DT > 0.0,
              "Push Box requires a positive time step");
inline constexpr double DT = CONTACT_BENCHMARK_DT;
inline constexpr double HALF_LENGTH = 0.5;
inline constexpr double HALF_WIDTH = 0.25;
inline constexpr double NOMINAL_MASS = 1.0;
inline constexpr double NOMINAL_FRICTION = 0.5;
inline constexpr double GRAVITY = 9.8;
inline constexpr double ROTATION_SCALE = 0.4;
inline constexpr double RADIUS =
    0.55901699437494742410;
inline constexpr double PI = 3.14159265358979323846;

using Problem = NMPCProblem<NX, NU, NC, HORIZON>;

struct PushBoxDynamics final : DynamicsModel<NX, NU> {
    double mass = NOMINAL_MASS;
    double friction = NOMINAL_FRICTION;

    void velocity(const Vec<NX>& x, const Vec<NU>& u,
                  Vec<NX>& xdot) const {
        const double theta = x[2];
        const double contact_x = u[0];
        const double contact_y = u[1];
        const double force_x = u[2] + u[4];
        const double force_y = u[3] + u[5];
        const double c = std::cos(theta);
        const double s = std::sin(theta);
        const double inverse_drag = 1.0 / (friction * mass * GRAVITY);
        xdot[0] = inverse_drag * (c * force_y - s * force_x);
        xdot[1] = inverse_drag * (s * force_y + c * force_x);
        xdot[2] =
            (-contact_y * force_y + contact_x * force_x) /
            (friction * mass * GRAVITY * ROTATION_SCALE * RADIUS);
    }

    Status discrete_step(const Vec<NX>& x, const Vec<NU>& u, double dt,
                         Vec<NX>& next) override {
        Vec<NX> xdot;
        velocity(x, u, xdot);
        for (int i = 0; i < NX; ++i)
            next[i] = x[i] + dt * xdot[i];
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
        const double translation = dt / (friction * mass * GRAVITY);
        const double rotation = dt /
            (friction * mass * GRAVITY * ROTATION_SCALE * RADIUS);

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
    double path_tracking_scale = 0.0;
    double tracking_scale = 1.0;
    double effort_scale = 1.0;

    PushBoxCost() {
        const double angle = 12.0 * 2.0 * PI / 18.0;
        target[0] = 3.0 * std::cos(angle);
        target[1] = 3.0 * std::sin(angle);
        target[2] = angle;
    }

    double stage_cost(const Vec<NX>& x, const Vec<NU>& u, int) override {
        double value = 0.0;
        for (int i = 0; i < NX; ++i) {
            const double error = x[i] - target[i];
            value += path_tracking_scale * error * error;
        }
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

    Status stage_gradient(const Vec<NX>& x, const Vec<NU>& u, int,
                          Vec<NX>& qx, Vec<NU>& qu) override {
        for (int i = 0; i < NX; ++i)
            qx[i] = 2.0 * path_tracking_scale * (x[i] - target[i]);
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
        for (int i = 0; i < NX; ++i)
            Qxx(i, i) = 2.0 * path_tracking_scale;
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
    static constexpr int pairs[10][2] = {
        {0, 4}, {1, 5}, {2, 6}, {3, 7},
        {0, 1}, {0, 2}, {0, 3}, {1, 2}, {1, 3}, {2, 3}};

    bool provides_adjoint_hessian() const override { return true; }

    int num_constraints(int k) const override {
        return k < HORIZON ? 8 : 0;
    }

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

inline ContactIPMParams solver_parameters() {
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
    return params;
}

inline void initialize_problem(Problem& problem, PushBoxDynamics& dynamics,
                               PushBoxCost& cost,
                               PushBoxContacts& contacts,
                               const Vec<NX>& initial_state) {
    problem.dynamics = &dynamics;
    problem.cost = &cost;
    problem.constraints = &contacts;
    problem.dt = DT;
    problem.init_bounds_free();
    problem.x0 = initial_state;
    for (int k = 0; k <= HORIZON; ++k) {
        problem.stages[k].x.zero();
        problem.stages[k].u.zero();
    }
    problem.stages[0].x = initial_state;
}

inline double translation_error(const Vec<NX>& state,
                                const Vec<NX>& target) {
    const double dx = state[0] - target[0];
    const double dy = state[1] - target[1];
    return std::sqrt(dx * dx + dy * dy);
}

inline double angular_error(const Vec<NX>& state,
                            const Vec<NX>& target) {
    return std::fabs(state[2] - target[2]);
}

}  // namespace contact_benchmark::push_box
