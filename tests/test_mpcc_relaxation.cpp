#include <cmath>
#include <cstdio>

#include "nmpc/contact_ipm.hpp"

using namespace nmpc;

namespace {

constexpr int NX = 1;
constexpr int NU = 2;
constexpr int NC = 3;
constexpr int HORIZON = 1;

struct StaticDynamics final : DynamicsModel<NX, NU> {
    Status discrete_step(const Vec<NX>& x, const Vec<NU>&,
                         double, Vec<NX>& x_next) override {
        x_next = x;
        return Status::SUCCESS;
    }

    Status linearize(const Vec<NX>&, const Vec<NU>&, double,
                     Mat<NX, NX>& A, Mat<NX, NU>& B) override {
        A.set_identity();
        B.zero();
        return Status::SUCCESS;
    }
};

struct ComplementarityCost final : CostModel<NX, NU> {
    double stage_cost(const Vec<NX>&, const Vec<NU>& u, int) override {
        const double d0 = u[0] - 1.0;
        const double d1 = u[1] - 0.2;
        return 0.5 * (d0 * d0 + d1 * d1);
    }

    double terminal_cost(const Vec<NX>&) override { return 0.0; }

    Status stage_gradient(const Vec<NX>&, const Vec<NU>& u, int,
                          Vec<NX>& qx, Vec<NU>& qu) override {
        qx.zero();
        qu[0] = u[0] - 1.0;
        qu[1] = u[1] - 0.2;
        return Status::SUCCESS;
    }

    Status stage_hessian(const Vec<NX>&, const Vec<NU>&, int,
                         Mat<NX, NX>& Qxx, Mat<NU, NU>& Quu,
                         Mat<NU, NX>& Qux) override {
        Qxx.zero();
        Quu.set_identity();
        Qux.zero();
        return Status::SUCCESS;
    }

    Status terminal_gradient(const Vec<NX>&, Vec<NX>& qx) override {
        qx.zero();
        return Status::SUCCESS;
    }

    Status terminal_hessian(const Vec<NX>&,
                            Mat<NX, NX>& Qxx) override {
        Qxx.zero();
        return Status::SUCCESS;
    }
};

struct ComplementarityConstraints final : ConstraintModel<NX, NU, NC> {
    int num_constraints(int k) const override { return k == 0 ? 2 : 0; }

    int num_complementarity_pairs(int k) const override {
        return k == 0 ? 1 : 0;
    }

    bool complementarity_pair(int k, int pair, int& first,
                              int& second) const override {
        if (k != 0 || pair != 0) return false;
        first = 0;
        second = 1;
        return true;
    }

    Status evaluate(const Vec<NX>&, const Vec<NU>& u, int,
                    Vec<NC>& g) override {
        g.zero();
        g[0] = -u[0];
        g[1] = -u[1];
        return Status::SUCCESS;
    }

    Status evaluate_terminal(const Vec<NX>&, Vec<NC>& g) override {
        g.zero();
        return Status::SUCCESS;
    }

    Status jacobian(const Vec<NX>&, const Vec<NU>&, int,
                    Mat<NC, NX>& Cx, Mat<NC, NU>& Cu) override {
        Cx.zero();
        Cu.zero();
        Cu(0, 0) = -1.0;
        Cu(1, 1) = -1.0;
        return Status::SUCCESS;
    }

    Status jacobian_terminal(const Vec<NX>&,
                             Mat<NC, NX>& Cx) override {
        Cx.zero();
        return Status::SUCCESS;
    }
};

using Problem = NMPCProblem<NX, NU, NC, HORIZON>;

bool near(double a, double b, double tolerance) {
    return std::fabs(a - b) <= tolerance;
}

int test_elastic_mpcc_row() {
    StaticDynamics dynamics;
    ComplementarityCost cost;
    ComplementarityConstraints constraints;
    Problem problem;
    problem.dynamics = &dynamics;
    problem.cost = &cost;
    problem.constraints = &constraints;
    problem.init_bounds_free();
    problem.x0[0] = 0.0;
    problem.stages[0].x[0] = 0.0;
    problem.stages[1].x[0] = 0.0;
    problem.stages[0].u[0] = 0.5;
    problem.stages[0].u[1] = 0.005;

    ContactIPMParams params;
    params.mu_init = 1e-2;
    // Deliberately leave a coarse generic floor; MPCC continuation must
    // derive a tighter internal floor from tol_mpcc.
    params.mu_min = 5e-4;
    params.mu_conv_threshold = 1e-6;
    params.max_iters = 200;
    params.tol_primal = 1e-7;
    params.tol_compl = 1e-7;
    params.tol_ineq = 1e-9;
    params.tol_stat = 1e-4;
    params.tol_mpcc = 1e-5;
    params.mpcc_relaxation_scale = 1.0;
    params.exact_hessian = false;
    params.verbosity = 0;

    ContactIPM<NX, NU, NC, HORIZON> solver;
    if (solver.configure(params) != Status::SUCCESS) return 1;
    const Status status = solver.solve(problem);
    if (status != Status::SUCCESS) {
        std::printf("elastic MPCC solve failed: %s\n", status_string(status));
        return 1;
    }

    const SolverStats& stats = solver.last_stats();
    const double u0 = problem.stages[0].u[0];
    const double u1 = problem.stages[0].u[1];
    const double sc = problem.stages[0].s[2];
    const double xi = problem.stages[0].lambda[2];
    const double product_row = u0 * u1 - stats.mpcc_relaxation;

    if (!(u0 >= 0.0 && u1 >= 0.0)) return 1;
    if (!(sc > 0.0 && xi > 0.0)) return 1;
    if (!near(product_row + sc, 0.0, 5e-7)) return 1;
    if (!near(sc * xi, stats.barrier_param, 5e-7)) return 1;
    if (stats.mpcc_complementarity > params.tol_mpcc) return 1;
    if (!(stats.barrier_param < params.mu_min)) return 1;
    if (!near(stats.mpcc_complementarity, std::fabs(u0 * u1), 1e-9))
        return 1;
    return 0;
}

int test_capacity_validation() {
    StaticDynamics dynamics;
    ComplementarityCost cost;

    struct TooSmallConstraints final : ConstraintModel<NX, NU, 2> {
        int num_constraints(int k) const override { return k == 0 ? 2 : 0; }
        int num_complementarity_pairs(int k) const override {
            return k == 0 ? 1 : 0;
        }
        bool complementarity_pair(int k, int p, int& a, int& b) const override {
            if (k != 0 || p != 0) return false;
            a = 0;
            b = 1;
            return true;
        }
        Status evaluate(const Vec<NX>&, const Vec<NU>&, int,
                        Vec<2>& g) override {
            g.zero();
            return Status::SUCCESS;
        }
        Status evaluate_terminal(const Vec<NX>&,
                                 Vec<2>& g) override {
            g.zero();
            return Status::SUCCESS;
        }
        Status jacobian(const Vec<NX>&, const Vec<NU>&, int,
                        Mat<2, NX>& Cx, Mat<2, NU>& Cu) override {
            Cx.zero();
            Cu.zero();
            return Status::SUCCESS;
        }
        Status jacobian_terminal(const Vec<NX>&,
                                 Mat<2, NX>& Cx) override {
            Cx.zero();
            return Status::SUCCESS;
        }
    } constraints;

    NMPCProblem<NX, NU, 2, HORIZON> problem;
    problem.dynamics = &dynamics;
    problem.cost = &cost;
    problem.constraints = &constraints;
    problem.init_bounds_free();

    ContactIPM<NX, NU, 2, HORIZON> solver;
    solver.configure();
    return solver.solve(problem) == Status::BAD_ARGUMENT ? 0 : 1;
}

int test_feasible_slack_floor_tracks_initial_barrier() {
    StaticDynamics dynamics;
    ComplementarityCost cost;
    ComplementarityConstraints constraints;
    Problem problem;
    problem.dynamics = &dynamics;
    problem.cost = &cost;
    problem.constraints = &constraints;
    problem.init_bounds_free();
    problem.stages[0].u[0] = 0.05;
    problem.stages[0].u[1] = 0.05;

    ContactIPMParams params;
    params.mu_init = 0.1;
    params.s_min_init = 0.01;
    params.delta_slack = 0.0;
    params.max_iters = 0;

    ContactIPM<NX, NU, NC, HORIZON> solver;
    if (solver.configure(params) != Status::SUCCESS) return 1;
    solver.solve(problem);
    for (int row = 0; row < NC; ++row) {
        if (problem.stages[0].s[row] + 1e-12 < params.mu_init)
            return 1;
    }
    return 0;
}

int test_mpcc_recovery_counts_all_attempts() {
    StaticDynamics dynamics;
    ComplementarityCost cost;
    ComplementarityConstraints constraints;
    Problem problem;
    problem.dynamics = &dynamics;
    problem.cost = &cost;
    problem.constraints = &constraints;
    problem.init_bounds_free();
    problem.stages[0].u[0] = 0.5;
    problem.stages[0].u[1] = 0.005;

    ContactIPMParams params;
    params.max_iters = 1;
    params.mpcc_recovery_max_iters = 1;
    params.tol_primal = 1e-30;
    params.tol_compl = 1e-30;
    params.tol_stat = 1e-30;
    params.tol_mpcc = 1e-30;

    ContactIPM<NX, NU, NC, HORIZON> solver;
    if (solver.configure(params) != Status::SUCCESS) return 1;
    if (solver.solve_mpcc_with_recovery(problem) == Status::SUCCESS) return 1;
    return solver.last_stats().inner_iterations == 5 ? 0 : 1;
}

}  // namespace

int main() {
    int failures = 0;
    failures += test_elastic_mpcc_row();
    failures += test_capacity_validation();
    failures += test_feasible_slack_floor_tracks_initial_barrier();
    failures += test_mpcc_recovery_counts_all_attempts();
    if (failures == 0)
        std::printf("MPCC elastic slack/barrier tests passed\n");
    return failures == 0 ? 0 : 1;
}