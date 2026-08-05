#include CONTACTIPM_NATIVE_CONFIG_HEADER

#include <algorithm>
#include <array>
#include <chrono>
#include <cmath>
#include <cstdlib>
#include <iomanip>
#include <iostream>
#include <numeric>
#include <string>
#include <vector>

#include "nmpc/contact_ipm.hpp"

using casadi_int = long long int;
using CasadiEval = int (*)(const double**, double**, casadi_int*, double*, int);
using CasadiWork = int (*)(casadi_int*, casadi_int*, casadi_int*, casadi_int*);
using CasadiCount = casadi_int (*)();

extern "C" {
#define DECLARE_CASADI_FUNCTION(name)                                      \
    int name(const double**, double**, casadi_int*, double*, int);         \
    int name##_work(casadi_int*, casadi_int*, casadi_int*, casadi_int*);   \
    casadi_int name##_n_in();                                               \
    casadi_int name##_n_out()
DECLARE_CASADI_FUNCTION(contactipm_f_disc);
DECLARE_CASADI_FUNCTION(contactipm_f_lin);
DECLARE_CASADI_FUNCTION(contactipm_stage_cost);
DECLARE_CASADI_FUNCTION(contactipm_term_cost);
DECLARE_CASADI_FUNCTION(contactipm_stage_grad);
DECLARE_CASADI_FUNCTION(contactipm_stage_hess);
DECLARE_CASADI_FUNCTION(contactipm_term_grad);
DECLARE_CASADI_FUNCTION(contactipm_term_hess);
DECLARE_CASADI_FUNCTION(contactipm_constr_eval);
DECLARE_CASADI_FUNCTION(contactipm_constr_eval_term);
DECLARE_CASADI_FUNCTION(contactipm_constr_jac);
DECLARE_CASADI_FUNCTION(contactipm_constr_jac_term);
DECLARE_CASADI_FUNCTION(contactipm_dyn_adj_hess);
DECLARE_CASADI_FUNCTION(contactipm_con_adj_hess);
DECLARE_CASADI_FUNCTION(contactipm_con_adj_hess_term);
#undef DECLARE_CASADI_FUNCTION
#define CASADI_CALL(name) name, name##_work, name##_n_in, name##_n_out

}

namespace {

namespace cfg = contactipm_native_config;
constexpr int NX = cfg::nx;
constexpr int NU = cfg::nu;
constexpr int NC = cfg::nc;
constexpr int N = cfg::horizon;

class CasadiCall {
public:
    CasadiCall(CasadiEval eval, CasadiWork work,
               CasadiCount input_count, CasadiCount output_count)
        : eval_(eval), input_count_(input_count()),
          output_count_(output_count()) {
        casadi_int arg = 0, result = 0, iw = 0, w = 0;
        work(&arg, &result, &iw, &w);
        args_.resize(static_cast<std::size_t>(arg), nullptr);
        results_.resize(static_cast<std::size_t>(result), nullptr);
        iw_.resize(static_cast<std::size_t>(iw));
        w_.resize(static_cast<std::size_t>(w));
    }

    int operator()(const double** args, double** results) {
        std::copy(args, args + input_count_, args_.begin());
        std::copy(results, results + output_count_, results_.begin());
        return eval_(args_.data(), results_.data(),
                     iw_.empty() ? nullptr : iw_.data(),
                     w_.empty() ? nullptr : w_.data(), 0);
    }

private:
    CasadiEval eval_;
    casadi_int input_count_;
    casadi_int output_count_;
    std::vector<const double*> args_;
    std::vector<double*> results_;
    std::vector<casadi_int> iw_;
    std::vector<double> w_;
};

template <int Rows, int Cols>
void add_dense(const std::array<double, Rows * Cols>& source,
               nmpc::Mat<Rows, Cols>& destination) {
    for (int index = 0; index < Rows * Cols; ++index)
        destination.data[index] += source[static_cast<std::size_t>(index)];
}

class NativeDynamics final : public nmpc::DynamicsModel<NX, NU> {
public:
    NativeDynamics()
        : step_(CASADI_CALL(contactipm_f_disc)),
          linearize_(CASADI_CALL(contactipm_f_lin)),
          adjoint_hessian_(CASADI_CALL(contactipm_dyn_adj_hess)) {}

    nmpc::Status discrete_step(const nmpc::Vec<NX>& x,
                               const nmpc::Vec<NU>& u, double,
                               nmpc::Vec<NX>& next) override {
        const double* args[] = {x.data, u.data};
        double* results[] = {next.data};
        return step_(args, results) == 0 ? nmpc::Status::SUCCESS
                                         : nmpc::Status::INTERNAL_ERROR;
    }

    nmpc::Status linearize(const nmpc::Vec<NX>& x,
                           const nmpc::Vec<NU>& u, double,
                           nmpc::Mat<NX, NX>& A,
                           nmpc::Mat<NX, NU>& B) override {
        const double* args[] = {x.data, u.data};
        double* results[] = {A.data, B.data};
        return linearize_(args, results) == 0 ? nmpc::Status::SUCCESS
                                              : nmpc::Status::INTERNAL_ERROR;
    }

    bool provides_adjoint_hessian() const override { return true; }

    nmpc::Status adjoint_hessian(const nmpc::Vec<NX>& x,
                                  const nmpc::Vec<NU>& u, double, int,
                                  const nmpc::Vec<NX>& p,
                                  nmpc::Mat<NX, NX>& Hxx,
                                  nmpc::Mat<NU, NX>& Hux,
                                  nmpc::Mat<NU, NU>& Huu) override {
        std::array<double, NX * NX> hxx{};
        std::array<double, NU * NX> hux{};
        std::array<double, NU * NU> huu{};
        const double* args[] = {x.data, u.data, p.data};
        double* results[] = {hxx.data(), hux.data(), huu.data()};
        if (adjoint_hessian_(args, results) != 0)
            return nmpc::Status::INTERNAL_ERROR;
        add_dense(hxx, Hxx);
        add_dense(hux, Hux);
        add_dense(huu, Huu);
        return nmpc::Status::SUCCESS;
    }

private:
    CasadiCall step_;
    CasadiCall linearize_;
    CasadiCall adjoint_hessian_;
};

class NativeCost final : public nmpc::CostModel<NX, NU> {
public:
    NativeCost()
        : stage_cost_(CASADI_CALL(contactipm_stage_cost)),
          term_cost_(CASADI_CALL(contactipm_term_cost)),
          stage_grad_(CASADI_CALL(contactipm_stage_grad)),
          stage_hess_(CASADI_CALL(contactipm_stage_hess)),
          term_grad_(CASADI_CALL(contactipm_term_grad)),
          term_hess_(CASADI_CALL(contactipm_term_hess)) {}

    double stage_cost(const nmpc::Vec<NX>& x,
                      const nmpc::Vec<NU>& u, int) override {
        double value = 0.0;
        const double* args[] = {x.data, u.data};
        double* results[] = {&value};
        stage_cost_(args, results);
        return value;
    }

    double terminal_cost(const nmpc::Vec<NX>& x) override {
        double value = 0.0;
        const double* args[] = {x.data};
        double* results[] = {&value};
        term_cost_(args, results);
        return value;
    }

    nmpc::Status stage_gradient(const nmpc::Vec<NX>& x,
                                const nmpc::Vec<NU>& u, int,
                                nmpc::Vec<NX>& qx,
                                nmpc::Vec<NU>& qu) override {
        const double* args[] = {x.data, u.data};
        double* results[] = {qx.data, qu.data};
        return stage_grad_(args, results) == 0 ? nmpc::Status::SUCCESS
                                               : nmpc::Status::INTERNAL_ERROR;
    }

    nmpc::Status stage_hessian(const nmpc::Vec<NX>& x,
                               const nmpc::Vec<NU>& u, int,
                               nmpc::Mat<NX, NX>& Qxx,
                               nmpc::Mat<NU, NU>& Quu,
                               nmpc::Mat<NU, NX>& Qux) override {
        const double* args[] = {x.data, u.data};
        double* results[] = {Qxx.data, Quu.data, Qux.data};
        return stage_hess_(args, results) == 0 ? nmpc::Status::SUCCESS
                                               : nmpc::Status::INTERNAL_ERROR;
    }

    nmpc::Status terminal_gradient(const nmpc::Vec<NX>& x,
                                   nmpc::Vec<NX>& qx) override {
        const double* args[] = {x.data};
        double* results[] = {qx.data};
        return term_grad_(args, results) == 0 ? nmpc::Status::SUCCESS
                                              : nmpc::Status::INTERNAL_ERROR;
    }

    nmpc::Status terminal_hessian(const nmpc::Vec<NX>& x,
                                  nmpc::Mat<NX, NX>& Qxx) override {
        const double* args[] = {x.data};
        double* results[] = {Qxx.data};
        return term_hess_(args, results) == 0 ? nmpc::Status::SUCCESS
                                              : nmpc::Status::INTERNAL_ERROR;
    }

private:
    CasadiCall stage_cost_;
    CasadiCall term_cost_;
    CasadiCall stage_grad_;
    CasadiCall stage_hess_;
    CasadiCall term_grad_;
    CasadiCall term_hess_;
};

class NativeConstraints final : public nmpc::ConstraintModel<NX, NU, NC> {
public:
    NativeConstraints()
        : evaluate_(CASADI_CALL(contactipm_constr_eval)),
          evaluate_term_(CASADI_CALL(contactipm_constr_eval_term)),
          jacobian_(CASADI_CALL(contactipm_constr_jac)),
          jacobian_term_(CASADI_CALL(contactipm_constr_jac_term)),
          adjoint_hessian_(CASADI_CALL(contactipm_con_adj_hess)),
          adjoint_hessian_term_(CASADI_CALL(contactipm_con_adj_hess_term)) {}

    int num_constraints(int k) const override { return k == N ? 0 : NC; }

    nmpc::Status evaluate(const nmpc::Vec<NX>& x,
                          const nmpc::Vec<NU>& u, int,
                          nmpc::Vec<NC>& g) override {
        const double* args[] = {x.data, u.data};
        double* results[] = {g.data};
        return evaluate_(args, results) == 0 ? nmpc::Status::SUCCESS
                                             : nmpc::Status::INTERNAL_ERROR;
    }

    nmpc::Status evaluate_terminal(const nmpc::Vec<NX>& x,
                                   nmpc::Vec<NC>& g) override {
        const double* args[] = {x.data};
        double* results[] = {g.data};
        return evaluate_term_(args, results) == 0
                   ? nmpc::Status::SUCCESS
                   : nmpc::Status::INTERNAL_ERROR;
    }

    nmpc::Status jacobian(const nmpc::Vec<NX>& x,
                          const nmpc::Vec<NU>& u, int,
                          nmpc::Mat<NC, NX>& Cx,
                          nmpc::Mat<NC, NU>& Cu) override {
        const double* args[] = {x.data, u.data};
        double* results[] = {Cx.data, Cu.data};
        return jacobian_(args, results) == 0 ? nmpc::Status::SUCCESS
                                             : nmpc::Status::INTERNAL_ERROR;
    }

    nmpc::Status jacobian_terminal(const nmpc::Vec<NX>& x,
                                   nmpc::Mat<NC, NX>& Cx) override {
        const double* args[] = {x.data};
        double* results[] = {Cx.data};
        return jacobian_term_(args, results) == 0
                   ? nmpc::Status::SUCCESS
                   : nmpc::Status::INTERNAL_ERROR;
    }

    bool provides_adjoint_hessian() const override { return true; }

    nmpc::Status adjoint_hessian(const nmpc::Vec<NX>& x,
                                  const nmpc::Vec<NU>& u, int,
                                  const nmpc::Vec<NC>& lambda,
                                  nmpc::Mat<NX, NX>& Hxx,
                                  nmpc::Mat<NU, NX>& Hux,
                                  nmpc::Mat<NU, NU>& Huu) override {
        std::array<double, NX * NX> hxx{};
        std::array<double, NU * NX> hux{};
        std::array<double, NU * NU> huu{};
        const double* args[] = {x.data, u.data, lambda.data};
        double* results[] = {hxx.data(), hux.data(), huu.data()};
        if (adjoint_hessian_(args, results) != 0)
            return nmpc::Status::INTERNAL_ERROR;
        add_dense(hxx, Hxx);
        add_dense(hux, Hux);
        add_dense(huu, Huu);
        return nmpc::Status::SUCCESS;
    }

    nmpc::Status adjoint_hessian_terminal(
        const nmpc::Vec<NX>& x, const nmpc::Vec<NC>& lambda,
        nmpc::Mat<NX, NX>& Hxx) override {
        std::array<double, NX * NX> hxx{};
        const double* args[] = {x.data, lambda.data};
        double* results[] = {hxx.data()};
        if (adjoint_hessian_term_(args, results) != 0)
            return nmpc::Status::INTERNAL_ERROR;
        add_dense(hxx, Hxx);
        return nmpc::Status::SUCCESS;
    }

private:
    CasadiCall evaluate_;
    CasadiCall evaluate_term_;
    CasadiCall jacobian_;
    CasadiCall jacobian_term_;
    CasadiCall adjoint_hessian_;
    CasadiCall adjoint_hessian_term_;
};

using Problem = nmpc::NMPCProblem<NX, NU, NC, N>;

void initialize_problem(Problem& problem, NativeDynamics& dynamics,
                        NativeCost& cost, NativeConstraints& constraints) {
    problem.dynamics = &dynamics;
    problem.cost = &cost;
    problem.constraints = cfg::has_constraints ? &constraints : nullptr;
    problem.dt = cfg::dt;
    problem.n_bound_x = 0;
    problem.n_bound_u = 0;

    for (int i = 0; i < NX; ++i) {
        problem.x0[i] = cfg::x0[static_cast<std::size_t>(i)];
        problem.x_lb[i] = cfg::x_lb[static_cast<std::size_t>(i)];
        problem.x_ub[i] = cfg::x_ub[static_cast<std::size_t>(i)];
        if (problem.x_lb[i] > -1e19 || problem.x_ub[i] < 1e19)
            ++problem.n_bound_x;
    }
    for (int i = 0; i < NU; ++i) {
        problem.u_lb[i] = cfg::u_lb[static_cast<std::size_t>(i)];
        problem.u_ub[i] = cfg::u_ub[static_cast<std::size_t>(i)];
        if (problem.u_lb[i] > -1e19 || problem.u_ub[i] < 1e19)
            ++problem.n_bound_u;
    }
    for (int k = 0; k <= N; ++k) {
        for (int i = 0; i < NX; ++i)
            problem.stages[k].x[i] = k == 0
                ? cfg::x0[static_cast<std::size_t>(i)]
                : cfg::x_guess[static_cast<std::size_t>(i)];
        problem.stages[k].u.zero();
        if (cfg::has_constraints) {
            for (int j = 0; j < NC; ++j) {
                problem.stages[k].s[j] = 1.0;
                problem.stages[k].lambda[j] = 0.1;
            }
        }
    }
}

double objective(Problem& problem, NativeCost& cost) {
    double value = 0.0;
    for (int k = 0; k < N; ++k)
        value += cost.stage_cost(problem.stages[k].x, problem.stages[k].u, k);
    return value + cost.terminal_cost(problem.stages[N].x);
}

double quantile(const std::vector<double>& sorted, double probability) {
    if (sorted.empty()) return 0.0;
    const double index = probability * static_cast<double>(sorted.size() - 1);
    const auto lower = static_cast<std::size_t>(std::floor(index));
    const auto upper = static_cast<std::size_t>(std::ceil(index));
    const double weight = index - static_cast<double>(lower);
    return sorted[lower] * (1.0 - weight) + sorted[upper] * weight;
}

struct Observation {
    nmpc::Status status;
    int iterations;
    double seconds;
    double objective;
};

Observation run_once(NativeDynamics& dynamics, NativeCost& cost,
                     NativeConstraints& constraints, int max_iterations) {
    Problem problem{};
    initialize_problem(problem, dynamics, cost, constraints);
    nmpc::ContactIPM<NX, NU, NC, N> solver;
    nmpc::ContactIPMParams params;
    params.mu_init = 0.1;
    params.mu_min = 1e-6;
    params.max_iters = max_iterations;
    params.tol_primal = 1e-4;
    params.tol_compl = 1e-4;
    params.tol_stat = 1e-4;
    params.tol_ineq = 1e-8;
    params.verbosity = 0;
    params.enable_preconditioner = problem.n_bound_x > 0
                                   || problem.n_bound_u > 0
                                   || cfg::has_constraints;
    params.bound_aware_preconditioner = problem.n_bound_x > 0
                                        || problem.n_bound_u > 0;
    solver.configure(params);

    const auto start = std::chrono::steady_clock::now();
    const nmpc::Status status = solver.solve(problem);
    const auto stop = std::chrono::steady_clock::now();
    return {status, solver.last_stats().inner_iterations,
            std::chrono::duration<double>(stop - start).count(),
            objective(problem, cost)};
}

}  // namespace

int main(int argc, char** argv) {
    int warmups = 1;
    int repetitions = 20;
    int max_iterations = 100;
    for (int i = 1; i < argc; ++i) {
        const std::string argument = argv[i];
        if (argument == "--warmups" && i + 1 < argc)
            warmups = std::atoi(argv[++i]);
        else if (argument == "--repetitions" && i + 1 < argc)
            repetitions = std::atoi(argv[++i]);
        else if (argument == "--max-iters" && i + 1 < argc)
            max_iterations = std::atoi(argv[++i]);
        else {
            std::cerr << "usage: " << argv[0]
                      << " [--warmups N] [--repetitions N] [--max-iters N]\n";
            return 2;
        }
    }
    if (warmups < 0 || repetitions < 1 || max_iterations < 1) return 2;

    NativeDynamics dynamics;
    NativeCost cost;
    NativeConstraints constraints;
    for (int i = 0; i < warmups; ++i)
        run_once(dynamics, cost, constraints, max_iterations);

    std::vector<double> times;
    std::vector<double> objectives;
    std::vector<int> iterations;
    int successes = 0;
    for (int i = 0; i < repetitions; ++i) {
        const Observation observation =
            run_once(dynamics, cost, constraints, max_iterations);
        if (observation.status == nmpc::Status::SUCCESS) ++successes;
        times.push_back(observation.seconds);
        objectives.push_back(observation.objective);
        iterations.push_back(observation.iterations);
    }
    std::sort(times.begin(), times.end());
    std::sort(objectives.begin(), objectives.end());
    std::sort(iterations.begin(), iterations.end());

    std::cout << std::setprecision(10)
              << "problem=" << cfg::problem_name << '\n'
              << "success=" << successes << '/' << repetitions << '\n'
              << "iterations_median=" << iterations[iterations.size() / 2] << '\n'
              << "objective_median=" << quantile(objectives, 0.5) << '\n'
              << "solver_time_median_s=" << quantile(times, 0.5) << '\n'
              << "solver_time_q1_s=" << quantile(times, 0.25) << '\n'
              << "solver_time_q3_s=" << quantile(times, 0.75) << '\n'
              << "solver_time_p90_s=" << quantile(times, 0.90) << '\n';
    return successes == repetitions ? 0 : 1;
}
