/**
 * @file bindings.cpp
 * @brief pybind11 bindings for ContactIPM solver with template dispatch.
 *
 * Creates Python callback implementations of DynamicsModel, CostModel,
 * and ConstraintModel that call CasADi Functions from C++.
 * Dispatches to the correct template instantiation based on problem dimensions.
 */

#include <pybind11/pybind11.h>
#include <pybind11/stl.h>
#include <pybind11/numpy.h>

#include <vector>
#include <string>
#include <chrono>
#include <memory>
#include <cstdlib>

#include "nmpc/contact_ipm.hpp"

namespace py = pybind11;
using namespace nmpc;

// ─────────────────────────────────────────────────────────────────────────────
//  Conversion helpers: Vec<N> <-> numpy
// ─────────────────────────────────────────────────────────────────────────────

template <int N>
py::array_t<double> vec_to_np(const Vec<N>& v) {
    py::array_t<double> arr(N);
    auto buf = arr.mutable_unchecked<1>();
    for (int i = 0; i < N; ++i) buf(i) = v.data[i];
    return arr;
}

template <int N>
void np_to_vec(py::array_t<double> arr, Vec<N>& v) {
    auto buf = arr.unchecked<1>();
    for (int i = 0; i < N; ++i) v.data[i] = buf(i);
}

// ─────────────────────────────────────────────────────────────────────────────
//  Python callback model implementations
// ─────────────────────────────────────────────────────────────────────────────

template <int NX, int NU>
struct PyDynModel : DynamicsModel<NX, NU> {
    py::object f_disc;   // CasADi Function: (x[NX], u[NU]) -> x_next[NX]
    py::object f_lin;    // CasADi Function: (x[NX], u[NU]) -> (A[NX,NX], B[NX,NU])
    py::object f_adj_hess; // (x, u, p[NX]) -> (Hxx[NX,NX], Hux[NU,NX], Huu[NU,NU])

    Status discrete_step(const Vec<NX>& x, const Vec<NU>& u,
                         double /*dt*/, Vec<NX>& x_next) override {
        auto x_np = vec_to_np(x);
        auto u_np = vec_to_np(u);
        py::array_t<double> result = f_disc(x_np, u_np).cast<py::array_t<double>>();
        np_to_vec(result, x_next);
        return Status::SUCCESS;
    }

    Status linearize(const Vec<NX>& x, const Vec<NU>& u, double /*dt*/,
                     Mat<NX, NX>& A, Mat<NX, NU>& B) override {
        auto x_np = vec_to_np(x);
        auto u_np = vec_to_np(u);
        py::tuple result = f_lin(x_np, u_np).cast<py::tuple>();
        auto A_np = result[0].cast<py::array_t<double>>();
        auto B_np = result[1].cast<py::array_t<double>>();
        auto A_buf = A_np.unchecked<2>();
        auto B_buf = B_np.unchecked<2>();
        for (int c = 0; c < NX; ++c)
            for (int r = 0; r < NX; ++r)
                A.data[c * NX + r] = A_buf(r, c);
        for (int c = 0; c < NU; ++c)
            for (int r = 0; r < NX; ++r)
                B.data[c * NX + r] = B_buf(r, c);
        return Status::SUCCESS;
    }

    bool provides_adjoint_hessian() const override {
        return f_adj_hess && !f_adj_hess.is_none();
    }

    // Adds  Σ_m p[m]·∇²f_m  into the (pre-populated) Hessian blocks.
    Status adjoint_hessian(const Vec<NX>& x, const Vec<NU>& u, double /*dt*/,
                           int /*k*/, const Vec<NX>& p,
                           Mat<NX, NX>& Hxx, Mat<NU, NX>& Hux,
                           Mat<NU, NU>& Huu) override {
        py::tuple r = f_adj_hess(vec_to_np(x), vec_to_np(u),
                                 vec_to_np(p)).cast<py::tuple>();
        auto Hxx_np = r[0].cast<py::array_t<double>>().unchecked<2>();
        auto Hux_np = r[1].cast<py::array_t<double>>().unchecked<2>();
        auto Huu_np = r[2].cast<py::array_t<double>>().unchecked<2>();
        for (int c = 0; c < NX; ++c)
            for (int rr = 0; rr < NX; ++rr) Hxx(rr, c) += Hxx_np(rr, c);
        for (int c = 0; c < NX; ++c)
            for (int rr = 0; rr < NU; ++rr) Hux(rr, c) += Hux_np(rr, c);
        for (int c = 0; c < NU; ++c)
            for (int rr = 0; rr < NU; ++rr) Huu(rr, c) += Huu_np(rr, c);
        return Status::SUCCESS;
    }
};

template <int NX, int NU>
struct PyCostModel : CostModel<NX, NU> {
    py::object stage_cost_fn;   // (x, u) -> scalar
    py::object term_cost_fn;   // (x,) -> scalar
    py::object stage_grad_fn;  // (x, u) -> (qx[NX], qu[NU])
    py::object stage_hess_fn;  // (x, u) -> (Qxx[NX,NX], Quu[NU,NU], Qux[NU,NX])
    py::object term_grad_fn;   // (x,) -> qx[NX]
    py::object term_hess_fn;   // (x,) -> Qxx[NX,NX]

    double stage_cost(const Vec<NX>& x, const Vec<NU>& u, int /*k*/) override {
        return stage_cost_fn(vec_to_np(x), vec_to_np(u)).cast<double>();
    }

    double terminal_cost(const Vec<NX>& x) override {
        return term_cost_fn(vec_to_np(x)).cast<double>();
    }

    Status stage_gradient(const Vec<NX>& x, const Vec<NU>& u, int /*k*/,
                          Vec<NX>& qx, Vec<NU>& qu) override {
        py::tuple r = stage_grad_fn(vec_to_np(x), vec_to_np(u)).cast<py::tuple>();
        np_to_vec(r[0].cast<py::array_t<double>>(), qx);
        np_to_vec(r[1].cast<py::array_t<double>>(), qu);
        return Status::SUCCESS;
    }

    Status stage_hessian(const Vec<NX>& x, const Vec<NU>& u, int /*k*/,
                         Mat<NX, NX>& Qxx, Mat<NU, NU>& Quu,
                         Mat<NU, NX>& Qux) override {
        py::tuple r = stage_hess_fn(vec_to_np(x), vec_to_np(u)).cast<py::tuple>();
        auto Qxx_np = r[0].cast<py::array_t<double>>().unchecked<2>();
        auto Quu_np = r[1].cast<py::array_t<double>>().unchecked<2>();
        auto Qux_np = r[2].cast<py::array_t<double>>().unchecked<2>();
        for (int c = 0; c < NX; ++c)
            for (int rr = 0; rr < NX; ++rr)
                Qxx.data[c * NX + rr] = Qxx_np(rr, c);
        for (int c = 0; c < NU; ++c)
            for (int rr = 0; rr < NU; ++rr)
                Quu.data[c * NU + rr] = Quu_np(rr, c);
        for (int c = 0; c < NX; ++c)
            for (int rr = 0; rr < NU; ++rr)
                Qux.data[c * NU + rr] = Qux_np(rr, c);
        return Status::SUCCESS;
    }

    Status terminal_gradient(const Vec<NX>& x, Vec<NX>& qx) override {
        np_to_vec(term_grad_fn(vec_to_np(x)).cast<py::array_t<double>>(), qx);
        return Status::SUCCESS;
    }

    Status terminal_hessian(const Vec<NX>& x, Mat<NX, NX>& Qxx) override {
        auto Qxx_np = term_hess_fn(vec_to_np(x)).cast<py::array_t<double>>().unchecked<2>();
        for (int c = 0; c < NX; ++c)
            for (int rr = 0; rr < NX; ++rr)
                Qxx.data[c * NX + rr] = Qxx_np(rr, c);
        return Status::SUCCESS;
    }
};

template <int NX, int NU, int NC>
struct PyConstrModel : ConstraintModel<NX, NU, NC> {
    py::object eval_fn;      // (x, u) -> g[NC]
    py::object eval_term_fn; // (x,) -> g[NC]
    py::object jac_fn;       // (x, u) -> (Cx[NC,NX], Cu[NC,NU])
    py::object jac_term_fn;  // (x,) -> Cx[NC,NX]
    py::object adj_hess_fn;      // (x, u, lam[NC]) -> (Hxx, Hux, Huu)
    py::object adj_hess_term_fn; // (x, lam[NC]) -> Hxx[NX,NX]
    int terminal_stage = -1;

    int num_constraints(int k) const override {
        return (k == terminal_stage) ? 0 : NC;
    }

    Status evaluate(const Vec<NX>& x, const Vec<NU>& u, int /*k*/,
                    Vec<NC>& g) override {
        np_to_vec(eval_fn(vec_to_np(x), vec_to_np(u)).cast<py::array_t<double>>(), g);
        return Status::SUCCESS;
    }

    Status evaluate_terminal(const Vec<NX>& x, Vec<NC>& g) override {
        np_to_vec(eval_term_fn(vec_to_np(x)).cast<py::array_t<double>>(), g);
        return Status::SUCCESS;
    }

    Status jacobian(const Vec<NX>& x, const Vec<NU>& u, int /*k*/,
                    Mat<NC, NX>& Cx, Mat<NC, NU>& Cu) override {
        py::tuple r = jac_fn(vec_to_np(x), vec_to_np(u)).cast<py::tuple>();
        auto Cx_np = r[0].cast<py::array_t<double>>().unchecked<2>();
        auto Cu_np = r[1].cast<py::array_t<double>>().unchecked<2>();
        for (int c = 0; c < NX; ++c)
            for (int rr = 0; rr < NC; ++rr)
                Cx.data[c * NC + rr] = Cx_np(rr, c);
        for (int c = 0; c < NU; ++c)
            for (int rr = 0; rr < NC; ++rr)
                Cu.data[c * NC + rr] = Cu_np(rr, c);
        return Status::SUCCESS;
    }

    Status jacobian_terminal(const Vec<NX>& x, Mat<NC, NX>& Cx) override {
        auto Cx_np = jac_term_fn(vec_to_np(x)).cast<py::array_t<double>>().unchecked<2>();
        for (int c = 0; c < NX; ++c)
            for (int rr = 0; rr < NC; ++rr)
                Cx.data[c * NC + rr] = Cx_np(rr, c);
        return Status::SUCCESS;
    }

    bool provides_adjoint_hessian() const override {
        return adj_hess_fn && !adj_hess_fn.is_none();
    }

    // Adds  Σ_j λ[j]·∇²g_j  into the (pre-populated) Hessian blocks.
    Status adjoint_hessian(const Vec<NX>& x, const Vec<NU>& u, int /*k*/,
                           const Vec<NC>& lambda,
                           Mat<NX, NX>& Hxx, Mat<NU, NX>& Hux,
                           Mat<NU, NU>& Huu) override {
        py::tuple r = adj_hess_fn(vec_to_np(x), vec_to_np(u),
                                  vec_to_np(lambda)).cast<py::tuple>();
        auto Hxx_np = r[0].cast<py::array_t<double>>().unchecked<2>();
        auto Hux_np = r[1].cast<py::array_t<double>>().unchecked<2>();
        auto Huu_np = r[2].cast<py::array_t<double>>().unchecked<2>();
        for (int c = 0; c < NX; ++c)
            for (int rr = 0; rr < NX; ++rr) Hxx(rr, c) += Hxx_np(rr, c);
        for (int c = 0; c < NX; ++c)
            for (int rr = 0; rr < NU; ++rr) Hux(rr, c) += Hux_np(rr, c);
        for (int c = 0; c < NU; ++c)
            for (int rr = 0; rr < NU; ++rr) Huu(rr, c) += Huu_np(rr, c);
        return Status::SUCCESS;
    }

    Status adjoint_hessian_terminal(const Vec<NX>& x, const Vec<NC>& lambda,
                                    Mat<NX, NX>& Hxx) override {
        if (!adj_hess_term_fn || adj_hess_term_fn.is_none()) return Status::SUCCESS;
        auto Hxx_np = adj_hess_term_fn(vec_to_np(x), vec_to_np(lambda))
                          .cast<py::array_t<double>>().unchecked<2>();
        for (int c = 0; c < NX; ++c)
            for (int rr = 0; rr < NX; ++rr) Hxx(rr, c) += Hxx_np(rr, c);
        return Status::SUCCESS;
    }
};

// ─────────────────────────────────────────────────────────────────────────────
//  Templated solve function
// ─────────────────────────────────────────────────────────────────────────────

template <int NX, int NU, int NC, int HORIZON>
std::tuple<py::array_t<double>, py::array_t<double>, py::dict>
solve_impl(
    py::object f_disc, py::object f_lin,
    py::object stage_cost_fn, py::object term_cost_fn,
    py::object stage_grad_fn, py::object stage_hess_fn,
    py::object term_grad_fn, py::object term_hess_fn,
    py::object constr_eval, py::object constr_eval_term,
    py::object constr_jac, py::object constr_jac_term,
    py::object dyn_adj_hess, py::object con_adj_hess, py::object con_adj_hess_term,
    bool has_constraints,
    py::array_t<double> x_lb_np, py::array_t<double> x_ub_np,
    py::array_t<double> u_lb_np, py::array_t<double> u_ub_np,
    py::array_t<double> x0_np,
    py::array_t<double> X_guess_np, py::array_t<double> U_guess_np,
    double mu_init, int max_iters, double tol_primal,
    double tol_compl, double tol_stat, int verbosity, double dt, double mu_min,
    double reg_max = 1e12, double inertia_min_pivot = 0.0)
{
    using Problem = NMPCProblem<NX, NU, NC, HORIZON>;

    // Create callback models
    auto dyn  = std::make_unique<PyDynModel<NX, NU>>();
    auto cost = std::make_unique<PyCostModel<NX, NU>>();
    std::unique_ptr<PyConstrModel<NX, NU, NC>> cons;

    dyn->f_disc = f_disc;
    dyn->f_lin  = f_lin;
    dyn->f_adj_hess = dyn_adj_hess;

    cost->stage_cost_fn  = stage_cost_fn;
    cost->term_cost_fn   = term_cost_fn;
    cost->stage_grad_fn  = stage_grad_fn;
    cost->stage_hess_fn  = stage_hess_fn;
    cost->term_grad_fn   = term_grad_fn;
    cost->term_hess_fn   = term_hess_fn;

    if (has_constraints) {
        cons = std::make_unique<PyConstrModel<NX, NU, NC>>();
        cons->terminal_stage = HORIZON;
        cons->eval_fn      = constr_eval;
        cons->eval_term_fn = constr_eval_term;
        cons->jac_fn       = constr_jac;
        cons->jac_term_fn  = constr_jac_term;
        cons->adj_hess_fn      = con_adj_hess;
        cons->adj_hess_term_fn = con_adj_hess_term;
    }

    // Set up problem
    Problem prob;
    prob.dynamics    = dyn.get();
    prob.cost        = cost.get();
    prob.constraints = has_constraints ? static_cast<ConstraintModel<NX, NU, NC>*>(cons.get()) : nullptr;
    prob.dt          = dt;

    np_to_vec(x0_np, prob.x0);

    // Bounds
    auto xlb = x_lb_np.unchecked<1>();
    auto xub = x_ub_np.unchecked<1>();
    auto ulb = u_lb_np.unchecked<1>();
    auto uub = u_ub_np.unchecked<1>();
    prob.n_bound_x = 0;
    prob.n_bound_u = 0;
    for (int i = 0; i < NX; ++i) {
        prob.x_lb[i] = xlb(i);
        prob.x_ub[i] = xub(i);
        if (xlb(i) > -1e19 || xub(i) < 1e19) prob.n_bound_x++;
    }
    for (int i = 0; i < NU; ++i) {
        prob.u_lb[i] = ulb(i);
        prob.u_ub[i] = uub(i);
        if (ulb(i) > -1e19 || uub(i) < 1e19) prob.n_bound_u++;
    }

    // Initial guess
    auto Xg = X_guess_np.unchecked<2>();
    auto Ug = U_guess_np.unchecked<2>();
    for (int k = 0; k <= HORIZON; ++k) {
        for (int i = 0; i < NX; ++i) prob.stages[k].x[i] = Xg(k, i);
        if (k < HORIZON)
            for (int i = 0; i < NU; ++i) prob.stages[k].u[i] = Ug(k, i);
        if (has_constraints) {
            for (int j = 0; j < NC; ++j) {
                prob.stages[k].s[j] = 1.0;
                prob.stages[k].lambda[j] = 0.1;
            }
        }
    }

    // Configure and solve
    auto solver = std::make_unique<ContactIPM<NX, NU, NC, HORIZON>>();
    ContactIPMParams pp;
    pp.mu_init     = mu_init;
    pp.max_iters   = max_iters;
    pp.tol_primal  = tol_primal;
    pp.tol_compl   = tol_compl;
    pp.tol_stat    = tol_stat;
    pp.tol_ineq    = 1e-8;
    pp.mu_min      = mu_min;
    pp.verbosity   = verbosity;
    pp.reg_max            = reg_max;
    pp.inertia_min_pivot  = inertia_min_pivot;
    // Select scaling from problem structure, not template dimensions.
    // Bound-aware scaling is used whenever finite variable bounds add
    // barrier curvature; otherwise use fixed derivative-based scaling.
    const bool has_variable_bounds =
        (prob.n_bound_x > 0) || (prob.n_bound_u > 0);
    pp.enable_preconditioner = has_variable_bounds || has_constraints;
    pp.bound_aware_preconditioner = has_variable_bounds;

    // Environment overrides are useful for controlled A/B experiments while
    // retaining the structure-based defaults above.
    {
        const char* pc = std::getenv("CONTACTIPM_PRECONDITIONER");
        if (pc && pc[0] == '0') pp.enable_preconditioner = false;
        else if (pc && pc[0] == '1') pp.enable_preconditioner = true;

        const char* min_it = std::getenv("CONTACTIPM_GN_EXACT_MIN_ITERS");
        if (min_it) pp.gn_exact_min_iters = std::max(1, std::atoi(min_it));
        const char* window = std::getenv("CONTACTIPM_GN_EXACT_STALL_WINDOW");
        if (window) pp.gn_exact_stall_window = std::max(2, std::atoi(window));

        const char* rollout = std::getenv("CONTACTIPM_NONLINEAR_ROLLOUT");
        if (rollout && rollout[0] == '0') pp.enable_nonlinear_rollout = false;
        else if (rollout && rollout[0] == '1') pp.enable_nonlinear_rollout = true;

    }
    // Exact (full-Lagrangian) Hessian override for controlled experiments:
    //   CONTACTIPM_EXACT_HESSIAN=1 forces exact-Newton curvature,
    //   CONTACTIPM_EXACT_HESSIAN=0 forces Gauss-Newton,
    //   unset             → use the C++ default (true, with reg_max fallback).
    {
        const char* eh = std::getenv("CONTACTIPM_EXACT_HESSIAN");
        if (eh && eh[0] == '1') pp.exact_hessian = true;
        else if (eh && eh[0] == '0') pp.exact_hessian = false;
        // else leave pp.exact_hessian at the C++ default
    }
    solver->configure(pp);

    auto t_start = std::chrono::high_resolution_clock::now();
    Status st = solver->solve(prob);
    auto t_end = std::chrono::high_resolution_clock::now();
    double solve_time = std::chrono::duration<double>(t_end - t_start).count();

    // Extract solution
    auto X_out = py::array_t<double>({HORIZON + 1, NX});
    auto U_out = py::array_t<double>({HORIZON, NU});
    auto X_buf = X_out.mutable_unchecked<2>();
    auto U_buf = U_out.mutable_unchecked<2>();
    for (int k = 0; k <= HORIZON; ++k) {
        for (int i = 0; i < NX; ++i) X_buf(k, i) = prob.stages[k].x[i];
        if (k < HORIZON)
            for (int i = 0; i < NU; ++i) U_buf(k, i) = prob.stages[k].u[i];
    }

    const auto& stats = solver->last_stats();
    py::dict result_stats;
    result_stats["status"]          = status_string(st);
    result_stats["iterations"]      = stats.inner_iterations;
    result_stats["time_solver"]     = solve_time;
    result_stats["cost"]            = stats.cost;
    result_stats["primal_infeas"]   = stats.primal_infeas;
    result_stats["dual_infeas"]     = stats.dual_infeas;
    result_stats["complementarity"] = stats.complementarity;
    result_stats["barrier_param"]   = stats.barrier_param;

    return std::make_tuple(X_out, U_out, result_stats);
}

// ─────────────────────────────────────────────────────────────────────────────
//  PyNMPCSolver wrapper (dispatches to template instantiations)
// ─────────────────────────────────────────────────────────────────────────────

class PyNMPCSolver {
public:
    PyNMPCSolver(int nx, int nu, int nc, int N, double dt)
        : nx_(nx), nu_(nu), nc_(nc), N_(N), dt_(dt) {}

    // Store callbacks and params for later dispatch
    void set_callbacks(
        py::object f_disc, py::object f_lin,
        py::object stage_cost, py::object term_cost,
        py::object stage_grad, py::object stage_hess,
        py::object term_grad, py::object term_hess,
        py::object constr_eval, py::object constr_eval_term,
        py::object constr_jac, py::object constr_jac_term,
        bool has_constraints,
        py::object dyn_adj_hess = py::none(),
        py::object con_adj_hess = py::none(),
        py::object con_adj_hess_term = py::none())
    {
        f_disc_ = f_disc; f_lin_ = f_lin;
        stage_cost_ = stage_cost; term_cost_ = term_cost;
        stage_grad_ = stage_grad; stage_hess_ = stage_hess;
        term_grad_ = term_grad; term_hess_ = term_hess;
        constr_eval_ = constr_eval; constr_eval_term_ = constr_eval_term;
        constr_jac_ = constr_jac; constr_jac_term_ = constr_jac_term;
        has_constraints_ = has_constraints;
        dyn_adj_hess_ = dyn_adj_hess;
        con_adj_hess_ = con_adj_hess;
        con_adj_hess_term_ = con_adj_hess_term;
    }

    void set_bounds(py::array_t<double> x_lb, py::array_t<double> x_ub,
                    py::array_t<double> u_lb, py::array_t<double> u_ub) {
        x_lb_ = x_lb; x_ub_ = x_ub; u_lb_ = u_lb; u_ub_ = u_ub;
    }

    void set_initial_state(py::array_t<double> x0) { x0_ = x0; }

    void set_params(double mu_init, int max_iters, double tol_primal,
                    double tol_compl, double tol_stat, int verbosity,
                    double mu_min = 5e-4,
                    double reg_max = 1e12,
                    double inertia_min_pivot = 0.0) {
        mu_init_ = mu_init; max_iters_ = max_iters;
        tol_primal_ = tol_primal; tol_compl_ = tol_compl;
        tol_stat_ = tol_stat; verbosity_ = verbosity;
        mu_min_ = mu_min;
        reg_max_ = reg_max;
        inertia_min_pivot_ = inertia_min_pivot;
    }

    std::tuple<py::array_t<double>, py::array_t<double>, py::dict>
    solve(py::array_t<double> X_guess, py::array_t<double> U_guess) {
        // Dispatch based on dimensions
        #define DISPATCH(NX, NU, NC, N) \
            return solve_impl<NX, NU, NC, N>( \
                f_disc_, f_lin_, stage_cost_, term_cost_, \
                stage_grad_, stage_hess_, term_grad_, term_hess_, \
                constr_eval_, constr_eval_term_, constr_jac_, constr_jac_term_, \
                dyn_adj_hess_, con_adj_hess_, con_adj_hess_term_, \
                has_constraints_, x_lb_, x_ub_, u_lb_, u_ub_, x0_, \
                X_guess, U_guess, \
                mu_init_, max_iters_, tol_primal_, tol_compl_, tol_stat_, \
                verbosity_, dt_, mu_min_, reg_max_, inertia_min_pivot_)

        if      (nx_ == 4  && nu_ == 1 && N_ == 25) { DISPATCH(4,  1, 1, 25); }
        else if (nx_ == 6  && nu_ == 4 && N_ == 25) { DISPATCH(6,  4, 1, 25); }
        else if (nx_ == 7  && nu_ == 4 && N_ == 50) { DISPATCH(7,  4, 1, 50); }
        else if (nx_ == 26 && nu_ == 2 && N_ == 25) { DISPATCH(26, 2, 1, 25); }
        else if (nx_ == 39 && nu_ == 3 && N_ == 25) { DISPATCH(39, 3, 1, 25); }
        else {
            py::dict stats;
            stats["status"] = "UnsupportedDimensions";
            stats["iterations"] = 0;
            stats["time_solver"] = 0.0;
            auto X = py::array_t<double>({N_ + 1, nx_});
            auto U = py::array_t<double>({N_, nu_});
            return std::make_tuple(X, U, stats);
        }
        #undef DISPATCH
    }

    int nx() const { return nx_; }
    int nu() const { return nu_; }
    int nc() const { return nc_; }
    int N()  const { return N_; }

private:
    int nx_, nu_, nc_, N_;
    double dt_;

    // Callbacks
    py::object f_disc_, f_lin_;
    py::object stage_cost_, term_cost_;
    py::object stage_grad_, stage_hess_;
    py::object term_grad_, term_hess_;
    py::object constr_eval_, constr_eval_term_;
    py::object constr_jac_, constr_jac_term_;
    py::object dyn_adj_hess_, con_adj_hess_, con_adj_hess_term_;
    bool has_constraints_ = false;

    // Bounds and state
    py::array_t<double> x_lb_, x_ub_, u_lb_, u_ub_, x0_;

    // Parameters
    double mu_init_ = 1.0;
    int max_iters_ = 100;
    double tol_primal_ = 1e-6;
    double tol_compl_ = 1e-6;
    double tol_stat_ = 0.5;
    int verbosity_ = 0;
    double mu_min_ = 5e-4;
    double reg_max_ = 1e12;
    double inertia_min_pivot_ = 0.0;
};

// ─────────────────────────────────────────────────────────────────────────────
//  pybind11 module
// ─────────────────────────────────────────────────────────────────────────────

PYBIND11_MODULE(contactipm, m) {
    m.doc() = "Python bindings for ContactIPM NMPC solver";

    py::class_<PyNMPCSolver>(m, "Solver")
        .def(py::init<int, int, int, int, double>(),
             py::arg("nx"), py::arg("nu"), py::arg("nc"), py::arg("N"), py::arg("dt"))

        .def("set_callbacks", &PyNMPCSolver::set_callbacks,
             py::arg("f_disc"), py::arg("f_lin"),
             py::arg("stage_cost"), py::arg("term_cost"),
             py::arg("stage_grad"), py::arg("stage_hess"),
             py::arg("term_grad"), py::arg("term_hess"),
             py::arg("constr_eval"), py::arg("constr_eval_term"),
             py::arg("constr_jac"), py::arg("constr_jac_term"),
             py::arg("has_constraints"),
             py::arg("dyn_adj_hess") = py::none(),
             py::arg("con_adj_hess") = py::none(),
             py::arg("con_adj_hess_term") = py::none())

        .def("set_bounds", &PyNMPCSolver::set_bounds,
             py::arg("x_lb"), py::arg("x_ub"),
             py::arg("u_lb"), py::arg("u_ub"))

        .def("set_initial_state", &PyNMPCSolver::set_initial_state,
             py::arg("x0"))

        .def("set_params", &PyNMPCSolver::set_params,
             py::arg("mu_init") = 1.0, py::arg("max_iters") = 100,
             py::arg("tol_primal") = 1e-6, py::arg("tol_compl") = 1e-6,
             py::arg("tol_stat") = 0.5, py::arg("verbosity") = 0,
             py::arg("mu_min") = 5e-4,
             py::arg("reg_max") = 1e12,
             py::arg("inertia_min_pivot") = 0.0)

        .def("solve", &PyNMPCSolver::solve,
             py::arg("X_guess"), py::arg("U_guess"))

        .def_property_readonly("nx", &PyNMPCSolver::nx)
        .def_property_readonly("nu", &PyNMPCSolver::nu)
        .def_property_readonly("nc", &PyNMPCSolver::nc)
        .def_property_readonly("N",  &PyNMPCSolver::N);

    m.attr("__version__") = "0.2.0";
}
