/**
 * @file    diag_quad_mpc.cpp
 * @brief   Isolate why quadcopter_mpc fails: test combinations of cost/dynamics/constraints
 */
#include <cstdio>
#include <cmath>
#include <memory>
#include "fatrop_benchmark_common.hpp"
#include "quadcopter_mpc.c"

constexpr int NX = 6, NU = 4, NC = 7, N = 25;
constexpr double DT = 0.08;

// ── Identity cost (condition = 1) ──────────────────────────────────
struct IdentityCost : nmpc::CostModel<NX, NU> {
    double stage_cost(const nmpc::Vec<NX>& x, const nmpc::Vec<NU>& u, int k) override {
        double s = 0;
        s += x[0]*x[0] + x[1]*x[1] + (x[2]-2.5)*(x[2]-2.5);
        s += x[3]*x[3] + x[4]*x[4] + x[5]*x[5];
        s += u[0]*u[0] + u[1]*u[1] + u[2]*u[2] + (u[3]-9.81)*(u[3]-9.81);
        return 0.5*s;
    }
    double terminal_cost(const nmpc::Vec<NX>& x) override {
        double s = 0;
        for (int i = 0; i < NX; ++i) { double v = (i==2) ? x[2]-2.5 : x[i]; s += v*v; }
        return 50*s;
    }
    nmpc::Status stage_gradient(const nmpc::Vec<NX>& x, const nmpc::Vec<NU>& u, int k,
                          nmpc::Vec<NX>& qx, nmpc::Vec<NU>& qu) override {
        qx[0]=x[0]; qx[1]=x[1]; qx[2]=x[2]-2.5; qx[3]=x[3]; qx[4]=x[4]; qx[5]=x[5];
        qu[0]=u[0]; qu[1]=u[1]; qu[2]=u[2]; qu[3]=u[3]-9.81;
        return nmpc::Status::SUCCESS;
    }
    nmpc::Status stage_hessian(const nmpc::Vec<NX>&, const nmpc::Vec<NU>&, int,
                         nmpc::Mat<NX,NX>& Qxx, nmpc::Mat<NU,NU>& Quu,
                         nmpc::Mat<NU,NX>& Qux) override {
        Qxx.zero(); Quu.zero(); Qux.zero();
        for (int i=0;i<NX;++i) Qxx(i,i)=1;
        for (int i=0;i<NU;++i) Quu(i,i)=1;
        return nmpc::Status::SUCCESS;
    }
    nmpc::Status terminal_gradient(const nmpc::Vec<NX>& x, nmpc::Vec<NX>& qx) override {
        qx[0]=100*x[0]; qx[1]=100*x[1]; qx[2]=100*(x[2]-2.5);
        qx[3]=100*x[3]; qx[4]=100*x[4]; qx[5]=100*x[5];
        return nmpc::Status::SUCCESS;
    }
    nmpc::Status terminal_hessian(const nmpc::Vec<NX>&, nmpc::Mat<NX,NX>& Qxx) override {
        Qxx.zero(); for (int i=0;i<NX;++i) Qxx(i,i)=100;
        return nmpc::Status::SUCCESS;
    }
};

// ── No-op constraints (all inactive) ──────────────────────────────
struct NoCons : nmpc::ConstraintModel<NX, NU, NC> {
    nmpc::Status evaluate(const nmpc::Vec<NX>&, const nmpc::Vec<NU>&, int,
                          nmpc::Vec<NC>& g) override {
        for (int i=0;i<NC;++i) g[i]=-1e6;
        return nmpc::Status::SUCCESS;
    }
    nmpc::Status evaluate_terminal(const nmpc::Vec<NX>&, nmpc::Vec<NC>& g) override {
        for (int i=0;i<NC;++i) g[i]=-1e6;
        return nmpc::Status::SUCCESS;
    }
    nmpc::Status jacobian(const nmpc::Vec<NX>&, const nmpc::Vec<NU>&, int,
                          nmpc::Mat<NC,NX>& Cx, nmpc::Mat<NC,NU>& Cu) override {
        Cx.zero(); Cu.zero();
        return nmpc::Status::SUCCESS;
    }
    nmpc::Status jacobian_terminal(const nmpc::Vec<NX>&, nmpc::Mat<NC,NX>& Cx) override {
        Cx.zero();
        return nmpc::Status::SUCCESS;
    }
};

// ── Simple Euler quadcopter dynamics ──────────────────────────────
struct SimpleQuadDyn : nmpc::DynamicsModel<NX, NU> {
    static constexpr double grav = 9.81;

    nmpc::Status discrete_step(const nmpc::Vec<NX>& x, const nmpc::Vec<NU>& u,
                               double dt, nmpc::Vec<NX>& xnext) override {
        // x = [px, py, pz, vx, vy, vz], u = [phi, theta, psi, a_thrust]
        // State order: pos(0-2), vel(3-5)
        // Control: euler angles(0-2), thrust(3)
        // NOTE: fatrop uses x[3..5] = EULER ANGLES, u = [phi, theta, psi, a]
        // Wait — need to match fatrop's state ordering.
        // Fatrop quadcopter: x=[px,py,pz,phi,theta,psi], u=[??]
        // Let me check the fatrop state definition...
        // Actually, from the CasADi code, the state is [px,py,pz,vx,vy,vz]
        // and controls include euler angles + thrust
        // But wait — x[3..5] are Euler angles per the cost: 1e3*psi^2
        // So state = [px, py, pz, phi, theta, psi] and u = [??]
        // This doesn't match typical quadcopter. Let me use FD on CasADi instead.
        return nmpc::Status::SUCCESS; // placeholder
    }

    nmpc::Status linearize(const nmpc::Vec<NX>& x, const nmpc::Vec<NU>& u,
                           double dt, nmpc::Mat<NX,NX>& A, nmpc::Mat<NX,NU>& B) override {
        return nmpc::Status::SUCCESS; // placeholder
    }
};

// ── FD wrapper around CasADi ──────────────────────────────────────
struct FDDynamics : nmpc::DynamicsModel<NX, NU> {
    nmpc::CodegenDynamics<NX, NU>* inner;
    nmpc::Status discrete_step(const nmpc::Vec<NX>& x, const nmpc::Vec<NU>& u,
                               double dt, nmpc::Vec<NX>& x_next) override {
        return inner->discrete_step(x, u, dt, x_next);
    }
    nmpc::Status linearize(const nmpc::Vec<NX>& x, const nmpc::Vec<NU>& u, double dt,
                           nmpc::Mat<NX,NX>& A, nmpc::Mat<NX,NU>& B) override {
        const double eps = 1e-6;
        nmpc::Vec<NX> xp, xm, fp, fm;
        for (int j = 0; j < NX; ++j) {
            xp = x; xp[j] += eps; xm = x; xm[j] -= eps;
            discrete_step(xp, u, dt, fp); discrete_step(xm, u, dt, fm);
            for (int i = 0; i < NX; ++i) A(i,j) = (fp[i]-fm[i])/(2*eps);
        }
        nmpc::Vec<NU> up, um;
        for (int j = 0; j < NU; ++j) {
            up = u; up[j] += eps; um = u; um[j] -= eps;
            discrete_step(x, up, dt, fp); discrete_step(x, um, dt, fm);
            for (int i = 0; i < NX; ++i) B(i,j) = (fp[i]-fm[i])/(2*eps);
        }
        return nmpc::Status::SUCCESS;
    }
};

void run_test(const char* label,
              nmpc::NMPCProblem<NX,NU,NC,N>& prob,
              nmpc::NMPCSolverPaper<NX,NU,NC,N>& solver,
              const nmpc::Vec<NX>& x0, const nmpc::Vec<NU>& u0) {
    // Forward-simulate initial guess
    prob.x0 = x0;
    prob.stages[0].x = x0;
    prob.stages[0].u = u0;
    for (int k = 0; k < N; ++k) {
        prob.dynamics->discrete_step(prob.stages[k].x, prob.stages[k].u, DT, prob.stages[k+1].x);
        prob.stages[k+1].u = u0;
    }

    printf("\n%s\n", label);
    auto st = solver.solve(prob);
    const auto& s = solver.last_stats();
    printf("  Status: %s | iter=%d | prim=%.2e | stat=%.2e | compl=%.2e | cost=%.3e\n",
           nmpc::status_string(st), s.inner_iterations,
           s.primal_infeas, s.dual_infeas, s.complementarity, s.cost);
}

void compare_jacobians(nmpc::CodegenDynamics<NX,NU>& dyn, FDDynamics& fd_dyn,
                       const nmpc::Vec<NX>& x, const nmpc::Vec<NU>& u, const char* label) {
    nmpc::Mat<NX,NX> A_cas, A_fd;
    nmpc::Mat<NX,NU> B_cas, B_fd;
    dyn.linearize(x, u, DT, A_cas, B_cas);
    fd_dyn.linearize(x, u, DT, A_fd, B_fd);
    double max_A = 0, max_B = 0;
    for (int i = 0; i < NX*NX; ++i) max_A = std::max(max_A, std::fabs(A_cas.data[i]-A_fd.data[i]));
    for (int i = 0; i < NX*NU; ++i) max_B = std::max(max_B, std::fabs(B_cas.data[i]-B_fd.data[i]));
    printf("  %s: max|dA|=%.3e max|dB|=%.3e\n", label, max_A, max_B);
}

int main() {
    printf("=== QuadCopterMPC ISOLATION TESTS ===\n");

    // CasADi models
    nmpc::CodegenDynamics<NX, NU> dyn;
    dyn.set_functions(f_expl, f_jac_x, f_jac_u);

    nmpc::CodegenCost<NX, NU> cost;
    cost.l_stage_fn = l_stage; cost.l_stage_grad_x_fn = l_stage_grad_x;
    cost.l_stage_grad_u_fn = l_stage_grad_u; cost.l_stage_hess_xx_fn = l_stage_hess_xx;
    cost.l_stage_hess_uu_fn = l_stage_hess_uu; cost.l_stage_hess_ux_fn = l_stage_hess_ux;
    cost.l_term_fn = l_term; cost.l_term_grad_fn = l_term_grad; cost.l_term_hess_fn = l_term_hess;

    nmpc::CodegenConstraints<NX, NU, NC> cons;
    cons.g_path_fn = g_path; cons.g_path_jac_x_fn = g_path_jac_x;
    cons.g_path_jac_u_fn = g_path_jac_u; cons.g_term_fn = g_term;
    cons.g_term_jac_fn = g_term_jac;

    // FD dynamics wrapper
    FDDynamics fd_dyn; fd_dyn.inner = &dyn;
    IdentityCost id_cost;
    NoCons no_cons;

    // Initial conditions
    nmpc::Vec<NX> x0; x0[0]=0; x0[1]=0; x0[2]=2.5; x0[3]=1; x0[4]=1; x0[5]=1;
    nmpc::Vec<NU> u0; u0[0]=9.81; u0[1]=M_PI/10; u0[2]=M_PI/10; u0[3]=M_PI/10;

    // ── Test 1: Original everything ──
    {
        nmpc::NMPCProblem<NX,NU,NC,N> prob;
        prob.dynamics = &dyn; prob.cost = &cost; prob.constraints = &cons; prob.dt = DT;
        auto solver = std::make_unique<nmpc::NMPCSolverPaper<NX,NU,NC,N>>();
        auto pp = fatrop_bench::default_params();
        pp.verbosity = 0; pp.max_iters = 500; pp.enable_refinement = false;
        solver->configure(pp);
        run_test("Test 1: ORIGINAL cost + CasADi dynamics + ORIGINAL constraints", prob, *solver, x0, u0);
    }

    // ── Test 2: Identity cost + CasADi dynamics + NO constraints ──
    {
        nmpc::NMPCProblem<NX,NU,NC,N> prob;
        prob.dynamics = &dyn; prob.cost = &id_cost; prob.constraints = &no_cons; prob.dt = DT;
        auto solver = std::make_unique<nmpc::NMPCSolverPaper<NX,NU,NC,N>>();
        auto pp = fatrop_bench::default_params();
        pp.verbosity = 0; pp.max_iters = 500; pp.enable_refinement = false;
        solver->configure(pp);
        run_test("Test 2: IDENTITY cost + CasADi dynamics + NO constraints", prob, *solver, x0, u0);
    }

    // ── Test 3: Identity cost + FD dynamics + NO constraints ──
    {
        nmpc::NMPCProblem<NX,NU,NC,N> prob;
        prob.dynamics = &fd_dyn; prob.cost = &id_cost; prob.constraints = &no_cons; prob.dt = DT;
        auto solver = std::make_unique<nmpc::NMPCSolverPaper<NX,NU,NC,N>>();
        auto pp = fatrop_bench::default_params();
        pp.verbosity = 0; pp.max_iters = 500; pp.enable_refinement = false;
        solver->configure(pp);
        run_test("Test 3: IDENTITY cost + FD dynamics + NO constraints", prob, *solver, x0, u0);
    }

    // ── Test 4: Original cost + FD dynamics + NO constraints ──
    {
        nmpc::NMPCProblem<NX,NU,NC,N> prob;
        prob.dynamics = &fd_dyn; prob.cost = &cost; prob.constraints = &no_cons; prob.dt = DT;
        auto solver = std::make_unique<nmpc::NMPCSolverPaper<NX,NU,NC,N>>();
        auto pp = fatrop_bench::default_params();
        pp.verbosity = 0; pp.max_iters = 500; pp.enable_refinement = false;
        solver->configure(pp);
        run_test("Test 4: ORIGINAL cost + FD dynamics + NO constraints", prob, *solver, x0, u0);
    }

    // ── Test 5: Identity cost + CasADi dynamics + ORIGINAL constraints ──
    {
        nmpc::NMPCProblem<NX,NU,NC,N> prob;
        prob.dynamics = &dyn; prob.cost = &id_cost; prob.constraints = &cons; prob.dt = DT;
        auto solver = std::make_unique<nmpc::NMPCSolverPaper<NX,NU,NC,N>>();
        auto pp = fatrop_bench::default_params();
        pp.verbosity = 0; pp.max_iters = 500; pp.enable_refinement = false;
        solver->configure(pp);
        run_test("Test 5: IDENTITY cost + CasADi dynamics + ORIGINAL constraints", prob, *solver, x0, u0);
    }

    // ── Jacobian comparison at multiple points ──
    printf("\n=== Jacobian comparison: CasADi RK4-sens vs FD-on-RK4 ===\n");
    {
        // Point 1: initial flat guess
        nmpc::Vec<NX> x1; x1[0]=0; x1[1]=0; x1[2]=2.5; x1[3]=1; x1[4]=1; x1[5]=1;
        nmpc::Vec<NU> u1; u1[0]=9.81; u1[1]=M_PI/10; u1[2]=M_PI/10; u1[3]=M_PI/10;
        compare_jacobians(dyn, fd_dyn, x1, u1, "Init point");
    }
    {
        // Point 2: nonzero velocities
        nmpc::Vec<NX> x2; x2[0]=1; x2[1]=-0.5; x2[2]=3.0; x2[3]=2; x2[4]=-1; x2[5]=0.5;
        nmpc::Vec<NU> u2; u2[0]=12; u2[1]=0.3; u2[2]=-0.2; u2[3]=0.5;
        compare_jacobians(dyn, fd_dyn, x2, u2, "Nonzero vel");
    }
    {
        // Point 3: large angles
        nmpc::Vec<NX> x3; x3[0]=2; x3[1]=1; x3[2]=2.5; x3[3]=5; x3[4]=3; x3[5]=-2;
        nmpc::Vec<NU> u3; u3[0]=15; u3[1]=1.0; u3[2]=0.8; u3[3]=1.5;
        compare_jacobians(dyn, fd_dyn, x3, u3, "Large values");
    }
    {
        // Point 4: Forward-simulated state at k=5
        nmpc::Vec<NX> x4; x4[0]=0; x4[1]=0; x4[2]=2.5; x4[3]=1; x4[4]=1; x4[5]=1;
        nmpc::Vec<NU> u4; u4[0]=9.81; u4[1]=M_PI/10; u4[2]=M_PI/10; u4[3]=M_PI/10;
        for (int k = 0; k < 5; ++k) {
            nmpc::Vec<NX> xn;
            dyn.discrete_step(x4, u4, DT, xn);
            x4 = xn;
        }
        compare_jacobians(dyn, fd_dyn, x4, u4, "After 5 steps");
        printf("    x=[%.3f %.3f %.3f %.3f %.3f %.3f]\n",
               x4[0], x4[1], x4[2], x4[3], x4[4], x4[5]);
    }

    return 0;
}
