/**
 * @file    pendulum_nmpc_paper.cpp
 * @brief   Inverted pendulum NMPC using the paper-aligned Mehrotra IPM solver.
 *
 * This demonstrates the paper-aligned solver (nmpc_ipm_paper.hpp) on the
 * same cart-pole problem as pendulum_nmpc.cpp, using the Mehrotra
 * predictor-corrector IPM directly on the barrier KKT (no SQP outer loop).
 *
 * The convexity assumption means the problem is benign enough that
 * Newton on the KKT converges without regularization or filter line search.
 */

#include <cstdio>
#include <cmath>
#include <chrono>

#ifndef M_PI
#define M_PI 3.14159265358979323846
#endif

#include "nmpc/nmpc_solver_paper.hpp"
#include "trajectory_dump.hpp"

using namespace nmpc;

constexpr int NX = 4;   // [x, θ, ẋ, θ̇]
constexpr int NU = 1;   // [F]
constexpr int NC = 1;   // no explicit constraints; bounds via variable bounds
constexpr int N  = 20;  // horizon

using VecX = Vec<NX>;
using VecU = Vec<NU>;
using VecC = Vec<1>;   // dummy (NC=1 but template needs >0)
using MatXX = Mat<NX, NX>;
using MatXU = Mat<NX, NU>;
using MatCX = Mat<1, NX>;
using MatCU = Mat<1, NU>;

// ── Simplified cart-pole dynamics (same as before) ──────────────────────────

constexpr double M_cart = 1.0, m_pole = 0.2, l_pole = 0.5, g_grav = 9.81;
constexpr double DT = 0.05;

struct CartPoleDyn : DynamicsModel<NX, NU> {
    Status discrete_step(const VecX& x, const VecU& u, double dt, VecX& nx) override {
        auto f = [](const VecX& s, const VecU& c, VecX& dx) {
            double t = s[1], dt = s[3], F = c[0];
            double st = std::sin(t), ct = std::cos(t);
            double den = M_cart + m_pole * st * st;
            dx[0] = s[2];
            dx[1] = s[3];
            dx[2] = (F + m_pole*st*(l_pole*dt*dt + g_grav*ct)) / std::max(den, 1e-6);
            dx[3] = (-F*ct - m_pole*l_pole*dt*dt*st*ct - (M_cart+m_pole)*g_grav*st)
                    / std::max(l_pole*den, 1e-9);
        };
        VecX k1,k2,k3,k4,tmp;
        f(x,u,k1);
        for(int i=0;i<NX;++i) tmp[i]=x[i]+0.5*dt*k1[i]; f(tmp,u,k2);
        for(int i=0;i<NX;++i) tmp[i]=x[i]+0.5*dt*k2[i]; f(tmp,u,k3);
        for(int i=0;i<NX;++i) tmp[i]=x[i]+dt*k3[i];     f(tmp,u,k4);
        for(int i=0;i<NX;++i) nx[i]=x[i]+(dt/6.0)*(k1[i]+2*k2[i]+2*k3[i]+k4[i]);
        return Status::SUCCESS;
    }
    Status linearize(const VecX& x, const VecU& u, double dt, MatXX& A, MatXU& B) override {
        const double eps = 1e-6;
        VecX xp,xm,fp,fm;
        for(int j=0;j<NX;++j){ xp=x;xp[j]+=eps; xm=x;xm[j]-=eps; discrete_step(xp,u,dt,fp); discrete_step(xm,u,dt,fm); for(int i=0;i<NX;++i) A(i,j)=(fp[i]-fm[i])/(2*eps); }
        VecU up,um;
        for(int j=0;j<NU;++j){ up=u;up[j]+=eps; um=u;um[j]-=eps; discrete_step(x,up,dt,fp); discrete_step(x,um,dt,fm); for(int i=0;i<NX;++i) B(i,j)=(fp[i]-fm[i])/(2*eps); }
        return Status::SUCCESS;
    }
};

struct CartPoleCost : CostModel<NX, NU> {
    static constexpr double wx=1.0, wt=10.0, wv=0.1, ww=0.5, wr=0.01;
    double stage_cost(const VecX& x, const VecU& u, int) override {
        double dx=x[0], dt=x[1]-M_PI, dv=x[2], dw=x[3];
        return 0.5*(wx*dx*dx + wt*dt*dt + wv*dv*dv + ww*dw*dw + wr*u[0]*u[0]);
    }
    double terminal_cost(const VecX& x) override {
        double dx=x[0], dt=x[1]-M_PI, dv=x[2], dw=x[3];
        return 5.0*(wx*dx*dx + wt*dt*dt + wv*dv*dv + ww*dw*dw);
    }
    Status stage_gradient(const VecX& x, const VecU& u, int, VecX& qx, VecU& qu) override {
        qx[0]=wx*x[0]; qx[1]=wt*(x[1]-M_PI); qx[2]=wv*x[2]; qx[3]=ww*x[3]; qu[0]=wr*u[0]; return Status::SUCCESS;
    }
    Status stage_hessian(const VecX&, const VecU&, int, MatXX& Qxx, Mat<NU,NU>& Quu, Mat<NU,NX>& Qux) override {
        Qxx.zero(); Qxx(0,0)=wx; Qxx(1,1)=wt; Qxx(2,2)=wv; Qxx(3,3)=ww; Quu.zero(); Quu(0,0)=wr; Qux.zero(); return Status::SUCCESS;
    }
    Status terminal_gradient(const VecX& x, VecX& qx) override {
        qx[0]=10*wx*x[0]; qx[1]=10*wt*(x[1]-M_PI); qx[2]=10*wv*x[2]; qx[3]=10*ww*x[3]; return Status::SUCCESS;
    }
    Status terminal_hessian(const VecX&, MatXX& Qxx) override {
        Qxx.zero(); Qxx(0,0)=10*wx; Qxx(1,1)=10*wt; Qxx(2,2)=10*wv; Qxx(3,3)=10*ww; return Status::SUCCESS;
    }
};


int main() {
    printf("═══════════════════════════════════════════════════════\n");
    printf("  Inverted Pendulum NMPC — Cart-Pole Swing-Up\n");
    printf("═══════════════════════════════════════════════════════\n\n");

    CartPoleDyn dyn; CartPoleCost cost;

    using Problem = NMPCProblem<NX, NU, NC, N>;
    Problem prob;
    prob.dynamics=&dyn; prob.cost=&cost; prob.constraints=nullptr; prob.dt=DT;

    // Variable bounds: |x| <= 2, |u| <= 30
    prob.init_bounds_free();
    prob.x_lb[0] = -2.0; prob.x_ub[0] = 2.0;
    prob.n_bound_x = 1;
    prob.u_lb[0] = -30.0; prob.u_ub[0] = 30.0;
    prob.n_bound_u = 1;

    // Initial state: slightly off upright, at rest
    VecX x0;
    x0[0]=0; x0[1]=M_PI-0.3; x0[2]=0; x0[3]=0;
    prob.x0 = x0;

    for(int k=0;k<=N;++k){
        prob.stages[k].x = x0;
        prob.stages[k].u.zero();
    }

    NMPCSolverPaper<NX, NU, NC, N> solver;
    PaperIPMParams pp;
    pp.mu_init=0.2; pp.max_iters=200;
    // Matched tolerances for fair comparison vs acados.
    pp.mu_min=1e-4;
    pp.tol_primal=1e-4; pp.tol_compl=1e-4; pp.tol_ineq=1e-4; pp.tol_stat = 0.02;
    pp.enable_preconditioner = true;  // diagonal Jacobi preconditioner
    pp.verbosity=2;
    solver.configure(pp);

    // ── Debug: single verbose solve ───────
    {
        Status dbg_st = solver.solve(prob);
        printf("\n=== DEBUG SOLVE ===\n");
        printf("Status: %s\n", status_string(dbg_st));
        printf("Iterations: %d\n", solver.last_stats().inner_iterations);
        printf("Cost: %.4f\n", solver.last_stats().cost);
    }

    // Reset for timing loop
    prob.x0 = x0;
    for(int k=0;k<=N;++k){
        prob.stages[k].x = x0;
        prob.stages[k].u.zero();
    }

    // ── Warm in-process timing loop (mirrors acados NTIMINGS=5) ───────
    StageData<NX, NU, NC> guess[N + 1];
    Vec<NX> x0_saved = prob.x0;
    for (int k = 0; k <= N; ++k) guess[k] = prob.stages[k];

    constexpr int NTIMINGS = 5;
    double times_ms[NTIMINGS];
    double min_ms = 1e12;
    Status st = Status::SUCCESS;
    for (int ii = 0; ii < NTIMINGS; ++ii) {
        prob.x0 = x0_saved;
        for (int k = 0; k <= N; ++k) prob.stages[k] = guess[k];
        PaperIPMParams quiet = pp; quiet.verbosity = 0;
        solver.configure(quiet);
        auto t_start = std::chrono::high_resolution_clock::now();
        st = solver.solve(prob);
        auto t_end = std::chrono::high_resolution_clock::now();
        times_ms[ii] = std::chrono::duration<double, std::milli>(t_end - t_start).count();
        if (times_ms[ii] < min_ms) min_ms = times_ms[ii];
    }
    for (int i = 0; i < NTIMINGS; ++i)
        for (int j = i + 1; j < NTIMINGS; ++j)
            if (times_ms[j] < times_ms[i]) { double tmp = times_ms[i]; times_ms[i] = times_ms[j]; times_ms[j] = tmp; }
    double median_ms = times_ms[NTIMINGS / 2];

    const auto& s = solver.last_stats();

    printf("\n=== SOLVE COMPLETE ===\n");
    printf("Status: %s\n", status_string(st));
    printf("Iterations: %d\n", s.inner_iterations);
    printf("Final mu: %.3e\n", s.barrier_param);
    printf("Primal inf: %.3e\n", s.primal_infeas);
    printf("Dual inf: %.3e\n", s.dual_infeas);
    printf("Complementarity: %.3e\n", s.complementarity);
    printf("Ineq viol: %.3e\n", s.condition_estimate);
    printf("SOC steps: %d\n", s.soc_steps);
    printf("Cost: %.4f\n", s.cost);
    printf("First u* = [%.3f]\n", prob.stages[0].u[0]);

    // ── Standardized summary (parsed by run_all.py) ──────────────────
    printf("\n=== BENCHMARK SUMMARY ===\n");
    printf("Status:          %s\n", status_string(st));
    printf("Iterations:      %d\n", s.inner_iterations);
    printf("Solve time:      %.3f ms\n", min_ms);
    printf("Median time:     %.3f ms\n", median_ms);
    printf("Primal inf:      %.3e\n", s.primal_infeas);
    printf("Stationarity:    %.3e\n", s.dual_infeas);
    printf("Complementarity: %.3e\n", s.complementarity);
    printf("Cost:            %.4f\n", s.cost);
    printf("First u*:        [%.3f]\n", prob.stages[0].u[0]);

    // Dump trajectory
    print_trajectory_table(prob);
    dump_trajectory_json(prob, "benchmarks/data/contactipm_pendulum.json",
                         "pendulum", s.inner_iterations, s.cost, DT);
    return 0;
}
