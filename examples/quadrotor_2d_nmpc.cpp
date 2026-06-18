/**
 * @file    quadrotor_2d_nmpc.cpp
 * @brief   2D Quadrotor hover — evaluates the paper-aligned NMPC solver.
 *
 * State:   [y, z, phi, vy, vz, dphi]   (6D: lateral, vertical, roll, velocities)
 * Control: [u1, u2]                     (2D: total thrust, differential torque)
 *
 * Dynamics (planar quadrotor):
 *   ẏ   = vy
 *   ż   = vz
 *   φ̇   = dphi
 *   v̇y  = -u1 * sin(φ) / m
 *   v̇z  =  u1 * cos(φ) / m - g
 *   dphi̇ = u2 * l / I_yy
 *
 * Cost: quadratic tracking of hover at [y_des, z_des] with φ=0.
 *
 * Constraints:  0 ≤ u1 ≤ 2mg  (thrust bounds, hover = mg)
 *                |u2| ≤ 0.5 mgl (torque limit)
 *                z ≥ 0 (ground)
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

// ── Dimensions ──────────────────────────────────────────────────────────────
constexpr int NX = 6;   // [y, z, phi, vy, vz, dphi]
constexpr int NU = 2;   // [u1, u2]
constexpr int NC = 6;   // u1≥0, u1≤2mg, |u2|≤0.5mgl, z≥0
constexpr int N  = 20;  // horizon

// ── Quadrotor parameters ────────────────────────────────────────────────────
constexpr double MASS = 0.5;      // kg
constexpr double L_ARM = 0.15;    // m (arm length)
constexpr double I_YY = 0.003;    // kg·m²
constexpr double GRAV = 9.81;
constexpr double DT   = 0.04;     // 40 ms
constexpr double HOVER_THRUST = MASS * GRAV;  // ~4.9 N

// ── Type aliases ────────────────────────────────────────────────────────────
using VecX = Vec<NX>;
using VecU = Vec<NU>;
using VecC = Vec<NC>;
using MatXX = Mat<NX, NX>;
using MatXU = Mat<NX, NU>;
using MatCX = Mat<NC, NX>;
using MatCU = Mat<NC, NU>;

// ── Dynamics ────────────────────────────────────────────────────────────────

struct QuadDyn : DynamicsModel<NX, NU> {
    Status discrete_step(const VecX& x, const VecU& u, double dt,
                         VecX& nx) override {
        // RK4
        auto f = [](const VecX& s, const VecU& c, VecX& dx) {
            double phi = s[2], dphi = s[5];
            double u1  = c[0], u2 = c[1];
            double sp = std::sin(phi), cp = std::cos(phi);

            dx[0] = s[3];                      // ẏ = vy
            dx[1] = s[4];                      // ż = vz
            dx[2] = s[5];                      // φ̇ = dphi
            dx[3] = -u1 * sp / MASS;           // v̇y
            dx[4] =  u1 * cp / MASS - GRAV;    // v̇z
            dx[5] =  u2 * L_ARM / I_YY;        // φ̈
        };
        VecX k1, k2, k3, k4, tmp;
        f(x, u, k1);
        for (int i = 0; i < NX; ++i) tmp[i] = x[i] + 0.5*dt*k1[i]; f(tmp, u, k2);
        for (int i = 0; i < NX; ++i) tmp[i] = x[i] + 0.5*dt*k2[i]; f(tmp, u, k3);
        for (int i = 0; i < NX; ++i) tmp[i] = x[i] + dt*k3[i];     f(tmp, u, k4);
        for (int i = 0; i < NX; ++i) nx[i] = x[i] + (dt/6.0)*(k1[i]+2*k2[i]+2*k3[i]+k4[i]);
        return Status::SUCCESS;
    }

    Status linearize(const VecX& x, const VecU& u, double dt,
                     MatXX& A, MatXU& B) override {
        const double eps = 1e-6;
        VecX xp, xm, fp, fm;
        for (int j = 0; j < NX; ++j) {
            xp = x; xp[j] += eps; xm = x; xm[j] -= eps;
            discrete_step(xp, u, dt, fp); discrete_step(xm, u, dt, fm);
            for (int i = 0; i < NX; ++i) A(i,j) = (fp[i]-fm[i])/(2*eps);
        }
        VecU up, um;
        for (int j = 0; j < NU; ++j) {
            up = u; up[j] += eps; um = u; um[j] -= eps;
            discrete_step(x, up, dt, fp); discrete_step(x, um, dt, fm);
            for (int i = 0; i < NX; ++i) B(i,j) = (fp[i]-fm[i])/(2*eps);
        }
        return Status::SUCCESS;
    }
};

// ── Cost: hover at [y_des=0, z_des=2] with φ=0 ─────────────────────────────

struct QuadCost : CostModel<NX, NU> {
    double y_des = 0.0, z_des = 2.0;

    static constexpr double w_y=5.0, w_z=10.0, w_phi=2.0;
    static constexpr double w_vy=0.5, w_vz=1.0, w_dphi=0.3;
    static constexpr double w_u=0.001;  // cheap control

    double stage_cost(const VecX& x, const VecU& u, int) override {
        double dy=x[0]-y_des, dz=x[1]-z_des, dp=x[2];
        return 0.5*(w_y*dy*dy + w_z*dz*dz + w_phi*dp*dp
                    + w_vy*x[3]*x[3] + w_vz*x[4]*x[4] + w_dphi*x[5]*x[5]
                    + w_u*(u[0]*u[0] + u[1]*u[1]));
    }
    double terminal_cost(const VecX& x) override {
        double dy=x[0]-y_des, dz=x[1]-z_des, dp=x[2];
        return 10.0*(w_y*dy*dy + w_z*dz*dz + w_phi*dp*dp
                     + w_vy*x[3]*x[3] + w_vz*x[4]*x[4] + w_dphi*x[5]*x[5]);
    }
    Status stage_gradient(const VecX& x, const VecU& u, int,
                          VecX& qx, VecU& qu) override {
        qx[0]=w_y*(x[0]-y_des); qx[1]=w_z*(x[1]-z_des); qx[2]=w_phi*x[2];
        qx[3]=w_vy*x[3]; qx[4]=w_vz*x[4]; qx[5]=w_dphi*x[5];
        qu[0]=w_u*u[0]; qu[1]=w_u*u[1];
        return Status::SUCCESS;
    }
    Status stage_hessian(const VecX&, const VecU&, int,
                         MatXX& Qxx, Mat<NU,NU>& Quu, Mat<NU,NX>& Qux) override {
        Qxx.zero(); Qxx(0,0)=w_y; Qxx(1,1)=w_z; Qxx(2,2)=w_phi;
        Qxx(3,3)=w_vy; Qxx(4,4)=w_vz; Qxx(5,5)=w_dphi;
        Quu.zero(); Quu(0,0)=w_u; Quu(1,1)=w_u; Qux.zero();
        return Status::SUCCESS;
    }
    Status terminal_gradient(const VecX& x, VecX& qx) override {
        qx[0]=10*w_y*(x[0]-y_des); qx[1]=10*w_z*(x[1]-z_des); qx[2]=10*w_phi*x[2];
        qx[3]=10*w_vy*x[3]; qx[4]=10*w_vz*x[4]; qx[5]=10*w_dphi*x[5];
        return Status::SUCCESS;
    }
    Status terminal_hessian(const VecX&, MatXX& Qxx) override {
        Qxx.zero(); Qxx(0,0)=10*w_y; Qxx(1,1)=10*w_z; Qxx(2,2)=10*w_phi;
        Qxx(3,3)=10*w_vy; Qxx(4,4)=10*w_vz; Qxx(5,5)=10*w_dphi;
        return Status::SUCCESS;
    }
};

// ── Constraints: thrust bounds + ground ─────────────────────────────────────

struct QuadCons : ConstraintModel<NX, NU, NC> {
    Status evaluate(const VecX& x, const VecU& u, int, VecC& g) override {
        // g ≤ 0 formulation:
        //   -u1 ≤ 0          →  g[0] = -u1
        //   u1 - 2mg ≤ 0     →  g[1] = u1 - 2*HOVER_THRUST
        //   -u2 - 0.5mgl ≤ 0 →  g[2] = -u2 - 0.5*HOVER_THRUST*L_ARM
        //   u2 - 0.5mgl ≤ 0  →  g[3] = u2 - 0.5*HOVER_THRUST*L_ARM
        //   -z ≤ 0           →  g[4] = -x[1]  (ground)
        //   reserve          →  g[5] = -1e10  (inactive)
        g[0] = -u[0];
        g[1] = u[0] - 2.0 * HOVER_THRUST;
        g[2] = -u[1] - 0.5 * HOVER_THRUST * L_ARM;
        g[3] = u[1] - 0.5 * HOVER_THRUST * L_ARM;
        g[4] = -x[1];    // z ≥ 0
        g[5] = -1e10;    // inactive
        return Status::SUCCESS;
    }
    Status evaluate_terminal(const VecX& x, VecC& g) override {
        g[0] = -1e10; g[1] = -1e10; g[2] = -1e10;
        g[3] = -1e10; g[4] = -x[1]; g[5] = -1e10;
        return Status::SUCCESS;
    }
    Status jacobian(const VecX&, const VecU&, int, MatCX& Cx, MatCU& Cu) override {
        Cx.zero(); Cu.zero();
        Cx(4,1) = -1.0;                // ∂g₄/∂z = -1
        Cu(0,0) = -1.0; Cu(1,0) = 1.0; // u1 bounds
        Cu(2,1) = -1.0; Cu(3,1) = 1.0; // u2 bounds
        return Status::SUCCESS;
    }
    Status jacobian_terminal(const VecX&, MatCX& Cx) override {
        Cx.zero(); Cx(4,1) = -1.0;
        return Status::SUCCESS;
    }
};

// ── Main: test both solvers on quadrotor hover ──────────────────────────────

int main() {
    printf("═══════════════════════════════════════════════════════\n");
    printf("  2D Quadrotor Hover — Solver 2 (Mehrotra PC IPM)\n");
    printf("═══════════════════════════════════════════════════════\n\n");

    QuadDyn dyn; QuadCost cost; QuadCons cons;

    using Problem = NMPCProblem<NX, NU, NC, N>;

    VecX x0;
    x0[0] = 0.5;   // y = 0.5m off
    x0[1] = 1.5;   // z = 1.5m (needs to climb to 2m)
    x0[2] = 0.1;   // φ = 0.1 rad (slightly tilted)
    x0[3] = 0.0;
    x0[4] = 0.0;   // vz = 0 (hovering)
    x0[5] = 0.0;   // dphi = 0

    Problem prob;
    prob.dynamics = &dyn; prob.cost = &cost; prob.constraints = &cons; prob.dt = DT;
    prob.x0 = x0;

    // Warm-start: simulate forward with hover thrust for better initial trajectory
    VecU u_hover; u_hover[0] = HOVER_THRUST; u_hover[1] = 0.0;
    prob.stages[0].x = x0;
    prob.stages[0].u = u_hover;
    for (int k = 0; k < N; ++k) {
        prob.stages[k].u = u_hover;
        VecX nx;
        dyn.discrete_step(prob.stages[k].x, prob.stages[k].u, DT, nx);
        prob.stages[k + 1].x = nx;
    }
    prob.stages[N].u.zero();
    prob.stages[N].u[0] = HOVER_THRUST;

    NMPCSolverPaper<NX, NU, NC, N> solver;
    PaperIPMParams pp;
    pp.mu_init = 0.2;
    pp.max_iters = 300;
    // Matched tolerances for fair comparison vs acados.
    pp.mu_min = 1e-4;
    pp.tol_primal = 1.5e-2; pp.tol_compl = 5e-2; pp.tol_ineq = 1e-2; pp.tol_stat = 2e-2;
    pp.kappa_eps = 5.0;  pp.max_same_mu = 5;  // Faster mu reduction
    pp.tau = 0.99;
    pp.soc_max = 4;             // enable SOC (layer 2)
    pp.verbosity = 2;
    solver.configure(pp);

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
    printf("First u* = [%.3f, %.3f]\n", prob.stages[0].u[0], prob.stages[0].u[1]);

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
    printf("First u*:        [%.3f, %.3f]\n", prob.stages[0].u[0], prob.stages[0].u[1]);

    // Dump trajectory
    print_trajectory_table(prob);
    dump_trajectory_json(prob, "benchmarks/data/contactipm_quadrotor.json",
                         "quadrotor", s.inner_iterations, s.cost, DT);

    return 0;
}
