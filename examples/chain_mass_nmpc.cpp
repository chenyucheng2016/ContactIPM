/**
 * @file    chain_mass_nmpc.cpp
 * @brief   Chain of masses NMPC benchmark — scalable state dimension.
 *
 * This is the standard MPC benchmark used by HPIPM, acados, and FORCES.
 *
 * Model: N masses connected by springs, controlled by forces.
 *   State:  [x_1, ..., x_N, v_1, ..., v_N]   → 2·N states
 *   Control: [F_1, ..., F_M]                   → M forces on first M masses
 *   Dynamics:  ẍ_i = (F_i + k·(x_{i+1} - 2x_i + x_{i-1}) - d·ẋ_i) / m
 *              (with x_0 = x_{N+1} = 0 as boundary condition)
 *
 * Cost: drive the masses to rest at origin with minimal control effort.
 * Constraints: |F_i| ≤ F_max,  |x_i| ≤ X_max
 *
 * Choose compile-time N_MASSES and N_ACTUATED to change problem size.
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

// ── Configurable dimensions ─────────────────────────────────────────────────
constexpr int N_MASSES    = 5;     // number of masses in chain
constexpr int N_ACTUATED  = 3;     // forces applied to first N masses
constexpr int NX = 2 * N_MASSES;   // [x_1..x_N, v_1..v_N]
constexpr int NU = N_ACTUATED;
constexpr int NC = 2 * NU + 2 * N_MASSES;  // force bounds + position bounds
constexpr int N  = 20;             // MPC horizon

constexpr double MASS     = 1.0;
constexpr double SPRING_K = 10.0;
constexpr double DAMPING  = 0.5;
constexpr double F_MAX    = 5.0;
constexpr double X_MAX    = 2.0;
constexpr double DT       = 0.05;

using VecX = Vec<NX>;
using VecU = Vec<NU>;
using VecC = Vec<NC>;
using MatXX = Mat<NX, NX>;
using MatXU = Mat<NX, NU>;
using MatCX = Mat<NC, NX>;
using MatCU = Mat<NC, NU>;

// ── Chain Dynamics ──────────────────────────────────────────────────────────

struct ChainDyn : DynamicsModel<NX, NU> {
    Status discrete_step(const VecX& x, const VecU& u, double dt,
                         VecX& nx) override {
        // Semi-implicit Euler on the spring-mass chain
        for (int i = 0; i < N_MASSES; ++i) {
            double x_i   = x[i];
            double x_im1 = (i > 0) ? x[i-1] : 0.0;
            double x_ip1 = (i < N_MASSES-1) ? x[i+1] : 0.0;
            double v_i   = x[N_MASSES + i];
            double F_i   = (i < N_ACTUATED) ? u[i] : 0.0;

            double spring_force = SPRING_K * (x_ip1 - 2.0*x_i + x_im1);
            double damp_force   = DAMPING * v_i;
            double a_i = (F_i + spring_force - damp_force) / MASS;

            // Update velocity first, then position (semi-implicit)
            double v_new = v_i + dt * a_i;
            nx[N_MASSES + i] = v_new;
            nx[i] = x_i + dt * v_new;
        }
        return Status::SUCCESS;
    }

    Status linearize(const VecX& x, const VecU& u, double dt,
                     MatXX& A, MatXU& B) override {
        const double eps = 1e-6;
        VecX xp, xm, fp, fm;
        for (int j = 0; j < NX; ++j) {
            xp = x; xp[j] += eps; xm = x; xm[j] -= eps;
            discrete_step(xp, u, dt, fp); discrete_step(xm, u, dt, fm);
            for (int i = 0; i < NX; ++i) A(i, j) = (fp[i] - fm[i]) / (2*eps);
        }
        VecU up, um;
        for (int j = 0; j < NU; ++j) {
            up = u; up[j] += eps; um = u; um[j] -= eps;
            discrete_step(x, up, dt, fp); discrete_step(x, um, dt, fm);
            for (int i = 0; i < NX; ++i) B(i, j) = (fp[i] - fm[i]) / (2*eps);
        }
        return Status::SUCCESS;
    }
};

// ── Cost: drive to origin ───────────────────────────────────────────────────

struct ChainCost : CostModel<NX, NU> {
    static constexpr double w_pos = 10.0, w_vel = 1.0, w_ctrl = 0.01;

    double stage_cost(const VecX& x, const VecU& u, int) override {
        double c = 0.0;
        for (int i = 0; i < N_MASSES; ++i) {
            c += w_pos * x[i]*x[i];
            c += w_vel * x[N_MASSES + i]*x[N_MASSES + i];
        }
        for (int i = 0; i < NU; ++i) c += w_ctrl * u[i]*u[i];
        return 0.5 * c;
    }
    double terminal_cost(const VecX& x) override {
        double c = 0.0;
        for (int i = 0; i < N_MASSES; ++i) {
            c += 20.0*w_pos * x[i]*x[i];
            c += 20.0*w_vel * x[N_MASSES + i]*x[N_MASSES + i];
        }
        return 0.5 * c;
    }
    Status stage_gradient(const VecX& x, const VecU& u, int,
                          VecX& qx, VecU& qu) override {
        for (int i = 0; i < N_MASSES; ++i) {
            qx[i] = w_pos * x[i];
            qx[N_MASSES + i] = w_vel * x[N_MASSES + i];
        }
        for (int i = 0; i < NU; ++i) qu[i] = w_ctrl * u[i];
        return Status::SUCCESS;
    }
    Status stage_hessian(const VecX&, const VecU&, int,
                         MatXX& Qxx, Mat<NU,NU>& Quu, Mat<NU,NX>& Qux) override {
        Qxx.zero();
        for (int i = 0; i < N_MASSES; ++i) {
            Qxx(i,i) = w_pos;
            Qxx(N_MASSES+i, N_MASSES+i) = w_vel;
        }
        Quu.zero(); for (int i = 0; i < NU; ++i) Quu(i,i) = w_ctrl;
        Qux.zero();
        return Status::SUCCESS;
    }
    Status terminal_gradient(const VecX& x, VecX& qx) override {
        for (int i = 0; i < N_MASSES; ++i) {
            qx[i] = 20*w_pos * x[i];
            qx[N_MASSES + i] = 20*w_vel * x[N_MASSES + i];
        }
        return Status::SUCCESS;
    }
    Status terminal_hessian(const VecX&, MatXX& Qxx) override {
        Qxx.zero();
        for (int i = 0; i < N_MASSES; ++i) {
            Qxx(i,i) = 20*w_pos;
            Qxx(N_MASSES+i, N_MASSES+i) = 20*w_vel;
        }
        return Status::SUCCESS;
    }
};

// ── Constraints: force bounds + position bounds ─────────────────────────────

struct ChainCons : ConstraintModel<NX, NU, NC> {
    Status evaluate(const VecX& x, const VecU& u, int, VecC& g) override {
        int idx = 0;
        // Force bounds: -F_max ≤ F_i ≤ F_max
        for (int i = 0; i < NU; ++i) {
            g[idx++] = -F_MAX - u[i];   // -F - u ≤ 0 → u ≥ -F
            g[idx++] = u[i] - F_MAX;    // u - F ≤ 0  → u ≤ F
        }
        // Position bounds: -X_max ≤ x_i ≤ X_max
        for (int i = 0; i < N_MASSES; ++i) {
            g[idx++] = -X_MAX - x[i];
            g[idx++] = x[i] - X_MAX;
        }
        return Status::SUCCESS;
    }
    Status evaluate_terminal(const VecX& x, VecC& g) override {
        int idx = 0;
        for (int i = 0; i < NU; ++i) { g[idx++] = -1e10; g[idx++] = -1e10; }
        for (int i = 0; i < N_MASSES; ++i) {
            g[idx++] = -X_MAX - x[i];
            g[idx++] = x[i] - X_MAX;
        }
        return Status::SUCCESS;
    }
    Status jacobian(const VecX&, const VecU&, int, MatCX& Cx, MatCU& Cu) override {
        Cx.zero(); Cu.zero();
        int idx = 0;
        for (int i = 0; i < NU; ++i) {
            Cu(idx, i)   = -1.0; idx++;
            Cu(idx, i)   =  1.0; idx++;
        }
        for (int i = 0; i < N_MASSES; ++i) {
            Cx(idx, i)   = -1.0; idx++;
            Cx(idx, i)   =  1.0; idx++;
        }
        return Status::SUCCESS;
    }
    Status jacobian_terminal(const VecX&, MatCX& Cx) override {
        Cx.zero();
        int idx = 0;
        for (int i = 0; i < NU; ++i) { idx += 2; }
        for (int i = 0; i < N_MASSES; ++i) {
            Cx(idx, i) = -1.0; idx++;
            Cx(idx, i) =  1.0; idx++;
        }
        return Status::SUCCESS;
    }
};

// ── Main ────────────────────────────────────────────────────────────────────

int main() {
    printf("═══════════════════════════════════════════════════════\n");
    printf("  Chain-of-Masses NMPC — nx=%d, nu=%d, nc=%d, N=%d\n",
           NX, NU, NC, N);
    printf("═══════════════════════════════════════════════════════\n\n");

    ChainDyn dyn; ChainCost cost; ChainCons cons;

    using Problem = NMPCProblem<NX, NU, NC, N>;
    Problem prob;
    prob.dynamics = &dyn; prob.cost = &cost; prob.constraints = &cons; prob.dt = DT;

    // Initial state: masses displaced in a sine pattern
    VecX x0;
    for (int i = 0; i < N_MASSES; ++i) {
        x0[i] = std::sin(M_PI * (i+1) / (N_MASSES+1)) * 1.5;  // sine wave, ±1.5m
        x0[N_MASSES + i] = 0.0;                                 // at rest
    }

    prob.x0 = x0;
    for (int k = 0; k <= N; ++k) {
        prob.stages[k].x = x0;
        prob.stages[k].u.zero();
    }

    NMPCSolverPaper<NX, NU, NC, N> solver;
    PaperIPMParams pp;
    pp.mu_init  = 10.0;
    pp.max_iters = 200;  pp.tol_primal=1e-3; pp.tol_compl=1e-2; pp.tol_ineq=1e-2;
    pp.kappa_eps = 5.0;  pp.max_same_mu = 15;
    pp.verbosity = 2;
    solver.configure(pp);

    auto t_start = std::chrono::high_resolution_clock::now();
    Status st = solver.solve(prob);
    auto t_end = std::chrono::high_resolution_clock::now();
    double solve_ms = std::chrono::duration<double, std::milli>(t_end - t_start).count();

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
    printf("Penalty weight: %.1f\n", s.penalty_weight);
    printf("Regularization: %.3e\n", s.regularization);
    printf("Cost: %.4f\n", s.cost);
    printf("Solve time: %.3f ms\n", solve_ms);
    printf("First u* = [");
    for (int i = 0; i < NU; ++i)
        printf("%.3f%s", prob.stages[0].u[i], (i<NU-1)?", ":"");
    printf("]\n");

    // Dump trajectory
    print_trajectory_table(prob);
    dump_trajectory_json(prob, "benchmarks/data/contactipm_chain_mass.json",
                         "chain_mass", s.inner_iterations, s.cost, DT);

    return 0;
}
