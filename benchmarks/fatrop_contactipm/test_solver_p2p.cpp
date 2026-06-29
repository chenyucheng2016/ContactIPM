#include <cstdio>
#include <cmath>
#include <memory>

#include "fatrop_benchmark_common.hpp"
#include "quadcopter_p2p.c"

constexpr int NX = 7, NU = 5, NC = 7, N = 25;
constexpr double T_INIT = 3.0;
constexpr double DT = T_INIT / N;

int main() {
    printf("Step 1: Setup dynamics\n"); fflush(stdout);
    
    auto dyn = std::make_unique<nmpc::CodegenDynamics<NX, NU>>();
    dyn->set_functions(f_expl, f_jac_x, f_jac_u);
    dyn->free_time_idx = 6;

    printf("Step 2: Setup cost\n"); fflush(stdout);
    auto cost = std::make_unique<nmpc::CodegenCost<NX, NU>>();
    cost->l_stage_fn = l_stage;
    cost->l_stage_grad_x_fn = l_stage_grad_x;
    cost->l_stage_grad_u_fn = l_stage_grad_u;
    cost->l_stage_hess_xx_fn = l_stage_hess_xx;
    cost->l_stage_hess_uu_fn = l_stage_hess_uu;
    cost->l_stage_hess_ux_fn = l_stage_hess_ux;
    cost->l_term_fn = l_term;
    cost->l_term_grad_fn = l_term_grad;
    cost->l_term_hess_fn = l_term_hess;

    printf("Step 3: Setup constraints\n"); fflush(stdout);
    auto cons = std::make_unique<nmpc::CodegenConstraints<NX, NU, NC>>();
    cons->g_path_fn = g_path;
    cons->g_path_jac_x_fn = g_path_jac_x;
    cons->g_path_jac_u_fn = g_path_jac_u;
    cons->g_term_fn = g_term;
    cons->g_term_jac_fn = g_term_jac;

    printf("Step 4: Create problem (heap)\n"); fflush(stdout);
    auto prob = std::make_unique<nmpc::NMPCProblem<NX, NU, NC, N>>();
    prob->dynamics = dyn.get();
    prob->cost = cost.get();
    prob->constraints = cons.get();
    prob->dt = DT;

    printf("Step 5: Initialize state\n"); fflush(stdout);
    nmpc::Vec<NX> x0;
    x0.zero();
    x0[2] = 2.5;
    x0[6] = T_INIT;

    nmpc::Vec<NU> u0;
    u0.zero();
    u0[0] = 9.81;

    prob->x0 = x0;
    // Linear interpolation from start to target position
    double pf[3] = {0.01, 5.0, 2.5};
    for (int k = 0; k <= N; ++k) {
        double alpha = (double)k / N;
        nmpc::Vec<NX> xk;
        xk.zero();
        xk[0] = alpha * pf[0];
        xk[1] = alpha * pf[1];
        xk[2] = x0[2];
        xk[6] = T_INIT;
        prob->stages[k].x = xk;
        prob->stages[k].u = u0;
    }

    printf("Step 6: Create solver\n"); fflush(stdout);
    auto solver = std::make_unique<nmpc::NMPCSolverPaper<NX, NU, NC, N>>();
    nmpc::PaperIPMParams pp = fatrop_bench::default_params();
    pp.verbosity = 3;
    solver->configure(pp);

    printf("Step 7: Calling solve...\n"); fflush(stdout);
    // Test individual model evaluations first
    printf("Step 7a: Testing discrete_step at k=0...\n"); fflush(stdout);
    {
        nmpc::Vec<NX> x_next;
        auto st2 = dyn->discrete_step(prob->stages[0].x, prob->stages[0].u, DT, x_next);
        printf("  discrete_step returned: %d, x_next=[%.4f, %.4f, %.4f, %.4f, %.4f, %.4f, %.4f]\n",
               (int)st2, x_next[0], x_next[1], x_next[2], x_next[3], x_next[4], x_next[5], x_next[6]);
        fflush(stdout);
    }
    printf("Step 7b: Testing linearize at k=0...\n"); fflush(stdout);
    {
        nmpc::Mat<NX,NX> A; nmpc::Mat<NX,NU> B;
        auto st2 = dyn->linearize(prob->stages[0].x, prob->stages[0].u, DT, A, B);
        printf("  linearize returned: %d\n  A(0,0)=%.6f A(0,2)=%.6f\n  B(0,0)=%.6f B(3,0)=%.6f\n",
               (int)st2, A(0,0), A(0,2), B(0,0), B(3,0));
        fflush(stdout);
    }
    printf("Step 7c: Now calling solve...\n"); fflush(stdout);
    // Manually evaluate model at initial guess
    printf("  Testing single RK4 at k=0 with dt=%f...\n", DT); fflush(stdout);
    {
        nmpc::Vec<NX> x_next;
        dyn->discrete_step(prob->stages[0].x, prob->stages[0].u, DT, x_next);
        printf("  x_next[0]=%.4f, x_next[6]=%.4f\n", x_next[0], x_next[6]); fflush(stdout);
    }
    printf("  Testing cost at k=0...\n"); fflush(stdout);
    {
        double c = cost->stage_cost(prob->stages[0].x, prob->stages[0].u, 0);
        printf("  cost=%.6f\n", c); fflush(stdout);
    }
    printf("  Testing constraint at k=0...\n"); fflush(stdout);
    {
        nmpc::Vec<NC> g;
        cons->evaluate(prob->stages[0].x, prob->stages[0].u, 0, g);
        printf("  g[0]=%.4f g[6]=%.4f\n", g[0], g[6]); fflush(stdout);
    }
    printf("  Testing gradient at k=0...\n"); fflush(stdout);
    {
        nmpc::Vec<NX> qx; nmpc::Vec<NU> qu;
        cost->stage_gradient(prob->stages[0].x, prob->stages[0].u, 0, qx, qu);
        printf("  qx[0]=%.3e qu[0]=%.3e\n", qx[0], qu[0]); fflush(stdout);
    }
    printf("  Testing Hessian at k=0...\n"); fflush(stdout);
    {
        nmpc::Mat<NX,NX> Qxx; nmpc::Mat<NU,NU> Quu; nmpc::Mat<NU,NX> Qux;
        cost->stage_hessian(prob->stages[0].x, prob->stages[0].u, 0, Qxx, Quu, Qux);
        printf("  Qxx(0,0)=%.3e Quu(0,0)=%.3e\n", Qxx(0,0), Quu(0,0)); fflush(stdout);
    }
    printf("  All individual evaluations passed!\n"); fflush(stdout);
    printf("  Calling solve now...\n"); fflush(stdout);
    auto st = solver->solve(*prob);
    printf("Step 8: Done! Status: %s\n", nmpc::status_string(st));
    fflush(stdout);

    return 0;
}
