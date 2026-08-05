"""Test ContactIPM solver - use IPOPT solution as warm start."""
import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parent))

import numpy as np
from casadi_benchmarks import CartPendulumMPC, solve_contactipm, solve_ipopt

print("=== Testing ContactIPM on cart_pendulum (warm-started) ===")
prob = CartPendulumMPC(N=25)

# Get IPOPT reference
print("Getting IPOPT reference...")
ipopt_result = solve_ipopt(prob, print_level=0, max_iter=300)
print(f"  IPOPT: iter={ipopt_result.iterations}, cost={ipopt_result.cost:.4f}")
print(f"  x_final = {ipopt_result.X[-1]}")

# Test ContactIPM with IPOPT solution as initial guess
print("\nRunning ContactIPM (warm-started from IPOPT, verbosity=0)...")
try:
    # Override the problem's initial guess with IPOPT solution
    prob_warm = CartPendulumMPC(N=25)
    prob_warm.x_init_guess = ipopt_result.X[0]  # start from IPOPT initial state
    
    # Actually, let's set the full trajectory as initial guess
    # by modifying the problem
    import casadi as ca
    from casadi_benchmarks.solvers import _build_contactipm_callbacks, compute_cost
    import contactipm
    import time
    
    nx = prob.nx
    nu = prob.nu
    N = prob.N
    dt = prob.T / N
    
    cb = _build_contactipm_callbacks(prob)
    solver = contactipm.Solver(nx, nu, 1, N, dt)
    solver.set_callbacks(
        cb['f_disc'], cb['f_lin'],
        cb['stage_cost'], cb['term_cost'],
        cb['stage_grad'], cb['stage_hess'],
        cb['term_grad'], cb['term_hess'],
        cb['constr_eval'], cb['constr_eval_term'],
        cb['constr_jac'], cb['constr_jac_term'],
        cb['has_constraints'],
    )
    solver.set_bounds(prob.x_lb, prob.x_ub, prob.u_lb, prob.u_ub)
    solver.set_initial_state(prob.x0)
    solver.set_params(mu_init=0.1, max_iters=200, tol_primal=1e-4, tol_compl=1e-4,
                      tol_stat=0.5, verbosity=0)
    
    # Use IPOPT solution as warm start
    X_guess = np.ascontiguousarray(ipopt_result.X)
    U_guess = np.ascontiguousarray(ipopt_result.U)
    
    t0 = time.time()
    X, U, stats = solver.solve(X_guess, U_guess)
    t1 = time.time()
    
    cost = compute_cost(np.asarray(X), np.asarray(U), prob)
    print(f"  ContactIPM: status={stats['status']}")
    print(f"  iterations={stats['iterations']}")
    print(f"  cost={cost:.4f}")
    print(f"  time={stats['time_solver']*1000:.2f}ms")
    print(f"  x_final = {np.asarray(X)[-1]}")
    print(f"  u[0] = {np.asarray(U)[0]}")
except Exception as e:
    print(f"  FAILED: {e}")
    import traceback
    traceback.print_exc()

# Also test with zero guess but more iterations and different params
print("\nRunning ContactIPM (cold-start, mu_init=0.1, verbosity=0)...")
try:
    prob2 = CartPendulumMPC(N=25)
    result = solve_contactipm(prob2, max_iters=300, mu_init=0.1, tol=1e-4, verbosity=0)
    print(f"  status={result.status}")
    print(f"  iterations={result.iterations}")
    print(f"  cost={result.cost:.4f}")
    print(f"  time={result.time_solver*1000:.2f}ms")
    print(f"  x_final = {result.X[-1]}")
except Exception as e:
    print(f"  FAILED: {e}")
    import traceback
    traceback.print_exc()

print("\nDone!")
