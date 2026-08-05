"""Full benchmark: IPOPT + acados + ContactIPM on all problems."""
import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parent))

import numpy as np
from casadi_benchmarks import (
    CartPendulumMPC, QuadcopterMPC, QuadcopterTrackingMPC,
    HangingChain2DMPC, HangingChain3DMPC,
    solve_ipopt, solve_acados, solve_contactipm,
)

problems = [
    ('cart_pendulum',    CartPendulumMPC,    25),
    ('quadcopter',       QuadcopterMPC,      25),
    ('quadcopter_track', QuadcopterTrackingMPC, 50),
    ('hanging_chain_2d', HangingChain2DMPC,  25),
    ('hanging_chain_3d', HangingChain3DMPC,  25),
]

print("=" * 100)
print("  CasADi NMPC Benchmark: IPOPT vs acados SQP vs ContactIPM")
print("=" * 100)
print()

all_results = []

for name, prob_fn, N in problems:
    print(f"--- {name} (N={N}) ---")
    prob = prob_fn(N=N)
    
    # IPOPT
    try:
        ipopt_res = solve_ipopt(prob, print_level=0, max_iter=300)
        print(f"  IPOPT:      iter={ipopt_res.iterations:3d}  "
              f"cost={ipopt_res.cost:10.4f}  time={ipopt_res.time_solver*1000:8.2f}ms  "
              f"status={ipopt_res.status}")
    except Exception as e:
        ipopt_res = None
        print(f"  IPOPT:      FAILED - {e}")
    
    # acados SQP
    try:
        acados_res = solve_acados(prob, mode='SQP', nlp_solver_max_iter=200)
        print(f"  acados SQP: iter={acados_res.iterations:3d}  "
              f"cost={acados_res.cost:10.4f}  time={acados_res.time_solver*1000:8.2f}ms  "
              f"status={acados_res.status}")
    except Exception as e:
        acados_res = None
        print(f"  acados SQP: FAILED - {e}")
    
    # ContactIPM (same initialization as IPOPT — fair comparison)
    try:
        cipm_res = solve_contactipm(prob, max_iters=200, mu_init=0.1, tol=1e-4,
                                     verbosity=0)
        print(f"  ContactIPM: iter={cipm_res.iterations:3d}  "
              f"cost={cipm_res.cost:10.4f}  time={cipm_res.time_solver*1000:8.2f}ms  "
              f"status={cipm_res.status}")
        
        # Compare costs
        if ipopt_res is not None:
            cost_diff = abs(cipm_res.cost - ipopt_res.cost) / max(abs(ipopt_res.cost), 1e-10)
            print(f"  Cost diff:  {cost_diff*100:.2f}%")
    except Exception as e:
        cipm_res = None
        print(f"  ContactIPM: FAILED - {e}")
        import traceback
        traceback.print_exc()
    
    all_results.append((name, ipopt_res, acados_res, cipm_res))
    print()

# Summary table
print("=" * 100)
print("SUMMARY TABLE")
print("=" * 100)
print(f"{'Problem':<20} {'Solver':<12} {'Iter':>5} {'Cost':>12} {'Time(ms)':>10} {'Status':<20}")
print("-" * 100)

for name, ipopt_res, acados_res, cipm_res in all_results:
    if ipopt_res is not None:
        print(f"{name:<20} {'IPOPT':<12} {ipopt_res.iterations:>5} "
              f"{ipopt_res.cost:>12.4f} {ipopt_res.time_solver*1000:>10.2f} "
              f"{ipopt_res.status:<20}")
    if acados_res is not None:
        print(f"{name:<20} {'acados SQP':<12} {acados_res.iterations:>5} "
              f"{acados_res.cost:>12.4f} {acados_res.time_solver*1000:>10.2f} "
              f"{acados_res.status:<20}")
    if cipm_res is not None:
        print(f"{name:<20} {'ContactIPM':<12} {cipm_res.iterations:>5} "
              f"{cipm_res.cost:>12.4f} {cipm_res.time_solver*1000:>10.2f} "
              f"{cipm_res.status:<20}")
    print("-" * 100)

print("\nDone!")
