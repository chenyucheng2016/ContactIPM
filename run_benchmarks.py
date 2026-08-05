"""Run all 5 CasADi benchmarks."""
import os
import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parent / "benchmarks"))
from casadi_benchmarks import (
    CartPendulumMPC, QuadcopterMPC, QuadcopterTrackingMPC,
    HangingChain2DMPC, HangingChain3DMPC,
    solve_contactipm, solve_ipopt
)

problems = [
    ('cart_pendulum',    CartPendulumMPC,    25, 550, False),
    ('quadcopter',       QuadcopterMPC,      25, 550, True),   # GN mode
    ('quadcopter_track', QuadcopterTrackingMPC, 50, 550, True),  # GN mode
    ('hanging_chain_2d', HangingChain2DMPC,  25, 550, False),
    ('hanging_chain_3d', HangingChain3DMPC,  25, 550, False),
]

print("=" * 90)
print("  CasADi NMPC Benchmark: IPOPT vs ContactIPM")
print("=" * 90)

for name, prob_fn, N, max_it, gn_mode in problems:
    # Set GN mode for quadcopter problems
    if gn_mode:
        os.environ['CONTACTIPM_EXACT_HESSIAN'] = '0'
    elif 'CONTACTIPM_EXACT_HESSIAN' in os.environ:
        del os.environ['CONTACTIPM_EXACT_HESSIAN']
    
    print(f"\n--- {name} (N={N}, {'GN' if gn_mode else 'exact'}) ---")
    prob = prob_fn(N=N)
    
    # IPOPT
    try:
        ipopt_res = solve_ipopt(prob, print_level=0, max_iter=300)
        print(f"  IPOPT:      cost={ipopt_res.cost:12.4f}  iter={ipopt_res.iterations:3d}  "
              f"time={ipopt_res.time_solver*1000:7.1f}ms  status={ipopt_res.status}")
    except Exception as e:
        print(f"  IPOPT:      FAILED - {e}")
        ipopt_res = None
    
    # ContactIPM
    res = solve_contactipm(prob, verbosity=0, max_iters=max_it)
    print(f"  ContactIPM: cost={res.cost:12.4f}  iter={res.iterations:3d}  "
          f"time={res.time_solver*1000:7.1f}ms  status={res.status}")
    
    if ipopt_res:
        diff = abs(res.cost - ipopt_res.cost)
        reldiff = diff / max(abs(ipopt_res.cost), 1e-10) * 100
        match = "OK" if reldiff < 0.01 else "MISMATCH"
        print(f"  Cost diff: {diff:.4f} ({reldiff:.2f}%) [{match}]")

print("\n" + "=" * 90)
