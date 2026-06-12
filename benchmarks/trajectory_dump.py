"""
Trajectory dump utility for acados OCP solvers.

Provides:
  print_acados_table()   – human-readable table to stdout
  dump_acados_trajectory() – structured JSON for comparison scripts

JSON format mirrors the C++ ContactIPM dump for consistent comparison.
"""
import json
import numpy as np


def print_acados_table(ocp_solver, N, NX, NU, cost_fn, constraint_fn):
    """
    Print human-readable trajectory table.

    Args:
        ocp_solver: AcadosOcpSolver instance (after solve)
        N: horizon length
        NX: state dimension
        NU: control dimension
        cost_fn: callable(x, u, k) -> float  (stage cost; ignored at k=N)
        constraint_fn: callable(x, u, k) -> np.ndarray  (g values, g<=0)
    """
    print("\n─── Trajectory Table ───")

    # Header
    hdr = "  k |"
    for i in range(NX):
        hdr += f" x[{i}]  "
    hdr += "|"
    for i in range(NU):
        hdr += f" u[{i}]  "
    hdr += "| cost_k  |"
    nc = len(constraint_fn(ocp_solver.get(0, 'x'), ocp_solver.get(0, 'u'), 0))
    for i in range(nc):
        hdr += f" g[{i}]  "
    print(hdr)
    print("----+" + "-------" * NX + "+" + "-------" * NU + "+---------+" + "-------" * nc)

    total_running = 0.0
    terminal_cost = 0.0

    for k in range(N + 1):
        xk = ocp_solver.get(k, 'x')
        uk = ocp_solver.get(k, 'u') if k < N else np.zeros(NU)

        if k < N:
            ck = cost_fn(xk, uk, k)
            total_running += ck
        else:
            ck = cost_fn(xk, uk, k)  # terminal cost
            terminal_cost = ck

        gk = constraint_fn(xk, uk, k)

        line = f"{k:3d} |"
        for i in range(NX):
            line += f" {xk[i]:6.3f}"
        line += " |"
        if k < N:
            for i in range(NU):
                line += f" {uk[i]:6.3f}"
        else:
            for i in range(NU):
                line += "   -   "
        line += f" | {ck:7.4f} |"
        for i in range(len(gk)):
            if gk[i] < -1e8:
                line += "   -   "
            else:
                line += f" {gk[i]:6.3f}"
        print(line)

    print("───")
    print(f"Running cost sum: {total_running:.6f}")
    print(f"Terminal cost:    {terminal_cost:.6f}")
    print(f"Total:            {total_running + terminal_cost:.6f}")


def dump_acados_trajectory(ocp_solver, N, NX, NU, problem_name,
                            cost_fn, constraint_fn, dt, iterations,
                            filepath=None):
    """
    Dump acados trajectory to JSON.

    Args:
        ocp_solver: AcadosOcpSolver instance (after solve)
        N: horizon length
        NX, NU: dimensions
        problem_name: e.g. 'pendulum'
        cost_fn: callable(x, u, k) -> float
        constraint_fn: callable(x, u, k) -> np.ndarray
        dt: timestep
        iterations: SQP iteration count
        filepath: output path (default: benchmarks/data/acados_{name}.json)
    """
    if filepath is None:
        filepath = f"benchmarks/data/acados_{problem_name}.json"

    nc = len(constraint_fn(ocp_solver.get(0, 'x'), ocp_solver.get(0, 'u'), 0))
    acados_internal_cost = ocp_solver.get_cost()

    data = {
        "solver": "acados_SQP_HPIPM",
        "problem": problem_name,
        "NX": NX, "NU": NU, "NC": nc, "N": N,
        "dt": dt,
        "iterations": int(iterations),
        "cost_total": 0.0,  # filled below with consistent per-node sum
        "cost_total_acados": float(acados_internal_cost),  # acados internal (time-integrated)
        "stages": []
    }

    running_sum = 0.0
    terminal_cost_val = 0.0

    for k in range(N + 1):
        xk = ocp_solver.get(k, 'x').tolist()
        uk = ocp_solver.get(k, 'u').tolist() if k < N else []

        xk_arr = ocp_solver.get(k, 'x')
        uk_arr = ocp_solver.get(k, 'u') if k < N else np.zeros(NU)

        if k < N:
            ck = cost_fn(xk_arr, uk_arr, k)
            running_sum += ck
        else:
            ck = cost_fn(xk_arr, uk_arr, k)
            terminal_cost_val = ck

        gk = constraint_fn(xk_arr, uk_arr, k)

        stage = {
            "k": k,
            "x": xk,
            "u": uk,
            "s": [],       # acados doesn't expose slacks directly
            "lambda": [],   # acados doesn't expose multipliers directly
            "cost": float(ck),
            "g": gk.tolist()
        }
        data["stages"].append(stage)

    data["cost_running"] = float(running_sum)
    data["cost_terminal"] = float(terminal_cost_val)
    # Use per-node sum as cost_total for consistency with ContactIPM
    # (acados internal get_cost() is time-integrated, not per-node)
    data["cost_total"] = float(running_sum + terminal_cost_val)

    with open(filepath, 'w') as f:
        json.dump(data, f, indent=2)
    print(f"Trajectory JSON saved to: {filepath}")
