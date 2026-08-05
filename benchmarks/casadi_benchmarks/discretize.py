"""
Discretization and NLP building for CasADi OCP problems.

Provides:
- rk4_discretize: Build discrete dynamics from continuous ODE using RK4
- build_nlp: Build a direct multiple-shooting NLP from an OCPProblem
- extract_solution: Extract state and control trajectories from NLP solution
"""

import casadi as ca
import numpy as np
from typing import Tuple, Optional
from .problems import OCPProblem


def rk4_discretize(f_expl: ca.Function, dt: float, M: int = 4) -> ca.Function:
    """
    Discretize continuous dynamics using RK4 integration.
    
    Args:
        f_expl: CasADi Function (x, u) -> xdot
        dt: time step
        M: number of RK4 substeps (default 4). For M > 1, the integration
           is applied M times with step size dt/M each, for higher accuracy.
    
    Returns:
        CasADi Function (x, u) -> x_next
    """
    x = ca.MX.sym('x', f_expl.size1_in(0))
    u = ca.MX.sym('u', f_expl.size1_in(1))
    
    # Apply RK4 M times with substep size dt/M for higher accuracy
    h = dt / M
    x_curr = x
    for _ in range(M):
        k1 = f_expl(x_curr, u)
        k2 = f_expl(x_curr + h/2 * k1, u)
        k3 = f_expl(x_curr + h/2 * k2, u)
        k4 = f_expl(x_curr + h * k3, u)
        x_curr = x_curr + (h/6) * (k1 + 2*k2 + 2*k3 + k4)
    
    x_next = x_curr
    
    return ca.Function('F_rk4', [x, u], [x_next], ['x', 'u'], ['x_next'])


def build_nlp(problem: OCPProblem, M_rk4: int = 4) -> dict:
    """
    Build a direct multiple-shooting NLP from an OCPProblem.
    
    Decision variables: [x_0, u_0, x_1, u_1, ..., x_N]
    Constraints:
        - x_0 = x0 (initial condition)
        - x_{k+1} = F(x_k, u_k) for k=0..N-1 (dynamics gaps)
        - g_path(x_k, u_k) in [lg, ug] for k=0..N-1 (path constraints)
    Bounds:
        - x_lb <= x_k <= x_ub for all k
        - u_lb <= u_k <= u_ub for k=0..N-1
    
    Args:
        problem: OCPProblem instance
        M_rk4: number of RK4 substeps per shooting interval
    
    Returns:
        dict with keys:
            'nlp': dict for ca.nlpsol (x, f, g, p)
            'x0_val': initial guess for decision variables
            'lbx': lower bounds on decision variables
            'ubx': upper bounds on decision variables
            'lbg': lower bounds on constraints
            'ubg': upper bounds on constraints
            'nx': number of states
            'nu': number of controls
            'N': number of shooting intervals
            'x_indices': list of (start, end) for each x_k in decision vector
            'u_indices': list of (start, end) for each u_k in decision vector
    """
    nx = problem.nx
    nu = problem.nu
    N = problem.N
    dt = problem.T / N
    
    # Discretize dynamics
    F = rk4_discretize(problem.f_expl, dt, M_rk4)
    
    # Create all symbolic variables first
    X = [ca.MX.sym(f'x_{k}', nx) for k in range(N + 1)]
    U = [ca.MX.sym(f'u_{k}', nu) for k in range(N)]
    
    # Build decision vector
    w = []
    for k in range(N + 1):
        w.append(X[k])
        if k < N:
            w.append(U[k])
    w = ca.vertcat(*w)
    
    # Initial guess and bounds
    w0 = []
    lbw = []
    ubw = []
    
    for k in range(N + 1):
        # State x_k
        # For k=0, use x0 (the actual initial state) to match ContactIPM
        # For k>0, use x_init_guess (or 0) as the initial guess
        if k == 0:
            w0.extend(problem.x0.tolist())
        elif problem.x_init_guess is not None:
            w0.extend(problem.x_init_guess.tolist())
        else:
            w0.extend([0.0] * nx)
        lbw.extend(problem.x_lb.tolist())
        ubw.extend(problem.x_ub.tolist())
        
        # Control u_k
        if k < N:
            w0.extend([0.0] * nu)
            lbw.extend(problem.u_lb.tolist())
            ubw.extend(problem.u_ub.tolist())
    
    # Constraints
    g = []
    lbg = []
    ubg = []
    
    # Initial condition: x_0 = x0
    g.append(X[0] - problem.x0)
    lbg.extend([0.0] * nx)
    ubg.extend([0.0] * nx)
    
    # Dynamics constraints: x_{k+1} - F(x_k, u_k) = 0
    for k in range(N):
        gap = X[k + 1] - F(X[k], U[k])
        g.append(gap)
        lbg.extend([0.0] * nx)
        ubg.extend([0.0] * nx)
    
    # Path constraints: lg <= g_path(x_k, u_k) <= ug
    if problem.g_path is not None:
        for k in range(N):
            g_k = problem.g_path(X[k], U[k])
            g.append(g_k)
            lbg.extend(problem.lg.tolist())
            ubg.extend(problem.ug.tolist())
    
    # Objective
    J = 0
    for k in range(N):
        J += problem.l_stage(X[k], U[k])
    J += problem.l_terminal(X[N])
    
    # Build NLP
    g = ca.vertcat(*g) if g else ca.DM([])
    
    nlp = {
        'x': w,
        'f': J,
        'g': g,
    }
    
    # Compute indices for extracting solutions
    x_indices = []
    u_indices = []
    idx = 0
    for k in range(N + 1):
        x_indices.append((idx, idx + nx))
        idx += nx
        if k < N:
            u_indices.append((idx, idx + nu))
            idx += nu
    
    return {
        'nlp': nlp,
        'x0_val': np.array(w0),
        'lbx': np.array(lbw),
        'ubx': np.array(ubw),
        'lbg': np.array(lbg) if lbg else np.array([]),
        'ubg': np.array(ubg) if ubg else np.array([]),
        'nx': nx,
        'nu': nu,
        'N': N,
        'x_indices': x_indices,
        'u_indices': u_indices,
    }


def extract_solution(sol: ca.DM, nlp_data: dict) -> Tuple[np.ndarray, np.ndarray]:
    """
    Extract state and control trajectories from NLP solution.
    
    Args:
        sol: solution from ca.nlpsol
        nlp_data: dict returned by build_nlp
    
    Returns:
        (X, U) where:
            X is shape (N+1, nx) - state trajectory
            U is shape (N, nu) - control trajectory
    """
    w_opt = sol['x'].full().flatten()
    
    nx = nlp_data['nx']
    nu = nlp_data['nu']
    N = nlp_data['N']
    
    X = np.zeros((N + 1, nx))
    U = np.zeros((N, nu))
    
    for k in range(N + 1):
        i_start, i_end = nlp_data['x_indices'][k]
        X[k] = w_opt[i_start:i_end]
    
    for k in range(N):
        i_start, i_end = nlp_data['u_indices'][k]
        U[k] = w_opt[i_start:i_end]
    
    return X, U


def compute_cost(X: np.ndarray, U: np.ndarray, problem: OCPProblem) -> float:
    """
    Compute total cost from trajectories.
    
    Args:
        X: state trajectory, shape (N+1, nx)
        U: control trajectory, shape (N, nu)
        problem: OCPProblem instance
    
    Returns:
        Total cost (sum of stage costs + terminal cost)
    """
    cost = 0.0
    N = problem.N
    
    for k in range(N):
        cost += float(problem.l_stage(X[k], U[k]))
    cost += float(problem.l_terminal(X[N]))
    
    return cost


if __name__ == "__main__":
    from .problems import CartPendulumMPC, QuadcopterMPC
    
    print("Testing NLP builder...")
    
    for prob_fn in [CartPendulumMPC, QuadcopterMPC]:
        prob = prob_fn()
        nlp_data = build_nlp(prob)
        
        print(f"\n{prob.name}:")
        print(f"  Decision variables: {nlp_data['nlp']['x'].size1()}")
        print(f"  Constraints: {nlp_data['nlp']['g'].size1()}")
        print(f"  x0 shape: {nlp_data['x0_val'].shape}")
        print(f"  lbx/ubx shape: {nlp_data['lbx'].shape}")
        print(f"  lbg/ubg shape: {nlp_data['lbg'].shape}")
    
    print("\nNLP builder validated successfully!")
