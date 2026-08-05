"""
Solver interfaces for CasADi benchmark problems.

Provides unified wrappers for IPOPT (via CasADi), acados, and ContactIPM.
"""

import casadi as ca
import numpy as np
import time
import os
import sys
import ctypes
from dataclasses import dataclass
from typing import Optional, Tuple

from .problems import OCPProblem
from .discretize import build_nlp, extract_solution, compute_cost, rk4_discretize


@dataclass
class SolveResult:
    """Result from a solver run."""
    solver: str
    problem_name: str
    status: str
    iterations: int
    time_solver: float  # seconds
    time_total: float   # seconds (including setup)
    cost: float
    X: np.ndarray       # state trajectory, shape (N+1, nx)
    U: np.ndarray       # control trajectory, shape (N, nu)


def solve_ipopt(problem: OCPProblem, 
                max_iter: int = 200,
                tol: float = 1e-8,
                mu_init: float = 1e2,
                print_level: int = 0) -> SolveResult:
    """
    Solve OCP using IPOPT via CasADi's nlpsol.
    
    Args:
        problem: OCPProblem instance
        max_iter: maximum IPOPT iterations
        tol: IPOPT convergence tolerance
        mu_init: initial barrier parameter
        print_level: IPOPT verbosity (0=silent, 5=verbose)
    
    Returns:
        SolveResult with solution and statistics
    """
    t_start = time.time()
    
    # Build NLP
    nlp_data = build_nlp(problem)
    nlp = nlp_data['nlp']
    
    # Create IPOPT solver
    opts = {
        'ipopt.max_iter': max_iter,
        'ipopt.tol': tol,
        'ipopt.mu_init': mu_init,
        'ipopt.print_level': print_level,
        'ipopt.linear_solver': 'mumps',
        'expand': True,
    }
    
    solver = ca.nlpsol('ipopt', 'ipopt', nlp, opts)
    
    # Solve
    t_solve_start = time.time()
    sol = solver(
        x0=nlp_data['x0_val'],
        lbx=nlp_data['lbx'],
        ubx=nlp_data['ubx'],
        lbg=nlp_data['lbg'],
        ubg=nlp_data['ubg'],
    )
    t_solve_end = time.time()
    
    # Extract solution
    X, U = extract_solution(sol, nlp_data)
    cost = compute_cost(X, U, problem)
    
    # Get statistics
    stats = solver.stats()
    iterations = stats.get('iter_count', 0)
    status = stats.get('return_status', 'Unknown')
    
    return SolveResult(
        solver='ipopt',
        problem_name=problem.name,
        status=status,
        iterations=iterations,
        time_solver=t_solve_end - t_solve_start,
        time_total=time.time() - t_start,
        cost=cost,
        X=X,
        U=U,
    )


def solve_acados(problem: OCPProblem,
                 mode: str = 'SQP',
                 qp_solver_cond: int = 4,
                 nlp_solver_max_iter: int = 200) -> SolveResult:
    """
    Solve OCP using acados.
    
    Args:
        problem: OCPProblem instance
        mode: 'SQP' or 'SQP_RTI'
        qp_solver_cond: QP solver condensing parameter
        nlp_solver_max_iter: maximum SQP iterations
    
    Returns:
        SolveResult with solution and statistics
    """
    try:
        from acados_template import AcadosOcp, AcadosOcpSolver, AcadosModel
    except ImportError:
        raise ImportError("acados_template not found. Install acados Python interface.")
    
    t_start = time.time()
    
    nx = problem.nx
    nu = problem.nu
    N = problem.N
    T = problem.T
    dt = T / N
    
    # Discretize dynamics
    F = rk4_discretize(problem.f_expl, dt, M=4)
    
    # Create acados OCP
    ocp = AcadosOcp()
    ocp.model = AcadosModel()
    ocp.model.name = problem.name
    
    # Define symbolic variables for acados
    x = ca.MX.sym('x', nx)
    u = ca.MX.sym('u', nu)
    
    # For acados with DISCRETE integrator, we need f_disc
    ocp.model.disc_dyn_expr = F(x, u)
    ocp.model.x = x
    ocp.model.u = u
    ocp.model.p = []
    
    # Cost
    ocp.cost.cost_type = 'EXTERNAL'
    ocp.cost.cost_type_e = 'EXTERNAL'
    ocp.model.cost_expr_ext_cost = problem.l_stage(x, u)
    ocp.model.cost_expr_ext_cost_e = problem.l_terminal(x)
    
    # Path constraints
    if problem.g_path is not None:
        ocp.model.con_h_expr = problem.g_path(x, u)
        ocp.constraints.lh = problem.lg
        ocp.constraints.uh = problem.ug
    
    # State and control bounds
    ocp.constraints.lbx = problem.x_lb
    ocp.constraints.ubx = problem.x_ub
    ocp.constraints.idxbx = np.arange(nx)
    
    ocp.constraints.lbu = problem.u_lb
    ocp.constraints.ubu = problem.u_ub
    ocp.constraints.idxbu = np.arange(nu)
    
    # Initial state
    ocp.constraints.x0 = problem.x0
    ocp.constraints.idxbx_0 = np.arange(nx)
    
    # Solver options
    ocp.solver_options.tf = T
    ocp.solver_options.N_horizon = N
    ocp.solver_options.integrator_type = 'DISCRETE'
    ocp.solver_options.nlp_solver_type = mode
    ocp.solver_options.qp_solver = 'PARTIAL_CONDENSING_HPIPM'
    ocp.solver_options.qp_solver_cond_N = N // qp_solver_cond
    ocp.solver_options.nlp_solver_max_iter = nlp_solver_max_iter
    ocp.solver_options.hessian_approx = 'EXACT' if mode == 'SQP' else 'GAUSS_NEWTON'
    ocp.solver_options.qp_solver_ric_alg = 1
    ocp.solver_options.hpipm_mode = 'BALANCE'
    
    # Create solver — fix Windows DLL name mismatch (Makefile generates lib*, Python expects no lib)
    if sys.platform == 'win32':
        try:
            from acados_template.utils import get_acados_path
            import shutil as _shutil
            import acados_template.utils as _acados_utils
            acados_path = get_acados_path()
            code_gen_dir = os.path.join(os.getcwd(), 'c_generated_code')
            os.makedirs(code_gen_dir, exist_ok=True)
            # Copy acados dependency DLLs next to the generated solver DLL
            for subdir in ['bin', 'lib']:
                src_dir = os.path.join(acados_path, subdir)
                for dll_name in ['blasfeo.dll', 'hpipm.dll', 'acados.dll']:
                    src = os.path.join(src_dir, dll_name)
                    dst = os.path.join(code_gen_dir, dll_name)
                    if os.path.isfile(src) and not os.path.isfile(dst):
                        _shutil.copy2(src, dst)
            # Copy MinGW runtime DLLs too
            cc_path = _shutil.which('cc')
            if cc_path:
                mingw_bin = os.path.dirname(cc_path)
                for dll_name in ['libgcc_s_seh-1.dll', 'libwinpthread-1.dll',
                                 'libstdc++-6.dll', 'libgomp-1.dll']:
                    src = os.path.join(mingw_bin, dll_name)
                    dst = os.path.join(code_gen_dir, dll_name)
                    if os.path.isfile(src) and not os.path.isfile(dst):
                        _shutil.copy2(src, dst)
            # Fix: Makefile generates "lib*.dll" but acados expects "*.dll" on Windows
            # Also fix winmode=8 which prevents finding dependency DLLs
            _orig_get_shared_lib = _acados_utils.get_shared_lib
            _added_dirs = set()
            def _patched_get_shared_lib(name, winmode=None):
                import ctypes as _ct
                # Fix lib prefix mismatch for OCP solver DLL
                if 'acados_ocp_solver_' in name and not os.path.isfile(name):
                    lib_name = name.replace('acados_ocp_solver_', 'libacados_ocp_solver_')
                    if os.path.isfile(lib_name):
                        name = lib_name
                # Add the DLL's directory to search path before loading
                dll_dir = os.path.dirname(name)
                if dll_dir and dll_dir not in _added_dirs:
                    os.add_dll_directory(dll_dir)
                    _added_dirs.add(dll_dir)
                # Also add acados bin/lib and MinGW bin
                for d in [os.path.join(acados_path, 'bin'),
                          os.path.join(acados_path, 'lib')]:
                    if d not in _added_dirs and os.path.isdir(d):
                        os.add_dll_directory(d)
                        _added_dirs.add(d)
                cc_path = _shutil.which('cc')
                if cc_path:
                    mingw_dir = os.path.dirname(cc_path)
                    if mingw_dir not in _added_dirs:
                        os.add_dll_directory(mingw_dir)
                        _added_dirs.add(mingw_dir)
                return _ct.WinDLL(name)
            _acados_utils.get_shared_lib = _patched_get_shared_lib
            import acados_template.acados_ocp_solver as _ocp_mod
            _ocp_mod.get_shared_lib = _patched_get_shared_lib
        except Exception:
            pass
    
    acados_solver = AcadosOcpSolver(ocp, json_file=f'{problem.name}.json')
    
    # Initialize with constant initialization (matching IPOPT's naive initialization)
    # x[k] = x_init_guess (or x0) for all k, u[k] = 0 for all k
    x_init = problem.x_init_guess if problem.x_init_guess is not None else problem.x0
    for k in range(N + 1):
        acados_solver.set(k, 'x', x_init)
    for k in range(N):
        acados_solver.set(k, 'u', np.zeros(nu))
    
    # Solve
    t_solve_start = time.time()
    status = acados_solver.solve()
    t_solve_end = time.time()
    
    # Extract solution
    X = np.zeros((N + 1, nx))
    U = np.zeros((N, nu))
    for k in range(N + 1):
        X[k] = acados_solver.get(k, 'x')
    for k in range(N):
        U[k] = acados_solver.get(k, 'u')
    
    cost = compute_cost(X, U, problem)
    
    # Get statistics
    iterations = acados_solver.get_stats('sqp_iter')
    time_solver = acados_solver.get_stats('time_tot') - acados_solver.get_stats('time_lin')
    
    return SolveResult(
        solver=f'acados_{mode}',
        problem_name=problem.name,
        status='Solved' if status == 0 else f'Failed({status})',
        iterations=iterations,
        time_solver=time_solver,
        time_total=time.time() - t_start,
        cost=cost,
        X=X,
        U=U,
    )


def _build_contactipm_callbacks(problem: OCPProblem, return_raw: bool = False):
    """
    Build all CasADi Function callbacks needed by ContactIPM.
    
    Creates: discrete dynamics, linearization, cost gradient/Hessian,
    and constraint evaluation/Jacobian functions.
    
    Returns:
        dict with all CasADi Functions and metadata
    """
    nx = problem.nx
    nu = problem.nu
    dt = problem.T / problem.N
    
    # Helper: wrap CasADi Function to return numpy arrays (not DM)
    def _wrap(fn, shape_out=None):
        """Wrap CasADi Function so outputs are numpy arrays with correct shapes.
        
        shape_out: None (auto), 'vec' (flatten to 1D), 'scalar' (float), or list of shapes
        """
        def wrapped(*args):
            result = fn(*args)
            if shape_out == 'scalar':
                return float(np.asarray(result).flatten()[0])
            elif shape_out == 'vec':
                return np.asarray(result).flatten()
            elif isinstance(shape_out, tuple):
                # Single output with specific shape
                return np.asarray(result).reshape(shape_out)
            elif isinstance(shape_out, list):
                # Multiple outputs with specific shapes
                if not isinstance(result, (list, tuple)):
                    result = [result]
                return tuple(np.asarray(r).reshape(s) for r, s in zip(result, shape_out))
            elif isinstance(result, (list, tuple)):
                return tuple(np.asarray(r) for r in result)
            else:
                return np.asarray(result)
        return wrapped
    
    # Symbolic variables
    xs = ca.MX.sym('x', nx)
    us = ca.MX.sym('u', nu)
    
    # ── Discrete dynamics via RK4 ──
    # Use rk4_discretize to ensure consistency with IPOPT
    f_disc_fn = rk4_discretize(problem.f_expl, dt, M=4)
    x_next_expr = f_disc_fn(xs, us)
    
    f_disc = f_disc_fn  # already a CasADi Function
    
    # ── Dynamics linearization: A = df_disc/dx, B = df_disc/du ──
    A_expr = ca.jacobian(x_next_expr, xs)
    B_expr = ca.jacobian(x_next_expr, us)
    f_lin = ca.Function('f_lin', [xs, us], [A_expr, B_expr])

    # ── Adjoint (contracted) Hessian of dynamics: Σ_m p_m·∇²_zz f_m ──
    # z = (x, u).  Sliced into (Hxx, Hux, Huu) blocks for the solver.
    z_sym = ca.vertcat(xs, us)
    p_adj = ca.MX.sym('p_adj', nx)
    Hf = ca.hessian(ca.dot(p_adj, x_next_expr), z_sym)[0]
    dyn_adj_hess_fn = ca.Function('dyn_adj_hess', [xs, us, p_adj],
                                  [Hf[0:nx, 0:nx],
                                   Hf[nx:nx + nu, 0:nx],
                                   Hf[nx:nx + nu, nx:nx + nu]])
    
    # ── Stage cost gradient and Hessian ──
    l_stage_expr = problem.l_stage(xs, us)
    qx_expr = ca.gradient(l_stage_expr, xs)
    qu_expr = ca.gradient(l_stage_expr, us)
    Qxx_expr = ca.jacobian(qx_expr, xs)
    Quu_expr = ca.jacobian(qu_expr, us)
    Qux_expr = ca.jacobian(qu_expr, xs)
    
    stage_cost_fn = problem.l_stage  # reuse existing
    stage_grad_fn = ca.Function('stage_grad', [xs, us], [qx_expr, qu_expr])
    stage_hess_fn = ca.Function('stage_hess', [xs, us], [Qxx_expr, Quu_expr, Qux_expr])
    
    # ── Terminal cost gradient and Hessian ──
    l_term_expr = problem.l_terminal(xs)
    qx_term_expr = ca.gradient(l_term_expr, xs)
    Qxx_term_expr = ca.jacobian(qx_term_expr, xs)
    
    term_cost_fn = problem.l_terminal  # reuse existing
    term_grad_fn = ca.Function('term_grad', [xs], [qx_term_expr])
    term_hess_fn = ca.Function('term_hess', [xs], [Qxx_term_expr])
    
    # ── Constraint callbacks ──
    has_constraints = problem.g_path is not None
    if has_constraints:
        g_expr = problem.g_path(xs, us)
        Cx_expr = ca.jacobian(g_expr, xs)
        Cu_expr = ca.jacobian(g_expr, us)
        
        constr_eval = problem.g_path  # reuse existing
        # Terminal constraint: zero (no terminal path constraints)
        constr_eval_term = ca.Function('constr_term', [xs], [ca.DM.zeros(problem.ng)])
        constr_jac = ca.Function('constr_jac', [xs, us], [Cx_expr, Cu_expr])
        constr_jac_term = ca.Function('constr_jac_term', [xs],
                                       [ca.DM.zeros(problem.ng, nx)])

        # ── Adjoint (contracted) Hessian of constraints: Σ_j λ_j·∇²_zz g_j ──
        lam_adj = ca.MX.sym('lam_adj', problem.ng)
        Hg = ca.hessian(ca.dot(lam_adj, g_expr), z_sym)[0]
        con_adj_hess_fn = ca.Function('con_adj_hess', [xs, us, lam_adj],
                                      [Hg[0:nx, 0:nx],
                                       Hg[nx:nx + nu, 0:nx],
                                       Hg[nx:nx + nu, nx:nx + nu]])
        # Terminal path constraint is zero in this suite → zero Hessian.
        con_adj_hess_term_fn = ca.Function('con_adj_hess_term', [xs, lam_adj],
                                           [ca.DM.zeros(nx, nx)])
    else:
        # Dummy constraint functions (never called when has_constraints=False)
        dummy_g = ca.DM.zeros(1)
        constr_eval = ca.Function('dummy_eval', [xs, us], [dummy_g])
        constr_eval_term = ca.Function('dummy_eval_term', [xs], [dummy_g])
        constr_jac = ca.Function('dummy_jac', [xs, us],
                                  [ca.DM.zeros(1, nx), ca.DM.zeros(1, nu)])
        constr_jac_term = ca.Function('dummy_jac_term', [xs], [ca.DM.zeros(1, nx)])
    
    raw_functions = {
        'f_disc': f_disc,
        'f_lin': f_lin,
        'stage_cost': problem.l_stage,
        'term_cost': problem.l_terminal,
        'stage_grad': stage_grad_fn,
        'stage_hess': stage_hess_fn,
        'term_grad': term_grad_fn,
        'term_hess': term_hess_fn,
        'constr_eval': constr_eval,
        'constr_eval_term': constr_eval_term,
        'constr_jac': constr_jac,
        'constr_jac_term': constr_jac_term,
        'dyn_adj_hess': dyn_adj_hess_fn,
        'con_adj_hess': con_adj_hess_fn if has_constraints else None,
        'con_adj_hess_term': con_adj_hess_term_fn if has_constraints else None,
        'has_constraints': has_constraints,
    }
    if return_raw:
        return raw_functions

    return {
        'f_disc': _wrap(f_disc, 'vec'),
        'f_lin': _wrap(f_lin, [(nx, nx), (nx, nu)]),
        'stage_cost': _wrap(problem.l_stage, 'scalar'),
        'term_cost': _wrap(problem.l_terminal, 'scalar'),
        'stage_grad': _wrap(stage_grad_fn, [(nx,), (nu,)]),
        'stage_hess': _wrap(stage_hess_fn, [(nx, nx), (nu, nu), (nu, nx)]),
        'term_grad': _wrap(term_grad_fn, 'vec'),
        'term_hess': _wrap(term_hess_fn, (nx, nx)),
        'constr_eval': _wrap(constr_eval, 'vec'),
        'constr_eval_term': _wrap(constr_eval_term, 'vec'),
        'constr_jac': _wrap(constr_jac, [(problem.ng, nx), (problem.ng, nu)]),
        'constr_jac_term': _wrap(constr_jac_term, (problem.ng, nx)),
        'dyn_adj_hess': _wrap(dyn_adj_hess_fn, [(nx, nx), (nu, nx), (nu, nu)]),
        'con_adj_hess': (_wrap(con_adj_hess_fn, [(nx, nx), (nu, nx), (nu, nu)])
                         if has_constraints else None),
        'con_adj_hess_term': (_wrap(con_adj_hess_term_fn, (nx, nx))
                              if has_constraints else None),
        'has_constraints': has_constraints,
    }


def solve_contactipm(problem: OCPProblem,
                     max_iters: int = 100,
                     mu_init: float = 0.1,
                     mu_min: float = 1e-6,
                     tol: float = 1e-4,
                     verbosity: int = 0,
                     X_init: Optional[np.ndarray] = None,
                     U_init: Optional[np.ndarray] = None) -> SolveResult:
    """
    Solve OCP using ContactIPM (via pybind11 bindings).
    
    Args:
        problem: OCPProblem instance
        max_iters: maximum IPM iterations
        mu_init: initial barrier parameter
        tol: convergence tolerance
        verbosity: 0=silent, 1=summary, 2=verbose
        X_init: initial state trajectory (N+1, nx), overrides default
        U_init: initial control trajectory (N, nu), overrides default
    
    Returns:
        SolveResult with solution and statistics
    """
    try:
        import contactipm
    except ImportError:
        raise ImportError("contactipm module not found. Build the pybind11 bindings first.")
    
    t_start = time.time()
    
    nx = problem.nx
    nu = problem.nu
    N = problem.N
    dt = problem.T / N
    
    # Build all CasADi derivative callbacks
    cb = _build_contactipm_callbacks(problem)
    
    # Create solver
    solver = contactipm.Solver(nx, nu, 1, N, dt)  # nc=1 (minimum)
    
    # Set callbacks
    solver.set_callbacks(
        cb['f_disc'], cb['f_lin'],
        cb['stage_cost'], cb['term_cost'],
        cb['stage_grad'], cb['stage_hess'],
        cb['term_grad'], cb['term_hess'],
        cb['constr_eval'], cb['constr_eval_term'],
        cb['constr_jac'], cb['constr_jac_term'],
        cb['has_constraints'],
        cb['dyn_adj_hess'], cb['con_adj_hess'], cb['con_adj_hess_term'],
    )
    
    # Set bounds
    solver.set_bounds(problem.x_lb, problem.x_ub, problem.u_lb, problem.u_ub)
    
    # Set initial state
    solver.set_initial_state(problem.x0)
    
    # Set parameters
    solver.set_params(
        mu_init=mu_init,
        max_iters=max_iters,
        tol_primal=tol,
        tol_compl=tol,
        tol_stat=tol,  # match IPOpt's tolerance
        verbosity=verbosity,
        mu_min=mu_min,
    )
    
    # Initial guess — same as IPOPT's naive initialization
    # x[k] = x_init_guess (or x0) for all k, u[k] = 0 for all k
    # IMPORTANT: x[0] must equal x0 (the actual initial state), not x_init_guess
    if X_init is not None and U_init is not None:
        X_guess = np.ascontiguousarray(X_init)
        U_guess = np.ascontiguousarray(U_init)
    else:
        if problem.x_init_guess is not None:
            X_guess = np.tile(problem.x_init_guess, (N + 1, 1))
        else:
            X_guess = np.tile(problem.x0, (N + 1, 1))
        # Override x[0] to match the actual initial state
        X_guess[0] = problem.x0
        U_guess = np.zeros((N, nu))
    
    # Solve
    t_solve_start = time.time()
    X, U, stats = solver.solve(
        np.ascontiguousarray(X_guess),
        np.ascontiguousarray(U_guess),
    )
    t_solve_end = time.time()
    
    # Compute cost from trajectories
    cost = compute_cost(X, U, problem)
    
    return SolveResult(
        solver='contactipm',
        problem_name=problem.name,
        status=stats.get('status', 'Unknown'),
        iterations=stats.get('iterations', 0),
        time_solver=stats.get('time_solver', t_solve_end - t_solve_start),
        time_total=time.time() - t_start,
        cost=cost,
        X=np.asarray(X),
        U=np.asarray(U),
    )


if __name__ == "__main__":
    from .problems import CartPendulumMPC, QuadcopterMPC
    
    print("Testing IPOPT solver...")
    prob = CartPendulumMPC()
    result = solve_ipopt(prob, print_level=5)
    print(f"\n{result.solver}: status={result.status}, iter={result.iterations}")
    print(f"  cost={result.cost:.4f}, time={result.time_solver*1000:.2f}ms")
    print(f"  x_final = {result.X[-1]}")
    print(f"  u[0] = {result.U[0]}")
