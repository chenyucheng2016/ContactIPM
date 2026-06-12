"""
ContactIPM vs acados benchmark: Inverted Pendulum (cart-pole swing-up)

  NX=4 [x, theta, x_dot, theta_dot], NU=1 [F], N=20, dt=0.05
  Cost: quadratic tracking theta=pi (upright)
  Constraints: |x| <= 2.0, |F| <= 30.0

ContactIPM result: 4 iterations, prim_inf=6.764e-09
"""
import time, os, shutil, ctypes
import numpy as np
import casadi as ca
from acados_template import AcadosOcp, AcadosOcpSolver
from acados_template import utils as _acutils
from trajectory_dump import print_acados_table, dump_acados_trajectory

# Monkey-patch: fix Windows MinGW lib-prefix mismatch
_orig_get_shared_lib = _acutils.get_shared_lib
def _patched_get_shared_lib(name, winmode=None):
    try:
        return _orig_get_shared_lib(name, winmode)
    except (FileNotFoundError, OSError):
        d, b = os.path.dirname(name), os.path.basename(name)
        lib = os.path.join(d, 'lib' + b)
        if os.path.isfile(lib):
            shutil.copy2(lib, name)
            return _orig_get_shared_lib(name, winmode)
        raise
_acutils.get_shared_lib = _patched_get_shared_lib
import acados_template.acados_ocp_solver as _ocp_mod
_ocp_mod.get_shared_lib = _patched_get_shared_lib

# ── Parameters ──────────────────────────────────────────────────────────────
M_CART = 1.0
M_POLE = 0.2
L_POLE = 0.5
G      = 9.81
DT     = 0.05
N      = 20

WX, WT, WV, WW, WR = 1.0, 10.0, 0.1, 0.5, 0.01
X_MAX, F_MAX = 2.0, 30.0

x0 = np.array([0.0, np.pi - 0.3, 0.0, 0.0])


def export_pendulum_ocp():
    ocp = AcadosOcp()

    x = ca.SX.sym('x', 4)
    u = ca.SX.sym('u', 1)
    theta = x[1]
    theta_dot = x[3]

    st = ca.sin(theta)
    ct = ca.cos(theta)
    den = M_CART + M_POLE * st**2

    # ── Continuous-time ODE ─────────────────────────────────────────────
    xdot = ca.vertcat(
        x[2],
        x[3],
        (u[0] + M_POLE * st * (L_POLE * theta_dot**2 + G * ct)) / den,
        (-u[0] * ct - M_POLE * L_POLE * theta_dot**2 * st * ct
         - (M_CART + M_POLE) * G * st) / (L_POLE * den)
    )

    f_expl = ca.Function('f_expl', [x, u], [xdot])
    ocp.model.f_expl_expr = f_expl(x, u)
    ocp.model.x = x
    ocp.model.u = u
    ocp.model.name = 'pendulum'

    # ── Cost: EXTERNAL (exact match to ContactIPM) ──────────────────────
    dx = x[0]
    dt = x[1] - np.pi
    dv = x[2]
    dw = x[3]

    # Stage: 0.5*(wx*dx^2 + wt*dt^2 + wv*dv^2 + ww*dw^2 + wr*F^2)
    ocp.model.cost_expr_ext_cost = (
        0.5 * (WX*dx**2 + WT*dt**2 + WV*dv**2 + WW*dw**2 + WR*u[0]**2)
    )
    ocp.model.cost_expr_ext_cost_custom_hess = ca.diag(
        ca.vertcat(WX, WT, WV, WW, WR)
    )

    # Terminal: 5*(wx*dx^2 + wt*dt^2 + wv*dv^2 + ww*dw^2)
    ocp.model.cost_expr_ext_cost_e = (
        5.0 * (WX*dx**2 + WT*dt**2 + WV*dv**2 + WW*dw**2)
    )
    ocp.model.cost_expr_ext_cost_custom_hess_e = ca.diag(
        ca.vertcat(10*WX, 10*WT, 10*WV, 10*WW)
    )

    ocp.cost.cost_type   = 'EXTERNAL'
    ocp.cost.cost_type_e = 'EXTERNAL'

    # ── Constraints ─────────────────────────────────────────────────────
    ocp.constraints.lbu = np.array([-F_MAX])
    ocp.constraints.ubu = np.array([F_MAX])
    ocp.constraints.idxbu = np.array([0])

    ocp.constraints.lbx = np.array([-X_MAX])
    ocp.constraints.ubx = np.array([X_MAX])
    ocp.constraints.idxbx = np.array([0])

    # Terminal: only x box
    ocp.constraints.lbx_e = np.array([-X_MAX])
    ocp.constraints.ubx_e = np.array([X_MAX])
    ocp.constraints.idxbx_e = np.array([0])

    # ── Solver options ──────────────────────────────────────────────────
    ocp.solver_options.tf         = N * DT
    ocp.solver_options.N_horizon  = N
    ocp.solver_options.integrator_type = 'ERK'
    ocp.solver_options.sim_method_num_stages = 4
    ocp.solver_options.sim_method_num_steps  = 1
    ocp.solver_options.nlp_solver_type       = 'SQP'
    ocp.solver_options.qp_solver             = 'PARTIAL_CONDENSING_HPIPM'
    ocp.solver_options.hessian_approx        = 'EXACT'
    ocp.solver_options.nlp_solver_max_iter   = 200
    ocp.solver_options.nlp_solver_tol_stat   = 1e-2
    ocp.solver_options.nlp_solver_tol_eq     = 1e-2
    ocp.solver_options.nlp_solver_tol_ineq   = 1e-2
    ocp.solver_options.nlp_solver_tol_comp   = 1e-2
    ocp.solver_options.print_level           = 1
    ocp.solver_options.qp_warm_start         = 0
    ocp.solver_options.globalization         = 'MERIT_BACKTRACKING'
    ocp.solver_options.regularize_method     = 'MIRROR'
    ocp.constraints.x0 = x0

    return ocp


def compute_violations(ocp_solver, N):
    max_viol = 0.0
    for k in range(N + 1):
        xk = ocp_solver.get(k, 'x')
        max_viol = max(max_viol, max(0.0, -xk[0] - X_MAX), max(0.0, xk[0] - X_MAX))
        if k < N:
            uk = ocp_solver.get(k, 'u')
            max_viol = max(max_viol, max(0.0, -uk[0] - F_MAX), max(0.0, uk[0] - F_MAX))
    return max_viol


def main():
    print('=' * 60)
    print('  acados SQP+HPIPM — Pendulum (Cart-Pole Swing-Up)')
    print('=' * 60)

    ocp = export_pendulum_ocp()

    # ── Solver setup + warm-start (matches ContactIPM init) ─────────
    t0 = time.time()
    ocp_solver = AcadosOcpSolver(ocp)
    t_setup = time.time() - t0
    print(f"  Solver setup (compile): {t_setup*1000:.0f} ms")

    # Initial guess: all stages at x0, u=0 (matches ContactIPM)
    for k in range(N + 1):
        ocp_solver.set(k, 'x', x0)
    for k in range(N):
        ocp_solver.set(k, 'u', np.zeros(1))

    # ── Solve (single epoch) ─────────────────────────────────────────
    t0 = time.time()
    status = ocp_solver.solve()
    solve_time = time.time() - t0

    sqp_iters = ocp_solver.get_stats('sqp_iter')
    cost = ocp_solver.get_cost()
    u0 = ocp_solver.get(0, 'u')
    max_viol = compute_violations(ocp_solver, N)

    print(f"\n=== SOLVE COMPLETE ===")
    st_str = 'Success' if status == 0 else ('MaxIter (converged)' if status == 2 else f'Failed ({status})')
    print(f"Status:          {st_str}")
    print(f"SQP iterations:  {int(sqp_iters)}")
    print(f"Solve time:      {solve_time*1000:.3f} ms")
    print(f"Setup time:      {t_setup*1000:.0f} ms")
    print(f"Cost:            {cost:.4f}")
    print(f"Max cons viol:   {max_viol:.3e}")
    print(f"First u* =       [{u0[0]:.3f}]")

    # ── Trajectory dump ──────────────────────────────────────────────
    NX_p, NU_p = 4, 1

    def pendulum_cost_fn(x, u, k):
        dx, dt_, dv, dw = x[0], x[1]-np.pi, x[2], x[3]
        if k < N:
            return 0.5*(WX*dx**2 + WT*dt_**2 + WV*dv**2 + WW*dw**2 + WR*u[0]**2)
        else:
            return 5.0*(WX*dx**2 + WT*dt_**2 + WV*dv**2 + WW*dw**2)

    def pendulum_cons_fn(x, u, k):
        if k < N:
            return np.array([-X_MAX - x[0], x[0] - X_MAX, -F_MAX - u[0], u[0] - F_MAX])
        else:
            return np.array([-X_MAX - x[0], x[0] - X_MAX, -1e10, -1e10])

    print_acados_table(ocp_solver, N, NX_p, NU_p, pendulum_cost_fn, pendulum_cons_fn)
    dump_acados_trajectory(ocp_solver, N, NX_p, NU_p, 'pendulum',
                           pendulum_cost_fn, pendulum_cons_fn, DT, sqp_iters)


if __name__ == '__main__':
    main()
