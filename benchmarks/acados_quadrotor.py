"""
ContactIPM vs acados benchmark: 2D Quadrotor Hover

  NX=6 [y, z, phi, vy, vz, dphi], NU=2 [u1, u2], N=20, dt=0.04
  Cost: quadratic tracking hover at (0, 2) with phi=0
  Constraints: 0 <= u1 <= 2mg, |u2| <= 0.5*mgl, z >= 0

ContactIPM result: 81 iterations, prim_inf=0.196
"""
import time, os, shutil
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
MASS   = 0.5        # kg
L_ARM  = 0.15       # m
I_YY   = 0.003      # kg*m^2
G      = 9.81
DT     = 0.04       # 40 ms
N      = 20
HOVER  = MASS * G   # ~4.905 N

W_Y, W_Z, W_PHI       = 5.0, 10.0, 2.0
W_VY, W_VZ, W_DPHI    = 0.5, 1.0, 0.3
W_U                     = 0.001

Y_DES, Z_DES = 0.0, 2.0

# Constraint bounds
U1_MAX = 2.0 * HOVER                 # 9.81
U2_MAX = 0.5 * HOVER * L_ARM         # 0.367875

x0 = np.array([0.5, 1.5, 0.1, 0.0, 0.0, 0.0])


def export_quadrotor_ocp():
    ocp = AcadosOcp()

    x = ca.SX.sym('x', 6)
    u = ca.SX.sym('u', 2)
    phi = x[2]

    sp = ca.sin(phi)
    cp = ca.cos(phi)

    # ── Continuous-time ODE ─────────────────────────────────────────────
    xdot = ca.vertcat(
        x[3],                       # y_dot = vy
        x[4],                       # z_dot = vz
        x[5],                       # phi_dot = dphi
        -u[0] * sp / MASS,          # vy_dot
         u[0] * cp / MASS - G,      # vz_dot
         u[1] * L_ARM / I_YY        # dphi_dot
    )

    f_expl = ca.Function('f_expl', [x, u], [xdot])
    ocp.model.f_expl_expr = f_expl(x, u)
    ocp.model.x = x
    ocp.model.u = u
    ocp.model.name = 'quadrotor_2d'

    # ── Cost ────────────────────────────────────────────────────────────
    dy = x[0] - Y_DES
    dz = x[1] - Z_DES
    dp = x[2]

    # Stage cost
    ocp.model.cost_expr_ext_cost = (
        0.5 * (W_Y*dy**2 + W_Z*dz**2 + W_PHI*dp**2
               + W_VY*x[3]**2 + W_VZ*x[4]**2 + W_DPHI*x[5]**2
               + W_U*(u[0]**2 + u[1]**2))
    )
    ocp.model.cost_expr_ext_cost_custom_hess = ca.diag(
        ca.vertcat(W_Y, W_Z, W_PHI, W_VY, W_VZ, W_DPHI, W_U, W_U)
    )

    # Terminal cost: 10x state weights, no control
    ocp.model.cost_expr_ext_cost_e = (
        10.0 * (W_Y*dy**2 + W_Z*dz**2 + W_PHI*dp**2
                + W_VY*x[3]**2 + W_VZ*x[4]**2 + W_DPHI*x[5]**2)
    )
    ocp.model.cost_expr_ext_cost_custom_hess_e = ca.diag(
        ca.vertcat(10*W_Y, 10*W_Z, 10*W_PHI, 10*W_VY, 10*W_VZ, 10*W_DPHI)
    )

    ocp.cost.cost_type   = 'EXTERNAL'
    ocp.cost.cost_type_e = 'EXTERNAL'

    # ── Constraints ─────────────────────────────────────────────────────
    ocp.constraints.lbu = np.array([0.0, -U2_MAX])
    ocp.constraints.ubu = np.array([U1_MAX, U2_MAX])
    ocp.constraints.idxbu = np.array([0, 1])

    ocp.constraints.lbx = np.array([0.0])
    ocp.constraints.ubx = np.array([1e10])
    ocp.constraints.idxbx = np.array([1])   # z index

    # Terminal: only z >= 0
    ocp.constraints.lbx_e = np.array([0.0])
    ocp.constraints.ubx_e = np.array([1e10])
    ocp.constraints.idxbx_e = np.array([1])

    # ── Solver options ──────────────────────────────────────────────────
    ocp.solver_options.tf         = N * DT
    ocp.solver_options.N_horizon  = N
    ocp.solver_options.integrator_type = 'ERK'
    ocp.solver_options.sim_method_num_stages = 4
    ocp.solver_options.sim_method_num_steps  = 1
    ocp.solver_options.nlp_solver_type       = 'SQP'
    ocp.solver_options.qp_solver             = 'PARTIAL_CONDENSING_HPIPM'
    ocp.solver_options.hessian_approx        = 'EXACT'
    ocp.solver_options.nlp_solver_max_iter   = 300
    ocp.solver_options.nlp_solver_tol_stat   = 5e-2
    ocp.solver_options.nlp_solver_tol_eq     = 1.5e-2
    ocp.solver_options.nlp_solver_tol_ineq   = 1e-2
    ocp.solver_options.nlp_solver_tol_comp   = 5e-2
    ocp.solver_options.print_level           = 1
    ocp.solver_options.qp_warm_start         = 0
    ocp.solver_options.globalization         = 'MERIT_BACKTRACKING'
    ocp.solver_options.regularize_method     = 'MIRROR'
    ocp.constraints.x0 = x0

    return ocp


def simulate_warm_start(x0, N, dt):
    """RK4 forward sim with hover thrust, matching ContactIPM warm-start."""
    u_hover = np.array([HOVER, 0.0])
    xs = [x0.copy()]
    us = [u_hover.copy() for _ in range(N)]
    for k in range(N):
        x = xs[-1]
        # RK4 with constant u_hover
        def f(s):
            phi = s[2]
            sp, cp = np.sin(phi), np.cos(phi)
            return np.array([
                s[3],                          # y_dot = vy
                s[4],                          # z_dot = vz
                s[5],                          # phi_dot = dphi
                -u_hover[0] * sp / MASS,       # vy_dot
                 u_hover[0] * cp / MASS - G,   # vz_dot
                 u_hover[1] * L_ARM / I_YY     # dphi_dot
            ])
        k1 = f(x)
        k2 = f(x + 0.5*dt*k1)
        k3 = f(x + 0.5*dt*k2)
        k4 = f(x + dt*k3)
        x_next = x + (dt/6.0)*(k1 + 2*k2 + 2*k3 + k4)
        xs.append(x_next)
    return xs, us


def compute_violations(ocp_solver, N):
    max_viol = 0.0
    for k in range(N + 1):
        xk = ocp_solver.get(k, 'x')
        max_viol = max(max_viol, max(0.0, -xk[1]))  # z >= 0
        if k < N:
            uk = ocp_solver.get(k, 'u')
            max_viol = max(max_viol,
                           max(0.0, -uk[0]),           # u1 >= 0
                           max(0.0, uk[0] - U1_MAX),   # u1 <= 2mg
                           max(0.0, -uk[1] - U2_MAX),   # u2 >= -u2_max
                           max(0.0, uk[1] - U2_MAX))    # u2 <= u2_max
    return max_viol


def main():
    print('=' * 60)
    print('  acados SQP+HPIPM — 2D Quadrotor Hover')
    print('=' * 60)

    ocp = export_quadrotor_ocp()

    # ── Solver setup ─────────────────────────────────────────────────
    t0 = time.time()
    ocp_solver = AcadosOcpSolver(ocp)
    t_setup = time.time() - t0
    print(f"  Solver setup (compile): {t_setup*1000:.0f} ms")

    # Warm-start: simulate forward with hover thrust (matches ContactIPM)
    xs_init, us_init = simulate_warm_start(x0, N, DT)
    for k in range(N + 1):
        ocp_solver.set(k, 'x', xs_init[k])
    for k in range(N):
        ocp_solver.set(k, 'u', us_init[k])

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
    print(f"First u* =       [{u0[0]:.3f}, {u0[1]:.3f}]")

    # ── Trajectory dump ──────────────────────────────────────────────
    NX_q, NU_q = 6, 2

    def quad_cost_fn(x, u, k):
        dy = x[0]-Y_DES; dz = x[1]-Z_DES; dp = x[2]
        if k < N:
            return 0.5*(W_Y*dy**2 + W_Z*dz**2 + W_PHI*dp**2
                        + W_VY*x[3]**2 + W_VZ*x[4]**2 + W_DPHI*x[5]**2
                        + W_U*(u[0]**2 + u[1]**2))
        else:
            return 10.0*(W_Y*dy**2 + W_Z*dz**2 + W_PHI*dp**2
                         + W_VY*x[3]**2 + W_VZ*x[4]**2 + W_DPHI*x[5]**2)

    def quad_cons_fn(x, u, k):
        if k < N:
            return np.array([
                -u[0],                          # -u1 <= 0
                u[0] - U1_MAX,                  # u1 <= 2mg
                -u[1] - U2_MAX,                 # -u2 <= 0.5mgl
                u[1] - U2_MAX,                  # u2 <= 0.5mgl
                -x[1],                          # -z <= 0
                -1e10                           # inactive
            ])
        else:
            return np.array([-1e10, -1e10, -1e10, -1e10, -x[1], -1e10])

    print_acados_table(ocp_solver, N, NX_q, NU_q, quad_cost_fn, quad_cons_fn)
    dump_acados_trajectory(ocp_solver, N, NX_q, NU_q, 'quadrotor',
                           quad_cost_fn, quad_cons_fn, DT, sqp_iters)


if __name__ == '__main__':
    main()
