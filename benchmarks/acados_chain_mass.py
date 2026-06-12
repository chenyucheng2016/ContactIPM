"""
ContactIPM vs acados benchmark: Chain of Masses

  5 masses, 3 actuated, NX=10, NU=3, N=20, dt=0.05
  Cost: drive to origin
  Constraints: |F_i| <= 5.0, |x_i| <= 2.0

ContactIPM result: 24 iterations, prim_inf=9.993e-04
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
N_MASSES   = 5
N_ACTUATED = 3
NX = 2 * N_MASSES   # 10
NU = N_ACTUATED      # 3
N  = 20
DT = 0.05

MASS_VAL = 1.0
SPRING_K = 10.0
DAMPING  = 0.5
F_MAX    = 5.0
X_MAX    = 2.0

W_POS, W_VEL, W_CTRL = 10.0, 1.0, 0.01

# Initial state: sine displacement
x0 = np.zeros(NX)
for i in range(N_MASSES):
    x0[i] = np.sin(np.pi * (i + 1) / (N_MASSES + 1)) * 1.5
    x0[N_MASSES + i] = 0.0


def export_chain_mass_ocp():
    ocp = AcadosOcp()

    x = ca.SX.sym('x', NX)
    u = ca.SX.sym('u', NU)

    # ── Continuous-time ODE: spring-mass chain ──────────────────────────
    xdot_list = []
    # Position derivatives: dx_i/dt = v_i
    for i in range(N_MASSES):
        xdot_list.append(x[N_MASSES + i])
    # Velocity derivatives: dv_i/dt = a_i
    for i in range(N_MASSES):
        x_i   = x[i]
        x_im1 = x[i - 1] if i > 0 else ca.SX(0.0)
        x_ip1 = x[i + 1] if i < N_MASSES - 1 else ca.SX(0.0)
        v_i   = x[N_MASSES + i]
        F_i   = u[i] if i < N_ACTUATED else ca.SX(0.0)

        spring = SPRING_K * (x_ip1 - 2.0 * x_i + x_im1)
        damp   = DAMPING * v_i
        a_i    = (F_i + spring - damp) / MASS_VAL
        xdot_list.append(a_i)

    xdot = ca.vertcat(*xdot_list)

    f_expl = ca.Function('f_expl', [x, u], [xdot])
    ocp.model.f_expl_expr = f_expl(x, u)
    ocp.model.x = x
    ocp.model.u = u
    ocp.model.name = 'chain_mass'

    # ── Cost ────────────────────────────────────────────────────────────
    # Stage: 0.5 * (sum w_pos*x_i^2 + w_vel*v_i^2 + w_ctrl*u_i^2)
    stage_cost = ca.SX(0.0)
    for i in range(N_MASSES):
        stage_cost += W_POS * x[i]**2
        stage_cost += W_VEL * x[N_MASSES + i]**2
    for i in range(NU):
        stage_cost += W_CTRL * u[i]**2
    ocp.model.cost_expr_ext_cost = 0.5 * stage_cost

    # Custom Hessian for stage
    Q_diag = []
    for i in range(N_MASSES):
        Q_diag.append(W_POS)
    for i in range(N_MASSES):
        Q_diag.append(W_VEL)
    for i in range(NU):
        Q_diag.append(W_CTRL)
    ocp.model.cost_expr_ext_cost_custom_hess = ca.diag(ca.vertcat(*Q_diag))

    # Terminal: 0.5 * (sum 20*w_pos*x_i^2 + 20*w_vel*v_i^2)
    term_cost = ca.SX(0.0)
    for i in range(N_MASSES):
        term_cost += 20.0 * W_POS * x[i]**2
        term_cost += 20.0 * W_VEL * x[N_MASSES + i]**2
    ocp.model.cost_expr_ext_cost_e = 0.5 * term_cost

    Q_diag_e = []
    for i in range(N_MASSES):
        Q_diag_e.append(20.0 * W_POS)
    for i in range(N_MASSES):
        Q_diag_e.append(20.0 * W_VEL)
    ocp.model.cost_expr_ext_cost_custom_hess_e = ca.diag(ca.vertcat(*Q_diag_e))

    ocp.cost.cost_type   = 'EXTERNAL'
    ocp.cost.cost_type_e = 'EXTERNAL'

    # ── Constraints ─────────────────────────────────────────────────────
    # Path: u box [-F_MAX, F_MAX], x box [-X_MAX, X_MAX]
    ocp.constraints.lbu = -F_MAX * np.ones(NU)
    ocp.constraints.ubu =  F_MAX * np.ones(NU)
    ocp.constraints.idxbu = np.arange(NU)

    ocp.constraints.lbx = -X_MAX * np.ones(N_MASSES)
    ocp.constraints.ubx =  X_MAX * np.ones(N_MASSES)
    ocp.constraints.idxbx = np.arange(N_MASSES)

    # Terminal: only x box (no force constraints)
    ocp.constraints.lbx_e = -X_MAX * np.ones(N_MASSES)
    ocp.constraints.ubx_e =  X_MAX * np.ones(N_MASSES)
    ocp.constraints.idxbx_e = np.arange(N_MASSES)

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
    ocp.solver_options.nlp_solver_tol_eq     = 1e-3
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
        for i in range(N_MASSES):
            max_viol = max(max_viol,
                           max(0.0, -xk[i] - X_MAX),
                           max(0.0, xk[i] - X_MAX))
        if k < N:
            uk = ocp_solver.get(k, 'u')
            for i in range(NU):
                max_viol = max(max_viol,
                               max(0.0, -uk[i] - F_MAX),
                               max(0.0, uk[i] - F_MAX))
    return max_viol


def main():
    print('=' * 60)
    print(f'  acados SQP+HPIPM — Chain of Masses (nx={NX}, nu={NU}, N={N})')
    print('=' * 60)

    ocp = export_chain_mass_ocp()

    # ── Solver setup ─────────────────────────────────────────────────
    t0 = time.time()
    ocp_solver = AcadosOcpSolver(ocp)
    t_setup = time.time() - t0
    print(f"  Solver setup (compile): {t_setup*1000:.0f} ms")

    # Initial guess: all stages at x0, u=0 (matches ContactIPM)
    for k in range(N + 1):
        ocp_solver.set(k, 'x', x0)
    for k in range(N):
        ocp_solver.set(k, 'u', np.zeros(NU))

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
    u_str = ', '.join(f'{u0[i]:.3f}' for i in range(NU))
    print(f"First u* =       [{u_str}]")

    # ── Trajectory dump ──────────────────────────────────────────────
    def chain_cost_fn(x, u, k):
        if k < N:
            c = 0.0
            for i in range(N_MASSES):
                c += W_POS * x[i]**2 + W_VEL * x[N_MASSES+i]**2
            for i in range(NU):
                c += W_CTRL * u[i]**2
            return 0.5 * c
        else:
            c = 0.0
            for i in range(N_MASSES):
                c += 20.0*W_POS * x[i]**2 + 20.0*W_VEL * x[N_MASSES+i]**2
            return 0.5 * c

    def chain_cons_fn(x, u, k):
        g_list = []
        if k < N:
            for i in range(NU):
                g_list.append(-F_MAX - u[i])
                g_list.append(u[i] - F_MAX)
        else:
            for i in range(NU):
                g_list.append(-1e10)
                g_list.append(-1e10)
        for i in range(N_MASSES):
            g_list.append(-X_MAX - x[i])
            g_list.append(x[i] - X_MAX)
        return np.array(g_list)

    print_acados_table(ocp_solver, N, NX, NU, chain_cost_fn, chain_cons_fn)
    dump_acados_trajectory(ocp_solver, N, NX, NU, 'chain_mass',
                           chain_cost_fn, chain_cons_fn, DT, sqp_iters)


if __name__ == '__main__':
    main()
