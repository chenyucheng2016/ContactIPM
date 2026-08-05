"""
Pure CasADi transcriptions of fixed-time NMPC benchmark problems.

Each problem class defines:
- f_expl: CasADi Function mapping (x, u) -> xdot (continuous dynamics)
- l_stage: CasADi Function mapping (x, u) -> scalar (stage cost)
- l_terminal: CasADi Function mapping x -> scalar (terminal cost)
- g_path: CasADi Function mapping (x, u) -> constraint values, with lg/ug bounds
- x0: initial state (numpy array)
- x_lb, x_ub, u_lb, u_ub: variable bounds (numpy arrays)
- nx, nu, ng: dimensions
- N, T: horizon length and total time
- x_init_guess: initial guess for states (numpy array, shape nx)
"""

import casadi as ca
import numpy as np
from dataclasses import dataclass, field
from typing import Optional


@dataclass
class OCPProblem:
    """Base class for OCP problem definitions in CasADi."""
    name: str
    nx: int
    nu: int
    ng: int          # number of path constraint outputs (can be 0)
    N: int           # number of shooting intervals
    T: float         # total horizon time
    x0: np.ndarray   # initial state, shape (nx,)
    x_lb: np.ndarray # state lower bounds, shape (nx,)
    x_ub: np.ndarray # state upper bounds, shape (nx,)
    u_lb: np.ndarray # control lower bounds, shape (nu,)
    u_ub: np.ndarray # control upper bounds, shape (nu,)
    f_expl: ca.Function        # (x, u) -> xdot
    l_stage: ca.Function       # (x, u) -> scalar
    l_terminal: ca.Function    # x -> scalar
    g_path: Optional[ca.Function] = None  # (x, u) -> g, with lg <= g <= ug
    lg: Optional[np.ndarray] = None       # path constraint lower bounds
    ug: Optional[np.ndarray] = None       # path constraint upper bounds
    x_init_guess: Optional[np.ndarray] = None  # initial state guess


# ─────────────────────────────────────────────────────────────────────────────
#  Cart Pendulum MPC
# ─────────────────────────────────────────────────────────────────────────────

def CartPendulumMPC(N: int = 25, T: float = 1.0) -> OCPProblem:
    """
    Cart-pendulum swing-up MPC.
    
    States: x (pos), dx (vel), theta (angle), omega (angular vel)
    Control: Fex (force on cart)
    
    Objective: swing up from theta=0 (hanging down) to theta=pi (upright),
    minimize force and terminal penalties on angular velocity and position.
    """
    # Physical constants
    g = 9.82
    L = 1.0
    m = 1.0
    I_pend = m * L**2 / 12.0
    m_cart = 0.5
    
    # Bounds
    max_f = 5.0
    max_x = 1.0
    max_v = 2.0
    
    # Symbolic variables
    x = ca.MX.sym('x', 4)   # [pos, vel, theta, omega]
    u = ca.MX.sym('u', 1)   # [Fex]
    
    pos = x[0]
    vel = x[1]
    theta = x[2]
    omega = x[3]
    Fex = u[0]
    
    # Compute cart acceleration (ddx) and angular acceleration (alpha)
    # Derived from Lagrangian mechanics:
    #   ddx = (Fex + m*sin(theta)*(L*omega^2/2 + g*cos(theta))) / (m_cart + m*sin^2(theta))
    #   alpha = (-ddx*cos(theta) - g*sin(theta)) * (0.5*m*L) / (I_pend + 0.25*m*L^2)
    st = ca.sin(theta)
    ct = ca.cos(theta)
    
    ddx = (Fex + m * st * (0.5 * L * omega**2 + g * ct)) / (m_cart + m * st**2)
    alpha = (-ddx * ct - g * st) * (0.5 * m * L) / (I_pend + 0.25 * m * L**2)
    
    # Continuous dynamics: xdot = f(x, u)
    xdot = ca.vertcat(vel, ddx, omega, alpha)
    f_expl = ca.Function('f_expl', [x, u], [xdot], ['x', 'u'], ['xdot'])
    
    # Stage cost: alpha * Fex^2 + beta * omega^2 + pospen * pos^2 + thetapen * (theta - pi)^2
    # The (theta - pi)^2 term drives the pendulum to swing up from theta=0 (down)
    # to theta=pi (upright).
    alpha_cost = 1e0
    beta = 1e1
    pospen = 1e1
    thetapen = 1e2  # strong penalty on angle deviation from upright
    l_stage_expr = alpha_cost * Fex**2 + beta * omega**2 + pospen * pos**2 + thetapen * (theta - np.pi)**2
    l_stage = ca.Function('l_stage', [x, u], [l_stage_expr], ['x', 'u'], ['l'])
    
    # Terminal cost: beta * omega^2 + pospen * pos^2 + thetapen * (theta - pi)^2
    l_term_expr = beta * omega**2 + pospen * pos**2 + thetapen * (theta - np.pi)**2
    l_terminal = ca.Function('l_terminal', [x], [l_term_expr], ['x'], ['l'])
    
    # Path constraints (applied at stages 0..N-1, i.e. include_last=False)
    # -max_f <= Fex <= max_f is handled via u_lb/u_ub
    # -max_x <= pos <= max_x and -max_v <= vel <= max_v are state bounds
    # (applied at stages 1..N, i.e. include_first=False, include_last=True)
    # We'll handle these via x_lb/x_ub
    
    # No explicit path constraints beyond variable bounds
    g_path = None
    lg = None
    ug = None
    
    # Initial state: pendulum hanging down (theta=0), at rest
    # This is the classic swing-up problem: start from stable equilibrium (down),
    # swing to unstable equilibrium (up), and stabilize there
    x0_val = np.array([0.0, 0.0, 0.0, 0.0])
    
    # Variable bounds
    x_lb = np.array([-max_x, -max_v, -1e20, -1e20])
    x_ub = np.array([max_x, max_v, 1e20, 1e20])
    u_lb = np.array([-max_f])
    u_ub = np.array([max_f])
    
    # Initial guess: pendulum hanging down (same as initial state)
    x_init = np.array([0.0, 0.0, 0.0, 0.0])
    
    return OCPProblem(
        name='cart_pendulum_mpc',
        nx=4, nu=1, ng=0,
        N=N, T=T,
        x0=x0_val,
        x_lb=x_lb, x_ub=x_ub,
        u_lb=u_lb, u_ub=u_ub,
        f_expl=f_expl,
        l_stage=l_stage,
        l_terminal=l_terminal,
        g_path=g_path, lg=lg, ug=ug,
        x_init_guess=x_init,
    )


# ─────────────────────────────────────────────────────────────────────────────
#  Quadcopter MPC
# ─────────────────────────────────────────────────────────────────────────────

def QuadcopterMPC(N: int = 25, T: float = 0.5) -> OCPProblem:
    """
    Quadcopter MPC with attitude and thrust control.
    
    States: p[3] (position), v[3] (velocity)
    Controls: at (thrust), eul[3] (roll, pitch, yaw)
    
    The rotation matrix maps body-frame thrust to world-frame acceleration.
    Path constraints enforce tilt limits and angular rate limits.
    """
    # Parameters
    # at is specific force (thrust/mass, m/s^2), bounded in acceleration units
    mass = 9.18
    g = 9.81
    atmin = -0.1  # small negative thrust allowed (m/s^2)
    atmax = 5.0 * g  # max specific force = 5g (m/s^2)
    tiltmax = 1.1 / 2
    dtiltmax = 6.0 / 2
    
    # Initial conditions
    p0 = np.array([0., 0., 2.5])
    v0 = np.array([1., 1., 1.])
    eul0 = np.array([np.pi/10, np.pi/10, np.pi/10])
    
    # Cost weights
    alpha = 1.0
    beta = 1.0
    gamma = 100.0
    delta = 10.0
    reg = 0.1  # control weight for Riccati stability
    
    # Symbolic variables
    x = ca.MX.sym('x', 6)   # [p_x, p_y, p_z, v_x, v_y, v_z]
    u = ca.MX.sym('u', 4)   # [at, phi, theta, psi]
    
    p = x[:3]
    v = x[3:]
    at = u[0]
    phi = u[1]
    theta_u = u[2]
    psi = u[3]
    
    # Rotation matrix (ZYX Euler angles)
    cr = ca.cos(phi)
    sr = ca.sin(phi)
    cp = ca.cos(theta_u)
    sp = ca.sin(theta_u)
    cy = ca.cos(psi)
    sy = ca.sin(psi)
    
    R = ca.vertcat(
        ca.horzcat(cy*cp, cy*sp*sr - sy*cr, cy*sp*cr + sy*sr),
        ca.horzcat(sy*cp, sy*sp*sr + cy*cr, sy*sp*cr - cy*sr),
        ca.horzcat(-sp,   cp*sr,            cp*cr)
    )
    
    # Acceleration in world frame
    g_vec = ca.vertcat(0, 0, -9.81)
    at_world = R @ ca.vertcat(0, 0, at)
    a = at_world + g_vec
    
    # Continuous dynamics
    xdot = ca.vertcat(v, a)
    f_expl = ca.Function('f_expl', [x, u], [xdot], ['x', 'u'], ['xdot'])
    
    # Stage cost
    # 1e3*psi^2 + alpha*|v|^2 + beta*|der(eul)|^2 + gamma*|p-p0|^2 + delta*|eul|^2
    # + regularization: reg*(at^2 + phi^2 + theta^2 + psi^2 + |p|^2 + |v|^2)
    # Note: der(eul) is not available directly in our formulation - we'll handle it
    # as a constraint on (eul_{k+1} - eul_k)/dt in the path constraints
    # For the stage cost, we omit der(eul) (it's implicitly penalized via eul changes)
    
    p0_ca = ca.DM(p0)
    stage_cost_expr = (1e3 * psi**2 
                       + alpha * ca.dot(v, v)
                       + gamma * ca.dot(p - p0_ca, p - p0_ca)
                       + delta * (phi**2 + theta_u**2 + psi**2)
                       + reg * (at**2 + phi**2 + theta_u**2 + psi**2))
    l_stage = ca.Function('l_stage', [x, u], [stage_cost_expr], ['x', 'u'], ['l'])
    
    # Terminal cost: reg*(|p|^2 + |v|^2)
    # Note: original rockit has 1e2*(psi-eul0[2])^2 at terminal node, but psi is a control
    # In our formulation, we omit this terminal control penalty
    term_cost_expr = reg * (ca.dot(p, p) + ca.dot(v, v))
    l_terminal = ca.Function('l_terminal', [x], [term_cost_expr], ['x'], ['l'])
    
    # Path constraints (applied at all shooting nodes k=0..N-1)
    # 1. Tilt constraint: cos(theta)*cos(phi) >= cos(tiltmax)
    #    -> g = cos(tiltmax) - cos(theta)*cos(phi) <= 0
    # 2. Control bounds on at are handled via u_lb/u_ub
    # 3. Euler angle bounds are handled via u_lb/u_ub
    
    tilt_constr = ca.cos(tiltmax) - cp * cr  # <= 0
    g_path_expr = ca.vertcat(tilt_constr)
    g_path = ca.Function('g_path', [x, u], [g_path_expr], ['x', 'u'], ['g'])
    lg = np.array([-1e20])  # no lower bound
    ug = np.array([0.0])    # g <= 0
    
    # Initial state
    x0_val = np.concatenate([p0, v0])
    
    # Variable bounds
    x_lb = np.full(6, -1e20)
    x_ub = np.full(6, 1e20)
    
    u_lb = np.array([atmin, -np.pi/2, -np.pi/2, -1e20])
    u_ub = np.array([atmax, np.pi/2, np.pi/2, 1e20])
    
    # Initial guess
    x_init = np.concatenate([p0, v0])
    
    return OCPProblem(
        name='quadcopter_mpc',
        nx=6, nu=4, ng=1,
        N=N, T=T,
        x0=x0_val,
        x_lb=x_lb, x_ub=x_ub,
        u_lb=u_lb, u_ub=u_ub,
        f_expl=f_expl,
        l_stage=l_stage,
        l_terminal=l_terminal,
        g_path=g_path, lg=lg, ug=ug,
        x_init_guess=x_init,
    )


# ─────────────────────────────────────────────────────────────────────────────
#  Quadcopter Tracking MPC
# ─────────────────────────────────────────────────────────────────────────────

def QuadcopterTrackingMPC(N: int = 50, T: float = 5.0) -> OCPProblem:
    """
    Quadcopter tracking a circular reference trajectory.
    
    States: p[3] (position), v[3] (velocity), theta (reference phase)
    Controls: at (thrust), eul[3] (roll, pitch, yaw)
    
    The reference is a unit circle in the xy-plane at z=2.5, centered at p0.
    Phase theta advances at 2*pi/T so theta_k = k * 2*pi/N, making the
    tracking reference autonomous via state augmentation.
    
    Transcribed from fatrop_benchmarks/quadcopter/tracking/problem_specification.py
    """
    # Parameters (same as quadcopter dynamics)
    # at is specific force (thrust/mass, m/s^2), bounded in acceleration units
    mass = 9.18
    g = 9.81
    atmin = -0.1  # small negative thrust allowed (m/s^2)
    atmax = 5.0 * g  # max specific force = 5g (m/s^2)
    tiltmax = 1.1 / 2
    
    # Initial conditions
    p0 = np.array([0., 0., 2.5])
    v0 = np.array([0., 0., 0.])
    eul0 = np.array([0., 0., 0.])
    
    # Reference circle parameters
    # track_k = p0 + [cos(theta_k), sin(theta_k), 0]
    # theta advances at rate 2*pi/T so full circle in T seconds
    dtheta_total = 2.0 * np.pi  # total phase sweep
    theta_dot = dtheta_total / T  # phase rate (constant)
    
    # Symbolic variables
    # State: [px, py, pz, vx, vy, vz, theta_phase]  (nx=7)
    # Control: [at, phi, theta_eul, psi]  (nu=4)
    x = ca.MX.sym('x', 7)
    u = ca.MX.sym('u', 4)
    
    p_pos = x[:3]
    v = x[3:6]
    theta_phase = x[6]
    
    at = u[0]
    phi = u[1]
    theta_u = u[2]
    psi = u[3]
    
    # Rotation matrix (ZYX Euler angles)
    cr = ca.cos(phi)
    sr = ca.sin(phi)
    cp = ca.cos(theta_u)
    sp = ca.sin(theta_u)
    cy = ca.cos(psi)
    sy = ca.sin(psi)
    
    R = ca.vertcat(
        ca.horzcat(cy*cp, cy*sp*sr - sy*cr, cy*sp*cr + sy*sr),
        ca.horzcat(sy*cp, sy*sp*sr + cy*cr, sy*sp*cr - cy*sr),
        ca.horzcat(-sp,   cp*sr,            cp*cr)
    )
    
    # Acceleration in world frame
    g_vec = ca.vertcat(0, 0, -9.81)
    at_world = R @ ca.vertcat(0, 0, at)
    a = at_world + g_vec
    
    # Continuous dynamics: [v, a, theta_dot]
    xdot = ca.vertcat(v, a, theta_dot)
    f_expl = ca.Function('f_expl', [x, u], [xdot], ['x', 'u'], ['xdot'])
    
    # Reference position from augmented phase state
    p0_ca = ca.DM(p0)
    p_ref = p0_ca + ca.vertcat(ca.cos(theta_phase), ca.sin(theta_phase), 0.0)
    
    # Stage cost: |p - p_ref|^2 + 1e-8 * |u|^2  (at stages 0..N-1)
    # The tracking term also appears at terminal node (include_last=True in rockit),
    # but since we cannot evaluate p_ref at terminal node without theta being integrated,
    # we split: stage cost has tracking + control reg, terminal cost has tracking only.
    track_err = p_pos - p_ref
    stage_cost_expr = ca.dot(track_err, track_err) + 0.1 * ca.dot(u, u)
    l_stage = ca.Function('l_stage', [x, u], [stage_cost_expr], ['x', 'u'], ['l'])
    
    # Terminal cost: |p - p_ref|^2  (tracking error at final node)
    track_err_term = p_pos - p_ref
    term_cost_expr = ca.dot(track_err_term, track_err_term)
    l_terminal = ca.Function('l_terminal', [x], [term_cost_expr], ['x'], ['l'])
    
    # Path constraints: tilt constraint cos(theta_eul)*cos(phi) >= cos(tiltmax)
    tilt_constr = ca.cos(tiltmax) - cp * cr  # <= 0
    g_path_expr = ca.vertcat(tilt_constr)
    g_path = ca.Function('g_path', [x, u], [g_path_expr], ['x', 'u'], ['g'])
    lg = np.array([-1e20])
    ug = np.array([0.0])
    
    # Initial state: [p0, v0, theta=0]
    x0_val = np.concatenate([p0, v0, [0.0]])
    
    # Variable bounds
    x_lb = np.full(7, -1e20)
    x_ub = np.full(7, 1e20)
    
    u_lb = np.array([atmin, -np.pi/2, -np.pi/2, -1e20])
    u_ub = np.array([atmax, np.pi/2, np.pi/2, 1e20])
    
    # Initial guess: hover at p0 with theta sweeping
    x_init = np.concatenate([p0, v0, [0.0]])
    
    return OCPProblem(
        name='quadcopter_tracking_mpc',
        nx=7, nu=4, ng=1,
        N=N, T=T,
        x0=x0_val,
        x_lb=x_lb, x_ub=x_ub,
        u_lb=u_lb, u_ub=u_ub,
        f_expl=f_expl,
        l_stage=l_stage,
        l_terminal=l_terminal,
        g_path=g_path, lg=lg, ug=ug,
        x_init_guess=x_init,
    )


# ─────────────────────────────────────────────────────────────────────────────
#  Hanging Chain MPC (2D and 3D)
# ─────────────────────────────────────────────────────────────────────────────

def _HangingChainMPC(dim: int, N: int = 25, T: float = 2.0, 
                     no_masses: int = 6) -> OCPProblem:
    """
    Hanging chain MPC with spring-damper dynamics.
    
    Chain of no_masses+1 masses connected by springs, last mass is controlled.
    States: p[0..no_masses] (positions), v[0..no_masses-1] (velocities)
    Control: u (velocity of last mass)
    """
    # Parameters
    D = 1.6
    L = 0.0055
    m = 0.03
    g = 9.81
    
    # Number of states and controls
    nx = (no_masses + 1) * dim + no_masses * dim  # positions + velocities
    nu = dim
    
    # End position (target)
    if dim == 3:
        x_end = np.array([1.0, 0.0, 0.0])
    else:
        x_end = np.array([1.0, 0.0])
    
    # Ground position
    ground = np.zeros(dim)
    
    # Initial positions: linearly interpolated from ground to x_end
    # no_masses+2 points total (ground + no_masses+1 masses)
    init_pos = np.zeros((no_masses + 2, dim))
    for i in range(no_masses + 2):
        alpha = i / (no_masses + 1)
        init_pos[i] = (1 - alpha) * ground + alpha * x_end
    
    # Symbolic variables
    # x = [p[0], p[1], ..., p[no_masses], v[0], ..., v[no_masses-1]]
    # Each p[i] and v[i] is dim-dimensional
    xs = ca.MX.sym('x', nx)
    us = ca.MX.sym('u', nu)
    
    # Extract positions and velocities
    p_all = []
    for i in range(no_masses + 1):
        p_all.append(xs[i*dim:(i+1)*dim])
    
    v_all = []
    for i in range(no_masses):
        v_all.append(xs[(no_masses + 1)*dim + i*dim : (no_masses + 1)*dim + (i+1)*dim])
    
    # Build full position array including ground and controlled endpoint
    # p_full = [ground, p[0], p[1], ..., p[no_masses-1], p[no_masses]]
    # But p[no_masses] is the last free mass, and u controls its derivative
    p_full = [ca.DM(ground)] + p_all
    
    # Compute spring forces
    # F[i] = D * (1 - L/dist(p_full[i], p_full[i+1])) * (p_full[i+1] - p_full[i])
    # for i = 0, 1, ..., no_masses
    F = []
    for i in range(no_masses + 1):
        xi_m1 = p_full[i]
        xi = p_full[i + 1]
        diff = xi - xi_m1
        # dist = sqrt(|diff|^2), with safe handling for dist=0
        dist_sq = ca.dot(diff, diff)
        dist = ca.if_else(dist_sq > 1e-10, ca.sqrt(dist_sq), 1e-5)
        Fi = D * (1 - L / dist) * diff
        F.append(Fi)
    
    # Gravity vector
    if dim == 3:
        g_vec = ca.DM([0, -g, 0])
    else:
        g_vec = ca.DM([0, -g])
    
    # Compute accelerations
    # For i = 0..no_masses-1:
    #   dv[i]/dt = (F[i+1] - F[i] + m*g_vec) / m
    #   dp[i]/dt = v[i]
    # dp[no_masses]/dt = u (control)
    
    xdot_list = []
    
    # dp[i]/dt for i = 0..no_masses-1
    for i in range(no_masses):
        xdot_list.append(v_all[i])
    
    # dp[no_masses]/dt = u
    xdot_list.append(us)
    
    # dv[i]/dt for i = 0..no_masses-1
    for i in range(no_masses):
        Ftot = F[i + 1] - F[i] + m * g_vec
        dv = Ftot / m
        xdot_list.append(dv)
    
    xdot = ca.vertcat(*xdot_list)
    f_expl = ca.Function('f_expl', [xs, us], [xdot], ['x', 'u'], ['xdot'])
    
    # Cost function
    # Stage cost: alpha*|p[no_masses] - x_end|^2 + beta*sum(|v[i]|^2) + gamma*|u|^2
    alpha_cost = 25.0
    beta_cost = 1.0
    gamma_cost = 0.01
    
    x_end_ca = ca.DM(x_end)
    stage_cost_expr = alpha_cost * ca.dot(p_all[-1] - x_end_ca, p_all[-1] - x_end_ca)
    for i in range(no_masses):
        stage_cost_expr += beta_cost * ca.dot(v_all[i], v_all[i])
    stage_cost_expr += gamma_cost * ca.dot(us, us)
    
    l_stage = ca.Function('l_stage', [xs, us], [stage_cost_expr], ['x', 'u'], ['l'])
    
    # Terminal cost: same as stage cost but without control term
    l_term_expr = alpha_cost * ca.dot(p_all[-1] - x_end_ca, p_all[-1] - x_end_ca)
    for i in range(no_masses):
        l_term_expr += beta_cost * ca.dot(v_all[i], v_all[i])
    
    l_terminal = ca.Function('l_terminal', [xs], [l_term_expr], ['x'], ['l'])
    
    # No path constraints beyond variable bounds
    g_path = None
    lg = None
    ug = None
    
    # Control bounds: -1 <= u <= 1
    u_lb = np.full(nu, -1.0)
    u_ub = np.full(nu, 1.0)
    
    # State bounds: no explicit bounds
    x_lb = np.full(nx, -1e20)
    x_ub = np.full(nx, 1e20)
    
    # Initial state
    # Build x0: [p[0], p[1], ..., p[no_masses], v[0], ..., v[no_masses-1]]
    # Positions from init_pos[1:] (skip ground)
    # Velocities: zeros (but we should simulate forward to get a better initial state)
    
    x0_list = []
    for i in range(no_masses + 1):
        x0_list.append(init_pos[i + 1])
    for i in range(no_masses):
        x0_list.append(np.zeros(dim))
    
    x0_val = np.concatenate(x0_list)
    
    # Initial guess (same as x0 for simplicity)
    x_init = x0_val.copy()
    
    return OCPProblem(
        name=f'hanging_chain_{dim}d_mpc',
        nx=nx, nu=nu, ng=0,
        N=N, T=T,
        x0=x0_val,
        x_lb=x_lb, x_ub=x_ub,
        u_lb=u_lb, u_ub=u_ub,
        f_expl=f_expl,
        l_stage=l_stage,
        l_terminal=l_terminal,
        g_path=g_path, lg=lg, ug=ug,
        x_init_guess=x_init,
    )


def HangingChain2DMPC(N: int = 25, T: float = 2.0, no_masses: int = 6) -> OCPProblem:
    """2D hanging chain MPC."""
    return _HangingChainMPC(dim=2, N=N, T=T, no_masses=no_masses)


def HangingChain3DMPC(N: int = 25, T: float = 2.0, no_masses: int = 6) -> OCPProblem:
    """3D hanging chain MPC."""
    return _HangingChainMPC(dim=3, N=N, T=T, no_masses=no_masses)


# ─────────────────────────────────────────────────────────────────────────────
#  Test: verify dimensions
# ─────────────────────────────────────────────────────────────────────────────

if __name__ == "__main__":
    print("Testing problem definitions...")
    
    for prob_fn in [CartPendulumMPC, QuadcopterMPC, QuadcopterTrackingMPC, HangingChain2DMPC, HangingChain3DMPC]:
        prob = prob_fn()
        print(f"\n{prob.name}:")
        print(f"  nx={prob.nx}, nu={prob.nu}, ng={prob.ng}")
        print(f"  N={prob.N}, T={prob.T}, dt={prob.T/prob.N:.4f}")
        print(f"  x0 shape: {prob.x0.shape}")
        print(f"  f_expl output shape: {prob.f_expl(prob.x0, np.zeros(prob.nu)).shape}")
        print(f"  l_stage({prob.x0}, 0) = {float(prob.l_stage(prob.x0, np.zeros(prob.nu))):.4f}")
        print(f"  l_terminal({prob.x0}) = {float(prob.l_terminal(prob.x0)):.4f}")
        if prob.g_path is not None:
            g_val = prob.g_path(prob.x0, np.zeros(prob.nu))
            print(f"  g_path shape: {g_val.shape}")
            print(f"  g_path({prob.x0}, 0) = {g_val.full().flatten()}")
    
    print("\nAll problems validated successfully!")
