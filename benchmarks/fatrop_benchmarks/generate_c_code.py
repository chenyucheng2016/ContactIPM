#!/usr/bin/env python3
"""
CasADi code generator for all 9 fatrop benchmarks.

Generates C functions for dynamics, cost, and constraints that can be
shared across ContactIPM, acados, and IPOPT solvers.

Usage:
    python generate_c_code.py [problem_name]
    If no argument given, generates for all problems.
"""

import casadi as cs
import numpy as np
import os
import sys


# ════════════════════════════════════════════════════════════════════════
#  Helpers
# ════════════════════════════════════════════════════════════════════════

def _add_all_functions(cg, fns):
    for fn in fns:
        cg.add(fn)



def _make_cost_functions(x_sym, u_sym, stage_expr, term_expr):
    """Create cost casadi Functions from symbolic expressions.
    All Jacobian/Hessian outputs are densified for direct C consumption."""
    fns = []
    fns.append(cs.Function('l_stage', [x_sym, u_sym], [stage_expr]))
    fns.append(cs.Function('l_stage_grad_x', [x_sym, u_sym],
                            [cs.gradient(stage_expr, x_sym)]))
    fns.append(cs.Function('l_stage_grad_u', [x_sym, u_sym],
                            [cs.gradient(stage_expr, u_sym)]))
    fns.append(cs.Function('l_stage_hess_xx', [x_sym, u_sym],
                            [cs.densify(cs.hessian(stage_expr, x_sym)[0])]))
    fns.append(cs.Function('l_stage_hess_uu', [x_sym, u_sym],
                            [cs.densify(cs.hessian(stage_expr, u_sym)[0])]))
    fns.append(cs.Function('l_stage_hess_ux', [x_sym, u_sym],
                            [cs.densify(cs.jacobian(cs.gradient(stage_expr, u_sym), x_sym))]))
    fns.append(cs.Function('l_term', [x_sym], [term_expr]))
    fns.append(cs.Function('l_term_grad', [x_sym], [cs.gradient(term_expr, x_sym)]))
    fns.append(cs.Function('l_term_hess', [x_sym],
                            [cs.densify(cs.hessian(term_expr, x_sym)[0])]))
    return fns


def _make_constraint_functions(x_sym, u_sym, path_expr, term_expr):
    """Create constraint casadi Functions.
    All Jacobian outputs are densified for direct C consumption."""
    fns = []
    fns.append(cs.Function('g_path', [x_sym, u_sym], [path_expr]))
    fns.append(cs.Function('g_path_jac_x', [x_sym, u_sym],
                            [cs.densify(cs.jacobian(path_expr, x_sym))]))
    fns.append(cs.Function('g_path_jac_u', [x_sym, u_sym],
                            [cs.densify(cs.jacobian(path_expr, u_sym))]))
    fns.append(cs.Function('g_term', [x_sym], [term_expr]))
    fns.append(cs.Function('g_term_jac', [x_sym],
                            [cs.densify(cs.jacobian(term_expr, x_sym))]))
    return fns


def _make_dynamics_functions(x_sym, u_sym, xdot):
    """Create dynamics casadi Functions.
    Jacobian outputs are densified for direct C consumption."""
    return [
        cs.Function('f_expl', [x_sym, u_sym], [xdot]),
        cs.Function('f_jac_x', [x_sym, u_sym], [cs.densify(cs.jacobian(xdot, x_sym))]),
        cs.Function('f_jac_u', [x_sym, u_sym], [cs.densify(cs.jacobian(xdot, u_sym))]),
    ]


def write_metadata(output_dir, meta):
    """Write problem metadata as a C header."""
    x0_str = ', '.join(f'{v}' for v in meta['x0'])
    u_init_str = ', '.join(f'{v}' for v in meta['u_init'])
    x_init_str = ', '.join(f'{v}' for v in meta['x_init'])
    header = f"""// Auto-generated problem metadata for {meta['name']}
#pragma once
#define PROB_NX     {meta['nx']}
#define PROB_NU     {meta['nu']}
#define PROB_NC     {meta['nc']}
#define PROB_N      {meta['N']}
#define PROB_T      {meta['T']}
#define PROB_DT     {meta['dt']}
#define PROB_FREE_TIME {meta.get('free_time', 0)}

static const double PROB_X0[] = {{{x0_str}}};
static const double PROB_U_INIT[] = {{{u_init_str}}};
static const double PROB_X_INIT[] = {{{x_init_str}}};
"""
    with open(os.path.join(output_dir, 'problem_meta.h'), 'w') as f:
        f.write(header)


def _generate_problem(name, x_sym, u_sym, xdot, stage_cost, term_cost,
                      path_constr, term_constr, meta, base_dir):
    """Common code-gen pipeline for any problem."""
    all_fns = (_make_dynamics_functions(x_sym, u_sym, xdot)
               + _make_cost_functions(x_sym, u_sym, stage_cost, term_cost)
               + _make_constraint_functions(x_sym, u_sym, path_constr, term_constr))

    cg = cs.CodeGenerator(name)
    _add_all_functions(cg, all_fns)

    output_dir = os.path.join(base_dir, '..', 'fatrop_codegen', name)
    os.makedirs(output_dir, exist_ok=True)
    cg.generate(output_dir + os.sep)
    write_metadata(output_dir, meta)
    print(f"  Generated {name} -> {output_dir}/")


# ════════════════════════════════════════════════════════════════════════
#  Cart Pendulum dynamics (shared by MPC and free-time variants)
# ════════════════════════════════════════════════════════════════════════

def _cart_pendulum_dynamics(x_sym, u_sym):
    """Returns xdot for cart-pole system.
    x = [pos, theta, vel, omega], u = [Fex]
    """
    g, L, m, m_cart = 9.82, 1.0, 1.0, 0.5
    I = m * L**2 / 12.0

    pos, theta, vel, omega = x_sym[0], x_sym[1], x_sym[2], x_sym[3]
    Fex = u_sym[0]
    st, ct = cs.sin(theta), cs.cos(theta)
    J_coeff = 1.0 / (I + 0.25 * m * L**2)

    ddx_sym = cs.SX.sym('ddx')
    alpha_expr = J_coeff * 0.5 * L * m * (-ddx_sym * ct - g * st)
    ddCOG_x = 0.5 * L * (alpha_expr * ct - omega**2 * st) + ddx_sym
    ddCOG_y = 0.5 * L * (alpha_expr * st + omega**2 * ct)
    F_x, F_y = m * ddCOG_x, m * (ddCOG_y + g)
    eq = -F_x + Fex - m_cart * ddx_sym
    J_cart = cs.jacobian(eq, ddx_sym)
    c_val = cs.substitute(eq, ddx_sym, 0.0)
    ddx_explicit = -c_val / J_cart
    alpha_explicit = cs.substitute(alpha_expr, ddx_sym, ddx_explicit)

    return cs.vertcat(vel, omega, ddx_explicit, alpha_explicit)


# ════════════════════════════════════════════════════════════════════════
#  Quadcopter dynamics (shared by MPC, P2P, obstacle variants)
# ════════════════════════════════════════════════════════════════════════

def _quadcopter_dynamics(p_sym, v_sym, at, phi, theta, psi):
    """Returns [v; accel] for quadcopter."""
    g = 9.81
    cr, sr = cs.cos(phi), cs.sin(phi)
    cp, sp = cs.cos(theta), cs.sin(theta)
    cy, sy = cs.cos(psi), cs.sin(psi)
    R = cs.vertcat(
        cs.horzcat(cy*cp, cy*sp*sr - sy*cr, cy*sp*cr + sy*sr),
        cs.horzcat(sy*cp, sy*sp*sr + cy*cr, sy*sp*cr - cy*sr),
        cs.horzcat(-sp, cp*sr, cp*cr))
    at_vec = R @ cs.vertcat(0, 0, at)
    accel = at_vec + cs.vertcat(0, 0, -g)
    return cs.vertcat(v_sym, accel)


def _quadcopter_path_constraints(phi, theta, cp, cr, at, atmin=0, atmax=None,
                                  tiltmax=None):
    """Standard quadcopter inequality constraints g <= 0."""
    if atmax is None: atmax = 9.18 * 5
    if tiltmax is None: tiltmax = 1.1 / 2
    return cs.vertcat(
        -phi - cs.pi/2, phi - cs.pi/2,
        -theta - cs.pi/2, theta - cs.pi/2,
        cs.cos(tiltmax) - cp * cr,
        -at + atmin, at - atmax)


# ════════════════════════════════════════════════════════════════════════
#  Hanging Chain dynamics
# ════════════════════════════════════════════════════════════════════════

def _hanging_chain_dynamics(x_sym, u_sym, no_masses, dim):
    """Build xdot for hanging chain.
    State: [p0(0:dim), p1(dim:2*dim), ..., p_{nm}((nm)*dim:(nm+1)*dim),
            v0((nm+1)*dim:...), ..., v_{nm-1}]
    Control: u(dim) = velocity of last mass
    """
    D, L, m, g = 1.6, 0.0055, 0.03, 9.81
    ground = cs.DM.zeros(dim, 1)

    # Extract position and velocity sub-vectors
    p = []
    for i in range(no_masses + 1):
        p.append(x_sym[i*dim:(i+1)*dim])
    v = []
    offset = (no_masses + 1) * dim
    for i in range(no_masses):
        v.append(x_sym[offset + i*dim : offset + (i+1)*dim])

    # All positions including ground anchor
    p_all = [ground] + p

    # Spring forces
    def dist(a, b):
        d = b - a
        sq = cs.dot(d, d)
        return cs.if_else(sq > 0, cs.sqrt(sq), 1e-8)

    F = []
    for i in range(no_masses + 1):
        xim1 = p_all[i]
        xi = p_all[i + 1]
        d = dist(xim1, xi)
        Fim1_i = D * (1 - L / d) * (xi - xim1)
        F.append(Fim1_i)

    # Accelerations
    grav = cs.DM.zeros(dim, 1)
    grav[1] = -g  # y-component

    xdot_parts = []
    # dp_i/dt = v_i for i = 0..no_masses-1
    for i in range(no_masses):
        xdot_parts.append(v[i])
    # dp_{no_masses}/dt = u (last mass velocity = control)
    xdot_parts.append(u_sym)

    # dv_i/dt for i = 0..no_masses-1
    for i in range(no_masses):
        Fi_ip1 = F[i + 1]
        Fim1_i = F[i]
        Ftot = Fi_ip1 - Fim1_i + m * grav
        xdot_parts.append(Ftot / m)

    return cs.vertcat(*xdot_parts)


# ════════════════════════════════════════════════════════════════════════
#  Truck-Trailer dynamics
# ════════════════════════════════════════════════════════════════════════

def _truck_trailer_dynamics(x_sym, u_sym):
    """Truck-trailer with 2 trailers.
    State: [theta2, x2, y2, theta1, theta0]
    Control: [delta0, v0]
    """
    L0, M0, W0 = 0.4, 0.1, 0.2
    L1, M1, W1 = 1.1, 0.2, 0.2
    L2, M2, W2 = 0.8, 0.1, 0.2

    theta2 = x_sym[0]; x2 = x_sym[1]; y2 = x_sym[2]
    theta1 = x_sym[3]; theta0 = x_sym[4]
    delta0 = u_sym[0]; v0 = u_sym[1]

    beta01 = theta0 - theta1
    beta12 = theta1 - theta2

    dtheta0 = v0 / L0 * cs.tan(delta0)
    dtheta1 = v0 / L1 * cs.sin(beta01) - M0 / L1 * cs.cos(beta01) * dtheta0
    v1 = v0 * cs.cos(beta01) + M0 * cs.sin(beta01) * dtheta0
    dtheta2 = v1 / L2 * cs.sin(beta12) - M1 / L2 * cs.cos(beta12) * dtheta1
    v2 = v1 * cs.cos(beta12) + M1 * cs.sin(beta12) * dtheta1

    return cs.vertcat(dtheta2, v2 * cs.cos(theta2), v2 * cs.sin(theta2),
                      dtheta1, dtheta0)


# ════════════════════════════════════════════════════════════════════════
#  Problem 1: CartPendulumMPC
# ════════════════════════════════════════════════════════════════════════

def generate_cart_pendulum_mpc(output_dir):
    NX, NU, NC, N = 4, 1, 6, 25
    T, dt = 2.5, 0.1
    alpha, beta, gamma, pospen = 1.0, 10.0, 10.0, 0.01
    max_f, max_x, max_v = 5.0, 1.0, 2.0

    x_sym = cs.SX.sym('x', NX)
    u_sym = cs.SX.sym('u', NU)
    pos, theta, vel, omega = x_sym[0], x_sym[1], x_sym[2], x_sym[3]
    Fex = u_sym[0]

    xdot = _cart_pendulum_dynamics(x_sym, u_sym)
    stage_cost = alpha * Fex**2 + beta * omega**2 + gamma * vel**2 + pospen * pos**2
    term_cost = beta * omega**2 + gamma * vel**2

    path_constr = cs.vertcat(
        -Fex - max_f, Fex - max_f,
        -pos - max_x, pos - max_x,
        -vel - max_v, vel - max_v)
    term_constr = cs.vertcat(
        -1e10, -1e10,
        -pos - max_x, pos - max_x,
        -vel - max_v, vel - max_v)

    meta = {'name': 'cart_pendulum_mpc', 'nx': NX, 'nu': NU, 'nc': NC,
            'N': N, 'T': T, 'dt': dt,
            'x0': [0.0, np.pi, 0.0, 1.0], 'u_init': [0.0],
            'x_init': [0.0, np.pi, 0.0, 1.0]}
    base = os.path.dirname(os.path.abspath(__file__))
    _generate_problem('cart_pendulum_mpc', x_sym, u_sym, xdot,
                      stage_cost, term_cost, path_constr, term_constr, meta, base)
    return meta


# ════════════════════════════════════════════════════════════════════════
#  Problem 2: CartPendulumTime (free-time swing-up)
# ════════════════════════════════════════════════════════════════════════

def generate_cart_pendulum_time(output_dir):
    """Free-time: x_aug = [pos,theta,vel,omega,T], u_aug = [Fex, dT]"""
    NX, NU, NC, N = 5, 2, 6, 100
    T_guess = 2.0
    dt = 1.0 / N  # normalized timestep
    max_f, max_x, max_v = 5.0, 1.0, 2.0

    x_sym = cs.SX.sym('x', NX)
    u_sym = cs.SX.sym('u', NU)
    # Physical sub-vectors
    x_phys = x_sym[0:4]
    u_phys = u_sym[0:1]

    xdot_phys = _cart_pendulum_dynamics(x_phys, u_phys)
    # Augmented: dT/dt = 0
    xdot = cs.vertcat(xdot_phys, 0.0)

    pos, theta, vel, omega = x_sym[0], x_sym[1], x_sym[2], x_sym[3]
    Fex = u_sym[0]

    # Cost: minimize T (terminal) + small stage regularization
    stage_cost = 1e-3 * Fex**2 + 1e-4 * u_sym[1]**2
    term_cost = x_sym[4]  # T

    # Constraints: same as MPC but with augmented state
    path_constr = cs.vertcat(
        -Fex - max_f, Fex - max_f,
        -pos - max_x, pos - max_x,
        -vel - max_v, vel - max_v)
    # Terminal: theta near pi, omega near 0, pos/vel bounds
    term_constr = cs.vertcat(
        -1e10, -1e10,
        -pos - max_x, pos - max_x,
        cs.cos(theta) + 0.95,  # theta near pi: cos(pi)=-1, so cos(theta)+0.95 <= 0 means cos(theta) <= -0.95
        vel**2 - max_v**2)

    x0 = [0.0, 0.0, 0.0, 0.0, T_guess]
    meta = {'name': 'cart_pendulum_time', 'nx': NX, 'nu': NU, 'nc': NC,
            'N': N, 'T': T_guess, 'dt': dt, 'free_time': 1,
            'x0': x0, 'u_init': [0.0, 0.0], 'x_init': x0}
    base = os.path.dirname(os.path.abspath(__file__))
    _generate_problem('cart_pendulum_time', x_sym, u_sym, xdot,
                      stage_cost, term_cost, path_constr, term_constr, meta, base)
    return meta


# ════════════════════════════════════════════════════════════════════════
#  Problem 3: QuadCopterMPC
# ════════════════════════════════════════════════════════════════════════

def generate_quadcopter_mpc(output_dir):
    NX, NU, N = 6, 4, 25
    # NC=1: only tilt constraint (phi/theta/at bounds handled as variable bounds)
    NC = 1
    T, dt = 2.0, 0.08

    x_sym = cs.SX.sym('x', NX)
    u_sym = cs.SX.sym('u', NU)
    p, v = x_sym[0:3], x_sym[3:6]
    at, phi, theta, psi = u_sym[0], u_sym[1], u_sym[2], u_sym[3]
    cp, cr = cs.cos(theta), cs.cos(phi)

    xdot = _quadcopter_dynamics(p, v, at, phi, theta, psi)

    p0 = cs.vertcat(0., 0., 2.5)
    eul = u_sym[1:4]
    stage_cost = (1e3 * psi**2 + cs.dot(v, v)
                  + 100.0 * cs.dot(p - p0, p - p0)
                  + 10.0 * cs.dot(eul, eul)
                  + 1e-4 * cs.dot(u_sym, u_sym))
    term_cost = 1e-4 * cs.dot(x_sym, x_sym)

    # Only tilt constraint (nonlinear); bounds handled as variable bounds
    tiltmax = 1.1 / 2
    path_constr = cs.cos(tiltmax) - cp * cr
    # Terminal: tilt constraint not applicable (controls absent at terminal)
    # Use g=0 (always satisfied with s=0) instead of g=-1e10 (requires s=1e10)
    term_constr = cs.vertcat(*([0.0] * NC))

    x0_val = [0., 0., 2.5, 1., 1., 1.]
    meta = {'name': 'quadcopter_mpc', 'nx': NX, 'nu': NU, 'nc': NC,
            'N': N, 'T': T, 'dt': dt,
            'x0': x0_val, 'u_init': [9.81, np.pi/10, np.pi/10, np.pi/10],
            'x_init': x0_val}
    base = os.path.dirname(os.path.abspath(__file__))
    _generate_problem('quadcopter_mpc', x_sym, u_sym, xdot,
                      stage_cost, term_cost, path_constr, term_constr, meta, base)
    return meta


# ════════════════════════════════════════════════════════════════════════
#  Problem 4: QuadCopterP2P (free-time point-to-point)
# ════════════════════════════════════════════════════════════════════════

def generate_quadcopter_p2p(output_dir):
    """Free-time: x_aug = [p(3),v(3),T], u_aug = [at,phi,theta,psi,dT]"""
    NX, NU, NC, N = 7, 5, 7, 25
    T_guess = 3.0
    dt = 1.0 / N

    x_sym = cs.SX.sym('x', NX)
    u_sym = cs.SX.sym('u', NU)
    p, v = x_sym[0:3], x_sym[3:6]
    at, phi, theta, psi = u_sym[0], u_sym[1], u_sym[2], u_sym[3]
    cp, cr = cs.cos(theta), cs.cos(phi)

    xdot_phys = _quadcopter_dynamics(p, v, at, phi, theta, psi)
    xdot = cs.vertcat(xdot_phys, 0.0)  # dT/dt = 0

    # Cost: minimize T + psi^2 regularization
    stage_cost = 1e2 * psi**2 + 1e-4 * cs.dot(u_sym, u_sym)
    term_cost = x_sym[6]  # T

    # Path constraints
    path_constr = _quadcopter_path_constraints(phi, theta, cp, cr, at)
    # Terminal: position near target, velocity near zero (controls absent at terminal)
    pf = cs.vertcat(0.01, 5., 2.5)
    term_constr = cs.vertcat(
        cs.dot(p - pf, p - pf) - 0.01,   # position error < 0.1
        cs.dot(v, v) - 0.01,              # velocity error < 0.1
        p[0] - 0.11, -p[0] - 0.11,       # px near 0.01
        p[1] - 5.1, -p[1] + 4.9,         # py near 5.0
        -1e10, -1e10)

    x0_val = [0., 0., 2.5, 0., 0., 0., T_guess]
    meta = {'name': 'quadcopter_p2p', 'nx': NX, 'nu': NU, 'nc': NC,
            'N': N, 'T': T_guess, 'dt': dt, 'free_time': 1,
            'x0': x0_val, 'u_init': [9.81, 0., 0., 0., 0.],
            'x_init': x0_val}
    base = os.path.dirname(os.path.abspath(__file__))
    _generate_problem('quadcopter_p2p', x_sym, u_sym, xdot,
                      stage_cost, term_cost, path_constr, term_constr, meta, base)
    return meta


# ════════════════════════════════════════════════════════════════════════
#  Problem 5: QuadCopterOneObs (free-time, one obstacle)
# ════════════════════════════════════════════════════════════════════════

def generate_quadcopter_one_obs(output_dir):
    """x_aug = [p(3),v(3),T], u_aug = [at,phi,theta,psi,slack]"""
    NX, NU, NC, N = 7, 5, 8, 50
    T_guess = 5.0
    dt = 1.0 / N

    p0_val = [0., 0., 7.5]
    pf_val = [10., 10., 7.5]
    p_obs = [0.5*p0_val[i] + 0.5*pf_val[i] for i in range(3)]
    r_obs, r_drone = 2.0, 0.30

    x_sym = cs.SX.sym('x', NX)
    u_sym = cs.SX.sym('u', NU)
    p, v = x_sym[0:3], x_sym[3:6]
    at, phi, theta, psi, slack = u_sym[0], u_sym[1], u_sym[2], u_sym[3], u_sym[4]
    cp, cr = cs.cos(theta), cs.cos(phi)

    xdot_phys = _quadcopter_dynamics(p, v, at, phi, theta, psi)
    xdot = cs.vertcat(xdot_phys, 0.0)

    # Obstacle distance in xy-plane
    dx, dy = p[0] - p_obs[0], p[1] - p_obs[1]
    obs_dist = cs.if_else(dx**2 + dy**2 > 0, cs.sqrt(dx**2 + dy**2), 1e-8)

    # Cost: minimize T + slack penalty + psi regularization
    stage_cost = 1e2 * slack + 1e2 * psi**2 + 1e-4 * cs.dot(u_sym[0:4], u_sym[0:4])
    term_cost = x_sym[6]  # T

    # Path constraints: standard + obstacle avoidance + slack >= 0
    base_constr = _quadcopter_path_constraints(phi, theta, cp, cr, at)
    path_constr = cs.vertcat(
        base_constr,
        -slack,  # slack >= 0
        -(obs_dist - (r_obs + r_drone) + slack))  # obs_dist + slack >= r_obs+r_drone

    pf = cs.vertcat(*pf_val)
    # Terminal: position + velocity near target (controls absent at terminal)
    term_constr = cs.vertcat(
        cs.dot(p - pf, p - pf) - 0.01,
        cs.dot(v, v) - 0.01,
        p[0] - pf_val[0] - 0.1, -p[0] + pf_val[0] - 0.1,
        p[1] - pf_val[1] - 0.1, -p[1] + pf_val[1] - 0.1,
        -1e10, -1e10)

    x0_val = p0_val + [0., 0., 0., T_guess]
    meta = {'name': 'quadcopter_one_obs', 'nx': NX, 'nu': NU, 'nc': NC,
            'N': N, 'T': T_guess, 'dt': dt, 'free_time': 1,
            'x0': x0_val, 'u_init': [9.81, 0., 0., 0., 0.],
            'x_init': x0_val}
    base = os.path.dirname(os.path.abspath(__file__))
    _generate_problem('quadcopter_one_obs', x_sym, u_sym, xdot,
                      stage_cost, term_cost, path_constr, term_constr, meta, base)
    return meta


# ════════════════════════════════════════════════════════════════════════
#  Problem 6: QuadCopterThreeObs (free-time, three obstacles)
# ════════════════════════════════════════════════════════════════════════

def generate_quadcopter_three_obs(output_dir):
    """x_aug = [p(3),v(3),T], u_aug = [at,phi,theta,psi,slack1,slack2,slack3]"""
    NX, NU, NC, N = 7, 7, 10, 100
    T_guess = 5.0
    dt = 1.0 / N

    p0_val = [0., 0., 7.5]
    pf_val = [15., 15., 7.5]
    r_obs, r_drone = 2.0, 0.30
    p_obs1 = [0.2*p0_val[i] + 0.8*pf_val[i] for i in range(3)]
    p_obs2 = [0.5*p0_val[i] + 0.5*pf_val[i] for i in range(3)]
    p_obs2[2] += 1.0
    p_obs3 = [0.8*p0_val[i] + 0.2*pf_val[i] for i in range(3)]
    p_obs3[0] -= 1.0

    x_sym = cs.SX.sym('x', NX)
    u_sym = cs.SX.sym('u', NU)
    p, v = x_sym[0:3], x_sym[3:6]
    at, phi, theta, psi = u_sym[0], u_sym[1], u_sym[2], u_sym[3]
    sl = [u_sym[4], u_sym[5], u_sym[6]]
    cp, cr = cs.cos(theta), cs.cos(phi)

    xdot_phys = _quadcopter_dynamics(p, v, at, phi, theta, psi)
    xdot = cs.vertcat(xdot_phys, 0.0)

    def obs_dist_xy(p_pos, obs):
        dx, dy = p_pos[0] - obs[0], p_pos[1] - obs[1]
        return cs.if_else(dx**2 + dy**2 > 0, cs.sqrt(dx**2 + dy**2), 1e-8)

    def obs_dist_xz(p_pos, obs):
        dx, dz = p_pos[0] - obs[0], p_pos[2] - obs[2]
        return cs.if_else(dx**2 + dz**2 > 0, cs.sqrt(dx**2 + dz**2), 1e-8)

    d1 = obs_dist_xy(p, p_obs1)
    d2 = obs_dist_xz(p, p_obs2)
    d3 = obs_dist_xy(p, p_obs3)

    stage_cost = 1e2 * (sl[0] + sl[1] + sl[2]) + 1e-4 * cs.dot(u_sym[0:4], u_sym[0:4])
    term_cost = x_sym[6]  # T

    base_constr = _quadcopter_path_constraints(phi, theta, cp, cr, at)
    path_constr = cs.vertcat(
        base_constr,
        -sl[0], -(d1 - (r_obs + r_drone) + sl[0]),
        -sl[1], -(d2 - (r_obs + r_drone) + sl[1]),
        -sl[2], -(d3 - (r_obs + r_drone) + sl[2]))

    pf = cs.vertcat(*pf_val)
    # Terminal: position + velocity near target (controls absent at terminal)
    term_constr = cs.vertcat(
        cs.dot(p - pf, p - pf) - 0.01,
        cs.dot(v, v) - 0.01,
        p[0] - pf_val[0] - 0.1, -p[0] + pf_val[0] - 0.1,
        p[1] - pf_val[1] - 0.1, -p[1] + pf_val[1] - 0.1,
        -1e10, -1e10,
        -1e10, -1e10, -1e10)

    x0_val = p0_val + [0., 0., 0., T_guess]
    meta = {'name': 'quadcopter_three_obs', 'nx': NX, 'nu': NU, 'nc': NC,
            'N': N, 'T': T_guess, 'dt': dt, 'free_time': 1,
            'x0': x0_val, 'u_init': [9.81, 0., 0., 0., 0., 0., 0.],
            'x_init': x0_val}
    base = os.path.dirname(os.path.abspath(__file__))
    _generate_problem('quadcopter_three_obs', x_sym, u_sym, xdot,
                      stage_cost, term_cost, path_constr, term_constr, meta, base)
    return meta


# ════════════════════════════════════════════════════════════════════════
#  Problem 7: TruckTrailerTime (free-time motion planning)
# ════════════════════════════════════════════════════════════════════════

def generate_truck_trailer_time(output_dir):
    """x_aug = [theta2,x2,y2,theta1,theta0,T], u_aug = [delta0,v0,dT]"""
    NX, NU, NC, N = 6, 3, 10, 50
    T_guess = 20.0
    dt = 1.0 / N

    x_sym = cs.SX.sym('x', NX)
    u_sym = cs.SX.sym('u', NU)
    x_phys = x_sym[0:5]
    u_phys = u_sym[0:2]

    xdot_phys = _truck_trailer_dynamics(x_phys, u_phys)
    xdot = cs.vertcat(xdot_phys, 0.0)

    theta2, x2, y2 = x_sym[0], x_sym[1], x_sym[2]
    theta1, theta0 = x_sym[3], x_sym[4]
    delta0, v0 = u_sym[0], u_sym[1]
    beta01 = theta0 - theta1
    beta12 = theta1 - theta2

    # Cost: minimize T + beta penalties
    stage_cost = beta01**2 + beta12**2 + 1e-4 * cs.dot(u_sym, u_sym)
    term_cost = x_sym[5]  # T

    # Path constraints
    speedf = 1.0
    path_constr = cs.vertcat(
        -0.2 * speedf - v0, v0 - 0.2 * speedf,         # speed bounds
        -np.pi/6 - delta0, delta0 - np.pi/6,            # steering bounds
        -np.pi/2 - beta01, beta01 - np.pi/2,            # joint angle bounds
        -np.pi/2 - beta12, beta12 - np.pi/2,            # joint angle bounds
        -1e10, -1e10)                                    # padding to NC=10

    # Terminal: target position + angles
    tf_x, tf_y = 0.0, -2.0
    tf_angle = np.pi / 2
    term_constr = cs.vertcat(
        (x2 - tf_x)**2 - 0.01,
        (y2 - tf_y)**2 - 0.01,
        (theta2 - tf_angle)**2 - 0.01,
        beta01**2 - 0.01,
        beta12**2 - 0.01,
        -1e10, -1e10, -1e10, -1e10, -1e10)

    x0_val = [0., 0., 0., 0., 0.1, T_guess]
    meta = {'name': 'truck_trailer_time', 'nx': NX, 'nu': NU, 'nc': NC,
            'N': N, 'T': T_guess, 'dt': dt, 'free_time': 1,
            'x0': x0_val, 'u_init': [0., -0.2, 0.],
            'x_init': x0_val}
    base = os.path.dirname(os.path.abspath(__file__))
    _generate_problem('truck_trailer_time', x_sym, u_sym, xdot,
                      stage_cost, term_cost, path_constr, term_constr, meta, base)
    return meta


# ════════════════════════════════════════════════════════════════════════
#  Problem 8: HangingChain2DMPC
# ════════════════════════════════════════════════════════════════════════

def generate_hanging_chain_2d(output_dir):
    no_masses, dim = 6, 2
    NX = (no_masses + 1) * dim + no_masses * dim  # 14 + 12 = 26
    NU = dim  # 2
    NC = 2 * dim  # 4 (u bounds + padding)
    N = 40
    T = 2.0
    dt = T / N

    alpha, beta_cost, gamma = 25.0, 1.0, 0.01
    x_end = np.array([1.0, 0.0])

    x_sym = cs.SX.sym('x', NX)
    u_sym = cs.SX.sym('u', NU)

    xdot = _hanging_chain_dynamics(x_sym, u_sym, no_masses, dim)

    # Stage cost: alpha*|p_last - x_end|^2 + beta*sum|v_i|^2 + gamma*|u|^2
    p_last = x_sym[no_masses * dim : (no_masses + 1) * dim]
    x_end_dm = cs.DM(x_end)
    stage_cost = alpha * cs.dot(p_last - x_end_dm, p_last - x_end_dm)
    offset_v = (no_masses + 1) * dim
    for i in range(no_masses):
        vi = x_sym[offset_v + i*dim : offset_v + (i+1)*dim]
        stage_cost += beta_cost * cs.dot(vi, vi)
    stage_cost += gamma * cs.dot(u_sym, u_sym)

    # Terminal cost: same position + velocity penalty
    term_cost = alpha * cs.dot(p_last - x_end_dm, p_last - x_end_dm)
    for i in range(no_masses):
        vi = x_sym[offset_v + i*dim : offset_v + (i+1)*dim]
        term_cost += beta_cost * cs.dot(vi, vi)

    # Path constraints: -1 <= u <= 1 for each dim
    path_parts = []
    for d in range(dim):
        path_parts.extend([-u_sym[d] - 1.0, u_sym[d] - 1.0])
    path_constr = cs.vertcat(*path_parts)
    # Terminal: same (u inactive but padded)
    term_constr = cs.vertcat(*([-1e10] * NC))

    # Initial state: linear interpolation from ground to x_end
    ground = np.zeros(dim)
    x0 = []
    for i in range(no_masses + 1):
        pi_val = ground + (i + 1) / (no_masses + 1) * (x_end - ground)
        x0.extend(pi_val.tolist())
    for i in range(no_masses):
        x0.extend([0.0] * dim)

    meta = {'name': 'hanging_chain_2d', 'nx': NX, 'nu': NU, 'nc': NC,
            'N': N, 'T': T, 'dt': dt,
            'x0': x0, 'u_init': [0.0] * dim, 'x_init': x0}
    base = os.path.dirname(os.path.abspath(__file__))
    _generate_problem('hanging_chain_2d', x_sym, u_sym, xdot,
                      stage_cost, term_cost, path_constr, term_constr, meta, base)
    return meta


# ════════════════════════════════════════════════════════════════════════
#  Problem 9: HangingChain3DMPC
# ════════════════════════════════════════════════════════════════════════

def generate_hanging_chain_3d(output_dir):
    no_masses, dim = 6, 3
    NX = (no_masses + 1) * dim + no_masses * dim  # 21 + 18 = 39
    NU = dim  # 3
    NC = 2 * dim  # 6 (u bounds)
    N = 40
    T, dt = 2.0, 2.0 / N

    alpha, beta_cost, gamma = 25.0, 1.0, 0.01
    x_end = np.array([1.0, 0.0, 0.0])

    x_sym = cs.SX.sym('x', NX)
    u_sym = cs.SX.sym('u', NU)

    xdot = _hanging_chain_dynamics(x_sym, u_sym, no_masses, dim)

    p_last = x_sym[no_masses * dim : (no_masses + 1) * dim]
    x_end_dm = cs.DM(x_end)
    stage_cost = alpha * cs.dot(p_last - x_end_dm, p_last - x_end_dm)
    offset_v = (no_masses + 1) * dim
    for i in range(no_masses):
        vi = x_sym[offset_v + i*dim : offset_v + (i+1)*dim]
        stage_cost += beta_cost * cs.dot(vi, vi)
    stage_cost += gamma * cs.dot(u_sym, u_sym)

    term_cost = alpha * cs.dot(p_last - x_end_dm, p_last - x_end_dm)
    for i in range(no_masses):
        vi = x_sym[offset_v + i*dim : offset_v + (i+1)*dim]
        term_cost += beta_cost * cs.dot(vi, vi)

    path_parts = []
    for d in range(dim):
        path_parts.extend([-u_sym[d] - 1.0, u_sym[d] - 1.0])
    path_constr = cs.vertcat(*path_parts)
    term_constr = cs.vertcat(*([-1e10] * NC))

    ground = np.zeros(dim)
    x0 = []
    for i in range(no_masses + 1):
        pi_val = ground + (i + 1) / (no_masses + 1) * (x_end - ground)
        x0.extend(pi_val.tolist())
    for i in range(no_masses):
        x0.extend([0.0] * dim)

    meta = {'name': 'hanging_chain_3d', 'nx': NX, 'nu': NU, 'nc': NC,
            'N': N, 'T': T, 'dt': dt,
            'x0': x0, 'u_init': [0.0] * dim, 'x_init': x0}
    base = os.path.dirname(os.path.abspath(__file__))
    _generate_problem('hanging_chain_3d', x_sym, u_sym, xdot,
                      stage_cost, term_cost, path_constr, term_constr, meta, base)
    return meta


# ════════════════════════════════════════════════════════════════════════
#  Main
# ════════════════════════════════════════════════════════════════════════

if __name__ == '__main__':
    problems = {
        'cart_pendulum_mpc': generate_cart_pendulum_mpc,
        'cart_pendulum_time': generate_cart_pendulum_time,
        'quadcopter_mpc': generate_quadcopter_mpc,
        'quadcopter_p2p': generate_quadcopter_p2p,
        'quadcopter_one_obs': generate_quadcopter_one_obs,
        'quadcopter_three_obs': generate_quadcopter_three_obs,
        'truck_trailer_time': generate_truck_trailer_time,
        'hanging_chain_2d': generate_hanging_chain_2d,
        'hanging_chain_3d': generate_hanging_chain_3d,
    }

    if len(sys.argv) > 1:
        names = sys.argv[1:]
    else:
        names = list(problems.keys())

    for name in names:
        if name not in problems:
            print(f"Unknown problem: {name}. Available: {list(problems.keys())}")
            continue
        print(f"Generating C code for: {name}")
        try:
            problems[name](None)
        except Exception as e:
            print(f"  ERROR generating {name}: {e}")

    print("\nDone.")
