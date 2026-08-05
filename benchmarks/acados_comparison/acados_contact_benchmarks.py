#!/usr/bin/env python3
"""acados baselines for the frozen CRISP and IMPACT contact benchmarks."""

from __future__ import annotations

import argparse
import json
import math
import os
import sys
from dataclasses import dataclass
from pathlib import Path
from typing import Callable

import casadi as ca
import numpy as np
from acados_template import AcadosModel, AcadosOcp, AcadosOcpSolver


ROOT = Path(__file__).resolve().parents[2]
GENERATED_ROOT = ROOT / ".deps" / "acados_generated"
ACADOS_INFTY = 1e15


@dataclass
class Benchmark:
    suite: str
    problem: str
    model: AcadosModel
    horizon: int
    dt: float
    initial_state: np.ndarray
    default_goal: np.ndarray
    initialization: str
    trajectory: Callable[[np.ndarray, np.ndarray], np.ndarray]

    @property
    def name(self) -> str:
        return f"contact_{self.suite}_{self.problem}"

    @property
    def directory(self) -> Path:
        return GENERATED_ROOT / self.name

    @property
    def json_path(self) -> Path:
        return self.directory / f"{self.name}.json"


def _model(name: str, nx: int, nu: int, np_: int) -> tuple:
    model = AcadosModel()
    model.name = name
    model.x = ca.SX.sym("x", nx)
    model.u = ca.SX.sym("u", nu)
    model.p = ca.SX.sym("p", np_)
    return model, model.x, model.u, model.p


def _set_constraints(
    model: AcadosModel,
    sides: list,
    products: list,
    equalities: list | None = None,
    terminal_sides: list | None = None,
) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    equalities = equalities or []
    model.con_h_expr = ca.vertcat(*(sides + products + equalities))
    model.con_h_expr_0 = model.con_h_expr
    lh = np.array(
        [0.0] * len(sides)
        + [-ACADOS_INFTY] * len(products)
        + [0.0] * len(equalities)
    )
    uh = np.array(
        [ACADOS_INFTY] * len(sides)
        + [0.0] * len(products)
        + [0.0] * len(equalities)
    )
    if terminal_sides:
        model.con_h_expr_e = ca.vertcat(*terminal_sides)
        lh_e = np.zeros(len(terminal_sides))
        uh_e = np.full(len(terminal_sides), ACADOS_INFTY)
    else:
        lh_e = np.array([])
        uh_e = np.array([])
    return lh, uh, lh_e, uh_e


def _crisp_cartpole() -> tuple[Benchmark, tuple]:
    name = "contact_crisp_cartpole_soft_walls"
    model, x, u, goal = _model(name, 4, 3, 4)
    dt = 0.02
    cart_mass, pole_mass, length, gravity = 1.0, 0.1, 0.8, 9.8
    cosine, sine = ca.cos(x[1]), ca.sin(x[1])
    denominator = -pole_mass * cosine**2 + cart_mass + pole_mass
    acceleration = (
        u[2] - u[1] + u[0] + u[1] * cosine**2
        - u[2] * cosine**2 - gravity * pole_mass * cosine * sine
        + length * pole_mass * x[3] ** 2 * sine
    ) / denominator
    angular_acceleration = -(
        u[1] * cart_mass * cosine - u[2] * cart_mass * cosine
        + pole_mass * u[0] * cosine - gravity * pole_mass**2 * sine
        - gravity * cart_mass * pole_mass * sine
        + length * pole_mass**2 * x[3] ** 2 * cosine * sine
    ) / (length * pole_mass * denominator)
    velocity_next = x[2] + dt * acceleration
    angular_velocity_next = x[3] + dt * angular_acceleration
    model.disc_dyn_expr = ca.vertcat(
        x[0] + dt * velocity_next,
        x[1] + dt * angular_velocity_next,
        velocity_next,
        angular_velocity_next,
    )
    model.cost_expr_ext_cost = 0.001 * u[0] ** 2
    model.cost_expr_ext_cost_e = 100.0 * ca.sumsqr(x - goal)
    left_gap = 1.0 - x[0] - length * sine + u[1] / 200.0
    right_gap = 1.0 + x[0] + length * sine + u[2] / 200.0
    sides = [u[1], u[2], left_gap, right_gap]
    products = [u[1] * left_gap, u[2] * right_gap]
    bounds = _set_constraints(model, sides, products)

    def trajectory(X: np.ndarray, U: np.ndarray) -> np.ndarray:
        controls = np.vstack([U, np.zeros((1, 3))])
        return np.hstack([X, controls])

    benchmark = Benchmark(
        "crisp", "cartpole_soft_walls", model, 99, dt,
        np.array([0.0, 3.14159, 0.0, 2.8]), np.zeros(4),
        "crisp_file", trajectory,
    )
    return benchmark, bounds


def _push_box(
    suite: str,
    horizon: int,
    dt: float,
    mass: float,
    gravity: float,
    half_x: float,
    half_y: float,
    rotation_scale: float,
    include_contact_point_cost: bool,
    initialization: str,
) -> tuple[Benchmark, tuple]:
    name = f"contact_{suite}_push_box"
    model, x, u, goal = _model(name, 3, 6, 3)
    inverse_drag = 1.0 / (0.5 * mass * gravity)
    force_x = u[3] + u[5]
    force_y = u[2] + u[4]
    cosine, sine = ca.cos(x[2]), ca.sin(x[2])
    radius = math.hypot(half_x, half_y)
    model.disc_dyn_expr = ca.vertcat(
        x[0] + dt * inverse_drag * (cosine * force_x - sine * force_y),
        x[1] + dt * inverse_drag * (sine * force_x + cosine * force_y),
        x[2] + dt * inverse_drag / (rotation_scale * radius)
        * (-u[1] * force_x + u[0] * force_y),
    )
    effort = ca.sumsqr(u) if include_contact_point_cost else ca.sumsqr(u[2:6])
    model.cost_expr_ext_cost = 0.001 * effort
    model.cost_expr_ext_cost_e = 100.0 * ca.sumsqr(x - goal)
    signed_force = [u[2], u[3], -u[4], -u[5]]
    gaps = [u[1] + half_y, u[0] + half_x, half_y - u[1], half_x - u[0]]
    products = [
        signed_force[index] * gaps[index] for index in range(4)
    ] + [
        signed_force[first] * signed_force[second]
        for first in range(4) for second in range(first + 1, 4)
    ]
    bounds = _set_constraints(model, signed_force + gaps, products)

    def trajectory(X: np.ndarray, U: np.ndarray) -> np.ndarray:
        controls = np.vstack([U, np.zeros((1, 6))])
        return np.hstack([X, controls])

    default_goal = (
        np.array([-1.5, -2.5980762113533156, 4.1887902047863905])
        if suite == "crisp" else np.array([1.0, 1.0, math.pi])
    )
    benchmark = Benchmark(
        suite, "push_box", model, horizon, dt, np.zeros(3),
        default_goal, initialization, trajectory,
    )
    return benchmark, bounds


def _crisp_transport() -> tuple[Benchmark, tuple]:
    name = "contact_crisp_transport"
    model, x, u, goal = _model(name, 4, 3, 4)
    dt = 0.02
    positive = u[0]
    negative = positive - x[2] + x[3]
    friction, active = u[1], u[2]
    payload_velocity_next = x[2] + dt * friction
    cart_velocity_next = x[3] + dt * (active - friction) / 2.0
    model.disc_dyn_expr = ca.vertcat(
        x[0] + dt * payload_velocity_next,
        x[1] + dt * cart_velocity_next,
        payload_velocity_next,
        cart_velocity_next,
    )
    model.cost_expr_ext_cost = 1e-4 * (friction**2 + active**2)
    error = x - goal
    model.cost_expr_ext_cost_e = (
        1e5 * (error[0] ** 2 + error[1] ** 2)
        + 10.0 * (error[2] ** 2 + error[3] ** 2)
    )
    limit = 0.2 * 1.0 * 9.81
    geometry = [x[0] - x[1] + 1.0, 1.0 - (x[0] - x[1])]
    sides = [
        positive, negative, limit - friction, friction + limit, *geometry
    ]
    products = [
        positive * negative,
        negative * (limit - friction),
        positive * (friction + limit),
    ]
    bounds = _set_constraints(
        model, sides, products, terminal_sides=geometry
    )

    def trajectory(X: np.ndarray, U: np.ndarray) -> np.ndarray:
        rows = np.zeros((X.shape[0], 8))
        rows[:, :4] = X
        rows[:-1, 4] = U[:, 0]
        rows[:-1, 5] = U[:, 0] - X[:-1, 2] + X[:-1, 3]
        rows[:-1, 6:8] = U[:, 1:3]
        return rows

    benchmark = Benchmark(
        "crisp", "transport", model, 199, dt,
        np.array([3.5, 3.0, -4.0, -4.0]),
        np.array([-0.5, 0.0, -2.0, -2.0]), "zeros", trajectory,
    )
    return benchmark, bounds


def _push_t_terms(u, retained: bool) -> tuple[list, list, list, list]:
    unit, dc = 0.05, 2.6429
    cx, cy = u[0], u[1]
    lambdas = [u[2 + i] for i in range(8)]
    v = [u[10 + i] if not retained else u[2 + i] for i in range(7)]
    coordinates = [
        cx - 2.0 * unit,
        cy - (4.0 - dc) * unit,
        cy - (3.0 - dc) * unit,
        cx - 0.5 * unit,
        cy + dc * unit,
        cx + 0.5 * unit,
        cx + 2.0 * unit,
    ]
    if retained:
        lambdas = [u[9 + i] for i in range(8)]
        w = [v[i] - coordinates[i] for i in range(7)]
        split_equalities: list = []
    else:
        w = [u[17 + i] for i in range(7)]
        split_equalities = [
            v[i] - w[i] - coordinates[i] for i in range(7)
        ]
    absolute = [v[i] + w[i] for i in range(7)]
    gaps = [
        (4.0 - dc) * unit - cy,
        absolute[0] + absolute[1] + absolute[2] - unit,
        absolute[0] + absolute[2] + absolute[3] - 1.5 * unit,
        absolute[2] + absolute[3] + absolute[4] - 3.0 * unit,
        absolute[3] + absolute[4] + absolute[5] - unit,
        absolute[2] + absolute[4] + absolute[5] - 3.0 * unit,
        absolute[2] + absolute[5] + absolute[6] - 1.5 * unit,
        absolute[1] + absolute[2] + absolute[6] - unit,
    ]
    signs = [-1.0, -1.0, 1.0, -1.0, 1.0, 1.0, 1.0, 1.0]
    forces = [signs[i] * lambdas[i] for i in range(8)]
    bounds = [
        cx + 2.0 * unit, 2.0 * unit - cx,
        cy + dc * unit, (4.0 - dc) * unit - cy,
    ]
    sides = v + w + forces + gaps + bounds
    products = (
        [v[i] * w[i] for i in range(7)]
        + [forces[i] * gaps[i] for i in range(8)]
        + [
            forces[first] * forces[second]
            for first in range(8) for second in range(first + 1, 8)
        ]
    )
    return sides, products, split_equalities, w


def _push_t(suite: str) -> tuple[Benchmark, tuple]:
    retained = suite == "crisp"
    horizon = 49 if retained else 50
    nu = 17 if retained else 24
    name = f"contact_{suite}_push_t"
    model, x, u, goal = _model(name, 3, nu, 3)
    dt = 0.05
    lambda_start = 9 if retained else 2
    lambdas = [u[lambda_start + i] for i in range(8)]
    odd_sum = sum(lambdas[0::2])
    even_sum = sum(lambdas[1::2])
    mass = 1.0 if retained else 0.1
    inverse_drag = 1.0 / (0.4 * mass * 9.8)
    cosine, sine = ca.cos(x[2]), ca.sin(x[2])
    model.disc_dyn_expr = ca.vertcat(
        x[0] + dt * inverse_drag * (cosine * even_sum - sine * odd_sum),
        x[1] + dt * inverse_drag * (sine * even_sum + cosine * odd_sum),
        x[2] + dt * inverse_drag / (0.4 * 2.8 * 0.05)
        * (-u[1] * even_sum + u[0] * odd_sum),
    )
    if retained:
        angle_cost = 2.0 - 2.0 * ca.cos(x[2] - goal[2])
        pose_cost = (x[0] - goal[0]) ** 2 + (x[1] - goal[1]) ** 2 + angle_cost
        model.cost_expr_ext_cost = pose_cost + 0.01 * ca.sumsqr(u[9:17])
        model.cost_expr_ext_cost_e = 100.0 * pose_cost
    else:
        model.cost_expr_ext_cost = 0.01 * ca.sumsqr(u)
        model.cost_expr_ext_cost_e = 100.0 * ca.sumsqr(x - goal)
    sides, products, equalities, reconstructed_w = _push_t_terms(u, retained)
    bounds = _set_constraints(model, sides, products, equalities)

    def trajectory(X: np.ndarray, U: np.ndarray) -> np.ndarray:
        if not retained:
            controls = np.vstack([U, np.zeros((1, 24))])
            return np.hstack([X, controls])
        rows = np.zeros((X.shape[0], 29))
        rows[:, :3] = X
        for k in range(horizon):
            control = U[k]
            cx, cy = control[:2]
            coordinates = np.array([
                cx - 0.10, cy - (4.0 - 2.6429) * 0.05,
                cy - (3.0 - 2.6429) * 0.05, cx - 0.025,
                cy + 2.6429 * 0.05, cx + 0.025, cx + 0.10,
            ])
            v = control[2:9]
            w = v - coordinates
            rows[k, 3:5] = control[:2]
            rows[k, 5:19:2] = v
            rows[k, 6:19:2] = w
            rows[k, 19:27] = control[9:17]
        coordinates = np.array([
            -0.10, -(4.0 - 2.6429) * 0.05,
            -(3.0 - 2.6429) * 0.05, -0.025,
            2.6429 * 0.05, 0.025, 0.10,
        ])
        terminal_v = np.maximum(coordinates, 0.0)
        rows[-1, 5:19:2] = terminal_v
        rows[-1, 6:19:2] = terminal_v - coordinates
        rows[:, 27] = np.cos(X[:, 2])
        rows[:, 28] = np.sin(X[:, 2])
        return rows

    default_goal = (
        np.array([0.01, 0.01, 0.01])
        if retained else np.array([0.0, 0.0, 0.0])
    )
    benchmark = Benchmark(
        suite, "push_t", model, horizon, dt, np.zeros(3),
        default_goal, "zeros" if retained else "linear", trajectory,
    )
    return benchmark, bounds


def _impact_cart_transport() -> tuple[Benchmark, tuple]:
    name = "contact_impact_cart_transport"
    model, x, u, goal = _model(name, 4, 4, 4)
    dt = 0.02
    friction, active, positive, negative = (u[i] for i in range(4))
    model.disc_dyn_expr = ca.vertcat(
        x[0] + dt * x[2],
        x[1] + dt * x[3],
        x[2] + dt * friction / 0.1,
        x[3] + dt * (active - friction) / 0.2,
    )
    model.cost_expr_ext_cost = 1e-6 * ca.sumsqr(u)
    model.cost_expr_ext_cost_e = 5000.0 * ca.sumsqr(x - goal)
    limit = 0.2 * 0.1 * 9.81
    sides = [
        positive, negative, limit - friction, friction + limit,
        1.0 - (x[1] - x[0]), 1.0 + (x[1] - x[0]),
    ]
    equalities = [x[2] - x[3] - positive + negative]
    products = [
        positive * negative,
        negative * (limit - friction),
        negative * (friction + limit),
    ]
    bounds = _set_constraints(
        model, sides, products, equalities,
        terminal_sides=[
            1.0 - (x[1] - x[0]), 1.0 + (x[1] - x[0])
        ],
    )

    def trajectory(X: np.ndarray, U: np.ndarray) -> np.ndarray:
        controls = np.vstack([U, np.zeros((1, 4))])
        return np.hstack([X, controls])

    benchmark = Benchmark(
        "impact", "cart_transport", model, 300, dt, np.zeros(4),
        np.array([1.0, 0.0, 0.0, 0.0]), "repeat", trajectory,
    )
    return benchmark, bounds


def make_benchmark(suite: str, problem: str) -> tuple[Benchmark, tuple]:
    if suite == "crisp":
        if problem == "cartpole_soft_walls":
            return _crisp_cartpole()
        if problem == "push_box":
            return _push_box(
                "crisp", 99, 0.02, 1.0, 9.8, 0.5, 0.25, 0.4,
                False, "zeros",
            )
        if problem == "transport":
            return _crisp_transport()
        if problem == "push_t":
            return _push_t("crisp")
    elif suite == "impact":
        if problem == "push_box":
            return _push_box(
                "impact", 50, 0.05, 0.1, 9.81, 0.3, 0.4, 0.5,
                True, "linear",
            )
        if problem == "push_t":
            return _push_t("impact")
        if problem == "cart_transport":
            return _impact_cart_transport()
    raise ValueError(f"unsupported benchmark: {suite}/{problem}")


def make_ocp(benchmark: Benchmark, bounds: tuple) -> AcadosOcp:
    lh, uh, lh_e, uh_e = bounds
    ocp = AcadosOcp()
    ocp.model = benchmark.model
    ocp.code_gen_opts.code_export_directory = str(benchmark.directory)
    ocp.code_gen_opts.json_file = str(benchmark.json_path)
    ocp.solver_options.N_horizon = benchmark.horizon
    ocp.solver_options.tf = benchmark.horizon * benchmark.dt
    ocp.solver_options.cost_scaling = np.ones(benchmark.horizon + 1)
    ocp.solver_options.integrator_type = "DISCRETE"
    ocp.solver_options.nlp_solver_type = "SQP"
    ocp.solver_options.qp_solver = "PARTIAL_CONDENSING_HPIPM"
    ocp.solver_options.hessian_approx = "GAUSS_NEWTON"
    ocp.solver_options.regularize_method = "CONVEXIFY"
    ocp.solver_options.globalization = "MERIT_BACKTRACKING"
    ocp.solver_options.nlp_solver_max_iter = 300
    ocp.solver_options.qp_solver_iter_max = 200
    ocp.solver_options.tol_stat = 1e-3
    ocp.solver_options.tol_eq = 1e-6
    ocp.solver_options.tol_ineq = 1e-8
    ocp.solver_options.tol_comp = 1e-6
    ocp.solver_options.print_level = 0
    ocp.cost.cost_type = "EXTERNAL"
    ocp.cost.cost_type_e = "EXTERNAL"
    ocp.constraints.x0 = benchmark.initial_state
    ocp.constraints.lh = lh
    ocp.constraints.uh = uh
    ocp.constraints.lh_0 = lh
    ocp.constraints.uh_0 = uh
    if lh_e.size:
        ocp.constraints.lh_e = lh_e
        ocp.constraints.uh_e = uh_e
    ocp.parameter_values = benchmark.default_goal
    return ocp


def generate(benchmark: Benchmark, bounds: tuple) -> None:
    benchmark.directory.mkdir(parents=True, exist_ok=True)
    ocp = make_ocp(benchmark, bounds)
    AcadosOcpSolver.generate(ocp, str(benchmark.json_path), verbose=True)
    print(benchmark.directory)


def prepare_linux_json(benchmark: Benchmark) -> None:
    document = json.loads(benchmark.json_path.read_text(encoding="utf-8"))
    acados_source = Path(os.environ["ACADOS_SOURCE_DIR"]).resolve()
    document["code_gen_opts"]["acados_include_path"] = str(
        acados_source / "include"
    )
    document["code_gen_opts"]["acados_lib_path"] = str(acados_source / "lib")
    document["code_gen_opts"]["code_export_directory"] = str(
        benchmark.directory
    )
    document["code_gen_opts"]["json_file"] = str(benchmark.json_path)
    document["code_gen_opts"]["os"] = "linux"
    document["code_gen_opts"]["shared_lib_ext"] = ".so"
    benchmark.json_path.write_text(
        json.dumps(document, indent=4) + "\n", encoding="utf-8"
    )


def _initial_guess(
    benchmark: Benchmark, start: np.ndarray, goal: np.ndarray
) -> tuple[np.ndarray, np.ndarray]:
    X = np.zeros((benchmark.horizon + 1, benchmark.model.x.rows()))
    U = np.zeros((benchmark.horizon, benchmark.model.u.rows()))
    if benchmark.initialization == "crisp_file":
        source = (
            ROOT / "benchmarks" / "CRISP" / "src" / "examples"
            / "pushbot" / "initial_guess_pushbot_example.txt"
        )
        guess = np.loadtxt(source).reshape(benchmark.horizon + 1, -1)
        X[:] = guess[:, : X.shape[1]]
        U[:] = guess[:-1, X.shape[1] : X.shape[1] + U.shape[1]]
    elif benchmark.initialization == "linear":
        for k, fraction in enumerate(np.linspace(0.0, 1.0, len(X))):
            X[k] = (1.0 - fraction) * start + fraction * goal
    elif benchmark.initialization == "repeat":
        X[:] = start
    elif benchmark.initialization != "zeros":
        raise ValueError(benchmark.initialization)
    X[0] = start
    return X, U


def solve(
    benchmark: Benchmark,
    start: np.ndarray,
    goal: np.ndarray,
    trajectory_path: Path,
    quiet: bool = False,
) -> dict:
    solver = AcadosOcpSolver(
        None,
        json_file=str(benchmark.json_path),
        generate=False,
        build=False,
        verbose=False,
    )
    solver.constraints_set(0, "lbx", start)
    solver.constraints_set(0, "ubx", start)
    X0, U0 = _initial_guess(benchmark, start, goal)
    for k in range(benchmark.horizon):
        solver.set(k, "x", X0[k])
        solver.set(k, "u", U0[k])
        solver.set(k, "p", goal)
    solver.set(benchmark.horizon, "x", X0[-1])
    solver.set(benchmark.horizon, "p", goal)
    status = solver.solve()
    X = np.vstack([
        solver.get(k, "x") for k in range(benchmark.horizon + 1)
    ])
    U = np.vstack([
        solver.get(k, "u") for k in range(benchmark.horizon)
    ])
    trajectory_path.parent.mkdir(parents=True, exist_ok=True)
    np.savetxt(trajectory_path, benchmark.trajectory(X, U), fmt="%.17g")
    objective = float(solver.get_cost())
    solver_seconds = float(solver.get_stats("time_tot"))
    iterations = int(solver.get_stats("sqp_iter"))
    residuals = np.asarray(solver.get_residuals()).reshape(-1)
    result = {
        "status": int(status),
        "reported_success": status == 0,
        "solver_seconds": solver_seconds,
        "iterations": iterations,
        "objective": objective,
        "residuals": residuals.tolist(),
        "trajectory": str(trajectory_path),
    }
    if not quiet:
        print(f"Status:              {status}")
        print(f"Reported success:    {'yes' if status == 0 else 'no'}")
        print(f"Solve time:          {solver_seconds:.9f} s")
        print(f"Iterations:          {iterations}")
        print(f"Objective:           {objective:.12e}")
        print("Residuals:           [" + ", ".join(
            f"{value:.3e}" for value in residuals
        ) + "]")
    return result


def _default_trajectory(benchmark: Benchmark, segment: int | None) -> Path:
    directory = Path(
        os.environ.get("ACADOS_BENCHMARK_TRAJECTORY_DIR", os.getcwd())
    )
    suffix = (
        f"_seg_{segment:02d}" if benchmark.problem == "push_t"
        and benchmark.suite == "crisp" and segment is not None else ""
    )
    return directory / f"acados_{benchmark.problem}{suffix}.txt"


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("action", choices=("generate", "prepare-linux", "solve"))
    parser.add_argument("--suite", choices=("crisp", "impact"), required=True)
    parser.add_argument("--problem", required=True)
    parser.add_argument("--start", nargs="+", type=float)
    parser.add_argument("--goal", nargs="+", type=float)
    parser.add_argument("--segment", type=int)
    parser.add_argument("--trajectory", type=Path)
    args = parser.parse_args()
    benchmark, bounds = make_benchmark(args.suite, args.problem)
    if args.action == "generate":
        generate(benchmark, bounds)
        return 0
    if args.action == "prepare-linux":
        prepare_linux_json(benchmark)
        return 0
    if args.start is None:
        if args.suite == "crisp" and args.problem == "push_t":
            if args.segment is None or not 0 <= args.segment < 50:
                parser.error("CRISP Push T solve requires --segment in [0, 49]")
            angle = 2.0 * math.pi * args.segment / 50.0
            radius = 0.25 + 0.25 * args.segment / 49.0
            start = np.array([
                radius * math.cos(angle), radius * math.sin(angle), angle
            ])
        else:
            start = benchmark.initial_state
    else:
        start = np.asarray(args.start)
    goal = (
        benchmark.default_goal if args.goal is None else np.asarray(args.goal)
    )
    if len(start) != benchmark.model.x.rows() or len(goal) != benchmark.model.p.rows():
        parser.error("start/goal dimension does not match the selected problem")
    trajectory = args.trajectory or _default_trajectory(
        benchmark, args.segment
    )
    result = solve(benchmark, start, goal, trajectory)
    return 0 if result["reported_success"] else 1


if __name__ == "__main__":
    raise SystemExit(main())
