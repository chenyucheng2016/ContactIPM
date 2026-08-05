"""Generate native C callbacks and metadata from a CasADi benchmark problem."""

import os
import argparse
from pathlib import Path

import casadi as ca
import numpy as np

from .problems import (
    CartPendulumMPC,
    HangingChain2DMPC,
    HangingChain3DMPC,
    QuadcopterMPC,
    QuadcopterTrackingMPC,
)
from .solvers import _build_contactipm_callbacks


PROBLEMS = {
    "cart_pendulum": (CartPendulumMPC, 25),
    "quadcopter": (QuadcopterMPC, 25),
    "quadcopter_tracking": (QuadcopterTrackingMPC, 50),
    "hanging_chain_2d": (HangingChain2DMPC, 25),
    "hanging_chain_3d": (HangingChain3DMPC, 25),
}

FUNCTION_NAMES = {
    "f_disc": "contactipm_f_disc",
    "f_lin": "contactipm_f_lin",
    "stage_cost": "contactipm_stage_cost",
    "term_cost": "contactipm_term_cost",
    "stage_grad": "contactipm_stage_grad",
    "stage_hess": "contactipm_stage_hess",
    "term_grad": "contactipm_term_grad",
    "term_hess": "contactipm_term_hess",
    "constr_eval": "contactipm_constr_eval",
    "constr_eval_term": "contactipm_constr_eval_term",
    "constr_jac": "contactipm_constr_jac",
    "constr_jac_term": "contactipm_constr_jac_term",
    "dyn_adj_hess": "contactipm_dyn_adj_hess",
    "con_adj_hess": "contactipm_con_adj_hess",
    "con_adj_hess_term": "contactipm_con_adj_hess_term",
}


def _alias(function: ca.Function, name: str) -> ca.Function:
    inputs = [
        ca.MX.sym(f"i{index}", function.size1_in(index), function.size2_in(index))
        for index in range(function.n_in())
    ]
    outputs = [ca.densify(output) for output in function.call(inputs)]
    return ca.Function(name, inputs, outputs)


def _format_array(values) -> str:
    return ", ".join(f"{float(value):.17g}" for value in np.asarray(values).flat)


def _dummy_constraint_functions(problem):
    nx = problem.nx
    nu = problem.nu
    x = ca.MX.sym("x", nx)
    u = ca.MX.sym("u", nu)
    lam = ca.MX.sym("lam", 1)
    return {
        "constr_eval": ca.Function("dummy_eval", [x, u], [ca.MX.zeros(1)]),
        "constr_eval_term": ca.Function(
            "dummy_eval_term", [x], [ca.MX.zeros(1)]
        ),
        "constr_jac": ca.Function(
            "dummy_jac", [x, u], [ca.MX.zeros(1, nx), ca.MX.zeros(1, nu)]
        ),
        "constr_jac_term": ca.Function(
            "dummy_jac_term", [x], [ca.MX.zeros(1, nx)]
        ),
        "con_adj_hess": ca.Function(
            "dummy_adj_hess",
            [x, u, lam],
            [ca.MX.zeros(nx, nx), ca.MX.zeros(nu, nx), ca.MX.zeros(nu, nu)],
        ),
        "con_adj_hess_term": ca.Function(
            "dummy_adj_hess_term", [x, lam], [ca.MX.zeros(nx, nx)]
        ),
    }


def generate(problem_key: str, output_dir: Path) -> None:
    constructor, horizon = PROBLEMS[problem_key]
    problem = constructor(N=horizon)
    functions = _build_contactipm_callbacks(problem, return_raw=True)
    if not functions["has_constraints"]:
        functions.update(_dummy_constraint_functions(problem))

    output_dir.mkdir(parents=True, exist_ok=True)
    previous_directory = Path.cwd()
    try:
        os.chdir(output_dir)
        generator = ca.CodeGenerator(f"{problem_key}_functions.c")
        for key, generated_name in FUNCTION_NAMES.items():
            generator.add(_alias(functions[key], generated_name))
        generator.generate()
    finally:
        os.chdir(previous_directory)

    nc = max(1, problem.ng)
    x_guess = problem.x_init_guess if problem.x_init_guess is not None else problem.x0
    header = f"""#pragma once
#include <array>

namespace contactipm_native_config {{
inline constexpr char problem_name[] = "{problem.name}";
inline constexpr int nx = {problem.nx};
inline constexpr int nu = {problem.nu};
inline constexpr int nc = {nc};
inline constexpr int horizon = {problem.N};
inline constexpr double dt = {problem.T / problem.N:.17g};
inline constexpr bool has_constraints = {"true" if problem.ng else "false"};
inline constexpr std::array<double, nx> x0 = {{{_format_array(problem.x0)}}};
inline constexpr std::array<double, nx> x_guess = {{{_format_array(x_guess)}}};
inline constexpr std::array<double, nx> x_lb = {{{_format_array(problem.x_lb)}}};
inline constexpr std::array<double, nx> x_ub = {{{_format_array(problem.x_ub)}}};
inline constexpr std::array<double, nu> u_lb = {{{_format_array(problem.u_lb)}}};
inline constexpr std::array<double, nu> u_ub = {{{_format_array(problem.u_ub)}}};
}}  // namespace contactipm_native_config
"""
    (output_dir / f"{problem_key}_config.hpp").write_text(header, encoding="utf-8")


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--problem", choices=PROBLEMS, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    args = parser.parse_args()
    generate(args.problem, args.output_dir)


if __name__ == "__main__":
    main()
