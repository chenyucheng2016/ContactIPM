"""
CasADi-based NMPC benchmark problems.

This package provides pure CasADi transcriptions of fixed-time NMPC problems
from the fatrop benchmark suite, enabling comparison of IPOPT, acados, and ContactIPM.
"""

from .problems import (
    CartPendulumMPC,
    QuadcopterMPC,
    QuadcopterTrackingMPC,
    HangingChain2DMPC,
    HangingChain3DMPC,
)

from .discretize import build_nlp, rk4_discretize, extract_solution, compute_cost

from .solvers import solve_ipopt, solve_acados, solve_contactipm, SolveResult

__all__ = [
    # Problems
    "CartPendulumMPC",
    "QuadcopterMPC",
    "QuadcopterTrackingMPC",
    "HangingChain2DMPC",
    "HangingChain3DMPC",
    # Discretization
    "build_nlp",
    "rk4_discretize",
    "extract_solution",
    "compute_cost",
    # Solvers
    "solve_ipopt",
    "solve_acados",
    "solve_contactipm",
    "SolveResult",
]
