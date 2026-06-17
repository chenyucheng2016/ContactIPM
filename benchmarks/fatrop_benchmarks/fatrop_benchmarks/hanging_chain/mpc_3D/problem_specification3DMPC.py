# import dynamics.py from parent directory
from benchmark_tools import SolveFatrop, SolveIpopt, SolveAcados
from hanging_chain.problem_specificationMPC import HangingChainMPC
import numpy as np
import casadi as cs
import rockit


class HangingChain3DMPC(HangingChainMPC):
    def __init__(self, T = 2.0, no_masses = 6):
        HangingChainMPC.__init__(self, T, no_masses, 3)
if __name__ == "__main__":
    SolveFatrop(HangingChain3DMPC(), 25)
    SolveAcados(HangingChain3DMPC(), 25)
    # SolveIpopt(HangingChain3DMPC(), 25)