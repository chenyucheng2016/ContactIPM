# import dynamics.py from parent directory
from benchmark_tools import SolveFatrop, SolveIpopt, SolveAcados
from hanging_chain.problem_specificationMPC import HangingChainMPC
import numpy as np
import casadi as cs
import rockit


class HangingChain2DMPC(HangingChainMPC):
    def __init__(self, T = 4.0, no_masses = 6):
        self.no_masses = no_masses
        HangingChainMPC.__init__(self, T, no_masses, 2)
if __name__ == "__main__":
    SolveFatrop(HangingChain2DMPC(), 40)
    SolveAcados(HangingChain2DMPC(), 40)
    SolveIpopt(HangingChain2DMPC(), 40)