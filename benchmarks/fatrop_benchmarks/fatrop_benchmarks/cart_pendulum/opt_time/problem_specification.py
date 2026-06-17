# import dynamics.py from parent directory
import sys
import os
sys.path.append(os.path.join(os.path.dirname(__file__), os.pardir))
sys.path.append(os.path.join(os.path.dirname(__file__), os.pardir, os.pardir))
from cart_pendulum.dynamics import *
from benchmark_tools import SolveFatrop, SolveIpopt
import numpy as np
import casadi as cs
import rockit


class CartPendulumTime(CartPendulumDynamics):
    def __init__(self):
        self.ocp = rockit.Ocp(T=rockit.FreeTime(1.))
        CartPendulumDynamics.__init__(self, self.ocp)
        # add objective
        self.ocp.add_objective(self.ocp.at_tf(self.ocp.T))
        # add boundary value constraints
        self.ocp.subject_to(self.ocp.at_t0(self.x == 0))
        self.ocp.subject_to(self.ocp.at_t0(self.theta == 0))
        self.ocp.subject_to(self.ocp.at_t0(self.omega == 0))
        self.ocp.subject_to(self.ocp.at_tf(self.theta == np.pi))
        self.ocp.subject_to(self.ocp.at_tf(self.omega == 0))


if __name__ == "__main__":
    SolveFatrop(CartPendulumTime(), 100)
    SolveIpopt(CartPendulumTime(), 100)