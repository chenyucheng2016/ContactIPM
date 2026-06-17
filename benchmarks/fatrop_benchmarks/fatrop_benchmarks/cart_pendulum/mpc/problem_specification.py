# import dynamics.py from parent directory
import sys
import os
sys.path.append(os.path.join(os.path.dirname(__file__), os.pardir))
from cart_pendulum.dynamics import *
import numpy as np
import casadi as cs
import rockit


class CartPendulumMPC(CartPendulumDynamics):
    def __init__(self, T = 2.5):
        self.ocp = rockit.Ocp(T=T)
        CartPendulumDynamics.__init__(self, self.ocp)
        # over write initial
        self.ocp.set_initial(self.theta, np.pi)
        # add objective
        alpha = 1e0 
        beta =  1e1
        gamma = 1e1 
        delta = 1e1 
        pospen = 1e-2 
        self.ocp.add_objective(self.ocp.sum(alpha*self.Fex**2, include_last=False))
        self.ocp.add_objective(self.ocp.sum(beta*cs.sum1(self.omega**2), include_last=True))
        self.ocp.add_objective(self.ocp.sum(gamma*cs.sum1(self.dx**2), include_last = True))
        # self.ocp.add_objective(self.ocp.sum(-delta*self.COG[1], include_last=True))
        self.ocp.add_objective(self.ocp.sum(pospen*self.x**2, include_last=True))
        # add boundary value constraints
        self.ocp.subject_to(self.ocp.at_t0(self.x == 0))
        self.ocp.subject_to(self.ocp.at_t0(self.dx == 0))
        self.ocp.subject_to(self.ocp.at_t0(self.theta == np.pi))
        # disturbance
        self.ocp.subject_to(self.ocp.at_t0(self.omega == 1.0))
        


from benchmark_tools import SolveFatrop, SolveIpopt, SolveAcados
if __name__ == "__main__":
    res, prob, stats = SolveFatrop(CartPendulumMPC(), 50)
    res2, prob2, stats = SolveAcados(CartPendulumMPC(), 50)
    print(res.sample(prob.Fex, grid = 'control')[1])
    print(res2.sample(prob2.Fex, grid = 'control')[1])
    # print(res.sample(prob.x, grid = 'control')[1])
    # print(res.sample(prob.dx, grid = 'control')[1])
    # print(res.sample(prob.theta, grid = 'control')[1])
    # print(res.sample(prob.omega, grid = 'control')[1])
    SolveIpopt(CartPendulumMPC(), 25)