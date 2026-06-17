# import dynamics.py from parent directory
import sys
import os
from benchmark_tools import SolveFatrop, SolveIpopt, SimulateOneStep, SolveAcados
from hanging_chain.dynamics import HangingChainDynamics, sum_of_squares
import numpy as np
import casadi as cs
import rockit


class HangingChainMPC(HangingChainDynamics):
    def __init__(self, T = 2.0, no_masses = 6, dim =2):
        self.ocp = rockit.Ocp(T=T)
        HangingChainDynamics.__init__(self, self.ocp, no_masses, dim)
        # parameters
        x_end = np.array([[1.0], [0.0], [0.0]]) if dim == 3 else np.array([[1.0], [0.0]])
        alpha = 25.0
        beta = 1.0
        gamma = 0.01
        # Initial conditions
        x0 = []
        init = np.linspace(self.ground, x_end, no_masses+2)
        for i in range(no_masses+1):
            # self.ocp.subject_to(self.ocp.at_t0(self.p[i] == init[i+1]))
            x0.append((self.p[i], init[i+1]))
        for i in range(no_masses):
            # self.ocp.subject_to(self.ocp.at_t0(self.v[i] == np.zeros((dim,1))))
            x0.append((self.v[i], np.zeros((dim,1))))
        # self.ocp.subject_to(self.ocp.at_t0(self.u ==  3*cs.DM([-.5, .5, .5] if dim == 3 else [-.5, .5])))
        u0 = 3*cs.DM([-.5, .5, .5] if dim == 3 else [-.5, .5])
        x0 = SimulateOneStep(self.ocp, x0, u0, 3, 0.05*3, "rk")
        self.ocp.subject_to(self.ocp.at_t0(self.ocp.x)== x0)
        # objective
        obj = alpha*sum_of_squares(self.p[-1] - x_end)
        for i in range(no_masses):
            obj += beta*sum_of_squares(self.v[i])
        obj += gamma*sum_of_squares(self.u)
        self.ocp.add_objective(self.ocp.sum(obj))
        # initial guess
        for i in range(no_masses):
            self.ocp.set_initial(self.p[i], init[i+1])

if __name__ == "__main__":
    res, prob, _ = SolveFatrop(HangingChainMPC(), 40)
    res2, prob2, _ = SolveAcados(HangingChainMPC(), 40)
    print(res.sample(prob.u, grid = 'control')[1])
    print(res2.sample(prob2.u, grid = 'control')[1])
    SolveIpopt(HangingChainMPC(), 40)