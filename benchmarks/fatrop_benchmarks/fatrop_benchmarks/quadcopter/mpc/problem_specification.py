# import dynamics.py from parent directory
import sys
import os
from quadcopter.dynamics import QuadcopterDynamics 
from benchmark_tools import SolveFatrop, SolveIpopt, SolveAcados
import numpy as np
import casadi as cs
import rockit


class QuadCopterMPC(QuadcopterDynamics):
    def __init__(self):
        self.ocp = rockit.Ocp(T=2.0)
        QuadcopterDynamics.__init__(self, self.ocp)
        # parameters
        # Initial & terminal conditions
        p0 = cs.vertcat(0., 0., 2.5)
        v0 = cs.vertcat(1., 1., 1.)
        eul0 = cs.vertcat(np.pi/10, np.pi/10, np.pi/10)

        # add objective
        alpha = 1.
        beta = 1.
        gamma = 100.
        delta = 10.
        # self.ocp.add_objective(self.ocp.sum(1e2*(self.psi- eul0[2])**2, include_last = True))
        self.ocp.add_objective(self.ocp.sum(1e3*self.psi**2 + alpha* cs.sum1(self.v**2) + beta*cs.sum1(self.ocp.der(self.eul)**2)  + gamma*cs.sum1((self.p-p0)**2)+ delta*cs.sum1(self.eul**2), include_last=False))
        reg = 1e-4
        self.ocp.add_objective(self.ocp.sum(reg*(self.ocp.u.T @ self.ocp.u), include_last=False))
        self.ocp.add_objective(self.ocp.sum(reg*(self.ocp.x.T @ self.ocp.x), include_last=True))
        # add boundary value constraints
        self.ocp.subject_to(self.ocp.at_t0(self.p) == p0)
        self.ocp.subject_to(self.ocp.at_t0(self.v) == v0)
        self.ocp.subject_to(self.ocp.at_t0(self.at) == 9.81)
        self.ocp.subject_to(self.ocp.at_t0(self.eul) == eul0)

if __name__ == "__main__":
    res, prob, _, = SolveFatrop(QuadCopterMPC(), 25)
    print(res.sample(prob.at, grid = 'control'))
    res, prob, _, = SolveAcados(QuadCopterMPC(), 25)
    print(res.sample(prob.at, grid = 'control'))
    SolveIpopt(QuadCopterMPC(), 25)