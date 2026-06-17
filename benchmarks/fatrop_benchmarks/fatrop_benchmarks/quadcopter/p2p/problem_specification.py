# import dynamics.py from parent directory
import sys
import os
from quadcopter.dynamics import QuadcopterDynamics 
from benchmark_tools import SolveFatrop, SolveIpopt
import numpy as np
import casadi as cs
import rockit


class QuadCopterP2P(QuadcopterDynamics):
    def __init__(self):
        self.ocp = rockit.Ocp(T=rockit.FreeTime(1.))
        QuadcopterDynamics.__init__(self, self.ocp)
        # parameters
        # Initial & terminal conditions
        p0 = cs.vertcat(0., 0., 2.5)
        v0 = cs.vertcat(0, 0, 0)
        eul0 = cs.vertcat(0, 0, 0)

        pf = cs.vertcat(0.01, 5., 2.5)
        vf = cs.vertcat(0., 0., 0.)
        eulf = cs.vertcat(0., 0., 0.)
        # add objective

        self.ocp.add_objective(self.ocp.sum(1e2*(self.psi- eul0[2])**2, include_last = True))
        self.ocp.add_objective(self.ocp.at_tf(self.ocp.T))
        # add boundary value constraints
        self.ocp.subject_to(self.ocp.at_t0(self.p) == p0)
        self.ocp.subject_to(self.ocp.at_t0(self.v) == v0)
        self.ocp.subject_to(self.ocp.at_t0(self.at) == 9.81)
        self.ocp.subject_to(self.ocp.at_t0(self.eul) == eul0)


        self.ocp.subject_to(self.ocp.at_tf(self.p) == pf)
        self.ocp.subject_to(self.ocp.at_tf(self.v) == vf)
        self.ocp.subject_to(self.ocp.at_tf(self.eul[[0,1]]) == eulf[[0,1]])
        # self.ocp.subject_to(self.psi  ==0, include_first=False)

if __name__ == "__main__":
    SolveFatrop(QuadCopterP2P(), 25)
    SolveIpopt(QuadCopterP2P(), 25)