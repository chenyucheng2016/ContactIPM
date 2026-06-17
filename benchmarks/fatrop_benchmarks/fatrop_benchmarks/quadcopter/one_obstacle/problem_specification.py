# import dynamics.py from parent directory
import sys
import os
from quadcopter.dynamics import QuadcopterDynamics 
from benchmark_tools import SolveFatrop, SolveIpopt
import numpy as np
import casadi as cs
import rockit

def distsq(p1, p2):
    return cs.sum1((p2-p1)**2)  
def csmax(x1, x2):
    return cs.if_else(x1>x2, x1, x2)
def sqrt_special(x):
    return cs.if_else(x>0.0, cs.sqrt(x), x)

class QuadCopterOneObs(QuadcopterDynamics):
    def __init__(self):
        self.ocp = rockit.Ocp(T=rockit.FreeTime(5.))
        QuadcopterDynamics.__init__(self, self.ocp)
        # parameters
        # Initial & terminal conditions
        p0 = cs.vertcat(0., 0., 7.5)
        v0 = cs.vertcat(0, 0, 0)
        eul0 = cs.vertcat(0, 0, 0)

        pf = cs.vertcat(10., 10., 7.5)
        vf = cs.vertcat(0, 0, 0)
        eulf = cs.vertcat(0, 0, 0)
        # add no-collision constraints
        p_obs = np.array(0.5*p0+0.5*pf)
        r_obs = 2.0 
        r_drone = 0.30
        self.softconstr = sqrt_special((distsq(self.p[:2], p_obs[:2])))-(r_obs+r_drone)
        n = self.ocp.control()
        self.ocp.add_objective(self.ocp.sum(1e2*(n)))
        self.ocp.subject_to(n>0, include_last=False)
        self.ocp.subject_to(self.softconstr+n > 0, include_last=False)

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

if __name__ == "__main__":
    res, prob, _ = SolveFatrop(QuadCopterOneObs(), 50)
    print("max constraint violation ", res.sample(prob.softconstr, grid = "control")[1].min())
    SolveIpopt(QuadCopterOneObs(), 50)