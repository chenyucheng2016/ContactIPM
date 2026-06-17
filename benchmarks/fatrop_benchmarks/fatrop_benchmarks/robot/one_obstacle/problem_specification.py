# import dynamics.py from parent directory
import sys
import os
from robot.dynamics import RobotDynamics 
from benchmark_tools import SolveFatrop, SolveIpopt
import numpy as np
import casadi as cs
import rockit


class RobotObs(RobotDynamics):
    def __init__(self):
        self.ocp = rockit.Ocp(T=rockit.FreeTime(1.))
        RobotDynamics.__init__(self, self.ocp)
        # parameters 
        n_joints = 7
        joinst_pos0 = np.array([0.0] + 2*[0.0] + [-0.40] + [0.00] + [3.0] + [0.0])
        target_pos = np.array([0.50, 0.00, 0.30])
        obstable_pos = np.array([0.5, 0.00, 0.6]) + np.array([0.00, 0.00, 0.00])
        obstacle_radius = np.array([0.20])
        joint0_p = self.ocp.parameter(n_joints)
        target_pos_p = self.ocp.parameter(3)
        self.ocp.set_value(joint0_p, joinst_pos0)
        self.ocp.set_value(target_pos_p, target_pos)
        # no collision constraints
        self.capsules = self.frankachain.GetAllCapsuleConstraints(
            obstable_pos, obstacle_radius, sqrt=True)
        n_capsules = self.capsules.shape[0]
        self.p = self.ocp.control(n_capsules)
        self.ocp.subject_to((self.capsules + self.p) > 0, include_last=False)
        self.ocp.subject_to(self.p>0, include_last=False)
        self.ocp.add_objective(self.ocp.sum(1e2*cs.sum1(self.p), include_last=False))
        # boundary constraints
        self.ocp.subject_to(self.ocp.at_t0(self.joints_theta == joint0_p))
        self.ocp.subject_to(self.ocp.at_tf(self.frankachain.GetFrameExpression(
            'TCP_frame')[:3, 3] == target_pos_p))
        # minimal time objective
        self.ocp.add_objective(self.ocp.at_tf(self.ocp.T))


if __name__ == "__main__":
    res, prob, _ = SolveFatrop(RobotObs(), 50)
    print("max constraint violation ", res.sample(prob.capsules, grid = "control")[1].min())
    SolveIpopt(RobotObs(), 50)