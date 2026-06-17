# import dynamics.py from parent directory
import sys
import os
from benchmark_tools import SolveFatrop, SolveIpopt
import numpy as np
from numpy import pi
import casadi as cs
import rockit
from truck_trailer.dynamicsT1 import TruckTrailerDynamics


class TruckTrailerTime(TruckTrailerDynamics):
    def __init__(self):
        self.ocp = rockit.Ocp(T=1.)
        TruckTrailerDynamics.__init__(self, self.ocp)
        self.ocp.set_initial(self.T, 20.)
        # parameters
        x2_t0 = 0.
        y2_t0 = 0.
        theta2_t0 = 0.
        theta1_t0 = 0.
        theta0_t0 = 0.
        x2_tf = 0.
        y2_tf = -2.
        theta2_tf = 2*pi/4
        theta1_tf = 2*pi/4
        theta0_tf = 2*pi/4
        # Initial constraints
        self.ocp.subject_to(self.ocp.at_t0(self.x2) == x2_t0)
        self.ocp.subject_to(self.ocp.at_t0(self.y2) == y2_t0)
        self.ocp.subject_to(self.ocp.at_t0(self.theta2) == theta2_t0)
        self.ocp.subject_to(self.ocp.at_t0(self.theta1) == theta1_t0)
        self.ocp.subject_to(self.ocp.at_t0(self.theta0) == theta0_t0)
        # Final constraint
        self.ocp.subject_to(self.ocp.at_tf(self.x2) == x2_tf)
        self.ocp.subject_to(self.ocp.at_tf(self.y2) == y2_tf)
        self.ocp.subject_to(self.ocp.at_tf(self.theta2) == 2*pi/4)
        self.ocp.subject_to(self.ocp.at_tf(self.beta12) == theta1_tf - theta2_tf)
        self.ocp.subject_to(self.ocp.at_tf(self.beta01) == theta0_tf - theta1_tf)

        # objective
        self.ocp.add_objective(self.ocp.at_tf(self.T))
        self.ocp.add_objective(self.ocp.sum(self.beta01**2, include_last=False))
        self.ocp.add_objective(self.ocp.sum(self.beta12**2, include_last=False))


if __name__ == "__main__":
    SolveFatrop(TruckTrailerTime(), 50)
    SolveIpopt(TruckTrailerTime(), 50, jit = False)