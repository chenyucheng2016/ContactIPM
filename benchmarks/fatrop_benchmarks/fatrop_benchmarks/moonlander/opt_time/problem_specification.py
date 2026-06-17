from benchmark_tools import SolveFatrop, SolveIpopt
from moonlander.dynamics import MoonLanderDynamics
import casadi as cs
import rockit
import numpy as np

class MoonLanderOptTime(MoonLanderDynamics):
    def __init__(self):
        self.ocp = rockit.Ocp(T=rockit.FreeTime(5.))
        MoonLanderDynamics.__init__(self, self.ocp)
        target = np.array([5., 5.]) 
        # Define the initial conditions
        self.ocp.subject_to(self.ocp.at_t0(self.p) == [0, 0])
        self.ocp.subject_to(self.ocp.at_t0(self.dp) == [0, 0])
        self.ocp.subject_to(self.ocp.at_t0(self.theta) == 0)
        self.ocp.subject_to(self.ocp.at_t0(self.dtheta) == 0)

        # Define the end conditions
        self.ocp.subject_to(self.ocp.at_tf(self.p) == target)
        self.ocp.subject_to(self.ocp.at_tf(self.dp) == [0, 0])

        # define the objective
        self.ocp.add_objective(self.ocp.at_tf(self.ocp.T))

if __name__ == "__main__":
    SolveFatrop(MoonLanderOptTime(), 50)
    SolveIpopt(MoonLanderOptTime(), 50)