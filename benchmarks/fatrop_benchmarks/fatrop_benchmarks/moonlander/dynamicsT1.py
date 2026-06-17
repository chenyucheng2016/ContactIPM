import rockit
import casadi as cs
import numpy as np
def transf(theta, p):
    return cs.vertcat(
        cs.horzcat(cs.cos(theta), -cs.sin(theta), p[0]),
                     cs.horzcat(cs.sin(theta), cs.cos(theta), p[1]),
                     cs.horzcat(0.0, 0.0, 1.))
class MoonLanderDynamics:
    def __init__(self, ocp:rockit.Ocp):
        # define the problems parameters
        m = 1.0
        g = 9.81
        I = 0.1
        D = 1.0
        max_thrust = 2*g 

        # Create an OCP object

        # Define the states
        self.p = ocp.state(2)
        self.dp = ocp.state(2)
        self.theta = ocp.state()
        self.dtheta = ocp.state()

        # Define the controls
        self.F1 = ocp.control()
        self.F2 = ocp.control()

        # compute the tranformation of the local frame attached to the rockt
        self.F_r = transf(self.theta, self.p)

        # Define the dynamics of the system in continuous time
        self.T = ocp.state()
        ocp.subject_to(ocp.at_t0(self.T)>0)
        ocp.set_der(self.T, 0.)
        ocp.set_der(self.p, self.dp * self.T)
        ocp.set_der(self.theta, self.dtheta * self.T)
        self.F_tot = (self.F_r @ cs.vertcat(0, self.F1 + self.F2, 1)) [:2]
        ocp.set_der(self.dp, (1/m * self.F_tot + cs.vertcat(0, -g))*self.T)
        ocp.set_der(self.dtheta, (1/I * D/2 * (self.F2 - self.F1))*self.T)

        # Define the path constraints
        ocp.subject_to((0<self.F1)<max_thrust, include_last=False)
        ocp.subject_to((0<self.F2)<max_thrust, include_last=False)


        # provide an initial guess
        ocp.set_initial(self.F1, 5.)
        ocp.set_initial(self.F2, 5.)