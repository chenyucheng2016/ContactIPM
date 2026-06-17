import rockit
import casadi as cs
import numpy as np


class CartPendulumDynamics:
    def __init__(self, ocp: rockit.Ocp):
        # Physical constants
        self.g = 9.82      # gravitation [m/s^2]
        self.L = 1.0       # pendulum length [m]
        self.m = 1.0       # pendulum mass [kg]
        self.I = self.m*self.L**2/12  # pendulum moment of inertia
        self.m_cart = 0.5  # cart mass [kg]

        max_f = 5.
        max_x = 1.
        max_v = 2.
        self.Fex = ocp.control()
        self.x = ocp.state()
        self.dx = ocp.state()
        self.theta = ocp.state()
        self.omega= ocp.state()
        # alpha = ocp.state()
        ddx = cs.MX.sym('ddx')
        self.COG = self.L/2 * cs.vertcat(cs.sin(self.theta), -
                               cs.cos(self.theta)) + cs.vertcat(self.x, 0)
        Endp = self.L * cs.vertcat(cs.sin(self.theta), -cs.cos(self.theta)) + cs.vertcat(self.x, 0)
        alpha = 1.0/(self.I+.25*self.m*self.L**2) * .5*self.L*self.m * \
            (-ddx*cs.cos(self.theta) - self.g*cs.sin(self.theta))
        ddCOG = cs.jacobian(cs.jacobian(self.COG, self.theta), self.theta)*self.omega*2 + cs.jacobian(self.COG,
                                                                                   self.theta)*alpha + cs.vertcat(ddx, 0)
        FXFY = self.m*ddCOG + cs.vertcat(0, self.m*self.g)
        eq = -FXFY[0] + self.Fex - self.m_cart*ddx
        J, c = cs.linear_coeff(eq, ddx)
        ddx_ = -1.0/J[0]*c
        alpha_ = cs.substitute(alpha, ddx, ddx_)
        ocp.set_der(self.x, self.dx)
        ocp.set_der(self.dx, ddx_)
        ocp.set_der(self.theta, self.omega)
        ocp.set_der(self.omega, alpha_)
        ocp.subject_to(-max_f < (self.Fex < max_f), include_last=False)
        ocp.subject_to(-max_x < (self.x < max_x), include_last=True, include_first=False)
        ocp.subject_to(-max_v < (self.dx < max_v), include_last=True, include_first=False)


if __name__ == '__main__':
    pass
