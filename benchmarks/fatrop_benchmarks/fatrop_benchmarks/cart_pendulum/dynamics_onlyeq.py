import rockit
import casadi as cs
import numpy as np
from benchmark_tools import SolveFatrop, SolveIpopt


class CartPendulumDynamics:
    def __init__(self, ocp: rockit.Ocp):
        # Physical constants
        self.g = 9.82      # gravitation [m/s^2]
        self.L = 1.0       # pendulum length [m]
        self.m = 1.0       # pendulum mass [kg]
        self.I = self.m*self.L**2/12  # pendulum moment of inertia
        self.m_cart = 0.5  # cart mass [kg]

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

class CartPendulumOnlyEq(CartPendulumDynamics):
    def __init__(self, T = 2.5):
        self.ocp = rockit.Ocp(T=T)
        CartPendulumDynamics.__init__(self, self.ocp)
        # over write initial
        self.ocp.set_initial(self.theta, np.pi)
        # add objective
        alpha = 1e0 
        self.ocp.add_objective(self.ocp.sum(alpha*self.Fex**2, include_last=False))

        # add boundary value constraints
        self.ocp.subject_to(self.ocp.at_t0(self.x == 0))
        self.ocp.subject_to(self.ocp.at_t0(self.dx == 0))
        self.ocp.subject_to(self.ocp.at_t0(self.theta == np.pi))
        # disturbance
        self.ocp.subject_to(self.ocp.at_t0(self.omega == 1.0))

        self.ocp.subject_to(self.ocp.at_tf(self.x == 0))
        self.ocp.subject_to(self.ocp.at_tf(self.dx == 0))
        self.ocp.subject_to(self.ocp.at_tf(self.theta == np.pi))
        self.ocp.subject_to(self.ocp.at_tf(self.omega == 0.0))


if __name__ == '__main__':
    # SolveIpopt(CartPendulumOnlyEq(), 100, jit = False)
    SolveFatrop(CartPendulumOnlyEq(), 100)
