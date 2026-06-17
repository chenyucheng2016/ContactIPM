import rockit
import casadi as cs
import numpy as np


class QuadcopterDynamics:
    def __init__(self, ocp: rockit.Ocp):
        # Parameters
        atmin = 0
        atmax = 9.18*5
        tiltmax = 1.1 / 2
        dtiltmax = 6. / 2
        # Dynamics
        self.p = ocp.state(3)
        self.v = ocp.state(3)
        self.at = ocp.control(order=1)
        self.eul = ocp.control(3, order=1)
        self.phi = self.eul[0]
        self.theta = self.eul[1]
        self.psi = self.eul[2]

        cr = np.cos(self.phi)
        sr = np.sin(self.phi)
        cp = np.cos(self.theta)
        sp = np.sin(self.theta)
        cy = np.cos(self.psi)
        sy = np.sin(self.psi)
        R = cs.vertcat(cs.horzcat(cy*cp, cy*sp*sr-sy*cr, cy*sp*cr + sy*sr),
                    cs.horzcat(sy*cp, sy*sp*sr + cy*cr, sy*sp*cr - cy*sr),
                    cs.horzcat(-sp, cp*sr, cp*cr))

        at_ = R @ cs.vertcat(0, 0, self.at)
        g = cs.vertcat(0, 0, -9.81)
        a = at_ + g
        self.T = ocp.state()
        ocp.subject_to(ocp.at_t0(self.T)>0.)
        ocp.set_der(self.T,0.)

        ocp.set_der(self.p, self.v * self.T)
        ocp.set_der(self.v, a * self.T)

        ocp.subject_to(-np.pi/2 <= (self.phi <= np.pi/2), include_last=False)
        ocp.subject_to(-np.pi/2 <= (self.theta <= np.pi/2), include_last=False)

        ocp.subject_to(cs.cos(self.theta)*cs.cos(self.phi) >= cs.cos(tiltmax), include_last=False)
        ocp.subject_to(-dtiltmax <= (ocp.der(self.phi) <= dtiltmax), include_last=False)
        ocp.subject_to(-dtiltmax <= (ocp.der(self.theta) <= dtiltmax), include_last=False)
        # ocp.subject_to(self.psi == 0, include_first=False, include_last=False)
        ocp.subject_to(atmin <= (self.at <= atmax), include_last=False, include_first=False)
        # a little bit of regularization
        # ocp.add_objective(ocp.sum(1e-8*ocp.u.T @ ocp.u))
        ocp.set_initial(self.at, 10)


if __name__ == '__main__':
    pass
