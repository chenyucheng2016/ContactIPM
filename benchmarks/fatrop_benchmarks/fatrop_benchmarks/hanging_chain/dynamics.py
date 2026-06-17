import rockit
import casadi as cs
import numpy as np

def _sqrt(val):
    return cs.if_else(val > 0, cs.sqrt(val), 0.0)


def sum_of_squares(p):
    return cs.sum1(p**2)


def dist(p1, p2):
    return _sqrt(sum_of_squares(p2-p1))

class HangingChainDynamics:
    def __init__(self, ocp: rockit.Ocp, no_masses = 6, dim =2):
        # problem parameters
        D = 1.6
        L = 0.0055
        m = 0.03
        g = 9.81
        self.ground = np.zeros((dim, 1))
        # OCP
        self.p = [ocp.state(dim) for i in range(no_masses+1)]
        self.v = [ocp.state(dim) for i in range(no_masses)]
        self.u = ocp.control(dim)

        # dynamics
        p_all = cs.horzcat(self.ground, *self.p)
        F = []
        for i in range(no_masses+1):
            xim1 = p_all[:, i]
            xi = p_all[:, i+1]
            Fim1_i = D*(1-L/dist(xim1, xi))*(xi-xim1)
            F.append(Fim1_i)
        for i in range(no_masses):
            Fi_ip1 = F[i+1]
            Fim1_i = F[i]
            Ftot = Fi_ip1 - Fim1_i + m*(cs.DM([[0], [-g], [0]]) if dim == 3 else cs.DM([[0], [-g]]))
            ocp.set_der(self.v[i], 1.0/m*Ftot)
            ocp.set_der(self.p[i], self.v[i])
        ocp.set_der(self.p[-1], self.u)
        ocp.subject_to(-1. <(self.u < 1.), include_first=True, include_last=False)

if __name__ == '__main__':
    pass
