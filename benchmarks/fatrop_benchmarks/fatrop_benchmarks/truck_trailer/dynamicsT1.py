import rockit
import casadi as cs
import numpy as np
import os 
import yaml
from numpy import pi, cos, sin, tan


class TruckTrailerDynamics:
    def __init__(self, ocp: rockit.Ocp):
        # find path of this file
        dir_path = os.path.dirname(os.path.realpath(__file__))
        # Parameters
        with open(dir_path + '/truck_trailer_para2.yaml', 'r') as file:
            para = yaml.safe_load(file)
        self.L0 = para['truck']['L']
        self.M0 = para['truck']['M']
        self.W0 = para['truck']['W']
        self.L1 = para['trailer1']['L']
        self.M1 = para['trailer1']['M']
        self.W1 = para['trailer1']['W']
        self.L2 = para['trailer2']['L']
        self.M2 = para['trailer2']['M']
        self.W2 = para['trailer2']['W']

        # Trailer model
        self.theta2 = ocp.state()
        self.x2     = ocp.state()
        self.y2     = ocp.state()

        self.theta1 = ocp.state()
        self.x1     = self.x2 + self.L2*cos(self.theta2) + self.M1*cos(self.theta1)
        self.y1     = self.y2 + self.L2*sin(self.theta2) + self.M1*sin(self.theta1)

        self.theta0 = ocp.state()
        self.x0     = self.x1 + self.L1*cos(self.theta1) + self.M0*cos(self.theta0)
        self.y0     = self.y1 + self.L1*sin(self.theta1) + self.M0*sin(self.theta0)

        self.delta0 = ocp.state()
        self.ddelta0 = ocp.control()
        self.v0     = ocp.state()
        self.dv0     = ocp.control()

        self.beta01 = self.theta0 - self.theta1
        self.beta12 = self.theta1 - self.theta2

        dtheta0 = self.v0/self.L0*tan(self.delta0)
        dtheta1 = self.v0/self.L1*sin(self.beta01) - self.M0/self.L1*cos(self.beta01)*dtheta0
        v1 = self.v0*cos(self.beta01) + self.M0*sin(self.beta01)*dtheta0

        dtheta2 = v1/self.L2*sin(self.beta12) - self.M1/self.L2*cos(self.beta12)*dtheta1
        v2 = v1*cos(self.beta12) + self.M1*sin(self.beta12)*dtheta1

        self.T = ocp.state()
        ocp.set_der(self.T, 0.)
        ocp.subject_to(ocp.at_t0(self.T) >= 0)

        ocp.set_der(self.theta2, dtheta2 * self.T)
        ocp.set_der(self.x2,     v2*cos(self.theta2)*self.T)
        ocp.set_der(self.y2,     v2*sin(self.theta2)*self.T)
        ocp.set_der(self.theta1, dtheta1*self.T)
        ocp.set_der(self.theta0, dtheta0*self.T)
        ocp.set_der(self.delta0, self.ddelta0*self.T)
        ocp.set_der(self.v0,     self.dv0*self.T)

        
        # Path constraints
        speedf = 1
        ocp.subject_to(-.2 * speedf <= (self.v0 <= .2* speedf), include_last=False)
        ocp.subject_to(-1 <= (self.dv0 <= 1), include_last=False)

        ocp.subject_to(-pi/6 <= (self.delta0 <= pi/6), include_last=False)
        ocp.subject_to(-pi/10<= (self.ddelta0 <= pi/10), include_last=False)

        ocp.subject_to(-pi/2 <= (self.beta01 <= pi/2), include_last=False)
        ocp.subject_to(-pi/2 <= (self.beta12 <= pi/2), include_last=False)
        # Initial guess
        ocp.set_initial(self.theta0, .1)
        ocp.set_initial(self.theta1, 0)
        ocp.set_initial(self.theta2, 0)
        ocp.set_initial(self.v0,     -.2)

if __name__ == '__main__':
    pass
