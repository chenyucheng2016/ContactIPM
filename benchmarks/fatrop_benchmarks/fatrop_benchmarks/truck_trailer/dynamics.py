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
        L0 = para['truck']['L']
        M0 = para['truck']['M']
        W0 = para['truck']['W']
        L1 = para['trailer1']['L']
        M1 = para['trailer1']['M']
        W1 = para['trailer1']['W']
        L2 = para['trailer2']['L']
        M2 = para['trailer2']['M']
        W2 = para['trailer2']['W']

        # Trailer model
        self.theta2 = ocp.state()
        self.x2     = ocp.state()
        self.y2     = ocp.state()

        self.theta1 = ocp.state()
        self.x1     = self.x2 + L2*cos(self.theta2) + M1*cos(self.theta1)
        self.y1     = self.y2 + L2*sin(self.theta2) + M1*sin(self.theta1)

        self.theta0 = ocp.state()
        self.x0     = self.x1 + L1*cos(self.theta1) + M0*cos(self.theta0)
        self.y0     = self.y1 + L1*sin(self.theta1) + M0*sin(self.theta0)

        self.delta0 = ocp.control(order=1)
        self.v0     = ocp.control(order=1)

        self.beta01 = self.theta0 - self.theta1
        self.beta12 = self.theta1 - self.theta2

        dtheta0 = self.v0/L0*tan(self.delta0)
        dtheta1 = self.v0/L1*sin(self.beta01) - M0/L1*cos(self.beta01)*dtheta0
        v1 = self.v0*cos(self.beta01) + M0*sin(self.beta01)*dtheta0

        dtheta2 = v1/L2*sin(self.beta12) - M1/L2*cos(self.beta12)*dtheta1
        v2 = v1*cos(self.beta12) + M1*sin(self.beta12)*dtheta1

        ocp.set_der(self.theta2, dtheta2)
        ocp.set_der(self.x2,     v2*cos(self.theta2))
        ocp.set_der(self.y2,     v2*sin(self.theta2))
        ocp.set_der(self.theta1, dtheta1)
        ocp.set_der(self.theta0, dtheta0)

        
        # Path constraints
        speedf = 1
        ocp.subject_to(-.2 * speedf <= (self.v0 <= .2* speedf), include_last=False)
        ocp.subject_to(-1 <= (ocp.der(self.v0) <= 1), include_last=False)

        ocp.subject_to(-pi/6 <= (self.delta0 <= pi/6), include_last=False)
        ocp.subject_to(-pi/10<= (ocp.der(self.delta0) <= pi/10), include_last=False)

        ocp.subject_to(-pi/2 <= (self.beta01 <= pi/2), include_last=False)
        ocp.subject_to(-pi/2 <= (self.beta12 <= pi/2), include_last=False)
        # Initial guess
        ocp.set_initial(self.theta0, .1)
        ocp.set_initial(self.theta1, 0)
        ocp.set_initial(self.theta2, 0)
        ocp.set_initial(self.v0,     -.2)

if __name__ == '__main__':
    pass
