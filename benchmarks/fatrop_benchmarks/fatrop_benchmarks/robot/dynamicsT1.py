import rockit
import casadi as cs
import numpy as np
from urdf2casadi import urdfparser as u2c
from robot.FrankaChain import FrankaChain
import os 


class RobotDynamics:
    def __init__(self, ocp: rockit.Ocp):
        N = 50
        # hyper params
        n_joints = 7
        dir_path = os.path.dirname(os.path.realpath(__file__))
        self.frankachain = FrankaChain('panda_link0', ['panda_link1', 'panda_link2', 'panda_link3', 'panda_link4',
                                                'panda_link5', 'panda_link6', 'panda_link7'], ['TCP_frame', 'panda_link8', 'panda_hand'], dir_path + '/panda_arm_model.urdf')

        max_vel = np.array([2.175,2.175, 2.175, 2.175,2.610, 2.610, 2.610]) 
        self.joints_theta = ocp.state(n_joints)
        joints_vel = ocp.control(n_joints)
        self.T = ocp.state()
        ocp.set_der(self.T, 0.)
        ocp.subject_to(self.T > 0)
        ocp.set_der(self.joints_theta, joints_vel * self.T)
        self.frankachain.set_up_expressions(self.joints_theta)
        ocp.subject_to(self.frankachain.joint_lower < (
            self.joints_theta < self.frankachain.joint_upper))
        ocp.subject_to(-max_vel < (joints_vel < max_vel), include_last=False)
        ocp.set_initial(self.joints_theta, 0.5*self.frankachain.joint_lower + 0.5*self.frankachain.joint_upper)

if __name__ == '__main__':
    pass
