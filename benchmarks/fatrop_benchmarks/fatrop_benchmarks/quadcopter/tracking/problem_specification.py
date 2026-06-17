# import dynamics.py from parent directory
import sys
import os
from quadcopter.dynamics import QuadcopterDynamics 
from benchmark_tools import SolveFatrop, SolveIpopt, SolveAcados
import numpy as np
import casadi as cs
import rockit


class QuadcopterTrack(QuadcopterDynamics):
    def __init__(self):
        self.ocp = rockit.Ocp(T=5.0)
        QuadcopterDynamics.__init__(self, self.ocp)
        # parameters
        # Initial & terminal conditions
        p0 = cs.vertcat(0., 0., 2.5)
        v0 = cs.vertcat(0, 0, 0)
        eul0 = cs.vertcat(0, 0, 0)

        pf = cs.vertcat(0., 5., 2.5)
        vf = cs.vertcat(0., 0., 0.)
        eulf = cs.vertcat(0., 0., 0.)

        
        # add objective

        # self.ocp.add_objective(self.ocp.sum(1e2*(self.psi- eul0[2])**2, include_last = True))
        # self.ocp.add_objective(self.ocp.at_tf(self.ocp.T))
        # control minimization
        self.ocp.add_objective(self.ocp.sum(1e0*cs.sumsqr(self.ocp.u)))
        # traking error minimization
        N = 50
        self.theta = np.linspace(0, 2*np.pi, N+1)
        # self.theta = 1.0/self.ocp.T * 2*np.pi * self.ocp.t

        self.track = self.ocp.parameter(3, 1, grid = 'control', include_last=True)
        track_points = p0 + np.vstack((np.cos(self.theta), np.sin(self.theta), np.zeros((1,N+1))))
        self.ocp.set_value(self.track, track_points)
        self.ocp.set_initial(self.p, track_points)
        self.ocp.add_objective(1e0*self.ocp.sum(cs.sumsqr(self.p - self.track), include_last=True))
        # add boundary value constraints
        self.ocp.subject_to(self.ocp.at_t0(self.p) == p0)
        # self.ocp.set_initial(self.p,  p0 + np.array([1.0, 0.0, 0.0]))
        self.ocp.subject_to(self.ocp.at_t0(self.v) == v0)
        # self.ocp.subject_to(self.ocp.at_t0(self.at) == 9.81)
        self.ocp.subject_to(self.ocp.at_t0(self.eul) == eul0)


        # self.ocp.subject_to(self.ocp.at_tf(self.p) == pf)
        # self.ocp.subject_to(self.ocp.at_tf(self.v) == vf)
        # self.ocp.subject_to(self.ocp.at_tf(self.eul[[0,1]]) == eulf[[0,1]])
        # self.ocp.subject_to(self.psi  ==0, include_first=False)

if __name__ == "__main__":
    res, prob, _ = SolveFatrop(QuadcopterTrack(), 50)
    p_sol = res.sample(prob.p, grid='control')[1]
    # visualize the 3D trajectory coordinates using matplotlib
    # import matplotlib.pyplot as plt
    # from mpl_toolkits.mplot3d import Axes3D
    # fig = plt.figure()
    # ax = fig.add_subplot(111, projection='3d')
    # # make the axes equal
    # max_range = np.array([p_sol[:,0].max()-p_sol[:,0].min(), p_sol[:,1].max()-p_sol[:,1].min(), p_sol[:,2].max()-p_sol[:,2].min()]).max() / 2.0
    # mid_x = (p_sol[:,0].max()+p_sol[:,0].min()) * 0.5
    # mid_y = (p_sol[:,1].max()+p_sol[:,1].min()) * 0.5
    # mid_z = (p_sol[:,2].max()+p_sol[:,2].min()) * 0.5
    # ax.set_xlim(mid_x - max_range, mid_x + max_range)
    # ax.set_ylim(mid_y - max_range, mid_y + max_range)
    # ax.set_zlim(mid_z - max_range, mid_z + max_range)
    # ax.plot(p_sol[:,0], p_sol[:,1], p_sol[:,2])
    # plt.show()
    SolveIpopt(QuadcopterTrack(), 50)
    SolveAcados(QuadcopterTrack(), 50)