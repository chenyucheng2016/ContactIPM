from matplotlib.animation import FuncAnimation
import matplotlib as mpl
import matplotlib.pyplot as plt
import matplotlib.patches as patches
import numpy as np
import time
import motion_planning.problem_specificationT1 as problem_specification
from benchmark_tools import SolveFatrop, SolveIpopt
from numpy import cos, sin
import casadi as cs
from dynamicsT1 import TruckTrailerDynamics



class Rectangle:
    def __init__(self, w_left, w_right, l_front, l_back, **args):

        #   (4)       y     (3)   ^
        #    |--------|------|    |  w_left
        #    |        o---x  |    x
        #    |               |    |
        #    |---------------|    |  w_right
        #   (1)             (2)   v
        #     <-------><---->
        #      l_back   l_front
        self.w_left = w_left
        self.w_right = w_right
        self.l_front = l_front
        self.l_back = l_back
        self.args = args

    def add_to_axes(self, ax):
        self.ax = ax
        self.rectangle = patches.Rectangle(
            (-self.l_back, -self.w_right), self.l_back + self.l_front, self.w_left + self.w_right, **self.args)
        self.ax.add_patch(self.rectangle)

    def update_position(self, x, y, theta):
        self.rectangle.set_transform(mpl.transforms.Affine2D().rotate_around(
            0.00, 0., (theta)).translate(x, y) + self.ax.transData)


class Wheel(Rectangle):
    def __init__(self, delta_x, delta_y):
        self.delta_x = delta_x 
        self.delta_y = delta_y 
        super().__init__(0.01, 0.01, 0.1, 0.1, color = "k")
    def update_position(self, x, y, theta, delta=0):
        super().update_position(x, y, theta)
        self.rectangle.set_transform(mpl.transforms.Affine2D().rotate_around(0,0,delta).translate(self.delta_x, self.delta_y).rotate_around(
            0.00, 0., (theta)).translate(x, y) + self.ax.transData)

    


class Vehicle:
    def __init__(self, w_left, w_right, l_front, l_back, **args):
        args["fill"] = False
        self.rectangle = Rectangle(w_left, w_right, l_front, l_back, **args)
        self.wheel1 = Wheel(0.0, w_left)
        self.wheel2 = Wheel(0.0 , -w_right)
        self.w_left = w_left
        self.w_right = w_right
        self.l_front = l_front
        self.l_back = l_back
        self.elements = [self.wheel1, self.wheel2, self.rectangle]
        if "color" in args:
            self.color = args["color"]
        else:
            self.color = "k"
        self.x = 0
        self.y = 0
        self.theta = 0
        self.back = np.array([0, 0])

    def add_to_axes(self, ax):
        for element in self.elements:
            element.add_to_axes(ax)
        self.xy,  = ax.plot([0], [0], 'x', color = self.color)

    def update_position(self, x, y, theta):
        for element in self.elements:
            element.update_position(x, y, theta)
        self.xy.set_data(x, y)
        self.x = x
        self.y = y
        self.theta = theta
        self.back = np.array([x - self.l_back * cos(theta), y - self.l_back * sin(theta)])

class Coupling:
    def __init__(self, veh1, veh2):
        self.veh1 = veh1
        self.veh2 = veh2
    def add_to_axes(self, ax):
        self.line,  = ax.plot([0, 0], [0, 0], color = "k")
        self.dot,  = ax.plot([0], [0], "o", color = "k")
    def update_position(self):
        self.line.set_data([self.veh1.back[0], self.veh2.x], [self.veh1.back[1], self.veh2.y])
        self.dot.set_data([self.veh1.back[0]], [self.veh1.back[1]])
class Path:
    def __init__(self,**args):
        self.args = args
    def add_to_axes(self, ax):
        self.line,  = ax.plot([], [], **self.args)
    def update(self, x, y):
        self.line.set_data(x, y)

class TruckTrailerVisualization:
    def __init__(self, prob: TruckTrailerDynamics, ax=None, fig=None):
        if ax is None:
            self.fig, self.ax = plt.subplots()
        else:
            self.ax = ax
            self.fig = fig
        self.ax.set_xlim([-1, 7.])
        self.ax.set_ylim([-4, 2.])
        self.ax.set_aspect('equal')
        self.ax.axis("off")
        self.truck = Vehicle(prob.W0/2, prob.W0/2, prob.L0, prob.M0, color = "grey")
        self.trailer1 = Vehicle(prob.W1/2, prob.W1/2, .8 * prob.L1, prob.M1, color ='r')
        self.coupling_truck_trailer = Coupling(self.truck, self.trailer1)
        self.trailer2 = Vehicle(prob.W2/2, prob.W2/2, .8 * prob.L2, prob.M2, color = 'g')
        self.coupling_trailer1_trailer2 = Coupling(self.trailer1, self.trailer2)
        self.steer_wheel = Wheel(prob.L0, 0)
        self.steer_wheel_path = Path(color = "grey")
        self.truck.add_to_axes(self.ax)
        self.trailer1.add_to_axes(self.ax)
        self.coupling_truck_trailer.add_to_axes(self.ax)
        self.trailer2.add_to_axes(self.ax)
        self.coupling_trailer1_trailer2.add_to_axes(self.ax)
        self.steer_wheel.add_to_axes(self.ax)
        self.steer_wheel_path.add_to_axes(self.ax)

    def update_visualization(self, x0s, y0s, theta0s, x1s, y1s, theta1s, x2s, y2s, theta2s, delta0s):

        self.truck.update_position(x0s, y0s, theta0s)
        self.trailer1.update_position(x1s, y1s, theta1s)
        self.coupling_truck_trailer.update_position()
        self.trailer2.update_position(x2s, y2s, theta2s)
        self.coupling_trailer1_trailer2.update_position()
        self.steer_wheel.update_position(x0s, y0s, theta0s, delta0s)
        self.ax.figure.canvas.draw()

class VisualizationFunctor:
    def __init__(self, vis:TruckTrailerVisualization, prob:TruckTrailerDynamics, res):
        self.vis = vis
        self.x0_res = res.sample(prob.x0, grid = 'control')[1]
        self.y0_res = res.sample(prob.y0, grid = 'control')[1]
        self.theta0_res = res.sample(prob.theta0, grid = 'control')[1]
        self.x1_res = res.sample(prob.x1, grid = 'control')[1]
        self.y1_res = res.sample(prob.y1, grid = 'control')[1]
        self.theta1_res = res.sample(prob.theta1, grid = 'control')[1]
        self.x2_res = res.sample(prob.x2, grid = 'control')[1]
        self.y2_res = res.sample(prob.y2, grid = 'control')[1]
        self.theta2_res = res.sample(prob.theta2, grid = 'control')[1]
        self.delta0_res = res.sample(prob.delta0, grid = 'control')[1]
        front_wheel_xy = cs.vertcat(prob.x0 + prob.L0 * cs.cos(prob.theta0), prob.y0 + prob.L0 * cs.sin(prob.theta0))
        self.front_wheel_path = res.sample(front_wheel_xy, grid = 'control')[1]
    def __call__(self, i):
        self.vis.steer_wheel_path.update(self.front_wheel_path[:i, 0], self.front_wheel_path[:i, 1])
        self.vis.update_visualization(self.x0_res[i], self.y0_res[i], self.theta0_res[i], self.x1_res[i], self.y1_res[i], self.theta1_res[i], self.x2_res[i], self.y2_res[i], self.theta2_res[i], self.delta0_res[i])
        # self.vis.ax.figure.canvas.draw()
    

if __name__ == "__main__":
    N = 100
    fig, ax = plt.subplots()
    res, prob, stat = SolveFatrop(problem_specification.TruckTrailerTime(), N)
    vis = TruckTrailerVisualization(prob, ax, fig)
    visfunctor = VisualizationFunctor(vis, prob, res)
    ani = FuncAnimation(
    vis.fig, visfunctor,
    frames=range(N))
    plt.show()
    # save the animation 
    ani.save('truck_trailer.mp4', writer='ffmpeg',dpi = 300, fps=30, extra_args=['-vcodec', 'libx264'])