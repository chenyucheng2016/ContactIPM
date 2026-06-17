# create a script that visualizes the cart-pendulum system using matplotlib. You can use the following snippet to visualize the system:
from matplotlib.animation import FuncAnimation
import matplotlib as mpl
import matplotlib.pyplot as plt
import matplotlib.patches as patches
import numpy as np
import time
import mpc.problem_specification
import opt_time.problem_specification
from benchmark_tools import SolveFatrop, SolveIpopt


class CartPendulumVisualization:
    def __init__(self, L=1.0, ax = None, fig = None, gray =False, first = True):
        self.L = L
        if ax is None:
            self.fig, self.ax = plt.subplots()
        else:
            self.ax = ax
            self.fig = fig
        self.ax.set_xlim([-1.5, 1.5])
        self.ax.set_ylim([-1.5, 1.5])
        self.ax.set_aspect('equal')
        self.ax.axis('off')

        # self.ax.grid()
        if gray:
            self.line = patches.Rectangle((-1.5, -0.005), 3., 0.01, fc='gray')
            self.cart = patches.Rectangle((-0.05, -0.05), 0.1, 0.1, fc='gray')
            self.pendulum = patches.Rectangle((-0.01, 0), 0.02, L, fc='gray')
        else:
            self.line = patches.Rectangle((-1.5, -0.005), 3., 0.01, fc='gray')
            self.cart = patches.Rectangle((-0.05, -0.05), 0.1, 0.1, fc='r')
            self.pendulum = patches.Rectangle((-0.01, 0), 0.02, L, fc='b')
        if first:
            self.ax.add_patch(self.line)
        self.ax.add_patch(self.cart)
        self.ax.add_patch(self.pendulum)

    def visualize(self, x, theta):
        self.cart.set_xy([x-0.05, -0.05])
        # rotate the pendulum over an angle theta
        self.pendulum.set_transform(mpl.transforms.Affine2D().rotate_around(
            0.00, 0., (-np.pi + theta)).translate(x, 0) + self.ax.transData)
        # set the position of the pendulum
        self.fig.canvas.draw()
        # self.fig.canvas.flush_events()


class VisualizationFunctor:
    def __init__(self, vis, x, theta, L=1.):
        self.x = x
        self.theta = theta
        self.vis: CartPendulumVisualization = vis

    def __call__(self, frame):
        self.vis.visualize(self.x[frame], self.theta[frame])


# use VisualizationFunctor to visualize the cart-pendulum system in an animation using FuncAnimation

if __name__ == '__main__':
    mode = "mpc"
    if mode == "time": 
        N = 100
        res, prob, _ = SolveFatrop(opt_time.problem_specification.CartPendulumTime(), N)
    if mode == "mpc":
        N = 50
        res, prob, _ = SolveFatrop(mpc.problem_specification.CartPendulumMPC(), N)
    _, res_theta = res.sample(prob.theta, grid='control')
    _, res_x = res.sample(prob.x, grid='control')
    vis = CartPendulumVisualization()
    ani = FuncAnimation(
        vis.fig, VisualizationFunctor(vis, res_x, res_theta),
        frames=range(N))
    plt.show(block=True)
    fig, ax = plt.subplots()
    for i in range(1, N, 2):
        vis = CartPendulumVisualization(fig = fig, ax = ax, gray=True, first = (i ==1))
        VisualizationFunctor(vis, res_x, res_theta)(i)
    vis = CartPendulumVisualization(fig = fig, ax = ax, gray=False, first = False)
    VisualizationFunctor(vis, res_x, res_theta)(0)
    vis = CartPendulumVisualization(fig = fig, ax = ax, gray=False, first = False)
    VisualizationFunctor(vis, res_x, res_theta)(N)
    plt.show()
    ## save the animation
    ani.save('cart_pendulum' + mode + '.mp4', writer='ffmpeg', dpi = 300, fps=30, extra_args=['-vcodec', 'libx264'])