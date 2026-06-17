import matplotlib as mpl
import matplotlib.pyplot as plt
import numpy as np
from benchmark_tools import SolveFatrop, SolveIpopt
import mpc_2D.problem_specification2DMPC as spec2D
import casadi as cs
from matplotlib.animation import FuncAnimation


class HangingChainVisualization:
    def __init__(self, no_masses, ax=None, fig = None, ground = np.array([0.0, 0.0])):
        p_all = np.zeros((no_masses+1, 2))
        self.ground = ground
        self.no_masses = no_masses
        if ax is None:
            self.fig, self.ax = plt.subplots()
        else:
            self.ax = ax
            self.fig = fig
        self.ax.axis("off")
        self.line, = self.ax.plot(p_all[:,0], p_all[:,1], '-o')
        self.ax.set_xlim((-.25, 1.25))
        self.ax.set_ylim((-2., .5))
        self.ax.set_aspect('equal')
        # plt.axis('off')
    def visualize(self, p):
        p_all = np.vstack((self.ground[np.newaxis, :], p.T))
        self.line.set_xdata(p_all[:,0])
        self.line.set_ydata(p_all[:,1])
        pass
class VisualizationFunctor:
    def __init__(self, vis, p_result):
        self.p_result = p_result
        self.vis: HangingChainVisualization = vis
    def __call__(self, frame):
        self.vis.visualize(self.p_result[frame])


if __name__ == "__main__":
    N = 50
    refine = 5 
    res, prob, _ = SolveIpopt(spec2D.HangingChain2DMPC() , N, jit = False)
    # res, prob, _ = SolveFatrop(spec2D.HangingChain2DMPC() , N)
    p_matrix = cs.horzcat(*prob.p)
    p_result = res.sample(p_matrix, grid = 'integrator', refine =refine)[1]
    vis = HangingChainVisualization(prob.no_masses)
    ani = FuncAnimation(
        vis.fig, VisualizationFunctor(vis, p_result),
        frames=range(N*refine))
    plt.show()
    ## save the animation
    ani.save('hanging_chain.mp4', writer='ffmpeg',dpi = 300, fps=30, extra_args=['-vcodec', 'libx264'])
